#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-7B-Instruct via LoRA on video→CadQuery pairs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, Trainer
from peft import prepare_model_for_kbit_training

# Monkey-patch the buggy fast_pos_embed_interpolate: it passes Tensor h/w to
# torch.linspace which requires int (transformers 4.57.x library bug).
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as _qvlm


def _patched_fast_pos_embed_interpolate(self, grid_thw):
    # Normalize to (N, 3) regardless of upstream shape
    if grid_thw.dim() == 3:
        grid_thw = grid_thw.reshape(-1, 3)
    grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]
    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h_i = int(h.item()) if hasattr(h, "item") else int(h)
        w_i = int(w.item()) if hasattr(w, "item") else int(w)
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h_i)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w_i)
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor
        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side
        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    dev = self.pos_embed.weight.device
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=dev)
    weight_tensor = torch.tensor(
        weight_list, dtype=self.pos_embed.weight.dtype, device=dev)
    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split(
        [int(h) * int(w) for h, w in zip(grid_hs, grid_ws)])

    permuted = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        ti, hi, wi = int(t), int(h), int(w)
        pos_embed = pos_embed.repeat(ti, 1)
        pos_embed = (
            pos_embed.view(ti, hi // merge_size, merge_size,
                           wi // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        permuted.append(pos_embed)
    return torch.cat(permuted)


_qvlm.Qwen3VLMoeVisionModel.fast_pos_embed_interpolate = _patched_fast_pos_embed_interpolate

sys.path.insert(0, str(Path(__file__).parent))
from data import VideoCadQueryDataset, SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_DEPTH
from train_common import (
    load_config,
    TokenizedDataset,
    build_lora,
    make_training_args,
    trainable_param_summary,
    vlm_collator,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3/config.yaml")
    parser.add_argument("--model_key", default="qwen",
                        help="key under cfg.models: 'qwen' = frozen-ViT baseline, "
                             "'qwen_vit' = also LoRA-adapts the vision transformer")
    parser.add_argument("--output_name", default=None,
                        help="adapter subdir under output_root (defaults to --model_key)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg.models[args.model_key]
    output_name = args.output_name or args.model_key
    use_depth = bool(cfg.raw.get("use_depth", False))
    depth_dir = cfg.raw.get("depth_frames_dir") if use_depth else None
    if use_depth:
        print(f"DEPTH conditioning ON: interleaving depth frames from {depth_dir}")

    # sim2real augmentation (train split only; val stays clean). Off unless the
    # config has an `augment:` block with enabled: true.
    from augment import FrameAugmentor
    train_aug = FrameAugmentor.from_config(cfg.raw.get("augment"))
    if train_aug is not None:
        print(f"AUGMENT ON (train only): {train_aug.cfg}")

    print(f"Loading processor and model: {mcfg['hf_id']}")
    processor = AutoProcessor.from_pretrained(mcfg["hf_id"], trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        mcfg["hf_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = build_lora(model, mcfg["lora_target_modules"], cfg)
    print(trainable_param_summary(model))

    def format_example(row):
        if "depth_frames" in row:
            user_content = []
            for rgb, dep in zip(row["frames"], row["depth_frames"]):
                user_content.append({"type": "image", "image": rgb})
                user_content.append({"type": "image", "image": dep})
            user_content.append({"type": "text", "text": USER_PROMPT_DEPTH})
        else:
            user_content = (
                [{"type": "image", "image": f} for f in row["frames"]] +
                [{"type": "text", "text": USER_PROMPT}]
            )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": row["code"]}]},
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Mask labels for everything except the assistant response
        labels = input_ids.clone()
        # Find the assistant turn start by tokenizing the prefix
        prefix_messages = messages[:-1]
        prefix_inputs = processor.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        prefix_len = prefix_inputs["input_ids"].shape[1]
        labels[:prefix_len] = -100

        out = {k: v.squeeze(0) for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["labels"] = labels
        return out

    train_ds = TokenizedDataset(
        VideoCadQueryDataset(
            Path(cfg.manifest_dir) / "train.jsonl",
            cfg.frames_dir, cfg.n_frames, depth_frames_dir=depth_dir,
            augment=train_aug),
        format_example,
    )
    val_ds = TokenizedDataset(
        VideoCadQueryDataset(
            Path(cfg.manifest_dir) / "validation.jsonl",
            cfg.frames_dir, cfg.n_frames, depth_frames_dir=depth_dir),
        format_example,
    )

    training_args = make_training_args(output_name, cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=vlm_collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
