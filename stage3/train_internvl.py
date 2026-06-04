#!/usr/bin/env python3
"""Fine-tune InternVL3-8B via LoRA on video→CadQuery pairs.

Uses the transformers-native InternVLForConditionalGeneration (the `-hf`
weights), loaded through AutoModelForImageTextToText — same path as Qwen/GLM.
The original OpenGVLab custom `InternVLChatModel` has a `forward(pixel_values,
...)` interface that doesn't line up with apply_chat_template, so we use the
native conversion instead. InternVL3's LLM backbone is Qwen2.5, so the LoRA
targets are the standard q/k/v/o + gate/up/down projections.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Trainer

sys.path.insert(0, str(Path(__file__).parent))
from data import VideoCadQueryDataset, SYSTEM_PROMPT, USER_PROMPT
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg.models["internvl"]

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

    eos_id = processor.tokenizer.eos_token_id

    def format_example(row):
        # Same approach as train_glm.py: run only the prompt (system + user)
        # through the chat template — it carries the images and yields
        # pixel_values — then tokenize the assistant code separately so the
        # target stays clean (templates stringify list content on non-user
        # turns into their Python repr).
        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": (
                [{"type": "image", "image": f} for f in row["frames"]] +
                [{"type": "text", "text": USER_PROMPT}]
            )},
        ]
        # crop_to_patches=False disables InternVL's dynamic hi-res tiling, which
        # would otherwise blow each frame up to ~13 tiles (8 frames -> ~26k
        # tokens, over both max_seq_len and the model context). One 448x448 tile
        # per frame (256 image tokens) is plenty for these CAD renders and keeps
        # us comparable to the single-image-per-frame budget of the other models.
        prompt_inputs = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            crop_to_patches=False,
        )
        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_mask = prompt_inputs["attention_mask"][0]

        code_ids = processor.tokenizer(
            row["code"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        code_ids = torch.cat([code_ids, torch.tensor([eos_id])])

        input_ids = torch.cat([prompt_ids, code_ids])
        attention_mask = torch.cat([prompt_mask, torch.ones_like(code_ids)])
        labels = torch.cat([
            torch.full((len(prompt_ids),), -100, dtype=input_ids.dtype),
            code_ids.clone(),
        ])

        out = {k: v.squeeze(0) for k, v in prompt_inputs.items()
               if k not in ("input_ids", "attention_mask")}
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["labels"] = labels
        return out

    train_ds = TokenizedDataset(
        VideoCadQueryDataset(
            Path(cfg.manifest_dir) / "train.jsonl",
            cfg.frames_dir, cfg.n_frames, augment=train_aug),
        format_example,
    )
    val_ds = TokenizedDataset(
        VideoCadQueryDataset(
            Path(cfg.manifest_dir) / "validation.jsonl",
            cfg.frames_dir, cfg.n_frames),
        format_example,
    )

    training_args = make_training_args("internvl", cfg)

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
