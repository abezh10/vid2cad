#!/usr/bin/env python3
"""Fine-tune Kimi-VL-A3B-Instruct via LoRA on video→CadQuery pairs.

Kimi-VL uses MoonViT and a custom chat template. The processor handles
both. Note: Kimi is MoE — LoRA adapts the MLA attention projections plus the
always-active shared experts; the 64 routed experts stay frozen (and are
pinned to eval to dodge the gate's noaux_tc training assert).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import transformers.activations as _activations
if not hasattr(_activations, "PytorchGELUTanh"):
    _activations.PytorchGELUTanh = _activations.GELUTanh

# Kimi-VL uses removed Cache APIs; alias them. get_usable_length is hit during
# the forward pass (training + eval); seen_tokens only during generation
# (prepare_inputs_for_generation) so it surfaces at eval time, not training.
from transformers.cache_utils import DynamicCache as _DynamicCache
if not hasattr(_DynamicCache, "get_usable_length"):
    _DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
if not hasattr(_DynamicCache, "seen_tokens"):
    _DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, Trainer
from peft import prepare_model_for_kbit_training

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


def patch_kimi_generation(model):
    """Fix the off-by-one attention-mask crop in Kimi's LM generation prep.

    DeepseekV3ForCausalLM.prepare_inputs_for_generation hardcodes
        max_cache_length = past_key_values.get_seq_length()
    i.e. the *current* cache length, not a bounded-cache capacity. Its
    "crop the mask if we'd exceed max cache length" branch then fires on
    every decode step (cache_length + 1 > cache_length) and truncates the
    attention_mask back to the past length — one token short of kv_seq_len —
    which trips the eager-attention size assert (mask 1751 vs expected 1752).
    For an unbounded DynamicCache there is no max length, so the crop must
    never fire: we rebuild the method with max_cache_length=None. Generation
    only (eval); training never calls generate(). Returns #classes patched.
    """
    from transformers.cache_utils import Cache

    def _pifg(self, input_ids, past_key_values=None, attention_mask=None,
              inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None  # DynamicCache is unbounded -> never crop
            if (attention_mask is not None
                    and attention_mask.shape[1] > input_ids.shape[1]):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if (max_cache_length is not None and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length):
                attention_mask = attention_mask[:, -max_cache_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    patched = set()
    for _m in model.modules():
        if type(_m).__name__ == "DeepseekV3ForCausalLM":
            cls = type(_m)
            if cls not in patched:
                cls.prepare_inputs_for_generation = _pifg
                patched.add(cls)
    return len(patched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3/config.yaml")
    parser.add_argument("--output_name", default="kimi",
                        help="subdir under output_root (use a fresh name to "
                             "avoid clobbering a previous run's adapter)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg.models["kimi"]
    n_frames = mcfg.get("n_frames", cfg.n_frames)

    # sim2real augmentation (train split only; val stays clean). Off unless the
    # config has an `augment:` block with enabled: true.
    from augment import FrameAugmentor
    train_aug = FrameAugmentor.from_config(cfg.raw.get("augment"))
    if train_aug is not None:
        print(f"AUGMENT ON (train only): {train_aug.cfg}")

    print(f"Loading processor and model: {mcfg['hf_id']} (4-bit QLoRA)")
    processor = AutoProcessor.from_pretrained(mcfg["hf_id"], trust_remote_code=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_id"],
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Kimi-VL's LM advertises supports_gradient_checkpointing but its decoder
    # forward never actually calls a checkpoint fn (dead flag) — so the built-in
    # enable path is a no-op. We wrap the decoder layers manually below; here we
    # just disable the KV cache (incompatible with training-time checkpointing).
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    model = build_lora(model, mcfg["lora_target_modules"], cfg)
    print(trainable_param_summary(model))

    # Kimi-VL's MoE is DeepSeek-V3 style, with two coupled obstacles to training:
    #   * the gate's "noaux_tc" routing hard-asserts `not self.training`;
    #   * DeepseekV3MoE.forward's training branch calls
    #     AddAuxiliaryLoss(y, aux_loss), but aux_loss is None whenever the gate
    #     isn't in its training path — so pinning only the gate would crash there.
    # LoRA adapts attention + the shared experts, so the router + 64 routed
    # experts are frozen and the inference routing (moe_infer) is numerically
    # what we want. Pin each MoE block (and its gate) to eval and neutralize
    # their .train() so the Trainer's model.train() can't flip them: the block
    # then takes the no-grad moe_infer path and skips the aux loss. Gradient
    # still reaches the attention LoRA via the residual stream, and the shared
    # experts run outside moe_infer's no_grad so their LoRA trains normally.
    n_moe = 0
    for _m in model.modules():
        is_block = hasattr(_m, "experts") and hasattr(_m, "gate")
        is_gate = getattr(_m, "topk_method", None) is not None
        if is_block or is_gate:
            _m.eval()
            _m.train = (lambda mod: (lambda mode=True: mod))(_m)
            n_moe += 1
    print(f"pinned {n_moe} MoE module(s) to eval mode")

    # The DeepSeek-V3 decoder forward calls each layer directly — its
    # supports_gradient_checkpointing flag is dead (no checkpoint branch). The
    # eager attention matrix (~700 MiB at this seq len) is the OOM site, and it
    # only fits if the other ~30 layers' activations aren't all retained. Wrap
    # each decoder layer's forward in non-reentrant checkpointing: stored
    # activations drop to the layer boundaries and the transient attn matrix
    # fits. Non-reentrant tolerates the no-grad embedding input (LoRA in the
    # attention adds the grad-requiring path that backprop follows).
    import torch.utils.checkpoint as _ckpt
    n_ckpt = 0
    for _m in model.modules():
        if type(_m).__name__ == "DeepseekV3DecoderLayer":
            def _make_ckpt_forward(orig):
                def _ckpt_forward(*a, **kw):
                    if not torch.is_grad_enabled():
                        return orig(*a, **kw)
                    return _ckpt.checkpoint(orig, *a, use_reentrant=False, **kw)
                return _ckpt_forward
            _m.forward = _make_ckpt_forward(_m.forward)
            n_ckpt += 1
    print(f"wrapped {n_ckpt} decoder layer(s) in gradient checkpointing")

    def format_example(row):
        # Build messages without embedded image objects — apply_chat_template only
        # produces the prompt string; we pass the flat image list directly to
        # the processor to avoid Kimi's nested-list issue.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": (
                [{"type": "image"} for _ in row["frames"]] +
                [{"type": "text", "text": USER_PROMPT}]
            )},
            {"role": "assistant", "content": [{"type": "text", "text": row["code"]}]},
        ]
        full_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prefix_text = processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[full_text], images=row["frames"], return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # prefix_len must be computed with the SAME image expansion as the full
        # sequence: processor(text, images=...) expands each {"type":"image"}
        # placeholder into ~hundreds of <|media_pad|> tokens. Tokenizing
        # prefix_text alone (no images) gives a too-short prefix, so the mask
        # would leave hundreds of image tokens unmasked as training targets and
        # the loss plateaus at ~ln(vocab) predicting unpredictable image pads.
        prefix_inputs = processor(text=[prefix_text], images=row["frames"], return_tensors="pt")
        prefix_len = prefix_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prefix_len] = -100

        out = {k: v.squeeze(0) for k, v in inputs.items()
               if k not in ("input_ids", "attention_mask")}
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["labels"] = labels
        return out

    train_ds = TokenizedDataset(
        VideoCadQueryDataset(
            Path(cfg.manifest_dir) / "train.jsonl",
            cfg.frames_dir, n_frames, augment=train_aug),
        format_example,
    )
    val_ds = TokenizedDataset(
        VideoCadQueryDataset(
            Path(cfg.manifest_dir) / "validation.jsonl",
            cfg.frames_dir, n_frames),
        format_example,
    )

    training_args = make_training_args(args.output_name, cfg)
    training_args.gradient_checkpointing = False

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
