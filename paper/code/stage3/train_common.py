"""Shared training utilities (config loading, trainer setup) for all VLMs.

Per-model scripts call `load_config()` then build their model + processor
and call `run_training()` with a model-specific `format_example` function.
"""

from __future__ import annotations

import json
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from data import VideoCadQueryDataset


@dataclass
class StageConfig:
    raw: dict

    def __getattr__(self, key):
        return self.raw[key]


def load_config(path: str | Path) -> StageConfig:
    with open(path) as f:
        return StageConfig(yaml.safe_load(f))


class TokenizedDataset(Dataset):
    """Wraps VideoCadQueryDataset, applies model-specific formatting on-the-fly."""

    def __init__(self, base: VideoCadQueryDataset, format_fn: Callable):
        self.base = base
        self.format_fn = format_fn

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        row = self.base[idx]
        return self.format_fn(row)


def vlm_collator(features: list[dict]) -> dict:
    """Collate VLM features.

    Token tensors (input_ids, attention_mask, labels) are padded and stacked.
    Vision tensors (pixel_values, image_grid_thw, etc.) are concatenated along
    dim 0 — models expect flat (total_patches, ...) shape, not batched.

    For batch_size=1 (our default), pixel_values/grid_thw pass through unchanged
    after the cat (single-element cat is a no-op).
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    batch = {}
    keys = features[0].keys()
    for k in keys:
        vals = [f[k] for f in features]

        if k == "input_ids" or k == "attention_mask":
            batch[k] = pad_sequence(vals, batch_first=True, padding_value=0)
        elif k == "labels":
            batch[k] = pad_sequence(vals, batch_first=True, padding_value=-100)
        else:
            # Concat along the patch/image dimension (dim 0).
            batch[k] = torch.cat(vals, dim=0)
    return batch


def build_lora(model, target_modules: list[str], cfg: StageConfig,
               task_type: str | None = "CAUSAL_LM"):
    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=task_type,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora)


def make_training_args(model_name: str, cfg: StageConfig) -> TrainingArguments:
    output_dir = Path(cfg.output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=(cfg.precision == "bf16"),
        fp16=(cfg.precision == "fp16"),
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )


def trainable_param_summary(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return (f"trainable: {trainable:,} ({100 * trainable / total:.2f}%) "
            f"of total {total:,}")
