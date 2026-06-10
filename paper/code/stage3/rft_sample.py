#!/usr/bin/env python3
"""RFT step 1 — sample K CadQuery candidates per TRAIN object from an SFT adapter.

Rejection-sampling fine-tuning (STaR): we draw K stochastic completions per
training object from the current best LoRA adapter, then (in rft_select.py) keep
the highest-IoU candidate as the new supervised target. This is self-distillation
— it only amplifies shapes the model already lands occasionally, but that's enough
to move geometric IoU without a reward model or RL infrastructure.

This script is the GPU half: it loads the base model + adapter exactly like
eval.py, builds the same chat prompt, and writes one JSONL line per object with
its K extracted-code candidates. Scoring/selection happens on CPU later.

Sharded for SLURM array jobs (--shard_id / --num_shards) and resumable (skips
objects already present in this shard's output file).

Output: <out_dir>/candidates_shard_<shard_id>.jsonl
  {"object_id": "...", "candidates": [{"code": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from data import VideoCadQueryDataset, SYSTEM_PROMPT, USER_PROMPT
from train_common import load_config
from eval import extract_code


def build_inputs(model, processor, args_model, frames):
    """Replicate eval.py's per-model prompt construction exactly."""
    if args_model == "kimi":
        img_content = [{"type": "image"} for _ in frames]
    else:
        img_content = [{"type": "image", "image": fr} for fr in frames]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": (
            img_content + [{"type": "text", "text": USER_PROMPT}]
        )},
    ]
    if args_model == "kimi":
        prefix_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prefix_text], images=frames, return_tensors="pt"
        ).to(model.device)
    else:
        tmpl_kwargs = {"crop_to_patches": False} if args_model == "internvl" else {}
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **tmpl_kwargs,
        ).to(model.device)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["qwen", "glm", "internvl", "kimi"])
    parser.add_argument("--checkpoint_dir", required=True, type=Path,
                        help="SFT LoRA adapter to sample from (e.g. stage3_runs/qwen)")
    parser.add_argument("--config", default="stage3/config.yaml")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="where candidate shards are written (round_dir/candidates)")
    parser.add_argument("--split", default="train",
                        help="manifest split to sample over (train.jsonl)")
    parser.add_argument("--k", type=int, default=8,
                        help="candidates to draw per object")
    parser.add_argument("--gen_batch", type=int, default=4,
                        help="num_return_sequences per generate() call (caps memory)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="limit objects (0 = all in shard); for quick tests")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    # Same import-time monkey-patches as eval.py.
    if args.model == "qwen":
        import train_qwen  # noqa: F401
    elif args.model == "kimi":
        import train_kimi  # noqa: F401

    cfg = load_config(args.config)
    mcfg = cfg.models[args.model]
    n_frames = mcfg.get("n_frames", cfg.n_frames)
    max_new = args.max_new_tokens or cfg.gen_max_new_tokens

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"candidates_shard_{args.shard_id}.jsonl"

    # Resume: collect object_ids already written in this shard.
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["object_id"])
                except Exception:
                    pass
    print(f"resume: {len(done)} objects already in {out_path.name}")

    print(f"Loading {args.model}: base={mcfg['hf_id']} adapter={args.checkpoint_dir}")
    processor = AutoProcessor.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    if args.model == "kimi":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            mcfg["hf_id"],
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        base = AutoModelForImageTextToText.from_pretrained(
            mcfg["hf_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(base, str(args.checkpoint_dir))
    model.eval()

    if args.model == "kimi":
        n_pat = train_kimi.patch_kimi_generation(model)
        print(f"patched Kimi generation prep on {n_pat} LM class(es)")

    ds = VideoCadQueryDataset(
        Path(cfg.manifest_dir) / f"{args.split}.jsonl", cfg.frames_dir, n_frames)

    # Deterministic contiguous shard over the manifest.
    all_idx = list(range(len(ds)))
    shard_idx = all_idx[args.shard_id::args.num_shards]
    if args.limit:
        shard_idx = shard_idx[:args.limit]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(shard_idx)} objects "
          f"(k={args.k} gen_batch={args.gen_batch} max_new={max_new} "
          f"temp={args.temperature} top_p={args.top_p})", flush=True)

    n_written = 0
    with open(out_path, "a") as fout:
        for pos, idx in enumerate(shard_idx):
            row = ds[idx]
            oid = row["object_id"]
            if oid in done:
                continue
            inputs = build_inputs(model, processor, args.model, row["frames"])
            prompt_len = inputs["input_ids"].shape[1]

            seqs = []
            remaining = args.k
            with torch.no_grad():
                while remaining > 0:
                    nb = min(args.gen_batch, remaining)
                    out = model.generate(
                        **inputs, max_new_tokens=max_new, do_sample=True,
                        num_return_sequences=nb,
                        temperature=args.temperature, top_p=args.top_p)
                    seqs.extend(list(out))
                    remaining -= nb

            cands = []
            for seq in seqs:
                gen_text = processor.decode(seq[prompt_len:], skip_special_tokens=True)
                cands.append({"code": extract_code(gen_text)})

            fout.write(json.dumps({"object_id": oid, "candidates": cands}) + "\n")
            fout.flush()
            n_written += 1
            if (pos + 1) % 10 == 0:
                print(f"  {pos+1}/{len(shard_idx)} sampled (written {n_written})",
                      flush=True)

    print(f"DONE shard {args.shard_id}: wrote {n_written} new objects to {out_path}")


if __name__ == "__main__":
    main()
