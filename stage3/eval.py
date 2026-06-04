#!/usr/bin/env python3
"""Evaluate a trained LoRA checkpoint on the test set.

For each test object:
1. Load n_frames frames (config-driven)
2. Generate CadQuery code (greedy)
3. Run the code in a subprocess; record execution success/error

Outputs:
  - per_object.jsonl  : one line per test object with generated code + status
  - summary.json      : execution success rate

The checkpoint dir holds a PEFT LoRA adapter (adapter_model.safetensors), not a
full model, so we load the base model from the model's hf_id and attach the
adapter via PeftModel.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from data import VideoCadQueryDataset, SYSTEM_PROMPT, USER_PROMPT, USER_PROMPT_DEPTH
from train_common import load_config


def extract_code(text: str) -> str:
    """Pull runnable CadQuery out of a raw generation.

    Thinking models (e.g. GLM-4.1V) wrap the answer in <think>...</think>
    reasoning and/or markdown fences. Instruct models (Qwen) emit raw code, for
    which every step below is a no-op. We strip reasoning, take a fenced block
    if present, then trim to the start of the actual program.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Kimi-VL-Thinking wraps reasoning in ◁think▷...◁/think▷ delimiters.
    text = re.sub(r"◁think▷.*?◁/think▷", "", text, flags=re.DOTALL)
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find('"""Generated'), text.find("import cadquery"))
              if i != -1]
    if starts:
        text = text[min(starts):]
    return text.strip()


def execute_script(code: str, timeout: int = 30) -> dict:
    """Run a CadQuery script in a subprocess and return status."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        script = td / "gen.py"
        script.write_text(code)
        try:
            r = subprocess.run(
                ["python3", str(script)],
                capture_output=True, text=True,
                timeout=timeout, cwd=td,
            )
            return {
                "exit_code": r.returncode,
                "stderr_tail": r.stderr[-500:] if r.stderr else "",
                "executed": r.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stderr_tail": "timeout", "executed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["qwen", "glm", "internvl", "kimi"])
    parser.add_argument("--checkpoint_dir", required=True, type=Path)
    parser.add_argument("--config", default="stage3/config.yaml")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of test examples (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="skip object_ids already in per_object.jsonl and append "
                             "(recovers a timed-out run instead of restarting)")
    parser.add_argument("--out_dir", type=Path, default=None)
    # Depth conditioning (must match how the checkpoint was trained): interleave
    # colormapped depth frames with the RGB frames per viewpoint.
    parser.add_argument("--use_depth", action="store_true",
                        help="interleave depth frames (depth-conditioned checkpoint)")
    parser.add_argument("--depth_frames_dir", type=Path, default=None,
                        help="override depth-frame dir (else cfg.depth_frames_dir)")
    # Best-of-N sampling. n_samples>1 draws N candidates per object and keeps
    # the first that executes (else the first). repetition_penalty /
    # no_repeat_ngram_size break the greedy workplane-repetition loops that
    # dominate the failure set; sampling lets a valid completion win.
    parser.add_argument("--n_samples", type=int, default=1,
                        help="best-of-N: sample N candidates, return first executable")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    # Some models need monkey-patches that live in their train_*.py modules;
    # importing the module applies them at import time. Qwen: vision-encoder
    # linspace fix. Kimi: PytorchGELUTanh alias + DynamicCache.get_usable_length.
    if args.model == "qwen":
        import train_qwen  # noqa: F401
    elif args.model == "kimi":
        import train_kimi  # noqa: F401

    cfg = load_config(args.config)
    mcfg = cfg.models[args.model]
    n_frames = mcfg.get("n_frames", cfg.n_frames)

    out_dir = args.out_dir or (args.checkpoint_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}: base={mcfg['hf_id']} adapter={args.checkpoint_dir}")
    processor = AutoProcessor.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    if args.model == "kimi":
        # Match training: 4-bit base + eager attention (Kimi's DeepSeek-V3 LM
        # registers only eager/flash_attention_2, and flash-attn isn't built).
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
        # Fix Kimi's generation-time attention-mask crop (off-by-one vs
        # kv_seq_len under DynamicCache). Class is only defined once the remote
        # code is loaded, so patch the live class after load.
        n_pat = train_kimi.patch_kimi_generation(model)
        print(f"patched Kimi generation prep on {n_pat} LM class(es)")

    depth_dir = None
    if args.use_depth:
        depth_dir = args.depth_frames_dir or cfg.raw.get("depth_frames_dir")
        if not depth_dir:
            sys.exit("--use_depth set but no depth_frames_dir (CLI or config)")
        print(f"DEPTH conditioning ON: interleaving depth frames from {depth_dir}")
    test_ds = VideoCadQueryDataset(
        Path(cfg.manifest_dir) / "test.jsonl",
        cfg.frames_dir, n_frames, depth_frames_dir=depth_dir,
    )
    n = len(test_ds) if args.limit == 0 else min(args.limit, len(test_ds))
    print(f"Evaluating on {n} test examples ...")

    max_new = args.max_new_tokens or cfg.gen_max_new_tokens
    # Optional token-level penalties. WARNING: repetition_penalty and
    # no_repeat_ngram_size corrupt CODE generation — code legitimately repeats
    # tokens ('cq', '.', 'Vector', ')'), so penalizing repeats pushes the model
    # onto wrong tokens (misspelled attrs, tabs-for-dots). A 1.2/ngram=3 run
    # scored 0/100 exec vs 88.9% greedy. Default OFF; leave them off for code.
    penalties = {}
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        penalties["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size:
        penalties["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    def greedy_kwargs():
        return {"max_new_tokens": max_new, "do_sample": False, **penalties}

    def sample_kwargs(num):
        return {"max_new_tokens": max_new, "do_sample": True,
                "num_return_sequences": num, "temperature": args.temperature,
                "top_p": args.top_p, **penalties}

    print(f"n_samples={args.n_samples} max_new={max_new} penalties={penalties} "
          f"temp={args.temperature} top_p={args.top_p}")

    per_object_path = out_dir / "per_object.jsonl"
    done_ids = set()
    if args.resume and per_object_path.exists():
        with open(per_object_path) as rf:
            for line in rf:
                try:
                    done_ids.add(json.loads(line)["object_id"])
                except Exception:
                    pass
        print(f"resume: {len(done_ids)} objects already in per_object.jsonl; skipping")
    write_mode = "a" if done_ids else "w"
    n_executed = 0
    n_exec_of_n_total = 0
    with open(per_object_path, write_mode) as f:
        for i in range(n):
            row = test_ds[i]
            if row["object_id"] in done_ids:
                continue
            # With depth on, interleave (rgb, depth) per viewpoint so the pose
            # pair is adjacent; else just the RGB frames.
            if "depth_frames" in row:
                images = []
                for r_, d_ in zip(row["frames"], row["depth_frames"]):
                    images.append(r_)
                    images.append(d_)
                prompt_text = USER_PROMPT_DEPTH
            else:
                images = row["frames"]
                prompt_text = USER_PROMPT
            # Kimi takes a flat image list passed to the processor (placeholders
            # in the message); the others embed the image objects in the message.
            if args.model == "kimi":
                img_content = [{"type": "image"} for _ in images]
            else:
                img_content = [{"type": "image", "image": im} for im in images]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": (
                    img_content + [{"type": "text", "text": prompt_text}]
                )},
            ]

            if args.model == "kimi":
                prefix_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[prefix_text], images=images, return_tensors="pt"
                ).to(model.device)
            else:
                # InternVL: disable dynamic hi-res tiling (matches training) so
                # each frame is one 448² tile rather than ~13 — else token counts
                # explode past the context window.
                tmpl_kwargs = {"crop_to_patches": False} if args.model == "internvl" else {}
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    **tmpl_kwargs,
                ).to(model.device)

            with torch.no_grad():
                if args.n_samples > 1:
                    # Greedy anchor + (N-1) sampled candidates. The anchor
                    # guarantees best-of-N >= the greedy baseline; the sampled
                    # candidates add shots at recovering the greedy failures
                    # (workplane-repetition collapse) without touching the rest.
                    seqs = list(model.generate(**inputs, **greedy_kwargs()))
                    seqs += list(model.generate(
                        **inputs, **sample_kwargs(args.n_samples - 1)))
                elif cfg.gen_do_sample:
                    seqs = list(model.generate(
                        **inputs, max_new_tokens=max_new, do_sample=True,
                        temperature=cfg.gen_temperature, **penalties))
                else:
                    seqs = list(model.generate(**inputs, **greedy_kwargs()))
            prompt_len = inputs["input_ids"].shape[1]

            # Decode + execute every candidate (1 when n_samples==1).
            cands = []  # (generated, code, exec_result)
            for seq in seqs:
                gen_text = processor.decode(seq[prompt_len:], skip_special_tokens=True)
                code_c = extract_code(gen_text)
                cands.append((gen_text, code_c, execute_script(code_c)))

            n_exec_of_n = sum(1 for _, _, r in cands if r["executed"])
            # best-of-N selection: first executable candidate, else the first.
            generated, code, exec_result = next(
                (c for c in cands if c[2]["executed"]), cands[0])
            if exec_result["executed"]:
                n_executed += 1
            n_exec_of_n_total += n_exec_of_n

            f.write(json.dumps({
                "object_id": row["object_id"],
                "generated": generated,
                "code": code,
                "ground_truth": row["code"],
                "n_samples": len(cands),
                "n_exec_of_n": n_exec_of_n,
                **exec_result,
            }) + "\n")
            f.flush()

            if (i + 1) % 25 == 0:
                rate = n_executed / (i + 1)
                print(f"  {i+1}/{n}  exec_rate={rate:.2%}", flush=True)

    # Recompute over the full per_object.jsonl so a --resume run reports the
    # complete test set, not just this invocation's newly-processed objects.
    all_rows = [json.loads(l) for l in open(per_object_path)]
    n_total = len(all_rows)
    n_executed = sum(1 for r in all_rows if r.get("executed"))
    n_exec_of_n_total = sum(r.get("n_exec_of_n", 0) for r in all_rows)
    n_samples_total = sum(r.get("n_samples", 1) for r in all_rows)
    summary = {
        "model": args.model,
        "checkpoint": str(args.checkpoint_dir),
        "n_evaluated": n_total,
        "n_executed": n_executed,
        "execution_rate": n_executed / n_total if n_total else 0.0,
        "n_samples": args.n_samples,
        # mean fraction of the N candidates that executed — a single-sample
        # (pass@1) proxy to compare against the best-of-N execution_rate above.
        "mean_exec_of_n": n_exec_of_n_total / n_samples_total if n_samples_total else 0.0,
        "gen": {
            "temperature": args.temperature, "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": max_new,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nExecution rate: {n_executed}/{n_total} = {n_executed / n_total:.2%}")
    print(f"Per-object: {per_object_path}")
    print(f"Summary:    {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
