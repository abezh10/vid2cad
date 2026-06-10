#!/usr/bin/env python3
"""Train-set geometry distribution + qualitative failure inspection for curves.

(1) bucket distribution over train.jsonl (is the train set boxy-dominated?)
(2) for a given eval, print GT vs predicted code for the lowest-IoU curved /
    angled objects to reveal the failure mode (wrong primitive? truncation?
    line-approximation of arcs? bad dims?).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_geometry import analyze_script, load_iou


def dist(manifest: Path, label: str):
    buckets = Counter()
    nonortho = 0
    n = 0
    tok_by_bucket = {}
    for line in open(manifest):
        row = json.loads(line)
        text = Path(row["script_path"]).read_text()
        f = analyze_script(text)
        buckets[f["bucket"]] += 1
        nonortho += int(f["non_orthogonal"])
        n += 1
        tok_by_bucket.setdefault(f["bucket"], []).append(len(text))
    print(f"\n=== {label}  (n={n}) ===")
    order = ["axis-aligned boxy", "axis circles/holes", "angled (diag edges)",
             "curved (arc/spline)", "tilted_wp"]
    for b in order:
        c = buckets.get(b, 0)
        chars = int(np.mean(tok_by_bucket[b])) if b in tok_by_bucket else 0
        print(f"  {b:<24} {c:>5}  ({100*c/n:4.1f}%)   avg_script_chars={chars}")
    print(f"  --> NON-ORTHOGONAL: {nonortho} ({100*nonortho/n:.1f}%)  "
          f"ORTHOGONAL: {n-nonortho} ({100*(n-nonortho)/n:.1f}%)")


def inspect(test: Path, eval_dir: Path, bucket: str, k: int):
    iou = load_iou(eval_dir)
    preds = {json.loads(l)["object_id"]: json.loads(l)
             for l in open(eval_dir / "per_object.jsonl")}
    rows = []
    for line in open(test):
        row = json.loads(line)
        oid = row["object_id"]
        f = analyze_script(Path(row["script_path"]).read_text())
        if f["bucket"] == bucket and oid in iou:
            rows.append((oid, iou[oid]["iou"], row["script_path"]))
    rows.sort(key=lambda r: r[1])
    print(f"\n{'#'*90}\n# {bucket}: {len(rows)} objs, showing {k} LOWEST-IoU cases\n{'#'*90}")
    for oid, iouv, spath in rows[:k]:
        gt = Path(spath).read_text()
        pred = preds.get(oid, {})
        code = pred.get("code", "")
        note = ""
        if not pred.get("executed", False):
            note = f"  [EXEC FAILED: {pred.get('stderr_tail','')[:120]}]"
        print(f"\n===== {oid}  IoU={iouv:.3f}{note} =====")
        print(f"--- GT ({len(gt)} chars) ---")
        print(gt)
        print(f"--- PRED ({len(code)} chars) ---")
        print(code[:1400])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="stage2b/train.jsonl", type=Path)
    ap.add_argument("--test", default="stage2b/test.jsonl", type=Path)
    ap.add_argument("--eval_dir", default="stage3_runs/qwen_vit_r64/eval", type=Path)
    ap.add_argument("--dist_only", action="store_true")
    ap.add_argument("--bucket", default="curved (arc/spline)")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    dist(args.train, "TRAIN")
    dist(args.test, "TEST")
    if not args.dist_only:
        inspect(args.test, args.eval_dir, args.bucket, args.k)


if __name__ == "__main__":
    main()
