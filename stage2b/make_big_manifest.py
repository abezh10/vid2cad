#!/usr/bin/env python3
"""Build the BIG training manifest: original train set + Stage-1 expansion.

Scans the expansion render root for completed videos (video.mp4), keeps only
ids listed in expansion_ids.json (train-split, leakage-checked at sampling
time), and emits stage2b/train_big.jsonl = original train.jsonl rows + new rows
whose frame_dir points at the expansion frame tree.

Run AFTER:
  1. the expansion render finishes (stage1_expansion/<bucket>/<id>/video.mp4)
  2. sample_frames.py extracted JPEGs into --frames_root

Also re-verifies zero overlap with the frozen val/test object ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_train", default="stage2b/train.jsonl", type=Path)
    ap.add_argument("--expansion_root", default="stage1_expansion", type=Path)
    ap.add_argument("--expansion_ids", default="stage1_output/expansion_ids.json", type=Path)
    ap.add_argument("--frames_root", default="stage2b/frames_exp", type=Path)
    ap.add_argument("--cadquery_dir", default="deepcad-data/cadquery_py", type=Path)
    ap.add_argument("--min_frames", type=int, default=24,
                    help="require at least this many extracted frames per object")
    ap.add_argument("--out", default="stage2b/train_big.jsonl", type=Path)
    args = ap.parse_args()

    orig_rows = [json.loads(l) for l in open(args.orig_train)]
    orig_ids = {r["object_id"] for r in orig_rows}

    # frozen val/test ids -> hard exclusion (belt & suspenders)
    frozen = set()
    for name in ("validation.jsonl", "test.jsonl"):
        p = args.orig_train.parent / name
        if p.exists():
            frozen |= {json.loads(l)["object_id"] for l in open(p)}

    allowed = set(json.load(open(args.expansion_ids))["train_expansion"])

    new_rows, n_no_video, n_no_frames, n_skip = [], 0, 0, 0
    for vid in sorted(args.expansion_root.glob("*/*/video.mp4")):
        oid = f"{vid.parent.parent.name}/{vid.parent.name}"
        if oid not in allowed or oid in frozen or oid in orig_ids:
            n_skip += 1
            continue
        fdir = args.frames_root / oid
        n_fr = len(list(fdir.glob("frame_*.jpg"))) if fdir.exists() else 0
        if n_fr < args.min_frames:
            n_no_frames += 1
            continue
        script = args.cadquery_dir / f"{oid}.py"
        if not script.exists():
            continue
        new_rows.append({
            "object_id": oid,
            "video_path": str(vid),
            "script_path": str(script),
            "split": "train",
            "frame_dir": str(fdir),
        })

    n_expected = len(allowed)
    print(f"expansion: {len(new_rows)} usable / {n_expected} sampled "
          f"(skipped: not-allowed/frozen {n_skip}, frames<{args.min_frames}: {n_no_frames})")

    all_rows = orig_rows + new_rows
    with open(args.out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"train_big: {len(orig_rows)} original + {len(new_rows)} expansion "
          f"= {len(all_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
