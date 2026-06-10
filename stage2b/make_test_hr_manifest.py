#!/usr/bin/env python3
"""Build test_hr.jsonl: the frozen 994 test objects with HIGH-RES frame dirs.

Same object ids / scripts / order semantics as stage2b/test.jsonl; only
frame_dir points at the high-res extraction (stage2b/frames_test_hr/<id>).
Used via eval.py --test_manifest stage2b/test_hr.jsonl for the resolution
ablation. Objects whose HR render/extraction failed are dropped (reported).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_test", default="stage2b/test.jsonl", type=Path)
    ap.add_argument("--hr_root", default="stage1_test_hr", type=Path)
    ap.add_argument("--frames_root", default="stage2b/frames_test_hr", type=Path)
    ap.add_argument("--min_frames", type=int, default=24)
    ap.add_argument("--out", default="stage2b/test_hr.jsonl", type=Path)
    args = ap.parse_args()

    kept, dropped = [], 0
    for line in open(args.orig_test):
        row = json.loads(line)
        oid = row["object_id"]
        vid = args.hr_root / oid / "video.mp4"
        fdir = args.frames_root / oid
        n_fr = len(list(fdir.glob("frame_*.jpg"))) if fdir.exists() else 0
        if not vid.exists() or n_fr < args.min_frames:
            dropped += 1
            continue
        kept.append({**row, "video_path": str(vid), "frame_dir": str(fdir)})

    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"test_hr: kept {len(kept)} / dropped {dropped} -> {args.out}")


if __name__ == "__main__":
    main()
