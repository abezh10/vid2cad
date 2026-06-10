#!/usr/bin/env python3
"""List completed videos under a Stage-1 render root as a JSONL manifest.

Feeds sample_frames.py for render roots other than the original stage1_output
(e.g. stage1_expansion/, stage1_test_hr/). Rows: {object_id, video_path}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    n = 0
    with open(args.out, "w") as f:
        for vid in sorted(args.root.glob("*/*/video.mp4")):
            oid = f"{vid.parent.parent.name}/{vid.parent.name}"
            f.write(json.dumps({"object_id": oid, "video_path": str(vid)}) + "\n")
            n += 1
    print(f"{n} videos under {args.root} -> {args.out}")


if __name__ == "__main__":
    main()
