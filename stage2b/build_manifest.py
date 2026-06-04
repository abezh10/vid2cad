#!/usr/bin/env python3
"""Pair completed Stage 1 videos with their source CadQuery scripts.

Walks stage1_output, finds objects with a video.mp4, locates the matching
.py script in deepcad-data/cadquery_py, and writes JSONL manifest files
respecting the train/val/test split from subset_split.json.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage1_output"))
    parser.add_argument("--cadquery_dir", type=Path,
                        default=Path("/scratch/users/abzhong/deepcad-data/cadquery_py"))
    parser.add_argument("--split_file", type=Path,
                        default=Path("/scratch/users/abzhong/stage1_output/subset_split.json"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split_file) as f:
        splits = json.load(f)

    completed = {
        f"{p.parent.parent.name}/{p.parent.name}"
        for p in args.stage1_dir.glob("*/*/video.mp4")
    }
    print(f"Found {len(completed)} completed videos")

    totals = {}
    for split_name, object_ids in splits.items():
        rows = []
        missing_video = missing_script = 0

        for obj_id in object_ids:
            if obj_id not in completed:
                missing_video += 1
                continue

            bucket, name = obj_id.split("/")
            script_path = args.cadquery_dir / bucket / f"{name}.py"
            if not script_path.exists():
                missing_script += 1
                continue

            video_path = args.stage1_dir / obj_id / "video.mp4"
            rows.append({
                "object_id": obj_id,
                "video_path": str(video_path),
                "script_path": str(script_path),
                "split": split_name,
            })

        out_path = args.out_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        totals[split_name] = len(rows)
        print(f"{split_name:>10}: {len(rows):>5} pairs   "
              f"(missing video: {missing_video}, missing script: {missing_script})  "
              f"-> {out_path.name}")

    summary_path = args.out_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(totals, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
