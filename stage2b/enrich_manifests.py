#!/usr/bin/env python3
"""Add frame_dir field to each row in train/val/test manifests.

Also drops rows whose frame directory doesn't have the expected count.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b"))
    parser.add_argument("--frames_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b/frames"))
    parser.add_argument("--n_frames", type=int, default=16)
    args = parser.parse_args()

    for split in ["train", "validation", "test"]:
        path = args.manifest_dir / f"{split}.jsonl"
        if not path.exists():
            continue

        with open(path) as f:
            rows = [json.loads(line) for line in f]

        new_rows = []
        dropped = 0
        for r in rows:
            frame_dir = args.frames_dir / r["object_id"]
            frames = sorted(frame_dir.glob("frame_*.jpg")) if frame_dir.exists() else []
            if len(frames) != args.n_frames:
                dropped += 1
                continue
            r["frame_dir"] = str(frame_dir)
            new_rows.append(r)

        with open(path, "w") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")

        print(f"{split:>10}: {len(new_rows):>5} kept, {dropped:>4} dropped (missing frames)")


if __name__ == "__main__":
    main()
