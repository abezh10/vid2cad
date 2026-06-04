#!/usr/bin/env python3
"""Sample uniformly spaced frames from videos for VLM input.

Uses opencv to read N frames per video and writes them as JPEGs into
stage2b/frames/<bucket>/<id>/frame_XX.jpg. Defaults to 16 frames.

Run mode:
  --serial : single process (debug)
  --workers N : multiprocessing pool over all manifest rows

Idempotent: skips objects whose frame directory already has N frames.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np


def sample_video(args):
    row, n_frames, out_root, quality = args
    obj_id = row["object_id"]
    video_path = Path(row["video_path"])
    out_dir = out_root / obj_id

    if out_dir.exists():
        existing = sorted(out_dir.glob("frame_*.jpg"))
        if len(existing) == n_frames:
            return obj_id, "skip", None

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return obj_id, "fail", f"cannot open {video_path}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        return obj_id, "fail", f"only {total} frames"

    indices = np.linspace(0, total - 1, n_frames).round().astype(int)
    saved = 0
    for i, target in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = out_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved += 1
    cap.release()

    if saved != n_frames:
        return obj_id, "fail", f"only saved {saved}/{n_frames}"
    return obj_id, "ok", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path,
                        help="JSONL manifest from build_manifest.py")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b/frames"))
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = [json.loads(line) for line in f]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Sampling {args.n_frames} frames from {len(rows)} videos "
          f"-> {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = [(r, args.n_frames, args.out_dir, args.quality) for r in rows]

    ok = skip = fail = 0
    fail_examples = []

    if args.serial:
        for w in work:
            oid, status, err = sample_video(w)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                if len(fail_examples) < 5:
                    fail_examples.append((oid, err))
    else:
        with mp.Pool(args.workers) as pool:
            for i, (oid, status, err) in enumerate(pool.imap_unordered(sample_video, work, chunksize=8)):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    if len(fail_examples) < 5:
                        fail_examples.append((oid, err))
                if (i + 1) % 200 == 0:
                    print(f"  progress: {i+1}/{len(work)}  ok={ok} skip={skip} fail={fail}")

    print(f"\nDone: ok={ok}  skip={skip}  fail={fail}")
    if fail_examples:
        print("Failure examples:")
        for oid, err in fail_examples:
            print(f"  {oid}: {err}")


if __name__ == "__main__":
    main()
