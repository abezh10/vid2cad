#!/usr/bin/env python3
"""Estimate progress + ETA for frame extraction.

Fast variant: walks bucket dirs once, counts object dirs with N entries,
samples a handful of jpgs for start/latest mtimes.

Usage:
  python3 stage2b/eta.py
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b/frames"))
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--target", type=int, default=9909,
                        help="Expected total objects (train+val+test)")
    args = parser.parse_args()

    completed = 0
    partial = 0
    sampled_mtimes = []  # mtimes from a few jpgs for rate estimate
    n_sample = 50

    for bucket in os.scandir(args.frames_dir):
        if not bucket.is_dir():
            continue
        for obj in os.scandir(bucket.path):
            if not obj.is_dir():
                continue
            count = 0
            first_jpg = None
            for entry in os.scandir(obj.path):
                if entry.name.startswith("frame_") and entry.name.endswith(".jpg"):
                    count += 1
                    if first_jpg is None:
                        first_jpg = entry
            if count == args.n_frames:
                completed += 1
                if first_jpg is not None and len(sampled_mtimes) < n_sample:
                    sampled_mtimes.append(first_jpg.stat().st_mtime)
            elif count > 0:
                partial += 1

    if not sampled_mtimes:
        print(f"Completed: {completed} / {args.target}")
        print("No frames yet — cannot estimate rate.")
        return

    now = time.time()
    # Use the earliest jpg as start, the latest as last write
    # (one stat per directory in our sample)
    all_mtimes_for_latest = []
    for bucket in os.scandir(args.frames_dir):
        if not bucket.is_dir():
            continue
        for obj in os.scandir(bucket.path):
            if obj.is_dir():
                try:
                    all_mtimes_for_latest.append(obj.stat().st_mtime)
                except FileNotFoundError:
                    pass

    start = min(sampled_mtimes)
    # Get latest from sampled set plus a fresh scan of dirs
    latest = max(all_mtimes_for_latest) if all_mtimes_for_latest else max(sampled_mtimes)
    elapsed = now - start
    objs_per_sec = completed / elapsed if elapsed > 0 else 0
    objs_per_min = objs_per_sec * 60
    remaining = args.target - completed
    eta_sec = remaining / objs_per_sec if objs_per_sec > 0 else 0
    pct = 100 * completed / args.target

    def fmt(secs):
        return str(timedelta(seconds=int(secs)))

    print(f"Started:    {datetime.fromtimestamp(start):%H:%M:%S}")
    print(f"Last write: {datetime.fromtimestamp(latest):%H:%M:%S}")
    print(f"Now:        {datetime.fromtimestamp(now):%H:%M:%S}")
    print(f"Elapsed:    {fmt(elapsed)}")
    print()
    print(f"Completed:  {completed:>5} / {args.target}  ({pct:.1f}%)")
    print(f"Partial:    {partial:>5} (frames present but incomplete)")
    print(f"Remaining:  {remaining:>5}")
    print(f"Throughput: {objs_per_min:.1f} objects/min")
    print()
    print(f"ETA:        {fmt(eta_sec)}")
    print(f"Done at:    {datetime.fromtimestamp(now + eta_sec):%H:%M:%S}")


if __name__ == "__main__":
    main()
