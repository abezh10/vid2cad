#!/usr/bin/env python3
"""Randomly sample completed Stage 1 videos for manual review."""

import argparse
import json
import random
import shutil
from pathlib import Path


def discover_completed(output_dir: Path) -> list[Path]:
    """Find all object directories that have a video.mp4."""
    results = []
    for bucket in sorted(output_dir.iterdir()):
        if not bucket.is_dir() or bucket.name.startswith("."):
            continue
        for obj in sorted(bucket.iterdir()):
            if obj.is_dir() and (obj / "video.mp4").exists():
                results.append(obj)
    return results


def main():
    parser = argparse.ArgumentParser(description="Sample completed Stage 1 videos for review")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Stage 1 output directory to scan")
    parser.add_argument("--sample_dir", required=True, type=Path,
                        help="Directory to copy sampled videos into")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of videos to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--copy-all", action="store_true",
                        help="Copy full output (depth, mask, metadata) not just video")
    args = parser.parse_args()

    completed = discover_completed(args.output_dir)
    print(f"Found {len(completed)} completed renders in {args.output_dir}")

    if len(completed) == 0:
        print("ERROR: no completed renders found")
        raise SystemExit(1)

    n = min(args.n, len(completed))
    rng = random.Random(args.seed)
    sampled = rng.sample(completed, n)
    sampled.sort()

    args.sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for obj_dir in sampled:
        rel = obj_dir.relative_to(args.output_dir)
        flat_name = str(rel).replace("/", "_")

        if args.copy_all:
            dest = args.sample_dir / flat_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(obj_dir, dest)
        else:
            shutil.copy2(obj_dir / "video.mp4", args.sample_dir / f"{flat_name}.mp4")

        manifest.append(str(rel))
        print(f"  {rel}")

    manifest_path = args.sample_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps({
        "source": str(args.output_dir),
        "n_total_completed": len(completed),
        "n_sampled": n,
        "seed": args.seed,
        "objects": manifest,
    }, indent=2))

    print(f"\nSampled {n} / {len(completed)} videos → {args.sample_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
