#!/usr/bin/env python3
"""Sample a subset from the DeepCAD train/val/test split for Stage 1 rendering."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", required=True, type=Path)
    parser.add_argument("--cadquery_dir", required=True, type=Path)
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest_out", required=True, type=Path)
    parser.add_argument("--split_out", required=True, type=Path)
    args = parser.parse_args()

    with open(args.split_file) as f:
        splits = json.load(f)

    rng = random.Random(args.seed)

    sampled = {}
    manifest_lines = []
    for split_name, n_want in [("train", args.n_train), ("validation", args.n_val), ("test", args.n_test)]:
        ids = splits[split_name]
        available = [oid for oid in ids if (args.cadquery_dir / f"{oid}.py").exists()]
        rng.shuffle(available)
        chosen = available[:n_want]
        sampled[split_name] = sorted(chosen)
        for oid in sampled[split_name]:
            manifest_lines.append(str(args.cadquery_dir / f"{oid}.py"))
        print(f"{split_name}: sampled {len(chosen)} / {len(available)} available")

    manifest_lines.sort()
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text("\n".join(manifest_lines) + "\n")
    print(f"Manifest: {len(manifest_lines)} entries → {args.manifest_out}")

    args.split_out.write_text(json.dumps(sampled, indent=2))
    print(f"Split: {args.split_out}")


if __name__ == "__main__":
    main()
