#!/usr/bin/env python3
"""Classify the ENTIRE translated CadQuery corpus (~213k scripts) by geometry.

Writes one JSONL row per script: object_id, bucket, binary flags, n_solids,
script_chars. This index drives difficulty-stratified sampling of the Stage-1
expansion render (favor curved/angled/tilted parts — the buckets the model is
weakest on — with FRESH data rather than duplication).

Pure regex (diag_geometry.analyze_script), parallelized; minutes on 16 cores.

Usage:
  python3 stage3/classify_corpus.py \
      --cadquery_dir deepcad-data/cadquery_py \
      --out deepcad-data/geometry_index.jsonl --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_geometry import analyze_script


def process(py_path: str) -> dict | None:
    p = Path(py_path)
    oid = f"{p.parent.name}/{p.stem}"
    try:
        text = p.read_text()
    except Exception:
        return None
    f = analyze_script(text)
    return {
        "object_id": oid,
        "bucket": f["bucket"],
        "non_orthogonal": f["non_orthogonal"],
        "has_angled_line": f["has_angled_line"],
        "has_curve": f["has_curve"],
        "has_tilted_wp": f["has_tilted_wp"],
        "n_solids": f["n_solids"],
        "script_chars": len(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadquery_dir", default="deepcad-data/cadquery_py", type=Path)
    ap.add_argument("--out", default="deepcad-data/geometry_index.jsonl", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(str(p) for p in args.cadquery_dir.glob("*/*.py"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"Classifying {len(paths)} scripts with {args.workers} workers ...")

    buckets = Counter()
    n_done = n_fail = 0
    with Pool(args.workers) as pool, open(args.out, "w") as fh:
        for rec in pool.imap_unordered(process, paths, chunksize=256):
            if rec is None:
                n_fail += 1
                continue
            fh.write(json.dumps(rec) + "\n")
            buckets[rec["bucket"]] += 1
            n_done += 1
            if n_done % 20000 == 0:
                print(f"  {n_done}/{len(paths)}", flush=True)

    print(f"\nDone: {n_done} classified, {n_fail} unreadable -> {args.out}")
    total = sum(buckets.values())
    for b, c in buckets.most_common():
        print(f"  {b:<24} {c:>7}  ({100*c/total:.1f}%)")


if __name__ == "__main__":
    main()
