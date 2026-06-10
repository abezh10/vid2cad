#!/usr/bin/env python3
"""Sample a difficulty-stratified EXPANSION set for Stage-1 rendering.

Draws ONLY from the official DeepCAD train split (no val/test contamination),
excludes every object already in the original 10k subset, and stratifies by
the geometry buckets from stage3/classify_corpus.py so the expansion is rich
in the geometry the model is weakest on (curved / angled / tilted parts) —
fresh hard data instead of duplication.

Default mix for --n_total 20000 (combined with the existing 7,924-object train
set this lands ~47% non-orthogonal, matching the duplication mix that won the
qwen_vit_r64_bal ablation, but with all-new objects):
    curved   30%   angled  22%   tilted  all available (~few hundred)
    circles  20%   boxy    rest

Outputs:
  --manifest_out : one .py path per line (input to submit_stage1_subset.sh)
  --ids_out      : {"train_expansion": [ids]} (for manifest building later)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BUCKETS = ["curved (arc/spline)", "angled (diag edges)", "tilted_wp",
           "axis circles/holes", "axis-aligned boxy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_file", default="deepcad-data/train_val_test_split.json", type=Path)
    ap.add_argument("--geometry_index", default="deepcad-data/geometry_index.jsonl", type=Path)
    ap.add_argument("--cadquery_dir", default="deepcad-data/cadquery_py", type=Path)
    ap.add_argument("--exclude_split", action="append", type=Path, default=None,
                    help="json file(s) of id-lists to exclude (repeatable). Any "
                         "dict-of-lists shape works (subset_split.json, "
                         "expansion_ids.json). Default: stage1_output/subset_split.json")
    ap.add_argument("--n_total", type=int, default=20000)
    ap.add_argument("--frac_curved", type=float, default=0.30)
    ap.add_argument("--frac_angled", type=float, default=0.22)
    ap.add_argument("--frac_circles", type=float, default=0.20)
    # tilted: take ALL available (rare); boxy: remainder
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--manifest_out", default="stage1_output/expansion_manifest.txt", type=Path)
    ap.add_argument("--ids_out", default="stage1_output/expansion_ids.json", type=Path)
    args = ap.parse_args()

    official = json.load(open(args.split_file))
    train_ids = set(official["train"])

    exclude_files = args.exclude_split or [Path("stage1_output/subset_split.json")]
    excluded = set()
    for ef in exclude_files:
        if ef.exists():
            ex = json.load(open(ef))
            for ids in ex.values():
                excluded.update(ids)
    print(f"official train: {len(train_ids)}   excluded (already rendered/sampled): {len(excluded)}")

    # geometry index -> available pool per bucket (train-split, not yet rendered)
    pool = defaultdict(list)
    n_index = 0
    for line in open(args.geometry_index):
        rec = json.loads(line)
        oid = rec["object_id"]
        n_index += 1
        if oid in train_ids and oid not in excluded:
            pool[rec["bucket"]].append(oid)
    print(f"geometry index rows: {n_index}")
    for b in BUCKETS:
        print(f"  available {b:<24} {len(pool[b]):>7}")

    rng = random.Random(args.seed)
    for b in BUCKETS:
        rng.shuffle(pool[b])

    want = {
        "curved (arc/spline)": int(args.n_total * args.frac_curved),
        "angled (diag edges)": int(args.n_total * args.frac_angled),
        "axis circles/holes": int(args.n_total * args.frac_circles),
        "tilted_wp": len(pool["tilted_wp"]),  # take all — rarest, weakest bucket
    }
    chosen = {}
    for b, n in want.items():
        chosen[b] = pool[b][: min(n, len(pool[b]))]
    n_so_far = sum(len(v) for v in chosen.values())
    n_boxy = max(0, args.n_total - n_so_far)
    chosen["axis-aligned boxy"] = pool["axis-aligned boxy"][: min(n_boxy, len(pool["axis-aligned boxy"]))]

    all_ids = sorted(set(oid for ids in chosen.values() for oid in ids))
    print(f"\nselected {len(all_ids)} objects:")
    nonortho = 0
    for b in BUCKETS:
        c = len(chosen.get(b, []))
        print(f"  {b:<24} {c:>6}  ({100*c/len(all_ids):.1f}%)")
        if b not in ("axis circles/holes", "axis-aligned boxy"):
            nonortho += c
    print(f"  NON-ORTHOGONAL          {nonortho:>6}  ({100*nonortho/len(all_ids):.1f}%)")

    # sanity: no leakage into val/test, no overlap with existing renders
    val_test = set(official["validation"]) | set(official["test"])
    leak = [o for o in all_ids if o in val_test or o in excluded]
    assert not leak, f"LEAKAGE: {leak[:5]}"

    lines = [str(args.cadquery_dir / f"{oid}.py") for oid in all_ids]
    missing = [l for l in lines if not Path(l).exists()]
    if missing:
        print(f"WARNING: {len(missing)} scripts missing on disk; dropping them")
        lines = [l for l in lines if Path(l).exists()]
    args.manifest_out.write_text("\n".join(lines) + "\n")
    args.ids_out.write_text(json.dumps({"train_expansion": all_ids}, indent=2))
    print(f"\nmanifest -> {args.manifest_out}  ({len(lines)} lines)")
    print(f"ids      -> {args.ids_out}")


if __name__ == "__main__":
    main()
