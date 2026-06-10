#!/usr/bin/env python3
"""Build a geometry-difficulty-balanced training manifest.

The train set is ~69% orthogonal (axis-aligned boxes + axis circles), which the
model already nails (median IoU ~0.79), and ~31% non-orthogonal (angled edges,
arcs/splines, tilted workplanes) where it collapses (median IoU ~0.16-0.44).
Curved/angled scripts are also ~1.8x longer, so they need more gradient steps to
learn — but the natural distribution starves them.

This duplicates the hard buckets in the manifest so each epoch gives the model
more practice on the geometry it fails, WITHOUT dropping any easy examples
(boxy stays at full count, so it can't regress — "improve all parts"). Duplication
(vs. loss-reweighting) keeps the HF Trainer/sampler untouched and is fully
reproducible.

Bucket priority + detection are shared with diag_geometry.analyze_script.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from diag_geometry import analyze_script

# per-bucket duplication factor (integer). 1 = keep as-is.
DEFAULT_FACTORS = {
    "axis-aligned boxy": 1,
    "axis circles/holes": 1,
    "angled (diag edges)": 2,
    "curved (arc/spline)": 2,
    "tilted_wp": 4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", default="stage2b/train.jsonl", type=Path)
    ap.add_argument("--out_manifest", default="stage2b/train_balanced.jsonl", type=Path)
    ap.add_argument("--factor", action="append", default=[],
                    help="override bucket factor, e.g. 'curved (arc/spline)=3'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    factors = dict(DEFAULT_FACTORS)
    for spec in args.factor:
        k, _, v = spec.rpartition("=")
        factors[k] = int(v)

    rows = [json.loads(l) for l in open(args.in_manifest)]
    out = []
    in_buckets = Counter()
    out_buckets = Counter()
    for row in rows:
        f = analyze_script(Path(row["script_path"]).read_text())
        b = f["bucket"]
        in_buckets[b] += 1
        n = factors.get(b, 1)
        for _ in range(n):
            out.append(row)
        out_buckets[b] += n

    import random
    random.Random(args.seed).shuffle(out)
    with open(args.out_manifest, "w") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")

    order = ["axis-aligned boxy", "axis circles/holes", "angled (diag edges)",
             "curved (arc/spline)", "tilted_wp"]
    n_in, n_out = len(rows), len(out)
    print(f"IN  {args.in_manifest}  n={n_in}")
    print(f"OUT {args.out_manifest}  n={n_out}  (x{n_out/n_in:.2f})")
    print(f"{'bucket':<24} {'in':>6} {'in%':>6} {'fac':>4} {'out':>6} {'out%':>6}")
    nonortho_in = nonortho_out = 0
    for b in order:
        i, o, fac = in_buckets[b], out_buckets[b], factors.get(b, 1)
        print(f"{b:<24} {i:>6} {100*i/n_in:>5.1f}% {fac:>4} {o:>6} {100*o/n_out:>5.1f}%")
        if b not in ("axis-aligned boxy", "axis circles/holes"):
            nonortho_in += i
            nonortho_out += o
    print(f"{'NON-ORTHOGONAL':<24} {nonortho_in:>6} {100*nonortho_in/n_in:>5.1f}% "
          f"{'':>4} {nonortho_out:>6} {100*nonortho_out/n_out:>5.1f}%")


if __name__ == "__main__":
    main()
