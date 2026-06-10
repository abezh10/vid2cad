#!/usr/bin/env python3
"""Stratify test-set IoU by GROUND-TRUTH geometry type.

Goal: quantify *where* the model loses IoU — specifically the hypothesis that
non-orthogonal geometry (diagonal/triangular edges, arcs/splines, tilted
workplanes, tiny features) scores worse than axis-aligned box/cylinder parts.

For every test object we parse its GT CadQuery script (the same DSL the model is
trained to emit) and extract structural geometry features, then join against the
per-object IoU from one or more geom_per_object.jsonl eval dirs.

Pure-text/regex parsing (no CadQuery exec needed) so it runs instantly on CPU.

Usage:
  python3 stage3/diag_geometry.py \
      --test stage2b/test.jsonl \
      --eval stage3_runs/qwen_vit_r64/eval:r64 \
      --eval stage3_runs/qwen/eval:baseline
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-6

# --- regexes over the translator's emitted DSL -----------------------------
RE_NORMAL = re.compile(r"normal=cq\.Vector\(([^)]*)\)")
RE_XDIR = re.compile(r"xDir=cq\.Vector\(([^)]*)\)")
RE_MOVETO = re.compile(r"\.moveTo\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)")
RE_LINETO = re.compile(r"\.lineTo\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)")
RE_CIRCLE = re.compile(r"\.circle\(\s*([-\d.eE]+)\s*\)")
RE_EXTRUDE = re.compile(r"\.extrude\(\s*([-\d.eE]+)")
# curved / angled sketch primitives beyond straight axis-aligned lines
RE_CURVE = re.compile(
    r"\.(threePointArc|radiusArc|sagittaArc|tangentArcPoint|tangentArc|"
    r"spline|ellipse|ellipseArc|polarLine|bezier)\(")
RE_POLYGON = re.compile(r"\.polygon\(\s*(\d+)")
RE_FILLET = re.compile(r"\.fillet\(")
RE_CHAMFER = re.compile(r"\.chamfer\(")


def _vec(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    return tuple(parts[:3]) if len(parts) >= 3 else (0.0, 0.0, 0.0)


def _axis_aligned(v: tuple[float, float, float]) -> bool:
    """True if v points (almost) exactly along ±X/±Y/±Z."""
    near0 = sum(1 for c in v if abs(c) < 1e-4)
    near1 = sum(1 for c in v if abs(abs(c) - 1.0) < 1e-4)
    return near0 == 2 and near1 == 1


def analyze_script(text: str) -> dict:
    """Extract structural geometry features from a GT CadQuery script."""
    f = {
        "has_angled_line": False,   # a polyline segment that moves in x AND y
        "has_curve": False,         # arc / spline / ellipse / polygon(n)
        "has_tilted_wp": False,     # workplane normal (or xDir) off the axes
        "has_circle": False,
        "has_fillet_chamfer": False,
        "n_solids": text.count(".extrude("),
        "n_lineto": len(RE_LINETO.findall(text)),
        "n_circle": len(RE_CIRCLE.findall(text)),
        "min_feature": None,        # smallest |circle radius| or |extrude depth|
    }

    if RE_CURVE.search(text):
        f["has_curve"] = True
    for m in RE_POLYGON.finditer(text):
        # polygon(3) / polygon(5)+ are non-rectangular; polygon(4) ~ rect
        if int(m.group(1)) != 4:
            f["has_curve"] = True
    if RE_FILLET.search(text) or RE_CHAMFER.search(text):
        f["has_fillet_chamfer"] = True
    if RE_CIRCLE.search(text):
        f["has_circle"] = True

    # tilted workplanes: any normal or xDir not on an axis
    for rgx in (RE_NORMAL, RE_XDIR):
        for m in rgx.finditer(text):
            if not _axis_aligned(_vec(m.group(1))):
                f["has_tilted_wp"] = True
                break

    # angled polyline segments: walk moveTo/lineTo per workplane block. We scan
    # the whole script as one stream of (move|line) ops; a new moveTo resets the
    # pen. A segment is "angled" if it changes both x and y by > eps.
    ops = []
    for m in re.finditer(r"\.(moveTo|lineTo)\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)", text):
        ops.append((m.group(1), float(m.group(2)), float(m.group(3))))
    pen = None
    for kind, x, y in ops:
        if kind == "moveTo":
            pen = (x, y)
        else:  # lineTo
            if pen is not None:
                dx, dy = abs(x - pen[0]), abs(y - pen[1])
                if dx > 1e-4 and dy > 1e-4:
                    f["has_angled_line"] = True
            pen = (x, y)

    # smallest feature dimension (proxy for "small feature")
    dims = [abs(float(x)) for x in RE_CIRCLE.findall(text)]
    dims += [abs(float(x)) for x in RE_EXTRUDE.findall(text)]
    dims = [d for d in dims if d > EPS]
    if dims:
        f["min_feature"] = min(dims)

    # derived: non-orthogonal if it has any angled/curved/tilted element
    f["non_orthogonal"] = (
        f["has_angled_line"] or f["has_curve"] or f["has_tilted_wp"])

    # primary bucket (priority order, mutually exclusive)
    if f["has_tilted_wp"]:
        f["bucket"] = "tilted_wp"
    elif f["has_curve"]:
        f["bucket"] = "curved (arc/spline)"
    elif f["has_angled_line"]:
        f["bucket"] = "angled (diag edges)"
    elif f["has_circle"]:
        f["bucket"] = "axis circles/holes"
    else:
        f["bucket"] = "axis-aligned boxy"
    return f


def load_iou(eval_dir: Path) -> dict[str, dict]:
    out = {}
    p = eval_dir / "geom_per_object.jsonl"
    for line in open(p):
        r = json.loads(line)
        out[r["object_id"]] = r
    return out


def fmt_stratum(name, rows, iou_maps, eval_names):
    """One table row: count + per-eval mean/median IoU."""
    n = len(rows)
    cells = [f"{name:<26} n={n:<4}"]
    for ev in eval_names:
        im = iou_maps[ev]
        ious = [im[o]["iou"] for o in rows if o in im]
        execd = [im[o].get("executed", False) for o in rows if o in im]
        if ious:
            mean = np.mean(ious)
            med = np.median(ious)
            exr = np.mean(execd)
            cells.append(f"{ev}: mIoU={mean:.3f} med={med:.3f} exec={exr:.0%}")
        else:
            cells.append(f"{ev}: (no data)")
    return "  ".join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="stage2b/test.jsonl", type=Path)
    ap.add_argument("--eval", action="append", required=True,
                    help="path:label, e.g. stage3_runs/qwen_vit_r64/eval:r64")
    ap.add_argument("--dump", type=Path, default=None,
                    help="optional: write per-object features+iou jsonl here")
    args = ap.parse_args()

    eval_dirs = {}
    eval_names = []
    for spec in args.eval:
        path, _, label = spec.partition(":")
        label = label or Path(path).parent.name
        eval_dirs[label] = Path(path)
        eval_names.append(label)
    iou_maps = {lab: load_iou(d) for lab, d in eval_dirs.items()}

    # analyze every test object's GT script
    feats = {}
    for line in open(args.test):
        row = json.loads(line)
        oid = row["object_id"]
        text = Path(row["script_path"]).read_text()
        feats[oid] = analyze_script(text)

    all_ids = list(feats)
    print(f"Analyzed {len(all_ids)} test objects.\n")

    # ---- buckets (mutually exclusive primary category) ----
    by_bucket = defaultdict(list)
    for oid, f in feats.items():
        by_bucket[f["bucket"]].append(oid)

    order = ["axis-aligned boxy", "axis circles/holes", "angled (diag edges)",
             "curved (arc/spline)", "tilted_wp"]
    print("=" * 100)
    print("PRIMARY GEOMETRY BUCKET (mutually exclusive, priority: tilted>curved>angled>circle>boxy)")
    print("=" * 100)
    for b in order:
        if by_bucket[b]:
            print(fmt_stratum(b, by_bucket[b], iou_maps, eval_names))

    # ---- binary flags (overlapping) ----
    print("\n" + "=" * 100)
    print("BINARY FEATURE FLAGS (overlapping)")
    print("=" * 100)
    flags = ["non_orthogonal", "has_angled_line", "has_curve", "has_tilted_wp",
             "has_circle", "has_fillet_chamfer"]
    for fl in flags:
        ids = [o for o, f in feats.items() if f[fl]]
        if ids:
            print(fmt_stratum(fl, ids, iou_maps, eval_names))
    # orthogonal complement
    ortho = [o for o, f in feats.items() if not f["non_orthogonal"]]
    print(fmt_stratum("ORTHOGONAL (complement)", ortho, iou_maps, eval_names))

    # ---- #solids ----
    print("\n" + "=" * 100)
    print("NUMBER OF SOLIDS (extrude count)")
    print("=" * 100)
    for lo, hi, lab in [(1, 1, "1 solid"), (2, 2, "2 solids"),
                        (3, 4, "3-4 solids"), (5, 99, "5+ solids")]:
        ids = [o for o, f in feats.items() if lo <= f["n_solids"] <= hi]
        if ids:
            print(fmt_stratum(lab, ids, iou_maps, eval_names))

    # ---- min feature size quartiles ----
    print("\n" + "=" * 100)
    print("SMALLEST FEATURE SIZE (min circle radius / extrude depth) — quartiles")
    print("=" * 100)
    sized = [(o, f["min_feature"]) for o, f in feats.items()
             if f["min_feature"] is not None]
    if sized:
        vals = sorted(v for _, v in sized)
        qs = np.quantile(vals, [0.25, 0.5, 0.75])
        edges = [(-1, qs[0], "Q1 smallest"), (qs[0], qs[1], "Q2"),
                 (qs[1], qs[2], "Q3"), (qs[2], 1e18, "Q4 largest")]
        for lo, hi, lab in edges:
            ids = [o for o, v in sized if lo < v <= hi]
            if ids:
                print(fmt_stratum(f"{lab} (<= {hi:.4g})", ids, iou_maps, eval_names))

    if args.dump:
        with open(args.dump, "w") as f:
            for oid, ft in feats.items():
                rec = {"object_id": oid, **ft}
                for ev in eval_names:
                    rec[f"iou_{ev}"] = iou_maps[ev].get(oid, {}).get("iou")
                f.write(json.dumps(rec) + "\n")
        print(f"\nWrote per-object dump -> {args.dump}")


if __name__ == "__main__":
    main()
