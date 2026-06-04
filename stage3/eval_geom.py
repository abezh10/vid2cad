#!/usr/bin/env python3
"""Geometric fidelity eval: Chamfer distance + volumetric IoU.

Reads the per_object.jsonl produced by eval.py, rebuilds a mesh from each
generated CadQuery script, and compares it to the ground-truth mesh
(stage1_output/<id>/object.stl) after normalizing both to a unit cube.

Metrics (both scale-invariant, computed on the normalized meshes):
  - chamfer : mean bidirectional surface point distance (lower = better)
  - iou     : volumetric IoU via sampled occupancy   (higher = better)

Failed/truncated generations count as iou=0, chamfer=None.

Output: geom_per_object.jsonl + geom_summary.json next to the input.
CPU-only; parallelized with a process pool.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

N_SURFACE = 4096      # surface points for Chamfer
N_OCC = 8192          # query points for volumetric IoU
BUILD_TIMEOUT = 45


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find('"""Generated'), text.find("import cadquery"))
              if i != -1]
    if starts:
        text = text[min(starts):]
    return text.strip()


def build_mesh(code: str) -> trimesh.Trimesh | None:
    """Run a CadQuery script in a subprocess, export `result` to STL, load it."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stl = td / "out.stl"
        runner = code + (
            "\n\nimport cadquery as _cq\n"
            f"_cq.exporters.export(result, {str(stl)!r})\n"
        )
        (td / "gen.py").write_text(runner)
        try:
            r = subprocess.run([sys.executable, str(td / "gen.py")],
                               capture_output=True, timeout=BUILD_TIMEOUT, cwd=td)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not stl.exists():
            return None
        try:
            m = trimesh.load(stl, force="mesh")
        except Exception:
            return None
        if m.is_empty or len(m.vertices) == 0:
            return None
        return m


def normalize(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center on bbox midpoint and uniformly scale longest axis to 1."""
    m = m.copy()
    lo, hi = m.bounds
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    if extent <= 0:
        return m
    m.apply_translation(-center)
    m.apply_scale(1.0 / extent)
    return m


def chamfer(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    pa = a.sample(N_SURFACE)
    pb = b.sample(N_SURFACE)
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    return float(da.mean() + db.mean())


def volumetric_iou(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    pts = (np.random.rand(N_OCC, 3) - 0.5) * 1.1  # cover normalized bounds + margin
    ia = a.contains(pts)
    ib = b.contains(pts)
    union = np.logical_or(ia, ib).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(ia, ib).sum() / union)


def process(args) -> dict:
    row, data_dir = args
    oid = row["object_id"]
    out = {"object_id": oid, "executed": row.get("executed", False),
           "iou": 0.0, "chamfer": None, "note": ""}

    gt_path = Path(data_dir) / oid / "object.stl"
    if not gt_path.exists():
        out["note"] = "no_gt"
        return out

    if not row.get("executed", False):
        out["note"] = "gen_failed"
        return out

    code = row.get("code") or extract_code(row["generated"])
    gen = build_mesh(code)
    if gen is None:
        out["note"] = "mesh_build_failed"
        return out
    try:
        gt = trimesh.load(gt_path, force="mesh")
    except Exception:
        out["note"] = "gt_load_failed"
        return out

    gen_n, gt_n = normalize(gen), normalize(gt)
    try:
        out["chamfer"] = chamfer(gen_n, gt_n)
        out["iou"] = volumetric_iou(gen_n, gt_n)
        out["note"] = "ok"
    except Exception as e:
        out["note"] = f"metric_error:{type(e).__name__}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, type=Path,
                    help="dir containing per_object.jsonl (e.g. stage3_runs/qwen/eval)")
    ap.add_argument("--data_dir", default="stage1_output", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_dir / "per_object.jsonl")]
    if args.limit:
        rows = rows[:args.limit]

    work = [(r, str(args.data_dir)) for r in rows]
    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process, work, chunksize=4)):
            results.append(res)
            if (i + 1) % 50 == 0:
                done = [r for r in results if r["note"] == "ok"]
                miou = np.mean([r["iou"] for r in done]) if done else 0
                print(f"  {i+1}/{len(work)}  scored={len(done)}  mIoU={miou:.3f}",
                      flush=True)

    with open(args.eval_dir / "geom_per_object.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    scored = [r for r in results if r["note"] == "ok"]
    ious = np.array([r["iou"] for r in scored]) if scored else np.array([0.0])
    chs = np.array([r["chamfer"] for r in scored if r["chamfer"] is not None])
    n = len(results)
    # "overall" treats every object (incl. failed gen) as iou=0
    overall_iou = sum(r["iou"] for r in results) / n if n else 0.0

    summary = {
        "eval_dir": str(args.eval_dir),
        "n_objects": n,
        "n_scored": len(scored),
        "iou_mean_scored": float(ious.mean()),
        "iou_median_scored": float(np.median(ious)),
        "iou_mean_overall": float(overall_iou),
        "iou_at_0.5": int((ious >= 0.5).sum()),
        "iou_at_0.7": int((ious >= 0.7).sum()),
        "chamfer_mean": float(chs.mean()) if len(chs) else None,
        "chamfer_median": float(np.median(chs)) if len(chs) else None,
    }
    (args.eval_dir / "geom_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
