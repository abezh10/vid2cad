#!/usr/bin/env python3
"""Diagnostic: how much of the IoU gap is pose (orientation) vs genuine shape?

The geom eval normalizes (center + uniform scale) but does NOT align rotation, so
a correctly-reconstructed part built in a different sketch frame than the GT can
score a low IoU purely from orientation mismatch. This script re-scores an
existing per_object.jsonl two ways, on the SAME fixed occupancy points:

  iou_base    : identity rotation (matches the production geom eval)
  iou_aligned : max IoU over the 24 proper cube rotations (axis swaps/flips)

A large (aligned - base) gap => much of the ~0.5 mIoU ceiling is orientation, and
the win is teaching canonical frame + reporting aligned IoU. A small gap => the
ceiling is genuine geometry, so prioritise 3D conditioning / RL.

CPU-only; no GPU and no re-generation. Reuses build_mesh / normalize from
eval_geom so meshes match the real eval exactly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from eval_geom import build_mesh, normalize, extract_code

N_OCC = 8192
# One fixed occupancy point set, shared across every object and every rotation,
# so iou_base and iou_aligned differ only by the rotation — not by sampling noise.
# All meshes are normalized into the same [-0.5, 0.5] cube before scoring.
_PTS = (np.random.RandomState(0).rand(N_OCC, 3) - 0.5) * 1.1


def cube_rotations() -> list[np.ndarray]:
    """The 24 proper rotations of the cube (signed axis permutations, det=+1)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                mats.append(M)
    return mats


_ROTS = cube_rotations()  # 24


def _iou(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    ia = a.contains(_PTS)
    ib = b.contains(_PTS)
    u = np.logical_or(ia, ib).sum()
    return float(np.logical_and(ia, ib).sum() / u) if u else 0.0


def _rotate(m: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    T = np.eye(4)
    T[:3, :3] = R
    out = m.copy()
    out.apply_transform(T)
    return out


def process(task) -> dict:
    row, data_dir = task
    oid = row["object_id"]
    out = {"object_id": oid, "iou_base": 0.0, "iou_aligned": 0.0,
           "identity_best": True, "note": ""}

    if not row.get("executed", False):
        out["note"] = "gen_failed"
        return out
    gt_path = Path(data_dir) / oid / "object.stl"
    if not gt_path.exists():
        out["note"] = "no_gt"
        return out

    code = row.get("code") or extract_code(row["generated"])
    gen = build_mesh(code)
    if gen is None:
        out["note"] = "mesh_build_failed"
        return out
    try:
        gt_n = normalize(trimesh.load(gt_path, force="mesh"))
    except Exception:
        out["note"] = "gt_load_failed"
        return out

    try:
        base = _iou(normalize(gen), gt_n)
        best = base
        best_is_identity = True
        for R in _ROTS:
            if np.allclose(R, np.eye(3)):
                continue
            iou_r = _iou(normalize(_rotate(gen, R)), gt_n)
            if iou_r > best:
                best = iou_r
                best_is_identity = False
        out["iou_base"] = base
        out["iou_aligned"] = best
        out["identity_best"] = best_is_identity
        out["note"] = "ok"
    except Exception as e:
        out["note"] = f"metric_error:{type(e).__name__}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="stage3_runs/qwen/eval", type=Path)
    ap.add_argument("--data_dir", default="stage1_output", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_dir / "per_object.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    work = [(r, str(args.data_dir)) for r in rows]

    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process, work, chunksize=2)):
            results.append(res)
            if (i + 1) % 50 == 0:
                ok = [r for r in results if r["note"] == "ok"]
                if ok:
                    b = np.mean([r["iou_base"] for r in ok])
                    a = np.mean([r["iou_aligned"] for r in ok])
                    print(f"  {i+1}/{len(work)}  scored={len(ok)}  "
                          f"base={b:.3f} aligned={a:.3f} (+{a-b:.3f})", flush=True)

    ok = [r for r in results if r["note"] == "ok"]
    base = np.array([r["iou_base"] for r in ok]) if ok else np.array([0.0])
    algn = np.array([r["iou_aligned"] for r in ok]) if ok else np.array([0.0])
    n_rot_helps = sum(1 for r in ok if not r["identity_best"])

    summary = {
        "eval_dir": str(args.eval_dir),
        "n_objects": len(results),
        "n_scored": len(ok),
        "iou_base_mean": float(base.mean()),
        "iou_aligned_mean": float(algn.mean()),
        "iou_gain_from_alignment": float(algn.mean() - base.mean()),
        "iou_base_median": float(np.median(base)),
        "iou_aligned_median": float(np.median(algn)),
        "frac_rotation_helps": n_rot_helps / len(ok) if ok else 0.0,
        # objects the rotation lifted by a lot => clear "right shape, wrong frame"
        "n_gain_over_0.2": int((algn - base > 0.2).sum()),
        "interpretation": "large gain => ceiling is pose; small gain => ceiling is shape",
    }
    (args.eval_dir / "diag_rotation_summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.eval_dir / "diag_rotation_per_object.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
