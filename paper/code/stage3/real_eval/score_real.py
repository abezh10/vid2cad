#!/usr/bin/env python3
"""Workstream B1 — geometric scoring for the REAL eval, with pose alignment.

Unlike the synthetic eval (where the model learned the sim canonical frame, so
identity rotation is valid — see diag_rotation.py: aligning only adds +0.03), a
real printed part is filmed at an ARBITRARY orientation. So we must align gen->GT
before scoring or IoU is meaningless. We align in two stages:

  1. coarse: best of the 24 proper cube rotations (axis swaps/flips)
  2. refine: ICP from that coarse init (rigid; continuous rotation/translation)

and report the aligned IoU + chamfer as the real geometric fidelity, alongside
the identity (unaligned) numbers for reference. Reuses build_mesh / normalize /
volumetric_iou / chamfer from eval_geom and the 24 rotations from diag_rotation,
so the reward matches the rest of the pipeline.

Input : <eval_dir>/per_object.jsonl  (from eval.py on the real round)
Output: <eval_dir>/real_geom_per_object.jsonl + real_geom_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # stage3/
from eval_geom import build_mesh, normalize, volumetric_iou, chamfer, extract_code
from diag_rotation import cube_rotations

_ROTS = cube_rotations()       # 24
N_ICP_PTS = 2048


def _rotate(m: trimesh.Trimesh, R: np.ndarray) -> trimesh.Trimesh:
    T = np.eye(4)
    T[:3, :3] = R
    out = m.copy()
    out.apply_transform(T)
    return out


def _icp_refine(gen_n: trimesh.Trimesh, gt_n: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """Rigid ICP from the current pose; returns the transformed gen or None."""
    try:
        src = gen_n.sample(N_ICP_PTS)
        tgt = gt_n.sample(N_ICP_PTS)
        matrix, _, _ = trimesh.registration.icp(
            src, tgt, initial=np.eye(4), max_iterations=30, scale=False)
        out = gen_n.copy()
        out.apply_transform(matrix)
        return out
    except Exception:
        return None


def process(task) -> dict:
    row, data_dir = task
    oid = row["object_id"]
    out = {"object_id": oid, "executed": row.get("executed", False),
           "iou_identity": 0.0, "iou_aligned": 0.0,
           "chamfer_aligned": None, "note": ""}

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
        identity = volumetric_iou(normalize(gen), gt_n)
        # coarse: best cube rotation
        best_iou, best_mesh = identity, normalize(gen)
        for R in _ROTS:
            cand = normalize(_rotate(gen, R))
            iou_r = volumetric_iou(cand, gt_n)
            if iou_r > best_iou:
                best_iou, best_mesh = iou_r, cand
        # refine: ICP from the best coarse pose
        refined = _icp_refine(best_mesh, gt_n)
        if refined is not None:
            rn = normalize(refined)
            iou_icp = volumetric_iou(rn, gt_n)
            if iou_icp > best_iou:
                best_iou, best_mesh = iou_icp, rn

        out["iou_identity"] = identity
        out["iou_aligned"] = best_iou
        out["chamfer_aligned"] = chamfer(best_mesh, gt_n)
        out["note"] = "ok"
    except Exception as e:
        out["note"] = f"metric_error:{type(e).__name__}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, type=Path,
                    help="dir with per_object.jsonl from eval.py on the real round")
    ap.add_argument("--data_dir", default="stage1_output", type=Path,
                    help="GT meshes: <data_dir>/<oid>/object.stl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_dir / "per_object.jsonl")]
    work = [(r, str(args.data_dir)) for r in rows]
    results = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(process, work, chunksize=1):
            results.append(r)

    ok = [r for r in results if r["note"] == "ok"]
    ident = np.array([r["iou_identity"] for r in ok]) if ok else np.array([0.0])
    algn = np.array([r["iou_aligned"] for r in ok]) if ok else np.array([0.0])
    chs = np.array([r["chamfer_aligned"] for r in ok if r["chamfer_aligned"] is not None])
    n = len(results)
    n_exec = sum(1 for r in results if r["executed"])

    summary = {
        "eval_dir": str(args.eval_dir),
        "n_objects": n,
        "execution_rate": n_exec / n if n else 0.0,
        "n_scored": len(ok),
        "iou_identity_mean": float(ident.mean()),
        "iou_aligned_mean": float(algn.mean()),
        "iou_aligned_median": float(np.median(algn)),
        "iou_aligned_at_0.5": int((algn >= 0.5).sum()),
        "iou_aligned_at_0.7": int((algn >= 0.7).sum()),
        "chamfer_aligned_median": float(np.median(chs)) if len(chs) else None,
        "note": "iou_aligned (cube-rot + ICP) is the real metric; identity is unaligned reference",
    }
    (args.eval_dir / "real_geom_summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.eval_dir / "real_geom_per_object.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
