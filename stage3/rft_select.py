#!/usr/bin/env python3
"""RFT step 2 — score candidates by IoU vs GT and build the curated round.

Reads the candidate shards from rft_sample.py, rebuilds a mesh for each
candidate, scores volumetric IoU against the ground-truth STL (the same reward
eval_geom.py uses), and keeps the single highest-IoU candidate per object as the
new supervised target. Objects whose best candidate fails to execute / build, or
falls below --min_iou, are dropped (STaR: train only on what the model can solve).

Emits a self-contained round directory that train_<model>.py can consume:
  <round_dir>/
    selected_code/<bucket>/<id>.py   curated targets (best candidate per object)
    train.jsonl                      curated manifest (object_id + script_path)
    validation.jsonl, test.jsonl     copied from the base manifest (stay GT)
    config.yaml                      base config w/ manifest_dir+output_root=round_dir
    select_summary.json              kept/dropped counts, IoU distribution

CPU-only; parallelized with a process pool. build_mesh / normalize /
volumetric_iou are imported from eval_geom so the reward matches the final eval.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from eval_geom import build_mesh, normalize, volumetric_iou


def score_object(task) -> dict:
    """Pick the highest-IoU candidate for one object."""
    oid, candidates, data_dir = task
    res = {"object_id": oid, "best_iou": 0.0, "best_code": None,
           "n_candidates": len(candidates), "n_executed": 0, "note": ""}

    gt_path = Path(data_dir) / oid / "object.stl"
    if not gt_path.exists():
        res["note"] = "no_gt"
        return res
    try:
        gt_n = normalize(trimesh.load(gt_path, force="mesh"))
    except Exception:
        res["note"] = "gt_load_failed"
        return res

    for cand in candidates:
        code = cand.get("code") or ""
        if not code.strip():
            continue
        mesh = build_mesh(code)
        if mesh is None:
            continue
        res["n_executed"] += 1
        try:
            iou = volumetric_iou(normalize(mesh), gt_n)
        except Exception:
            continue
        if iou > res["best_iou"]:
            res["best_iou"] = iou
            res["best_code"] = code

    res["note"] = "ok" if res["best_code"] is not None else "no_valid_candidate"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_dir", required=True, type=Path,
                    help="dir of candidates_shard_*.jsonl from rft_sample.py")
    ap.add_argument("--round_dir", required=True, type=Path,
                    help="output round dir (curated manifest + config land here)")
    ap.add_argument("--base_config", default="stage3/config.yaml")
    ap.add_argument("--data_dir", default="stage1_output", type=Path,
                    help="GT meshes: <data_dir>/<oid>/object.stl")
    ap.add_argument("--min_iou", type=float, default=0.5,
                    help="drop objects whose best candidate IoU < this")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Merge all shards; if an object appears twice (overlapping reruns), keep the
    # entry with more candidates.
    by_oid: dict[str, list] = {}
    shard_files = sorted(args.candidates_dir.glob("candidates_shard_*.jsonl"))
    if not shard_files:
        sys.exit(f"no candidate shards in {args.candidates_dir}")
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                rec = json.loads(line)
                oid, cands = rec["object_id"], rec["candidates"]
                if oid not in by_oid or len(cands) > len(by_oid[oid]):
                    by_oid[oid] = cands
    items = list(by_oid.items())
    if args.limit:
        items = items[:args.limit]
    print(f"loaded {len(items)} objects from {len(shard_files)} shards", flush=True)

    work = [(oid, cands, str(args.data_dir)) for oid, cands in items]
    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(score_object, work, chunksize=2)):
            results.append(res)
            if (i + 1) % 50 == 0:
                kept = [r for r in results if r["best_code"] and r["best_iou"] >= args.min_iou]
                miou = np.mean([r["best_iou"] for r in kept]) if kept else 0.0
                print(f"  {i+1}/{len(work)}  kept={len(kept)}  mean_best_iou={miou:.3f}",
                      flush=True)

    # Build curated round directory.
    args.round_dir.mkdir(parents=True, exist_ok=True)
    code_root = args.round_dir / "selected_code"
    kept_rows = []
    for r in results:
        if not r["best_code"] or r["best_iou"] < args.min_iou:
            continue
        oid = r["object_id"]                 # "<bucket>/<id>"
        code_path = code_root / f"{oid}.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(r["best_code"])
        kept_rows.append({"object_id": oid, "script_path": str(code_path),
                          "split": "train", "rft_iou": r["best_iou"]})

    (args.round_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in kept_rows))

    # Validation/test stay ground-truth: copy from the base manifest dir.
    base_cfg = yaml.safe_load(open(args.base_config))
    base_manifest = Path(base_cfg["manifest_dir"])
    for split in ("validation.jsonl", "test.jsonl"):
        src = base_manifest / split
        if src.exists():
            shutil.copy(src, args.round_dir / split)

    # Round config: identical to base except data + output point at round_dir.
    round_cfg = dict(base_cfg)
    round_cfg["manifest_dir"] = str(args.round_dir)
    round_cfg["output_root"] = str(args.round_dir)
    (args.round_dir / "config.yaml").write_text(yaml.safe_dump(round_cfg, sort_keys=False))

    ious = np.array([r["best_iou"] for r in results])
    kept_ious = np.array([row["rft_iou"] for row in kept_rows]) if kept_rows else np.array([0.0])
    summary = {
        "candidates_dir": str(args.candidates_dir),
        "round_dir": str(args.round_dir),
        "min_iou": args.min_iou,
        "n_objects": len(results),
        "n_kept": len(kept_rows),
        "n_no_valid_candidate": sum(1 for r in results if r["note"] == "no_valid_candidate"),
        "n_no_gt": sum(1 for r in results if r["note"] == "no_gt"),
        "best_iou_mean_all": float(ious.mean()),
        "best_iou_mean_kept": float(kept_ious.mean()),
        "best_iou_median_kept": float(np.median(kept_ious)),
        "kept_iou_at_0.7": int((kept_ious >= 0.7).sum()),
        "kept_iou_at_0.9": int((kept_ious >= 0.9).sum()),
    }
    (args.round_dir / "select_summary.json").write_text(json.dumps(summary, indent=2))
    with open(args.round_dir / "select_per_object.jsonl", "w") as f:
        for r in results:
            r2 = {k: v for k, v in r.items() if k != "best_code"}
            f.write(json.dumps(r2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nCurated round at {args.round_dir}")
    print(f"  train.jsonl: {len(kept_rows)} objects")
    print(f"  next: train_<model>.py --config {args.round_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
