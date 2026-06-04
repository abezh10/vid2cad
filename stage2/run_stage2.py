#!/usr/bin/env python3
"""Stage 2: Depth maps + camera poses → normalized point cloud."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def setup_logging(log_path: Path, object_id: str) -> logging.Logger:
    obj_logger = logging.getLogger(f"stage2.{object_id.replace('/', '_')}")
    obj_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    obj_logger.addHandler(fh)
    obj_logger.addHandler(ch)
    return obj_logger


def is_complete(obj_dir: Path) -> bool:
    return (
        (obj_dir / "point_cloud_norm_8192.ply").exists()
        and (obj_dir / "point_cloud_norm_256.ply").exists()
        and (obj_dir / "stage2_metadata.json").exists()
    )


def stage1_complete(obj_dir: Path, num_frames: int = 60) -> bool:
    if not (obj_dir / "cameras.json").exists():
        return False
    if not (obj_dir / "metadata.json").exists():
        return False
    depth_dir = obj_dir / "depth"
    mask_dir = obj_dir / "mask"
    if not depth_dir.exists() or not mask_dir.exists():
        return False
    if len(list(depth_dir.glob("frame_*.exr"))) < num_frames:
        return False
    return len(list(mask_dir.glob("frame_*.exr"))) >= num_frames


def compare_bbox(points_mm: np.ndarray, obj_dir: Path,
                 metadata: dict, obj_logger: logging.Logger) -> dict:
    pc_min = points_mm.min(axis=0)
    pc_max = points_mm.max(axis=0)
    pc_dims = pc_max - pc_min

    from stage2.fuse import SCENE_UNIT_TO_MM
    stl_dims = np.array(metadata["bbox_dims_mm"]) * SCENE_UNIT_TO_MM

    diff = np.abs(pc_dims - stl_dims)
    rel_diff = diff / (stl_dims + 1e-6)
    max_rel = float(rel_diff.max())

    if max_rel > 0.05:
        obj_logger.warning(
            "Bbox mismatch: PC dims=[%.1f, %.1f, %.1f] vs STL dims=[%.1f, %.1f, %.1f] (max rel diff %.1f%%)",
            *pc_dims, *stl_dims, max_rel * 100)
    else:
        obj_logger.info("Bbox match: max relative diff %.1f%%", max_rel * 100)

    return {
        "point_cloud_dims_mm": pc_dims.tolist(),
        "stl_dims_mm": stl_dims.tolist(),
        "max_relative_diff": max_rel,
        "match": max_rel <= 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2: depth + cameras → point cloud")
    parser.add_argument("--object_id", required=True,
                        help="Object ID relative to data_dir, e.g. 0000/00000007")
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Root Stage 1 output directory")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if outputs exist")
    args = parser.parse_args()

    obj_dir = args.data_dir / args.object_id
    if not obj_dir.exists():
        print(f"ERROR: Object directory not found: {obj_dir}", file=sys.stderr)
        sys.exit(1)

    obj_logger = setup_logging(obj_dir / "stage2.log", args.object_id)
    obj_logger.info("Starting Stage 2 for %s", args.object_id)

    if not args.force and is_complete(obj_dir):
        obj_logger.info("All outputs present — skipping (use --force to reprocess)")
        return

    if not stage1_complete(obj_dir):
        obj_logger.error("Stage 1 outputs incomplete for %s", args.object_id)
        sys.exit(2)

    from stage2.fuse import load_cameras, load_metadata, fuse_depth_maps
    from stage2.normalize import downsample_and_save

    cameras = load_cameras(obj_dir / "cameras.json")
    metadata = load_metadata(obj_dir / "metadata.json")

    points_mm = fuse_depth_maps(obj_dir, cameras, metadata, obj_logger)
    if len(points_mm) < 50:
        obj_logger.error("Fusion produced only %d points — aborting", len(points_mm))
        sys.exit(3)

    bbox_result = compare_bbox(points_mm, obj_dir, metadata, obj_logger)

    norm_meta = downsample_and_save(points_mm, obj_dir, obj_logger)

    stage2_meta = {
        "object_id": args.object_id,
        "bbox_comparison": bbox_result,
        **norm_meta,
    }
    (obj_dir / "stage2_metadata.json").write_text(json.dumps(stage2_meta, indent=2))

    if not is_complete(obj_dir):
        obj_logger.error("Stage 2 finished but expected outputs are missing")
        sys.exit(4)

    obj_logger.info("Stage 2 complete for %s", args.object_id)


if __name__ == "__main__":
    main()
