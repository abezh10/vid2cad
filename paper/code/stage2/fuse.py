"""Depth map loading, masking, back-projection, and multi-view fusion."""

import json
import logging
from pathlib import Path

import Imath
import numpy as np
import OpenEXR

SCENE_UNIT_TO_MM = 1000.0

logger = logging.getLogger(__name__)


def read_exr_channel(path: Path, width: int, height: int) -> np.ndarray:
    exr = OpenEXR.InputFile(str(path))
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(exr.header()["channels"].keys())
    ch_name = "R" if "R" in channels else channels[0]
    buf = exr.channel(ch_name, pt)
    return np.frombuffer(buf, dtype=np.float32).reshape(height, width)


def load_cameras(cameras_path: Path) -> list[dict]:
    with open(cameras_path) as f:
        return json.load(f)


def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path) as f:
        return json.load(f)


def backproject_frame(
    depth: np.ndarray,
    mask: np.ndarray,
    cam: dict,
    max_depth: float,
) -> np.ndarray:
    """Back-project masked depth pixels to world-space 3D points.

    All inputs/outputs are in scene units. Caller converts to mm.
    """
    h, w = depth.shape
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]

    valid = (mask > 0.8) & (depth > 1e-6) & (depth < max_depth)
    if valid.sum() < 10:
        return np.empty((0, 3), dtype=np.float32)

    vs, us = np.where(valid)
    ds = depth[vs, us]

    x_cam = (us - cx) * ds / fx
    y_cam = (vs - cy) * ds / fy
    z_cam = ds

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)

    w2c = np.array(cam["world_to_camera"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)

    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points
    quantized = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(idx)]


def fuse_depth_maps(
    obj_dir: Path,
    cameras: list[dict],
    metadata: dict,
    obj_logger: logging.Logger,
) -> np.ndarray:
    """Fuse all depth maps into a single point cloud in mm.

    Returns Nx3 float32 array in millimetres.
    """
    width = cameras[0]["width"]
    height = cameras[0]["height"]
    num_frames = metadata["num_frames"]
    orbit_radius = metadata["orbit_radius_mm"]
    longest_dim = metadata["longest_dim_mm"]
    max_depth = orbit_radius * 5.0

    depth_dir = obj_dir / "depth"
    mask_dir = obj_dir / "mask"

    all_points = []
    skipped = 0

    for cam in cameras:
        frame = cam["frame"]
        depth_path = depth_dir / f"frame_{frame:04d}.exr"
        mask_path = mask_dir / f"frame_{frame:04d}.exr"

        if not depth_path.exists() or not mask_path.exists():
            obj_logger.warning("Missing frame %d (depth=%s, mask=%s)",
                               frame, depth_path.exists(), mask_path.exists())
            skipped += 1
            continue

        depth = read_exr_channel(depth_path, width, height)
        mask = read_exr_channel(mask_path, width, height)

        pts = backproject_frame(depth, mask, cam, max_depth)
        if len(pts) < 10:
            skipped += 1
            continue

        all_points.append(pts)

    if skipped > 0:
        obj_logger.info("Skipped %d / %d frames", skipped, num_frames)

    if not all_points:
        obj_logger.error("No valid frames produced any points")
        return np.empty((0, 3), dtype=np.float32)

    raw = np.concatenate(all_points, axis=0)
    obj_logger.info("Back-projected %d raw points from %d frames",
                    len(raw), num_frames - skipped)

    raw_mm = raw * SCENE_UNIT_TO_MM

    voxel_size = longest_dim * SCENE_UNIT_TO_MM / 256.0
    deduped = voxel_downsample(raw_mm, voxel_size)
    obj_logger.info("Voxel downsample (%.2f mm): %d → %d points",
                    voxel_size, len(raw_mm), len(deduped))

    median = np.median(deduped, axis=0)
    mad = np.median(np.abs(deduped - median), axis=0)
    mad = np.maximum(mad, 1e-6)
    inlier = np.all(np.abs(deduped - median) < 10.0 * mad, axis=1)
    if inlier.sum() > 100:
        removed = len(deduped) - inlier.sum()
        if removed > 0:
            obj_logger.info("Outlier filter (6×MAD): removed %d points", removed)
        deduped = deduped[inlier]

    bbox_min = deduped.min(axis=0)
    bbox_max = deduped.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    stl_center_mm = np.array([
        (metadata["bbox_min_mm"][i] + metadata["bbox_max_mm"][i]) / 2.0
        for i in range(3)
    ]) * SCENE_UNIT_TO_MM

    dist = np.linalg.norm(bbox_center - stl_center_mm)
    obj_logger.info("Point cloud center vs STL center distance: %.2f mm", dist)

    return deduped
