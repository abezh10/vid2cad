"""Point cloud normalization, downsampling, and PLY I/O."""

import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)


def farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    n = len(points)
    if n <= k:
        return np.arange(n)

    indices = np.zeros(k, dtype=np.int64)
    dists = np.full(n, np.inf, dtype=np.float64)
    indices[0] = np.random.randint(n)

    for i in range(1, k):
        last = points[indices[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        indices[i] = np.argmax(dists)

    return indices


def upsample_to(points: np.ndarray, target: int, sigma: float = 0.001,
                obj_logger: logging.Logger | None = None) -> np.ndarray:
    n = len(points)
    if n >= target:
        return points
    if obj_logger:
        obj_logger.warning("Upsampling %d → %d points (σ=%.4f)", n, target, sigma)
    deficit = target - n
    idx = np.random.choice(n, size=deficit, replace=True)
    jitter = np.random.normal(0, sigma, size=(deficit, 3)).astype(np.float32)
    extra = points[idx] + jitter
    return np.concatenate([points, extra], axis=0)


def normalize_point_cloud(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center at bbox midpoint and scale to fit in [-0.5, 0.5]³.

    Returns (normalized_points, centroid_mm, scale_factor).
    """
    bbox_min = points_mm.min(axis=0)
    bbox_max = points_mm.max(axis=0)
    centroid = (bbox_min + bbox_max) / 2.0
    centered = points_mm - centroid
    extent = bbox_max - bbox_min
    scale = extent.max()
    if scale < 1e-8:
        scale = 1.0
    normalized = centered / scale
    return normalized.astype(np.float32), centroid, float(scale)


def save_ply(path: Path, points: np.ndarray) -> None:
    vertex = np.array(
        [(p[0], p[1], p[2]) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(str(path))


def load_ply(path: Path) -> np.ndarray:
    data = PlyData.read(str(path))
    v = data["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)


def downsample_and_save(
    points_mm: np.ndarray,
    obj_dir: Path,
    obj_logger: logging.Logger,
) -> dict:
    """Normalize, downsample to 8192 and 256, save all PLY files.

    Returns metadata dict with normalization params and point counts.
    """
    save_ply(obj_dir / "point_cloud_raw.ply", points_mm)
    obj_logger.info("Saved raw point cloud: %d points", len(points_mm))

    norm_pts, centroid, scale = normalize_point_cloud(points_mm)
    obj_logger.info("Normalized: centroid=[%.2f, %.2f, %.2f] mm, scale=%.2f mm",
                    centroid[0], centroid[1], centroid[2], scale)

    rng_state = np.random.get_state()
    np.random.seed(42)

    upsampled = len(norm_pts) < 8192
    pts_for_fps = upsample_to(norm_pts, 8192, sigma=0.001, obj_logger=obj_logger)

    MAX_FPS_INPUT = 50000
    if len(pts_for_fps) > MAX_FPS_INPUT:
        idx_sub = np.random.choice(len(pts_for_fps), MAX_FPS_INPUT, replace=False)
        pts_for_fps = pts_for_fps[idx_sub]
        obj_logger.info("Pre-subsampled %d → %d for FPS", len(norm_pts), MAX_FPS_INPUT)

    idx_8192 = farthest_point_sample(pts_for_fps, 8192)
    pts_8192 = pts_for_fps[idx_8192]
    pts_8192 = pts_8192[np.lexsort(pts_8192.T[::-1])]
    save_ply(obj_dir / "point_cloud_norm_8192.ply", pts_8192)
    obj_logger.info("Saved 8192-point cloud (range: [%.3f, %.3f])",
                    pts_8192.min(), pts_8192.max())

    idx_256 = farthest_point_sample(pts_8192, 256)
    pts_256 = pts_8192[idx_256]
    pts_256 = pts_256[np.lexsort(pts_256.T[::-1])]
    save_ply(obj_dir / "point_cloud_norm_256.ply", pts_256)
    obj_logger.info("Saved 256-point cloud")

    np.random.set_state(rng_state)

    return {
        "raw_point_count": len(points_mm),
        "upsampled": upsampled,
        "normalization": {
            "centroid_mm": centroid.tolist(),
            "scale_mm": scale,
        },
        "point_cloud_8192_bbox": {
            "min": pts_8192.min(axis=0).tolist(),
            "max": pts_8192.max(axis=0).tolist(),
        },
    }
