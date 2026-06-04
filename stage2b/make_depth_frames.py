#!/usr/bin/env python3
"""Render Stage-1 depth EXRs as colormapped JPGs aligned to the sampled RGB frames.

The RGB frames in stage2b/frames/<oid>/frame_XX.jpg (XX=00..15) were sampled from
the 60-frame orbit video at indices np.linspace(0, 59, 16).round(). The Stage-1
depth EXRs live at stage1_output/<oid>/depth/frame_NNNN.exr (NNNN=0001..0060,
1-based) with a matching mask at stage1_output/<oid>/mask/frame_NNNN.exr. Depth is
linear Z in scene units, replicated across R/G/B; mask is 1.0 on the CAD object.

For each of the 16 RGB indices we read the corresponding depth+mask EXR, keep only
foreground (mask > 0.5), normalize depth with per-object 2nd/98th percentiles for a
consistent color scale across viewpoints, apply TURBO, and paint background black.
Output: stage2b/depth_frames/<oid>/frame_XX.jpg, index-aligned to the RGB frames.

Idempotent (skips objects whose depth-frame dir already has N frames).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
import Imath

N_FRAMES = 16
VIDEO_FRAMES = 60
# 0-based video indices the RGB jpgs were sampled at -> 1-based EXR file numbers.
_RGB_IDX = np.linspace(0, VIDEO_FRAMES - 1, N_FRAMES).round().astype(int)
_EXR_NUM = [int(i) + 1 for i in _RGB_IDX]
_FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)


def _read_exr_r(path: Path) -> np.ndarray:
    f = OpenEXR.InputFile(str(path))
    h = f.header()
    dw = h["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    raw = f.channel("R", _FLOAT)
    return np.frombuffer(raw, dtype=np.float32).reshape(H, W)


def make_object(args) -> tuple[str, str, str | None]:
    oid, stage1_root, out_root, quality = args
    depth_dir = stage1_root / oid / "depth"
    mask_dir = stage1_root / oid / "mask"
    out_dir = out_root / oid

    if out_dir.exists() and len(list(out_dir.glob("frame_*.jpg"))) == N_FRAMES:
        return oid, "skip", None
    if not depth_dir.exists() or not mask_dir.exists():
        return oid, "fail", "missing depth/mask dir"

    # First pass: load masked foreground depths to fix a per-object color scale.
    depths, masks = [], []
    fg_vals = []
    for num in _EXR_NUM:
        dp = depth_dir / f"frame_{num:04d}.exr"
        mp_ = mask_dir / f"frame_{num:04d}.exr"
        if not dp.exists() or not mp_.exists():
            return oid, "fail", f"missing frame {num:04d}"
        d = _read_exr_r(dp)
        m = _read_exr_r(mp_) > 0.5
        depths.append(d)
        masks.append(m)
        if m.any():
            fg_vals.append(d[m])

    if not fg_vals:
        return oid, "fail", "no foreground in any frame"
    allfg = np.concatenate(fg_vals)
    lo, hi = np.percentile(allfg, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-6

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (d, m) in enumerate(zip(depths, masks)):
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)
        rgb = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        rgb[~m] = 0  # background -> black
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.jpg"),
                    rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return oid, "ok", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSONL manifest (rows with object_id)")
    ap.add_argument("--stage1_root", type=Path,
                    default=Path("/scratch/users/abzhong/stage1_output"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/scratch/users/abzhong/stage2b/depth_frames"))
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = [json.loads(line) for line in f]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Depth frames for {len(rows)} objects -> {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = [(r["object_id"], args.stage1_root, args.out_dir, args.quality)
            for r in rows]

    ok = skip = fail = 0
    fails = []
    if args.serial:
        it = (make_object(w) for w in work)
        for oid, status, err in it:
            ok += status == "ok"; skip += status == "skip"; fail += status == "fail"
            if status == "fail" and len(fails) < 8:
                fails.append((oid, err))
    else:
        with mp.Pool(args.workers) as pool:
            for i, (oid, status, err) in enumerate(
                    pool.imap_unordered(make_object, work, chunksize=8)):
                ok += status == "ok"; skip += status == "skip"; fail += status == "fail"
                if status == "fail" and len(fails) < 8:
                    fails.append((oid, err))
                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(work)}  ok={ok} skip={skip} fail={fail}")

    print(f"\nDone: ok={ok}  skip={skip}  fail={fail}")
    for oid, err in fails:
        print(f"  FAIL {oid}: {err}")


if __name__ == "__main__":
    main()
