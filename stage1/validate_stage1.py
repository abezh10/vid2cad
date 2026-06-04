#!/usr/bin/env python3
"""Scan Stage 1 outputs, report success/failure counts, save stage1_summary.json."""

import argparse
import json
import statistics
from pathlib import Path


def check_object(obj_dir: Path, num_frames: int) -> dict:
    result = {
        "object_id": str(obj_dir.relative_to(obj_dir.parent.parent)),
        "success": False,
        "missing": [],
        "depth_frames_found": 0,
        "mask_frames_found": 0,
    }

    if not (obj_dir / "video.mp4").exists():
        result["missing"].append("video.mp4")
    if not (obj_dir / "cameras.json").exists():
        result["missing"].append("cameras.json")
    if not (obj_dir / "metadata.json").exists():
        result["missing"].append("metadata.json")

    depth_dir = obj_dir / "depth"
    if not depth_dir.exists():
        result["missing"].append("depth/")
    else:
        found = len(list(depth_dir.glob("frame_*.exr")))
        result["depth_frames_found"] = found
        if found < num_frames:
            result["missing"].append(f"depth/frame_*.exr ({found}/{num_frames})")

    mask_dir = obj_dir / "mask"
    if not mask_dir.exists():
        result["missing"].append("mask/")
    else:
        found = len(list(mask_dir.glob("frame_*.exr")))
        result["mask_frames_found"] = found
        if found < num_frames:
            result["missing"].append(f"mask/frame_*.exr ({found}/{num_frames})")

    result["success"] = len(result["missing"]) == 0

    if result["success"]:
        try:
            meta = json.loads((obj_dir / "metadata.json").read_text())
            result["longest_dim_mm"] = meta.get("longest_dim_mm")
            result["bbox_dims_mm"]   = meta.get("bbox_dims_mm")
        except Exception:
            pass

    return result


def discover_object_dirs(output_dir: Path) -> list:
    """Find leaf directories that look like rendered object outputs."""
    dirs = []
    for bucket in sorted(output_dir.iterdir()):
        if not bucket.is_dir() or bucket.name.startswith("."):
            continue
        for obj in sorted(bucket.iterdir()):
            if obj.is_dir():
                dirs.append(obj)
    if not dirs:
        # Flat layout fallback
        dirs = [d for d in sorted(output_dir.iterdir())
                if d.is_dir() and not d.name.startswith(".")]
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 1 outputs")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--summary_out", type=Path,
                        help="Where to write stage1_summary.json (default: output_dir)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-object failure details")
    args = parser.parse_args()

    obj_dirs = discover_object_dirs(args.output_dir)
    print(f"Scanning {len(obj_dirs)} object directories …")

    success, failure = 0, 0
    failure_reasons: dict = {}
    longest_dims: list = []
    all_results = []

    for obj_dir in obj_dirs:
        r = check_object(obj_dir, args.num_frames)
        all_results.append(r)
        if r["success"]:
            success += 1
            if r.get("longest_dim_mm") is not None:
                longest_dims.append(r["longest_dim_mm"])
        else:
            failure += 1
            for m in r["missing"]:
                key = m.split("(")[0].strip()
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
            if args.verbose:
                print(f"  FAIL {r['object_id']}: {r['missing']}")

    sep = "=" * 52
    print(f"\n{sep}")
    print("Stage 1 Validation Summary")
    print(sep)
    print(f"  Total:    {len(obj_dirs)}")
    print(f"  Success:  {success}")
    print(f"  Failure:  {failure}")

    if failure_reasons:
        print("\n  Failure breakdown:")
        for reason, cnt in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")

    mean_dim = std_dim = None
    if longest_dims:
        mean_dim = statistics.mean(longest_dims)
        std_dim  = statistics.stdev(longest_dims) if len(longest_dims) > 1 else 0.0
        print(f"\n  Bounding-box longest dimension (mm) across {len(longest_dims)} objects:")
        print(f"    mean = {mean_dim:.2f}")
        print(f"    std  = {std_dim:.2f}")
        print(f"    min  = {min(longest_dims):.2f}")
        print(f"    max  = {max(longest_dims):.2f}")

    summary = {
        "total":   len(obj_dirs),
        "success": success,
        "failure": failure,
        "failure_reasons": failure_reasons,
    }
    if mean_dim is not None:
        summary["longest_dim_mm_n"]    = len(longest_dims)
        summary["longest_dim_mm_mean"] = round(mean_dim, 4)
        summary["longest_dim_mm_std"]  = round(std_dim,  4)
        summary["longest_dim_mm_min"]  = round(min(longest_dims), 4)
        summary["longest_dim_mm_max"]  = round(max(longest_dims), 4)

    out_path = (args.summary_out or args.output_dir) / "stage1_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary written to {out_path}")


if __name__ == "__main__":
    main()
