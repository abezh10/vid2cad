#!/usr/bin/env python3
"""Scan Stage 2 outputs, report success/failure counts, save stage2_summary.json."""

import argparse
import json
import statistics
from pathlib import Path


def check_object(obj_dir: Path) -> dict:
    result = {
        "object_id": str(obj_dir.relative_to(obj_dir.parent.parent)),
        "success": False,
        "missing": [],
    }

    for fname in ["point_cloud_norm_8192.ply", "point_cloud_norm_256.ply",
                   "point_cloud_raw.ply", "stage2_metadata.json"]:
        if not (obj_dir / fname).exists():
            result["missing"].append(fname)

    result["success"] = len(result["missing"]) == 0

    if result["success"]:
        try:
            meta = json.loads((obj_dir / "stage2_metadata.json").read_text())
            result["raw_point_count"] = meta.get("raw_point_count")
            result["upsampled"] = meta.get("upsampled", False)
            result["bbox_match"] = meta.get("bbox_comparison", {}).get("match", None)
            result["max_rel_diff"] = meta.get("bbox_comparison", {}).get("max_relative_diff", None)
        except Exception:
            pass

    return result


def discover_stage2_dirs(data_dir: Path) -> list:
    dirs = []
    for bucket in sorted(data_dir.iterdir()):
        if not bucket.is_dir() or bucket.name.startswith("."):
            continue
        for obj in sorted(bucket.iterdir()):
            if obj.is_dir() and (obj / "cameras.json").exists():
                dirs.append(obj)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 2 outputs")
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    obj_dirs = discover_stage2_dirs(args.data_dir)
    print(f"Scanning {len(obj_dirs)} object directories …")

    success = failure = 0
    fallback_count = upsampled_count = 0
    bbox_match_count = 0
    raw_counts: list[int] = []
    rel_diffs: list[float] = []
    failure_reasons: dict = {}

    for obj_dir in obj_dirs:
        r = check_object(obj_dir)
        if r["success"]:
            success += 1
            if r.get("raw_point_count") is not None:
                raw_counts.append(r["raw_point_count"])
            if r.get("upsampled"):
                upsampled_count += 1
            if r.get("bbox_match"):
                bbox_match_count += 1
            if r.get("max_rel_diff") is not None:
                rel_diffs.append(r["max_rel_diff"])
        else:
            failure += 1
            for m in r["missing"]:
                failure_reasons[m] = failure_reasons.get(m, 0) + 1
            if args.verbose:
                print(f"  FAIL {r['object_id']}: {r['missing']}")

    total = len(obj_dirs)
    sep = "=" * 52
    print(f"\n{sep}")
    print("Stage 2 Validation Summary")
    print(sep)
    print(f"  Total objects:   {total}")
    print(f"  Stage 2 done:    {success}")
    print(f"  Stage 2 missing: {failure}")
    print(f"  Upsampled:       {upsampled_count}")
    print(f"  Bbox match:      {bbox_match_count} / {success}")

    if failure_reasons:
        print("\n  Missing files:")
        for reason, cnt in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}")

    summary: dict = {
        "total": total,
        "success": success,
        "failure": failure,
        "failure_reasons": failure_reasons,
        "upsampled_count": upsampled_count,
        "bbox_match_count": bbox_match_count,
    }

    if raw_counts:
        mean_c = statistics.mean(raw_counts)
        std_c = statistics.stdev(raw_counts) if len(raw_counts) > 1 else 0.0
        print(f"\n  Raw point count ({len(raw_counts)} objects):")
        print(f"    mean = {mean_c:.0f}")
        print(f"    std  = {std_c:.0f}")
        print(f"    min  = {min(raw_counts)}")
        print(f"    max  = {max(raw_counts)}")
        summary["raw_count_mean"] = round(mean_c, 1)
        summary["raw_count_std"] = round(std_c, 1)
        summary["raw_count_min"] = min(raw_counts)
        summary["raw_count_max"] = max(raw_counts)

    if rel_diffs:
        mean_d = statistics.mean(rel_diffs)
        print(f"\n  Bbox relative diff ({len(rel_diffs)} objects):")
        print(f"    mean = {mean_d:.1%}")
        print(f"    max  = {max(rel_diffs):.1%}")
        summary["bbox_rel_diff_mean"] = round(mean_d, 4)
        summary["bbox_rel_diff_max"] = round(max(rel_diffs), 4)

    out_path = args.data_dir / "stage2_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary written to {out_path}")


if __name__ == "__main__":
    main()
