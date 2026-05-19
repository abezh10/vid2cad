"""Merge multiple shard manifests into a single canonical manifest + filtered split.

Usage:
    python3 merge_manifests.py shard_manifest_*.json \
        --out deepcad-data/cadquery_py/manifest.json
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SPLIT_FILE = REPO_ROOT / "deepcad-data" / "train_val_test_split.json"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "deepcad-data" / "cadquery_py" / "manifest.json"
DEFAULT_SPLIT_OUT = REPO_ROOT / "deepcad-data" / "cadquery_py" / "translated_split.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", help="Shard manifest paths")
    parser.add_argument("--out", default=str(DEFAULT_MANIFEST_OUT))
    parser.add_argument("--split-out", default=str(DEFAULT_SPLIT_OUT))
    args = parser.parse_args()

    successes: set[str] = set()
    failures: dict[str, str] = {}
    for mp in args.manifests:
        m = json.loads(Path(mp).read_text())
        for sid in m.get("successful", []):
            successes.add(sid)
            failures.pop(sid, None)
        for fid, reason in m.get("failed", {}).items():
            if fid in successes:
                continue  # another shard succeeded
            failures[fid] = reason

    by_bucket: Counter[str] = Counter()
    for reason in failures.values():
        bucket = reason.split(":", 2)[1].strip() if ":" in reason else reason
        bucket = bucket.split(":", 1)[0].strip()
        by_bucket[bucket] += 1

    merged = {
        "stats": {
            "total": len(successes) + len(failures),
            "successful": len(successes),
            "failed": len(failures),
            "by_failure_bucket": dict(by_bucket.most_common()),
        },
        "successful": sorted(successes),
        "failed": failures,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2))
    print(f"Merged {len(args.manifests)} shards -> {out}")
    print(f"  successful: {len(successes)}")
    print(f"  failed:     {len(failures)}")

    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text())
        filtered = {k: [x for x in v if x in successes] for k, v in split.items()}
        Path(args.split_out).write_text(json.dumps(filtered, indent=2))
        print(f"Filtered split: {args.split_out}")
        for k, v in filtered.items():
            print(f"  {k}: {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
