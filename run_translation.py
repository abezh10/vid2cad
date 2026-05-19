"""Translate the DeepCAD/Fusion JSON dataset into CADQuery Python files.

Outputs:
  - deepcad-data/cadquery_py/<bucket>/<id>.py  -- one file per successful design
  - deepcad-data/cadquery_py/manifest.json     -- success/failure log
  - deepcad-data/cadquery_py/translated_split.json -- official split filtered to successes
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from translator.translate import translate_design, TranslationError
from translator.validate import validate_file_subprocess


REPO_ROOT = Path(__file__).resolve().parent
JSON_ROOT = REPO_ROOT / "deepcad-data" / "cad_json"
OUT_ROOT = REPO_ROOT / "deepcad-data" / "cadquery_py"
SPLIT_FILE = REPO_ROOT / "deepcad-data" / "train_val_test_split.json"
MANIFEST_FILE = OUT_ROOT / "manifest.json"
SPLIT_OUT = OUT_ROOT / "translated_split.json"


def design_id_from_path(path: Path) -> str:
    """e.g. /.../cad_json/0013/00137104.json -> 0013/00137104"""
    return f"{path.parent.name}/{path.stem}"


def output_path_for(design_id: str) -> Path:
    return OUT_ROOT / f"{design_id}.py"


def process_one(json_path_str: str, validate: bool) -> dict:
    """Translate one JSON file and (optionally) validate the output.

    Returns a manifest record: {id, status, reason?, out_path?}.
    """
    json_path = Path(json_path_str)
    design_id = design_id_from_path(json_path)
    out_path = output_path_for(design_id)
    record = {"id": design_id}
    try:
        with open(json_path) as fp:
            data = json.load(fp)
    except Exception as e:  # noqa: BLE001
        record["status"] = "failed"
        record["reason"] = f"json load: {type(e).__name__}: {e}"
        return record

    try:
        code = translate_design(data)
    except TranslationError as e:
        record["status"] = "failed"
        record["reason"] = f"translate: {e}"
        return record
    except Exception as e:  # noqa: BLE001
        record["status"] = "failed"
        record["reason"] = f"translate-exc: {type(e).__name__}: {e}"
        return record

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to the final location, then validate in a subprocess. On failure, remove.
    with open(out_path, "w") as fp:
        fp.write(code)

    if validate:
        ok, msg = validate_file_subprocess(str(out_path))
        if not ok:
            try:
                out_path.unlink()
            except OSError:
                pass
            record["status"] = "failed"
            record["reason"] = f"validate: {msg}"
            return record

    record["status"] = "success"
    record["out_path"] = str(out_path.relative_to(REPO_ROOT))
    return record


def find_jsons(root: Path, limit: int | None = None) -> list[str]:
    paths = sorted(str(p) for p in root.rglob("*.json"))
    return paths[:limit] if limit else paths


_UUID_RE = re.compile(r"['\"]F[A-Za-z0-9_]{10,}['\"]")


def _bucket_reason(reason: str) -> str:
    """Collapse a failure message into a coarse bucket for stats.

    Strips entity UUIDs and noisy traceback fragments so similar failures group.
    """
    s = _UUID_RE.sub("<uuid>", reason)
    # First two segments separated by ":" (e.g. "translate: extrude <uuid>: empty profiles")
    parts = [p.strip() for p in s.split(":", 3)]
    if len(parts) >= 3:
        return f"{parts[0]}: {parts[2]}"
    return s[:80]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip executing generated code (faster but only translation-level failures caught)."
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N designs (debug).")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="If set, process only these design ids (e.g. 0000/00000007).")
    parser.add_argument("--serial", action="store_true",
                        help="Run serially in one process (useful for debugging).")
    parser.add_argument("--shard", default=None,
                        help="Process only shard 'M/N' (1-indexed) of the JSON list. "
                             "Use to split work across nodes; merge with merge_manifests.py.")
    parser.add_argument("--manifest-out", default=None,
                        help="Override output path for manifest.json (useful for sharded runs).")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.ids:
        json_paths = [str(JSON_ROOT / f"{i}.json") for i in args.ids]
    else:
        json_paths = find_jsons(JSON_ROOT, limit=args.limit)

    if args.shard:
        try:
            m_str, n_str = args.shard.split("/", 1)
            m, n = int(m_str), int(n_str)
            assert 1 <= m <= n
        except Exception:
            print(f"--shard must be 'M/N' with 1<=M<=N, got {args.shard!r}", file=sys.stderr)
            return 2
        json_paths = [p for i, p in enumerate(json_paths) if i % n == (m - 1)]
        print(f"Shard {m}/{n}: {len(json_paths)} designs")

    if not json_paths:
        print(f"No JSON files found under {JSON_ROOT}", file=sys.stderr)
        return 1

    print(f"Translating {len(json_paths)} designs (workers={args.workers}, validate={not args.no_validate})")

    records: list[dict] = []
    if args.serial or args.workers <= 1:
        for p in json_paths:
            records.append(process_one(p, validate=not args.no_validate))
    else:
        import time
        total = len(json_paths)
        report_every = max(1, total // 200)  # ~0.5% increments
        start = time.time()
        last_report = start
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, p, not args.no_validate): p for p in json_paths}
            print(f"Submitted {total} tasks, waiting for results...", flush=True)
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "id": design_id_from_path(Path(futures[fut])),
                        "status": "failed",
                        "reason": f"worker-exc: {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
                    }
                records.append(rec)
                now = time.time()
                if i % report_every == 0 or now - last_report >= 30:
                    elapsed = now - start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (total - i) / rate if rate > 0 else float("inf")
                    print(
                        f"  [{i}/{total}] {100*i/total:5.1f}%  "
                        f"rate={rate:.1f}/s  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                        flush=True,
                    )
                    last_report = now

    successes = [r for r in records if r["status"] == "success"]
    failures = [r for r in records if r["status"] != "success"]
    fail_counter: Counter[str] = Counter()
    for r in failures:
        fail_counter[_bucket_reason(r.get("reason", "unknown"))] += 1

    manifest = {
        "stats": {
            "total": len(records),
            "successful": len(successes),
            "failed": len(failures),
            "by_failure_bucket": dict(fail_counter.most_common()),
        },
        "successful": sorted(r["id"] for r in successes),
        "failed": {r["id"]: r.get("reason", "?") for r in failures},
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else MANIFEST_FILE
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Success: {len(successes)} / {len(records)} ({100*len(successes)/max(1,len(records)):.1f}%)")
    if fail_counter:
        print("Failure buckets:")
        for k, v in fail_counter.most_common(10):
            print(f"  {v:5d}  {k}")

    # Filter the official split file by translated successes (if present)
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as fp:
            split = json.load(fp)
        success_set = {r["id"] for r in successes}
        filtered = {k: [x for x in v if x in success_set] for k, v in split.items()}
        SPLIT_OUT.write_text(json.dumps(filtered, indent=2))
        print(f"Wrote filtered split: {SPLIT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
