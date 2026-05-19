"""Validate generated CADQuery .py files in batch.

Strategy: spawn N batch validator subprocesses (one per worker).
Feed each a queue of paths via stdin so the cadquery import cost is amortized.
If a worker crashes (segfault), restart it; treat the last-sent path as failed.

Outputs:
  - Updates manifest.json: sets validation status, removes invalid .py files
  - Re-writes translated_split.json filtered by validation success
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import queue
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUT_ROOT = REPO_ROOT / "deepcad-data" / "cadquery_py"
MANIFEST_FILE = OUT_ROOT / "manifest.json"
SPLIT_FILE = REPO_ROOT / "deepcad-data" / "train_val_test_split.json"
SPLIT_OUT = OUT_ROOT / "translated_split.json"
VALIDATOR = REPO_ROOT / "translator" / "_validate_subprocess.py"


def design_id_from_path(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


class Validator:
    """Manages a single batch validator subprocess; restarts on crash."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(VALIDATOR), "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
        )

    def validate(self, path: str, timeout_s: float = 180.0) -> tuple[bool, str]:
        """Send one path and read the response. Restart on crash/timeout."""
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self.proc.stdin.write(path + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._start()
            return False, "validator stdin broken"

        # Read with timeout via thread
        result_q: queue.Queue[str] = queue.Queue()

        def reader():
            try:
                line = self.proc.stdout.readline()
                result_q.put(line if line else "")
            except Exception as e:  # noqa: BLE001
                result_q.put("")

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            line = result_q.get(timeout=timeout_s)
        except queue.Empty:
            try:
                self.proc.kill()
            except Exception:
                pass
            self._start()
            return False, "timeout"

        if not line:
            self._start()
            return False, "validator died (likely segfault)"
        # Format: "<path>\tOK" or "<path>\tERR: msg"
        try:
            recv_path, status = line.rstrip("\n").split("\t", 1)
        except ValueError:
            return False, f"bad protocol: {line!r}"
        if status == "OK":
            return True, "ok"
        if status.startswith("ERR: "):
            return False, status[5:]
        return False, status

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def worker_thread(worker_id: int, in_q: queue.Queue, out_q: queue.Queue, timeout_s: float):
    v = Validator(worker_id)
    try:
        while True:
            item = in_q.get()
            if item is None:
                in_q.task_done()
                break
            path = item
            ok, msg = v.validate(path, timeout_s=timeout_s)
            out_q.put((path, ok, msg))
            in_q.task_done()
    finally:
        v.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-failed-files", action="store_true",
                        help="Don't delete .py files that fail validation.")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Per-file validation timeout in seconds.")
    parser.add_argument("--shard", default=None,
                        help="Validate only shard 'M/N' (1-indexed) of the manifest's "
                             "successes. Combine across nodes with merge_manifests.py.")
    parser.add_argument("--manifest-in", default=None,
                        help="Read input manifest from this path instead of the default.")
    parser.add_argument("--manifest-out", default=None,
                        help="Write output manifest to this path (useful for sharded runs).")
    parser.add_argument("--retry-only", choices=["timeout", "any"], default=None,
                        help="If set, only re-validate previously-failed entries of this kind. "
                             "'timeout' targets the timeout long-tail; 'any' retries all failures.")
    args = parser.parse_args()

    manifest_in = Path(args.manifest_in) if args.manifest_in else MANIFEST_FILE
    if not manifest_in.exists():
        print(f"ERROR: {manifest_in} not found. Run translation first.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_in.read_text())

    if args.retry_only:
        candidates = []
        for design_id, reason in manifest.get("failed", {}).items():
            if args.retry_only == "any" or "timeout" in reason:
                candidates.append(design_id)
        paths = [str(OUT_ROOT / f"{i}.py") for i in candidates]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        successful_ids = manifest.get("successful", [])
        paths = [str(OUT_ROOT / f"{i}.py") for i in successful_ids]
        paths = [p for p in paths if os.path.exists(p)]

    if args.shard:
        try:
            m_str, n_str = args.shard.split("/", 1)
            m, n = int(m_str), int(n_str)
            assert 1 <= m <= n
        except Exception:
            print(f"--shard must be 'M/N' with 1<=M<=N, got {args.shard!r}", file=sys.stderr)
            return 2
        paths = [p for i, p in enumerate(paths) if i % n == (m - 1)]
        print(f"Shard {m}/{n}: {len(paths)} files")
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("No translated .py files to validate.")
        return 0

    print(f"Validating {len(paths)} designs across {args.workers} workers...")
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()

    threads = []
    for wid in range(args.workers):
        t = threading.Thread(target=worker_thread, args=(wid, in_q, out_q, args.timeout), daemon=True)
        t.start()
        threads.append(t)

    for p in paths:
        in_q.put(p)
    for _ in range(args.workers):
        in_q.put(None)

    results: dict[str, tuple[bool, str]] = {}
    fail_counter: Counter[str] = Counter()
    t0 = time.time()
    log_every = max(1, len(paths) // 50)
    for i in range(len(paths)):
        path, ok, msg = out_q.get()
        results[path] = (ok, msg)
        if not ok:
            bucket = msg.split(":", 1)[0]
            fail_counter[bucket] += 1
        if (i + 1) % log_every == 0 or i + 1 == len(paths):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(paths) - i - 1) / rate if rate else 0
            print(f"  [{i+1}/{len(paths)}] rate={rate:.1f}/s ETA={eta:.0f}s", flush=True)

    for t in threads:
        t.join(timeout=2)

    # Write back manifest with validation status. Preserve un-validated entries as successes.
    failures: dict[str, str] = dict(manifest.get("failed", {}))
    successes_set = set(manifest.get("successful", []))
    for path, (ok, msg) in results.items():
        design_id = design_id_from_path(Path(path))
        if ok:
            successes_set.add(design_id)
            failures.pop(design_id, None)
        else:
            successes_set.discard(design_id)
            failures[design_id] = f"validate: {msg}"
            if not args.keep_failed_files:
                try:
                    os.remove(path)
                except OSError:
                    pass

    successes = sorted(successes_set)
    by_bucket: Counter[str] = Counter()
    for reason in failures.values():
        bucket = reason.split(":", 2)[1].strip() if ":" in reason else reason
        bucket = bucket.split(":", 1)[0].strip()
        by_bucket[bucket] += 1

    manifest = {
        "stats": {
            "total": len(successes) + len(failures),
            "successful": len(successes),
            "failed": len(failures),
            "by_failure_bucket": dict(by_bucket.most_common()),
        },
        "successful": successes,
        "failed": failures,
    }
    manifest_out = Path(args.manifest_out) if args.manifest_out else MANIFEST_FILE
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Serializing manifest ({len(successes) + len(failures)} records)...", flush=True)
    manifest_out.write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"\nValidation complete. Manifest: {manifest_out}", flush=True)
    print(f"  Validated successes: {len(successes)}")
    print(f"  Validation failures: {sum(1 for _, (ok, _) in results.items() if not ok)}")
    print(f"  Top buckets: {fail_counter.most_common(8)}")

    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text())
        success_set = set(successes)
        filtered = {k: [x for x in v if x in success_set] for k, v in split.items()}
        SPLIT_OUT.write_text(json.dumps(filtered, separators=(",", ":")))
        print(f"  Filtered split: {SPLIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
