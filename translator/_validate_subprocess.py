"""Subprocess validator. Two modes:

Single-file mode:
    python _validate_subprocess.py <path>
    -> prints OK or ERR: <msg>

Batch mode:
    python _validate_subprocess.py --batch
    -> reads paths line-by-line on stdin; for each, prints "<path>\\t<status>" on stdout.
       status = "OK" or "ERR: <msg>"

Batch mode amortizes the cadquery import cost across many files. A crash in one
file (segfault) terminates the batch, so callers should restart and skip the
last submitted path.
"""
import sys
import os
import resource
import signal
import io
from contextlib import redirect_stdout, redirect_stderr


def _set_limits():
    # Soft memory cap to avoid OOM on bad geometry (4 GB).
    try:
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    except (ValueError, OSError):
        pass


def _validate_path(path: str, cq) -> tuple[bool, str]:
    try:
        with open(path) as fp:
            code = fp.read()
    except Exception as e:  # noqa: BLE001
        return False, f"read failed: {e}"
    ns: dict = {}
    # Suppress any prints from generated code so they don't pollute our protocol
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(compile(code, path, "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return False, f"exec: {type(e).__name__}: {str(e)[:200]}"
    result = ns.get("result")
    if result is None:
        return False, "result is None"
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            val = result.val()
            if val is None:
                return False, "no underlying solid"
            bb = val.BoundingBox()
            if bb.DiagonalLength <= 0:
                return False, "degenerate bounding box"
    except Exception as e:  # noqa: BLE001
        return False, f"postcheck: {type(e).__name__}: {str(e)[:200]}"
    return True, "ok"


def main_single(path: str) -> int:
    _set_limits()
    try:
        import cadquery as cq
    except Exception as e:  # noqa: BLE001
        print(f"ERR: cadquery import: {e}")
        return 1
    ok, msg = _validate_path(path, cq)
    print("OK" if ok else f"ERR: {msg}")
    return 0 if ok else 1


def main_batch() -> int:
    _set_limits()
    try:
        import cadquery as cq
    except Exception as e:  # noqa: BLE001
        # Emit a single error line and exit so caller can recover
        sys.stdout.write(f"\tERR: cadquery import: {e}\n")
        sys.stdout.flush()
        return 1
    for line in sys.stdin:
        path = line.rstrip("\n")
        if not path:
            continue
        ok, msg = _validate_path(path, cq)
        sys.stdout.write(f"{path}\t{'OK' if ok else 'ERR: ' + msg}\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--batch":
        raise SystemExit(main_batch())
    if len(sys.argv) == 2:
        raise SystemExit(main_single(sys.argv[1]))
    print("ERR: usage", file=sys.stderr)
    raise SystemExit(2)
