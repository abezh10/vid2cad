"""Validate generated CADQuery code by executing it in an isolated subprocess.

Running cadquery code in-process risks segfaults that kill the worker pool.
We exec it in a child Python process and read OK/ERR from stdout.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_VALIDATOR = Path(__file__).resolve().parent / "_validate_subprocess.py"


def validate_code_subprocess(code: str, timeout_s: float = 30.0) -> tuple[bool, str]:
    """Run `code` in a child Python process and check it produces a valid result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fp:
        fp.write(code)
        path = fp.name
    try:
        return validate_file_subprocess(path, timeout_s=timeout_s)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def validate_file_subprocess(path: str, timeout_s: float = 30.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, str(_VALIDATOR), path],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode == 0 and out.startswith("OK"):
        return True, "ok"
    if proc.returncode < 0:
        return False, f"crash signal={-proc.returncode}; stdout={out!r}; stderr={err[:200]!r}"
    msg = out if out else (err if err else f"rc={proc.returncode}")
    if msg.startswith("ERR: "):
        msg = msg[5:]
    return False, msg


# Backward-compat alias used elsewhere.
def validate_generated_code(code: str) -> tuple[bool, str]:
    return validate_code_subprocess(code)
