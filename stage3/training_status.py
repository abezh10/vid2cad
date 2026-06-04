#!/usr/bin/env python3
"""Snapshot of all Stage 3 training runs.

For each model: reads the latest log, extracts the last loss line, the
current step, and projects ETA from step throughput.

Usage:
  python3 stage3/training_status.py
"""

import argparse
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


STEP_RE = re.compile(r"'loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)")
ALT_STEP_RE = re.compile(r"\{'loss':\s*([\d.]+),\s*'grad_norm':\s*([\d.e+-]+),\s*'learning_rate':\s*([\d.e+-]+),\s*'epoch':\s*([\d.]+)\}")
PROGRESS_RE = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(\S+)<(\S+),\s*(\S+)\]")


def slurm_state(job_id: str) -> dict:
    try:
        out = subprocess.check_output(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%M|%R"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            # Job finished, look at sacct
            out = subprocess.check_output(
                ["sacct", "-j", job_id, "-n", "-X", "-P",
                 "--format=State,Elapsed,Reason"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                parts = out.split("|")
                return {"state": parts[0], "elapsed": parts[1] if len(parts) > 1 else "?", "reason": "ended"}
            return {"state": "UNKNOWN", "elapsed": "?", "reason": "?"}
        state, elapsed, reason = out.split("|")
        return {"state": state, "elapsed": elapsed, "reason": reason}
    except Exception as e:
        return {"state": "?", "elapsed": "?", "reason": str(e)}


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"step": None, "total": None, "loss": None, "epoch": None, "eta": None}

    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return {"step": None, "total": None, "loss": None, "epoch": None, "eta": None}

    last_loss = None
    last_epoch = None
    for m in ALT_STEP_RE.finditer(text):
        last_loss = float(m.group(1))
        last_epoch = float(m.group(4))

    last_pct = last_step = total = elapsed_str = eta_str = it_str = None
    for m in PROGRESS_RE.finditer(text):
        last_pct = int(m.group(1))
        last_step = int(m.group(2))
        total = int(m.group(3))
        elapsed_str = m.group(4)
        eta_str = m.group(5)
        it_str = m.group(6)

    return {
        "step": last_step,
        "total": total,
        "loss": last_loss,
        "epoch": last_epoch,
        "eta": eta_str,
        "elapsed": elapsed_str,
        "iter_rate": it_str,
    }


def find_job_id(model: str, logs_dir: Path) -> str | None:
    logs = sorted(logs_dir.glob(f"{model}_*.out"))
    if not logs:
        return None
    return logs[-1].stem.split("_", 1)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=Path, default=Path("logs/stage3"))
    parser.add_argument("--models", nargs="+",
                        default=["kimi", "qwen", "glm", "internvl"])
    args = parser.parse_args()

    print(f"=== Stage 3 training status @ {datetime.now():%H:%M:%S} ===\n")
    header = f"{'model':<10} {'jobid':<10} {'state':<10} {'step':>14} {'epoch':>6} {'loss':>9} {'rate':>10} {'eta':>10}"
    print(header)
    print("-" * len(header))

    for model in args.models:
        job_id = find_job_id(model, args.logs_dir)
        if job_id is None:
            print(f"{model:<10} {'(none)':<10}")
            continue

        log_out = args.logs_dir / f"{model}_{job_id}.out"
        log_err = args.logs_dir / f"{model}_{job_id}.err"
        parsed = parse_log(log_out)
        if parsed["step"] is None:
            parsed = parse_log(log_err)

        sl = slurm_state(job_id)
        step_str = (f"{parsed['step']}/{parsed['total']}"
                    if parsed["step"] is not None else "—")
        epoch_str = f"{parsed['epoch']:.2f}" if parsed["epoch"] is not None else "—"
        loss_str = f"{parsed['loss']:.4f}" if parsed["loss"] is not None else "—"
        rate_str = parsed["iter_rate"] or "—"
        eta_str = parsed["eta"] or "—"

        print(f"{model:<10} {job_id:<10} {sl['state']:<10} {step_str:>14} "
              f"{epoch_str:>6} {loss_str:>9} {rate_str:>10} {eta_str:>10}")


if __name__ == "__main__":
    main()
