#!/usr/bin/env python3
"""Emit a Stage-1 render manifest for the FROZEN 994 test objects at high res.

The original 640x480 test renders stay untouched (all paper numbers remain
comparable). This re-renders the same objects into a SEPARATE root
(stage1_test_hr/) at the new resolution for an optional high-res eval:
same checkpoint, same parts — only input resolution changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", default="stage2b/test.jsonl", type=Path)
    ap.add_argument("--cadquery_dir", default="deepcad-data/cadquery_py", type=Path)
    ap.add_argument("--manifest_out", default="stage1_output/test_hr_manifest.txt", type=Path)
    args = ap.parse_args()

    lines = []
    for line in open(args.test_manifest):
        oid = json.loads(line)["object_id"]
        p = args.cadquery_dir / f"{oid}.py"
        if p.exists():
            lines.append(str(p))
    args.manifest_out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} test objects -> {args.manifest_out}")


if __name__ == "__main__":
    main()
