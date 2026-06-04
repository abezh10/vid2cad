#!/usr/bin/env python3
"""Validate CadQuery scripts paired in manifests.

Parses each .py with ast, counts characters/lines/tokens (whitespace-split),
and flags syntax errors. Writes scripts_summary.json with stats per split.
"""

import argparse
import ast
import json
import statistics
from pathlib import Path


def stats(label, values):
    if not values:
        return f"{label}: no data"
    return (f"{label}: n={len(values)} mean={statistics.mean(values):.0f} "
            f"median={statistics.median(values):.0f} "
            f"min={min(values)} max={max(values)} "
            f"p95={sorted(values)[int(len(values)*0.95)]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b"))
    parser.add_argument("--out_path", type=Path,
                        default=Path("/scratch/users/abzhong/stage2b/scripts_summary.json"))
    args = parser.parse_args()

    summary = {}

    for split in ["train", "validation", "test"]:
        path = args.manifest_dir / f"{split}.jsonl"
        if not path.exists():
            continue

        with open(path) as f:
            rows = [json.loads(line) for line in f]

        lines_list = []
        chars_list = []
        words_list = []
        syntax_errors = []

        for r in rows:
            try:
                code = Path(r["script_path"]).read_text()
            except Exception as e:
                syntax_errors.append((r["object_id"], f"read error: {e}"))
                continue

            try:
                ast.parse(code)
            except SyntaxError as e:
                syntax_errors.append((r["object_id"], f"syntax error: {e.msg}"))
                continue

            lines_list.append(code.count("\n") + 1)
            chars_list.append(len(code))
            words_list.append(len(code.split()))

        print(f"\n== {split} ({len(rows)} pairs) ==")
        print(f"  syntax errors: {len(syntax_errors)}")
        print(f"  {stats('lines', lines_list)}")
        print(f"  {stats('chars', chars_list)}")
        print(f"  {stats('whitespace tokens', words_list)}")

        summary[split] = {
            "total": len(rows),
            "syntax_errors": len(syntax_errors),
            "syntax_error_ids": [oid for oid, _ in syntax_errors[:10]],
            "lines_mean": round(statistics.mean(lines_list), 1) if lines_list else None,
            "chars_mean": round(statistics.mean(chars_list), 1) if chars_list else None,
            "chars_max": max(chars_list) if chars_list else None,
            "chars_p95": sorted(chars_list)[int(len(chars_list) * 0.95)] if chars_list else None,
        }

    args.out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten {args.out_path}")


if __name__ == "__main__":
    main()
