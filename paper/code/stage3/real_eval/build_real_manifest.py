#!/usr/bin/env python3
"""Workstream B1 — turn real phone videos of 3D-printed parts into an eval round.

You 3D-print a handful of TEST-set parts (their CadQuery is the ground truth),
shoot a short orbit video of each, then this script extracts frames + emits a
self-contained round dir that the EXISTING eval harness consumes unchanged:

  real_eval/<run>/
    frames/<bucket>/<id>/frame_XX.jpg   uniformly-sampled frames (like stage2b)
    test.jsonl                          object_id + script_path (GT = test CadQuery)
    config.yaml                         base config w/ manifest_dir+frames_dir=<run>

Then:
  python3 stage3/eval.py --model qwen --checkpoint_dir <adapter> \
      --config real_eval/<run>/config.yaml --out_dir real_eval/<run>/eval_qwen
  python3 stage3/real_eval/score_real.py --eval_dir real_eval/<run>/eval_qwen

Mapping each video to its object_id (which printed part it is) — two ways:
  --mapping map.json   {"<video filename or path>": "<bucket>/<id>", ...}
  --mapping map.csv    rows: video,object_id
or rely on the filename convention <bucket>__<id>.<ext>  (double underscore = '/').

Frame extraction mirrors stage2b/sample_frames.py (uniform N frames via OpenCV).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def load_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        return {str(k): v for k, v in json.loads(path.read_text()).items()}
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip().lower() not in ("video", "path"):
                out[row[0].strip()] = row[1].strip()
    return out


def resolve_object_id(video: Path, mapping: dict[str, str]) -> str | None:
    # explicit mapping by full path, then by filename, then by stem
    for key in (str(video), video.name, video.stem):
        if key in mapping:
            return mapping[key]
    # filename convention: <bucket>__<id>.<ext>  ->  <bucket>/<id>
    if "__" in video.stem:
        return video.stem.replace("__", "/")
    return None


def extract_frames(video: Path, out_dir: Path, n_frames: int, quality: int) -> int:
    """Write n_frames uniformly-spaced JPEGs (frame_00.jpg ...). Returns count."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # some phone codecs don't report frame count — decode sequentially
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            return 0
        idxs = [int(i * len(frames) / n_frames) for i in range(n_frames)]
        out_dir.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(idxs):
            cv2.imwrite(str(out_dir / f"frame_{j:02d}.jpg"), frames[i],
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
        return n_frames
    idxs = [int(i * total / n_frames) for i in range(n_frames)]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"frame_{j:02d}.jpg"), fr,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        written += 1
    cap.release()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True, type=Path,
                    help="dir of real orbit videos (one per printed part)")
    ap.add_argument("--run_dir", required=True, type=Path,
                    help="output round dir, e.g. real_eval/run1")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="JSON/CSV mapping video -> object_id (else filename convention)")
    ap.add_argument("--base_config", default="stage3/config.yaml", type=Path)
    ap.add_argument("--test_manifest", default="stage2b/test.jsonl", type=Path,
                    help="source of GT script_path per object_id")
    ap.add_argument("--n_frames", type=int, default=16,
                    help="frames to extract; the dataset subsamples to model n_frames")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    import yaml

    # object_id -> GT script_path, from the frozen test manifest.
    gt = {}
    with open(args.test_manifest) as f:
        for line in f:
            r = json.loads(line)
            gt[r["object_id"]] = r["script_path"]

    mapping = load_mapping(args.mapping)
    videos = sorted(p for p in args.videos_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"no videos found in {args.videos_dir}")

    frames_root = args.run_dir / "frames"
    rows, skipped = [], []
    for v in videos:
        oid = resolve_object_id(v, mapping)
        if oid is None:
            skipped.append((v.name, "no object_id mapping"))
            continue
        if oid not in gt:
            skipped.append((v.name, f"object_id {oid} not in test manifest"))
            continue
        n = extract_frames(v, frames_root / oid, args.n_frames, args.quality)
        if n < args.n_frames:
            skipped.append((v.name, f"only {n}/{args.n_frames} frames extracted"))
            continue
        rows.append({"object_id": oid, "script_path": gt[oid],
                     "split": "test", "frame_dir": str(frames_root / oid),
                     "source_video": str(v)})
        print(f"  {v.name} -> {oid}  ({n} frames)")

    if not rows:
        sys.exit("no usable videos — check --mapping / filenames / object_ids")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "test.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    # Round config: base config but data points at this run dir. eval.py reads
    # manifest_dir/test.jsonl and frames_dir, so both must point here.
    base_cfg = yaml.safe_load(open(args.base_config))
    base_cfg["manifest_dir"] = str(args.run_dir)
    base_cfg["frames_dir"] = str(frames_root)
    (args.run_dir / "config.yaml").write_text(yaml.safe_dump(base_cfg, sort_keys=False))

    summary = {"n_videos": len(videos), "n_usable": len(rows),
               "n_skipped": len(skipped), "skipped": skipped}
    (args.run_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(rows)} usable / {len(videos)} videos. Skipped: {len(skipped)}")
    for name, why in skipped:
        print(f"  SKIP {name}: {why}")
    print(f"\nRound dir: {args.run_dir}")
    print(f"Next: eval.py --config {args.run_dir/'config.yaml'} --out_dir {args.run_dir}/eval_<model>")


if __name__ == "__main__":
    main()
