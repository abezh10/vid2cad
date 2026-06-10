"""Shared dataset loader for video → CadQuery training.

Model-agnostic — produces per-row dicts with:
  object_id : str
  frames    : list[PIL.Image]            (N frames, RGB)
  code      : str                        (target CadQuery script)

Per-model training scripts wrap this in a chat template + processor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


SYSTEM_PROMPT = (
    "You are an expert CAD modeller. The user will give you a short orbit "
    "video of a CAD part. Output a CadQuery Python script that reconstructs "
    "the part. Output only the script, no explanation."
)

USER_PROMPT = (
    "Here is an orbit-view video of a CAD part as a sequence of frames. "
    "Write the CadQuery script that produces this part."
)

# Used when depth frames are supplied. RGB and depth are interleaved per
# viewpoint (rgb, depth, rgb, depth, ...) so the two views of each pose are
# adjacent; the prompt explains the colormap and the background masking.
USER_PROMPT_DEPTH = (
    "Here is an orbit-view video of a CAD part. The frames come in pairs: for "
    "each viewpoint you are shown the rendered RGB frame followed immediately by "
    "its depth map. In the depth map the part is colored by distance to the "
    "camera (blue/purple = nearer, red = farther) and the background is black. "
    "Use both the appearance and the depth geometry to write the CadQuery script "
    "that produces this part."
)


class VideoCadQueryDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        frames_dir: str | Path,
        n_frames: int = 16,
        depth_frames_dir: str | Path | None = None,
        augment: Callable | None = None,
    ):
        manifest_path = Path(manifest_path)
        self.frames_dir = Path(frames_dir)
        self.n_frames = n_frames
        self.depth_frames_dir = Path(depth_frames_dir) if depth_frames_dir else None
        # Optional sim2real augmentation (FrameAugmentor). Applied to the RGB
        # frames only — never depth (its color encodes geometry). Pass only on
        # the TRAIN dataset; leave None for val/eval so they stay clean.
        self.augment = augment

        with open(manifest_path) as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.rows)

    def _select(self, paths: list[Path]) -> list[Path]:
        """Uniformly pick n_frames from the (sorted) frame paths."""
        if len(paths) > self.n_frames:
            step = len(paths) / self.n_frames
            return [paths[int(i * step)] for i in range(self.n_frames)]
        return paths

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        obj_id = row["object_id"]

        frame_dir = self.frames_dir / obj_id
        frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
        if len(frame_paths) < self.n_frames:
            raise RuntimeError(
                f"{obj_id}: only {len(frame_paths)} frames, need {self.n_frames}"
            )
        frame_paths = self._select(frame_paths)
        frames = [Image.open(p).convert("RGB") for p in frame_paths]
        if self.augment is not None:
            frames = self.augment(frames)
        code = Path(row["script_path"]).read_text()

        out = {
            "object_id": obj_id,
            "frames": frames,
            "code": code,
        }

        if self.depth_frames_dir is not None:
            depth_dir = self.depth_frames_dir / obj_id
            depth_paths = sorted(depth_dir.glob("frame_*.jpg"))
            # A handful of objects (~0.2%) have empty masks (degenerate/zero-size
            # renders) so no depth frames were produced. Fall back to RGB-only for
            # those rather than dropping them, keeping the set identical to the
            # RGB-only baseline. Downstream branches on the depth_frames key.
            if len(depth_paths) >= self.n_frames:
                depth_paths = self._select(depth_paths)
                out["depth_frames"] = [Image.open(p).convert("RGB") for p in depth_paths]

        return out
