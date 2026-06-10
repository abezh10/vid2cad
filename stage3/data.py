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

# --- Geometry-aware conditioning variant ("geom") --------------------------
# Conditions the model on HOW to represent non-orthogonal geometry (the bucket
# it collapses on: diagonal/triangular edges, arcs/splines, small features,
# multi-solid parts) so it stops simplifying complex parts into one box. Applied
# identically at train and eval. General guidance — does not bias toward boxes,
# so it should not regress the axis-aligned parts.
SYSTEM_PROMPT_GEOM = (
    "You are an expert CAD modeller. The user gives you a short orbit video of a "
    "single CAD part as frames. Reconstruct it as a CadQuery Python script.\n"
    "Build the part as a sequence of sketch-then-extrude solids combined with "
    "union/cut, using these primitives:\n"
    "- workplane: cq.Plane(origin, xDir, normal) then cq.Workplane(plane)\n"
    "- straight profile: .moveTo(x,y).lineTo(...)....close() — use lineTo for "
    "EVERY straight edge, including slanted, diagonal and triangular edges; do "
    "not force right angles where the edge is angled\n"
    "- circular profile or hole: .circle(r)\n"
    "- curved/rounded edge: .threePointArc(mid_pt, end_pt), chaining arcs around "
    "fillets, rounded ends and arc segments\n"
    "- solid: .extrude(depth); use .cut(...) for holes and pockets\n"
    "Before writing, read the geometry carefully:\n"
    "- count every distinct solid and feature and model all of them — never "
    "simplify a multi-feature part down to a single block\n"
    "- reproduce slanted and curved edges with lineTo / threePointArc instead of "
    "approximating them as axis-aligned rectangles\n"
    "- preserve small features (small holes, thin tabs, narrow slots) at their "
    "true relative size\n"
    "Output only the script, no explanation."
)
USER_PROMPT_GEOM = USER_PROMPT


def get_prompts(variant: str = "default") -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a prompt variant.

    'default' = the original generic prompt (all prior runs / paper baselines).
    'geom'    = geometry-aware conditioning (above). MUST match between the
                checkpoint's training and its eval.
    """
    if variant == "geom":
        return SYSTEM_PROMPT_GEOM, USER_PROMPT_GEOM
    if variant in ("default", "", None):
        return SYSTEM_PROMPT, USER_PROMPT
    raise ValueError(f"unknown prompt_variant: {variant!r}")

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
        max_side: int | None = None,
    ):
        manifest_path = Path(manifest_path)
        self.frames_dir = Path(frames_dir)
        self.n_frames = n_frames
        self.depth_frames_dir = Path(depth_frames_dir) if depth_frames_dir else None
        # Optional sim2real augmentation (FrameAugmentor). Applied to the RGB
        # frames only — never depth (its color encodes geometry). Pass only on
        # the TRAIN dataset; leave None for val/eval so they stay clean.
        self.augment = augment
        # Optional resolution cap: downscale frames so max(w, h) <= max_side
        # (keeps aspect). Lets high-res renders (e.g. 960x720) train at a
        # controlled token budget; None = native resolution unchanged.
        self.max_side = max_side

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

    def _load(self, p: Path) -> Image.Image:
        im = Image.open(p).convert("RGB")
        if self.max_side and max(im.size) > self.max_side:
            w, h = im.size
            s = self.max_side / max(w, h)
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        return im

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        obj_id = row["object_id"]

        # Manifest rows carry an explicit frame_dir (build_manifest.py always
        # writes one); honor it so mixed-source sets (original 640x480 tree +
        # expansion renders elsewhere) can train together. Fall back to the
        # conventional <frames_dir>/<object_id> layout.
        frame_dir = Path(row["frame_dir"]) if row.get("frame_dir") else (
            self.frames_dir / obj_id)
        frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
        if len(frame_paths) < self.n_frames:
            raise RuntimeError(
                f"{obj_id}: only {len(frame_paths)} frames, need {self.n_frames}"
            )
        frame_paths = self._select(frame_paths)
        frames = [self._load(p) for p in frame_paths]
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
