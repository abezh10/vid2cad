#!/usr/bin/env python3
"""Train-time image augmentation for sim2real robustness (Workstream B2).

The models are trained on clean Blender/EEVEE renders but must run on real phone
video. This module roughens the synthetic frames toward the real distribution
WITHOUT re-rendering: photometric shifts (exposure / white balance / gamma),
optical+sensor degradation (blur, downscale, noise), and codec artifacts (JPEG
recompression), plus a light geometric jitter (crop + small rotation).

Design — clip-consistent, not frame-independent:
  A real capture is ONE device under ONE lighting condition, so exposure, white
  balance, focus blur, downscale factor and JPEG quality are sampled ONCE per
  clip and applied identically to every frame. Only sensor noise is re-sampled
  per frame (it is genuinely independent shot-to-shot). Frame-independent
  photometric jitter would read as flicker and hurt more than help.

Pure PIL + numpy (no torchvision dependency). Applied to the RGB frames only;
never to depth maps (their color encodes geometry) and never at eval time.

Pipeline order mirrors a real camera: geometry -> optics (WB/exposure/gamma,
blur) -> sampling (downscale) -> sensor noise -> codec (JPEG).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Default augmentation config. Every op is gated by a probability `p`; magnitudes
# are uniform ranges. Tuned to be visible but not destructive — the part must
# stay recognizable. Override per-field via config.yaml `augment:` block.
DEFAULT_AUGMENT = {
    "enabled": True,
    # geometric (clip-level; identical box/angle across the orbit)
    "crop": {"p": 0.7, "scale_min": 0.80, "scale_max": 1.0},
    "rotate": {"p": 0.5, "deg": 5.0},
    # photometric (clip-level)
    "brightness": {"p": 0.8, "min": 0.70, "max": 1.30},
    "contrast": {"p": 0.7, "min": 0.75, "max": 1.30},
    "color": {"p": 0.7, "min": 0.70, "max": 1.30},
    "white_balance": {"p": 0.7, "gain": 0.12},   # per-channel ±12% gain
    "gamma": {"p": 0.5, "min": 0.80, "max": 1.25},
    # optics / sensor / codec
    "blur": {"p": 0.5, "radius_max": 1.4},
    "motion_blur": {"p": 0.3, "ksize_max": 9},
    "downscale": {"p": 0.5, "scale_min": 0.45, "scale_max": 0.9},
    "noise": {"p": 0.7, "sigma_max": 12.0},      # per-frame, gaussian, 0-255 scale
    "jpeg": {"p": 0.8, "q_min": 25, "q_max": 80},
}


def _merge(base: dict, override: dict | None) -> dict:
    if not override:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    out = {}
    for k, v in base.items():
        if isinstance(v, dict) and isinstance(override.get(k), dict):
            out[k] = {**v, **override[k]}
        elif k in override:
            out[k] = override[k]
        else:
            out[k] = dict(v) if isinstance(v, dict) else v
    # allow brand-new keys from override (forward compat)
    for k, v in override.items():
        if k not in out:
            out[k] = v
    return out


@dataclass
class FrameAugmentor:
    """Callable: list[PIL.Image] -> list[PIL.Image], clip-consistent."""

    cfg: dict = field(default_factory=lambda: dict(DEFAULT_AUGMENT))

    @classmethod
    def from_config(cls, override: dict | None) -> "FrameAugmentor | None":
        """Build from a config `augment:` block, or None if disabled/absent."""
        if not override or not override.get("enabled", False):
            return None
        return cls(cfg=_merge(DEFAULT_AUGMENT, override))

    # -- individual ops -----------------------------------------------------

    @staticmethod
    def _crop_box(w, h, scale, rng):
        cw, ch = int(round(w * scale)), int(round(h * scale))
        x0 = rng.integers(0, w - cw + 1)
        y0 = rng.integers(0, h - ch + 1)
        return (x0, y0, x0 + cw, y0 + ch)

    @staticmethod
    def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
        lut = [min(255, int((i / 255.0) ** gamma * 255.0 + 0.5)) for i in range(256)]
        return img.point(lut * 3)

    @staticmethod
    def _apply_wb(img: Image.Image, gains) -> Image.Image:
        r, g, b = img.split()
        r = r.point(lambda i, gr=gains[0]: min(255, int(i * gr)))
        g = g.point(lambda i, gg=gains[1]: min(255, int(i * gg)))
        b = b.point(lambda i, gb=gains[2]: min(255, int(i * gb)))
        return Image.merge("RGB", (r, g, b))

    @staticmethod
    def _motion_kernel(ksize: int, angle_deg: float) -> np.ndarray:
        k = np.zeros((ksize, ksize), np.float32)
        k[ksize // 2, :] = 1.0
        # rotate the horizontal streak to `angle`
        from PIL import Image as _I
        ki = _I.fromarray((k * 255).astype(np.uint8)).rotate(angle_deg, resample=_I.BILINEAR)
        k = np.asarray(ki, np.float32)
        s = k.sum()
        return k / s if s > 0 else None

    def _motion_blur(self, img: Image.Image, ksize: int, angle: float) -> Image.Image:
        ker = self._motion_kernel(ksize, angle)
        if ker is None:
            return img
        arr = np.asarray(img, np.float32)
        # separable-free direct conv via FFT-light: use scipy if present else PIL fallback
        try:
            from scipy.ndimage import convolve
            for c in range(3):
                arr[..., c] = convolve(arr[..., c], ker, mode="reflect")
        except Exception:
            return img.filter(ImageFilter.GaussianBlur(radius=ksize / 4.0))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # -- main ---------------------------------------------------------------

    def __call__(self, frames: list[Image.Image]) -> list[Image.Image]:
        if not frames:
            return frames
        c = self.cfg
        rng = np.random.default_rng()
        w, h = frames[0].size

        # ---- sample clip-level params once ----
        crop_box = None
        if rng.random() < c["crop"]["p"]:
            scale = rng.uniform(c["crop"]["scale_min"], c["crop"]["scale_max"])
            crop_box = self._crop_box(w, h, scale, rng)
        rot_deg = (rng.uniform(-c["rotate"]["deg"], c["rotate"]["deg"])
                   if rng.random() < c["rotate"]["p"] else 0.0)
        bright = (rng.uniform(c["brightness"]["min"], c["brightness"]["max"])
                  if rng.random() < c["brightness"]["p"] else None)
        contr = (rng.uniform(c["contrast"]["min"], c["contrast"]["max"])
                 if rng.random() < c["contrast"]["p"] else None)
        color = (rng.uniform(c["color"]["min"], c["color"]["max"])
                 if rng.random() < c["color"]["p"] else None)
        wb = None
        if rng.random() < c["white_balance"]["p"]:
            g = c["white_balance"]["gain"]
            wb = [1.0 + rng.uniform(-g, g) for _ in range(3)]
        gamma = (rng.uniform(c["gamma"]["min"], c["gamma"]["max"])
                 if rng.random() < c["gamma"]["p"] else None)
        blur_r = (rng.uniform(0.3, c["blur"]["radius_max"])
                  if rng.random() < c["blur"]["p"] else None)
        mblur = None
        if rng.random() < c["motion_blur"]["p"]:
            ks = int(rng.integers(3, max(4, c["motion_blur"]["ksize_max"]) + 1)) | 1
            mblur = (ks, rng.uniform(0, 180))
        down = (rng.uniform(c["downscale"]["scale_min"], c["downscale"]["scale_max"])
                if rng.random() < c["downscale"]["p"] else None)
        noise_sigma = (rng.uniform(2.0, c["noise"]["sigma_max"])
                       if rng.random() < c["noise"]["p"] else None)
        jpeg_q = (int(rng.integers(c["jpeg"]["q_min"], c["jpeg"]["q_max"] + 1))
                  if rng.random() < c["jpeg"]["p"] else None)

        out = []
        for fr in frames:
            im = fr
            # geometry
            if crop_box is not None:
                im = im.crop(crop_box).resize((w, h), Image.BILINEAR)
            if rot_deg:
                im = im.rotate(rot_deg, resample=Image.BILINEAR, fillcolor=(127, 127, 127))
            # photometric
            if wb is not None:
                im = self._apply_wb(im, wb)
            if bright is not None:
                im = ImageEnhance.Brightness(im).enhance(bright)
            if contr is not None:
                im = ImageEnhance.Contrast(im).enhance(contr)
            if color is not None:
                im = ImageEnhance.Color(im).enhance(color)
            if gamma is not None:
                im = self._apply_gamma(im, gamma)
            # optics
            if blur_r is not None:
                im = im.filter(ImageFilter.GaussianBlur(radius=blur_r))
            if mblur is not None:
                im = self._motion_blur(im, mblur[0], mblur[1])
            # sampling (downscale then back up — softens like a low-res sensor)
            if down is not None:
                sw, sh = max(8, int(w * down)), max(8, int(h * down))
                im = im.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            # sensor noise (per-frame, independent)
            if noise_sigma is not None:
                arr = np.asarray(im, np.float32)
                arr += rng.normal(0, noise_sigma, arr.shape)
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            # codec
            if jpeg_q is not None:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=jpeg_q)
                buf.seek(0)
                im = Image.open(buf).convert("RGB")
            out.append(im)
        return out


if __name__ == "__main__":
    # Headless self-test: synthetic frames through the pipeline, shapes preserved.
    import sys
    aug = FrameAugmentor.from_config({"enabled": True})
    assert aug is not None
    base = [Image.fromarray(
        (np.indices((480, 640)).sum(0) % 256).astype(np.uint8)
    ).convert("RGB") for _ in range(8)]
    out = aug(base)
    assert len(out) == 8 and all(o.size == (640, 480) for o in out), "size mismatch"
    # disabled path
    assert FrameAugmentor.from_config({"enabled": False}) is None
    assert FrameAugmentor.from_config(None) is None
    # If a real frame path is given, write an augmented strip for visual QA.
    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1]).convert("RGB")
        strip = aug([src] * 6)
        W, H = src.size
        canvas = Image.new("RGB", (W, H * 6))
        for i, im in enumerate(strip):
            canvas.paste(im, (0, i * H))
        canvas.save("aug_preview.png")
        print("wrote aug_preview.png")
    print("augment.py self-test OK")
