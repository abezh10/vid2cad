#!/usr/bin/env python3
"""vid2cad — editable videos showing patches + transformer blocks.

Produces (in stage3/figures/):
  orbit_raw.mp4      copy of the real Blender orbit render (clean footage to edit)
  orbit_patches.mp4  the SAME orbit with a 16x16 ViT patch grid burned on the moving part
  patchify_anim.mp4  animated: frame -> patch sweep -> tokens -> transformer blocks -> LLM
  patchify_frames/   the animation as individual PNGs (drop into any editor)

Pure OpenCV + numpy (no system ffmpeg needed). Edit OID / sizes / colors below.

Usage: python3 stage3/figures/make_patch_videos.py
"""
import shutil
from pathlib import Path

import cv2
import numpy as np

OID = "0000/00000908"
SRC_VIDEO = f"/scratch/users/abzhong/stage1_output/{OID}/video.mp4"
OUT = Path("/scratch/users/abzhong/stage3/figures")
PC, PR = 16, 12                      # patch grid cols, rows
CYAN = (255, 224, 0)                 # BGR
YELLOW = (0, 212, 255)
GREEN = (106, 160, 63)
PURPLE = (167, 94, 123)
DARK = (40, 40, 40)


def grid(img, pc=PC, pr=PR, color=CYAN, thick=1, alpha=0.9):
    h, w = img.shape[:2]
    ov = img.copy()
    for c in range(pc + 1):
        x = int(round(c * w / pc)); cv2.line(ov, (x, 0), (x, h), color, thick, cv2.LINE_AA)
    for r in range(pr + 1):
        y = int(round(r * h / pr)); cv2.line(ov, (0, y), (w, y), color, thick, cv2.LINE_AA)
    return cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)


def orbit_patches():
    cap = cv2.VideoCapture(SRC_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    vw = cv2.VideoWriter(str(OUT / "orbit_patches.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for loop in range(2):                      # 2 loops for a longer editable clip
        for i, fr in enumerate(frames):
            g = grid(fr)
            # sweep a highlighted patch around so the eye tracks "patch -> token"
            k = (i * 3) % (PC * PR)
            pc_, pr_ = k % PC, k // PC
            x0, y0 = int(pc_ * w / PC), int(pr_ * h / PR)
            x1, y1 = int((pc_ + 1) * w / PC), int((pr_ + 1) * h / PR)
            cv2.rectangle(g, (x0, y0), (x1, y1), YELLOW, 2, cv2.LINE_AA)
            cv2.rectangle(g, (0, 0), (w, 26), DARK, -1)
            cv2.putText(g, "ViT patches (16x16 px) on the orbit frames",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(g)
    vw.release()
    print("wrote", OUT / "orbit_patches.mp4")
    shutil.copyfile(SRC_VIDEO, OUT / "orbit_raw.mp4")
    print("wrote", OUT / "orbit_raw.mp4")
    return frames[len(frames) // 2]


def rrect(img, x, y, w, h, color, fill=False, t=2):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1 if fill else t, cv2.LINE_AA)


def patchify_anim(base_frame):
    W, H, fps = 1280, 600, 24
    fdir = OUT / "patchify_frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    vw = cv2.VideoWriter(str(OUT / "patchify_anim.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    fw, fh, fx, fy = 460, 345, 40, 150          # frame panel
    frame = cv2.resize(base_frame, (fw, fh))
    tx, ty, tgw = 600, 150, 9                    # token mini-grid cell size
    bx, by, bw, bh = 980, 175, 200, 70           # transformer block boxes

    SWEEP, PULSE, OUTRO = 70, 48, 26
    T = SWEEP + PULSE + OUTRO

    def header(cv, title):
        cv2.rectangle(cv, (0, 0), (W, 46), DARK, -1)
        cv2.putText(cv, title, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    for t in range(T):
        cv = np.full((H, W, 3), 255, np.uint8)
        # --- frame panel + grid ---
        panel = grid(frame.copy())
        n_patches = PC * PR
        visited = min(n_patches, int((t / max(1, SWEEP)) * n_patches)) if t < SWEEP else n_patches
        ov = panel.copy()
        for k in range(visited):
            c, r = k % PC, k // PC
            x0, y0 = int(c * fw / PC), int(r * fh / PR)
            x1, y1 = int((c + 1) * fw / PC), int((r + 1) * fh / PR)
            cv2.rectangle(ov, (x0, y0), (x1, y1), GREEN, -1)
        panel = cv2.addWeighted(ov, 0.28, panel, 0.72, 0)
        if t < SWEEP and visited > 0:
            k = visited - 1; c, r = k % PC, k // PC
            x0, y0 = int(c * fw / PC), int(r * fh / PR)
            x1, y1 = int((c + 1) * fw / PC), int((r + 1) * fh / PR)
            cv2.rectangle(panel, (x0, y0), (x1, y1), YELLOW, 2, cv2.LINE_AA)
        cv[fy:fy + fh, fx:fx + fw] = panel
        cv2.putText(cv, "1. patchify frame", (fx, fy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, DARK, 2, cv2.LINE_AA)

        # --- token mini-grid ---
        cv2.putText(cv, "2. patch tokens + pos", (tx, fy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, DARK, 2, cv2.LINE_AA)
        for k in range(n_patches):
            c, r = k % PC, k // PC
            x = tx + c * (tgw + 2); y = ty + r * (tgw + 2)
            col = GREEN if k < visited else (210, 210, 210)
            cv2.rectangle(cv, (x, y), (x + tgw, y + tgw), col, -1, cv2.LINE_AA)

        # --- transformer blocks ---
        cv2.putText(cv, "3. transformer blocks", (bx, fy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, DARK, 2, cv2.LINE_AA)
        pulse_i = -1
        if SWEEP <= t < SWEEP + PULSE:
            pulse_i = int((t - SWEEP) / PULSE * 3) % 3
        for i in range(3):
            yb = by + i * (bh + 16)
            active = (i == pulse_i) or (t >= SWEEP + PULSE)
            rrect(cv, bx, yb, bw, bh, PURPLE if active else (225, 215, 240), fill=True)
            rrect(cv, bx, yb, bw, bh, (120, 60, 95), t=2)
            txtcol = (255, 255, 255) if active else (90, 50, 80)
            cv2.putText(cv, "MHSA + MLP", (bx + 28, yb + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, txtcol, 2, cv2.LINE_AA)
        cv2.putText(cv, "x 27  (d=1152)", (bx + 30, by + 3 * (bh + 16) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 60, 95), 2, cv2.LINE_AA)

        # arrows
        cv2.arrowedLine(cv, (fx + fw + 12, fy + fh // 2), (tx - 14, ty + 55),
                        DARK, 2, cv2.LINE_AA, tipLength=0.03)
        cv2.arrowedLine(cv, (tx + 16 * (tgw + 2) + 12, ty + 55),
                        (bx - 14, by + bh), DARK, 2, cv2.LINE_AA, tipLength=0.04)

        # outro: -> LLM -> code
        if t >= SWEEP + PULSE:
            cv2.putText(cv, "-> MLP connector -> LLM decoder -> CadQuery code",
                        (fx, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 110, 40), 2, cv2.LINE_AA)

        header(cv, f"vid2cad: how the Vision Transformer reads each frame   (object {OID})")
        vw.write(cv)
        cv2.imwrite(str(fdir / f"frame_{t:04d}.png"), cv)
    vw.release()
    print("wrote", OUT / "patchify_anim.mp4", f"({T} frames)")
    print("wrote", fdir, "(editable PNG sequence)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mid = orbit_patches()
    patchify_anim(mid)


if __name__ == "__main__":
    main()
