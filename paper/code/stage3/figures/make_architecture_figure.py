#!/usr/bin/env python3
"""vid2cad architecture figure (paper-ready, editable).

Builds a single horizontal pipeline figure from REAL data:
  orbit frames -> patchify (ViT) -> transformer blocks -> connector -> LLM -> CadQuery
Outputs PNG (raster preview) + SVG (vector, editable in Inkscape/Illustrator).

Usage:  python3 stage3/figures/make_architecture_figure.py
Edit:   change OID / FRAMES_DIR / CODE_SNIPPET below, or restyle the boxes.
"""
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

OID = "0000/00000908"
FRAMES_DIR = f"/scratch/users/abzhong/stage2b/frames/{OID}"
OUT = Path("/scratch/users/abzhong/stage3/figures")
PATCH_COLS, PATCH_ROWS = 16, 12          # visual patch grid (16x16 px on 640x480-ish)

CODE_SNIPPET = (
    "import cadquery as cq\n"
    "result = None\n"
    "\n"
    "wp_1 = cq.Workplane(cq.Plane(\n"
    "    origin=cq.Vector(0,0,0),\n"
    "    normal=cq.Vector(0,-1,0)))\n"
    "wp_1 = (wp_1.moveTo(0.0048, 0.0079)\n"
    "        .lineTo(0.1762, 0.0079)\n"
    "        .lineTo(0.1762, 0.1032)\n"
    "        .lineTo(0.0048, 0.1032).close())\n"
    "body_1 = wp_1.extrude(0.0079)\n"
    "result = body_1\n"
    "#  + outer wall (body_2) -> union\n"
)

# palette
C_VIT = "#E8E2F6"; E_VIT = "#7B5EA7"
C_CON = "#DDEEF6"; E_CON = "#3B82A6"
C_LLM = "#FCE8C8"; E_LLM = "#C8902A"
C_TOK = "#CFE8D8"; E_TOK = "#3FA06A"
ARROW = "#444444"
mono = FontProperties(family="monospace", size=7.0)


def load_frames(n=16):
    fs = sorted(glob.glob(f"{FRAMES_DIR}/*.jpg") + glob.glob(f"{FRAMES_DIR}/*.png"))
    return [plt.imread(f) for f in fs[:n]]


def rbox(ax, x, y, w, h, fc, ec, lw=1.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, label=None, rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                        lw=1.8, color=ARROW, connectionstyle=f"arc3,rad={rad}", zorder=2)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, label, ha="center", va="bottom",
                fontsize=7.5, color=ARROW, style="italic")


def main():
    frames = load_frames()
    fig = plt.figure(figsize=(17.5, 6.2), dpi=200)
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    bg.set_xlim(0, 1); bg.set_ylim(0, 1)

    bg.text(0.5, 0.965, "vid2cad — video → CadQuery via a Vision-Language Model",
            ha="center", va="top", fontsize=15, fontweight="bold")
    bg.text(0.5, 0.925, f"real example: DeepCAD object {OID}  —  model input: 8 frames sampled from the 60-frame orbit",
            ha="center", va="top", fontsize=9.5, color="#555")

    yc = 0.52  # vertical center of the pipeline

    # ---- 1. Filmstrip (orbit video) ----
    fs_x, fs_w = 0.035, 0.135
    n_strip = 5
    strip_idx = np.linspace(0, len(frames) - 1, n_strip).astype(int)
    cell_h = 0.085
    for i, fi in enumerate(strip_idx):
        ax = fig.add_axes([fs_x, yc - 0.18 + i * (cell_h + 0.006), fs_w, cell_h])
        ax.imshow(frames[fi]); ax.axis("off")
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#222"); s.set_linewidth(0.8)
    bg.text(fs_x + fs_w / 2, yc - 0.235, "orbit video\n(360°, 8 frames sampled)",
            ha="center", va="top", fontsize=8.5, fontweight="bold")

    # ---- 2. Patchify panel (real frame + grid) ----
    pf_x, pf_w, pf_h = 0.225, 0.165, 0.27
    ax = fig.add_axes([pf_x, yc - pf_h / 2, pf_w, pf_h])
    ax.imshow(frames[len(frames) // 2], extent=[0, 1, 0, 1])
    for c in range(PATCH_COLS + 1):
        ax.axvline(c / PATCH_COLS, color="#00E0FF", lw=0.6, alpha=0.85)
    for r in range(PATCH_ROWS + 1):
        ax.axhline(r / PATCH_ROWS, color="#00E0FF", lw=0.6, alpha=0.85)
    # highlight one patch
    ax.add_patch(plt.Rectangle((6 / PATCH_COLS, 6 / PATCH_ROWS), 1 / PATCH_COLS,
                 1 / PATCH_ROWS, fc="none", ec="#FFD400", lw=2.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for s in ax.spines.values():
        s.set_visible(True)
    bg.text(pf_x + pf_w / 2, yc - pf_h / 2 - 0.03,
            "patchify: split each frame into\n16×16 px patches → tokens",
            ha="center", va="top", fontsize=8.5, fontweight="bold")

    # ---- 3. Token sequence + embed ----
    tk_x = 0.415
    for i in range(7):
        rbox(bg, tk_x, yc - 0.155 + i * 0.045, 0.028, 0.034, C_TOK, E_TOK, lw=1.1)
    bg.text(tk_x + 0.014, yc + 0.20, "patch\nembeddings\n+ 2D pos.", ha="center",
            va="bottom", fontsize=8, color=E_TOK, fontweight="bold")

    # ---- 4. ViT transformer blocks ----
    vt_x, vt_w = 0.49, 0.115
    for i, dy in enumerate([0.07, 0.0, -0.07]):
        rbox(bg, vt_x, yc + dy - 0.035, vt_w, 0.062, C_VIT, E_VIT)
        bg.text(vt_x + vt_w / 2, yc + dy, "MHSA  +  MLP", ha="center", va="center",
                fontsize=8.5, color="#3A2A55")
    bg.text(vt_x + vt_w / 2, yc + 0.155, "Vision Transformer (ViT)", ha="center",
            va="bottom", fontsize=9.5, fontweight="bold", color=E_VIT)
    bg.text(vt_x + vt_w / 2, yc - 0.135, "× 27 blocks,  d = 1152,  patch 16",
            ha="center", va="top", fontsize=8, color="#3A2A55")
    # LoRA badge on the ViT (the qwen_vit experiment)
    rbox(bg, vt_x + vt_w - 0.045, yc + 0.085, 0.055, 0.03, "#FFF3D6", "#C8902A", lw=1.2)
    bg.text(vt_x + vt_w - 0.0175, yc + 0.10, "LoRA*", ha="center", va="center",
            fontsize=7.5, color="#9A6B12", fontweight="bold")

    # ---- 5. Connector / merger ----
    cn_x, cn_w = 0.635, 0.075
    rbox(bg, cn_x, yc - 0.05, cn_w, 0.10, C_CON, E_CON)
    bg.text(cn_x + cn_w / 2, yc, "MLP\nconnector\n(merger)", ha="center", va="center",
            fontsize=8.5, color="#1E5571")

    # ---- 6. LLM decoder ----
    lm_x, lm_w = 0.74, 0.115
    rbox(bg, lm_x, yc - 0.105, lm_w, 0.21, C_LLM, E_LLM, lw=2.0)
    bg.text(lm_x + lm_w / 2, yc + 0.055, "LLM decoder", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#7A560F")
    bg.text(lm_x + lm_w / 2, yc - 0.01, "Qwen3-VL-8B", ha="center", va="center",
            fontsize=9, color="#7A560F")
    bg.text(lm_x + lm_w / 2, yc - 0.06, "attends jointly over\nimage + text tokens",
            ha="center", va="center", fontsize=7.5, color="#7A560F", style="italic")
    rbox(bg, lm_x + lm_w - 0.045, yc + 0.075, 0.05, 0.03, "#FFF3D6", "#C8902A", lw=1.2)
    bg.text(lm_x + lm_w - 0.02, yc + 0.09, "LoRA", ha="center", va="center",
            fontsize=7.5, color="#9A6B12", fontweight="bold")

    # ---- 7. Output code ----
    co_x, co_w, co_h = 0.875, 0.115, 0.34
    rbox(bg, co_x, yc - co_h / 2, co_w, co_h, "#1E1E1E", "#000", lw=1.2)
    bg.text(co_x + 0.008, yc + co_h / 2 - 0.018, CODE_SNIPPET, ha="left", va="top",
            fontproperties=mono, color="#E6E6E6", linespacing=1.25)
    bg.text(co_x + co_w / 2, yc - co_h / 2 - 0.03,
            "CadQuery program\n(editable parametric CAD)", ha="center", va="top",
            fontsize=8.5, fontweight="bold")

    # ---- arrows ----
    arrow(bg, fs_x + fs_w + 0.004, yc, pf_x - 0.006, yc, "sample")
    arrow(bg, pf_x + pf_w + 0.004, yc, tk_x - 0.006, yc, "linear\nembed")
    arrow(bg, tk_x + 0.03, yc, vt_x - 0.006, yc)
    arrow(bg, vt_x + vt_w + 0.004, yc, cn_x - 0.006, yc)
    arrow(bg, cn_x + cn_w + 0.004, yc, lm_x - 0.006, yc)
    arrow(bg, lm_x + lm_w + 0.004, yc, co_x - 0.006, yc, "decode")

    # ---- footnote (the ablation) ----
    bg.text(0.5, 0.045,
            "* LoRA on the ViT is the controlled ablation (qwen_vit): baseline runs adapt only the LLM (ViT frozen); "
            "qwen_vit also adapts the 27 ViT blocks.\n"
            "Multi-view fusion is implicit — the LLM self-attends over the concatenated per-frame patch tokens; there is no dedicated 3D module.",
            ha="center", va="bottom", fontsize=7.8, color="#555")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure_architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / "figure_architecture.svg", bbox_inches="tight", facecolor="white")
    print("wrote", OUT / "figure_architecture.png")
    print("wrote", OUT / "figure_architecture.svg")


if __name__ == "__main__":
    main()
