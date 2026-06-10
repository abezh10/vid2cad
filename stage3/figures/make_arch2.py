"""Detailed vid2cad architecture figure (CVPR figure*-width, high-res, no overlaps).
Real example: DeepCAD object 0000/00000908. Qwen3-VL-8B ViT: 27 blocks, d=1152,
patch 16, 16 heads, MLP 4304, 2x2 spatial merge -> 4096-d LLM tokens.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw

ROOT = Path("/scratch/users/abzhong")
OUT = ROOT / "stage3/figures/data"; OUT.mkdir(parents=True, exist_ok=True)
OBJ = "0000/00000908"
XL, YL = 28.0, 12.0
TRAIN, MERGE, VIT, IO = "#f3c969", "#bcd9ef", "#d9ccef", "#dfeae0"

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, XL); ax.set_ylim(0, YL); ax.axis("off")

def box(x0, x1, y0, y1, fc, ec="#33486b", lw=1.4, r=0.08):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0, zorder=2,
                 boxstyle=f"round,pad=0.02,rounding_size={r}", lw=lw, ec=ec, fc=fc))

def arrow(x0, x1, y, lab=None, lw=1.8, c="#1f3b63"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=15, lw=lw, color=c, zorder=3))
    if lab: ax.text((x0+x1)/2, y+0.42, lab, ha="center", va="bottom",
                    fontsize=8.5, style="italic", color="#555")

def pill(cx, cy, txt, w=0.86):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-0.19), w, 0.38, zorder=4,
                 boxstyle="round,pad=0.02,rounding_size=0.1", lw=1.1,
                 ec="#c8531b", fc="#ffe7c2"))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=7.5,
            weight="bold", color="#9c3d12", zorder=5)

def img_inset(x0, y0, w, h, arr):
    axi = ax.inset_axes([x0/XL, y0/YL, w/XL, h/YL]); axi.imshow(arr); axi.axis("off")

# title
ax.text(XL/2, 11.45, "vid2cad — video → CadQuery via a Vision-Language Model",
        ha="center", fontsize=15.5, weight="bold")
ax.text(XL/2, 10.78, f"real example: DeepCAD object {OBJ}  (8 of 60 orbit frames shown)",
        ha="center", fontsize=10, color="#666")

# 1) orbit video frames
fr_idx = [0, 2, 5, 8, 12]
h, gap, top = 1.22, 0.18, 9.55
for k, i in enumerate(fr_idx):
    arr = np.asarray(Image.open(ROOT/f"stage2b/frames/{OBJ}/frame_{i:02d}.jpg"))
    img_inset(0.35, top - (k+1)*h - k*gap, 1.85, h, arr)
ax.text(1.27, 1.75, "orbit video\n(360°, 8 frames sampled)", ha="center",
        fontsize=9.3, weight="bold", color="#10233d")
arrow(2.45, 3.35, 6.0, "sample")

# 2) patchify frame with grid
im = Image.open(ROOT/f"stage2b/frames/{OBJ}/frame_05.jpg").convert("RGB")
dr = ImageDraw.Draw(im); W, H = im.size; p = 16
for x in range(0, W+1, p): dr.line([(x, 0), (x, H)], fill=(120, 210, 235), width=1)
for y in range(0, H+1, p): dr.line([(0, y), (W, y)], fill=(120, 210, 235), width=1)
rr, cc = 16, 19
dr.rectangle([cc*p, rr*p, (cc+1)*p, (rr+1)*p], outline=(255, 205, 0), width=3)
img_inset(3.55, 4.95, 3.2, 2.4, np.asarray(im))
ax.text(5.15, 3.95, "patchify: split each frame into\n16×16 px patches → tokens",
        ha="center", fontsize=8.8, weight="bold", color="#10233d")
arrow(6.95, 7.85, 6.0)
ax.text(7.4, 6.62, "linear\nembed", ha="center", fontsize=7.8, style="italic", color="#555")

# 3) patch-embedding tokens
ax.text(8.55, 8.18, "patch embeddings\n+ 2D pos.", ha="center", fontsize=8.4,
        weight="bold", color="#2e7d4f")
for k in range(6):
    box(8.15, 8.95, 4.30+k*0.58, 4.78+k*0.58, "#bfe3c8", ec="#2e7d4f", lw=1.1, r=0.1)
arrow(9.05, 9.5, 6.0)

# 4) Vision Transformer (contained module)
box(9.6, 13.72, 4.05, 8.12, "#ece6f7", ec="#5a3da8", lw=1.3, r=0.07)
ax.text(11.85, 9.12, "Vision Transformer (ViT)", ha="center", fontsize=10.2,
        weight="bold", color="#5a3da8")
for k, yy in enumerate([5.02, 5.96, 6.90]):
    box(9.85, 13.47, yy, yy+0.82, VIT, ec="#5a3da8", lw=1.2)
    ax.text(11.66, yy+0.41, "MHSA  +  MLP", ha="center", va="center",
            fontsize=9.3, weight="bold", color="#3a2470")
pill(13.04, 8.12, "LoRA*", w=0.94)
ax.text(11.66, 4.55, "× 27 blocks  ·  d = 1152  ·  patch 16", ha="center",
        fontsize=8.2, color="#3a2470")
arrow(13.78, 14.0, 6.0)

# 5) merger
box(14.05, 16.45, 4.45, 7.65, MERGE, ec="#1f6f9c")
ax.text(15.25, 6.62, "MLP connector\n(merger)", ha="center", va="center",
        fontsize=8.6, weight="bold", color="#0d3c57")
ax.text(15.25, 5.42, "2×2 token merge", ha="center", va="center",
        fontsize=7.5, color="#0d3c57")
arrow(16.5, 16.68, 6.0)

# 6) LLM decoder
box(16.7, 20.3, 4.35, 7.65, TRAIN)
ax.text(18.5, 7.05, "LLM decoder", ha="center", fontsize=11.2, weight="bold", color="#10233d")
ax.text(18.5, 6.42, "Qwen3-VL-8B", ha="center", fontsize=9.6, color="#10233d")
ax.text(18.5, 5.45, "attends jointly over\nimage + text tokens", ha="center",
        fontsize=8.4, style="italic", color="#5a4a1f")
pill(19.5, 7.74, "LoRA")
arrow(20.4, 20.98, 6.0)
ax.text(20.69, 6.86, "decode", ha="center", va="bottom", fontsize=6.6,
        style="italic", color="#777")

# 7) CadQuery program
code = [
 "import cadquery as cq", "result = None",
 "plane_wp_1 = cq.Plane(", "    origin=cq.Vector(0,0,0),",
 "    normal=cq.Vector(0,-1,0))", "wp_1 = cq.Workplane(plane_wp_1)",
 "wp_1 = (wp_1.moveTo(0.0048, 0.0079)", "    .lineTo(0.1762, 0.0079)",
 "    .lineTo(0.1762, 0.1032)", "    .lineTo(0.0048, 0.1032).close())",
 "body_1 = wp_1.extrude(0.0079)", "result = body_1",
 "#  + outer wall (body_2) -> union",
]
box(21.1, 27.7, 2.9, 9.0, "#23232b", ec="#23232b", r=0.05)
ax.text(21.4, 8.6, "\n".join(code), va="top", ha="left", fontfamily="monospace",
        fontsize=7.5, color="#e8e8ec", zorder=4, linespacing=1.42)
ax.text(24.4, 2.55, "CadQuery program\n(editable parametric CAD)", ha="center",
        va="top", fontsize=9.4, weight="bold", color="#10233d")

# footnote
ax.text(XL/2, 0.78,
    "* LoRA on the ViT is the controlled ablation (qwen_vit): the baseline adapts only the LLM (ViT frozen); qwen_vit also adapts all 27 ViT blocks.\n"
    "Multi-view fusion is implicit — the LLM self-attends over the concatenated per-frame patch tokens; there is no dedicated 3D module.",
    ha="center", va="center", fontsize=8, color="#777", linespacing=1.3)

fig.savefig(OUT / "fig_architecture_detailed.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "fig_architecture_detailed.svg", bbox_inches="tight")
print("wrote", OUT / "fig_architecture_detailed.png", "+ .svg")
