"""Vertical ViT-depth figure: all 27 Qwen3-VL vision blocks + one expanded block.
Shows where LoRA is applied (qkv/proj in MHSA, fc1/fc2 in MLP) and the residual path.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/scratch/users/abzhong/stage3/figures/data"); OUT.mkdir(parents=True, exist_ok=True)
GOLD, LN, ATTN, MLP, ADD = "#f3c969", "#e8eef5", "#d9ccef", "#cfe3d6", "#fbe3c8"

fig, ax = plt.subplots(figsize=(6.8, 8.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x0, x1, y0, y1, fc, ec="#33486b", lw=1.3, r=0.06, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0, zorder=z,
                 boxstyle=f"round,pad=0.02,rounding_size={r}", lw=lw, ec=ec, fc=fc))

def up(x, y0, y1, lw=1.6, c="#1f3b63", ls="-"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=12, lw=lw, color=c, ls=ls, zorder=3))

def pill(cx, cy, txt="LoRA", w=0.62):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-0.16), w, 0.32, zorder=5,
                 boxstyle="round,pad=0.02,rounding_size=0.07", lw=1.0, ec="#c8531b", fc="#ffe7c2"))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=6.8, weight="bold", color="#9c3d12", zorder=6)

# ---------------- left: 27-block stack ----------------
ax.text(2.25, 9.42, "Vision encoder depth", ha="center", fontsize=11, weight="bold", color="#5a3da8")
box(0.85, 3.65, 0.55, 1.25, "#dfeae0"); ax.text(2.25, 0.9, "patch tokens (+2D pos)\nd = 1152,  patch 16",
        ha="center", va="center", fontsize=7.8, color="#10233d")
y0, n, bh, gp = 1.55, 27, 0.18, 0.045
for k in range(n):
    yy = y0 + k*(bh+gp)
    box(1.1, 3.4, yy, yy+bh, GOLD, ec="#c8531b", lw=0.8, r=0.04)
ytop = y0 + (n-1)*(bh+gp) + bh
ax.text(2.25, y0+bh/2, "block 1", ha="center", va="center", fontsize=6.6, color="#7a3b12")
ax.text(2.25, ytop-bh/2, "block 27", ha="center", va="center", fontsize=6.6, color="#7a3b12")
up(2.25, 1.28, 1.6); up(2.25, ytop+0.04, ytop+0.42)
box(0.85, 3.65, ytop+0.45, ytop+1.05, "#bcd9ef", ec="#1f6f9c")
ax.text(2.25, ytop+0.75, "→ merger (2×2)\n→ 4096-d → LLM", ha="center", va="center",
        fontsize=7.6, color="#0d3c57")
# x27 brace
ax.annotate("", xy=(0.7, ytop), xytext=(0.7, y0),
            arrowprops=dict(arrowstyle="-", color="#c8531b", lw=1.4))
ax.text(0.42, (y0+ytop)/2, "× 27", rotation=90, ha="center", va="center",
        fontsize=9, weight="bold", color="#c8531b")

# ---------------- dashed connector to exploded block ----------------
kmid = 13; ymid = y0 + kmid*(bh+gp) + bh/2
ax.add_patch(FancyArrowPatch((3.4, ymid), (5.1, 5.1), arrowstyle="-|>", mutation_scale=11,
             lw=1.2, color="#888", ls=(0, (4, 3)), zorder=1))

# ---------------- right: one block expanded ----------------
ax.text(6.9, 9.42, "Inside one ViT block", ha="center", fontsize=11, weight="bold", color="#5a3da8")
cx = 6.9
def slab(y0, y1, fc, title, sub=None, fs=8.6):
    box(5.15, 8.55, y0, y1, fc)
    ax.text(cx, (y0+y1)/2 + (0.12 if sub else 0), title, ha="center", va="center",
            fontsize=fs, weight="bold", color="#10233d")
    if sub: ax.text(cx, (y0+y1)/2 - 0.18, sub, ha="center", va="center", fontsize=7.0, color="#444")

ax.text(cx, 1.15, "token sequence in", ha="center", fontsize=7.6, style="italic", color="#555")
up(cx, 1.32, 1.55)
slab(1.55, 2.05, LN, "LayerNorm")
up(cx, 2.05, 2.35)
slab(2.35, 3.55, ATTN, "Multi-Head Self-Attention", "16 heads,  d = 1152"); pill(9.2, 2.95)
ax.text(cx, 2.62, "LoRA on  Wq, Wk, Wv, Wo", ha="center", fontsize=6.6, color="#5a3da8")
up(cx, 3.55, 3.85)
slab(3.85, 4.35, ADD, "⊕  residual add")
up(cx, 4.35, 4.65)
slab(4.65, 5.15, LN, "LayerNorm")
up(cx, 5.15, 5.45)
slab(5.45, 6.75, MLP, "MLP  (1152 → 4304 → 1152)", "GELU"); pill(9.2, 6.1)
ax.text(cx, 5.72, "LoRA on  fc1, fc2", ha="center", fontsize=6.6, color="#2e7d4f")
up(cx, 6.75, 7.05)
slab(7.05, 7.55, ADD, "⊕  residual add")
up(cx, 7.55, 8.05)
ax.text(cx, 8.25, "token sequence out", ha="center", fontsize=7.6, style="italic", color="#555")

# residual skip arcs (right side)
for (ya, yb) in [(1.45, 4.10), (4.50, 7.30)]:
    ax.add_patch(FancyArrowPatch((4.95, ya), (5.0, yb), arrowstyle="-|>", mutation_scale=10,
                 lw=1.2, color="#999", connectionstyle="arc3,rad=-0.45", zorder=1))
ax.text(4.35, 5.75, "residual\npath", ha="center", fontsize=6.6, color="#999", rotation=90)

fig.suptitle("Qwen3-VL-8B vision transformer  (LoRA-adapted tower)",
             fontsize=12.5, weight="bold", y=0.99)
fig.savefig(OUT / "fig_vit_stack.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "fig_vit_stack.svg", bbox_inches="tight")
print("wrote", OUT / "fig_vit_stack.png", "+ .svg")
