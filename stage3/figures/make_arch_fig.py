"""Architecture diagram for the Methods section:
orbit video -> sampled frames -> LoRA-adapted ViT -> visual tokens ->
LoRA-adapted multimodal LLM (self-attention over visual+text) -> CadQuery program
-> execute -> solid ; scored by IoU/Chamfer.
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path("/scratch/users/abzhong")
OUT = ROOT / "stage3/figures/data"; OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 3.7))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.7); ax.axis("off")
TRAIN, FROZEN, IO = "#f3c969", "#cdd8e6", "#dfeae0"

def block(x, w, title, sub, color, star=False, y=1.25, h=1.25, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                 linewidth=1.4, edgecolor="#33486b", facecolor=color, zorder=2))
    ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
            fontsize=fs, weight="bold", color="#10233d", zorder=3)
    if sub:
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center",
                fontsize=7.8, color="#33486b", zorder=3)
    if star:
        ax.text(x + w - 0.16, y + h - 0.16, "★", ha="center", va="center",
                fontsize=12, color="#c8531b", zorder=4)
    return x + w

def arrow(x0, x1, y=1.875, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=14, lw=1.6, color="#1f3b63", zorder=1))
    if label:
        ax.text((x0+x1)/2, y+0.16, label, ha="center", va="bottom",
                fontsize=7.4, style="italic", color="#444")

# input frames thumbnail
fr = [np.asarray(Image.open(ROOT/f"stage2b/frames/0000/00002065/frame_{i:02d}.jpg")) for i in (3,7,11)]
strip = np.concatenate(fr, axis=1)
axim = ax.inset_axes([0.03, 0.34, 0.16, 0.34])
axim.imshow(strip); axim.axis("off")
ax.text(1.15, 2.75, "Orbit video", ha="center", fontsize=10, weight="bold", color="#10233d")
ax.text(1.15, 1.05, "8 sampled frames", ha="center", fontsize=7.8, color="#33486b")

x = 2.35
arrow(2.15, x)
x = block(x, 2.05, "Vision Transformer", "patchify + self-attn;\nLoRA-adapted", TRAIN, star=True)
arrow(x, x+0.55, label="visual tokens"); x += 0.55
x = block(x, 2.25, "Multimodal LLM", "causal self-attn over\nvisual + text tokens;\nLoRA-adapted", TRAIN, star=True)
arrow(x, x+0.45);
x += 0.45
x = block(x, 2.0, "CadQuery\nprogram", "autoregressive\ntext decode", IO)
arrow(x, x+0.4); x += 0.4
x = block(x, 1.55, "Execute", "CadQuery\n→ solid", IO)

# text prompt feeding the LLM from below
ax.add_patch(FancyBboxPatch((4.95, 0.15), 2.25, 0.7, boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1.0, edgecolor="#8a93a0", facecolor="#f6f7f9", zorder=2))
ax.text(6.07, 0.5, "Task / system prompt", ha="center", va="center", fontsize=7.8, color="#555")
ax.add_patch(FancyArrowPatch((6.07, 0.85), (6.07, 1.25), arrowstyle="-|>",
             mutation_scale=11, lw=1.3, color="#8a93a0", zorder=1))

# eval feedback
ax.add_patch(FancyBboxPatch((9.55, 0.15), 2.1, 0.7, boxstyle="round,pad=0.02,rounding_size=0.06",
             linewidth=1.0, edgecolor="#8a93a0", facecolor="#f6f7f9", linestyle="--", zorder=2))
ax.text(10.6, 0.5, "IoU / Chamfer\nvs. GT solid", ha="center", va="center", fontsize=7.6, color="#555")
ax.add_patch(FancyArrowPatch((10.55, 1.25), (10.6, 0.86), arrowstyle="-|>",
             mutation_scale=11, lw=1.3, color="#8a93a0", linestyle="--", zorder=1))

# legend
ax.add_patch(plt.Rectangle((0.15, 3.25), 0.28, 0.28, facecolor=TRAIN, edgecolor="#33486b"))
ax.text(0.5, 3.39, "★ LoRA-adapted (trainable)", fontsize=8, va="center")
ax.add_patch(plt.Rectangle((4.0, 3.25), 0.28, 0.28, facecolor=IO, edgecolor="#33486b"))
ax.text(4.35, 3.39, "frozen base weights / deterministic", fontsize=8, va="center")

fig.tight_layout()
fig.savefig(OUT / "fig_architecture.png", dpi=170, bbox_inches="tight")
print("wrote", OUT / "fig_architecture.png")
