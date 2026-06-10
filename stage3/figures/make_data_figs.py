"""Generate Data-section figures for the vid2cad paper.

Outputs (into stage3/figures/data/):
  fig_data_funnel.png  - dataset construction funnel with real counts
  fig_script_dist.png  - script-length / char distributions of GT programs
  fig_io_example.png   - example input (orbit frames) -> output (CadQuery) pairs
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path("/scratch/users/abzhong")
OUT = ROOT / "stage3/figures/data"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Figure 1: dataset construction pipeline, grouped into phases (rule-separated)
# ----------------------------------------------------------------------------
groups = [
    ("Source\ncorpus", "#cfe0f1", [
        ("DeepCAD JSON designs",          215093, "raw corpus"),
        ("Valid CadQuery programs",       213102, "99.1% translate, 0 syntax errors"),
    ]),
    ("Benchmark\nsplit", "#a9cae9", [
        ("DeepCAD official split",        178238, "161,240 / 8,946 / 8,052"),
    ]),
    ("Rendered\n(this work)", "#6fa8d6", [
        ("Rendered subset",                10000, "compute-budget sample"),
        ("Usable (video, program) pairs",   9909, "99.1% usable: 7,924 / 991 / 994"),
    ]),
]
rows = []
for gi, (gname, gcolor, items) in enumerate(groups):
    for it in items:
        rows.append((gi, gname, gcolor, *it))
n = len(rows)

fig, ax = plt.subplots(figsize=(9.6, 5.8))
ax.set_xlim(0, 12); ax.axis("off")
row_h, gap = 1.0, 0.55
ys, y, prev = [], 0.0, None
for gi, *_ in rows:
    if prev is not None and gi != prev:
        y -= gap
    y -= row_h
    ys.append(y)
    prev = gi
ax.set_ylim(y - 0.25, 0.55)
cx, boxspan = 6.4, 5.6
log = {c: np.log10(c) for _, _, _, _, c, _ in rows}
lo, hi = min(log.values()), max(log.values())

# group bands, labels, and separating rules (like the grouped table)
for gi, (gname, gcolor, items) in enumerate(groups):
    idxs = [k for k, r in enumerate(rows) if r[0] == gi]
    top = ys[idxs[0]] + row_h
    bot = ys[idxs[-1]]
    ax.add_patch(plt.Rectangle((0.3, bot - 0.08), 11.4, (top - bot) + 0.16,
                               facecolor=gcolor, alpha=0.20, edgecolor="none", zorder=0))
    ax.text(1.0, (top + bot) / 2, gname, ha="center", va="center",
            fontsize=9.5, weight="bold", color="#1c4a78", linespacing=0.9)
    if gi > 0:
        ax.plot([0.3, 11.7], [top + gap / 2, top + gap / 2],
                color="#9aa7b4", lw=0.9, ls="-", zorder=1)

for k, (gi, gname, gcolor, name, count, note) in enumerate(rows):
    yc = ys[k]
    w = boxspan * (0.62 + 0.38 * (np.log10(count) - lo) / (hi - lo))
    box = FancyBboxPatch((cx - w / 2, yc + 0.15), w, 0.70,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=1.3, edgecolor="#1f3b63", facecolor=gcolor, zorder=2)
    ax.add_patch(box)
    ax.text(cx, yc + 0.60, name, ha="center", va="center", fontsize=10.5,
            weight="bold", color="#0b1c33", zorder=3)
    ax.text(cx, yc + 0.33, f"{count:,}", ha="center", va="center", fontsize=12,
            weight="bold", color="#0b1c33", zorder=3)
    ax.text(cx + boxspan / 2 + 0.45, yc + 0.5, note, ha="left", va="center",
            fontsize=8.2, color="#3a3a3a", style="italic", zorder=3)
    if k < n - 1:
        ax.annotate("", xy=(cx, ys[k + 1] + 0.87), xytext=(cx, yc + 0.11),
                    arrowprops=dict(arrowstyle="-|>", color="#1f3b63", lw=1.5), zorder=1)

ax.set_title("Dataset construction pipeline", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_data_funnel.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_data_funnel.svg", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "fig_data_funnel.png")

# ----------------------------------------------------------------------------
# Figure 2: GT program complexity distributions
# ----------------------------------------------------------------------------
lines = np.load("/tmp/lines.npy")
chars = np.load("/tmp/chars.npy")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.hist(lines, bins=range(10, 100, 4), color="#4878a8", edgecolor="white")
ax.axvline(np.median(lines), color="#b22222", ls="--", lw=1.6,
           label=f"median = {np.median(lines):.0f}")
ax.axvline(lines.mean(), color="#d18b00", ls="--", lw=1.6,
           label=f"mean = {lines.mean():.1f}")
ax.set_xlabel("lines per CadQuery program")
ax.set_ylabel("# programs")
ax.set_title("Program length (lines)", fontsize=11, weight="bold")
ax.legend(fontsize=9, frameon=False)

ax = axes[1]
ax.hist(np.clip(chars, 0, 3000), bins=range(0, 3100, 120),
        color="#5a9367", edgecolor="white")
ax.axvline(np.median(chars), color="#b22222", ls="--", lw=1.6,
           label=f"median = {np.median(chars):.0f}")
ax.axvline(np.percentile(chars, 95), color="#d18b00", ls="--", lw=1.6,
           label=f"p95 = {np.percentile(chars,95):.0f}")
ax.set_xlabel("characters per program (clipped at 3,000)")
ax.set_ylabel("# programs")
ax.set_title("Program size (characters)", fontsize=11, weight="bold")
ax.legend(fontsize=9, frameon=False)

fig.suptitle("Ground-truth CadQuery programs (n = 9,909; 0 syntax errors)",
             fontsize=12, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_script_dist.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_script_dist.svg", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "fig_script_dist.png")

# ----------------------------------------------------------------------------
# Figure 3: example input (orbit frames) -> output (CadQuery program)
# ----------------------------------------------------------------------------
def load_script(obj_id):
    txt = (ROOT / f"deepcad-data/cadquery_py/{obj_id}.py").read_text().splitlines()
    out = []
    for ln in txt:
        s = ln.strip()
        if s.startswith('"""') or s == "" or s == "result = None":
            continue
        out.append(ln.rstrip())
    return "\n".join(out)

def best_frames(obj_id, k=5):
    """Pick the k highest-contrast frames (washed-out/top-down frames have low std)."""
    fd = ROOT / f"stage2b/frames/{obj_id}"
    stds = []
    for i in range(16):
        im = np.asarray(Image.open(fd / f"frame_{i:02d}.jpg").convert("L"), dtype=np.float32)
        stds.append((im.std(), i))
    top = sorted([i for _, i in sorted(stds, reverse=True)[:k]])
    return top

def frame_strip(obj_id, idxs):
    imgs = [Image.open(ROOT / f"stage2b/frames/{obj_id}/frame_{i:02d}.jpg") for i in idxs]
    w, h = imgs[0].size
    strip = Image.new("RGB", (w * len(imgs), h), "white")
    for k, im in enumerate(imgs):
        strip.paste(im, (k * w, 0))
    return np.asarray(strip)

examples = ["0000/00002065", "0002/00021681"]
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.28, wspace=0.06)
for r, obj in enumerate(examples):
    axf = fig.add_subplot(gs[r, 0])
    axf.imshow(frame_strip(obj, best_frames(obj, 5)))
    axf.set_xticks([]); axf.set_yticks([])
    axf.set_ylabel(f"{obj}", fontsize=9, color="#333")
    if r == 0:
        axf.set_title("Input: orbit video  (5 of 16 sampled frames)",
                      fontsize=11, weight="bold")
    axs = fig.add_subplot(gs[r, 1])
    axs.axis("off")
    if r == 0:
        axs.set_title("Output: CadQuery program", fontsize=11, weight="bold")
    axs.text(0.0, 1.0, load_script(obj), fontfamily="monospace", fontsize=6.4,
             va="top", ha="left", color="#0b1c33",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f7fa", ec="#c5cfdb"))
fig.suptitle("Example (video → program) training pairs",
             fontsize=13, weight="bold", y=0.99)
fig.savefig(OUT / "fig_io_example.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_io_example.svg", bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "fig_io_example.png")
