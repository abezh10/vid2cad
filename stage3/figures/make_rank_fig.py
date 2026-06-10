"""ViT LoRA-rank capacity scaling figure for the Experiments section."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/scratch/users/abzhong/stage3/figures/data"); OUT.mkdir(parents=True, exist_ok=True)
ranks   = [0, 16, 32, 64]
overall = [0.469, 0.527, 0.542, 0.559]
median  = [0.564, 0.655, 0.671, 0.682]

fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.plot(ranks, median,  "-o", color="#c8531b", lw=2, ms=7, label="median IoU (scored)")
ax.plot(ranks, overall, "-s", color="#1f3b63", lw=2, ms=7, label="mIoU (overall, n=994)")
for x, y in zip(ranks, overall): ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                              xytext=(0, -14), ha="center", fontsize=7.5, color="#1f3b63")
for x, y in zip(ranks, median):  ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                                              xytext=(0, 8), ha="center", fontsize=7.5, color="#c8531b")
ax.axvspan(-3, 1, color="#dddddd", alpha=0.5)
ax.text(0, 0.50, "frozen\nViT", ha="center", fontsize=8, color="#555")
ax.annotate("", xy=(64, 0.559), xytext=(0, 0.469),
            arrowprops=dict(arrowstyle="-", ls=":", color="#888"))
ax.text(40, 0.498, "+0.090 overall mIoU,\nno overfitting", fontsize=8, color="#444")
ax.set_xlabel("ViT LoRA rank  (0 = frozen vision encoder)")
ax.set_ylabel("IoU")
ax.set_xticks(ranks)
ax.set_title("Adapting more of the vision encoder\nmonotonically improves shape fidelity",
             fontsize=10.5, weight="bold")
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_ylim(0.44, 0.71); ax.set_xlim(-4, 70)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fig_rank_scaling.png", dpi=170, bbox_inches="tight")
fig.savefig(OUT / "fig_rank_scaling.svg", bbox_inches="tight")
print("wrote", OUT / "fig_rank_scaling.png", "+ .svg")
