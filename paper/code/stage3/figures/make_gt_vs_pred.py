#!/usr/bin/env python3
"""Ground-truth vs predicted geometry, from the BEST model's real test predictions.

For a spread of test objects (excellent -> failure), renders:
   [ input frame ]  [ GT solid ]  [ predicted solid ]    labeled with IoU
The predicted solid is built by executing the model's generated CadQuery; GT is the
Stage-1 STL. Both normalized to the unit cube and shown from the same viewpoint.

Output: stage3/figures/gt_vs_pred_grid.png (+ .svg) and per-object panels.
Usage:  python3 stage3/figures/make_gt_vs_pred.py
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

sys.path.insert(0, "/scratch/users/abzhong/stage3")
from eval_geom import build_mesh, normalize, extract_code

EVAL = Path("/scratch/users/abzhong/stage3_runs/qwen_vit_r64/eval")
OUT = Path("/scratch/users/abzhong/stage3/figures")
DATA = Path("/scratch/users/abzhong/stage1_output")
FRAMES = Path("/scratch/users/abzhong/stage2b/frames")

# (object_id, IoU, tier) — chosen across the quality range from geom_per_object.jsonl
EXAMPLES = [
    ("0002/00027709", 1.000, "excellent"),
    ("0003/00035415", 1.000, "excellent"),
    ("0087/00879546", 0.900, "good"),
    ("0023/00234584", 0.898, "good"),
    ("0055/00550771", 0.597, "medium"),
    ("0072/00725658", 0.596, "medium"),
    ("0001/00010900", 0.246, "failure"),
    ("0050/00507636", 0.242, "failure"),
]
TIER_COLOR = {"excellent": "#1f8a3b", "good": "#3b82a6",
              "medium": "#c8902a", "failure": "#b03030"}
GT_C = (0.55, 0.60, 0.68)
PR_C = (0.42, 0.55, 0.85)


def load_codes():
    m = {}
    for line in open(EVAL / "per_object.jsonl"):
        r = json.loads(line)
        m[r["object_id"]] = r.get("code") or extract_code(r.get("generated", ""))
    return m


def render_mesh(ax, mesh, base_color, elev=24, azim=-58):
    if mesh is None or len(mesh.faces) == 0:
        ax.text(0.5, 0.5, 0.5, "no solid", ha="center", color="red");
        ax.set_xlim(-.5,.5); ax.set_ylim(-.5,.5); ax.set_zlim(-.5,.5); ax.axis("off"); return
    tris = mesh.vertices[mesh.faces]
    n = mesh.face_normals
    light = np.array([0.3, 0.4, 0.85]); light /= np.linalg.norm(light)
    inten = 0.4 + 0.6 * np.clip(n @ light, 0, 1)
    cols = np.clip(np.array(base_color)[None, :] * inten[:, None], 0, 1)
    pc = Poly3DCollection(tris, facecolors=cols, edgecolors=(0, 0, 0, 0.12), linewidths=0.2)
    ax.add_collection3d(pc)
    ax.set_xlim(-.5, .5); ax.set_ylim(-.5, .5); ax.set_zlim(-.5, .5)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.axis("off")


def main():
    codes = load_codes()
    nrow = len(EXAMPLES)
    fig = plt.figure(figsize=(8.6, 2.45 * nrow), dpi=170)
    fig.suptitle("Ground truth vs. prediction — best model (rank-64 ViT adaptation), real test outputs",
                 y=0.997, fontsize=12, fontweight="bold")
    for i, (oid, iou, tier) in enumerate(EXAMPLES):
        # input frame
        ax0 = fig.add_subplot(nrow, 3, 3 * i + 1)
        fs = sorted(glob.glob(str(FRAMES / oid / "*.jpg")))
        if fs:
            ax0.imshow(plt.imread(fs[len(fs)//2]))
        ax0.axis("off")
        ax0.text(-0.06, 0.5, f"{tier.upper()}\nIoU {iou:.2f}", transform=ax0.transAxes,
                 ha="right", va="center", fontsize=9.5, fontweight="bold",
                 color=TIER_COLOR[tier], rotation=0)
        if i == 0: ax0.set_title("input (1 of 8 frames)", fontsize=10)

        # GT
        ax1 = fig.add_subplot(nrow, 3, 3 * i + 2, projection="3d")
        try:
            gt = normalize(trimesh.load(DATA / oid / "object.stl", force="mesh"))
            render_mesh(ax1, gt, GT_C)
        except Exception as e:
            ax1.text2D(0.5, 0.5, "GT load failed", ha="center"); ax1.axis("off")
        if i == 0: ax1.set_title("ground truth", fontsize=10)

        # prediction
        ax2 = fig.add_subplot(nrow, 3, 3 * i + 3, projection="3d")
        try:
            pred = build_mesh(codes.get(oid, ""))
            render_mesh(ax2, normalize(pred) if pred is not None else None, PR_C)
        except Exception:
            render_mesh(ax2, None, PR_C)
        if i == 0: ax2.set_title("prediction (model CadQuery)", fontsize=10)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.965, bottom=0.005,
                        wspace=0.02, hspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "gt_vs_pred_grid.png", dpi=170, facecolor="white")
    fig.savefig(OUT / "gt_vs_pred_grid.svg", facecolor="white")
    print("wrote", OUT / "gt_vs_pred_grid.png")
    print("wrote", OUT / "gt_vs_pred_grid.svg")


if __name__ == "__main__":
    main()
