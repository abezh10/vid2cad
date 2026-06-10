"""Qualitative reconstruction figure: input frame -> GT solid -> predicted solid.
Renders real CadQuery solids (executed via cadquery, rendered via trimesh+mpl)
for a success, a partial, and a valid-but-wrong failure case from the best model.
"""
import json, signal, tempfile, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import cadquery as cq
import trimesh
warnings.filterwarnings("ignore")

ROOT = Path("/scratch/users/abzhong")
RUN = ROOT / "stage3_runs/qwen_vit_r64/eval"
OUT = ROOT / "stage3/figures/data"; OUT.mkdir(parents=True, exist_ok=True)

iou = {}
for l in open(RUN / "geom_per_object.jsonl"):
    d = json.loads(l)
    if d.get("iou") is not None:
        iou[d["object_id"]] = d["iou"]
po = {json.loads(l)["object_id"]: json.loads(l) for l in open(RUN / "per_object.jsonl")}
ok = {o: iou[o] for o in iou if o in po and str(po[o].get("executed")) == "True"}

def near(t):  # closest executed object to target IoU
    return min(ok.items(), key=lambda x: abs(x[1] - t))[0]

examples = [("Success", near(0.92)), ("Partial", "0072/00724057"),
            ("Valid code,\nwrong solid", "0002/00021948")]

class TO(Exception): pass
def _h(s, f): raise TO()
signal.signal(signal.SIGALRM, _h)

def code_to_mesh(code):
    ns = {}
    signal.alarm(25)
    try:
        exec(code, ns)
        res = ns.get("result")
        if res is None:
            return None
        f = tempfile.mktemp(suffix=".stl")
        cq.exporters.export(res, f)
    except Exception:
        return None
    finally:
        signal.alarm(0)
    m = trimesh.load(f, force="mesh")
    return m

def draw(ax, mesh, color):
    if mesh is None or len(mesh.faces) == 0:
        ax.text2D(0.5, 0.5, "(no solid)", ha="center"); ax.axis("off"); return
    v = mesh.vertices - mesh.vertices.mean(0)
    v = v / (np.abs(v).max() + 1e-9)
    ax.add_collection3d(Poly3DCollection(v[mesh.faces], facecolor=color,
                        edgecolor="#33486b", linewidths=0.15, alpha=1.0))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(26, -58); ax.axis("off")

def best_frame(obj):
    fd = ROOT / f"stage2b/frames/{obj}"
    stds = [(np.asarray(Image.open(fd / f"frame_{i:02d}.jpg").convert("L"),
            dtype=np.float32).std(), i) for i in range(16)]
    return ROOT / f"stage2b/frames/{obj}/frame_{max(stds)[1]:02d}.jpg"

fig = plt.figure(figsize=(8.6, 8.2))
gs = fig.add_gridspec(3, 3, wspace=0.05, hspace=0.12)
cols = ["Input frame", "Ground-truth solid", "Predicted solid (ours)"]
for r, (label, obj) in enumerate(examples):
    axi = fig.add_subplot(gs[r, 0])
    axi.imshow(Image.open(best_frame(obj))); axi.set_xticks([]); axi.set_yticks([])
    axi.set_ylabel(f"{label}\nIoU={ok.get(obj,0):.2f}", fontsize=10, weight="bold")
    if r == 0: axi.set_title(cols[0], fontsize=11, weight="bold")
    axg = fig.add_subplot(gs[r, 1], projection="3d")
    draw(axg, code_to_mesh(po[obj]["ground_truth"]), "#cdd8e6")
    if r == 0: axg.set_title(cols[1], fontsize=11, weight="bold")
    axp = fig.add_subplot(gs[r, 2], projection="3d")
    draw(axp, code_to_mesh(po[obj]["code"]), "#9fb8d6")
    if r == 0: axp.set_title(cols[2], fontsize=11, weight="bold")

fig.suptitle("Qualitative reconstructions (Qwen3-VL, ViT-adapted rank 64)",
             fontsize=13, weight="bold", y=0.97)
fig.savefig(OUT / "fig_qualitative.png", dpi=160, bbox_inches="tight")
fig.savefig(OUT / "fig_qualitative.svg", bbox_inches="tight")
print("wrote", OUT / "fig_qualitative.png", "| examples:",
      [(l.split(chr(10))[0], o, round(ok.get(o,0),2)) for l, o in examples])
