# vid2cad

Reconstruct a parametric CAD model from a short orbit **video** of a part.

The pipeline turns the DeepCAD corpus into rendered orbit videos, then fine-tunes
vision-language models (VLMs) to translate those videos back into executable
[CadQuery](https://github.com/CadQuery/cadquery) Python scripts. Reconstruction
quality is measured geometrically (volumetric IoU + Chamfer distance) against the
ground-truth solid. The longer-term target is **sim2real** reconstruction of real,
mechanical/manufactured parts captured with an ArUco rig.

```
DeepCAD JSON ──▶ CadQuery .py ──▶ STL ──▶ Blender orbit render ──▶ video + depth + masks + camera poses
   (Stage 0)        (Stage 0)              (Stage 1)
                                                │
                          ┌─────────────────────┴───────────────────────┐
                          ▼                                              ▼
              TSDF point-cloud fusion                        sampled RGB / depth frames
                  (Stage 2)                                        (Stage 2b)
                          │                                              │
                          └──────────────▶ VLM fine-tune ◀──────────────┘
                                       video → CadQuery
                                          (Stage 3)
```

## Pipeline stages

### Stage 0 — Translation (`translator/`, `run_all.sh`)
DeepCAD's ~215k JSON CAD designs are translated into executable CadQuery `.py`
scripts (`run_translation.py`), then validated by executing each in a subprocess
and confirming it produces a valid solid (`run_validation.py`, with a slow-retry
pass for timeouts). Output: `deepcad-data/cadquery_py/<bucket>/<id>.py`.

### Stage 1 — Rendering (`stage1/`)
Each CadQuery script is executed to an STL, placed in a scene on an ArUco-marked
ground plane, and rendered as a 60-frame **orbit video** with Blender EEVEE.
Per object we save `video.mp4`, per-frame **depth EXRs**, **mask EXRs**, and
`cameras.json` (OpenCV-convention intrinsics + world-to-camera poses).
Run as a Slurm array job over a 10k-object subset.
*Unit note:* all `_mm` fields are Blender scene units (1 unit = 1000 real mm).

### Stage 2 — Point-cloud fusion (`stage2/`)
Depth maps + masks + camera poses are fused via TSDF integration into a metric
point cloud, then centered/scaled to `[-0.5, 0.5]³` and downsampled (8192- and
256-point PLYs) for CAD-Recode-style point→CAD conditioning. Masks isolate the
part from the ground/marker/background. Output written next to each object.

### Stage 2b — Frame & manifest prep (`stage2b/`)
Samples a fixed set of RGB frames from each orbit video into JPGs, builds the
`train/validation/test.jsonl` manifests Stage 3 consumes, and (for the depth
experiment) produces TURBO-colormapped **depth frames** aligned to the RGB
frames (`make_depth_frames.py`).

### Stage 3 — VLM fine-tuning & evaluation (`stage3/`)
LoRA-fine-tunes VLMs (Qwen3-VL-8B, GLM-4.1V, InternVL3-8B, Kimi-VL) on
`(orbit frames → CadQuery script)` pairs. `eval.py` runs the held-out test set,
executes each predicted script, and `eval_geom.py` scores **volumetric IoU +
Chamfer** (scale-invariant, no rotation alignment) against the ground-truth solid.

Experiment tracks under Stage 3:
- **Depth conditioning** — interleave depth frames with RGB (`config_depth.yaml`).
- **RFT** (rejection-sampling self-distillation) — sample K candidates per object,
  keep the best-IoU executable ones, and SFT on them (`rft_sample.py` →
  `rft_select.py` → retrain).
- **ViT adaptation** (`qwen_vit`) — LoRA also adapts the vision transformer, not
  just the LLM. This is the strongest single lever so far: it raises *per-part
  shape fidelity*, not just execution rate. See `stage3/results_table.md`.

## Running it

The pipeline runs on Stanford FarmShare with Slurm-submitted array jobs.
Quick version (Stanford FarmShare):

```bash
module load python/3.13.5          # required before any python3/pip call
# Stage 1 (render a subset)
bash stage1/submit_stage1_subset.sh --manifest stage1_output/subset_manifest.txt \
    --output_dir stage1_output --aruco_marker aruco_marker.png
# Stage 2 (point clouds)
bash stage2/submit_stage2.sh --data_dir stage1_output
# Stage 3 (train + eval) — e.g. the ViT-adaptation run
sbatch stage3/train_qwen_vit.sbatch
```

Results and per-table analysis live in `stage3/results_table.md`.
