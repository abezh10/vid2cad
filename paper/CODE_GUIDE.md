# Source-code guide — vid2cad (CS231n submission)

The most important source, grouped by paper section. Paths are under `paper/code/`
(copied verbatim from the live repo). ★ = files that implement the paper's core claims.

---

## 1. The model & the headline contribution — ViT adaptation  ★

The thesis ("the vision encoder is the bottleneck") is implemented as **which LoRA
target modules are adapted**. One config entry is the whole experiment.

| File | What it is | Look here |
|---|---|---|
| ★ `stage3/config.yaml` | Shared recipe + per-model LoRA targets. | The **`qwen` vs `qwen_vit`** entries: `qwen_vit` adds `qkv, proj, linear_fc1, linear_fc2` — the 27 ViT blocks + merger. *This diff is the experiment.* |
| ★ `stage3/train_qwen.py` (223 ln) | Qwen3-VL trainer. `--model_key` selects frozen-ViT (`qwen`) vs adapted-ViT (`qwen_vit`); `--lora_rank/--n_frames/--epochs` drive the capacity sweep. | `main()` arg parsing + `build_lora(model, mcfg["lora_target_modules"], cfg)`. |
| `stage3/train_common.py` | Config loader, `build_lora`, tokenized dataset, collator, training args — shared by all 4 model trainers. | `build_lora` (PEFT target-module matching). |
| `stage3/data.py` | `VideoCadQueryDataset`: loads N frames (+ optional depth, + optional augmentation) paired with GT CadQuery. | `__getitem__`; the `augment` hook. |

*(The other three trainers — `train_glm/internvl/kimi.py` — are in the repo; they share
this scaffold and are only needed to reproduce the 4-encoder comparison, Table A.)*

## 2. Evaluation & metrics  ★

| File | What it is | Look here |
|---|---|---|
| ★ `stage3/eval.py` | Generate CadQuery from test frames; measure **execution rate**; supports greedy and best-of-N. | generation + exec-in-subprocess. |
| ★ `stage3/eval_geom.py` (195 ln) | Build GT & predicted meshes, normalize, compute **volumetric IoU + Chamfer**. The primary metric. | `build_mesh`, `normalize`, `volumetric_iou`, `chamfer`. |
| `stage3/diag_rotation.py` | 24-cube-rotation diagnostic proving low IoU is genuine shape error, not pose. | `cube_rotations()`. |

## 3. RFT / STaR self-training (data engine)

| File | What it is |
|---|---|
| `stage3/rft_sample.py` | Sample K candidates per train object from the current model. |
| ★ `stage3/rft_select.py` (179 ln) | Keep the highest-IoU candidate (≥0.5) as the new target → curated `train.jsonl`. The STaR loop. |

## 4. Sim2real (Workstream B)

| File | What it is |
|---|---|
| ★ `stage3/augment.py` (241 ln) | Clip-consistent train-time augmentation (photometric + sensor/codec + geometric) to close the render→real gap without re-rendering. |
| `stage3/real_eval/build_real_manifest.py` | Real phone videos → self-contained eval round dir. |
| `stage3/real_eval/score_real.py` | Pose-aligned IoU (24 rotations + ICP) for uncontrolled real capture. |

## 5. The data factory (Stages 0–2b) — how training data is made

| File | Stage | What it is |
|---|---|---|
| `translator/translate.py`, `emit.py` | 0 | DeepCAD JSON construction sequence → runnable CadQuery program. |
| ★ `stage1/render_scene.py` (825 ln) | 1 | Blender orbit renderer with **domain randomization** (Nishita sky, randomized lights/materials, camera shake, ArUco marker). Emits video + depth EXR + cryptomatte masks + camera intrinsics/extrinsics. |
| `stage2/fuse.py`, `normalize.py` | 2 | Multi-view depth → TSDF fusion → normalized point cloud (FPS). |
| `stage2b/sample_frames.py`, `build_manifest.py` | 2b | Sample N frames; build (frames → code) train/val/test manifests. |

## 6. Reproduce the figures

| File | Produces |
|---|---|
| `stage3/figures/make_architecture_figure.py` | `figure_architecture.{png,svg}` |
| `stage3/figures/make_patch_videos.py` | `patchify_anim.mp4`, `orbit_patches.mp4` |
| ★ `stage3/figures/make_gt_vs_pred.py` | `gt_vs_pred_grid.{png,svg}` — GT vs predicted solids from real test outputs. |

---

## Minimal reproduction path (the headline result)

```bash
module load python/3.13.5
# 1. frozen-ViT baseline  -> stage3_runs/qwen        (mIoU 0.469)
python3 stage3/train_qwen.py --model_key qwen
# 2. adapted-ViT          -> stage3_runs/qwen_vit     (mIoU 0.527)
python3 stage3/train_qwen.py --model_key qwen_vit
# 3. capacity sweep       -> rank 64 is best          (mIoU 0.559)
python3 stage3/train_qwen.py --model_key qwen_vit --lora_rank 64 --lora_alpha 128 --output_name qwen_vit_r64
# 4. score any of them
python3 stage3/eval.py --model qwen --checkpoint_dir stage3_runs/<name> --config stage3/config.yaml --out_dir stage3_runs/<name>/eval
python3 stage3/eval_geom.py --eval_dir stage3_runs/<name>/eval --data_dir stage1_output --workers 16
```

All numbers: `paper/results_table.md`.
