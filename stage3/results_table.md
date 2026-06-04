# vid2cad — results (paper-ready)

All numbers are greedy decode on the **frozen 994-object test split** (`stage2b/test.jsonl`),
unless noted. Meshes are unit-normalized (centered, scaled to a unit cube) before scoring.

## Metric definitions (read first)

- **Execution rate** — fraction of the 994 generated CadQuery programs that run and
  produce a valid solid. Measures *code validity*, not shape correctness.
- **Volumetric IoU** — intersection-over-union of the generated solid vs. the
  ground-truth solid (voxelized). The primary *shape-fidelity* metric.
  - **mIoU (overall)** — mean IoU over **all 994** objects, with execution/build
    failures counted as **0**. Fixed denominator → **comparable across models. PRIMARY METRIC.**
  - **mIoU (scored)** — mean IoU over only the objects that executed *and* scored.
    Denominator varies with execution rate, so it is **not** comparable across models
    and inflates models that fail on hard parts. Reported for reference only.
- **IoU@0.5 / @0.7** — count of test objects whose IoU clears 0.5 / 0.7 (out of 994).
- **Chamfer (median)** — median symmetric Chamfer distance on unit-normalized surfaces
  (lower = better); complements IoU (captures surface error where IoU saturates).

## Table A — Vision encoder comparison (4 VLMs, identical LoRA recipe)

Identical recipe: LoRA r=16/α=32, 3 epochs, eff. batch 8, lr 1e-4 cosine, bf16,
8 frames (Kimi 4 + 4-bit QLoRA). **The vision encoder is the main thing that differs**
— but so do frame count / quantization / LoRA coverage, so this table is
*observational*; the controlled test is Table C.

| Model | Vision encoder (ViT) | Frames | Exec % | **mIoU (overall)** | mIoU (scored) | median | IoU≥0.5 | IoU≥0.7 | Chamfer med |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-VL-8B** | Qwen3-VL ViT (27 blk, p16, native-res, 2D-RoPE) | 8 | 88.9 | **0.469** | 0.533 | 0.564 | 488 | 326 | 0.059 |
| InternVL3-8B | InternViT (1 tile/frame = 256 tok) | 8 | 95.5 | 0.447 | 0.473 | 0.483 | 448 | 251 | 0.075 |
| GLM-4.1V-9B | ViT (EVA/AIMv2-lineage), attn-only LoRA | 8 | 90.0 | 0.410 | 0.463 | 0.450 | 405 | 246 | 0.083 |
| Kimi-VL-A3B | MoonViT (native-res, SigLIP-init) | 4 | 92.7 | 0.220 | 0.246 | 0.173 | 122 | 55 | 0.231 |

**Takeaways:** (1) Qwen3-VL's native-resolution ViT wins on shape fidelity.
(2) Execution rate ≠ fidelity — InternVL has the highest exec (95.5%) but lower IoU
than Qwen: the models write runnable code that builds the *wrong* solid, so geometry
(not syntax) is the bottleneck.

## Table B — RFT / STaR self-training (Qwen3-VL, round 1)

Rejection-sampling fine-tuning: sample K candidates per train object, keep the
highest-IoU one (≥0.5) as the new target, retrain. Curated set = 5,579 of 7,924 objects.

| | Baseline (SFT) | **RFT round 1** | Δ |
|---|---|---|---|
| Execution rate | 88.9% | **95.7%** | **+6.8 pts** |
| **mIoU (overall)** | 0.469 | **0.500** | **+0.031** |
| mIoU (scored) | 0.533 | 0.526 | −0.007 (artifact*) |
| n_scored (denominator) | 874 | 944 | +70 |
| IoU≥0.5 | 488 | **517** | +29 |
| IoU≥0.7 | 326 | **353** | +27 |
| Chamfer median | 0.059 | 0.058 | ~flat |

*The scored-mIoU dip is a **denominator artifact**: RFT makes 70 additional (harder)
parts execute and enter the scored pool, lowering the per-part mean even though overall
performance rises. The fixed-denominator overall mIoU (+0.031) and the threshold counts
(+29 @0.5, +27 @0.7) are the honest read.

**Mechanism:** STaR improves *coverage and code validity* (more parts solved, more
cross the quality bars) but **not peak per-part fidelity** (scored-mIoU flat). The
residual shape error on already-solved parts needs a different lever → Table C.

## Table C — Controlled ViT-adaptation ablation (the thesis)  ✅

`qwen_vit` is byte-identical to the Qwen baseline (same data/recipe/LLM-LoRA) **except
LoRA also adapts the 27-block vision transformer** (qkv, proj, linear_fc1, linear_fc2
+ merger = 117 vision layers). Single-variable test of: *does adapting the ViT to the
synthetic-CAD render domain raise shape fidelity?*  **Answer: yes, substantially.**

| | Frozen ViT (baseline) | **Adapted ViT (qwen_vit)** | Δ |
|---|---|---|---|
| Execution rate | 88.9% | **91.4%** | +2.5 pts |
| **mIoU (overall)** | 0.469 | **0.527** | **+0.058** |
| mIoU (scored) | 0.533 | **0.584** | **+0.050** |
| median IoU (scored) | 0.564 | **0.655** | **+0.091** |
| IoU≥0.5 | 488 | **563** | +75 |
| IoU≥0.7 | 326 | **416** | +90 |
| Chamfer median | 0.059 | **0.044** | −0.015 (better) |

**This is the cleanest result in the project.** Unlike RFT, the *scored* mIoU rose
(+0.050) **even though execution rate also rose** — so it is **not** a denominator
artifact; it is a genuine per-part fidelity gain. The median jumped +0.091 — the
typical part is markedly better. Adapting the ViT to the render domain (vs. its
natural-photo pretraining) lets it attend to the edges/faces/symmetry that determine
the solid.

## Headline comparison (Qwen3-VL, greedy, n=994)

| Intervention | Exec % | **mIoU (overall)** | mIoU (scored) | median | IoU≥0.7 | Chamfer med | what it buys |
|---|---|---|---|---|---|---|---|
| SFT baseline (frozen ViT) | 88.9 | 0.469 | 0.533 | 0.564 | 326 | 0.059 | — |
| + RFT / STaR (data engine) | **95.7** | 0.500 | 0.526 | 0.553 | 353 | 0.058 | **coverage + validity** |
| + ViT adaptation (qwen_vit) | 91.4 | **0.527** | **0.584** | **0.655** | **416** | **0.044** | **per-part fidelity** |

**The two interventions are complementary by mechanism** — RFT broadens *which* parts
the model can produce (exec ↑, more parts solved), ViT-adaptation deepens *how well* it
produces them (scored/median IoU ↑). They attack different terms, so **combining them
(RFT-curated targets + ViT-adapted encoder) is the obvious next run and likely additive.**

## Table D — Sim2real gap (real phone video)  ⏳ PENDING

Real orbit videos of 3D-printed test-set parts, scored with **pose-aligned** IoU
(24 cube rotations + ICP, since real capture orientation is uncontrolled). Gap =
synthetic mIoU − real mIoU on the same parts. Compare frozen vs augmentation-trained
adapters (Workstream B2). Awaiting captured videos.
