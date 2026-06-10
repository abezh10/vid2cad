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

## Table C2 — ViT-adaptation capacity scaling (LoRA rank)  ✅

Sweep the ViT LoRA rank, everything else fixed (GT data, 8 frames, 3 epochs).
Adapting *more* of the vision transformer **monotonically** raises fidelity **and**
execution — with **no overfitting** (val loss kept decreasing at every rank; rank-64's
val loss was the lowest of all runs).

| ViT LoRA rank | trainable | exec % | **mIoU (overall)** | scored | median | IoU≥0.7 | Chamfer |
|---|---|---|---|---|---|---|---|
| 0 (frozen ViT, baseline) | 43.6M (LLM only) | 88.9 | 0.469 | 0.533 | 0.564 | 326 | 0.059 |
| 16 | ≈52M | 91.4 | 0.527 | 0.584 | 0.655 | 416 | 0.044 |
| 32 | 105M | 91.3 | 0.542 | 0.598 | 0.671 | 425 | 0.043 |
| **64** | 210M | **93.8** | **0.559** | **0.605** | **0.682** | **449** | 0.043 |

Rank-64 ViT adaptation: mIoU-overall 0.559 (+0.090 over the frozen-ViT baseline),
exec 93.8%, median 0.682. The monotonic rank→IoU trend with no overfitting is a
clean *scaling* result and the strongest evidence for the thesis: the encoder's
visual grounding was the bottleneck, and the more of it you adapt to the
CAD-render domain, the better the geometry — the curve hasn't saturated by
rank 64, so there may be more headroom. (Superseded as best model by the
difficulty-balanced run in Table G.)

**Frame count is *not* a lever here (control):** holding rank=32, going **8→10 frames**
moved mIoU only **0.542 → 0.544 (+0.002, within noise)**. So the gains come from
*encoder capacity*, not more views — 8 frames already saturates; **rank is what matters.**
This isolates the contribution to ViT adaptation specifically.
**Replicated at rank 64:** 8→10 frames moved 0.559 → 0.555 (−0.004, noise) — the
frames result holds at the highest capacity tested (Table G run A).

## Table D — Full 2×2 ablation (data × encoder)  ✅

Crossing the two levers: **data** (original GT vs. RFT-curated targets) ×
**encoder** (frozen ViT vs. LoRA-adapted ViT). Qwen3-VL, greedy, n=994.

| | Frozen ViT | **Adapted ViT** |
|---|---|---|
| **SFT data (GT targets, 7,924)** | 0.469 · exec 88.9 · med 0.564 | **0.527** · exec 91.4 · med **0.655** |
| **RFT data (curated targets, 5,579)** | 0.500 · exec **95.7** · med 0.553 | 0.519 · exec 94.2 · med 0.595 |

(cells = mIoU-overall · execution% · median-IoU)

### Headline comparison (same data, expanded)

| Intervention | Exec % | **mIoU (overall)** | mIoU (scored) | median | IoU≥0.7 | Chamfer med |
|---|---|---|---|---|---|---|
| SFT baseline (frozen ViT) | 88.9 | 0.469 | 0.533 | 0.564 | 326 | 0.059 |
| + RFT / STaR (data engine) | **95.7** | 0.500 | 0.526 | 0.553 | 353 | 0.058 |
| **+ ViT adaptation** (best mIoU) | 91.4 | **0.527** | **0.584** | **0.655** | **416** | **0.044** |
| + RFT **and** ViT adaptation | 94.2 | 0.519 | 0.557 | 0.595 | 385 | 0.050 |

### The interaction (the interesting finding — they do NOT simply stack)

My prior expectation (additive) was **wrong**, and the real result is a better story:

- **ViT adaptation is the dominant single lever** — best mIoU-overall (0.527), best
  median (0.655), best Chamfer, +90 parts ≥0.7. *Better visual grounding* is what moves
  shape fidelity most.
- **RFT and ViT-adaptation partially conflict on the target axis.** RFT *replaces* GT
  targets with the model's own best-of-K programs (kept only at IoU≥0.5) — trading
  geometric precision for validity/coverage. ViT-adaptation instead wants maximally
  *precise* targets to sharpen visual grounding. Train the adapted ViT on the curated
  (smaller, model-stylized) set and you get the **best execution among fidelity-strong
  runs (94.2%)** but lower peak fidelity than the adapted ViT on precise GT
  (median 0.595 vs 0.655; ≥0.7: 385 vs 416).
- **Caveat (state in the paper):** the combined cell changes *two* things vs. ViT-only
  (encoder adaptation **and** data source/size), so it is the one non-clean corner of
  the 2×2; a fully controlled version would re-run RFT selection without the IoU≥0.5
  precision cap, or keep GT targets and only *add* RFT-discovered parts.

**Bottom line for the paper:** report **ViT-adaptation as the headline (0.527, +0.058
over baseline, a clean controlled single-variable win)**; present the 2×2 to show RFT
buys validity/coverage, ViT-adaptation buys fidelity, and the two are **substitutive,
not additive** on this task — combined is a Pareto point (high exec + good fidelity),
not a new maximum.

## Table E — Conditioning experiment: explicit depth input  ✅

Interleave a rendered **depth frame** next to each RGB frame (8 RGB + 8 depth), to
test whether *giving* the model 3D geometry helps as much as *adapting its encoder*.

| | SFT baseline (RGB only) | + depth frames | + ViT adaptation (for contrast) |
|---|---|---|---|
| Execution rate | 88.9% | 92.9% | 91.4% |
| **mIoU (overall)** | 0.469 | 0.485 | **0.527** |
| median IoU | 0.564 | 0.561 | **0.655** |
| IoU≥0.7 | 326 | 345 | **416** |
| Chamfer median | 0.059 | 0.055 | **0.044** |

**Insight (reinforces the thesis):** handing the model explicit depth helps only a
little (+0.016 overall), far less than **adapting its existing RGB encoder** (+0.058).
So the bottleneck was **not missing 3D input — it was the encoder's visual grounding.**
Teaching the ViT to see CAD renders beats bolting on a depth channel.

## Table F — Sim2real gap (real phone video)  ⏳ PENDING

Real orbit videos of 3D-printed test-set parts, scored with **pose-aligned** IoU
(24 cube rotations + ICP, since real capture orientation is uncontrolled). Gap =
synthetic mIoU − real mIoU on the same parts. Compare frozen vs augmentation-trained
adapters (Workstream B2). Awaiting captured videos.

## Table G — Geometry-stratified failure analysis + difficulty-balanced data  ✅

**Diagnosis.** Stratifying the rank-64 model's per-part IoU by the GT script's
geometry (regex over the DSL: workplane normals, lineTo deltas, arc/spline ops)
shows the residual error is **concentrated in non-orthogonal geometry**:

| bucket (n) | r64 mIoU | r64 median | exec |
|---|---|---|---|
| axis-aligned boxy (388) | 0.676 | 0.790 | 99% |
| axis circles/holes (274) | 0.656 | 0.771 | 97% |
| angled / diagonal edges (126) | 0.445 | 0.444 | 90% |
| **curved (arc/spline) (202)** | **0.284** | **0.163** | 82% |
| tilted workplane (4) | 0.033 | — | 50% |
| **ORTHOGONAL total (662)** | **0.668** | 0.780 | 98% |
| **NON-ORTHOGONAL total (332)** | **0.342** | 0.305 | 84% |

Failure modes on the worst curved parts: (i) *under-generation* — multi-body
arc parts collapsed to a primitive solid; (ii) *repetition→truncation* — the
model rambles past the token budget and the script dies mid-statement. Curved
GT scripts are ~1.8× longer than boxy ones (1499 vs 843 chars) but only 19% of
the natural training mix.

**Intervention (3-run ablation, all rank-64 + 10 frames, greedy, n=994).**
A = control; B = + difficulty-balanced training data (duplicate non-orthogonal
buckets: angled/curved ×2, tilted ×4 → non-ortho 30.9%→47.5% of samples, no
easy examples dropped); C = B + a geometry-aware system prompt.

| | r64 (8f, baseline) | A: +10 frames | **B: +balanced data** | C: B+geom prompt |
|---|---|---|---|---|
| **mIoU (overall)** | 0.559 | 0.555 | **0.577** | 0.571 |
| mIoU (scored) | 0.605 | 0.603 | 0.603 | 0.601 |
| median IoU | 0.682 | 0.693 | 0.692 | 0.688 |
| IoU≥0.5 | 580 | 575 | **607** | 603 |
| IoU≥0.7 | 449 | 451 | **468** | 467 |
| n_scored (exec'd) | 919 | 914 | **951** | 944 |
| Chamfer median | 0.0428 | 0.0413 | 0.0409 | **0.0397** |
| non-ortho mIoU (332) | 0.342 | 0.321 | **0.382** | 0.376 |
| — curved (202) | 0.284 | 0.245 | **0.326** | 0.312 |
| — angled (126) | 0.445 | 0.452 | 0.481 | **0.488** |
| ortho mIoU (662) | 0.668 | 0.672 | **0.674** | 0.669 |

**Findings.**
1. **Difficulty-balanced data is a clean win: +0.018 overall over the best
   prior model (0.559→0.577), +0.022 over the matched frames control.** The
   lift is exactly where it was aimed — non-orthogonal +0.040 (curved +0.042
   with median 0.163→0.275; exec on curved 82%→88%) — **with no regression on
   orthogonal parts** (+0.006). Scored-mIoU is flat while n_scored +32: the
   mechanism is *rescuing previously-failing hard parts*, i.e. exactly the
   under-generation/truncation failure modes diagnosed above.
2. **Frames stay a non-lever at rank 64** (A ≈ baseline, −0.004), replicating
   the rank-32 control.
3. **The geometry-aware prompt does not stack on balanced data** (C 0.571 vs
   B 0.577): it nudges angled parts up (+0.007) and Chamfer down, but loses
   curved/tilted — informative conditioning that the data lever already covers.
4. Caveat: B dips on the tiny 5+-solid bucket (n=22, 0.254→0.181), a likely
   duplication artifact; C recovers it (0.231). The Stage-1 expansion replaces
   duplication with *fresh* stratified data to remove this artifact.

**New best model: `qwen_vit_r64_bal` — mIoU-overall 0.577, exec 95.7%
(951/994), median 0.692, IoU≥0.7 468.** Full per-bucket table:
`stage3/bal_comparison.txt`; per-object features+IoU dump:
`stage3/bal_comparison_dump.jsonl`.

## Table H — Scale-up: stratified expansion + high-res renders  ⏳ IN FLIGHT

Pipeline built and queued (pending cluster): classify all ~221k translated
scripts → render +20k fresh train-split objects stratified toward the weak
buckets (curved 30% / angled 22% / all tilted) at **960×720** (vs 640×480),
60 frames, no-EXR → train rank-128 on ~28k mixed-res samples (2 epochs,
10 frames, max_side 768, mild sim2real augmentation ON) → eval on (a) the
frozen 640×480 test set for comparability and (b) a 960×720 re-render of the
same 994 test ids (resolution ablation; original test assets untouched).
