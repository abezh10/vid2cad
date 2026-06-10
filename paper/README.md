# vid2cad — paper submission bundle (CS231n)

**Video → CadQuery via a vision-language model.** Thesis: *the vision encoder is the
bottleneck* — adapting the ViT to the synthetic-CAD render domain is the dominant lever
for shape fidelity.

## Headline result (Qwen3-VL, greedy, n=994 held-out test)

| Model | mIoU (overall) | exec % | median IoU |
|---|---|---|---|
| SFT baseline (frozen ViT) | 0.469 | 88.9 | 0.564 |
| + RFT / STaR | 0.500 | 95.7 | 0.553 |
| + ViT adaptation (rank 16) | 0.527 | 91.4 | 0.655 |
| **+ ViT adaptation (rank 64)** | **0.559** | **93.8** | **0.682** |

Full tables (4-encoder comparison, RFT, 2×2 data×encoder, capacity scaling, depth):
**`results_table.md`**.

## What's in this bundle

```
paper/
  README.md            ← you are here
  CODE_GUIDE.md        ← annotated map of the most important source code
  results_table.md     ← all quantitative results (Tables A–F)
  code/                ← the key source files (verbatim), in repo structure
  figures/
    figure_architecture.{png,svg}   pipeline: video→patches→ViT→LLM→CadQuery
    gt_vs_pred_grid.{png,svg}        GT vs predicted solids, 8 real test objects (IoU 1.0→0.24)
    patchify_anim.mp4                animated patchify→transformer-blocks
    orbit_patches.mp4                real orbit with ViT patch grid overlaid
    orbit_raw.mp4                    clean orbit footage
    videos/                          orbit videos of the 8 GT-vs-pred example objects
                                     (named <bucket>_<id>_iou<score>.mp4)
```

## Figures, at a glance

- **`figure_architecture`** — the method, built from a real example (object 0000/00000908):
  8 orbit frames → 16×16 ViT patches → 27 transformer blocks → MLP connector → LLM →
  CadQuery. LoRA badges mark the frozen-vs-adapted ablation.
- **`gt_vs_pred_grid`** — the results, qualitatively: for 8 test objects spanning
  excellent→failure, the input frame, the ground-truth solid, and the model's predicted
  solid (built by executing its generated CadQuery), each labeled with IoU. Shows the
  model nailing simple prisms/cylinders (IoU≈1.0) and failing on stacked/compound forms.
- **`videos/`** — the actual orbit inputs for those 8 objects, so a reader can see what
  the model saw.

SVGs are editable (Inkscape/Illustrator); every figure is regenerable via
`code/stage3/figures/*.py`.
