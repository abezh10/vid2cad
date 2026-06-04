# Milestone 3 — Preliminary Results (May 29)
### vid2cad: reconstructing editable CAD from a short orbit video

---

## Slide 0 — Recap / what changed since M2 *(context)*

- **Goal:** given a short orbit video of a single CAD part, output an *editable* CadQuery Python program that rebuilds it — not a mesh or point cloud.
- **Pipeline now built end-to-end:**
  - Stage 0: DeepCAD JSON → CadQuery `.py` (213k programs translated & validated)
  - Stage 1: CadQuery → STL → Blender orbit render → video + frames + depth/cameras
  - Stage 2: depth + poses → fused, normalized point cloud (the *geometric* route)
  - **Stage 2b/3 — the main new development:** fine-tune video-capable VLMs to emit CadQuery **directly from frames**, skipping the point cloud
- **Why the shift:** direct video→code is simpler, preserves construction history/editability, and lets us benchmark several frontier VLMs head-to-head on the same task.
- **Training data (stage2b manifest):** 7,924 train / 991 val / **994 test**, each = 8 sampled frames + ground-truth CadQuery script (mean ~30 lines, ~970 chars, 0 GT syntax errors).

---

## Slide 1 — Results

- **Experimental design:** 4 candidate VLMs, **identical recipe** for a fair comparison — LoRA (r=16, α=32), 3 epochs, effective batch 8, lr 1e-4 cosine, bf16, 8 frames (4 for Kimi), on a single L40S (46 GB). Only ~16M params trained (0.16%).
- **Qwen3-VL-8B — first model fully through the pipeline:**
  - Clean convergence: training loss **0.99 → 0.13** over 2,973 steps. *(fig1)*
  - **Execution rate: 88.9%** greedy — 884 of 994 held-out parts produce CadQuery that runs without error. *(fig2)* **Best-of-N decoding lifts this to 100% on a 100-object matched slice** (greedy 93% → best-of-4 100%, all 7 failures recovered), with no retraining. *(fig5)*
  - **Geometric fidelity (12-object pilot, 11 scored):** mean **IoU 0.50**, median **IoU 0.57**; 6/11 ≥ 0.5, 3/11 ≥ 0.7; median Chamfer 0.081 (unit-normalized). *(fig4)*
- **GLM-4.1V-9B:** trained cleanly (loss 0.98 → 0.15) — currently **re-training** after an eval-formatting fix (see analysis). Eval pending.
- **Kimi-VL-A3B & InternVL3-8B:** training **blocked** by upstream modeling bugs — no checkpoints yet.

| Model | Status | Exec rate | mIoU (n=11) | median IoU | median Chamfer |
|---|---|---|---|---|---|
| **Qwen3-VL-8B** | trained + evaluated | **88.9%** | 0.50 | 0.57 | 0.081 |
| GLM-4.1V-9B-Thinking | re-training (eval pending) | — | — | — | — |
| Kimi-VL-A3B-Thinking | blocked (MoE train assert) | — | — | — | — |
| InternVL3-8B | blocked (forward signature) | — | — | — | — |

---

## Slide 2 — Comparisons / baselines

- **Cross-model (primary comparison):** same data + recipe across 4 VLMs isolates *model* effect. 1 of 4 complete today; loss curves for Qwen and GLM are near-identical, so differences should surface at the **code-validity & geometry** stage, not in training loss. *(fig1)*
- **Approach comparison:** direct video→code (Stage 3) vs. the geometric point-cloud→CAD-Recode route (Stages 1–2). The geometric pipeline is built; a side-by-side on the same test parts is the next baseline to land.
- **Zero-shot reference (planned):** un-fine-tuned base VLMs on the same prompt — needed to quantify how much of the 88.9% comes from fine-tuning vs. the base model's prior CAD knowledge.
- **Honest status:** we do **not** yet have baseline numbers to plot against Qwen — the comparison framework exists, the competing numbers are the immediate next deliverable.

---

## Slide 3 — Analysis / insights

- **Fine-tuned Qwen learns the task well:** ~89% of generations are valid, runnable CadQuery from vision alone — strong evidence the frames carry enough signal to drive program synthesis.
- **Failures are overwhelmingly *code*, not *geometry*:** of the 110 non-executing outputs — **81 SyntaxError, 21 NameError**, only **8 geometry-kernel failures** (OCC `StdFail_NotDone` / null object). *(fig3)* → ~93% of failures are surface-level Python errors, which are far more tractable to fix than wrong-shape reasoning.
- **Root cause of the code failures = greedy *repetition collapse*, not truncation.** The failing generations don't stop because they hit the token cap — they fall into a loop, emitting workplane after workplane (`wp_10`, `wp_13`, …) until cut off. Two pieces of evidence: (1) raising the cap 1024→3072 recovered **0** failures (and slightly regressed); (2) on a 100-object slice, switching greedy→**best-of-N sampling recovered all 7 failures (93%→100%)**. So this is a *decoding-degeneration* problem, fixable without retraining. *(fig5)*
- **Token penalties are the wrong cure — code is token-repetitive.** `repetition_penalty 1.2 + no_repeat_ngram 3` scored **0/100**: penalizing repeats shoves the model off structurally-required tokens (`cq`, `.`, `Vector`, `)`), producing `cq`+TAB+`Vector`, `cq.Direction`, unmatched parens. The fix that works is **plain sampling + a greedy anchor + best-of-N** (no penalties). *(fig5)*
- **Valid ≠ correct:** execution rate measures *code validity*; IoU measures *shape*. Pilot shows the gap — most parts land at IoU 0.5–1.0, but one part ran fine yet scored IoU 0.0 (Chamfer 0.74) — right syntax, wrong solid.
- **Thinking models need careful output handling:** GLM's first eval produced empty parsed code because `<think>…</think>` wrapping wasn't stripped before execution — a harness bug, not a model failure (hence the "malformed_old" run and the re-train).

---

## Slide 4 — Limitations

- **Comparison incomplete:** only Qwen is fully evaluated; GLM re-training; **Kimi & InternVL blocked** (Kimi's MoE gate asserts `not self.training`; InternVL's `forward()` doesn't receive `pixel_values` from our collator).
- **Geometric eval is a 12-object pilot, not the full 994** — IoU/Chamfer are preliminary and high-variance. (The occupancy backend was only just unblocked — earlier runs failed with a missing-module error, so the n=11 numbers are the very first geometric signal.)
- **No baseline numbers yet** (zero-shot VLM, point-cloud→CAD-Recode) — can't yet attribute the gain to fine-tuning or to the direct-generation approach.
- **Best-of-N is so far validated on 100 objects, not the full 994** — the 100%/recovery numbers are a strong proof-of-mechanism, but the full-set best-of-N headline is still pending. Eval is also 8 frames, LoRA-only (no full fine-tune), limited temporal coverage.
- **Code-validity is an easy metric** — the harder, more meaningful target is geometric correctness at scale.

---

## Slide 5 — Next steps

- **Land the real numbers:** run the full 994-object geometric eval (IoU + Chamfer) on Qwen now that the occupancy backend works — replaces the 12-object pilot.
- **Complete the comparison:** finish GLM re-train + eval; **unblock Kimi & InternVL** (patch Kimi's training-mode gate; route `pixel_values` through InternVL's collator) so all 4 models report on the identical test set.
- **Add baselines:** zero-shot base VLMs + the point-cloud→CAD-Recode pipeline, to isolate the value of fine-tuning and of direct video→code.
- **Scale up the decoding fix:** run best-of-N over the full 994-object test set to land the real headline execution rate (100-object slice already shows greedy 93%→100%). The mechanism is settled — repetition collapse, not truncation; token penalties hurt — so this is now an engineering/run-cost step, not a research question.
- **Push selection from "first-executable" to "best-IoU":** with N candidates per part already executing, pick the one with highest volumetric IoU (not just the first that runs) to raise *geometric* fidelity for free. Add more frames and a qualitative gallery of reconstructed vs. ground-truth parts for the final report.

---

### Figures (in `milestone3/`)
- `fig1_training_loss.png` — LoRA fine-tuning loss, Qwen vs. GLM
- `fig2_execution_rate.png` — test-set code-validity, all 4 models
- `fig3_failure_breakdown.png` — Qwen output breakdown / failure modes
- `fig4_geom_iou.png` — per-part volumetric IoU (geometric pilot)
- `fig5_decoding_ablation.png` — decoding ablation (greedy 93% / rep-penalty boN 0% / clean best-of-4 100%) — **Slides 1 & 3**
- `results_table.csv` — the summary table above
