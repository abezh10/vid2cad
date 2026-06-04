# Real eval (sim2real measurement) — Workstream B1

Measures how the synthetic-trained models do on **real phone video** of physical
parts, against the same CadQuery ground truth used in the synthetic test set. The
gap between this and the synthetic numbers *is* the sim2real gap.

## One-time prep
1. Pick parts from the **test split** (`stage2b/test.jsonl`) and 3D-print them —
   their CadQuery is the ground truth, and the GT mesh already exists at
   `stage1_output/<bucket>/<id>/object.stl`.
2. Shoot a short orbit video of each printed part (slow ~360°, part centered).
3. Name each file `<bucket>__<id>.mp4` (double underscore = the `/` in the
   object_id), or pass a `--mapping` JSON/CSV of `video -> <bucket>/<id>`.

## Run
```bash
module load python/3.13.5
cd /scratch/users/abzhong

# 1. frames + self-contained round dir (mirrors the rft_select pattern)
python3 stage3/real_eval/build_real_manifest.py \
    --videos_dir <dir-of-videos> --run_dir real_eval/run1 --n_frames 16

# 2. generate CadQuery from the REAL frames (GPU) — same harness as synthetic.
#    Add best-of-N (--n_samples 4 --temperature 0.8 --top_p 0.95) for the
#    deployment-style number.
python3 stage3/eval.py --model qwen --checkpoint_dir stage3_runs/qwen \
    --config real_eval/run1/config.yaml --out_dir real_eval/run1/eval_qwen

# 3. score with pose alignment (real orientation is arbitrary → cube-rot + ICP)
python3 stage3/real_eval/score_real.py --eval_dir real_eval/run1/eval_qwen
```

`real_geom_summary.json` reports `execution_rate`, `iou_aligned_mean/median`,
`iou_aligned_at_0.5/0.7`, and `chamfer_aligned_median`. Compare these to the
synthetic `geom_summary.json` **for the same object_ids** to quantify the gap.

## Notes
- Use **aligned** IoU here, not identity: unlike the synthetic eval (model learned
  the sim canonical frame), a real part is filmed at an arbitrary pose. `score_real`
  aligns via the 24 cube rotations + ICP refine before scoring.
- To test whether B2 augmentation closes the gap: re-run step 2 with the
  augmentation-trained adapter and compare `iou_aligned_mean` on the same videos.
- GT mesh dir override: `score_real.py --data_dir stage1_output` (default).
