#!/bin/bash
# Launch the BIG run: rank-128 ViT-LoRA on train_big.jsonl (original 7,924 @
# 640x480 + ~20k expansion @ 960x720, mixed-res, max_side-capped), 10 frames,
# 2 epochs, mild sim2real augmentation, with the full eval->geom chain plus the
# high-res test eval. Run AFTER stage2b/train_big.jsonl exists.
#
# Usage: bash stage3/launch_big_training.sh [prompt_variant]   (default|geom)
set -euo pipefail
cd /scratch/users/abzhong
PV="${1:-default}"
NAME="qwen_vit_big"

test -s stage2b/train_big.jsonl || { echo "train_big.jsonl missing — run the render pipeline first"; exit 1; }

TJ=$(sbatch --parsable \
    --export=ALL,NAME=$NAME,RANK=128,ALPHA=256,NFRAMES=10,EPOCHS=2,TRAIN_MANIFEST=stage2b/train_big.jsonl,PROMPT_VARIANT=$PV,MAX_SIDE=768,DL_WORKERS=4,AUGMENT=mild \
    stage3/train_qwen_vit_bal.sbatch)
echo "train: $TJ (rank128 a256 f10 e2 max_side768 prompt=$PV augment=mild)"

# eval on the FROZEN 640x480 test set (comparable to all paper numbers)
EJ=$(sbatch --parsable --dependency=afterok:"$TJ" \
    --export=ALL,NAME=$NAME,NFRAMES=10,PROMPT_VARIANT=$PV \
    stage3/eval_qwen_vit_bal.sbatch)
GJ=$(sbatch --parsable --dependency=afterok:"$EJ" \
    --export=ALL,NAME=$NAME stage3/geom_qwen_vit_sweep.sbatch)

# additional eval on the HIGH-RES re-render of the same frozen test ids
EHJ=$(sbatch --parsable --dependency=afterok:"$TJ" \
    --export=ALL,NAME=$NAME,NFRAMES=10,PROMPT_VARIANT=$PV,TEST_MANIFEST=stage2b/test_hr.jsonl,OUT_SUFFIX=eval_hr,MAX_SIDE=768 \
    stage3/eval_qwen_vit_bal.sbatch)
GHJ=$(sbatch --parsable --dependency=afterok:"$EHJ" \
    --export=ALL,NAME=$NAME,EVAL_SUBDIR=eval_hr stage3/geom_qwen_vit_sweep.sbatch)

echo "eval=$EJ geom=$GJ eval_hr=$EHJ geom_hr=$GHJ"
echo "train=$TJ eval=$EJ geom=$GJ eval_hr=$EHJ geom_hr=$GHJ" > stage3/.big_jobids
