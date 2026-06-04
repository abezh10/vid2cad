#!/bin/bash
#SBATCH --job-name=rft_sel
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage3/rft_select_%j.out
#SBATCH --error=logs/stage3/rft_select_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# RFT selection (CPU). Scores every candidate by IoU vs GT, keeps the best per
# object, and writes the curated round dir (train.jsonl + config.yaml). Run
# after all rft_sample.sbatch array tasks finish.
#
# Env overrides: MODEL ROUND_DIR MIN_IOU LIMIT

set -euo pipefail
MODEL="${MODEL:-qwen}"
ROUND_DIR="${ROUND_DIR:-stage3_rft/$MODEL/round1}"
MIN_IOU="${MIN_IOU:-0.5}"
LIMIT="${LIMIT:-0}"

echo "=== rft select model=$MODEL round=$ROUND_DIR min_iou=$MIN_IOU on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"

python3 stage3/rft_select.py --candidates_dir "$ROUND_DIR/candidates" \
    --round_dir "$ROUND_DIR" --base_config stage3/config.yaml \
    --data_dir stage1_output --min_iou "$MIN_IOU" --workers 16 --limit "$LIMIT"

echo "=== done at $(date) ==="
