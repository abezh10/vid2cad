#!/bin/bash
#SBATCH --job-name=rft_geom
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/rft_geom_%j.out
#SBATCH --error=logs/stage3/rft_geom_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# IoU/Chamfer over an RFT eval dir. Run after rft_eval.sbatch.
# Env overrides: MODEL ROUND_DIR

set -euo pipefail
MODEL="${MODEL:-qwen}"
ROUND_DIR="${ROUND_DIR:-stage3_rft/$MODEL/round1}"

echo "=== rft geom model=$MODEL ($ROUND_DIR/$MODEL/eval) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong

python3 stage3/eval_geom.py --eval_dir "$ROUND_DIR/$MODEL/eval" \
    --data_dir stage1_output --workers 16

echo "=== done at $(date) ==="
