#!/bin/bash
#SBATCH --job-name=diag_rot
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/diag_rotation_%j.out
#SBATCH --error=logs/stage3/diag_rotation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Rotation-alignment diagnostic (CPU). Quantifies how much of the IoU gap is
# orientation vs genuine shape. Override EVAL_DIR for other models, e.g.
# EVAL_DIR=stage3_runs/internvl/eval sbatch stage3/diag_rotation.sbatch

set -euo pipefail
EVAL_DIR="${EVAL_DIR:-stage3_runs/qwen/eval}"
LIMIT="${LIMIT:-0}"
echo "=== rotation diagnostic ($EVAL_DIR limit=$LIMIT) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"

python3 stage3/diag_rotation.py --eval_dir "$EVAL_DIR" --data_dir stage1_output \
    --limit "$LIMIT" --workers 16

echo "=== done at $(date) ==="
