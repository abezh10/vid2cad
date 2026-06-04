#!/bin/bash
#SBATCH --job-name=gm_kimi
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/geom_kimi_%j.out
#SBATCH --error=logs/stage3/geom_kimi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Geometric eval (IoU + Chamfer) for Kimi. Override with
# EVAL_DIR=stage3_runs/kimi_fixed/eval for the corrected retrain.

set -euo pipefail
EVAL_DIR="${EVAL_DIR:-stage3_runs/kimi/eval}"
echo "=== geom eval kimi ($EVAL_DIR) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong

python3 stage3/eval_geom.py --eval_dir "$EVAL_DIR" --data_dir stage1_output --workers 16

echo "=== done at $(date) ==="
