#!/bin/bash
#SBATCH --job-name=vit_geom
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/vit_geom_%j.out
#SBATCH --error=logs/stage3/vit_geom_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Geometric scoring (IoU/Chamfer) for a ViT-sweep eval dir. Env: NAME.
# Compare stage3_runs/$NAME/eval/geom_summary.json against the rank-16/8-frame
# qwen_vit baseline (stage3_runs/qwen_vit/eval/geom_summary.json).
set -euo pipefail
NAME="${NAME:?set NAME}"
echo "=== vit-sweep geom $NAME on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong
python3 stage3/eval_geom.py --eval_dir "stage3_runs/$NAME/eval" \
    --data_dir stage1_output --workers 16
echo "=== done at $(date) ==="
