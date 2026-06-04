#!/bin/bash
#SBATCH --job-name=gm_qwen_vit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/geom_qwen_vit_%j.out
#SBATCH --error=logs/stage3/geom_qwen_vit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Geometric scoring (IoU/Chamfer) for the ViT-adapted run. Compare
# stage3_runs/qwen_vit/eval/geom_summary.json against the frozen-ViT baseline
# stage3_runs/qwen/eval/geom_summary.json -> the controlled ViT-adaptation delta.
set -euo pipefail
echo "=== geom eval qwen_vit on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong

python3 stage3/eval_geom.py --eval_dir stage3_runs/qwen_vit/eval \
    --data_dir stage1_output --workers 16

echo "=== done at $(date) ==="
