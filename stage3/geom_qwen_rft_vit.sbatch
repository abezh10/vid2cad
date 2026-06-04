#!/bin/bash
#SBATCH --job-name=gm_rft_vit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/geom_qwen_rft_vit_%j.out
#SBATCH --error=logs/stage3/geom_qwen_rft_vit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Geometric scoring for the combined RFT+ViT run. Compare
# stage3_rft/qwen/round1/qwen_vit/eval/geom_summary.json against the three
# corners of the 2x2 ablation (baseline / RFT-only / ViT-only).
set -euo pipefail
echo "=== geom qwen RFT+ViT on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong

python3 stage3/eval_geom.py \
    --eval_dir stage3_rft/qwen/round1/qwen_vit/eval \
    --data_dir stage1_output --workers 16

echo "=== done at $(date) ==="
