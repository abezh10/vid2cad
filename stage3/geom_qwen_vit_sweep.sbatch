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

# Geometric scoring (IoU/Chamfer) for a ViT-sweep eval dir.
# Env: NAME [EVAL_SUBDIR=eval] (eval_hr for the high-res test eval).
set -euo pipefail
NAME="${NAME:?set NAME}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval}"
echo "=== vit-sweep geom $NAME ($EVAL_SUBDIR) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong
python3 stage3/eval_geom.py --eval_dir "stage3_runs/$NAME/$EVAL_SUBDIR" \
    --data_dir stage1_output --workers 16
echo "=== done at $(date) ==="
