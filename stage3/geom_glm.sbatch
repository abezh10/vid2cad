#!/bin/bash
#SBATCH --job-name=gm_glm
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/geom_glm_%j.out
#SBATCH --error=logs/stage3/geom_glm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== geom eval glm (stage3_runs/glm/eval) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong

python3 stage3/eval_geom.py --eval_dir stage3_runs/glm/eval --data_dir stage1_output --workers 16

echo "=== done at $(date) ==="
