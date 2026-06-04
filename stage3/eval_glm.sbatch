#!/bin/bash
#SBATCH --job-name=ev_glm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/eval_glm_%j.out
#SBATCH --error=logs/stage3/eval_glm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== eval glm on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/glm"

python3 stage3/eval.py --model glm --checkpoint_dir stage3_runs/glm --config stage3/config.yaml

echo "=== done at $(date) ==="
