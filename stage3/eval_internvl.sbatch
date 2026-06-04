#!/bin/bash
#SBATCH --job-name=ev_internvl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/eval_internvl_%j.out
#SBATCH --error=logs/stage3/eval_internvl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Greedy baseline eval for InternVL3-8B (execution rate), directly comparable to
# the Qwen 88.93% greedy baseline. eval.py applies crop_to_patches=False for
# internvl (matches training; avoids the hi-res tiling token explosion).

set -euo pipefail
echo "=== eval internvl on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/internvl"

python3 stage3/eval.py --model internvl --checkpoint_dir stage3_runs/internvl \
    --config stage3/config.yaml --out_dir stage3_runs/internvl/eval

echo "=== done at $(date) ==="
