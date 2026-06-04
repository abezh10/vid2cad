#!/bin/bash
#SBATCH --job-name=ev_qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/eval_qwen_%j.out
#SBATCH --error=logs/stage3/eval_qwen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== eval qwen on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen --checkpoint_dir stage3_runs/qwen --config stage3/config.yaml --out_dir stage3_runs/qwen/eval_tok3072

echo "=== done at $(date) ==="
