#!/bin/bash
#SBATCH --job-name=s3_internvl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/internvl_%j.out
#SBATCH --error=logs/stage3/internvl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== internvl training on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/internvl"
mkdir -p "${HF_HOME}"

python3 stage3/train_internvl.py --config stage3/config.yaml

echo "=== done at $(date) ==="
