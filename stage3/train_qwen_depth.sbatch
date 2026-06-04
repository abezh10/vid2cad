#!/bin/bash
#SBATCH --job-name=s3_qwen_d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/qwen_depth_%j.out
#SBATCH --error=logs/stage3/qwen_depth_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Depth-conditioning experiment: Qwen fine-tuned on interleaved RGB+depth frames
# (config_depth.yaml -> stage3_runs_depth/qwen). Compare against the RGB-only
# baseline stage3_runs/qwen. Doubles the per-object image count (8 RGB + 8 depth),
# so VRAM headroom is tighter than the baseline run.

set -euo pipefail
echo "=== qwen DEPTH training on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"
mkdir -p "${HF_HOME}"

python3 stage3/train_qwen.py --config stage3/config_depth.yaml

echo "=== done at $(date) ==="
