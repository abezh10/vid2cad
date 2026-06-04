#!/bin/bash
#SBATCH --job-name=s3_qwen_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/qwen_vit_%j.out
#SBATCH --error=logs/stage3/qwen_vit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# CS231n ViT-adaptation ablation: same recipe + data as the frozen-ViT baseline
# (stage3_runs/qwen), but LoRA ALSO adapts the Qwen3-VL vision transformer
# (qwen_vit entry in config.yaml). Output -> stage3_runs/qwen_vit. Compare
# IoU/exec against stage3_runs/qwen to isolate the effect of adapting the ViT.

set -euo pipefail
echo "=== qwen ViT-adapt training on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"
mkdir -p "${HF_HOME}"

python3 stage3/train_qwen.py --config stage3/config.yaml --model_key qwen_vit

echo "=== done at $(date) ==="
