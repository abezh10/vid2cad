#!/bin/bash
#SBATCH --job-name=s3_rft_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/qwen_rft_vit_%j.out
#SBATCH --error=logs/stage3/qwen_rft_vit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# COMBINED run: RFT-curated data (stage3_rft/qwen/round1 manifest) + ViT-adapted
# encoder (qwen_vit targets). Tests whether the two wins stack. Output ->
# stage3_rft/qwen/round1/qwen_vit. Compare against:
#   stage3_runs/qwen           (SFT baseline, frozen ViT)
#   stage3_rft/qwen/round1/qwen (RFT only, frozen ViT)
#   stage3_runs/qwen_vit       (ViT-adapt only, SFT data)
set -euo pipefail
echo "=== qwen RFT+ViT combined training on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"
mkdir -p "${HF_HOME}"

python3 stage3/train_qwen.py \
    --config stage3_rft/qwen/round1/config.yaml \
    --model_key qwen_vit --output_name qwen_vit

echo "=== done at $(date) ==="
