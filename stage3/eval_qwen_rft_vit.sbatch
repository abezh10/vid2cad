#!/bin/bash
#SBATCH --job-name=ev_rft_vit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/eval_qwen_rft_vit_%j.out
#SBATCH --error=logs/stage3/eval_qwen_rft_vit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Eval the combined RFT+ViT adapter. --model qwen = Qwen3-VL base; the adapter
# at stage3_rft/qwen/round1/qwen_vit carries both LLM + ViT LoRA. Greedy decode,
# same 994 test set (round config manifest_dir = round dir).
set -euo pipefail
echo "=== eval qwen RFT+ViT on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen \
    --checkpoint_dir stage3_rft/qwen/round1/qwen_vit \
    --config stage3_rft/qwen/round1/config.yaml \
    --out_dir stage3_rft/qwen/round1/qwen_vit/eval

echo "=== done at $(date) ==="
