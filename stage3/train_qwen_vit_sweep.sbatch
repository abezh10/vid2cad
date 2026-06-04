#!/bin/bash
#SBATCH --job-name=vit_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/vit_sweep_%j.out
#SBATCH --error=logs/stage3/vit_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ViT-adaptation capacity sweep. Identical data/recipe to stage3_runs/qwen_vit
# (LoRA also adapts the Qwen3-VL vision tower) but with rank / frame-count
# overrides. Env: NAME RANK ALPHA [NFRAMES=8] [EPOCHS=3].
set -euo pipefail
NAME="${NAME:?set NAME}"
RANK="${RANK:?set RANK}"
ALPHA="${ALPHA:?set ALPHA}"
NFRAMES="${NFRAMES:-8}"
EPOCHS="${EPOCHS:-3}"

echo "=== vit-sweep $NAME (rank=$RANK alpha=$ALPHA frames=$NFRAMES epochs=$EPOCHS) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"
mkdir -p "${HF_HOME}"

python3 stage3/train_qwen.py --config stage3/config.yaml --model_key qwen_vit \
    --output_name "$NAME" --lora_rank "$RANK" --lora_alpha "$ALPHA" \
    --n_frames "$NFRAMES" --epochs "$EPOCHS"

echo "=== done at $(date) ==="
