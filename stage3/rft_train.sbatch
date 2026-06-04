#!/bin/bash
#SBATCH --job-name=rft_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/rft_train_%j.out
#SBATCH --error=logs/stage3/rft_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# RFT fine-tune (GPU). SFT the same per-model train_<model>.py on the curated
# round manifest, using the round config (manifest_dir + output_root point at
# the round dir, so the new adapter lands at <ROUND_DIR>/<MODEL>). Run after
# rft_select.sbatch.
#
# Env overrides: MODEL ROUND_DIR

set -euo pipefail
MODEL="${MODEL:-qwen}"
ROUND_DIR="${ROUND_DIR:-stage3_rft/$MODEL/round1}"
ROUND_CFG="$ROUND_DIR/config.yaml"

echo "=== rft train model=$MODEL round=$ROUND_DIR cfg=$ROUND_CFG on $(hostname) at $(date) ==="
if [ ! -f "$ROUND_CFG" ]; then echo "missing $ROUND_CFG — run rft_select first" >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/$MODEL"
mkdir -p "$HF_HOME"

python3 "stage3/train_${MODEL}.py" --config "$ROUND_CFG"

echo "=== done at $(date) ==="
