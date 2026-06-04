#!/bin/bash
#SBATCH --job-name=rft_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/rft_eval_%j.out
#SBATCH --error=logs/stage3/rft_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Eval an RFT-trained adapter on the round test set (identical to the base test
# set, so the IoU is directly comparable to the RGB-only baseline). The round
# config's manifest_dir points at the round dir; the adapter lives at
# <ROUND_DIR>/<MODEL>. Run after rft_train.sbatch.
#
# Env overrides: MODEL ROUND_DIR

set -euo pipefail
MODEL="${MODEL:-qwen}"
ROUND_DIR="${ROUND_DIR:-stage3_rft/$MODEL/round1}"
ROUND_CFG="$ROUND_DIR/config.yaml"

echo "=== rft eval model=$MODEL round=$ROUND_DIR on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/$MODEL"

python3 stage3/eval.py --model "$MODEL" \
    --checkpoint_dir "$ROUND_DIR/$MODEL" \
    --config "$ROUND_CFG" \
    --out_dir "$ROUND_DIR/$MODEL/eval"

echo "=== done at $(date) ==="
