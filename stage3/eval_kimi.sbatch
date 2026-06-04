#!/bin/bash
#SBATCH --job-name=ev_kimi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage3/eval_kimi_%j.out
#SBATCH --error=logs/stage3/eval_kimi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Greedy baseline eval for Kimi-VL-A3B-Thinking (execution rate), comparable to
# the Qwen 88.93% / InternVL 95.47% greedy baselines. Override the checkpoint
# with CKPT=stage3_runs/kimi_fixed to eval the corrected retrain.

set -euo pipefail
CKPT="${CKPT:-stage3_runs/kimi}"
LIMIT="${LIMIT:-0}"   # 0 = full test set; >0 limits examples (quick patch check)
echo "=== eval kimi (ckpt=$CKPT limit=$LIMIT) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/kimi"

python3 stage3/eval.py --model kimi --checkpoint_dir "$CKPT" \
    --config stage3/config.yaml --out_dir "$CKPT/eval" --limit "$LIMIT" --resume

echo "=== done at $(date) ==="
