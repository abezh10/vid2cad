#!/bin/bash
#SBATCH --job-name=rft_smpl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage3/rft_sample_%A_%a.out
#SBATCH --error=logs/stage3/rft_sample_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3

# RFT candidate sampling (GPU array). Each array task samples one contiguous
# shard of the TRAIN manifest from the SFT adapter. Override the shard count
# from the command line, e.g. 4 shards / 4 concurrent (respects the 4-GPU cap):
#   sbatch --array=0-3%4 --export=MODEL=qwen stage3/rft_sample.sbatch
#
# Env overrides: MODEL CKPT ROUND_DIR K GEN_BATCH TEMP TOP_P LIMIT

set -euo pipefail
MODEL="${MODEL:-qwen}"
CKPT="${CKPT:-stage3_runs/$MODEL}"
ROUND_DIR="${ROUND_DIR:-stage3_rft/$MODEL/round1}"
K="${K:-8}"
GEN_BATCH="${GEN_BATCH:-4}"
TEMP="${TEMP:-0.9}"
TOP_P="${TOP_P:-0.95}"
LIMIT="${LIMIT:-0}"

echo "=== rft sample model=$MODEL ckpt=$CKPT round=$ROUND_DIR ==="
echo "    shard=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} k=$K on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/$MODEL"
mkdir -p "$HF_HOME"

python3 stage3/rft_sample.py --model "$MODEL" --checkpoint_dir "$CKPT" \
    --config stage3/config.yaml --out_dir "$ROUND_DIR/candidates" \
    --k "$K" --gen_batch "$GEN_BATCH" --temperature "$TEMP" --top_p "$TOP_P" \
    --limit "$LIMIT" \
    --shard_id "${SLURM_ARRAY_TASK_ID:-0}" --num_shards "${SLURM_ARRAY_TASK_COUNT:-1}"

echo "=== done shard ${SLURM_ARRAY_TASK_ID:-0} at $(date) ==="
