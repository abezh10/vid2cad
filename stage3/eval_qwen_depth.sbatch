#!/bin/bash
#SBATCH --job-name=ev_qwen_d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/eval_qwen_depth_%j.out
#SBATCH --error=logs/stage3/eval_qwen_depth_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Eval the depth-conditioned Qwen checkpoint. --use_depth interleaves the depth
# frames exactly as training did; without it the inputs would not match the
# adapter. Geom (IoU/Chamfer) is run separately over the produced eval dir.

set -euo pipefail
echo "=== eval qwen DEPTH on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen \
    --checkpoint_dir stage3_runs_depth/qwen \
    --config stage3/config_depth.yaml --use_depth \
    --out_dir stage3_runs_depth/qwen/eval

echo "=== done at $(date) ==="
