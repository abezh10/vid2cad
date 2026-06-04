#!/bin/bash
#SBATCH --job-name=s3_kimi_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/kimi_fixed_%j.out
#SBATCH --error=logs/stage3/kimi_fixed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Retrain with corrected LoRA targets: the original run targeted
# q_proj/k_proj/v_proj/o_proj, but Kimi's DeepSeek-V3 MLA attention has no
# k_proj/v_proj (K/V go through kv_a_proj_with_mqa + kv_b_proj) so only q_proj
# and o_proj matched -> 0.05% trainable -> loss plateaued at ~4.5. Fixed config
# now targets the real MLA projections + the always-active shared experts.
# Writes to stage3_runs/kimi_fixed so the original (weak) adapter is preserved.

set -euo pipefail
echo "=== kimi FIXED retrain on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/kimi"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${HF_HOME}"

python3 stage3/train_kimi.py --config stage3/config.yaml --output_name kimi_fixed

echo "=== done at $(date) ==="
