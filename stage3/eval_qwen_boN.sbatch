#!/bin/bash
#SBATCH --job-name=ev_qwen_boN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/eval_qwen_boN_%j.out
#SBATCH --error=logs/stage3/eval_qwen_boN_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Quick best-of-N decoding test on the first 100 test objects. Design: a GREEDY
# anchor candidate (guarantees best-of-N >= the 88.9% greedy baseline) plus 3
# sampled candidates for diversity. NO repetition_penalty / no_repeat_ngram —
# those corrupt code (a 1.2/ngram=3 run scored 0/100). Writes to a fresh dir so
# the greedy baseline in stage3_runs/qwen/eval/ stays intact for before/after.

set -euo pipefail
echo "=== best-of-N eval qwen on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen --checkpoint_dir stage3_runs/qwen \
    --config stage3/config.yaml --out_dir stage3_runs/qwen/eval_boN_test2 \
    --limit 100 \
    --n_samples 4 --temperature 0.8 --top_p 0.95 \
    --max_new_tokens 1536

echo "=== done at $(date) ==="
