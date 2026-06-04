#!/bin/bash
#SBATCH --job-name=ev_qwen_boNfull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage3/eval_qwen_boN_full_%j.out
#SBATCH --error=logs/stage3/eval_qwen_boN_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Full 994-object best-of-N eval, same config that scored 100/100 on the 100-obj
# subset: 4 sampled candidates, temp 0.8 / top_p 0.95, NO repetition_penalty /
# no_repeat_ngram (those corrupt CadQuery's repetitive syntax). Fresh out_dir so
# the greedy baseline (eval/) and the 100-obj test (eval_boN_test2/) stay intact.

set -euo pipefail
echo "=== full best-of-N eval qwen on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen --checkpoint_dir stage3_runs/qwen \
    --config stage3/config.yaml --out_dir stage3_runs/qwen/eval_boN_full \
    --limit 0 \
    --n_samples 4 --temperature 0.8 --top_p 0.95 \
    --max_new_tokens 1536

echo "=== done at $(date) ==="
