#!/bin/bash
#SBATCH --job-name=ev_qwen_boN2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage3/eval_qwen_boN2_full_%j.out
#SBATCH --error=logs/stage3/eval_qwen_boN2_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Full 994-object best-of-2 eval. Same validated decoding (temp 0.8 / top_p 0.95,
# NO repetition_penalty / no_repeat_ngram), but 2 candidates instead of 4 to halve
# cost: per-sample exec is ~96%, so best-of-2 ~= 99.8% — nearly identical to
# best-of-4 at half the GPU time (~7-8h vs ~13h). The best-of-4 full run was
# killed by its 6h wall at ~100/994; this fits comfortably in 12h. Fresh out_dir.

set -euo pipefail
echo "=== full best-of-2 eval qwen on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong

export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen --checkpoint_dir stage3_runs/qwen \
    --config stage3/config.yaml --out_dir stage3_runs/qwen/eval_boN2_full \
    --limit 0 \
    --n_samples 2 --temperature 0.8 --top_p 0.95 \
    --max_new_tokens 1536

echo "=== done at $(date) ==="
