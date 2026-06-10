#!/bin/bash
#SBATCH --job-name=vit_bal_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/stage3/vit_bal_eval_%j.out
#SBATCH --error=logs/stage3/vit_bal_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Eval a balanced/prompt-conditioned checkpoint on the 994-object test set.
# Env: NAME [NFRAMES=10] [PROMPT_VARIANT=default]. n_frames AND prompt_variant
# MUST match how the checkpoint was trained.
set -euo pipefail
NAME="${NAME:?set NAME}"
NFRAMES="${NFRAMES:-10}"
PROMPT_VARIANT="${PROMPT_VARIANT:-default}"
TEST_MANIFEST="${TEST_MANIFEST:-}"   # override test set (e.g. stage2b/test_hr.jsonl)
OUT_SUFFIX="${OUT_SUFFIX:-eval}"     # eval subdir name (eval | eval_hr)
MAX_SIDE="${MAX_SIDE:-}"             # frame-resolution cap; match training
echo "=== eval $NAME (frames=$NFRAMES prompt=$PROMPT_VARIANT manifest=${TEST_MANIFEST:-default} out=$OUT_SUFFIX) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

EXTRA=()
if [[ -n "$TEST_MANIFEST" ]]; then EXTRA+=(--test_manifest "$TEST_MANIFEST"); fi
if [[ -n "$MAX_SIDE" ]]; then EXTRA+=(--max_side "$MAX_SIDE"); fi

python3 stage3/eval.py --model qwen --checkpoint_dir "stage3_runs/$NAME" \
    --config stage3/config.yaml --out_dir "stage3_runs/$NAME/$OUT_SUFFIX" \
    --n_frames "$NFRAMES" --prompt_variant "$PROMPT_VARIANT" --resume "${EXTRA[@]}"

echo "=== done at $(date) ==="
