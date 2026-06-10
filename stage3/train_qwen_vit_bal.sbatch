#!/bin/bash
#SBATCH --job-name=vit_bal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=44:00:00
#SBATCH --output=logs/stage3/vit_bal_%j.out
#SBATCH --error=logs/stage3/vit_bal_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ViT-adaptation run with an optional geometry-difficulty-balanced train manifest.
# Same data/recipe/LLM+ViT-LoRA as the qwen_vit sweep, with rank/frame/manifest
# overrides. Env: NAME RANK ALPHA [NFRAMES=8] [EPOCHS=3] [TRAIN_MANIFEST=].
#   - leave TRAIN_MANIFEST empty  -> original stage2b/train.jsonl (control)
#   - set it to train_balanced    -> oversample non-orthogonal geometry
set -euo pipefail
NAME="${NAME:?set NAME}"
RANK="${RANK:?set RANK}"
ALPHA="${ALPHA:?set ALPHA}"
NFRAMES="${NFRAMES:-8}"
EPOCHS="${EPOCHS:-3}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
PROMPT_VARIANT="${PROMPT_VARIANT:-default}"
MAX_STEPS="${MAX_STEPS:-}"
MAX_SIDE="${MAX_SIDE:-}"
DL_WORKERS="${DL_WORKERS:-}"
AUGMENT="${AUGMENT:-}"          # ''=off (config.yaml) | mild (config_aug_mild.yaml)

CONFIG="stage3/config.yaml"
if [[ "$AUGMENT" == "mild" ]]; then
    CONFIG="stage3/config_aug_mild.yaml"
fi

echo "=== vit_bal $NAME (rank=$RANK alpha=$ALPHA frames=$NFRAMES epochs=$EPOCHS prompt=$PROMPT_VARIANT) ==="
echo "    config=$CONFIG max_side='${MAX_SIDE:-native}' augment='${AUGMENT:-off}'"
echo "    train_manifest='${TRAIN_MANIFEST:-<default train.jsonl>}' on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"
mkdir -p "${HF_HOME}"

EXTRA=()
if [[ -n "$TRAIN_MANIFEST" ]]; then
    EXTRA+=(--train_manifest "$TRAIN_MANIFEST")
fi
if [[ -n "$MAX_STEPS" ]]; then
    EXTRA+=(--max_steps "$MAX_STEPS")
fi
if [[ -n "$MAX_SIDE" ]]; then
    EXTRA+=(--max_side "$MAX_SIDE")
fi
if [[ -n "$DL_WORKERS" ]]; then
    EXTRA+=(--dataloader_workers "$DL_WORKERS")
fi

python3 stage3/train_qwen.py --config "$CONFIG" --model_key qwen_vit \
    --output_name "$NAME" --lora_rank "$RANK" --lora_alpha "$ALPHA" \
    --n_frames "$NFRAMES" --epochs "$EPOCHS" \
    --prompt_variant "$PROMPT_VARIANT" "${EXTRA[@]}"

echo "=== done at $(date) ==="
