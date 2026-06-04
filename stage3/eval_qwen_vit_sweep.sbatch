#!/bin/bash
#SBATCH --job-name=vit_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/vit_eval_%j.out
#SBATCH --error=logs/stage3/vit_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Eval a ViT-sweep checkpoint on the 994-object test set. Env: NAME [NFRAMES=8].
# Adapter rank loads from the saved adapter_config.json; n_frames MUST match
# what the checkpoint was trained with.
set -euo pipefail
NAME="${NAME:?set NAME}"
NFRAMES="${NFRAMES:-8}"
echo "=== vit-sweep eval $NAME (frames=$NFRAMES) on $(hostname) at $(date) ==="
module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
export HF_HOME="/scratch/users/abzhong/.hf_cache/qwen"

python3 stage3/eval.py --model qwen --checkpoint_dir "stage3_runs/$NAME" \
    --config stage3/config.yaml --out_dir "stage3_runs/$NAME/eval" --n_frames "$NFRAMES"

echo "=== done at $(date) ==="
