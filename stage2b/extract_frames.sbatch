#!/bin/bash
#SBATCH --job-name=extract_fr
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1/extract_fr_%j.out
#SBATCH --error=logs/stage1/extract_fr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Post-render frame extraction for a render root (expansion or test-HR).
# Env: ROOT (e.g. stage1_expansion), FRAMES_OUT (e.g. stage2b/frames_exp),
#      NFRAMES (default 24). Idempotent (sample_frames skips complete dirs).
set -euo pipefail
ROOT="${ROOT:?set ROOT}"
FRAMES_OUT="${FRAMES_OUT:?set FRAMES_OUT}"
NFRAMES="${NFRAMES:-24}"
module load python/3.13.5
cd /scratch/users/abzhong

LIST="${FRAMES_OUT%/}.videos.jsonl"
echo "=== list videos under $ROOT at $(date) ==="
python3 stage2b/list_videos.py --root "$ROOT" --out "$LIST"

echo "=== extract $NFRAMES frames/video -> $FRAMES_OUT at $(date) ==="
python3 stage2b/sample_frames.py --manifest "$LIST" \
    --out_dir "$FRAMES_OUT" --n_frames "$NFRAMES" --workers 16

echo "=== done at $(date) ==="
