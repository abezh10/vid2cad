#!/bin/bash
#SBATCH --job-name=depth_frames
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/stage2/depth_frames_%j.out
#SBATCH --error=logs/stage2/depth_frames_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Render Stage-1 depth EXRs as colormapped JPGs aligned to the sampled RGB
# frames, for every object in the train/validation/test manifests. CPU-only;
# idempotent (skips objects already done). Feeds the depth-conditioning
# experiment (stage3/config_depth.yaml).

set -euo pipefail
module load python/3.13.5
cd /scratch/users/abzhong

echo "=== depth frames on $(hostname) at $(date) ==="
for m in train validation test; do
    echo "--- $m ---"
    python3 stage2b/make_depth_frames.py \
        --manifest "stage2b/${m}.jsonl" --workers 32
done
echo "=== done at $(date) ==="
