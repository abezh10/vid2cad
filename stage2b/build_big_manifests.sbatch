#!/bin/bash
#SBATCH --job-name=big_manifest
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1/big_manifest_%j.out
#SBATCH --error=logs/stage1/big_manifest_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Final manifest assembly after expansion + test-HR frames are extracted:
#   stage2b/train_big.jsonl  (original 7,924 + usable expansion rows)
#   stage2b/test_hr.jsonl    (frozen test ids, high-res frame dirs)
# Env overrides (tranche-2 chain): EXP_IDS (allowed-ids json), OUT (train manifest),
# SKIP_TEST_HR=1 (test-HR already built by the tranche-1 chain).
set -euo pipefail
module load python/3.13.5
cd /scratch/users/abzhong
EXP_IDS="${EXP_IDS:-stage1_output/expansion_ids.json}"
OUT="${OUT:-stage2b/train_big.jsonl}"

python3 stage2b/make_big_manifest.py \
    --expansion_root stage1_expansion \
    --expansion_ids "$EXP_IDS" \
    --frames_root stage2b/frames_exp \
    --out "$OUT"

if [[ "${SKIP_TEST_HR:-0}" != "1" ]]; then
    python3 stage2b/make_test_hr_manifest.py \
        --hr_root stage1_test_hr \
        --frames_root stage2b/frames_test_hr \
        --out stage2b/test_hr.jsonl
fi

echo "=== done at $(date) ==="
