#!/bin/bash
#SBATCH --job-name=prep_exp
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/stage1/prep_exp_%j.out
#SBATCH --error=logs/stage1/prep_exp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Expansion prep (CPU): classify the full ~221k translated corpus by geometry,
# then sample the difficulty-stratified +20k expansion manifest and the 994
# test-HR re-render manifest. Idempotent.
set -euo pipefail
module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"

echo "=== classify corpus at $(date) ==="
python3 stage3/classify_corpus.py \
    --cadquery_dir deepcad-data/cadquery_py \
    --out deepcad-data/geometry_index.jsonl --workers 16

echo "=== sample expansion at $(date) ==="
python3 stage1/prepare_expansion.py \
    --n_total "${N_TOTAL:-20000}" \
    --manifest_out stage1_output/expansion_manifest.txt \
    --ids_out stage1_output/expansion_ids.json

echo "=== test-HR manifest at $(date) ==="
python3 stage1/prepare_test_hr.py \
    --manifest_out stage1_output/test_hr_manifest.txt

echo "=== prep done at $(date) ==="
