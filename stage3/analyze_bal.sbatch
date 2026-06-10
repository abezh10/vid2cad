#!/bin/bash
#SBATCH --job-name=bal_analyze
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/stage3/bal_analyze_%j.out
#SBATCH --error=logs/stage3/bal_analyze_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Final stratified comparison across the baseline + 3 ablation models, run once
# all geom-scoring finishes. Writes the per-bucket table to stage3/bal_comparison.txt
# so results are waiting whenever the pipeline completes (~2 days out).
set -euo pipefail
module load python/3.13.5
cd /scratch/users/abzhong
export PYTHONPATH="/scratch/users/abzhong/stage3"
echo "=== bal analysis on $(hostname) at $(date) ==="

python3 stage3/diag_geometry.py \
    --eval stage3_runs/qwen_vit_r64/eval:r64_base \
    --eval stage3_runs/qwen_vit_r64_f10/eval:A_f10 \
    --eval stage3_runs/qwen_vit_r64_bal/eval:B_bal \
    --eval stage3_runs/qwen_vit_r64_balp/eval:C_balp \
    --dump stage3/bal_comparison_dump.jsonl | tee stage3/bal_comparison.txt

echo "" | tee -a stage3/bal_comparison.txt
echo "=== overall geom summaries ===" | tee -a stage3/bal_comparison.txt
for d in qwen_vit_r64 qwen_vit_r64_f10 qwen_vit_r64_bal qwen_vit_r64_balp; do
    echo "--- $d ---" | tee -a stage3/bal_comparison.txt
    cat "stage3_runs/$d/eval/geom_summary.json" 2>/dev/null | tee -a stage3/bal_comparison.txt
    echo "" | tee -a stage3/bal_comparison.txt
done
echo "=== analysis done at $(date) ==="
