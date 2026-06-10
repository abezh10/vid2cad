#!/bin/bash
# Chain eval (afterok: train) -> geom (afterok: eval) for the 3 balanced/prompt
# ablation models, then a final analysis job (afterok: all 3 geoms). The whole
# pipeline runs hands-off once training succeeds; results land in
# stage3/bal_comparison.txt. Reads train job ids from stage3/.bal_jobids (A B C).
set -euo pipefail
cd /scratch/users/abzhong
read -r JA JB JC < stage3/.bal_jobids
echo "train jobs: A=$JA B=$JB C=$JC"

GEOM_IDS=()
: > stage3/.bal_jobids_downstream

# name  frames  prompt_variant  train_jobid
chain() {
  local name="$1" nf="$2" pv="$3" tj="$4"
  local ej gj
  ej=$(sbatch --parsable --dependency=afterok:"$tj" \
        --export=ALL,NAME="$name",NFRAMES="$nf",PROMPT_VARIANT="$pv" \
        stage3/eval_qwen_vit_bal.sbatch)
  gj=$(sbatch --parsable --dependency=afterok:"$ej" \
        --export=ALL,NAME="$name" \
        stage3/geom_qwen_vit_sweep.sbatch)
  GEOM_IDS+=("$gj")
  echo "$name : eval=$ej (afterok:$tj)  geom=$gj (afterok:$ej)"
  echo "$name eval=$ej geom=$gj" >> stage3/.bal_jobids_downstream
}

chain qwen_vit_r64_f10  10 default "$JA"
chain qwen_vit_r64_bal  10 default "$JB"
chain qwen_vit_r64_balp 10 geom    "$JC"

# final analysis after all three geoms succeed
DEP=$(IFS=:; echo "${GEOM_IDS[*]}")
aj=$(sbatch --parsable --dependency=afterok:"$DEP" stage3/analyze_bal.sbatch)
echo "analysis=$aj (afterok:$DEP)"
echo "analysis=$aj" >> stage3/.bal_jobids_downstream
echo "--- downstream chain submitted ---"
cat stage3/.bal_jobids_downstream