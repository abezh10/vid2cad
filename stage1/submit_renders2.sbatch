#!/bin/bash
#SBATCH --job-name=submit_renders2
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/stage1/submit_renders2_%j.out
#SBATCH --error=logs/stage1/submit_renders2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Tranche-2 expansion: +30k stratified objects into the SAME stage1_expansion
# root, at 960x720 no-EXR. Runs afterany the tranche-1 render array so the
# 1024-task MaxSubmit budget is free again. Chains: render array -> frame
# extraction (idempotent over the whole root) -> train_big2.jsonl with the
# MERGED tranche-1+2 id set (expansion_ids_all.json). test-HR is NOT rebuilt.
set -euo pipefail
cd /scratch/users/abzhong

echo "=== submitting tranche-2 render at $(date) ==="
EXP_OUT=$(bash stage1/submit_stage1_subset.sh \
    --manifest stage1_output/expansion2_manifest.txt \
    --output_dir stage1_expansion \
    --aruco_marker aruco_marker.png \
    --resolution 960 720 --no_exr \
    --num-tasks 1000 --time 08:00:00)
echo "$EXP_OUT"
EXP_JOB=$(echo "$EXP_OUT" | grep -oE "Submitted array job [0-9]+" | grep -oE "[0-9]+")

echo "=== chaining extraction + merged manifest (exp2=$EXP_JOB) ==="
EX=$(sbatch --parsable --dependency=afterany:"$EXP_JOB" \
     --export=ALL,ROOT=stage1_expansion,FRAMES_OUT=stage2b/frames_exp,NFRAMES=24 \
     stage2b/extract_frames.sbatch)
BM=$(sbatch --parsable --dependency=afterok:"$EX" \
     --export=ALL,EXP_IDS=stage1_output/expansion_ids_all.json,OUT=stage2b/train_big2.jsonl,SKIP_TEST_HR=1 \
     stage2b/build_big_manifests.sbatch)

echo "tranche2_render=$EXP_JOB extract=$EX build=$BM" > stage1_output/.render2_jobids
cat stage1_output/.render2_jobids
echo "=== chained at $(date) ==="
