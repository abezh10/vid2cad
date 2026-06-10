#!/bin/bash
#SBATCH --job-name=submit_renders
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/stage1/submit_renders_%j.out
#SBATCH --error=logs/stage1/submit_renders_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Runs AFTER prep_expansion.sbatch (afterok): submits the two Stage-1 render
# arrays (expansion @960x720 no-EXR, test-HR @960x720 no-EXR) and chains
# frame-extraction + manifest-building behind them. Task split respects
# MaxSubmitPerUser=1024: 800 expansion tasks (~25 obj x ~11min = ~4.5h) +
# 200 test-HR tasks (~5 obj = ~1h).
set -euo pipefail
cd /scratch/users/abzhong

echo "=== submitting expansion render at $(date) ==="
EXP_OUT=$(bash stage1/submit_stage1_subset.sh \
    --manifest stage1_output/expansion_manifest.txt \
    --output_dir stage1_expansion \
    --aruco_marker aruco_marker.png \
    --resolution 960 720 --no_exr \
    --num-tasks 800 --time 08:00:00)
echo "$EXP_OUT"
EXP_JOB=$(echo "$EXP_OUT" | grep -oE "Submitted array job [0-9]+" | grep -oE "[0-9]+")

echo "=== submitting test-HR render at $(date) ==="
THR_OUT=$(bash stage1/submit_stage1_subset.sh \
    --manifest stage1_output/test_hr_manifest.txt \
    --output_dir stage1_test_hr \
    --aruco_marker aruco_marker.png \
    --resolution 960 720 --no_exr \
    --num-tasks 200 --time 04:00:00)
echo "$THR_OUT"
THR_JOB=$(echo "$THR_OUT" | grep -oE "Submitted array job [0-9]+" | grep -oE "[0-9]+")

echo "=== chaining extraction + manifests (exp=$EXP_JOB thr=$THR_JOB) ==="
EX1=$(sbatch --parsable --dependency=afterany:"$EXP_JOB" \
      --export=ALL,ROOT=stage1_expansion,FRAMES_OUT=stage2b/frames_exp,NFRAMES=24 \
      stage2b/extract_frames.sbatch)
EX2=$(sbatch --parsable --dependency=afterany:"$THR_JOB" \
      --export=ALL,ROOT=stage1_test_hr,FRAMES_OUT=stage2b/frames_test_hr,NFRAMES=24 \
      stage2b/extract_frames.sbatch)
BM=$(sbatch --parsable --dependency=afterok:"$EX1":"$EX2" \
      stage2b/build_big_manifests.sbatch)

echo "expansion_render=$EXP_JOB test_hr_render=$THR_JOB" > stage1_output/.render_jobids
echo "extract_exp=$EX1 extract_thr=$EX2 build_manifests=$BM" >> stage1_output/.render_jobids
cat stage1_output/.render_jobids
echo "=== all chained at $(date) ==="
