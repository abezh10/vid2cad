#!/bin/bash
#SBATCH --job-name=vid2cad
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=360G
#SBATCH --time=5:00:00
#SBATCH --output=logs/vid2cad-%j.out
#SBATCH --error=logs/vid2cad-%j.err

set -euo pipefail

mkdir -p logs

echo "=== Job $SLURM_JOB_ID started on $(hostname) at $(date) ==="
echo "CPUs allocated:  $SLURM_CPUS_PER_TASK"
echo "Memory request:  ${SLURM_MEM_PER_NODE:-unset} MB"
echo "Working dir:     $SLURM_SUBMIT_DIR"
echo

module load python/3.13.5

cd "$SLURM_SUBMIT_DIR"

bash run_all.sh "$SLURM_CPUS_PER_TASK"

echo
echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
