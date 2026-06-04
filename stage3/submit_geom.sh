#!/bin/bash
# Submit a geometric eval (Chamfer + volumetric IoU) for a model's text-eval
# output. CPU-only — runs on the normal partition.
#
# Usage:
#   bash stage3/submit_geom.sh qwen
#   bash stage3/submit_geom.sh glm
#   bash stage3/submit_geom.sh qwen stage3_runs/qwen/eval_tok1024   # explicit dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-}"
case "$MODEL" in
    qwen|glm) ;;
    *) echo "Usage: $0 {qwen|glm} [eval_dir]"; exit 1 ;;
esac
EVAL_DIR="${2:-stage3_runs/${MODEL}/eval}"
if [[ ! -f "${REPO_ROOT}/${EVAL_DIR}/per_object.jsonl" ]]; then
    echo "No per_object.jsonl in ${EVAL_DIR}; run the text eval first."
    exit 1
fi

mkdir -p logs/stage3
NCPU=16

SBATCH_FILE="${REPO_ROOT}/stage3/geom_${MODEL}.sbatch"
cat > "$SBATCH_FILE" <<SBATCH
#!/bin/bash
#SBATCH --job-name=gm_${MODEL}
#SBATCH --partition=normal
#SBATCH --cpus-per-task=${NCPU}
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3/geom_${MODEL}_%j.out
#SBATCH --error=logs/stage3/geom_${MODEL}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== geom eval ${MODEL} (${EVAL_DIR}) on \$(hostname) at \$(date) ==="
module load python/3.13.5
cd ${REPO_ROOT}

python3 stage3/eval_geom.py --eval_dir ${EVAL_DIR} --data_dir stage1_output --workers ${NCPU}

echo "=== done at \$(date) ==="
SBATCH

JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $NF}')
echo "Submitted ${MODEL} geom eval as job ${JOB_ID}"
echo "Logs: logs/stage3/geom_${MODEL}_${JOB_ID}.{out,err}"
