#!/bin/bash
# Submit a Stage 3 evaluation job for one trained model on 1x L40S.
# Runs eval.py over the full test set: generates CadQuery code per object and
# records execution success rate.
#
# Usage:
#   bash stage3/submit_eval.sh qwen
#   bash stage3/submit_eval.sh glm
#   bash stage3/submit_eval.sh qwen eval_tok3072   # write to a named subdir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-}"
case "$MODEL" in
    qwen|glm) ;;
    *) echo "Usage: $0 {qwen|glm} [out_subdir]"; exit 1 ;;
esac
OUT_SUBDIR="${2:-eval}"

CKPT="${REPO_ROOT}/stage3_runs/${MODEL}"
if [[ ! -f "${CKPT}/adapter_model.safetensors" ]]; then
    echo "No trained adapter at ${CKPT}; train ${MODEL} first."
    exit 1
fi

mkdir -p logs/stage3

SBATCH_FILE="${REPO_ROOT}/stage3/eval_${MODEL}.sbatch"
cat > "$SBATCH_FILE" <<SBATCH
#!/bin/bash
#SBATCH --job-name=ev_${MODEL}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage3/eval_${MODEL}_%j.out
#SBATCH --error=logs/stage3/eval_${MODEL}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== eval ${MODEL} on \$(hostname) at \$(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd ${REPO_ROOT}

export PYTHONPATH="${REPO_ROOT}/stage3"
export HF_HOME="${REPO_ROOT}/.hf_cache/${MODEL}"

python3 stage3/eval.py --model ${MODEL} --checkpoint_dir stage3_runs/${MODEL} --config stage3/config.yaml --out_dir stage3_runs/${MODEL}/${OUT_SUBDIR}

echo "=== done at \$(date) ==="
SBATCH

JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $NF}')
echo "Submitted ${MODEL} eval as job ${JOB_ID}"
echo "Logs: logs/stage3/eval_${MODEL}_${JOB_ID}.{out,err}"
