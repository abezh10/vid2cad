#!/bin/bash
# Submit a Stage 3 training job for one model on 1× L40S.
#
# Usage:
#   bash stage3/submit_train.sh kimi
#   bash stage3/submit_train.sh qwen
#   bash stage3/submit_train.sh glm
#   bash stage3/submit_train.sh internvl
#
# Submit all four in parallel:
#   for m in kimi qwen glm internvl; do bash stage3/submit_train.sh $m; done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 {kimi|qwen|glm|internvl}"
    exit 1
fi

case "$MODEL" in
    kimi|qwen|glm|internvl) ;;
    *) echo "Unknown model: $MODEL"; exit 1 ;;
esac

mkdir -p logs/stage3

SBATCH_FILE="${REPO_ROOT}/stage3/train_${MODEL}.sbatch"
cat > "$SBATCH_FILE" <<SBATCH
#!/bin/bash
#SBATCH --job-name=s3_${MODEL}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage3/${MODEL}_%j.out
#SBATCH --error=logs/stage3/${MODEL}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== ${MODEL} training on \$(hostname) at \$(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module load python/3.13.5
cd ${REPO_ROOT}

export PYTHONPATH="${REPO_ROOT}/stage3"
export HF_HOME="${REPO_ROOT}/.hf_cache/${MODEL}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "\${HF_HOME}"

python3 stage3/train_${MODEL}.py --config stage3/config.yaml

echo "=== done at \$(date) ==="
SBATCH

JOB_ID=$(sbatch "$SBATCH_FILE" | awk '{print $NF}')
echo "Submitted ${MODEL} as job ${JOB_ID}"
echo "Logs: logs/stage3/${MODEL}_${JOB_ID}.{out,err}"
