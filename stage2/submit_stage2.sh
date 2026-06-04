#!/bin/bash
# Submit Stage 2 as a SLURM array job.
# Discovers completed Stage 1 objects and processes them in chunks.
#
# Usage:
#   bash stage2/submit_stage2.sh --data_dir stage1_output
#   bash stage2/submit_stage2.sh --manifest stage1_output/subset_manifest.txt --data_dir stage1_output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
DATA_DIR=""
MANIFEST=""
PARTITION="normal"
NUM_TASKS=500
CPUS=2
MEM="8G"
TIME_LIMIT="04:00:00"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2";   shift 2 ;;
        --manifest)   MANIFEST="$2";   shift 2 ;;
        --partition)  PARTITION="$2";  shift 2 ;;
        --cpus)       CPUS="$2";       shift 2 ;;
        --mem)        MEM="$2";        shift 2 ;;
        --time)       TIME_LIMIT="$2"; shift 2 ;;
        --num-tasks)  NUM_TASKS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "ERROR: --data_dir is required."
    exit 1
fi

mkdir -p logs/stage2

# ---------- build manifest of completed Stage 1 objects ----------
STAGE2_MANIFEST="${DATA_DIR}/stage2_input_manifest.txt"

if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
    # Use provided manifest (from prepare_subset.py), extract object IDs
    # Manifest has paths like /scratch/.../cadquery_py/0000/00000007.py
    # We need object IDs like 0000/00000007
    python3 -c "
from pathlib import Path
import sys
manifest = Path('$MANIFEST')
data_dir = Path('$DATA_DIR')
ids = []
for line in manifest.read_text().strip().split('\n'):
    parts = Path(line).stem  # 00000007
    bucket = Path(line).parent.name  # 0000
    obj_id = f'{bucket}/{parts}'
    obj_dir = data_dir / obj_id
    if (obj_dir / 'cameras.json').exists():
        ids.append(obj_id)
Path('$STAGE2_MANIFEST').write_text('\n'.join(sorted(ids)) + '\n')
print(f'Found {len(ids)} Stage 1 completed objects from manifest')
"
else
    echo "Auto-discovering completed Stage 1 objects …"
    find "$DATA_DIR" -name "cameras.json" -path "*/????/????????/cameras.json" \
        | sed "s|${DATA_DIR}/||; s|/cameras.json||" \
        | sort > "$STAGE2_MANIFEST"
fi

N=$(wc -l < "$STAGE2_MANIFEST")
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no completed Stage 1 objects found"
    exit 1
fi

LAST_TASK=$(( NUM_TASKS < N ? NUM_TASKS - 1 : N - 1 ))
ACTUAL_TASKS=$(( LAST_TASK + 1 ))
CHUNK=$(( (N + ACTUAL_TASKS - 1) / ACTUAL_TASKS ))
MAX_CONCURRENT=$(( 512 / CPUS ))
if (( MAX_CONCURRENT > 128 )); then MAX_CONCURRENT=128; fi

echo "Found $N objects → $ACTUAL_TASKS tasks × ~$CHUNK objects/task"
echo "Resources: ${CPUS} CPUs, ${MEM} RAM, time ${TIME_LIMIT}"
echo "Max concurrent: ${MAX_CONCURRENT}"

# ---------- write sbatch worker ----------
SBATCH_FILE="${DATA_DIR}/stage2_array.sbatch"
cat > "$SBATCH_FILE" <<'OUTER'
#!/bin/bash
#SBATCH --job-name=stage2
OUTER

cat >> "$SBATCH_FILE" <<SBATCH
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=logs/stage2/%A_%a.out
#SBATCH --error=logs/stage2/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== task \$SLURM_ARRAY_TASK_ID / \$SLURM_ARRAY_JOB_ID on \$(hostname) at \$(date) ==="

module load python/3.13.5
export PYTHONPATH="${SCRIPT_DIR}/.."

MANIFEST="${STAGE2_MANIFEST}"
N=${N}
CHUNK=${CHUNK}
START=\$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=\$(( START + CHUNK - 1 ))
if (( END > N )); then END=${N}; fi

echo "Processing manifest lines \$START .. \$END (\$(( END - START + 1 )) objects)"

OK=0
FAIL=0
for (( LINE=START; LINE<=END; LINE++ )); do
    OBJECT_ID=\$(sed -n "\${LINE}p" "\$MANIFEST")
    [[ -z "\$OBJECT_ID" ]] && continue

    echo "--- [\$(( LINE - START + 1 ))/\$(( END - START + 1 ))] \$OBJECT_ID ---"

    python3 ${SCRIPT_DIR}/run_stage2.py \\
        --object_id "\$OBJECT_ID" \\
        --data_dir  "${DATA_DIR}" \\
    && OK=\$(( OK + 1 )) || FAIL=\$(( FAIL + 1 ))
done

echo ""
echo "=== task \$SLURM_ARRAY_TASK_ID summary: ok=\$OK fail=\$FAIL total=\$(( OK + FAIL )) ==="
echo "=== done at \$(date) ==="
SBATCH

# ---------- submit ----------
JOB_ID=$(sbatch \
    --array="0-${LAST_TASK}%${MAX_CONCURRENT}" \
    "$SBATCH_FILE" | awk '{print $NF}')

echo ""
echo "Submitted array job ${JOB_ID} (tasks 0-${LAST_TASK}, max ${MAX_CONCURRENT} concurrent)"
echo ""
echo "Monitor:   squeue -u \$USER"
echo "Progress:  find ${DATA_DIR} -name point_cloud_norm_8192.ply | wc -l"
echo "Validate:  python3 ${SCRIPT_DIR}/validate_stage2.py --data_dir ${DATA_DIR}"
