#!/bin/bash
# Submit Stage 1 for a pre-built manifest (subset of objects).
#
# Usage:
#   bash submit_stage1_subset.sh \
#       --manifest  stage1_output/subset_manifest.txt \
#       --output_dir stage1_output \
#       --aruco_marker aruco_marker.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
MANIFEST=""
OUTPUT_DIR=""
ARUCO_MARKER=""
BLENDER_EXE="/scratch/users/abzhong/blender/blender"
ENGINE="BLENDER_EEVEE"
NUM_FRAMES=60
FPS=24
PARTITION="normal"
NUM_TASKS=500
CPUS=4
MEM="4G"
TIME_LIMIT="06:00:00"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)     MANIFEST="$2";     shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --aruco_marker) ARUCO_MARKER="$2"; shift 2 ;;
        --blender)      BLENDER_EXE="$2";  shift 2 ;;
        --engine)       ENGINE="$2";       shift 2 ;;
        --num_frames)   NUM_FRAMES="$2";   shift 2 ;;
        --fps)          FPS="$2";          shift 2 ;;
        --time)         TIME_LIMIT="$2";   shift 2 ;;
        --partition)    PARTITION="$2";    shift 2 ;;
        --cpus)         CPUS="$2";         shift 2 ;;
        --mem)          MEM="$2";          shift 2 ;;
        --num-tasks)    NUM_TASKS="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MANIFEST" || -z "$OUTPUT_DIR" || -z "$ARUCO_MARKER" ]]; then
    echo "ERROR: --manifest, --output_dir, and --aruco_marker are required."
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then echo "ERROR: manifest not found: $MANIFEST"; exit 1; fi
if [[ ! -f "$ARUCO_MARKER" ]]; then echo "ERROR: ArUco marker not found: $ARUCO_MARKER"; exit 1; fi
if [[ ! -x "$BLENDER_EXE" ]]; then echo "ERROR: Blender not found: $BLENDER_EXE"; exit 1; fi

mkdir -p "$OUTPUT_DIR" logs/stage1

N=$(wc -l < "$MANIFEST")
if [[ "$N" -eq 0 ]]; then echo "ERROR: manifest is empty"; exit 1; fi

# Derive INPUT_DIR from the first manifest entry
INPUT_DIR=$(head -1 "$MANIFEST" | xargs dirname | xargs dirname)

LAST_TASK=$(( NUM_TASKS < N ? NUM_TASKS - 1 : N - 1 ))
ACTUAL_TASKS=$(( LAST_TASK + 1 ))
CHUNK=$(( (N + ACTUAL_TASKS - 1) / ACTUAL_TASKS ))
MAX_CONCURRENT=$(( 512 / CPUS ))
if (( MAX_CONCURRENT > 128 )); then MAX_CONCURRENT=128; fi

echo "Manifest: $N objects → $ACTUAL_TASKS tasks × ~$CHUNK objects/task"
echo "Resources: ${CPUS} CPUs, ${MEM} RAM, time ${TIME_LIMIT}"
echo "Max concurrent: ${MAX_CONCURRENT}"

# ---------- write sbatch worker ----------
SBATCH_FILE="${OUTPUT_DIR}/stage1_subset.sbatch"
cat > "$SBATCH_FILE" <<'OUTER'
#!/bin/bash
#SBATCH --job-name=stage1-sub
OUTER

cat >> "$SBATCH_FILE" <<SBATCH
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=logs/stage1/%A_%a.out
#SBATCH --error=logs/stage1/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== task \$SLURM_ARRAY_TASK_ID / \$SLURM_ARRAY_JOB_ID on \$(hostname) at \$(date) ==="

module load python/3.13.5

MANIFEST="${MANIFEST}"
INPUT_DIR="${INPUT_DIR}"
N=${N}
CHUNK=${CHUNK}
START=\$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=\$(( START + CHUNK - 1 ))
if (( END > N )); then END=${N}; fi

echo "Processing manifest lines \$START .. \$END (\$(( END - START + 1 )) objects)"

OK=0
FAIL=0
for (( LINE=START; LINE<=END; LINE++ )); do
    OBJECT_PATH=\$(sed -n "\${LINE}p" "\$MANIFEST")
    [[ -z "\$OBJECT_PATH" ]] && continue

    OBJECT_ID="\${OBJECT_PATH#\${INPUT_DIR}/}"
    OBJECT_ID="\${OBJECT_ID%.py}"

    echo "--- [\$(( LINE - START + 1 ))/\$(( END - START + 1 ))] \$OBJECT_ID ---"

    python3 ${SCRIPT_DIR}/run_stage1.py \\
        --object_id    "\$OBJECT_ID" \\
        --input_dir    "${INPUT_DIR}" \\
        --output_dir   "${OUTPUT_DIR}" \\
        --aruco_marker "${ARUCO_MARKER}" \\
        --blender      "${BLENDER_EXE}" \\
        --engine       "${ENGINE}" \\
        --num_frames   ${NUM_FRAMES} \\
        --fps          ${FPS} \\
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
echo "Monitor:  squeue -u \$USER"
echo "Progress: find ${OUTPUT_DIR} -name video.mp4 | wc -l"
echo "Validate: python3 ${SCRIPT_DIR}/validate_stage1.py --output_dir ${OUTPUT_DIR}"
