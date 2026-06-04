#!/bin/bash
# Submit Stage 1 as a SLURM array job — one task per CadQuery script.
#
# Prerequisites: run setup_stage1_env.sh first.
#
# Usage:
#   bash submit_stage1.sh \
#       --input_dir  deepcad-data/cadquery_py \
#       --output_dir /scratch/users/abzhong/stage1_output \
#       --aruco_marker aruco_marker.png \
#       [--blender /scratch/users/abzhong/blender/blender] \
#       [--engine BLENDER_EEVEE] \
#       [--num_frames 60] [--fps 24] [--time 01:00:00]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- defaults ----------
INPUT_DIR=""
OUTPUT_DIR=""
ARUCO_MARKER=""
BLENDER_EXE="/scratch/users/abzhong/blender/blender"
ENGINE="BLENDER_EEVEE"
NUM_FRAMES=60
FPS=24
TIME_LIMIT="01:00:00"
PARTITION="normal"
CPUS=8           # Blender EEVEE uses all available threads for tile rendering
MEM="24G"        # 3 GB/CPU, within the 4 GB/CPU cluster limit
BATCH_SIZE=1000  # FarmShare MaxArraySize = 1024
MAX_CONCURRENT=128  # Slurm MaxJobs per user on the normal QOS

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)    INPUT_DIR="$2";    shift 2 ;;
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
        --batch-size)      BATCH_SIZE="$2";      shift 2 ;;
        --max-concurrent)  MAX_CONCURRENT="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$ARUCO_MARKER" ]]; then
    echo "ERROR: --input_dir, --output_dir, and --aruco_marker are required."
    exit 1
fi

# Validate that key files/dirs exist before submitting
if [[ ! -f "$ARUCO_MARKER" ]]; then
    echo "ERROR: ArUco marker not found: $ARUCO_MARKER"
    echo "       Run: python3 ${SCRIPT_DIR}/generate_aruco.py --output $ARUCO_MARKER"
    exit 1
fi
if [[ ! -x "$BLENDER_EXE" ]]; then
    echo "ERROR: Blender executable not found or not executable: $BLENDER_EXE"
    echo "       Run: bash ${SCRIPT_DIR}/setup_stage1_env.sh"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" logs/stage1

# ---------- build manifest ----------
MANIFEST="${OUTPUT_DIR}/stage1_input_manifest.txt"
find "$INPUT_DIR" -name "*.py" | sort > "$MANIFEST"
N=$(wc -l < "$MANIFEST")

if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no .py scripts found under $INPUT_DIR"
    exit 1
fi

N_BATCHES=$(( (N + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Found $N scripts → submitting $N_BATCHES array job(s) of up to $BATCH_SIZE tasks (${CPUS} CPUs, ${MEM} RAM each, max ${MAX_CONCURRENT} concurrent) …"

# ---------- write self-contained sbatch worker ----------
# The worker reads: MANIFEST, BATCH_OFFSET (both exported by sbatch --export)
SBATCH_FILE="${OUTPUT_DIR}/stage1_array.sbatch"
cat > "$SBATCH_FILE" <<SBATCH
#!/bin/bash
#SBATCH --job-name=stage1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --output=logs/stage1/%A_%a.out
#SBATCH --error=logs/stage1/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
echo "=== batch_offset=\${BATCH_OFFSET} task=\$SLURM_ARRAY_TASK_ID job=\$SLURM_ARRAY_JOB_ID host=\$(hostname) ==="

module load python/3.13.5

# Global manifest line = batch offset + local task id  (manifest is 1-indexed)
MANIFEST_LINE=\$(( BATCH_OFFSET + SLURM_ARRAY_TASK_ID + 1 ))
OBJECT_PATH=\$(sed -n "\${MANIFEST_LINE}p" "${MANIFEST}")
if [[ -z "\$OBJECT_PATH" ]]; then
    echo "ERROR: no manifest entry at line \$MANIFEST_LINE"
    exit 1
fi

OBJECT_ID="\${OBJECT_PATH#${INPUT_DIR}/}"
OBJECT_ID="\${OBJECT_ID%.py}"
echo "Object ID: \$OBJECT_ID"

python3 ${SCRIPT_DIR}/run_stage1.py \
    --object_id    "\$OBJECT_ID" \
    --input_dir    "${INPUT_DIR}" \
    --output_dir   "${OUTPUT_DIR}" \
    --aruco_marker "${ARUCO_MARKER}" \
    --blender      "${BLENDER_EXE}" \
    --engine       "${ENGINE}" \
    --num_frames   ${NUM_FRAMES} \
    --fps          ${FPS}

echo "=== task \$SLURM_ARRAY_TASK_ID done at \$(date) ==="
SBATCH

# ---------- submit one array job per batch ----------
JOB_IDS=()
for (( batch=0; batch<N_BATCHES; batch++ )); do
    OFFSET=$(( batch * BATCH_SIZE ))
    # Tasks in this batch: 0 … min(BATCH_SIZE,remaining)-1
    REMAINING=$(( N - OFFSET ))
    LAST=$(( (REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE) - 1 ))

    JOB_ID=$(sbatch \
        --array="0-${LAST}%${MAX_CONCURRENT}" \
        --export=ALL,BATCH_OFFSET="${OFFSET}" \
        "$SBATCH_FILE" | awk '{print $NF}')
    JOB_IDS+=("$JOB_ID")
    echo "  Batch $((batch+1))/${N_BATCHES}: offset=${OFFSET}, tasks 0-${LAST} → job ${JOB_ID}"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} array job(s): ${JOB_IDS[*]}"
echo "Monitor:  squeue -u \$USER"
echo "Validate: python3 ${SCRIPT_DIR}/validate_stage1.py --output_dir ${OUTPUT_DIR}"
