#!/bin/bash
# Set up everything Stage 1 needs on FarmShare.
#
# What this does:
#   1. pip-installs opencv-python into your user site-packages
#      (cadquery 2.7.0 and numpy are already present via python/3.13.5)
#   2. Downloads a portable Blender 3.6 LTS binary to BLENDER_DIR
#      (skip with --skip-blender if you already have one)
#
# Usage:
#   bash setup_stage1_env.sh [--blender-dir /path/to/blender] [--skip-blender]

set -euo pipefail

BLENDER_DIR="/scratch/users/abzhong/blender"
SKIP_BLENDER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --blender-dir)  BLENDER_DIR="$2"; shift 2 ;;
        --skip-blender) SKIP_BLENDER=1;   shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------- 1. Python packages ----------
echo "=== Loading python/3.13.5 ==="
module load python/3.13.5

echo ""
echo "=== Checking existing packages ==="
python3 -c "import cadquery; print('  cadquery', cadquery.__version__, '✓')"
python3 -c "import numpy;    print('  numpy   ', numpy.__version__,    '✓')"

if python3 -c "import cv2" 2>/dev/null; then
    python3 -c "import cv2; print('  cv2     ', cv2.__version__,       '✓')"
else
    echo "  cv2 not found — installing …"
    pip install --user opencv-python
    python3 -c "import cv2; print('  cv2     ', cv2.__version__,       '✓')"
fi

# ---------- 2. Blender portable binary ----------
if [[ "$SKIP_BLENDER" -eq 1 ]]; then
    echo ""
    echo "=== Skipping Blender download (--skip-blender) ==="
else
    BLENDER_BIN="${BLENDER_DIR}/blender"
    if [[ -x "$BLENDER_BIN" ]]; then
        echo ""
        echo "=== Blender already present: $BLENDER_BIN ==="
        "$BLENDER_BIN" --version | head -1
    else
        echo ""
        echo "=== Downloading Blender 3.6 LTS → ${BLENDER_DIR} ==="
        mkdir -p "$BLENDER_DIR"

        ARCHIVE="blender-3.6.22-linux-x64.tar.xz"
        URL="https://download.blender.org/release/Blender3.6/${ARCHIVE}"

        echo "  Fetching $URL …"
        wget -q --show-progress -O "/tmp/${ARCHIVE}" "$URL"

        echo "  Extracting …"
        tar -xf "/tmp/${ARCHIVE}" -C "$BLENDER_DIR" --strip-components=1
        rm "/tmp/${ARCHIVE}"

        echo "  Verifying …"
        "$BLENDER_BIN" --version | head -1
        echo "  Blender installed at: $BLENDER_BIN"
    fi
fi

# ---------- summary ----------
BLENDER_BIN="${BLENDER_DIR}/blender"
echo ""
echo "=== Setup complete ==="
echo ""
echo "Run a smoke test with:"
echo "  module load python/3.13.5"
echo "  python3 stage1/generate_aruco.py --output aruco_marker.png"
echo "  python3 stage1/run_stage1.py \\"
echo "      --object_id 0000/00000007 \\"
echo "      --input_dir deepcad-data/cadquery_py \\"
echo "      --output_dir /scratch/users/abzhong/stage1_output \\"
echo "      --aruco_marker aruco_marker.png \\"
echo "      --blender ${BLENDER_BIN}"
