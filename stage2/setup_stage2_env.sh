#!/bin/bash
# Install Stage 2 Python dependencies (run once).
#
# Usage:
#   bash stage2/setup_stage2_env.sh

set -euo pipefail

echo "=== Stage 2 environment setup ==="

module load python/3.13.5

PACKAGES=(OpenEXR Imath numpy-stl plyfile trimesh)

for pkg in "${PACKAGES[@]}"; do
    mod="${pkg//-/_}"
    if python3 -c "import ${mod}" 2>/dev/null; then
        echo "  ${pkg} already installed"
    else
        echo "  Installing ${pkg} …"
        pip install --user "${pkg}"
    fi
done

echo ""
echo "Verifying imports …"
python3 -c "
import OpenEXR;        print(f'  OpenEXR {OpenEXR.__version__}')
import Imath;          print(f'  Imath ok')
import stl;            print(f'  numpy-stl ok')
import plyfile;        print(f'  plyfile ok')
import numpy as np;    print(f'  numpy {np.__version__}')
"

echo ""
echo "Stage 2 environment ready."
