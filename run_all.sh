#!/usr/bin/env bash
# End-to-end: install deps, translate JSON -> CADQuery .py, validate, finalize manifest.
# Usage:  bash run_all.sh [WORKERS]
set -euo pipefail

WORKERS="${1:-$(getconf _NPROCESSORS_ONLN)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[1/4] installing requirements"
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "[2/4] translating JSON -> CADQuery .py (workers=$WORKERS)"
python3 run_translation.py --no-validate --workers "$WORKERS"

echo "[3/4] validating with short timeout (workers=$WORKERS)"
python3 run_validation.py --workers "$WORKERS" --timeout 30

echo "[4/4] retrying timeouts with longer budget"
python3 run_validation.py --workers "$WORKERS" --timeout 300 --retry-only timeout

echo "Done. Manifest:           deepcad-data/cadquery_py/manifest.json"
echo "      Filtered split:     deepcad-data/cadquery_py/translated_split.json"
echo "      Generated CADQuery: deepcad-data/cadquery_py/<bucket>/<id>.py"
