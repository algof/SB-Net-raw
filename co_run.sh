#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt

# Allow overriding datasets via env var, default to all
DATASETS_ARG=${DATASETS_ARG:-all}
echo "Running datasets: ${DATASETS_ARG}"

python scripts/run_all.py --datasets "${DATASETS_ARG}"