#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Native Docker deployment (NOT Flower local simulation).
python scripts/run_flower_docker.py "$@"
