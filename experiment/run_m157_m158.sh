#!/bin/bash
# M157-M158 SOTA Baseline Experiment Runner
# For ags1: 2×EPYC 9354, 1.5TB RAM, A6000+H100
# Conda env: reuse base (gcc 13+, OpenMP)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/m157_m158_sota_baseline_experiment.cpp"
BIN="${SCRIPT_DIR}/m157_m158"
LOGS="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${LOGS}/m157_m158_${TIMESTAMP}.log"

echo "=== M157-M158 SOTA Baseline ==="
echo "Date: $(date)"
echo "Host: $(hostname 2>/dev/null || echo unknown)"
echo "CPU:  $(nproc) cores"
echo "RAM:  $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo N/A)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
echo ""

# Build
g++ -std=c++17 -O2 -fopenmp -march=native -Wno-unused-parameter -o "${BIN}" "${SRC}" -lpthread
echo "Build OK ($(du -h "${BIN}" | cut -f1))"

# Run at multiple scales
for SCALE in 14 16 18 20; do
    echo ""
    echo "--- Scale ${SCALE} ---"
    if command -v numactl &>/dev/null; then
        numactl --cpunodebind=1 --membind=1 "${BIN}" --scale ${SCALE} --threads 32 --debug 1 2>&1 | tee -a "${LOG}"
    else
        "${BIN}" --scale ${SCALE} --threads 32 --debug 1 2>&1 | tee -a "${LOG}"
    fi
done

echo ""
echo "Log: ${LOG}"
echo "Done."
