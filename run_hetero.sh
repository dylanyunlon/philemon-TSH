#!/bin/bash
# run_hetero.sh — Run Philemon-TSH heterogeneous benchmark with full logging
#
# Usage:
#   cd /data/jiacheng/system/cache/temp/nips2026/0510
#   bash run_hetero.sh            # default: all experiments
#   bash run_hetero.sh 2>&1 | tee results/hetero_$(date +%Y%m%d_%H%M%S).log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/hetero_bench"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${RESULTS_DIR}/hetero_${TIMESTAMP}.log"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH Heterogeneous Benchmark Runner         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Pre-flight checks
echo "=== System Info ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo ""

# GPU status
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# NUMA topology (all GPUs should be on NUMA1 for ags1)
echo "=== NUMA Binding ==="
echo "Binding to NUMA1 (GPU-local node)"
echo ""

# Set CUDA devices visible (ensure correct order: 0=A6000, 1=A6000, 2=H100)
export CUDA_VISIBLE_DEVICES=0,1,2

# Disable ECC correction overhead on A6000 (if writable — usually not)
# nvidia-smi -i 0 --ecc-config=0 2>/dev/null || true
# nvidia-smi -i 1 --ecc-config=0 2>/dev/null || true

# Set persistence mode for stable clocks
nvidia-smi -pm 1 2>/dev/null || true

# Lock GPU clocks for reproducible results (may require root)
for gpu in 0 1 2; do
    nvidia-smi -i $gpu -lgc $(nvidia-smi -i $gpu --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null || echo "0") 2>/dev/null || true
done

echo "=== Running Benchmark ==="
echo "Output: ${LOG}"
echo ""

# Run with NUMA binding to node 1 (where GPUs are)
if command -v numactl &>/dev/null; then
    numactl --cpunodebind=1 --membind=1 "${BIN}" 2>&1 | tee "${LOG}"
else
    echo "WARNING: numactl not found, running without NUMA binding"
    "${BIN}" 2>&1 | tee "${LOG}"
fi

echo ""
echo "=== Results saved to ${LOG} ==="

# Reset GPU clocks
for gpu in 0 1 2; do
    nvidia-smi -i $gpu -rgc 2>/dev/null || true
done
