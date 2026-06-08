#!/bin/bash
# M175-M176: Tiered Memory Experiment runner for ags1
# Usage: ./experiment/run_m175_m176.sh [ci]
#
# ags1: 2×AMD EPYC 9354 (128 cores), 1.5TB RAM, 2×A6000+1×H100
# Run on NUMA node1 (GPU side) for HBM locality simulation.
#
# Results: experiment/results/m175_tiered_memory.csv
#          experiment/results/m175_tiered_memory.tex

set -e
cd "$(dirname "$0")/.."

MODE="${1:-full}"
BIN=/tmp/m175_m176

echo "════════════════════════════════════════════════════════"
echo "  M175-M176: Tiered Memory Experiment"
echo "  Mode: $MODE"
echo "════════════════════════════════════════════════════════"

# ─── Build ────────────────────────────────────────────────────────────────────
echo "[1/3] Building..."
g++ -std=c++17 -O2 -fopenmp -march=native \
    -o "$BIN" experiment/m175_m176_tiered_memory.cpp -lpthread
echo "  Build: OK"

mkdir -p experiment/results

# ─── Run ──────────────────────────────────────────────────────────────────────
if [ "$MODE" = "ci" ]; then
    echo "[2/3] Running CI mode (scales 14,16)..."
    "$BIN" --ci --debug 1
elif [ "$MODE" = "full" ]; then
    echo "[2/3] Running full mode (scales 20,22,24,26)..."
    # On ags1: use NUMA node1 for GPU-side memory locality
    if command -v numactl &>/dev/null; then
        echo "  Using numactl --cpunodebind=1 --membind=1"
        numactl --cpunodebind=1 --membind=1 \
            "$BIN" \
            --scales 20,22,24,26 \
            --threads 128 \
            --ef 16 \
            --iters 10 \
            --mig-threads 8 \
            --debug 1
    else
        "$BIN" \
            --scales 20,22,24,26 \
            --threads 128 \
            --ef 16 \
            --iters 10 \
            --mig-threads 8 \
            --debug 1
    fi
elif [ "$MODE" = "medium" ]; then
    echo "[2/3] Running medium mode (scales 16,18,20)..."
    "$BIN" \
        --scales 16,18,20 \
        --threads 32 \
        --ef 16 \
        --iters 10 \
        --mig-threads 4 \
        --debug 1
else
    echo "ERROR: unknown mode '$MODE' (use: ci | medium | full)"
    exit 1
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Results:"
if [ -f experiment/results/m175_tiered_memory.csv ]; then
    echo "  CSV: experiment/results/m175_tiered_memory.csv"
    head -20 experiment/results/m175_tiered_memory.csv
fi
if [ -f experiment/results/m175_tiered_memory.tex ]; then
    echo "  LaTeX: experiment/results/m175_tiered_memory.tex"
fi
echo ""
echo "════════════════════════════════════════════════════════"
echo "  M175-M176 Complete"
echo "════════════════════════════════════════════════════════"
