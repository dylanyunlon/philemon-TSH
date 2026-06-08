#!/bin/bash
# M177-M178: Streaming + Compaction Experiment Runner
# Run on ags1 (128 cores, 1.5TB RAM) for paper-quality data
#
# Usage: ./experiment/run_m177_m178.sh [--scale 14] [--threads 128] [--flushes 256]

set -e
cd "$(dirname "$0")/.."

# ─── Config ───────────────────────────────────────────────────────────────────
SCALE="${SCALE:-14}"
THREADS="${THREADS:-4}"
FLUSHES="${FLUSHES:-256}"
PARTS_PER_FLUSH="${PARTS:-30}"
EDGES_PER_PART="${EDGES:-5000}"
DEBUG="${DEBUG:-1}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiment/results"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  M177-M178: Streaming + Compaction (RQ5) — ags1 Run          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Config: scale=${SCALE}  threads=${THREADS}  flushes=${FLUSHES}"
echo "  parts/flush=${PARTS_PER_FLUSH}  edges/part=${EDGES_PER_PART}"
echo ""

# ─── Build ────────────────────────────────────────────────────────────────────
echo "=== Building m177_m178 ==="
g++ -std=c++17 -O2 -fopenmp -fstack-protector-strong \
    -o /tmp/m177_m178_runner \
    experiment/m177_m178_streaming_compaction.cpp \
    -lpthread

echo "  Build: OK"

# ─── Validate (quick smoke test) ──────────────────────────────────────────────
echo ""
echo "=== Smoke test (flushes=32) ==="
/tmp/m177_m178_runner \
    --flushes 32 \
    --parts 10 \
    --threads "${THREADS}" \
    --debug 1 | grep -E "PASS|FAIL|Summary"

# ─── Main experiment (paper-scale) ────────────────────────────────────────────
echo ""
echo "=== Main experiment (paper-scale) ==="
if command -v numactl &> /dev/null; then
    # ags1: NUMA node1 (GPU side) for best latency
    numactl --cpunodebind=1 --membind=1 \
        /tmp/m177_m178_runner \
            --flushes "${FLUSHES}" \
            --parts "${PARTS_PER_FLUSH}" \
            --scale "${SCALE}" \
            --threads "${THREADS}" \
            --debug "${DEBUG}"
else
    /tmp/m177_m178_runner \
        --flushes "${FLUSHES}" \
        --parts "${PARTS_PER_FLUSH}" \
        --scale "${SCALE}" \
        --threads "${THREADS}" \
        --debug "${DEBUG}"
fi

# ─── Archive results ──────────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"
if [ -f "${RESULTS_DIR}/m177_streaming.csv" ]; then
    cp "${RESULTS_DIR}/m177_streaming.csv" \
       "${RESULTS_DIR}/m177_streaming_${TIMESTAMP}.csv"
    echo ""
    echo "  Results archived: ${RESULTS_DIR}/m177_streaming_${TIMESTAMP}.csv"
fi

echo ""
echo "=== M177-M178 complete ==="
