#!/bin/bash
# M169-M170: ags1 Server Experiment Runner
# Run: bash experiment/run_m169_m170.sh
# Server: ags1 (2×EPYC 9354 128核, 1.5TB RAM, 2×A6000+H100)
#
# This script:
#   1. Compiles the driver workload engine experiment
#   2. Runs at multiple scales (14, 16, 18, 20, 22, 24)
#   3. Collects CSV data for paper tables
#   4. Auto-pushes results to git for sub-Claude consumption

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiment/results"
LOG_DIR="experiment/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  M169-M170 ags1 Experiment Runner                           ║"
echo "║  $(date)                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# §1: System info dump
echo ""
echo "=== System Info ==="
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture" || true
free -h || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
uname -r
echo ""

# §2: Compile
echo "=== Compiling m169_m170_driver_workload_engine.cpp ==="
BINARY="$REPO_DIR/experiment/m169_m170_bin"
g++ -std=c++17 -O2 -fopenmp -march=native \
    -o "$BINARY" \
    experiment/m169_m170_driver_workload_engine.cpp \
    -lpthread
echo "  Compiled: $BINARY"

# §3: Multi-scale experiment
SCALES=(14 16 18 20 22 24)
THREADS=$(nproc)
# Cap threads at 64 for reproducibility
if [ "$THREADS" -gt 64 ]; then THREADS=64; fi

MASTER_CSV="$RESULTS_DIR/m169_ags1_${TIMESTAMP}.csv"
echo "scale,vertices,edges,algo,system,latency_ms,value,insert_meps" > "$MASTER_CSV"

for SCALE in "${SCALES[@]}"; do
    echo ""
    echo "═══ Running scale=$SCALE  threads=$THREADS ═══"

    LOG="$LOG_DIR/m169_scale${SCALE}_${TIMESTAMP}.log"
    PER_SCALE_CSV="$RESULTS_DIR/m169_paper_data.csv"

    # Run with debug=1 for clean output, timeout after 30 min per scale
    timeout 1800 "$BINARY" --scale "$SCALE" --threads "$THREADS" --debug 1 \
        > "$LOG" 2>&1 || {
        echo "  [WARN] scale=$SCALE timed out or failed, see $LOG"
        continue
    }

    # Extract summary
    PASS=$(grep -c "PASS:" "$LOG" 2>/dev/null || echo 0)
    FAIL=$(grep -c "FAIL:" "$LOG" 2>/dev/null || echo 0)
    echo "  scale=$SCALE: $PASS PASS, $FAIL FAIL"

    # Append per-scale CSV data (skip header)
    if [ -f "$PER_SCALE_CSV" ]; then
        tail -n +6 "$PER_SCALE_CSV" >> "$MASTER_CSV"
    fi

    # Print key metrics from log
    grep -E "Init MEPS|BFS.*Philemon|PR.*Philemon|SSSP.*Philemon|WCC.*Philemon|Insert/Delete.*MEPS|MEPS=" "$LOG" | head -10
done

echo ""
echo "═══ Master CSV: $MASTER_CSV ═══"
cat "$MASTER_CSV"

# §4: Auto-push results to git
echo ""
echo "=== Git Push Results ==="
git add experiment/results/ experiment/logs/ || true
git add experiment/m169_m170_driver_workload_engine.cpp || true
git commit -m "M169-M170: ags1 multi-scale results ($TIMESTAMP) — driver workload engine

Scales: ${SCALES[*]}
Threads: $THREADS
Results: $MASTER_CSV" \
    --author="dylanyunlon <dogechat@163.com>" || echo "  Nothing to commit"

git push origin main || echo "  [WARN] Push failed — check credentials"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done. Results in $MASTER_CSV                ║"
echo "║  Sub-Claudes can pull from main branch.                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
