#!/bin/bash
# M169-M180 Unified ags1 Experiment Runner
# Run: bash experiment/run_ags1_m169_m180.sh
# Server: ags1 (2×EPYC 9354 128核, 1.5TB RAM, 2×A6000+H100)
set -euo pipefail
cd "$(dirname "$0")/.."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p experiment/results experiment/logs
echo "Philemon-TSH: M169-M180 ags1 Runner ($TIMESTAMP)"
THREADS=$(nproc); [ "$THREADS" -gt 64 ] && THREADS=64
TOTAL_PASS=0; TOTAL_FAIL=0
run_exp() {
    local SRC=$1 NAME=$2 ARGS="${3:-}"
    local BIN="/tmp/$(basename $SRC .cpp)"
    echo "═══ $NAME ═══"
    g++ -std=c++17 -O2 -fopenmp -march=native -o "$BIN" "$SRC" -lpthread 2>&1 || { echo "COMPILE FAIL"; return; }
    LOG="experiment/logs/${NAME}_${TIMESTAMP}.log"
    timeout 1800 "$BIN" $ARGS > "$LOG" 2>&1 || true
    P=$(grep -c "PASS:" "$LOG" 2>/dev/null || echo 0)
    F=$(grep -c "FAIL:" "$LOG" 2>/dev/null || echo 0)
    TOTAL_PASS=$((TOTAL_PASS+P)); TOTAL_FAIL=$((TOTAL_FAIL+F))
    echo "  $P PASS, $F FAIL"
}
run_exp experiment/m169_m170_driver_workload_engine.cpp M169-M170 "--scale 14 --threads $THREADS --debug 1"
run_exp experiment/m171_m172_neograph_engine.cpp M171-M172 "--scale 14 --threads $THREADS --debug 1"
run_exp experiment/m173_m174_largescale_experiment.cpp M173-M174 "--scale 14 --threads $THREADS"
run_exp experiment/m175_m176_tiered_memory.cpp M175-M176 "--scale 14 --threads $THREADS --debug 1"
run_exp experiment/m177_m178_streaming_compaction.cpp M177-M178 "--scale 14 --threads $THREADS --debug 1"
run_exp experiment/m179_m180_final_integration.cpp M179-M180 ""
echo "Total: $TOTAL_PASS PASS, $TOTAL_FAIL FAIL"
git add experiment/results/ experiment/logs/ 2>/dev/null || true
git commit -m "ags1 run ($TIMESTAMP): $TOTAL_PASS pass $TOTAL_FAIL fail" --author="dylanyunlon <dogechat@163.com>" 2>/dev/null || true
git push origin main 2>/dev/null || echo "Push failed"
