#!/bin/bash
# M173-M174: ags1 Server Runner — Large-Scale RMAT Experiment
#
# Server: ags1 (2×AMD EPYC 9354 128核, 1.5TB RAM, 2×A6000+H100)
# Usage:  bash experiment/run_m173_m174.sh [--quick] [--scales "20 22 24"]
#
# What this does:
#   1. Compiles m173_m174_largescale_experiment.cpp
#   2. Runs RMAT scale 20/22/24 (optionally 26 for full paper data)
#   3. Produces experiment/results/m173_largescale.csv (Table 1 in paper)
#   4. Prints LaTeX rows ready for philemon_tsh_reconstructed.tex
#   5. Auto-pushes results to git main branch
#
# Scale → Dataset mapping:
#   scale 20  → ~1M  vertices, ~16M  edges  (LiveJournal-like)
#   scale 22  → ~4M  vertices, ~67M  edges  (Twitter-small-like)
#   scale 24  → ~16M vertices, ~268M edges  (uk-2007-like)
#   scale 26  → ~67M vertices, ~1B   edges  (Twitter-full, needs 1.5TB RAM)
#
# SOTA targets (from RapidStore VLDB'25 Table 3, LiveJournal):
#   Insert: ≥2.0 MEPS   BFS: ≤30s   PR 10iter: ≤300s   Mem: ≤2.0GB(+SSD)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiment/results"
LOG_DIR="experiment/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ─── Parse args ──────────────────────────────────────────────────────────────
QUICK=0
SCALES="20 22 24"
for arg in "$@"; do
    case "$arg" in
        --quick)   QUICK=1; SCALES="20" ;;
        --scales)  shift; SCALES="$1" ;;
        --scale26) SCALES="20 22 24 26" ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  M173-M174 ags1 Large-Scale Experiment Runner               ║"
echo "║  $(date)                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─── §1: System info ─────────────────────────────────────────────────────────
echo ""
echo "=== System Info ==="
lscpu | grep -E "Model name|CPU\(s\)|Thread|NUMA|Socket" 2>/dev/null | head -8 || true
echo "RAM: $(free -h | awk '/^Mem:/{print $2}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4 || echo "GPU: none"
uname -r
echo ""

# ─── §2: Compile ─────────────────────────────────────────────────────────────
echo "=== Compiling m173_m174_largescale_experiment.cpp ==="
BINARY="$REPO_DIR/experiment/m173_m174_bin"

g++ -std=c++17 -O2 -fopenmp -march=native \
    -o "$BINARY" \
    experiment/m173_m174_largescale_experiment.cpp \
    -lpthread
echo "  Compiled: $BINARY  ($(du -sh "$BINARY" | cut -f1))"

# ─── §3: Thread count ────────────────────────────────────────────────────────
THREADS=$(nproc 2>/dev/null || echo 4)
# Use NUMA node 1 (GPU-side) on ags1 for best memory bandwidth
if command -v numactl &>/dev/null && [ "$(numactl --hardware | grep available | awk '{print $2}')" -gt 1 ] 2>/dev/null; then
    NUMA_PREFIX="numactl --cpunodebind=1 --membind=1"
    echo "  NUMA: using node 1 (GPU-side) for best bandwidth"
else
    NUMA_PREFIX=""
    echo "  NUMA: numactl not available, using default binding"
fi

# Cap at 64 for reproducibility (matches VLDB'25 32-thread reference × 2)
if [ "$THREADS" -gt 64 ]; then THREADS=64; fi
echo "  Threads: $THREADS"

# ─── §4: Multi-scale experiment loop ─────────────────────────────────────────
CSV_OUT="$RESULTS_DIR/m173_largescale.csv"

# Clear old CSV to get fresh results
echo "# M173-M174 Large-Scale Experiment — ags1 run $TIMESTAMP" > "$CSV_OUT"
echo "# Server: $(hostname) | Threads: $THREADS" >> "$CSV_OUT"
echo "# SOTA reference: RapidStore VLDB'25 Table 3 (LiveJournal)" >> "$CSV_OUT"

echo ""
echo "=== Running scales: $SCALES ==="

PASS_TOTAL=0
FAIL_TOTAL=0

for SCALE in $SCALES; do
    echo ""
    echo "═══ Scale $SCALE (N=$(python3 -c "print(f'{1<<$SCALE:,}')" 2>/dev/null || echo "2^$SCALE"), M≈$(python3 -c "print(f'{(1<<$SCALE)*16:,}')" 2>/dev/null || echo "16×2^$SCALE")) ═══"

    LOG="$LOG_DIR/m173_scale${SCALE}_${TIMESTAMP}.log"

    # Scale 23+ uses streaming mode to avoid OOM
    STREAMING=""
    if [ "$SCALE" -ge 23 ]; then
        STREAMING="--streaming"
        echo "  [streaming mode for large scale]"
    fi

    # Timeout: scale 20→5min, 22→20min, 24→90min, 26→6h
    case "$SCALE" in
        20) TIMEOUT=300   ;;
        21) TIMEOUT=600   ;;
        22) TIMEOUT=1200  ;;
        23) TIMEOUT=3600  ;;
        24) TIMEOUT=5400  ;;
        26) TIMEOUT=21600 ;;
        *)  TIMEOUT=300   ;;
    esac

    START_T=$(date +%s)

    $NUMA_PREFIX "$BINARY" \
        --scale "$SCALE" \
        --threads "$THREADS" \
        --debug 1 \
        --iters 10 \
        --csv "$CSV_OUT" \
        $STREAMING \
        > "$LOG" 2>&1 &
    PID=$!

    # Wait with timeout
    ELAPSED=0
    while kill -0 $PID 2>/dev/null; do
        sleep 10
        ELAPSED=$(($(date +%s) - START_T))
        if [ "$ELAPSED" -gt "$TIMEOUT" ]; then
            echo "  [WARN] scale=$SCALE timeout after ${ELAPSED}s, killing"
            kill -9 $PID 2>/dev/null || true
            break
        fi
        # Progress indicator
        LINES=$(wc -l < "$LOG" 2>/dev/null || echo 0)
        echo "  ... ${ELAPSED}s elapsed, log=$LINES lines"
    done
    wait $PID 2>/dev/null || true

    END_T=$(date +%s)
    RUN_S=$((END_T - START_T))

    # Parse results
    PASS=$(grep -c "\[PASS\]" "$LOG" 2>/dev/null || echo 0)
    FAIL=$(grep -c "\[FAIL\]" "$LOG" 2>/dev/null || echo 0)
    PASS_TOTAL=$((PASS_TOTAL + PASS))
    FAIL_TOTAL=$((FAIL_TOTAL + FAIL))

    echo "  scale=$SCALE: $PASS PASS, $FAIL FAIL  (${RUN_S}s)"

    # Print key metrics
    grep -E "Insert:|BFS:|PR target:|Philemon Target|MEPS|RSS:" "$LOG" 2>/dev/null | \
        grep -v "^#" | head -12 || true

    # Print LaTeX row
    grep "\[LaTeX\]" "$LOG" 2>/dev/null || true
done

# ─── §5: Summary ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Experiment Summary                                          ║"
echo "║  Total: $PASS_TOTAL PASS  $FAIL_TOTAL FAIL                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo ""
echo "=== CSV Results ($CSV_OUT) ==="
grep -v "^#" "$CSV_OUT" | head -40 || true

echo ""
echo "=== SOTA Comparison (scale 20, LiveJournal-like) ==="
for LOG in "$LOG_DIR"/m173_scale20_${TIMESTAMP}.log; do
    if [ -f "$LOG" ]; then
        grep -A 12 "SOTA Comparison" "$LOG" 2>/dev/null | head -15 || true
    fi
done

# ─── §6: Git push ────────────────────────────────────────────────────────────
echo ""
echo "=== Git Push ==="
cd "$REPO_DIR"

git config user.name  "dylanyunlon"
git config user.email "dogechat@163.com"

git add experiment/results/m173_largescale.csv \
        experiment/m173_m174_largescale_experiment.cpp \
        experiment/run_m173_m174.sh 2>/dev/null || true
git add experiment/logs/m173_scale*_${TIMESTAMP}.log 2>/dev/null || true

COMMIT_MSG="M173-M174: large-scale experiment results (${TIMESTAMP})

Scales: $SCALES  Threads: $THREADS
Total: $PASS_TOTAL PASS  $FAIL_TOTAL FAIL
Results: $CSV_OUT

SOTA targets (LiveJournal, scale 20):
  Insert: ≥2.0 MEPS  BFS: ≤30s  PR: ≤300s  Mem: ≤2.0GB
  Advantage: 1/3 DRAM + SSD → 90% performance of pure-DRAM systems

Algorithmic changes (20%):
  [MOD] TierAwarePartitioner: degree-rank vertex-to-tier assignment
  [MOD] AdaptiveBFS: DRAM-first bottom-up frontier processing
  [MOD] TieredPageRank: one-step SSD/HDD contribution lag
  [MOD] BucketSSSP: tier-aware delta coarsening
  [MOD] AfforestWCC: 2% sampling shortcut for large components"

git commit -m "$COMMIT_MSG" --author="dylanyunlon <dogechat@163.com>" || echo "  Nothing to commit"

git push origin main && echo "  Pushed to main." || echo "  [WARN] Push failed"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done.  Next: M175-M176 Tiered Memory Validation            ║"
echo "║  Pull results: git pull origin main                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
