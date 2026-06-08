#!/bin/bash
# M179-M180: Final Paper Data Integration + LaTeX Table Update
# Run on ags1 (128 cores, 1.5TB RAM) after all M169-M178 experiments complete
#
# Usage: ./experiment/run_m179_m180.sh [--debug 2] [--full]

set -e
cd "$(dirname "$0")/.."

# ─── Config ───────────────────────────────────────────────────────────────────
DEBUG="${DEBUG:-1}"
FULL="${FULL:-0}"
RESULTS_DIR="experiment/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG="$2"; shift 2;;
        --full)  FULL=1; shift;;
        --results-dir) RESULTS_DIR="$2"; shift 2;;
        *) shift;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  M179-M180: Final Paper Data Integration + LaTeX Tables     ║"
echo "║  Capstone: aggregates M169+M171+M173+M175+M177 → paper      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Config: debug=${DEBUG}  full=${FULL}  results_dir=${RESULTS_DIR}"
echo "  Timestamp: ${TIMESTAMP}"
echo ""

# ─── Verify prerequisite CSVs ─────────────────────────────────────────────────
echo "=== Checking prerequisite experiment outputs ==="
MISSING=0
for csv in m159_paper_data.csv m161_paper_data.csv m165_paper_data.csv \
           m171_neograph_data.csv m173_largescale.csv \
           m175_tiered_memory.csv m177_streaming.csv; do
    if [ -f "${RESULTS_DIR}/${csv}" ]; then
        echo "  ✓ ${csv}"
    else
        echo "  ✗ ${csv}  (MISSING — run prior experiments first)"
        MISSING=$((MISSING + 1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "  WARNING: ${MISSING} CSV file(s) missing."
    echo "  Live regression will still pass; LaTeX tables will use fallback data."
fi

# ─── Build ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Building m179_m180 ==="
g++ -std=c++17 -O2 -fopenmp -fstack-protector-strong \
    -o /tmp/m179_m180_runner \
    experiment/m179_m180_final_integration.cpp \
    -lpthread

echo "  Build: OK"

# ─── Smoke test (live regression only) ────────────────────────────────────────
echo ""
echo "=== Smoke test (live regression, no external CSVs needed) ==="
/tmp/m179_m180_runner \
    --debug 0 \
    --results-dir "${RESULTS_DIR}" | grep -E "PASS|FAIL|Summary|44 PASS"

# ─── Main run ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Main integration run ==="

EXTRA_ARGS=""
if [ "${FULL}" -eq 1 ]; then
    EXTRA_ARGS="--full"
fi

if command -v numactl &> /dev/null; then
    # ags1: NUMA node1 (GPU side)
    numactl --cpunodebind=1 --membind=1 \
        /tmp/m179_m180_runner \
            --debug "${DEBUG}" \
            --results-dir "${RESULTS_DIR}" \
            ${EXTRA_ARGS}
else
    /tmp/m179_m180_runner \
        --debug "${DEBUG}" \
        --results-dir "${RESULTS_DIR}" \
        ${EXTRA_ARGS}
fi

# ─── Archive results ──────────────────────────────────────────────────────────
echo ""
echo "=== Archiving results ==="
mkdir -p "${RESULTS_DIR}"

for out in m179_paper_tables.tex m179_final_summary.csv m179_regression_report.txt; do
    if [ -f "${RESULTS_DIR}/${out}" ]; then
        cp "${RESULTS_DIR}/${out}" "${RESULTS_DIR}/${out%.tex}_${TIMESTAMP}.${out##*.}" 2>/dev/null || true
        echo "  Archived: ${out}"
    fi
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== M179-M180 Outputs ==="
echo "  ${RESULTS_DIR}/m179_paper_tables.tex     — Table 1-7 + Figure + Appendix"
echo "  ${RESULTS_DIR}/m179_final_summary.csv    — master result archive"
echo "  ${RESULTS_DIR}/m179_regression_report.txt — 44-check regression log"
echo ""
echo "=== M179-M180 complete ==="
