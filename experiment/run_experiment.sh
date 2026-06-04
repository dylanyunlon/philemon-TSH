#!/bin/bash
# run_experiment.sh — Philemon-TSH 全流程实验运行脚本
#
# 对标 upstream/rapidstore/run.sh 的 numactl + vtune 风格
# 在无GPU/numactl环境下优雅降级
#
# 用法:
#   bash run_experiment.sh                    # 默认合成数据
#   bash run_experiment.sh config.cfg         # 指定配置文件
#   bash run_experiment.sh "" 10000 50000 2   # CLI: vertices edges debug_level

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/philemon_experiment.cpp"
BIN="${SCRIPT_DIR}/philemon_experiment"
CFG="${SCRIPT_DIR}/philemon_experiment.cfg"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOGS_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${LOGS_DIR}/experiment_${TIMESTAMP}.log"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH Experiment Runner Script                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ─── System Info ─────────────────────────────────────────────────
echo "=== System Info ==="
echo "Date:     $(date)"
echo "Hostname: $(hostname 2>/dev/null || echo 'unknown')"
echo "Kernel:   $(uname -r)"
echo "CPU:      $(nproc) cores"
echo "RAM:      $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'N/A')"
echo ""

# GPU 状态 (可选)
echo "=== GPU Status ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader 2>/dev/null || echo "nvidia-smi query failed"
else
    echo "No nvidia-smi (CPU-only mode)"
fi
echo ""

# NUMA 信息 (可选)
echo "=== NUMA Topology ==="
if command -v numactl &>/dev/null; then
    numactl --hardware 2>/dev/null | head -5 || echo "numactl hardware query failed"
else
    echo "numactl not available"
fi
echo ""

# ─── Conda 环境检测 ──────────────────────────────────────────────
echo "=== Compiler ==="
g++ --version 2>/dev/null | head -1 || echo "g++ not found"
echo ""

# ─── Build ───────────────────────────────────────────────────────
echo "=== Building ==="
echo "Source: ${SRC}"
echo "Output: ${BIN}"

g++ -std=c++17 -O2 -pthread -Wall -Wextra -Wno-unused-parameter \
    -DPHILE_DEBUG=1 \
    -o "${BIN}" "${SRC}"

echo "Build OK ($(du -h "${BIN}" | cut -f1))"
echo ""

# ─── Run ─────────────────────────────────────────────────────────
echo "=== Running Experiment ==="
echo "Config: ${1:-${CFG}}"
echo "Log:    ${LOG}"
echo ""

# 选择运行方式: 有numactl就绑NUMA, 没有就直接跑
RUN_CMD="${BIN}"
if [ $# -ge 1 ] && [ -n "$1" ]; then
    RUN_CMD="${BIN} $1"
elif [ -f "${CFG}" ]; then
    RUN_CMD="${BIN} ${CFG}"
fi

# 追加 CLI 参数 (vertices, edges, debug_level)
if [ $# -ge 2 ]; then RUN_CMD="${RUN_CMD} $2"; fi
if [ $# -ge 3 ]; then RUN_CMD="${RUN_CMD} $3"; fi
if [ $# -ge 4 ]; then RUN_CMD="${RUN_CMD} $4"; fi

echo "Command: ${RUN_CMD}"
echo ""

if command -v numactl &>/dev/null; then
    echo "[NUMA] Binding to node 0"
    numactl --cpunodebind=0 --membind=0 ${RUN_CMD} 2>&1 | tee "${LOG}"
else
    ${RUN_CMD} 2>&1 | tee "${LOG}"
fi

echo ""
echo "=== Experiment finished ==="
echo "Log:     ${LOG}"
echo "Results: ${RESULTS_DIR}/run_result.json"
echo ""

# ─── 后处理: 从日志提取关键指标 ──────────────────────────────────
echo "=== Key Metrics ==="
grep -E '^\[TIMER·END\]' "${LOG}" 2>/dev/null || echo "(no timer data)"
echo ""
grep -E '^\[.*·RESULT\]' "${LOG}" 2>/dev/null || echo "(no result data)"
echo ""
grep -E '^\[TIER·ACCESS\]' "${LOG}" 2>/dev/null || echo "(no tier data)"
echo ""
echo "Done."
