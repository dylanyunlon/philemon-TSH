#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# auto_experiment_push.sh — ags1服务器: 运行实验 → 自动push到git
# 
# 用法: bash experiment/auto_experiment_push.sh
# 前置: conda activate walking3
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOGS_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH Auto Experiment → Git Push                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date)] Starting on $(hostname)"

# ─── 1. Pull最新代码 ───
echo ">>> [1/5] git pull..."
git pull --rebase origin main || git pull origin main

# ─── 2. 编译实验 ───
echo ">>> [2/5] Compiling..."
g++ -std=c++17 -O2 -pthread \
    -o experiment/philemon_experiment \
    experiment/philemon_experiment.cpp 2>&1 | tail -5
echo "  Build OK"

# ─── 3. 运行实验（多种规模）───
echo ">>> [3/5] Running experiments..."
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# 小规模验证
echo "  [3a] Small scale (V=2000 E=10000)..."
./experiment/philemon_experiment > "${LOGS_DIR}/run_small_${TIMESTAMP}.log" 2>&1
echo "  Done: $(tail -1 ${LOGS_DIR}/run_small_${TIMESTAMP}.log)"

# 中规模
echo "  [3b] Medium scale (V=50000 E=500000)..."
./experiment/philemon_experiment "" 50000 500000 1 > "${LOGS_DIR}/run_medium_${TIMESTAMP}.log" 2>&1
echo "  Done: $(tail -1 ${LOGS_DIR}/run_medium_${TIMESTAMP}.log)"

# 大规模（如果内存允许）
echo "  [3c] Large scale (V=200000 E=2000000)..."
./experiment/philemon_experiment "" 200000 2000000 0 > "${LOGS_DIR}/run_large_${TIMESTAMP}.log" 2>&1 || echo "  Large scale failed (memory?)"

# ─── 4. 收集结果到CSV ───
echo ">>> [4/5] Collecting results..."
python3 << 'PYEOF'
import json, os, csv, glob

results_dir = os.environ.get("RESULTS_DIR", "experiment/results")
logs_dir = os.environ.get("LOGS_DIR", "experiment/logs")
timestamp = os.environ.get("TIMESTAMP", "unknown")

# 读取JSON结果
json_path = os.path.join(results_dir, "run_result.json")
if os.path.exists(json_path):
    with open(json_path) as f:
        data = json.load(f)
    print(f"  Latest result: V={data['config']['num_vertices']} "
          f"E={data['config']['num_edges']} RSS={data['memory_rss_kb']}KB")
    for algo, ms in data.get("timings_ms", {}).items():
        print(f"    {algo}: {ms}ms")

# 写入汇总CSV
csv_path = os.path.join(results_dir, f"auto_run_{timestamp}.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "vertices", "edges", "algorithm", "threads", "time_ms", "rss_kb"])
    if os.path.exists(json_path):
        for algo, ms in data.get("timings_ms", {}).items():
            parts = algo.rsplit("_T", 1)
            algo_name = parts[0] if len(parts)==2 else algo
            threads = int(parts[1]) if len(parts)==2 else 1
            w.writerow([timestamp, data['config']['num_vertices'],
                       data['config']['num_edges'], algo_name, threads, ms,
                       data['memory_rss_kb']])
print(f"  Saved: {csv_path}")
PYEOF

# ─── 5. Git push ───
echo ">>> [5/5] Git push..."
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"
git add experiment/results/ experiment/logs/
git commit -m "auto: experiment data ${TIMESTAMP} from $(hostname)" || echo "  Nothing to commit"
git push origin main 2>&1 || echo "  Push failed (will retry later)"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Auto experiment complete.                                ║"
echo "║  Claude workers: git pull to get latest data              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
