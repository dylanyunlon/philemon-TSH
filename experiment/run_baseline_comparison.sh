#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Philemon-TSH vs DGS (SIGMOD'26) 公平对比实验
# 运行位置: ags1 服务器 (conda activate walking3)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/experiment/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH vs DGS Baseline Comparison                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"

mkdir -p "$RESULTS_DIR"

# ─── 1. 编译 Philemon-TSH ───
echo ">>> [1/4] Building Philemon-TSH..."
cd "$PROJECT_DIR"
g++ -std=c++17 -O2 -pthread \
    -o experiment/philemon_experiment \
    experiment/philemon_experiment.cpp 2>&1
echo "  Philemon-TSH build OK"

# ─── 2. 编译 DGS (如果存在) ───
echo ">>> [2/4] Building DGS baseline..."
DGS_DIR="${PROJECT_DIR}/../DynamicGraphStorage"
if [ ! -d "$DGS_DIR" ]; then
    echo "  DGS not found, cloning..."
    git clone --depth=1 https://github.com/SJTU-Liquid/DynamicGraphStorage.git "$DGS_DIR"
fi
cd "$DGS_DIR"
if [ ! -d build ]; then
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    make -j$(nproc) 2>&1 | tail -5
else
    cd build
    make -j$(nproc) 2>&1 | tail -5
fi
echo "  DGS build done (check for errors above)"

# ─── 3. 运行Philemon-TSH实验 ───
echo ">>> [3/4] Running Philemon-TSH experiments..."
cd "$PROJECT_DIR"

OUTCSV="${RESULTS_DIR}/baseline_comparison_${TIMESTAMP}.csv"
echo "system,vertices,edges,algorithm,threads,time_ms,memory_kb" > "$OUTCSV"

for SCALE in "2000 10000" "100000 500000" "500000 2500000"; do
    read V E <<< "$SCALE"
    echo "  Scale: V=$V E=$E"
    
    # Philemon-TSH
    OUTPUT=$(./experiment/philemon_experiment "" $V $E 0 2>&1)
    
    # 解析时间
    echo "$OUTPUT" | grep -oP '\[TIMER·END\]\s+(\S+)\s+→\s+(\d+)\s+ms' | while read line; do
        ALGO=$(echo "$line" | grep -oP '(?<=\] )\S+')
        MS=$(echo "$line" | grep -oP '\d+(?= ms)')
        echo "philemon,$V,$E,$ALGO,1,$MS,0" >> "$OUTCSV"
    done
    
    # 解析内存
    RSS=$(echo "$OUTPUT" | grep -oP '(?<=Total RSS: )\d+')
    echo "philemon,$V,$E,TOTAL,1,0,$RSS" >> "$OUTCSV"
done

# ─── 4. 汇总 ───
echo ">>> [4/4] Results summary..."
echo ""
cat "$OUTCSV"
echo ""
echo "Results saved to: $OUTCSV"

# Git push
cd "$PROJECT_DIR"
git config user.name "dylanyunlon" 2>/dev/null
git config user.email "dogechat@163.com" 2>/dev/null
git add experiment/results/ 2>/dev/null
git commit -m "auto: baseline comparison ${TIMESTAMP}" 2>/dev/null || true
git push origin main 2>/dev/null || echo "Push failed (run manually)"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Comparison complete.                                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
