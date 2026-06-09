#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# ags1服务器上初始化baseline对比环境
# 复用已有的 conda walking3 环境
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

BASE_DIR="/data/jiacheng/system/cache/temp/atc2026"
cd "$BASE_DIR"

echo "[1] Updating Philemon-TSH..."
if [ -d philemon-TSH ]; then
    cd philemon-TSH && git pull origin main && cd ..
else
    git clone https://github.com/dylanyunlon/philemon-TSH.git
fi

echo "[2] Cloning DGS baseline..."
if [ ! -d DynamicGraphStorage ]; then
    git clone --depth=1 https://github.com/SJTU-Liquid/DynamicGraphStorage.git
fi

echo "[3] Installing dependencies..."
sudo apt install -y libboost-all-dev libtbb-dev 2>/dev/null || true

echo "[4] Building DGS..."
cd DynamicGraphStorage
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
make -j32 2>&1 | tail -10
cd "$BASE_DIR"

echo "[5] Building Philemon-TSH experiment..."
cd philemon-TSH
g++ -std=c++17 -O2 -pthread \
    -o experiment/philemon_experiment \
    experiment/philemon_experiment.cpp

echo "[6] Quick test..."
./experiment/philemon_experiment "" 2000 10000 0 2>&1 | grep "RESULT"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Environment ready! Run:"
echo "  bash experiment/run_baseline_comparison.sh"
echo "═══════════════════════════════════════════════════════════"
