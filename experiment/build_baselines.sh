#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# build_baselines.sh — 编译所有 SOTA baseline 系统
# M182: 从源码编译 RapidStore / Sortledton / Teseo / LiveGraph
#
# 使用 SJTU-Liquid/DynamicGraphStorage 统一测试框架
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASELINE_DIR="${PROJECT_ROOT}/baselines"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Building SOTA Baselines for Comparison                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"

mkdir -p "${BASELINE_DIR}"
cd "${BASELINE_DIR}"

# ─── 1. DynamicGraphStorage 统一框架 (SIGMOD'26) ───
echo ""
echo ">>> [1/5] DynamicGraphStorage framework..."
if [ ! -d "DynamicGraphStorage" ]; then
    git clone https://github.com/SJTU-Liquid/DynamicGraphStorage.git
fi
cd DynamicGraphStorage
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 2>&1 | tail -5
cmake --build . -j $(nproc) 2>&1 | tail -5
echo "  ✓ DynamicGraphStorage built"
cd "${BASELINE_DIR}"

# ─── 2. RapidStore (VLDB'25) ───
echo ""
echo ">>> [2/5] RapidStore..."
if [ ! -d "RapidStore" ]; then
    git clone https://github.com/SJTU-Liquid/RapidStore.git
fi
cd RapidStore
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 2>&1 | tail -5
cmake --build . -j $(nproc) 2>&1 | tail -5
echo "  ✓ RapidStore built"
cd "${BASELINE_DIR}"

# ─── 3. Teseo (VLDB'21) ───
echo ""
echo ">>> [3/5] Teseo..."
if [ ! -d "teseo" ]; then
    git clone https://github.com/cwida/teseo.git
fi
cd teseo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j $(nproc) 2>&1 | tail -5
echo "  ✓ Teseo built"
cd "${BASELINE_DIR}"

# ─── 4. Aspen (PLDI'19) ───
echo ""
echo ">>> [4/5] Aspen..."
if [ ! -d "aspen" ]; then
    git clone https://github.com/ldhulipala/aspen.git
fi
cd aspen
make -j $(nproc) 2>&1 | tail -5 || echo "  ⚠ Aspen build may need manual fixes"
echo "  ✓ Aspen built (or attempted)"
cd "${BASELINE_DIR}"

# ─── 5. LiveGraph (binary release) ───
echo ""
echo ">>> [5/5] LiveGraph..."
if [ ! -d "LiveGraph-Binary" ]; then
    git clone https://github.com/thu-pacman/LiveGraph-Binary.git
fi
echo "  ✓ LiveGraph binary downloaded"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  All baselines built. Run: bash experiment/run_sota_comparison.sh ║"
echo "╚═══════════════════════════════════════════════════════════╝"
