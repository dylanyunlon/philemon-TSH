#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# setup_ags1.sh — Philemon-TSH 实验环境初始化 (ags1 服务器)
# M181: 复用已有 conda walking3 环境
#
# 用法: bash experiment/setup_ags1.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="walking3"
WORK_DIR="/data/jiacheng/system/cache/temp/atc2026"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH ATC'26 — Environment Setup (ags1)          ║"
echo "║  Reusing conda env: ${CONDA_ENV}                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ─── 1. 检查硬件 ───
echo ""
echo ">>> [1/5] Checking hardware..."
nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# ─── 2. 激活 conda walking3 ───
echo ""
echo ">>> [2/5] Activating conda env: ${CONDA_ENV}..."
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"

# ─── 3. 安装 Philemon 额外依赖 (C++ 编译需要) ───
echo ""
echo ">>> [3/5] Installing C++ build deps into ${CONDA_ENV}..."
pip install matplotlib pandas numpy scipy 2>/dev/null || true
# cmake/ninja 如果 conda 里没有就从 conda 装
which cmake >/dev/null 2>&1 || conda install -y cmake ninja

# ─── 4. 克隆/更新项目 ───
echo ""
echo ">>> [4/5] Setting up project..."
mkdir -p "${WORK_DIR}" && cd "${WORK_DIR}"

if [ -d "philemon-TSH" ]; then
    echo "  Project exists, pulling latest..."
    cd philemon-TSH && git pull origin main
else
    echo "  Cloning project..."
    git clone https://github.com/dylanyunlon/philemon-TSH.git
    cd philemon-TSH
fi

git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# ─── 5. 编译 Philemon-TSH ───
echo ""
echo ">>> [5/5] Building Philemon-TSH..."
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CUDA_ARCHITECTURES="86;90" \
    -DCMAKE_CXX_FLAGS="-O3 -march=znver4"
cmake --build . -j $(nproc)
cd ..

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                          ║"
echo "║                                                           ║"
echo "║  Usage:                                                   ║"
echo "║    conda activate ${CONDA_ENV}                            ║"
echo "║    cd ${WORK_DIR}/philemon-TSH                            ║"
echo "║    bash experiment/build_baselines.sh                     ║"
echo "║    bash experiment/run_sota_comparison.sh                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
