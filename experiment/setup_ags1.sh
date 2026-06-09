#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# setup_ags1.sh — Philemon-TSH 实验环境初始化 (ags1 服务器)
# M181: 环境搭建
#
# 用法: bash experiment/setup_ags1.sh
# 前置: 已有 conda base 环境
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="atc26"
CUDA_VERSION="12.4"
WORK_DIR="/data/jiacheng/system/cache/temp/atc2026"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH ATC'26 — Environment Setup (ags1)          ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ─── 1. 检查硬件 ───
echo ""
echo ">>> [1/7] Checking hardware..."
nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  Current CUDA toolkit: $(nvcc --version 2>/dev/null | tail -1 || echo 'not installed')"

# ─── 2. 创建 conda 环境 ───
echo ""
echo ">>> [2/7] Creating conda environment: ${ENV_NAME}..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment ${ENV_NAME} already exists, activating..."
else
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# 激活环境 (在脚本中需要 source)
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# ─── 3. 安装 CUDA toolkit ───
echo ""
echo ">>> [3/7] Installing CUDA ${CUDA_VERSION} toolkit..."
# 使用 conda 安装 CUDA toolkit (不影响系统 driver)
conda install -y -c nvidia cuda-toolkit=${CUDA_VERSION} cuda-nvcc=${CUDA_VERSION}

# 验证
echo "  New nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo 'FAILED')"

# ─── 4. 安装 C++ 构建工具 ───
echo ""
echo ">>> [4/7] Installing build tools..."
conda install -y cmake ninja gxx_linux-64 make
pip install matplotlib pandas numpy scipy

# ─── 5. 克隆/更新项目 ───
echo ""
echo ">>> [5/7] Setting up project..."
cd "${WORK_DIR}" 2>/dev/null || mkdir -p "${WORK_DIR}" && cd "${WORK_DIR}"

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

# ─── 6. 编译 Philemon-TSH ───
echo ""
echo ">>> [6/7] Building Philemon-TSH..."
mkdir -p build && cd build
cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CUDA_ARCHITECTURES="86;90" \
    -DCMAKE_CXX_FLAGS="-O3 -march=znver4"
cmake --build . -j $(nproc)
cd ..

# ─── 7. 下载数据集 ───
echo ""
echo ">>> [7/7] Downloading datasets..."
mkdir -p data/ldbc data/snap data/rmat

# SNAP datasets
for ds in email-Enron wiki-Vote soc-LiveJournal1; do
    if [ ! -f "data/snap/${ds}.txt" ]; then
        echo "  Downloading ${ds}..."
        wget -q "https://snap.stanford.edu/data/${ds}.txt.gz" -O "data/snap/${ds}.txt.gz" 2>/dev/null && \
            gunzip -f "data/snap/${ds}.txt.gz" || echo "  WARNING: ${ds} download failed"
    fi
done

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                          ║"
echo "║                                                           ║"
echo "║  Next steps:                                              ║"
echo "║  1. conda activate ${ENV_NAME}                            ║"
echo "║  2. bash experiment/build_baselines.sh                    ║"
echo "║  3. bash experiment/run_sota_comparison.sh                ║"
echo "╚═══════════════════════════════════════════════════════════╝"
