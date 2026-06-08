#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# run_ags1_real_sota.sh — 真实SOTA对比 (无sudo, 用conda)
#
# 在ags1上运行:
#   conda activate base
#   bash experiment/run_ags1_real_sota.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

TS=$(date +%Y%m%d_%H%M%S)
BUILD="$REPO_DIR/build_sota"
DATA="$REPO_DIR/datasets"
RESULTS="experiment/results"
LOGS="experiment/logs"
mkdir -p "$BUILD" "$DATA" "$RESULTS" "$LOGS"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  真实SOTA对比 (conda, 无sudo)                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── §1: conda依赖 ────────────────────────────────────────────
echo "═══ §1: 环境准备 ═══"
# 检测conda
if command -v conda &>/dev/null; then
    echo "  conda: $(conda --version)"
    # 安装依赖到当前conda环境
    conda install -y cmake boost tbb -c conda-forge 2>&1 | tail -3
else
    echo "  [WARN] conda未找到, 尝试module load或手动安装"
fi

# 验证工具
CMAKE=$(which cmake 2>/dev/null || echo "")
NVCC=$(which nvcc 2>/dev/null || echo "")
echo "  cmake: ${CMAKE:-MISSING}"
echo "  nvcc: ${NVCC:-MISSING}"
echo "  g++: $(g++ --version | head -1)"
echo ""

# ─── §2: 下载LiveJournal ──────────────────────────────────────
echo "═══ §2: 数据集 ═══"
LJ="$DATA/soc-LiveJournal1.txt"
if [ ! -f "$LJ" ]; then
    echo "  下载LiveJournal..."
    wget -q "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz" -O "$DATA/lj.txt.gz" 2>&1 || {
        echo "  下载失败, 用RMAT代替"
        LJ=""
    }
    [ -f "$DATA/lj.txt.gz" ] && gunzip "$DATA/lj.txt.gz" && mv "$DATA/lj.txt" "$LJ"
fi
[ -f "$LJ" ] && echo "  LiveJournal: $(wc -l < "$LJ") lines" || echo "  用RMAT代替"

# ─── §3: 编译CUDA hetero_bench ────────────────────────────────
echo ""
echo "═══ §3: CUDA实验 (真实GPU tier) ═══"
HETERO_BIN="$BUILD/hetero_bench"

if [ -n "$NVCC" ]; then
    echo "  编译 hetero_bench.cu → H100(HBM) + A6000(GDDR) + DRAM..."
    
    # CUDA 11.5: sm_86(A6000), compute_80 PTX(H100 JIT)
    $NVCC -std=c++17 -O2 \
        -arch=sm_86 \
        -gencode=arch=compute_80,code=compute_80 \
        -Xcompiler "-pthread -fopenmp -O2" \
        -I src \
        -o "$HETERO_BIN" \
        src/cuda/hetero_bench.cu 2>&1 | tail -5
    
    if [ -f "$HETERO_BIN" ]; then
        echo "  ✓ hetero_bench 编译成功"
        echo ""
        echo "  运行GPU tier实验..."
        export CUDA_VISIBLE_DEVICES=0,1,2
        timeout 600 "$HETERO_BIN" > "$LOGS/hetero_${TS}.log" 2>&1 || true
        
        echo "  结果:"
        grep -E "bandwidth|latency|PASS|FAIL|tier|HBM|GDDR|DRAM|throughput" \
            "$LOGS/hetero_${TS}.log" 2>/dev/null | head -20
        echo ""
        # GPU利用率验证
        nvidia-smi --query-compute-apps=pid,name,used_gpu_memory \
            --format=csv,noheader 2>/dev/null | head -5
    else
        echo "  ✗ 编译失败 — 检查CUDA include paths"
        # 尝试简化编译(不含项目头文件)
        echo "  尝试独立编译..."
        $NVCC -std=c++17 -O2 -arch=sm_86 \
            -o "$HETERO_BIN" src/cuda/hetero_bench.cu 2>&1 | tail -10
    fi
else
    echo "  nvcc不可用. 尝试: module load cuda 或 export PATH=/usr/local/cuda/bin:\$PATH"
fi

# ─── §4: 编译RapidStore (CSR + NeoGraph baseline) ─────────────
echo ""
echo "═══ §4: RapidStore Baseline ═══"
RS_BUILD="$BUILD/rapidstore"

if [ -n "$CMAKE" ]; then
    mkdir -p "$RS_BUILD"
    cd "$RS_BUILD"
    
    # 用conda路径
    CONDA_PFX="${CONDA_PREFIX:-/usr}"
    
    echo "  cmake配置..."
    $CMAKE "$REPO_DIR/upstream/rapidstore" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$CONDA_PFX" \
        -DBOOST_ROOT="$CONDA_PFX" \
        -DTBB_DIR="$CONDA_PFX/lib/cmake/TBB" \
        2>&1 | tail -10
    
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "  make (只编译CSR和NeoGraph wrapper)..."
        # 先编译能编译的
        make -j$(nproc) reader graph utils 2>&1 | tail -3 || true
        # CSR不依赖外部SOTA库
        make -j$(nproc) csr_wrapper.out 2>&1 | tail -5 || true
        make -j$(nproc) neo_wrapper.out 2>&1 | tail -5 || true
        
        for bin in csr_wrapper.out neo_wrapper.out; do
            FOUND=$(find "$RS_BUILD" -name "$bin" -executable 2>/dev/null | head -1)
            if [ -n "$FOUND" ]; then
                echo "  ✓ $bin"
            else
                echo "  ✗ $bin 编译失败"
            fi
        done
    else
        echo "  cmake配置失败. 可能缺少boost/tbb."
        echo "  运行: conda install -y cmake boost tbb -c conda-forge"
    fi
    cd "$REPO_DIR"
else
    echo "  cmake不可用. 运行: conda install -y cmake -c conda-forge"
fi

# ─── §5: 编译Philemon CPU实验 ─────────────────────────────────
echo ""
echo "═══ §5: Philemon CPU实验 ═══"
THREADS=$(nproc)
[ "$THREADS" -gt 64 ] && THREADS=64

PHI_BIN="$BUILD/m169_driver"
g++ -std=c++17 -O2 -fopenmp -march=native \
    -o "$PHI_BIN" experiment/m169_m170_driver_workload_engine.cpp -lpthread 2>&1

if [ -f "$PHI_BIN" ]; then
    echo "  ✓ Philemon CPU编译成功"
    
    for SCALE in 14 18 20; do
        echo ""
        echo "  --- scale=$SCALE, threads=$THREADS ---"
        timeout 300 "$PHI_BIN" --scale $SCALE --threads $THREADS --debug 1 \
            > "$LOGS/phi_s${SCALE}_${TS}.log" 2>&1 || true
        
        P=$(grep -c "PASS:" "$LOGS/phi_s${SCALE}_${TS}.log" 2>/dev/null || echo 0)
        F=$(grep -c "FAIL:" "$LOGS/phi_s${SCALE}_${TS}.log" 2>/dev/null || echo 0)
        MEPS=$(grep "Init MEPS" "$LOGS/phi_s${SCALE}_${TS}.log" 2>/dev/null | head -1)
        echo "    $P PASS, $F FAIL  $MEPS"
    done
fi

# ─── §6: 运行RapidStore baseline ──────────────────────────────
echo ""
echo "═══ §6: RapidStore Baseline运行 ═══"
CSR_BIN=$(find "$RS_BUILD" -name "csr_wrapper.out" -executable 2>/dev/null | head -1)
NEO_BIN=$(find "$RS_BUILD" -name "neo_wrapper.out" -executable 2>/dev/null | head -1)

if [ -n "$CSR_BIN" ]; then
    echo "  运行CSR baseline..."
    # 配置: 让它用RMAT而不是LiveJournal(先验证能跑)
    timeout 300 numactl --cpunodebind=0 --membind=0 "$CSR_BIN" \
        > "$LOGS/csr_baseline_${TS}.log" 2>&1 || true
    echo "  $(tail -10 "$LOGS/csr_baseline_${TS}.log" 2>/dev/null | head -5)"
fi

if [ -n "$NEO_BIN" ]; then
    echo "  运行NeoGraph baseline..."
    timeout 300 numactl --cpunodebind=0 --membind=0 "$NEO_BIN" \
        > "$LOGS/neo_baseline_${TS}.log" 2>&1 || true
    echo "  $(tail -10 "$LOGS/neo_baseline_${TS}.log" 2>/dev/null | head -5)"
fi

# ─── §7: 汇总 ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  完成. 日志: $LOGS/*_${TS}.log          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "如果依赖缺失:"
echo "  conda install -y cmake boost tbb -c conda-forge"
echo "  export PATH=/usr/local/cuda/bin:\$PATH  # 如果nvcc找不到"
echo ""
echo "下载SOTA库源码(可选,编译真实baseline):"
echo "  cd $BUILD"
echo "  git clone https://github.com/PerFuchs/sortledton.git"
echo "  git clone https://github.com/cwida/teseo.git"
echo "  git clone https://github.com/nicknash/LiveGraph.git"

# push
git add experiment/results/ experiment/logs/ 2>/dev/null || true
git commit -m "ags1 real run ($TS)" --author="dylanyunlon <dogechat@163.com>" 2>/dev/null || true
git push origin main 2>/dev/null || true
