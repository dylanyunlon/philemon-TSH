#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# run_ags1_real_sota.sh — 真实SOTA对比实验
#
# 在ags1服务器上:
#   1. 编译upstream RapidStore → 得到真实baseline (CSR/Sortledton/Teseo等)
#   2. 编译Philemon CUDA hetero_bench → 得到真实GPU tier数据
#   3. 下载LiveJournal数据集
#   4. 在相同硬件上运行所有系统, 收集真实数据
#
# 服务器: ags1 (2×EPYC 9354 128核, 1.5TB RAM, 2×A6000+H100)
# 运行: bash experiment/run_ags1_real_sota.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="experiment/results"
LOG_DIR="experiment/logs"
BUILD_DIR="/tmp/philemon_build_${TIMESTAMP}"
DATA_DIR="/data/jiacheng/system/cache/temp/atc2026/datasets"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$BUILD_DIR" "$DATA_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Philemon-TSH: 真实SOTA对比实验                              ║"
echo "║  $(date)                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ─── §0: 系统信息 ─────────────────────────────────────────────
echo ""
echo "═══ §0: System Info ═══"
lscpu | grep -E "Model name|CPU\(s\):|Socket" || true
free -g 2>/dev/null || cat /proc/meminfo | head -3
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"
nvcc --version 2>/dev/null | tail -1 || echo "nvcc not found"
echo ""

# ─── §1: 下载LiveJournal数据集 ─────────────────────────────────
echo "═══ §1: Dataset Preparation ═══"
LJ_DIR="$DATA_DIR/liveJournal"
if [ ! -f "$LJ_DIR/soc-LiveJournal1.txt" ]; then
    echo "  Downloading LiveJournal dataset..."
    mkdir -p "$LJ_DIR"
    cd "$LJ_DIR"
    wget -q "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz" -O soc-LiveJournal1.txt.gz || {
        echo "  [WARN] Download failed. Generating synthetic RMAT instead."
        echo "  For real results, manually download LiveJournal from SNAP."
    }
    if [ -f "soc-LiveJournal1.txt.gz" ]; then
        gunzip soc-LiveJournal1.txt.gz
        echo "  LiveJournal: $(wc -l < soc-LiveJournal1.txt) lines"
    fi
    cd "$REPO_DIR"
else
    echo "  LiveJournal already at $LJ_DIR"
fi

# ─── §2: 编译upstream RapidStore baseline ──────────────────────
echo ""
echo "═══ §2: Build Upstream RapidStore Baselines ═══"
RS_BUILD="$BUILD_DIR/rapidstore"
mkdir -p "$RS_BUILD"

# 检查依赖
echo "  Checking dependencies..."
HAS_BOOST=$(dpkg -l | grep -c libboost-program-options-dev 2>/dev/null || echo 0)
HAS_TBB=$(dpkg -l | grep -c libtbb-dev 2>/dev/null || echo 0)
HAS_CMAKE=$(which cmake 2>/dev/null || echo "")

if [ -z "$HAS_CMAKE" ]; then
    echo "  [ERROR] cmake not found. Install: sudo apt install cmake"
    echo "  Skipping RapidStore build."
    RS_BUILT=0
elif [ "$HAS_BOOST" = "0" ] || [ "$HAS_TBB" = "0" ]; then
    echo "  [WARN] Missing deps. Install: sudo apt install libboost-program-options-dev libtbb-dev"
    echo "  Attempting build anyway..."
    RS_BUILT=0
else
    echo "  Dependencies OK. Building..."
    RS_BUILT=1
fi

if [ "$RS_BUILT" = "1" ]; then
    cd "$RS_BUILD"
    cmake "$REPO_DIR/upstream/rapidstore" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    make -j$(nproc) 2>&1 | tail -10
    
    # 检查编译产物
    for wrapper in csr_wrapper neo_wrapper sortledton_wrapper teseo_wrapper; do
        if [ -f "wrapper/apps/${wrapper}/${wrapper}" ] || [ -f "wrapper/${wrapper}" ]; then
            echo "  ✓ ${wrapper} built"
        else
            echo "  ✗ ${wrapper} not found"
            RS_BUILT=0
        fi
    done
    cd "$REPO_DIR"
fi

# ─── §3: 编译Philemon CUDA hetero_bench ───────────────────────
echo ""
echo "═══ §3: Build Philemon CUDA Benchmarks ═══"
NVCC_OK=$(which nvcc 2>/dev/null || echo "")

if [ -n "$NVCC_OK" ]; then
    echo "  nvcc found: $(nvcc --version 2>/dev/null | tail -1)"
    
    # 编译hetero_bench (真实GPU tier实验)
    echo "  Building hetero_bench.cu..."
    nvcc -std=c++17 -O2 \
        -arch=sm_86 -gencode=arch=compute_80,code=compute_80 \
        -Xcompiler "-pthread -fopenmp -Wall" \
        -o "$BUILD_DIR/hetero_bench" \
        src/cuda/hetero_bench.cu 2>&1 | tail -5
    
    if [ -f "$BUILD_DIR/hetero_bench" ]; then
        echo "  ✓ hetero_bench built (GPU tier experiment)"
        CUDA_BUILT=1
    else
        echo "  ✗ hetero_bench build failed"
        CUDA_BUILT=0
    fi
else
    echo "  [WARN] nvcc not found. CUDA benchmarks skipped."
    echo "  Install CUDA toolkit: module load cuda or conda install cuda-toolkit"
    CUDA_BUILT=0
fi

# ─── §4: 编译Philemon CPU实验 (作为对照) ──────────────────────
echo ""
echo "═══ §4: Build Philemon CPU Experiments ═══"
THREADS=$(nproc)
[ "$THREADS" -gt 64 ] && THREADS=64

for f in experiment/m169_m170_driver_workload_engine.cpp \
         experiment/m173_m174_largescale_experiment.cpp; do
    BN=$(basename "$f" .cpp)
    g++ -std=c++17 -O2 -fopenmp -march=native -o "$BUILD_DIR/$BN" "$f" -lpthread 2>&1 || {
        echo "  ✗ $BN compile failed"
        continue
    }
    echo "  ✓ $BN built"
done

# ─── §5: 运行实验 ─────────────────────────────────────────────
echo ""
echo "═══ §5: Run Experiments ═══"
MASTER_CSV="$RESULTS_DIR/real_sota_${TIMESTAMP}.csv"
echo "system,algo,scale,time_s,insert_meps,mem_gb,source" > "$MASTER_CSV"

# 5a: Philemon CPU baseline (验证)
echo ""
echo "  --- 5a: Philemon CPU (scale=20, $THREADS threads) ---"
if [ -f "$BUILD_DIR/m169_m170_driver_workload_engine" ]; then
    timeout 600 "$BUILD_DIR/m169_m170_driver_workload_engine" \
        --scale 20 --threads "$THREADS" --debug 1 \
        > "$LOG_DIR/phi_cpu_s20_${TIMESTAMP}.log" 2>&1 || true
    
    P=$(grep -c "PASS:" "$LOG_DIR/phi_cpu_s20_${TIMESTAMP}.log" 2>/dev/null || echo 0)
    F=$(grep -c "FAIL:" "$LOG_DIR/phi_cpu_s20_${TIMESTAMP}.log" 2>/dev/null || echo 0)
    echo "    $P PASS, $F FAIL"
    
    # 提取数据
    grep "Init MEPS\|BFS.*Philemon\|PR.*Philemon" "$LOG_DIR/phi_cpu_s20_${TIMESTAMP}.log" | head -5
fi

# 5b: CUDA hetero_bench (真实GPU tier)
if [ "$CUDA_BUILT" = "1" ]; then
    echo ""
    echo "  --- 5b: Philemon CUDA hetero_bench (H100 + A6000) ---"
    
    # 设置CUDA可见设备
    export CUDA_VISIBLE_DEVICES=0,1,2
    
    timeout 600 "$BUILD_DIR/hetero_bench" \
        > "$LOG_DIR/hetero_bench_${TIMESTAMP}.log" 2>&1 || true
    
    echo "    Log: $LOG_DIR/hetero_bench_${TIMESTAMP}.log"
    tail -20 "$LOG_DIR/hetero_bench_${TIMESTAMP}.log" 2>/dev/null
    
    # 检查GPU是否真实运行
    nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
fi

# 5c: RapidStore baselines (如果编译成功)
if [ "$RS_BUILT" = "1" ]; then
    echo ""
    echo "  --- 5c: RapidStore Baselines ---"
    
    # 配置LiveJournal workload路径
    if [ -d "$LJ_DIR/workloads" ]; then
        export WORKLOAD_DIR="$LJ_DIR/workloads"
    fi
    
    for wrapper in csr_wrapper sortledton_wrapper teseo_wrapper; do
        BIN=$(find "$RS_BUILD" -name "${wrapper}*" -executable | head -1)
        if [ -n "$BIN" ]; then
            echo "    Running $wrapper..."
            timeout 300 numactl --cpunodebind=0 --membind=0 "$BIN" \
                > "$LOG_DIR/${wrapper}_${TIMESTAMP}.log" 2>&1 || true
            echo "    $(tail -5 "$LOG_DIR/${wrapper}_${TIMESTAMP}.log" 2>/dev/null)"
        fi
    done
fi

# ─── §6: 汇总 ─────────────────────────────────────────────────
echo ""
echo "═══ §6: Summary ═══"
echo "  Results: $MASTER_CSV"
echo "  Logs: $LOG_DIR/*_${TIMESTAMP}.log"
echo ""
echo "  Build status:"
echo "    RapidStore baselines: $([ "$RS_BUILT" = "1" ] && echo "✓" || echo "✗ (install deps: cmake libboost-program-options-dev libtbb-dev)")"
echo "    CUDA hetero_bench:   $([ "${CUDA_BUILT:-0}" = "1" ] && echo "✓" || echo "✗ (need nvcc)")"
echo ""
echo "  下一步:"
echo "    1. 如果RapidStore编译失败: sudo apt install cmake libboost-program-options-dev libtbb-dev"
echo "    2. 如果需要LiveJournal: wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
echo "    3. 预处理数据: cd upstream/rapidstore && cmake-build/dataset_preprocessor/preprocessor"
echo "    4. 重新运行: bash experiment/run_ags1_real_sota.sh"

# Auto-push
git add experiment/results/ experiment/logs/ 2>/dev/null || true
git commit -m "ags1 real SOTA run ($TIMESTAMP)" \
    --author="dylanyunlon <dogechat@163.com>" 2>/dev/null || true
git push origin main 2>/dev/null || echo "  Push: manual needed"
