#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_ags1_gpu_bench.sh — 在ags1上运行全部GPU实验, 结果写入git
#
# 使用: 
#   cd /data/jiacheng/system/cache/temp/nips2026
#   git clone https://github.com/dylanyunlon/philemon-TSH.git  (或 git pull)
#   cd philemon-TSH
#   bash experiment/run_ags1_gpu_bench.sh
#
# 结果会自动commit+push到main分支
# ═══════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

LOG_DIR="experiment/ags1_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════"
echo " Philemon-TSH GPU Benchmark Suite — ags1"
echo " $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo " CUDA: $(nvcc --version 2>/dev/null | tail -1)"
echo "═══════════════════════════════════════════════════════"

# ─── Step 0: Hardware snapshot ───────────────────────────────────
echo ""
echo ">>> Step 0: Hardware snapshot"
{
    echo "=== lscpu ==="
    lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
    echo ""
    echo "=== Memory ==="
    free -h
    echo ""
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv
    echo ""
    echo "=== Topology ==="
    nvidia-smi topo -m
    echo ""
    echo "=== NUMA ==="
    numactl --hardware 2>/dev/null | head -15
    echo ""
    echo "=== Driver & CUDA ==="
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
    nvcc --version 2>/dev/null | tail -1
} > "$LOG_DIR/hardware_${TIMESTAMP}.txt" 2>&1
echo "  Saved: $LOG_DIR/hardware_${TIMESTAMP}.txt"

# ─── Step 1: Build GPU benchmarks ───────────────────────────────
echo ""
echo ">>> Step 1: Build GPU benchmarks"

# hetero_bench: core tier allocation + bandwidth + migration
echo "  Building hetero_bench..."
nvcc -std=c++17 -O2 \
    -arch=sm_86 -gencode=arch=compute_80,code=compute_80 \
    -gencode=arch=compute_90,code=sm_90 \
    -Xcompiler "-pthread -fopenmp -Wall" -lineinfo \
    -o hetero_bench src/cuda/hetero_bench.cu 2>&1 | tee "$LOG_DIR/build_hetero_${TIMESTAMP}.txt"

# walking_hetero_bench: migration matrix + hotness placement + concurrent migrate
echo "  Building walking_hetero_bench..."
nvcc -std=c++17 -O2 \
    -arch=sm_86 -gencode=arch=compute_80,code=compute_80 \
    -gencode=arch=compute_90,code=sm_90 \
    -Xcompiler "-pthread -fopenmp -Wall" -lineinfo -DWALKING_CUDA=1 \
    -o walking_hetero_bench src/cuda/walking_hetero_bench.cu 2>&1 | tee "$LOG_DIR/build_walking_${TIMESTAMP}.txt"

# walking_integration: full system integration on GPU
echo "  Building walking_integration..."
nvcc -std=c++17 -O2 \
    -arch=sm_86 -gencode=arch=compute_80,code=compute_80 \
    -gencode=arch=compute_90,code=sm_90 \
    -Xcompiler "-pthread -fopenmp -Wall" -lineinfo -DWALKING_CUDA=1 \
    -o walking_integration src/cuda/walking_integration.cu 2>&1 | tee "$LOG_DIR/build_integration_${TIMESTAMP}.txt"

echo "  Build complete."

# ─── Step 2: CPU-only experiments (M133-M140) ───────────────────
echo ""
echo ">>> Step 2: CPU experiments (validation + paper data)"

for exp in m133_m134_sota_benchmark m135_m136_scaling_experiment m137_m138_algorithm_walking m139_m140_migration_latency; do
    SRC="experiment/${exp}.cpp"
    if [ -f "$SRC" ]; then
        echo "  Compiling $exp..."
        g++ -std=c++17 -O2 -pthread -o "${exp}_bench" "$SRC" 2>&1
        echo "  Running $exp (--latex)..."
        ./"${exp}_bench" --latex > "$LOG_DIR/${exp}_${TIMESTAMP}.txt" 2>&1
        echo "  Done: $LOG_DIR/${exp}_${TIMESTAMP}.txt"
        rm -f "${exp}_bench"
    fi
done

# ─── Step 3: GPU experiments ─────────────────────────────────────
echo ""
echo ">>> Step 3: GPU experiments on ags1"

# E1-E6: hetero_bench (tier allocation, bandwidth, cross-tier query, migration, scaling)
echo "  Running hetero_bench..."
CUDA_VISIBLE_DEVICES=0,1,2 ./hetero_bench 2>&1 | tee "$LOG_DIR/hetero_bench_${TIMESTAMP}.txt"

# Migration matrix + hotness placement + concurrent migration
echo "  Running walking_hetero_bench..."
CUDA_VISIBLE_DEVICES=0,1,2 ./walking_hetero_bench 2>&1 | tee "$LOG_DIR/walking_hetero_${TIMESTAMP}.txt"

# Full integration (if it needs all GPUs)
echo "  Running walking_integration..."
CUDA_VISIBLE_DEVICES=0,1,2 ./walking_integration 2>&1 | tee "$LOG_DIR/walking_integration_${TIMESTAMP}.txt"

# ─── Step 4: Collect results and push ────────────────────────────
echo ""
echo ">>> Step 4: Commit & push results"

# Cleanup binaries
rm -f hetero_bench walking_hetero_bench walking_integration

git add "$LOG_DIR/"
git commit -m "ags1 GPU benchmark results ${TIMESTAMP}

Hardware: 2×EPYC 9354 + 2×A6000 + H100 NVL, 1.5TB DDR5
Experiments: hetero_bench + walking_hetero + walking_integration + M133-M140 CPU" \
  --author="dylanyunlon <dogechat@163.com>"

git push origin main

echo ""
echo "═══════════════════════════════════════════════════════"
echo " All experiments complete. Results pushed to main."
echo " Log dir: $LOG_DIR/"
echo "═══════════════════════════════════════════════════════"
