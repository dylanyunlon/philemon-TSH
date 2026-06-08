#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_ags1_gpu_bench.sh — ags1 GPU实验 (修复: CUDA 11.5兼容)
#
# CUDA 11.5 supports: sm_86 (A6000) + compute_80 PTX (JIT to H100)
# No compute_90 (requires CUDA 12.0+)
#
# 使用:
#   cd philemon-TSH && bash experiment/run_ags1_gpu_bench.sh
# ═══════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

LOG_DIR="experiment/ags1_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════"
echo " Philemon-TSH GPU Benchmark — ags1"
echo " $(date)"
echo " CUDA: $(nvcc --version 2>/dev/null | grep release)"
echo " GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "═══════════════════════════════════════════════════════"

# ─── Hardware snapshot ───────────────────────────────────────────
{
    lscpu | grep -E "Model name|Socket|Core|Thread|NUMA|CPU\(s\):|Architecture"
    echo ""; free -h
    echo ""; nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current --format=csv
    echo ""; nvidia-smi topo -m 2>/dev/null
    echo ""; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
    nvcc --version 2>/dev/null | tail -1
} > "$LOG_DIR/hardware_${TS}.txt" 2>&1
echo "  Hardware → $LOG_DIR/hardware_${TS}.txt"

# ─── NVCC flags: sm_86 (A6000) + compute_80 PTX (H100 JIT) ─────
# CUDA 11.5: NO compute_90/sm_90 support
NVFLAGS="-std=c++17 -O2 -arch=sm_86 -gencode=arch=compute_80,code=compute_80 -Xcompiler -pthread,-fopenmp -lineinfo"

# ─── Build ───────────────────────────────────────────────────────
echo ""
echo ">>> Building GPU benchmarks (CUDA 11.5, sm_86 + compute_80 PTX)..."

echo "  [1/3] hetero_bench..."
nvcc $NVFLAGS -o hetero_bench src/cuda/hetero_bench.cu 2>&1 | tee "$LOG_DIR/build_hetero_${TS}.txt"
echo "  OK"

echo "  [2/3] walking_hetero_bench..."
nvcc $NVFLAGS -DWALKING_CUDA=1 -o walking_hetero_bench src/cuda/walking_hetero_bench.cu 2>&1 | tee "$LOG_DIR/build_walking_${TS}.txt"
echo "  OK"

echo "  [3/3] walking_integration..."
nvcc $NVFLAGS -DWALKING_CUDA=1 -o walking_integration src/cuda/walking_integration.cu 2>&1 | tee "$LOG_DIR/build_integration_${TS}.txt"
echo "  OK"

# ─── CPU experiments (M133-M140) ─────────────────────────────────
echo ""
echo ">>> CPU experiments..."
for exp in m133_m134_sota_benchmark m135_m136_scaling_experiment m137_m138_algorithm_walking m139_m140_migration_latency; do
    SRC="experiment/${exp}.cpp"
    if [ -f "$SRC" ]; then
        echo "  $exp..."
        g++ -std=c++17 -O2 -pthread -march=native -o "/tmp/${exp}" "$SRC" 2>&1
        "/tmp/${exp}" --latex > "$LOG_DIR/${exp}_${TS}.txt" 2>&1
        echo "    Done ($(wc -l < "$LOG_DIR/${exp}_${TS}.txt") lines)"
        rm -f "/tmp/${exp}"
    fi
done

# ─── GPU experiments ─────────────────────────────────────────────
echo ""
echo ">>> GPU experiments..."

# Pin to NUMA1 where all GPUs are
NUMA_PREFIX="numactl --cpunodebind=1 --membind=1"

echo "  [1/3] hetero_bench (E1-E6: tier alloc + bandwidth + migration + scaling)..."
CUDA_VISIBLE_DEVICES=0,1,2 $NUMA_PREFIX ./hetero_bench 2>&1 | tee "$LOG_DIR/hetero_bench_${TS}.txt"
echo "  Done"

echo "  [2/3] walking_hetero_bench (migration matrix + hotness + concurrent)..."
CUDA_VISIBLE_DEVICES=0,1,2 $NUMA_PREFIX ./walking_hetero_bench 2>&1 | tee "$LOG_DIR/walking_hetero_${TS}.txt"
echo "  Done"

echo "  [3/3] walking_integration (full system)..."
CUDA_VISIBLE_DEVICES=0,1,2 $NUMA_PREFIX ./walking_integration 2>&1 | tee "$LOG_DIR/walking_integration_${TS}.txt"
echo "  Done"

# ─── Cleanup & push ─────────────────────────────────────────────
rm -f hetero_bench walking_hetero_bench walking_integration

echo ""
echo ">>> Committing results..."
git add "$LOG_DIR/"
git commit -m "ags1 GPU+CPU benchmark ${TS}

CUDA 11.5 / sm_86+compute_80 PTX / H100 NVL + 2×A6000
CPU: 2×EPYC 9354 (128 threads) / 1.5TB DDR5
Experiments: hetero_bench + walking_hetero + walking_integration + M133-M140" \
  --author="dylanyunlon <dogechat@163.com>"

git push origin main

echo ""
echo "═══════════════════════════════════════════════════════"
echo " All done. Results in $LOG_DIR/"
echo "═══════════════════════════════════════════════════════"
