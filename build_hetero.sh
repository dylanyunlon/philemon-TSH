#!/bin/bash
# build_hetero.sh — Build Philemon-TSH heterogeneous GPU benchmark
#
# Target: ags1 (A6000×2 + H100 NVL, CUDA 11.5, Driver 550)
#
# Usage:
#   cd /data/jiacheng/system/cache/temp/nips2026/0510
#   bash build_hetero.sh
#
# Notes:
#   - sm_86 = A6000 (Ampere GA102)
#   - sm_90 = H100  (Hopper GH100)
#   - CUDA 11.5 supports sm_86; sm_90 requires CUDA 11.8+
#     If sm_90 fails, we fall back to compute_90 PTX JIT
#   - Uses -Xcompiler -pthread for std::thread / std::shared_mutex

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${SCRIPT_DIR}/src/cuda/hetero_bench.cu"
OUT="${SCRIPT_DIR}/hetero_bench"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║  Building Philemon-TSH Heterogeneous Benchmark       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Detect CUDA toolkit version
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "CUDA toolkit: ${CUDA_VER}"

# Determine arch flags based on CUDA version
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

ARCH_FLAGS="-arch=sm_86"

# sm_90 (H100) support: CUDA 11.8+ for native, 11.0+ for PTX JIT
if [ "$CUDA_MAJOR" -gt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    ARCH_FLAGS="${ARCH_FLAGS} -gencode=arch=compute_90,code=sm_90"
    echo "H100 sm_90: native support"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 5 ]; then
    # CUDA 11.5-11.7: use PTX JIT for H100
    # The driver (550.x) supports sm_90 JIT from compute_80 PTX
    ARCH_FLAGS="${ARCH_FLAGS} -gencode=arch=compute_80,code=compute_80"
    echo "H100 sm_90: PTX JIT via compute_80 (driver will JIT to sm_90)"
else
    echo "WARNING: CUDA ${CUDA_VER} may not support H100. Trying compute_80 PTX."
    ARCH_FLAGS="${ARCH_FLAGS} -gencode=arch=compute_80,code=compute_80"
fi

echo "Arch flags: ${ARCH_FLAGS}"
echo ""

# Build
echo "Compiling: nvcc ${ARCH_FLAGS} ..."
nvcc -std=c++17 -O2 \
     ${ARCH_FLAGS} \
     -Xcompiler "-pthread -fopenmp -Wall" \
     -lineinfo \
     -o "${OUT}" \
     "${SRC}"

echo ""
echo "Build successful: ${OUT}"
echo "Size: $(du -h "${OUT}" | cut -f1)"
echo ""
echo "Run with:"
echo "  numactl --cpunodebind=1 --membind=1 ${OUT}"
echo ""
echo "(NUMA1 binding recommended since all GPUs are on NUMA node 1)"
