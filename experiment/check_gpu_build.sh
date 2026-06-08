#!/bin/bash
# Quick GPU build check for ags1 (CUDA 11.5 compatible)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "CUDA version:"
nvcc --version | grep release

echo ""
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

NVFLAGS="-std=c++17 -O2 -arch=sm_86 -gencode=arch=compute_80,code=compute_80 -Xcompiler -pthread"

echo ""
echo "Compiling hetero_bench.cu..."
nvcc $NVFLAGS -o /tmp/test_hetero src/cuda/hetero_bench.cu 2>&1
echo "  OK"

echo ""
echo "Compiling walking_hetero_bench.cu..."
nvcc $NVFLAGS -DWALKING_CUDA=1 -o /tmp/test_walking src/cuda/walking_hetero_bench.cu 2>&1
echo "  OK"

echo ""
echo "Compiling walking_integration.cu..."
nvcc $NVFLAGS -DWALKING_CUDA=1 -o /tmp/test_integ src/cuda/walking_integration.cu 2>&1
echo "  OK"

echo ""
echo "All builds passed. Ready for: bash experiment/run_ags1_gpu_bench.sh"
rm -f /tmp/test_hetero /tmp/test_walking /tmp/test_integ
