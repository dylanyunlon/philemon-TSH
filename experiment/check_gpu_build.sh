#!/bin/bash
# Quick build check for ags1 — run this first to verify compilation
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Checking nvcc..."
nvcc --version

echo ""
echo "Checking GPU visibility..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "Test compile: hetero_bench.cu"
nvcc -std=c++17 -O2 \
    -arch=sm_86 -gencode=arch=compute_80,code=compute_80 \
    -Xcompiler "-pthread" \
    -o /tmp/hetero_bench_test src/cuda/hetero_bench.cu 2>&1
echo "  OK"

echo ""
echo "Test compile: walking_hetero_bench.cu"
nvcc -std=c++17 -O2 \
    -arch=sm_86 -gencode=arch=compute_80,code=compute_80 \
    -Xcompiler "-pthread" -DWALKING_CUDA=1 \
    -o /tmp/walking_hetero_test src/cuda/walking_hetero_bench.cu 2>&1
echo "  OK"

echo ""
echo "All GPU builds OK. Ready to run full benchmark."
rm -f /tmp/hetero_bench_test /tmp/walking_hetero_test
