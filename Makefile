# Makefile — Philemon-TSH: Temporal Subgraph on Heterogeneous Memory
#
# Targets:
#   make cpu          — CPU-only benchmark (dev VM, no GPU needed)
#   make cuda         — CUDA heterogeneous benchmark (ags1: A6000×2 + H100)
#   make all          — both
#   make clean

CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -pthread -Wall
NVFLAGS  := -std=c++17 -O2 -Xcompiler "-pthread -fopenmp -Wall" -lineinfo

# Detect CUDA version for arch flags
CUDA_VER := $(shell nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR := $(shell echo $(CUDA_VER) | cut -d. -f1)
CUDA_MINOR := $(shell echo $(CUDA_VER) | cut -d. -f2)

# sm_86 for A6000, compute_80 PTX for H100 JIT (CUDA 11.5 compatible)
ARCH_FLAGS := -arch=sm_86 -gencode=arch=compute_80,code=compute_80

.PHONY: all cpu cuda clean

all: cpu cuda

cpu: philemon_bench

cuda: hetero_bench

philemon_bench: src/bench/philemon_bench.cpp src/core/tiered_allocator.hpp \
                src/bridge/temporal_bridge.hpp src/scheduler/migration_scheduler.hpp
	$(CXX) $(CXXFLAGS) -I src -o $@ src/bench/philemon_bench.cpp

hetero_bench: src/cuda/hetero_bench.cu
	$(NVCC) $(NVFLAGS) $(ARCH_FLAGS) -o $@ $<

clean:
	rm -f philemon_bench hetero_bench
