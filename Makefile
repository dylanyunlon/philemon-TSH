# Makefile — Philemon-TSH: Temporal Subgraph on Heterogeneous Memory
#
# Targets:
#   make cpu              — CPU-only benchmark (dev VM, no GPU needed)
#   make integration      — Integration bench (TEM-Graph + RapidStore + algorithms)
#   make cuda             — CUDA heterogeneous benchmark (ags1: A6000×2 + H100)
#   make all              — cpu + integration + cuda
#   make test             — Quick integration test (small graph)
#   make clean

CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -pthread -Wall -Wextra -Wno-unused-parameter
NVFLAGS  := -std=c++17 -O2 -Xcompiler "-pthread -fopenmp -Wall" -lineinfo

# Include paths — all source directories
INCLUDES := -I src -I src/core -I src/bridge -I src/index -I src/wrapper \
            -I src/algorithms -I src/executor -I src/debug

# Detect CUDA version for arch flags
CUDA_VER := $(shell nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//')
CUDA_MAJOR := $(shell echo $(CUDA_VER) | cut -d. -f1)

# sm_86 for A6000, compute_80 PTX for H100 JIT (CUDA 11.5 compatible)
ARCH_FLAGS := -arch=sm_86 -gencode=arch=compute_80,code=compute_80

# Source files
CORE_HEADERS := src/core/tiered_allocator.hpp src/core/seqlock.hpp \
                src/core/slab_allocator.hpp src/core/tier_ptr.hpp \
                src/core/async_migrator.hpp src/core/partition_index.hpp \
                src/core/temporal_edge.hpp

INDEX_HEADERS := src/index/interval.hpp src/index/dll_list.hpp \
                 src/index/tem_graph.hpp src/index/tem_graph_impl.hpp

WRAPPER_HEADERS := src/wrapper/rapidstore_wrapper.hpp \
                   src/wrapper/graph_edge.hpp src/wrapper/edge_stream.hpp

ALGO_HEADERS := src/algorithms/tiered_bfs.hpp src/algorithms/tiered_pagerank.hpp \
                src/algorithms/tiered_sssp.hpp src/algorithms/tiered_wcc.hpp \
                src/algorithms/tiered_tc.hpp

EXEC_HEADERS := src/executor/thread_pool_base.hpp src/executor/spin_lock.hpp \
                src/executor/query_executor.hpp

DEBUG_HEADERS := src/debug/philemon_debug.hpp

ALL_HEADERS := $(CORE_HEADERS) $(INDEX_HEADERS) $(WRAPPER_HEADERS) \
               $(ALGO_HEADERS) $(EXEC_HEADERS) $(DEBUG_HEADERS) \
               src/bridge/temporal_bridge.hpp src/scheduler/migration_scheduler.hpp

.PHONY: all cpu integration cuda test clean

all: cpu integration

cpu: philemon_bench

integration: integration_bench

cuda: hetero_bench

# CPU-only benchmark (Phase 1–3 core system)
philemon_bench: src/bench/philemon_bench.cpp $(CORE_HEADERS) \
                src/bridge/temporal_bridge.hpp src/scheduler/migration_scheduler.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Integration benchmark (M011–M016: index + wrapper + algorithms + executor)
integration_bench: src/bench/integration_bench.cpp $(ALL_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# CUDA heterogeneous benchmark
hetero_bench: src/cuda/hetero_bench.cu
	$(NVCC) $(NVFLAGS) $(ARCH_FLAGS) -o $@ $<

# Quick test with small graph
test: integration_bench
	./integration_bench 1000 5000 100 2 1

clean:
	rm -f philemon_bench integration_bench hetero_bench
