# Makefile — Philemon-TSH: Temporal Subgraph on Heterogeneous Memory
#
# Targets:
#   make cpu              — CPU-only benchmark (dev VM, no GPU needed)
#   make integration      — Integration bench (TEM-Graph + RapidStore + algorithms)
#   make cuda             — CUDA heterogeneous benchmark (ags1: A6000×2 + H100)
#   make ldbc             — LDBC SNB loader + cost model bench (M017-M020)
#   make all              — cpu + integration + cuda
#   make test             — Quick integration test (small graph)
#   make ldbc_test        — Quick LDBC bench with synthetic data
#   make clean

CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O2 -pthread -Wall -Wextra -Wno-unused-parameter
NVFLAGS  := -std=c++17 -O2 -Xcompiler "-pthread -fopenmp -Wall" -lineinfo

# Include paths — all source directories
INCLUDES := -I src -I src/core -I src/bridge -I src/index -I src/wrapper \
            -I src/algorithms -I src/executor -I src/debug \
            -I src/loader -I src/cost_model

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
                src/algorithms/tiered_tc.hpp \
                src/algorithms/cross_tier_bfs.hpp src/algorithms/cross_tier_sssp.hpp

EXEC_HEADERS := src/executor/thread_pool_base.hpp src/executor/spin_lock.hpp \
                src/executor/query_executor.hpp

DEBUG_HEADERS := src/debug/philemon_debug.hpp

LOADER_HEADERS := src/loader/ldbc_types.hpp src/loader/ldbc_loader.hpp \
                  src/loader/ldbc_driver.hpp

COST_HEADERS := src/cost_model/tier_cost_model.hpp

ALL_HEADERS := $(CORE_HEADERS) $(INDEX_HEADERS) $(WRAPPER_HEADERS) \
               $(ALGO_HEADERS) $(EXEC_HEADERS) $(DEBUG_HEADERS) \
               $(LOADER_HEADERS) $(COST_HEADERS) \
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

# LDBC SNB loader + cost model bench (M017-M020)
ldbc_bench: src/bench/ldbc_bench.cpp $(ALL_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

ldbc: ldbc_bench

# Quick LDBC bench with synthetic data
ldbc_test: ldbc_bench
	./ldbc_bench "" 4 2.0 2

clean:
	rm -f philemon_bench integration_bench hetero_bench ldbc_bench
