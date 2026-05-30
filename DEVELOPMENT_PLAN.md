# Philemon-TSH: 38-Claude Development Plan

## Project Goal

Produce publication-quality data (2000+ pts/curve, 3+ seeds, 5+ methods) benchmarking temporal subgraph processing on heterogeneous memory (HBM/GDDR/DRAM), integrating patterns from 20 big-tech reference repositories.

## Data Demo Target (commit 294c91b)

```
reversed_figure_data.json:      3000 steps × 3 seeds × 5 methods (perplexity vs steps)
gradient_norm_24k_data.json:    2000 steps × 3 seeds × 4 methods (gradient norm vs steps)
ppl_vs_time_1B_30k_data.json:   2000 steps × 3 seeds × 5 methods (perplexity vs time)
reversed_figure18_data.json:    panels with parameter/momentum norms
```

Our generated data (matching X-axis scale):
```
philemon_query_latency_2000.json:   2000 steps × 3 seeds × 3 methods × 2 query types
philemon_qps_2000.json:             2000 steps × 3 seeds × 3 methods
philemon_memory_util_2000.json:     2000 steps × 3 seeds × 3 tiers
philemon_migration_cost_2000.json:  2000 steps × 3 seeds
```

---

## Pattern Lineage: "From C, the Good Example"

### C (the good example) = RapidStore `wrapper::snapshot_edges`
```cpp
// upstream/rapidstore/wrapper/wrapper.h:240
template<class S, class F>
void snapshot_edges(S &s, uint64_t index, F&& callback, bool logical) {
    s->edges(index, callback, logical);
}
```
Template-dispatched callback traversal: the visitor pattern for graph edges. Every tier-aware query in Philemon-TSH follows this callback dispatch.

### D = TieredAllocator, letting E = TemporalBridge allocate F = partition memory across tiers, and G = query with tier-aware scanning
Pattern source — NCCL `ncclMemAlloc` (nccl/src/allocator.cc:14):
```cpp
ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
    // ...dispatches to cudaMalloc, cuMemCreate, or cuMemMap
    // based on CUDA version and handle types
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // ...tiered fallback: VMM → pool → fallback malloc
}
```
Our TieredAllocator waterfall (HBM→GDDR→DRAM) mirrors NCCL's allocation dispatch.

### H = SeqLock introduces I = wait-free reads, so J = query readers can K = scan without blocking, while L = adaptive density optimizes M = partition granularity
Pattern source — abseil `Mutex::lock_shared` (abseil-cpp/absl/synchronization/mutex.h:269):
```cpp
void lock_shared() ABSL_SHARED_LOCK_FUNCTION();
void ReaderLock() ABSL_SHARED_LOCK_FUNCTION() { lock_shared(); }
void WriterLock() ABSL_EXCLUSIVE_LOCK_FUNCTION() { lock(); }
```
We replace abseil's reader-writer lock with a seqlock: readers never block, they optimistically read and retry on collision.

### N = SlabAllocator integrates O = per-tier size-class pools, P = allocate supports Q = O(1) bitmask slots, R = compact enhances S = memory stability
Pattern source — NCCL page-based pooling (nccl/src/allocator.cc:370-400):
```cpp
page->freeMask = uint64_t(-1)>>(64 - pageSize/pageObjSize);
int slot = popFirstOneBit(&page->freeMask);
devObj = (char*)page->devObjs + slot*pageObjSize;
if (page->freeMask == 0) *pagePtr = page->next; // Remove full page
```
Our SlabPage uses identical 64-bit bitmask with `__builtin_ctzll`.

Pattern source — PyTorch CachingAllocator (pytorch/c10/cuda/CUDACachingAllocator.cpp:201,3583):
```cpp
struct Block {
    size_t size; void* ptr; Block* prev; Block* next;
    bool allocated; BlockPool* pool;
};
size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    dst->size += subsumed_size; delete src;
    return subsumed_size;
}
```

Pattern source — TensorFlow Arena (tensorflow/core/lib/core/arena.h:67):
```cpp
void* GetMemory(const size_t size, const int align) {
    if (size > 0 && size < remaining_ && align == 1) {  // fast path
        void* result = freestart_;
        freestart_ += size; remaining_ -= size;
        return result;
    }
    return GetMemoryFallback(size, align);  // slow path: new block
}
```

### T = SeqLock+slab completes U = fragmentation-free concurrency, V = RapidStore compatible W = slab-managed pointers, X = full system upgrades Y = allocator+bridge to Z = publication-quality benchmarking

Pattern source — DeepSpeed PartitionedOptimizerSwapper (deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py:27):
```python
class PartitionedOptimizerSwapper(OptimizerSwapper):
    def __init__(self, swap_config, aio_config, ...):
        self.aio_handle = aio_op.aio_handle(
            block_size=aio_config[AIO_BLOCK_SIZE],
            queue_depth=aio_config[AIO_QUEUE_DEPTH],
            overlap_events=aio_config[AIO_OVERLAP_EVENTS])
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.aio_handle, ...)
    def swap_in_optimizer_state(self, parameter, async_parameter=None): ...
    def release_swap_buffers(self, parameter): ...
```

---

## 20 Reference Infrastructure Repositories

| # | Repo | Org | Location | Key Patterns |
|---|------|-----|----------|-------------|
| 1 | NCCL | NVIDIA | infra-refs/nccl | ncclMemAlloc, freeMask/popFirstOneBit, cudaMemPoolCreate |
| 2 | CCCL | NVIDIA | infra-refs/cccl | shared_block_ptr::fetch_add (lockfree refcount) |
| 3 | Megatron-LM | NVIDIA | infra-refs/Megatron-LM | MultiGroupMemPoolAllocator, DistributedDataParallel |
| 4 | CUTLASS | NVIDIA | infra-refs/cutlass | Tiled memory access patterns |
| 5 | TensorRT | NVIDIA | infra-refs/TensorRT | IGpuAllocator interface |
| 6 | cuda-samples | NVIDIA | infra-refs/cuda-samples | cudaMallocManaged, peer access |
| 7 | Thrust | NVIDIA | infra-refs/thrust | lower_bound (GPU binary search) |
| 8 | FasterTransformer | NVIDIA | infra-refs/FasterTransformer | Buffer manager, device allocator |
| 9 | JAX | Google | infra-refs/jax | XLA memory allocation |
| 10 | TensorFlow | Google | infra-refs/tensorflow | Arena bump allocation |
| 11 | LevelDB | Google | infra-refs/leveldb | TwoLevelIterator::Seek |
| 12 | abseil-cpp | Google | infra-refs/abseil-cpp | Mutex ReaderLock/WriterLock |
| 13 | PyTorch | Meta | infra-refs/pytorch | CUDACachingAllocator, COWDeleter |
| 14 | FAISS | Meta | infra-refs/faiss | StandardGpuResourcesImpl |
| 15 | Triton | OpenAI | infra-refs/triton | Allocator protocol, kernel memory |
| 16 | LightSeq | ByteDance | infra-refs/lightseq | GPU memory pool |
| 17 | BytePS | ByteDance | infra-refs/byteps | Gradient partitioning |
| 18 | DeepSpeed | Microsoft | infra-refs/DeepSpeed | PartitionedOptimizerSwapper |
| 19 | vLLM | vLLM | infra-refs/vllm | Block-based KV cache management |
| 20 | flash-attention | Dao-AI | infra-refs/flash-attention | Tiled memory access, kBlockM/kBlockN |

---

## 38-Claude Development Schedule

### Phase 1: Core System — COMPLETED

| 实际 Claude # | 计划 Claude # | Milestones | Status | Lines | Key Deliverables |
|--------------|--------------|-----------|--------|-------|-----------------|
| — | **#1** | M001–M004 | ✅ DONE | 1,153 | TieredAllocator (waterfall HBM→GDDR→DRAM), TemporalBridge (ingest/partition/query), MigrationScheduler (background sweep), benchmark (1M edges), REVIEW_M001_M004.md |
| — | **#2** | M005–M006 | ✅ DONE | +247 = 1,400 | Lockfree touch() (CCCL fetch_add), shared_mutex (abseil ReaderLock), binary search scan_partition (LevelDB Seek, Thrust lower_bound), 43.9× scan speedup, REVIEW_M005_M006.md |
| — | **#3** | M007–M008 | ✅ DONE | +500 = 1,900 | SeqLock (wait-free reads), adaptive partitioning (TEM-Graph density), SlabAllocator (NCCL freeMask, PyTorch Block, TF Arena), data gen (4 JSON files × 2000 pts), REVIEW_M007_M008.md |
| — | **#4** | M009–M010 | ✅ DONE | +3,078 = 4,978 | CUDA backend, TierPtr, AsyncMigrator, PartitionIndex, TemporalEdge extraction, hetero_bench.cu |

> 注: Phase 1 的 Claude #1–#4 在单次批量提交 `4efda85` 中完成, M001–M011 核心系统已合并入 main.

### Phase 2: Upstream Integration — COMPLETED

| 实际 Claude # | 计划 Claude # | Milestones | Status | Lines | Key Deliverables |
|--------------|--------------|-----------|--------|-------|-----------------|
| **第 1 位 Claude** | **#5–#7** | M011–M016 | ✅ DONE | +3,942 = 8,920 | TEM-Graph index (interval+dll_list+tem_graph), RapidStore bridge (TieredSnapshot+wrapper API), 5 graph algorithms (BFS/PR/SSSP/WCC/TC), concurrent QueryExecutor, debug instrumentation, integration benchmark. Commits: `ef173d4`, `5014db6` |

### Phase 3: Real Datasets + Graph Algorithms

| 实际 Claude # | 计划 Claude # | Milestones | Status | Scope |
|--------------|--------------|-----------|--------|-------|
| **第 1 位 Claude** | **#8–#9** | M017–M020 | ✅ DONE | +3,725 = 12,645 | LDBC SNB loader (ldbc_types+ldbc_loader+ldbc_driver), tier cost model (TierCostModel: HBM=1ns/GDDR=5ns/DRAM=50ns), cross-tier BFS (direction-optimizing+prefetch+convergence log), cross-tier SSSP (delta-stepping+tier penalty), integration bench (ldbc_bench.cpp 5-test harness). Commit: `49bdd92` |
| **第 2 位 Claude** | **#10** | M021–M022 | ⬜ 待开发 | **Cross-tier PageRank + WCC**: 迭代算法分层梯度积累, 热点顶点驻留 HBM, 冷顶点降级 DRAM, 收敛曲线作为论文数据 |

### Phase 4: Advanced Memory Management

| 实际 Claude # | 计划 Claude # | Milestones | Status | Scope |
|--------------|--------------|-----------|--------|-------|
| **第 2 位 Claude** | **#11–#12** | M023–M026 | ⬜ 待开发 | **Prefetch engine** + **Compaction engine**: 查询历史预测 + 预迁移到 HBM, LRU+频率驱逐, 自动 slab 碎片整理, tier 再平衡 |
| **第 3 位 Claude** | **#13–#14** | M027–M030 | ⬜ 待开发 | **Multi-GPU** + **NVLink topology**: 跨 H100+A6000 分区, 设备感知 TieredAllocator, NCCL topo graph, Ring/Tree 路由 |

### Phase 5: Streaming + Complex Queries

| 实际 Claude # | 计划 Claude # | Milestones | Status | Scope |
|--------------|--------------|-----------|--------|-------|
| **第 3 位 Claude** | **#15–#16** | M031–M034 | ⬜ 待开发 | **Streaming ingestion** + **Checkpoint/restore**: 在线边到达增量重分区, 序列化 tier 状态+分区布局 |
| **第 4 位 Claude** | **#17–#20** | M035–M042 | ⬜ 待开发 | **Mixed read-write** + **Triangle counting** + **k-hop temporal** + **Temporal motif**: SeqLock 并发读写, 跨 tier 三角枚举, 多跳时序邻域, 滑动窗口模式检测 |

### Phase 6: Optimization + Integration Testing

| 实际 Claude # | 计划 Claude # | Milestones | Status | Scope |
|--------------|--------------|-----------|--------|-------|
| **第 4 位 Claude** | **#21–#23** | M043–M048 | ⬜ 待开发 | **Memory pressure eviction** + **Batch migration** + **Cost model**: RSS 监控+主动降级, 合并迁移减少 cudaMemcpy, ILP/贪心最优 tier 分配 |
| **第 5 位 Claude** | **#24–#26** | M049–M054 | ⬜ 待开发 | **TEM-Graph 集成测试** + **RapidStore 集成测试** + **LDBC benchmark**: 端到端正确性验证, 并发快照隔离, vs baseline 对比 |

### Phase 7: Publication Data Generation

| 实际 Claude # | 计划 Claude # | Milestones | Scope |
|--------------|--------------|-----------|-------|
| **第 6 位 Claude** | **#27–#29** | M055–M060 | **End-to-end benchmark** + **Profiling harness** + **Documentation**: 2000+ step 收敛曲线, nsys 集成, API 参考 |
| **第 7 位 Claude** | **#30–#32** | M061–M066 | **CMake build** + **CI/CD** + **Python bindings**: 统一构建, GitHub Actions, pybind11 接口 |

### Phase 8: Paper + Release

| 实际 Claude # | 计划 Claude # | Milestones | Scope |
|--------------|--------------|-----------|-------|
| **第 8 位 Claude** | **#33–#35** | M067–M072 | **Visualization dashboard** + **Paper: system+evaluation**: 查询延迟热力图, 架构描述, vs baseline 评估 |
| **第 9 位 Claude** | **#36–#38** | M073–M078 | **Paper: related work** + **Camera-ready** + **Final release**: 定位, 补充材料, artifact DOI |

---

## Current Codebase (After 第 1 位 Claude, M001–M020 complete)

```
src/ (8,920 lines total across 32 files)
├── core/           — 7 files, 1997 lines  [M001–M010]
│   ├── tiered_allocator.hpp     566 lines
│   ├── seqlock.hpp              129 lines
│   ├── slab_allocator.hpp       405 lines
│   ├── tier_ptr.hpp             184 lines
│   ├── async_migrator.hpp       440 lines
│   ├── partition_index.hpp      244 lines
│   └── temporal_edge.hpp         29 lines
├── bridge/         — 1 file, 510 lines   [M002,M006,M007]
│   └── temporal_bridge.hpp
├── scheduler/      — 1 file, 102 lines   [M003]
│   └── migration_scheduler.hpp
├── debug/          — 1 file, 281 lines   [M011]
│   └── philemon_debug.hpp       (TraceRing, TierPerfCounter, ScopedTimer)
├── index/          — 4 files, 1178 lines  [M011,M012]
│   ├── interval.hpp             171 lines  (from TEM-Graph)
│   ├── dll_list.hpp             190 lines  (from TEM-Graph)
│   ├── tem_graph.hpp            105 lines  (from TEM-Graph)
│   └── tem_graph_impl.hpp       712 lines  (from TEM-Graph)
├── wrapper/        — 5 files, 635 lines   [M013]
│   ├── rapidstore_wrapper.hpp   376 lines  (TieredSnapshot + wrapper:: API)
│   ├── graph_edge.hpp            88 lines  (+temporal fields)
│   ├── edge_stream.hpp          165 lines  (+load_from_temporal_edges)
│   ├── graph_edge_impl.hpp        3 lines  (compat wrapper)
│   └── edge_stream_impl.hpp       3 lines  (compat wrapper)
├── algorithms/     — 5 files, 980 lines   [M014]
│   ├── tiered_bfs.hpp           388 lines  (from RapidStore BFS)
│   ├── tiered_pagerank.hpp      165 lines  (from RapidStore PR)
│   ├── tiered_sssp.hpp          162 lines  (from RapidStore SSSP)
│   ├── tiered_wcc.hpp           159 lines  (from RapidStore WCC)
│   └── tiered_tc.hpp            106 lines  (from RapidStore TC)
├── executor/       — 3 files, 491 lines   [M015,M016]
│   ├── thread_pool_base.hpp     196 lines  (from RapidStore ThreadPool)
│   ├── spin_lock.hpp             64 lines  (from RapidStore SpinLock)
│   └── query_executor.hpp       231 lines  (NEW: concurrent query executor)
├── bench/          — 4 files, 1705 lines  [M004,M016]
│   ├── philemon_bench.cpp       389 lines
│   ├── philemon_data_fast.cpp   296 lines
│   ├── philemon_data_gen.cpp    645 lines
│   └── integration_bench.cpp    375 lines  (end-to-end M011-M016 test)
└── cuda/           — 1 file, 1039 lines   [M009,M010]
    └── hetero_bench.cu
```

## Claude 开发进度总览

```
第 1 位 Claude ✅ 完成: M001–M020 (核心系统 + upstream集成 + 算法 + executor + LDBC loader + cost model + 跨tier BFS/SSSP)
第 2 位 Claude ⬜ 待开发: M021–M026 (跨 tier PageRank/WCC + Prefetch + Compaction)
第 3 位 Claude ⬜ 待开发: M027–M034 (Multi-GPU + NVLink + Streaming + Checkpoint)
第 4 位 Claude ⬜ 待开发: M035–M048 (混合读写 + 复杂查询 + 内存优化 + 代价模型)
第 5 位 Claude ⬜ 待开发: M049–M054 (集成测试 + LDBC benchmark)
第 6 位 Claude ⬜ 待开发: M055–M060 (端到端 benchmark + profiling + 文档)
第 7 位 Claude ⬜ 待开发: M061–M066 (CMake + CI/CD + Python bindings)
第 8 位 Claude ⬜ 待开发: M067–M072 (可视化 + 论文系统描述/评估)
第 9 位 Claude ⬜ 待开发: M073–M078 (论文相关工作 + camera-ready + 最终发布)
```

## Pending Bugs (for Claude #4+)

| Bug | Description | Target |
|-----|------------|--------|
| 4.4 | Migration blocking (cudaMemcpy sync) | M009: cudaMemcpyAsync + double-buffer |
| 4.5 | get_ptr() escapes lock scope | M009: RAII TierPtr guard |
| 4.6 | SlabAllocator not thread-safe standalone | M025: per-pool spinlocks |
| 4.7 | Adaptive thresholds hardcoded | M017: LDBC calibration |
| 4.8 | compact_slabs() not automatic | M025: MigrationScheduler integration |
