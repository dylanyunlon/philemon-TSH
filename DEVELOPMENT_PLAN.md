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

### Phase 1: Core System (Claude #1–#3) — COMPLETED

| Claude # | Milestones | Status | Lines | Key Deliverables |
|----------|-----------|--------|-------|-----------------|
| **#1** | M001–M004 | ✅ DONE | 1,153 | TieredAllocator (waterfall HBM→GDDR→DRAM), TemporalBridge (ingest/partition/query), MigrationScheduler (background sweep), benchmark (1M edges), REVIEW_M001_M004.md |
| **#2** | M005–M006 | ✅ DONE | +247 = 1,400 | Lockfree touch() (CCCL fetch_add), shared_mutex (abseil ReaderLock), binary search scan_partition (LevelDB Seek, Thrust lower_bound), 43.9× scan speedup, REVIEW_M005_M006.md |
| **#3** | M007–M008 | ✅ DONE | +500 = 1,900 | SeqLock (wait-free reads), adaptive partitioning (TEM-Graph density), SlabAllocator (NCCL freeMask, PyTorch Block, TF Arena), data gen (4 JSON files × 2000 pts), REVIEW_M007_M008.md |

### Phase 2: CUDA + Upstream Integration (Claude #4–#7)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#4** | M009–M010 | **CUDA backend**: Replace posix_memalign with cudaMalloc/cudaMallocHost. cudaMemcpyAsync + double-buffered migration (DeepSpeed AsyncTensorSwapper pattern). RAII TierPtr guard for get_ptr() (Bug 4.5). NVLink P2P copy via cudaMemcpyPeer. Pattern: NCCL ncclMemAlloc → cuMemCreate → cuMemMap. |
| **#5** | M011–M012 | **TEM-Graph index integration**: Build TEM-Graph interval index within each partition. contains_query/contained_query accelerate temporal subgraph extraction. Replace linear scan_partition with index-based O(output) traversal. Pattern: TEM-Graph build_index → doubly-linked list + successor pointers. |
| **#6** | M013–M014 | **RapidStore bridge**: Expose tiered partitions as RapidStore snapshots. Wrap TieredAllocator behind RapidStore's wrapper::snapshot_edges API. Enable RapidStore algorithms (BFS, SSSP) on tiered data. Pattern: RapidStore wrapper → template dispatch. |
| **#7** | M015–M016 | **Concurrent query executor**: Thread pool with per-partition parallelism. Work-stealing across tiers (HBM queries first). Batched query interface for throughput. Pattern: Megatron DistributedDataParallel → bucket-based parallelism. |

### Phase 3: Real Datasets + Graph Algorithms (Claude #8–#10)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#8** | M017–M018 | **LDBC SNB loader**: Parse LDBC temporal graph (person-knows-person with timestamps). Calibrate adaptive partition thresholds from real data distributions. Benchmark on LDBC SF-1, SF-10, SF-100. |
| **#9** | M019–M020 | **Cross-tier BFS/SSSP**: Algorithms that span HBM+GDDR+DRAM. Automatic partition prefetch when BFS frontier crosses tier boundary. Cost model: HBM access 1ns, GDDR 5ns, DRAM 50ns. |
| **#10** | M021–M022 | **Cross-tier PageRank + WCC**: Iterative algorithms with tiered gradient accumulation. Hot vertices stay in HBM, cold vertices demoted to DRAM. Convergence curves as publication data. |

### Phase 4: Advanced Memory Management (Claude #11–#14)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#11** | M023–M024 | **Prefetch engine**: Predict next-access partition from query history. Pre-migrate to HBM before query arrives. LRU + frequency-based eviction policy. |
| **#12** | M025–M026 | **Compaction engine**: Automatic slab defragmentation (Bug 4.8). Tier rebalancing when usage exceeds 80%. compact_slabs() in MigrationScheduler loop. |
| **#13** | M027–M028 | **Multi-GPU support**: Partition across H100 + A6000 devices. Device-aware TieredAllocator with per-GPU HBM pools. Pattern: Megatron MultiGroupMemPoolAllocator. |
| **#14** | M029–M030 | **NVLink topology**: NCCL topo graph for optimal partition placement. Ring/tree routing for inter-device migration. Pattern: NCCL topology detection. |

### Phase 5: Streaming + Complex Queries (Claude #15–#20)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#15** | M031–M032 | **Streaming ingestion**: Online edge arrival with incremental re-partitioning. Amortized flush: batch edges, sort, merge into existing partitions. |
| **#16** | M033–M034 | **Checkpoint/restore**: Serialize tier state + partition layout to persistent storage. Resume from checkpoint after restart. |
| **#17** | M035–M036 | **Mixed read-write workload**: Concurrent insert + temporal query. SeqLock enables non-blocking reads during write bursts. |
| **#18** | M037–M038 | **Triangle counting**: Per-partition triangle enumeration with cross-tier edge lookup. Aggregate across tiers. |
| **#19** | M039–M040 | **k-hop temporal neighborhood**: Multi-hop BFS with temporal constraints. Each hop may cross tier boundaries. |
| **#20** | M041–M042 | **Temporal motif mining**: Detect recurring temporal patterns across partitions. Sliding window over time range. |

### Phase 6: Optimization + Integration Testing (Claude #21–#26)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#21** | M043–M044 | **Memory pressure eviction**: RSS monitoring + proactive demotion. OOM-triggered release (PyTorch pattern). |
| **#22** | M045–M046 | **Batch migration**: Coalesce multiple partition moves into one transfer. Reduce cudaMemcpy call overhead. |
| **#23** | M047–M048 | **Cost model**: bandwidth × latency × query-miss penalty. Optimal tier assignment via ILP or greedy. |
| **#24** | M049–M050 | **TEM-Graph integration tests**: End-to-end contains_query on tiered data. Correctness validation against baseline. |
| **#25** | M051–M052 | **RapidStore integration tests**: End-to-end snapshot_edges on tiered partitions. Concurrent snapshot isolation. |
| **#26** | M053–M054 | **LDBC benchmark suite**: Interactive Short queries on tiered graph. Compare vs TEM-Graph-only and RapidStore-only baselines. |

### Phase 7: Publication Data Generation (Claude #27–#32)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#27** | M055–M056 | **End-to-end benchmark**: Temporal PageRank convergence across tiers. 2000+ step convergence curves. |
| **#28** | M057–M058 | **Profiling harness**: nsys integration, bandwidth utilization metrics. Per-tier throughput breakdown. |
| **#29** | M059–M060 | **Documentation**: API reference, architecture diagrams, deployment guide. |
| **#30** | M061–M062 | **CMake build system**: Unified build with upstream TEM-Graph + RapidStore. CUDA optional. |
| **#31** | M063–M064 | **CI/CD pipeline**: GitHub Actions. Automated benchmark regression. Performance gates. |
| **#32** | M065–M066 | **Python bindings**: pybind11 for TemporalBridge + query interface. Jupyter notebook demos. |

### Phase 8: Paper + Release (Claude #33–#38)

| Claude # | Milestones | Scope |
|----------|-----------|-------|
| **#33** | M067–M068 | **Visualization dashboard**: Query latency heatmap by time range × tier. Interactive React component. |
| **#34** | M069–M070 | **Paper: system description**: Architecture, design decisions, pattern lineage from 20 repos. |
| **#35** | M071–M072 | **Paper: evaluation**: vs baseline TEM-Graph, vs RapidStore-only. Scalability to 100M edges. |
| **#36** | M073–M074 | **Paper: related work + conclusion**: Position in temporal graph processing landscape. |
| **#37** | M075–M076 | **Camera-ready**: Supplementary material, artifact packaging, reproducibility scripts. |
| **#38** | M077–M078 | **Final integration test**: Release tagging, artifact DOI, README for reproducibility. |

---

## Current Codebase (After Claude #1 restart, incorporating #1–#3 work)

```
src/
├── core/
│   ├── tiered_allocator.hpp     474 lines  [M001,M005,M008]
│   ├── seqlock.hpp              129 lines  [M007]
│   └── slab_allocator.hpp       405 lines  [M008]
├── bridge/
│   └── temporal_bridge.hpp      480 lines  [M002,M006,M007]
├── scheduler/
│   └── migration_scheduler.hpp  102 lines  [M003]
└── bench/
    ├── philemon_bench.cpp       310 lines  [M004,M008]
    ├── philemon_data_fast.cpp   296 lines  [data gen]
    └── philemon_data_gen.cpp    645 lines  [data gen, full]
                                ──────
                          Total: 2,841 lines

upstream/
├── temgraph/          TEM-Graph temporal interval index
└── rapidstore/        RapidStore dynamic graph storage

infra-refs/ (20 repos, shallow clones)

data output (4 JSON files, 2000 pts/curve × 3 seeds each):
├── philemon_query_latency_2000.json     481 KB
├── philemon_qps_2000.json              404 KB
├── philemon_memory_util_2000.json      135 KB
└── philemon_migration_cost_2000.json   206 KB
```

## Pending Bugs (for Claude #4+)

| Bug | Description | Target |
|-----|------------|--------|
| 4.4 | Migration blocking (cudaMemcpy sync) | M009: cudaMemcpyAsync + double-buffer |
| 4.5 | get_ptr() escapes lock scope | M009: RAII TierPtr guard |
| 4.6 | SlabAllocator not thread-safe standalone | M025: per-pool spinlocks |
| 4.7 | Adaptive thresholds hardcoded | M017: LDBC calibration |
| 4.8 | compact_slabs() not automatic | M025: MigrationScheduler integration |
