/**
 * hetero_bench.cu — Philemon-TSH Heterogeneous Memory Benchmark
 *
 * Target: ags1 server
 *   GPU0: NVIDIA RTX A6000  (49 GB GDDR6)   PCIe NODE to NUMA1
 *   GPU1: NVIDIA RTX A6000  (49 GB GDDR6)   PCIe NODE to NUMA1
 *   GPU2: NVIDIA H100 NVL   (96 GB HBM2e)   PCIe Gen5 to NUMA1
 *   CPU:  2× AMD EPYC 9354  (128 threads)   ~1.5 TB DDR5
 *
 * Topology: all GPUs on NUMA1, no NVLink, PCIe-only peer paths.
 *
 * Experiments:
 *   E1: Per-tier allocation + bandwidth measurement (H2D, D2H, D2D)
 *   E2: Temporal edge partitioning across 4 tiers (H100-HBM, A6000-0-GDDR,
 *       A6000-1-GDDR, Host-DRAM) with placement policy
 *   E3: Cross-tier temporal subgraph query (gather from all tiers → host)
 *   E4: Migration latency (promote DRAM→HBM, demote HBM→GDDR, etc.)
 *   E5: Concurrent query throughput under background migration
 *   E6: Scaling: 1M → 10M → 50M → 100M edges
 *
 * Build:
 *   nvcc -std=c++17 -O2 -arch=sm_86 -gencode=arch=compute_90,code=sm_90 \
 *        -Xcompiler -pthread,-fopenmp \
 *        -o hetero_bench src/cuda/hetero_bench.cu
 *
 * (sm_86 for A6000, sm_90 for H100)
 *
 * Milestone: M009–M010 (Claude #4 equivalent, accelerated)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <unordered_map>
#include <string>

// ════════════════════════════════════════════════════════════════════════════
//  CUDA error checking
// ════════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ════════════════════════════════════════════════════════════════════════════
//  Data structures (mirrors the CPU-only version but GPU-aware)
// ════════════════════════════════════════════════════════════════════════════

struct TemporalEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_start;
    int32_t  ts_end;
};

// Memory tier — now maps to actual devices
enum class DeviceTier : uint8_t {
    H100_HBM   = 0,  // GPU2: H100 NVL  (HBM2e, ~3.35 TB/s)
    A6000_0    = 1,  // GPU0: A6000      (GDDR6, ~768 GB/s)
    A6000_1    = 2,  // GPU1: A6000      (GDDR6, ~768 GB/s)
    HOST_DRAM  = 3,  // CPU:  DDR5       (~80 GB/s per channel)
    TIER_COUNT = 4
};

static const char* tier_name(DeviceTier t) {
    switch (t) {
        case DeviceTier::H100_HBM:  return "H100-HBM";
        case DeviceTier::A6000_0:   return "A6000[0]-GDDR";
        case DeviceTier::A6000_1:   return "A6000[1]-GDDR";
        case DeviceTier::HOST_DRAM: return "Host-DRAM";
        default: return "UNKNOWN";
    }
}

static int tier_to_gpu(DeviceTier t) {
    switch (t) {
        case DeviceTier::H100_HBM:  return 2;
        case DeviceTier::A6000_0:   return 0;
        case DeviceTier::A6000_1:   return 1;
        case DeviceTier::HOST_DRAM: return -1; // host
        default: return -1;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Partition descriptor
// ════════════════════════════════════════════════════════════════════════════

struct Partition {
    uint64_t   id;
    DeviceTier tier;
    void*      dev_ptr;         // device or host pointer
    size_t     size_bytes;
    uint64_t   edge_count;
    int32_t    ts_lo, ts_hi;

    // Access tracking
    std::atomic<uint64_t> access_count{0};
    std::atomic<uint64_t> last_access_ns{0};

    // CUDA stream for async operations on this partition
    cudaStream_t stream;

    Partition()
        : id(0), tier(DeviceTier::HOST_DRAM), dev_ptr(nullptr),
          size_bytes(0), edge_count(0), ts_lo(0), ts_hi(0), stream(nullptr) {}
};

// ════════════════════════════════════════════════════════════════════════════
//  HeteroAllocator — real CUDA memory across devices
// ════════════════════════════════════════════════════════════════════════════

class HeteroAllocator {
public:
    HeteroAllocator() {
        // Query device properties
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        printf("[HeteroAllocator] Found %d CUDA devices\n", device_count);

        for (int d = 0; d < device_count && d < 3; ++d) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
            printf("  GPU%d: %s  VRAM=%.1f GB  SM=%d.%d  PCIe-Gen=%d\n",
                   d, prop.name,
                   prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                   prop.major, prop.minor,
                   prop.pciBusID > 0 ? 0 : 0);  // PCIe gen not in prop
            gpu_total_mem_[d] = prop.totalGlobalMem;
            gpu_used_mem_[d] = 0;
        }

        // Enable peer access between GPUs where possible
        for (int i = 0; i < device_count && i < 3; ++i) {
            for (int j = 0; j < device_count && j < 3; ++j) {
                if (i == j) continue;
                int can_access = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    CUDA_CHECK(cudaSetDevice(i));
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err == cudaSuccess) {
                        printf("  Peer access: GPU%d → GPU%d enabled\n", i, j);
                    } else if (err != cudaErrorPeerAccessAlreadyEnabled) {
                        printf("  Peer access: GPU%d → GPU%d FAILED: %s\n",
                               i, j, cudaGetErrorString(err));
                    }
                    cudaGetLastError(); // clear error
                }
            }
        }
        CUDA_CHECK(cudaSetDevice(0)); // reset
    }

    // Allocate on a specific tier
    void* allocate(DeviceTier tier, size_t size) {
        void* ptr = nullptr;
        int gpu = tier_to_gpu(tier);

        if (gpu >= 0) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaMalloc(&ptr, size));
            CUDA_CHECK(cudaMemset(ptr, 0, size));
            gpu_used_mem_[gpu] += size;
        } else {
            // Host pinned memory (for async DMA)
            CUDA_CHECK(cudaMallocHost(&ptr, size));
            memset(ptr, 0, size);
            host_used_ += size;
        }
        return ptr;
    }

    void deallocate(DeviceTier tier, void* ptr, size_t size) {
        int gpu = tier_to_gpu(tier);
        if (gpu >= 0) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaFree(ptr));
            gpu_used_mem_[gpu] -= size;
        } else {
            CUDA_CHECK(cudaFreeHost(ptr));
            host_used_ -= size;
        }
    }

    // Copy between any two tiers (sync)
    void copy_sync(DeviceTier dst_tier, void* dst,
                   DeviceTier src_tier, const void* src,
                   size_t size) {
        cudaMemcpyKind kind = cudaMemcpyDefault;
        // cudaMemcpyDefault handles all cases with UVA
        CUDA_CHECK(cudaMemcpy(dst, src, size, kind));
    }

    // Copy between any two tiers (async on stream)
    void copy_async(DeviceTier dst_tier, void* dst,
                    DeviceTier src_tier, const void* src,
                    size_t size, cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
    }

    void print_usage() const {
        printf("  Memory usage:\n");
        for (int d = 0; d < 3; ++d) {
            printf("    GPU%d: %.2f / %.2f GB\n",
                   d,
                   gpu_used_mem_[d] / (1024.0*1024.0*1024.0),
                   gpu_total_mem_[d] / (1024.0*1024.0*1024.0));
        }
        printf("    Host pinned: %.2f MB\n", host_used_ / (1024.0*1024.0));
    }

private:
    size_t gpu_total_mem_[3] = {};
    size_t gpu_used_mem_[3]  = {};
    size_t host_used_ = 0;
};

// ════════════════════════════════════════════════════════════════════════════
//  Synthetic edge generator
// ════════════════════════════════════════════════════════════════════════════

static std::vector<TemporalEdge>
generate_edges(size_t n, int32_t ts_min, int32_t ts_max, uint64_t max_vertex) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int32_t> ts_dist(ts_min, ts_max);
    std::vector<TemporalEdge> edges(n);

    for (size_t i = 0; i < n; ++i) {
        double u1 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        double u2 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        edges[i].source      = static_cast<uint64_t>(std::pow(u1, 2.0) * max_vertex);
        edges[i].destination = static_cast<uint64_t>(std::pow(u2, 2.0) * max_vertex);
        if (edges[i].source == edges[i].destination)
            edges[i].destination = (edges[i].destination + 1) % max_vertex;
        edges[i].weight   = 1.0;
        edges[i].ts_start = ts_dist(rng);
        int32_t dur = std::uniform_int_distribution<int32_t>(1, 100)(rng);
        edges[i].ts_end   = std::min(edges[i].ts_start + dur, ts_max);
    }
    return edges;
}

// ════════════════════════════════════════════════════════════════════════════
//  Timer utility
// ════════════════════════════════════════════════════════════════════════════

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
    float end(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(stop, s));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ════════════════════════════════════════════════════════════════════════════
//  E1: Bandwidth measurement between all tier pairs
// ════════════════════════════════════════════════════════════════════════════

static void experiment_bandwidth(HeteroAllocator& alloc) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  E1: Cross-Tier Bandwidth Matrix (GB/s)                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    const size_t SIZES[] = {
        1ULL << 20,   // 1 MB
        16ULL << 20,  // 16 MB
        64ULL << 20,  // 64 MB
        256ULL << 20, // 256 MB
    };
    const char* SIZE_LABELS[] = {"1MB", "16MB", "64MB", "256MB"};

    DeviceTier tiers[] = {
        DeviceTier::H100_HBM, DeviceTier::A6000_0,
        DeviceTier::A6000_1, DeviceTier::HOST_DRAM
    };

    for (int si = 0; si < 4; ++si) {
        size_t sz = SIZES[si];
        printf("  Transfer size: %s\n", SIZE_LABELS[si]);
        printf("  %16s", "src \\ dst");
        for (auto dst : tiers) printf("  %14s", tier_name(dst));
        printf("\n");

        for (auto src : tiers) {
            printf("  %16s", tier_name(src));
            void* src_ptr = alloc.allocate(src, sz);

            for (auto dst : tiers) {
                if (src == dst) {
                    printf("  %14s", "—");
                } else {
                    void* dst_ptr = alloc.allocate(dst, sz);

                    // Warmup
                    alloc.copy_sync(dst, dst_ptr, src, src_ptr, sz);

                    // Measure (5 iterations)
                    CudaTimer timer;
                    const int ITERS = 5;
                    timer.begin();
                    for (int i = 0; i < ITERS; ++i) {
                        alloc.copy_sync(dst, dst_ptr, src, src_ptr, sz);
                    }
                    float ms = timer.end();

                    double gb = (double)sz * ITERS / (1024.0*1024.0*1024.0);
                    double bw = gb / (ms / 1000.0);
                    printf("  %11.2f GB/s", bw);

                    alloc.deallocate(dst, dst_ptr, sz);
                }
            }
            printf("\n");
            alloc.deallocate(src, src_ptr, sz);
        }
        printf("\n");
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  E2: Temporal edge partitioning across 4 tiers
// ════════════════════════════════════════════════════════════════════════════

struct PartitionSet {
    std::vector<Partition> parts;
    std::shared_mutex mu;
    HeteroAllocator& alloc;

    PartitionSet(HeteroAllocator& a) : alloc(a) {}

    ~PartitionSet() {
        for (auto& p : parts) {
            if (p.dev_ptr) {
                alloc.deallocate(p.tier, p.dev_ptr, p.size_bytes);
            }
            if (p.stream) {
                cudaStreamDestroy(p.stream);
            }
        }
    }
};

static void partition_and_place(
    PartitionSet& ps,
    std::vector<TemporalEdge>& edges,
    size_t partition_cap)
{
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  E2: Partition + Tier Placement (%zu edges)                  ║\n",
           edges.size());
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    // Sort by ts_start
    std::sort(edges.begin(), edges.end(),
        [](const TemporalEdge& a, const TemporalEdge& b) {
            if (a.ts_start != b.ts_start) return a.ts_start < b.ts_start;
            return a.ts_end < b.ts_end;
        });

    auto t_sort = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(t_sort - t0).count();

    // Placement policy:
    //   - First 25% of partitions (hottest / most recent) → H100 HBM
    //   - Next 25% → A6000[0] GDDR
    //   - Next 25% → A6000[1] GDDR
    //   - Remaining → Host DRAM
    size_t total_parts = (edges.size() + partition_cap - 1) / partition_cap;

    size_t created = 0;
    for (size_t i = 0; i < edges.size(); i += partition_cap) {
        size_t end = std::min(i + partition_cap, edges.size());
        size_t count = end - i;
        size_t sz = count * sizeof(TemporalEdge);

        int32_t lo = edges[i].ts_start;
        int32_t hi = edges[end - 1].ts_end;
        for (size_t j = i; j < end; ++j) {
            hi = std::max(hi, edges[j].ts_end);
        }

        // Tier assignment based on partition index
        DeviceTier tier;
        double frac = (double)created / total_parts;
        if (frac < 0.25) {
            tier = DeviceTier::H100_HBM;
        } else if (frac < 0.50) {
            tier = DeviceTier::A6000_0;
        } else if (frac < 0.75) {
            tier = DeviceTier::A6000_1;
        } else {
            tier = DeviceTier::HOST_DRAM;
        }

        void* ptr = ps.alloc.allocate(tier, sz);

        // Upload edges to device
        if (tier_to_gpu(tier) >= 0) {
            CUDA_CHECK(cudaSetDevice(tier_to_gpu(tier)));
            CUDA_CHECK(cudaMemcpy(ptr, &edges[i], sz, cudaMemcpyHostToDevice));
        } else {
            memcpy(ptr, &edges[i], sz);
        }

        Partition part;
        part.id         = created + 1;
        part.tier       = tier;
        part.dev_ptr    = ptr;
        part.size_bytes = sz;
        part.edge_count = count;
        part.ts_lo      = lo;
        part.ts_hi      = hi;
        CUDA_CHECK(cudaStreamCreate(&part.stream));

        ps.parts.push_back(std::move(part));
        ++created;
    }

    auto t_place = std::chrono::high_resolution_clock::now();
    double place_ms = std::chrono::duration<double, std::milli>(t_place - t_sort).count();

    printf("  Sort: %.2f ms  |  Placement: %.2f ms  |  Partitions: %zu\n\n",
           sort_ms, place_ms, created);

    // Print partition layout
    printf("  %-6s  %-14s  %-20s  %-10s  %-10s\n",
           "ID", "Tier", "Timestamp Range", "Edges", "Size (MB)");
    printf("  %-6s  %-14s  %-20s  %-10s  %-10s\n",
           "------", "--------------", "--------------------",
           "----------", "----------");

    size_t tier_edges[4] = {};
    size_t tier_bytes[4] = {};

    for (auto& p : ps.parts) {
        char ts_range[32];
        snprintf(ts_range, sizeof(ts_range), "[%d, %d]", p.ts_lo, p.ts_hi);

        printf("  %-6lu  %-14s  %-20s  %-10lu  %-10.2f\n",
               (unsigned long)p.id,
               tier_name(p.tier),
               ts_range,
               (unsigned long)p.edge_count,
               p.size_bytes / (1024.0 * 1024.0));

        int ti = static_cast<int>(p.tier);
        tier_edges[ti] += p.edge_count;
        tier_bytes[ti] += p.size_bytes;
    }

    printf("\n  Tier Summary:\n");
    for (int i = 0; i < 4; ++i) {
        if (tier_edges[i] > 0) {
            printf("    %-14s  %zu edges  (%.2f MB)\n",
                   tier_name(static_cast<DeviceTier>(i)),
                   tier_edges[i],
                   tier_bytes[i] / (1024.0*1024.0));
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  E3: Cross-tier temporal subgraph query
// ════════════════════════════════════════════════════════════════════════════

struct QueryResult {
    uint64_t edge_count;
    double   gather_ms;     // D2H transfer time
    double   scan_ms;       // host-side scan time
    double   total_ms;
};

static QueryResult cross_tier_query(
    PartitionSet& ps,
    int32_t ts_lo, int32_t ts_hi,
    bool verbose = false)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: Find overlapping partitions
    std::vector<size_t> overlap_ids;
    {
        std::shared_lock<std::shared_mutex> lk(ps.mu);
        for (size_t i = 0; i < ps.parts.size(); ++i) {
            auto& p = ps.parts[i];
            if (p.ts_lo <= ts_hi && p.ts_hi >= ts_lo) {
                overlap_ids.push_back(i);
                p.access_count.fetch_add(1, std::memory_order_relaxed);
                auto now = std::chrono::steady_clock::now().time_since_epoch();
                p.last_access_ns.store(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()),
                    std::memory_order_relaxed);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Step 2: Gather data from all tiers to host (async, pipelined)
    // For each overlapping partition on GPU, initiate D2H transfer
    struct GatherBuf {
        std::vector<TemporalEdge> host_edges;
        size_t part_idx;
    };
    std::vector<GatherBuf> gathered(overlap_ids.size());
    std::vector<cudaEvent_t> events(overlap_ids.size());

    for (size_t gi = 0; gi < overlap_ids.size(); ++gi) {
        auto& p = ps.parts[overlap_ids[gi]];
        gathered[gi].part_idx = overlap_ids[gi];
        gathered[gi].host_edges.resize(p.edge_count);

        CUDA_CHECK(cudaEventCreate(&events[gi]));

        if (tier_to_gpu(p.tier) >= 0) {
            // GPU partition: async D2H
            CUDA_CHECK(cudaSetDevice(tier_to_gpu(p.tier)));
            CUDA_CHECK(cudaMemcpyAsync(
                gathered[gi].host_edges.data(),
                p.dev_ptr,
                p.size_bytes,
                cudaMemcpyDeviceToHost,
                p.stream));
            CUDA_CHECK(cudaEventRecord(events[gi], p.stream));
        } else {
            // Host partition: direct memcpy
            memcpy(gathered[gi].host_edges.data(), p.dev_ptr, p.size_bytes);
            CUDA_CHECK(cudaEventRecord(events[gi]));
        }
    }

    // Wait for all transfers
    for (size_t gi = 0; gi < events.size(); ++gi) {
        CUDA_CHECK(cudaEventSynchronize(events[gi]));
        cudaEventDestroy(events[gi]);
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // Step 3: Scan gathered edges with binary search
    uint64_t total_matched = 0;
    for (auto& gb : gathered) {
        auto& edges = gb.host_edges;
        // Binary search: lower_bound on ts_start >= ts_lo
        auto first = std::lower_bound(
            edges.begin(), edges.end(), ts_lo,
            [](const TemporalEdge& e, int32_t val) {
                return e.ts_start < val;
            });
        for (auto it = first; it != edges.end(); ++it) {
            if (it->ts_start > ts_hi) break;
            if (it->ts_end <= ts_hi) {
                ++total_matched;
            }
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    double gather_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double scan_ms   = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double total_ms  = std::chrono::duration<double, std::milli>(t3 - t0).count();

    if (verbose) {
        printf("    [%d,%d] → %zu partitions, %lu edges, "
               "gather=%.2fms scan=%.2fms total=%.2fms\n",
               ts_lo, ts_hi, overlap_ids.size(),
               (unsigned long)total_matched,
               gather_ms, scan_ms, total_ms);
    }

    return {total_matched, gather_ms, scan_ms, total_ms};
}

static void experiment_query(PartitionSet& ps) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  E3: Cross-Tier Temporal Subgraph Query                     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    struct QSpec { const char* label; int32_t lo, hi; };
    QSpec queries[] = {
        {"narrow  [1000,1050]", 1000, 1050},
        {"medium  [2000,3000]", 2000, 3000},
        {"wide    [0,5000]",    0,    5000},
        {"full    [0,10000]",   0,    10000},
    };

    printf("  %-22s  %10s  %10s  %10s  %10s  %10s\n",
           "Query", "Edges", "Gather(ms)", "Scan(ms)", "Total(ms)", "ns/edge");
    printf("  %-22s  %10s  %10s  %10s  %10s  %10s\n",
           "----------------------", "----------", "----------",
           "----------", "----------", "----------");

    for (auto& q : queries) {
        const int WARMUP = 3;
        const int ITERS  = 20;

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            cross_tier_query(ps, q.lo, q.hi);
        }

        // Measure
        double sum_gather = 0, sum_scan = 0, sum_total = 0;
        uint64_t edges = 0;
        for (int i = 0; i < ITERS; ++i) {
            auto r = cross_tier_query(ps, q.lo, q.hi);
            edges      = r.edge_count;
            sum_gather += r.gather_ms;
            sum_scan   += r.scan_ms;
            sum_total  += r.total_ms;
        }

        double avg_gather = sum_gather / ITERS;
        double avg_scan   = sum_scan   / ITERS;
        double avg_total  = sum_total  / ITERS;
        double ns_per     = edges > 0 ? avg_total * 1e6 / edges : 0;

        printf("  %-22s  %10lu  %10.3f  %10.3f  %10.3f  %10.2f\n",
               q.label, (unsigned long)edges,
               avg_gather, avg_scan, avg_total, ns_per);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  E4: Migration latency between tiers
// ════════════════════════════════════════════════════════════════════════════

static void experiment_migration(HeteroAllocator& alloc) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  E4: Migration Latency (async copy + pointer swap)          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    const size_t EDGE_COUNTS[] = {100000, 500000, 1000000, 5000000};

    DeviceTier tiers[] = {
        DeviceTier::H100_HBM, DeviceTier::A6000_0,
        DeviceTier::A6000_1, DeviceTier::HOST_DRAM
    };

    printf("  %-12s  %-16s  %-16s  %-10s  %-14s\n",
           "Edges", "From", "To", "ms", "BW (GB/s)");
    printf("  %-12s  %-16s  %-16s  %-10s  %-14s\n",
           "------------", "----------------", "----------------",
           "----------", "--------------");

    for (size_t ec : EDGE_COUNTS) {
        size_t sz = ec * sizeof(TemporalEdge);

        for (auto src : tiers) {
            for (auto dst : tiers) {
                if (src == dst) continue;

                // Only test key migration paths
                bool interesting =
                    (src == DeviceTier::HOST_DRAM && dst == DeviceTier::H100_HBM) ||
                    (src == DeviceTier::HOST_DRAM && dst == DeviceTier::A6000_0) ||
                    (src == DeviceTier::H100_HBM  && dst == DeviceTier::HOST_DRAM) ||
                    (src == DeviceTier::A6000_0   && dst == DeviceTier::H100_HBM) ||
                    (src == DeviceTier::A6000_0   && dst == DeviceTier::A6000_1) ||
                    (src == DeviceTier::H100_HBM  && dst == DeviceTier::A6000_0);
                if (!interesting) continue;

                void* src_ptr = alloc.allocate(src, sz);
                void* dst_ptr = alloc.allocate(dst, sz);

                // Warmup
                alloc.copy_sync(dst, dst_ptr, src, src_ptr, sz);

                // Measure async migration
                cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreate(&stream));

                CudaTimer timer;
                const int ITERS = 5;
                timer.begin(stream);
                for (int i = 0; i < ITERS; ++i) {
                    alloc.copy_async(dst, dst_ptr, src, src_ptr, sz, stream);
                }
                float ms = timer.end(stream);
                ms /= ITERS;

                double gb = sz / (1024.0*1024.0*1024.0);
                double bw = gb / (ms / 1000.0);

                printf("  %-12zu  %-16s  %-16s  %10.3f  %14.2f\n",
                       ec, tier_name(src), tier_name(dst), ms, bw);

                cudaStreamDestroy(stream);
                alloc.deallocate(dst, dst_ptr, sz);
                alloc.deallocate(src, src_ptr, sz);
            }
        }
        printf("  %s\n", "·····································"
               "·····························");
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  E5: Concurrent query + background migration
// ════════════════════════════════════════════════════════════════════════════

static void experiment_concurrent(PartitionSet& ps) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  E5: Concurrent Query + Background Migration                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    const int NUM_THREADS = 8;
    const int QUERIES_PER_THREAD = 5000;

    // Phase 1: queries only (no migration)
    {
        std::atomic<uint64_t> total_edges{0};
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            threads.emplace_back([&, tid]() {
                std::mt19937 rng(tid * 1000 + 42);
                std::uniform_int_distribution<int32_t> lo_dist(0, 9000);
                uint64_t local = 0;
                for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
                    int32_t lo = lo_dist(rng);
                    int32_t hi = lo + 50 + (q % 200);
                    auto r = cross_tier_query(ps, lo, hi);
                    local += r.edge_count;
                }
                total_edges.fetch_add(local, std::memory_order_relaxed);
            });
        }
        for (auto& t : threads) t.join();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uint64_t total_q = NUM_THREADS * QUERIES_PER_THREAD;
        printf("  Query-only: %lu queries in %.1f ms (%.0f QPS)\n",
               (unsigned long)total_q, ms, total_q * 1000.0 / ms);
        printf("    Total edges scanned: %lu\n\n",
               (unsigned long)total_edges.load());
    }

    // Phase 2: queries + background migration (promote DRAM→H100)
    {
        std::atomic<uint64_t> total_edges{0};
        std::atomic<uint64_t> migrations_done{0};
        std::atomic<bool> stop_migration{false};

        auto t0 = std::chrono::high_resolution_clock::now();

        // Migration thread: continuously promote hottest DRAM partitions to H100
        std::thread migrator([&]() {
            while (!stop_migration.load(std::memory_order_acquire)) {
                std::unique_lock<std::shared_mutex> lk(ps.mu);
                for (auto& p : ps.parts) {
                    if (p.tier == DeviceTier::HOST_DRAM &&
                        p.access_count.load(std::memory_order_relaxed) > 5) {
                        // Promote to H100 HBM
                        void* new_ptr = ps.alloc.allocate(DeviceTier::H100_HBM, p.size_bytes);
                        if (new_ptr) {
                            ps.alloc.copy_sync(DeviceTier::H100_HBM, new_ptr,
                                              p.tier, p.dev_ptr, p.size_bytes);
                            ps.alloc.deallocate(p.tier, p.dev_ptr, p.size_bytes);
                            p.dev_ptr = new_ptr;
                            p.tier = DeviceTier::H100_HBM;
                            migrations_done.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });

        // Query threads
        std::vector<std::thread> threads;
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            threads.emplace_back([&, tid]() {
                std::mt19937 rng(tid * 2000 + 42);
                std::uniform_int_distribution<int32_t> lo_dist(0, 9000);
                uint64_t local = 0;
                for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
                    int32_t lo = lo_dist(rng);
                    int32_t hi = lo + 50 + (q % 200);
                    auto r = cross_tier_query(ps, lo, hi);
                    local += r.edge_count;
                }
                total_edges.fetch_add(local, std::memory_order_relaxed);
            });
        }
        for (auto& t : threads) t.join();
        stop_migration.store(true, std::memory_order_release);
        migrator.join();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uint64_t total_q = NUM_THREADS * QUERIES_PER_THREAD;
        printf("  Query + migration: %lu queries in %.1f ms (%.0f QPS)\n",
               (unsigned long)total_q, ms, total_q * 1000.0 / ms);
        printf("    Migrations completed: %lu\n",
               (unsigned long)migrations_done.load());
        printf("    Total edges scanned: %lu\n",
               (unsigned long)total_edges.load());
    }

    // Show final partition layout after migration
    printf("\n  Post-migration tier distribution:\n");
    size_t tier_edges[4] = {};
    for (auto& p : ps.parts) {
        tier_edges[static_cast<int>(p.tier)] += p.edge_count;
    }
    for (int i = 0; i < 4; ++i) {
        if (tier_edges[i] > 0) {
            printf("    %-14s  %zu edges\n",
                   tier_name(static_cast<DeviceTier>(i)), tier_edges[i]);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  E6: Scaling experiment
// ════════════════════════════════════════════════════════════════════════════

static void experiment_scaling(HeteroAllocator& alloc) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  E6: Edge Count Scaling (1M → 10M → 50M → 100M)            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    const size_t EDGE_COUNTS[] = {1000000, 10000000, 50000000, 100000000};
    const size_t PARTITION_CAP = 500000; // 500K edges per partition

    printf("  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s  %-12s\n",
           "Edges", "Parts", "Sort(ms)", "Place(ms)", "Query(ms)", "Edges/Q",
           "Throughput");
    printf("  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s  %-12s\n",
           "----------", "----------", "----------", "----------",
           "----------", "----------", "------------");

    for (size_t n : EDGE_COUNTS) {
        // Check if we have enough VRAM
        size_t total_bytes = n * sizeof(TemporalEdge);
        size_t available = 0;
        for (int d = 0; d < 3; ++d) {
            size_t free_mem = 0, total_mem = 0;
            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            available += free_mem;
        }
        // Reserve 20% headroom
        if (total_bytes > available * 0.8) {
            printf("  %-10zu  SKIPPED (need %.1f GB, available %.1f GB)\n",
                   n,
                   total_bytes / (1024.0*1024.0*1024.0),
                   available / (1024.0*1024.0*1024.0));
            continue;
        }

        auto edges = generate_edges(n, 0, 10000, 1000000);

        // Sort
        auto ts = std::chrono::high_resolution_clock::now();
        std::sort(edges.begin(), edges.end(),
            [](const TemporalEdge& a, const TemporalEdge& b) {
                if (a.ts_start != b.ts_start) return a.ts_start < b.ts_start;
                return a.ts_end < b.ts_end;
            });
        auto te = std::chrono::high_resolution_clock::now();
        double sort_ms = std::chrono::duration<double, std::milli>(te - ts).count();

        // Partition + place
        PartitionSet ps(alloc);
        ts = std::chrono::high_resolution_clock::now();

        size_t total_parts = (edges.size() + PARTITION_CAP - 1) / PARTITION_CAP;
        for (size_t i = 0; i < edges.size(); i += PARTITION_CAP) {
            size_t end = std::min(i + PARTITION_CAP, edges.size());
            size_t count = end - i;
            size_t sz = count * sizeof(TemporalEdge);
            size_t idx = i / PARTITION_CAP;

            DeviceTier tier;
            double frac = (double)idx / total_parts;
            if (frac < 0.25)      tier = DeviceTier::H100_HBM;
            else if (frac < 0.50) tier = DeviceTier::A6000_0;
            else if (frac < 0.75) tier = DeviceTier::A6000_1;
            else                  tier = DeviceTier::HOST_DRAM;

            void* ptr = alloc.allocate(tier, sz);
            if (tier_to_gpu(tier) >= 0) {
                CUDA_CHECK(cudaSetDevice(tier_to_gpu(tier)));
                CUDA_CHECK(cudaMemcpy(ptr, &edges[i], sz, cudaMemcpyHostToDevice));
            } else {
                memcpy(ptr, &edges[i], sz);
            }

            Partition part;
            part.id = idx + 1;
            part.tier = tier;
            part.dev_ptr = ptr;
            part.size_bytes = sz;
            part.edge_count = count;
            part.ts_lo = edges[i].ts_start;
            part.ts_hi = edges[end-1].ts_end;
            for (size_t j = i; j < end; ++j)
                part.ts_hi = std::max(part.ts_hi, edges[j].ts_end);
            CUDA_CHECK(cudaStreamCreate(&part.stream));
            ps.parts.push_back(std::move(part));
        }

        te = std::chrono::high_resolution_clock::now();
        double place_ms = std::chrono::duration<double, std::milli>(te - ts).count();

        // Free edge vector to save memory
        edges.clear();
        edges.shrink_to_fit();

        // Query: medium range
        const int ITERS = 10;
        double sum_ms = 0;
        uint64_t qedges = 0;
        for (int i = 0; i < ITERS; ++i) {
            auto r = cross_tier_query(ps, 2000, 3000);
            sum_ms += r.total_ms;
            qedges = r.edge_count;
        }
        double avg_query = sum_ms / ITERS;

        printf("  %-10zu  %-10zu  %-10.1f  %-10.1f  %-10.2f  %-10lu  %.0f Medges/s\n",
               n, ps.parts.size(), sort_ms, place_ms,
               avg_query, (unsigned long)qedges,
               qedges / (avg_query / 1000.0) / 1e6);

        // Explicit cleanup (PartitionSet destructor handles it)
    }
}


// ════════════════════════════════════════════════════════════════════════════
//  MAIN
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Philemon-TSH — Heterogeneous GPU Benchmark                 ║\n");
    printf("║  A6000 × 2 + H100 NVL × 1 + Host DRAM                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    // Print system info
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    printf("System: %d CUDA devices\n", dev_count);
    for (int d = 0; d < dev_count; ++d) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        printf("  GPU%d: %-24s  SM=%d.%d  VRAM=%.1f/%.1f GB free\n",
               d, prop.name, prop.major, prop.minor,
               free_mem / (1024.0*1024.0*1024.0),
               total_mem / (1024.0*1024.0*1024.0));
    }
    printf("\n");

    HeteroAllocator alloc;

    // ── E1: Bandwidth ─────────────────────────────────────────────
    experiment_bandwidth(alloc);

    // ── E2: Partition + Place ─────────────────────────────────────
    const size_t NUM_EDGES = 10'000'000;  // 10M edges for main experiments
    const size_t PARTITION_CAP = 500'000; // 500K edges per partition
    auto edges = generate_edges(NUM_EDGES, 0, 10000, 1000000);
    printf("  Generated %zu edges (%.2f MB)\n\n",
           edges.size(), edges.size() * sizeof(TemporalEdge) / (1024.0*1024.0));

    PartitionSet ps(alloc);
    partition_and_place(ps, edges, PARTITION_CAP);
    edges.clear(); edges.shrink_to_fit();

    alloc.print_usage();

    // ── E3: Query ─────────────────────────────────────────────────
    experiment_query(ps);

    // ── E4: Migration ─────────────────────────────────────────────
    experiment_migration(alloc);

    // ── E5: Concurrent ────────────────────────────────────────────
    experiment_concurrent(ps);

    // ── E6: Scaling ───────────────────────────────────────────────
    // Clean up main partition set first to free GPU memory
    ps.parts.clear();

    experiment_scaling(alloc);

    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  All experiments complete.                                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
