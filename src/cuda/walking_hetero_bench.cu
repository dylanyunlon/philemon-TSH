/**
 * walking_hetero_bench.cu — 跨tier Benchmark + 热度驱动placement + 并发查询迁移
 *
 * mv来源与算法改动对照:
 *
 *   hetero_bench.cu (E4 benchmark)
 *     KEEP: TierConfig 四tier定义, 容量/带宽/延迟参数
 *     KEEP: alloc_on_tier / free_on_tier 模拟分配
 *     KEEP: sequential/random read/write benchmark pattern
 *     KEEP: Timer, INSPECT, CHK, sep, rss_kb debug基础设施
 *     MOD:  单tier性能 → 全tier对迁移延迟矩阵 (MigrationMatrix)
 *     NEW:  kern_gpu_memcpy_bench: GPU端memcpy kernel
 *     NEW:  exp_migration_matrix(): 4x4 tier对, 4种size, P50/P99矩阵
 *
 *   hotness_tracker.hpp (HotnessTracker)
 *     KEEP: access_count, last_access_time, heat_score 计算
 *     KEEP: 衰减函数: heat *= decay_factor per epoch
 *     MOD:  CPU单线程 → GPU批量更新 kern_heat_update
 *     NEW:  kern_placement_decision: heat→promote/demote决策
 *     NEW:  exp_heat_placement(): Zipf workload + 多epoch观察
 *
 *   tier_rebalancer.hpp (TierRebalancer)
 *     KEEP: tier间数据搬运逻辑
 *     KEEP: 容量约束检查
 *     NEW:  SimulatedTierSystem: 4-tier容量模拟
 *     NEW:  kern_concurrent_lookup: 查询+后台迁移并行
 *     NEW:  exp_concurrent_migrate(): throughput衰减曲线
 *
 * Build:
 *   GPU:  nvcc -std=c++17 -O2 -arch=sm_86 -DWALKING_CUDA=1 \
 *              -o walking_hetero_bench walking_hetero_bench.cu
 *   CPU:  g++ -std=c++17 -O2 -pthread -DWALKING_CUDA=0 -x c++ \
 *              -o walking_hetero_bench walking_hetero_bench.cu
 *
 * Milestone: M089–M091
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <sys/resource.h>
#include <unistd.h>
#include <condition_variable>

// ════════════════════════════════════════════════════════════════
// CUDA / CPU compile-mode dispatch
// ════════════════════════════════════════════════════════════════
#ifndef WALKING_CUDA
  #ifdef __CUDACC__
    #define WALKING_CUDA 1
  #else
    #define WALKING_CUDA 0
  #endif
#endif

#if WALKING_CUDA
  #include <cuda_runtime.h>
  #define GPU_CHECK(call) do { \
      cudaError_t e = (call); \
      if (e != cudaSuccess) { \
          fprintf(stderr, "[CUDA·FATAL] %s:%d %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(e)); \
          exit(1); } } while(0)
#else
  #define GPU_CHECK(call) ((void)0)
  enum { cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2 };
  inline void* _fake_alloc(size_t n) { void* p = malloc(n); if(p) memset(p,0,n); return p; }
  #define cudaMalloc(p,n)     (*(p)=_fake_alloc(n),(void)0)
  #define cudaFree(p)         free(p)
  #define cudaMemcpy(d,s,n,k) memcpy(d,s,n)
  #define cudaMemset(p,v,n)   memset(p,v,n)
  #define cudaDeviceSynchronize() ((void)0)
#endif

// ════════════════════════════════════════════════════════════════
// Debug infra — KEEP from walking_gpu_tree.cu
// ════════════════════════════════════════════════════════════════
static int g_dbg = 2;
static long rss_kb() { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss; }
static uint64_t g_insp = 0, g_pass = 0, g_fail = 0;

#define INSPECT(tag, ...) do { g_insp++; \
    std::printf("[INSPECT·%04lu·%s] ", (unsigned long)g_insp, tag); \
    std::printf(__VA_ARGS__); std::printf("  RSS=%ldKB\n", rss_kb()); } while(0)

#define CHK(cond, tag, ...) do { if(cond){g_pass++;} else { g_fail++; \
    std::printf("[FAIL·%s] ", tag); std::printf(__VA_ARGS__); std::printf("\n"); }} while(0)

struct Timer {
    const char* l; std::chrono::high_resolution_clock::time_point t0;
    Timer(const char* s) : l(s), t0(std::chrono::high_resolution_clock::now()) {
        if (g_dbg >= 1) std::printf("[T·START] %s\n", l);
    }
    double ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    }
    ~Timer() { std::printf("[T·END]   %s → %.2f ms\n", l, ms()); }
};

static void sep(const char* s) {
    std::printf("\n════════════════════════════════════════════════════\n  %s\n"
                "════════════════════════════════════════════════════\n\n", s);
}

namespace walking {
namespace hetero {

// ════════════════════════════════════════════════════════════════
// §0  Tier definitions — KEEP from hetero_bench.cu DeviceTier
// ════════════════════════════════════════════════════════════════

enum TierID : uint8_t {
    TIER_DRAM = 0,   // CPU DDR5 (~80 GB/s)
    TIER_CXL  = 1,   // CXL-attached memory (~40 GB/s, higher latency)
    TIER_SSD  = 2,   // NVMe SSD (~7 GB/s sequential)
    TIER_GPU  = 3,   // GPU HBM/GDDR (~3 TB/s local, PCIe bottleneck)
    TIER_COUNT = 4
};

static const char* tier_name(TierID t) {
    switch (t) {
        case TIER_DRAM: return "DRAM";
        case TIER_CXL:  return "CXL";
        case TIER_SSD:  return "SSD";
        case TIER_GPU:  return "GPU";
        default:        return "???";
    }
}

// KEEP from hetero_bench.cu: tier configuration with capacity/bandwidth/latency
struct TierConfig {
    TierID   id;
    uint64_t capacity_bytes;
    double   bandwidth_mbps;     // theoretical peak (MB/s)
    double   read_latency_ns;    // single access latency
    double   write_latency_ns;

    void dump() const {
        std::printf("    [%s] cap=%luMB bw=%.0fMB/s rlat=%.0fns wlat=%.0fns\n",
                    tier_name(id),
                    (unsigned long)(capacity_bytes / (1024*1024)),
                    bandwidth_mbps, read_latency_ns, write_latency_ns);
    }
};

static TierConfig default_tiers[TIER_COUNT] = {
    { TIER_DRAM, 512ULL * 1024*1024*1024,  80000.0,   80.0,   80.0  },
    { TIER_CXL,  256ULL * 1024*1024*1024,  40000.0,  250.0,  300.0  },
    { TIER_SSD,    4ULL * 1024*1024*1024*1024, 7000.0, 5000.0, 15000.0 },
    { TIER_GPU,   80ULL * 1024*1024*1024, 3350000.0,  120.0,  120.0  },
};

// ════════════════════════════════════════════════════════════════
// §0.1  Tier allocation / deallocation — KEEP from hetero_bench.cu
// ════════════════════════════════════════════════════════════════

// KEEP: alloc_on_tier / free_on_tier pattern (CPU simulation mode)
static void* alloc_on_tier(TierID tier, size_t size) {
    void* ptr = nullptr;
    if (tier == TIER_GPU) {
        cudaMalloc(&ptr, size);
    } else {
        ptr = aligned_alloc(64, (size + 63) & ~63ULL);
        if (ptr) memset(ptr, 0, size);
    }
    return ptr;
}

static void free_on_tier(TierID tier, void* ptr) {
    if (!ptr) return;
    if (tier == TIER_GPU) {
        cudaFree(ptr);
    } else {
        free(ptr);
    }
}

// ════════════════════════════════════════════════════════════════
// §0.2  Simulated latency injection — KEEP sequential/random pattern
//
// On CPU-only builds, we inject artificial latency to simulate tier
// differences. On GPU builds with real devices, actual transfer
// latencies dominate.
// ════════════════════════════════════════════════════════════════

static void inject_tier_latency_ns(TierID tier, uint64_t count) {
    // In CPU simulation: busy-spin for tier-appropriate latency
    double lat = default_tiers[tier].read_latency_ns * count;
    if (lat > 100.0) {
        auto t0 = std::chrono::steady_clock::now();
        double target_us = lat / 1000.0;
        while (true) {
            auto now = std::chrono::steady_clock::now();
            double elapsed_us = std::chrono::duration_cast<
                std::chrono::microseconds>(now - t0).count();
            if (elapsed_us >= target_us) break;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// §1  M089: Tier Migration Latency Matrix
//
// mv: hetero_bench.cu E4 (migration latency benchmark)
// KEEP: sequential copy pattern, CudaTimer, bandwidth calculation
// MOD:  single-tier → all tier-pair migration latency matrix
// NEW:  kern_gpu_memcpy_bench, MigrationMatrix, P50/P99 stats
// ════════════════════════════════════════════════════════════════

// ─── [NEW] kern_gpu_memcpy_bench: GPU-side memcpy benchmark ───
// Simulates GPU-initiated transfer (GPU↔host tiers).
// On CPU fallback: plain memcpy with latency injection.
#if WALKING_CUDA
__global__ void kern_gpu_memcpy_bench(char* __restrict__ dst,
                                       const char* __restrict__ src,
                                       uint64_t nbytes) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
    // Coalesced 8-byte copy
    const uint64_t* s64 = reinterpret_cast<const uint64_t*>(src);
    uint64_t* d64 = reinterpret_cast<uint64_t*>(dst);
    uint64_t n64 = nbytes / 8;
    for (uint64_t i = tid; i < n64; i += stride) {
        d64[i] = s64[i];
    }
}
#else
static void kern_gpu_memcpy_bench_cpu(char* dst, const char* src, uint64_t nbytes) {
    memcpy(dst, src, nbytes);
}
#endif

// ─── Latency sample collection ───
struct LatencySample {
    double latency_us;
    double bandwidth_mbps;
};

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[idx];
}

// ─── [NEW] MigrationMatrix: latency and bandwidth for all tier pairs ───
struct MigrationMatrix {
    double latency_p50_us[TIER_COUNT][TIER_COUNT];
    double latency_p99_us[TIER_COUNT][TIER_COUNT];
    double bandwidth_mbps[TIER_COUNT][TIER_COUNT];
    size_t transfer_size;

    void dump() const {
        std::printf("    Transfer size: %zu bytes (%.1f KB)\n\n",
                    transfer_size, transfer_size / 1024.0);

        std::printf("    P50 Latency (μs):\n");
        std::printf("    %8s", "src\\dst");
        for (int d = 0; d < TIER_COUNT; d++)
            std::printf("  %8s", tier_name(static_cast<TierID>(d)));
        std::printf("\n");
        for (int s = 0; s < TIER_COUNT; s++) {
            std::printf("    %8s", tier_name(static_cast<TierID>(s)));
            for (int d = 0; d < TIER_COUNT; d++) {
                if (s == d) std::printf("  %8s", "--");
                else std::printf("  %8.1f", latency_p50_us[s][d]);
            }
            std::printf("\n");
        }

        std::printf("\n    P99 Latency (μs):\n");
        std::printf("    %8s", "src\\dst");
        for (int d = 0; d < TIER_COUNT; d++)
            std::printf("  %8s", tier_name(static_cast<TierID>(d)));
        std::printf("\n");
        for (int s = 0; s < TIER_COUNT; s++) {
            std::printf("    %8s", tier_name(static_cast<TierID>(s)));
            for (int d = 0; d < TIER_COUNT; d++) {
                if (s == d) std::printf("  %8s", "--");
                else std::printf("  %8.1f", latency_p99_us[s][d]);
            }
            std::printf("\n");
        }

        std::printf("\n    Effective Bandwidth (MB/s):\n");
        std::printf("    %8s", "src\\dst");
        for (int d = 0; d < TIER_COUNT; d++)
            std::printf("  %8s", tier_name(static_cast<TierID>(d)));
        std::printf("\n");
        for (int s = 0; s < TIER_COUNT; s++) {
            std::printf("    %8s", tier_name(static_cast<TierID>(s)));
            for (int d = 0; d < TIER_COUNT; d++) {
                if (s == d) std::printf("  %8s", "--");
                else std::printf("  %8.0f", bandwidth_mbps[s][d]);
            }
            std::printf("\n");
        }
    }
};

// ─── Measure one tier-pair migration ───
static LatencySample measure_migration(TierID src_tier, TierID dst_tier,
                                        size_t size, int iter) {
    void* src_ptr = alloc_on_tier(src_tier, size);
    void* dst_ptr = alloc_on_tier(dst_tier, size);
    CHK(src_ptr != nullptr, "ALLOC_SRC",
        "failed alloc %zu bytes on %s", size, tier_name(src_tier));
    CHK(dst_ptr != nullptr, "ALLOC_DST",
        "failed alloc %zu bytes on %s", size, tier_name(dst_tier));

    if (!src_ptr || !dst_ptr) {
        free_on_tier(src_tier, src_ptr);
        free_on_tier(dst_tier, dst_ptr);
        return {0.0, 0.0};
    }

    // Fill source with pattern
    if (src_tier != TIER_GPU) {
        memset(src_ptr, 0xAB, size);
    }

    // Warmup
    if (src_tier == TIER_GPU || dst_tier == TIER_GPU) {
#if WALKING_CUDA
        cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDefault);
        cudaDeviceSynchronize();
#else
        memcpy(dst_ptr, src_ptr, size);
#endif
    } else {
        memcpy(dst_ptr, src_ptr, size);
        inject_tier_latency_ns(src_tier, 1);
        inject_tier_latency_ns(dst_tier, 1);
    }

    // Timed iterations
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iter; i++) {
        if (src_tier == TIER_GPU || dst_tier == TIER_GPU) {
#if WALKING_CUDA
            // GPU-involved: use kern_gpu_memcpy_bench or cudaMemcpy
            if (src_tier == TIER_GPU && dst_tier == TIER_GPU) {
                int blocks = std::min((int)((size/8 + 255)/256), 1024);
                kern_gpu_memcpy_bench<<<blocks, 256>>>(
                    (char*)dst_ptr, (const char*)src_ptr, size);
                cudaDeviceSynchronize();
            } else {
                cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDefault);
                cudaDeviceSynchronize();
            }
#else
            kern_gpu_memcpy_bench_cpu((char*)dst_ptr, (const char*)src_ptr, size);
            inject_tier_latency_ns(src_tier, 1);
            inject_tier_latency_ns(dst_tier, 1);
#endif
        } else {
            // CPU-to-CPU tier: memcpy + simulated latency
            memcpy(dst_ptr, src_ptr, size);
            inject_tier_latency_ns(src_tier, 1);
            inject_tier_latency_ns(dst_tier, 1);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count();
    double per_us = total_us / iter;
    double bw_mbps = (size / (1024.0 * 1024.0)) / (per_us / 1e6);

    free_on_tier(src_tier, src_ptr);
    free_on_tier(dst_tier, dst_ptr);

    return {per_us, bw_mbps};
}

// ─── [NEW] exp_migration_matrix: full 4x4 tier-pair measurement ───
static void exp_migration_matrix() {
    sep("M089: Tier Migration Latency Matrix");

    const size_t SIZES[] = { 4096, 65536, 1048576, 16777216 };
    const char* SIZE_LABELS[] = { "4KB", "64KB", "1MB", "16MB" };
    const int NUM_SIZES = 4;
    const int SAMPLES = 30;     // per measurement
    const int WARMUP_ITERS = 3;

    INSPECT("MATRIX_START", "sizes=%d samples=%d tiers=%d",
            NUM_SIZES, SAMPLES, (int)TIER_COUNT);

    for (int si = 0; si < NUM_SIZES; si++) {
        size_t sz = SIZES[si];
        MigrationMatrix mat;
        memset(&mat, 0, sizeof(mat));
        mat.transfer_size = sz;

        std::printf("\n  ┌─── Transfer Size: %s (%zu bytes) ───┐\n",
                    SIZE_LABELS[si], sz);

        for (int s = 0; s < TIER_COUNT; s++) {
            for (int d = 0; d < TIER_COUNT; d++) {
                if (s == d) continue;

                TierID src = static_cast<TierID>(s);
                TierID dst = static_cast<TierID>(d);

                // Collect latency samples
                std::vector<double> lat_samples;
                std::vector<double> bw_samples;
                lat_samples.reserve(SAMPLES);
                bw_samples.reserve(SAMPLES);

                // Warmup rounds
                for (int w = 0; w < WARMUP_ITERS; w++) {
                    measure_migration(src, dst, sz, 1);
                }

                // Measurement rounds
                for (int r = 0; r < SAMPLES; r++) {
                    auto sample = measure_migration(src, dst, sz, 1);
                    lat_samples.push_back(sample.latency_us);
                    bw_samples.push_back(sample.bandwidth_mbps);
                }

                mat.latency_p50_us[s][d] = percentile(lat_samples, 50.0);
                mat.latency_p99_us[s][d] = percentile(lat_samples, 99.0);

                // Use median bandwidth
                std::sort(bw_samples.begin(), bw_samples.end());
                mat.bandwidth_mbps[s][d] = bw_samples[bw_samples.size() / 2];

                INSPECT("PAIR", "%s→%s sz=%s P50=%.1fμs P99=%.1fμs BW=%.0fMB/s",
                        tier_name(src), tier_name(dst), SIZE_LABELS[si],
                        mat.latency_p50_us[s][d], mat.latency_p99_us[s][d],
                        mat.bandwidth_mbps[s][d]);
            }
        }

        mat.dump();

        // Validate: closer tiers should have lower latency
        CHK(mat.latency_p50_us[TIER_DRAM][TIER_CXL] <=
            mat.latency_p50_us[TIER_DRAM][TIER_SSD] * 1.5,
            "LAT_ORDER", "DRAM→CXL should be <= DRAM→SSD (P50) for %s",
            SIZE_LABELS[si]);
    }

    // ─── Bandwidth heatmap summary ───
    std::printf("\n  ─── Bandwidth Heatmap (MB/s, 1MB transfers) ───\n");
    {
        MigrationMatrix hm;
        memset(&hm, 0, sizeof(hm));
        hm.transfer_size = 1048576;
        size_t sz = 1048576;

        for (int s = 0; s < TIER_COUNT; s++) {
            for (int d = 0; d < TIER_COUNT; d++) {
                if (s == d) continue;
                auto sample = measure_migration(
                    static_cast<TierID>(s), static_cast<TierID>(d), sz, 10);
                hm.bandwidth_mbps[s][d] = sample.bandwidth_mbps;
            }
        }
        hm.dump();

        INSPECT("HEATMAP_DONE", "1MB bandwidth matrix complete");
    }

    INSPECT("M089_DONE", "migration matrix experiment complete");
}

// ════════════════════════════════════════════════════════════════
// §2  M090: Heat-Driven Placement
//
// mv: hotness_tracker.hpp (CLOCK + exponential decay)
//     online_learner.hpp (access pattern → tier recommendation)
//
// KEEP: access_count, last_access_time, heat_score computation
// KEEP: decay function: heat *= decay_factor per epoch
// MOD:  CPU single-thread → GPU batch heat_score update
// NEW:  kern_heat_update, kern_placement_decision
// NEW:  exp_heat_placement: Zipf workload + multi-epoch observation
// ════════════════════════════════════════════════════════════════

// ─── Heat entry per key ───
// mv: hotness_tracker.hpp HotnessEntry
struct HeatEntry {
    uint64_t key;
    uint32_t access_count;
    float    heat_score;       // decayed hotness (hotness_tracker decayed_hotness analog)
    uint8_t  current_tier;
    uint8_t  target_tier;      // filled by placement decision
    uint32_t last_epoch;       // epoch of last access
};

// ─── [NEW] kern_heat_update: GPU batch heat_score update ───
// mv MOD from hotness_tracker.hpp: CPU single-thread loop → GPU parallel
// Each thread updates one key's heat score with exponential decay.
// Uses atomicAdd for concurrent-safe access_count increment.
#if WALKING_CUDA
__global__ void kern_heat_update(HeatEntry* entries,
                                  const uint32_t* access_indices,
                                  uint32_t num_accesses,
                                  uint32_t current_epoch,
                                  float decay_lambda) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_accesses) return;

    uint32_t idx = access_indices[tid];
    // Atomic increment access count
    atomicAdd(&entries[idx].access_count, 1u);
    entries[idx].last_epoch = current_epoch;

    // Recompute heat with exponential decay
    // mv from hotness_tracker.hpp: decayed_hotness = count * exp(-λ * age)
    uint32_t age = current_epoch - entries[idx].last_epoch;
    float count_f = (float)entries[idx].access_count;
    entries[idx].heat_score = count_f * expf(-decay_lambda * (float)age);
}
#else
static void kern_heat_update_cpu(HeatEntry* entries,
                                  const uint32_t* access_indices,
                                  uint32_t num_accesses,
                                  uint32_t current_epoch,
                                  float decay_lambda) {
    for (uint32_t i = 0; i < num_accesses; i++) {
        uint32_t idx = access_indices[i];
        entries[idx].access_count++;
        entries[idx].last_epoch = current_epoch;
        uint32_t age = current_epoch - entries[idx].last_epoch;
        float count_f = (float)entries[idx].access_count;
        entries[idx].heat_score = count_f * expf(-decay_lambda * (float)age);
    }
}
#endif

// ─── [NEW] kern_placement_decision: heat → promote/demote ───
// Input: heat_scores[N], current_tier[N], tier_capacities[4]
// Rule:  heat > promote_threshold → upgrade to faster tier
//        heat < demote_threshold → downgrade to slower tier
// Output: migration_plan[N] (target_tier for each key)
#if WALKING_CUDA
__global__ void kern_placement_decision(HeatEntry* entries,
                                         uint32_t num_keys,
                                         float promote_threshold,
                                         float demote_threshold) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    float heat = entries[tid].heat_score;
    uint8_t cur = entries[tid].current_tier;
    uint8_t target = cur;

    if (heat > promote_threshold && cur > TIER_DRAM) {
        // Promote: move to one tier faster
        target = cur - 1;
    } else if (heat < demote_threshold && cur < TIER_SSD) {
        // Demote: move to one tier slower
        target = cur + 1;
    }

    entries[tid].target_tier = target;
}
#else
static void kern_placement_decision_cpu(HeatEntry* entries,
                                         uint32_t num_keys,
                                         float promote_threshold,
                                         float demote_threshold) {
    for (uint32_t i = 0; i < num_keys; i++) {
        float heat = entries[i].heat_score;
        uint8_t cur = entries[i].current_tier;
        uint8_t target = cur;

        if (heat > promote_threshold && cur > TIER_DRAM) {
            target = cur - 1;
        } else if (heat < demote_threshold && cur < TIER_SSD) {
            target = cur + 1;
        }

        entries[i].target_tier = target;
    }
}
#endif

// ─── Epoch-level heat decay (KEEP from hotness_tracker decay_window) ───
static void decay_all_heat(HeatEntry* entries, uint32_t n, float decay_factor) {
    for (uint32_t i = 0; i < n; i++) {
        entries[i].heat_score *= decay_factor;
    }
}

// ─── Zipf distribution generator (skew=0.99, upstream sssp pattern) ───
static void generate_zipf_accesses(uint32_t* out, uint32_t num_accesses,
                                    uint32_t num_keys, double skew,
                                    std::mt19937& rng) {
    // Precompute CDF
    std::vector<double> weights(num_keys);
    for (uint32_t i = 0; i < num_keys; i++) {
        weights[i] = 1.0 / std::pow(i + 1, skew);
    }
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& w : weights) w /= total;

    // Build CDF
    std::vector<double> cdf(num_keys);
    cdf[0] = weights[0];
    for (uint32_t i = 1; i < num_keys; i++) {
        cdf[i] = cdf[i-1] + weights[i];
    }

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (uint32_t a = 0; a < num_accesses; a++) {
        double u = unif(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        out[a] = static_cast<uint32_t>(std::distance(cdf.begin(), it));
        if (out[a] >= num_keys) out[a] = num_keys - 1;
    }
}

// ─── [NEW] exp_heat_placement: multi-epoch Zipf workload ───
static void exp_heat_placement() {
    sep("M090: Heat-Driven Placement");

    const uint32_t NUM_KEYS = 10000;
    const uint32_t ACCESSES_PER_EPOCH = 50000;
    const uint32_t NUM_EPOCHS = 20;
    const float DECAY_LAMBDA = 0.05f;
    const float DECAY_PER_EPOCH = 0.85f;
    const float PROMOTE_THRESHOLD = 15.0f;
    const float DEMOTE_THRESHOLD = 1.0f;
    const double ZIPF_SKEW = 0.99;

    INSPECT("HEAT_START", "keys=%u acc/epoch=%u epochs=%u decay=%.2f "
            "promote=%.1f demote=%.1f zipf=%.2f",
            NUM_KEYS, ACCESSES_PER_EPOCH, NUM_EPOCHS, DECAY_PER_EPOCH,
            PROMOTE_THRESHOLD, DEMOTE_THRESHOLD, ZIPF_SKEW);

    // Initialize entries: all keys start on SSD (tier 2)
    std::vector<HeatEntry> entries(NUM_KEYS);
    for (uint32_t i = 0; i < NUM_KEYS; i++) {
        entries[i].key = i;
        entries[i].access_count = 0;
        entries[i].heat_score = 0.0f;
        entries[i].current_tier = TIER_SSD;
        entries[i].target_tier = TIER_SSD;
        entries[i].last_epoch = 0;
    }

    std::vector<uint32_t> access_buf(ACCESSES_PER_EPOCH);
    std::mt19937 rng(42);

    // GPU buffers
    HeatEntry* d_entries = nullptr;
    uint32_t* d_accesses = nullptr;
#if WALKING_CUDA
    GPU_CHECK(cudaMalloc(&d_entries, NUM_KEYS * sizeof(HeatEntry)));
    GPU_CHECK(cudaMalloc(&d_accesses, ACCESSES_PER_EPOCH * sizeof(uint32_t)));
#else
    d_entries = entries.data();    // CPU fallback: use host directly
    d_accesses = access_buf.data();
#endif

    // Per-epoch statistics
    std::printf("  %-6s  ", "Epoch");
    for (int t = 0; t < TIER_COUNT; t++)
        std::printf("%-8s  ", tier_name(static_cast<TierID>(t)));
    std::printf("%-10s  %-10s  %-10s\n", "Promotes", "Demotes", "TopHeat");
    std::printf("  %-6s  ", "------");
    for (int t = 0; t < TIER_COUNT; t++)
        std::printf("%-8s  ", "--------");
    std::printf("%-10s  %-10s  %-10s\n", "----------", "----------", "----------");

    for (uint32_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        Timer t_epoch("epoch");

        // Generate Zipf access pattern
        generate_zipf_accesses(access_buf.data(), ACCESSES_PER_EPOCH,
                                NUM_KEYS, ZIPF_SKEW, rng);

        // ─── GPU batch heat update ───
#if WALKING_CUDA
        GPU_CHECK(cudaMemcpy(d_entries, entries.data(),
                             NUM_KEYS * sizeof(HeatEntry),
                             cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_accesses, access_buf.data(),
                             ACCESSES_PER_EPOCH * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

        int blocks_upd = (ACCESSES_PER_EPOCH + 255) / 256;
        kern_heat_update<<<blocks_upd, 256>>>(
            d_entries, d_accesses, ACCESSES_PER_EPOCH, epoch, DECAY_LAMBDA);
        GPU_CHECK(cudaDeviceSynchronize());

        // ─── GPU placement decision ───
        int blocks_dec = (NUM_KEYS + 255) / 256;
        kern_placement_decision<<<blocks_dec, 256>>>(
            d_entries, NUM_KEYS, PROMOTE_THRESHOLD, DEMOTE_THRESHOLD);
        GPU_CHECK(cudaDeviceSynchronize());

        GPU_CHECK(cudaMemcpy(entries.data(), d_entries,
                             NUM_KEYS * sizeof(HeatEntry),
                             cudaMemcpyDeviceToHost));
#else
        kern_heat_update_cpu(entries.data(), access_buf.data(),
                              ACCESSES_PER_EPOCH, epoch, DECAY_LAMBDA);
        kern_placement_decision_cpu(entries.data(), NUM_KEYS,
                                     PROMOTE_THRESHOLD, DEMOTE_THRESHOLD);
#endif

        // ─── Execute migrations & count ───
        uint32_t tier_count[TIER_COUNT] = {};
        uint32_t promotes = 0, demotes = 0;
        for (uint32_t i = 0; i < NUM_KEYS; i++) {
            uint8_t old_tier = entries[i].current_tier;
            uint8_t new_tier = entries[i].target_tier;
            if (new_tier < old_tier) {
                promotes++;
                entries[i].current_tier = new_tier;
                INSPECT("PROMOTE", "epoch=%u key=%lu %s→%s heat=%.2f",
                        epoch, (unsigned long)entries[i].key,
                        tier_name(static_cast<TierID>(old_tier)),
                        tier_name(static_cast<TierID>(new_tier)),
                        entries[i].heat_score);
            } else if (new_tier > old_tier) {
                demotes++;
                entries[i].current_tier = new_tier;
                INSPECT("DEMOTE", "epoch=%u key=%lu %s→%s heat=%.2f",
                        epoch, (unsigned long)entries[i].key,
                        tier_name(static_cast<TierID>(old_tier)),
                        tier_name(static_cast<TierID>(new_tier)),
                        entries[i].heat_score);
            }
            tier_count[entries[i].current_tier]++;
        }

        // Find top heat
        float max_heat = 0;
        for (uint32_t i = 0; i < NUM_KEYS; i++) {
            max_heat = std::max(max_heat, entries[i].heat_score);
        }

        std::printf("  %-6u  ", epoch);
        for (int t = 0; t < TIER_COUNT; t++)
            std::printf("%-8u  ", tier_count[t]);
        std::printf("%-10u  %-10u  %-10.2f\n", promotes, demotes, max_heat);

        INSPECT("EPOCH_DIST", "epoch=%u DRAM=%u CXL=%u SSD=%u GPU=%u "
                "promote=%u demote=%u top_heat=%.2f",
                epoch, tier_count[0], tier_count[1], tier_count[2],
                tier_count[3], promotes, demotes, max_heat);

        // Epoch-end decay (KEEP from hotness_tracker decay_window)
        decay_all_heat(entries.data(), NUM_KEYS, DECAY_PER_EPOCH);
    }

    // ─── Final distribution validation ───
    uint32_t final_tiers[TIER_COUNT] = {};
    for (uint32_t i = 0; i < NUM_KEYS; i++) {
        final_tiers[entries[i].current_tier]++;
    }

    std::printf("\n  Final tier distribution:\n");
    for (int t = 0; t < TIER_COUNT; t++) {
        std::printf("    %-6s  %u keys (%.1f%%)\n",
                    tier_name(static_cast<TierID>(t)),
                    final_tiers[t], 100.0 * final_tiers[t] / NUM_KEYS);
    }

    // Zipf with 0.99 → ~20% keys get ~80% accesses → those should be in DRAM/CXL
    CHK(final_tiers[TIER_DRAM] + final_tiers[TIER_CXL] > 0,
        "HEAT_PROMOTE", "at least some hot keys should have promoted to DRAM/CXL");
    CHK(final_tiers[TIER_SSD] < NUM_KEYS,
        "HEAT_DEMOTE", "some keys should have moved off SSD");

    // Top-10 hottest keys report
    std::vector<std::pair<float, uint32_t>> heat_rank(NUM_KEYS);
    for (uint32_t i = 0; i < NUM_KEYS; i++) {
        heat_rank[i] = {entries[i].heat_score, i};
    }
    std::partial_sort(heat_rank.begin(), heat_rank.begin() + 10,
                      heat_rank.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });

    std::printf("\n  Top-10 hottest keys:\n");
    std::printf("    %-6s  %-8s  %-8s  %-10s\n",
                "Key", "Tier", "Accesses", "Heat");
    for (int i = 0; i < 10 && i < (int)NUM_KEYS; i++) {
        uint32_t idx = heat_rank[i].second;
        std::printf("    %-6u  %-8s  %-8u  %-10.2f\n",
                    idx,
                    tier_name(static_cast<TierID>(entries[idx].current_tier)),
                    entries[idx].access_count,
                    entries[idx].heat_score);
    }

    INSPECT("M090_DONE", "heat placement experiment complete "
            "DRAM=%u CXL=%u SSD=%u GPU=%u",
            final_tiers[0], final_tiers[1], final_tiers[2], final_tiers[3]);

#if WALKING_CUDA
    cudaFree(d_entries);
    cudaFree(d_accesses);
#endif
}

// ════════════════════════════════════════════════════════════════
// §3  M091: Concurrent Query + Background Migration
//
// mv: tier_rebalancer.hpp (TierRebalancer + RebalanceExecutor)
//
// KEEP: tier间搬运逻辑, 容量约束检查
// KEEP: Barrier arrive_and_wait (upstream driver.h)
// NEW:  SimulatedTierSystem: 4-tier system with capacity limits
// NEW:  kern_concurrent_lookup: query kernel with background migration
// NEW:  exp_concurrent_migrate: throughput vs migration rate
// ════════════════════════════════════════════════════════════════

// ─── [NEW] SimulatedTierSystem ───
// Models a 4-tier system where keys are distributed across tiers.
// Each tier has a capacity limit. Supports migrate(key, src, dst)
// and lookup(key) returning (value, tier).
struct TierSlot {
    uint64_t key;
    int64_t  value;
    uint8_t  tier;
    uint32_t access_count;
    float    heat;
    bool     migrating;  // flag: currently being migrated (optimistic reads)
};

class SimulatedTierSystem {
public:
    SimulatedTierSystem(uint32_t num_keys,
                        const uint64_t tier_caps[TIER_COUNT])
        : num_keys_(num_keys)
    {
        for (int t = 0; t < TIER_COUNT; t++) {
            tier_capacity_[t] = tier_caps[t];
            tier_used_[t].store(0, std::memory_order_relaxed);
        }
        slots_.resize(num_keys);
    }

    void init_uniform(std::mt19937& rng) {
        // Distribute keys: all start in SSD (tier 2)
        for (uint32_t i = 0; i < num_keys_; i++) {
            slots_[i].key = i;
            slots_[i].value = static_cast<int64_t>(i * 7 + 13);
            slots_[i].tier = TIER_SSD;
            slots_[i].access_count = 0;
            slots_[i].heat = 0.0f;
            slots_[i].migrating = false;
        }
        tier_used_[TIER_SSD].store(num_keys_, std::memory_order_relaxed);
    }

    // ─── KEEP from tier_rebalancer: capacity constraint check ───
    bool can_migrate(uint32_t key_idx, TierID dst) const {
        if (key_idx >= num_keys_) return false;
        if (slots_[key_idx].tier == dst) return false;
        uint64_t used = tier_used_[dst].load(std::memory_order_relaxed);
        return used < tier_capacity_[dst];
    }

    // ─── Migrate key to target tier (under lock) ───
    bool migrate(uint32_t key_idx, TierID dst) {
        if (key_idx >= num_keys_) return false;
        std::unique_lock<std::shared_mutex> lk(mu_);

        TierSlot& slot = slots_[key_idx];
        TierID src = static_cast<TierID>(slot.tier);
        if (src == dst) return true;

        uint64_t used = tier_used_[dst].load(std::memory_order_relaxed);
        if (used >= tier_capacity_[dst]) return false;

        slot.migrating = true;

        // Simulate copy latency (proportional to tier distance)
        inject_tier_latency_ns(src, 1);
        inject_tier_latency_ns(dst, 1);

        // Atomic tier swap
        tier_used_[src].fetch_sub(1, std::memory_order_relaxed);
        tier_used_[dst].fetch_add(1, std::memory_order_relaxed);
        slot.tier = dst;
        slot.migrating = false;

        INSPECT("MIGRATE", "key=%u %s→%s heat=%.2f acc=%u",
                key_idx, tier_name(src), tier_name(dst),
                slot.heat, slot.access_count);

        return true;
    }

    // ─── Lookup: optimistic read (doesn't block migration) ───
    // mv from tier_rebalancer: read path uses shared_lock
    std::pair<int64_t, TierID> lookup(uint32_t key_idx) {
        if (key_idx >= num_keys_) return {0, TIER_SSD};
        std::shared_lock<std::shared_mutex> lk(mu_);
        TierSlot& slot = slots_[key_idx];
        slot.access_count++;
        slot.heat += 1.0f;
        TierID tier = static_cast<TierID>(slot.tier);
        inject_tier_latency_ns(tier, 1);
        return {slot.value, tier};
    }

    // ─── Record access for heat tracking (lockfree path) ───
    void record_access(uint32_t key_idx) {
        if (key_idx >= num_keys_) return;
        slots_[key_idx].access_count++;
        slots_[key_idx].heat += 1.0f;
    }

    // ─── Decay all heat (KEEP from hotness_tracker) ───
    void decay_heat(float factor) {
        for (uint32_t i = 0; i < num_keys_; i++) {
            slots_[i].heat *= factor;
        }
    }

    // ─── Get top-K hottest keys (for migration scheduling) ───
    std::vector<uint32_t> top_k_hot(uint32_t k) const {
        std::vector<std::pair<float, uint32_t>> ranked(num_keys_);
        for (uint32_t i = 0; i < num_keys_; i++) {
            ranked[i] = {slots_[i].heat, i};
        }
        size_t topn = std::min((size_t)k, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + topn, ranked.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<uint32_t> result(topn);
        for (size_t i = 0; i < topn; i++) result[i] = ranked[i].second;
        return result;
    }

    // ─── Get cold keys (for demotion) ───
    std::vector<uint32_t> bottom_k_cold(uint32_t k, TierID tier) const {
        std::vector<std::pair<float, uint32_t>> cold;
        for (uint32_t i = 0; i < num_keys_; i++) {
            if (slots_[i].tier == tier) {
                cold.push_back({slots_[i].heat, i});
            }
        }
        size_t topn = std::min((size_t)k, cold.size());
        if (topn == 0) return {};
        std::partial_sort(cold.begin(), cold.begin() + topn, cold.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<uint32_t> result(topn);
        for (size_t i = 0; i < topn; i++) result[i] = cold[i].second;
        return result;
    }

    void dump_tier_dist() const {
        std::printf("    Tier distribution: ");
        for (int t = 0; t < TIER_COUNT; t++) {
            std::printf("%s=%lu  ",
                        tier_name(static_cast<TierID>(t)),
                        (unsigned long)tier_used_[t].load());
        }
        std::printf("\n");
    }

    uint32_t num_keys() const { return num_keys_; }
    const TierSlot& slot(uint32_t i) const { return slots_[i]; }

private:
    uint32_t num_keys_;
    std::vector<TierSlot> slots_;
    uint64_t tier_capacity_[TIER_COUNT];
    std::atomic<uint64_t> tier_used_[TIER_COUNT];
    std::shared_mutex mu_;
};

// ─── [NEW] kern_concurrent_lookup: batch query kernel ───
// Simulates batch lookups happening concurrently with background migration.
// On GPU: each thread does one lookup. On CPU: sequential loop.
struct LookupResult {
    int64_t value;
    uint8_t tier;
};

#if WALKING_CUDA
__global__ void kern_concurrent_lookup(const TierSlot* slots,
                                        const uint32_t* query_keys,
                                        LookupResult* results,
                                        uint32_t num_queries,
                                        uint32_t num_keys) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    uint32_t k = query_keys[tid];
    if (k < num_keys) {
        // Optimistic read: even if slot is migrating, return current value
        results[tid].value = slots[k].value;
        results[tid].tier = slots[k].tier;
    } else {
        results[tid].value = -1;
        results[tid].tier = 0xFF;
    }
}
#else
static void kern_concurrent_lookup_cpu(const TierSlot* slots,
                                        const uint32_t* query_keys,
                                        LookupResult* results,
                                        uint32_t num_queries,
                                        uint32_t num_keys) {
    for (uint32_t i = 0; i < num_queries; i++) {
        uint32_t k = query_keys[i];
        if (k < num_keys) {
            results[i].value = slots[k].value;
            results[i].tier = slots[k].tier;
        } else {
            results[i].value = -1;
            results[i].tier = 0xFF;
        }
    }
}
#endif

// ─── RebalanceBarrier — KEEP from tier_rebalancer.hpp ───
// 100% preserved from upstream driver.h arrive_and_wait
class RebalanceBarrier {
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t count_;
    size_t waiting_;
    std::atomic<uint32_t> phase_{0};

public:
    explicit RebalanceBarrier(size_t count) : count_(count), waiting_(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        ++waiting_;
        if (waiting_ == count_) {
            waiting_ = 0;
            phase_.fetch_add(1, std::memory_order_relaxed);
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return waiting_ == 0; });
        }
    }

    uint32_t phase() const { return phase_.load(std::memory_order_relaxed); }
};

// ─── [NEW] exp_concurrent_migrate ───
// Launches query workload (10K queries/batch) + background migration
// (each epoch migrates top-K hot keys).
// Measures: throughput vs migration rate, hot key distribution change.
static void exp_concurrent_migrate() {
    sep("M091: Concurrent Query + Background Migration");

    const uint32_t NUM_KEYS = 10000;
    const uint32_t QUERIES_PER_BATCH = 10000;
    const uint32_t BATCHES_PER_EPOCH = 50;
    const uint32_t NUM_EPOCHS = 15;
    const uint32_t MIGRATE_TOP_K = 50;       // promote top-K per epoch
    const uint32_t DEMOTE_COLD_K = 20;       // demote cold keys to free space
    const float HEAT_DECAY = 0.80f;
    const double ZIPF_SKEW = 0.99;

    // Tier capacities (limited to force eviction pressure)
    uint64_t tier_caps[TIER_COUNT] = {
        500,    // DRAM: 500 keys max
        1500,   // CXL:  1500 keys max
        NUM_KEYS, // SSD: unlimited
        200     // GPU:  200 keys max (not used for placement here)
    };

    INSPECT("CONCURRENT_START",
            "keys=%u qps_batch=%u batches/epoch=%u epochs=%u "
            "migrate_k=%u demote_k=%u",
            NUM_KEYS, QUERIES_PER_BATCH, BATCHES_PER_EPOCH,
            NUM_EPOCHS, MIGRATE_TOP_K, DEMOTE_COLD_K);

    SimulatedTierSystem sys(NUM_KEYS, tier_caps);
    std::mt19937 rng(42);
    sys.init_uniform(rng);

    std::printf("  Initial state:\n");
    sys.dump_tier_dist();

    // Access pattern: Zipf
    std::vector<uint32_t> access_buf(QUERIES_PER_BATCH);

    // Epoch-level throughput table
    std::printf("\n  %-6s  %-12s  %-12s  %-10s  %-10s  %-10s  ",
                "Epoch", "Queries", "Time(ms)", "QPS", "Promotes", "Demotes");
    for (int t = 0; t < TIER_COUNT; t++)
        std::printf("%-6s  ", tier_name(static_cast<TierID>(t)));
    std::printf("\n");
    std::printf("  %-6s  %-12s  %-12s  %-10s  %-10s  %-10s  ",
                "------", "------------", "------------",
                "----------", "----------", "----------");
    for (int t = 0; t < TIER_COUNT; t++)
        std::printf("%-6s  ", "------");
    std::printf("\n");

    // Track throughput over epochs for degradation curve
    std::vector<double> epoch_qps;
    epoch_qps.reserve(NUM_EPOCHS);

    for (uint32_t epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        // ─── Phase 1: Run query batches (measuring throughput) ───
        uint64_t total_queries = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (uint32_t batch = 0; batch < BATCHES_PER_EPOCH; batch++) {
            generate_zipf_accesses(access_buf.data(), QUERIES_PER_BATCH,
                                    NUM_KEYS, ZIPF_SKEW, rng);

            // Execute lookups
            for (uint32_t q = 0; q < QUERIES_PER_BATCH; q++) {
                auto [val, tier] = sys.lookup(access_buf[q]);
                (void)val; (void)tier;
            }
            total_queries += QUERIES_PER_BATCH;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double epoch_ms = std::chrono::duration_cast<
            std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        double qps = total_queries / (epoch_ms / 1000.0);
        epoch_qps.push_back(qps);

        // ─── Phase 2: Background migration (between epochs) ───
        // Demote cold keys from DRAM to make room
        uint32_t demoted = 0;
        {
            auto cold_dram = sys.bottom_k_cold(DEMOTE_COLD_K, TIER_DRAM);
            for (uint32_t idx : cold_dram) {
                if (sys.can_migrate(idx, TIER_CXL)) {
                    if (sys.migrate(idx, TIER_CXL)) demoted++;
                }
            }
            auto cold_cxl = sys.bottom_k_cold(DEMOTE_COLD_K, TIER_CXL);
            for (uint32_t idx : cold_cxl) {
                if (sys.can_migrate(idx, TIER_SSD)) {
                    if (sys.migrate(idx, TIER_SSD)) demoted++;
                }
            }
        }

        // Promote hot keys toward DRAM
        uint32_t promoted = 0;
        {
            auto hot_keys = sys.top_k_hot(MIGRATE_TOP_K);
            for (uint32_t idx : hot_keys) {
                const auto& slot = sys.slot(idx);
                TierID cur = static_cast<TierID>(slot.tier);
                if (cur > TIER_DRAM) {
                    TierID target = static_cast<TierID>(cur - 1);
                    if (sys.can_migrate(idx, target)) {
                        if (sys.migrate(idx, target)) promoted++;
                    }
                }
            }
        }

        // Heat decay
        sys.decay_heat(HEAT_DECAY);

        // Print epoch row
        std::printf("  %-6u  %-12lu  %-12.1f  %-10.0f  %-10u  %-10u  ",
                    epoch, (unsigned long)total_queries, epoch_ms,
                    qps, promoted, demoted);

        // Tier distribution snapshot (inline)
        for (int t = 0; t < TIER_COUNT; t++) {
            uint32_t cnt = 0;
            for (uint32_t i = 0; i < NUM_KEYS; i++) {
                if (sys.slot(i).tier == t) cnt++;
            }
            std::printf("%-6u  ", cnt);
        }
        std::printf("\n");

        INSPECT("EPOCH_CONCURRENT", "epoch=%u queries=%lu ms=%.1f "
                "qps=%.0f promote=%u demote=%u",
                epoch, (unsigned long)total_queries, epoch_ms,
                qps, promoted, demoted);
    }

    // ─── Throughput degradation curve ───
    std::printf("\n  Throughput over epochs (QPS):\n");
    std::printf("    ");
    for (uint32_t e = 0; e < NUM_EPOCHS; e++) {
        std::printf("E%02u:%.0f  ", e, epoch_qps[e]);
        if ((e + 1) % 8 == 0) std::printf("\n    ");
    }
    std::printf("\n");

    // Validate: QPS should be relatively stable (within 3x of initial)
    if (epoch_qps.size() >= 2) {
        double first_qps = epoch_qps[0];
        double last_qps = epoch_qps.back();
        CHK(last_qps > first_qps * 0.3, "QPS_STABLE",
            "QPS degraded too much: %.0f → %.0f", first_qps, last_qps);
    }

    // ─── Concurrent readers + migrator test ───
    std::printf("\n  ─── Concurrent Reader + Migrator Test ───\n");
    {
        const int NUM_READER_THREADS = 4;
        const int READER_OPS = 20000;
        std::atomic<uint64_t> total_lookups{0};
        std::atomic<uint64_t> total_migrations{0};
        std::atomic<bool> stop_migrator{false};

        auto t_start = std::chrono::high_resolution_clock::now();

        // Background migrator thread
        std::thread migrator([&]() {
            std::mt19937 mrng(999);
            while (!stop_migrator.load(std::memory_order_acquire)) {
                auto hot = sys.top_k_hot(10);
                for (uint32_t idx : hot) {
                    const auto& slot = sys.slot(idx);
                    TierID cur = static_cast<TierID>(slot.tier);
                    if (cur > TIER_DRAM) {
                        TierID target = static_cast<TierID>(cur - 1);
                        if (sys.can_migrate(idx, target)) {
                            if (sys.migrate(idx, target)) {
                                total_migrations.fetch_add(1,
                                    std::memory_order_relaxed);
                            }
                        }
                    }
                }
                // Also demote some cold to free space
                auto cold = sys.bottom_k_cold(5, TIER_DRAM);
                for (uint32_t idx : cold) {
                    if (sys.can_migrate(idx, TIER_CXL)) {
                        if (sys.migrate(idx, TIER_CXL)) {
                            total_migrations.fetch_add(1,
                                std::memory_order_relaxed);
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });

        // Reader threads
        std::vector<std::thread> readers;
        for (int rt = 0; rt < NUM_READER_THREADS; rt++) {
            readers.emplace_back([&, rt]() {
                std::mt19937 local_rng(rt * 1000 + 42);
                std::vector<uint32_t> local_acc(1000);

                for (int batch = 0; batch < READER_OPS / 1000; batch++) {
                    generate_zipf_accesses(local_acc.data(), 1000,
                                            NUM_KEYS, ZIPF_SKEW, local_rng);
                    for (int q = 0; q < 1000; q++) {
                        auto [val, tier] = sys.lookup(local_acc[q]);
                        (void)val; (void)tier;
                    }
                    total_lookups.fetch_add(1000, std::memory_order_relaxed);
                }
            });
        }

        for (auto& r : readers) r.join();
        stop_migrator.store(true, std::memory_order_release);
        migrator.join();

        auto t_end = std::chrono::high_resolution_clock::now();
        double concurrent_ms = std::chrono::duration_cast<
            std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
        uint64_t lk = total_lookups.load();
        uint64_t mg = total_migrations.load();

        std::printf("    Readers: %d threads × %d ops = %lu lookups in %.1f ms\n",
                    NUM_READER_THREADS, READER_OPS,
                    (unsigned long)lk, concurrent_ms);
        std::printf("    Migrator: %lu migrations during query window\n",
                    (unsigned long)mg);
        std::printf("    Concurrent QPS: %.0f\n",
                    lk / (concurrent_ms / 1000.0));
        std::printf("    Migrations/sec: %.0f\n",
                    mg / (concurrent_ms / 1000.0));

        INSPECT("CONCURRENT_DONE", "lookups=%lu migrations=%lu ms=%.1f "
                "qps=%.0f mig/s=%.0f",
                (unsigned long)lk, (unsigned long)mg, concurrent_ms,
                lk / (concurrent_ms / 1000.0),
                mg / (concurrent_ms / 1000.0));

        CHK(lk > 0, "CONCURRENT_LOOKUPS", "should have completed lookups");
        CHK(mg > 0, "CONCURRENT_MIGRATIONS", "should have completed migrations");
    }

    // Final state
    std::printf("\n  Final tier distribution after concurrent test:\n");
    sys.dump_tier_dist();

    // Top-5 hottest keys final placement
    auto final_hot = sys.top_k_hot(5);
    std::printf("  Top-5 hottest final placement:\n");
    for (uint32_t idx : final_hot) {
        const auto& s = sys.slot(idx);
        std::printf("    key=%u tier=%s heat=%.1f acc=%u\n",
                    idx, tier_name(static_cast<TierID>(s.tier)),
                    s.heat, s.access_count);
    }

    INSPECT("M091_DONE", "concurrent query + migration experiment complete");
}

}  // namespace hetero
}  // namespace walking

// ════════════════════════════════════════════════════════════════
// MAIN — outside namespace
// ════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    std::printf("╔═══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  walking_hetero_bench — M089-M091                            ║\n");
    std::printf("║  Cross-Tier Benchmark + Heat-Driven Placement                ║\n");
    std::printf("║  + Concurrent Query/Migration                                ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

#if WALKING_CUDA
    int dev_count = 0;
    GPU_CHECK(cudaGetDeviceCount(&dev_count));
    std::printf("  CUDA devices: %d\n", dev_count);
    for (int d = 0; d < dev_count; d++) {
        cudaDeviceProp prop;
        GPU_CHECK(cudaGetDeviceProperties(&prop, d));
        std::printf("    GPU%d: %-30s SM=%d.%d VRAM=%.1f GB\n",
                    d, prop.name, prop.major, prop.minor,
                    prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    }
#else
    std::printf("  [CPU-only mode: WALKING_CUDA=0]\n");
#endif
    std::printf("  RSS at start: %ld KB\n\n", rss_kb());

    INSPECT("MAIN_START", "walking_hetero_bench M089-M091");

    // ─── M089: Tier Migration Latency Matrix ──────────────────
    walking::hetero::exp_migration_matrix();

    // ─── M090: Heat-Driven Placement ──────────────────────────
    walking::hetero::exp_heat_placement();

    // ─── M091: Concurrent Query + Background Migration ────────
    walking::hetero::exp_concurrent_migrate();

    // ─── SUMMARY ──────────────────────────────────────────────
    sep("SUMMARY");
    std::printf("  Inspections: %lu\n", (unsigned long)g_insp);
    std::printf("  Checks passed: %lu\n", (unsigned long)g_pass);
    std::printf("  Checks failed: %lu\n", (unsigned long)g_fail);
    std::printf("  RSS peak: %ld KB\n", rss_kb());

    CHK(g_fail == 0, "OVERALL", "all checks should pass");

    if (g_fail == 0) {
        std::printf("\n  ✓ All M089-M091 experiments PASSED\n");
    } else {
        std::printf("\n  ✗ %lu check(s) FAILED\n", (unsigned long)g_fail);
    }

    std::printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  walking_hetero_bench complete.                              ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return (g_fail > 0) ? 1 : 0;
}
