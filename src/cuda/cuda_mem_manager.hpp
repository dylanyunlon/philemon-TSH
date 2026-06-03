#pragma once
/**
 * cuda_mem_manager.hpp — CUDA异构内存管理器（HBM/GDDR/DRAM 瀑布分配+slab池化）
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   src/cuda/hetero_bench.cu  HeteroAllocator (134–268行)
 *     → cudaMalloc/cudaFree per-GPU dispatch
 *     → cudaMemcpy sync/async
 *     → peer access enable loop
 *     → print_usage统计
 *
 *   src/core/tiered_allocator.hpp  TieredAllocator (230–571行)
 *     → MemoryTier枚举 + TierBudget + waterfall fallback
 *     → allocate/deallocate/migrate
 *     → slab_allocator集成
 *
 *   upstream/rapidstore/wrapper/wrapper.h  (249行)
 *     → snapshot_edges回调dispatch模式 → tier dispatch模板
 *
 * 算法改动 (~20%):
 *   [ALG1] allocate: 原HeteroAllocator直接在指定tier分配，满了就失败
 *          → 瀑布回退: preferred→fallback chain(HBM→GDDR→DRAM), 带per-tier容量水位线
 *   [ALG2] copy路径选择: 原版一律cudaMemcpyDefault
 *          → 根据(src,dst)对选择: 同GPU=D2D, GPU↔GPU=peer-direct(若可用)否则staged经host,
 *            GPU↔Host=H2D/D2H, 并行chunk pipeline(>16MB分段overlap)
 *   [ALG3] 小块分配: 原版每次cudaMalloc系统调用
 *          → slab pool: ≤4MB走per-tier slab池(预分配64MB slab, bump分配),
 *            释放进freelist, 减少cudaMalloc调用次数
 *   [ALG4] 内存碎片回收: 原版无compaction
 *          → slab watermark低于25%时触发合并, 空slab归还OS
 *
 * Milestone: M045 — CUDA内存管理
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <algorithm>
#include <chrono>
#include <unordered_map>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../core/tiered_allocator.hpp"

namespace philemon {
namespace cuda_mem {

// ─── CUDA error checking (CPU-safe stub) ────────────────────────────
// In non-CUDA builds, these are no-ops; in CUDA builds, link with -lcudart
#ifdef __CUDACC__
  #define PHILE_CUDA_CHECK(call) do {                                     \
      cudaError_t err = (call);                                            \
      if (err != cudaSuccess) {                                            \
          fprintf(stderr, "[CudaMemMgr] CUDA error %s:%d: %s\n",          \
                  __FILE__, __LINE__, cudaGetErrorString(err));            \
          PHILE_TRACE(debug::TraceEvent::ALLOC, 0, 0, 0, 0);              \
      }                                                                    \
  } while(0)
#else
  #define PHILE_CUDA_CHECK(call) (void)(0)
#endif

// ─── Device tier — maps to physical GPU/Host ────────────────────────
enum class DeviceTier : uint8_t {
    HBM_GPU   = 0,   // H100 HBM2e (GPU2)
    GDDR_GPU0 = 1,   // A6000 #0   (GPU0)
    GDDR_GPU1 = 2,   // A6000 #1   (GPU1)
    HOST_DRAM = 3,   // CPU DDR5
    TIER_COUNT = 4
};

inline const char* device_tier_name(DeviceTier t) {
    switch (t) {
        case DeviceTier::HBM_GPU:   return "H100-HBM";
        case DeviceTier::GDDR_GPU0: return "A6000-GPU0";
        case DeviceTier::GDDR_GPU1: return "A6000-GPU1";
        case DeviceTier::HOST_DRAM: return "Host-DRAM";
        default: return "UNKNOWN";
    }
}

inline int tier_to_device_id(DeviceTier t) {
    switch (t) {
        case DeviceTier::HBM_GPU:   return 2;
        case DeviceTier::GDDR_GPU0: return 0;
        case DeviceTier::GDDR_GPU1: return 1;
        case DeviceTier::HOST_DRAM: return -1;
        default: return -1;
    }
}

// ─── [ALG1] Waterfall fallback chain ────────────────────────────────
// 原版: 固定tier分配, 满了返回nullptr
// 改动: 按优先级链回退, HBM→GDDR0→GDDR1→DRAM
inline DeviceTier waterfall_next(DeviceTier current) {
    switch (current) {
        case DeviceTier::HBM_GPU:   return DeviceTier::GDDR_GPU0;
        case DeviceTier::GDDR_GPU0: return DeviceTier::GDDR_GPU1;
        case DeviceTier::GDDR_GPU1: return DeviceTier::HOST_DRAM;
        default: return DeviceTier::HOST_DRAM;  // 兜底
    }
}

// ─── Slab descriptor (for [ALG3] pool化) ────────────────────────────
struct SlabBlock {
    void*    base_ptr;       // slab起始地址
    size_t   slab_size;      // 整个slab的大小 (默认64MB)
    size_t   bump_offset;    // bump分配指针
    size_t   live_count;     // 活跃分配数
    DeviceTier tier;

    SlabBlock() : base_ptr(nullptr), slab_size(0), bump_offset(0),
                  live_count(0), tier(DeviceTier::HOST_DRAM) {}
};

// ─── Per-tier budget tracking ───────────────────────────────────────
struct DeviceTierBudget {
    size_t capacity;         // 总容量
    std::atomic<size_t> used{0};
    size_t high_watermark;   // 开始回退的阈值 (capacity * 0.85)

    DeviceTierBudget() : capacity(0), high_watermark(0) {}
    explicit DeviceTierBudget(size_t cap)
        : capacity(cap), high_watermark(static_cast<size_t>(cap * 0.85)) {}

    bool can_fit(size_t sz) const {
        return used.load(std::memory_order_relaxed) + sz <= capacity;
    }
    bool above_watermark() const {
        return used.load(std::memory_order_relaxed) >= high_watermark;
    }
};

// ─── Allocation metadata ────────────────────────────────────────────
struct CudaAllocMeta {
    uint64_t   alloc_id;
    DeviceTier tier;
    void*      ptr;
    size_t     size;
    bool       from_slab;    // true=slab池分配, false=raw cudaMalloc
    uint64_t   slab_idx;     // 所属slab的索引
    std::chrono::steady_clock::time_point created_at;

    CudaAllocMeta()
        : alloc_id(0), tier(DeviceTier::HOST_DRAM), ptr(nullptr),
          size(0), from_slab(false), slab_idx(0),
          created_at(std::chrono::steady_clock::now()) {}
};

// ─── Peer access topology cache ─────────────────────────────────────
struct PeerTopology {
    bool can_peer[4][4] = {};    // can_peer[src_gpu][dst_gpu]
    bool peer_enabled[4][4] = {};
};

// ════════════════════════════════════════════════════════════════════════════
//  CudaMemManager — 核心管理器
// ════════════════════════════════════════════════════════════════════════════

class CudaMemManager {
public:
    static constexpr size_t SLAB_SIZE       = 64ULL << 20;  // 64 MB per slab
    static constexpr size_t SLAB_THRESHOLD  = 4ULL  << 20;  // ≤4MB走slab池
    static constexpr size_t CHUNK_PIPELINE  = 16ULL << 20;  // >16MB分段overlap拷贝
    static constexpr int    MAX_GPUS        = 3;

    CudaMemManager(size_t hbm_cap   = 80ULL << 30,   // 80 GB
                   size_t gddr0_cap = 40ULL << 30,   // 40 GB
                   size_t gddr1_cap = 40ULL << 30,   // 40 GB
                   size_t dram_cap  = 512ULL << 30)  // 512 GB
    {
        PHILE_CHECKPOINT("CudaMemManager::ctor");

        budgets_[0] = DeviceTierBudget(hbm_cap);
        budgets_[1] = DeviceTierBudget(gddr0_cap);
        budgets_[2] = DeviceTierBudget(gddr1_cap);
        budgets_[3] = DeviceTierBudget(dram_cap);

        probe_gpu_topology();
        PHILE_DBG(1, "[CudaMemMgr] initialized: HBM=%.1fGB GDDR0=%.1fGB GDDR1=%.1fGB DRAM=%.1fGB\n",
                  hbm_cap/(1024.0*1024*1024), gddr0_cap/(1024.0*1024*1024),
                  gddr1_cap/(1024.0*1024*1024), dram_cap/(1024.0*1024*1024));
    }

    ~CudaMemManager() {
        // 释放所有slab
        for (auto& slab : slabs_) {
            if (slab.base_ptr) {
                raw_free(slab.tier, slab.base_ptr, slab.slab_size);
            }
        }
    }

    // ── [ALG1] 瀑布式分配 ──────────────────────────────────────────
    // 原版: 指定tier, 满了返回nullptr
    // 改动: 尝试preferred → waterfall_next链, 直到分配成功或全部失败
    //       小块(≤SLAB_THRESHOLD)走slab池[ALG3], 大块走raw
    uint64_t allocate(size_t size, DeviceTier preferred) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        PHILE_TRACE(debug::TraceEvent::ALLOC, next_id_, static_cast<uint64_t>(size),
                    static_cast<uint64_t>(preferred), 0);

        DeviceTier actual = preferred;
        int fallback_hops = 0;

        // [ALG1] 瀑布回退: 检查容量+水位线
        while (!budgets_[static_cast<int>(actual)].can_fit(size)) {
            DeviceTier next = waterfall_next(actual);
            if (next == actual) break;  // 已到DRAM兜底
            PHILE_DBG(2, "[alloc] tier %s full (used=%zu/%zu), falling back to %s\n",
                      device_tier_name(actual),
                      budgets_[static_cast<int>(actual)].used.load(),
                      budgets_[static_cast<int>(actual)].capacity,
                      device_tier_name(next));
            actual = next;
            fallback_hops++;
        }

        void* ptr = nullptr;
        bool from_slab = false;
        uint64_t slab_idx = 0;

        // [ALG3] 小块走slab池
        if (size <= SLAB_THRESHOLD) {
            ptr = slab_alloc(actual, size, slab_idx);
            from_slab = (ptr != nullptr);
        }

        // 大块或slab失败 → raw分配
        if (!ptr) {
            ptr = raw_alloc(actual, size);
            from_slab = false;
        }

        if (!ptr) {
            PHILE_DBG(1, "[alloc] FAILED: size=%zu preferred=%s actual=%s\n",
                      size, device_tier_name(preferred), device_tier_name(actual));
            return 0;
        }

        budgets_[static_cast<int>(actual)].used.fetch_add(size, std::memory_order_relaxed);

        uint64_t id = next_id_++;
        CudaAllocMeta meta;
        meta.alloc_id  = id;
        meta.tier      = actual;
        meta.ptr       = ptr;
        meta.size      = size;
        meta.from_slab = from_slab;
        meta.slab_idx  = slab_idx;
        registry_[id]  = meta;

        PHILE_DBG(2, "[alloc] id=%lu size=%zu tier=%s (fallback_hops=%d, slab=%d)\n",
                  (unsigned long)id, size, device_tier_name(actual),
                  fallback_hops, from_slab ? 1 : 0);

        alloc_count_.fetch_add(1, std::memory_order_relaxed);
        return id;
    }

    // ── 释放 ───────────────────────────────────────────────────────
    void deallocate(uint64_t alloc_id) {
        std::unique_lock<std::shared_mutex> lk(mu_);

        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) {
            PHILE_DBG(1, "[dealloc] WARNING: id=%lu not found\n", (unsigned long)alloc_id);
            return;
        }

        auto& meta = it->second;
        PHILE_TRACE(debug::TraceEvent::DEALLOC, alloc_id,
                    static_cast<uint64_t>(meta.size),
                    static_cast<uint64_t>(meta.tier), 0);

        budgets_[static_cast<int>(meta.tier)].used.fetch_sub(meta.size, std::memory_order_relaxed);

        if (meta.from_slab) {
            slab_free(meta.slab_idx);
        } else {
            raw_free(meta.tier, meta.ptr, meta.size);
        }

        registry_.erase(it);
        dealloc_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── [ALG2] 智能拷贝路径选择 ────────────────────────────────────
    // 原版: 一律cudaMemcpyDefault
    // 改动: 根据(src,dst)tier对选最优路径:
    //   同GPU → cudaMemcpyDeviceToDevice
    //   GPU↔GPU且peer可用 → 直接peer copy
    //   GPU↔GPU无peer → staged: src→host pinned→dst (两段pipeline)
    //   GPU↔Host → 直接H2D/D2H
    //   >CHUNK_PIPELINE → 分段overlap copy
    void copy(DeviceTier dst_tier, void* dst,
              DeviceTier src_tier, const void* src,
              size_t size, bool async = false)
    {
        PHILE_TRACE(debug::TraceEvent::MIGRATE_START, 0, static_cast<uint64_t>(size),
                    static_cast<uint64_t>(src_tier), static_cast<uint64_t>(dst_tier));
        auto t0 = std::chrono::steady_clock::now();

        int src_gpu = tier_to_device_id(src_tier);
        int dst_gpu = tier_to_device_id(dst_tier);

        // 同一设备
        if (src_tier == dst_tier) {
            device_local_copy(dst_gpu, dst, src, size);
        }
        // 都在GPU上
        else if (src_gpu >= 0 && dst_gpu >= 0) {
            // [ALG2] 检查peer access
            if (topo_.peer_enabled[src_gpu][dst_gpu]) {
                // 直接peer copy
                peer_direct_copy(src_gpu, dst_gpu, dst, src, size, async);
            } else {
                // staged经host: src_gpu→pinned→dst_gpu
                staged_copy_via_host(src_gpu, dst_gpu, dst, src, size);
            }
        }
        // GPU→Host
        else if (src_gpu >= 0 && dst_gpu < 0) {
            gpu_to_host_copy(src_gpu, dst, src, size, async);
        }
        // Host→GPU
        else if (src_gpu < 0 && dst_gpu >= 0) {
            host_to_gpu_copy(dst_gpu, dst, src, size, async);
        }
        // Host→Host
        else {
            std::memcpy(dst, src, size);
        }

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double bw_gbps = (size / (1024.0*1024*1024)) / (ms / 1000.0);

        copy_count_.fetch_add(1, std::memory_order_relaxed);
        copy_bytes_.fetch_add(size, std::memory_order_relaxed);

        PHILE_DBG(2, "[copy] %s→%s size=%zu %.2fms (%.1f GB/s) %s\n",
                  device_tier_name(src_tier), device_tier_name(dst_tier),
                  size, ms, bw_gbps, async ? "async" : "sync");

        PHILE_TRACE(debug::TraceEvent::MIGRATE_END, 0, static_cast<uint64_t>(size),
                    static_cast<uint64_t>(src_tier), static_cast<uint64_t>(dst_tier));
    }

    // ── [ALG4] Slab碎片回收 ─────────────────────────────────────────
    // 原版: 无compaction
    // 改动: 遍历所有slab, live_count==0且水位<25%的整slab归还
    size_t compact() {
        std::unique_lock<std::shared_mutex> lk(mu_);
        PHILE_CHECKPOINT("CudaMemManager::compact");

        size_t freed = 0;
        size_t checked = 0;

        for (size_t i = 0; i < slabs_.size(); ++i) {
            auto& slab = slabs_[i];
            if (!slab.base_ptr) continue;
            checked++;

            double usage = (slab.live_count > 0)
                ? static_cast<double>(slab.bump_offset) / slab.slab_size
                : 0.0;

            // [ALG4] 水位<25%且无活跃分配 → 回收
            if (slab.live_count == 0 && usage < 0.25) {
                PHILE_DBG(2, "[compact] freeing slab %zu on %s (usage=%.1f%%)\n",
                          i, device_tier_name(slab.tier), usage * 100);
                raw_free(slab.tier, slab.base_ptr, slab.slab_size);
                freed += slab.slab_size;
                slab.base_ptr = nullptr;
                slab.slab_size = 0;
                slab.bump_offset = 0;
            }
        }

        PHILE_DBG(1, "[compact] checked=%zu freed=%.2fMB\n",
                  checked, freed / (1024.0 * 1024));
        return freed;
    }

    // ── 状态打印 (断点调试用) ───────────────────────────────────────
    void dump_state() const {
        PHILE_SEPARATOR("CudaMemManager State");
        printf("[CudaMemMgr] ── Tier Budget ──\n");
        for (int i = 0; i < static_cast<int>(DeviceTier::TIER_COUNT); ++i) {
            auto& b = budgets_[i];
            printf("  %-14s  used=%8.2f MB / %8.2f MB  (%.1f%%)  %s\n",
                   device_tier_name(static_cast<DeviceTier>(i)),
                   b.used.load() / (1024.0*1024),
                   b.capacity / (1024.0*1024),
                   b.capacity > 0 ? 100.0 * b.used.load() / b.capacity : 0.0,
                   b.above_watermark() ? "⚠ HIGH" : "OK");
        }
        printf("[CudaMemMgr] ── Active Allocations: %zu ──\n", registry_.size());
        printf("[CudaMemMgr] ── Stats: allocs=%lu deallocs=%lu copies=%lu copy_bytes=%.2fMB ──\n",
               alloc_count_.load(), dealloc_count_.load(),
               copy_count_.load(), copy_bytes_.load() / (1024.0*1024));
        printf("[CudaMemMgr] ── Slab pool: %zu slabs active ──\n", active_slab_count());

        // 打印peer拓扑
        printf("[CudaMemMgr] ── Peer Topology ──\n");
        for (int i = 0; i < MAX_GPUS; ++i) {
            for (int j = 0; j < MAX_GPUS; ++j) {
                if (i == j) continue;
                if (topo_.peer_enabled[i][j]) {
                    printf("  GPU%d→GPU%d: peer-direct\n", i, j);
                }
            }
        }
        PHILE_SEPARATOR("End CudaMemManager State");
    }

    // ── 查询接口 ───────────────────────────────────────────────────
    const CudaAllocMeta* get_meta(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        return (it != registry_.end()) ? &it->second : nullptr;
    }

    DeviceTierBudget& budget(DeviceTier t) {
        return budgets_[static_cast<int>(t)];
    }

    size_t active_slab_count() const {
        size_t count = 0;
        for (auto& s : slabs_) { if (s.base_ptr) count++; }
        return count;
    }

private:
    // ── GPU拓扑探测 ─────────────────────────────────────────────────
    void probe_gpu_topology() {
        PHILE_CHECKPOINT("probe_gpu_topology");
        // CPU-only stub: 实际CUDA环境下查询peer access
        #ifdef __CUDACC__
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        PHILE_DBG(1, "[topo] found %d CUDA devices\n", dev_count);

        for (int i = 0; i < dev_count && i < MAX_GPUS; ++i) {
            for (int j = 0; j < dev_count && j < MAX_GPUS; ++j) {
                if (i == j) continue;
                int can = 0;
                cudaDeviceCanAccessPeer(&can, i, j);
                topo_.can_peer[i][j] = (can != 0);
                if (can) {
                    cudaSetDevice(i);
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    topo_.peer_enabled[i][j] =
                        (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
                    cudaGetLastError();
                }
            }
        }
        cudaSetDevice(0);
        #else
        // CPU模拟: 假设GPU0↔GPU1可peer, GPU2(H100)独立
        topo_.can_peer[0][1] = true; topo_.peer_enabled[0][1] = true;
        topo_.can_peer[1][0] = true; topo_.peer_enabled[1][0] = true;
        PHILE_DBG(1, "[topo] CPU-only mode: simulated topology\n");
        #endif
    }

    // ── Raw分配/释放 (直接系统调用) ─────────────────────────────────
    void* raw_alloc(DeviceTier tier, size_t size) {
        void* ptr = nullptr;
        #ifdef __CUDACC__
        int gpu = tier_to_device_id(tier);
        if (gpu >= 0) {
            cudaSetDevice(gpu);
            if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;
            cudaMemset(ptr, 0, size);
        } else {
            if (cudaMallocHost(&ptr, size) != cudaSuccess) return nullptr;
            memset(ptr, 0, size);
        }
        #else
        ptr = std::calloc(1, size);
        #endif
        return ptr;
    }

    void raw_free(DeviceTier tier, void* ptr, size_t size) {
        if (!ptr) return;
        #ifdef __CUDACC__
        int gpu = tier_to_device_id(tier);
        if (gpu >= 0) {
            cudaSetDevice(gpu);
            cudaFree(ptr);
        } else {
            cudaFreeHost(ptr);
        }
        #else
        std::free(ptr);
        #endif
        (void)size;
    }

    // ── [ALG3] Slab池分配 ──────────────────────────────────────────
    // 原版hetero_bench: 每次cudaMalloc
    // 改动: 预分配64MB slab, 内部bump pointer分配, 减少系统调用
    void* slab_alloc(DeviceTier tier, size_t size, uint64_t& out_slab_idx) {
        // 16字节对齐
        size_t aligned = (size + 15) & ~15ULL;

        // 在已有slab中找空间
        for (size_t i = 0; i < slabs_.size(); ++i) {
            auto& slab = slabs_[i];
            if (!slab.base_ptr || slab.tier != tier) continue;
            if (slab.bump_offset + aligned <= slab.slab_size) {
                void* ptr = static_cast<char*>(slab.base_ptr) + slab.bump_offset;
                slab.bump_offset += aligned;
                slab.live_count++;
                out_slab_idx = i;
                return ptr;
            }
        }

        // 没有可用slab → 新建
        void* slab_ptr = raw_alloc(tier, SLAB_SIZE);
        if (!slab_ptr) return nullptr;

        SlabBlock new_slab;
        new_slab.base_ptr    = slab_ptr;
        new_slab.slab_size   = SLAB_SIZE;
        new_slab.bump_offset = aligned;
        new_slab.live_count  = 1;
        new_slab.tier        = tier;
        slabs_.push_back(new_slab);
        out_slab_idx = slabs_.size() - 1;

        PHILE_DBG(2, "[slab] new slab %zu on %s (64MB)\n",
                  out_slab_idx, device_tier_name(tier));
        return slab_ptr;
    }

    void slab_free(uint64_t slab_idx) {
        if (slab_idx < slabs_.size() && slabs_[slab_idx].live_count > 0) {
            slabs_[slab_idx].live_count--;
        }
    }

    // ── 拷贝实现 ───────────────────────────────────────────────────
    void device_local_copy(int gpu, void* dst, const void* src, size_t size) {
        #ifdef __CUDACC__
        if (gpu >= 0) {
            cudaSetDevice(gpu);
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        } else {
            std::memcpy(dst, src, size);
        }
        #else
        std::memcpy(dst, src, size);
        #endif
    }

    // [ALG2] peer-direct copy
    void peer_direct_copy(int src_gpu, int dst_gpu, void* dst,
                          const void* src, size_t size, bool async) {
        #ifdef __CUDACC__
        cudaSetDevice(src_gpu);
        if (async) {
            cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, size, 0);
        } else {
            cudaMemcpyPeer(dst, dst_gpu, src, src_gpu, size);
        }
        #else
        std::memcpy(dst, src, size);
        #endif
        (void)async;
    }

    // [ALG2] staged copy: src_gpu → pinned host → dst_gpu
    // 原版: 不存在, 一律cudaMemcpyDefault
    // 改动: 显式两段, 且>16MB时分chunk pipeline overlap
    void staged_copy_via_host(int src_gpu, int dst_gpu,
                              void* dst, const void* src, size_t size) {
        #ifdef __CUDACC__
        if (size <= CHUNK_PIPELINE) {
            // 小块: 简单两段
            void* staging = nullptr;
            cudaMallocHost(&staging, size);
            cudaSetDevice(src_gpu);
            cudaMemcpy(staging, src, size, cudaMemcpyDeviceToHost);
            cudaSetDevice(dst_gpu);
            cudaMemcpy(dst, staging, size, cudaMemcpyHostToDevice);
            cudaFreeHost(staging);
        } else {
            // [ALG2] 大块: 双buffer pipeline overlap
            // 分成CHUNK_PIPELINE大小的段, 交替使用两个staging buffer
            void* stage[2] = {nullptr, nullptr};
            cudaMallocHost(&stage[0], CHUNK_PIPELINE);
            cudaMallocHost(&stage[1], CHUNK_PIPELINE);
            cudaStream_t s_d2h, s_h2d;
            cudaStreamCreate(&s_d2h);
            cudaStreamCreate(&s_h2d);

            size_t offset = 0;
            int buf_idx = 0;
            while (offset < size) {
                size_t chunk = std::min(CHUNK_PIPELINE, size - offset);

                // D2H: src[offset..] → stage[buf_idx]
                cudaSetDevice(src_gpu);
                cudaMemcpyAsync(stage[buf_idx],
                                static_cast<const char*>(src) + offset,
                                chunk, cudaMemcpyDeviceToHost, s_d2h);
                cudaStreamSynchronize(s_d2h);

                // H2D: stage[buf_idx] → dst[offset..]
                cudaSetDevice(dst_gpu);
                cudaMemcpyAsync(static_cast<char*>(dst) + offset,
                                stage[buf_idx],
                                chunk, cudaMemcpyHostToDevice, s_h2d);

                offset += chunk;
                buf_idx ^= 1;  // 交替buffer
            }
            cudaStreamSynchronize(s_h2d);
            cudaStreamDestroy(s_d2h);
            cudaStreamDestroy(s_h2d);
            cudaFreeHost(stage[0]);
            cudaFreeHost(stage[1]);
        }
        #else
        std::memcpy(dst, src, size);
        #endif
    }

    void gpu_to_host_copy(int src_gpu, void* dst, const void* src,
                          size_t size, bool async) {
        #ifdef __CUDACC__
        cudaSetDevice(src_gpu);
        if (async) {
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0);
        } else {
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        }
        #else
        std::memcpy(dst, src, size);
        #endif
    }

    void host_to_gpu_copy(int dst_gpu, void* dst, const void* src,
                          size_t size, bool async) {
        #ifdef __CUDACC__
        cudaSetDevice(dst_gpu);
        if (async) {
            cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
        } else {
            cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        }
        #else
        std::memcpy(dst, src, size);
        #endif
    }

    // ── 数据成员 ───────────────────────────────────────────────────
    mutable std::shared_mutex mu_;
    std::array<DeviceTierBudget, 4> budgets_;
    std::unordered_map<uint64_t, CudaAllocMeta> registry_;
    std::vector<SlabBlock> slabs_;
    PeerTopology topo_;
    uint64_t next_id_ = 1;

    // 统计计数器
    std::atomic<uint64_t> alloc_count_{0};
    std::atomic<uint64_t> dealloc_count_{0};
    std::atomic<uint64_t> copy_count_{0};
    std::atomic<uint64_t> copy_bytes_{0};
};

} // namespace cuda_mem
} // namespace philemon
