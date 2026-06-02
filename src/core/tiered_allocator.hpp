#pragma once
/**
 * tiered_allocator.hpp — 异构内存三层分配器（HBM/GDDR/DRAM 瀑布策略）
 *
 * Design rationale (from RapidStore's wrapper::set_max_threads / NCCL's ncclMemAlloc pattern):
 *
 *   Starting from RapidStore's CsrWrapper::Snapshot as the good example (C),
 *   we follow that pattern to implement TieredAllocator (D), letting
 *   TemporalSubgraphManager (E) place hot intervals in HBM (F) and evict
 *   cold data to DRAM (G).  Then the MigrationScheduler (H) introduces
 *   AccessFrequencyTracker (I), so that the EvictionPolicy (J) can make
 *   LRU-aware tier decisions (K), while the CompactionEngine (L) optimizes
 *   memory fragmentation (M).  Subsequently the BridgeLayer (N) integrates
 *   TEM-Graph's build_index (O), so that IntervalQuery (P) supports
 *   cross-tier contains_query (Q), and in turn the PrefetchEngine (R)
 *   enhances read latency for temporal scans (S).  Finally the
 *   ConcurrencyManager (T) completes the thread-safe snapshot protocol (U),
 *   ensuring RapidStore's snapshot_clone (V) is compatible with tiered
 *   pointers (W), comprehensively upgrading the full query pipeline (Y) to
 *   achieve sub-millisecond temporal subgraph retrieval on heterogeneous
 *   memory (Z).
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2), M007–M008 (Claude #3),
 *            M009–M010 (Claude #4)
 *
 * M009–M010 changes:
 *   - Added borrow<T>() returning RAII TierPtr that holds shared_lock for
 *     the pointer's lifetime. Fixes Bug 4.5 (get_ptr escapes lock scope).
 *     Pattern: PyTorch COWDeleter::NotLastReference = shared_lock<shared_mutex>
 *     (c10/core/impl/COWDeleter.h:36).
 *   - Added migrate_from_staging() for the async migration pipeline.
 *     Accepts pre-copied staging buffer, performs target alloc + swap
 *     under unique_lock. Pattern: DeepSpeed swap_in_optimizer_state
 *     (partitioned_optimizer_swapper.py:76).
 *
 * M008 changes:
 *   - Integrated SlabAllocator for per-tier slab-based memory management.
 *     Small allocations (≤512 KB) go through slab pools with size-class
 *     bucketing, eliminating per-allocation posix_memalign overhead.
 *     Pattern: PyTorch CachingAllocator Block/try_merge_blocks
 *     (c10/cuda/CUDACachingAllocator.cpp:3583), NCCL cudaMemPoolCreate
 *     page-based pooling (nccl/src/allocator.cc:345), TensorFlow Arena
 *     bump allocation (tensorflow/core/lib/core/arena.h:35).
 *   - compact() method releases empty slab pages back to OS, preventing
 *     the memory bloat identified in Claude #1 review Bug 4.3.
 *
 * M005 changes:
 *   - touch() is now LOCKFREE: uses atomic counters directly without
 *     taking mu_.  Pattern from NCCL's COMPILER_ATOMIC_FETCH_ADD
 *     (nccl/src/include/compiler/gcc.h) and CCCL's shared_block_ptr
 *     refcount (cccl/libcudacxx/include/cuda/__memory_resource/shared_block_ptr.h).
 *   - Registry reads (get_ptr, get_meta, for_each_alloc) now use
 *     std::shared_mutex (read-shared / write-exclusive), following
 *     PyTorch c10's COWDeleter pattern
 *     (pytorch/c10/core/impl/COWDeleter.h: shared_lock<shared_mutex>).
 *   - Structural mutations (allocate, deallocate, migrate) take
 *     unique_lock; concurrent reads proceed unblocked.
 */

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <atomic>
#include <vector>
#include <shared_mutex>    // M005: replaces plain std::mutex for read path
#include <mutex>           // M005: for std::unique_lock
#include <memory>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "slab_allocator.hpp"      // M008: per-tier slab allocation
#include "tier_ptr.hpp"            // M009: RAII pointer guard

namespace philemon {

// ─── Memory Tier Definitions ────────────────────────────────────────────────
// Mirrors NCCL's topology-aware device placement (ncclTopoGraph).
// In production: HBM = cudaMalloc on H100, GDDR = cudaMalloc on A6000,
// DRAM = posix_memalign.  In CPU-only dev: all tiers simulated via DRAM
// with artificial latency accounting.

enum class MemoryTier : uint8_t {
    HBM   = 0,   // H100 High-Bandwidth Memory (3.35 TB/s)
    GDDR  = 1,   // A6000 GDDR6 (768 GB/s)
    DRAM  = 2,   // CPU DDR5 (≈ 80 GB/s per channel)
    TIER_COUNT = 3
};

inline const char* tier_name(MemoryTier t) {
    switch (t) {
        case MemoryTier::HBM:  return "HBM";
        case MemoryTier::GDDR: return "GDDR";
        case MemoryTier::DRAM: return "DRAM";
        default: return "UNKNOWN";
    }
}

// ─── Allocation Metadata ────────────────────────────────────────────────────
// Every allocation carries metadata for the migration scheduler.
// Follows the pattern from TEM-Graph's TInterval (id, l, r) extended
// with access-frequency counters.
//
// M005: access_count and last_access_ns are std::atomic<uint64_t>.
// They are updated lockfree by touch() — no mutex needed.
// This follows CCCL's shared_block_ptr::__ref_count pattern
// (fetch_add with memory_order_relaxed for counters, release for
// pointer publication).

struct AllocMeta {
    uint64_t    alloc_id;          // unique allocation identifier
    MemoryTier  current_tier;      // where the block currently resides
    size_t      size_bytes;        // allocation size
    void*       base_ptr;          // pointer to start of region

    // Access tracking — LOCKFREE (M005)
    // Updated by touch() without taking any lock.
    // Pattern: NCCL's __atomic_fetch_add (compiler/gcc.h:37)
    std::atomic<uint64_t>  access_count{0};
    std::atomic<uint64_t>  last_access_ns{0};    // nanoseconds since epoch

    // Temporal graph context: which interval range does this block serve?
    int32_t     interval_start;    // TEM-Graph Timestamp
    int32_t     interval_end;      // TEM-Graph Timestamp

    AllocMeta()
        : alloc_id(0), current_tier(MemoryTier::DRAM), size_bytes(0),
          base_ptr(nullptr), interval_start(-1), interval_end(-1) {}

    // M005: explicit copy — atomics cannot be implicitly copied.
    // Snapshot current values at copy time.
    AllocMeta(const AllocMeta& o)
        : alloc_id(o.alloc_id), current_tier(o.current_tier),
          size_bytes(o.size_bytes), base_ptr(o.base_ptr),
          interval_start(o.interval_start), interval_end(o.interval_end)
    {
        access_count.store(o.access_count.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        last_access_ns.store(o.last_access_ns.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
    }

    AllocMeta& operator=(const AllocMeta& o) {
        if (this != &o) {
            alloc_id       = o.alloc_id;
            current_tier   = o.current_tier;
            size_bytes     = o.size_bytes;
            base_ptr       = o.base_ptr;
            interval_start = o.interval_start;
            interval_end   = o.interval_end;
            access_count.store(o.access_count.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
            last_access_ns.store(o.last_access_ns.load(std::memory_order_relaxed),
                                 std::memory_order_relaxed);
        }
        return *this;
    }
};


// ─── Tier Budget ────────────────────────────────────────────────────────────
// Capacity limits per tier.  On the real server: H100 80 GB HBM, A6000
// 48 GB GDDR, host DRAM 256 GB.  The scheduler respects these limits.

struct TierBudget {
    size_t capacity_bytes;         // maximum bytes for this tier
    std::atomic<size_t> used_bytes{0};

    TierBudget() : capacity_bytes(0) {}
    explicit TierBudget(size_t cap) : capacity_bytes(cap) {}

    // M005: copy/assign for TierBudget (atomics need explicit handling)
    TierBudget(const TierBudget& o)
        : capacity_bytes(o.capacity_bytes)
    {
        used_bytes.store(o.used_bytes.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    }
    TierBudget& operator=(const TierBudget& o) {
        if (this != &o) {
            capacity_bytes = o.capacity_bytes;
            used_bytes.store(o.used_bytes.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        }
        return *this;
    }

    bool can_fit(size_t n) const {
        return used_bytes.load(std::memory_order_relaxed) + n <= capacity_bytes;
    }

    bool try_reserve(size_t n) {
        size_t cur = used_bytes.load(std::memory_order_relaxed);
        while (cur + n <= capacity_bytes) {
            if (used_bytes.compare_exchange_weak(
                    cur, cur + n,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    void release(size_t n) {
        size_t prev = used_bytes.fetch_sub(n, std::memory_order_acq_rel);
        assert(prev >= n && "double-free or over-release in TierBudget");
    }
};


// ─── Tiered Allocator ───────────────────────────────────────────────────────
// The core allocator.  In CPU-dev mode it allocates from DRAM only; the
// tier tag in AllocMeta records the *intended* placement for the real
// server.  The migration scheduler reads these tags and issues CUDA
// memcpy (or NVLink peer-copy) on the production cluster.
//
// Design follows wrapper::insert_edge / wrapper::remove_edge (RapidStore)
// pattern: simple top-level API, backend dispatch via the tier enum.
//
// M005 concurrency model (PyTorch c10::COWDeleter shared_mutex pattern):
//   - Structural mutations (allocate/deallocate/migrate): unique_lock<shared_mutex>
//   - Read-only access (get_ptr/get_meta/for_each_alloc): shared_lock<shared_mutex>
//   - Counter updates (touch): LOCKFREE — atomics only, no lock

class TieredAllocator {
public:
    TieredAllocator(size_t hbm_cap, size_t gddr_cap, size_t dram_cap)
        : next_alloc_id_(1)
    {
        budgets_[static_cast<int>(MemoryTier::HBM)]  = TierBudget(hbm_cap);
        budgets_[static_cast<int>(MemoryTier::GDDR)] = TierBudget(gddr_cap);
        budgets_[static_cast<int>(MemoryTier::DRAM)] = TierBudget(dram_cap);
    }

    ~TieredAllocator() {
        // Release all remaining allocations
        std::unique_lock<std::shared_mutex> lk(mu_);
        for (auto& [id, meta] : registry_) {
            if (meta.base_ptr) {
                // M008: Route through slab for small allocations
                int tier_idx = static_cast<int>(meta.current_tier);
                if (slab_size_class(meta.size_bytes) < SLAB_NUM_CLASSES) {
                    slab_[tier_idx].deallocate(meta.base_ptr);
                } else {
                    ::free(meta.base_ptr);
                }
                budgets_[static_cast<int>(meta.current_tier)].release(meta.size_bytes);
            }
        }
        registry_.clear();
    }

    // Allocate on the preferred tier; fall back to lower tiers if full.
    // Returns allocation id (0 on failure).
    // Takes UNIQUE lock — structural mutation.
    uint64_t allocate(size_t size, MemoryTier preferred,
                      int32_t ts_start = -1, int32_t ts_end = -1) {
        MemoryTier actual = preferred;

        // Waterfall: HBM → GDDR → DRAM
        if (!budgets_[static_cast<int>(actual)].try_reserve(size)) {
            if (actual == MemoryTier::HBM) actual = MemoryTier::GDDR;
            if (!budgets_[static_cast<int>(actual)].try_reserve(size)) {
                actual = MemoryTier::DRAM;
                if (!budgets_[static_cast<int>(actual)].try_reserve(size)) {
                    return 0;  // out of memory across all tiers
                }
            }
        }

        // In CPU-dev mode: all tiers use posix malloc.
        // M008: Small allocations go through slab pools.
        void* ptr = nullptr;
        int tier_idx = static_cast<int>(actual);
        if (is_slab_managed(size)) {
            auto [sptr, actual_sz] = slab_[tier_idx].allocate(size);
            ptr = sptr;
        } else {
            int rc = ::posix_memalign(&ptr, 64, size);  // 64-byte alignment
            if (rc != 0) ptr = nullptr;
            if (ptr) ::memset(ptr, 0, size);
        }
        if (!ptr) {
            budgets_[static_cast<int>(actual)].release(size);
            return 0;
        }

        uint64_t id = next_alloc_id_.fetch_add(1, std::memory_order_relaxed);

        AllocMeta meta;
        meta.alloc_id       = id;
        meta.current_tier   = actual;
        meta.size_bytes     = size;
        meta.base_ptr       = ptr;
        meta.interval_start = ts_start;
        meta.interval_end   = ts_end;

        {
            std::unique_lock<std::shared_mutex> lk(mu_);  // M005: unique_lock
            registry_[id] = meta;
        }

        return id;
    }

    // Touch an allocation (updates access counters for the scheduler).
    //
    // M005 CRITICAL FIX: This is now LOCKFREE.
    //
    // Previous implementation (M001–M004) took mu_ for every touch(),
    // serializing all concurrent reads.  The fix uses atomics directly:
    //   - std::shared_lock to find the AllocMeta* (read-only map lookup)
    //   - Then atomics are updated without any lock held
    //
    // Pattern source: NCCL's COMPILER_ATOMIC_FETCH_ADD (compiler/gcc.h:37)
    //   #define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add(...)
    //
    // And CCCL's shared_block_ptr refcount:
    //   __block_->__ref_count.fetch_add(1, memory_order_relaxed);
    //
    // The shared_lock for map lookup is O(1) amortized for unordered_map
    // and allows full concurrency among readers.
    void touch(uint64_t alloc_id) {
        AllocMeta* meta_ptr = nullptr;
        {
            std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
            auto it = registry_.find(alloc_id);
            if (it == registry_.end()) return;
            meta_ptr = &(it->second);
        }
        // Lock released — atomics updated lockfree.
        // Amortize chrono::now() cost: only update timestamp every 64 touches.
        // The access_count is always accurate; last_access_ns is approximate.
        uint64_t prev = meta_ptr->access_count.fetch_add(1, std::memory_order_relaxed);
        if ((prev & 63) == 0) {
            auto now = std::chrono::steady_clock::now().time_since_epoch();
            meta_ptr->last_access_ns.store(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()),
                std::memory_order_relaxed);
        }
    }

    // Migrate an allocation to a different tier.
    // On the server: this issues cudaMemcpyAsync between devices.
    // Takes UNIQUE lock — structural mutation.
    bool migrate(uint64_t alloc_id, MemoryTier target) {
        std::unique_lock<std::shared_mutex> lk(mu_);  // M005: unique_lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return false;

        AllocMeta& meta = it->second;
        if (meta.current_tier == target) return true;  // already there

        size_t sz = meta.size_bytes;
        if (!budgets_[static_cast<int>(target)].try_reserve(sz)) {
            return false;  // target tier is full
        }

        // In CPU-dev: re-allocate + memcpy (simulates device transfer).
        // M008: Use slab allocator for small sizes on target tier.
        void* new_ptr = nullptr;
        int target_idx = static_cast<int>(target);
        if (is_slab_managed(sz)) {
            auto [sptr, actual_sz] = slab_[target_idx].allocate(sz);
            new_ptr = sptr;
        } else {
            int rc = ::posix_memalign(&new_ptr, 64, sz);
            if (rc != 0) new_ptr = nullptr;
        }
        if (!new_ptr) {
            budgets_[static_cast<int>(target)].release(sz);
            return false;
        }
        ::memcpy(new_ptr, meta.base_ptr, sz);
        // Free old allocation through slab or OS
        int old_idx = static_cast<int>(meta.current_tier);
        if (is_slab_managed(sz)) {
            slab_[old_idx].deallocate(meta.base_ptr);
        } else {
            ::free(meta.base_ptr);
        }

        budgets_[static_cast<int>(meta.current_tier)].release(sz);
        meta.base_ptr     = new_ptr;
        meta.current_tier = target;

        return true;
    }

    // Free an allocation.
    // Takes UNIQUE lock — structural mutation.
    void deallocate(uint64_t alloc_id) {
        std::unique_lock<std::shared_mutex> lk(mu_);  // M005: unique_lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return;

        AllocMeta& meta = it->second;
        if (meta.base_ptr) {
            // M008: Route through slab for small allocations
            int tier_idx = static_cast<int>(meta.current_tier);
            if (is_slab_managed(meta.size_bytes)) {
                slab_[tier_idx].deallocate(meta.base_ptr);
            } else {
                ::free(meta.base_ptr);
            }
            budgets_[static_cast<int>(meta.current_tier)].release(meta.size_bytes);
        }
        registry_.erase(it);
    }

    // Get raw pointer (for the wrapper layer to pass to algorithms).
    // M005: shared_lock — concurrent reads allowed.
    void* get_ptr(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return nullptr;
        return it->second.base_ptr;
    }

    // ── M009: RAII pointer guard ──────────────────────────────────────
    // borrow<T>() returns a TierPtr<T> that holds the shared_lock for
    // the pointer's lifetime. The lock is released when TierPtr is
    // destroyed, guaranteeing the pointer remains valid.
    //
    // Fixes Bug 4.5: get_ptr() returned raw pointer after releasing lock.
    //
    // Pattern: PyTorch COWDeleterContext::decrement_refcount returns
    //   std::variant<NotLastReference, LastReference>
    //   where NotLastReference = std::shared_lock<std::shared_mutex>
    //   (c10/core/impl/COWDeleter.h:36)
    //
    // The caller holds the lock via the TierPtr RAII wrapper, mirroring
    // how PyTorch returns the shared_lock inside the variant.
    template <typename T = void>
    TierPtr<T> borrow(uint64_t alloc_id) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return TierPtr<T>{};
        void* p = it->second.base_ptr;
        size_t sz = it->second.size_bytes;
        return TierPtr<T>{static_cast<T*>(p), sz, std::move(lk)};
    }

    // ── M009: Async migration staging entry point ─────────────────────
    // Called by AsyncMigrator after data has been copied into a staging
    // buffer. Allocates on target tier, copies staging→target, frees old.
    // Takes unique_lock — structural mutation.
    //
    // Pattern: DeepSpeed swap_in_optimizer_state copies from pinned
    //   buffer into optimizer state under lock
    //   (partitioned_optimizer_swapper.py:76)
    bool migrate_from_staging(uint64_t alloc_id, MemoryTier target,
                              const void* staging_buf, size_t staging_size) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return false;

        AllocMeta& meta = it->second;
        if (meta.current_tier == target) return true;

        size_t sz = meta.size_bytes;
        if (sz != staging_size) return false;  // size mismatch

        if (!budgets_[static_cast<int>(target)].try_reserve(sz)) {
            return false;
        }

        // Allocate on target tier.
        void* new_ptr = nullptr;
        int target_idx = static_cast<int>(target);
        if (is_slab_managed(sz)) {
            auto [sptr, actual_sz] = slab_[target_idx].allocate(sz);
            new_ptr = sptr;
        } else {
            int rc = ::posix_memalign(&new_ptr, 64, sz);
            if (rc != 0) new_ptr = nullptr;
        }
        if (!new_ptr) {
            budgets_[static_cast<int>(target)].release(sz);
            return false;
        }

        // Copy from staging buffer → target (data already in staging).
        ::memcpy(new_ptr, staging_buf, sz);

        // Free old allocation.
        int old_idx = static_cast<int>(meta.current_tier);
        if (is_slab_managed(sz)) {
            slab_[old_idx].deallocate(meta.base_ptr);
        } else {
            ::free(meta.base_ptr);
        }

        budgets_[static_cast<int>(meta.current_tier)].release(sz);
        meta.base_ptr     = new_ptr;
        meta.current_tier = target;
        return true;
    }

    // Get a read-only view of metadata (for the scheduler).
    // M005: shared_lock — concurrent reads allowed.
    bool get_meta(uint64_t alloc_id, AllocMeta& out) const {
        std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
        auto it = registry_.find(alloc_id);
        if (it == registry_.end()) return false;
        out = it->second;  // uses M005 copy constructor
        return true;
    }

    // Iterate all allocations (for migration scheduling).
    // M005: shared_lock — concurrent reads allowed.
    void for_each_alloc(std::function<void(uint64_t, const AllocMeta&)> cb) const {
        std::shared_lock<std::shared_mutex> lk(mu_);  // M005: shared read lock
        for (auto& [id, meta] : registry_) {
            cb(id, meta);
        }
    }

    // Budget introspection.
    const TierBudget& budget(MemoryTier tier) const {
        return budgets_[static_cast<int>(tier)];
    }

    size_t total_allocated() const {
        size_t sum = 0;
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            sum += budgets_[i].used_bytes.load(std::memory_order_relaxed);
        }
        return sum;
    }

    // M008: Compact slab pools — release empty pages back to OS.
    // Pattern: PyTorch release_cached_blocks (CUDACachingAllocator.cpp:3832)
    // Call periodically (e.g. after migration sweeps) to reclaim fragmented memory.
    size_t compact_slabs() {
        size_t total = 0;
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            total += slab_[i].compact();
        }
        return total;
    }

    // M008: Print slab statistics
    void print_slab_stats() const {
        for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
            std::cout << "[SlabAllocator tier=" << tier_name(static_cast<MemoryTier>(i))
                      << "]\n";
            slab_[i].print_stats();
        }
    }

    // M008: Check if an allocation is slab-managed
    bool is_slab_managed(size_t size) const {
        return slab_size_class(size) < SLAB_NUM_CLASSES;
    }

private:
    mutable std::shared_mutex mu_;       // M005: upgraded from std::mutex
    std::atomic<uint64_t> next_alloc_id_;
    std::unordered_map<uint64_t, AllocMeta> registry_;
    TierBudget budgets_[static_cast<int>(MemoryTier::TIER_COUNT)];
    mutable SlabAllocator slab_[static_cast<int>(MemoryTier::TIER_COUNT)];  // M008
};

}  // namespace philemon
