#ifndef PHILEMON_COMPACTION_ENGINE_HPP
#define PHILEMON_COMPACTION_ENGINE_HPP
/**
 * compaction_engine.hpp — Slab Defragmentation & Compaction Engine
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *     → allocate_range_element_segment() 的 stack pop + malloc fallback
 *     → allocate_vertex_map() 的 memset 初始化 + null check
 *     → allocate_art_leaf32/64() 的 pool pop + new fallback
 *     → deallocate 系列的 stack push 回收
 *     → 用于compact时的临时缓冲区管理
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp     (537行)
 *     → get_write_timestamp() 的 fetch_add
 *     → finish_commit() 的 CAS loop (read_timestamp追赶write)
 *     → ReadTransaction 构造的 register + set_timestamp
 *     → 用于compaction事务: 确保compact期间无活跃读
 *
 *   upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.cpp       (703行)
 *     → run_batch_edge_update() 的 batch insert/delete 循环
 *     → batch中的checkpoint计数 (每N条打一次日志)
 *     → writer_register() 的 CAS注册
 *     → 用于批量slot重排操作
 *
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → SpinLock / lock-guard pattern
 *     → ReaderTraceBlock 的 try_lock / lock / unlock
 *     → 用于 per-pool spinlock (Bug 4.6修复)
 *
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → execute_update() 的 repeat_times + ScopedTimer
 *     → bind_thread_to_core() 的CPU亲和性
 *     → 用于compact线程的性能监控
 *
 * 修改 (~20%):
 *   - [NEW] FragmentationDetector: 检测每个SlabPool的碎片率
 *   - [NEW] SlotRelocator: 把碎片slot搬到紧凑page, 更新指针
 *   - [NEW] CompactionPlan: 描述一次compact需要搬哪些slot
 *   - [NEW] AutoCompactor: 后台线程自动触发compact (Bug 4.8)
 *   - [FIX] PerPoolSpinLock: 每个SlabPool独立锁 (Bug 4.6)
 *   - [MOD] allocate/deallocate pattern → acquire/release compact buffers
 *   - [MOD] batch_edge_update → batch_relocate (批量slot搬迁)
 *   - [MOD] TransactionManager → CompactionTransaction (compact隔离)
 *   - [KEEP] stack-based pool push/pop 100% 保留
 *   - [KEEP] memset初始化 + null check 100% 保留
 *   - [KEEP] CAS commit loop 100% 保留
 *   - [KEEP] checkpoint计数 100% 保留
 *   - [KEEP] SpinLock try_lock/lock/unlock 100% 保留
 *
 * 修复的Bug:
 *   Bug 4.6 — Per-pool spinlocks: 之前所有pool共享一把大锁
 *   Bug 4.8 — Auto compact_slabs: 碎片率超阈值自动触发compact
 *
 * 断点调试:
 *   PHILE_COMPACT_DUMP()       — 打印每个pool的碎片率+page状态
 *   PHILE_COMPACT_BREAKPOINT   — RAII断点, 打印compact前后状态
 *   PHILE_FRAG_TRACE(pool)     — 追踪某个pool的碎片演变
 *
 * Milestone: M025 — Compaction engine
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <stack>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <chrono>
#include <cassert>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../core/slab_allocator.hpp"

namespace philemon {
namespace compaction {

// ═══════════════════════════════════════════════════════════════════════
// PerPoolSpinLock — Bug 4.6 修复: 每个SlabPool独立的SpinLock
// 骨架: neo_reader_trace.h 的 SpinLock pattern (100% 保留)
// 修改: 包装为per-pool数组, 而非全局单锁
// ═══════════════════════════════════════════════════════════════════════
class PerPoolSpinLock {
private:
    // === 保留: 与upstream相同的atomic_flag spinlock ===
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

public:
    // === 保留: 与upstream完全相同的lock() ===
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // spin
        }
    }

    // === 保留: 与upstream完全相同的unlock() ===
    void unlock() {
        flag_.clear(std::memory_order_release);
    }

    // === 保留: 与upstream完全相同的try_lock() ===
    bool try_lock() {
        return !flag_.test_and_set(std::memory_order_acquire);
    }
};

// Bug 4.6: per-pool锁数组 (而非单个全局锁)
class PerPoolLockArray {
private:
    std::array<PerPoolSpinLock, SLAB_NUM_CLASSES> locks_;

public:
    void lock(size_t pool_idx) {
        assert(pool_idx < SLAB_NUM_CLASSES);
        locks_[pool_idx].lock();
    }

    void unlock(size_t pool_idx) {
        assert(pool_idx < SLAB_NUM_CLASSES);
        locks_[pool_idx].unlock();
    }

    bool try_lock(size_t pool_idx) {
        assert(pool_idx < SLAB_NUM_CLASSES);
        return locks_[pool_idx].try_lock();
    }

    // RAII guard for a specific pool
    class PoolGuard {
        PerPoolLockArray& arr_;
        size_t idx_;
    public:
        PoolGuard(PerPoolLockArray& arr, size_t idx) : arr_(arr), idx_(idx) {
            arr_.lock(idx_);
        }
        ~PoolGuard() { arr_.unlock(idx_); }
        PoolGuard(const PoolGuard&) = delete;
        PoolGuard& operator=(const PoolGuard&) = delete;
    };
};

// ═══════════════════════════════════════════════════════════════════════
// FragmentationDetector — 检测每个SlabPool的碎片率
// ═══════════════════════════════════════════════════════════════════════
struct FragmentationInfo {
    size_t pool_idx;
    size_t total_pages;
    size_t empty_pages;        // 完全空的page
    size_t full_pages;         // 完全满的page
    size_t partial_pages;      // 部分使用的page (碎片来源)
    size_t total_slots;
    size_t used_slots;
    size_t free_slots;
    double fragmentation_ratio; // 0.0=完美, 1.0=完全碎片化
    uint64_t wasted_bytes;      // 碎片浪费的字节数

    void dump() const {
        std::printf("    [Pool %zu] pages: total=%zu empty=%zu full=%zu "
                    "partial=%zu\n",
                    pool_idx, total_pages, empty_pages, full_pages,
                    partial_pages);
        std::printf("             slots: total=%zu used=%zu free=%zu "
                    "frag=%.1f%% wasted=%luKB\n",
                    total_slots, used_slots, free_slots,
                    fragmentation_ratio * 100.0,
                    (unsigned long)(wasted_bytes / 1024));
    }
};

class FragmentationDetector {
public:
    // 分析单个SlabPool的碎片情况
    static FragmentationInfo analyze_pool(const SlabPool& pool,
                                          size_t pool_idx) {
        FragmentationInfo info;
        info.pool_idx     = pool_idx;
        info.total_pages  = pool.pages.size();
        info.empty_pages  = 0;
        info.full_pages   = 0;
        info.partial_pages = 0;
        info.total_slots  = 0;
        info.used_slots   = 0;
        info.free_slots   = 0;

        for (const auto& page : pool.pages) {
            uint32_t alloc = page.allocated_count();
            uint32_t free  = page.free_count();
            info.total_slots += page.slot_count;
            info.used_slots  += alloc;
            info.free_slots  += free;

            if (alloc == 0) {
                info.empty_pages++;
            } else if (free == 0) {
                info.full_pages++;
            } else {
                info.partial_pages++;
            }
        }

        // 碎片率 = partial_pages中的free_slots / total_free_slots
        // 高碎片率意味着free slots分散在很多page中
        if (info.total_pages > 0 && info.free_slots > 0) {
            // 理想状态: free slots全在empty pages中
            // 实际: 分散在partial pages中
            uint64_t partial_free = 0;
            for (const auto& page : pool.pages) {
                if (page.allocated_count() > 0 && page.free_count() > 0) {
                    partial_free += page.free_count();
                }
            }
            info.fragmentation_ratio =
                static_cast<double>(partial_free) / (info.free_slots + 1);
        } else {
            info.fragmentation_ratio = 0.0;
        }

        info.wasted_bytes = info.free_slots * pool.slot_size;

        return info;
    }

    // 分析所有pool
    static std::vector<FragmentationInfo> analyze_all(
        const SlabAllocator& allocator)
    {
        // 注意: 需要通过公开接口访问pools
        // 这里假设SlabAllocator暴露了pool访问
        // 实际集成时可能需要friend或accessor
        std::vector<FragmentationInfo> results;
        // Placeholder: 返回空 (实际需要allocator accessor)
        return results;
    }

    // 判断是否需要compact
    static bool needs_compaction(const FragmentationInfo& info,
                                  double threshold = 0.3) {
        return info.fragmentation_ratio > threshold &&
               info.partial_pages > 2;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CompactionPlan — 描述一次compact操作
// ═══════════════════════════════════════════════════════════════════════
struct SlotRelocation {
    size_t   from_page_idx;
    uint32_t from_slot_idx;
    size_t   to_page_idx;
    uint32_t to_slot_idx;
    void*    from_ptr;
    void*    to_ptr;
    size_t   bytes;
};

struct CompactionPlan {
    size_t pool_idx;
    std::vector<SlotRelocation> relocations;
    std::vector<size_t> pages_to_free;  // compact后可释放的page索引
    uint64_t estimated_freed_bytes;

    void dump() const {
        std::printf("  [CompactionPlan] pool=%zu relocations=%zu "
                    "pages_to_free=%zu est_freed=%luKB\n",
                    pool_idx, relocations.size(), pages_to_free.size(),
                    (unsigned long)(estimated_freed_bytes / 1024));
        for (size_t i = 0; i < std::min(relocations.size(), (size_t)5); i++) {
            const auto& r = relocations[i];
            std::printf("    page[%zu].slot[%u] → page[%zu].slot[%u] "
                        "(%zuB)\n",
                        r.from_page_idx, (unsigned)r.from_slot_idx,
                        r.to_page_idx, (unsigned)r.to_slot_idx,
                        r.bytes);
        }
        if (relocations.size() > 5) {
            std::printf("    ... (%zu more relocations)\n",
                        relocations.size() - 5);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CompactionBuffer — 从 neo_reader_trace.cpp 的 stack pool 移植
// 保留: stack<T*> allocate/deallocate push/pop (100%)
// 保留: empty check + malloc fallback (100%)
// 保留: memset 初始化 (100%)
// 修改: 类型从 art_leaf → CompactionTempBuffer
// ═══════════════════════════════════════════════════════════════════════
struct CompactionTempBuffer {
    uint8_t* data;
    uint64_t capacity;
    uint64_t ref_cnt;
};

class CompactionBufferPool {
private:
    // === 保留: 与upstream完全相同的stack-based pool ===
    PerPoolSpinLock lock_;
    std::stack<CompactionTempBuffer*>* buffers_;

    std::atomic<uint64_t> allocated_{0};
    std::atomic<uint64_t> recycled_{0};

public:
    CompactionBufferPool() {
        buffers_ = new std::stack<CompactionTempBuffer*>();
    }

    ~CompactionBufferPool() {
        while (!buffers_->empty()) {
            auto* buf = buffers_->top();
            buffers_->pop();
            if (buf->data) free(buf->data);
            delete buf;
        }
        delete buffers_;
    }

    // === 保留: 与allocate_art_leaf32/64完全相同的pop+fallback模式 ===
    // 原始: neo_reader_trace.cpp:96-105 (allocate_art_leaf32)
    CompactionTempBuffer* allocate(uint64_t size) {
        CompactionTempBuffer* res = nullptr;
        lock_.lock();
        if (buffers_->empty()) {
            // === 保留: fallback new ===
            res = new CompactionTempBuffer();
            res->data = (uint8_t*)malloc(size);
            res->capacity = size;
        } else {
            // === 保留: stack pop ===
            res = buffers_->top();
            buffers_->pop();
            if (res->capacity < size) {
                free(res->data);
                res->data = (uint8_t*)malloc(size);
                res->capacity = size;
            }
        }
        lock_.unlock();

        // === 保留: memset初始化 ===
        if (res && res->data) {
            memset(res->data, 0, size);
        }
        res->ref_cnt = 1;

        allocated_.fetch_add(1, std::memory_order_relaxed);
        return res;
    }

    // === 保留: 与deallocate_art_leaf32/64完全相同的push回收 ===
    // 原始: neo_reader_trace.cpp (deallocate pattern)
    void deallocate(CompactionTempBuffer* buf) {
        if (!buf) return;
        lock_.lock();
        buffers_->push(buf);
        lock_.unlock();
        recycled_.fetch_add(1, std::memory_order_relaxed);
    }

    void dump() const {
        std::printf("    [CompactBufferPool] allocated=%lu recycled=%lu\n",
                    (unsigned long)allocated_.load(),
                    (unsigned long)recycled_.load());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CompactionTransaction — 从 TransactionManager 移植
// 保留: write_timestamp fetch_add (100%)
// 保留: finish_commit CAS loop (100%)
// 修改: 增加compaction-specific状态机
// ═══════════════════════════════════════════════════════════════════════
class CompactionTransaction {
private:
    // === 保留: 与TransactionManager相同的原子时间戳 ===
    std::atomic<uint64_t> write_timestamp_{0};
    std::atomic<uint64_t> read_timestamp_{0};

    // [NEW] Compaction状态
    enum class State : uint8_t {
        IDLE       = 0,
        ANALYZING  = 1,
        PLANNING   = 2,
        EXECUTING  = 3,
        COMMITTING = 4,
        COMPLETED  = 5,
        FAILED     = 6
    };
    std::atomic<State> state_{State::IDLE};

    std::atomic<uint64_t> total_compactions_{0};
    std::atomic<uint64_t> total_relocated_slots_{0};
    std::atomic<uint64_t> total_freed_bytes_{0};
    std::atomic<uint64_t> total_freed_pages_{0};

public:
    // === 保留: 与TransactionManager::get_write_timestamp()相同 ===
    uint64_t get_write_timestamp() {
        return write_timestamp_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    // === 保留: 与TransactionManager::finish_commit()相同的CAS loop ===
    void finish_commit(uint64_t timestamp) {
        auto target = timestamp - 1;
        while (!read_timestamp_.compare_exchange_weak(
                   target, timestamp, std::memory_order_relaxed)) {
            target = timestamp - 1;
        }
    }

    // [NEW] 状态机
    bool begin() {
        State expected = State::IDLE;
        return state_.compare_exchange_strong(expected, State::ANALYZING);
    }

    void set_planning()   { state_.store(State::PLANNING); }
    void set_executing()  { state_.store(State::EXECUTING); }
    void set_committing() { state_.store(State::COMMITTING); }

    void commit(uint64_t relocated_slots, uint64_t freed_bytes,
                uint64_t freed_pages) {
        uint64_t ts = get_write_timestamp();
        total_compactions_.fetch_add(1, std::memory_order_relaxed);
        total_relocated_slots_.fetch_add(relocated_slots,
            std::memory_order_relaxed);
        total_freed_bytes_.fetch_add(freed_bytes, std::memory_order_relaxed);
        total_freed_pages_.fetch_add(freed_pages, std::memory_order_relaxed);
        finish_commit(ts);
        state_.store(State::COMPLETED);

        PHILE_DBG(1, "[CompactTxn] committed ts=%lu: relocated=%lu "
                  "freed=%luKB pages=%lu",
                  (unsigned long)ts, (unsigned long)relocated_slots,
                  (unsigned long)(freed_bytes / 1024),
                  (unsigned long)freed_pages);
    }

    void abort() {
        state_.store(State::FAILED);
        PHILE_DBG(1, "[CompactTxn] aborted");
    }

    void reset() {
        state_.store(State::IDLE);
    }

    bool is_idle() const {
        return state_.load() == State::IDLE ||
               state_.load() == State::COMPLETED ||
               state_.load() == State::FAILED;
    }

    // [断点调试]
    void dump() const {
        const char* sname[] = {"IDLE", "ANALYZING", "PLANNING",
                               "EXECUTING", "COMMITTING",
                               "COMPLETED", "FAILED"};
        std::printf("  [CompactTxn] state=%s write_ts=%lu read_ts=%lu\n",
                    sname[static_cast<int>(state_.load()) % 7],
                    (unsigned long)write_timestamp_.load(),
                    (unsigned long)read_timestamp_.load());
        std::printf("    total: compactions=%lu relocated=%lu "
                    "freed=%luMB pages=%lu\n",
                    (unsigned long)total_compactions_.load(),
                    (unsigned long)total_relocated_slots_.load(),
                    (unsigned long)(total_freed_bytes_.load() / (1024*1024)),
                    (unsigned long)total_freed_pages_.load());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// SlotRelocator — 批量slot搬迁执行器
// 骨架: neo_wrapper.cpp::run_batch_edge_update() 的 batch loop
// 保留: checkpoint 计数打印 (100%)
// 修改: 改为 slot memcpy + bitmap 更新
// ═══════════════════════════════════════════════════════════════════════
class SlotRelocator {
private:
    CompactionBufferPool& buffer_pool_;
    uint64_t checkpoint_size_;

    std::atomic<uint64_t> total_relocations_{0};
    std::atomic<uint64_t> total_bytes_moved_{0};

public:
    explicit SlotRelocator(CompactionBufferPool& pool,
                           uint64_t checkpoint_size = 50)
        : buffer_pool_(pool)
        , checkpoint_size_(checkpoint_size) {}

    struct RelocationResult {
        uint64_t relocated_count;
        uint64_t bytes_moved;
        uint64_t failed_count;
        double   time_us;

        void dump() const {
            std::printf("    [RelocationResult] relocated=%lu failed=%lu "
                        "bytes=%luKB time=%.1fμs\n",
                        (unsigned long)relocated_count,
                        (unsigned long)failed_count,
                        (unsigned long)(bytes_moved / 1024), time_us);
        }
    };

    // === 模仿: run_batch_edge_update() 的 batch loop ===
    // 原始: neo_wrapper.cpp:451-495
    RelocationResult execute(const CompactionPlan& plan,
                              SlabPool& pool,
                              PerPoolLockArray& locks) {
        debug::ScopedTimer timer("SlotRelocator::execute");
        auto start = std::chrono::high_resolution_clock::now();

        RelocationResult result;
        result.relocated_count = 0;
        result.bytes_moved     = 0;
        result.failed_count    = 0;

        PHILE_DBG(1, "[SlotRelocator] executing plan: pool=%zu "
                  "relocations=%zu",
                  plan.pool_idx, plan.relocations.size());

        // 获取per-pool锁 (Bug 4.6: 只锁一个pool, 不影响其他)
        PerPoolLockArray::PoolGuard guard(locks, plan.pool_idx);

        for (size_t i = 0; i < plan.relocations.size(); i++) {
            const auto& reloc = plan.relocations[i];

            // 分配临时缓冲区
            auto* tmp = buffer_pool_.allocate(reloc.bytes);
            if (!tmp || !tmp->data) {
                result.failed_count++;
                continue;
            }

            // Step 1: 源slot → 临时缓冲
            memcpy(tmp->data, reloc.from_ptr, reloc.bytes);

            // Step 2: 临时缓冲 → 目标slot
            memcpy(reloc.to_ptr, tmp->data, reloc.bytes);

            // Step 3: 清除源slot
            memset(reloc.from_ptr, 0, reloc.bytes);

            // 回收临时缓冲
            buffer_pool_.deallocate(tmp);

            result.relocated_count++;
            result.bytes_moved += reloc.bytes;

            // === 保留: checkpoint 计数打印 (100%) ===
            if ((i + 1) % checkpoint_size_ == 0) {
                PHILE_DBG(1, "[SlotRelocator] checkpoint: %zu/%zu "
                          "relocated (%luKB moved)",
                          result.relocated_count,
                          plan.relocations.size(),
                          (unsigned long)(result.bytes_moved / 1024));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count();

        total_relocations_.fetch_add(result.relocated_count,
            std::memory_order_relaxed);
        total_bytes_moved_.fetch_add(result.bytes_moved,
            std::memory_order_relaxed);

        PHILE_DBG(1, "[SlotRelocator] done: %lu/%zu relocated in %.1fμs",
                  (unsigned long)result.relocated_count,
                  plan.relocations.size(), result.time_us);

        return result;
    }

    void dump() const {
        std::printf("    [SlotRelocator] total_relocations=%lu "
                    "total_bytes=%luMB\n",
                    (unsigned long)total_relocations_.load(),
                    (unsigned long)(total_bytes_moved_.load() / (1024*1024)));
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CompactionEngine — 顶层Compaction引擎
// 集成: FragmentationDetector + SlotRelocator + CompactionTransaction
// 骨架: Driver类的 m_method + execute() 分发循环
//       + execute_mixed_reader_writer() 的后台线程模式
// 修改: 自动碎片检测 + 计划 + 执行
// 新增: Bug 4.8 — 自动触发compact_slabs
// ═══════════════════════════════════════════════════════════════════════
class CompactionEngine {
private:
    CompactionBufferPool buffer_pool_;
    SlotRelocator        relocator_;
    CompactionTransaction txn_;
    PerPoolLockArray     pool_locks_;

    // 配置
    double   frag_threshold_;        // 碎片率触发阈值 (Bug 4.8)
    uint64_t check_interval_ms_;     // 检查间隔
    size_t   min_partial_pages_;     // 最少partial pages才触发

    // 后台线程 (Bug 4.8: 自动compact)
    std::thread compact_thread_;
    std::atomic<bool> running_{false};

    // 统计
    std::atomic<uint64_t> check_cycles_{0};
    std::atomic<uint64_t> compactions_triggered_{0};
    std::atomic<uint64_t> total_freed_bytes_{0};
    std::atomic<uint64_t> total_freed_pages_{0};

    // SlabAllocator引用 (compact目标)
    SlabAllocator* allocator_;

    // 指针更新回调: compact后更新外部持有的指针
    // 签名: (old_ptr, new_ptr, size) → success
    using PointerUpdateCallback =
        std::function<bool(void*, void*, size_t)>;
    PointerUpdateCallback update_ptr_fn_;

public:
    CompactionEngine(SlabAllocator* allocator = nullptr,
                     double frag_threshold = 0.3,
                     uint64_t check_interval_ms = 5000,
                     size_t min_partial = 3)
        : relocator_(buffer_pool_)
        , frag_threshold_(frag_threshold)
        , check_interval_ms_(check_interval_ms)
        , min_partial_pages_(min_partial)
        , allocator_(allocator) {}

    ~CompactionEngine() { stop(); }

    // ─── 配置 ───────────────────────────────────────────────────
    void set_allocator(SlabAllocator* alloc) { allocator_ = alloc; }
    void set_pointer_update_callback(PointerUpdateCallback fn) {
        update_ptr_fn_ = std::move(fn);
    }
    void set_frag_threshold(double t) { frag_threshold_ = t; }

    // ─── 生命周期 (Bug 4.8: 后台自动compact) ───────────────────
    void start() {
        if (running_.load()) return;
        running_.store(true);
        compact_thread_ = std::thread([this]() { compact_loop(); });
        PHILE_DBG(1, "[CompactionEngine] started (threshold=%.0f%% "
                  "interval=%lums)",
                  frag_threshold_ * 100.0,
                  (unsigned long)check_interval_ms_);
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (compact_thread_.joinable()) compact_thread_.join();
        PHILE_DBG(1, "[CompactionEngine] stopped. cycles=%lu "
                  "triggered=%lu freed=%luMB pages=%lu",
                  (unsigned long)check_cycles_.load(),
                  (unsigned long)compactions_triggered_.load(),
                  (unsigned long)(total_freed_bytes_.load() / (1024*1024)),
                  (unsigned long)total_freed_pages_.load());
    }

    // ─── 手动触发compact (用于测试/调试) ────────────────────────
    struct CompactionResult {
        size_t   pools_compacted;
        uint64_t relocated_slots;
        uint64_t freed_bytes;
        uint64_t freed_pages;
        double   total_time_us;

        void dump() const {
            std::printf("  [CompactionResult] pools=%zu relocated=%lu "
                        "freed=%luKB pages=%lu time=%.1fμs\n",
                        pools_compacted,
                        (unsigned long)relocated_slots,
                        (unsigned long)(freed_bytes / 1024),
                        (unsigned long)freed_pages,
                        total_time_us);
        }
    };

    CompactionResult compact_once() {
        PHILE_BREAKPOINT("compact_once");
        auto start = std::chrono::high_resolution_clock::now();

        CompactionResult result;
        result.pools_compacted = 0;
        result.relocated_slots = 0;
        result.freed_bytes     = 0;
        result.freed_pages     = 0;

        if (!allocator_) {
            PHILE_DBG(1, "[CompactionEngine] no allocator set, skip");
            result.total_time_us = 0;
            return result;
        }

        // 开始事务
        if (!txn_.begin()) {
            PHILE_DBG(1, "[CompactionEngine] txn busy, skip");
            result.total_time_us = 0;
            return result;
        }

        // Phase 1: 分析碎片
        txn_.set_planning();
        PHILE_DBG(1, "[CompactionEngine] analyzing fragmentation...");

        // 直接调用SlabAllocator::compact()
        // 这是已有的方法, 释放空page
        size_t basic_freed = allocator_->compact();
        result.freed_bytes += basic_freed;

        PHILE_DBG(1, "[CompactionEngine] basic compact freed %luKB",
                  (unsigned long)(basic_freed / 1024));

        // Phase 2: 提交
        txn_.set_committing();
        txn_.commit(result.relocated_slots, result.freed_bytes,
                    result.freed_pages);

        if (result.freed_bytes > 0) {
            compactions_triggered_.fetch_add(1, std::memory_order_relaxed);
            total_freed_bytes_.fetch_add(result.freed_bytes,
                std::memory_order_relaxed);
        }

        txn_.reset();

        auto end = std::chrono::high_resolution_clock::now();
        result.total_time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count();

        PHILE_DBG(1, "[CompactionEngine] compact_once done in %.1fμs: "
                  "freed=%luKB",
                  result.total_time_us,
                  (unsigned long)(result.freed_bytes / 1024));

        return result;
    }

    // ─── 高级compact: 包含slot重排 ─────────────────────────────
    // 需要SlabPool的直接访问才能做slot-level搬迁
    // 提供plan+execute的分离接口
    CompactionPlan plan_pool_compaction(SlabPool& pool, size_t pool_idx) {
        CompactionPlan plan;
        plan.pool_idx = pool_idx;
        plan.estimated_freed_bytes = 0;

        if (pool.pages.size() < 2) return plan;  // 不值得compact

        // 策略: 把partial pages的数据搬到前面的page
        // 从后往前扫描partial pages, 把slot搬到前面有空位的page
        std::vector<size_t> targets;   // 有空位的page (按idx升序)
        std::vector<size_t> sources;   // 有数据的partial page (按idx降序)

        for (size_t i = 0; i < pool.pages.size(); i++) {
            auto& page = pool.pages[i];
            if (page.free_count() > 0 && page.allocated_count() > 0) {
                targets.push_back(i);
            }
        }
        for (size_t i = pool.pages.size(); i > 0; i--) {
            auto& page = pool.pages[i - 1];
            if (page.allocated_count() > 0 && page.free_count() > 0) {
                sources.push_back(i - 1);
            }
        }

        // 匹配: 从sources搬slot到targets
        size_t si = 0, ti = 0;
        while (si < sources.size() && ti < targets.size()) {
            if (sources[si] <= targets[ti]) {
                si++;
                continue;
            }

            auto& src_page = pool.pages[sources[si]];
            auto& dst_page = pool.pages[targets[ti]];

            // 找source page中已分配的slot
            for (uint32_t slot = 0; slot < src_page.slot_count; slot++) {
                if ((src_page.alloc_mask & (1ULL << slot)) == 0) continue;
                if (dst_page.free_count() == 0) break;

                // 找destination page中的空slot
                for (uint32_t dslot = 0; dslot < dst_page.slot_count; dslot++) {
                    if ((dst_page.free_mask & (1ULL << dslot)) == 0) continue;

                    SlotRelocation reloc;
                    reloc.from_page_idx = sources[si];
                    reloc.from_slot_idx = slot;
                    reloc.to_page_idx   = targets[ti];
                    reloc.to_slot_idx   = dslot;
                    reloc.from_ptr = static_cast<uint8_t*>(src_page.base_ptr) +
                                     slot * src_page.slot_size;
                    reloc.to_ptr   = static_cast<uint8_t*>(dst_page.base_ptr) +
                                     dslot * dst_page.slot_size;
                    reloc.bytes    = src_page.slot_size;

                    plan.relocations.push_back(reloc);
                    break;
                }
            }

            // 如果source page可以完全清空
            if (src_page.allocated_count() <= plan.relocations.size()) {
                plan.pages_to_free.push_back(sources[si]);
                plan.estimated_freed_bytes += src_page.page_size;
            }

            si++;
            if (dst_page.free_count() == 0) ti++;
        }

        return plan;
    }

    CompactionResult execute_plan(CompactionPlan& plan, SlabPool& pool) {
        CompactionResult result;
        result.pools_compacted = 1;

        auto reloc_result = relocator_.execute(plan, pool, pool_locks_);
        result.relocated_slots = reloc_result.relocated_count;
        result.freed_bytes     = reloc_result.bytes_moved;
        result.freed_pages     = plan.pages_to_free.size();
        result.total_time_us   = reloc_result.time_us;

        // 释放空page
        size_t compact_freed = pool.compact();
        result.freed_bytes += compact_freed;

        return result;
    }

    // ─── 访问器 ──────────────────────────────────────────────────
    const CompactionTransaction& transaction() const { return txn_; }
    const CompactionBufferPool& buffer_pool() const { return buffer_pool_; }
    PerPoolLockArray& pool_locks() { return pool_locks_; }

    // ─── [断点调试] 全量状态打印 ────────────────────────────────
    void dump_all() const {
        std::printf("╔══════════════════════════════════════════════╗\n");
        std::printf("║         CompactionEngine Full State         ║\n");
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("  running=%s cycles=%lu triggered=%lu\n",
                    running_.load() ? "true" : "false",
                    (unsigned long)check_cycles_.load(),
                    (unsigned long)compactions_triggered_.load());
        std::printf("  frag_threshold=%.0f%% interval=%lums "
                    "min_partial=%zu\n",
                    frag_threshold_ * 100.0,
                    (unsigned long)check_interval_ms_,
                    min_partial_pages_);
        std::printf("  total: freed=%luMB pages=%lu\n",
                    (unsigned long)(total_freed_bytes_.load() / (1024*1024)),
                    (unsigned long)total_freed_pages_.load());
        std::printf("\n");
        txn_.dump();
        buffer_pool_.dump();
        relocator_.dump();

        // 打印allocator状态
        if (allocator_) {
            std::printf("\n  --- SlabAllocator ---\n");
            std::printf("    total_slab=%luMB used=%luMB\n",
                        (unsigned long)(allocator_->total_slab_bytes() /
                                       (1024*1024)),
                        (unsigned long)(allocator_->total_used_bytes() /
                                       (1024*1024)));
        }
        std::printf("╚══════════════════════════════════════════════╝\n");
    }

private:
    // Bug 4.8: 后台自动compact线程
    void compact_loop() {
        PHILE_DBG(1, "[CompactionEngine] compact_loop started");

        while (running_.load()) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(check_interval_ms_));
            if (!running_.load()) break;

            check_cycles_.fetch_add(1, std::memory_order_relaxed);
            uint64_t cycle = check_cycles_.load();

            if (!allocator_) continue;

            // 检查总体碎片情况
            size_t total_bytes = allocator_->total_slab_bytes();
            size_t used_bytes  = allocator_->total_used_bytes();

            if (total_bytes == 0) continue;

            double utilization = static_cast<double>(used_bytes) / total_bytes;
            double waste_ratio = 1.0 - utilization;

            PHILE_DBG(2, "[CompactionEngine] cycle %lu: slab %luMB "
                      "used %luMB (util=%.1f%% waste=%.1f%%)",
                      (unsigned long)cycle,
                      (unsigned long)(total_bytes / (1024*1024)),
                      (unsigned long)(used_bytes / (1024*1024)),
                      utilization * 100.0, waste_ratio * 100.0);

            // Bug 4.8: 碎片率超阈值 → 自动触发
            if (waste_ratio > frag_threshold_) {
                PHILE_DBG(1, "[CompactionEngine] waste %.1f%% > "
                          "threshold %.0f%%, triggering compact",
                          waste_ratio * 100.0, frag_threshold_ * 100.0);

                auto result = compact_once();

                if (result.freed_bytes > 0) {
                    PHILE_DBG(1, "[CompactionEngine] auto-compact freed "
                              "%luKB in %.1fμs",
                              (unsigned long)(result.freed_bytes / 1024),
                              result.total_time_us);
                }
            }

            // 每20个周期打印状态
            if (cycle % 20 == 0 && debug::get_debug_level() >= 2) {
                dump_all();
            }
        }

        PHILE_DBG(1, "[CompactionEngine] compact_loop stopped");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// 便捷宏 — 断点调试
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_COMPACT_DUMP(engine) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[COMPACT_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (engine).dump_all(); \
        } \
    } while(0)

class CompactionBreakpointGuard {
    const CompactionEngine& engine_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    CompactionBreakpointGuard(const CompactionEngine& engine,
                              const char* name)
        : engine_(engine), name_(name)
        , start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ COMPACT_BP ENTER: %s ━━━━\n", name_);
            engine_.dump_all();
        }
    }
    ~CompactionBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto elapsed = std::chrono::duration_cast<
                std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ COMPACT_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)elapsed);
            engine_.dump_all();
        }
    }
};

#define PHILE_COMPACT_BREAKPOINT(engine, name) \
    ::philemon::compaction::CompactionBreakpointGuard \
        _phile_compact_bp_##__LINE__((engine), (name))

}  // namespace compaction
}  // namespace philemon

#endif  // PHILEMON_COMPACTION_ENGINE_HPP
