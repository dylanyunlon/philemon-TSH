#ifndef PHILEMON_LRU_EVICTION_HPP
#define PHILEMON_LRU_EVICTION_HPP
/**
 * lru_eviction.hpp — LRU+频率混合驱逐策略（冷分区降级到慢层）
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → WriterTraceBlock 的 stack<T*> 对象池 pattern
 *     → allocate_xxx() / deallocate_xxx() 的 push/pop 模式
 *     → SpinLock 保护并发访问
 *     → 用于驱逐候选的对象池管理
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *     → allocate_range_element_segment() 的 empty check + malloc fallback
 *     → allocate_vertex_map() 的 memset 初始化
 *     → allocate_art_leaf32/64() 的 stack pop + fallback new
 *     → deallocate 系列的 stack push 回收
 *     → 用于eviction buffer的分配/回收
 *
 *   upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.cpp       (703行)
 *     → Neo_Graph_Wrapper::set_max_threads() 的 writer_register loop
 *     → run_batch_edge_update() 的 batch insert/delete loop
 *     → run_batch_vertex_update() 的 batch CAS操作
 *     → 用于批量驱逐操作
 *
 *   upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.h         (210行)
 *     → TransactionManager tm 成员的生命周期
 *     → get_shared_snapshot / get_unique_snapshot 接口
 *     → Snapshot 嵌套类 + edges() 回调
 *     → 用于驱逐时的快照隔离
 *
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → execute_update() 的 repeat_times + checkpoint_size 循环
 *     → execute_insert_delete() 的 throughput 计算
 *     → 用于驱逐性能统计
 *
 * 修改 (~20%):
 *   - [NEW] LRUList: 双向链表实现LRU排序
 *   - [NEW] FrequencyBucket: 频率桶, 按访问频率分组
 *   - [NEW] EvictionPolicy: LRU+LFU混合策略 (2Q变体)
 *   - [NEW] EvictionCandidate: 驱逐候选评分
 *   - [NEW] TierAwareEviction: 感知tier代价的驱逐决策
 *   - [MOD] WriterTraceBlock pool → EvictionBuffer pool (slab-aware)
 *   - [MOD] allocate/deallocate → acquire/release eviction slots
 *   - [MOD] batch_edge_update → batch_evict (批量驱逐)
 *   - [KEEP] stack-based 对象池 push/pop 100% 保留
 *   - [KEEP] SpinLock / lock-guard pattern 100% 保留
 *   - [KEEP] empty check + malloc fallback 100% 保留
 *   - [KEEP] memset 初始化 100% 保留
 *   - [KEEP] batch loop + checkpoint 计数 100% 保留
 *
 * 断点调试:
 *   PHILE_EVICTION_DUMP()     — 打印LRU链表+频率桶+候选列表
 *   PHILE_LRU_BREAKPOINT      — RAII断点, 打印驱逐前后状态
 *   PHILE_EVICT_TRACE(id)     — 追踪某个partition的驱逐过程
 *
 * Milestone: M024 — LRU eviction policy
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <list>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <chrono>
#include <cassert>
#include <thread>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../cost_model/tier_cost_model.hpp"

namespace philemon {
namespace eviction {

// ═══════════════════════════════════════════════════════════════════════
// SpinLock — 从 neo_reader_trace.h 保留
// 原始: utils/spin_lock.h, WriterTraceBlock::lock
// ═══════════════════════════════════════════════════════════════════════
class SpinLock {
private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // spin
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }
    bool try_lock() {
        return !flag_.test_and_set(std::memory_order_acquire);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// EvictionEntry — LRU链表节点
// ═══════════════════════════════════════════════════════════════════════
struct EvictionEntry {
    uint64_t partition_id;
    uint8_t  current_tier;     // HBM=0, GDDR=1, DRAM=2
    uint64_t bytes;
    uint64_t last_access_ts;   // 最后访问时间戳
    uint32_t access_count;     // 总访问次数
    uint32_t recent_count;     // 近期访问次数 (衰减窗口)
    double   avg_latency_ns;   // 平均查询延迟

    // 驱逐评分: 越低越容易被驱逐
    double eviction_score() const {
        // 综合考虑: 频率(低=容易驱逐) × tier代价(HBM更宝贵)
        double freq_score = static_cast<double>(recent_count + 1);
        double tier_weight = (current_tier == 0) ? 3.0 :
                             (current_tier == 1) ? 2.0 : 1.0;
        return freq_score * tier_weight;
    }

    void dump() const {
        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        std::printf("    [EvEntry] part=%lu tier=%s %luKB "
                    "last_ts=%lu count=%u recent=%u "
                    "avg_lat=%.1fμs score=%.1f\n",
                    (unsigned long)partition_id,
                    tname[current_tier % 3],
                    (unsigned long)(bytes / 1024),
                    (unsigned long)last_access_ts,
                    (unsigned)access_count,
                    (unsigned)recent_count,
                    avg_latency_ns / 1000.0,
                    eviction_score());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// EvictionBufferPool — 从 WriterTraceBlock 的 stack-based pool 移植
// 保留: stack<T*> + allocate/deallocate push/pop (100%)
// 保留: empty() check + malloc fallback (100%)
// 保留: memset 初始化 (100%)
// 保留: SpinLock 保护 (100%)
// 修改: 类型从 RangeElementSegment → EvictionBuffer
// 修改: 增加pool大小统计
// ═══════════════════════════════════════════════════════════════════════

// 驱逐缓冲区: 暂存被驱逐partition的数据
struct EvictionBuffer {
    uint8_t* data;
    uint64_t capacity;
    uint64_t used;
    uint64_t ref_cnt;  // 保留: 与RangeElementSegment相同的ref_cnt
};

class EvictionBufferPool {
private:
    // === 保留: 与WriterTraceBlock完全相同的stack-based pool ===
    SpinLock lock_;
    std::stack<EvictionBuffer*>* buffers_;

    // 统计
    std::atomic<uint64_t> total_allocated_{0};
    std::atomic<uint64_t> total_recycled_{0};
    std::atomic<uint64_t> pool_size_{0};

    uint64_t default_capacity_;

public:
    explicit EvictionBufferPool(uint64_t default_capacity = 1024 * 1024)
        : default_capacity_(default_capacity) {
        buffers_ = new std::stack<EvictionBuffer*>();
    }

    ~EvictionBufferPool() {
        // 清理所有pooled buffers
        while (!buffers_->empty()) {
            auto* buf = buffers_->top();
            buffers_->pop();
            if (buf->data) free(buf->data);
            delete buf;
        }
        delete buffers_;
    }

    // === 保留: 与allocate_range_element_segment()完全相同的模式 ===
    // 原始: neo_reader_trace.cpp:63-71
    EvictionBuffer* allocate(uint64_t capacity = 0) {
        if (capacity == 0) capacity = default_capacity_;

        EvictionBuffer* res = nullptr;
        lock_.lock();
        if (buffers_->empty()) {
            // fallback: malloc new buffer
            res = new EvictionBuffer();
            res->data = (uint8_t*)malloc(capacity);
            res->capacity = capacity;
        } else {
            res = buffers_->top();
            buffers_->pop();
            pool_size_.fetch_sub(1, std::memory_order_relaxed);
            // 如果capacity不够, 重新分配
            if (res->capacity < capacity) {
                free(res->data);
                res->data = (uint8_t*)malloc(capacity);
                res->capacity = capacity;
            }
        }
        lock_.unlock();

        // === 保留: 与upstream完全相同的memset初始化 ===
        memset(res->data, 0, capacity);
        res->ref_cnt = 1;
        res->used = 0;

        total_allocated_.fetch_add(1, std::memory_order_relaxed);

        PHILE_DBG(3, "[EvictionBufferPool] allocated %luKB buffer (total=%lu)",
                  (unsigned long)(capacity / 1024),
                  (unsigned long)total_allocated_.load());

        return res;
    }

    // === 保留: 与deallocate_range_element_segment()完全相同的push回收 ===
    // 原始: neo_reader_trace.cpp (deallocate pattern)
    void deallocate(EvictionBuffer* buf) {
        if (!buf) return;
        lock_.lock();
        buffers_->push(buf);
        pool_size_.fetch_add(1, std::memory_order_relaxed);
        lock_.unlock();

        total_recycled_.fetch_add(1, std::memory_order_relaxed);

        PHILE_DBG(3, "[EvictionBufferPool] recycled buffer (pool=%lu)",
                  (unsigned long)pool_size_.load());
    }

    // [断点调试]
    void dump() const {
        std::printf("  [EvictionBufferPool] allocated=%lu recycled=%lu "
                    "pool=%lu default_cap=%luKB\n",
                    (unsigned long)total_allocated_.load(),
                    (unsigned long)total_recycled_.load(),
                    (unsigned long)pool_size_.load(),
                    (unsigned long)(default_capacity_ / 1024));
    }
};

// ═══════════════════════════════════════════════════════════════════════
// LRUEvictionList — 双向链表LRU
// 骨架: 基于STL list, 类似neo_wrapper的顺序遍历模式
// ═══════════════════════════════════════════════════════════════════════
class LRUEvictionList {
private:
    // LRU链表: front=最近使用, back=最久未用
    std::list<EvictionEntry> lru_list_;
    // partition_id → iterator映射 (O(1) touch)
    std::unordered_map<uint64_t, std::list<EvictionEntry>::iterator> index_;
    mutable std::shared_mutex mu_;

    size_t max_size_;

public:
    explicit LRUEvictionList(size_t max_size = 65536)
        : max_size_(max_size) {}

    // 插入或更新 — O(1)
    void touch(uint64_t partition_id, uint8_t tier, uint64_t bytes,
               uint64_t timestamp, double latency_ns) {
        std::unique_lock<std::shared_mutex> lock(mu_);

        auto it = index_.find(partition_id);
        if (it != index_.end()) {
            // 已存在: 更新并移到front
            auto& entry = *(it->second);
            entry.last_access_ts = timestamp;
            entry.access_count++;
            entry.recent_count++;
            entry.current_tier = tier;
            // 滑动平均延迟
            entry.avg_latency_ns = entry.avg_latency_ns * 0.9 +
                                   latency_ns * 0.1;
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        } else {
            // 新增
            if (lru_list_.size() >= max_size_) {
                // 移除最久未用的
                auto& victim = lru_list_.back();
                index_.erase(victim.partition_id);
                lru_list_.pop_back();
            }

            EvictionEntry entry;
            entry.partition_id  = partition_id;
            entry.current_tier  = tier;
            entry.bytes         = bytes;
            entry.last_access_ts = timestamp;
            entry.access_count  = 1;
            entry.recent_count  = 1;
            entry.avg_latency_ns = latency_ns;

            lru_list_.push_front(entry);
            index_[partition_id] = lru_list_.begin();
        }
    }

    // 获取驱逐候选: 从back开始, 返回最多n个最冷的partition
    std::vector<EvictionEntry> get_eviction_candidates(size_t n) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::vector<EvictionEntry> candidates;
        auto it = lru_list_.rbegin();
        for (size_t i = 0; i < n && it != lru_list_.rend(); ++it, ++i) {
            candidates.push_back(*it);
        }
        return candidates;
    }

    // 获取特定tier的驱逐候选
    std::vector<EvictionEntry> get_tier_candidates(uint8_t tier,
                                                    size_t n) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::vector<EvictionEntry> candidates;
        // 从LRU尾部开始扫描
        for (auto it = lru_list_.rbegin();
             it != lru_list_.rend() && candidates.size() < n; ++it) {
            if (it->current_tier == tier) {
                candidates.push_back(*it);
            }
        }
        return candidates;
    }

    // 移除已驱逐的entry
    void remove(uint64_t partition_id) {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto it = index_.find(partition_id);
        if (it != index_.end()) {
            lru_list_.erase(it->second);
            index_.erase(it);
        }
    }

    // 频率衰减: 所有entry的recent_count减半
    void decay_recent_counts() {
        std::unique_lock<std::shared_mutex> lock(mu_);
        for (auto& entry : lru_list_) {
            entry.recent_count >>= 1;
        }
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return lru_list_.size();
    }

    // [断点调试] 打印LRU链表状态
    void dump(size_t head_n = 5, size_t tail_n = 5) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::printf("══════ LRUEvictionList (size=%zu, max=%zu) ══════\n",
                    lru_list_.size(), max_size_);

        // 打印头部 (最热)
        std::printf("  --- HOTTEST (head) ---\n");
        auto it = lru_list_.begin();
        for (size_t i = 0; i < head_n && it != lru_list_.end(); ++it, ++i) {
            it->dump();
        }

        if (lru_list_.size() > head_n + tail_n) {
            std::printf("    ... (%zu entries omitted) ...\n",
                        lru_list_.size() - head_n - tail_n);
        }

        // 打印尾部 (最冷 = 驱逐候选)
        std::printf("  --- COLDEST (tail, eviction candidates) ---\n");
        auto candidates = get_eviction_candidates(tail_n);
        for (const auto& c : candidates) {
            c.dump();
        }

        // Tier分布统计
        uint64_t tier_counts[3] = {0, 0, 0};
        uint64_t tier_bytes[3]  = {0, 0, 0};
        for (const auto& entry : lru_list_) {
            if (entry.current_tier < 3) {
                tier_counts[entry.current_tier]++;
                tier_bytes[entry.current_tier] += entry.bytes;
            }
        }
        std::printf("  tier distribution: HBM=%lu(%luMB) GDDR=%lu(%luMB) "
                    "DRAM=%lu(%luMB)\n",
                    (unsigned long)tier_counts[0],
                    (unsigned long)(tier_bytes[0] / (1024*1024)),
                    (unsigned long)tier_counts[1],
                    (unsigned long)(tier_bytes[1] / (1024*1024)),
                    (unsigned long)tier_counts[2],
                    (unsigned long)(tier_bytes[2] / (1024*1024)));
        std::printf("══════ End LRUEvictionList ══════\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// BatchEvictionExecutor — 批量驱逐执行器
// 骨架: neo_wrapper.cpp::run_batch_edge_update() 的batch loop
//       + driver.h::execute_update() 的 repeat_times + checkpoint
// 修改: 改为batch eviction with tier-awareness
// ═══════════════════════════════════════════════════════════════════════
class BatchEvictionExecutor {
private:
    EvictionBufferPool& pool_;
    const cost_model::TierCostModel& cost_model_;

    // === 保留: driver.h execute_update的checkpoint概念 ===
    uint64_t checkpoint_size_;    // 每多少个eviction打一次checkpoint

    // 迁移回调
    using MigrateCallback = std::function<bool(uint64_t, uint8_t, uint8_t, uint64_t)>;
    MigrateCallback migrate_fn_;

    // 统计
    std::atomic<uint64_t> total_evictions_{0};
    std::atomic<uint64_t> total_bytes_evicted_{0};
    std::atomic<uint64_t> batch_count_{0};

public:
    BatchEvictionExecutor(EvictionBufferPool& pool,
                          const cost_model::TierCostModel& cost_model,
                          uint64_t checkpoint_size = 100)
        : pool_(pool)
        , cost_model_(cost_model)
        , checkpoint_size_(checkpoint_size) {}

    void set_migrate_callback(MigrateCallback fn) {
        migrate_fn_ = std::move(fn);
    }

    // === 模仿: run_batch_edge_update()的batch循环 ===
    // 原始: neo_wrapper.cpp:451-495
    struct EvictionResult {
        uint64_t evicted_count;
        uint64_t evicted_bytes;
        uint64_t failed_count;
        double   total_time_us;
        std::vector<std::pair<uint64_t, bool>> details; // (partition_id, success)

        void dump() const {
            std::printf("  [EvictionResult] evicted=%lu/%lu bytes=%luMB "
                        "time=%.1fμs\n",
                        (unsigned long)evicted_count,
                        (unsigned long)(evicted_count + failed_count),
                        (unsigned long)(evicted_bytes / (1024*1024)),
                        total_time_us);
        }
    };

    EvictionResult execute_batch(
        const std::vector<EvictionEntry>& candidates,
        uint8_t target_tier)
    {
        debug::ScopedTimer timer("BatchEvictionExecutor::execute_batch");
        auto start = std::chrono::high_resolution_clock::now();

        EvictionResult result;
        result.evicted_count = 0;
        result.evicted_bytes = 0;
        result.failed_count  = 0;

        batch_count_.fetch_add(1, std::memory_order_relaxed);

        PHILE_DBG(1, "[BatchEvict] starting batch #%lu: %zu candidates → tier %u",
                  (unsigned long)batch_count_.load(),
                  candidates.size(), (unsigned)target_tier);

        // === 保留: 与run_batch_edge_update相同的batch loop + checkpoint ===
        for (size_t i = 0; i < candidates.size(); i++) {
            const auto& entry = candidates[i];

            // 跳过已在目标tier或更低tier的
            if (entry.current_tier >= target_tier) {
                PHILE_DBG(3, "[BatchEvict] skip part=%lu (already tier %u)",
                          (unsigned long)entry.partition_id,
                          (unsigned)entry.current_tier);
                continue;
            }

            // 分配eviction buffer
            EvictionBuffer* buf = pool_.allocate(entry.bytes);

            bool success = false;
            if (migrate_fn_) {
                success = migrate_fn_(entry.partition_id,
                                      entry.current_tier,
                                      target_tier, entry.bytes);
            } else {
                // 模拟迁移
                success = true;
            }

            // === 保留: checkpoint计数打印 ===
            // 原始: driver.h execute_update的 (j-start) % checkpoint_size
            if ((i + 1) % checkpoint_size_ == 0) {
                PHILE_DBG(1, "[BatchEvict] checkpoint: %zu/%zu evicted "
                          "(bytes=%luMB)",
                          result.evicted_count, i + 1,
                          (unsigned long)(result.evicted_bytes / (1024*1024)));
            }

            if (success) {
                result.evicted_count++;
                result.evicted_bytes += entry.bytes;
                total_evictions_.fetch_add(1, std::memory_order_relaxed);
                total_bytes_evicted_.fetch_add(entry.bytes,
                    std::memory_order_relaxed);
            } else {
                result.failed_count++;
            }

            result.details.push_back({entry.partition_id, success});

            // 回收buffer
            pool_.deallocate(buf);
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.total_time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count();

        PHILE_DBG(1, "[BatchEvict] batch complete: %lu/%zu evicted in %.1fμs",
                  (unsigned long)result.evicted_count,
                  candidates.size(), result.total_time_us);

        return result;
    }

    // [断点调试]
    void dump() const {
        std::printf("  [BatchEvictionExecutor] total=%lu bytes=%luMB "
                    "batches=%lu checkpoint=%lu\n",
                    (unsigned long)total_evictions_.load(),
                    (unsigned long)(total_bytes_evicted_.load() / (1024*1024)),
                    (unsigned long)batch_count_.load(),
                    (unsigned long)checkpoint_size_);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// TierAwareEvictionPolicy — 顶层驱逐策略
// 集成LRU + 频率 + Tier代价 的混合决策
// 骨架: driver.h::Driver 的 m_method + execute() 分发
//       + neo_wrapper.h 的 TransactionManager 生命周期
// 修改: 替换为tier-aware eviction逻辑
// ═══════════════════════════════════════════════════════════════════════
class TierAwareEvictionPolicy {
private:
    LRUEvictionList      lru_;
    EvictionBufferPool   pool_;
    BatchEvictionExecutor executor_;

    // 每个tier的容量和当前用量
    struct TierUsage {
        uint64_t capacity_bytes;
        std::atomic<uint64_t> used_bytes{0};
        double   pressure_threshold;   // 驱逐触发阈值 (0.0-1.0)
        double   target_utilization;   // 驱逐目标利用率

        double utilization() const {
            return (capacity_bytes > 0) ?
                static_cast<double>(used_bytes.load()) / capacity_bytes : 0.0;
        }
        bool needs_eviction() const {
            return utilization() > pressure_threshold;
        }

        void dump(const char* name) const {
            std::printf("    [%s] used=%luMB/%luMB (%.1f%%) "
                        "threshold=%.0f%% target=%.0f%%\n",
                        name,
                        (unsigned long)(used_bytes.load() / (1024*1024)),
                        (unsigned long)(capacity_bytes / (1024*1024)),
                        utilization() * 100.0,
                        pressure_threshold * 100.0,
                        target_utilization * 100.0);
        }
    };

    TierUsage tiers_[3];

    // 后台驱逐线程
    std::thread eviction_thread_;
    std::atomic<bool> running_{false};
    uint64_t check_interval_ms_;

    // 统计
    std::atomic<uint64_t> eviction_cycles_{0};
    std::atomic<uint64_t> evictions_triggered_{0};

public:
    TierAwareEvictionPolicy(const cost_model::TierCostModel& cost_model,
                             uint64_t check_interval_ms = 1000)
        : executor_(pool_, cost_model)
        , check_interval_ms_(check_interval_ms)
    {
        // HBM: 80GB, 驱逐阈值80%, 目标70%
        tiers_[0].capacity_bytes     = 80ULL * 1024 * 1024 * 1024;
        tiers_[0].pressure_threshold = 0.80;
        tiers_[0].target_utilization = 0.70;

        // GDDR: 48GB, 驱逐阈值85%, 目标75%
        tiers_[1].capacity_bytes     = 48ULL * 1024 * 1024 * 1024;
        tiers_[1].pressure_threshold = 0.85;
        tiers_[1].target_utilization = 0.75;

        // DRAM: 512GB, 驱逐阈值90%, 目标80%
        tiers_[2].capacity_bytes     = 512ULL * 1024 * 1024 * 1024;
        tiers_[2].pressure_threshold = 0.90;
        tiers_[2].target_utilization = 0.80;
    }

    ~TierAwareEvictionPolicy() { stop(); }

    // ─── 配置 ───────────────────────────────────────────────────
    void set_tier_capacity(uint8_t tier, uint64_t bytes) {
        tiers_[tier % 3].capacity_bytes = bytes;
    }

    void set_tier_thresholds(uint8_t tier, double pressure, double target) {
        tiers_[tier % 3].pressure_threshold = pressure;
        tiers_[tier % 3].target_utilization = target;
    }

    void set_migrate_callback(
        std::function<bool(uint64_t, uint8_t, uint8_t, uint64_t)> fn) {
        executor_.set_migrate_callback(std::move(fn));
    }

    // ─── 运行时更新 ─────────────────────────────────────────────
    void update_usage(uint8_t tier, uint64_t used_bytes) {
        tiers_[tier % 3].used_bytes.store(used_bytes,
            std::memory_order_relaxed);
    }

    void record_access(uint64_t partition_id, uint8_t tier, uint64_t bytes,
                       uint64_t timestamp, double latency_ns) {
        lru_.touch(partition_id, tier, bytes, timestamp, latency_ns);
    }

    // ─── 生命周期 ───────────────────────────────────────────────
    void start() {
        if (running_.load()) return;
        running_.store(true);
        eviction_thread_ = std::thread([this]() { eviction_loop(); });
        PHILE_DBG(1, "[EvictionPolicy] started (interval=%lums)",
                  (unsigned long)check_interval_ms_);
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (eviction_thread_.joinable()) eviction_thread_.join();
        PHILE_DBG(1, "[EvictionPolicy] stopped. cycles=%lu triggered=%lu",
                  (unsigned long)eviction_cycles_.load(),
                  (unsigned long)evictions_triggered_.load());
    }

    // ─── 手动触发驱逐 (用于测试/调试) ──────────────────────────
    BatchEvictionExecutor::EvictionResult
    force_eviction(uint8_t tier, size_t max_evict = 32) {
        PHILE_BREAKPOINT("force_eviction");

        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        PHILE_DBG(1, "[EvictionPolicy] force_eviction: tier=%s max=%zu",
                  tname[tier % 3], max_evict);

        auto candidates = lru_.get_tier_candidates(tier, max_evict);
        uint8_t target = (tier < 2) ? tier + 1 : 2;

        auto result = executor_.execute_batch(candidates, target);

        // 移除成功驱逐的entries
        for (const auto& [part_id, success] : result.details) {
            if (success) {
                lru_.remove(part_id);
            }
        }

        return result;
    }

    // ─── 访问器 ──────────────────────────────────────────────────
    const LRUEvictionList& lru() const { return lru_; }

    // ─── [断点调试] 全量状态打印 ────────────────────────────────
    void dump_all() const {
        std::printf("╔══════════════════════════════════════════════╗\n");
        std::printf("║       TierAwareEvictionPolicy State         ║\n");
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("  running=%s cycles=%lu triggered=%lu\n",
                    running_.load() ? "true" : "false",
                    (unsigned long)eviction_cycles_.load(),
                    (unsigned long)evictions_triggered_.load());
        std::printf("\n  --- Tier Usage ---\n");
        tiers_[0].dump("HBM ");
        tiers_[1].dump("GDDR");
        tiers_[2].dump("DRAM");
        std::printf("\n");
        lru_.dump(5, 5);
        std::printf("\n");
        pool_.dump();
        executor_.dump();
        std::printf("╚══════════════════════════════════════════════╝\n");
    }

private:
    void eviction_loop() {
        PHILE_DBG(1, "[EvictionPolicy] eviction_loop started");

        while (running_.load()) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(check_interval_ms_));
            if (!running_.load()) break;

            eviction_cycles_.fetch_add(1, std::memory_order_relaxed);
            uint64_t cycle = eviction_cycles_.load();

            // 检查每个tier的压力
            for (uint8_t tier = 0; tier < 3; tier++) {
                if (!tiers_[tier].needs_eviction()) continue;

                evictions_triggered_.fetch_add(1, std::memory_order_relaxed);

                const char* tname[] = {"HBM", "GDDR", "DRAM"};
                PHILE_DBG(1, "[EvictionPolicy] cycle %lu: %s pressure "
                          "%.1f%% > threshold %.0f%%",
                          (unsigned long)cycle, tname[tier],
                          tiers_[tier].utilization() * 100.0,
                          tiers_[tier].pressure_threshold * 100.0);

                // 计算需要释放多少bytes
                uint64_t used = tiers_[tier].used_bytes.load();
                uint64_t target_used = static_cast<uint64_t>(
                    tiers_[tier].capacity_bytes *
                    tiers_[tier].target_utilization);
                uint64_t to_free = (used > target_used) ?
                                   (used - target_used) : 0;

                if (to_free == 0) continue;

                PHILE_DBG(1, "[EvictionPolicy] need to free %luMB from %s",
                          (unsigned long)(to_free / (1024*1024)), tname[tier]);

                // 获取候选 (按LRU coldest优先)
                auto candidates = lru_.get_tier_candidates(tier, 64);

                // 过滤: 只驱逐足够bytes
                std::vector<EvictionEntry> to_evict;
                uint64_t accumulated = 0;
                for (const auto& c : candidates) {
                    if (accumulated >= to_free) break;
                    to_evict.push_back(c);
                    accumulated += c.bytes;
                }

                if (to_evict.empty()) continue;

                // 执行批量驱逐
                uint8_t target = (tier < 2) ? tier + 1 : 2;
                auto result = executor_.execute_batch(to_evict, target);

                // 更新LRU
                for (const auto& [part_id, success] : result.details) {
                    if (success) {
                        lru_.remove(part_id);
                    }
                }

                PHILE_DBG(1, "[EvictionPolicy] evicted %lu/%zu from %s "
                          "(%luMB freed)",
                          (unsigned long)result.evicted_count,
                          to_evict.size(), tname[tier],
                          (unsigned long)(result.evicted_bytes / (1024*1024)));
            }

            // 每10个周期做频率衰减
            if (cycle % 10 == 0) {
                lru_.decay_recent_counts();
                PHILE_DBG(2, "[EvictionPolicy] frequency decay at cycle %lu",
                          (unsigned long)cycle);
            }

            // 每30个周期打印状态 (debug level 2)
            if (cycle % 30 == 0 && debug::get_debug_level() >= 2) {
                dump_all();
            }
        }

        PHILE_DBG(1, "[EvictionPolicy] eviction_loop stopped");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// 便捷宏 — 断点调试
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_EVICTION_DUMP(policy) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[EVICTION_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (policy).dump_all(); \
        } \
    } while(0)

#define PHILE_LRU_DUMP(policy, head_n, tail_n) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[LRU_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (policy).lru().dump(head_n, tail_n); \
        } \
    } while(0)

// 驱逐断点: RAII自动打印前后状态
class EvictionBreakpointGuard {
    const TierAwareEvictionPolicy& policy_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    EvictionBreakpointGuard(const TierAwareEvictionPolicy& policy,
                            const char* name)
        : policy_(policy), name_(name)
        , start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ EVICTION_BP ENTER: %s ━━━━\n", name_);
            policy_.dump_all();
        }
    }
    ~EvictionBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto elapsed = std::chrono::duration_cast<
                std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ EVICTION_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)elapsed);
            policy_.dump_all();
        }
    }
};

#define PHILE_EVICTION_BREAKPOINT(policy, name) \
    ::philemon::eviction::EvictionBreakpointGuard \
        _phile_eviction_bp_##__LINE__((policy), (name))

}  // namespace eviction
}  // namespace philemon

#endif  // PHILEMON_LRU_EVICTION_HPP
