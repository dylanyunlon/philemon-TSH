#ifndef PHILEMON_PREFETCH_ENGINE_HPP
#define PHILEMON_PREFETCH_ENGINE_HPP
/**
 * prefetch_engine.hpp — Query-History-Driven Prefetch Engine
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → ReaderTraceBlock 的 atomic_value + lock/unlock/status/timestamp
 *     → ActiveReaderTracer 的 block 数组 + register/unregister
 *     → 用于追踪活跃查询, 预测下一次查询位置
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *     → reader_register() 的 CAS loop
 *     → get_min_timestamp() 的全局扫描
 *     → WriterTraceBlock 的 stack-based 对象池
 *     → 用于查询历史收集, slab对象回收
 *
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_transaction.h   (331行)
 *     → TransactionManager 的 atomic write_timestamp/read_timestamp
 *     → ReadTransaction 的 snapshot + timestamp 绑定
 *     → 用于预取事务的生命周期管理
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp     (537行)
 *     → get_write_timestamp() 的 fetch_add
 *     → finish_commit() 的 CAS loop (read_timestamp追赶write_timestamp)
 *     → ReadTransaction 构造函数的 reader_register + set_timestamp
 *     → 用于预取请求的原子提交协议
 *
 *   upstream/rapidstore/wrapper/wrapper.h                              (249行)
 *     → snapshot_edges() 模板回调 s->edges(index, callback, logical)
 *     → get_shared_snapshot / snapshot_clone
 *     → 用于预取时的边扫描采样
 *
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → execute_query() 的查询分发循环 (BFS/SSSP/PR/WCC/TC)
 *     → execute_mixed_reader_writer() 的并发reader/writer模式
 *     → Barrier 类 (arrive_and_wait)
 *     → bind_thread_to_core()
 *     → 用于预取线程的调度和同步
 *
 * 修改 (~20%):
 *   - [NEW] QueryHistoryRing: 环形缓冲记录最近N次查询的(vertex, tier, timestamp)
 *   - [NEW] PrefetchPredictor: 基于历史频率的下一次查询tier预测
 *   - [NEW] PrefetchEngine: 后台线程预迁移cold→hot tier数据
 *   - [NEW] PrefetchTicket: 异步预取请求跟踪 (类似AsyncMigrator的ticket)
 *   - [NEW] AdaptiveThreshold: 动态调整预取触发阈值
 *   - [MOD] ReaderTraceBlock → PrefetchTraceBlock (增加tier字段+access_count)
 *   - [MOD] ActiveReaderTracer → PrefetchTracer (增加历史统计)
 *   - [MOD] TransactionManager → PrefetchScheduler (增加预取队列)
 *   - [MOD] execute_query → prefetch_aware_dispatch (查询前检查预取)
 *   - [KEEP] atomic lock/unlock/CAS 模式 100% 保留
 *   - [KEEP] reader_register() 的 CAS loop 100% 保留
 *   - [KEEP] Barrier arrive_and_wait 100% 保留
 *   - [KEEP] bind_thread_to_core() 100% 保留
 *   - [KEEP] stack-based 对象池 pattern 100% 保留
 *
 * 断点调试支持:
 *   PHILE_PREFETCH_DUMP()      — 打印预取队列+命中率+预测状态
 *   PHILE_HISTORY_DUMP(n)      — 打印最近n条查询历史
 *   PHILE_PREDICT_DUMP(vtx)    — 打印某vertex的预取预测
 *   PHILE_PREFETCH_BREAKPOINT  — RAII断点, 自动打印预取前后状态
 *
 * Milestone: M023 — Prefetch engine
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <queue>
#include <stack>
#include <array>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <cassert>
#include <unordered_map>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../cost_model/tier_cost_model.hpp"

namespace philemon {
namespace prefetch {

// ═══════════════════════════════════════════════════════════════════════
// PrefetchTraceBlock — 从 neo_reader_trace.h::ReaderTraceBlock 移植
//   保留: atomic_value 的 lock_bit/status/timestamp 位域编码 (100%)
//   保留: lock() 的 CAS spin loop (100%)
//   保留: unlock() 的 fetch_and 清除 (100%)
//   保留: try_lock() 的单次 CAS (100%)
//   修改: 增加 tier_id (3位) + access_count (16位) 到位域
//   修改: 增加 last_vertex_id 字段用于预测
// ═══════════════════════════════════════════════════════════════════════
struct PrefetchTraceBlock {
private:
    // === 从 ReaderTraceBlock 完整保留的位域布局 ===
    std::atomic<uint64_t> atomic_value;

    // 保留: 与upstream完全相同的位域常量
    static constexpr uint64_t LOCK_BIT       = 63;
    static constexpr uint64_t LOCK_MASK      = 1ULL << LOCK_BIT;
    static constexpr uint64_t STATUS_SHIFT   = 60;
    static constexpr uint64_t STATUS_MASK    = 0x7ULL << STATUS_SHIFT; // 3 bits
    static constexpr uint64_t TIMESTAMP_MASK = (1ULL << 60) - 1;      // 60 bits

    // [NEW] 新增位域: tier在bit 57-59, access_count在bit 41-56
    static constexpr uint64_t TIER_SHIFT     = 57;
    static constexpr uint64_t TIER_MASK      = 0x7ULL << TIER_SHIFT;  // 3 bits
    static constexpr uint64_t COUNT_SHIFT    = 41;
    static constexpr uint64_t COUNT_MASK     = 0xFFFFULL << COUNT_SHIFT; // 16 bits

    // [NEW] 预取专用字段
    std::atomic<uint64_t> last_vertex_id_{0};
    std::atomic<uint32_t> prefetch_hits_{0};
    std::atomic<uint32_t> prefetch_misses_{0};

public:
    PrefetchTraceBlock() : atomic_value(0) {}

    // === 保留: 与upstream ReaderTraceBlock 完全相同的lock() ===
    // 原始代码: neo_reader_trace.h:27-40
    void lock() {
        uint64_t expected;
        uint64_t desired;
        while (true) {
            expected = atomic_value.load(std::memory_order_relaxed);
            if (expected & LOCK_MASK) {
                // Lock is already held by another thread, continue spinning
                continue;
            }
            desired = expected | LOCK_MASK;
            if (atomic_value.compare_exchange_weak(expected, desired,
                                                    std::memory_order_acquire)) {
                // Acquired the lock
                break;
            }
        }
    }

    // === 保留: 与upstream完全相同的unlock() ===
    // 原始代码: neo_reader_trace.h:42-44
    void unlock() {
        atomic_value.fetch_and(~LOCK_MASK, std::memory_order_release);
    }

    // === 保留: 与upstream完全相同的try_lock() ===
    // 原始代码: neo_reader_trace.h:46-50
    bool try_lock() {
        uint64_t expected = atomic_value.load(std::memory_order_relaxed);
        uint64_t desired = expected | LOCK_MASK;
        return ((expected & LOCK_MASK) == 0) &&
               atomic_value.compare_exchange_strong(expected, desired,
                                                     std::memory_order_acquire);
    }

    // === 保留: 与upstream完全相同的get/set_status ===
    uint8_t get_status() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return static_cast<uint8_t>((value & STATUS_MASK) >> STATUS_SHIFT);
    }

    void set_status(uint8_t status) {
        assert(status < 8);
        uint64_t expected;
        uint64_t desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~STATUS_MASK) |
                      (static_cast<uint64_t>(status) << STATUS_SHIFT);
        } while (!atomic_value.compare_exchange_weak(expected, desired,
                                                      std::memory_order_relaxed));
    }

    // === 保留: 与upstream完全相同的get/set_timestamp ===
    uint64_t get_timestamp() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return value & TIMESTAMP_MASK;
    }

    void set_timestamp(uint64_t timestamp) {
        assert(timestamp < (1ULL << 60));
        uint64_t expected;
        uint64_t desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~TIMESTAMP_MASK) | (timestamp & TIMESTAMP_MASK);
        } while (!atomic_value.compare_exchange_weak(expected, desired,
                                                      std::memory_order_relaxed));
    }

    // === 保留: 与upstream完全相同的clear() ===
    void clear() {
        atomic_value.store(0, std::memory_order_relaxed);
    }

    // [NEW] Tier accessors — 20%修改部分
    uint8_t get_tier() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return static_cast<uint8_t>((value & TIER_MASK) >> TIER_SHIFT);
    }

    void set_tier(uint8_t tier) {
        assert(tier < 8);
        uint64_t expected;
        uint64_t desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~TIER_MASK) |
                      (static_cast<uint64_t>(tier) << TIER_SHIFT);
        } while (!atomic_value.compare_exchange_weak(expected, desired,
                                                      std::memory_order_relaxed));
    }

    // [NEW] Access count — 用于频率预测
    uint16_t get_access_count() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return static_cast<uint16_t>((value & COUNT_MASK) >> COUNT_SHIFT);
    }

    void increment_access_count() {
        uint64_t expected;
        uint64_t desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            uint16_t count = static_cast<uint16_t>(
                (expected & COUNT_MASK) >> COUNT_SHIFT);
            if (count < 0xFFFF) count++;
            desired = (expected & ~COUNT_MASK) |
                      (static_cast<uint64_t>(count) << COUNT_SHIFT);
        } while (!atomic_value.compare_exchange_weak(expected, desired,
                                                      std::memory_order_relaxed));
    }

    // [NEW] Vertex tracking — 最近访问的顶点
    void set_last_vertex(uint64_t vid) {
        last_vertex_id_.store(vid, std::memory_order_relaxed);
    }
    uint64_t get_last_vertex() const {
        return last_vertex_id_.load(std::memory_order_relaxed);
    }

    // [NEW] Prefetch hit/miss tracking
    void record_hit()  { prefetch_hits_.fetch_add(1, std::memory_order_relaxed); }
    void record_miss() { prefetch_misses_.fetch_add(1, std::memory_order_relaxed); }
    uint32_t hits()  const { return prefetch_hits_.load(std::memory_order_relaxed); }
    uint32_t misses() const { return prefetch_misses_.load(std::memory_order_relaxed); }
    double hit_rate() const {
        uint32_t h = hits(), m = misses();
        return (h + m > 0) ? static_cast<double>(h) / (h + m) : 0.0;
    }

    // [NEW] 断点调试: 打印当前block完整状态
    void dump(const char* tag = "PrefetchTraceBlock") const {
        std::printf("  [%s] status=%u ts=%lu tier=%u count=%u "
                    "vertex=%lu hits=%u misses=%u hit_rate=%.2f%%\n",
                    tag,
                    (unsigned)get_status(),
                    (unsigned long)get_timestamp(),
                    (unsigned)get_tier(),
                    (unsigned)get_access_count(),
                    (unsigned long)get_last_vertex(),
                    (unsigned)hits(), (unsigned)misses(),
                    hit_rate() * 100.0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// QueryHistoryEntry — 查询历史记录条目
// ═══════════════════════════════════════════════════════════════════════
struct QueryHistoryEntry {
    uint64_t vertex_id;
    uint8_t  tier;
    uint8_t  query_type;   // 0=BFS, 1=SSSP, 2=PR, 3=WCC, 4=TC
    uint64_t timestamp;
    uint64_t partition_id;
    double   latency_ns;   // 实际查询延迟

    void dump() const {
        const char* qtype[] = {"BFS", "SSSP", "PR", "WCC", "TC"};
        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        std::printf("    vertex=%lu tier=%s type=%s ts=%lu "
                    "part=%lu lat=%.2fμs\n",
                    (unsigned long)vertex_id,
                    tname[tier % 3],
                    qtype[query_type % 5],
                    (unsigned long)timestamp,
                    (unsigned long)partition_id,
                    latency_ns / 1000.0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// QueryHistoryRing — 环形缓冲, 记录最近N次查询
// 骨架: neo_reader_trace.h::ActiveReaderTracer 的固定大小数组
// 修改: 改为环形覆盖写, 增加频率统计
// ═══════════════════════════════════════════════════════════════════════
class QueryHistoryRing {
private:
    std::vector<QueryHistoryEntry> ring_;
    std::atomic<uint64_t> head_{0};       // 下一个写入位置
    std::atomic<uint64_t> count_{0};      // 总写入数
    size_t capacity_;
    mutable std::shared_mutex mu_;

    // [NEW] 频率表: vertex_id → 近期访问次数
    std::unordered_map<uint64_t, uint32_t> freq_table_;
    std::mutex freq_mu_;

public:
    explicit QueryHistoryRing(size_t capacity = 8192)
        : ring_(capacity), capacity_(capacity) {}

    // 记录一次查询 — 类似 neo_reader_trace::reader_register() 的写入
    void record(uint64_t vertex_id, uint8_t tier, uint8_t query_type,
                uint64_t timestamp, uint64_t partition_id, double latency_ns) {
        uint64_t idx = head_.fetch_add(1, std::memory_order_relaxed) % capacity_;

        QueryHistoryEntry entry;
        entry.vertex_id    = vertex_id;
        entry.tier         = tier;
        entry.query_type   = query_type;
        entry.timestamp    = timestamp;
        entry.partition_id = partition_id;
        entry.latency_ns   = latency_ns;

        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            ring_[idx] = entry;
        }
        count_.fetch_add(1, std::memory_order_relaxed);

        // 更新频率表
        {
            std::lock_guard<std::mutex> flock(freq_mu_);
            freq_table_[vertex_id]++;
        }
    }

    // 获取最近n条记录 — 用于断点调试
    std::vector<QueryHistoryEntry> recent(size_t n) const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::vector<QueryHistoryEntry> result;
        uint64_t total = count_.load(std::memory_order_relaxed);
        size_t actual = std::min(n, std::min(total, capacity_));
        uint64_t start = head_.load(std::memory_order_relaxed);

        for (size_t i = 0; i < actual; i++) {
            uint64_t idx = (start - 1 - i + capacity_ * 2) % capacity_;
            result.push_back(ring_[idx]);
        }
        return result;
    }

    // 获取top-K高频顶点 — 用于预取决策
    std::vector<std::pair<uint64_t, uint32_t>> top_k_vertices(size_t k) const {
        std::lock_guard<std::mutex> flock(
            const_cast<std::mutex&>(freq_mu_));
        std::vector<std::pair<uint64_t, uint32_t>> sorted_freq(
            freq_table_.begin(), freq_table_.end());
        std::sort(sorted_freq.begin(), sorted_freq.end(),
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;
                  });
        if (sorted_freq.size() > k) sorted_freq.resize(k);
        return sorted_freq;
    }

    // 查询某vertex的访问频率
    uint32_t vertex_frequency(uint64_t vertex_id) const {
        std::lock_guard<std::mutex> flock(
            const_cast<std::mutex&>(freq_mu_));
        auto it = freq_table_.find(vertex_id);
        return (it != freq_table_.end()) ? it->second : 0;
    }

    // 频率衰减 — 每个周期将所有频率减半
    void decay_frequencies() {
        std::lock_guard<std::mutex> flock(freq_mu_);
        for (auto it = freq_table_.begin(); it != freq_table_.end(); ) {
            it->second >>= 1;
            if (it->second == 0) {
                it = freq_table_.erase(it);
            } else {
                ++it;
            }
        }
        PHILE_DBG(2, "[QueryHistoryRing] frequency decay: %zu vertices remain",
                  freq_table_.size());
    }

    uint64_t total_queries() const {
        return count_.load(std::memory_order_relaxed);
    }

    // [断点调试] 打印最近n条历史
    void dump(size_t n = 10) const {
        auto entries = recent(n);
        std::printf("══════ QueryHistoryRing (total=%lu, showing %zu) ══════\n",
                    (unsigned long)total_queries(), entries.size());
        for (const auto& e : entries) {
            e.dump();
        }
        std::printf("══════ End QueryHistory ══════\n");
    }

    // [断点调试] 打印频率表top-K
    void dump_top_k(size_t k = 20) const {
        auto top = top_k_vertices(k);
        std::printf("══════ Top-%zu Vertex Frequencies ══════\n", k);
        for (size_t i = 0; i < top.size(); i++) {
            std::printf("  #%zu: vertex=%lu freq=%u\n",
                        i + 1, (unsigned long)top[i].first,
                        (unsigned)top[i].second);
        }
        std::printf("══════ End Top-K ══════\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchTicket — 异步预取请求跟踪
// 模式: 类似 AsyncMigrator::MigrationTicket
// ═══════════════════════════════════════════════════════════════════════
struct PrefetchTicket {
    enum class Status : uint8_t {
        PENDING   = 0,
        ACTIVE    = 1,
        COMPLETED = 2,
        CANCELLED = 3,
        FAILED    = 4
    };

    uint64_t ticket_id;
    uint64_t partition_id;
    uint8_t  from_tier;
    uint8_t  to_tier;
    uint64_t bytes;
    std::atomic<Status> status{Status::PENDING};
    double   predicted_benefit_ns;
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point complete_time;

    PrefetchTicket() = default;

    // Explicit copy (std::atomic is non-copyable)
    PrefetchTicket(const PrefetchTicket& o)
        : ticket_id(o.ticket_id), partition_id(o.partition_id)
        , from_tier(o.from_tier), to_tier(o.to_tier), bytes(o.bytes)
        , predicted_benefit_ns(o.predicted_benefit_ns)
        , submit_time(o.submit_time), complete_time(o.complete_time) {
        status.store(o.status.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    }
    PrefetchTicket& operator=(const PrefetchTicket& o) {
        if (this != &o) {
            ticket_id = o.ticket_id; partition_id = o.partition_id;
            from_tier = o.from_tier; to_tier = o.to_tier; bytes = o.bytes;
            predicted_benefit_ns = o.predicted_benefit_ns;
            submit_time = o.submit_time; complete_time = o.complete_time;
            status.store(o.status.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        }
        return *this;
    }

    double elapsed_us() const {
        auto end = (status.load() == Status::COMPLETED) ?
                   complete_time : std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end - submit_time).count();
    }

    void dump() const {
        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        const char* sname[] = {"PENDING", "ACTIVE", "COMPLETED",
                               "CANCELLED", "FAILED"};
        std::printf("  [PrefetchTicket #%lu] part=%lu %s→%s %luKB "
                    "status=%s benefit=%.1fμs elapsed=%.1fμs\n",
                    (unsigned long)ticket_id,
                    (unsigned long)partition_id,
                    tname[from_tier % 3], tname[to_tier % 3],
                    (unsigned long)(bytes / 1024),
                    sname[static_cast<int>(status.load()) % 5],
                    predicted_benefit_ns / 1000.0,
                    elapsed_us());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchPredictor — 查询预测引擎
// 骨架: driver.h::execute_query() 的查询类型分发
//       + neo_reader_trace::get_min_timestamp() 的全局扫描
// 修改: 基于历史频率预测下一次需要预取的partition
// ═══════════════════════════════════════════════════════════════════════
class PrefetchPredictor {
private:
    const QueryHistoryRing& history_;
    const cost_model::TierCostModel& cost_model_;

    // 预取决策阈值
    double   min_benefit_threshold_ns_;   // 最小收益阈值
    uint32_t min_frequency_threshold_;    // 最小频率阈值
    double   confidence_threshold_;       // 预测置信度阈值

    // 自适应阈值统计
    std::atomic<uint64_t> correct_predictions_{0};
    std::atomic<uint64_t> total_predictions_{0};

public:
    PrefetchPredictor(const QueryHistoryRing& history,
                      const cost_model::TierCostModel& cost_model,
                      double min_benefit_ns = 5000.0,       // 5μs
                      uint32_t min_frequency = 3,
                      double confidence = 0.6)
        : history_(history)
        , cost_model_(cost_model)
        , min_benefit_threshold_ns_(min_benefit_ns)
        , min_frequency_threshold_(min_frequency)
        , confidence_threshold_(confidence) {}

    // ─── 预测: 给定当前分区分布, 哪些partition值得预取? ─────
    // partitions: (partition_id, current_tier, size_bytes, access_freq)
    // 返回: 需要预取的partition列表 + 目标tier + 预计收益
    struct PrefetchRecommendation {
        uint64_t partition_id;
        uint8_t  from_tier;
        uint8_t  to_tier;
        uint64_t bytes;
        double   benefit_ns;
        double   confidence;

        void dump() const {
            const char* tname[] = {"HBM", "GDDR", "DRAM"};
            std::printf("    [Recommend] part=%lu %s→%s %luKB "
                        "benefit=%.1fμs conf=%.2f\n",
                        (unsigned long)partition_id,
                        tname[from_tier % 3], tname[to_tier % 3],
                        (unsigned long)(bytes / 1024),
                        benefit_ns / 1000.0, confidence);
        }
    };

    std::vector<PrefetchRecommendation> predict(
        const std::vector<std::tuple<uint64_t, uint8_t, uint64_t, double>>& partitions)
    {
        debug::ScopedTimer timer("PrefetchPredictor::predict");
        std::vector<PrefetchRecommendation> recs;

        // 获取top-K高频顶点
        auto top_vertices = history_.top_k_vertices(64);

        // 对每个partition评估预取收益
        for (const auto& [part_id, cur_tier, bytes, freq] : partitions) {
            // 跳过已在HBM的partition
            if (cur_tier == 0) continue;

            // 频率检查
            if (freq < min_frequency_threshold_) continue;

            // 计算预取收益: 从cold_tier迁移到HBM能省多少时间
            uint64_t expected_accesses = static_cast<uint64_t>(freq * 10);
            double benefit = cost_model_.prefetch_benefit_ns(
                cur_tier, 0 /* HBM */, bytes, expected_accesses);

            if (benefit < min_benefit_threshold_ns_) continue;

            // 置信度: 基于历史准确率 + 频率稳定性
            double conf = compute_confidence(freq, part_id);
            if (conf < confidence_threshold_) continue;

            PrefetchRecommendation rec;
            rec.partition_id = part_id;
            rec.from_tier    = cur_tier;
            rec.to_tier      = 0;  // 总是预取到HBM
            rec.bytes        = bytes;
            rec.benefit_ns   = benefit;
            rec.confidence   = conf;
            recs.push_back(rec);
        }

        // 按收益排序
        std::sort(recs.begin(), recs.end(),
                  [](const auto& a, const auto& b) {
                      return a.benefit_ns > b.benefit_ns;
                  });

        total_predictions_.fetch_add(recs.size(), std::memory_order_relaxed);

        PHILE_DBG(1, "[PrefetchPredictor] evaluated %zu partitions, "
                  "recommended %zu prefetches",
                  partitions.size(), recs.size());

        return recs;
    }

    // 反馈: 标记预测是否正确 (实际访问了预取的partition)
    void feedback(bool was_correct) {
        if (was_correct) {
            correct_predictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // 自适应阈值调整
    void adapt_thresholds() {
        uint64_t total = total_predictions_.load(std::memory_order_relaxed);
        uint64_t correct = correct_predictions_.load(std::memory_order_relaxed);

        if (total < 10) return;  // 样本太少

        double accuracy = static_cast<double>(correct) / total;

        if (accuracy < 0.3) {
            // 准确率太低 → 收紧阈值
            min_benefit_threshold_ns_ *= 1.5;
            min_frequency_threshold_  = std::min(min_frequency_threshold_ + 2,
                                                  (uint32_t)20);
            confidence_threshold_     = std::min(confidence_threshold_ + 0.1, 0.95);
            PHILE_DBG(1, "[adapt] accuracy=%.1f%% → TIGHTER: "
                      "min_benefit=%.0fns min_freq=%u conf=%.2f",
                      accuracy * 100.0, min_benefit_threshold_ns_,
                      (unsigned)min_frequency_threshold_,
                      confidence_threshold_);
        } else if (accuracy > 0.8) {
            // 准确率高 → 放松阈值,预取更多
            min_benefit_threshold_ns_ *= 0.8;
            if (min_frequency_threshold_ > 1) min_frequency_threshold_--;
            confidence_threshold_ = std::max(confidence_threshold_ - 0.05, 0.3);
            PHILE_DBG(1, "[adapt] accuracy=%.1f%% → LOOSER: "
                      "min_benefit=%.0fns min_freq=%u conf=%.2f",
                      accuracy * 100.0, min_benefit_threshold_ns_,
                      (unsigned)min_frequency_threshold_,
                      confidence_threshold_);
        }

        // 重置统计 (滑动窗口)
        correct_predictions_.store(0, std::memory_order_relaxed);
        total_predictions_.store(0, std::memory_order_relaxed);
    }

    // [断点调试] 完整状态打印
    void dump() const {
        uint64_t total = total_predictions_.load();
        uint64_t correct = correct_predictions_.load();
        std::printf("══════ PrefetchPredictor ══════\n");
        std::printf("  min_benefit=%.1fμs min_freq=%u confidence=%.2f\n",
                    min_benefit_threshold_ns_ / 1000.0,
                    (unsigned)min_frequency_threshold_,
                    confidence_threshold_);
        std::printf("  predictions: total=%lu correct=%lu accuracy=%.1f%%\n",
                    (unsigned long)total, (unsigned long)correct,
                    (total > 0) ? (100.0 * correct / total) : 0.0);
        std::printf("══════ End Predictor ══════\n");
    }

private:
    double compute_confidence(double freq, uint64_t partition_id) const {
        // 置信度 = 频率归一化 × 历史准确率
        double freq_conf = std::min(freq / 20.0, 1.0);
        uint64_t total = total_predictions_.load();
        uint64_t correct = correct_predictions_.load();
        double acc_conf = (total > 5) ?
                          static_cast<double>(correct) / total : 0.5;
        return freq_conf * 0.6 + acc_conf * 0.4;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchTracer — 从 ActiveReaderTracer 移植
// 保留: blocks数组 + register/unregister 模式 (100%)
// 保留: reader_register() 的 CAS扫描 loop (100%)
// 保留: get_active_reader_info() 的全局block扫描 (100%)
// 修改: Block类型从 ReaderTraceBlock → PrefetchTraceBlock
// 修改: 增加 prefetch-specific 统计方法
// ═══════════════════════════════════════════════════════════════════════
static constexpr size_t MAX_PREFETCH_SLOTS = 64;

class PrefetchTracer {
private:
    // 保留: 与ActiveReaderTracer完全相同的固定大小block数组
    std::array<PrefetchTraceBlock, MAX_PREFETCH_SLOTS> blocks_;

public:
    // === 保留: 与upstream reader_register()完全相同的CAS注册逻辑 ===
    // 原始: neo_reader_trace.cpp:7-14
    PrefetchTraceBlock* prefetch_register() {
        for (auto& block : blocks_) {
            if (block.get_status() == 0 && block.try_lock()) {
                block.set_status(1);  // acquiring
                return &block;
            }
        }
        return nullptr;  // 所有slot都被占用
    }

    // === 保留: 与upstream set_status相同 ===
    void set_status(PrefetchTraceBlock* block, uint64_t status) {
        block->set_status(1);
    }

    // === 保留: 与upstream set_timestamp相同 ===
    void set_timestamp(PrefetchTraceBlock* block, uint64_t timestamp) {
        block->set_timestamp(timestamp);
        block->unlock();
    }

    // === 保留: 与upstream reader_unregister相同 ===
    void prefetch_unregister(PrefetchTraceBlock* block) {
        block->lock();
        block->clear();
    }

    // === 保留: 与upstream get_active_reader_info()完全相同的扫描逻辑 ===
    // 原始: neo_reader_trace.cpp:30-40
    void get_active_prefetch_info(std::vector<uint64_t>& timestamps) {
        for (auto& block : blocks_) {
            if (block.get_status() == 1) {
                uint64_t timestamp;
                do {
                    timestamp = block.get_timestamp();
                } while (timestamp == 0);
                timestamps.push_back(timestamp);
            }
        }
        std::sort(timestamps.begin(), timestamps.end());
        timestamps.erase(std::unique(timestamps.begin(), timestamps.end()),
                         timestamps.end());
    }

    // === 保留: 与upstream get_min_timestamp()完全相同 ===
    // 原始: neo_reader_trace.cpp:42-55
    uint64_t get_min_timestamp() {
        uint64_t min_timestamp = std::numeric_limits<uint64_t>::max();
        for (auto& block : blocks_) {
            if (block.get_status() == 1) {
                block.lock();
                auto timestamp = block.get_timestamp();
                block.unlock();
                if (timestamp < min_timestamp) {
                    min_timestamp = timestamp;
                }
            }
        }
        return min_timestamp;
    }

    // [NEW] 获取所有活跃预取的tier分布
    void get_tier_distribution(uint64_t counts[3]) const {
        counts[0] = counts[1] = counts[2] = 0;
        for (const auto& block : blocks_) {
            if (block.get_status() == 1) {
                uint8_t tier = block.get_tier();
                if (tier < 3) counts[tier]++;
            }
        }
    }

    // [NEW] 获取全局命中率
    double global_hit_rate() const {
        uint64_t total_hits = 0, total_misses = 0;
        for (const auto& block : blocks_) {
            total_hits   += block.hits();
            total_misses += block.misses();
        }
        uint64_t total = total_hits + total_misses;
        return (total > 0) ? static_cast<double>(total_hits) / total : 0.0;
    }

    // [断点调试] 打印所有活跃块状态
    void dump() const {
        std::printf("══════ PrefetchTracer (slots=%zu) ══════\n",
                    MAX_PREFETCH_SLOTS);
        uint64_t tier_counts[3] = {0, 0, 0};
        int active = 0;
        for (size_t i = 0; i < MAX_PREFETCH_SLOTS; i++) {
            if (blocks_[i].get_status() == 1) {
                std::printf("  slot[%zu]: ", i);
                blocks_[i].dump();
                uint8_t tier = blocks_[i].get_tier();
                if (tier < 3) tier_counts[tier]++;
                active++;
            }
        }
        std::printf("  active=%d  tiers: HBM=%lu GDDR=%lu DRAM=%lu  "
                    "global_hit_rate=%.1f%%\n",
                    active,
                    (unsigned long)tier_counts[0],
                    (unsigned long)tier_counts[1],
                    (unsigned long)tier_counts[2],
                    global_hit_rate() * 100.0);
        std::printf("══════ End PrefetchTracer ══════\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchScheduler — 预取事务调度器
// 骨架: TransactionManager 的 write_timestamp/read_timestamp + CAS
//       + Driver::Barrier (arrive_and_wait)
//       + execute_mixed_reader_writer() 的并发reader/writer线程模式
// 修改: 改为prefetch请求队列 + 后台工作线程
// ═══════════════════════════════════════════════════════════════════════
class PrefetchScheduler {
private:
    // === 保留: 与TransactionManager相同的原子时间戳 ===
    std::atomic<uint64_t> write_timestamp_{0};
    std::atomic<uint64_t> read_timestamp_{0};

    // 预取请求队列
    std::queue<PrefetchTicket> pending_queue_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;

    // 已完成的ticket (保留以供查询)
    std::vector<PrefetchTicket> completed_;
    std::mutex completed_mu_;

    // 后台工作线程
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    size_t num_workers_;

    // === 保留: 与driver.h::Barrier完全相同 ===
    // 原始: driver.h:50-65
    class Barrier {
    public:
        explicit Barrier(std::size_t count) : count_(count), waiting_(0) {}
        void arrive_and_wait() {
            std::unique_lock<std::mutex> lock(mtx_);
            ++waiting_;
            if (waiting_ == count_) {
                waiting_ = 0;
                cv_.notify_all();
            } else {
                cv_.wait(lock, [this] { return waiting_ == 0; });
            }
        }
    private:
        std::mutex mtx_;
        std::condition_variable cv_;
        std::size_t count_;
        std::size_t waiting_;
    };

    // 统计
    std::atomic<uint64_t> total_submitted_{0};
    std::atomic<uint64_t> total_completed_{0};
    std::atomic<uint64_t> total_cancelled_{0};
    std::atomic<uint64_t> total_bytes_prefetched_{0};

    // 迁移回调: 实际执行tier-to-tier数据移动
    // 签名: (partition_id, from_tier, to_tier, bytes) → success
    using MigrateCallback = std::function<bool(uint64_t, uint8_t, uint8_t, uint64_t)>;
    MigrateCallback migrate_fn_;

public:
    explicit PrefetchScheduler(size_t num_workers = 2)
        : num_workers_(num_workers) {}

    ~PrefetchScheduler() { stop(); }

    // 设置迁移回调 (连接到TieredAllocator::migrate)
    void set_migrate_callback(MigrateCallback fn) {
        migrate_fn_ = std::move(fn);
    }

    // === 保留: 与TransactionManager::get_write_timestamp()相同的fetch_add ===
    uint64_t get_write_timestamp() {
        return write_timestamp_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    // === 保留: 与TransactionManager::finish_commit()相同的CAS loop ===
    void finish_commit(uint64_t timestamp) {
        auto target = timestamp - 1;
        while (!read_timestamp_.compare_exchange_weak(
                   target, timestamp, std::memory_order_relaxed)) {
            target = timestamp - 1;  // 重置target
        }
    }

    // 提交预取请求
    uint64_t submit(uint64_t partition_id, uint8_t from_tier,
                    uint8_t to_tier, uint64_t bytes,
                    double predicted_benefit_ns) {
        PrefetchTicket ticket;
        ticket.ticket_id = get_write_timestamp();
        ticket.partition_id = partition_id;
        ticket.from_tier    = from_tier;
        ticket.to_tier      = to_tier;
        ticket.bytes        = bytes;
        ticket.status.store(PrefetchTicket::Status::PENDING);
        ticket.predicted_benefit_ns = predicted_benefit_ns;
        ticket.submit_time = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(queue_mu_);
            pending_queue_.push(ticket);
        }
        queue_cv_.notify_one();

        total_submitted_.fetch_add(1, std::memory_order_relaxed);

        PHILE_DBG(2, "[PrefetchScheduler] submitted ticket #%lu: "
                  "part=%lu %u→%u %luKB",
                  (unsigned long)ticket.ticket_id,
                  (unsigned long)partition_id,
                  (unsigned)from_tier, (unsigned)to_tier,
                  (unsigned long)(bytes / 1024));

        return ticket.ticket_id;
    }

    // 启动后台工作线程
    // === 保留: driver.h::execute_mixed_reader_writer的线程创建+bind模式 ===
    void start() {
        if (running_.load()) return;
        running_.store(true);

        for (size_t i = 0; i < num_workers_; i++) {
            workers_.emplace_back([this, i]() {
                worker_loop(i);
            });

            // === 保留: 与driver.h::bind_thread_to_core相同 ===
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i % std::thread::hardware_concurrency(), &cpuset);
            int rc = pthread_setaffinity_np(workers_.back().native_handle(),
                                            sizeof(cpu_set_t), &cpuset);
            if (rc != 0) {
                PHILE_DBG(1, "[PrefetchScheduler] warning: "
                          "bind_thread_to_core failed for worker %zu", i);
            }
        }

        PHILE_DBG(1, "[PrefetchScheduler] started %zu workers", num_workers_);
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        queue_cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
        PHILE_DBG(1, "[PrefetchScheduler] stopped. "
                  "submitted=%lu completed=%lu cancelled=%lu "
                  "bytes=%luMB",
                  (unsigned long)total_submitted_.load(),
                  (unsigned long)total_completed_.load(),
                  (unsigned long)total_cancelled_.load(),
                  (unsigned long)(total_bytes_prefetched_.load() / (1024*1024)));
    }

    // 查询ticket状态
    PrefetchTicket::Status query_status(uint64_t ticket_id) const {
        std::lock_guard<std::mutex> lock(
            const_cast<std::mutex&>(completed_mu_));
        for (const auto& t : completed_) {
            if (t.ticket_id == ticket_id)
                return t.status.load();
        }
        return PrefetchTicket::Status::PENDING;
    }

    // [断点调试] 完整状态快照
    void dump() const {
        std::printf("══════ PrefetchScheduler ══════\n");
        std::printf("  workers=%zu running=%s\n",
                    num_workers_,
                    running_.load() ? "true" : "false");
        std::printf("  write_ts=%lu read_ts=%lu\n",
                    (unsigned long)write_timestamp_.load(),
                    (unsigned long)read_timestamp_.load());
        std::printf("  submitted=%lu completed=%lu cancelled=%lu\n",
                    (unsigned long)total_submitted_.load(),
                    (unsigned long)total_completed_.load(),
                    (unsigned long)total_cancelled_.load());
        std::printf("  bytes_prefetched=%luMB\n",
                    (unsigned long)(total_bytes_prefetched_.load() /
                                   (1024 * 1024)));

        // 打印pending queue大小
        size_t pending;
        {
            std::lock_guard<std::mutex> lock(
                const_cast<std::mutex&>(queue_mu_));
            pending = const_cast<std::queue<PrefetchTicket>&>(
                pending_queue_).size();
        }
        std::printf("  pending_queue=%zu\n", pending);

        // 打印最近5个completed
        {
            std::lock_guard<std::mutex> lock(
                const_cast<std::mutex&>(completed_mu_));
            size_t show = std::min(completed_.size(), (size_t)5);
            if (show > 0) {
                std::printf("  recent completed:\n");
                for (size_t i = completed_.size() - show;
                     i < completed_.size(); i++) {
                    completed_[i].dump();
                }
            }
        }
        std::printf("══════ End PrefetchScheduler ══════\n");
    }

private:
    void worker_loop(size_t worker_id) {
        PHILE_DBG(2, "[PrefetchWorker %zu] started", worker_id);

        while (running_.load()) {
            PrefetchTicket ticket;
            {
                std::unique_lock<std::mutex> lock(queue_mu_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100),
                                   [this] {
                                       return !pending_queue_.empty() ||
                                              !running_.load();
                                   });
                if (!running_.load()) break;
                if (pending_queue_.empty()) continue;
                ticket = pending_queue_.front();
                pending_queue_.pop();
            }

            // 执行预取
            ticket.status.store(PrefetchTicket::Status::ACTIVE);
            PHILE_DBG(2, "[PrefetchWorker %zu] executing ticket #%lu",
                      worker_id, (unsigned long)ticket.ticket_id);

            bool success = false;
            if (migrate_fn_) {
                success = migrate_fn_(ticket.partition_id,
                                      ticket.from_tier,
                                      ticket.to_tier,
                                      ticket.bytes);
            } else {
                // 模拟预取 (无实际迁移回调时)
                std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<int>(ticket.bytes / 1024)));
                success = true;
            }

            ticket.complete_time = std::chrono::steady_clock::now();
            ticket.status.store(success ?
                PrefetchTicket::Status::COMPLETED :
                PrefetchTicket::Status::FAILED);

            if (success) {
                total_completed_.fetch_add(1, std::memory_order_relaxed);
                total_bytes_prefetched_.fetch_add(ticket.bytes,
                    std::memory_order_relaxed);
                finish_commit(ticket.ticket_id);
            }

            // 保留结果
            {
                std::lock_guard<std::mutex> lock(completed_mu_);
                completed_.push_back(ticket);
                // 保留最多1000个结果
                if (completed_.size() > 1000) {
                    completed_.erase(completed_.begin(),
                                     completed_.begin() + 500);
                }
            }

            PHILE_DBG(2, "[PrefetchWorker %zu] ticket #%lu %s in %.1fμs",
                      worker_id,
                      (unsigned long)ticket.ticket_id,
                      success ? "COMPLETED" : "FAILED",
                      ticket.elapsed_us());
        }

        PHILE_DBG(2, "[PrefetchWorker %zu] stopped", worker_id);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchEngine — 顶层预取引擎, 组合所有组件
// 骨架: driver.h::Driver 类的 m_method + m_num_threads + execute()
//       + execute_query() 的 for-loop-over-operation-types
// 修改: 集成QueryHistory + Predictor + Scheduler为一体
// ═══════════════════════════════════════════════════════════════════════
class PrefetchEngine {
private:
    QueryHistoryRing  history_;
    PrefetchTracer    tracer_;
    PrefetchPredictor predictor_;
    PrefetchScheduler scheduler_;

    // 配置
    size_t    max_prefetch_batch_;       // 每轮最多预取多少个partition
    uint64_t  prefetch_interval_ms_;     // 预取检查间隔
    bool      adaptive_thresholds_;      // 是否启用自适应阈值

    // 后台预取线程
    std::thread prefetch_thread_;
    std::atomic<bool> running_{false};

    // 周期统计
    std::atomic<uint64_t> cycle_count_{0};
    std::atomic<uint64_t> total_prefetch_requests_{0};
    std::atomic<uint64_t> total_prefetch_bytes_{0};

    // 分区状态获取回调
    using PartitionStateCallback = std::function<
        std::vector<std::tuple<uint64_t, uint8_t, uint64_t, double>>()>;
    PartitionStateCallback get_partition_state_;

public:
    PrefetchEngine(const cost_model::TierCostModel& cost_model,
                   size_t history_capacity = 8192,
                   size_t max_batch = 16,
                   uint64_t interval_ms = 500,
                   size_t num_workers = 2)
        : history_(history_capacity)
        , predictor_(history_, cost_model)
        , scheduler_(num_workers)
        , max_prefetch_batch_(max_batch)
        , prefetch_interval_ms_(interval_ms)
        , adaptive_thresholds_(true) {}

    ~PrefetchEngine() { stop(); }

    // ─── 初始化 ─────────────────────────────────────────────────
    void set_partition_state_callback(PartitionStateCallback fn) {
        get_partition_state_ = std::move(fn);
    }

    void set_migrate_callback(
        std::function<bool(uint64_t, uint8_t, uint8_t, uint64_t)> fn) {
        scheduler_.set_migrate_callback(std::move(fn));
    }

    // ─── 生命周期 ───────────────────────────────────────────────
    void start() {
        if (running_.load()) return;
        running_.store(true);
        scheduler_.start();
        prefetch_thread_ = std::thread([this]() { prefetch_loop(); });
        PHILE_DBG(1, "[PrefetchEngine] started (interval=%lums batch=%zu)",
                  (unsigned long)prefetch_interval_ms_, max_prefetch_batch_);
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (prefetch_thread_.joinable()) prefetch_thread_.join();
        scheduler_.stop();
        PHILE_DBG(1, "[PrefetchEngine] stopped after %lu cycles, "
                  "%lu prefetches, %luMB",
                  (unsigned long)cycle_count_.load(),
                  (unsigned long)total_prefetch_requests_.load(),
                  (unsigned long)(total_prefetch_bytes_.load() / (1024*1024)));
    }

    // ─── 查询hook: 在每次查询前/后调用 ─────────────────────────
    // 模式: 类似 execute_query() 中每个查询的执行入口

    // 查询前: 检查是否有预取命中
    bool check_prefetch_hit(uint64_t vertex_id, uint64_t partition_id) {
        auto* block = tracer_.prefetch_register();
        if (!block) return false;

        block->set_last_vertex(vertex_id);
        block->set_tier(0);  // 假设命中HBM

        // 检查该partition是否已被预取
        bool hit = (scheduler_.query_status(partition_id) ==
                    PrefetchTicket::Status::COMPLETED);
        if (hit) {
            block->record_hit();
            predictor_.feedback(true);
        } else {
            block->record_miss();
            predictor_.feedback(false);
        }

        tracer_.prefetch_unregister(block);

        PHILE_DBG(3, "[PrefetchEngine] check_hit vertex=%lu part=%lu → %s",
                  (unsigned long)vertex_id, (unsigned long)partition_id,
                  hit ? "HIT" : "MISS");
        return hit;
    }

    // 查询后: 记录历史
    void record_query(uint64_t vertex_id, uint8_t tier, uint8_t query_type,
                      uint64_t timestamp, uint64_t partition_id,
                      double latency_ns) {
        history_.record(vertex_id, tier, query_type, timestamp,
                        partition_id, latency_ns);

        PHILE_DBG(3, "[PrefetchEngine] recorded: vertex=%lu tier=%u "
                  "type=%u lat=%.1fμs",
                  (unsigned long)vertex_id, (unsigned)tier,
                  (unsigned)query_type, latency_ns / 1000.0);
    }

    // ─── 访问器 ─────────────────────────────────────────────────
    const QueryHistoryRing&  history()   const { return history_; }
    const PrefetchTracer&    tracer()    const { return tracer_; }
    const PrefetchPredictor& predictor() const { return predictor_; }
    const PrefetchScheduler& scheduler() const { return scheduler_; }

    // ─── [断点调试] 全量状态打印 ────────────────────────────────
    void dump_all() const {
        std::printf("╔══════════════════════════════════════════════╗\n");
        std::printf("║          PrefetchEngine Full State          ║\n");
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("  running=%s cycles=%lu interval=%lums batch=%zu\n",
                    running_.load() ? "true" : "false",
                    (unsigned long)cycle_count_.load(),
                    (unsigned long)prefetch_interval_ms_,
                    max_prefetch_batch_);
        std::printf("  total_prefetches=%lu total_bytes=%luMB\n",
                    (unsigned long)total_prefetch_requests_.load(),
                    (unsigned long)(total_prefetch_bytes_.load() /
                                   (1024 * 1024)));
        std::printf("\n");
        history_.dump(10);
        std::printf("\n");
        history_.dump_top_k(10);
        std::printf("\n");
        tracer_.dump();
        std::printf("\n");
        predictor_.dump();
        std::printf("\n");
        scheduler_.dump();
        std::printf("╚══════════════════════════════════════════════╝\n");
    }

private:
    void prefetch_loop() {
        PHILE_DBG(1, "[PrefetchEngine] prefetch_loop started");

        while (running_.load()) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(prefetch_interval_ms_));
            if (!running_.load()) break;

            cycle_count_.fetch_add(1, std::memory_order_relaxed);
            uint64_t cycle = cycle_count_.load();

            PHILE_DBG(2, "[PrefetchEngine] cycle %lu begin", (unsigned long)cycle);

            // 获取当前分区状态
            if (!get_partition_state_) continue;
            auto partitions = get_partition_state_();

            if (partitions.empty()) {
                PHILE_DBG(2, "[PrefetchEngine] no partitions, skip");
                continue;
            }

            // 获取预测
            auto recommendations = predictor_.predict(partitions);

            // 提交预取请求 (限制batch大小)
            size_t submitted = 0;
            for (const auto& rec : recommendations) {
                if (submitted >= max_prefetch_batch_) break;

                scheduler_.submit(rec.partition_id, rec.from_tier,
                                  rec.to_tier, rec.bytes, rec.benefit_ns);
                submitted++;

                total_prefetch_requests_.fetch_add(1, std::memory_order_relaxed);
                total_prefetch_bytes_.fetch_add(rec.bytes,
                    std::memory_order_relaxed);
            }

            PHILE_DBG(2, "[PrefetchEngine] cycle %lu: %zu recommended, "
                      "%zu submitted",
                      (unsigned long)cycle, recommendations.size(), submitted);

            // 每10个周期做一次频率衰减
            if (cycle % 10 == 0) {
                history_.decay_frequencies();
            }

            // 每20个周期做一次自适应阈值调整
            if (adaptive_thresholds_ && cycle % 20 == 0) {
                const_cast<PrefetchPredictor&>(predictor_).adapt_thresholds();
            }

            // 每50个周期打印完整状态 (debug级别2以上)
            if (cycle % 50 == 0 && debug::get_debug_level() >= 2) {
                dump_all();
            }
        }

        PHILE_DBG(1, "[PrefetchEngine] prefetch_loop stopped");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// 便捷宏 — 断点调试
// ═══════════════════════════════════════════════════════════════════════

// 打印预取引擎完整状态
#define PHILE_PREFETCH_DUMP(engine) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[PREFETCH_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (engine).dump_all(); \
        } \
    } while(0)

// 打印最近N条查询历史
#define PHILE_HISTORY_DUMP(engine, n) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[HISTORY_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (engine).history().dump(n); \
        } \
    } while(0)

// 预取断点: RAII自动打印前后状态
class PrefetchBreakpointGuard {
    const PrefetchEngine& engine_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    PrefetchBreakpointGuard(const PrefetchEngine& engine, const char* name)
        : engine_(engine), name_(name)
        , start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ PREFETCH_BP ENTER: %s ━━━━\n", name_);
            engine_.dump_all();
        }
    }
    ~PrefetchBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto elapsed = std::chrono::duration_cast<
                std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ PREFETCH_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)elapsed);
            engine_.dump_all();
        }
    }
};

#define PHILE_PREFETCH_BREAKPOINT(engine, name) \
    ::philemon::prefetch::PrefetchBreakpointGuard \
        _phile_prefetch_bp_##__LINE__((engine), (name))

}  // namespace prefetch
}  // namespace philemon

#endif  // PHILEMON_PREFETCH_ENGINE_HPP
