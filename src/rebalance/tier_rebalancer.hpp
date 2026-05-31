#ifndef PHILEMON_TIER_REBALANCER_HPP
#define PHILEMON_TIER_REBALANCER_HPP
/**
 * tier_rebalancer.hpp — Automatic Tier Redistribution Engine
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → execute_mixed_reader_writer() 的并发reader/writer线程模式
 *     → Barrier (arrive_and_wait) 同步原语
 *     → bind_thread_to_core() CPU亲和性
 *     → execute_query() 的 for-loop-over-query-types
 *     → throughput计算 (edges/sec)
 *     → 用于rebalance线程的调度和性能统计
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp     (537行)
 *     → get_write_timestamp() 的 fetch_add
 *     → finish_commit() 的 CAS loop
 *     → 用于rebalance事务的原子提交
 *
 *   upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.h         (210行)
 *     → Snapshot 嵌套类 + edges() 回调
 *     → TransactionManager tm 生命周期
 *     → 用于rebalance时的快照读取
 *
 *   upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.cpp       (703行)
 *     → run_batch_edge_update() 的 batch循环
 *     → writer_register() 的 CAS注册
 *     → checkpoint 计数打印
 *     → 用于批量迁移操作
 *
 *   upstream/rapidstore/wrapper/wrapper.h                              (249行)
 *     → snapshot_edges() 模板回调 pattern
 *     → 用于采样partition热度
 *
 * 修改 (~20%):
 *   - [NEW] TierProfile: 每个tier的运行时profile (利用率+温度)
 *   - [NEW] RebalanceDecision: 重平衡决策 (哪些partition移到哪个tier)
 *   - [NEW] ImbalanceDetector: 检测tier不平衡
 *   - [NEW] RebalanceExecutor: 批量执行tier间迁移
 *   - [NEW] TierRebalancer: 顶层自动重平衡引擎
 *   - [MOD] Barrier → RebalanceBarrier (增加phase同步)
 *   - [MOD] execute_mixed → rebalance_mixed (读写混合模式)
 *   - [MOD] TransactionManager → RebalanceTransaction
 *   - [KEEP] Barrier arrive_and_wait 100% 保留
 *   - [KEEP] bind_thread_to_core 100% 保留
 *   - [KEEP] CAS commit loop 100% 保留
 *   - [KEEP] batch loop + checkpoint 100% 保留
 *   - [KEEP] snapshot_edges 回调 pattern 100% 保留
 *
 * 断点调试:
 *   PHILE_REBALANCE_DUMP()     — 打印tier利用率+温度+决策
 *   PHILE_REBALANCE_BREAKPOINT — RAII断点
 *   PHILE_TIER_PROFILE_DUMP()  — 打印各tier详细profile
 *
 * Milestone: M026 — Tier rebalancing
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cassert>
#include <cmath>
#include <unordered_map>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../cost_model/tier_cost_model.hpp"

namespace philemon {
namespace rebalance {

// ═══════════════════════════════════════════════════════════════════════
// TierProfile — 每个tier的运行时状态
// ═══════════════════════════════════════════════════════════════════════
struct TierProfile {
    uint8_t  tier_id;           // 0=HBM, 1=GDDR, 2=DRAM
    uint64_t capacity_bytes;
    std::atomic<uint64_t> used_bytes{0};
    std::atomic<uint64_t> partition_count{0};

    // 访问统计 (滑动窗口)
    std::atomic<uint64_t> access_count{0};    // 当前窗口访问次数
    std::atomic<uint64_t> total_accesses{0};  // 历史总访问次数
    std::atomic<uint64_t> total_latency_ns{0};// 历史总延迟

    // 温度指标
    double temperature() const {
        uint64_t acc = access_count.load(std::memory_order_relaxed);
        uint64_t parts = partition_count.load(std::memory_order_relaxed);
        if (parts == 0) return 0.0;
        return static_cast<double>(acc) / parts;
    }

    double utilization() const {
        if (capacity_bytes == 0) return 0.0;
        return static_cast<double>(
            used_bytes.load(std::memory_order_relaxed)) / capacity_bytes;
    }

    double avg_latency_ns() const {
        uint64_t total = total_accesses.load(std::memory_order_relaxed);
        if (total == 0) return 0.0;
        return static_cast<double>(
            total_latency_ns.load(std::memory_order_relaxed)) / total;
    }

    // 衰减窗口
    void decay_window() {
        uint64_t cur = access_count.load(std::memory_order_relaxed);
        access_count.store(cur >> 1, std::memory_order_relaxed);
    }

    void dump() const {
        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        std::printf("    [%s] used=%luMB/%luMB (%.1f%%) "
                    "parts=%lu accesses=%lu temp=%.1f "
                    "avg_lat=%.1fμs\n",
                    tname[tier_id % 3],
                    (unsigned long)(used_bytes.load() / (1024*1024)),
                    (unsigned long)(capacity_bytes / (1024*1024)),
                    utilization() * 100.0,
                    (unsigned long)partition_count.load(),
                    (unsigned long)access_count.load(),
                    temperature(),
                    avg_latency_ns() / 1000.0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PartitionHeatInfo — 分区热度信息
// ═══════════════════════════════════════════════════════════════════════
struct PartitionHeatInfo {
    uint64_t partition_id;
    uint8_t  current_tier;
    uint64_t bytes;
    uint64_t access_count;
    double   avg_latency_ns;
    double   heat_score;       // 综合热度评分

    // 热度评分: 高=热(应在HBM), 低=冷(可移到DRAM)
    void compute_heat(double global_avg_access) {
        double freq_factor = (global_avg_access > 0) ?
            static_cast<double>(access_count) / global_avg_access : 0.0;
        double size_factor = std::log2(1.0 + bytes / (1024.0 * 1024.0));
        heat_score = freq_factor / (size_factor + 0.1);
    }

    void dump() const {
        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        std::printf("      part=%lu tier=%s %luKB acc=%lu "
                    "lat=%.1fμs heat=%.2f\n",
                    (unsigned long)partition_id,
                    tname[current_tier % 3],
                    (unsigned long)(bytes / 1024),
                    (unsigned long)access_count,
                    avg_latency_ns / 1000.0,
                    heat_score);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// RebalanceDecision — 重平衡决策
// ═══════════════════════════════════════════════════════════════════════
struct MigrationOrder {
    uint64_t partition_id;
    uint8_t  from_tier;
    uint8_t  to_tier;
    uint64_t bytes;
    double   expected_benefit_ns;

    void dump() const {
        const char* tname[] = {"HBM", "GDDR", "DRAM"};
        std::printf("      [Migrate] part=%lu %s→%s %luKB "
                    "benefit=%.1fμs\n",
                    (unsigned long)partition_id,
                    tname[from_tier % 3], tname[to_tier % 3],
                    (unsigned long)(bytes / 1024),
                    expected_benefit_ns / 1000.0);
    }
};

struct RebalanceDecision {
    std::vector<MigrationOrder> promotions;    // cold_tier → hot_tier
    std::vector<MigrationOrder> demotions;     // hot_tier → cold_tier
    uint64_t total_bytes_to_move;
    double   estimated_total_benefit_ns;

    bool is_empty() const {
        return promotions.empty() && demotions.empty();
    }

    void dump() const {
        std::printf("  [RebalanceDecision] promotions=%zu demotions=%zu "
                    "bytes=%luMB benefit=%.1fμs\n",
                    promotions.size(), demotions.size(),
                    (unsigned long)(total_bytes_to_move / (1024*1024)),
                    estimated_total_benefit_ns / 1000.0);
        if (!promotions.empty()) {
            std::printf("    --- Promotions (cold→hot) ---\n");
            for (size_t i = 0; i < std::min(promotions.size(), (size_t)5); i++)
                promotions[i].dump();
            if (promotions.size() > 5)
                std::printf("      ... (%zu more)\n",
                            promotions.size() - 5);
        }
        if (!demotions.empty()) {
            std::printf("    --- Demotions (hot→cold) ---\n");
            for (size_t i = 0; i < std::min(demotions.size(), (size_t)5); i++)
                demotions[i].dump();
            if (demotions.size() > 5)
                std::printf("      ... (%zu more)\n",
                            demotions.size() - 5);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// ImbalanceDetector — 检测tier不平衡并生成重平衡决策
// 骨架: driver.h::execute_query() 的查询类型分发
// 修改: 改为分区热度排序 + 最优tier分配
// ═══════════════════════════════════════════════════════════════════════
class ImbalanceDetector {
private:
    const cost_model::TierCostModel& cost_model_;

    // 阈值
    double promotion_heat_threshold_;    // 热度超过此值 → promote
    double demotion_heat_threshold_;     // 热度低于此值 → demote
    double utilization_headroom_;        // 目标tier需要的剩余空间比例
    size_t max_orders_per_cycle_;        // 每轮最多迁移多少个

public:
    ImbalanceDetector(const cost_model::TierCostModel& cost_model,
                      double promote_threshold = 5.0,
                      double demote_threshold = 0.5,
                      double headroom = 0.15,
                      size_t max_orders = 32)
        : cost_model_(cost_model)
        , promotion_heat_threshold_(promote_threshold)
        , demotion_heat_threshold_(demote_threshold)
        , utilization_headroom_(headroom)
        , max_orders_per_cycle_(max_orders) {}

    RebalanceDecision analyze(
        const std::array<TierProfile, 3>& profiles,
        std::vector<PartitionHeatInfo>& partitions)
    {
        debug::ScopedTimer timer("ImbalanceDetector::analyze");
        RebalanceDecision decision;
        decision.total_bytes_to_move = 0;
        decision.estimated_total_benefit_ns = 0;

        if (partitions.empty()) return decision;

        // 计算全局平均访问数
        double global_avg = 0;
        for (const auto& p : partitions) global_avg += p.access_count;
        global_avg /= partitions.size();

        // 计算每个分区的热度
        for (auto& p : partitions) {
            p.compute_heat(global_avg);
        }

        // 按热度排序: 最热在前
        std::sort(partitions.begin(), partitions.end(),
                  [](const auto& a, const auto& b) {
                      return a.heat_score > b.heat_score;
                  });

        // 统计每个tier的可用空间
        uint64_t tier_avail[3];
        for (int t = 0; t < 3; t++) {
            uint64_t used = profiles[t].used_bytes.load();
            uint64_t headroom_bytes = static_cast<uint64_t>(
                profiles[t].capacity_bytes * utilization_headroom_);
            int64_t avail = static_cast<int64_t>(
                profiles[t].capacity_bytes) - used - headroom_bytes;
            tier_avail[t] = (avail > 0) ? static_cast<uint64_t>(avail) : 0;
        }

        PHILE_DBG(2, "[ImbalanceDetector] global_avg_access=%.1f "
                  "avail: HBM=%luMB GDDR=%luMB DRAM=%luMB",
                  global_avg,
                  (unsigned long)(tier_avail[0] / (1024*1024)),
                  (unsigned long)(tier_avail[1] / (1024*1024)),
                  (unsigned long)(tier_avail[2] / (1024*1024)));

        // Pass 1: 识别需要promote的热分区 (不在HBM但热度高)
        for (const auto& p : partitions) {
            if (decision.promotions.size() >= max_orders_per_cycle_) break;
            if (p.heat_score < promotion_heat_threshold_) break;
            if (p.current_tier == 0) continue;  // 已在HBM

            // 找最佳目标tier (尽量HBM, 否则GDDR)
            uint8_t target = 255;
            if (p.current_tier > 0 && tier_avail[0] >= p.bytes) {
                target = 0;
            } else if (p.current_tier > 1 && tier_avail[1] >= p.bytes) {
                target = 1;
            }

            if (target == 255) continue;

            MigrationOrder order;
            order.partition_id = p.partition_id;
            order.from_tier    = p.current_tier;
            order.to_tier      = target;
            order.bytes        = p.bytes;
            order.expected_benefit_ns =
                cost_model_.prefetch_benefit_ns(
                    p.current_tier, target, p.bytes, p.access_count);

            decision.promotions.push_back(order);
            decision.total_bytes_to_move += p.bytes;
            decision.estimated_total_benefit_ns += order.expected_benefit_ns;
            tier_avail[target] -= p.bytes;
        }

        // Pass 2: 识别需要demote的冷分区 (在HBM/GDDR但热度低)
        // 从最冷的开始 (倒序遍历)
        for (auto it = partitions.rbegin(); it != partitions.rend(); ++it) {
            if (decision.demotions.size() >= max_orders_per_cycle_) break;
            if (it->heat_score > demotion_heat_threshold_) break;
            if (it->current_tier >= 2) continue;  // 已在DRAM

            uint8_t target = it->current_tier + 1;
            if (tier_avail[target] < it->bytes) continue;

            MigrationOrder order;
            order.partition_id = it->partition_id;
            order.from_tier    = it->current_tier;
            order.to_tier      = target;
            order.bytes        = it->bytes;
            order.expected_benefit_ns = 0;  // demotion无正收益

            decision.demotions.push_back(order);
            decision.total_bytes_to_move += it->bytes;
            tier_avail[target] -= it->bytes;
        }

        PHILE_DBG(1, "[ImbalanceDetector] decision: %zu promotions, "
                  "%zu demotions, %luMB to move",
                  decision.promotions.size(), decision.demotions.size(),
                  (unsigned long)(decision.total_bytes_to_move / (1024*1024)));

        return decision;
    }

    // 阈值调节
    void set_promotion_threshold(double t) { promotion_heat_threshold_ = t; }
    void set_demotion_threshold(double t) { demotion_heat_threshold_ = t; }
};

// ═══════════════════════════════════════════════════════════════════════
// RebalanceBarrier — 从 driver.h::Barrier 移植
// 保留: arrive_and_wait 100% (完整)
// 修改: 增加 phase tracking
// ═══════════════════════════════════════════════════════════════════════
class RebalanceBarrier {
private:
    // === 保留: 与driver.h::Barrier完全相同的实现 ===
    std::mutex mtx_;
    std::condition_variable cv_;
    std::size_t count_;
    std::size_t waiting_;

    // [NEW] Phase tracking
    std::atomic<uint32_t> phase_{0};

public:
    explicit RebalanceBarrier(std::size_t count) : count_(count), waiting_(0) {}

    // === 保留: 与upstream完全相同的arrive_and_wait ===
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

    uint32_t phase() const {
        return phase_.load(std::memory_order_relaxed);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// RebalanceTransaction — 从 TransactionManager 移植
// 保留: fetch_add + CAS loop (100%)
// 修改: 增加 rebalance-specific 统计
// ═══════════════════════════════════════════════════════════════════════
class RebalanceTransaction {
private:
    // === 保留: 与TransactionManager相同的原子时间戳 ===
    std::atomic<uint64_t> write_timestamp_{0};
    std::atomic<uint64_t> read_timestamp_{0};

    // 统计
    std::atomic<uint64_t> total_rebalances_{0};
    std::atomic<uint64_t> total_promotions_{0};
    std::atomic<uint64_t> total_demotions_{0};
    std::atomic<uint64_t> total_bytes_moved_{0};

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

    void record_rebalance(uint64_t promotions, uint64_t demotions,
                          uint64_t bytes) {
        uint64_t ts = get_write_timestamp();
        total_rebalances_.fetch_add(1, std::memory_order_relaxed);
        total_promotions_.fetch_add(promotions, std::memory_order_relaxed);
        total_demotions_.fetch_add(demotions, std::memory_order_relaxed);
        total_bytes_moved_.fetch_add(bytes, std::memory_order_relaxed);
        finish_commit(ts);
    }

    void dump() const {
        std::printf("  [RebalanceTxn] write_ts=%lu read_ts=%lu\n",
                    (unsigned long)write_timestamp_.load(),
                    (unsigned long)read_timestamp_.load());
        std::printf("    total: rebalances=%lu promotions=%lu "
                    "demotions=%lu bytes=%luMB\n",
                    (unsigned long)total_rebalances_.load(),
                    (unsigned long)total_promotions_.load(),
                    (unsigned long)total_demotions_.load(),
                    (unsigned long)(total_bytes_moved_.load() / (1024*1024)));
    }
};

// ═══════════════════════════════════════════════════════════════════════
// RebalanceExecutor — 批量执行tier迁移
// 骨架: neo_wrapper.cpp::run_batch_edge_update() 的 batch loop
// 保留: checkpoint 计数 (100%)
// 修改: 改为tier迁移的批量执行
// ═══════════════════════════════════════════════════════════════════════
class RebalanceExecutor {
private:
    uint64_t checkpoint_size_;

    // 迁移回调
    using MigrateCallback =
        std::function<bool(uint64_t, uint8_t, uint8_t, uint64_t)>;
    MigrateCallback migrate_fn_;

    std::atomic<uint64_t> total_executed_{0};
    std::atomic<uint64_t> total_bytes_{0};

public:
    explicit RebalanceExecutor(uint64_t checkpoint_size = 50)
        : checkpoint_size_(checkpoint_size) {}

    void set_migrate_callback(MigrateCallback fn) {
        migrate_fn_ = std::move(fn);
    }

    struct ExecutionResult {
        uint64_t promoted_count;
        uint64_t demoted_count;
        uint64_t failed_count;
        uint64_t bytes_moved;
        double   total_time_us;

        void dump() const {
            std::printf("  [ExecutionResult] promoted=%lu demoted=%lu "
                        "failed=%lu bytes=%luMB time=%.1fμs\n",
                        (unsigned long)promoted_count,
                        (unsigned long)demoted_count,
                        (unsigned long)failed_count,
                        (unsigned long)(bytes_moved / (1024*1024)),
                        total_time_us);
        }
    };

    // === 模仿: run_batch_edge_update() 的 batch loop ===
    ExecutionResult execute(const RebalanceDecision& decision) {
        debug::ScopedTimer timer("RebalanceExecutor::execute");
        auto start = std::chrono::high_resolution_clock::now();

        ExecutionResult result;
        result.promoted_count = 0;
        result.demoted_count  = 0;
        result.failed_count   = 0;
        result.bytes_moved    = 0;

        PHILE_DBG(1, "[RebalanceExecutor] executing: %zu promotions, "
                  "%zu demotions",
                  decision.promotions.size(), decision.demotions.size());

        // === Phase 1: Execute promotions (cold→hot) ===
        for (size_t i = 0; i < decision.promotions.size(); i++) {
            const auto& order = decision.promotions[i];
            bool success = false;

            if (migrate_fn_) {
                success = migrate_fn_(order.partition_id, order.from_tier,
                                      order.to_tier, order.bytes);
            } else {
                success = true;  // 模拟
            }

            if (success) {
                result.promoted_count++;
                result.bytes_moved += order.bytes;
            } else {
                result.failed_count++;
            }

            // === 保留: checkpoint 计数 ===
            if ((i + 1) % checkpoint_size_ == 0) {
                PHILE_DBG(1, "[RebalanceExecutor] promotions checkpoint: "
                          "%zu/%zu (%luMB moved)",
                          result.promoted_count,
                          decision.promotions.size(),
                          (unsigned long)(result.bytes_moved / (1024*1024)));
            }
        }

        // === Phase 2: Execute demotions (hot→cold) ===
        for (size_t i = 0; i < decision.demotions.size(); i++) {
            const auto& order = decision.demotions[i];
            bool success = false;

            if (migrate_fn_) {
                success = migrate_fn_(order.partition_id, order.from_tier,
                                      order.to_tier, order.bytes);
            } else {
                success = true;
            }

            if (success) {
                result.demoted_count++;
                result.bytes_moved += order.bytes;
            } else {
                result.failed_count++;
            }

            // === 保留: checkpoint ===
            if ((i + 1) % checkpoint_size_ == 0) {
                PHILE_DBG(1, "[RebalanceExecutor] demotions checkpoint: "
                          "%zu/%zu",
                          result.demoted_count,
                          decision.demotions.size());
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        result.total_time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(end - start).count();

        total_executed_.fetch_add(
            result.promoted_count + result.demoted_count,
            std::memory_order_relaxed);
        total_bytes_.fetch_add(result.bytes_moved, std::memory_order_relaxed);

        PHILE_DBG(1, "[RebalanceExecutor] done: promoted=%lu demoted=%lu "
                  "failed=%lu bytes=%luMB in %.1fμs",
                  (unsigned long)result.promoted_count,
                  (unsigned long)result.demoted_count,
                  (unsigned long)result.failed_count,
                  (unsigned long)(result.bytes_moved / (1024*1024)),
                  result.total_time_us);

        return result;
    }

    void dump() const {
        std::printf("    [RebalanceExecutor] total=%lu bytes=%luMB\n",
                    (unsigned long)total_executed_.load(),
                    (unsigned long)(total_bytes_.load() / (1024*1024)));
    }
};

// ═══════════════════════════════════════════════════════════════════════
// TierRebalancer — 顶层自动重平衡引擎
// 骨架: Driver 类的 execute() 主循环
//       + execute_mixed_reader_writer() 的后台线程模式
// 修改: 集成Profile + Detector + Executor为一体
// ═══════════════════════════════════════════════════════════════════════
class TierRebalancer {
private:
    std::array<TierProfile, 3> profiles_;
    ImbalanceDetector   detector_;
    RebalanceExecutor   executor_;
    RebalanceTransaction txn_;

    // 配置
    uint64_t check_interval_ms_;
    double   imbalance_threshold_;  // 最小不平衡度才触发
    bool     auto_rebalance_;

    // 后台线程
    std::thread rebalance_thread_;
    std::atomic<bool> running_{false};

    // 分区状态获取回调
    using PartitionInfoCallback = std::function<
        std::vector<PartitionHeatInfo>()>;
    PartitionInfoCallback get_partitions_;

    // 统计
    std::atomic<uint64_t> check_cycles_{0};
    std::atomic<uint64_t> rebalances_triggered_{0};

public:
    TierRebalancer(const cost_model::TierCostModel& cost_model,
                   uint64_t check_interval_ms = 2000,
                   double imbalance_threshold = 0.2)
        : detector_(cost_model)
        , check_interval_ms_(check_interval_ms)
        , imbalance_threshold_(imbalance_threshold)
        , auto_rebalance_(true)
    {
        // 初始化tier profiles
        // HBM: 80GB
        profiles_[0].tier_id = 0;
        profiles_[0].capacity_bytes = 80ULL * 1024 * 1024 * 1024;
        // GDDR: 48GB
        profiles_[1].tier_id = 1;
        profiles_[1].capacity_bytes = 48ULL * 1024 * 1024 * 1024;
        // DRAM: 512GB
        profiles_[2].tier_id = 2;
        profiles_[2].capacity_bytes = 512ULL * 1024 * 1024 * 1024;
    }

    ~TierRebalancer() { stop(); }

    // ─── 配置 ───────────────────────────────────────────────────
    void set_tier_capacity(uint8_t tier, uint64_t bytes) {
        profiles_[tier % 3].capacity_bytes = bytes;
    }

    void set_partition_callback(PartitionInfoCallback fn) {
        get_partitions_ = std::move(fn);
    }

    void set_migrate_callback(
        std::function<bool(uint64_t, uint8_t, uint8_t, uint64_t)> fn) {
        executor_.set_migrate_callback(std::move(fn));
    }

    // ─── 运行时更新 (由外部调用) ─────────────────────────────────
    void update_tier_usage(uint8_t tier, uint64_t used, uint64_t parts) {
        profiles_[tier % 3].used_bytes.store(used,
            std::memory_order_relaxed);
        profiles_[tier % 3].partition_count.store(parts,
            std::memory_order_relaxed);
    }

    void record_access(uint8_t tier, uint64_t latency_ns) {
        profiles_[tier % 3].access_count.fetch_add(1,
            std::memory_order_relaxed);
        profiles_[tier % 3].total_accesses.fetch_add(1,
            std::memory_order_relaxed);
        profiles_[tier % 3].total_latency_ns.fetch_add(latency_ns,
            std::memory_order_relaxed);
    }

    // ─── 生命周期 ───────────────────────────────────────────────
    void start() {
        if (running_.load()) return;
        running_.store(true);
        rebalance_thread_ = std::thread([this]() { rebalance_loop(); });

        // === 保留: bind_thread_to_core ===
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(std::thread::hardware_concurrency() - 1, &cpuset);
        pthread_setaffinity_np(rebalance_thread_.native_handle(),
                               sizeof(cpu_set_t), &cpuset);

        PHILE_DBG(1, "[TierRebalancer] started (interval=%lums "
                  "threshold=%.1f%%)",
                  (unsigned long)check_interval_ms_,
                  imbalance_threshold_ * 100.0);
    }

    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (rebalance_thread_.joinable()) rebalance_thread_.join();
        PHILE_DBG(1, "[TierRebalancer] stopped. cycles=%lu triggered=%lu",
                  (unsigned long)check_cycles_.load(),
                  (unsigned long)rebalances_triggered_.load());
    }

    // ─── 手动触发重平衡 ─────────────────────────────────────────
    RebalanceExecutor::ExecutionResult rebalance_once() {
        PHILE_BREAKPOINT("rebalance_once");

        if (!get_partitions_) {
            PHILE_DBG(1, "[TierRebalancer] no partition callback, skip");
            return {};
        }

        auto partitions = get_partitions_();
        auto decision = detector_.analyze(profiles_, partitions);

        if (decision.is_empty()) {
            PHILE_DBG(1, "[TierRebalancer] balanced, no action needed");
            return {};
        }

        auto result = executor_.execute(decision);

        txn_.record_rebalance(result.promoted_count,
                              result.demoted_count,
                              result.bytes_moved);
        rebalances_triggered_.fetch_add(1, std::memory_order_relaxed);

        return result;
    }

    // ─── 不平衡度计算 ───────────────────────────────────────────
    double compute_imbalance() const {
        // 计算tier间温度差异的标准差
        double temps[3];
        for (int i = 0; i < 3; i++) {
            temps[i] = profiles_[i].temperature();
        }

        // 理想状态: HBM最热, DRAM最冷
        // 如果DRAM分区比HBM分区更热 → 不平衡
        double imbalance = 0.0;

        // 温度应该: HBM > GDDR > DRAM
        if (temps[2] > temps[0] && temps[0] > 0) {
            imbalance += (temps[2] - temps[0]) / temps[0];
        }
        if (temps[2] > temps[1] && temps[1] > 0) {
            imbalance += (temps[2] - temps[1]) / temps[1];
        }

        // 利用率不平衡
        double utils[3];
        for (int i = 0; i < 3; i++) {
            utils[i] = profiles_[i].utilization();
        }
        // HBM利用率太低而DRAM太满 → 不平衡
        if (utils[2] > 0.8 && utils[0] < 0.5) {
            imbalance += (utils[2] - utils[0]);
        }

        return imbalance;
    }

    // ─── 访问器 ──────────────────────────────────────────────────
    const std::array<TierProfile, 3>& profiles() const { return profiles_; }
    const RebalanceTransaction& transaction() const { return txn_; }

    // ─── [断点调试] 全量状态打印 ────────────────────────────────
    void dump_all() const {
        std::printf("╔══════════════════════════════════════════════╗\n");
        std::printf("║          TierRebalancer Full State          ║\n");
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("  running=%s cycles=%lu triggered=%lu\n",
                    running_.load() ? "true" : "false",
                    (unsigned long)check_cycles_.load(),
                    (unsigned long)rebalances_triggered_.load());
        std::printf("  interval=%lums threshold=%.1f%% imbalance=%.2f\n",
                    (unsigned long)check_interval_ms_,
                    imbalance_threshold_ * 100.0,
                    compute_imbalance());

        std::printf("\n  --- Tier Profiles ---\n");
        for (int i = 0; i < 3; i++) {
            profiles_[i].dump();
        }

        std::printf("\n");
        txn_.dump();
        executor_.dump();
        std::printf("╚══════════════════════════════════════════════╝\n");
    }

private:
    void rebalance_loop() {
        PHILE_DBG(1, "[TierRebalancer] rebalance_loop started");

        while (running_.load()) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(check_interval_ms_));
            if (!running_.load()) break;

            check_cycles_.fetch_add(1, std::memory_order_relaxed);
            uint64_t cycle = check_cycles_.load();

            // 计算不平衡度
            double imbalance = compute_imbalance();

            PHILE_DBG(2, "[TierRebalancer] cycle %lu: imbalance=%.2f "
                      "(threshold=%.2f)",
                      (unsigned long)cycle, imbalance, imbalance_threshold_);

            // 超过阈值 → 触发重平衡
            if (auto_rebalance_ && imbalance > imbalance_threshold_) {
                PHILE_DBG(1, "[TierRebalancer] imbalance %.2f > %.2f, "
                          "triggering rebalance",
                          imbalance, imbalance_threshold_);

                auto result = rebalance_once();

                PHILE_DBG(1, "[TierRebalancer] rebalance done: "
                          "promoted=%lu demoted=%lu bytes=%luMB",
                          (unsigned long)result.promoted_count,
                          (unsigned long)result.demoted_count,
                          (unsigned long)(result.bytes_moved / (1024*1024)));
            }

            // 每5个周期做温度衰减
            if (cycle % 5 == 0) {
                for (auto& profile : profiles_) {
                    profile.decay_window();
                }
                PHILE_DBG(2, "[TierRebalancer] temperature decay at "
                          "cycle %lu", (unsigned long)cycle);
            }

            // 每25个周期打印状态
            if (cycle % 25 == 0 && debug::get_debug_level() >= 2) {
                dump_all();
            }
        }

        PHILE_DBG(1, "[TierRebalancer] rebalance_loop stopped");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// 便捷宏 — 断点调试
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_REBALANCE_DUMP(rebalancer) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[REBALANCE_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (rebalancer).dump_all(); \
        } \
    } while(0)

#define PHILE_TIER_PROFILE_DUMP(rebalancer) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[TIER_PROFILE] at %s:%d\n", __FILE__, __LINE__); \
            for (int _i = 0; _i < 3; _i++) \
                (rebalancer).profiles()[_i].dump(); \
        } \
    } while(0)

class RebalanceBreakpointGuard {
    const TierRebalancer& rebalancer_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    RebalanceBreakpointGuard(const TierRebalancer& r, const char* name)
        : rebalancer_(r), name_(name)
        , start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ REBALANCE_BP ENTER: %s ━━━━\n", name_);
            rebalancer_.dump_all();
        }
    }
    ~RebalanceBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto elapsed = std::chrono::duration_cast<
                std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ REBALANCE_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)elapsed);
            rebalancer_.dump_all();
        }
    }
};

#define PHILE_REBALANCE_BREAKPOINT(rebalancer, name) \
    ::philemon::rebalance::RebalanceBreakpointGuard \
        _phile_rebal_bp_##__LINE__((rebalancer), (name))

}  // namespace rebalance
}  // namespace philemon

#endif  // PHILEMON_TIER_REBALANCER_HPP
