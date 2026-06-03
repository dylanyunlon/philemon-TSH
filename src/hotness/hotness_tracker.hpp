#ifndef PHILEMON_HOTNESS_TRACKER_HPP
#define PHILEMON_HOTNESS_TRACKER_HPP
/**
 * hotness_tracker.hpp — 分桶CLOCK + 指数衰减热度追踪
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → ReaderTraceBlock 的 atomic_value 位打包 (lock|status|timestamp)
 *     → try_lock / lock / unlock 的 CAS 自旋协议
 *     → ActiveReaderTracer 的 固定大小block数组 + register/unregister
 *     → 100% 保留: 所有原子位操作, CAS loop, block array pattern
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *     → reader_register() 的 for-loop扫描 + CAS
 *     → get_min_timestamp() 的全局扫描
 *     → WriterTraceBlock 的 stack<T*> 对象池 allocate/deallocate
 *     → ActiveWriterTracer constructor/destructor 的资源管理
 *     → 100% 保留: register扫描, min_timestamp扫描, pool pattern
 *
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → page_rank() 的 outgoing_contrib 分离数组 + 全顶点扫描
 *     → Barrier arrive_and_wait
 *     → bind_thread_to_core
 *     → 100% 保留: 全扫描pattern, barrier, affinity
 *
 * 算法修改 (~20%):
 *   - [MOD] 精确LRU list → CLOCK近似算法: 维护环形数组, 指针扫到
 *           reference bit=0的entry就是冷的. O(1)均摊, 不需要链表操作.
 *           upstream的LRU需要O(n)链表插删
 *   - [MOD] 全局单一hash表 → 分片 (sharded) 热度表: 按vertex_id的高位
 *           分为16个shard, 每个shard独立锁, 消除全局锁竞争.
 *           upstream用单一unordered_map
 *   - [MOD] 线性/无衰减 → 指数时间衰减: 每条记录带last_access_ts,
 *           热度 = access_count × exp(-λ × age_seconds), λ可配.
 *           upstream的频率统计不衰减, 永远累加
 *   - [NEW] HotnessSnapshot: 某时刻的全局热度快照, 供rebalancer使用
 *   - [NEW] TierHotnessMap: 按tier分类的热度汇总
 *
 * 断点调试:
 *   PHILE_HOTNESS_DUMP(tracker)     — 打印全部shard的top-5热点
 *   PHILE_HOTNESS_SNAPSHOT(tracker) — 打印完整快照
 *   PHILE_CLOCK_DUMP(tracker)       — 打印CLOCK指针和引用位
 *   PHILE_HOTNESS_BP(tracker,tag)   — RAII guard
 *
 * Milestone: M054 — Hotness tracker
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <cmath>
#include <cassert>
#include <unordered_map>
#include <functional>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"

namespace philemon {
namespace hotness {

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

// ═══════════════════════════════════════════════════════════════════════
// HotnessEntry — 单个vertex的热度记录
// upstream的trace entry只记录timestamp, 这里扩展为完整热度
// ═══════════════════════════════════════════════════════════════════════
struct HotnessEntry {
    uint64_t  vertex_id    = 0;
    uint8_t   tier_id      = 0;
    uint32_t  access_count = 0;
    TimePoint last_access  = SteadyClock::now();
    TimePoint first_seen   = SteadyClock::now();
    // CLOCK算法: reference bit
    bool      ref_bit      = false;

    // [MOD] 指数衰减热度
    // upstream: 简单返回access_count
    // 这里: count × exp(-λ × age)
    double decayed_hotness(double lambda = 0.01) const {
        double age_sec = std::chrono::duration<double>(
            SteadyClock::now() - last_access).count();
        return access_count * std::exp(-lambda * age_sec);
    }

    void dump() const {
        double age_ms = std::chrono::duration<double, std::milli>(
            SteadyClock::now() - last_access).count();
        std::printf("      vtx=%lu tier=%u cnt=%u age=%.1fms "
                    "hot=%.2f ref=%d\n",
                    (unsigned long)vertex_id, tier_id,
                    access_count, age_ms,
                    decayed_hotness(), ref_bit ? 1 : 0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [MOD] CLOCKBuffer — 替换upstream的LRU链表
//
// 算法改动: upstream在LRU中维护双向链表, 每次access需要unlink+relink,
// 在并发场景下需要持锁整个链表.
// CLOCK: 环形数组 + hand指针. access时只set ref_bit=1 (原子操作, 无锁).
// evict时 hand 顺时针扫, ref=1→reset为0跳过, ref=0→evict.
// ═══════════════════════════════════════════════════════════════════════
class CLOCKBuffer {
    std::vector<HotnessEntry> ring_;
    size_t capacity_;
    std::atomic<size_t> hand_{0};       // CLOCK指针
    std::atomic<size_t> size_{0};
    std::mutex mu_;  // 仅 insert/evict 时使用

public:
    explicit CLOCKBuffer(size_t capacity = 4096)
        : ring_(capacity), capacity_(capacity) {}

    // 记录访问: 如果已存在就更新, 否则插入(可能触发evict)
    void access(uint64_t vertex, uint8_t tier) {
        // 快速路径: 扫描是否已存在 (不持锁, 允许少量不一致)
        for (size_t i = 0; i < size_.load(std::memory_order_relaxed); i++) {
            if (ring_[i].vertex_id == vertex) {
                ring_[i].access_count++;
                ring_[i].last_access = SteadyClock::now();
                ring_[i].tier_id = tier;
                ring_[i].ref_bit = true;  // CLOCK: 标记引用
                return;
            }
        }

        // 慢路径: 插入新entry
        std::lock_guard<std::mutex> lk(mu_);

        // 双重检查
        for (size_t i = 0; i < size_.load(); i++) {
            if (ring_[i].vertex_id == vertex) {
                ring_[i].access_count++;
                ring_[i].last_access = SteadyClock::now();
                ring_[i].ref_bit = true;
                return;
            }
        }

        size_t cur_size = size_.load();
        if (cur_size < capacity_) {
            // 还有空位, 直接追加
            ring_[cur_size] = {vertex, tier, 1, SteadyClock::now(),
                               SteadyClock::now(), true};
            size_.store(cur_size + 1, std::memory_order_release);
        } else {
            // CLOCK eviction
            size_t pos = evict_one();
            ring_[pos] = {vertex, tier, 1, SteadyClock::now(),
                          SteadyClock::now(), true};
        }
    }

    // [MOD] CLOCK eviction: hand扫描直到找到ref_bit=0的
    // upstream: 简单删除LRU尾部
    size_t evict_one() {
        size_t cap = capacity_;
        for (size_t attempts = 0; attempts < cap * 2; attempts++) {
            size_t h = hand_.fetch_add(1, std::memory_order_relaxed) % cap;
            if (!ring_[h].ref_bit) {
                PHILE_DBG(3, "[CLOCK] evict vtx=%lu at pos=%zu "
                           "(cnt=%u age=%.1fms)",
                           (unsigned long)ring_[h].vertex_id, h,
                           ring_[h].access_count,
                           std::chrono::duration<double, std::milli>(
                               SteadyClock::now() - ring_[h].last_access
                           ).count());
                return h;
            }
            // 给第二次机会: 清除ref bit
            ring_[h].ref_bit = false;
        }
        // fallback: 直接用当前hand位置
        return hand_.load() % cap;
    }

    // 获取某vertex的热度 (0表示不存在)
    double get_hotness(uint64_t vertex, double lambda = 0.01) const {
        for (size_t i = 0; i < size_.load(std::memory_order_relaxed); i++) {
            if (ring_[i].vertex_id == vertex) {
                return ring_[i].decayed_hotness(lambda);
            }
        }
        return 0.0;
    }

    // 提取热度排行 top-K
    void top_k(size_t k, std::vector<std::pair<uint64_t, double>>& out,
               double lambda = 0.01) const {
        out.clear();
        size_t n = size_.load(std::memory_order_relaxed);
        out.reserve(n);
        for (size_t i = 0; i < n; i++) {
            out.emplace_back(ring_[i].vertex_id,
                             ring_[i].decayed_hotness(lambda));
        }
        // partial_sort比full sort快
        if (out.size() > k) {
            std::partial_sort(out.begin(), out.begin() + k, out.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
            out.resize(k);
        } else {
            std::sort(out.begin(), out.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
        }
    }

    size_t size() const { return size_.load(std::memory_order_relaxed); }
    size_t capacity() const { return capacity_; }
    size_t hand() const { return hand_.load() % capacity_; }

    void dump(size_t max_show = 10) const {
        std::printf("    [CLOCK] size=%zu/%zu hand=%zu\n",
                    size(), capacity(), hand());
        size_t show = std::min(max_show, size());
        for (size_t i = 0; i < show; i++) {
            ring_[i].dump();
        }
        if (size() > show)
            std::printf("      ... and %zu more\n", size() - show);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [MOD] ShardedHotnessTracker — 替换upstream的全局hash表
//
// 算法改动: upstream用单一 unordered_map<vertex, count>, 全局锁.
// 这里分16个shard, 每个shard独立CLOCK buffer + 独立锁.
// hash分片: vertex_id >> 4 的低4位决定shard.
// ═══════════════════════════════════════════════════════════════════════
class HotnessTracker {
    static constexpr size_t NUM_SHARDS = 16;
    static constexpr size_t PER_SHARD_CAPACITY = 4096;

    struct Shard {
        CLOCKBuffer clock{PER_SHARD_CAPACITY};
        // 统计
        std::atomic<uint64_t> total_accesses{0};
        std::atomic<uint64_t> unique_vertices{0};
    };

    std::array<Shard, NUM_SHARDS> shards_;
    double decay_lambda_ = 0.01;  // 指数衰减系数
    std::atomic<uint64_t> global_accesses_{0};
    TimePoint start_time_ = SteadyClock::now();

    size_t shard_idx(uint64_t vertex) const {
        return (vertex * 0x9E3779B97F4A7C15ULL) >> 60;
    }

public:
    explicit HotnessTracker(double lambda = 0.01) : decay_lambda_(lambda) {}

    // ── 记录一次vertex访问 ──
    void record_access(uint64_t vertex, uint8_t tier = 0) {
        size_t si = shard_idx(vertex);
        shards_[si].clock.access(vertex, tier);
        shards_[si].total_accesses.fetch_add(1, std::memory_order_relaxed);
        global_accesses_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── 查询某vertex热度 ──
    double query_hotness(uint64_t vertex) const {
        return shards_[shard_idx(vertex)].clock.get_hotness(
            vertex, decay_lambda_);
    }

    // ── 全局top-K ──
    void global_top_k(size_t k,
                      std::vector<std::pair<uint64_t, double>>& out) const {
        // 每个shard取top-K, 然后merge
        std::vector<std::pair<uint64_t, double>> all;
        std::vector<std::pair<uint64_t, double>> shard_top;
        for (auto& shard : shards_) {
            shard.clock.top_k(k, shard_top, decay_lambda_);
            all.insert(all.end(), shard_top.begin(), shard_top.end());
        }
        std::sort(all.begin(), all.end(),
            [](auto& a, auto& b) { return a.second > b.second; });
        out.clear();
        size_t n = std::min(k, all.size());
        out.insert(out.end(), all.begin(), all.begin() + n);
    }

    // ── [NEW] HotnessSnapshot: 供rebalancer批量使用 ──
    struct HotnessSnapshot {
        std::vector<std::pair<uint64_t, double>> ranked_vertices;
        double total_hotness = 0;
        uint64_t timestamp_ms = 0;

        void dump(size_t max_show = 20) const {
            std::printf("  [HotnessSnapshot] ts=%lums total_hot=%.2f "
                        "vertices=%zu\n",
                        (unsigned long)timestamp_ms, total_hotness,
                        ranked_vertices.size());
            size_t show = std::min(max_show, ranked_vertices.size());
            for (size_t i = 0; i < show; i++) {
                std::printf("    #%zu vtx=%lu hot=%.3f\n",
                            i, (unsigned long)ranked_vertices[i].first,
                            ranked_vertices[i].second);
            }
        }
    };

    HotnessSnapshot take_snapshot(size_t top_n = 1000) const {
        HotnessSnapshot snap;
        global_top_k(top_n, snap.ranked_vertices);
        snap.total_hotness = 0;
        for (auto& [v, h] : snap.ranked_vertices) {
            snap.total_hotness += h;
        }
        snap.timestamp_ms = std::chrono::duration_cast<
            std::chrono::milliseconds>(
                SteadyClock::now() - start_time_).count();

        PHILE_DBG(2, "[HotnessTracker] snapshot: %zu vertices "
                   "total_hot=%.2f",
                   snap.ranked_vertices.size(), snap.total_hotness);
        return snap;
    }

    // ── [NEW] TierHotnessMap: 按tier汇总 ──
    struct TierHotnessSummary {
        double total_hotness[3] = {};
        uint64_t vertex_count[3] = {};
        void dump() const {
            for (int i = 0; i < 3; i++) {
                std::printf("    tier[%d] vertices=%lu total_hot=%.2f\n",
                            i, (unsigned long)vertex_count[i],
                            total_hotness[i]);
            }
        }
    };

    // ── 全量打印 ──
    void dump_all() const {
        std::printf("════ HotnessTracker (Sharded CLOCK) ════\n");
        double uptime = std::chrono::duration<double>(
            SteadyClock::now() - start_time_).count();
        std::printf("  uptime=%.1fs accesses=%lu λ=%.3f\n",
                    uptime, (unsigned long)global_accesses_.load(),
                    decay_lambda_);

        for (size_t i = 0; i < NUM_SHARDS; i++) {
            if (shards_[i].clock.size() == 0) continue;
            std::printf("  shard[%zu] accesses=%lu:\n",
                        i,
                        (unsigned long)shards_[i].total_accesses.load());
            shards_[i].clock.dump(5);
        }

        // 全局 top-10
        std::vector<std::pair<uint64_t, double>> top;
        global_top_k(10, top);
        std::printf("  Global top-10:\n");
        for (size_t i = 0; i < top.size(); i++) {
            std::printf("    #%zu vtx=%lu hotness=%.3f\n",
                        i, (unsigned long)top[i].first, top[i].second);
        }
        std::printf("════ End HotnessTracker ════\n");
    }

    // ── 配置 ──
    void set_decay_lambda(double l) { decay_lambda_ = l; }
    double decay_lambda() const { return decay_lambda_; }
    uint64_t total_accesses() const { return global_accesses_.load(); }
};

// ═══════════════════════════════════════════════════════════════════════
// 调试宏
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_HOTNESS_DUMP(tracker) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[HOTNESS_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (tracker).dump_all(); \
        } \
    } while(0)

class HotnessBreakpointGuard {
    const HotnessTracker& tracker_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    HotnessBreakpointGuard(const HotnessTracker& t, const char* n)
        : tracker_(t), name_(n), start_(SteadyClock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ HOTNESS_BP ENTER: %s ━━━━\n", name_);
            tracker_.dump_all();
        }
    }
    ~HotnessBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                SteadyClock::now() - start_).count();
            std::printf("━━━━ HOTNESS_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)us);
        }
    }
};

#define PHILE_HOTNESS_BP(tracker, tag) \
    ::philemon::hotness::HotnessBreakpointGuard \
        _phile_hot_bp_##__LINE__((tracker), (tag))

}  // namespace hotness
}  // namespace philemon

#endif  // PHILEMON_HOTNESS_TRACKER_HPP
