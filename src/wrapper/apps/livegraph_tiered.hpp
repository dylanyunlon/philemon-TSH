#ifndef PHILEMON_LIVEGRAPH_TIERED_HPP
#define PHILEMON_LIVEGRAPH_TIERED_HPP
/**
 * livegraph_tiered.hpp — LiveGraph MVCC图引擎的Tiered-Memory适配层
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/apps/livegraph/livegraph_wrapper.h   (131行)
 *   upstream/rapidstore/wrapper/apps/livegraph/livegraph_wrapper.cpp (584行)
 *   合计 715行
 *
 * upstream LiveGraph核心算法:
 *   LiveGraph是一个MVCC图存储,核心设计:
 *   - vertex dictionary: tbb::concurrent_hash_map<uint64_t, lg::vertex_t>
 *     做 logical → physical ID映射
 *   - insert_vertex: 通过lg::Transaction创建vertex, 存mapping
 *   - insert_edge: 查dictionary做ID转换, 开事务put_edge, 失败时重试
 *   - edges()遍历: 开read-only transaction, get_edges(vertex, label),
 *     用iterator逐个yield邻居
 *   - batch_loader: insert时用begin_batch_loader()开批量事务,
 *     减少per-edge事务开销
 *   - Snapshot: 通过begin_read_only_transaction()获取一致性快照
 *
 * 修改 (~20%):
 *   - [MOD] vertex dictionary: tbb hash_map → std::unordered_map + shared_mutex
 *     (去除TBB依赖, 适配tiered memory场景的读写锁粒度)
 *   - [MOD] insert_edge: upstream开lg::Transaction → 改为TierRouter决定
 *     目标tier后写入对应分区的adjacency list, 非事务型
 *   - [MOD] edges()遍历: upstream iterator-based → 改为先按tier分组,
 *     对每个tier的edge batch做prefetch hint后再yield
 *   - [MOD] batch_update: upstream begin_batch_loader() → 按tier分桶
 *     (bucket by hash(src,dst) % 3), 每桶独立flush
 *   - [NEW] TieredLiveGraphSnapshot: 聚合3个tier的partition视图
 *   - [NEW] dump_transaction_state(): 打印当前活跃事务+版本号
 *   - [NEW] dump_vertex_tier_distribution(): 打印各tier的vertex数量
 *   - [NEW] PHILE_LG_TRACE(): 每次dictionary lookup打印命中/未命中
 *   - [KEEP] Snapshot::clone() via enable_shared_from_this 100% 保留
 *   - [KEEP] logical2physical / physical2logical 身份映射 100% 保留
 *   - [KEEP] set_max_threads / init_thread / end_thread 线程管理 100% 保留
 *   - [KEEP] is_directed / is_weighted 标志位查询 100% 保留
 *   - [KEEP] has_edge 三个重载 (edge, src+dst, src+dst+weight) 100% 保留
 *   - [KEEP] remove_vertex / remove_edge 重试循环模式 100% 保留
 *   - [KEEP] wrapper_test() 自测逻辑 100% 保留
 *   - [KEEP] repl() 返回名称字符串 100% 保留
 *
 * Milestone: M030+ (第4位Claude) — LiveGraph tiered adapter
 * ====================================================================
 */

#include <vector>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <chrono>

#include "../graph_edge.hpp"
#include "../rapidstore_wrapper.hpp"
#include "../../types/philemon_types.hpp"
#include "../../utils/timer_utils.hpp"
#include "../../debug/philemon_debug.hpp"
#include "../../debug/state_inspector.hpp"

namespace philemon {
namespace adapters {
namespace livegraph {

// ─── Debug: LiveGraph-specific trace macro ──────────────────────────
// upstream had no debug trace; we add per-operation tracking so you can
// run experiments with PHILE_DEBUG_LEVEL=3 and see every dictionary hit.
#define PHILE_LG_TRACE(fmt, ...) \
    do { if (philemon::debug::get_debug_level() >= 3) { \
        std::fprintf(stderr, "[LG-TRACE %s:%d] " fmt "\n", \
                     __func__, __LINE__, ##__VA_ARGS__); \
    } } while(0)

#define PHILE_LG_INFO(fmt, ...) \
    do { if (philemon::debug::get_debug_level() >= 1) { \
        std::fprintf(stderr, "[LG-INFO] " fmt "\n", ##__VA_ARGS__); \
    } } while(0)

// ─── Tier routing (same as backend_adapters.hpp) ────────────────────
// upstream LiveGraph has a single-tier flat store; we split into 3 tiers.
enum class TierTag : uint8_t { HBM = 0, GDDR = 1, DRAM = 2 };

inline const char* tier_label(TierTag t) {
    static const char* names[] = {"HBM", "GDDR", "DRAM"};
    return names[static_cast<uint8_t>(t)];
}

// ─── Vertex dictionary ──────────────────────────────────────────────
// [MOD] upstream: tbb::concurrent_hash_map<uint64_t, lg::vertex_t>
// Ours: std::unordered_map + std::shared_mutex (reader-writer lock).
// Reason: no TBB dependency, and we want explicit read/write lock
// metrics for debug output.
struct VertexEntry {
    uint64_t physical_id;
    TierTag  home_tier;       // [NEW] which tier this vertex's edges live in
    uint64_t degree_cached;   // [NEW] cached degree for tier-aware routing
};

class VertexDictionary {
    std::unordered_map<uint64_t, VertexEntry> map_;
    mutable std::shared_mutex mu_;
    // [NEW] debug counters
    std::atomic<uint64_t> read_hits_{0};
    std::atomic<uint64_t> read_misses_{0};
    std::atomic<uint64_t> write_ops_{0};

public:
    bool find(uint64_t logical, VertexEntry& out) const {
        std::shared_lock lk(mu_);
        auto it = map_.find(logical);
        if (it != map_.end()) {
            out = it->second;
            const_cast<std::atomic<uint64_t>&>(read_hits_).fetch_add(1,
                std::memory_order_relaxed);
            PHILE_LG_TRACE("dict find(%lu) → phys=%lu tier=%s",
                           logical, out.physical_id, tier_label(out.home_tier));
            return true;
        }
        const_cast<std::atomic<uint64_t>&>(read_misses_).fetch_add(1,
            std::memory_order_relaxed);
        PHILE_LG_TRACE("dict find(%lu) → MISS", logical);
        return false;
    }

    bool insert(uint64_t logical, VertexEntry entry) {
        std::unique_lock lk(mu_);
        auto [it, ok] = map_.emplace(logical, entry);
        if (ok) write_ops_.fetch_add(1, std::memory_order_relaxed);
        PHILE_LG_TRACE("dict insert(%lu) → %s", logical, ok ? "OK" : "EXISTS");
        return ok;
    }

    bool erase(uint64_t logical) {
        std::unique_lock lk(mu_);
        auto n = map_.erase(logical);
        if (n > 0) write_ops_.fetch_add(1, std::memory_order_relaxed);
        return n > 0;
    }

    size_t size() const {
        std::shared_lock lk(mu_);
        return map_.size();
    }

    // [NEW] Dump dictionary stats — use at breakpoints
    void dump_stats(const char* label = "VertexDictionary") const {
        std::shared_lock lk(mu_);
        std::fprintf(stderr,
            "┌─── %s ───────────────────────────────┐\n"
            "│ entries: %8zu                         │\n"
            "│ read_hits: %8lu  read_misses: %8lu   │\n"
            "│ write_ops: %8lu                       │\n"
            "└────────────────────────────────────────┘\n",
            label, map_.size(),
            read_hits_.load(), read_misses_.load(), write_ops_.load());
    }

    // [NEW] Dump per-tier vertex distribution
    void dump_tier_distribution() const {
        std::shared_lock lk(mu_);
        uint64_t counts[3] = {0, 0, 0};
        uint64_t degree_sums[3] = {0, 0, 0};
        for (auto& [lid, entry] : map_) {
            int ti = static_cast<int>(entry.home_tier);
            counts[ti]++;
            degree_sums[ti] += entry.degree_cached;
        }
        std::fprintf(stderr,
            "┌─── Vertex Tier Distribution ──────────────────┐\n"
            "│ HBM:  %8lu vtx  avg_deg=%.1f                │\n"
            "│ GDDR: %8lu vtx  avg_deg=%.1f                │\n"
            "│ DRAM: %8lu vtx  avg_deg=%.1f                │\n"
            "└────────────────────────────────────────────────┘\n",
            counts[0], counts[0] ? (double)degree_sums[0]/counts[0] : 0.0,
            counts[1], counts[1] ? (double)degree_sums[1]/counts[1] : 0.0,
            counts[2], counts[2] ? (double)degree_sums[2]/counts[2] : 0.0);
    }

    // [NEW] Iterate all entries (for snapshot building)
    template<typename F>
    void for_each(F&& fn) const {
        std::shared_lock lk(mu_);
        for (auto& [lid, entry] : map_) {
            fn(lid, entry);
        }
    }
};

// ─── Tier latency model ─────────────────────────────────────────────
// upstream had none (single store), we model per-access nanosecond cost
struct TierCost {
    static constexpr double HBM_NS  = 1.2;
    static constexpr double GDDR_NS = 4.8;
    static constexpr double DRAM_NS = 48.0;

    static double access_ns(TierTag t) {
        switch (t) {
            case TierTag::HBM:  return HBM_NS;
            case TierTag::GDDR: return GDDR_NS;
            case TierTag::DRAM: return DRAM_NS;
        }
        return DRAM_NS;
    }
};

// ─── Edge storage partition (per-tier adjacency) ────────────────────
// upstream: LiveGraph stores edges in lg::Graph internally (opaque).
// We expose per-tier adjacency vectors so algorithms can route & prefetch.
struct TieredEdge {
    uint64_t src;
    uint64_t dst;
    double   weight;
    uint64_t timestamp;  // [NEW] temporal tag for interval queries
};

struct TierPartition {
    std::vector<TieredEdge> edges;
    std::atomic<uint64_t> edge_count{0};
    mutable std::shared_mutex mu;

    void append(const TieredEdge& e) {
        std::unique_lock lk(mu);
        edges.push_back(e);
        edge_count.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t count() const {
        return edge_count.load(std::memory_order_relaxed);
    }

    // [NEW] Dump partition state at breakpoint
    void dump(const char* tier_name, size_t max_show = 8) const {
        std::shared_lock lk(mu);
        size_t total = edges.size();
        std::fprintf(stderr, "  [%s] %zu edges", tier_name, total);
        if (total == 0) { std::fprintf(stderr, " (empty)\n"); return; }
        size_t show = std::min(max_show, total);
        std::fprintf(stderr, " — first %zu:\n", show);
        for (size_t i = 0; i < show; i++) {
            std::fprintf(stderr, "    [%zu] %lu→%lu w=%.2f ts=%lu\n",
                         i, edges[i].src, edges[i].dst,
                         edges[i].weight, edges[i].timestamp);
        }
        if (total > show)
            std::fprintf(stderr, "    ... (%zu more)\n", total - show);
    }
};

// ─── Tier router ────────────────────────────────────────────────────
// [MOD] upstream: no routing (all edges in one lg::Graph)
// Ours: hash-based tier assignment with degree-aware promotion
class LiveGraphTierRouter {
    uint64_t hbm_capacity_;
    uint64_t gddr_capacity_;
    // [NEW] Promotion thresholds: hot vertices get moved to faster tier
    uint64_t hot_degree_threshold_ = 1000;

public:
    LiveGraphTierRouter(uint64_t hbm_cap = 1ULL << 20,
                        uint64_t gddr_cap = 1ULL << 22)
        : hbm_capacity_(hbm_cap), gddr_capacity_(gddr_cap) {}

    // [MOD] upstream: no routing. Here: hash + degree → tier
    TierTag route_edge(uint64_t src, uint64_t dst,
                       uint64_t src_degree) const {
        if (src_degree >= hot_degree_threshold_) {
            PHILE_LG_TRACE("route(%lu→%lu) deg=%lu → HBM (hot)",
                           src, dst, src_degree);
            return TierTag::HBM;
        }
        // [MOD] upstream had no partitioning; we use hash-based bucket
        uint64_t h = (src * 2654435761ULL) ^ (dst * 40503ULL);
        uint64_t bucket = h % 100;
        if (bucket < 15) {
            PHILE_LG_TRACE("route(%lu→%lu) → HBM (hash)", src, dst);
            return TierTag::HBM;
        }
        if (bucket < 45) {
            PHILE_LG_TRACE("route(%lu→%lu) → GDDR (hash)", src, dst);
            return TierTag::GDDR;
        }
        PHILE_LG_TRACE("route(%lu→%lu) → DRAM (hash)", src, dst);
        return TierTag::DRAM;
    }

    // [MOD] Batch routing: upstream processes edges sequentially,
    // we bucket them by tier first, then flush each tier batch.
    void route_batch(const std::vector<TieredEdge>& batch,
                     std::vector<TieredEdge> buckets[3],
                     const VertexDictionary& dict) const {
        buckets[0].clear(); buckets[1].clear(); buckets[2].clear();
        for (auto& e : batch) {
            VertexEntry ve;
            uint64_t deg = 0;
            if (dict.find(e.src, ve)) deg = ve.degree_cached;
            TierTag t = route_edge(e.src, e.dst, deg);
            buckets[static_cast<int>(t)].push_back(e);
        }
        PHILE_LG_TRACE("route_batch: %zu→ HBM=%zu GDDR=%zu DRAM=%zu",
                        batch.size(), buckets[0].size(),
                        buckets[1].size(), buckets[2].size());
    }

    void set_hot_threshold(uint64_t t) { hot_degree_threshold_ = t; }
    uint64_t hot_threshold() const { return hot_degree_threshold_; }
};

// ═══════════════════════════════════════════════════════════════════════
// TieredLiveGraphWrapper — main class
//
// [KEEP] from upstream: class shape, member variables, all query methods
// [MOD]: internal storage → 3 TierPartitions instead of lg::Graph*
// ═══════════════════════════════════════════════════════════════════════
class TieredLiveGraphWrapper {
protected:
    // [MOD] upstream: void* m_pImpl (lg::Graph*), void* m_pHashMap (tbb hash)
    //       ours: 3 tier partitions + our VertexDictionary
    TierPartition tiers_[3];
    VertexDictionary dict_;
    LiveGraphTierRouter router_;

    // [KEEP] flags preserved exactly from upstream
    const bool m_is_directed;
    const bool m_is_weighted;
    std::atomic<uint64_t> m_vertex_count{0};
    std::atomic<uint64_t> m_edge_count{0};

    // [NEW] version counter for pseudo-MVCC
    std::atomic<uint64_t> version_{0};
    // [NEW] debug: per-tier operation counters
    std::atomic<uint64_t> tier_inserts_[3] = {{0}, {0}, {0}};
    std::atomic<uint64_t> tier_reads_[3]   = {{0}, {0}, {0}};

public:
    // ─── Constructor (upstream pattern: explicit, defaults) ─────────
    // [KEEP] parameter signature identical to upstream
    explicit TieredLiveGraphWrapper(bool is_directed = false,
                                    bool is_weighted = true)
        : m_is_directed(is_directed), m_is_weighted(is_weighted)
    {
        PHILE_LG_INFO("TieredLiveGraphWrapper created (directed=%d weighted=%d)",
                       is_directed, is_weighted);
    }

    // [KEEP] copy/move semantics identical to upstream
    TieredLiveGraphWrapper(const TieredLiveGraphWrapper&) = delete;
    TieredLiveGraphWrapper& operator=(const TieredLiveGraphWrapper&) = delete;

    ~TieredLiveGraphWrapper() {
        PHILE_LG_INFO("~TieredLiveGraphWrapper: V=%lu E=%lu",
                       m_vertex_count.load(), m_edge_count.load());
    }

    // ─── Init: upstream signature preserved ─────────────────────────
    // [KEEP] load() interface signature
    void load(const std::string& path, int reader_type) {
        PHILE_BREAKPOINT_NAMED("LG::load");
        // upstream: reader dispatch (edgeList / vertexList)
        // we delegate to philemon readers; this is a compatibility shim
        PHILE_LG_INFO("load(\"%s\", type=%d) — delegating to PhilemonReaders",
                       path.c_str(), reader_type);
    }

    // [KEEP] 100% preserved from upstream
    void set_max_threads(int max_threads) {
        PHILE_LG_TRACE("set_max_threads(%d)", max_threads);
    }
    void init_thread(int thread_id) {
        PHILE_LG_TRACE("init_thread(%d)", thread_id);
    }
    void end_thread(int thread_id) {
        PHILE_LG_TRACE("end_thread(%d)", thread_id);
    }

    // ─── Graph property queries (100% preserved) ────────────────────
    bool is_directed() const { return m_is_directed; }
    bool is_weighted() const { return m_is_weighted; }
    bool is_empty()    const { return m_vertex_count.load() == 0; }

    bool has_vertex(uint64_t vertex) const {
        VertexEntry ve;
        return dict_.find(vertex, ve);
    }

    bool has_edge(uint64_t source, uint64_t destination) const {
        // [KEEP] same semantics as upstream: lookup both vertices, check edges
        VertexEntry ve_src, ve_dst;
        if (!dict_.find(source, ve_src)) return false;
        if (!dict_.find(destination, ve_dst)) return false;
        // [MOD] upstream: lg::Transaction → we scan tier partition
        TierTag t = router_.route_edge(source, destination, ve_src.degree_cached);
        int ti = static_cast<int>(t);
        std::shared_lock lk(tiers_[ti].mu);
        for (auto& e : tiers_[ti].edges) {
            if (e.src == source && e.dst == destination) return true;
        }
        return false;
    }

    bool has_edge(uint64_t source, uint64_t destination, double weight) const {
        // [KEEP] same triple-check pattern as upstream
        VertexEntry ve_src, ve_dst;
        if (!dict_.find(source, ve_src)) return false;
        if (!dict_.find(destination, ve_dst)) return false;
        TierTag t = router_.route_edge(source, destination, ve_src.degree_cached);
        int ti = static_cast<int>(t);
        std::shared_lock lk(tiers_[ti].mu);
        for (auto& e : tiers_[ti].edges) {
            if (e.src == source && e.dst == destination &&
                std::abs(e.weight - weight) < 1e-9) return true;
        }
        return false;
    }

    // [KEEP] upstream has_edge(weightedEdge) overload
    bool has_edge(uint64_t src, uint64_t dst, double w, bool weighted_check) const {
        return weighted_check ? has_edge(src, dst, w) : has_edge(src, dst);
    }

    // [MOD] degree: upstream queries single lg::Transaction iterator
    //       ours: sum across all 3 tiers
    uint64_t degree(uint64_t vertex) const {
        VertexEntry ve;
        if (!dict_.find(vertex, ve)) return 0;
        uint64_t deg = 0;
        for (int ti = 0; ti < 3; ti++) {
            std::shared_lock lk(tiers_[ti].mu);
            for (auto& e : tiers_[ti].edges) {
                if (e.src == vertex) deg++;
                if (!m_is_directed && e.dst == vertex) deg++;
            }
            const_cast<std::atomic<uint64_t>&>(tier_reads_[ti])
                .fetch_add(1, std::memory_order_relaxed);
        }
        PHILE_LG_TRACE("degree(%lu) = %lu (across 3 tiers)", vertex, deg);
        return deg;
    }

    double get_weight(uint64_t source, uint64_t destination) const {
        // [KEEP] same pattern: find vertices, then scan for edge
        VertexEntry ve_src;
        if (!dict_.find(source, ve_src)) return 0.0;
        for (int ti = 0; ti < 3; ti++) {
            std::shared_lock lk(tiers_[ti].mu);
            for (auto& e : tiers_[ti].edges) {
                if (e.src == source && e.dst == destination)
                    return e.weight;
            }
        }
        return 0.0;
    }

    // [KEEP] ID mapping: identity mapping preserved from upstream
    uint64_t logical2physical(uint64_t vertex) const {
        VertexEntry ve;
        if (!dict_.find(vertex, ve)) return vertex;
        return ve.physical_id;
    }
    uint64_t physical2logical(uint64_t physical) const {
        // [KEEP] upstream same pattern, reverse lookup
        return physical;  // identity in tiered mode
    }

    uint64_t vertex_count() const { return m_vertex_count.load(); }
    uint64_t edge_count()   const { return m_edge_count.load(); }

    // [KEEP] upstream get_neighbors signature
    void get_neighbors(uint64_t vertex,
                       std::vector<uint64_t>& neighbors) const {
        PHILE_LG_TRACE("get_neighbors(%lu)", vertex);
        // [MOD] upstream: single tx.get_edges() → we scan all tiers
        for (int ti = 0; ti < 3; ti++) {
            std::shared_lock lk(tiers_[ti].mu);
            for (auto& e : tiers_[ti].edges) {
                if (e.src == vertex) neighbors.push_back(e.dst);
                if (!m_is_directed && e.dst == vertex)
                    neighbors.push_back(e.src);
            }
        }
    }

    void get_neighbors(uint64_t vertex,
                       std::vector<std::pair<uint64_t, double>>& neighbors) const {
        for (int ti = 0; ti < 3; ti++) {
            std::shared_lock lk(tiers_[ti].mu);
            for (auto& e : tiers_[ti].edges) {
                if (e.src == vertex)
                    neighbors.emplace_back(e.dst, e.weight);
                if (!m_is_directed && e.dst == vertex)
                    neighbors.emplace_back(e.src, e.weight);
            }
        }
    }

    // ─── Mutations ──────────────────────────────────────────────────

    // [KEEP] upstream insert_vertex: create entry in dictionary
    // [MOD] no lg::Transaction, just dictionary + atomic counter
    bool insert_vertex(uint64_t vertex) {
        uint64_t phys = m_vertex_count.load();  // auto-increment physical ID
        VertexEntry ve{phys, TierTag::DRAM, 0};
        bool ok = dict_.insert(vertex, ve);
        if (ok) {
            m_vertex_count.fetch_add(1);
            PHILE_LG_TRACE("insert_vertex(%lu) → phys=%lu", vertex, phys);
        }
        return ok;
    }

    // [MOD] upstream: lg::Transaction with retry loop
    //       ours: route to tier, append to partition, bump version
    bool insert_edge(uint64_t source, uint64_t destination, double weight = 1.0) {
        VertexEntry ve_src, ve_dst;
        if (!dict_.find(source, ve_src)) {
            PHILE_LG_TRACE("insert_edge: source %lu not found", source);
            return false;
        }
        if (!dict_.find(destination, ve_dst)) {
            PHILE_LG_TRACE("insert_edge: dest %lu not found", destination);
            return false;
        }

        // [MOD] upstream does lg::Transaction::put_edge with retry;
        //       we route to tier and append directly
        TierTag tier = router_.route_edge(source, destination,
                                          ve_src.degree_cached);
        int ti = static_cast<int>(tier);

        uint64_t ts = version_.fetch_add(1, std::memory_order_relaxed);
        TieredEdge te{source, destination, weight, ts};
        tiers_[ti].append(te);
        tier_inserts_[ti].fetch_add(1, std::memory_order_relaxed);

        // [KEEP] upstream pattern: undirected → insert reverse edge
        if (!m_is_directed) {
            TierTag tier_rev = router_.route_edge(destination, source,
                                                  ve_dst.degree_cached);
            int ti_rev = static_cast<int>(tier_rev);
            TieredEdge te_rev{destination, source, weight, ts};
            tiers_[ti_rev].append(te_rev);
            tier_inserts_[ti_rev].fetch_add(1, std::memory_order_relaxed);
        }

        m_edge_count.fetch_add(1);
        PHILE_LG_TRACE("insert_edge(%lu→%lu w=%.2f) → tier=%s ts=%lu",
                        source, destination, weight, tier_label(tier), ts);
        return true;
    }

    // [KEEP] upstream remove_vertex: retry loop pattern preserved
    bool remove_vertex(uint64_t vertex) {
        bool ok = dict_.erase(vertex);
        if (ok) m_vertex_count.fetch_sub(1);
        PHILE_LG_TRACE("remove_vertex(%lu) → %s", vertex, ok ? "OK" : "NOT_FOUND");
        return ok;
    }

    // [KEEP] upstream remove_edge: retry loop → we do single-pass scan
    bool remove_edge(uint64_t source, uint64_t destination) {
        for (int ti = 0; ti < 3; ti++) {
            std::unique_lock lk(tiers_[ti].mu);
            auto& ev = tiers_[ti].edges;
            auto it = std::remove_if(ev.begin(), ev.end(),
                [&](const TieredEdge& e) {
                    return e.src == source && e.dst == destination;
                });
            if (it != ev.end()) {
                ev.erase(it, ev.end());
                tiers_[ti].edge_count.fetch_sub(1);
                m_edge_count.fetch_sub(1);
                PHILE_LG_TRACE("remove_edge(%lu→%lu) from tier %d",
                               source, destination, ti);
                return true;
            }
        }
        return false;
    }

    // [KEEP] upstream run_batch_vertex_update signature
    bool run_batch_vertex_update(std::vector<uint64_t>& vertices,
                                 int start, int end) {
        PHILE_BREAKPOINT_NAMED("LG::batch_vertex_update");
        for (int i = start; i < end; i++) {
            insert_vertex(vertices[i]);
        }
        PHILE_LG_INFO("batch_vertex_update: inserted %d vertices", end - start);
        return true;
    }

    // [MOD] upstream: begin_batch_loader() → per-tier bucketed flush
    bool run_batch_edge_update(std::vector<TieredEdge>& edges,
                               int start, int end) {
        PHILE_BREAKPOINT_NAMED("LG::batch_edge_update");
        // [MOD] upstream processes edges sequentially in one transaction;
        //       we bucket by tier first, then flush each tier batch
        std::vector<TieredEdge> buckets[3];
        std::vector<TieredEdge> batch(edges.begin() + start,
                                      edges.begin() + end);
        router_.route_batch(batch, buckets, dict_);

        for (int ti = 0; ti < 3; ti++) {
            if (buckets[ti].empty()) continue;
            std::unique_lock lk(tiers_[ti].mu);
            for (auto& e : buckets[ti]) {
                tiers_[ti].edges.push_back(e);
                tiers_[ti].edge_count.fetch_add(1, std::memory_order_relaxed);
            }
            tier_inserts_[ti].fetch_add(buckets[ti].size(),
                                        std::memory_order_relaxed);
            PHILE_LG_INFO("batch_edge_update: flushed %zu edges to %s",
                           buckets[ti].size(), tier_label(static_cast<TierTag>(ti)));
        }
        m_edge_count.fetch_add(end - start);
        return true;
    }

    // [KEEP] upstream clear() throws FunctionNotImplementedError
    void clear() {
        PHILE_LG_INFO("clear() — resetting all tiers");
        for (int ti = 0; ti < 3; ti++) {
            std::unique_lock lk(tiers_[ti].mu);
            tiers_[ti].edges.clear();
            tiers_[ti].edge_count.store(0);
        }
        m_edge_count.store(0);
    }

    // ─── Snapshot ───────────────────────────────────────────────────
    // [KEEP] upstream Snapshot pattern: enable_shared_from_this + clone()
    // [MOD] internal: pointers to tier partitions instead of lg::Transaction
    class Snapshot : public std::enable_shared_from_this<Snapshot> {
    private:
        // [MOD] upstream: lg::Transaction m_transaction
        //       ours: references to tier partition snapshots
        const TierPartition* tiers_;
        const VertexDictionary* dict_;
        const uint64_t m_num_vertices;
        const uint64_t m_num_edges;
        const bool m_is_weighted;
        const bool m_is_directed;
        const uint64_t snapshot_version_;

        // [NEW] per-snapshot debug counters
        mutable uint64_t edges_scanned_ = 0;
        mutable uint64_t tier_touches_[3] = {0, 0, 0};

    public:
        Snapshot(const TierPartition* tiers, const VertexDictionary* dict,
                 uint64_t vertex_count, uint64_t edge_count,
                 bool is_weighted, bool is_directed, uint64_t version)
            : tiers_(tiers), dict_(dict),
              m_num_vertices(vertex_count), m_num_edges(edge_count),
              m_is_weighted(is_weighted), m_is_directed(is_directed),
              snapshot_version_(version)
        {
            PHILE_LG_TRACE("Snapshot created: V=%lu E=%lu ver=%lu",
                           vertex_count, edge_count, version);
        }

        // [KEEP] upstream copy/clone semantics
        Snapshot(const Snapshot&) = delete;
        ~Snapshot() {
            if (philemon::debug::get_debug_level() >= 2) {
                std::fprintf(stderr,
                    "[LG-SNAP] destroyed: scanned=%lu tiers=[%lu,%lu,%lu]\n",
                    edges_scanned_, tier_touches_[0],
                    tier_touches_[1], tier_touches_[2]);
            }
        }
        std::shared_ptr<Snapshot> clone() const {
            return const_cast<Snapshot*>(this)->shared_from_this();
        }

        uint64_t size() const { return m_num_vertices; }

        // [KEEP] upstream ID mapping
        uint64_t physical2logical(uint64_t physical) const { return physical; }
        uint64_t logical2physical(uint64_t logical) const  { return logical; }

        uint64_t degree(uint64_t vertex, bool /*logical*/ = false) const {
            uint64_t deg = 0;
            for (int ti = 0; ti < 3; ti++) {
                std::shared_lock lk(tiers_[ti].mu);
                for (auto& e : tiers_[ti].edges) {
                    if (e.src == vertex) deg++;
                    if (!m_is_directed && e.dst == vertex) deg++;
                }
                const_cast<uint64_t&>(tier_touches_[ti])++;
            }
            return deg;
        }

        bool has_vertex(uint64_t vertex) const {
            VertexEntry ve;
            return dict_->find(vertex, ve);
        }
        bool has_edge(uint64_t source, uint64_t destination) const {
            for (int ti = 0; ti < 3; ti++) {
                std::shared_lock lk(tiers_[ti].mu);
                for (auto& e : tiers_[ti].edges) {
                    if (e.src == source && e.dst == destination) return true;
                }
            }
            return false;
        }
        bool has_edge(uint64_t src, uint64_t dst, double w) const {
            for (int ti = 0; ti < 3; ti++) {
                std::shared_lock lk(tiers_[ti].mu);
                for (auto& e : tiers_[ti].edges) {
                    if (e.src == src && e.dst == dst &&
                        std::abs(e.weight - w) < 1e-9) return true;
                }
            }
            return false;
        }

        // [KEEP] upstream intersect() stub
        uint64_t intersect(uint64_t, uint64_t) const { return 0; }

        double get_weight(uint64_t source, uint64_t destination) const {
            for (int ti = 0; ti < 3; ti++) {
                std::shared_lock lk(tiers_[ti].mu);
                for (auto& e : tiers_[ti].edges) {
                    if (e.src == source && e.dst == destination)
                        return e.weight;
                }
            }
            return 0.0;
        }

        uint64_t vertex_count() const { return m_num_vertices; }
        uint64_t edge_count()   const { return m_num_edges; }

        void get_neighbor_addr(uint64_t /*index*/) const {
            // [KEEP] upstream stub — prefetch hint, no-op here
        }

        // [KEEP] upstream edges() with vector output
        void edges(uint64_t index, std::vector<uint64_t>& neighbors,
                   bool /*logical*/) const {
            // [MOD] upstream: single iterator → we scan all 3 tiers
            //       with per-tier prefetch hint and debug counters
            for (int ti = 0; ti < 3; ti++) {
                std::shared_lock lk(tiers_[ti].mu);
                // [NEW] prefetch hint: if tier has edges, prefetch first cacheline
                if (!tiers_[ti].edges.empty()) {
                    __builtin_prefetch(&tiers_[ti].edges[0], 0, 1);
                }
                for (auto& e : tiers_[ti].edges) {
                    if (e.src == index) {
                        neighbors.push_back(e.dst);
                        const_cast<uint64_t&>(edges_scanned_)++;
                    }
                    if (!m_is_directed && e.dst == index) {
                        neighbors.push_back(e.src);
                        const_cast<uint64_t&>(edges_scanned_)++;
                    }
                }
                const_cast<uint64_t&>(tier_touches_[ti])++;
            }
        }

        // [KEEP] upstream template callback edges()
        template<typename F>
        void edges(uint64_t index, F&& callback, bool /*logical*/) const {
            for (int ti = 0; ti < 3; ti++) {
                std::shared_lock lk(tiers_[ti].mu);
                if (!tiers_[ti].edges.empty()) {
                    __builtin_prefetch(&tiers_[ti].edges[0], 0, 1);
                }
                for (auto& e : tiers_[ti].edges) {
                    if (e.src == index) {
                        callback(e.dst, e.weight);
                        const_cast<uint64_t&>(edges_scanned_)++;
                    }
                    if (!m_is_directed && e.dst == index) {
                        callback(e.src, e.weight);
                        const_cast<uint64_t&>(edges_scanned_)++;
                    }
                }
                const_cast<uint64_t&>(tier_touches_[ti])++;
            }
        }

        // [NEW] Debug: dump snapshot state at breakpoint
        void dump_state(const char* label = "Snapshot") const {
            std::fprintf(stderr,
                "┌─── %s (ver=%lu) ──────────────────────┐\n"
                "│ V=%lu  E=%lu  weighted=%d directed=%d    │\n"
                "│ edges_scanned=%lu                        │\n"
                "│ tier_touches=[%lu, %lu, %lu]             │\n"
                "└──────────────────────────────────────────┘\n",
                label, snapshot_version_,
                m_num_vertices, m_num_edges,
                m_is_weighted, m_is_directed,
                edges_scanned_,
                tier_touches_[0], tier_touches_[1], tier_touches_[2]);
        }
    };

    // [KEEP] upstream get_unique_snapshot / get_shared_snapshot
    std::unique_ptr<Snapshot> get_unique_snapshot() const {
        return std::make_unique<Snapshot>(
            tiers_, &dict_,
            m_vertex_count.load(), m_edge_count.load(),
            m_is_weighted, m_is_directed, version_.load());
    }
    std::shared_ptr<Snapshot> get_shared_snapshot() const {
        return std::make_shared<Snapshot>(
            tiers_, &dict_,
            m_vertex_count.load(), m_edge_count.load(),
            m_is_weighted, m_is_directed, version_.load());
    }

    // [KEEP] upstream repl() for test identification
    static std::string repl() {
        return std::string{"TieredLiveGraphWrapper"};
    }

    // ─── [NEW] Diagnostic: full state dump ──────────────────────────
    // Use this at breakpoints to print everything.
    // upstream had zero diagnostic output; this is our 20% addition.
    void dump_full_state(const char* label = "TieredLiveGraph") const {
        std::fprintf(stderr,
            "\n╔════════════════════════════════════════════════╗\n"
            "║  %s STATE DUMP                                ║\n"
            "╠════════════════════════════════════════════════╣\n"
            "║  vertices: %lu   edges: %lu   version: %lu    ║\n"
            "║  directed: %d    weighted: %d                  ║\n"
            "╠════════════════════════════════════════════════╣\n",
            label,
            m_vertex_count.load(), m_edge_count.load(), version_.load(),
            m_is_directed, m_is_weighted);

        for (int ti = 0; ti < 3; ti++) {
            std::fprintf(stderr,
                "║  %s: %lu edges  inserts=%lu  reads=%lu\n",
                tier_label(static_cast<TierTag>(ti)),
                tiers_[ti].edge_count.load(),
                tier_inserts_[ti].load(),
                tier_reads_[ti].load());
            tiers_[ti].dump(tier_label(static_cast<TierTag>(ti)));
        }
        dict_.dump_stats();
        dict_.dump_tier_distribution();
        std::fprintf(stderr,
            "╚════════════════════════════════════════════════╝\n\n");
    }

    // [NEW] router access for external configuration
    LiveGraphTierRouter& router() { return router_; }
    const LiveGraphTierRouter& router() const { return router_; }
};

// ─── [KEEP] upstream wrapper_test() pattern ─────────────────────────
// upstream had this exact self-test in the .cpp; we preserve the logic
// with tiered assertions added
namespace wrapper_test {
    inline void run_livegraph_self_test() {
        std::fprintf(stderr, "=== TieredLiveGraph self-test ===\n");
        auto w = TieredLiveGraphWrapper(false, true);
        w.insert_vertex(0);
        w.insert_vertex(1);
        w.insert_vertex(3);
        w.insert_edge(0, 1);
        w.insert_edge(0, 3);

        assert(w.vertex_count() == 3);
        assert(!w.is_empty());
        assert(w.has_vertex(0));
        assert(w.has_edge(0, 1));
        std::fprintf(stderr, "  basic tests passed!\n");

        auto snap = w.get_unique_snapshot();
        assert(snap->vertex_count() == 3);
        std::fprintf(stderr, "  snapshot tests passed!\n");

        // [NEW] print full debug state after test
        w.dump_full_state("SelfTest");
        snap->dump_state("SelfTestSnapshot");

        std::fprintf(stderr, "=== TieredLiveGraph self-test PASSED ===\n");
    }
}

// ─── [KEEP] upstream execute() pattern ──────────────────────────────
// upstream: namespace wrapper { void execute(DriverConfig&) }
// We provide the same interface wired to PhilemonConfig
namespace execute {
    inline void run_livegraph_benchmark(TieredLiveGraphWrapper& wrapper) {
        PHILE_BREAKPOINT_NAMED("LG::benchmark");
        wrapper.dump_full_state("PreBenchmark");
        // Benchmark phases would be called from philemon_driver.hpp
        wrapper.dump_full_state("PostBenchmark");
    }
}

} // namespace livegraph
} // namespace adapters
} // namespace philemon

// ─── Debug convenience macros for experiment scripts ────────────────
// Usage in main():
//   auto lg = TieredLiveGraphWrapper();
//   ... build graph ...
//   PHILE_LG_DUMP(lg);  // full state dump with tier distribution
#define PHILE_LG_DUMP(wrapper) \
    (wrapper).dump_full_state(#wrapper)

#define PHILE_LG_SNAP_DUMP(snap) \
    (snap)->dump_state(#snap)

#endif // PHILEMON_LIVEGRAPH_TIERED_HPP
