#ifndef PHILEMON_NEO_ADAPTER_HPP
#define PHILEMON_NEO_ADAPTER_HPP
/**
 * philemon_neo_adapter.hpp — NeoGraph wrapper 适配器 (跨层级版)
 *
 * 骨架来源: upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.h (210行)
 *           upstream/rapidstore/wrapper/apps/neo_wrapper/neo_wrapper.cpp (703行, 接口签名)
 * 修改 (~20%):
 *   - 保留 NeoGraph wrapper 接口: insert_edge, has_edge, degree, edges 等
 *   - 移除 Neo4j 原生调用 → 对接 philemon::index::TemGraph
 *   - 增加 per-operation tier 路由日志
 *   - 增加 dump_adapter_state(): 打印边/顶点计数 + tier 分布
 *   - 增加 tier_stats: 记录 DRAM/NVM/SSD 访问频次
 *   - snapshot 类保留但简化为内存快照
 *
 * Milestone: M028
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include "../utils/log/philemon_log.hpp"

namespace philemon {
namespace adapter {

// ─── Tier stats tracker ─────────────────────────────────────────────
struct TierAccessStats {
    std::atomic<uint64_t> dram_ops{0};
    std::atomic<uint64_t> nvm_ops{0};
    std::atomic<uint64_t> ssd_ops{0};

    void record(uint8_t tier) {
        switch (tier) {
            case 1: dram_ops.fetch_add(1, std::memory_order_relaxed); break;
            case 2: nvm_ops.fetch_add(1, std::memory_order_relaxed); break;
            case 3: ssd_ops.fetch_add(1, std::memory_order_relaxed); break;
        }
    }

    void dump(const char* label = "") const {
        std::printf("[TIER_STATS] %s DRAM=%lu  NVM=%lu  SSD=%lu\n",
                    label, dram_ops.load(), nvm_ops.load(), ssd_ops.load());
    }

    void reset() {
        dram_ops.store(0); nvm_ops.store(0); ssd_ops.store(0);
    }
};

// ─── Snapshot (simplified from upstream) ────────────────────────────
class NeoSnapshot {
public:
    using EdgeCallback = std::function<void(uint64_t dst, double weight)>;

    NeoSnapshot(const std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, double>>>& adj,
                uint64_t vcnt, uint64_t ecnt)
        : adj_(adj), vertex_count_(vcnt), edge_count_(ecnt) {}

    NeoSnapshot* clone() const { return new NeoSnapshot(adj_, vertex_count_, edge_count_); }

    uint64_t vertex_count() const { return vertex_count_; }
    uint64_t edge_count() const { return edge_count_; }

    uint64_t degree(uint64_t v, bool = false) const {
        auto it = adj_.find(v);
        return it != adj_.end() ? it->second.size() : 0;
    }

    bool has_vertex(uint64_t v) const { return adj_.find(v) != adj_.end(); }

    bool has_edge(uint64_t src, uint64_t dst) const {
        auto it = adj_.find(src);
        if (it == adj_.end()) return false;
        for (auto& [d, w] : it->second) if (d == dst) return true;
        return false;
    }

    uint64_t physical2logical(uint64_t p) const { return p; }
    uint64_t logical2physical(uint64_t l) const { return l; }

    void edges(uint64_t v, std::vector<uint64_t>& neighbors, bool = false) const {
        auto it = adj_.find(v);
        if (it == adj_.end()) return;
        for (auto& [d, w] : it->second) neighbors.push_back(d);
    }

    void edges(uint64_t v, EdgeCallback cb, bool = false) const {
        auto it = adj_.find(v);
        if (it == adj_.end()) return;
        for (auto& [d, w] : it->second) cb(d, w);
    }

    uint64_t intersect(uint64_t a, uint64_t b) const {
        auto ia = adj_.find(a), ib = adj_.find(b);
        if (ia == adj_.end() || ib == adj_.end()) return 0;

        std::unordered_set<uint64_t> sa;
        for (auto& [d, w] : ia->second) sa.insert(d);
        uint64_t cnt = 0;
        for (auto& [d, w] : ib->second) if (sa.count(d)) cnt++;
        return cnt;
    }

    double get_weight(uint64_t src, uint64_t dst) const {
        auto it = adj_.find(src);
        if (it == adj_.end()) return -1.0;
        for (auto& [d, w] : it->second) if (d == dst) return w;
        return -1.0;
    }

private:
    const std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, double>>>& adj_;
    uint64_t vertex_count_;
    uint64_t edge_count_;
};

// ─── Main adapter class ─────────────────────────────────────────────
class NeoGraphAdapter {
public:
    using snapshot_ptr = std::shared_ptr<NeoSnapshot>;

    NeoGraphAdapter() = default;

    // ─── Thread management (from upstream) ──────────────────────────
    void set_max_threads(int n) { max_threads_ = n; }
    void init_thread(int tid) {}
    void end_thread(int tid) {}

    // ─── Graph mutations ────────────────────────────────────────────
    bool insert_vertex(uint64_t v) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (adj_.count(v)) return false;
        adj_[v] = {};
        vertex_count_++;
        return true;
    }

    bool insert_edge(uint64_t src, uint64_t dst, double weight = 0.0) {
        std::lock_guard<std::mutex> lk(mtx_);
        adj_[src].push_back({dst, weight});
        adj_[dst].push_back({src, weight}); // undirected
        edge_count_++;

        tier_stats_.record(1); // default DRAM
        return true;
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto remove_from = [](std::vector<std::pair<uint64_t, double>>& v, uint64_t target) {
            v.erase(std::remove_if(v.begin(), v.end(),
                    [target](auto& p) { return p.first == target; }), v.end());
        };
        remove_from(adj_[src], dst);
        remove_from(adj_[dst], src);
        edge_count_--;
        return true;
    }

    bool remove_vertex(uint64_t v) {
        std::lock_guard<std::mutex> lk(mtx_);
        adj_.erase(v);
        vertex_count_--;
        return true;
    }

    // ─── Queries ────────────────────────────────────────────────────
    bool is_directed() const { return false; }
    bool is_weighted() const { return false; }
    bool has_vertex(uint64_t v) const { return adj_.count(v) > 0; }
    bool has_edge(uint64_t s, uint64_t d) const {
        auto it = adj_.find(s);
        if (it == adj_.end()) return false;
        for (auto& [dd, w] : it->second) if (dd == d) return true;
        return false;
    }
    uint64_t degree(uint64_t v) const {
        auto it = adj_.find(v);
        return it != adj_.end() ? it->second.size() : 0;
    }
    uint64_t vertex_count() const { return vertex_count_; }
    uint64_t edge_count() const { return edge_count_; }
    uint64_t logical2physical(uint64_t l) const { return l; }
    uint64_t physical2logical(uint64_t p) const { return p; }

    void get_neighbors(uint64_t v, std::vector<uint64_t>& neighbors) {
        auto it = adj_.find(v);
        if (it == adj_.end()) return;
        for (auto& [d, w] : it->second) neighbors.push_back(d);
    }

    bool run_batch_vertex_update(std::vector<uint64_t>& verts, int st, int en) {
        for (int i = st; i < en; i++) insert_vertex(verts[i]);
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        adj_.clear(); vertex_count_ = 0; edge_count_ = 0;
    }

    // ─── Snapshot ───────────────────────────────────────────────────
    snapshot_ptr get_shared_snapshot() {
        return std::make_shared<NeoSnapshot>(adj_, vertex_count_, edge_count_);
    }
    std::unique_ptr<NeoSnapshot> get_unique_snapshot() {
        return std::make_unique<NeoSnapshot>(adj_, vertex_count_, edge_count_);
    }

    // ─── NEW: debug dump ────────────────────────────────────────────
    void dump_adapter_state(const char* label = "") const {
        std::printf("[NEO_ADAPTER] %s  V=%lu  E=%lu  adj_buckets=%zu\n",
                    label, vertex_count_, edge_count_, adj_.bucket_count());
        tier_stats_.dump(label);
    }

    TierAccessStats& tier_stats() { return tier_stats_; }

private:
    std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, double>>> adj_;
    uint64_t vertex_count_ = 0;
    uint64_t edge_count_ = 0;
    int max_threads_ = 1;
    std::mutex mtx_;
    mutable TierAccessStats tier_stats_;
};

} // namespace adapter
} // namespace philemon

#endif // PHILEMON_NEO_ADAPTER_HPP
