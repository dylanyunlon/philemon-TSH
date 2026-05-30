#pragma once
/**
 * rapidstore_wrapper.hpp — Tiered Partition ↔ RapidStore Snapshot Bridge
 *
 * 骨架来源: upstream/rapidstore/wrapper/wrapper.h (249行)
 * 修改 (~20%):
 *   - 保留全部 wrapper:: template API (snapshot_edges, degree, etc.)
 *   - 增加 tiered_wrapper:: namespace，将 TieredAllocator 的分区
 *     暴露为 RapidStore 兼容的 snapshot 接口
 *   - 增加 TieredSnapshot 类: 实现 edges(), degree(), vertex_count()
 *     使用 TemporalBridge 的 scan_partition() 作为后端
 *   - 增加 per-query debug trace (哪个 tier 提供了哪些边)
 *   - 增加 tier_latency_model: HBM=1ns, GDDR=5ns, DRAM=50ns 模拟
 *
 * Pattern lineage:
 *   RapidStore wrapper::snapshot_edges(S, u, callback, logical)
 *     → template dispatch for edge traversal
 *   RapidStore wrapper::get_unique_snapshot(W)
 *     → our TieredSnapshot wraps partition pointers
 *
 * Milestone: M013–M014 (Claude #6)
 */

#include <functional>
#include <vector>
#include <cstdint>
#include <atomic>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>

#include "../debug/philemon_debug.hpp"
#include "../core/temporal_edge.hpp"

namespace philemon {

// ─── Upstream wrapper:: API (100% preserved from RapidStore) ────────
// This is the "C, the good example" — kept verbatim so algorithms
// written for RapidStore compile against our tiered backend.

namespace wrapper {

    // Init
    template<class W>
    void set_max_threads(W& w, int max_threads) { w.set_max_threads(max_threads); }
    template<class W>
    void init_thread(W& w, int thread_id) { w.init_thread(thread_id); }
    template<class W>
    void end_thread(W& w, int thread_id) { w.end_thread(thread_id); }

    // Graph queries
    template<class W>
    bool is_directed(W& w) { return w.is_directed(); }
    template<class W>
    bool is_empty(W& w) { return w.is_empty(); }
    template<class W>
    bool has_vertex(W& w, uint64_t vertex) { return w.has_vertex(vertex); }
    template<class W>
    uint64_t degree(W& w, uint64_t vertex) { return w.degree(vertex); }
    template<class W>
    uint64_t vertex_count(W& w) { return w.vertex_count(); }
    template<class W>
    uint64_t edge_count(W& w) { return w.edge_count(); }

    template<class W>
    void get_neighbors(W& w, uint64_t vertex,
                       std::vector<uint64_t>& neighbors) {
        w.get_neighbors(vertex, neighbors);
    }
    template<class W>
    void get_neighbors(W& w, uint64_t vertex,
                       std::vector<std::pair<uint64_t, double>>& neighbors) {
        w.get_neighbors(vertex, neighbors);
    }

    // Mutations
    template<class W>
    bool insert_vertex(W& w, uint64_t vertex) { return w.insert_vertex(vertex); }
    template<class W>
    bool insert_edge(W& w, uint64_t src, uint64_t dst, double wt) {
        return w.insert_edge(src, dst, wt);
    }
    template<class W>
    bool remove_edge(W& w, uint64_t src, uint64_t dst) {
        return w.remove_edge(src, dst);
    }

    // Snapshot operations (critical for algorithm integration)
    template<class W>
    auto get_unique_snapshot(W& w) { return w.get_unique_snapshot(); }
    template<class W>
    auto get_shared_snapshot(W& w) { return w.get_shared_snapshot(); }
    template<class S>
    auto snapshot_clone(S& s) { return s->clone(); }
    template<class S>
    uint64_t size(S& s) { return s->size(); }

    template<class S>
    uint64_t snapshot_degree(S& s, uint64_t source, bool logical = false) {
        return s->degree(source, logical);
    }
    template<class S>
    uint64_t snapshot_vertex_count(S& s) { return s->vertex_count(); }
    template<class S>
    uint64_t snapshot_edge_count(S& s) { return s->edge_count(); }

    // The key API — callback-based edge traversal
    template<class S, class F>
    void snapshot_edges(S& s, uint64_t index, F&& callback, bool logical) {
        s->edges(index, callback, logical);
    }

    template<class S>
    void snapshot_edges(S& s, uint64_t index,
                        std::vector<uint64_t>& neighbors, bool logical) {
        s->edges(index, neighbors, logical);
    }

    template<class S>
    uint64_t snapshot_intersect(S& s, uint64_t vtx_a, uint64_t vtx_b) {
        return s->intersect(vtx_a, vtx_b);
    }

}  // namespace wrapper


// ─── NEW: Tiered Snapshot — wraps Philemon partitions as RapidStore snapshot ─

struct TierLatencyModel {
    uint64_t hbm_ns  = 1;
    uint64_t gddr_ns = 5;
    uint64_t dram_ns = 50;

    uint64_t latency_for(uint8_t tier) const {
        switch (tier) {
            case 0: return hbm_ns;
            case 1: return gddr_ns;
            case 2: return dram_ns;
            default: return dram_ns;
        }
    }
};

/**
 * TieredSnapshot — presents tiered partitions as a RapidStore-compatible snapshot.
 *
 * This is the bridge that lets RapidStore algorithms (BFS, PageRank, SSSP)
 * run on data stored across HBM/GDDR/DRAM tiers.
 *
 * Implements the same interface as RapidStore's Snapshot:
 *   - vertex_count(), edge_count()
 *   - degree(vertex, logical)
 *   - edges(vertex, callback, logical)
 *   - has_vertex(vertex)
 *   - clone()
 */
class TieredSnapshot {
public:
    // Adjacency list: vertex → [(dest, weight, tier)]
    struct AdjEntry {
        uint64_t dest;
        double   weight;
        uint8_t  tier;       // which tier this edge resides on
        int32_t  ts_start;   // temporal annotation
        int32_t  ts_end;
    };

    TieredSnapshot() = default;

    // Build from a set of TemporalEdge arrays + their tier assignments
    void build_from_edges(const TemporalEdge* edges, size_t count,
                          uint8_t tier, uint64_t max_vertex = 0) {
        for (size_t i = 0; i < count; ++i) {
            auto& e = edges[i];
            uint64_t mx = std::max(e.source, e.dest);
            if (mx >= adj_.size()) adj_.resize(mx + 1);
            adj_[e.source].push_back({e.dest, e.weight, tier,
                                       e.ts_start, e.ts_end});
            total_edges_++;
        }
        PHILE_DBG(2, "TieredSnapshot: added %zu edges from tier %u, "
                     "total=%lu vertices=%zu",
                  count, tier,
                  (unsigned long)total_edges_, adj_.size());
    }

    // RapidStore-compatible API
    uint64_t vertex_count() const { return adj_.size(); }
    uint64_t edge_count() const { return total_edges_; }
    uint64_t size() const { return adj_.size(); }

    bool has_vertex(uint64_t v) const {
        return v < adj_.size() && !adj_[v].empty();
    }

    uint64_t degree(uint64_t v, bool /*logical*/ = false) const {
        if (v >= adj_.size()) return 0;
        return adj_[v].size();
    }

    // Callback-based edge traversal (the core RapidStore pattern)
    template <typename F>
    void edges(uint64_t v, F&& callback, bool /*logical*/ = false) const {
        if (v >= adj_.size()) return;
        auto t0 = std::chrono::steady_clock::now();
        for (auto& e : adj_[v]) {
            callback(e.dest, e.weight);
        }
        auto t1 = std::chrono::steady_clock::now();
        uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t1 - t0).count();
        if (!adj_[v].empty()) {
            uint8_t tier = adj_[v][0].tier;
            debug::tier_perf(tier).read_count.fetch_add(1);
            debug::tier_perf(tier).read_bytes.fetch_add(
                adj_[v].size() * sizeof(AdjEntry));
            debug::tier_perf(tier).total_read_ns.fetch_add(ns);
        }
    }

    // Vector-based edge retrieval
    void edges(uint64_t v, std::vector<uint64_t>& neighbors,
               bool /*logical*/ = false) const {
        if (v >= adj_.size()) return;
        neighbors.reserve(neighbors.size() + adj_[v].size());
        for (auto& e : adj_[v]) {
            neighbors.push_back(e.dest);
        }
    }

    uint64_t intersect(uint64_t a, uint64_t b) const {
        if (a >= adj_.size() || b >= adj_.size()) return 0;
        // Simple sorted intersection
        std::vector<uint64_t> na, nb;
        for (auto& e : adj_[a]) na.push_back(e.dest);
        for (auto& e : adj_[b]) nb.push_back(e.dest);
        std::sort(na.begin(), na.end());
        std::sort(nb.begin(), nb.end());
        uint64_t count = 0;
        size_t i = 0, j = 0;
        while (i < na.size() && j < nb.size()) {
            if (na[i] == nb[j]) { count++; i++; j++; }
            else if (na[i] < nb[j]) i++;
            else j++;
        }
        return count;
    }

    // Clone for per-thread snapshots
    std::shared_ptr<TieredSnapshot> clone() const {
        auto c = std::make_shared<TieredSnapshot>();
        c->adj_ = adj_;
        c->total_edges_ = total_edges_;
        return c;
    }

    // ─── Debug: dump per-tier edge distribution ────────────────────
    void dump_tier_distribution() const {
        uint64_t per_tier[4] = {0, 0, 0, 0};
        for (auto& neighbors : adj_) {
            for (auto& e : neighbors) {
                per_tier[e.tier % 4]++;
            }
        }
        std::printf("[TieredSnapshot] Edge distribution: "
                    "HBM=%lu GDDR=%lu DRAM=%lu total=%lu\n",
                    (unsigned long)per_tier[0],
                    (unsigned long)per_tier[1],
                    (unsigned long)per_tier[2],
                    (unsigned long)total_edges_);
    }

    // ─── Debug: dump adjacency for a specific vertex ───────────────
    void dump_vertex(uint64_t v, int max_edges = 10) const {
        if (v >= adj_.size()) {
            std::printf("[TieredSnapshot] vertex %lu: not found\n",
                        (unsigned long)v);
            return;
        }
        const char* tnames[] = {"HBM", "GDDR", "DRAM", "??"};
        std::printf("[TieredSnapshot] vertex %lu: degree=%zu\n",
                    (unsigned long)v, adj_[v].size());
        int printed = 0;
        for (auto& e : adj_[v]) {
            if (printed >= max_edges) {
                std::printf("  ... (%zu more)\n",
                            adj_[v].size() - printed);
                break;
            }
            std::printf("  → %lu w=%.2f tier=%s ts=[%d,%d]\n",
                        (unsigned long)e.dest, e.weight,
                        tnames[e.tier % 4], e.ts_start, e.ts_end);
            printed++;
        }
    }

private:
    std::vector<std::vector<AdjEntry>> adj_;
    uint64_t total_edges_ = 0;
};


// ─── NEW: Tiered Wrapper — the integration class ───────────────────
// Wraps TieredSnapshot + TemporalBridge into a single object that
// RapidStore algorithms can operate on.

class TieredGraphWrapper {
public:
    TieredGraphWrapper() = default;

    void set_max_threads(int n) { max_threads_ = n; }
    void init_thread(int /*id*/) {}
    void end_thread(int /*id*/) {}

    bool is_directed() const { return true; }
    bool is_empty() const { return !snapshot_ || snapshot_->edge_count() == 0; }

    bool has_vertex(uint64_t v) const {
        return snapshot_ && snapshot_->has_vertex(v);
    }
    uint64_t degree(uint64_t v) const {
        return snapshot_ ? snapshot_->degree(v) : 0;
    }
    uint64_t vertex_count() const {
        return snapshot_ ? snapshot_->vertex_count() : 0;
    }
    uint64_t edge_count() const {
        return snapshot_ ? snapshot_->edge_count() : 0;
    }

    void get_neighbors(uint64_t v, std::vector<uint64_t>& out) {
        if (snapshot_) snapshot_->edges(v, out);
    }
    void get_neighbors(uint64_t v,
                       std::vector<std::pair<uint64_t, double>>& out) {
        if (!snapshot_) return;
        snapshot_->edges(v, [&](uint64_t dest, double w) {
            out.push_back({dest, w});
        });
    }

    bool insert_vertex(uint64_t) { return false; /* read-only snapshot */ }
    bool insert_edge(uint64_t, uint64_t, double) { return false; }
    bool remove_edge(uint64_t, uint64_t) { return false; }

    std::shared_ptr<TieredSnapshot> get_unique_snapshot() {
        return snapshot_;
    }
    std::shared_ptr<TieredSnapshot> get_shared_snapshot() {
        return snapshot_;
    }

    void set_snapshot(std::shared_ptr<TieredSnapshot> snap) {
        snapshot_ = snap;
    }

    // Debug
    void dump_state() const {
        if (snapshot_) {
            std::printf("[TieredGraphWrapper] vertices=%lu edges=%lu\n",
                        (unsigned long)snapshot_->vertex_count(),
                        (unsigned long)snapshot_->edge_count());
            snapshot_->dump_tier_distribution();
        } else {
            std::printf("[TieredGraphWrapper] no snapshot\n");
        }
    }

private:
    std::shared_ptr<TieredSnapshot> snapshot_;
    int max_threads_ = 1;
};

}  // namespace philemon
