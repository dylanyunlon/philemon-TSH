#ifndef PHILEMON_CROSS_TIER_WCC_HPP
#define PHILEMON_CROSS_TIER_WCC_HPP
/**
 * cross_tier_wcc.hpp — Cross-Tier Weakly Connected Components
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/algorithms/WCC.h     (149行)
 *   upstream/rapidstore/algorithms/WCC.cpp            (137行)
 *   upstream/rapidstore/wrapper/driver.h:784-834      (51行, UnionFind+wcc())
 *   upstream/rapidstore/wrapper/wrapper.h             (249行, snapshot API)
 *
 * 修改 (~20%):
 *   - [NEW] 增加 WCCRoundStats: 每轮 label propagation 的 tier 统计
 *   - [NEW] 增加 UnionFindTiered: 在 driver.h UnionFind 基础上增加 rank+路径压缩
 *   - [NEW] 增加 component_tier_map: 追踪每个component跨几个tier
 *   - [NEW] 增加 dump_all_state(): 打印完整的 component/label 状态
 *   - [NEW] 增加 convergence_log: 每轮 change_count+component_count
 *   - [MOD] 原 omp parallel → std::thread (与项目统一)
 *   - [MOD] 原 log_info → PHILE_DBG
 *   - [KEEP] label propagation 核心 100% 保留
 *   - [KEEP] path compression (comp[i]=comp[comp[i]]) 100% 保留
 *   - [KEEP] high→low union logic 100% 保留
 *
 * Milestone: M022 — Cross-tier WCC with component tier tracking
 * ====================================================================
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <functional>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"
#include "../cost_model/tier_cost_model.hpp"

namespace philemon {
namespace algorithms {

// ═══════════════════════════════════════════════════════════════════════
// Per-round statistics — for debug + paper data
// (NEW: not in upstream, added for convergence tracking)
// ═══════════════════════════════════════════════════════════════════════
struct WCCRoundStats {
    int      round;
    uint64_t changes;            // # vertices that changed component
    uint64_t num_components;     // distinct components after this round
    uint64_t largest_component;  // size of largest component
    double   time_ms;
    // Tier distribution of changed vertices
    uint64_t changes_hbm;
    uint64_t changes_gddr;
    uint64_t changes_dram;
    // Cross-tier stats
    uint64_t cross_tier_edges;   // edges spanning different tiers

    void dump() const {
        std::printf("  [WCC round %d] changes=%lu components=%lu "
                    "largest=%lu time=%.3fms\n",
                    round, (unsigned long)changes,
                    (unsigned long)num_components,
                    (unsigned long)largest_component, time_ms);
        std::printf("           tier_changes: HBM=%lu GDDR=%lu DRAM=%lu "
                    "cross_tier_edges=%lu\n",
                    (unsigned long)changes_hbm,
                    (unsigned long)changes_gddr,
                    (unsigned long)changes_dram,
                    (unsigned long)cross_tier_edges);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// UnionFindTiered — Enhanced union-find from upstream driver.h:784-808
//
// Original from driver.h (preserved):
//   - root[] array, find() with path compression, unite()
// NEW:
//   - rank-based union (union by rank for balanced trees)
//   - tier tracking per root
//   - component_count() utility
//   - dump_state() for breakpoint debugging
// ═══════════════════════════════════════════════════════════════════════
class UnionFindTiered {
public:
    std::vector<uint64_t> root;   // From upstream driver.h
    std::vector<uint64_t> rank_;  // NEW: union by rank
    std::vector<uint8_t>  tier;   // NEW: simulated tier of each vertex

    // Constructor from upstream driver.h (preserved + rank init)
    UnionFindTiered(uint64_t size) : root(size), rank_(size, 0), tier(size) {
        for (uint64_t i = 0; i < size; i++) {
            root[i] = i;
            // NEW: simulate tier assignment
            tier[i] = (i < size / 3) ? 0 :
                      (i < size * 2 / 3) ? 1 : 2;
        }
    }

    // From upstream driver.h::find() (path compression preserved)
    uint64_t find(uint64_t x) {
        if (x == root[x]) {
            return x;
        }
        return root[x] = find(root[x]);  // path compression
    }

    // From upstream driver.h::unite() + NEW: union by rank
    void unite(uint64_t x, uint64_t y) {
        uint64_t rootX = find(x);
        uint64_t rootY = find(y);
        if (rootX != rootY) {
            // NEW: union by rank (upstream just did root[rootY] = rootX)
            if (rank_[rootX] < rank_[rootY]) {
                root[rootX] = rootY;
            } else if (rank_[rootX] > rank_[rootY]) {
                root[rootY] = rootX;
            } else {
                root[rootY] = rootX;
                rank_[rootX]++;
            }
        }
    }

    // NEW: count distinct components
    uint64_t component_count() {
        std::unordered_set<uint64_t> roots;
        for (uint64_t i = 0; i < root.size(); i++) {
            roots.insert(find(i));
        }
        return roots.size();
    }

    // NEW: get component sizes
    std::unordered_map<uint64_t, uint64_t> component_sizes() {
        std::unordered_map<uint64_t, uint64_t> sizes;
        for (uint64_t i = 0; i < root.size(); i++) {
            sizes[find(i)]++;
        }
        return sizes;
    }

    // NEW: count cross-tier edges (edges where src and dst are in different tiers)
    uint64_t count_cross_tier_roots() {
        uint64_t cross = 0;
        for (uint64_t i = 0; i < root.size(); i++) {
            uint64_t r = find(i);
            if (tier[i] != tier[r]) cross++;
        }
        return cross;
    }

    // NEW: dump state for breakpoint debugging
    void dump_state(const std::string& phase) const {
        if (debug::get_debug_level() < 3) return;

        std::printf("\n╔══════════════════════════════════════════════════╗\n");
        std::printf("║ UF STATE DUMP: %s                          ║\n",
                    phase.c_str());
        std::printf("╠══════════════════════════════════════════════════╣\n");

        // Root array (first 30)
        std::printf("║ root[0..29]: ");
        for (uint64_t i = 0; i < std::min(root.size(), (size_t)30); i++) {
            std::printf("%lu ", (unsigned long)root[i]);
        }
        std::printf("\n");

        // Rank stats
        uint64_t max_rank = 0;
        for (auto r : rank_) max_rank = std::max(max_rank, r);
        std::printf("║ max_rank=%lu\n", (unsigned long)max_rank);

        // Tier distribution of roots
        uint64_t tier_roots[3] = {0, 0, 0};
        for (uint64_t i = 0; i < root.size(); i++) {
            if (root[i] == i) tier_roots[tier[i]]++;
        }
        std::printf("║ root_tier_dist: HBM=%lu GDDR=%lu DRAM=%lu\n",
                    (unsigned long)tier_roots[0],
                    (unsigned long)tier_roots[1],
                    (unsigned long)tier_roots[2]);

        std::printf("╚══════════════════════════════════════════════════╝\n\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CrossTierWCC — WCC with tier-aware component tracking
//
// Class structure from upstream WCC.h (wccExperiments), modified:
//   - Added m_cost_model, m_round_log
//   - Added UnionFind-based verification path
//   - Added comprehensive state dumping
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
class CrossTierWCC {
    // ─── From upstream WCC.h (100% preserved) ────────────────────
    const int m_num_threads;
    F& m_method;
    S m_snapshot;

    // ─── NEW: Tier-aware extensions ──────────────────────────────
    cost_model::TierCostModel m_cost_model;
    std::vector<WCCRoundStats> m_round_log;

public:
    // ─── Constructor: upstream + new params ───────────────────────
    CrossTierWCC(int num_threads, F& method, S snapshot,
                 cost_model::TierCostModel cost_model = {})
        : m_num_threads(num_threads), m_method(method),
          m_snapshot(snapshot), m_cost_model(cost_model) {}

    ~CrossTierWCC() {}

    // ─── Main entry (from upstream WCC.h::run_wcc, modified) ─────
    void run_wcc(std::vector<std::pair<uint64_t, int64_t>>& results);

    // ─── NEW: Access convergence log ─────────────────────────────
    const std::vector<WCCRoundStats>& round_log() const { return m_round_log; }

    // ─── NEW: Dump convergence ───────────────────────────────────
    void dump_convergence() const;

    // ─── NEW: Dump complete WCC state for breakpoint debugging ───
    void dump_all_state(const std::string& phase,
                        const uint64_t* comp, uint64_t N,
                        int current_round) const;

private:
    // ─── Core: from upstream WCC.h::wcc() ────────────────────────
    std::unique_ptr<uint64_t[]> wcc();

    // ─── From upstream driver.h::wcc() — UnionFind version ───────
    void wcc_unionfind(std::vector<int>& result);

    // ─── NEW: Collect per-round stats ────────────────────────────
    WCCRoundStats collect_round_stats(int round, uint64_t changes,
                                      const uint64_t* comp, uint64_t N,
                                      double time_ms);
};

// ═══════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════

// ─── dump_all_state: NEW, breakpoint-style state dump ────────────────
template <class F, class S>
void CrossTierWCC<F,S>::dump_all_state(
    const std::string& phase,
    const uint64_t* comp, uint64_t N,
    int current_round) const
{
    if (debug::get_debug_level() < 3) return;

    std::printf("\n╔══════════════════════════════════════════════════╗\n");
    std::printf("║ WCC STATE DUMP: %s (round=%d)              ║\n",
                phase.c_str(), current_round);
    std::printf("╠══════════════════════════════════════════════════╣\n");

    // Component array (first 30)
    std::printf("║ comp[0..29]: ");
    for (uint64_t i = 0; i < std::min(N, (uint64_t)30); i++) {
        std::printf("%lu ", (unsigned long)comp[i]);
    }
    std::printf("\n");

    // Count distinct components
    std::unordered_map<uint64_t, uint64_t> comp_sizes;
    for (uint64_t i = 0; i < N; i++) {
        comp_sizes[comp[i]]++;
    }
    std::printf("║ distinct_components=%zu\n", comp_sizes.size());

    // Top-5 largest components
    std::vector<std::pair<uint64_t, uint64_t>> sorted_comps(
        comp_sizes.begin(), comp_sizes.end());
    std::partial_sort(sorted_comps.begin(),
                      sorted_comps.begin() +
                          std::min(sorted_comps.size(), (size_t)5),
                      sorted_comps.end(),
                      [](auto& a, auto& b) { return a.second > b.second; });
    std::printf("║ top-5 components:\n");
    for (size_t i = 0; i < std::min(sorted_comps.size(), (size_t)5); i++) {
        std::printf("║   comp[%lu] size=%lu\n",
                    (unsigned long)sorted_comps[i].first,
                    (unsigned long)sorted_comps[i].second);
    }

    // Tier perf
    std::printf("║ tier_perf:\n");
    for (int t = 0; t < 3; t++) {
        auto& perf = debug::tier_perf(t);
        std::printf("║   tier[%d] reads=%lu writes=%lu bytes=%lu\n",
                    t, (unsigned long)perf.read_count.load(),
                    (unsigned long)perf.write_count.load(),
                    (unsigned long)perf.bytes_transferred.load());
    }

    std::printf("╚══════════════════════════════════════════════════╝\n\n");
}

// ═══════════════════════════════════════════════════════════════════════
// wcc() — Label propagation core
//
// From upstream WCC.h::wcc() — 100% preserved core algorithm,
// with NEW: per-round stats, cross-tier tracking, state dump
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
std::unique_ptr<uint64_t[]> CrossTierWCC<F,S>::wcc() {
    debug::ScopedTimer timer("CrossTierWCC::compute");
    m_round_log.clear();

    // ─── From upstream WCC.h: init component array ───────────────
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    std::unique_ptr<uint64_t[]> ptr_components { new uint64_t[N] };
    uint64_t* comp = ptr_components.get();

    // From upstream: initialize each vertex as its own component
    for (uint64_t i = 0; i < N; i++) {
        comp[i] = i;
    }

    PHILE_DBG(1, "[CrossTierWCC] N=%lu threads=%d",
              (unsigned long)N, m_num_threads);

    // NEW: state dump after init
    dump_all_state("after_init", comp, N, -1);

    // ─── From upstream WCC.h: label propagation loop ─────────────
    bool change = true;
    int round = 0;
    std::vector<std::thread> threads;
    uint64_t chunk_size = (N + m_num_threads - 1) / m_num_threads;

    while (change) {
        auto round_start = std::chrono::high_resolution_clock::now();
        change = false;

        // NEW: per-tier change counter
        std::atomic<uint64_t> tier_changes[3] = {{0}, {0}, {0}};
        std::atomic<uint64_t> total_changes{0};
        std::atomic<uint64_t> cross_tier_count{0};

        // From upstream WCC.h: parallel label propagation
        // (structure 100% preserved, +tier tracking)
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &change, &comp, N, chunk_size,
                                  &tier_changes, &total_changes,
                                  &cross_tier_count](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap = wrapper::snapshot_clone(m_snapshot);
                uint64_t start = chunk_size * tid;
                uint64_t end = std::min(start + chunk_size, N);

                for (uint64_t u = start; u < end; u++) {
                    // From upstream WCC.h: edge callback for label propagation
                    // (100% preserved logic)
                    wrapper::snapshot_edges(snap, u,
                        [comp, &change, u, N, &total_changes,
                         &tier_changes, &cross_tier_count]
                        (uint64_t v, double w) {
                            uint64_t comp_u = comp[u];
                            uint64_t comp_v = comp[v];
                            if (comp_u == comp_v) {
                                return;  // from upstream: skip same-component
                            }

                            // From upstream: high→low union
                            uint64_t high_comp = std::max(comp_u, comp_v);
                            uint64_t low_comp = std::min(comp_u, comp_v);

                            // From upstream: bounds check
                            if (high_comp >= N || low_comp >= N) return;

                            if (high_comp == comp[high_comp]) {
                                change = true;
                                comp[high_comp] = low_comp;
                                total_changes.fetch_add(1,
                                    std::memory_order_relaxed);

                                // NEW: track tier of changed vertex
                                uint8_t vtx_tier = (high_comp < N / 3) ? 0 :
                                    (high_comp < N * 2 / 3) ? 1 : 2;
                                tier_changes[vtx_tier].fetch_add(1,
                                    std::memory_order_relaxed);

                                // NEW: track cross-tier edges
                                uint8_t u_tier = (u < N / 3) ? 0 :
                                    (u < N * 2 / 3) ? 1 : 2;
                                uint8_t v_tier = (v < N / 3) ? 0 :
                                    (v < N * 2 / 3) ? 1 : 2;
                                if (u_tier != v_tier) {
                                    cross_tier_count.fetch_add(1,
                                        std::memory_order_relaxed);
                                }
                            }
                        }, false);
                }

                wrapper::end_thread(m_method, tid);
            }, i);
        }

        for (auto& t : threads) t.join();
        threads.clear();

        // ─── From upstream WCC.h: pointer jumping / path compression ─
        // (100% preserved)
        for (uint64_t i = 0; i < N; i++) {
            while (comp[i] != comp[comp[i]]) {
                comp[i] = comp[comp[i]];
            }
        }

        auto round_end = std::chrono::high_resolution_clock::now();
        double round_ms = std::chrono::duration<double, std::milli>(
            round_end - round_start).count();

        // ─── NEW: Collect and log round stats ────────────────────
        auto stats = collect_round_stats(round, total_changes.load(),
                                          comp, N, round_ms);
        stats.changes_hbm  = tier_changes[0].load();
        stats.changes_gddr = tier_changes[1].load();
        stats.changes_dram = tier_changes[2].load();
        stats.cross_tier_edges = cross_tier_count.load();
        m_round_log.push_back(stats);

        if (debug::get_debug_level() >= 2) {
            stats.dump();
        }

        // NEW: state dump at interesting rounds
        if (debug::get_debug_level() >= 3 &&
            (round < 3 || !change)) {
            dump_all_state("after_round", comp, N, round);
        }

        round++;
        PHILE_DBG(2, "[WCC round %d] changes=%lu", round,
                  (unsigned long)total_changes.load());
    }

    PHILE_DBG(1, "[CrossTierWCC] converged in %d rounds", round);
    return ptr_components;
}

// ─── wcc_unionfind: from upstream driver.h::wcc() + UnionFind ────────
// Sequential verification baseline (80% preserved from driver.h)
template <class F, class S>
void CrossTierWCC<F,S>::wcc_unionfind(std::vector<int>& result) {
    debug::ScopedTimer timer("WCC::unionfind_baseline");

    // From upstream driver.h:811-834
    uint64_t size = wrapper::snapshot_vertex_count(m_snapshot);
    UnionFindTiered uf(size);

    PHILE_DBG(2, "[WCC::UF] N=%lu", (unsigned long)size);

    for (uint64_t source = 0; source < size; source++) {
        // From upstream: edge callback for union-find
        std::function<uint64_t(uint64_t, double)> cb =
            [source, &uf](uint64_t destination, double weight) -> uint64_t {
                uf.unite(source, destination);
                return 0;
            };
        wrapper::snapshot_edges(m_snapshot, source, cb, false);
        result[source] = -1;
    }

    // From upstream: assign component IDs
    uint64_t component_cnt = 0;
    for (uint64_t i = 0; i < size; i++) {
        uint64_t r = uf.find(i);
        if (result[r] == -1) {
            result[r] = component_cnt++;
        }
        result[i] = result[r];
    }

    PHILE_DBG(1, "[WCC::UF] found %lu components",
              (unsigned long)component_cnt);

    // NEW: dump UF state
    uf.dump_state("after_wcc_complete");
}

// ─── collect_round_stats: NEW ────────────────────────────────────────
template <class F, class S>
WCCRoundStats CrossTierWCC<F,S>::collect_round_stats(
    int round, uint64_t changes,
    const uint64_t* comp, uint64_t N,
    double time_ms)
{
    WCCRoundStats stats;
    stats.round = round;
    stats.changes = changes;
    stats.time_ms = time_ms;
    stats.cross_tier_edges = 0;

    // Count distinct components and find largest
    std::unordered_map<uint64_t, uint64_t> sizes;
    for (uint64_t i = 0; i < N; i++) {
        sizes[comp[i]]++;
    }
    stats.num_components = sizes.size();
    stats.largest_component = 0;
    for (auto& [id, sz] : sizes) {
        stats.largest_component = std::max(stats.largest_component, sz);
    }

    return stats;
}

// ═══════════════════════════════════════════════════════════════════════
// run_wcc: main entry point
// From upstream WCC.h::run_wcc() (structure preserved)
// + NEW: convergence dump, tier perf, UF verification
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void CrossTierWCC<F,S>::run_wcc(
    std::vector<std::pair<uint64_t, int64_t>>& results)
{
    debug::ScopedTimer timer("CrossTierWCC::run_wcc");

    // Reset tier perf counters
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto components = wcc();

    // ─── From upstream WCC.h: result collection (preserved) ──────
    uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    results.resize(N);

    for (uint64_t u = 0; u < N; u++) {
        results[u] = {u, static_cast<int64_t>(components[u])};
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    // ─── NEW: Component statistics ───────────────────────────────
    std::unordered_map<uint64_t, uint64_t> comp_sizes;
    for (uint64_t u = 0; u < N; u++) {
        comp_sizes[components[u]]++;
    }

    PHILE_DBG(1, "[CrossTierWCC] total: %ld ms, N=%lu, "
              "components=%zu, rounds=%zu",
              (long)ms.count(), (unsigned long)N,
              comp_sizes.size(), m_round_log.size());

    // ─── NEW: Top-10 components ──────────────────────────────────
    if (debug::get_debug_level() >= 1) {
        std::vector<std::pair<uint64_t, uint64_t>> sorted(
            comp_sizes.begin(), comp_sizes.end());
        std::partial_sort(sorted.begin(),
                          sorted.begin() + std::min(sorted.size(), (size_t)10),
                          sorted.end(),
                          [](auto& a, auto& b) {
                              return a.second > b.second;
                          });
        std::printf("  [WCC top-10 components]:\n");
        for (size_t i = 0; i < std::min(sorted.size(), (size_t)10); i++) {
            std::printf("    #%zu: comp=%lu size=%lu\n",
                        i + 1,
                        (unsigned long)sorted[i].first,
                        (unsigned long)sorted[i].second);
        }
    }

    // ─── NEW: Component size distribution ────────────────────────
    if (debug::get_debug_level() >= 2) {
        uint64_t singleton = 0, small = 0, medium = 0, large = 0;
        for (auto& [id, sz] : comp_sizes) {
            if (sz == 1) singleton++;
            else if (sz < 10) small++;
            else if (sz < 1000) medium++;
            else large++;
        }
        std::printf("  [WCC size distribution] singleton=%lu small(2-9)=%lu "
                    "medium(10-999)=%lu large(1000+)=%lu\n",
                    (unsigned long)singleton, (unsigned long)small,
                    (unsigned long)medium, (unsigned long)large);
    }

    // Print convergence + tier perf
    dump_convergence();
    debug::print_all_tier_perf();

    // ─── NEW: Optional UF verification ───────────────────────────
    if (debug::get_debug_level() >= 3 && N <= 100000) {
        PHILE_DBG(3, "[WCC] Running UnionFind verification...");
        std::vector<int> uf_result(N, -1);
        wcc_unionfind(uf_result);
        // Compare component counts (don't compare IDs as they can differ)
        std::unordered_set<int> uf_comps(uf_result.begin(), uf_result.end());
        PHILE_DBG(1, "[WCC verify] label_prop=%zu vs union_find=%zu components",
                  comp_sizes.size(), uf_comps.size());
    }
}

// ─── dump_convergence: NEW ───────────────────────────────────────────
template <class F, class S>
void CrossTierWCC<F,S>::dump_convergence() const {
    if (m_round_log.empty()) return;

    std::printf("──── WCC Convergence Log (%zu rounds) ────\n",
                m_round_log.size());
    std::printf("  %-6s %-10s %-12s %-12s %-9s %-7s %-7s %-7s %-10s\n",
                "Round", "Changes", "Components", "Largest",
                "time_ms", "HBM", "GDDR", "DRAM", "XTier");
    for (const auto& s : m_round_log) {
        std::printf("  %-6d %-10lu %-12lu %-12lu %-9.3f %-7lu %-7lu %-7lu %-10lu\n",
                    s.round,
                    (unsigned long)s.changes,
                    (unsigned long)s.num_components,
                    (unsigned long)s.largest_component,
                    s.time_ms,
                    (unsigned long)s.changes_hbm,
                    (unsigned long)s.changes_gddr,
                    (unsigned long)s.changes_dram,
                    (unsigned long)s.cross_tier_edges);
    }
    std::printf("──── End WCC Convergence ────\n");
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_CROSS_TIER_WCC_HPP
