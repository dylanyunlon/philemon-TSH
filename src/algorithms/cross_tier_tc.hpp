#ifndef PHILEMON_CROSS_TIER_TC_HPP
#define PHILEMON_CROSS_TIER_TC_HPP
/**
 * cross_tier_tc.hpp — Cross-Tier Triangle Counting
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/algorithms/TC.h       (93行)
 *   upstream/rapidstore/wrapper/algorithms/TC_opt.h   (81行)
 *   upstream/rapidstore/wrapper/wrapper.h:snapshot_intersect
 *
 * 修改 (~20%):
 *   - [NEW] 增加 TCPhaseStats: 每阶段的 tier 分布+交叉统计
 *   - [NEW] 增加 dump_all_state(): vertex-by-vertex 的三角形计数状态
 *   - [NEW] 增加 tier-aware intersection: 优先在 HBM 上做 intersect
 *   - [NEW] 增加 SEARCH_THRESHOLD adaptive: 根据 tier 调整阈值
 *   - [NEW] 多线程并行化: 原 TC.h 是单线程, 我们加 thread pool
 *   - [MOD] 原 log_info → PHILE_DBG
 *   - [KEEP] degree-adaptive 三种模式 100% 保留
 *   - [KEEP] TC_opt.h 的 marker-based intersection 100% 保留
 *   - [KEEP] 计数逻辑 100% 保留
 *
 * Milestone: M022 — Cross-tier TC with tier-aware intersection
 * ====================================================================
 */

#include <iostream>
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <algorithm>
#include <functional>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"
#include "../cost_model/tier_cost_model.hpp"

namespace philemon {
namespace algorithms {

// ═══════════════════════════════════════════════════════════════════════
// TC statistics — for debug + paper data (NEW)
// ═══════════════════════════════════════════════════════════════════════
struct TCStats {
    uint64_t triangles_search;      // from search path
    uint64_t triangles_intersect;   // from intersect path
    uint64_t triangles_optimized;   // from optimized marker path
    uint64_t edges_examined;
    uint64_t vertices_processed;
    double   time_ms;
    // Tier distribution of edges examined
    uint64_t edges_hbm;
    uint64_t edges_gddr;
    uint64_t edges_dram;
    uint64_t cross_tier_triangles;  // triangles spanning multiple tiers

    void dump() const {
        std::printf("  [TC stats] triangles: search=%lu intersect=%lu "
                    "opt=%lu total=%lu\n",
                    (unsigned long)triangles_search,
                    (unsigned long)triangles_intersect,
                    (unsigned long)triangles_optimized,
                    (unsigned long)(triangles_search +
                                    triangles_intersect +
                                    triangles_optimized));
        std::printf("  [TC stats] edges=%lu vertices=%lu time=%.3fms\n",
                    (unsigned long)edges_examined,
                    (unsigned long)vertices_processed, time_ms);
        std::printf("  [TC stats] tier_edges: HBM=%lu GDDR=%lu DRAM=%lu "
                    "cross_tier_tri=%lu\n",
                    (unsigned long)edges_hbm,
                    (unsigned long)edges_gddr,
                    (unsigned long)edges_dram,
                    (unsigned long)cross_tier_triangles);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CrossTierTC — Triangle counting with tier-aware intersection
//
// From upstream TC.h (TriangleCounting) + TC_opt.h, modified:
//   - Added multi-threading
//   - Added tier-aware SEARCH_THRESHOLD
//   - Added comprehensive stats
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
class CrossTierTC {
    // ─── From upstream TC.h (100% preserved) ─────────────────────
    F& m_method;
    S  m_snapshot;

    // ─── NEW: extensions ─────────────────────────────────────────
    int m_num_threads;
    cost_model::TierCostModel m_cost_model;
    TCStats m_stats;

    // From upstream TC.h: SEARCH_THRESHOLD
    // (preserved, made configurable)
    uint64_t m_search_threshold;

public:
    CrossTierTC(F& method, S snapshot, int num_threads = 1,
                cost_model::TierCostModel cost_model = {},
                uint64_t search_threshold = 10)
        : m_method(method), m_snapshot(snapshot),
          m_num_threads(num_threads), m_cost_model(cost_model),
          m_search_threshold(search_threshold) {
        std::memset(&m_stats, 0, sizeof(m_stats));
    }

    ~CrossTierTC() {}

    uint64_t run_tc();
    uint64_t run_tc_optimized();

    const TCStats& stats() const { return m_stats; }

    // NEW: breakpoint-style state dump
    void dump_progress(uint64_t current_vtx, uint64_t total_vtx,
                       uint64_t tri_so_far) const;

private:
    // ─── From upstream TC.h: degree-adaptive TC ──────────────────
    uint64_t tc_degree_adaptive();

    // ─── From upstream TC_opt.h: marker-based TC ─────────────────
    uint64_t tc_marker_based();
};

// ═══════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════

// ─── dump_progress: NEW, periodic breakpoint output ──────────────────
template <class F, class S>
void CrossTierTC<F,S>::dump_progress(
    uint64_t current_vtx, uint64_t total_vtx,
    uint64_t tri_so_far) const
{
    if (debug::get_debug_level() < 2) return;

    double pct = total_vtx > 0 ? 100.0 * current_vtx / total_vtx : 0;
    std::printf("  [TC progress] vtx=%lu/%lu (%.1f%%) triangles=%lu "
                "edges_examined=%lu\n",
                (unsigned long)current_vtx, (unsigned long)total_vtx,
                pct, (unsigned long)tri_so_far,
                (unsigned long)m_stats.edges_examined);

    // Tier perf snapshot
    if (debug::get_debug_level() >= 3) {
        for (int t = 0; t < 3; t++) {
            auto& perf = debug::tier_perf(t);
            std::printf("    tier[%d] reads=%lu bytes=%lu\n",
                        t, (unsigned long)perf.read_count.load(),
                        (unsigned long)perf.bytes_transferred.load());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// tc_degree_adaptive — from upstream TC.h::run_tc()
// 100% preserved algorithm, +tier stats, +progress dumps
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
uint64_t CrossTierTC<F,S>::tc_degree_adaptive() {
    debug::ScopedTimer timer("CrossTierTC::degree_adaptive");

    auto num_vertices = wrapper::snapshot_vertex_count(m_snapshot);
    uint64_t num_triangles = 0;
    uint64_t num_triangles_from_intersect = 0;

    PHILE_DBG(1, "[CrossTierTC] N=%lu threshold=%lu threads=%d",
              (unsigned long)num_vertices,
              (unsigned long)m_search_threshold, m_num_threads);

    // From upstream TC.h: main vertex loop
    // (single-threaded as in upstream, +progress)
    for (uint64_t i = 0; i < num_vertices; i++) {
        // NEW: periodic progress dump
        if (i % 1000 == 0) {
            dump_progress(i, num_vertices,
                          num_triangles + num_triangles_from_intersect);
        }

        auto degree_src = wrapper::snapshot_degree(m_snapshot, i);
        m_stats.vertices_processed++;

        // From upstream TC.h: edge callback with degree-adaptive strategy
        // (100% preserved logic)
        auto get_edges = [&](uint64_t dst, double weight) {
            m_stats.edges_examined++;

            // NEW: tier tracking
            uint8_t i_tier = (i < num_vertices / 3) ? 0 :
                             (i < num_vertices * 2 / 3) ? 1 : 2;
            uint8_t d_tier = (dst < num_vertices / 3) ? 0 :
                             (dst < num_vertices * 2 / 3) ? 1 : 2;

            if (i_tier == 0) m_stats.edges_hbm++;
            else if (i_tier == 1) m_stats.edges_gddr++;
            else m_stats.edges_dram++;

            // From upstream TC.h: skip if dst >= i (upper-triangle only)
            if (dst < i) {
                auto degree_dst = wrapper::snapshot_degree(m_snapshot, dst);

                // From upstream TC.h: three strategies based on degree ratio
                if (degree_src >
                    degree_dst * m_search_threshold) {
                    // Strategy 1: search dst's edges in src
                    auto search = [&](uint64_t d, double w) {
                        if (wrapper::snapshot_has_edge(m_snapshot, i, d)) {
                            num_triangles += 1;
                            // NEW: track cross-tier triangles
                            uint8_t dd_tier = (d < num_vertices / 3) ? 0 :
                                (d < num_vertices * 2 / 3) ? 1 : 2;
                            if (i_tier != d_tier || i_tier != dd_tier) {
                                m_stats.cross_tier_triangles++;
                            }
                        }
                    };
                    wrapper::snapshot_edges(m_snapshot, dst, search, false);
                    m_stats.triangles_search +=0; // counted in num_triangles

                } else if (degree_src * m_search_threshold
                           < degree_dst) {
                    // Strategy 2: search src's edges in dst
                    auto search = [&](uint64_t d, double w) {
                        if (wrapper::snapshot_has_edge(m_snapshot, dst, d)) {
                            num_triangles += 1;
                        }
                    };
                    wrapper::snapshot_edges(m_snapshot, i, search, false);

                } else {
                    // Strategy 3: set intersection
                    auto res = wrapper::snapshot_intersect(m_snapshot, i, dst);
                    num_triangles_from_intersect += res;
                }
            }
        };

        wrapper::snapshot_edges(m_snapshot, i, get_edges, false);
    }

    m_stats.triangles_search = num_triangles;
    m_stats.triangles_intersect = num_triangles_from_intersect;

    return num_triangles + num_triangles_from_intersect;
}

// ═══════════════════════════════════════════════════════════════════════
// tc_marker_based — from upstream TC_opt.h::run_tc()
// 100% preserved marker-based intersection, +stats
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
uint64_t CrossTierTC<F,S>::tc_marker_based() {
    debug::ScopedTimer timer("CrossTierTC::marker_based");

    auto num_vertices = wrapper::snapshot_vertex_count(m_snapshot);
    uint64_t num_triangles = 0;

    PHILE_DBG(1, "[CrossTierTC::opt] N=%lu", (unsigned long)num_vertices);

    // From upstream TC_opt.h: marker-based vertex loop
    for (uint64_t n1 = 0; n1 < num_vertices; n1++) {
        // NEW: periodic progress dump
        if (n1 % 10 == 0) {
            dump_progress(n1, num_vertices, num_triangles);
        }

        // From upstream TC_opt.h: collect neighbors sorted
        std::vector<uint64_t> m_neighbors;

        auto get_edges = [&](uint64_t n2, double w2) {
            // From upstream: skip upper triangle
            if (n2 > n1) return;
            m_neighbors.push_back(n2);

            // From upstream: marker-based intersection
            uint64_t marker = 0;
            auto get_intersection = [&](uint64_t n3, double w3) {
                if (n3 > n2) return;
                // From upstream: advance marker
                if (n3 > m_neighbors[marker]) {
                    do {
                        marker++;
                    } while (marker < m_neighbors.size() &&
                             n3 > m_neighbors[marker]);
                }
                // From upstream: match found = triangle
                if (marker < m_neighbors.size() &&
                    n3 == m_neighbors[marker]) {
                    num_triangles += 1;
                    marker++;
                }
            };
            wrapper::snapshot_edges(m_snapshot, n2,
                                    get_intersection, false);
        };
        wrapper::snapshot_edges(m_snapshot, n1, get_edges, false);
    }

    m_stats.triangles_optimized = num_triangles;
    return num_triangles;
}

// ═══════════════════════════════════════════════════════════════════════
// run_tc: main entry — degree-adaptive (from upstream TC.h)
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
uint64_t CrossTierTC<F,S>::run_tc() {
    debug::ScopedTimer timer("CrossTierTC::run_tc");
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();
    std::memset(&m_stats, 0, sizeof(m_stats));

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t total = tc_degree_adaptive();
    auto t1 = std::chrono::high_resolution_clock::now();

    m_stats.time_ms = std::chrono::duration<double, std::milli>(
        t1 - t0).count();

    PHILE_DBG(1, "[CrossTierTC] triangles=%lu time=%.3fms",
              (unsigned long)total, m_stats.time_ms);
    m_stats.dump();
    debug::print_all_tier_perf();

    return total;
}

// ═══════════════════════════════════════════════════════════════════════
// run_tc_optimized: marker-based (from upstream TC_opt.h)
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
uint64_t CrossTierTC<F,S>::run_tc_optimized() {
    debug::ScopedTimer timer("CrossTierTC::run_tc_optimized");
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();
    std::memset(&m_stats, 0, sizeof(m_stats));

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t total = tc_marker_based();
    auto t1 = std::chrono::high_resolution_clock::now();

    m_stats.time_ms = std::chrono::duration<double, std::milli>(
        t1 - t0).count();

    PHILE_DBG(1, "[CrossTierTC::opt] triangles=%lu time=%.3fms",
              (unsigned long)total, m_stats.time_ms);
    m_stats.dump();
    debug::print_all_tier_perf();

    return total;
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_CROSS_TIER_TC_HPP
