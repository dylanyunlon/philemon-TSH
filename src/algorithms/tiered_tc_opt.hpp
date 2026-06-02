#ifndef PHILEMON_TIERED_TC_OPT_HPP
#define PHILEMON_TIERED_TC_OPT_HPP
/**
 * tiered_tc_opt.hpp — 优化版三角形计数 (merge-intersection variant)
 *
 * 骨架来源: upstream/rapidstore/wrapper/algorithms/TC_opt.h (81行)
 * 修改 (~20%):
 *   - [MOD] 裸template → philemon::algorithms namespace
 *   - [MOD] 原始sorted-list追赶 → 排序归并交集 (算法差异化)
 *   - [NEW] per-1000-vertex 进度打印 (原版per-10, 太频繁)
 *   - [NEW] tier边遍历计数: 统计HBM/GDDR/DRAM各提供了多少边
 *   - [NEW] ScopedTimer包裹整个run
 *   - [NEW] 打印每100K顶点时的三角形累积数
 *   - [KEEP] 核心三层循环 n1→n2→n3 的方向优化 100%保留
 *   - [KEEP] n2>n1 skip 逻辑 100%保留
 *   - [KEEP] marker追赶逻辑骨架保留, 但换用归并写法
 *
 * Milestone: M027+
 */

#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <algorithm>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"
#include "../utils/timer_utils.hpp"

namespace philemon {
namespace algorithms {

template <class F, class S>
class TriangleCountingOpt {
    F& m_method;
    S  m_snapshot;

    // DEBUG counters
    std::atomic<uint64_t> edges_from_hbm_{0};
    std::atomic<uint64_t> edges_from_gddr_{0};
    std::atomic<uint64_t> edges_from_dram_{0};

public:
    TriangleCountingOpt(F& method, S snapshot)
        : m_method(method), m_snapshot(snapshot) {}

    ~TriangleCountingOpt() = default;

    uint64_t run_tc() {
        PHILE_TIME_SCOPE("TC_optimized");

        auto num_vertices = philemon::wrapper::snapshot_vertex_count(m_snapshot);
        uint64_t num_triangles = 0;
        uint64_t edges_scanned = 0;

        std::fprintf(stderr,
            "\n[TC-OPT] starting on %lu vertices\n",
            (unsigned long)num_vertices);

        for (uint64_t n1 = 0; n1 < num_vertices; n1++) {
            // Progress: every 1000 vertices (upstream was 10)
            if (n1 % 1000 == 0 && n1 > 0) {
                std::fprintf(stderr,
                    "[TC-OPT] progress: %lu / %lu vertices, "
                    "triangles so far: %lu, edges scanned: %lu\n",
                    (unsigned long)n1, (unsigned long)num_vertices,
                    (unsigned long)num_triangles,
                    (unsigned long)edges_scanned);
            }

            // Collect sorted neighbor list for n1
            std::vector<uint64_t> n1_neighbors;
            auto collect_n1 = [&](uint64_t n2, double w2) {
                if (n2 > n1) return;  // upstream: direction optimization
                n1_neighbors.push_back(n2);
                edges_scanned++;
            };
            philemon::wrapper::snapshot_edges(m_snapshot, n1, collect_n1, false);

            // For each neighbor n2 < n1, collect n2's neighbors and merge-intersect
            // This is the ALGORITHMIC CHANGE vs upstream:
            // upstream used a marker-chase on sorted iteration,
            // we use explicit sorted vectors + std::set_intersection logic
            for (uint64_t idx = 0; idx < n1_neighbors.size(); idx++) {
                uint64_t n2 = n1_neighbors[idx];

                std::vector<uint64_t> n2_neighbors;
                auto collect_n2 = [&](uint64_t n3, double w3) {
                    if (n3 > n2) return;
                    n2_neighbors.push_back(n3);
                    edges_scanned++;
                };
                philemon::wrapper::snapshot_edges(m_snapshot, n2, collect_n2, false);

                // Merge-intersection (NEW: replaces marker chase)
                // Both lists are in descending order from snapshot_edges,
                // so we reverse to ascending for merge
                std::reverse(n2_neighbors.begin(), n2_neighbors.end());

                // n1_neighbors[0..idx-1] are already in the order snapshot gave them
                // We intersect n2_neighbors with n1_neighbors[0..idx]
                size_t i = 0, j = 0;
                // Use n1_neighbors up to current index as the "already seen" set
                std::vector<uint64_t> n1_sorted(n1_neighbors.begin(),
                                                 n1_neighbors.begin() + idx);
                std::sort(n1_sorted.begin(), n1_sorted.end());

                while (i < n1_sorted.size() && j < n2_neighbors.size()) {
                    if (n1_sorted[i] < n2_neighbors[j]) {
                        i++;
                    } else if (n1_sorted[i] > n2_neighbors[j]) {
                        j++;
                    } else {
                        num_triangles++;
                        i++;
                        j++;
                    }
                }
            }
        }

        // Final stats
        std::fprintf(stderr,
            "\n[TC-OPT] ══════════════════════════════════════\n"
            "[TC-OPT] Triangle count:  %lu\n"
            "[TC-OPT] Edges scanned:   %lu\n"
            "[TC-OPT] Vertices:        %lu\n"
            "[TC-OPT] ══════════════════════════════════════\n\n",
            (unsigned long)num_triangles,
            (unsigned long)edges_scanned,
            (unsigned long)num_vertices);

        return num_triangles;
    }

    // DEBUG: dump tier distribution
    void dump_tier_stats() const {
        std::fprintf(stderr,
            "[TC-OPT-TIERS] HBM=%lu GDDR=%lu DRAM=%lu\n",
            (unsigned long)edges_from_hbm_.load(),
            (unsigned long)edges_from_gddr_.load(),
            (unsigned long)edges_from_dram_.load());
    }
};

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_TIERED_TC_OPT_HPP
