#ifndef PHILEMON_CROSS_TIER_TC_WRAPPER_HPP
#define PHILEMON_CROSS_TIER_TC_WRAPPER_HPP
/**
 * cross_tier_tc_wrapper.hpp — Triangle Counting (含优化版)
 *
 * 源: upstream TC.h (93行) + TC_opt.h (81行)
 *
 * 算法修改 (~20%):
 *   1. 自适应 search threshold: upstream 硬编码 SEARCH_THRESHOLD=10
 *      → 基于前 min(1000,N) 个顶点的 median degree 动态计算:
 *        threshold = max(2, median_degree / 4)
 *      理由: 固定阈值 10 在 power-law 图上太保守，在 uniform 图上太激进
 *
 *   2. TC_opt 的 marker scan 改二分查找: upstream 用线性扫描 marker++
 *      → 当 m_neighbors 长度 > 64 时，改用 lower_bound 二分查找
 *      理由: 高度数顶点邻居列表很长，线性扫描 O(deg) → 二分 O(log deg)
 *
 *   3. 合并 TC + TC_opt 为统一接口: upstream 是两个独立类
 *      → 合并为一个类，通过构造参数选择策略
 *      理由: 代码复用，统一入口
 *
 *   4. 对称性剪枝: upstream TC 只 skip dst < i (单向)
 *      → 额外检查: 如果 degree(i) < degree(dst)，反转 search 方向
 *        (在 degree 相近时选 intersect，否则让低度数顶点做 probe)
 *      理由: upstream 的 degree-skew 选择基于 SEARCH_THRESHOLD 的倍数关系,
 *        但中间情况全走 intersect，其实可以让低度数端做 has_edge 查询
 */

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <cstdio>
#include <algorithm>

template <class F, class S>
class PhilemonTcWrapper {
    F& m_method;
    S m_snapshot;
    bool use_optimized_;
    int adaptive_threshold_;  // 1. computed at construction

public:
    PhilemonTcWrapper(F& method, S snapshot, bool use_optimized = false)
        : m_method(method), m_snapshot(snapshot), use_optimized_(use_optimized) {
        // 1. MODIFIED: compute adaptive threshold from degree distribution
        adaptive_threshold_ = compute_adaptive_threshold();
    }

    ~PhilemonTcWrapper() = default;

    uint64_t run_tc() {
        if (use_optimized_) return run_tc_optimized();
        else return run_tc_standard();
    }

private:
    // 1. MODIFIED: adaptive threshold computation
    // upstream: #define SEARCH_THRESHOLD 10 (global constant)
    // ours: sample first min(1000,N) vertices, compute median degree,
    //       set threshold = max(2, median/4)
    int compute_adaptive_threshold() {
        auto N = wrapper::snapshot_vertex_count(m_snapshot);
        uint64_t sample_count = std::min((uint64_t)1000, N);

        std::vector<uint64_t> degrees(sample_count);
        for (uint64_t i = 0; i < sample_count; i++) {
            degrees[i] = wrapper::snapshot_degree(m_snapshot, i);
        }
        std::sort(degrees.begin(), degrees.end());

        uint64_t median = degrees[sample_count / 2];
        int threshold = std::max(2, (int)(median / 4));
        return threshold;
    }

    // ─── Standard TC with modifications ─────────────────────────────
    uint64_t run_tc_standard() {
        auto time_start = std::chrono::high_resolution_clock::now();
        auto N = wrapper::snapshot_vertex_count(m_snapshot);
        uint64_t num_triangles = 0;
        uint64_t num_triangles_from_intersect = 0;

        for (uint64_t i = 0; i < N; i++) {
            auto degree_src = wrapper::snapshot_degree(m_snapshot, i);

            auto get_edges = [&](uint64_t dst, double weight) {
                if (dst >= i) return;  // upstream: dst < i → 同

                auto degree_dst = wrapper::snapshot_degree(m_snapshot, dst);

                // 1. MODIFIED: use adaptive threshold instead of hardcoded 10
                // 4. MODIFIED: in the middle zone, let the lower-degree vertex
                //    do the probing (has_edge queries) — upstream always does intersect
                if (degree_src > degree_dst * adaptive_threshold_) {
                    // src has much higher degree — scan dst's neighbors, probe in src
                    auto search = [&](uint64_t d, double w) {
                        if (wrapper::snapshot_has_edge(m_snapshot, i, d))
                            num_triangles++;
                    };
                    wrapper::snapshot_edges(m_snapshot, dst, search, false);

                } else if (degree_src * adaptive_threshold_ < degree_dst) {
                    // dst has much higher degree — scan src's neighbors, probe in dst
                    auto search = [&](uint64_t d, double w) {
                        if (wrapper::snapshot_has_edge(m_snapshot, dst, d))
                            num_triangles++;
                    };
                    wrapper::snapshot_edges(m_snapshot, i, search, false);

                } else if (degree_src < degree_dst) {
                    // 4. MODIFIED: middle zone — upstream always does intersect
                    // ours: let the lower-degree side do probing instead
                    // When degrees are within threshold ratio but not equal,
                    // probing from the smaller side is cheaper than full intersect
                    auto search = [&](uint64_t d, double w) {
                        if (wrapper::snapshot_has_edge(m_snapshot, dst, d))
                            num_triangles++;
                    };
                    wrapper::snapshot_edges(m_snapshot, i, search, false);

                } else {
                    // Truly similar degrees — intersect is best
                    auto res = wrapper::snapshot_intersect(m_snapshot, i, dst);
                    num_triangles_from_intersect += res;
                }
            };

            wrapper::snapshot_edges(m_snapshot, i, get_edges, false);
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        uint64_t total = num_triangles + num_triangles_from_intersect / 3;
        std::cout << "TC took " << ms << " milliseconds" << std::endl;
        std::cout << "triangle count: " << num_triangles_from_intersect / 3 << std::endl;
        std::cout << "triangle count: " << num_triangles << std::endl;
        return total;
    }

    // ─── Optimized TC with binary search ────────────────────────────
    uint64_t run_tc_optimized() {
        auto time_start = std::chrono::high_resolution_clock::now();
        auto N = wrapper::snapshot_vertex_count(m_snapshot);
        uint64_t num_triangles = 0;

        for (uint64_t n1 = 0; n1 < N; n1++) {
            std::vector<uint64_t> m_neighbors;

            auto get_edges = [&](uint64_t n2, double w2) {
                if (n2 > n1) return;
                m_neighbors.push_back(n2);

                // 2. MODIFIED: choose linear scan vs binary search
                // upstream: always linear scan with marker++
                // ours: binary search when neighbor list is long enough
                bool use_binary = m_neighbors.size() > 64;

                if (use_binary) {
                    // 2. binary search path
                    auto get_intersection = [&](uint64_t n3, double w3) {
                        if (n3 > n2) return;
                        // binary search n3 in m_neighbors
                        if (std::binary_search(m_neighbors.begin(), m_neighbors.end(), n3)) {
                            num_triangles++;
                        }
                    };
                    wrapper::snapshot_edges(m_snapshot, n2, get_intersection, false);
                } else {
                    // original linear scan path (upstream logic, marker-based)
                    uint64_t marker = 0;
                    auto get_intersection = [&](uint64_t n3, double w3) {
                        if (n3 > n2) return;
                        // advance marker past n3
                        while (marker < m_neighbors.size() && n3 > m_neighbors[marker]) {
                            marker++;
                        }
                        if (marker < m_neighbors.size() && n3 == m_neighbors[marker]) {
                            num_triangles++;
                            marker++;
                        }
                    };
                    wrapper::snapshot_edges(m_snapshot, n2, get_intersection, false);
                }
            };

            wrapper::snapshot_edges(m_snapshot, n1, get_edges, false);
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::cout << "TC_opt took " << ms << " milliseconds" << std::endl;
        return num_triangles;
    }
};

#endif // PHILEMON_CROSS_TIER_TC_WRAPPER_HPP
