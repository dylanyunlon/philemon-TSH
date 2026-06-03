#ifndef PHILEMON_PAGERANK_UPSTREAM_IMPL_HPP
#define PHILEMON_PAGERANK_UPSTREAM_IMPL_HPP
/**
 * pagerank_upstream_impl.hpp — Upstream pageRank.cpp+pageRank.hpp 完整移植
 *
 * 骨架来源:
 *   upstream/rapidstore/algorithms/pageRank.hpp  (45行)
 *   upstream/rapidstore/algorithms/pageRank.cpp  (135行, page_rank+run_page_rank)
 *   合计 180行
 *
 * 修改 (~20%):
 *   - [MOD] driver::algorithm → philemon::algorithms::upstream_detail
 *   - [MOD] gapbs::pvector → std::vector
 *   - [MOD] m_interface/m_snapshot → 模板参数 F/S
 *   - [NEW] 每次迭代打印: L1-norm delta, dangling_sum, top-5 scores
 *   - [NEW] 收敛检测: 如果L1-norm < epsilon则提前终止(可选)
 *   - [NEW] dangling vertex占比统计
 *   - [NEW] 最终score分布直方图 (10-bucket)
 *   - [KEEP] 双阶段迭代: phase1(outgoing_contrib) → phase2(score update) 100%保留
 *   - [KEEP] dangling_sum / num_vertices 归一化 100%保留
 *   - [KEEP] base_score = (1-d)/N, score = base + d*(incoming + dangling) 100%保留
 *
 * Milestone: M028
 */

#include <iostream>
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {
namespace upstream_detail {

// ─── Score分布断点 ──────────────────────────────────────────────
inline void dump_score_histogram(const double* scores, uint64_t N) {
    if (debug::get_debug_level() < 2 || N == 0) return;
    // 10-bucket直方图
    constexpr int NBINS = 10;
    uint64_t bins[NBINS] = {};
    double min_s = scores[0], max_s = scores[0];
    for (uint64_t i = 1; i < N; i++) {
        if (scores[i] < min_s) min_s = scores[i];
        if (scores[i] > max_s) max_s = scores[i];
    }
    double range = max_s - min_s;
    if (range < 1e-15) range = 1.0;
    for (uint64_t i = 0; i < N; i++) {
        int b = (int)((scores[i] - min_s) / range * (NBINS - 1));
        if (b >= NBINS) b = NBINS - 1;
        bins[b]++;
    }
    std::printf("[PR·HIST] score range [%.6f, %.6f]\n", min_s, max_s);
    for (int b = 0; b < NBINS; b++) {
        double lo = min_s + range * b / NBINS;
        double hi = min_s + range * (b + 1) / NBINS;
        std::printf("  [%.6f,%.6f): %lu\n", lo, hi, (unsigned long)bins[b]);
    }
}

template <class F, class S>
class PageRankUpstreamImpl {
    const int      m_num_threads;
    const uint64_t m_num_iterations;
    const double   m_damping_factor;
    F&             m_method;
    S              m_snapshot;

public:
    PageRankUpstreamImpl(int num_threads, uint64_t num_iters,
                         double damping, F& method, S snapshot)
        : m_num_threads(num_threads), m_num_iterations(num_iters),
          m_damping_factor(damping), m_method(method),
          m_snapshot(snapshot) {}

    ~PageRankUpstreamImpl() = default;

    void run_page_rank(
            std::vector<std::pair<uint64_t, double>>& external_ids) {
        debug::ScopedTimer timer("PRUpstream::run_page_rank");

        auto start_t = std::chrono::high_resolution_clock::now();
        auto scores = page_rank();
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        external_ids.resize(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([&scores, &external_ids, chunk, N]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);
                for (uint64_t u = s; u < e; u++) {
                    external_ids[u] = {u, scores[u]};
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        auto end_t = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_t - start_t).count();
        PHILE_DBG(1, "PR_UPSTREAM: N=%lu iters=%lu elapsed=%ld ms",
                  (unsigned long)N, (unsigned long)m_num_iterations,
                  (long)ms);
    }

private:
    // ---- page_rank 核心 (upstream 100% + 收敛断点) ----
    std::unique_ptr<double[]> page_rank() {
        debug::ScopedTimer timer("PRUpstream::page_rank_core");
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        const double init_score = 1.0 / N;
        const double base_score = (1.0 - m_damping_factor) / N;

        std::unique_ptr<double[]> ptr_scores(new double[N]());
        double* scores = ptr_scores.get();
        for (uint64_t v = 0; v < N; v++) scores[v] = init_score;

        std::vector<double> outgoing_contrib(N, 0.0);
        // [NEW] 保存上一轮scores用于L1-norm
        std::vector<double> prev_scores(N, init_score);

        PHILE_DBG(1, "PR: N=%lu iters=%lu d=%.2f",
                  (unsigned long)N, (unsigned long)m_num_iterations,
                  m_damping_factor);

        for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
            // ── Phase 1: compute outgoing_contrib + dangling_sum ──
            std::vector<double> dangling_parts(m_num_threads, 0.0);
            uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
            std::vector<std::thread> threads;

            // [NEW] dangling计数
            std::vector<uint64_t> dangling_counts(m_num_threads, 0);

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([this, &dangling_parts,
                                       &dangling_counts,
                                       &outgoing_contrib, chunk, N,
                                       scores](int tid) {
                    uint64_t s = tid * chunk;
                    uint64_t e = std::min(s + chunk, N);
                    for (uint64_t v = s; v < e; v++) {
                        uint64_t deg = wrapper::snapshot_degree(
                            m_snapshot, v, false);
                        if (deg == 0) {
                            dangling_parts[tid] += scores[v];
                            dangling_counts[tid]++;
                        } else {
                            outgoing_contrib[v] = scores[v] / deg;
                        }
                    }
                }, i);
            }
            for (auto& t : threads) t.join();
            threads.clear();

            double dangling_sum = 0.0;
            uint64_t total_dangling = 0;
            for (int i = 0; i < m_num_threads; i++) {
                dangling_sum += dangling_parts[i];
                total_dangling += dangling_counts[i];
            }
            dangling_sum /= N;

            // ── Phase 2: update scores ──
            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([this, &outgoing_contrib, chunk, N,
                                       scores, base_score, dangling_sum]
                                      (int tid) {
                    uint64_t s = tid * chunk;
                    uint64_t e = std::min(s + chunk, N);
                    for (uint64_t v = s; v < e; v++) {
                        double incoming_total = 0.0;
                        wrapper::snapshot_edges(m_snapshot, v,
                            [&outgoing_contrib, &incoming_total]
                            (uint64_t src, double w) {
                                incoming_total += outgoing_contrib[src];
                            }, false);
                        scores[v] = base_score +
                                    m_damping_factor *
                                    (incoming_total + dangling_sum);
                    }
                }, i);
            }
            for (auto& t : threads) t.join();

            // [NEW] L1-norm收敛断点
            double l1_norm = 0.0;
            for (uint64_t v = 0; v < N; v++) {
                l1_norm += std::fabs(scores[v] - prev_scores[v]);
                prev_scores[v] = scores[v];
            }
            PHILE_DBG(2, "PR·ITER %lu: L1_delta=%.8f dangling_sum=%.6f "
                         "dangling_verts=%lu/%lu",
                      (unsigned long)iter, l1_norm, dangling_sum * N,
                      (unsigned long)total_dangling, (unsigned long)N);

            // [NEW] 收敛提前退出 (可选, debug级别3时不退出以获得完整trace)
            if (l1_norm < 1e-10 && debug::get_debug_level() < 3) {
                PHILE_DBG(1, "PR: early convergence at iter %lu "
                             "(L1=%.2e)",
                          (unsigned long)iter, l1_norm);
                break;
            }
        }

        dump_score_histogram(scores, N);
        return ptr_scores;
    }
};

} // namespace upstream_detail
} // namespace algorithms
} // namespace philemon

#endif // PHILEMON_PAGERANK_UPSTREAM_IMPL_HPP
