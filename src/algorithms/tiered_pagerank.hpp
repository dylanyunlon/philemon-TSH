#ifndef PHILEMON_TIERED_PAGERANK_HPP
#define PHILEMON_TIERED_PAGERANK_HPP
/**
 * tiered_pagerank.hpp — PageRank on tiered memory snapshots
 *
 * 骨架来源: upstream/rapidstore/wrapper/algorithms/PR.h (174行)
 * 修改 (~20%):
 *   - 包裹在 philemon::algorithms namespace
 *   - 移除 gapbs::pvector → 使用 std::vector<double>
 *   - 移除 log_info → PHILE_DBG
 *   - 增加 per-iteration convergence 打印
 *   - 增加 per-tier access 统计
 *   - 核心 PageRank 迭代算法 100% 保留
 *
 * Milestone: M014 (Claude #5–6)
 */

#include <iostream>
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {

template <class F, class S>
class TieredPageRank {
    const int m_num_threads;
    const uint64_t m_num_iterations;
    const double m_damping_factor;
    F& m_method;
    S m_snapshot;

public:
    TieredPageRank(int num_threads, uint64_t num_iterations,
                   double damping_factor, F& method, S snapshot)
        : m_num_threads(num_threads), m_num_iterations(num_iterations),
          m_damping_factor(damping_factor), m_method(method),
          m_snapshot(snapshot) {}

    ~TieredPageRank() {}

    void run_page_rank(std::vector<std::pair<uint64_t, double>>& results);

private:
    std::unique_ptr<double[]> page_rank();
};

template <class F, class S>
std::unique_ptr<double[]> TieredPageRank<F,S>::page_rank() {
    debug::ScopedTimer timer("PageRank::compute");
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    const double init_score = 1.0 / N;
    const double base_score = (1.0 - m_damping_factor) / N;

    std::unique_ptr<double[]> scores(new double[N]());
    for (uint64_t v = 0; v < N; v++) scores[v] = init_score;
    std::vector<double> outgoing_contrib(N, 0.0);

    PHILE_DBG(1, "PageRank: N=%lu iters=%lu damping=%.2f",
              (unsigned long)N, (unsigned long)m_num_iterations,
              m_damping_factor);

    for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
        debug::ScopedTimer iter_timer("PR::iteration");
        std::vector<double> dangling_sums(m_num_threads, 0.0);
        double dangling_sum = 0.0;
        uint64_t chunk_size = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        // Phase 1: compute outgoing contributions
        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &dangling_sums, &outgoing_contrib,
                                  chunk_size, N, &scores](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap = wrapper::snapshot_clone(m_snapshot);
                uint64_t start = tid * chunk_size;
                uint64_t end = std::min(start + chunk_size, N);
                for (uint64_t v = start; v < end; v++) {
                    uint64_t deg = wrapper::snapshot_degree(snap, v, false);
                    if (deg == 0) {
                        dangling_sums[tid] += scores[v];
                    } else {
                        outgoing_contrib[v] = scores[v] / deg;
                    }
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();
        threads.clear();

        for (int i = 0; i < m_num_threads; i++)
            dangling_sum += dangling_sums[i];
        dangling_sum /= N;

        // Phase 2: accumulate incoming scores
        double max_delta = 0.0;
        std::vector<double> deltas(m_num_threads, 0.0);

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &outgoing_contrib, chunk_size, N,
                                  &scores, base_score, dangling_sum,
                                  &deltas](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap = wrapper::snapshot_clone(m_snapshot);
                uint64_t start = tid * chunk_size;
                uint64_t end = std::min(start + chunk_size, N);
                for (uint64_t v = start; v < end; v++) {
                    double incoming = 0.0;
                    wrapper::snapshot_edges(snap, v,
                        [&](uint64_t src, double w) {
                            if (src == v) return;
                            incoming += outgoing_contrib[src];
                        }, false);
                    double new_score = base_score +
                        m_damping_factor * (incoming + dangling_sum);
                    double d = std::abs(new_score - scores[v]);
                    if (d > deltas[tid]) deltas[tid] = d;
                    scores[v] = new_score;
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();

        for (auto& d : deltas) max_delta = std::max(max_delta, d);
        PHILE_DBG(2, "PR iter %lu: max_delta=%.8f dangling=%.6f",
                  (unsigned long)iter, max_delta, dangling_sum * N);
    }
    return scores;
}

template <class F, class S>
void TieredPageRank<F,S>::run_page_rank(
    std::vector<std::pair<uint64_t, double>>& results) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto scores = page_rank();

    uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    results.resize(N);
    for (uint64_t u = 0; u < N; u++) {
        results[u] = {u, scores[u]};
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    PHILE_DBG(1, "Tiered PageRank took %ld ms", (long)ms.count());

    if (debug::get_debug_level() >= 1) debug::print_all_tier_perf();
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_TIERED_PAGERANK_HPP
