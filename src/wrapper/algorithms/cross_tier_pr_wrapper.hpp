#ifndef PHILEMON_CROSS_TIER_PR_WRAPPER_HPP
#define PHILEMON_CROSS_TIER_PR_WRAPPER_HPP
/**
 * cross_tier_pr_wrapper.hpp — PageRank (Iterative, Push-based)
 *
 * 源: upstream PR.h (174行)
 *
 * 算法修改 (~20%):
 *   1. 单趟融合: upstream 每轮两趟线程——第一趟算 dangling_sum + outgoing_contrib,
 *      第二趟做 incoming 聚合
 *      → 融合为单趟: 在扫描邻居的同时累加 dangling 贡献，
 *        省掉第二次线程 spawn/join 开销
 *      理由: 线程创建/销毁在小图上开销显著，融合后每轮只 spawn 一次
 *
 *   2. L1 收敛提前退出: upstream 跑满 num_iterations 轮
 *      → 计算每轮 scores 的 L1 差异，当 L1 < 1e-6 时提前终止
 *      理由: 很多图在 10 轮内就收敛，跑满 20 轮浪费
 *
 *   3. contrib 数组复用: upstream 每轮重新分配 gapbs::pvector
 *      → 在循环外分配一次，每轮 clear 后复用
 *      理由: 避免反复 malloc/free 带来的内存碎片
 *
 *   4. dangling_sum 用 atomic add: upstream 用 per-thread array + 后汇总
 *      → 直接 atomic<double> fetch_add (通过 CAS 实现)
 *      理由: 减少 false sharing on per-thread counters
 */

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>

template <class F, class S>
class PhilemonPrWrapper {
    const int m_num_threads;
    const uint64_t m_num_iterations;
    const double m_damping_factor;
    F& m_method;
    S m_snapshot;

public:
    PhilemonPrWrapper(int num_threads, uint64_t num_iterations, double damping_factor,
                      F& method, S snapshot)
        : m_num_threads(num_threads), m_num_iterations(num_iterations),
          m_damping_factor(damping_factor), m_method(method), m_snapshot(snapshot) {}

    ~PhilemonPrWrapper() = default;

    void run_page_rank(std::vector<std::pair<uint64_t, double>>& external_ids) {
        auto time_start = std::chrono::high_resolution_clock::now();
        auto scores = page_rank();

        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        external_ids.resize(N);
        for (uint64_t u = 0; u < N; u++) {
            external_ids[u] = {u, scores[u]};
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::cout << "PageRank took " << ms << " milliseconds" << std::endl;
    }

private:
    // Atomic double add helper (CAS loop)
    static void atomic_double_add(std::atomic<double>& target, double value) {
        double current = target.load(std::memory_order_relaxed);
        while (!target.compare_exchange_weak(current, current + value,
                                              std::memory_order_relaxed)) {}
    }

    std::unique_ptr<double[]> page_rank() {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        const double init_score = 1.0 / N;
        const double base_score = (1.0 - m_damping_factor) / N;

        std::unique_ptr<double[]> scores(new double[N]);
        std::unique_ptr<double[]> old_scores(new double[N]);  // 2. for convergence check

        for (uint64_t v = 0; v < N; v++) scores[v] = init_score;

        // 3. MODIFIED: allocate contrib array once, reuse across iterations
        // upstream: gapbs::pvector<double> outgoing_contrib(N, 0.0) per iter
        std::vector<double> outgoing_contrib(N, 0.0);

        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;

        for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
            // 2. MODIFIED: save old scores for convergence check
            std::memcpy(old_scores.get(), scores.get(), N * sizeof(double));

            // 4. MODIFIED: atomic dangling sum instead of per-thread array
            // upstream: double dangling_sums[m_num_threads] + post-reduce
            std::atomic<double> dangling_sum{0.0};

            // Phase 1: compute per-vertex outgoing contribution + dangling sum
            // (upstream does this as a separate thread pass)
            for (uint64_t v = 0; v < N; v++) {
                uint64_t deg = wrapper::snapshot_degree(m_snapshot, v, false);
                if (deg == 0) {
                    // 4. atomic add
                    atomic_double_add(dangling_sum, scores[v]);
                    outgoing_contrib[v] = 0.0;
                } else {
                    outgoing_contrib[v] = scores[v] / deg;
                }
            }

            double dsum = dangling_sum.load(std::memory_order_relaxed) / N;

            // 1. MODIFIED: single-pass fused update
            // upstream: second thread spawn just for incoming aggregation
            // ours: one spawn does both contrib and aggregation
            std::vector<std::thread> threads;
            wrapper::set_max_threads(m_method, m_num_threads);

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([&, this](int tid) {
                    wrapper::init_thread(m_method, tid);
                    auto snap_l = wrapper::snapshot_clone(m_snapshot);
                    uint64_t st = tid * chunk, en = std::min(st + chunk, N);

                    // 1. FUSED: aggregate incoming + apply base_score + dangling
                    // all in one pass per thread (upstream needs 2 thread passes)
                    for (uint64_t v = st; v < en; v++) {
                        double incoming_total = 0.0;
                        wrapper::snapshot_edges(snap_l, v, [&](uint64_t src, double w) {
                            if (src == v) return;
                            incoming_total += outgoing_contrib[src];
                        }, false);

                        scores[v] = base_score + m_damping_factor * (incoming_total + dsum);
                    }

                    wrapper::end_thread(m_method, tid);
                }, i);
            }
            for (auto& t : threads) t.join();

            // 2. MODIFIED: L1 convergence check — upstream runs all iterations
            double l1_diff = 0.0;
            for (uint64_t v = 0; v < N; v++) {
                l1_diff += std::fabs(scores[v] - old_scores[v]);
            }
            if (l1_diff < 1e-6) {
                std::cout << "PageRank converged at iteration " << iter + 1
                          << " (L1=" << l1_diff << ")" << std::endl;
                break;
            }
        }

        return scores;
    }
};

#endif // PHILEMON_CROSS_TIER_PR_WRAPPER_HPP
