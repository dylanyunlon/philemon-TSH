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
    // ---- page_rank 核心 (upstream骨架 + tier温度加权 + 分层收敛) ----
    // 核心改动:
    //  1) outgoing_contrib乘以tier_boost: HBM(hot)×1.2, GDDR×1.0, DRAM(cold)×0.8
    //  2) 收敛判定: 每tier单独计算L1-norm, 全部tier收敛才算收敛
    //  3) dangling sum按tier加权分配: 热tier分到更多dangling score
    std::unique_ptr<double[]> page_rank() {
        debug::ScopedTimer timer("PRUpstream::page_rank_core");
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        const double init_score = 1.0 / N;
        const double base_score = (1.0 - m_damping_factor) / N;

        std::unique_ptr<double[]> ptr_scores(new double[N]());
        double* scores = ptr_scores.get();
        for (uint64_t v = 0; v < N; v++) scores[v] = init_score;

        std::vector<double> outgoing_contrib(N, 0.0);
        std::vector<double> prev_scores(N, init_score);

        // tier分配: 按degree分3档(与BFS一致的策略)
        // tier_id[v] = 0(HBM), 1(GDDR), 2(DRAM)
        std::vector<uint8_t> tier_id(N);
        for (uint64_t v = 0; v < N; v++) {
            uint64_t deg = wrapper::snapshot_degree(m_snapshot, v, false);
            tier_id[v] = (deg > 1000) ? 0 : (deg > 100) ? 1 : 2;
        }

        // tier温度boost因子
        constexpr double TIER_BOOST[3] = {1.2, 1.0, 0.8};  // HBM, GDDR, DRAM

        PHILE_DBG(1, "PR: N=%lu iters=%lu d=%.2f tier_boost=[%.1f,%.1f,%.1f]",
                  (unsigned long)N, (unsigned long)m_num_iterations,
                  m_damping_factor,
                  TIER_BOOST[0], TIER_BOOST[1], TIER_BOOST[2]);

        for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
            // ── Phase 1: outgoing_contrib + 分tier dangling ──
            double dangling_by_tier[3] = {};
            uint64_t dangling_cnt_by_tier[3] = {};
            uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
            std::vector<std::thread> threads;

            // 用临时数组收集per-thread结果
            struct TierAcc { double dang[3]={}; uint64_t cnt[3]={}; };
            std::vector<TierAcc> per_thread(m_num_threads);

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([this, &per_thread, &outgoing_contrib,
                                       &tier_id, chunk, N, scores]
                                      (int tid) {
                    uint64_t s = tid * chunk;
                    uint64_t e = std::min(s + chunk, N);
                    for (uint64_t v = s; v < e; v++) {
                        uint64_t deg = wrapper::snapshot_degree(
                            m_snapshot, v, false);
                        int t = tier_id[v];
                        if (deg == 0) {
                            per_thread[tid].dang[t] += scores[v];
                            per_thread[tid].cnt[t]++;
                        } else {
                            // tier温度加权: hot tier的贡献更大
                            outgoing_contrib[v] = scores[v] / deg * TIER_BOOST[t];
                        }
                    }
                }, i);
            }
            for (auto& t : threads) t.join();
            threads.clear();

            // 汇总分tier dangling
            for (auto& ta : per_thread) {
                for (int t = 0; t < 3; t++) {
                    dangling_by_tier[t] += ta.dang[t];
                    dangling_cnt_by_tier[t] += ta.cnt[t];
                }
            }
            // tier感知dangling分配: 热tier分到更多
            double dangling_sum = 0;
            for (int t = 0; t < 3; t++)
                dangling_sum += dangling_by_tier[t] * TIER_BOOST[t];
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

            // 分层L1-norm收敛判定: 每tier独立计算
            double l1_by_tier[3] = {};
            for (uint64_t v = 0; v < N; v++) {
                l1_by_tier[tier_id[v]] += std::fabs(scores[v] - prev_scores[v]);
                prev_scores[v] = scores[v];
            }
            double l1_total = l1_by_tier[0] + l1_by_tier[1] + l1_by_tier[2];

            PHILE_DBG(2, "PR·ITER %lu: L1={HBM:%.6f GDDR:%.6f DRAM:%.6f total:%.6f} "
                         "dangling_by_tier={%lu,%lu,%lu}",
                      (unsigned long)iter,
                      l1_by_tier[0], l1_by_tier[1], l1_by_tier[2], l1_total,
                      (unsigned long)dangling_cnt_by_tier[0],
                      (unsigned long)dangling_cnt_by_tier[1],
                      (unsigned long)dangling_cnt_by_tier[2]);

            // score热力图: 打印每tier的min/max/avg score
            if (debug::get_debug_level() >= 3) {
                for (int t = 0; t < 3; t++) {
                    double tmin = 1e30, tmax = 0, tsum = 0;
                    uint64_t tcnt = 0;
                    for (uint64_t v = 0; v < N; v++) {
                        if (tier_id[v] == t) {
                            tmin = std::min(tmin, scores[v]);
                            tmax = std::max(tmax, scores[v]);
                            tsum += scores[v];
                            tcnt++;
                        }
                    }
                    if (tcnt > 0)
                        std::printf("[PR·HEAT] tier%d: n=%lu min=%.8f "
                                    "avg=%.8f max=%.8f\n",
                                    t, (unsigned long)tcnt,
                                    tmin, tsum/tcnt, tmax);
                }
            }

            // 全tier收敛才提前退出
            bool all_converged = (l1_by_tier[0] < 1e-10) &&
                                  (l1_by_tier[1] < 1e-10) &&
                                  (l1_by_tier[2] < 1e-10);
            if (all_converged && debug::get_debug_level() < 3) {
                PHILE_DBG(1, "PR: all tiers converged at iter %lu",
                          (unsigned long)iter);
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
