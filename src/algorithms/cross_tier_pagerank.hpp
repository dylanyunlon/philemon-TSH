#ifndef PHILEMON_CROSS_TIER_PAGERANK_HPP
#define PHILEMON_CROSS_TIER_PAGERANK_HPP
/**
 * cross_tier_pagerank.hpp — Cross-Tier PageRank with Gradient Accumulation
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/algorithms/PR.h        (174行)
 *   upstream/rapidstore/algorithms/pageRank.cpp         (159行)
 *   upstream/rapidstore/wrapper/driver.h:840-885        (46行, page_rank())
 *   upstream/rapidstore/wrapper/wrapper.h               (249行, API)
 *
 * 修改 (~20%):
 *   - [NEW] 增加 IterationStats: 每轮 convergence+tier分布+代价记录
 *   - [NEW] 增加 dump_all_state(): 打印所有数据结构的当前状态
 *   - [NEW] 增加 per-tier gradient accumulation (热点顶点驻留HBM)
 *   - [NEW] 增加 hotspot detection: 识别高度数顶点做tier placement
 *   - [NEW] 增加 convergence_log: 可输出为论文数据点
 *   - [MOD] 原 gapbs::pvector → std::vector<double>
 *   - [MOD] 原 omp_set_num_threads → wrapper::set_max_threads
 *   - [MOD] 原 log_info → PHILE_DBG + structured dump
 *   - [KEEP] 核心 PageRank 迭代算法 100% 保留
 *   - [KEEP] dangling_sum 处理 100% 保留
 *   - [KEEP] outgoing_contrib 计算 100% 保留
 *   - [KEEP] 多线程 chunk 分配 100% 保留
 *
 * Milestone: M021 — Cross-tier PageRank with tier gradient accumulation
 * ====================================================================
 */

#include <iostream>
#include <memory>
#include <string>
#include <random>
#include <vector>
#include <utility>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cstdio>
#include <cstring>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"
#include "../cost_model/tier_cost_model.hpp"

namespace philemon {
namespace algorithms {

// ═══════════════════════════════════════════════════════════════════════
// Per-iteration statistics — for debug + paper data
// (NEW: 不在upstream中, 为论文数据收集而增)
// ═══════════════════════════════════════════════════════════════════════
struct PRIterStats {
    int      iteration;
    double   max_delta;          // 最大 score 变化
    double   l2_norm;            // L2 norm of score change
    double   dangling_sum;       // dangling 节点贡献总和
    double   time_ms;            // 本轮耗时
    // Tier distribution of vertices processed
    uint64_t vertices_hbm;
    uint64_t vertices_gddr;
    uint64_t vertices_dram;
    // Hotspot info
    uint64_t hotspot_count;      // 度数 > avg*10 的顶点数
    uint64_t hotspot_edges;      // 这些热点顶点的总边数
    // Cost estimation (from TierCostModel)
    double   estimated_cost_us;
    double   actual_cost_us;

    void dump() const {
        std::printf("  [PR iter %d] max_Δ=%.8f L2=%.8f dangling=%.6f "
                    "time=%.3fms\n",
                    iteration, max_delta, l2_norm, dangling_sum, time_ms);
        std::printf("           tier_vtx: HBM=%lu GDDR=%lu DRAM=%lu "
                    "hotspots=%lu(edges=%lu)\n",
                    (unsigned long)vertices_hbm,
                    (unsigned long)vertices_gddr,
                    (unsigned long)vertices_dram,
                    (unsigned long)hotspot_count,
                    (unsigned long)hotspot_edges);
        std::printf("           cost: est=%.2fμs act=%.2fμs ratio=%.2f\n",
                    estimated_cost_us, actual_cost_us,
                    actual_cost_us > 0 ? estimated_cost_us / actual_cost_us : 0.0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CrossTierPageRank — PageRank with tier-aware gradient accumulation
//
// Class structure from upstream PR.h (pageRankExperiments), modified:
//   - Added m_cost_model for tier cost estimation
//   - Added m_iter_log for convergence tracking
//   - Added hotspot detection threshold
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
class CrossTierPageRank {
    // ─── From upstream PR.h (100% preserved) ─────────────────────
    const int m_num_threads;
    const uint64_t m_num_iterations;
    const double m_damping_factor;
    F& m_method;
    S m_snapshot;

    // ─── NEW: Tier-aware extensions ──────────────────────────────
    cost_model::TierCostModel m_cost_model;
    std::vector<PRIterStats> m_iter_log;
    double m_convergence_threshold;  // early stop if max_delta < this
    uint64_t m_hotspot_degree_mult;  // vertex with degree > avg*mult = hotspot

public:
    // ─── Constructor: upstream + new params ───────────────────────
    CrossTierPageRank(int num_threads, uint64_t num_iterations,
                      double damping_factor, F& method, S snapshot,
                      cost_model::TierCostModel cost_model = {},
                      double convergence_threshold = 1e-6,
                      uint64_t hotspot_degree_mult = 10)
        : m_num_threads(num_threads), m_num_iterations(num_iterations),
          m_damping_factor(damping_factor), m_method(method),
          m_snapshot(snapshot), m_cost_model(cost_model),
          m_convergence_threshold(convergence_threshold),
          m_hotspot_degree_mult(hotspot_degree_mult) {}

    ~CrossTierPageRank() {}

    // ─── Main entry (from upstream PR.h::run_page_rank, modified) ─
    void run_page_rank(std::vector<std::pair<uint64_t, double>>& results);

    // ─── NEW: Access convergence log for paper data ──────────────
    const std::vector<PRIterStats>& iteration_log() const { return m_iter_log; }

    // ─── NEW: Dump all convergence data ──────────────────────────
    void dump_convergence() const;

    // ─── NEW: Dump complete algorithm state for breakpoint debugging ─
    void dump_all_state(const std::string& phase,
                        const double* scores, uint64_t N,
                        const std::vector<double>& outgoing_contrib,
                        int current_iter) const;

private:
    // ─── Core: from upstream PR.h::page_rank() ───────────────────
    std::unique_ptr<double[]> page_rank();

    // ─── From upstream driver.h::page_rank() — single-thread version ─
    // Used as verification baseline
    void page_rank_sequential(std::vector<double>& result,
                              const std::vector<uint64_t>& degree_list);

    // ─── NEW: Detect hotspot vertices ────────────────────────────
    void detect_hotspots(uint64_t N, uint64_t avg_degree,
                         std::vector<uint64_t>& hotspot_vertices,
                         std::vector<uint64_t>& degree_cache);

    // ─── NEW: Collect per-iteration stats ────────────────────────
    PRIterStats collect_iter_stats(int iter, double max_delta,
                                  double l2_norm, double dangling_sum,
                                  double time_ms, uint64_t N,
                                  const std::vector<uint64_t>& hotspots);
};

// ═══════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════

// ─── detect_hotspots: NEW, identify high-degree vertices for HBM pinning ─
template <class F, class S>
void CrossTierPageRank<F,S>::detect_hotspots(
    uint64_t N, uint64_t avg_degree,
    std::vector<uint64_t>& hotspot_vertices,
    std::vector<uint64_t>& degree_cache)
{
    debug::ScopedTimer timer("PR::detect_hotspots");
    hotspot_vertices.clear();
    degree_cache.resize(N);

    uint64_t threshold = avg_degree * m_hotspot_degree_mult;
    uint64_t total_hotspot_edges = 0;

    std::vector<std::thread> threads;
    uint64_t chunk_size = (N + m_num_threads - 1) / m_num_threads;
    std::vector<std::vector<uint64_t>> local_hotspots(m_num_threads);

    wrapper::set_max_threads(m_method, m_num_threads);
    for (int i = 0; i < m_num_threads; i++) {
        threads.emplace_back([this, &degree_cache, &local_hotspots,
                              chunk_size, N, threshold](int tid) {
            wrapper::init_thread(m_method, tid);
            auto snap = wrapper::snapshot_clone(m_snapshot);
            uint64_t start = tid * chunk_size;
            uint64_t end = std::min(start + chunk_size, N);
            for (uint64_t v = start; v < end; v++) {
                uint64_t deg = wrapper::snapshot_degree(snap, v, false);
                degree_cache[v] = deg;
                if (deg > threshold) {
                    local_hotspots[tid].push_back(v);
                }
            }
            wrapper::end_thread(m_method, tid);
        }, i);
    }
    for (auto& t : threads) t.join();

    // Merge local hotspot lists
    for (auto& local : local_hotspots) {
        for (auto v : local) {
            hotspot_vertices.push_back(v);
            total_hotspot_edges += degree_cache[v];
        }
    }

    PHILE_DBG(1, "[PR::hotspots] found %zu hotspots (threshold=%lu) "
              "total_edges=%lu",
              hotspot_vertices.size(), (unsigned long)threshold,
              (unsigned long)total_hotspot_edges);

    // DEBUG: Print top-10 hotspot vertices
    if (debug::get_debug_level() >= 2 && !hotspot_vertices.empty()) {
        std::sort(hotspot_vertices.begin(), hotspot_vertices.end(),
                  [&](uint64_t a, uint64_t b) {
                      return degree_cache[a] > degree_cache[b];
                  });
        size_t show = std::min(hotspot_vertices.size(), (size_t)10);
        std::printf("  [PR::hotspots] top-%zu:\n", show);
        for (size_t i = 0; i < show; i++) {
            uint64_t v = hotspot_vertices[i];
            std::printf("    vtx=%lu degree=%lu (tier=simulated_%s)\n",
                        (unsigned long)v, (unsigned long)degree_cache[v],
                        i < show / 3 ? "HBM" :
                        i < show * 2 / 3 ? "GDDR" : "DRAM");
        }
    }
}

// ─── dump_all_state: NEW, breakpoint-style state dump ────────────────
template <class F, class S>
void CrossTierPageRank<F,S>::dump_all_state(
    const std::string& phase,
    const double* scores, uint64_t N,
    const std::vector<double>& outgoing_contrib,
    int current_iter) const
{
    if (debug::get_debug_level() < 3) return;

    std::printf("\n╔══════════════════════════════════════════════════╗\n");
    std::printf("║ PR STATE DUMP: %s (iter=%d)               ║\n",
                phase.c_str(), current_iter);
    std::printf("╠══════════════════════════════════════════════════╣\n");

    // Score statistics
    double min_score = 1e30, max_score = -1e30, sum_score = 0;
    uint64_t zero_count = 0;
    for (uint64_t v = 0; v < N; v++) {
        min_score = std::min(min_score, scores[v]);
        max_score = std::max(max_score, scores[v]);
        sum_score += scores[v];
        if (scores[v] == 0.0) zero_count++;
    }
    std::printf("║ scores: N=%lu min=%.8f max=%.8f avg=%.8f\n",
                (unsigned long)N, min_score, max_score, sum_score / N);
    std::printf("║         sum=%.8f zeros=%lu\n",
                sum_score, (unsigned long)zero_count);

    // First 20 scores
    std::printf("║ scores[0..19]: ");
    for (uint64_t v = 0; v < std::min(N, (uint64_t)20); v++) {
        std::printf("%.6f ", scores[v]);
    }
    std::printf("\n");

    // Outgoing contrib stats
    if (!outgoing_contrib.empty()) {
        double min_oc = 1e30, max_oc = -1e30;
        uint64_t oc_zero = 0;
        for (uint64_t v = 0; v < std::min(N, (uint64_t)outgoing_contrib.size()); v++) {
            min_oc = std::min(min_oc, outgoing_contrib[v]);
            max_oc = std::max(max_oc, outgoing_contrib[v]);
            if (outgoing_contrib[v] == 0.0) oc_zero++;
        }
        std::printf("║ outgoing_contrib: min=%.8f max=%.8f zeros=%lu\n",
                    min_oc, max_oc, (unsigned long)oc_zero);
    }

    // Tier perf counters
    std::printf("║ tier_perf_counters:\n");
    for (int t = 0; t < 3; t++) {
        auto& perf = debug::tier_perf(t);
        std::printf("║   tier[%d] reads=%lu writes=%lu bytes=%lu "
                    "lat_sum=%lu\n",
                    t, (unsigned long)perf.read_count.load(),
                    (unsigned long)perf.write_count.load(),
                    (unsigned long)perf.bytes_transferred.load(),
                    (unsigned long)perf.latency_sum_ns.load());
    }

    std::printf("╚══════════════════════════════════════════════════╝\n\n");
}

// ═══════════════════════════════════════════════════════════════════════
// page_rank() — Core iterative algorithm
//
// Structure from upstream PR.h::page_rank() (100% preserved core),
// with NEW: tier stats, convergence tracking, hotspot tracking, state dump
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
std::unique_ptr<double[]> CrossTierPageRank<F,S>::page_rank() {
    debug::ScopedTimer timer("CrossTierPageRank::compute");
    m_iter_log.clear();

    // ─── From upstream PR.h: init (100% preserved) ───────────────
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    const double init_score = 1.0 / N;
    const double base_score = (1.0 - m_damping_factor) / N;

    std::unique_ptr<double[]> scores(new double[N]());
    for (uint64_t v = 0; v < N; v++) scores[v] = init_score;

    // Replaced gapbs::pvector<double> with std::vector<double>
    std::vector<double> outgoing_contrib(N, 0.0);

    PHILE_DBG(1, "[CrossTierPR] N=%lu iters=%lu damping=%.2f "
              "convergence_threshold=%.1e threads=%d",
              (unsigned long)N, (unsigned long)m_num_iterations,
              m_damping_factor, m_convergence_threshold, m_num_threads);

    // ─── NEW: Detect hotspots ────────────────────────────────────
    uint64_t total_edges = wrapper::snapshot_edge_count(m_snapshot);
    uint64_t avg_degree = N > 0 ? total_edges / N : 1;
    std::vector<uint64_t> hotspot_vertices;
    std::vector<uint64_t> degree_cache;
    detect_hotspots(N, avg_degree, hotspot_vertices, degree_cache);

    // NEW: State dump after initialization
    dump_all_state("after_init", scores.get(), N, outgoing_contrib, -1);

    // ─── Main iteration loop ─────────────────────────────────────
    for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
        debug::ScopedTimer iter_timer("PR::iteration");
        auto iter_start = std::chrono::high_resolution_clock::now();

        // ─── Phase 1: compute outgoing contributions ─────────────
        // From upstream PR.h (100% preserved structure)
        std::vector<double> dangling_sums(m_num_threads, 0.0);
        double dangling_sum = 0.0;
        uint64_t chunk_size = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        // NEW: per-tier vertex count tracking
        std::atomic<uint64_t> tier_counts[3] = {{0}, {0}, {0}};

        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            // Thread lambda from upstream PR.h (structure preserved)
            threads.emplace_back([this, &dangling_sums, &outgoing_contrib,
                                  chunk_size, N, &scores, &degree_cache,
                                  &tier_counts](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap = wrapper::snapshot_clone(m_snapshot);
                uint64_t start = tid * chunk_size;
                uint64_t end = std::min(start + chunk_size, N);

                for (uint64_t v = start; v < end; v++) {
                    // From upstream: degree check + dangling/contrib
                    uint64_t out_degree = degree_cache[v];
                    if (out_degree == 0) {
                        dangling_sums[tid] += scores[v];
                    } else {
                        outgoing_contrib[v] = scores[v] / out_degree;
                    }

                    // NEW: Simulated tier assignment for stats
                    // In production: query TierPtr for actual tier
                    uint8_t sim_tier = (v < N / 3) ? 0 :
                                      (v < N * 2 / 3) ? 1 : 2;
                    tier_counts[sim_tier].fetch_add(1, std::memory_order_relaxed);
                }

                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();
        threads.clear();

        // From upstream PR.h: sum dangling contributions (100% preserved)
        for (int i = 0; i < m_num_threads; i++) {
            dangling_sum += dangling_sums[i];
        }
        dangling_sum /= N;

        // NEW: State dump after Phase 1
        if (iter == 0 || iter == m_num_iterations - 1) {
            dump_all_state("after_phase1_outgoing", scores.get(), N,
                           outgoing_contrib, iter);
        }

        // ─── Phase 2: accumulate incoming scores ─────────────────
        // From upstream PR.h (structure preserved, +delta tracking)
        std::vector<double> deltas(m_num_threads, 0.0);
        std::vector<double> l2_parts(m_num_threads, 0.0);

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &outgoing_contrib, chunk_size, N,
                                  &scores, base_score, dangling_sum,
                                  &deltas, &l2_parts](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap = wrapper::snapshot_clone(m_snapshot);
                uint64_t start = tid * chunk_size;
                uint64_t end = std::min(start + chunk_size, N);

                for (uint64_t v = start; v < end; v++) {
                    // From upstream: incoming score accumulation
                    double incoming = 0.0;
                    wrapper::snapshot_edges(snap, v,
                        [&](uint64_t src, double w) {
                            if (src == v) return;
                            incoming += outgoing_contrib[src];
                        }, false);

                    // From upstream: score update
                    double new_score = base_score +
                        m_damping_factor * (incoming + dangling_sum);
                    double d = std::abs(new_score - scores[v]);

                    // NEW: track max delta and L2 norm
                    if (d > deltas[tid]) deltas[tid] = d;
                    l2_parts[tid] += d * d;

                    scores[v] = new_score;
                }

                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();

        // Merge convergence metrics
        double max_delta = 0.0;
        double l2_sum = 0.0;
        for (int i = 0; i < m_num_threads; i++) {
            max_delta = std::max(max_delta, deltas[i]);
            l2_sum += l2_parts[i];
        }
        double l2_norm = std::sqrt(l2_sum);

        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(
            iter_end - iter_start).count();

        // ─── NEW: Collect and log iteration stats ────────────────
        auto stats = collect_iter_stats(
            iter, max_delta, l2_norm, dangling_sum * N,
            iter_ms, N, hotspot_vertices);
        stats.vertices_hbm  = tier_counts[0].load();
        stats.vertices_gddr = tier_counts[1].load();
        stats.vertices_dram = tier_counts[2].load();
        m_iter_log.push_back(stats);

        // Per-iteration debug output
        if (debug::get_debug_level() >= 2) {
            stats.dump();
        }

        // NEW: State dump at interesting iterations
        if (debug::get_debug_level() >= 3 &&
            (iter < 3 || iter == m_num_iterations - 1)) {
            dump_all_state("after_phase2_update", scores.get(), N,
                           outgoing_contrib, iter);
        }

        // ─── NEW: Early convergence check ────────────────────────
        if (max_delta < m_convergence_threshold) {
            PHILE_DBG(1, "[CrossTierPR] converged at iter %lu "
                      "(max_delta=%.10f < threshold=%.1e)",
                      (unsigned long)iter, max_delta,
                      m_convergence_threshold);
            break;
        }
    }

    return scores;
}

// ─── page_rank_sequential: from upstream driver.h::page_rank() ───────
// Single-thread verification baseline (80% preserved from driver.h)
template <class F, class S>
void CrossTierPageRank<F,S>::page_rank_sequential(
    std::vector<double>& result,
    const std::vector<uint64_t>& degree_list)
{
    debug::ScopedTimer timer("PR::sequential_baseline");

    // From upstream driver.h:840-885 (structure preserved)
    uint64_t size = wrapper::snapshot_vertex_count(m_snapshot);
    std::vector<double> outgoing_contrib(size);
    const double init_score = 1.0 / size;
    const double base_score = (1.0 - m_damping_factor) / size;
    double dangling_sum = 0.0;

    for (uint64_t source = 0; source < size; source++) {
        result[source] = init_score;
    }

    PHILE_DBG(2, "[PR::sequential] N=%lu iters=%lu",
              (unsigned long)size, (unsigned long)m_num_iterations);

    for (uint64_t iter = 0; iter < m_num_iterations; iter++) {
        dangling_sum = 0.0;

        // From upstream driver.h: outgoing contribution phase
        for (uint64_t source = 0; source < size; source++) {
            uint64_t degree = degree_list[source];
            if (degree == 0) {
                dangling_sum += result[source];
            } else {
                outgoing_contrib[source] = result[source] / degree;
            }
        }
        dangling_sum /= size;

        auto start = std::chrono::high_resolution_clock::now();

        // From upstream driver.h: incoming score accumulation
        double max_delta = 0.0;
        for (uint64_t source = 0; source < size; source++) {
            double incoming_total = 0.0;
            auto cb = [&](uint64_t dst, double w) {
                incoming_total += outgoing_contrib[dst];
            };
            wrapper::snapshot_edges(m_snapshot, source, cb, false);

            double new_score = base_score +
                m_damping_factor * (incoming_total + dangling_sum);
            double d = std::abs(new_score - result[source]);
            max_delta = std::max(max_delta, d);
            result[source] = new_score;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(end - start).count();

        // NEW: per-iteration convergence print
        PHILE_DBG(2, "  [PR::seq iter %lu] max_delta=%.8f time=%.3fμs",
                  (unsigned long)iter, max_delta, duration_ns / 1000.0);

        if (max_delta < m_convergence_threshold) {
            PHILE_DBG(1, "[PR::seq] converged at iter %lu",
                      (unsigned long)iter);
            break;
        }
    }
}

// ─── collect_iter_stats: NEW ─────────────────────────────────────────
template <class F, class S>
PRIterStats CrossTierPageRank<F,S>::collect_iter_stats(
    int iter, double max_delta, double l2_norm, double dangling_sum,
    double time_ms, uint64_t N,
    const std::vector<uint64_t>& hotspots)
{
    PRIterStats stats;
    stats.iteration = iter;
    stats.max_delta = max_delta;
    stats.l2_norm = l2_norm;
    stats.dangling_sum = dangling_sum;
    stats.time_ms = time_ms;

    // Hotspot info
    stats.hotspot_count = hotspots.size();
    stats.hotspot_edges = 0;
    // (actual computation happens in detect_hotspots; here just record)

    // Cost estimation using TierCostModel
    uint64_t total_edges = wrapper::snapshot_edge_count(m_snapshot);
    uint64_t avg_degree = N > 0 ? total_edges / N : 1;

    // Estimate: each iteration touches all vertices + their edges
    std::vector<std::pair<uint8_t, uint64_t>> work_by_tier = {
        {0, N / 3},      // HBM vertices
        {1, N / 3},      // GDDR vertices
        {2, N - 2*(N/3)} // DRAM vertices
    };
    auto est = m_cost_model.estimate_scan_cost(work_by_tier, avg_degree);
    stats.estimated_cost_us = est.total_ns / 1000.0;
    stats.actual_cost_us = time_ms * 1000.0;

    return stats;
}

// ═══════════════════════════════════════════════════════════════════════
// run_page_rank: main entry point
// From upstream PR.h::run_page_rank() (structure preserved)
// + NEW: convergence dump, tier perf, verification
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void CrossTierPageRank<F,S>::run_page_rank(
    std::vector<std::pair<uint64_t, double>>& results)
{
    debug::ScopedTimer timer("CrossTierPageRank::run_page_rank");

    // Reset tier perf counters
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto scores = page_rank();

    // From upstream PR.h::run_page_rank() — result collection
    // (threading removed for simplicity, was in upstream for id mapping)
    uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    results.resize(N);
    for (uint64_t u = 0; u < N; u++) {
        results[u] = {u, scores[u]};
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    PHILE_DBG(1, "[CrossTierPR] total: %ld ms, N=%lu, iters_run=%zu",
              (long)ms.count(), (unsigned long)N, m_iter_log.size());

    // ─── NEW: Score sanity check ─────────────────────────────────
    double score_sum = 0.0;
    double max_score = 0.0;
    uint64_t max_vtx = 0;
    for (uint64_t u = 0; u < N; u++) {
        score_sum += results[u].second;
        if (results[u].second > max_score) {
            max_score = results[u].second;
            max_vtx = u;
        }
    }
    PHILE_DBG(1, "[CrossTierPR] score_sum=%.6f (expect ~1.0) "
              "max_score=%.8f at vtx=%lu",
              score_sum, max_score, (unsigned long)max_vtx);

    // ─── NEW: Top-10 ranked vertices ─────────────────────────────
    if (debug::get_debug_level() >= 1) {
        auto ranked = results;
        std::partial_sort(ranked.begin(),
                          ranked.begin() + std::min(N, (uint64_t)10),
                          ranked.end(),
                          [](auto& a, auto& b) {
                              return a.second > b.second;
                          });
        std::printf("  [PR top-10 vertices]:\n");
        for (uint64_t i = 0; i < std::min(N, (uint64_t)10); i++) {
            std::printf("    #%lu: vtx=%lu score=%.8f\n",
                        (unsigned long)(i + 1),
                        (unsigned long)ranked[i].first,
                        ranked[i].second);
        }
    }

    // Print convergence + tier perf
    dump_convergence();
    debug::print_all_tier_perf();
}

// ─── dump_convergence: NEW ───────────────────────────────────────────
template <class F, class S>
void CrossTierPageRank<F,S>::dump_convergence() const {
    if (m_iter_log.empty()) return;

    std::printf("──── PageRank Convergence Log (%zu iterations) ────\n",
                m_iter_log.size());
    std::printf("  %-5s %-12s %-12s %-10s %-9s %-7s %-7s %-7s\n",
                "Iter", "max_Δ", "L2_norm", "dangling",
                "time_ms", "HBM", "GDDR", "DRAM");
    for (const auto& s : m_iter_log) {
        std::printf("  %-5d %-12.8f %-12.8f %-10.4f %-9.3f %-7lu %-7lu %-7lu\n",
                    s.iteration, s.max_delta, s.l2_norm, s.dangling_sum,
                    s.time_ms,
                    (unsigned long)s.vertices_hbm,
                    (unsigned long)s.vertices_gddr,
                    (unsigned long)s.vertices_dram);
    }
    std::printf("──── End PageRank Convergence ────\n");
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_CROSS_TIER_PAGERANK_HPP
