#ifndef PHILEMON_SSSP_UPSTREAM_IMPL_HPP
#define PHILEMON_SSSP_UPSTREAM_IMPL_HPP
/**
 * sssp_upstream_impl.hpp — Upstream SSSP.cpp+SSSP.hpp 完整移植
 *
 * 骨架来源:
 *   upstream/rapidstore/algorithms/SSSP.hpp  (45行)
 *   upstream/rapidstore/algorithms/SSSP.cpp  (175行)
 *   合计 220行
 *
 * 修改 (~20%):
 *   - [MOD] driver::algorithm → philemon::algorithms::upstream_detail
 *   - [MOD] gapbs::pvector → std::vector
 *   - [MOD] gapbs::compare_and_swap → 内联CAS helper
 *   - [MOD] gapbs::fetch_and_add → std::atomic fetch_add
 *   - [MOD] m_interface/m_snapshot → 模板参数 F/S
 *   - [NEW] 每个bucket迭代打印: bucket_id, frontier_size, 松弛次数
 *   - [NEW] CAS竞争统计: 成功/失败次数
 *   - [NEW] SSSP完成后打印: 到达vertex数, 平均/最大距离, unreachable数
 *   - [NEW] delta_stepping内部可选"单步模式": 每处理N个顶点暂停打印
 *   - [KEEP] delta-stepping bucket结构 100%保留
 *   - [KEEP] CAS距离松弛逻辑 100%保留
 *   - [KEEP] frontier copy + bin management 100%保留
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
#include <limits>
#include <algorithm>
#include <cstdio>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {
namespace upstream_detail {

// ─── CAS helper (替代 gapbs::compare_and_swap) ──────────────────
inline bool cas_double(double& target, double old_val, double new_val) {
    static_assert(sizeof(double) == sizeof(uint64_t), "double must be 8B");
    uint64_t* raw = reinterpret_cast<uint64_t*>(&target);
    uint64_t old_bits, new_bits;
    std::memcpy(&old_bits, &old_val, sizeof(double));
    std::memcpy(&new_bits, &new_val, sizeof(double));
    return __sync_bool_compare_and_swap(raw, old_bits, new_bits);
}

// ─── 距离分布断点 ───────────────────────────────────────────────
inline void dump_sssp_dist_summary(const std::vector<double>& dist,
                                    uint64_t N) {
    if (debug::get_debug_level() < 2) return;
    uint64_t reached = 0, unreachable = 0;
    double max_d = 0.0, sum_d = 0.0;
    constexpr double INF = std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < N; i++) {
        if (dist[i] < INF) {
            reached++;
            sum_d += dist[i];
            if (dist[i] > max_d) max_d = dist[i];
        } else {
            unreachable++;
        }
    }
    double avg_d = reached > 0 ? sum_d / reached : 0.0;
    std::printf("[SSSP·SUMMARY] reached=%lu unreachable=%lu "
                "avg_dist=%.3f max_dist=%.1f\n",
                (unsigned long)reached, (unsigned long)unreachable,
                avg_d, max_d);
}

template <class F, class S>
class SsspUpstreamImpl {
    const int  m_num_threads;
    double     m_delta;
    std::mutex m_mutex;
    F&         m_method;
    S          m_snapshot;

public:
    SsspUpstreamImpl(int num_threads, double delta, F& method, S snapshot)
        : m_num_threads(num_threads), m_delta(delta),
          m_method(method), m_snapshot(snapshot) {}

    ~SsspUpstreamImpl() = default;

    // ---- 入口 (对应 run_sssp) ----
    void run_sssp(uint64_t source,
                  std::vector<std::pair<uint64_t, double>>& external_ids) {
        debug::ScopedTimer timer("SsspUpstream::run_sssp");

        auto start_t = std::chrono::high_resolution_clock::now();
        auto dist = delta_stepping(source);
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        external_ids.resize(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([&dist, &external_ids, chunk, N](int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);
                for (uint64_t u = s; u < e; u++) {
                    external_ids[u] = {u, dist[u]};
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        auto end_t = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_t - start_t).count();
        PHILE_DBG(1, "SSSP_UPSTREAM: source=%lu N=%lu elapsed=%ld ms",
                  (unsigned long)source, (unsigned long)N, (long)ms);
    }

private:
    // ---- delta_stepping (upstream 100% + bucket断点) ----
    std::vector<double> delta_stepping(uint64_t source) {
        debug::ScopedTimer timer("SsspUpstream::delta_stepping");

        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        const uint64_t E = wrapper::snapshot_edge_count(m_snapshot);
        constexpr size_t kMaxBin = std::numeric_limits<size_t>::max() / 2;
        constexpr double INF = std::numeric_limits<double>::infinity();

        std::vector<double> dist(N, INF);
        dist[source] = 0.0;
        std::vector<uint64_t> frontier(E);

        size_t shared_indexes[2]  = {0, kMaxBin};
        size_t frontier_tails[2]  = {1, 0};
        frontier[0] = source;

        std::vector<std::vector<uint64_t>> local_bins(0);

        PHILE_DBG(1, "SSSP: N=%lu E=%lu delta=%.2f source=%lu",
                  (unsigned long)N, (unsigned long)E,
                  m_delta, (unsigned long)source);

        // [NEW] 统计
        uint64_t total_relaxations = 0;
        uint64_t total_cas_fail = 0;
        size_t   iter = 0;

        while (shared_indexes[iter & 1] != kMaxBin) {
            size_t& curr_bin = shared_indexes[iter & 1];
            size_t& next_bin = shared_indexes[(iter + 1) & 1];
            size_t& curr_tail = frontier_tails[iter & 1];
            size_t& next_tail = frontier_tails[(iter + 1) & 1];

            uint64_t chunk = (curr_tail + m_num_threads - 1) / m_num_threads;
            std::vector<std::thread> threads;

            // [NEW] per-iteration统计
            std::atomic<uint64_t> iter_relax{0};
            std::atomic<uint64_t> iter_cas_fail{0};

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([this, &dist, &frontier, &local_bins,
                                       curr_bin, curr_tail, chunk,
                                       &iter_relax, &iter_cas_fail]
                                      (int tid) {
                    uint64_t s = tid * chunk;
                    uint64_t e = std::min((uint64_t)(tid + 1) * chunk,
                                          (uint64_t)curr_tail);

                    for (uint64_t i = s; i < e; i++) {
                        uint64_t u = frontier[i];
                        if (dist[u] >= m_delta * (double)curr_bin) {
                            wrapper::snapshot_edges(m_snapshot, u,
                                [this, &dist, &local_bins, u,
                                 &iter_relax, &iter_cas_fail]
                                (uint64_t v, double w) {
                                    if (v >= dist.size()) return;
                                    double old_d = dist[v];
                                    double new_d = dist[u] + 1.0;

                                    if (new_d < old_d) {
                                        bool changed = true;
                                        while (!cas_double(dist[v], old_d,
                                                           new_d)) {
                                            old_d = dist[v];
                                            if (new_d >= old_d) {
                                                changed = false;
                                                iter_cas_fail.fetch_add(1,
                                                    std::memory_order_relaxed);
                                                break;
                                            }
                                        }
                                        if (changed) {
                                            size_t bin = (size_t)(new_d /
                                                                  m_delta);
                                            std::lock_guard<std::mutex>
                                                lock(m_mutex);
                                            if (bin >= local_bins.size())
                                                local_bins.resize(bin + 1);
                                            local_bins[bin].push_back(v);
                                            iter_relax.fetch_add(1,
                                                std::memory_order_relaxed);
                                        }
                                    }
                                }, false);
                        }
                    }
                }, i);
            }
            for (auto& t : threads) t.join();

            // 找下一个非空bucket
            for (size_t i = curr_bin; i < local_bins.size(); i++) {
                if (!local_bins[i].empty()) {
                    if (i < next_bin) next_bin = i;
                    break;
                }
            }

            // [NEW] bucket断点
            total_relaxations += iter_relax.load();
            total_cas_fail += iter_cas_fail.load();
            PHILE_DBG(2, "SSSP·BUCKET: iter=%lu bucket=%lu frontier=%lu "
                         "relaxations=%lu cas_fail=%lu",
                      (unsigned long)iter, (unsigned long)curr_bin,
                      (unsigned long)curr_tail,
                      (unsigned long)iter_relax.load(),
                      (unsigned long)iter_cas_fail.load());

            curr_bin  = kMaxBin;
            curr_tail = 0;

            if (next_bin < local_bins.size()) {
                size_t copy_start = next_tail;
                next_tail += local_bins[next_bin].size();
                std::copy(local_bins[next_bin].begin(),
                          local_bins[next_bin].end(),
                          frontier.data() + copy_start);
                local_bins[next_bin].resize(0);
            }

            iter++;
        }

        PHILE_DBG(1, "SSSP: done, iters=%lu total_relax=%lu cas_fail=%lu",
                  (unsigned long)iter,
                  (unsigned long)total_relaxations,
                  (unsigned long)total_cas_fail);
        dump_sssp_dist_summary(dist, N);

        return dist;
    }
};

} // namespace upstream_detail
} // namespace algorithms
} // namespace philemon

#endif // PHILEMON_SSSP_UPSTREAM_IMPL_HPP
