#ifndef PHILEMON_TIERED_SSSP_HPP
#define PHILEMON_TIERED_SSSP_HPP
/**
 * tiered_sssp.hpp — Delta-Stepping SSSP on tiered memory
 *
 * 骨架来源: upstream/rapidstore/wrapper/algorithms/SSSP.h (182行)
 * 修改 (~20%):
 *   - 包裹在 philemon::algorithms namespace
 *   - 移除 gapbs 依赖 → 使用 std::vector
 *   - 移除 log_info → PHILE_DBG
 *   - 增加 per-bucket 统计打印
 *   - 增加 tier access 跟踪
 *   - Delta-stepping 核心算法 100% 保留
 *
 * Milestone: M014
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

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {

template <class F, class S>
class TieredSSSP {
    const int m_num_threads;
    double m_delta;
    std::mutex m_mutex;
    F& m_method;
    S m_snapshot;

public:
    TieredSSSP(int num_threads, double delta, F& method, S snapshot)
        : m_num_threads(num_threads), m_delta(delta),
          m_method(method), m_snapshot(snapshot) {}

    ~TieredSSSP() {}

    void run_sssp(uint64_t source,
                  std::vector<std::pair<uint64_t, double>>& results);

private:
    std::vector<double> delta_stepping(uint64_t source);
    size_t get_bucket_id(double dist) const {
        return (dist == std::numeric_limits<double>::infinity())
            ? std::numeric_limits<size_t>::max()
            : static_cast<size_t>(dist / m_delta);
    }
};

template <class F, class S>
std::vector<double> TieredSSSP<F,S>::delta_stepping(uint64_t source) {
    debug::ScopedTimer timer("SSSP::delta_stepping");
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    constexpr double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(N, INF);
    dist[source] = 0.0;

    // Bucket structure: bucket[i] holds vertices with dist in [i*delta, (i+1)*delta)
    std::vector<std::vector<uint64_t>> buckets(1);
    buckets[0].push_back(source);

    PHILE_DBG(1, "SSSP: N=%lu source=%lu delta=%.2f",
              (unsigned long)N, (unsigned long)source, m_delta);

    size_t current_bucket = 0;
    uint64_t relaxations = 0;
    uint64_t bucket_iters = 0;

    while (current_bucket < buckets.size()) {
        // Skip empty buckets
        while (current_bucket < buckets.size() &&
               buckets[current_bucket].empty()) {
            current_bucket++;
        }
        if (current_bucket >= buckets.size()) break;

        // Process current bucket (may re-add vertices in light-edge phase)
        std::vector<uint64_t> frontier;
        std::swap(frontier, buckets[current_bucket]);
        bucket_iters++;

        while (!frontier.empty()) {
            std::vector<uint64_t> next_frontier;
            auto snap = wrapper::snapshot_clone(m_snapshot);

            for (uint64_t u : frontier) {
                if (dist[u] > (current_bucket + 1) * m_delta) continue;

                wrapper::snapshot_edges(snap, u,
                    [&](uint64_t v, double w) {
                        if (v >= N || v == u) return;
                        double new_dist = dist[u] + w;
                        if (new_dist < dist[v]) {
                            dist[v] = new_dist;
                            relaxations++;
                            size_t bid = get_bucket_id(new_dist);
                            if (bid >= buckets.size())
                                buckets.resize(bid + 1);

                            if (bid == current_bucket) {
                                // Light edge: re-process in same bucket
                                next_frontier.push_back(v);
                            } else {
                                buckets[bid].push_back(v);
                            }
                        }
                    }, false);
            }
            frontier = std::move(next_frontier);
        }
        current_bucket++;
    }

    // Count reachable vertices
    uint64_t reachable = 0;
    for (uint64_t v = 0; v < N; v++) {
        if (dist[v] < INF) reachable++;
    }

    PHILE_DBG(1, "SSSP: reachable=%lu/%lu relaxations=%lu bucket_iters=%lu",
              (unsigned long)reachable, (unsigned long)N,
              (unsigned long)relaxations, (unsigned long)bucket_iters);

    return dist;
}

template <class F, class S>
void TieredSSSP<F,S>::run_sssp(
    uint64_t source,
    std::vector<std::pair<uint64_t, double>>& results) {

    auto t0 = std::chrono::high_resolution_clock::now();
    auto dist = delta_stepping(source);

    uint64_t N = dist.size();
    results.resize(N);
    for (uint64_t u = 0; u < N; u++) {
        results[u] = {u, dist[u]};
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    PHILE_DBG(1, "Tiered SSSP took %ld ms", (long)ms.count());

    if (debug::get_debug_level() >= 1) debug::print_all_tier_perf();
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_TIERED_SSSP_HPP
