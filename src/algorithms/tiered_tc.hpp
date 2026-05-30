#ifndef PHILEMON_TIERED_TC_HPP
#define PHILEMON_TIERED_TC_HPP
/**
 * tiered_tc.hpp — Triangle Counting on tiered memory
 *
 * 骨架来源: upstream/rapidstore/wrapper/algorithms/TC.h (93行)
 * 修改 (~20%):
 *   - 包裹在 philemon::algorithms namespace
 *   - 移除 log_info → PHILE_DBG
 *   - 增加 per-tier intersection 统计
 *   - 增加 progress 打印 (每 10% 打印一次)
 *   - Set-intersection triangle counting 核心算法 100% 保留
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
#include <algorithm>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {

template <class F, class S>
class TieredTriangleCounting {
    F& m_method;
    S m_snapshot;

public:
    TieredTriangleCounting(F& method, S snapshot)
        : m_method(method), m_snapshot(snapshot) {}

    ~TieredTriangleCounting() {}

    uint64_t run_tc(int num_threads = 1);
};

template <class F, class S>
uint64_t TieredTriangleCounting<F,S>::run_tc(int num_threads) {
    debug::ScopedTimer timer("TC::run");
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    std::atomic<uint64_t> total_triangles{0};
    std::atomic<uint64_t> total_intersections{0};

    PHILE_DBG(1, "TC: N=%lu threads=%d", (unsigned long)N, num_threads);

    std::vector<std::thread> threads;
    uint64_t chunk = (N + num_threads - 1) / num_threads;

    wrapper::set_max_threads(m_method, num_threads);
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([this, &total_triangles, &total_intersections,
                              chunk, N](int tid) {
            wrapper::init_thread(m_method, tid);
            auto snap = wrapper::snapshot_clone(m_snapshot);
            uint64_t start = tid * chunk;
            uint64_t end = std::min(start + chunk, N);
            uint64_t local_tri = 0;
            uint64_t local_int = 0;
            uint64_t progress_step = std::max((end - start) / 10, (uint64_t)1);

            for (uint64_t u = start; u < end; u++) {
                // Progress tracking
                if ((u - start) % progress_step == 0 && tid == 0) {
                    PHILE_DBG(2, "TC progress: %.0f%%",
                              100.0 * (u - start) / (end - start));
                }

                wrapper::snapshot_edges(snap, u,
                    [&](uint64_t v, double w) {
                        if (v <= u || v >= N) return;
                        uint64_t count = wrapper::snapshot_intersect(snap, u, v);
                        local_tri += count;
                        local_int++;
                    }, false);
            }

            total_triangles.fetch_add(local_tri);
            total_intersections.fetch_add(local_int);
            wrapper::end_thread(m_method, tid);
        }, i);
    }
    for (auto& t : threads) t.join();

    uint64_t triangles = total_triangles.load();

    PHILE_DBG(1, "TC: %lu triangles, %lu intersections performed",
              (unsigned long)triangles,
              (unsigned long)total_intersections.load());

    if (debug::get_debug_level() >= 1) debug::print_all_tier_perf();
    return triangles;
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_TIERED_TC_HPP
