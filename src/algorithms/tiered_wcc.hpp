#ifndef PHILEMON_TIERED_WCC_HPP
#define PHILEMON_TIERED_WCC_HPP
/**
 * tiered_wcc.hpp — Weakly Connected Components on tiered memory
 *
 * 骨架来源: upstream/rapidstore/wrapper/algorithms/WCC.h (149行)
 * 修改 (~20%):
 *   - 包裹在 philemon::algorithms namespace
 *   - 移除 log_info → PHILE_DBG
 *   - 增加 per-iteration component 统计
 *   - 增加 tier 访问跟踪
 *   - Label-propagation 核心算法 100% 保留
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
#include <unordered_set>
#include <algorithm>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {

template <class F, class S>
class TieredWCC {
    const int m_num_threads;
    F& m_method;
    S m_snapshot;

public:
    TieredWCC(int num_threads, F& method, S snapshot)
        : m_num_threads(num_threads), m_method(method),
          m_snapshot(snapshot) {}

    ~TieredWCC() {}

    void run_wcc(std::vector<std::pair<uint64_t, uint64_t>>& results);

private:
    std::vector<uint64_t> wcc();
};

template <class F, class S>
std::vector<uint64_t> TieredWCC<F,S>::wcc() {
    debug::ScopedTimer timer("WCC::compute");
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

    // Initialize: each vertex is its own component
    std::vector<std::atomic<uint64_t>> comp(N);
    for (uint64_t v = 0; v < N; v++) {
        comp[v].store(v);
    }

    PHILE_DBG(1, "WCC: N=%lu threads=%d", (unsigned long)N, m_num_threads);

    bool changed = true;
    uint64_t iteration = 0;

    while (changed) {
        changed = false;
        std::vector<bool> thread_changed(m_num_threads, false);
        std::vector<std::thread> threads;
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;

        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &comp, chunk, N,
                                  &thread_changed](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap = wrapper::snapshot_clone(m_snapshot);
                uint64_t start = tid * chunk;
                uint64_t end = std::min(start + chunk, N);

                for (uint64_t u = start; u < end; u++) {
                    uint64_t my_comp = comp[u].load();
                    wrapper::snapshot_edges(snap, u,
                        [&](uint64_t v, double w) {
                            if (v >= N) return;
                            uint64_t v_comp = comp[v].load();
                            // Link to smaller component ID
                            uint64_t expected = comp[u].load();
                            if (v_comp < expected) {
                                if (comp[u].compare_exchange_strong(
                                        expected, v_comp)) {
                                    thread_changed[tid] = true;
                                }
                            }
                        }, false);
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();
        for (auto& tc : thread_changed) if (tc) changed = true;

        // Pointer jumping: comp[u] = comp[comp[u]]
        for (uint64_t u = 0; u < N; u++) {
            uint64_t c = comp[u].load();
            while (c != comp[c].load()) {
                c = comp[c].load();
            }
            comp[u].store(c);
        }

        iteration++;
        if (debug::get_debug_level() >= 2) {
            // Count distinct components
            std::unordered_set<uint64_t> unique;
            for (uint64_t v = 0; v < N; v++) unique.insert(comp[v].load());
            PHILE_DBG(2, "WCC iter %lu: %zu components",
                      (unsigned long)iteration, unique.size());
        }
    }

    // Convert to non-atomic result
    std::vector<uint64_t> result(N);
    std::unordered_set<uint64_t> unique;
    for (uint64_t v = 0; v < N; v++) {
        result[v] = comp[v].load();
        unique.insert(result[v]);
    }

    PHILE_DBG(1, "WCC: %zu components in %lu iterations",
              unique.size(), (unsigned long)iteration);
    return result;
}

template <class F, class S>
void TieredWCC<F,S>::run_wcc(
    std::vector<std::pair<uint64_t, uint64_t>>& results) {

    auto t0 = std::chrono::high_resolution_clock::now();
    auto comp = wcc();

    results.resize(comp.size());
    for (uint64_t u = 0; u < comp.size(); u++) {
        results[u] = {u, comp[u]};
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    PHILE_DBG(1, "Tiered WCC took %ld ms", (long)ms.count());

    if (debug::get_debug_level() >= 1) debug::print_all_tier_perf();
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_TIERED_WCC_HPP
