#ifndef PHILEMON_WCC_UPSTREAM_IMPL_HPP
#define PHILEMON_WCC_UPSTREAM_IMPL_HPP
/**
 * wcc_upstream_impl.hpp — Upstream WCC.cpp+WCC.hpp 完整移植
 *
 * 骨架来源:
 *   upstream/rapidstore/algorithms/WCC.hpp  (46行)
 *   upstream/rapidstore/algorithms/WCC.cpp  (120行, wcc+run_wcc)
 *   合计 166行
 *
 * 修改 (~20%):
 *   - [MOD] driver::algorithm → philemon::algorithms::upstream_detail
 *   - [MOD] m_interface/m_snapshot → 模板参数 F/S
 *   - [MOD] omp parallel for → 手动std::thread (保留pointer jumping语义)
 *   - [NEW] 每轮label propagation打印: 轮次, 变化顶点数, 最大连通分量大小
 *   - [NEW] pointer jumping打印: 跳转次数
 *   - [NEW] wcc完成后: 分量数目直方图 (top-5大分量)
 *   - [KEEP] label propagation: comp[high] = low 100%保留
 *   - [KEEP] pointer jumping: comp[i] = comp[comp[i]] 100%保留
 *   - [KEEP] convergence检测 100%保留
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
#include <unordered_map>
#include <cstdio>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {
namespace upstream_detail {

// ─── 分量分布断点 ───────────────────────────────────────────────
inline void dump_component_distribution(const uint64_t* comp, uint64_t N) {
    if (debug::get_debug_level() < 2) return;
    std::unordered_map<uint64_t, uint64_t> sizes;
    for (uint64_t i = 0; i < N; i++) sizes[comp[i]]++;

    std::vector<std::pair<uint64_t, uint64_t>> sorted_comp(
        sizes.begin(), sizes.end());
    std::sort(sorted_comp.begin(), sorted_comp.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    std::printf("[WCC·DIST] total_components=%lu\n",
                (unsigned long)sizes.size());
    int show = std::min((int)sorted_comp.size(), 5);
    for (int i = 0; i < show; i++) {
        std::printf("  comp[%lu] size=%lu (%.2f%%)\n",
                    (unsigned long)sorted_comp[i].first,
                    (unsigned long)sorted_comp[i].second,
                    100.0 * sorted_comp[i].second / N);
    }
}

template <class F, class S>
class WccUpstreamImpl {
    const int m_num_threads;
    F&        m_method;
    S         m_snapshot;

public:
    WccUpstreamImpl(int num_threads, F& method, S snapshot)
        : m_num_threads(num_threads),
          m_method(method), m_snapshot(snapshot) {}

    ~WccUpstreamImpl() = default;

    void run_wcc(std::vector<std::pair<uint64_t, int64_t>>& external_ids) {
        debug::ScopedTimer timer("WccUpstream::run_wcc");

        auto start_t = std::chrono::high_resolution_clock::now();
        auto ptr_comp = wcc();
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        external_ids.resize(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([&ptr_comp, &external_ids, chunk, N]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);
                for (uint64_t u = s; u < e; u++) {
                    external_ids[u] = {u, (int64_t)ptr_comp[u]};
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        auto end_t = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_t - start_t).count();
        PHILE_DBG(1, "WCC_UPSTREAM: N=%lu elapsed=%ld ms",
                  (unsigned long)N, (long)ms);
    }

private:
    // ---- wcc 核心 (upstream 100% + 每轮断点) ----
    std::unique_ptr<uint64_t[]> wcc() {
        debug::ScopedTimer timer("WccUpstream::wcc_core");
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        std::unique_ptr<uint64_t[]> ptr_comp(new uint64_t[N]);
        uint64_t* comp = ptr_comp.get();

        // 初始化: 每个顶点是自己的分量
        for (uint64_t i = 0; i < N; i++) comp[i] = i;

        bool change = true;
        uint64_t round = 0;
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;

        PHILE_DBG(1, "WCC: N=%lu threads=%d", (unsigned long)N, m_num_threads);

        while (change) {
            change = false;
            std::vector<std::thread> threads;

            // [NEW] per-thread变化计数
            std::vector<uint64_t> change_counts(m_num_threads, 0);

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([this, &change, comp, N, chunk,
                                       &change_counts]
                                      (int tid) {
                    uint64_t s = chunk * tid;
                    uint64_t e = std::min(s + chunk, N);
                    uint64_t local_changes = 0;

                    for (uint64_t u = s; u < e; u++) {
                        wrapper::snapshot_edges(m_snapshot, u,
                            [comp, &change, u, N, &local_changes]
                            (uint64_t v, double w) {
                                uint64_t cu = comp[u];
                                uint64_t cv = comp[v];
                                if (cu == cv) return;

                                uint64_t hi = std::max(cu, cv);
                                uint64_t lo = std::min(cu, cv);
                                if (hi >= N || lo >= N) return;

                                if (hi == comp[hi]) {
                                    change = true;
                                    comp[hi] = lo;
                                    local_changes++;
                                }
                            });
                    }
                    change_counts[tid] = local_changes;
                }, i);
            }
            for (auto& t : threads) t.join();

            // Pointer jumping (upstream 100%)
            uint64_t jump_count = 0;
            for (uint64_t i = 0; i < N; i++) {
                while (comp[i] != comp[comp[i]]) {
                    comp[i] = comp[comp[i]];
                    jump_count++;
                }
            }

            // [NEW] 每轮断点
            uint64_t total_changes = 0;
            for (auto c : change_counts) total_changes += c;
            round++;
            PHILE_DBG(2, "WCC·ROUND %lu: changes=%lu jumps=%lu "
                         "converged=%s",
                      (unsigned long)round, (unsigned long)total_changes,
                      (unsigned long)jump_count,
                      change ? "no" : "YES");
        }

        PHILE_DBG(1, "WCC: converged in %lu rounds", (unsigned long)round);
        dump_component_distribution(comp, N);

        return ptr_comp;
    }
};

} // namespace upstream_detail
} // namespace algorithms
} // namespace philemon

#endif // PHILEMON_WCC_UPSTREAM_IMPL_HPP
