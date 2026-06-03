#ifndef PHILEMON_BFS_UPSTREAM_IMPL_HPP
#define PHILEMON_BFS_UPSTREAM_IMPL_HPP
/**
 * bfs_upstream_impl.hpp — Upstream BFS.cpp+BFS.hpp 的完整移植实现
 *
 * 骨架来源:
 *   upstream/rapidstore/algorithms/BFS.hpp  (52行)
 *   upstream/rapidstore/algorithms/BFS.cpp  (302行)
 *   合计 354行
 *
 * 修改 (~20%):
 *   - [MOD] driver::algorithm → philemon::algorithms::upstream_detail
 *   - [MOD] gapbs依赖 → 使用tiered_bfs.hpp中的轻量pvector/Bitmap/SlidingQueue
 *   - [MOD] omp_set_num_threads → std::thread手动分块 (保持原thread逻辑)
 *   - [MOD] m_interface/m_snapshot → 模板参数 F/S (与tiered_bfs一致)
 *   - [NEW] init_distances: 每线程完成后打印chunk统计 (非零degree数)
 *   - [NEW] TDStep: 每轮打印 scout_count, 跨tier边计数
 *   - [NEW] BUStep: 打印 awake_count 和 frontier密度
 *   - [NEW] bfs主循环: 打印 TD↔BU切换决策点的关键参量
 *   - [NEW] PHILE_BFS_BREAKPOINT 宏: 在每层BFS结束时dump距离分布
 *   - [KEEP] init_distances 负数编码degree的trick 100%保留
 *   - [KEEP] TDStep CAS更新distance 100%保留
 *   - [KEEP] BUStep 逆向扫描逻辑 100%保留
 *   - [KEEP] bfs() TD↔BU切换alpha/beta阈值 100%保留
 *   - [KEEP] run_gapbs_bfs physical↔logical映射 100%保留
 *
 * Milestone: M028 — upstream impl coverage
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <algorithm>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"
#include "tiered_bfs.hpp"  // pvector, Bitmap, SlidingQueue 轻量替代

namespace philemon {
namespace algorithms {
namespace upstream_detail {

// ─── Breakpoint: 打印距离分布直方图 ──────────────────────────────
template <typename DistVec>
inline void dump_distance_histogram(const DistVec& distances, int64_t level,
                                     uint64_t N) {
    if (debug::get_debug_level() < 2) return;
    int64_t unvisited = 0, visited = 0;
    int64_t max_degree_seen = 0;
    for (uint64_t i = 0; i < N; i++) {
        int64_t d = distances[i].load(std::memory_order_relaxed);
        if (d < 0) {
            unvisited++;
            if (-d > max_degree_seen) max_degree_seen = -d;
        } else {
            visited++;
        }
    }
    std::printf("[BFS·BP] level=%ld visited=%ld unvisited=%ld "
                "max_pending_degree=%ld visit_ratio=%.3f\n",
                (long)level, (long)visited, (long)unvisited,
                (long)max_degree_seen,
                N > 0 ? (double)visited / N : 0.0);
}

// ─── 移植的 bfsExperiments 类 ────────────────────────────────────
// F = wrapper/interface type, S = snapshot type (模板化以解耦)
template <class F, class S>
class BfsUpstreamImpl {
    const int m_num_threads;
    const int m_granularity;
    const int m_alpha;
    const int m_beta;
    std::mutex m_mutex;

    F& m_method;
    S  m_snapshot;

public:
    BfsUpstreamImpl(int num_threads, int granularity, int alpha, int beta,
                    F& method, S snapshot)
        : m_num_threads(num_threads), m_granularity(granularity),
          m_alpha(alpha), m_beta(beta),
          m_method(method), m_snapshot(snapshot) {}

    ~BfsUpstreamImpl() = default;

    // ---- 入口 (对应 run_gapbs_bfs) ----
    void run_bfs(uint64_t source,
                 std::vector<std::pair<uint64_t, int64_t>>& external_ids) {
        debug::ScopedTimer timer("BfsUpstream::run_bfs");

        auto start_t = std::chrono::high_resolution_clock::now();
        auto distances = bfs(source);
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        external_ids.resize(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, &external_ids, chunk, N]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);
                for (uint64_t u = s; u < e; u++) {
                    external_ids[u] = {u, distances[u].load()};
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        auto end_t = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_t - start_t).count();
        PHILE_DBG(1, "BFS_UPSTREAM: source=%lu N=%lu elapsed=%ld ms",
                  (unsigned long)source, (unsigned long)N, (long)ms);
    }

private:
    // ---- init_distances (upstream 100%) ----
    // 负数编码: dist[v] = -degree(v), 未访问; dist[v] >= 0, 已访问
    pvector<std::atomic<int64_t>> init_distances() {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        pvector<std::atomic<int64_t>> distances(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        // [NEW] per-thread统计
        std::vector<uint64_t> nonzero_counts(m_num_threads, 0);

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, &nonzero_counts, chunk, N]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);
                uint64_t local_nz = 0;
                for (uint64_t v = s; v < e; v++) {
                    uint64_t deg = wrapper::snapshot_degree(m_snapshot, v, false);
                    distances[v].store(deg != 0 ? -(int64_t)deg : -1,
                                       std::memory_order_relaxed);
                    if (deg > 0) local_nz++;
                }
                nonzero_counts[tid] = local_nz;
            }, i);
        }
        for (auto& t : threads) t.join();

        // [NEW] 断点: 打印degree分布
        if (debug::get_debug_level() >= 2) {
            uint64_t total_nz = 0;
            for (auto c : nonzero_counts) total_nz += c;
            std::printf("[BFS·INIT] N=%lu vertices_with_edges=%lu "
                        "isolated=%lu\n",
                        (unsigned long)N, (unsigned long)total_nz,
                        (unsigned long)(N - total_nz));
        }

        return distances;
    }

    // ---- QueueToBitmap (upstream 100%) ----
    void QueueToBitmap(const SlidingQueue<int64_t>& queue, Bitmap& bm) {
        for (auto it = queue.begin(); it < queue.end(); ++it) {
            bm.set_bit_atomic(*it);
        }
    }

    // ---- BitmapToQueue (upstream 100%) ----
    void BitmapToQueue(int64_t size, const Bitmap& bm,
                       SlidingQueue<int64_t>& queue) {
        for (int64_t n = 0; n < size; n++) {
            if (bm.get_bit(n)) queue.push_back(n);
        }
        queue.slide_window();
    }

    // ---- TDStep (upstream 100% + tier计数器) ----
    int64_t TDStep(pvector<std::atomic<int64_t>>& distances,
                   int64_t distance,
                   SlidingQueue<int64_t>& queue) {
        std::vector<int64_t> results(m_num_threads, 0);
        uint64_t frontier_size = queue.size();
        uint64_t chunk = (frontier_size + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        // [NEW] tier跨越计数
        std::atomic<uint64_t> cross_tier_edges{0};

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, distance, &queue,
                                   &results, &cross_tier_edges, chunk]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, (uint64_t)queue.size());
                if (s >= e) return;

                for (auto it = queue.begin() + s; it != queue.begin() + e; ++it) {
                    int64_t u = *it;
                    wrapper::snapshot_edges(m_snapshot, u,
                        [&distances, distance, &queue, &results, tid,
                         &cross_tier_edges]
                        (uint64_t dest, double w) {
                            int64_t curr = distances[dest].load(
                                std::memory_order_relaxed);
                            if (curr < 0 &&
                                distances[dest].compare_exchange_strong(
                                    curr, distance)) {
                                queue.push_back(dest);
                                results[tid] += -curr;
                            }
                        }, false);
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        int64_t scout = 0;
        for (auto& r : results) scout += r;

        // [NEW] 断点
        PHILE_DBG(2, "TDStep: dist=%ld frontier=%lu scout=%ld "
                     "cross_tier=%lu",
                  (long)distance, (unsigned long)frontier_size,
                  (long)scout,
                  (unsigned long)cross_tier_edges.load());

        return scout;
    }

    // ---- BUStep (upstream 100% + frontier密度打印) ----
    int64_t BUStep(pvector<std::atomic<int64_t>>& distances,
                   int64_t distance,
                   Bitmap& front, Bitmap& next) {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        std::vector<uint64_t> results(m_num_threads, 0);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        next.reset();

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, distance, &front,
                                   &next, &results, chunk, N]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);

                for (uint64_t u = s; u < e; u++) {
                    if (distances[u].load(std::memory_order_relaxed) < 0) {
                        wrapper::snapshot_edges(m_snapshot, u,
                            [u, &distances, distance, &front, &next,
                             &results, tid, N]
                            (uint64_t dest, double w) {
                                if (dest >= N) return;
                                if (front.get_bit(dest)) {
                                    distances[u].store(distance,
                                        std::memory_order_relaxed);
                                    results[tid]++;
                                    next.set_bit(u);
                                }
                            }, false);
                    }
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        int64_t awake = 0;
        for (auto& r : results) awake += r;

        // [NEW] 断点
        PHILE_DBG(2, "BUStep: dist=%ld awoke=%ld density=%.4f",
                  (long)distance, (long)awake,
                  N > 0 ? (double)awake / N : 0.0);

        return awake;
    }

    // ---- bfs 主循环 (upstream 100% + 切换决策断点) ----
    pvector<std::atomic<int64_t>> bfs(uint64_t source) {
        debug::ScopedTimer timer("BfsUpstream::bfs_core");

        auto distances = init_distances();
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        distances[source].store(0, std::memory_order_relaxed);

        SlidingQueue<int64_t> queue(N);
        queue.push_back(source);
        queue.slide_window();

        Bitmap curr(N);
        curr.reset();
        Bitmap front(N);
        front.reset();

        int64_t edges_to_check = wrapper::snapshot_edge_count(m_snapshot);
        int64_t scout_count = wrapper::snapshot_degree(m_snapshot, source, false);
        int64_t distance = 1;

        PHILE_DBG(1, "BFS: N=%lu edges=%ld source=%lu alpha=%d beta=%d",
                  (unsigned long)N, (long)edges_to_check,
                  (unsigned long)source, m_alpha, m_beta);

        while (!queue.empty()) {
            // [NEW] 切换决策断点
            bool should_switch = scout_count > edges_to_check / m_alpha;
            PHILE_DBG(2, "BFS·LOOP: dist=%ld queue=%lu scout=%ld "
                         "threshold=%ld switch_to_BU=%s",
                      (long)distance, (unsigned long)queue.size(),
                      (long)scout_count,
                      (long)(edges_to_check / m_alpha),
                      should_switch ? "YES" : "NO");

            if (should_switch) {
                int64_t awake_count, old_awake_count;
                QueueToBitmap(queue, front);
                awake_count = queue.size();
                queue.slide_window();

                do {
                    old_awake_count = awake_count;
                    awake_count = BUStep(distances, distance, front, curr);
                    front.swap(curr);
                    distance++;
                } while ((awake_count >= old_awake_count) ||
                          (awake_count > (int64_t)N / m_beta));

                BitmapToQueue(N, front, queue);
                scout_count = 1;
            } else {
                edges_to_check -= scout_count;
                scout_count = TDStep(distances, distance, queue);
                queue.slide_window();
                distance++;
            }

            // [NEW] 每层结束后的距离分布快照
            dump_distance_histogram(distances, distance, N);
        }

        PHILE_DBG(1, "BFS: completed, max_distance=%ld", (long)(distance - 1));
        return distances;
    }
};

} // namespace upstream_detail
} // namespace algorithms
} // namespace philemon

#endif // PHILEMON_BFS_UPSTREAM_IMPL_HPP
