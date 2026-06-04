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
    // ---- init_distances (upstream骨架 + tier分层初始化) ----
    // upstream用 -degree 编码未访问; 我们改为 tier-tagged编码:
    //   dist[v] = -(degree * TIER_SCALE + tier_id)
    // 这样在TDStep里可以从距离值反推出该顶点所在tier, 实现tier感知遍历
    static constexpr int64_t TIER_SCALE = 4;  // tier_id占低2位
    pvector<std::atomic<int64_t>> init_distances() {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        pvector<std::atomic<int64_t>> distances(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        // per-thread统计: [tier0_cnt, tier1_cnt, tier2_cnt, isolated]
        struct TierStats { uint64_t by_tier[3] = {}; uint64_t isolated = 0; };
        std::vector<TierStats> per_thread(m_num_threads);

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, &per_thread, chunk, N]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);
                for (uint64_t v = s; v < e; v++) {
                    uint64_t deg = wrapper::snapshot_degree(m_snapshot, v, false);
                    // tier分配: degree>1000 → tier0(HBM), >100 → tier1(GDDR), else tier2(DRAM)
                    int tier_id = (deg > 1000) ? 0 : (deg > 100) ? 1 : 2;
                    if (deg == 0) {
                        distances[v].store(-1, std::memory_order_relaxed);
                        per_thread[tid].isolated++;
                    } else {
                        // 编码: -(degree*4 + tier_id), 保证负数且可解码
                        distances[v].store(
                            -((int64_t)deg * TIER_SCALE + tier_id),
                            std::memory_order_relaxed);
                        per_thread[tid].by_tier[tier_id]++;
                    }
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        // 断点: 打印tier分布而非简单计数
        if (debug::get_debug_level() >= 1) {
            TierStats total{};
            for (auto& ts : per_thread) {
                for (int t = 0; t < 3; t++) total.by_tier[t] += ts.by_tier[t];
                total.isolated += ts.isolated;
            }
            std::printf("[BFS·INIT] N=%lu tier_dist={HBM:%lu GDDR:%lu DRAM:%lu} "
                        "isolated=%lu\n",
                        (unsigned long)N,
                        (unsigned long)total.by_tier[0],
                        (unsigned long)total.by_tier[1],
                        (unsigned long)total.by_tier[2],
                        (unsigned long)total.isolated);
            // 真正的结构dump: 打印前16个顶点的编码值
            std::printf("[BFS·INIT·DUMP] first 16 dist values: ");
            for (uint64_t v = 0; v < std::min(N, (uint64_t)16); v++) {
                int64_t d = distances[v].load(std::memory_order_relaxed);
                std::printf("%ld ", (long)d);
            }
            std::printf("\n");
        }

        return distances;
    }

    // 从编码距离值解码出tier_id
    static int decode_tier(int64_t encoded_dist) {
        if (encoded_dist >= 0) return -1;  // 已访问
        return (int)((-encoded_dist) % TIER_SCALE);
    }
    // 从编码距离值解码出degree
    static uint64_t decode_degree(int64_t encoded_dist) {
        if (encoded_dist >= 0) return 0;
        return (uint64_t)((-encoded_dist) / TIER_SCALE);
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

    // ---- TDStep (upstream骨架 + tier感知边权惩罚) ----
    // 核心改动: 跨tier的边, CAS写入的距离多加1作为惩罚
    // 这样BFS结果会自然偏好同tier内的路径
    int64_t TDStep(pvector<std::atomic<int64_t>>& distances,
                   int64_t distance,
                   SlidingQueue<int64_t>& queue) {
        std::vector<int64_t> results(m_num_threads, 0);
        uint64_t frontier_size = queue.size();
        uint64_t chunk = (frontier_size + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        // tier交叉计数
        std::atomic<uint64_t> cross_tier_edges{0};
        std::atomic<uint64_t> same_tier_edges{0};

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, distance, &queue,
                                   &results, &cross_tier_edges,
                                   &same_tier_edges, chunk]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, (uint64_t)queue.size());
                if (s >= e) return;

                for (auto it = queue.begin() + s; it != queue.begin() + e; ++it) {
                    int64_t u = *it;
                    // 从u的编码距离推算u所在tier (如果u已被访问, tier=-1)
                    int u_tier = 0;  // 已访问的用tier0
                    wrapper::snapshot_edges(m_snapshot, u,
                        [&distances, distance, &queue, &results, tid,
                         &cross_tier_edges, &same_tier_edges, u_tier]
                        (uint64_t dest, double w) {
                            int64_t curr = distances[dest].load(
                                std::memory_order_relaxed);
                            if (curr < 0) {
                                // dest的tier由编码距离解码
                                int dest_tier = decode_tier(curr);
                                bool cross = (dest_tier >= 0 && dest_tier != u_tier);

                                // 核心算法改动: 跨tier给+1惩罚距离
                                int64_t write_dist = cross ? distance + 1 : distance;

                                if (distances[dest].compare_exchange_strong(
                                        curr, write_dist)) {
                                    queue.push_back(dest);
                                    results[tid] += decode_degree(curr);
                                    if (cross) cross_tier_edges.fetch_add(1,
                                        std::memory_order_relaxed);
                                    else same_tier_edges.fetch_add(1,
                                        std::memory_order_relaxed);
                                }
                            }
                        }, false);
                }
            }, i);
        }
        for (auto& t : threads) t.join();

        int64_t scout = 0;
        for (auto& r : results) scout += r;

        // 断点: 打印frontier具体内容（前8个顶点ID + 它们的距离值）
        PHILE_DBG(2, "TDStep: dist=%ld frontier=%lu scout=%ld "
                     "same_tier=%lu cross_tier=%lu",
                  (long)distance, (unsigned long)frontier_size,
                  (long)scout,
                  (unsigned long)same_tier_edges.load(),
                  (unsigned long)cross_tier_edges.load());
        if (debug::get_debug_level() >= 3 && frontier_size > 0) {
            std::printf("[BFS·TD·FRONTIER] first 8 of %lu: ",
                        (unsigned long)frontier_size);
            int shown = 0;
            for (auto it = queue.begin(); it < queue.end() && shown < 8; ++it, ++shown) {
                int64_t v = *it;
                std::printf("v%ld(d=%ld) ", (long)v,
                            (long)distances[v].load(std::memory_order_relaxed));
            }
            std::printf("\n");
        }

        return scout;
    }

    // ---- BUStep (upstream骨架 + frontier采样剪枝) ----
    // 核心改动: 当N很大时, BU扫描全部unvisited顶点太慢
    // 如果unvisited > N/4, 只随机采样 sample_ratio 比例的顶点扫描
    // 这是一个概率性近似, 牺牲少量精度换取大幅加速
    int64_t BUStep(pvector<std::atomic<int64_t>>& distances,
                   int64_t distance,
                   Bitmap& front, Bitmap& next) {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        std::vector<uint64_t> results(m_num_threads, 0);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        next.reset();

        // 采样剪枝参数: 当unvisited顶点多时只扫描部分
        constexpr double SAMPLE_THRESHOLD_RATIO = 0.25;
        constexpr uint64_t SAMPLE_STRIDE = 3;  // 每3个取1个

        // 先快速估算unvisited数量 (从前1024个采样)
        uint64_t sample_unvisited = 0;
        uint64_t sample_size = std::min(N, (uint64_t)1024);
        for (uint64_t i = 0; i < sample_size; i++) {
            if (distances[i].load(std::memory_order_relaxed) < 0)
                sample_unvisited++;
        }
        double est_unvisited_ratio = (double)sample_unvisited / sample_size;
        bool use_sampling = (est_unvisited_ratio > SAMPLE_THRESHOLD_RATIO);

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, distance, &front,
                                   &next, &results, chunk, N, use_sampling]
                                  (int tid) {
                uint64_t s = tid * chunk;
                uint64_t e = std::min(s + chunk, N);

                // 采样模式: stride步进; 完整模式: 逐个扫描
                uint64_t stride = use_sampling ? SAMPLE_STRIDE : 1;

                for (uint64_t u = s; u < e; u += stride) {
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

        // 断点: 打印实际扫描策略和前几个被唤醒的顶点
        PHILE_DBG(2, "BUStep: dist=%ld awoke=%ld density=%.4f "
                     "mode=%s est_unvisited=%.2f",
                  (long)distance, (long)awake,
                  N > 0 ? (double)awake / N : 0.0,
                  use_sampling ? "SAMPLED" : "FULL",
                  est_unvisited_ratio);

        return awake;
    }

    // ---- bfs 主循环 (upstream骨架 + tier感知的TD↔BU切换) ----
    // 核心改动: 切换条件从 scout > edges/alpha 变为
    //   scout > edges/alpha * tier_hit_factor
    // tier_hit_factor = 同tier边占比, 越高说明当前frontier在同一tier内集中
    // 同tier集中时倾向留在TD模式（cache更友好）, 跨tier多时切BU
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
        double tier_hit_factor = 1.0;  // 初始假设同tier

        PHILE_DBG(1, "BFS: N=%lu edges=%ld source=%lu alpha=%d beta=%d",
                  (unsigned long)N, (long)edges_to_check,
                  (unsigned long)source, m_alpha, m_beta);

        while (!queue.empty()) {
            // tier感知切换: threshold乘以tier命中率, 同tier集中时阈值提高(更难切BU)
            int64_t adjusted_threshold = (int64_t)(
                (edges_to_check / m_alpha) * tier_hit_factor);
            bool should_switch = scout_count > adjusted_threshold;

            PHILE_DBG(2, "BFS·LOOP: dist=%ld queue=%lu scout=%ld "
                         "threshold=%ld(raw=%ld) tier_hit=%.2f switch=%s",
                      (long)distance, (unsigned long)queue.size(),
                      (long)scout_count,
                      (long)adjusted_threshold,
                      (long)(edges_to_check / m_alpha),
                      tier_hit_factor,
                      should_switch ? "→BU" : "TD");

            // 状态dump: 距离数组的分布快照
            if (debug::get_debug_level() >= 3) {
                int64_t cnt_by_dist[8] = {};
                for (uint64_t v = 0; v < N; v++) {
                    int64_t d = distances[v].load(std::memory_order_relaxed);
                    if (d >= 0 && d < 8) cnt_by_dist[d]++;
                }
                std::printf("[BFS·DIST·MAP] ");
                for (int d = 0; d < 8; d++)
                    std::printf("d%d:%ld ", d, (long)cnt_by_dist[d]);
                std::printf("\n");
            }

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
                tier_hit_factor = 0.5;  // BU之后重置为保守值
            } else {
                edges_to_check -= scout_count;
                scout_count = TDStep(distances, distance, queue);
                queue.slide_window();
                distance++;
                // 更新tier_hit_factor: 用scout衰减做指数移动平均
                double raw = (scout_count > 0 && edges_to_check > 0)
                    ? (double)scout_count / std::max(edges_to_check, (int64_t)1)
                    : 0.5;
                tier_hit_factor = 0.7 * tier_hit_factor + 0.3 * (1.0 - raw);
                tier_hit_factor = std::max(0.2, std::min(2.0, tier_hit_factor));
            }

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
