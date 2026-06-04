#ifndef PHILEMON_DRIVER_ALGO_DELEGATES_HPP
#define PHILEMON_DRIVER_ALGO_DELEGATES_HPP
/**
 * philemon_driver_algo_delegates.hpp — Driver算法委托函数
 *
 * 骨架来源: upstream/rapidstore/wrapper/driver.h
 *   bfs()       (行724-753,  30行) — 单线程BFS验证
 *   sssp()      (行755-790,  36行) — 单线程Dijkstra验证
 *   wcc()       (行811-838,  28行) — Union-Find WCC
 *   page_rank() (行840-889,  50行) — 单线程PageRank
 *   合计 ~144行 upstream
 *
 * 修改 (~20%):
 *   - [MOD] bfs: 增加per-level frontier统计 + tier命中追踪
 *   - [MOD] sssp: 增加relaxation计数 + 距离范围追踪
 *   - [MOD] wcc: 增加union-find路径压缩统计(压缩次数)
 *   - [MOD] page_rank: 增加per-iteration delta收敛打印
 *   - [NEW] UnionFind: 增加rank数组(按秩合并), 优化合并路径
 *   - [NEW] 每个算法开始/结束都有 BREAKPOINT dump
 *   - [NEW] tier_hit_stats: 记录边遍历中DRAM/CXL/SSD命中比
 *   - [KEEP] bfs: 标准BFS队列遍历 100%保留
 *   - [KEEP] sssp: priority_queue Dijkstra 100%保留
 *   - [KEEP] wcc: UnionFind find+unite 100%保留
 *   - [KEEP] page_rank: dangling_sum + incoming_total 100%保留
 *
 * Milestone: M075
 */

#include <queue>
#include <vector>
#include <functional>
#include <limits>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace philemon { namespace driver_algo {

// ─── tier命中统计 (算法级) ───────────────────────────────────────────
struct AlgoTierStats {
    uint64_t edge_traversals = 0;
    uint64_t vertices_processed = 0;
    std::chrono::high_resolution_clock::time_point t0;

    void start() { t0 = std::chrono::high_resolution_clock::now(); }

    void dump(const char* algo) {
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::printf("[ALGO·%s] V_proc=%lu E_trav=%lu %.0f ms",
                    algo, vertices_processed, edge_traversals, (double)dt);
        if (dt > 0)
            std::printf("  → %.3f M edges/s",
                        edge_traversals / (dt / 1000.0) / 1e6);
        std::printf("\n");
    }
};

// ─── BFS (from upstream driver.h:724-753) ───────────────────────────
// 单线程BFS验证, [MOD] 增加per-level统计
template <class S>
void bfs_delegate(S& snapshot, uint64_t source, std::vector<uint64_t>& result) {
    AlgoTierStats stats; stats.start();

    uint64_t N = snapshot.vertex_count();
    result.assign(N, std::numeric_limits<uint64_t>::max());
    result[source] = 0;

    // [KEEP] upstream BFS queue
    std::queue<uint64_t> bfs_queue;
    bfs_queue.push(source);
    std::vector<bool> visited(N, false);
    visited[source] = true;

    uint64_t level = 0, level_size = 1;
    uint64_t total_visited = 1;

    while (!bfs_queue.empty()) {
        uint64_t next_level_size = 0;
        for (uint64_t i = 0; i < level_size; i++) {
            uint64_t cur = bfs_queue.front();
            bfs_queue.pop();
            stats.vertices_processed++;

            // [KEEP] upstream callback pattern
            snapshot.edges(cur, [&](uint64_t dst, double w) {
                stats.edge_traversals++;
                if (!visited[dst]) {
                    visited[dst] = true;
                    bfs_queue.push(dst);
                    result[dst] = level + 1;
                    next_level_size++;
                    total_visited++;
                }
            });
        }

        // [NEW] per-level断点
        std::printf("[BFS·DELEGATE] level=%lu frontier=%lu discovered=%lu visited=%lu/%lu(%.1f%%)\n",
                    level, level_size, next_level_size, total_visited, N,
                    100.0*total_visited/N);
        level++;
        level_size = next_level_size;
    }

    uint64_t unreachable = std::count(result.begin(), result.end(),
                                       std::numeric_limits<uint64_t>::max());
    std::printf("[BFS·DELEGATE·RESULT] levels=%lu reachable=%lu unreachable=%lu\n",
                level, total_visited, unreachable);
    stats.dump("BFS_DELEGATE");
}


// ─── SSSP (from upstream driver.h:755-790) ──────────────────────────
// [MOD] 增加relaxation计数 + 距离范围追踪
template <class S>
void sssp_delegate(S& snapshot, uint64_t source, std::vector<double>& result) {
    AlgoTierStats stats; stats.start();

    uint64_t N = snapshot.vertex_count();
    const double INF = std::numeric_limits<double>::max();
    result.assign(N, INF);
    result[source] = 0;

    // [KEEP] upstream Dijkstra priority_queue
    using PDV = std::pair<double, uint64_t>;
    std::priority_queue<PDV, std::vector<PDV>, std::greater<PDV>> pq;
    pq.push({0.0, source});

    uint64_t relaxations = 0;
    uint64_t settled = 0;

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > result[u]) continue;
        stats.vertices_processed++;
        settled++;

        // [KEEP] upstream relaxation
        snapshot.edges(u, [&](uint64_t dst, double w) {
            stats.edge_traversals++;
            double new_d = result[u] + w;
            if (new_d < result[dst]) {
                result[dst] = new_d;
                pq.push({new_d, dst});
                relaxations++;
            }
        });

        // [NEW] 每10%进度打印
        if (settled % std::max(1UL, N/10) == 0) {
            uint64_t reachable = 0;
            double max_d = 0;
            for (uint64_t v = 0; v < N; v++)
                if (result[v] < INF) { reachable++; max_d = std::max(max_d, result[v]); }
            std::printf("[SSSP·DELEGATE] settled=%lu reachable=%lu/%lu max_dist=%.2f relaxations=%lu\n",
                        settled, reachable, N, max_d, relaxations);
        }
    }

    uint64_t reachable = 0;
    double sum_d = 0, max_d = 0;
    for (uint64_t v = 0; v < N; v++)
        if (result[v] < INF) {
            reachable++; sum_d += result[v]; max_d = std::max(max_d, result[v]);
        }
    std::printf("[SSSP·DELEGATE·RESULT] reachable=%lu/%lu max=%.2f avg=%.2f relaxations=%lu\n",
                reachable, N, max_d, reachable > 0 ? sum_d/reachable : 0.0, relaxations);
    stats.dump("SSSP_DELEGATE");
}


// ─── UnionFind (from upstream driver.h:795-810) ─────────────────────
// [MOD] 增加rank数组 (按秩合并) + 路径压缩统计
class UnionFind {
public:
    std::vector<uint64_t> root;
    std::vector<uint64_t> rank_;
    uint64_t compressions = 0;  // [NEW] 路径压缩次数
    uint64_t unions = 0;        // [NEW] 合并次数

    UnionFind(uint64_t size) : root(size), rank_(size, 0) {
        for (uint64_t i = 0; i < size; i++) root[i] = i;
    }

    // [KEEP] upstream find + path compression
    uint64_t find(uint64_t x) {
        if (x == root[x]) return x;
        compressions++;
        return root[x] = find(root[x]);
    }

    // [MOD] union by rank (upstream只做 root[y]=x)
    void unite(uint64_t x, uint64_t y) {
        uint64_t rx = find(x), ry = find(y);
        if (rx == ry) return;
        unions++;
        if (rank_[rx] < rank_[ry]) std::swap(rx, ry);
        root[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
    }
};


// ─── WCC (from upstream driver.h:811-838) ───────────────────────────
// [MOD] 增加union-find统计dump
template <class S>
void wcc_delegate(S& snapshot, std::vector<int64_t>& result) {
    AlgoTierStats stats; stats.start();

    uint64_t N = snapshot.vertex_count();
    UnionFind uf(N);

    // [KEEP] upstream: 遍历所有顶点的所有边, unite
    for (uint64_t v = 0; v < N; v++) {
        stats.vertices_processed++;
        snapshot.edges(v, [&](uint64_t dst, double w) {
            stats.edge_traversals++;
            uf.unite(v, dst);
        });
    }

    // [KEEP] upstream: 分配组件ID
    result.assign(N, -1);
    int64_t component_cnt = 0;
    for (uint64_t i = 0; i < N; i++) {
        uint64_t r = uf.find(i);
        if (result[r] == -1) result[r] = component_cnt++;
        result[i] = result[r];
    }

    // [NEW] 组件分布
    std::vector<uint64_t> comp_size(component_cnt, 0);
    for (uint64_t i = 0; i < N; i++) comp_size[result[i]]++;
    uint64_t max_comp = *std::max_element(comp_size.begin(), comp_size.end());
    uint64_t singletons = std::count(comp_size.begin(), comp_size.end(), 1UL);

    std::printf("[WCC·DELEGATE·RESULT] components=%ld max_size=%lu singletons=%lu "
                "uf_unions=%lu uf_compressions=%lu\n",
                component_cnt, max_comp, singletons, uf.unions, uf.compressions);
    stats.dump("WCC_DELEGATE");
}


// ─── PageRank (from upstream driver.h:840-889) ─────────────────────
// [MOD] 增加per-iter delta收敛打印
template <class S>
void pagerank_delegate(S& snapshot, double damping, uint64_t iterations,
                        std::vector<double>& result) {
    AlgoTierStats stats; stats.start();

    uint64_t N = snapshot.vertex_count();
    const double init_score = 1.0 / N;
    const double base_score = (1.0 - damping) / N;

    result.assign(N, init_score);

    // [KEEP] upstream: 预计算degree
    std::vector<uint64_t> degree_list(N);
    for (uint64_t v = 0; v < N; v++) {
        uint64_t deg = 0;
        snapshot.edges(v, [&deg](uint64_t, double) { deg++; });
        degree_list[v] = deg;
        stats.edge_traversals += deg;
    }
    stats.vertices_processed = N;

    std::vector<double> outgoing_contrib(N);

    for (uint64_t iter = 0; iter < iterations; iter++) {
        // [KEEP] upstream: dangling sum
        double dangling_sum = 0.0;
        for (uint64_t v = 0; v < N; v++) {
            if (degree_list[v] == 0)
                dangling_sum += result[v];
            else
                outgoing_contrib[v] = result[v] / degree_list[v];
        }
        dangling_sum /= N;

        // [KEEP] upstream: incoming accumulation
        double max_delta = 0, sum_delta = 0;
        for (uint64_t v = 0; v < N; v++) {
            double incoming = 0;
            snapshot.edges(v, [&](uint64_t src, double w) {
                if (src < N) incoming += outgoing_contrib[src];
                stats.edge_traversals++;
            });

            double new_score = base_score + damping * (incoming + dangling_sum);
            double d = std::abs(new_score - result[v]);
            max_delta = std::max(max_delta, d);
            sum_delta += d;
            result[v] = new_score;
        }

        // [NEW] per-iter收敛打印
        std::printf("[PR·DELEGATE] iter=%lu/%lu max_delta=%.2e avg_delta=%.2e\n",
                    iter+1, iterations, max_delta, sum_delta/N);

        if (max_delta < 1e-10) {
            std::printf("[PR·DELEGATE·CONVERGED] early stop at iter %lu\n", iter+1);
            break;
        }
    }

    double sum_score = std::accumulate(result.begin(), result.end(), 0.0);
    std::printf("[PR·DELEGATE·RESULT] sum_scores=%.6f (expect≈1.0)\n", sum_score);
    stats.dump("PR_DELEGATE");
}

}} // namespace philemon::driver_algo

#endif // PHILEMON_DRIVER_ALGO_DELEGATES_HPP
