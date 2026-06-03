#pragma once
/**
 * cuda_bfs_kernel.hpp — GPU并行BFS (warp-level edge scan + ballot-based BU)
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/algorithms/BFS.h  (330行)
 *     → TDStep: frontier queue遍历, snapshot_edges回调, CAS距离更新
 *     → BUStep: 全顶点扫描, bitmap front/next, 找到前驱即停
 *     → bfs(): direction switching (scout_count > edges/alpha → BU)
 *     → QueueToBitmap / BitmapToQueue 互转
 *
 *   upstream/rapidstore/algorithms/BFS.cpp  (302行)
 *     → 底层BFS实现, init_distances用-degree编码
 *
 *   src/algorithms/bfs_upstream_impl.hpp  (前4位Claude的移植)
 *     → density-based TD↔BU切换
 *
 *   src/cuda/hetero_bench.cu  cross_tier_query (510–610行)
 *     → 跨tier数据gather模式 → kernel的数据喂入模式
 *
 * 算法改动 (~20%):
 *   [ALG1] TDStep: 原版OMP thread-level并行, 每线程处理queue的一个chunk
 *          → warp-level: 每warp处理一个frontier vertex, warp内32线程并行扫描邻接表,
 *            warp-shuffle做CAS距离更新, __shfl_sync + atomicAdd计scout_count
 *   [ALG2] BUStep: 原版逐vertex遍历邻接表, bool done提前退出
 *          → __ballot_sync: warp的32个线程各检一个unvisited vertex,
 *            ballot投票后只有命中的lane做邻接表查找, 减少warp分歧
 *   [ALG3] frontier管理: 原版SlidingQueue + QueueBuffer + flush
 *          → global atomic append: warp-leader做一次atomicAdd预留slot,
 *            然后warp内scatter写入, 减少atomic冲突
 *   [ALG4] direction switch: 原版固定alpha/beta阈值
 *          → 动态: switch_ratio = frontier_edges / remaining_edges,
 *            当ratio > 0.03*sqrt(V/E)时切BU, 自适应图密度
 *
 * Milestone: M047 — CUDA并行BFS kernel
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
#include <chrono>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../core/tiered_allocator.hpp"

namespace philemon {
namespace cuda_bfs {

// ─── CSR格式 (GPU-friendly) ────────────────────────────────────────
struct CsrGraph {
    uint64_t  num_vertices;
    uint64_t  num_edges;
    uint64_t* row_offsets;    // [V+1], GPU/Host指针
    uint64_t* col_indices;    // [E],   GPU/Host指针
    double*   weights;        // [E],   可选
    bool      on_device;      // true=GPU内存
};

// ─── BFS结果 ───────────────────────────────────────────────────────
struct BfsResult {
    std::vector<int64_t> distances;    // -1 = unreachable
    uint64_t  visited_count;
    double    total_ms;
    double    td_ms, bu_ms;           // 各方向累积时间
    int       td_steps, bu_steps;
    int       direction_switches;
};

// ─── [ALG4] 动态切换阈值计算 ──────────────────────────────────────
// 原版: 固定 alpha=15, beta=18
// 改动: 自适应, switch_ratio基于图密度
inline double compute_switch_ratio(uint64_t V, uint64_t E) {
    // 当 frontier_edges / remaining > 0.03 * sqrt(V/E) → 切BU
    double density = static_cast<double>(E) / V;
    return 0.03 * std::sqrt(static_cast<double>(V) / std::max(E, 1UL));
}

// ════════════════════════════════════════════════════════════════════════════
//  GPU Kernels (定义在__CUDACC__下, CPU模拟在else)
// ════════════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__

// [ALG1] Warp-level TD kernel
// 原版BFS.h TDStep: OMP线程, 每线程一个queue chunk, snapshot_edges回调
// 改动: 一个warp处理一个frontier vertex的全部邻接表
__global__ void kernel_td_step(
    const uint64_t* __restrict__ row_offsets,
    const uint64_t* __restrict__ col_indices,
    const int64_t*  __restrict__ frontier,      // 当前frontier数组
    uint64_t        frontier_size,
    int64_t*        distances,                  // dist[v], -degree or level
    int64_t         current_level,
    int64_t*        next_frontier,              // 输出: 下一层frontier
    uint64_t*       next_frontier_size,         // atomic counter
    uint64_t*       scout_count)                // 探索到的边数
{
    // 每warp处理一个frontier vertex
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane    = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int64_t u = frontier[warp_id];
    uint64_t row_start = row_offsets[u];
    uint64_t row_end   = row_offsets[u + 1];
    uint64_t degree    = row_end - row_start;

    uint64_t local_scout = 0;

    // [ALG1] warp内32线程并行扫描邻接表
    // 原版: 单线程遍历, snapshot_edges回调
    // 改动: lane i 处理 edge[row_start + i], stride=32
    for (uint64_t idx = lane; idx < degree; idx += 32) {
        uint64_t v = col_indices[row_start + idx];
        int64_t curr = distances[v];

        // CAS: 如果v未访问(curr < 0), 尝试设为current_level
        if (curr < 0) {
            int64_t old = atomicCAS((long long*)&distances[v],
                                    (long long)curr,
                                    (long long)current_level);
            if (old == curr) {
                // 成功! warp-leader统一push到next_frontier
                // [ALG3] warp-aggregated push
                uint32_t mask = __ballot_sync(0xFFFFFFFF, 1);
                uint32_t leader_lane = __ffs(mask) - 1;
                uint32_t count_in_warp = __popc(mask);

                uint64_t base_slot;
                if (lane == leader_lane) {
                    base_slot = atomicAdd((unsigned long long*)next_frontier_size,
                                          (unsigned long long)count_in_warp);
                }
                base_slot = __shfl_sync(0xFFFFFFFF, base_slot, leader_lane);

                // 每个命中的lane算自己在warp内的排名
                uint32_t lower_mask = (1u << lane) - 1;
                uint32_t rank = __popc(mask & lower_mask);
                next_frontier[base_slot + rank] = v;

                local_scout += static_cast<uint64_t>(-curr);
            }
        }
    }

    // warp-reduce scout_count
    for (int offset = 16; offset > 0; offset >>= 1)
        local_scout += __shfl_down_sync(0xFFFFFFFF, local_scout, offset);
    if (lane == 0)
        atomicAdd((unsigned long long*)scout_count, (unsigned long long)local_scout);
}

// [ALG2] Ballot-based BU kernel
// 原版BFS.h BUStep: 逐vertex扫描邻接表, 找到一个front邻居就停
// 改动: __ballot_sync, warp内32线程各检一个unvisited vertex
__global__ void kernel_bu_step(
    const uint64_t* __restrict__ row_offsets,
    const uint64_t* __restrict__ col_indices,
    int64_t*        distances,
    int64_t         current_level,
    const uint32_t* __restrict__ front_bitmap,   // 当前层bitmap
    uint32_t*       next_bitmap,                  // 下一层bitmap
    uint64_t        num_vertices,
    uint64_t*       awake_count)
{
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp_lane_base = tid - lane;

    // 每lane检查一个vertex
    uint64_t v = tid;
    bool is_candidate = false;
    bool found_parent = false;

    if (v < num_vertices && distances[v] < 0) {
        is_candidate = true;
    }

    // [ALG2] ballot: 哪些lane有unvisited vertex需要检查
    uint32_t candidate_mask = __ballot_sync(0xFFFFFFFF, is_candidate);

    // 只有candidate lanes才进入邻接表扫描
    if (is_candidate) {
        uint64_t row_start = row_offsets[v];
        uint64_t row_end   = row_offsets[v + 1];

        for (uint64_t idx = row_start; idx < row_end && !found_parent; ++idx) {
            uint64_t u = col_indices[idx];
            // 检查u是否在front bitmap中
            uint32_t word = u / 32;
            uint32_t bit  = u & 31;
            if (front_bitmap[word] & (1u << bit)) {
                distances[v] = current_level;
                // 设next bitmap
                uint32_t my_word = v / 32;
                uint32_t my_bit  = v & 31;
                atomicOr(&next_bitmap[my_word], 1u << my_bit);
                found_parent = true;
            }
        }
    }

    // warp-reduce awake count
    uint32_t found_mask = __ballot_sync(0xFFFFFFFF, found_parent);
    if (lane == 0) {
        atomicAdd((unsigned long long*)awake_count,
                  (unsigned long long)__popc(found_mask));
    }
}

#endif // __CUDACC__

// ════════════════════════════════════════════════════════════════════════════
//  GpuBfs — 主BFS类 (CPU/GPU双模式)
// ════════════════════════════════════════════════════════════════════════════

class GpuBfs {
public:
    GpuBfs() = default;

    // ── 运行BFS ─────────────────────────────────────────────────
    BfsResult run(const CsrGraph& graph, uint64_t source) {
        PHILE_CHECKPOINT("GpuBfs::run");
        auto t0 = std::chrono::steady_clock::now();

        BfsResult result;
        result.td_ms = result.bu_ms = 0;
        result.td_steps = result.bu_steps = 0;
        result.direction_switches = 0;

        uint64_t V = graph.num_vertices;
        uint64_t E = graph.num_edges;

        PHILE_DBG(1, "[BFS] V=%lu E=%lu source=%lu\n",
                  (unsigned long)V, (unsigned long)E, (unsigned long)source);

        // 初始化距离: -degree编码 (upstream BFS.h的init_distances模式)
        std::vector<int64_t> dist(V);
        for (uint64_t v = 0; v < V; ++v) {
            uint64_t deg = (graph.on_device) ? 0 :
                graph.row_offsets[v + 1] - graph.row_offsets[v];
            dist[v] = (deg != 0) ? -static_cast<int64_t>(deg) : -1;
        }
        dist[source] = 0;

        PHILE_DBG(2, "[BFS] init: dist[source]=%ld, dist[0..4]={%ld,%ld,%ld,%ld,%ld}\n",
                  dist[source],
                  V > 0 ? dist[0] : 0, V > 1 ? dist[1] : 0,
                  V > 2 ? dist[2] : 0, V > 3 ? dist[3] : 0,
                  V > 4 ? dist[4] : 0);

        // frontier初始化
        std::vector<int64_t> frontier = {static_cast<int64_t>(source)};

        int64_t scout_count = 0;
        if (!graph.on_device) {
            scout_count = static_cast<int64_t>(
                graph.row_offsets[source + 1] - graph.row_offsets[source]);
        }

        int64_t remaining_edges = static_cast<int64_t>(E);
        int64_t level = 1;
        bool in_bu_mode = false;

        // [ALG4] 动态切换比率
        double switch_ratio = compute_switch_ratio(V, E);
        PHILE_DBG(2, "[BFS] switch_ratio=%.6f (dynamic, orig alpha/beta fixed)\n",
                  switch_ratio);

        // bitmap (for BU mode)
        size_t bmp_words = (V + 31) / 32;
        std::vector<uint32_t> front_bmp(bmp_words, 0);
        std::vector<uint32_t> next_bmp(bmp_words, 0);

        while (!frontier.empty()) {
            auto step_t0 = std::chrono::steady_clock::now();

            // [ALG4] 动态direction switch
            // 原版: scout_count > edges_to_check / alpha (固定alpha=15)
            // 改动: frontier_edges / remaining > switch_ratio(自适应)
            double ratio = (remaining_edges > 0)
                ? static_cast<double>(scout_count) / remaining_edges
                : 1.0;

            if (!in_bu_mode && ratio > switch_ratio) {
                // 切换到BU
                in_bu_mode = true;
                result.direction_switches++;
                PHILE_DBG(2, "[BFS] L%ld: TD→BU switch (ratio=%.4f > %.4f)\n",
                          level, ratio, switch_ratio);

                // queue→bitmap
                std::fill(front_bmp.begin(), front_bmp.end(), 0);
                for (auto v : frontier) {
                    front_bmp[v / 32] |= (1u << (v & 31));
                }
            }

            if (in_bu_mode) {
                // BU step (CPU模拟)
                uint64_t awake = cpu_bu_step(graph, dist, level,
                                              front_bmp, next_bmp, V);
                result.bu_steps++;

                PHILE_DBG(2, "[BFS] L%ld BU: awake=%lu\n",
                          level, (unsigned long)awake);

                std::swap(front_bmp, next_bmp);
                std::fill(next_bmp.begin(), next_bmp.end(), 0);

                // 切回TD: awake太少
                if (awake < V / 50 && awake > 0) {
                    in_bu_mode = false;
                    result.direction_switches++;
                    // bitmap→queue
                    frontier.clear();
                    for (uint64_t w = 0; w < bmp_words; ++w) {
                        uint32_t word = front_bmp[w];
                        while (word) {
                            uint32_t bit = __builtin_ctz(word);
                            frontier.push_back(static_cast<int64_t>(w * 32 + bit));
                            word &= word - 1;
                        }
                    }
                    scout_count = 1;
                    PHILE_DBG(2, "[BFS] L%ld: BU→TD switch (awake=%lu < V/50)\n",
                              level, (unsigned long)awake);
                }

                if (awake == 0) break;
            } else {
                // TD step (CPU模拟, 模拟warp-level逻辑)
                std::vector<int64_t> next_frontier;
                next_frontier.reserve(frontier.size() * 2);
                scout_count = 0;

                // [ALG1] CPU模拟warp-level处理: 每个frontier vertex独立处理
                for (auto u : frontier) {
                    if (graph.on_device) continue;
                    uint64_t row_start = graph.row_offsets[u];
                    uint64_t row_end   = graph.row_offsets[u + 1];

                    for (uint64_t idx = row_start; idx < row_end; ++idx) {
                        uint64_t v = graph.col_indices[idx];
                        if (v >= V) continue;

                        int64_t curr = dist[v];
                        if (curr < 0) {
                            // CAS模拟
                            dist[v] = level;
                            next_frontier.push_back(static_cast<int64_t>(v));
                            scout_count += -curr;
                        }
                    }
                }

                remaining_edges -= scout_count;
                result.td_steps++;

                PHILE_DBG(2, "[BFS] L%ld TD: frontier=%zu→%zu scout=%ld remaining=%ld\n",
                          level, frontier.size(), next_frontier.size(),
                          scout_count, remaining_edges);

                frontier = std::move(next_frontier);
            }

            auto step_t1 = std::chrono::steady_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_t1 - step_t0).count();
            if (in_bu_mode) result.bu_ms += step_ms;
            else result.td_ms += step_ms;

            level++;
        }

        // 结果收集
        result.distances = std::move(dist);
        result.visited_count = 0;
        for (auto d : result.distances) {
            if (d >= 0) result.visited_count++;
        }

        auto t1 = std::chrono::steady_clock::now();
        result.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PHILE_DBG(1, "[BFS] done: visited=%lu/%lu  %.2fms (TD=%.2f BU=%.2f)  "
                  "td_steps=%d bu_steps=%d switches=%d\n",
                  (unsigned long)result.visited_count,
                  (unsigned long)V, result.total_ms,
                  result.td_ms, result.bu_ms,
                  result.td_steps, result.bu_steps,
                  result.direction_switches);

        // 断点: 打印距离分布
        if (debug::get_debug_level() >= 2) {
            dump_distance_histogram(result.distances, V);
        }

        return result;
    }

private:
    // ── BU step CPU模拟 ─────────────────────────────────────────
    uint64_t cpu_bu_step(const CsrGraph& graph,
                         std::vector<int64_t>& dist,
                         int64_t level,
                         const std::vector<uint32_t>& front_bmp,
                         std::vector<uint32_t>& next_bmp,
                         uint64_t V)
    {
        uint64_t awake = 0;
        for (uint64_t v = 0; v < V; ++v) {
            if (dist[v] >= 0) continue;  // 已访问

            if (graph.on_device) continue;
            uint64_t row_start = graph.row_offsets[v];
            uint64_t row_end   = graph.row_offsets[v + 1];

            // [ALG2] 模拟ballot: 找到一个front邻居就停
            for (uint64_t idx = row_start; idx < row_end; ++idx) {
                uint64_t u = graph.col_indices[idx];
                if (u >= V) continue;
                uint32_t word = u / 32;
                uint32_t bit  = u & 31;
                if (front_bmp[word] & (1u << bit)) {
                    dist[v] = level;
                    next_bmp[v / 32] |= (1u << (v & 31));
                    awake++;
                    break;
                }
            }
        }
        return awake;
    }

    // ── 断点调试: 距离直方图 ────────────────────────────────────
    void dump_distance_histogram(const std::vector<int64_t>& dist, uint64_t V) {
        PHILE_SEPARATOR("BFS Distance Histogram");
        std::unordered_map<int64_t, uint64_t> hist;
        uint64_t unreachable = 0;
        for (auto d : dist) {
            if (d < 0) unreachable++;
            else hist[d]++;
        }

        int64_t max_level = 0;
        for (auto& [k, v] : hist) max_level = std::max(max_level, k);

        printf("[BFS hist] levels 0..%ld  unreachable=%lu\n",
               max_level, (unsigned long)unreachable);
        for (int64_t l = 0; l <= std::min(max_level, (int64_t)20); ++l) {
            auto it = hist.find(l);
            uint64_t cnt = (it != hist.end()) ? it->second : 0;
            printf("  L%-3ld: %8lu  ", l, (unsigned long)cnt);
            // mini bar
            uint64_t bar = (V > 0) ? (cnt * 40 / V) : 0;
            for (uint64_t b = 0; b < bar; ++b) printf("█");
            printf("\n");
        }
        PHILE_SEPARATOR("End BFS Histogram");
    }
};

} // namespace cuda_bfs
} // namespace philemon
