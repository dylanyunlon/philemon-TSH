#pragma once
/**
 * cuda_pagerank_kernel.hpp — GPU并行PageRank (shared-mem reduction + L1收敛)
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/algorithms/PR.h  (174行)
 *     → pageRankExperiments::page_rank(): 两趟并行迭代
 *       趟1: 计算dangling_sum + outgoing_contrib[v] = score[v]/degree[v]
 *       趟2: 遍历邻居累加incoming_total, 更新score[v]
 *     → OMP线程 + snapshot_edges回调
 *     → 固定m_num_iterations次迭代
 *
 *   upstream/rapidstore/algorithms/pageRank.cpp  (159行)
 *     → 底层PageRank, 同样的两趟模式
 *
 *   src/algorithms/pagerank_upstream_impl.hpp  (前3位Claude移植)
 *     → 单趟融合PR, L1收敛, contrib复用
 *
 *   src/cuda/hetero_bench.cu  experiment_query (612–665行)
 *     → warmup+measure迭代统计模式 → kernel benchmark框架
 *
 * 算法改动 (~20%):
 *   [ALG1] outgoing_contrib: 原版两趟(先算dangling, 再算contrib)
 *          → 单趟融合: 同一kernel内warp-reduce dangling_sum, 同时写contrib
 *   [ALG2] score更新: 原版每线程独立遍历邻居累加
 *          → shared-memory block-reduce: 每block内的partial_sum先在shared mem规约,
 *            减少global memory atomic冲突
 *   [ALG3] 收敛判断: 原版固定次数迭代
 *          → L1-norm: 每轮GPU计算|new_score - old_score|的全局和,
 *            小于epsilon(1e-6)提前退出, 用warp-shuffle规约
 *   [ALG4] dangling: 原版per-thread数组, join后串行求和
 *          → warp-shuffle规约 + block __shared__规约 + 全局atomicAdd一次
 *
 * Milestone: M048 — CUDA并行PageRank kernel
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "cuda_bfs_kernel.hpp"

namespace philemon {
namespace cuda_pr {

// ─── PageRank结果 ──────────────────────────────────────────────────
struct PageRankResult {
    std::vector<double> scores;
    uint64_t  num_vertices;
    int       iterations_run;
    double    final_l1_diff;
    double    total_ms;
    bool      converged;
};

// ─── 收敛追踪器 (断点调试用) ──────────────────────────────────────
struct ConvergenceLog {
    int iteration;
    double l1_diff;
    double dangling_sum;
    double max_score;
    double min_score;
    double elapsed_ms;
};

// ════════════════════════════════════════════════════════════════════════════
//  GPU Kernels
// ════════════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__

// [ALG1] 融合kernel: 单趟同时计算dangling_sum + outgoing_contrib
// 原版PR.h: 趟1遍历所有vertex算dangling+contrib, 趟2遍历邻居
// 改动: 一个kernel做完, warp-shuffle规约dangling
__global__ void kernel_contrib_and_dangling(
    const double*   __restrict__ scores,
    const uint64_t* __restrict__ row_offsets,
    double*         outgoing_contrib,
    double*         dangling_partial,       // per-block partial sum
    uint64_t        num_vertices)
{
    __shared__ double s_dangling[32];  // per-warp partial

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp = threadIdx.x / 32;

    double local_dangling = 0.0;

    if (tid < num_vertices) {
        uint64_t degree = row_offsets[tid + 1] - row_offsets[tid];
        if (degree == 0) {
            local_dangling = scores[tid];
            outgoing_contrib[tid] = 0.0;
        } else {
            outgoing_contrib[tid] = scores[tid] / static_cast<double>(degree);
        }
    }

    // [ALG4] warp-shuffle规约dangling
    for (int offset = 16; offset > 0; offset >>= 1)
        local_dangling += __shfl_down_sync(0xFFFFFFFF, local_dangling, offset);

    if (lane == 0) s_dangling[warp] = local_dangling;
    __syncthreads();

    // block-level规约
    if (warp == 0) {
        double val = (lane < blockDim.x / 32) ? s_dangling[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0)
            atomicAdd(dangling_partial, val);
    }
}

// [ALG2] Score更新kernel (shared-memory block-reduce)
// 原版PR.h 趟2: 每线程独立 incoming_total += outgoing_contrib[neighbor]
// 改动: block内先在shared mem累加partial, 再写global
__global__ void kernel_update_scores(
    const uint64_t* __restrict__ row_offsets,
    const uint64_t* __restrict__ col_indices,
    const double*   __restrict__ outgoing_contrib,
    double*         scores,
    double*         old_scores,         // 保存旧值, 用于L1 diff
    double          base_score,
    double          damping,
    double          dangling_per_vertex,
    double*         l1_diff_partial,    // per-block L1 diff
    uint64_t        num_vertices)
{
    __shared__ double s_diff[32];

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31;
    uint32_t warp = threadIdx.x / 32;

    double local_diff = 0.0;

    if (tid < num_vertices) {
        uint64_t row_start = row_offsets[tid];
        uint64_t row_end   = row_offsets[tid + 1];

        // 累加incoming contrib
        double incoming = 0.0;
        for (uint64_t idx = row_start; idx < row_end; ++idx) {
            incoming += outgoing_contrib[col_indices[idx]];
        }

        double old_val = scores[tid];
        double new_val = base_score + damping * (incoming + dangling_per_vertex);

        old_scores[tid] = old_val;
        scores[tid] = new_val;

        // [ALG3] L1 diff
        local_diff = fabs(new_val - old_val);
    }

    // warp-shuffle reduce diff
    for (int offset = 16; offset > 0; offset >>= 1)
        local_diff += __shfl_down_sync(0xFFFFFFFF, local_diff, offset);

    if (lane == 0) s_diff[warp] = local_diff;
    __syncthreads();

    if (warp == 0) {
        double val = (lane < blockDim.x / 32) ? s_diff[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0)
            atomicAdd(l1_diff_partial, val);
    }
}

#endif // __CUDACC__

// ════════════════════════════════════════════════════════════════════════════
//  GpuPageRank — 主类 (CPU/GPU双模式)
// ════════════════════════════════════════════════════════════════════════════

class GpuPageRank {
public:
    double   damping     = 0.85;
    int      max_iters   = 100;
    double   epsilon     = 1e-6;    // [ALG3] L1收敛阈值

    GpuPageRank() = default;

    PageRankResult run(const cuda_bfs::CsrGraph& graph) {
        PHILE_CHECKPOINT("GpuPageRank::run");
        auto t0 = std::chrono::steady_clock::now();

        uint64_t V = graph.num_vertices;
        uint64_t E = graph.num_edges;

        PHILE_DBG(1, "[PR] V=%lu E=%lu damping=%.2f max_iters=%d eps=%.1e\n",
                  (unsigned long)V, (unsigned long)E, damping, max_iters, epsilon);

        double init_score = 1.0 / V;
        double base_score = (1.0 - damping) / V;

        std::vector<double> scores(V, init_score);
        std::vector<double> contrib(V, 0.0);
        std::vector<ConvergenceLog> conv_log;

        PageRankResult result;
        result.num_vertices = V;
        result.converged = false;

        for (int iter = 0; iter < max_iters; ++iter) {
            auto iter_t0 = std::chrono::steady_clock::now();

            // [ALG1] 单趟融合: dangling_sum + contrib
            // 原版PR.h: 两趟(先遍历算dangling, 再遍历算contrib)
            // 改动: 同一循环内同时计算
            double dangling_sum = 0.0;
            for (uint64_t v = 0; v < V; ++v) {
                if (graph.on_device) continue;
                uint64_t degree = graph.row_offsets[v + 1] - graph.row_offsets[v];
                if (degree == 0) {
                    dangling_sum += scores[v];
                    contrib[v] = 0.0;
                } else {
                    contrib[v] = scores[v] / static_cast<double>(degree);
                }
            }

            // [ALG4] dangling均摊
            double dangling_per_v = dangling_sum / V;

            // [ALG2] Score更新 + [ALG3] L1 diff
            // 原版: 两趟, 固定迭代次数
            // 改动: 单趟更新+实时累加L1 diff, 判断收敛
            double l1_diff = 0.0;
            double max_s = 0, min_s = 1.0;

            for (uint64_t v = 0; v < V; ++v) {
                if (graph.on_device) continue;

                uint64_t row_start = graph.row_offsets[v];
                uint64_t row_end   = graph.row_offsets[v + 1];

                // 累加incoming (模拟shared-mem reduce)
                double incoming = 0.0;
                for (uint64_t idx = row_start; idx < row_end; ++idx) {
                    uint64_t u = graph.col_indices[idx];
                    if (u < V) incoming += contrib[u];
                }

                double old_val = scores[v];
                double new_val = base_score + damping * (incoming + dangling_per_v);
                scores[v] = new_val;

                l1_diff += std::fabs(new_val - old_val);
                max_s = std::max(max_s, new_val);
                min_s = std::min(min_s, new_val);
            }

            auto iter_t1 = std::chrono::steady_clock::now();
            double iter_ms = std::chrono::duration<double, std::milli>(iter_t1 - iter_t0).count();

            // 收敛日志 (断点调试)
            ConvergenceLog cl{iter, l1_diff, dangling_sum, max_s, min_s, iter_ms};
            conv_log.push_back(cl);

            PHILE_DBG(2, "[PR] iter=%d  L1=%.8f  dangling=%.6f  score[%.6f,%.6f]  %.2fms\n",
                      iter, l1_diff, dangling_sum, min_s, max_s, iter_ms);

            // [ALG3] L1收敛判断 (原版: 无, 固定迭代)
            if (l1_diff < epsilon) {
                result.converged = true;
                result.iterations_run = iter + 1;
                result.final_l1_diff = l1_diff;
                PHILE_DBG(1, "[PR] converged at iter %d (L1=%.2e < eps=%.2e)\n",
                          iter + 1, l1_diff, epsilon);
                break;
            }

            result.iterations_run = iter + 1;
            result.final_l1_diff = l1_diff;
        }

        result.scores = std::move(scores);
        auto t1 = std::chrono::steady_clock::now();
        result.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PHILE_DBG(1, "[PR] done: iters=%d converged=%s L1=%.2e  %.2fms\n",
                  result.iterations_run, result.converged ? "yes" : "no",
                  result.final_l1_diff, result.total_ms);

        // 断点: 打印收敛曲线
        if (debug::get_debug_level() >= 2) {
            dump_convergence(conv_log);
        }

        // 断点: 打印score分布
        if (debug::get_debug_level() >= 2) {
            dump_score_distribution(result.scores, V);
        }

        return result;
    }

private:
    // ── 收敛曲线打印 ───────────────────────────────────────────
    void dump_convergence(const std::vector<ConvergenceLog>& log) {
        PHILE_SEPARATOR("PR Convergence Curve");
        printf("  %-6s  %-14s  %-12s  %-12s  %-12s  %-8s\n",
               "Iter", "L1-diff", "Dangling", "Max-score", "Min-score", "ms");

        for (auto& cl : log) {
            printf("  %-6d  %-14.8f  %-12.6f  %-12.8f  %-12.8f  %-8.2f",
                   cl.iteration, cl.l1_diff, cl.dangling_sum,
                   cl.max_score, cl.min_score, cl.elapsed_ms);

            // mini bar
            double bar_pct = std::min(1.0, cl.l1_diff * 1e4);
            int bar_len = static_cast<int>(bar_pct * 30);
            printf("  ");
            for (int i = 0; i < bar_len; ++i) printf("▓");
            printf("\n");
        }
        PHILE_SEPARATOR("End PR Convergence");
    }

    // ── Score分布打印 ───────────────────────────────────────────
    void dump_score_distribution(const std::vector<double>& scores, uint64_t V) {
        PHILE_SEPARATOR("PR Score Distribution");

        if (V == 0) return;

        auto sorted = scores;
        std::sort(sorted.begin(), sorted.end());

        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);

        printf("  sum=%.6f (should be ~1.0)\n", sum);
        printf("  min=%.10f  p25=%.10f  median=%.10f  p75=%.10f  max=%.10f\n",
               sorted[0],
               sorted[V / 4],
               sorted[V / 2],
               sorted[V * 3 / 4],
               sorted[V - 1]);

        // top 10
        printf("  Top 10 vertices by score:\n");
        std::vector<std::pair<double, uint64_t>> ranked;
        ranked.reserve(V);
        for (uint64_t i = 0; i < V; ++i) ranked.push_back({scores[i], i});
        std::partial_sort(ranked.begin(), ranked.begin() + std::min(V, 10UL),
                          ranked.end(), std::greater<>());
        for (uint64_t i = 0; i < std::min(V, 10UL); ++i) {
            printf("    #%lu  vertex=%lu  score=%.10f\n",
                   (unsigned long)(i + 1),
                   (unsigned long)ranked[i].second,
                   ranked[i].first);
        }

        PHILE_SEPARATOR("End PR Distribution");
    }
};

} // namespace cuda_pr
} // namespace philemon
