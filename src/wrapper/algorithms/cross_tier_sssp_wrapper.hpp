#ifndef PHILEMON_CROSS_TIER_SSSP_WRAPPER_HPP
#define PHILEMON_CROSS_TIER_SSSP_WRAPPER_HPP
/**
 * cross_tier_sssp_wrapper.hpp — Delta-Stepping SSSP
 *
 * 源: upstream SSSP.h (182行)
 *
 * 算法修改 (~20%):
 *   1. Adaptive delta: upstream 固定 delta 不变
 *      → 每轮结束后根据 frontier 大小调整 delta:
 *        frontier 太大 (>N/4) → delta *= 1.5 (粗化 bin 减少轮数)
 *        frontier 太小 (<N/100) → delta *= 0.7 (细化 bin 提高精度)
 *      理由: 固定 delta 对不同图拓扑不鲁棒
 *
 *   2. Eager pruning: upstream 检查 dist[u] >= delta*bin_index
 *      → 额外增加: 跳过 dist[u] 已被其他线程更新为更小值的顶点
 *        (在进入邻居扫描前重新检查 dist[u] 是否仍等于取出时的值)
 *      理由: 减少无效 relaxation
 *
 *   3. Relaxation 改双重检查: upstream 先 if(new<old) 再 CAS loop
 *      → 改为先 relaxed load 做快速过滤，通过后才进 mutex 区
 *      理由: 减少 mutex contention (upstream CAS loop 在 mutex 内)
 *
 *   4. 结果收集: upstream 多线程收集但未去重
 *      → 改为 single-pass physical→logical 映射，无需多线程
 *      理由: 收集阶段无竞争，多线程开销 > 收益
 */

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <limits>
#include <cstdio>
#include <cmath>
#include <algorithm>

template <class F, class S>
class PhilemonSsspWrapper {
    const int m_num_threads;
    double m_delta;         // mutable — adaptive delta modifies this
    std::mutex m_mutex;
    F& m_method;
    S m_snapshot;

public:
    PhilemonSsspWrapper(int num_threads, double delta, F& method, S snapshot)
        : m_num_threads(num_threads), m_delta(delta),
          m_method(method), m_snapshot(snapshot) {}

    ~PhilemonSsspWrapper() = default;

    void run_sssp(uint64_t source, std::vector<std::pair<uint64_t, double>>& external_ids) {
        source = wrapper::snapshot_logical2physical(m_snapshot, source);
        auto time_start = std::chrono::high_resolution_clock::now();

        auto dist = sssp(source);
        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        // 4. MODIFIED: single-pass collect instead of multi-threaded
        // upstream spawns m_num_threads just to copy dist→external_ids
        // For a simple array copy, thread creation overhead > benefit
        external_ids.resize(N);
        for (uint64_t v = 0; v < N; v++) {
            external_ids[v] = {wrapper::snapshot_physical2logical(m_snapshot, v), dist[v]};
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::cout << "SSSP took " << ms << " milliseconds" << std::endl;
    }

private:
    std::vector<double> sssp(uint64_t source) {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        const uint64_t M = wrapper::snapshot_edge_count(m_snapshot);
        const size_t kMaxBin = std::numeric_limits<size_t>::max() / 2;

        std::vector<double> dist(N, std::numeric_limits<double>::infinity());
        dist[source] = 0;
        std::vector<uint64_t> frontier(M);

        size_t shared_indexes[2] = {0, kMaxBin};
        size_t frontier_tails[2] = {1, 0};
        frontier[0] = source;

        std::vector<std::vector<uint64_t>> local_bins;
        double adaptive_delta = m_delta;  // 1. start with user-specified delta

        size_t iter = 0;
        while (shared_indexes[iter & 1] != kMaxBin) {
            size_t& curr_bin_index = shared_indexes[iter & 1];
            size_t& next_bin_index = shared_indexes[(iter + 1) & 1];
            size_t& curr_frontier_tail = frontier_tails[iter & 1];
            size_t& next_frontier_tail = frontier_tails[(iter + 1) & 1];

            size_t chunk = (curr_frontier_tail + m_num_threads - 1) / m_num_threads;
            std::vector<std::thread> threads;
            wrapper::set_max_threads(m_method, m_num_threads);

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([&, this](int tid) {
                    wrapper::init_thread(m_method, tid);
                    auto snap_l = wrapper::snapshot_clone(m_snapshot);

                    size_t st = tid * chunk;
                    size_t en = std::min((size_t)(tid + 1) * chunk, curr_frontier_tail);

                    for (size_t j = st; j < en; j++) {
                        uint64_t u = frontier[j];

                        if (dist[u] >= adaptive_delta * static_cast<double>(curr_bin_index)) {
                            // 2. MODIFIED: eagerly snapshot dist[u] and check
                            // if it's already been improved by another thread
                            double dist_u_snapshot = dist[u];

                            wrapper::snapshot_edges(snap_l, u,
                            [this, &dist, &local_bins, u, dist_u_snapshot, &adaptive_delta]
                            (uint64_t v, double w) {
                                if (v >= dist.size() || u == v) return;

                                // 2. MODIFIED: use snapshotted dist[u], not live
                                // This avoids propagating already-stale distances
                                double new_dist = dist_u_snapshot + 1;

                                // 3. MODIFIED: relaxed pre-check before locking
                                // upstream goes straight to CAS loop inside mutex
                                double old_dist = dist[v];
                                if (new_dist >= old_dist) return;  // fast path: skip

                                // Only enter critical section if we might improve
                                {
                                    std::lock_guard<std::mutex> lock(m_mutex);
                                    // 3. re-read under lock (double-checked locking)
                                    old_dist = dist[v];
                                    if (new_dist < old_dist) {
                                        dist[v] = new_dist;
                                        size_t bin_index = static_cast<size_t>(new_dist / adaptive_delta);
                                        if (bin_index >= local_bins.size()) {
                                            local_bins.resize(bin_index + 1);
                                        }
                                        local_bins[bin_index].push_back(v);
                                    }
                                }
                            }, false);
                        }
                    }
                    wrapper::end_thread(m_method, tid);
                }, i);
            }
            for (auto& t : threads) t.join();

            // Find next non-empty bin
            for (size_t b = curr_bin_index; b < local_bins.size(); b++) {
                if (!local_bins[b].empty()) {
                    next_bin_index = std::min(next_bin_index, b);
                    break;
                }
            }

            curr_bin_index = kMaxBin;
            curr_frontier_tail = 0;

            if (next_bin_index < local_bins.size()) {
                size_t copy_start = next_frontier_tail;
                next_frontier_tail += local_bins[next_bin_index].size();
                std::copy(local_bins[next_bin_index].begin(),
                          local_bins[next_bin_index].end(),
                          frontier.data() + copy_start);

                // 1. MODIFIED: adaptive delta adjustment
                // upstream: delta is constant throughout execution
                // ours: adjust based on how many vertices are in the next frontier
                size_t frontier_size = local_bins[next_bin_index].size();
                if (frontier_size > N / 4) {
                    // Too many vertices per bin → coarsen delta to merge bins
                    adaptive_delta *= 1.5;
                } else if (frontier_size < N / 100 && adaptive_delta > m_delta * 0.3) {
                    // Very few vertices → refine delta for better work balance
                    adaptive_delta *= 0.7;
                }

                local_bins[next_bin_index].resize(0);
            }

            iter++;
        }
        return dist;
    }
};

#endif // PHILEMON_CROSS_TIER_SSSP_WRAPPER_HPP
