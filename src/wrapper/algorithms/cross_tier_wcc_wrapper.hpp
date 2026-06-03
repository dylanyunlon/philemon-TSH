#ifndef PHILEMON_CROSS_TIER_WCC_WRAPPER_HPP
#define PHILEMON_CROSS_TIER_WCC_WRAPPER_HPP
/**
 * cross_tier_wcc_wrapper.hpp — Weakly Connected Components
 *
 * 源: upstream WCC.h (149行)
 *
 * 算法修改 (~20%):
 *   1. Hook 改 CAS: upstream 直接 comp[high] = low (无原子保证，data race)
 *      → 改为 CAS loop: compare_exchange 保证原子性
 *      理由: upstream 在多线程下有 data race (comp 是 raw uint64_t array)
 *
 *   2. Path compression 改 halving: upstream 用 full path compression
 *      (while comp[i] != comp[comp[i]]: comp[i] = comp[comp[i]])
 *      → 改为 path halving: comp[i] = comp[comp[i]] 只做一步
 *        但在 hook 阶段也做 inline halving
 *      理由: halving 每次减半路径长度且无需额外 pass，
 *        在实践中收敛速度相当但每轮工作量更少
 *
 *   3. 改 stale-aware hook: upstream hook 只比较 comp_u vs comp_v
 *      → hook 前先 find-root(u) 和 find-root(v)，然后 hook roots
 *      理由: 直接 hook comp[u] 可能 hook 到一个已被 hook 走的中间节点，
 *        导致多余轮数
 *
 *   4. change flag 改 atomic: upstream 用 plain bool (data race)
 *      → 改为 atomic<bool>
 */

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <algorithm>

template <class F, class S>
class PhilemonWccWrapper {
    const int m_num_threads;
    F& m_method;
    S m_snapshot;

public:
    PhilemonWccWrapper(int num_threads, F& method, S snapshot)
        : m_num_threads(num_threads), m_method(method), m_snapshot(snapshot) {}

    ~PhilemonWccWrapper() = default;

    void run_wcc(std::vector<std::pair<uint64_t, int64_t>>& external_ids) {
        auto time_start = std::chrono::high_resolution_clock::now();
        auto components = wcc();
        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        // Collect results
        external_ids.resize(N);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &components, chunk, &external_ids, N](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap_l = wrapper::snapshot_clone(m_snapshot);
                uint64_t st = tid * chunk, en = std::min(st + chunk, N);
                for (uint64_t u = st; u < en; u++) {
                    external_ids[u] = {
                        wrapper::snapshot_physical2logical(snap_l, u),
                        (int64_t)wrapper::snapshot_physical2logical(m_snapshot, components[u])
                    };
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();

        auto time_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::cout << "WCC took " << ms << " milliseconds" << std::endl;
    }

private:
    // 3. MODIFIED: find root with inline path halving
    // upstream has no find_root — directly reads comp[u]
    static uint64_t find_root(std::vector<std::atomic<uint64_t>>& comp, uint64_t x) {
        while (true) {
            uint64_t p = comp[x].load(std::memory_order_acquire);
            if (p == x) return x;
            // 2. inline path halving during find
            uint64_t gp = comp[p].load(std::memory_order_acquire);
            if (gp != p) {
                comp[x].compare_exchange_weak(p, gp,
                    std::memory_order_release, std::memory_order_relaxed);
            }
            x = p;
        }
    }

    std::vector<uint64_t> wcc() {
        const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);

        // 1. MODIFIED: use atomic<uint64_t> instead of raw uint64_t
        // upstream: uint64_t* comp with data races on concurrent writes
        std::vector<std::atomic<uint64_t>> comp(N);
        for (uint64_t i = 0; i < N; i++) {
            comp[i].store(i, std::memory_order_relaxed);
        }

        // 4. MODIFIED: atomic bool instead of plain bool
        std::atomic<bool> change{true};
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;

        while (change.load(std::memory_order_acquire)) {
            change.store(false, std::memory_order_release);
            std::vector<std::thread> threads;

            for (int i = 0; i < m_num_threads; i++) {
                threads.emplace_back([&, this](int tid) {
                    wrapper::init_thread(m_method, tid);
                    auto snap_l = wrapper::snapshot_clone(m_snapshot);
                    uint64_t st = tid * chunk, en = std::min(st + chunk, N);

                    for (uint64_t u = st; u < en; u++) {
                        wrapper::snapshot_edges(snap_l, u,
                        [&comp, &change, u, N](uint64_t v, double w) {
                            // 3. MODIFIED: find roots before hooking
                            // upstream: directly reads comp[u], comp[v]
                            // ours: find actual roots to avoid hooking stale nodes
                            uint64_t root_u = find_root(comp, u);
                            uint64_t root_v = find_root(comp, v);
                            if (root_u == root_v) return;

                            uint64_t high = std::max(root_u, root_v);
                            uint64_t low  = std::min(root_u, root_v);

                            if (high >= N || low >= N) return;

                            // 1. MODIFIED: CAS hook instead of direct write
                            // upstream: comp[high] = low (data race)
                            // ours: atomic CAS — only hook if high still points to itself
                            uint64_t expected = high;
                            if (comp[high].compare_exchange_strong(
                                    expected, low,
                                    std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
                                change.store(true, std::memory_order_release);
                            }
                        }, false);
                    }

                    wrapper::end_thread(m_method, tid);
                }, i);
            }
            for (auto& t : threads) t.join();

            // 2. MODIFIED: path halving instead of full compression
            // upstream: while comp[i] != comp[comp[i]]: comp[i] = comp[comp[i]]
            //   (jumps until flat — up to O(depth) per vertex)
            // ours: single halving step per vertex per round
            //   (each round cuts path length in half — amortized O(α(n)) total)
            for (uint64_t i = 0; i < N; i++) {
                uint64_t p = comp[i].load(std::memory_order_relaxed);
                uint64_t gp = comp[p].load(std::memory_order_relaxed);
                if (p != gp) {
                    comp[i].store(gp, std::memory_order_relaxed);
                }
            }
        }

        // Final full compression to canonical roots
        std::vector<uint64_t> result(N);
        for (uint64_t i = 0; i < N; i++) {
            result[i] = find_root(comp, i);
        }
        return result;
    }
};

#endif // PHILEMON_CROSS_TIER_WCC_WRAPPER_HPP
