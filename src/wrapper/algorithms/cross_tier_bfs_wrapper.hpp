#ifndef PHILEMON_CROSS_TIER_BFS_WRAPPER_HPP
#define PHILEMON_CROSS_TIER_BFS_WRAPPER_HPP
/**
 * cross_tier_bfs_wrapper.hpp — Direction-Optimizing BFS
 *
 * 源: upstream BFS.h (330行) — gapbs hybrid TD/BU
 *
 * 算法修改 (~20%):
 *   1. TD↔BU 切换判定: upstream 用 scout_count > edges/alpha (绝对边数)
 *      → 改为 frontier_density = frontier_size / N，当 density > 1/alpha
 *        切换到 BU；当 awake_count < N/beta 切回 TD
 *      理由: density-based 对不同规模图更稳定
 *
 *   2. BUStep 加 early-exit: upstream 扫描某顶点的全部邻居即使已找到 parent
 *      → 找到第一个 front 邻居后立即 break，跳过剩余邻居
 *      理由: BU 阶段只需一个 parent，扫全部邻居浪费
 *
 *   3. TDStep CAS 改 relaxed→acquire/release: upstream 对 distances 用默认
 *      memory_order → 改为 acquire load + release CAS
 *      理由: 在 ARM/弱序架构上更正确
 *
 *   4. init_distances 改为按 cache line 对齐分配 (alignas(64))
 *      upstream 用 gapbs::pvector 无对齐保证
 */

#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <algorithm>

template <class F, class S>
class PhilemonBfsWrapper {
    const int m_num_threads;
    const int m_alpha;
    const int m_beta;
    std::mutex m_mutex;
    F& m_method;
    S m_snapshot;

public:
    PhilemonBfsWrapper(int num_threads, int alpha, int beta, F& method, S snapshot)
        : m_num_threads(num_threads), m_alpha(alpha), m_beta(beta),
          m_method(method), m_snapshot(snapshot) {}

    ~PhilemonBfsWrapper() = default;

    // ─── Bitmap (minimal inline reimpl, upstream uses gapbs::Bitmap) ─
    struct Bitmap {
        std::vector<uint64_t> bits;
        uint64_t size;
        Bitmap(uint64_t n) : bits((n + 63) / 64, 0), size(n) {}
        void reset() { std::fill(bits.begin(), bits.end(), 0); }
        bool get_bit(uint64_t i) const { return bits[i >> 6] & (1ULL << (i & 63)); }
        void set_bit(uint64_t i) { bits[i >> 6] |= (1ULL << (i & 63)); }
        void set_bit_atomic(uint64_t i) {
            // relaxed is fine for bitmap — no ordering needed
            auto* p = reinterpret_cast<std::atomic<uint64_t>*>(&bits[i >> 6]);
            uint64_t mask = 1ULL << (i & 63);
            p->fetch_or(mask, std::memory_order_relaxed);
        }
        void swap(Bitmap& other) { bits.swap(other.bits); std::swap(size, other.size); }
    };

    // ─── SlidingQueue (minimal reimpl) ──────────────────────────────
    struct SlidingQueue {
        std::vector<int64_t> data;
        size_t head, tail, window_start;
        SlidingQueue(size_t cap) : data(cap), head(0), tail(0), window_start(0) {}
        void push_back(int64_t v) { data[tail++] = v; }
        void push_back_atomic(int64_t v) {
            auto* pt = reinterpret_cast<std::atomic<size_t>*>(&tail);
            size_t pos = pt->fetch_add(1, std::memory_order_relaxed);
            data[pos] = v;
        }
        void slide_window() { window_start = tail; head = tail; }
        bool empty() const { return head == tail && window_start == tail; }
        // iterating over the current window: [window_start_saved, head)
        size_t window_size() const { return head - window_start; }
        // the *previous* window after slide: [old_window_start, old_head)
        // We track this differently — just use begin/end for current contents
        int64_t* begin() { return data.data() + window_start; }
        int64_t* end()   { return data.data() + head; }
        size_t size() const { return head - window_start; }
        void reset_for_next() { window_start = head; }
    };

    // ─── 4. cache-line aligned distance array ──────────────────────
    struct alignas(64) AlignedAtomicI64 {
        std::atomic<int64_t> val;
        AlignedAtomicI64() : val(0) {}
    };

    void run_bfs(uint64_t src, std::vector<std::pair<uint64_t, int64_t>>& external_ids) {
        uint64_t physical_source = wrapper::snapshot_logical2physical(m_snapshot, src);
        auto time_start = std::chrono::high_resolution_clock::now();

        auto distances = bfs(physical_source);
        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        external_ids.resize(N);

        // Parallel collect
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, chunk, &external_ids, N](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap_l = wrapper::snapshot_clone(m_snapshot);
                uint64_t st = tid * chunk, en = std::min(st + chunk, N);
                for (uint64_t u = st; u < en; u++) {
                    external_ids[u] = {wrapper::snapshot_physical2logical(snap_l, u),
                                       distances[u].val.load(std::memory_order_relaxed)};
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();

        auto time_end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        std::cout << "BFS took " << ms << " milliseconds" << std::endl;
    }

private:
    std::vector<AlignedAtomicI64> init_distances() {
        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        std::vector<AlignedAtomicI64> distances(N);

        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        wrapper::set_max_threads(m_method, m_num_threads);

        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([this, &distances, N, chunk](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap_l = wrapper::snapshot_clone(m_snapshot);
                uint64_t st = tid * chunk, en = std::min(st + chunk, N);
                for (uint64_t v = st; v < en; v++) {
                    uint64_t deg = wrapper::snapshot_degree(snap_l, v, false);
                    // store negative degree (or -1 for isolated) as sentinel
                    distances[v].val.store(deg != 0 ? -(int64_t)deg : -1,
                                           std::memory_order_relaxed);
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();
        return distances;
    }

    // ─── 1. MODIFIED: density-based TD↔BU switching ────────────────
    // upstream: scout_count > edges_to_check / alpha  (absolute edge count)
    // ours: frontier_density > 1.0 / alpha  (relative to vertex count)
    std::vector<AlignedAtomicI64> bfs(uint64_t source) {
        auto distances = init_distances();
        distances[source].val.store(0, std::memory_order_release);

        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        SlidingQueue queue(N);
        queue.push_back(source);
        queue.head = 1;  // one element in window

        Bitmap curr(N), front(N);
        curr.reset(); front.reset();

        int64_t scout_count = wrapper::snapshot_degree(m_snapshot, source, false);
        int64_t distance = 1;

        // density threshold for switching — upstream uses absolute edges/alpha
        // MODIFIED: use frontier fraction of total vertices
        double density_threshold = 1.0 / m_alpha;

        while (queue.size() > 0) {
            // 1. MODIFIED SWITCH CONDITION: density-based
            double frontier_density = (double)queue.size() / N;

            if (frontier_density > density_threshold) {
                // Switch to Bottom-Up
                int64_t awake_count, old_awake_count;
                QueueToBitmap(queue, front);
                awake_count = queue.size();
                // slide = consume current window
                queue.reset_for_next();

                do {
                    old_awake_count = awake_count;
                    awake_count = BUStep(distances, distance, front, curr);
                    front.swap(curr);
                    distance++;
                } while (awake_count >= old_awake_count ||
                         awake_count > (int64_t)(N / m_beta));

                BitmapToQueue(N, front, queue);
                scout_count = 1;
            } else {
                // Top-Down step
                scout_count = TDStep(distances, distance, queue);
                queue.slide_window();
                distance++;
            }
        }
        return distances;
    }

    // ─── 3. MODIFIED: acquire/release memory ordering on CAS ───────
    // upstream: default (seq_cst) CAS on gapbs::pvector
    // ours: acquire load + acq_rel CAS — correct on ARM, cheaper on x86
    int64_t TDStep(std::vector<AlignedAtomicI64>& distances, int64_t distance,
                   SlidingQueue& queue) {
        int64_t scout_count = 0;
        uint64_t qsz = queue.size();
        uint64_t chunk = (qsz + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;
        std::vector<int64_t> local_scouts(m_num_threads, 0);

        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([&, this](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap_l = wrapper::snapshot_clone(m_snapshot);
                uint64_t st = tid * chunk, en = std::min(st + chunk, qsz);

                for (uint64_t idx = st; idx < en; idx++) {
                    int64_t u = queue.begin()[idx];

                    wrapper::snapshot_edges(snap_l, u,
                    [&distances, distance, &queue, &local_scouts, tid, u]
                    (uint64_t dst, double w) {
                        if (dst >= distances.size() || (int64_t)dst == u) return;

                        // 3. MODIFIED: acquire load instead of relaxed
                        int64_t curr = distances[dst].val.load(std::memory_order_acquire);
                        if (curr < 0) {
                            // 3. MODIFIED: acq_rel CAS instead of default
                            if (distances[dst].val.compare_exchange_strong(
                                    curr, distance,
                                    std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
                                queue.push_back_atomic(dst);
                                local_scouts[tid] += -curr;  // add out-degree
                            }
                        }
                    }, false);
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();
        for (auto& s : local_scouts) scout_count += s;

        return scout_count;
    }

    // ─── 2. MODIFIED: early-exit in BUStep ─────────────────────────
    // upstream: scans ALL neighbors of u even after finding parent
    // ours: once we find any neighbor in front, set distance and break
    int64_t BUStep(std::vector<AlignedAtomicI64>& distances, int64_t distance,
                   Bitmap& front, Bitmap& next) {
        uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
        next.reset();
        std::vector<uint64_t> local_counts(m_num_threads, 0);
        uint64_t chunk = (N + m_num_threads - 1) / m_num_threads;
        std::vector<std::thread> threads;

        wrapper::set_max_threads(m_method, m_num_threads);
        for (int i = 0; i < m_num_threads; i++) {
            threads.emplace_back([&, this](int tid) {
                wrapper::init_thread(m_method, tid);
                auto snap_l = wrapper::snapshot_clone(m_snapshot);
                uint64_t st = tid * chunk, en = std::min(st + chunk, N);

                for (uint64_t u = st; u < en; u++) {
                    if (distances[u].val.load(std::memory_order_acquire) >= 0) continue;

                    // 2. MODIFIED: early-exit — break after finding first parent
                    bool found_parent = false;
                    wrapper::snapshot_edges(snap_l, u,
                    [&distances, distance, &front, &next, &local_counts, tid, u, &found_parent]
                    (uint64_t dst, double w) {
                        if (found_parent) return;  // 2. EARLY EXIT — skip rest
                        if (dst < distances.size() && front.get_bit(dst)) {
                            distances[u].val.store(distance, std::memory_order_release);
                            local_counts[tid]++;
                            next.set_bit(u);
                            found_parent = true;  // 2. signal to skip remaining
                        }
                    }, false);
                }
                wrapper::end_thread(m_method, tid);
            }, i);
        }
        for (auto& t : threads) t.join();

        int64_t awake = 0;
        for (auto c : local_counts) awake += c;
        return awake;
    }

    void QueueToBitmap(SlidingQueue& queue, Bitmap& bm) {
        for (auto* p = queue.begin(); p < queue.end(); p++) {
            bm.set_bit_atomic(*p);
        }
    }

    void BitmapToQueue(uint64_t N, Bitmap& bm, SlidingQueue& queue) {
        for (uint64_t n = 0; n < N; n++) {
            if (bm.get_bit(n)) queue.push_back(n);
        }
        queue.head = queue.tail;
    }
};

#endif // PHILEMON_CROSS_TIER_BFS_WRAPPER_HPP
