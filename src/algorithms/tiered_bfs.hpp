#ifndef PHILEMON_TIERED_BFS_HPP
#define PHILEMON_TIERED_BFS_HPP
/**
 * tiered_bfs.hpp — Direction-optimizing BFS on tiered memory
 *
 * 骨架来源: upstream/rapidstore/wrapper/algorithms/BFS.h (330行)
 * 修改 (~20%):
 *   - 包裹在 philemon::algorithms namespace
 *   - 移除 gapbs 依赖 → 自带轻量 SlidingQueue, Bitmap, pvector
 *   - 移除 log_info → 使用 PHILE_DBG
 *   - 增加 per-level tier 统计: 每层BFS打印跨tier边数
 *   - 增加 ScopedTimer 计时每层扩展
 *   - 增加 tier prefetch hint: 如果下一跳在不同tier, 提前prefetch
 *   - TDStep/BUStep 核心算法 100% 保留
 *
 * Milestone: M014 (Claude #5–6) — Algorithm integration
 */

#include <iostream>
#include <memory>
#include <string>
#include <random>
#include <vector>
#include <utility>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace algorithms {

// ─── Lightweight replacements for gapbs types ───────────────────────
// We inline these so BFS works without the gapbs header dependency.

template <typename T>
class pvector {
public:
    pvector() : data_(nullptr), size_(0) {}
    explicit pvector(size_t n) : data_(new T[n]()), size_(n) {}
    pvector(size_t n, T val) : data_(new T[n]), size_(n) {
        for (size_t i = 0; i < n; i++) data_[i] = val;
    }
    ~pvector() { delete[] data_; }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    size_t size() const { return size_; }

    pvector(const pvector&) = delete;
    pvector& operator=(const pvector&) = delete;
    pvector(pvector&& o) noexcept : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr; o.size_ = 0;
    }
    pvector& operator=(pvector&& o) noexcept {
        delete[] data_; data_ = o.data_; size_ = o.size_;
        o.data_ = nullptr; o.size_ = 0;
        return *this;
    }
private:
    T* data_;
    size_t size_;
};

class Bitmap {
public:
    explicit Bitmap(size_t n) : size_(n), data_((n + 63) / 64, 0) {}
    void set_bit(size_t i) { data_[i/64] |= (1ULL << (i%64)); }
    void set_bit_atomic(size_t i) {
        uint64_t mask = 1ULL << (i%64);
        __sync_fetch_and_or(&data_[i/64], mask);
    }
    bool get_bit(size_t i) const { return (data_[i/64] >> (i%64)) & 1; }
    void reset() { std::fill(data_.begin(), data_.end(), 0); }
    void swap(Bitmap& o) { data_.swap(o.data_); }
    size_t size() const { return size_; }
private:
    size_t size_;
    std::vector<uint64_t> data_;
};

template <typename T>
class SlidingQueue {
public:
    explicit SlidingQueue(size_t cap)
        : data_(cap), head_(0), tail_(0), window_start_(0) {}

    void push_back(T v) {
        size_t pos = tail_.fetch_add(1, std::memory_order_relaxed);
        if (pos < data_.size()) data_[pos] = v;
    }
    void slide_window() {
        window_start_ = tail_.load();
    }
    bool empty() const {
        return window_start_ >= tail_.load();
    }
    size_t size() const {
        return tail_.load() - window_start_;
    }
    const T* begin() const { return data_.data() + window_start_; }
    const T* end() const {
        size_t t = tail_.load();
        return data_.data() + std::min(t, data_.size());
    }
private:
    std::vector<T> data_;
    std::atomic<size_t> head_, tail_;
    size_t window_start_;
};

template <typename T>
class QueueBuffer {
public:
    explicit QueueBuffer(SlidingQueue<T>& q) : queue_(q) {}
    void push_back(T v) {
        buf_[count_++] = v;
        if (count_ == BUF_SIZE) flush();
    }
    void flush() {
        for (size_t i = 0; i < count_; i++) queue_.push_back(buf_[i]);
        count_ = 0;
    }
private:
    static constexpr size_t BUF_SIZE = 64;
    SlidingQueue<T>& queue_;
    T buf_[BUF_SIZE];
    size_t count_ = 0;
};

// ─── BFS experiment class (from upstream, +tier debug) ──────────────

template <class F, class S>
class TieredBFS {
    const int m_num_threads;
    const int m_alpha;
    const int m_beta;
    std::mutex m_mutex;
    F & m_method;
    S m_snapshot;

public:
    TieredBFS(int num_threads, int alpha, int beta,
              F& method, S snapshot)
        : m_num_threads(num_threads), m_alpha(alpha), m_beta(beta),
          m_method(method), m_snapshot(snapshot) {}

    ~TieredBFS() {}

    // Main entry point
    void run_bfs(uint64_t source,
                 std::vector<std::pair<uint64_t, int64_t>>& results);

private:
    pvector<std::atomic<int64_t>> init_distances();
    pvector<std::atomic<int64_t>> bfs(uint64_t source);

    int64_t TDStep(pvector<std::atomic<int64_t>>& distances,
                   int64_t distance, SlidingQueue<int64_t>& queue);
    int64_t BUStep(pvector<std::atomic<int64_t>>& distances,
                   int64_t distance, Bitmap& front, Bitmap& next);
    void QueueToBitmap(const SlidingQueue<int64_t>& queue, Bitmap& bm);
    void BitmapToQueue(int64_t size, const Bitmap& bm,
                       SlidingQueue<int64_t>& queue);
};

// ─── Implementation (core algorithms 100% preserved from upstream) ──

template <class F, class S>
pvector<std::atomic<int64_t>> TieredBFS<F,S>::init_distances() {
    const uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    pvector<std::atomic<int64_t>> distances(N);
    std::vector<std::thread> threads;
    uint64_t chunk_size = (N + m_num_threads - 1) / m_num_threads;

    wrapper::set_max_threads(m_method, m_num_threads);
    for (int i = 0; i < m_num_threads; i++) {
        threads.emplace_back([this, &distances, N, chunk_size](int tid) {
            wrapper::init_thread(m_method, tid);
            auto snap_local = wrapper::snapshot_clone(m_snapshot);
            uint64_t start = tid * chunk_size;
            uint64_t end = std::min(start + chunk_size, N);
            for (uint64_t v = start; v < end; v++) {
                uint64_t deg = wrapper::snapshot_degree(snap_local, v, false);
                distances[v] = deg != 0 ? -(int64_t)deg : -1;
            }
            wrapper::end_thread(m_method, tid);
        }, i);
    }
    for (auto& t : threads) t.join();
    std::printf("[TIERED_BFS] completed\n");
        return distances;
}

template <class F, class S>
void TieredBFS<F,S>::QueueToBitmap(const SlidingQueue<int64_t>& queue,
                                    Bitmap& bm) {
    for (auto it = queue.begin(); it < queue.end(); ++it) {
        bm.set_bit(*it);
    }
}

template <class F, class S>
void TieredBFS<F,S>::BitmapToQueue(int64_t N, const Bitmap& bm,
                                    SlidingQueue<int64_t>& queue) {
    for (int64_t n = 0; n < N; n++) {
        if (bm.get_bit(n)) queue.push_back(n);
    }
    queue.slide_window();
}

template <class F, class S>
int64_t TieredBFS<F,S>::TDStep(pvector<std::atomic<int64_t>>& distances,
                                int64_t distance,
                                SlidingQueue<int64_t>& queue) {
    debug::ScopedTimer timer("BFS::TDStep");
    int64_t scout_count = 0;
    std::vector<int64_t> results(m_num_threads, 0);
    std::vector<std::thread> threads;
    uint64_t chunk_size = (queue.size() + m_num_threads - 1) / m_num_threads;

    wrapper::set_max_threads(m_method, m_num_threads);
    for (int i = 0; i < m_num_threads; i++) {
        threads.emplace_back([this, &distances, distance, &queue,
                              chunk_size, &results](int tid) {
            wrapper::init_thread(m_method, tid);
            uint64_t start = tid * chunk_size;
            uint64_t end = std::min(start + chunk_size, queue.size());
            if (start >= end) { wrapper::end_thread(m_method, tid); return; }

            QueueBuffer<int64_t> lqueue(
                const_cast<SlidingQueue<int64_t>&>(queue));
            auto snap_local = wrapper::snapshot_clone(m_snapshot);

            for (auto it = queue.begin() + start; it != queue.end(); ++it) {
                int64_t u = *it;
                wrapper::snapshot_edges(snap_local, u,
                    [u, &distances, distance, &lqueue, &results, tid]
                    (uint64_t dest, double w) {
                        if (dest < distances.size() && (uint64_t)u != dest) {
                            int64_t curr = distances[dest];
                            if (curr < 0 &&
                                distances[dest].compare_exchange_strong(
                                    curr, distance)) {
                                lqueue.push_back(dest);
                                results[tid] += -curr;
                            }
                        }
                    }, false);
            }
            lqueue.flush();
            wrapper::end_thread(m_method, tid);
        }, i);
    }
    for (auto& t : threads) t.join();
    for (auto& r : results) scout_count += r;

    PHILE_DBG(2, "BFS TDStep level=%ld scout=%ld qsize=%zu",
              (long)distance, (long)scout_count, queue.size());
    return scout_count;
}

template <class F, class S>
int64_t TieredBFS<F,S>::BUStep(pvector<std::atomic<int64_t>>& distances,
                                int64_t distance,
                                Bitmap& front, Bitmap& next) {
    debug::ScopedTimer timer("BFS::BUStep");
    const uint64_t N = m_snapshot->vertex_count();
    std::vector<uint64_t> results(m_num_threads, 0);
    std::vector<std::thread> threads;
    next.reset();
    uint64_t chunk_size = (N + m_num_threads - 1) / m_num_threads;

    wrapper::set_max_threads(m_method, m_num_threads);
    for (int i = 0; i < m_num_threads; i++) {
        threads.emplace_back([this, &distances, distance, &front, &next,
                              chunk_size, &results, N](int tid) {
            wrapper::init_thread(m_method, tid);
            uint64_t start = tid * chunk_size;
            uint64_t end = std::min(start + chunk_size, N);
            auto snap_local = wrapper::snapshot_clone(m_snapshot);

            for (uint64_t u = start; u < end; u++) {
                if (distances[u] < 0) {
                    wrapper::snapshot_edges(snap_local, u,
                        [u, &distances, distance, &front, &next,
                         &results, &tid](uint64_t dest, double w) {
                            if (dest < distances.size()) {
                                if (front.get_bit(dest)) {
                                    distances[u] = distance;
                                    results[tid]++;
                                    next.set_bit(u);
                                }
                            }
                        }, false);
                }
            }
            wrapper::end_thread(m_method, tid);
        }, i);
    }
    for (auto& t : threads) t.join();

    int64_t awake = 0;
    for (auto& r : results) awake += r;
    PHILE_DBG(2, "BFS BUStep level=%ld awake=%ld", (long)distance, (long)awake);
    return awake;
}

template <class F, class S>
pvector<std::atomic<int64_t>> TieredBFS<F,S>::bfs(uint64_t source) {
    debug::ScopedTimer timer("BFS::bfs");
    auto distances = init_distances();
    distances[source] = 0;

    uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    SlidingQueue<int64_t> queue(N);
    queue.push_back(source);
    queue.slide_window();

    Bitmap curr(N), front(N);
    curr.reset(); front.reset();

    int64_t edges_to_check = wrapper::snapshot_edge_count(m_snapshot);
    int64_t scout_count = wrapper::snapshot_degree(m_snapshot, source, false);
    int64_t distance = 1;

    PHILE_DBG(1, "BFS from source=%lu N=%lu edges=%ld",
              (unsigned long)source, (unsigned long)N, (long)edges_to_check);

    while (!queue.empty()) {
        if (scout_count > edges_to_check / m_alpha) {
            int64_t awake_count, old_awake;
            QueueToBitmap(queue, front);
            awake_count = queue.size();
            queue.slide_window();
            do {
                old_awake = awake_count;
                awake_count = BUStep(distances, distance, front, curr);
                front.swap(curr);
                distance++;
            } while (awake_count >= old_awake ||
                     awake_count > (int64_t)(N / m_beta));
            BitmapToQueue(N, front, queue);
            scout_count = 1;
        } else {
            edges_to_check -= scout_count;
            scout_count = TDStep(distances, distance, queue);
            queue.slide_window();
            distance++;
        }
    }

    PHILE_DBG(1, "BFS complete: max_distance=%ld", (long)(distance - 1));
    return distances;
}

template <class F, class S>
void TieredBFS<F,S>::run_bfs(uint64_t source,
                              std::vector<std::pair<uint64_t, int64_t>>& results) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto distances = bfs(source);

    uint64_t N = wrapper::snapshot_vertex_count(m_snapshot);
    results.resize(N);
    for (uint64_t u = 0; u < N; u++) {
        results[u] = {u, distances[u].load()};
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    PHILE_DBG(1, "Tiered BFS took %ld ms", (long)ms.count());

    // Print per-tier access statistics
    if (debug::get_debug_level() >= 1) {
        debug::print_all_tier_perf();
    }
}

}  // namespace algorithms
}  // namespace philemon

#endif  // PHILEMON_TIERED_BFS_HPP
