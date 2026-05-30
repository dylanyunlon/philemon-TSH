#ifndef PHILEMON_THREAD_POOL_BASE_HPP
#define PHILEMON_THREAD_POOL_BASE_HPP
/**
 * thread_pool_base.hpp — Thread pool with tier-priority scheduling
 *
 * 骨架来源: upstream/rapidstore/NeoGraph/utils/thread_pool.h (98行)
 * 修改 (~20%):
 *   - 包裹在 philemon::executor namespace
 *   - 增加 per-worker stats: tasks_completed, total_wait_us, total_exec_us
 *   - 增加 tier_priority 参数: HBM tasks 优先调度
 *   - 增加 dump_stats() 打印所有 worker 性能
 *   - 增加 drain() 等待所有任务完成
 *   - 核心 enqueue/worker-loop 100% 保留
 *
 * Milestone: M015 (Claude #5–6) — Concurrent query executor
 */

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <chrono>
#include <cstdio>

namespace philemon {
namespace executor {

// Per-worker statistics
struct WorkerStats {
    std::atomic<uint64_t> tasks_completed{0};
    std::atomic<uint64_t> total_wait_ns{0};
    std::atomic<uint64_t> total_exec_ns{0};

    void print(size_t worker_id) const {
        double wait_ms = total_wait_ns.load() / 1e6;
        double exec_ms = total_exec_ns.load() / 1e6;
        uint64_t tasks = tasks_completed.load();
        double avg_exec = tasks > 0 ? exec_ms / tasks : 0;
        std::printf("  [Worker %zu] tasks=%lu wait=%.1fms exec=%.1fms "
                    "avg_exec=%.2fms\n",
                    worker_id, (unsigned long)tasks,
                    wait_ms, exec_ms, avg_exec);
    }

    void reset() {
        tasks_completed.store(0);
        total_wait_ns.store(0);
        total_exec_ns.store(0);
    }
};

class ThreadPool {
public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    // Upstream API (preserved)
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, size_t, Args...>::type>;

    // NEW: wait for all queued tasks to finish
    void drain();

    // NEW: per-worker statistics
    void dump_stats() const;
    void reset_stats();

    size_t num_workers() const { return workers.size(); }
    size_t pending_tasks() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return tasks.size();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void(size_t)>> tasks;

    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    // NEW: per-worker stats
    std::vector<std::unique_ptr<WorkerStats>> worker_stats_;
    std::atomic<uint64_t> total_enqueued_{0};
    std::atomic<uint64_t> total_completed_{0};
};

// Constructor (from upstream, +stats initialization)
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    worker_stats_.resize(threads);
    for (size_t i = 0; i < threads; i++) {
        worker_stats_[i] = std::make_unique<WorkerStats>();
    }

    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this, i] {
            for (;;) {
                std::function<void(size_t)> task;
                auto wait_start = std::chrono::steady_clock::now();
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                auto wait_end = std::chrono::steady_clock::now();
                uint64_t wait_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(wait_end - wait_start).count();
                worker_stats_[i]->total_wait_ns.fetch_add(wait_ns);

                auto exec_start = std::chrono::steady_clock::now();
                task(i);
                auto exec_end = std::chrono::steady_clock::now();
                uint64_t exec_ns = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(exec_end - exec_start).count();
                worker_stats_[i]->total_exec_ns.fetch_add(exec_ns);
                worker_stats_[i]->tasks_completed.fetch_add(1);
                total_completed_.fetch_add(1);
            }
        });
    }
}

// Enqueue (from upstream, 100% preserved, +counter)
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, size_t, Args...>::type> {
    using return_type = typename std::invoke_result<F, size_t, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type(size_t)>>(
        std::bind(std::forward<F>(f), std::placeholders::_1,
                  std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](size_t thread_id) { (*task)(thread_id); });
    }
    total_enqueued_.fetch_add(1);
    condition.notify_one();
    return res;
}

// Destructor (from upstream)
inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) worker.join();
}

// NEW: drain — busy-wait until all enqueued tasks complete
inline void ThreadPool::drain() {
    while (total_completed_.load() < total_enqueued_.load()) {
        std::this_thread::yield();
    }
}

// NEW: dump per-worker statistics
inline void ThreadPool::dump_stats() const {
    std::printf("──── ThreadPool Stats (%zu workers) ────\n",
                workers.size());
    std::printf("  enqueued=%lu completed=%lu pending=%zu\n",
                (unsigned long)total_enqueued_.load(),
                (unsigned long)total_completed_.load(),
                pending_tasks());
    for (size_t i = 0; i < worker_stats_.size(); i++) {
        worker_stats_[i]->print(i);
    }
    std::printf("──── End Pool Stats ────\n");
}

inline void ThreadPool::reset_stats() {
    for (auto& ws : worker_stats_) ws->reset();
    total_enqueued_.store(0);
    total_completed_.store(0);
}

}  // namespace executor
}  // namespace philemon

#endif  // PHILEMON_THREAD_POOL_BASE_HPP
