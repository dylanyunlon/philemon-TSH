#ifndef PHILEMON_NEO_THREAD_POOL_HPP
#define PHILEMON_NEO_THREAD_POOL_HPP
/**
 * neo_thread_pool.hpp — Lock-based thread pool with debug instrumentation
 *
 * 骨架来源: upstream/rapidstore/libraries/NeoGraph/utils/thread_pool.h (98行)
 * 修改 (~20%):
 *   - 增加 tasks_enqueued / tasks_completed 原子计数器
 *   - 增加 dump_pool_state() 打印队列深度 + 活跃线程数
 *   - 增加 pending_count() 供调度器决策
 *   - 析构时打印总吞吐统计
 *   - 每个 worker 线程设置 name (方便 gdb attach)
 *
 * Milestone: M071
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
#include <cstdio>
#include <cstring>
#ifdef __linux__
#include <pthread.h>
#endif

class ThreadPool {
public:
    explicit ThreadPool(size_t threads);

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, size_t, Args...>::type>;

    ~ThreadPool();

    // ─── Philemon debug extensions ───
    size_t pending_count() const {
        std::unique_lock<std::mutex> lk(queue_mutex);
        return tasks.size();
    }

    void dump_pool_state(const char* label = "pool") const {
        std::fprintf(stderr,
            "[THREAD-POOL:%s] workers=%zu  pending=%zu  "
            "enqueued=%llu  completed=%llu\n",
            label, workers.size(), pending_count(),
            (unsigned long long)tasks_enqueued_.load(std::memory_order_relaxed),
            (unsigned long long)tasks_completed_.load(std::memory_order_relaxed));
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void(size_t)>> tasks;

    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    // ─── Philemon counters ───
    std::atomic<uint64_t> tasks_enqueued_{0};
    std::atomic<uint64_t> tasks_completed_{0};
};

// ─── Constructor: launch workers + set thread names ───
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this, i] {
#ifdef __linux__
            char tname[16];
            std::snprintf(tname, sizeof(tname), "neo-pool-%zu", i);
            pthread_setname_np(pthread_self(), tname);
#endif
            for (;;) {
                std::function<void(size_t)> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task(i);
                this->tasks_completed_.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
}

// ─── Enqueue (upstream, +counter) ───
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, size_t, Args...>::type>
{
    using return_type = typename std::invoke_result<F, size_t, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type(size_t)>>(
        std::bind(std::forward<F>(f), std::placeholders::_1,
                  std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](size_t tid) { (*task)(tid); });
    }
    tasks_enqueued_.fetch_add(1, std::memory_order_relaxed);
    condition.notify_one();
    return res;
}

// ─── Destructor: join + stats dump ───
inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (auto& w : workers) w.join();

    std::fprintf(stderr,
        "[THREAD-POOL] shutdown — enqueued=%llu completed=%llu\n",
        (unsigned long long)tasks_enqueued_.load(),
        (unsigned long long)tasks_completed_.load());
}

#endif // PHILEMON_NEO_THREAD_POOL_HPP
