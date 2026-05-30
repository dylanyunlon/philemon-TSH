#ifndef PHILEMON_SPIN_LOCK_HPP
#define PHILEMON_SPIN_LOCK_HPP
/**
 * spin_lock.hpp — Lightweight spinlock for hot-path synchronization
 *
 * 骨架来源: upstream/rapidstore/NeoGraph/utils/spin_lock.h (23行)
 * 修改 (~10%):
 *   - 包裹在 philemon::executor namespace
 *   - 增加 contention counter (debug mode)
 *
 * Milestone: M015
 */

#include <atomic>
#include <thread>

namespace philemon {
namespace executor {

class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            contention_.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

    bool try_lock() {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    uint64_t contention_count() const {
        return contention_.load(std::memory_order_relaxed);
    }

    void reset_contention() {
        contention_.store(0, std::memory_order_relaxed);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    std::atomic<uint64_t> contention_{0};
};

// RAII guard
class SpinGuard {
public:
    explicit SpinGuard(SpinLock& lock) : lock_(lock) { lock_.lock(); }
    ~SpinGuard() { lock_.unlock(); }
    SpinGuard(const SpinGuard&) = delete;
    SpinGuard& operator=(const SpinGuard&) = delete;
private:
    SpinLock& lock_;
};

}  // namespace executor
}  // namespace philemon

#endif  // PHILEMON_SPIN_LOCK_HPP
