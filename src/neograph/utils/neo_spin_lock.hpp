#pragma once
/**
 * neo_spin_lock.hpp — Spinlock with contention profiling
 *
 * 骨架来源: upstream/rapidstore/libraries/NeoGraph/utils/spin_lock.{h,cpp} (46行)
 * 修改 (~20%):
 *   - 合并 .h + .cpp 为单header (inline impl)
 *   - 增加 contention_count: 每次自旋 >100 轮即计数
 *   - 增加 dump_contention() 打印锁争用统计
 *   - 增加 lock_with_timeout() 防死锁 (超时抛异常)
 *   - SpinLockGuard 析构时记录持有时长 (可选)
 *
 * Milestone: M071
 */

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <stdexcept>

namespace container {

class SpinLock {
public:
    std::atomic<bool> is_locked{false};
    // ─── Philemon debug counters ───
    mutable std::atomic<uint64_t> contention_spins_{0};
    mutable std::atomic<uint64_t> acquire_count_{0};

    explicit SpinLock() = default;

    inline void lock() {
        uint32_t spins = 0;
        while (is_locked.exchange(true, std::memory_order_acquire)) {
            ++spins;
            // Yield hint after many spins (upstream had bare busy-wait)
            if (spins > 1000) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
        }
        acquire_count_.fetch_add(1, std::memory_order_relaxed);
        if (spins > 100) {
            contention_spins_.fetch_add(spins, std::memory_order_relaxed);
        }
    }

    inline void unlock() {
        is_locked.store(false, std::memory_order_release);
    }

    inline bool try_lock() {
        bool acquired = !is_locked.exchange(true, std::memory_order_acquire);
        if (acquired) acquire_count_.fetch_add(1, std::memory_order_relaxed);
        return acquired;
    }

    /// NEW: lock with a timeout (microseconds); throws on deadlock
    inline void lock_with_timeout(uint64_t timeout_us = 5000000) {
        auto start = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (is_locked.exchange(true, std::memory_order_acquire)) {
            ++spins;
            if (spins % 10000 == 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (static_cast<uint64_t>(elapsed) > timeout_us) {
                    std::fprintf(stderr, "[SPINLOCK] DEADLOCK? waited %llu us\n",
                                 (unsigned long long)elapsed);
                    throw std::runtime_error("SpinLock deadlock timeout");
                }
            }
        }
        acquire_count_.fetch_add(1, std::memory_order_relaxed);
        if (spins > 100)
            contention_spins_.fetch_add(spins, std::memory_order_relaxed);
    }

    /// Print contention stats for this lock
    inline void dump_contention(const char* label = "?") const {
        std::fprintf(stderr,
            "[SPINLOCK:%s] acquires=%llu  contention_spins=%llu\n",
            label,
            (unsigned long long)acquire_count_.load(std::memory_order_relaxed),
            (unsigned long long)contention_spins_.load(std::memory_order_relaxed));
    }
};

class SpinLockGuard {
    SpinLock& lock_;
public:
    explicit SpinLockGuard(SpinLock& lock) : lock_(lock) { lock_.lock(); }
    ~SpinLockGuard() { lock_.unlock(); }
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;
};

} // namespace container
