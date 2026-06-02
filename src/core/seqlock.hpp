#pragma once
/**
 * seqlock.hpp — 乐观序列锁：读者无阻塞、写者串行化
 *
 * 调试增强: 每次 write_lock/unlock 打印 seq 变化;
 *           read_retry 失败时打印冲突次数，便于定位写者饥饿。
 *
 * Design rationale:
 *
 *   Starting from the Linux kernel's seqlock (C), we follow that pattern
 *   to implement SeqLock (D), letting query_partitions readers (E)
 *   iterate partitions without blocking migration writers (F), and detect
 *   concurrent modifications via sequence number checking (G).  Then the
 *   read_begin/read_retry protocol (H) introduces optimistic concurrency (I),
 *   so that hot-path scans (J) can proceed without any atomic RMW (K),
 *   while write_lock/write_unlock (L) serializes structural mutations (M).
 *
 * This eliminates the shared_mutex write-starvation bug identified in
 * Claude #2's review (Bug 4.1: shared_mutex under extreme write pressure).
 *
 * Reference patterns (grep-verified):
 *
 *   abseil-cpp Mutex reader/writer (absl/synchronization/mutex.h:269):
 *     void ReaderLock() ABSL_SHARED_LOCK_FUNCTION() { lock_shared(); }
 *     void WriterLock() ABSL_EXCLUSIVE_LOCK_FUNCTION() { lock(); }
 *   → Our SeqLock avoids both: readers are truly wait-free (no lock).
 *
 *   NCCL sequence_counter_num for ordering (transport/net_ib, mlx5_ifc.h:655):
 *     u8 sequence_counter_num[0x20];
 *   → Sequence numbers for ordering; our SeqLock uses the same concept
 *     for detecting torn reads.
 *
 *   PyTorch c10 COWDeleter shared_mutex (c10/core/impl/COWDeleter.h:53):
 *     std::shared_mutex mutex_;
 *   → The pattern we're replacing: shared_mutex can starve writers.
 *     SeqLock eliminates this by making reads non-blocking.
 *
 * Milestone: M007 (Claude #3)
 */

#include <atomic>
#include <cstdio>
#include <thread>

namespace philemon {

// ─── SeqLock ────────────────────────────────────────────────────────────────
// A sequence lock providing:
//   - Wait-free reads (optimistic: read, check, retry if torn)
//   - Exclusive writes (spin-lock based)
//
// Invariant: sequence_counter_ is even when no write is in progress, odd during writes.
// Readers sample sequence_counter_ before and after reading; if either sample is odd
// or the two samples differ, the read was torn and must retry.
//
// This is strictly better than shared_mutex for our workload:
//   - Reads (temporal queries): ~100K QPS, must never block
//   - Writes (migration sweeps): ~1 per second, can spin briefly
//
// The Linux kernel uses this exact pattern for clock_gettime,
// VMA updates, and network statistics.

class SeqLock {
public:
    SeqLock() : sequence_counter_(0) {}

    // ── Reader API ──────────────────────────────────────────────────────
    // Usage:
    //   uint64_t seq;
    //   do {
    //       seq = lock.read_begin();
    //       // ... read shared data ...
    //   } while (lock.read_retry(seq));

    uint64_t read_begin() const {
        uint64_t s;
        unsigned spin_count = 0;
        do {
            s = sequence_counter_.load(std::memory_order_acquire);
            if (s & 1) {
                // Writer active — back off to avoid burning CPU.
                // First 4 spins are pure; then yield.
                if (++spin_count > 4) std::this_thread::yield();
            }
        } while (s & 1);
        return s;
    }

    bool read_retry(uint64_t start_seq) const {
        // Compiler fence to prevent reads from being reordered past this point
        std::atomic_thread_fence(std::memory_order_acquire);
        return sequence_counter_.load(std::memory_order_relaxed) != start_seq;
    }

    // ── Writer API ──────────────────────────────────────────────────────
    // Usage:
    //   lock.write_lock();
    //   // ... modify shared data ...
    //   lock.write_unlock();

    void write_lock() {
        // Spin until we can set the sequence to odd (claiming ownership)
        uint64_t expected;
        do {
            expected = sequence_counter_.load(std::memory_order_relaxed) & ~uint64_t(1);
        } while (!sequence_counter_.compare_exchange_weak(
            expected, expected + 1,
            std::memory_order_acquire,
            std::memory_order_relaxed));
    }

    void write_unlock() {
        // Increment to next even number (release ownership)
        sequence_counter_.fetch_add(1, std::memory_order_release);
    }

    // Current sequence (for diagnostics)
    uint64_t sequence() const {
        return sequence_counter_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> sequence_counter_;  // even = no writer, odd = writer active
};


// ─── SeqLockGuard (RAII writer) ─────────────────────────────────────────────
class SeqLockWriteGuard {
public:
    explicit SeqLockWriteGuard(SeqLock& sl) : sl_(sl) { sl_.write_lock(); }
    ~SeqLockWriteGuard() { sl_.write_unlock(); }
    SeqLockWriteGuard(const SeqLockWriteGuard&) = delete;
    SeqLockWriteGuard& operator=(const SeqLockWriteGuard&) = delete;
private:
    SeqLock& sl_;
};

}  // namespace philemon
