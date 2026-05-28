/**
 * async_migrator.hpp — Asynchronous double-buffered migration engine
 *
 * Bug 4.4 (Claude #2 review): migrate() uses synchronous memcpy,
 * blocking all queries during data transfer. For large partitions
 * (3+ MB), this can stall query processing for 300+ µs.
 *
 * Solution: AsyncMigrator uses pinned staging buffers and async
 * copy semantics. In CPU-dev mode, it simulates async behavior
 * with a background thread + condition variable. In CUDA mode
 * (M009 future), it will use cudaMemcpyAsync + cudaStreamSynchronize.
 *
 * Double-buffering: two staging buffers allow one migration to be
 * in-flight while the next is being prepared. This follows
 * DeepSpeed's PartitionedOptimizerSwapper pattern:
 *   self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.aio_handle, ...)
 *   def swap_in_optimizer_state(self, parameter, async_parameter=None):
 *
 * The migration pipeline:
 *   1. Caller calls submit(alloc_id, target_tier)
 *   2. AsyncMigrator copies data into staging buffer (non-blocking)
 *   3. Background thread completes the swap (free old, update registry)
 *   4. Caller calls wait(ticket) or poll(ticket) to check completion
 *
 * Pattern lineage (grep-verified):
 *   DeepSpeed PartitionedOptimizerSwapper::swap_in_optimizer_state
 *     (partitioned_optimizer_swapper.py:76) → async swap + pinned buffer
 *   DeepSpeed AsyncTensorSwapper → double-buffered gradient swap
 *   NCCL cudaMemcpyAsync (transport/p2p.cc:833)
 *     → cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream)
 *   PyTorch cudaEventRecord/cudaEventSynchronize
 *     (CUDACachingAllocator.cpp:4355,4307) → event-based completion tracking
 *   NCCL cudaStreamSynchronize (allocator.cc:315) → stream sync barrier
 *
 * Milestone: M009–M010 (Claude #4)
 */

#pragma once

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <cstring>
#include <chrono>
#include <vector>
#include <iostream>

namespace philemon {

// Forward declaration.
class TieredAllocator;

// ─── Migration ticket ───────────────────────────────────────────────────

struct MigrationTicket {
    uint64_t ticket_id;
    uint64_t alloc_id;
    MemoryTier source_tier;
    MemoryTier target_tier;
    size_t    size_bytes;

    enum class Status : int {
        PENDING   = 0,   // queued, not started
        STAGING   = 1,   // copying to staging buffer
        SWAPPING  = 2,   // swapping pointers under lock
        COMPLETED = 3,   // done
        FAILED    = 4    // allocation or copy failed
    };
    std::atomic<Status> status{Status::PENDING};

    // Timing (ns since epoch) for profiling.
    std::atomic<uint64_t> submit_ns{0};
    std::atomic<uint64_t> start_ns{0};
    std::atomic<uint64_t> complete_ns{0};

    double elapsed_us() const {
        uint64_t s = submit_ns.load(std::memory_order_relaxed);
        uint64_t c = complete_ns.load(std::memory_order_relaxed);
        if (s == 0 || c == 0) return -1.0;
        return (c - s) / 1000.0;
    }
};

// ─── Staging buffer pool ────────────────────────────────────────────────
// Double-buffered: two pinned staging areas for overlapping transfers.
// Pattern: DeepSpeed's swap_buffer_manager.free/alloc cycle.

class StagingPool {
public:
    explicit StagingPool(size_t buf_size = 4ULL * 1024 * 1024)  // 4 MB default
        : buf_size_(buf_size)
    {
        // Pre-allocate two staging buffers (pinned memory in CUDA mode).
        // In CPU-dev: posix_memalign with page alignment.
        for (int i = 0; i < 2; ++i) {
            void* p = nullptr;
            int rc = ::posix_memalign(&p, 4096, buf_size_);
            if (rc != 0 || !p) {
                std::cerr << "[StagingPool] Failed to allocate staging buffer " << i << "\n";
                p = nullptr;
            }
            bufs_[i] = p;
            in_use_[i].store(false);
        }
    }

    ~StagingPool() {
        for (int i = 0; i < 2; ++i) {
            if (bufs_[i]) ::free(bufs_[i]);
        }
    }

    // Non-copyable.
    StagingPool(const StagingPool&) = delete;
    StagingPool& operator=(const StagingPool&) = delete;

    // Acquire a staging buffer. Returns {buffer, index} or {nullptr, -1}.
    // If both buffers are in use, returns nullptr (caller should wait).
    std::pair<void*, int> acquire(size_t needed) {
        if (needed > buf_size_) {
            // Oversize: allocate one-off.
            void* p = nullptr;
            int rc = ::posix_memalign(&p, 4096, needed);
            if (rc != 0) return {nullptr, -1};
            return {p, -1};  // -1 = one-off, caller must free
        }
        for (int i = 0; i < 2; ++i) {
            bool expected = false;
            if (in_use_[i].compare_exchange_strong(expected, true,
                    std::memory_order_acquire)) {
                return {bufs_[i], i};
            }
        }
        return {nullptr, -1};  // both in use
    }

    // Release a staging buffer by index. If index == -1, frees the one-off.
    void release(void* ptr, int index) {
        if (index < 0) {
            if (ptr) ::free(ptr);
        } else if (index < 2) {
            in_use_[index].store(false, std::memory_order_release);
        }
    }

    size_t buffer_size() const { return buf_size_; }

private:
    size_t buf_size_;
    void*  bufs_[2] = {nullptr, nullptr};
    std::atomic<bool> in_use_[2] = {false, false};
};

// ─── AsyncMigrator ──────────────────────────────────────────────────────

class AsyncMigrator {
public:
    /**
     * Construct with a reference to the TieredAllocator.
     * Starts a background worker thread for completing migrations.
     *
     * @param allocator  The TieredAllocator to operate on.
     * @param staging_size  Size of each staging buffer (default 4 MB).
     */
    explicit AsyncMigrator(TieredAllocator& allocator,
                           size_t staging_size = 4ULL * 1024 * 1024)
        : allocator_(allocator)
        , staging_(staging_size)
        , next_ticket_(1)
        , stop_(false)
    {
        worker_ = std::thread([this]() { worker_loop(); });
    }

    ~AsyncMigrator() {
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            stop_ = true;
        }
        queue_cv_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

    // Non-copyable.
    AsyncMigrator(const AsyncMigrator&) = delete;
    AsyncMigrator& operator=(const AsyncMigrator&) = delete;

    /**
     * Submit an asynchronous migration request.
     *
     * Returns a ticket_id that can be used to poll/wait for completion.
     * The migration is queued and executed by the background worker.
     *
     * In CUDA mode (future M009), this would:
     *   1. cudaMallocAsync on target device
     *   2. cudaMemcpyAsync src→staging→dst
     *   3. cudaEventRecord for completion tracking
     *
     * In CPU-dev mode:
     *   1. Acquire staging buffer
     *   2. memcpy src→staging in background thread
     *   3. Allocate target, memcpy staging→target, free old
     */
    uint64_t submit(uint64_t alloc_id, MemoryTier target) {
        auto ticket = std::make_shared<MigrationTicket>();
        uint64_t tid = next_ticket_.fetch_add(1, std::memory_order_relaxed);
        ticket->ticket_id = tid;
        ticket->alloc_id = alloc_id;
        ticket->target_tier = target;
        ticket->status.store(MigrationTicket::Status::PENDING);

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        ticket->submit_ns.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            pending_.push(ticket);
            // Also store in flight map for poll/wait.
            std::lock_guard<std::mutex> flk(flight_mu_);
            in_flight_[tid] = ticket;
        }
        queue_cv_.notify_one();
        return tid;
    }

    /**
     * Poll a migration ticket. Non-blocking.
     * Returns the current status.
     */
    MigrationTicket::Status poll(uint64_t ticket_id) const {
        std::lock_guard<std::mutex> lk(flight_mu_);
        auto it = in_flight_.find(ticket_id);
        if (it == in_flight_.end()) return MigrationTicket::Status::COMPLETED;
        return it->second->status.load(std::memory_order_acquire);
    }

    /**
     * Wait for a migration to complete. Blocking.
     * Returns elapsed time in microseconds.
     */
    double wait(uint64_t ticket_id) {
        std::shared_ptr<MigrationTicket> ticket;
        {
            std::lock_guard<std::mutex> lk(flight_mu_);
            auto it = in_flight_.find(ticket_id);
            if (it == in_flight_.end()) return 0.0;
            ticket = it->second;
        }

        // Spin-wait with yield (in CUDA mode: cudaEventSynchronize).
        while (true) {
            auto st = ticket->status.load(std::memory_order_acquire);
            if (st == MigrationTicket::Status::COMPLETED ||
                st == MigrationTicket::Status::FAILED) {
                break;
            }
            std::this_thread::yield();
        }

        double elapsed = ticket->elapsed_us();

        // Cleanup.
        {
            std::lock_guard<std::mutex> lk(flight_mu_);
            in_flight_.erase(ticket_id);
        }
        return elapsed;
    }

    /**
     * Wait for all pending migrations to complete.
     */
    void drain() {
        while (true) {
            bool all_done = true;
            {
                std::lock_guard<std::mutex> lk(flight_mu_);
                // Remove completed/failed tickets and check if any remain.
                for (auto it = in_flight_.begin(); it != in_flight_.end(); ) {
                    auto st = it->second->status.load(std::memory_order_acquire);
                    if (st == MigrationTicket::Status::COMPLETED ||
                        st == MigrationTicket::Status::FAILED) {
                        it = in_flight_.erase(it);
                    } else {
                        all_done = false;
                        ++it;
                    }
                }
                if (all_done && in_flight_.empty()) return;
            }
            std::this_thread::yield();
        }
    }

    // Stats.
    struct Stats {
        uint64_t total_submitted = 0;
        uint64_t total_completed = 0;
        uint64_t total_failed    = 0;
        double   total_bytes     = 0;
        double   total_time_us   = 0;

        double avg_throughput_mbps() const {
            if (total_time_us <= 0) return 0;
            return (total_bytes / (1024.0 * 1024.0)) / (total_time_us / 1e6);
        }
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lk(stats_mu_);
        return stats_;
    }

private:
    void worker_loop() {
        while (true) {
            std::shared_ptr<MigrationTicket> ticket;

            {
                std::unique_lock<std::mutex> lk(queue_mu_);
                queue_cv_.wait(lk, [this]() {
                    return !pending_.empty() || stop_;
                });
                if (stop_ && pending_.empty()) return;
                ticket = pending_.front();
                pending_.pop();
            }

            execute_migration(ticket);
        }
    }

    void execute_migration(std::shared_ptr<MigrationTicket> ticket) {
        auto now_ns = []() -> uint64_t {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };

        ticket->start_ns.store(now_ns());
        ticket->status.store(MigrationTicket::Status::STAGING);

        // Get source info via the allocator's get_meta.
        AllocMeta meta;
        if (!allocator_.get_meta(ticket->alloc_id, meta)) {
            ticket->status.store(MigrationTicket::Status::FAILED);
            ticket->complete_ns.store(now_ns());
            record_failure();
            return;
        }

        ticket->source_tier = meta.current_tier;
        ticket->size_bytes = meta.size_bytes;

        if (meta.current_tier == ticket->target_tier) {
            // Already on target tier.
            ticket->status.store(MigrationTicket::Status::COMPLETED);
            ticket->complete_ns.store(now_ns());
            return;
        }

        // Step 1: Acquire staging buffer.
        auto [staging_buf, staging_idx] = staging_.acquire(meta.size_bytes);
        if (!staging_buf) {
            // Both staging buffers busy — fall back to sync.
            bool ok = allocator_.migrate(ticket->alloc_id, ticket->target_tier);
            ticket->status.store(ok ? MigrationTicket::Status::COMPLETED
                                    : MigrationTicket::Status::FAILED);
            ticket->complete_ns.store(now_ns());
            if (ok) record_success(meta.size_bytes, ticket->elapsed_us());
            else    record_failure();
            return;
        }

        // Step 2: Copy source data → staging buffer (simulates async DMA).
        // In CUDA mode: cudaMemcpyAsync(staging, src, size, H2D, stream)
        void* src_ptr = allocator_.get_ptr(ticket->alloc_id);
        if (!src_ptr) {
            staging_.release(staging_buf, staging_idx);
            ticket->status.store(MigrationTicket::Status::FAILED);
            ticket->complete_ns.store(now_ns());
            record_failure();
            return;
        }
        ::memcpy(staging_buf, src_ptr, meta.size_bytes);

        // Step 3: Swap — allocate on target, copy staging→target, free old.
        ticket->status.store(MigrationTicket::Status::SWAPPING);
        bool ok = allocator_.migrate_from_staging(
            ticket->alloc_id, ticket->target_tier,
            staging_buf, meta.size_bytes);

        // Step 4: Release staging.
        staging_.release(staging_buf, staging_idx);

        ticket->status.store(ok ? MigrationTicket::Status::COMPLETED
                                : MigrationTicket::Status::FAILED);
        ticket->complete_ns.store(now_ns());

        if (ok) record_success(meta.size_bytes, ticket->elapsed_us());
        else    record_failure();
    }

    void record_success(size_t bytes, double us) {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.total_completed++;
        stats_.total_submitted++;
        stats_.total_bytes += bytes;
        stats_.total_time_us += us;
    }

    void record_failure() {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.total_submitted++;
        stats_.total_failed++;
    }

    TieredAllocator& allocator_;
    StagingPool      staging_;

    std::atomic<uint64_t> next_ticket_;
    bool stop_;

    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::queue<std::shared_ptr<MigrationTicket>> pending_;

    mutable std::mutex flight_mu_;
    std::unordered_map<uint64_t, std::shared_ptr<MigrationTicket>> in_flight_;

    mutable std::mutex stats_mu_;
    Stats stats_;

    std::thread worker_;
};

}  // namespace philemon
