#pragma once
/**
 * migration_scheduler.hpp — Background thread for cross-tier data migration
 *
 * Follows NCCL's ncclTopoGraph ring/tree scheduling pattern:
 * periodically evaluate partition hotness, issue migrations between
 * HBM ↔ GDDR ↔ DRAM to keep hot temporal subgraphs close to compute.
 *
 * Also draws from Megatron's DistributedDataParallel._make_param_hook
 * pattern of deferred, batched operations.
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2)
 */

#include "../core/tiered_allocator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

namespace philemon {

struct MigrationStats {
    std::atomic<uint64_t> total_migrations{0};
    std::atomic<uint64_t> hbm_to_gddr{0};
    std::atomic<uint64_t> hbm_to_dram{0};
    std::atomic<uint64_t> gddr_to_hbm{0};
    std::atomic<uint64_t> gddr_to_dram{0};
    std::atomic<uint64_t> dram_to_hbm{0};
    std::atomic<uint64_t> dram_to_gddr{0};
    std::atomic<uint64_t> sweep_count{0};

    void print() const {
        std::cout << "[MigrationStats] sweeps=" << sweep_count.load()
                  << " total_migrations=" << total_migrations.load()
                  << " HBM→GDDR=" << hbm_to_gddr.load()
                  << " HBM→DRAM=" << hbm_to_dram.load()
                  << " GDDR→HBM=" << gddr_to_hbm.load()
                  << " GDDR→DRAM=" << gddr_to_dram.load()
                  << " DRAM→HBM=" << dram_to_hbm.load()
                  << " DRAM→GDDR=" << dram_to_gddr.load()
                  << "\n";
    }
};


class MigrationScheduler {
public:
    MigrationScheduler(TemporalBridge& bridge,
                       std::chrono::milliseconds interval_ms = std::chrono::milliseconds(500))
        : bridge_(bridge)
        , interval_(interval_ms)
        , running_(false)
    {}

    ~MigrationScheduler() {
        stop();
    }

    void start() {
        if (running_.load()) return;
        running_.store(true, std::memory_order_release);
        worker_ = std::thread([this]() { run_loop(); });
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    size_t sweep_once() {
        size_t n = bridge_.migration_sweep();
        stats_.sweep_count.fetch_add(1, std::memory_order_relaxed);
        stats_.total_migrations.fetch_add(n, std::memory_order_relaxed);
        return n;
    }

    const MigrationStats& stats() const { return stats_; }

    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

private:
    void run_loop() {
        while (running_.load(std::memory_order_acquire)) {
            sweep_once();
            std::this_thread::sleep_for(interval_);
        }
    }

    TemporalBridge&             bridge_;
    std::chrono::milliseconds   interval_;
    std::atomic<bool>           running_;
    std::thread                 worker_;
    MigrationStats              stats_;
};

}  // namespace philemon
