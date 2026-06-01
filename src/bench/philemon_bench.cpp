/**
 * philemon_bench.cpp — 端到端集成基准测试（带全量断点调试）
 *
 * Wires TieredAllocator + TemporalBridge + MigrationScheduler into an
 * end-to-end pipeline that:
 *   1. Generates synthetic temporal edges (LDBC-like power-law).
 *   2. Partitions and places them on tiered memory.
 *   3. Runs temporal subgraph queries.
 *   4. Triggers migration sweeps.
 *   5. Reports throughput, latency, memory tier usage.
 *   6. M005: Measures concurrent query throughput (multi-threaded).
 *   7. M006: Compares binary-search vs linear-scan latency.
 *   8. M009: Tests RAII TierPtr and async migration pipeline.
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -I src -o philemon_bench src/bench/philemon_bench.cpp
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2)
 */

#include "../core/tiered_allocator.hpp"
#include "../core/async_migrator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include "../scheduler/migration_scheduler.hpp"
#include "../debug/philemon_debug.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <thread>
#include <vector>
#include <sys/resource.h>

using namespace philemon;

// ─── Utility: peak RSS ──────────────────────────────────────────────────────
static double get_peak_rss_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss / 1024.0;
    }
    return -1.0;
}

// ─── Synthetic edge generator ───────────────────────────────────────────────
static std::vector<TemporalEdge>
generate_edges(size_t n, int32_t ts_min, int32_t ts_max, uint64_t max_vertex) {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int32_t> ts_dist(ts_min, ts_max);
    std::vector<TemporalEdge> edges;
    edges.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        double u1 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        double u2 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        uint64_t src = static_cast<uint64_t>(std::pow(u1, 2.0) * max_vertex);
        uint64_t dst = static_cast<uint64_t>(std::pow(u2, 2.0) * max_vertex);
        if (src == dst) dst = (dst + 1) % max_vertex;

        int32_t t0 = ts_dist(rng);
        int32_t dur = std::uniform_int_distribution<int32_t>(1, 100)(rng);
        int32_t t1  = std::min(t0 + dur, ts_max);

        edges.emplace_back(src, dst, 1.0, t0, t1);
    }
    return edges;
}


int main() {
    const char* dbg_env = std::getenv("PHILEMON_DEBUG");
    int dbg_level = dbg_env ? std::atoi(dbg_env) : 1;
    philemon::debug::set_debug_level(dbg_level);

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "   Philemon-TSH — 异构内存时间子图处理系统\n";
    std::cout << "   端到端基准测试 (debug_level=" << dbg_level << ")\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";

    // ── 1. Configure tier budgets ─────────────────────────────────────────
    const size_t HBM_CAP  =  512ULL * 1024 * 1024;
    const size_t GDDR_CAP = 1024ULL * 1024 * 1024;
    const size_t DRAM_CAP = 2048ULL * 1024 * 1024;

    TieredAllocator allocator(HBM_CAP, GDDR_CAP, DRAM_CAP);

    // ── 2. Configure placement policy ─────────────────────────────────────
    TierPlacementPolicy policy(
        100'000'000ULL,   // 100 ms hot threshold
        1'000'000'000ULL  // 1 s warm threshold
    );

    // ── 3. Create bridge ──────────────────────────────────────────────────
    const size_t PARTITION_SIZE = 100'000;
    TemporalBridge bridge(allocator, policy, PARTITION_SIZE);

    // ── 4. Generate + ingest edges ────────────────────────────────────────
    const size_t NUM_EDGES    = 1'000'000;
    const int32_t TS_MIN      = 0;
    const int32_t TS_MAX      = 10'000;
    const uint64_t MAX_VERTEX = 100'000;

    std::cout << "[1] Generating " << NUM_EDGES << " synthetic temporal edges...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto edges = generate_edges(NUM_EDGES, TS_MIN, TS_MAX, MAX_VERTEX);
    auto t1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Generated in " << std::fixed << std::setprecision(2)
              << gen_ms << " ms\n";

    if (dbg_level >= 1) {
        std::printf("    ── 边数据采样（首/尾各5条）──\n");
        for (size_t i = 0; i < 5 && i < edges.size(); i++)
            edges[i].dump_edge("    HEAD");
        for (size_t i = edges.size() > 5 ? edges.size()-5 : 0; i < edges.size(); i++)
            edges[i].dump_edge("    TAIL");
    }
    PHILE_CHECKPOINT("after_edge_generation");

    // ── 5. Ingest into bridge ─────────────────────────────────────────────
    std::cout << "[2] Ingesting edges into TemporalBridge...\n";
    t0 = std::chrono::high_resolution_clock::now();
    bridge.add_edges(edges);
    t1 = std::chrono::high_resolution_clock::now();
    double ingest_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Ingested in " << std::fixed << std::setprecision(2)
              << ingest_ms << " ms\n\n";

    // ── 6. Partition + tier-place ─────────────────────────────────────────
    std::cout << "[3] Flushing partitions (sort + allocate + place)...\n";
    t0 = std::chrono::high_resolution_clock::now();
    size_t nparts = bridge.flush_partitions();
    t1 = std::chrono::high_resolution_clock::now();
    double part_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Created " << nparts << " partitions in "
              << std::fixed << std::setprecision(2) << part_ms << " ms\n";

    // Print partition summary
    auto parts_snap = bridge.partitions_snapshot();
    for (auto& p : parts_snap) {
        std::cout << "    Partition alloc=" << p.alloc_id
                  << " ts=[" << p.ts_lo << "," << p.ts_hi << "]"
                  << " edges=" << p.edge_count
                  << " tier=" << tier_name(p.tier()) << "\n";
    }

    // Memory usage
    std::cout << "\n    Tier usage:\n";
    for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
        auto& b = allocator.budget(static_cast<MemoryTier>(i));
        double used_mb = b.used_bytes.load() / (1024.0 * 1024.0);
        double cap_mb  = b.capacity_bytes / (1024.0 * 1024.0);
        std::cout << "      " << tier_name(static_cast<MemoryTier>(i))
                  << ": " << std::fixed << std::setprecision(2)
                  << used_mb << " / " << cap_mb << " MB\n";
    }

    // ── 7. Temporal subgraph queries (M006: binary search) ────────────────
    std::cout << "\n[4] Running temporal subgraph queries (M006: binary search)...\n";

    struct QuerySpec {
        const char* label;
        int32_t lo, hi;
    };
    QuerySpec queries[] = {
        {"narrow [1000,1050]", 1000, 1050},
        {"medium [2000,3000]", 2000, 3000},
        {"wide   [0,5000]",    0,    5000},
        {"full   [0,10000]",   0,    10000},
    };

    for (auto& q : queries) {
        uint64_t count = 0;
        t0 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 100; ++iter) {
            count = 0;
            bridge.temporal_subgraph_query(q.lo, q.hi,
                [&count](const TemporalEdge& e) {
                    ++count;
                });
        }
        t1 = std::chrono::high_resolution_clock::now();
        double q_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

        std::cout << "    " << q.label
                  << "  →  " << count << " edges"
                  << "  in " << std::fixed << std::setprecision(1) << q_us << " µs"
                  << "  (" << std::setprecision(2)
                  << (count > 0 ? q_us * 1000.0 / count : 0.0)
                  << " ns/edge)\n";
    }

    // ── 8. M005: Concurrent query throughput test ─────────────────────────
    std::cout << "\n[5] M005: Concurrent query throughput (4 threads, 10K queries each)...\n";
    {
        const int NUM_THREADS = 4;
        const int QUERIES_PER_THREAD = 10'000;
        std::atomic<uint64_t> total_edges_found{0};

        t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int tid = 0; tid < NUM_THREADS; ++tid) {
            threads.emplace_back([&, tid]() {
                std::mt19937 rng(tid * 1000 + 42);
                std::uniform_int_distribution<int32_t> lo_dist(0, 9000);
                uint64_t local_count = 0;
                for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
                    int32_t lo = lo_dist(rng);
                    int32_t hi = lo + 50 + (q % 200);  // varying window sizes
                    bridge.temporal_subgraph_query(lo, hi,
                        [&local_count](const TemporalEdge& e) {
                            ++local_count;
                        });
                }
                total_edges_found.fetch_add(local_count, std::memory_order_relaxed);
            });
        }
        for (auto& t : threads) t.join();
        t1 = std::chrono::high_resolution_clock::now();

        double conc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        uint64_t total_queries = NUM_THREADS * QUERIES_PER_THREAD;
        double qps = total_queries * 1000.0 / conc_ms;

        std::cout << "    " << total_queries << " queries in "
                  << std::fixed << std::setprecision(1) << conc_ms << " ms"
                  << " (" << std::setprecision(0) << qps << " QPS)"
                  << "  total edges scanned: " << total_edges_found.load() << "\n";
    }

    // ── 9. Migration sweep ────────────────────────────────────────────────
    std::cout << "\n[6] Running migration sweep...\n";
    MigrationScheduler scheduler(bridge, std::chrono::milliseconds(100));
    size_t migrated = scheduler.sweep_once();
    std::cout << "    Migrated " << migrated << " partitions\n";
    scheduler.stats().print();

    // Re-check tier usage after migration
    std::cout << "\n    Tier usage after migration:\n";
    for (int i = 0; i < static_cast<int>(MemoryTier::TIER_COUNT); ++i) {
        auto& b = allocator.budget(static_cast<MemoryTier>(i));
        double used_mb = b.used_bytes.load() / (1024.0 * 1024.0);
        double cap_mb  = b.capacity_bytes / (1024.0 * 1024.0);
        std::cout << "      " << tier_name(static_cast<MemoryTier>(i))
                  << ": " << std::fixed << std::setprecision(2)
                  << used_mb << " / " << cap_mb << " MB\n";
    }

    // ── 10. M008: Slab allocator statistics ──────────────────────────────
    std::cout << "\n[7] M008: Slab allocator statistics...\n";
    allocator.print_slab_stats();

    // ── 11. M008: Compact slabs after migration ────────────────────────
    std::cout << "\n[8] M008: Compacting slab pools...\n";
    size_t compacted = allocator.compact_slabs();
    std::cout << "    Released " << (compacted / 1024) << " KB from empty slab pages\n";

    // ── 12. M007: Adaptive partitioning test with skewed data ──────────
    {
        std::cout << "\n[9] M007: Adaptive partitioning (skewed temporal distribution)...\n";
        TieredAllocator alloc2(HBM_CAP, GDDR_CAP, DRAM_CAP);
        // M007: Use smaller partition cap to show adaptive behavior
        TemporalBridge bridge2(alloc2, policy, 10000 /*10K edges per partition*/);

        // Generate skewed data: 90% of edges in last 10% of time range
        std::vector<TemporalEdge> skewed;
        skewed.reserve(100000);
        std::mt19937_64 rng2(123);
        for (size_t i = 0; i < 10000; ++i) {
            // 10% sparse: timestamps 0-9000
            int32_t t0 = std::uniform_int_distribution<int32_t>(0, 9000)(rng2);
            int32_t t1 = t0 + std::uniform_int_distribution<int32_t>(1, 50)(rng2);
            skewed.emplace_back(i % 1000, (i + 1) % 1000, 1.0, t0, std::min(t1, 10000));
        }
        for (size_t i = 0; i < 90000; ++i) {
            // 90% dense: timestamps 9000-10000
            int32_t t0 = std::uniform_int_distribution<int32_t>(9000, 10000)(rng2);
            int32_t t1 = t0 + std::uniform_int_distribution<int32_t>(1, 20)(rng2);
            skewed.emplace_back(i % 1000, (i + 1) % 1000, 1.0, t0, std::min(t1, 10000));
        }

        bridge2.add_edges(skewed);
        auto t2 = std::chrono::high_resolution_clock::now();
        size_t nparts2 = bridge2.flush_partitions();
        auto t3 = std::chrono::high_resolution_clock::now();
        double part2_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "    Skewed data: 100K edges (10K sparse + 90K dense)\n";
        std::cout << "    Created " << nparts2 << " adaptive partitions in "
                  << std::fixed << std::setprecision(2) << part2_ms << " ms\n";

        auto snap = bridge2.partitions_snapshot();
        for (auto& p : snap) {
            std::cout << "    Partition alloc=" << p.alloc_id
                      << " ts=[" << p.ts_lo << "," << p.ts_hi << "]"
                      << " edges=" << p.edge_count
                      << " tier=" << tier_name(p.tier()) << "\n";
        }

        // Query narrow window in dense region
        uint64_t count2 = 0;
        t2 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; ++iter) {
            count2 = 0;
            bridge2.temporal_subgraph_query(9500, 9550,
                [&count2](const TemporalEdge&) { ++count2; });
        }
        t3 = std::chrono::high_resolution_clock::now();
        double q2_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / 1000.0;
        std::cout << "    Query [9500,9550] in dense zone: " << count2 << " edges in "
                  << std::fixed << std::setprecision(1) << q2_us << " µs\n";
    }

    // ═══ [10] M009: RAII TierPtr test ═══════════════════════════════════
    std::cout << "\n[10] M009: RAII TierPtr (borrow) test...\n";
    {
        // borrow() returns TierPtr holding shared_lock.
        // Pointer is valid only while TierPtr is alive.
        auto partitions = bridge.query_partitions(0, 10000);
        if (!partitions.empty()) {
            uint64_t first_alloc = partitions[0]->alloc_id;
            auto ptr = allocator.borrow<TemporalEdge>(first_alloc);
            if (ptr) {
                size_t n = ptr.count();
                // Safe access under RAII guard.
                uint64_t checksum = 0;
                for (size_t i = 0; i < std::min(n, (size_t)100); ++i) {
                    checksum += ptr.get()[i].source;
                }
                std::cout << "    TierPtr: " << n << " edges, checksum=" << checksum
                          << " (lock held during access)\n";
            }
            // ~TierPtr releases shared_lock here.
        }
        std::cout << "    TierPtr released — lock freed\n";
    }

    // ═══ [11] M009: Async migration test ══════════════════════════════
    std::cout << "\n[11] M009: Async migration test...\n";
    {
        // Create a separate allocator + bridge for async test.
        TieredAllocator async_alloc(512ULL * 1024 * 1024,
                                    1024ULL * 1024 * 1024,
                                    2048ULL * 1024 * 1024);
        TierPlacementPolicy async_pol(100'000'000ULL, 1'000'000'000ULL);
        TemporalBridge async_bridge(async_alloc, async_pol, 50000);

        // Ingest edges.
        auto async_edges = generate_edges(100000, 0, 10000, 100000);
        async_bridge.add_edges(async_edges);
        async_bridge.flush_partitions();

        auto async_parts = async_bridge.query_partitions(0, 10000);
        std::cout << "    " << async_parts.size() << " partitions created\n";

        // Create AsyncMigrator and submit migrations.
        AsyncMigrator migrator(async_alloc, 2ULL * 1024 * 1024);

        std::vector<uint64_t> tickets;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (auto* part : async_parts) {
            if (part->tier() != MemoryTier::HBM) {
                uint64_t tid = migrator.submit(part->alloc_id, MemoryTier::HBM);
                tickets.push_back(tid);
            }
        }
        std::cout << "    Submitted " << tickets.size() << " async migrations\n";

        // Wait for all.
        migrator.drain();

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto stats = migrator.stats();
        std::cout << "    Completed " << stats.total_completed << " migrations in "
                  << std::fixed << std::setprecision(1) << elapsed_ms << " ms\n";
        std::cout << "    Throughput: " << std::setprecision(1)
                  << stats.avg_throughput_mbps() << " MB/s\n";
        if (stats.total_failed > 0) {
            std::cout << "    Failed: " << stats.total_failed << "\n";
        }

        // Verify all on HBM now.
        double hbm_mb = async_alloc.budget(MemoryTier::HBM).used_bytes.load() / (1024.0*1024.0);
        std::cout << "    HBM usage after migration: " << std::setprecision(2)
                  << hbm_mb << " MB\n";
    }

    // ── Summary ───────────────────────────────────────────────────────
    double peak_mb = get_peak_rss_mb();
    PHILE_SEPARATOR("FINAL SUMMARY");
    philemon::debug::global_trace().dump_last(15);
    philemon::debug::print_all_tier_perf();
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "  Peak RSS: " << std::fixed << std::setprecision(1)
              << peak_mb << " MB\n";
    std::cout << "  Total allocated across tiers: "
              << std::setprecision(2)
              << allocator.total_allocated() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    return 0;
}
