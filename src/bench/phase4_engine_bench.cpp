/**
 * phase4_engine_bench.cpp — Phase 4 Engine Integration Benchmark
 *
 * ====================================================================
 * 测试对象:
 *   M023 — PrefetchEngine      (预取引擎)
 *   M024 — LRU Eviction Policy (LRU淘汰策略)
 *   M025 — CompactionEngine    (碎片整理引擎)
 *   M026 — TierRebalancer      (层级重平衡)
 *
 * 骨架来源:
 *   src/bench/cross_tier_bench.cpp  (测试框架结构)
 *   src/bench/integration_bench.cpp (MockGraphBackend)
 *   upstream/rapidstore/wrapper/driver.h (ScopedTimer, execute pattern)
 *
 * 编译: g++ -std=c++17 -O2 -o phase4_bench phase4_engine_bench.cpp \
 *       -I../.. -lpthread
 * 运行: ./phase4_bench [debug_level=1] [num_partitions=100]
 *
 * Milestone: M023-M026 集成benchmark
 * ====================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <functional>
#include <thread>
#include <numeric>
#include <tuple>

// ─── Project includes ────────────────────────────────────────────────
#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../cost_model/tier_cost_model.hpp"
#include "../core/slab_allocator.hpp"
#include "../prefetch/prefetch_engine.hpp"
#include "../eviction/lru_eviction.hpp"
#include "../compaction/compaction_engine.hpp"
#include "../rebalance/tier_rebalancer.hpp"

using namespace philemon;

// ═══════════════════════════════════════════════════════════════════════
// Test harness
// ═══════════════════════════════════════════════════════════════════════
struct TestResult {
    std::string name;
    bool   passed;
    double time_us;
    std::string detail;
};

static std::vector<TestResult> g_results;

template <typename Func>
void run_test(const std::string& name, Func&& fn) {
    std::printf("\n═══ TEST: %s ═══\n", name.c_str());
    auto start = std::chrono::high_resolution_clock::now();
    TestResult result;
    result.name = name;
    try {
        result.detail = fn();
        result.passed = true;
    } catch (const std::exception& e) {
        result.detail = std::string("EXCEPTION: ") + e.what();
        result.passed = false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.time_us = std::chrono::duration_cast<
        std::chrono::microseconds>(end - start).count();
    std::printf("  → %s (%.1fμs) %s\n",
                result.passed ? "PASS" : "FAIL",
                result.time_us,
                result.detail.c_str());
    g_results.push_back(result);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: PrefetchEngine — 查询历史 + 预测
// ═══════════════════════════════════════════════════════════════════════
std::string test_prefetch_engine() {
    cost_model::TierCostModel cost_model;
    prefetch::PrefetchEngine engine(cost_model, 256, 8, 100, 1);

    // 记录查询: record_query(vertex, tier, query_type, ts, part_id, lat)
    for (uint64_t q = 0; q < 50; q++) {
        engine.record_query(q % 10, uint8_t(q % 3), uint8_t(q % 5),
                            q, q % 20, 100.0 + q);
    }

    // 设置模拟迁移回调
    std::atomic<uint64_t> migrate_count{0};
    engine.set_migrate_callback(
        [&](uint64_t, uint8_t, uint8_t, uint64_t) -> bool {
            migrate_count.fetch_add(1);
            return true;
        });

    // 设置分区状态回调
    engine.set_partition_state_callback(
        [&]() -> std::vector<std::tuple<uint64_t, uint8_t, uint64_t, double>> {
            std::vector<std::tuple<uint64_t, uint8_t, uint64_t, double>> parts;
            for (uint64_t i = 0; i < 20; i++) {
                parts.emplace_back(i, uint8_t(i % 3), 1024*1024ULL, 0.0);
            }
            return parts;
        });

    // 验证历史记录
    size_t hist_size = engine.history().recent(50).size();

    char buf[256];
    snprintf(buf, sizeof(buf),
             "queries_recorded=50 history_size=%zu",
             hist_size);
    return std::string(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: LRU Eviction — LRU列表操作 + 淘汰候选
// ═══════════════════════════════════════════════════════════════════════
std::string test_lru_eviction() {
    eviction::LRUEvictionList lru;

    // 插入100个条目 via touch()
    for (uint64_t i = 0; i < 100; i++) {
        lru.touch(i, uint8_t(i % 3), (i + 1) * 1024, i, 50.0 + i);
    }

    // Touch一些条目 (模拟访问)
    for (uint64_t i = 0; i < 20; i++) {
        lru.touch(i * 5, uint8_t((i * 5) % 3), (i + 1) * 1024, 200 + i, 30.0);
    }

    // 获取DRAM tier的淘汰候选
    auto candidates = lru.get_tier_candidates(2, 10);

    // 移除一些
    size_t removed = 0;
    for (auto& c : candidates) {
        lru.remove(c.partition_id);
        removed++;
    }

    size_t remaining = lru.size();

    char buf[256];
    snprintf(buf, sizeof(buf),
             "inserted=100 touched=20 candidates=%zu removed=%zu "
             "remaining=%zu",
             candidates.size(), removed, remaining);
    return std::string(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: CompactionEngine — 碎片检测 + compact
// ═══════════════════════════════════════════════════════════════════════
std::string test_compaction_engine() {
    SlabAllocator allocator;
    compaction::CompactionEngine engine(&allocator, 0.3, 1000, 3);

    // 分配并释放一些内存制造碎片
    std::vector<void*> ptrs;
    for (int i = 0; i < 200; i++) {
        auto [ptr, sz] = allocator.allocate(1024);  // 1KB
        if (ptr) ptrs.push_back(ptr);
    }

    // 释放偶数位置的 (制造碎片)
    size_t freed = 0;
    for (size_t i = 0; i < ptrs.size(); i += 2) {
        allocator.deallocate(ptrs[i]);
        ptrs[i] = nullptr;
        freed++;
    }

    size_t before = allocator.total_slab_bytes();

    // 执行compact
    auto result = engine.compact_once();

    size_t after = allocator.total_slab_bytes();

    // 清理
    for (auto p : ptrs) {
        if (p) allocator.deallocate(p);
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
             "alloc=200 freed=%zu before=%zuKB after=%zuKB "
             "compact_freed=%luKB time=%.1fμs",
             freed, before / 1024, after / 1024,
             (unsigned long)(result.freed_bytes / 1024),
             result.total_time_us);
    return std::string(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 4: TierRebalancer — 不平衡检测 + 重平衡
// ═══════════════════════════════════════════════════════════════════════
std::string test_tier_rebalancer() {
    cost_model::TierCostModel cost_model;
    rebalance::TierRebalancer rebalancer(cost_model, 500, 0.2);

    // 设置tier容量 (较小便于测试)
    rebalancer.set_tier_capacity(0, 1ULL * 1024 * 1024 * 1024);
    rebalancer.set_tier_capacity(1, 2ULL * 1024 * 1024 * 1024);
    rebalancer.set_tier_capacity(2, 8ULL * 1024 * 1024 * 1024);

    // 模拟不平衡: DRAM有很多热数据
    rebalancer.update_tier_usage(0, 200ULL * 1024 * 1024,  10);
    rebalancer.update_tier_usage(1, 500ULL * 1024 * 1024,  20);
    rebalancer.update_tier_usage(2, 6ULL * 1024 * 1024 * 1024, 100);

    // 模拟访问: DRAM分区很热
    for (int i = 0; i < 1000; i++) {
        rebalancer.record_access(2, 500);
    }
    for (int i = 0; i < 50; i++) {
        rebalancer.record_access(0, 10);
    }

    double imbalance = rebalancer.compute_imbalance();

    // 设置分区和迁移回调
    std::atomic<uint64_t> migrate_count{0};
    rebalancer.set_partition_callback(
        [&]() -> std::vector<rebalance::PartitionHeatInfo> {
            std::vector<rebalance::PartitionHeatInfo> parts;
            for (uint64_t i = 0; i < 20; i++) {
                rebalance::PartitionHeatInfo info;
                info.partition_id = i;
                info.current_tier = 2;
                info.bytes = 50ULL * 1024 * 1024;
                info.access_count = 100 + i * 10;
                info.avg_latency_ns = 500.0;
                parts.push_back(info);
            }
            for (uint64_t i = 20; i < 30; i++) {
                rebalance::PartitionHeatInfo info;
                info.partition_id = i;
                info.current_tier = 0;
                info.bytes = 20ULL * 1024 * 1024;
                info.access_count = 2;
                info.avg_latency_ns = 10.0;
                parts.push_back(info);
            }
            return parts;
        });

    rebalancer.set_migrate_callback(
        [&](uint64_t, uint8_t, uint8_t, uint64_t) -> bool {
            migrate_count.fetch_add(1);
            return true;
        });

    auto result = rebalancer.rebalance_once();

    char buf[256];
    snprintf(buf, sizeof(buf),
             "imbalance=%.2f promoted=%lu demoted=%lu "
             "migrated=%lu bytes=%luMB time=%.1fμs",
             imbalance,
             (unsigned long)result.promoted_count,
             (unsigned long)result.demoted_count,
             (unsigned long)migrate_count.load(),
             (unsigned long)(result.bytes_moved / (1024*1024)),
             result.total_time_us);
    return std::string(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 5: 集成测试 — 所有引擎协同
// ═══════════════════════════════════════════════════════════════════════
std::string test_integrated_engines() {
    cost_model::TierCostModel cost_model;
    SlabAllocator allocator;

    prefetch::PrefetchEngine  prefetch(cost_model, 256, 8, 100, 1);
    compaction::CompactionEngine compactor(&allocator, 0.3, 2000, 3);
    rebalance::TierRebalancer rebalancer(cost_model, 1000, 0.2);
    rebalancer.set_tier_capacity(0, 1ULL * 1024 * 1024 * 1024);
    rebalancer.set_tier_capacity(1, 2ULL * 1024 * 1024 * 1024);
    rebalancer.set_tier_capacity(2, 8ULL * 1024 * 1024 * 1024);

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> vertex_dist(0, 999);
    std::uniform_int_distribution<int> tier_dist(0, 2);

    // Phase A: 分配内存 + 记录查询
    std::vector<void*> allocs;
    for (int i = 0; i < 100; i++) {
        auto [ptr, sz] = allocator.allocate(2048);
        if (ptr) allocs.push_back(ptr);
        uint64_t v = vertex_dist(rng);
        prefetch.record_query(v, uint8_t(v % 3), uint8_t(v % 5),
                              i, i % 20, 100.0);
    }

    // Phase B: 释放一些制造碎片
    for (size_t i = 0; i < allocs.size(); i += 3) {
        allocator.deallocate(allocs[i]);
        allocs[i] = nullptr;
    }

    // Phase C: 模拟访问
    for (int i = 0; i < 500; i++) {
        rebalancer.record_access(tier_dist(rng), rng() % 1000);
    }

    // Phase D: compact
    auto compact_result = compactor.compact_once();

    // Phase E: 清理
    for (auto p : allocs) {
        if (p) allocator.deallocate(p);
    }

    char buf[256];
    snprintf(buf, sizeof(buf),
             "alloc=100 queries=100 compact_freed=%zuKB "
             "slab_total=%zuKB",
             (size_t)(compact_result.freed_bytes / 1024),
             allocator.total_slab_bytes() / 1024);
    return std::string(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 6: Debug macros
// ═══════════════════════════════════════════════════════════════════════
std::string test_debug_macros() {
    cost_model::TierCostModel cost_model;
    SlabAllocator allocator;
    compaction::CompactionEngine compactor(&allocator);
    rebalance::TierRebalancer rebalancer(cost_model);
    rebalancer.set_tier_capacity(0, 1ULL * 1024 * 1024 * 1024);
    rebalancer.set_tier_capacity(1, 2ULL * 1024 * 1024 * 1024);
    rebalancer.set_tier_capacity(2, 8ULL * 1024 * 1024 * 1024);

    auto saved = debug::get_debug_level();
    debug::set_debug_level(2);

    PHILE_COMPACT_DUMP(compactor);
    PHILE_REBALANCE_DUMP(rebalancer);
    PHILE_TIER_PROFILE_DUMP(rebalancer);

    {
        PHILE_COMPACT_BREAKPOINT(compactor, "test_bp_compact");
    }
    {
        PHILE_REBALANCE_BREAKPOINT(rebalancer, "test_bp_rebalance");
    }

    debug::set_debug_level(saved);
    return "all debug macros executed";
}

// ═══════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    int debug_level = (argc > 1) ? std::atoi(argv[1]) : 1;
    debug::set_debug_level(debug_level);

    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  Philemon-TSH Phase 4 Engine Integration Benchmark  ║\n");
    std::printf("║  M023-M026: Prefetch/Eviction/Compact/Rebalance     ║\n");
    std::printf("╠══════════════════════════════════════════════════════╣\n");
    std::printf("║  Debug level: %d                                     ║\n",
                debug_level);
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    run_test("T1: PrefetchEngine",      test_prefetch_engine);
    run_test("T2: LRU Eviction",        test_lru_eviction);
    run_test("T3: CompactionEngine",    test_compaction_engine);
    run_test("T4: TierRebalancer",      test_tier_rebalancer);
    run_test("T5: Integrated Engines",  test_integrated_engines);
    run_test("T6: Debug Macros",        test_debug_macros);

    // 汇总
    std::printf("\n╔══════════════════════════════════════════════════════╗\n");
    std::printf("║                   SUMMARY                           ║\n");
    std::printf("╠══════════════════════════════════════════════════════╣\n");
    size_t passed = 0, failed = 0;
    double total_us = 0;
    for (auto& r : g_results) {
        std::printf("  %-30s %s  %.1fμs\n",
                    r.name.c_str(),
                    r.passed ? "PASS" : "FAIL",
                    r.time_us);
        if (r.passed) passed++; else failed++;
        total_us += r.time_us;
    }
    std::printf("╠══════════════════════════════════════════════════════╣\n");
    std::printf("  Total: %zu passed, %zu failed, %.1fμs\n",
                passed, failed, total_us);
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    return (failed > 0) ? 1 : 0;
}
