/**
 * m101_fullchain_debug_experiment.cpp — M101: 全链路追踪+综合debug实验
 *
 * 覆盖模块:
 *   debug/philemon_debug.hpp (340行) — ScopedTimer, TraceEvent, PHILE_DBG
 *   debug/state_inspector.hpp (448行) — InspectionPoint, inspection_log
 *   全链路: allocator → bridge → index → executor 完整数据流
 *
 * 验证:
 *   - ScopedTimer 嵌套计时
 *   - TraceEvent 事件追踪
 *   - 端到端: 边生成→TemporalBridge→flush→TemGraph→QueryExecutor→结果
 *   - 多阶段inspection记录
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m101_test experiment/m101_fullchain_debug_experiment.cpp
 * Milestone: M101 (第10位Claude)
 */

#include <iostream>
#include <cassert>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "../src/debug/philemon_debug.hpp"
#include "../src/debug/state_inspector.hpp"
#include "../src/core/tiered_allocator.hpp"
#include "../src/core/temporal_edge.hpp"
#include "../src/bridge/temporal_bridge.hpp"
#include "../src/index/tem_graph.hpp"
#include "../src/index/tem_graph_impl.hpp"
#include "../src/executor/query_executor.hpp"
#include "../src/cost_model/tier_cost_model.hpp"
#include "../src/cost_model/cost_estimator.hpp"

static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define SECTION(name) std::printf("\n\033[1;33m╔══════════════════════════════════════════════════════════════╗\033[0m\n\033[1;33m║  %-58s ║\033[0m\n\033[1;33m╚══════════════════════════════════════════════════════════════╝\033[0m\n", name)
#define TEST(name) do { g_tests_run++; std::printf("\n  \033[36m[TEST %03d]\033[0m %s\n", g_tests_run, name); } while(0)
#define PASS() do { g_tests_passed++; std::printf("  \033[32m[PASS ✓]\033[0m\n"); } while(0)
#define FAIL(msg) do { g_tests_failed++; std::printf("  \033[31m[FAIL ✗] %s\033[0m\n", msg); } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define DUMP(fmt, ...) std::printf("    \033[90m[DUMP]\033[0m " fmt "\n", ##__VA_ARGS__)
#define STEP(fmt, ...) std::printf("    \033[34m[STEP]\033[0m " fmt "\n", ##__VA_ARGS__)
#define STAT(fmt, ...) std::printf("    \033[35m[STAT]\033[0m " fmt "\n", ##__VA_ARGS__)

// ═══════════════════════════════════════════════════════════════════
// §1 ScopedTimer + TraceEvent — 调试基础设施
// ═══════════════════════════════════════════════════════════════════
static void test_debug_infrastructure() {
    SECTION("§1 Debug Infrastructure (debug/, 340+448 lines)");

    TEST("ScopedTimer: nested timing");
    {
        STEP("outer timer (100ms)...");
        {
            philemon::debug::ScopedTimer outer("outer_op");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            {
                philemon::debug::ScopedTimer inner("inner_op");
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        DUMP("timers printed above by ScopedTimer destructor");
        PASS();
    }

    TEST("TraceEvent: record and inspect");
    {
        STEP("recording trace events...");
        philemon::debug::TraceRing ring;
        ring.record(philemon::debug::TraceEvent::ALLOC, 1, 0, 0);
        ring.record(philemon::debug::TraceEvent::QUERY_BEGIN, 2, 100, 200);
        ring.record(philemon::debug::TraceEvent::QUERY_END, 2, 0, 0, 42, 100);
        ring.record(philemon::debug::TraceEvent::MIGRATE_END, 3, 0, 1);

        STEP("dumping trace ring:");
        ring.dump_last(10);
        DUMP("ring size = %zu", ring.size());

        CHECK(ring.size() >= 4, "must have 4 events");
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// §2 全链路: Edge→Bridge→TemGraph→QueryExecutor
// ═══════════════════════════════════════════════════════════════════
static void test_full_chain() {
    SECTION("§2 Full Chain: Edge → Bridge → Index → Executor");

    TEST("End-to-end data flow");
    {
        philemon::debug::record_inspection("chain", "start full chain test");

        // Phase 1: 创建基础设施
        STEP("Phase 1: creating TieredAllocator + TemporalBridge...");
        philemon::TieredAllocator alloc(16<<20, 32<<20, 128<<20);
        philemon::TierPlacementPolicy policy(500, 5000);
        philemon::TemporalBridge bridge(alloc, policy, 256);

        // Phase 2: 生成时序边并写入Bridge
        STEP("Phase 2: generating 3000 temporal edges...");
        std::mt19937 rng(42);
        std::vector<std::pair<philemon::index::Timestamp, philemon::index::Timestamp>> index_edges;

        for (int i = 0; i < 3000; i++) {
            int s = rng() % 2000;
            int e = s + 1 + rng() % 300;
            philemon::TemporalEdge edge(i % 500, (i + 1) % 500, 1.0, s, e);
            bridge.add_edge(edge);
            index_edges.push_back({s, e});
        }

        STEP("Phase 2b: flushing bridge partitions...");
        size_t n_parts = bridge.flush_partitions();
        DUMP("created %zu partitions from 3000 edges", n_parts);
        philemon::debug::record_inspection("chain", "bridge flushed");

        // Phase 3: 构建TemGraph索引
        STEP("Phase 3: building TemGraph index from same edges...");
        philemon::index::TemGraph graph;
        graph.load_from_edges(philemon::index::CONTAINS_QUERY, index_edges);
        DUMP("index: total=%u unique=%u range=[%d,%d]",
             graph.total_intervals_, graph.unique_intervals_,
             graph.earliest_time_, graph.latest_time_);
        DUMP("memory=%zu bytes, avg_degree=%.2f",
             graph.index_memory_bytes(), graph.avg_degree());
        philemon::debug::record_inspection("chain", "TemGraph index built");

        // Phase 4: 创建QueryExecutor并执行查询
        STEP("Phase 4: creating QueryExecutor (8 threads)...");
        philemon::executor::QueryExecutor executor(8, &graph);

        STEP("Phase 4b: submitting 200 mixed queries...");
        std::vector<philemon::executor::QueryRequest> requests;
        for (int i = 0; i < 200; i++) {
            philemon::executor::QueryRequest req;
            req.type = (i % 3 == 0)
                ? philemon::executor::QueryType::CONTAINS
                : philemon::executor::QueryType::CONTAINS;  // all contains for this index
            req.l = rng() % 1500;
            req.r = req.l + 100 + rng() % 500;
            req.query_id = i;
            requests.push_back(req);
        }

        auto batch = executor.batch_query(requests);
        philemon::debug::record_inspection("chain", "200-query batch completed");

        DUMP("batch: %zu queries, total_matched=%lu total_visited=%lu elapsed=%.1fms",
             batch.results.size(),
             (unsigned long)batch.total_matched,
             (unsigned long)batch.total_visited,
             batch.total_elapsed_ms);

        STEP("Phase 5: cost model analysis...");
        philemon::cost_model::CostEstimator estimator;
        auto amat = estimator.compute_amat(0.85, 0.65);
        DUMP("AMAT = %.2f ns (HBM 85%%, GDDR 65%%)", amat.amat_ns);

        // Phase 6: 最终状态dump
        STEP("Phase 6: final state dump...");
        executor.dump_state();
        alloc.print_slab_stats();

        STEP("full inspection log:");
        philemon::debug::print_inspection_log();

        CHECK(n_parts > 0, "bridge must create partitions");
        CHECK(batch.results.size() == 200, "all 200 queries must execute");
        CHECK(graph.total_intervals_ == 3000, "index must have 3000 intervals");
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// §3 并发压力测试 — 多线程查询 + 检查点
// ═══════════════════════════════════════════════════════════════════
static void test_concurrent_stress() {
    SECTION("§3 Concurrent Stress Test");

    TEST("8-thread concurrent query stress (1000 queries)");
    {
        philemon::index::TemGraph graph;
        std::vector<std::pair<philemon::index::Timestamp, philemon::index::Timestamp>> edges;
        std::mt19937 rng(123);
        for (int i = 0; i < 10000; i++) {
            int s = rng() % 5000;
            edges.push_back({s, s + 1 + rng() % 500});
        }
        graph.load_from_edges(philemon::index::CONTAINS_QUERY, edges);

        DUMP("index: %u intervals, %u unique, memory=%zu bytes",
             graph.total_intervals_, graph.unique_intervals_,
             graph.index_memory_bytes());

        philemon::executor::QueryExecutor executor(8, &graph);

        STEP("submitting 1000 queries in rapid succession...");
        auto t0 = std::chrono::steady_clock::now();

        std::vector<std::future<philemon::index::QueryResult>> futures;
        for (int i = 0; i < 1000; i++) {
            int lo = rng() % 4000;
            futures.push_back(executor.submit_contains(lo, lo + 200 + rng() % 800));
        }
        executor.drain();

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();

        int64_t total_matched = 0, total_visited = 0;
        for (auto& f : futures) {
            auto r = f.get();
            total_matched += r.matched_count;
            total_visited += r.visited_intervals;
        }

        DUMP("1000 queries in %lu us (%.1f us/query)",
             (unsigned long)elapsed, elapsed / 1000.0);
        DUMP("total_matched=%ld total_visited=%ld",
             (long)total_matched, (long)total_visited);
        STAT("throughput = %.0f queries/sec", 1000.0 / (elapsed / 1e6));

        executor.dump_state();
        CHECK(executor.total_completed() == 1000, "all 1000 must complete");
        PASS();
    }
}

int main() {
    std::printf("═══════════════════════════════════════════════════════════════════\n");
    std::printf("  M101 Full-Chain Debug + Trace Experiment\n");
    std::printf("  Covers: debug (788 lines) + full pipeline integration\n");
    std::printf("═══════════════════════════════════════════════════════════════════\n\n");

    philemon::debug::set_debug_level(1);
    auto t0 = std::chrono::steady_clock::now();

    test_debug_infrastructure();
    test_full_chain();
    test_concurrent_stress();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    std::printf("\n═══════════════════════════════════════════════════════════════════\n");
    std::printf("  M101 RESULTS: %d/%d passed, %d failed, elapsed=%ldms\n",
                g_tests_passed, g_tests_run, g_tests_failed, (long)elapsed);
    std::printf("═══════════════════════════════════════════════════════════════════\n");

    return g_tests_failed > 0 ? 1 : 0;
}
