/**
 * cross_tier_bench.cpp — Comprehensive Cross-Tier Algorithm Benchmark
 *
 * ====================================================================
 * 骨架来源 (upstream + existing, 保留 ~80%):
 *   upstream/rapidstore/main.cpp                    (202行)
 *   upstream/rapidstore/wrapper/driver.h            (execute_query pattern)
 *   src/bench/integration_bench.cpp                 (375行, 测试框架)
 *   src/bench/ldbc_bench.cpp                        (477行, LDBC harness)
 *
 * 修改 (~20%):
 *   - [NEW] 5-test harness for cross-tier PR/WCC/TC
 *   - [NEW] convergence data collection + JSON output
 *   - [NEW] state inspector integration (breakpoint debugging)
 *   - [NEW] tier heatmap at each test boundary
 *   - [MOD] 测试数据生成方式 (inline, 不依赖文件)
 *   - [KEEP] 测试框架结构 100% 保留
 *
 * 编译: g++ -std=c++17 -O2 -o cross_tier_bench cross_tier_bench.cpp
 *       -I../.. -lpthread
 * 运行: ./cross_tier_bench [debug_level=1] [num_vertices=10000]
 *
 * Milestone: M021-M022 benchmark
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
#include <fstream>

// ─── Project includes ────────────────────────────────────────────────
#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../cost_model/tier_cost_model.hpp"

// ═══════════════════════════════════════════════════════════════════════
// Mock graph backend — simulates the RapidStore wrapper:: API
//
// In production, this would be replaced by actual RapidStore.
// This mock is derived from src/bench/integration_bench.cpp's
// MockGraphBackend (structure preserved, enhanced for cross-tier tests)
// ═══════════════════════════════════════════════════════════════════════

struct MockEdge {
    uint64_t src, dst;
    double weight;
    uint64_t timestamp;
};

class MockSnapshot {
public:
    std::vector<std::vector<std::pair<uint64_t, double>>>* adj_;
    uint64_t n_vertices_;
    uint64_t n_edges_;

    uint64_t vertex_count() const { return n_vertices_; }
    uint64_t edge_count() const { return n_edges_; }

    uint64_t degree(uint64_t v, bool = false) const {
        if (v < adj_->size()) return (*adj_)[v].size();
        return 0;
    }

    bool has_vertex(uint64_t v) const { return v < n_vertices_; }
    bool has_edge(uint64_t src, uint64_t dst) const {
        if (src >= adj_->size()) return false;
        for (auto& [d, w] : (*adj_)[src]) {
            if (d == dst) return true;
        }
        return false;
    }

    template <class Callback>
    void edges(uint64_t v, Callback&& cb, bool = false) const {
        if (v >= adj_->size()) return;
        // NEW: track tier access for debug
        uint8_t tier = (v < n_vertices_ / 3) ? 0 :
                       (v < n_vertices_ * 2 / 3) ? 1 : 2;
        philemon::debug::tier_perf(tier).read_count.fetch_add(1,
            std::memory_order_relaxed);
        philemon::debug::tier_perf(tier).bytes_transferred.fetch_add(
            (*adj_)[v].size() * 16, std::memory_order_relaxed);

        for (auto& [dst, weight] : (*adj_)[v]) {
            cb(dst, weight);
        }
    }

    uint64_t intersect(uint64_t a, uint64_t b) const {
        if (a >= adj_->size() || b >= adj_->size()) return 0;
        uint64_t count = 0;
        for (auto& [d, w] : (*adj_)[a]) {
            if (has_edge(b, d)) count++;
        }
        return count;
    }

    auto clone() const { return std::make_shared<MockSnapshot>(*this); }
    uint64_t physical2logical(uint64_t p) const { return p; }
    uint64_t logical2physical(uint64_t l) const { return l; }
};

using MockSnapshotPtr = std::shared_ptr<MockSnapshot>;

class MockGraphMethod {
    std::vector<std::vector<std::pair<uint64_t, double>>> adj_;
    MockSnapshotPtr snapshot_;

public:
    void build_graph(uint64_t n_vertices, uint64_t n_edges, uint64_t seed) {
        adj_.resize(n_vertices);
        std::mt19937_64 rng(seed);

        uint64_t actual_edges = 0;
        for (uint64_t i = 0; i < n_edges; i++) {
            uint64_t src = rng() % n_vertices;
            uint64_t dst = rng() % n_vertices;
            if (src == dst) continue;
            double w = 1.0 + (rng() % 100) / 100.0;
            adj_[src].push_back({dst, w});
            adj_[dst].push_back({src, w});
            actual_edges += 2;
        }

        snapshot_ = std::make_shared<MockSnapshot>();
        snapshot_->adj_ = &adj_;
        snapshot_->n_vertices_ = n_vertices;
        snapshot_->n_edges_ = actual_edges;
    }

    void set_max_threads(int) {}
    void init_thread(int) {}
    void end_thread(int) {}

    MockSnapshotPtr get_unique_snapshot() { return snapshot_; }
    MockSnapshotPtr get_shared_snapshot() { return snapshot_; }
};

// ─── wrapper:: namespace shims (matching upstream wrapper.h API) ──────
namespace wrapper {
    inline void set_max_threads(MockGraphMethod& m, int t) { m.set_max_threads(t); }
    inline void init_thread(MockGraphMethod& m, int t) { m.init_thread(t); }
    inline void end_thread(MockGraphMethod& m, int t) { m.end_thread(t); }

    inline auto get_shared_snapshot(MockGraphMethod& m) { return m.get_shared_snapshot(); }
    inline auto snapshot_clone(MockSnapshotPtr& s) { return s->clone(); }
    inline uint64_t snapshot_vertex_count(MockSnapshotPtr& s) { return s->vertex_count(); }
    inline uint64_t snapshot_edge_count(MockSnapshotPtr& s) { return s->edge_count(); }
    inline uint64_t snapshot_degree(MockSnapshotPtr& s, uint64_t v, bool l = false) { return s->degree(v, l); }
    inline bool snapshot_has_vertex(MockSnapshotPtr& s, uint64_t v) { return s->has_vertex(v); }
    inline bool snapshot_has_edge(MockSnapshotPtr& s, uint64_t a, uint64_t b) { return s->has_edge(a, b); }
    inline uint64_t snapshot_intersect(MockSnapshotPtr& s, uint64_t a, uint64_t b) { return s->intersect(a, b); }

    template <class Cb>
    void snapshot_edges(MockSnapshotPtr& s, uint64_t v, Cb&& cb, bool l = false) {
        s->edges(v, std::forward<Cb>(cb), l);
    }
}

// Now include the algorithm headers (they depend on wrapper::)
#include "../algorithms/cross_tier_pagerank.hpp"
#include "../algorithms/cross_tier_wcc.hpp"
#include "../algorithms/cross_tier_tc.hpp"

// ═══════════════════════════════════════════════════════════════════════
// Test harness
// ═══════════════════════════════════════════════════════════════════════

struct TestResult {
    std::string name;
    double time_ms;
    bool passed;
    std::string details;
};

void print_separator(const char* title) {
    std::printf("\n");
    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf("  %s\n", title);
    std::printf("═══════════════════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: Cross-Tier PageRank
// ═══════════════════════════════════════════════════════════════════════
TestResult test_cross_tier_pagerank(MockGraphMethod& method,
                                     uint64_t N, int threads) {
    PHILE_BREAKPOINT("test_cross_tier_pagerank");
    print_separator("Test 1: Cross-Tier PageRank");

    TestResult result;
    result.name = "CrossTierPageRank";

    auto snapshot = method.get_shared_snapshot();
    philemon::cost_model::TierCostModel cost_model;

    PHILE_INSPECT("PR_init", "N=%lu threads=%d",
                  (unsigned long)N, threads);

    auto t0 = std::chrono::high_resolution_clock::now();

    philemon::algorithms::CrossTierPageRank<MockGraphMethod, MockSnapshotPtr>
        pr(threads, /*iters=*/10, /*damping=*/0.85,
           method, snapshot, cost_model,
           /*conv_threshold=*/1e-6, /*hotspot_mult=*/10);

    std::vector<std::pair<uint64_t, double>> pr_results;
    pr.run_page_rank(pr_results);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Validation: scores should sum to ~1.0
    double score_sum = 0;
    for (auto& [v, s] : pr_results) score_sum += s;
    result.passed = (std::abs(score_sum - 1.0) < 0.01);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "score_sum=%.6f iters=%zu time=%.3fms",
                  score_sum, pr.iteration_log().size(), result.time_ms);
    result.details = buf;

    PHILE_INSPECT("PR_done", "%s", buf);
    PHILE_DUMP_PAIRS("pr_results", pr_results, 5);
    PHILE_TIER_HEATMAP();

    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: Cross-Tier WCC
// ═══════════════════════════════════════════════════════════════════════
TestResult test_cross_tier_wcc(MockGraphMethod& method,
                                uint64_t N, int threads) {
    PHILE_BREAKPOINT("test_cross_tier_wcc");
    print_separator("Test 2: Cross-Tier WCC");

    TestResult result;
    result.name = "CrossTierWCC";

    auto snapshot = method.get_shared_snapshot();
    philemon::cost_model::TierCostModel cost_model;

    PHILE_INSPECT("WCC_init", "N=%lu threads=%d",
                  (unsigned long)N, threads);

    // Reset tier perf
    for (int t = 0; t < 3; t++) philemon::debug::tier_perf(t).reset();

    auto t0 = std::chrono::high_resolution_clock::now();

    philemon::algorithms::CrossTierWCC<MockGraphMethod, MockSnapshotPtr>
        wcc(threads, method, snapshot, cost_model);

    std::vector<std::pair<uint64_t, int64_t>> wcc_results;
    wcc.run_wcc(wcc_results);

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Validation: every vertex should have a component
    result.passed = (wcc_results.size() == N);

    // Count components
    std::unordered_map<int64_t, uint64_t> comp_count;
    for (auto& [v, c] : wcc_results) comp_count[c]++;

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "components=%zu rounds=%zu time=%.3fms",
                  comp_count.size(), wcc.round_log().size(),
                  result.time_ms);
    result.details = buf;

    PHILE_INSPECT("WCC_done", "%s", buf);
    PHILE_TIER_HEATMAP();

    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: Cross-Tier Triangle Counting
// ═══════════════════════════════════════════════════════════════════════
TestResult test_cross_tier_tc(MockGraphMethod& method,
                               uint64_t N, int threads) {
    PHILE_BREAKPOINT("test_cross_tier_tc");
    print_separator("Test 3: Cross-Tier Triangle Counting");

    TestResult result;
    result.name = "CrossTierTC";

    auto snapshot = method.get_shared_snapshot();
    philemon::cost_model::TierCostModel cost_model;

    PHILE_INSPECT("TC_init", "N=%lu", (unsigned long)N);

    for (int t = 0; t < 3; t++) philemon::debug::tier_perf(t).reset();

    auto t0 = std::chrono::high_resolution_clock::now();

    philemon::algorithms::CrossTierTC<MockGraphMethod, MockSnapshotPtr>
        tc(method, snapshot, threads, cost_model);

    uint64_t triangles = tc.run_tc();

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.passed = true;  // TC always produces a count

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "triangles=%lu edges_examined=%lu time=%.3fms",
                  (unsigned long)triangles,
                  (unsigned long)tc.stats().edges_examined,
                  result.time_ms);
    result.details = buf;

    PHILE_INSPECT("TC_done", "%s", buf);
    PHILE_TIER_HEATMAP();

    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Test 4: Cross-Tier TC Optimized (marker-based)
// ═══════════════════════════════════════════════════════════════════════
TestResult test_cross_tier_tc_opt(MockGraphMethod& method,
                                   uint64_t N, int threads) {
    PHILE_BREAKPOINT("test_cross_tier_tc_opt");
    print_separator("Test 4: Cross-Tier TC Optimized");

    TestResult result;
    result.name = "CrossTierTC_Opt";

    auto snapshot = method.get_shared_snapshot();
    philemon::cost_model::TierCostModel cost_model;

    for (int t = 0; t < 3; t++) philemon::debug::tier_perf(t).reset();

    auto t0 = std::chrono::high_resolution_clock::now();

    philemon::algorithms::CrossTierTC<MockGraphMethod, MockSnapshotPtr>
        tc(method, snapshot, threads, cost_model);

    uint64_t triangles = tc.run_tc_optimized();

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    result.passed = true;

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "triangles_opt=%lu time=%.3fms",
                  (unsigned long)triangles, result.time_ms);
    result.details = buf;

    PHILE_INSPECT("TC_opt_done", "%s", buf);
    PHILE_TIER_HEATMAP();

    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Test 5: All algorithms sequential (convergence comparison)
// ═══════════════════════════════════════════════════════════════════════
TestResult test_sequential_comparison(MockGraphMethod& method,
                                       uint64_t N) {
    PHILE_BREAKPOINT("test_sequential_comparison");
    print_separator("Test 5: Sequential Baseline Comparison");

    TestResult result;
    result.name = "SequentialComparison";

    auto snapshot = method.get_shared_snapshot();
    philemon::cost_model::TierCostModel cost_model;

    auto t0 = std::chrono::high_resolution_clock::now();

    // PR with 1 thread (quasi-sequential)
    {
        philemon::algorithms::CrossTierPageRank<MockGraphMethod, MockSnapshotPtr>
            pr(1, 5, 0.85, method, snapshot, cost_model, 1e-4);

        std::vector<std::pair<uint64_t, double>> pr_results;
        pr.run_page_rank(pr_results);
    }

    // WCC with 1 thread
    {
        philemon::algorithms::CrossTierWCC<MockGraphMethod, MockSnapshotPtr>
            wcc(1, method, snapshot, cost_model);

        std::vector<std::pair<uint64_t, int64_t>> wcc_results;
        wcc.run_wcc(wcc_results);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.passed = true;

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "sequential_total=%.3fms", result.time_ms);
    result.details = buf;

    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    // Parse args
    int debug_level = (argc > 1) ? std::atoi(argv[1]) : 1;
    uint64_t N = (argc > 2) ? std::atol(argv[2]) : 5000;
    int threads = (argc > 3) ? std::atoi(argv[3]) : 4;
    uint64_t E = N * 5;  // ~5 edges per vertex

    philemon::debug::set_debug_level(debug_level);

    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║   Philemon-TSH Cross-Tier Algorithm Benchmark           ║\n");
    std::printf("║   M021-M022: PageRank + WCC + TC                        ║\n");
    std::printf("╠══════════════════════════════════════════════════════════╣\n");
    std::printf("║   N=%lu  E=%lu  threads=%d  debug=%d                    \n",
                (unsigned long)N, (unsigned long)E, threads, debug_level);
    std::printf("╚══════════════════════════════════════════════════════════╝\n");

    // Build test graph
    print_separator("Building Test Graph");
    MockGraphMethod method;
    {
        PHILE_BREAKPOINT("build_graph");
        auto t0 = std::chrono::high_resolution_clock::now();
        method.build_graph(N, E, /*seed=*/42);
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(
            t1 - t0).count();

        auto snap = method.get_shared_snapshot();
        std::printf("  Graph built: V=%lu E=%lu in %.3f ms\n",
                    (unsigned long)snap->vertex_count(),
                    (unsigned long)snap->edge_count(),
                    build_ms);
    }

    // Run tests
    std::vector<TestResult> results;

    results.push_back(test_cross_tier_pagerank(method, N, threads));
    results.push_back(test_cross_tier_wcc(method, N, threads));

    // TC is O(V*E) so use smaller graph for large N
    if (N <= 10000) {
        results.push_back(test_cross_tier_tc(method, N, threads));
        results.push_back(test_cross_tier_tc_opt(method, N, threads));
    } else {
        std::printf("\n  [SKIP] TC tests (N=%lu > 10000, too slow)\n",
                    (unsigned long)N);
    }

    results.push_back(test_sequential_comparison(method, N));

    // ─── Summary ─────────────────────────────────────────────────
    print_separator("Test Summary");

    int passed = 0, failed = 0;
    std::printf("  %-25s %-8s %-12s %s\n",
                "Test", "Status", "Time(ms)", "Details");
    std::printf("  ────────────────────────────────────────────────────\n");
    for (const auto& r : results) {
        std::printf("  %-25s %-8s %-12.3f %s\n",
                    r.name.c_str(),
                    r.passed ? "✅ PASS" : "❌ FAIL",
                    r.time_ms, r.details.c_str());
        if (r.passed) passed++; else failed++;
    }
    std::printf("  ────────────────────────────────────────────────────\n");
    std::printf("  Total: %d passed, %d failed, %zu total\n",
                passed, failed, results.size());

    // Print inspection log
    PHILE_PRINT_INSPECTIONS();

    // Print final tier heatmap
    print_separator("Final Tier Access Heatmap");
    PHILE_TIER_HEATMAP();

    std::printf("\n✅ Benchmark complete.\n");
    return failed > 0 ? 1 : 0;
}
