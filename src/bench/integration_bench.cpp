/**
 * integration_bench.cpp — End-to-end benchmark wiring all M011-M016 components
 *
 * Wires together:
 *   1. Data generation (synthetic temporal graph)
 *   2. TEM-Graph interval index (contains/contained queries)
 *   3. TieredSnapshot (RapidStore-compatible tiered graph)
 *   4. Graph algorithms (BFS, PageRank, SSSP, WCC, TC)
 *   5. QueryExecutor (concurrent temporal queries)
 *   6. Debug instrumentation (TraceRing, TierPerfCounter)
 *
 * Usage:
 *   ./integration_bench [num_vertices] [num_edges] [num_queries] [num_threads]
 *
 * Default: 10000 vertices, 50000 edges, 1000 queries, 4 threads
 *
 * Milestone: M016 integration test
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>

// Philemon headers
#include "../debug/philemon_debug.hpp"
#include "../core/temporal_edge.hpp"
#include "../index/interval.hpp"
#include "../index/dll_list.hpp"
#include "../index/tem_graph.hpp"
#include "../index/tem_graph_impl.hpp"
#include "../wrapper/rapidstore_wrapper.hpp"
#include "../wrapper/graph_edge.hpp"
#include "../wrapper/edge_stream.hpp"
#include "../executor/thread_pool_base.hpp"
#include "../executor/spin_lock.hpp"
#include "../executor/query_executor.hpp"
// Algorithms
#include "../algorithms/tiered_bfs.hpp"
#include "../algorithms/tiered_pagerank.hpp"
#include "../algorithms/tiered_sssp.hpp"
#include "../algorithms/tiered_wcc.hpp"
#include "../algorithms/tiered_tc.hpp"

using namespace philemon;

// ─── Synthetic data generator ───────────────────────────────────────

struct BenchConfig {
    uint64_t num_vertices = 10000;
    uint64_t num_edges    = 50000;
    uint64_t num_queries  = 1000;
    int      num_threads  = 4;
    int      debug_level  = 1;
    int      time_range   = 100000;
};

struct SyntheticGraph {
    std::vector<TemporalEdge>                        edges;
    std::vector<std::pair<index::Timestamp, index::Timestamp>> intervals;
    uint64_t max_vertex = 0;
};

SyntheticGraph generate_graph(const BenchConfig& cfg) {
    debug::ScopedTimer timer("generate_graph");
    SyntheticGraph g;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> vertex_dist(0, cfg.num_vertices - 1);
    std::uniform_int_distribution<int32_t> ts_dist(0, cfg.time_range);
    std::uniform_real_distribution<double> weight_dist(0.1, 10.0);

    g.edges.reserve(cfg.num_edges);
    g.intervals.reserve(cfg.num_edges);

    for (uint64_t i = 0; i < cfg.num_edges; i++) {
        uint64_t src = vertex_dist(rng);
        uint64_t dst = vertex_dist(rng);
        if (src == dst) dst = (dst + 1) % cfg.num_vertices;

        int32_t t0 = ts_dist(rng);
        int32_t t1 = t0 + std::abs((int)(ts_dist(rng) % 1000)) + 1;
        double w = weight_dist(rng);

        TemporalEdge e;
        e.source   = src;
        e.destination   = dst;
        e.weight   = w;
        e.ts_begin = t0;
        e.ts_finish   = t1;
        g.edges.push_back(e);
        g.intervals.push_back({t0, t1});
        g.max_vertex = std::max(g.max_vertex, std::max(src, dst));
    }

    PHILE_DBG(1, "Generated: %lu vertices, %lu edges, ts_range=[0,%d]",
              (unsigned long)(g.max_vertex + 1),
              (unsigned long)g.edges.size(), cfg.time_range);
    return g;
}


// ─── Phase 1: TEM-Graph Index ───────────────────────────────────────

void bench_temgraph_index(const SyntheticGraph& g, const BenchConfig& cfg,
                          index::TemGraph& index_out) {
    std::printf("\n══════ Phase 1: TEM-Graph Index Build + Query ══════\n");

    // Build index from in-memory edges
    {
        debug::ScopedTimer timer("index_build");
        index_out.load_from_edges(index::CONTAINS_QUERY, g.intervals);
    }

    index_out.dump_index_state(10);

    // Run sample queries
    std::printf("\n── Sample contains queries ──\n");
    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> ts_dist(0, cfg.time_range);
    uint64_t total_matched = 0;

    for (int i = 0; i < 5; i++) {
        int32_t l = ts_dist(rng);
        int32_t r = l + 500;
        auto result = index_out.contains_query_traced(l, r);
        result.dump("sample");
        total_matched += result.matched_count;
    }
    PHILE_DBG(1, "Sample queries: total_matched=%lu", (unsigned long)total_matched);
}


// ─── Phase 2: TieredSnapshot + Algorithms ───────────────────────────

void bench_algorithms(const SyntheticGraph& g, const BenchConfig& cfg) {
    std::printf("\n══════ Phase 2: Graph Algorithms on TieredSnapshot ══════\n");

    // Build tiered snapshot — distribute edges across tiers by recency
    // Hot (recent ts_end) → HBM(0), warm → GDDR(1), cold → DRAM(2)
    auto snapshot = std::make_shared<TieredSnapshot>();

    // Sort edges by ts_end to assign tiers
    std::vector<size_t> order(g.edges.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return g.edges[a].ts_finish > g.edges[b].ts_finish;  // newest first
    });

    size_t n = g.edges.size();
    size_t hbm_end  = n / 5;       // top 20% → HBM
    size_t gddr_end = n / 2;       // next 30% → GDDR
                                    // rest 50% → DRAM

    // Build per-tier edge arrays
    std::vector<TemporalEdge> hbm_edges, gddr_edges, dram_edges;
    for (size_t i = 0; i < n; i++) {
        if (i < hbm_end)       hbm_edges.push_back(g.edges[order[i]]);
        else if (i < gddr_end) gddr_edges.push_back(g.edges[order[i]]);
        else                   dram_edges.push_back(g.edges[order[i]]);
    }

    snapshot->build_from_edges(hbm_edges.data(), hbm_edges.size(), 0);
    snapshot->build_from_edges(gddr_edges.data(), gddr_edges.size(), 1);
    snapshot->build_from_edges(dram_edges.data(), dram_edges.size(), 2);
    snapshot->dump_tier_distribution();

    // Create wrapper for algorithms
    TieredGraphWrapper wrapper;
    wrapper.set_snapshot(snapshot);
    wrapper.dump_state();

    // Reset tier counters before each algorithm
    auto reset_counters = []() {
        for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();
    };

    // --- BFS ---
    if (snapshot->vertex_count() > 0) {
        std::printf("\n── BFS ──\n");
        reset_counters();
        std::vector<std::pair<uint64_t, int64_t>> bfs_results;
        algorithms::TieredBFS<TieredGraphWrapper,
                              std::shared_ptr<TieredSnapshot>>
            bfs(cfg.num_threads, 15, 18, wrapper, snapshot);
        bfs.run_bfs(0, bfs_results);

        // Count reachable vertices
        uint64_t reachable = 0;
        int64_t max_dist = 0;
        for (auto& [v, d] : bfs_results) {
            if (d >= 0) { reachable++; max_dist = std::max(max_dist, d); }
        }
        std::printf("  BFS: reachable=%lu max_dist=%ld\n",
                    (unsigned long)reachable, (long)max_dist);
    }

    // --- PageRank ---
    {
        std::printf("\n── PageRank (10 iterations) ──\n");
        reset_counters();
        std::vector<std::pair<uint64_t, double>> pr_results;
        algorithms::TieredPageRank<TieredGraphWrapper,
                                    std::shared_ptr<TieredSnapshot>>
            pr(cfg.num_threads, 10, 0.85, wrapper, snapshot);
        pr.run_page_rank(pr_results);

        // Top-5 PageRank vertices
        std::sort(pr_results.begin(), pr_results.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        std::printf("  Top-5 PR:");
        for (int i = 0; i < std::min(5, (int)pr_results.size()); i++) {
            std::printf(" v%lu=%.6f", (unsigned long)pr_results[i].first,
                        pr_results[i].second);
        }
        std::printf("\n");
    }

    // --- SSSP ---
    if (snapshot->vertex_count() > 0) {
        std::printf("\n── SSSP (delta=1.0) ──\n");
        reset_counters();
        std::vector<std::pair<uint64_t, double>> sssp_results;
        algorithms::TieredSSSP<TieredGraphWrapper,
                                std::shared_ptr<TieredSnapshot>>
            sssp(cfg.num_threads, 1.0, wrapper, snapshot);
        sssp.run_sssp(0, sssp_results);

        uint64_t reachable = 0;
        for (auto& [v, d] : sssp_results) {
            if (d < 1e18) reachable++;
        }
        std::printf("  SSSP: reachable=%lu\n", (unsigned long)reachable);
    }

    // --- WCC ---
    {
        std::printf("\n── WCC ──\n");
        reset_counters();
        std::vector<std::pair<uint64_t, uint64_t>> wcc_results;
        algorithms::TieredWCC<TieredGraphWrapper,
                               std::shared_ptr<TieredSnapshot>>
            wcc(cfg.num_threads, wrapper, snapshot);
        wcc.run_wcc(wcc_results);
    }

    // --- TC (only on small graphs) ---
    if (cfg.num_vertices <= 5000) {
        std::printf("\n── Triangle Counting ──\n");
        reset_counters();
        algorithms::TieredTriangleCounting<TieredGraphWrapper,
                                            std::shared_ptr<TieredSnapshot>>
            tc(wrapper, snapshot);
        uint64_t triangles = tc.run_tc(cfg.num_threads);
        std::printf("  TC: %lu triangles\n", (unsigned long)triangles);
    } else {
        std::printf("\n── TC: skipped (vertices=%lu > 5000) ──\n",
                    (unsigned long)cfg.num_vertices);
    }
}


// ─── Phase 3: Concurrent Query Executor ─────────────────────────────

void bench_executor(index::TemGraph& index, const BenchConfig& cfg) {
    std::printf("\n══════ Phase 3: Concurrent Query Executor ══════\n");

    executor::QueryExecutor exec(cfg.num_threads, &index);

    // Generate random query workload
    std::mt19937 rng(999);
    std::uniform_int_distribution<int32_t> ts_dist(0, cfg.time_range);

    std::vector<executor::QueryRequest> requests;
    requests.reserve(cfg.num_queries);
    for (uint64_t i = 0; i < cfg.num_queries; i++) {
        int32_t l = ts_dist(rng);
        int32_t r = l + 200 + (ts_dist(rng) % 800);
        executor::QueryRequest req;
        req.type     = (i % 2 == 0) ? executor::QueryType::CONTAINS
                                     : executor::QueryType::CONTAINED;
        req.l        = l;
        req.r        = r;
        req.query_id = i;
        requests.push_back(req);
    }

    PHILE_DBG(1, "Submitting %lu queries to executor...",
              (unsigned long)requests.size());

    auto batch = exec.batch_query(requests);
    batch.dump();
    exec.dump_state();

    // Trace ring summary
    debug::global_trace().dump_last(10);
}


// ─── Phase 4: EdgeStream Integration ────────────────────────────────

void bench_edge_stream(const SyntheticGraph& g) {
    std::printf("\n══════ Phase 4: EdgeStream Compatibility ══════\n");
    debug::ScopedTimer timer("edge_stream_test");

    driver::graph::edgeStream stream;
    stream.load_from_temporal_edges(g.edges.data(), g.edges.size());
    stream.dump_stream_stats("loaded");

    stream.permute_stream();
    stream.dump_stream_stats("permuted");

    stream.sort();
    stream.remove_duplicates();
    stream.dump_stream_stats("deduped");

    // Verify edge retrieval
    driver::graph::weightedEdge e;
    int count = 0;
    while (stream.get_next_edge(e)) count++;
    std::printf("  Sequential scan: %d edges retrieved\n", count);
}


// ─── Main ───────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (argc > 1) cfg.num_vertices = std::atol(argv[1]);
    if (argc > 2) cfg.num_edges    = std::atol(argv[2]);
    if (argc > 3) cfg.num_queries  = std::atol(argv[3]);
    if (argc > 4) cfg.num_threads  = std::atoi(argv[4]);
    if (argc > 5) cfg.debug_level  = std::atoi(argv[5]);

    debug::set_debug_level(cfg.debug_level);

    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  Philemon-TSH Integration Benchmark (M011–M016)     ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n");
    std::printf("  vertices=%lu edges=%lu queries=%lu threads=%d debug=%d\n\n",
                (unsigned long)cfg.num_vertices,
                (unsigned long)cfg.num_edges,
                (unsigned long)cfg.num_queries,
                cfg.num_threads, cfg.debug_level);

    // Generate data
    auto graph = generate_graph(cfg);

    // Phase 1: TEM-Graph index
    index::TemGraph tem_index;
    bench_temgraph_index(graph, cfg, tem_index);

    // Phase 2: Algorithms on TieredSnapshot
    bench_algorithms(graph, cfg);

    // Phase 3: Concurrent executor
    bench_executor(tem_index, cfg);

    // Phase 4: EdgeStream compatibility
    bench_edge_stream(graph);

    // Final summary
    std::printf("\n╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  All Phases Complete                                 ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    debug::print_all_tier_perf();
    index::print_peak_memory_usage("integration_bench");

    return 0;
}
