/**
 * test_integration.cpp — 端到端集成测试
 *
 * M040: config加载→建图→运行算法→验证结果的全流程
 *
 * Pipeline:
 *   1. 生成合成时序图 (vertices + temporal edges)
 *   2. TieredAllocator分配
 *   3. TieredLiveGraphWrapper加载
 *   4. TemGraph interval index查询
 *   5. BFS/WCC on snapshot
 *   6. QueryPhaseTimer计时
 *   7. CostModel tier specs
 *   8. Debug infrastructure
 *   9. Full pipeline < 5000ms
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <functional>
#include <fstream>

// Core
#include "core/temporal_edge.hpp"
#include "core/tiered_allocator.hpp"
#include "core/seqlock.hpp"

// Index
#include "index/interval.hpp"
#include "index/dll_list.hpp"
#include "index/tem_graph.hpp"
#include "index/tem_graph_impl.hpp"

// Wrapper
#include "wrapper/apps/livegraph_tiered.hpp"
#include "wrapper/apps/backend_adapters.hpp"
#include "wrapper/graph_edge.hpp"
#include "wrapper/rapidstore_wrapper.hpp"

// Entry — only QueryPhaseTimer (avoid driver_entry which pulls broken driver)
#include "entry/temporal_query_driver.hpp"

// Debug
#include "debug/philemon_debug.hpp"
#include "debug/state_inspector.hpp"

// Utils
#include "utils/config_parser.hpp"

// Cost model
#include "cost_model/tier_cost_model.hpp"

using namespace philemon;

// ═══════════════════════════════════════════════════════════════════
// Synthetic graph generator
// ═══════════════════════════════════════════════════════════════════

struct SyntheticGraph {
    uint64_t num_vertices;
    std::vector<TemporalEdge> edges;
    int time_range;

    std::vector<std::pair<int, int>> to_intervals() const {
        std::vector<std::pair<int, int>> result;
        result.reserve(edges.size());
        for (auto& e : edges) {
            result.emplace_back(e.ts_begin, e.ts_finish);
        }
        return result;
    }

    static SyntheticGraph generate(uint64_t nv, uint64_t ne,
                                    int t_range = 10000, uint64_t seed = 42) {
        SyntheticGraph sg;
        sg.num_vertices = nv;
        sg.time_range = t_range;

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<uint64_t> vdist(0, nv - 1);
        std::uniform_int_distribution<int> tdist(0, t_range - 100);
        std::uniform_int_distribution<int> span_dist(10, 100);
        std::uniform_real_distribution<double> wdist(0.1, 10.0);

        sg.edges.reserve(ne);
        for (uint64_t i = 0; i < ne; i++) {
            TemporalEdge e;
            e.source = vdist(rng);
            e.destination = vdist(rng);
            while (e.destination == e.source) e.destination = vdist(rng);
            e.weight = wdist(rng);
            e.ts_begin = tdist(rng);
            e.ts_finish = e.ts_begin + span_dist(rng);
            sg.edges.push_back(e);
        }
        return sg;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Integration Test Fixture
// ═══════════════════════════════════════════════════════════════════

class IntegrationTest : public ::testing::Test {
protected:
    static constexpr uint64_t NV = 100;
    static constexpr uint64_t NE = 500;
    SyntheticGraph sg;

    void SetUp() override {
        debug::set_debug_level(0);
        sg = SyntheticGraph::generate(NV, NE);
    }
};

// ─── Test 1: TieredAllocator + graph loading ────────────────────────

TEST_F(IntegrationTest, AllocatorAndGraphLoad) {
    TieredAllocator allocator{1024*1024, 4*1024*1024, 16*1024*1024};
    uint64_t part_id = allocator.allocate(NE * sizeof(TemporalEdge), MemoryTier::DRAM);
    EXPECT_GT(part_id, 0u);

    AllocMeta meta;
    EXPECT_TRUE(allocator.get_meta(part_id, meta));
    EXPECT_EQ(meta.size_bytes, NE * sizeof(TemporalEdge));
}

// ─── Test 2: LiveGraph full lifecycle ───────────────────────────────

TEST_F(IntegrationTest, LiveGraphFullLifecycle) {
    adapters::livegraph::TieredLiveGraphWrapper lgw(false, true);

    std::set<uint64_t> vertex_set;
    for (auto& e : sg.edges) {
        vertex_set.insert(e.source);
        vertex_set.insert(e.destination);
    }
    for (auto v : vertex_set) lgw.insert_vertex(v);
    EXPECT_EQ(lgw.vertex_count(), vertex_set.size());

    for (auto& e : sg.edges)
        lgw.insert_edge(e.source, e.destination, e.weight);
    EXPECT_EQ(lgw.edge_count(), NE);

    auto snap = lgw.get_unique_snapshot();
    EXPECT_EQ(snap->vertex_count(), vertex_set.size());
}

// ─── Test 3: TemGraph interval queries ──────────────────────────────

TEST_F(IntegrationTest, TemGraphIntervalQueries) {
    index::TemGraph tg;

    // CONTAINS query
    tg.load_from_edges(index::CONTAINS_QUERY, sg.to_intervals());
    EXPECT_EQ(tg.total_intervals_, (int64_t)NE);

    int contains_result = tg.contains_query(500, 510);
    EXPECT_GE(contains_result, 0);

    // Wide range should find most intervals
    int wide_result = tg.contains_query(0, 20000);
    EXPECT_GT(wide_result, 0);
}

// ─── Test 4: BFS on LiveGraph snapshot ──────────────────────────────

TEST_F(IntegrationTest, BFSOnLiveGraphSnapshot) {
    adapters::livegraph::TieredLiveGraphWrapper lgw(false, true);

    std::set<uint64_t> vertex_set;
    for (auto& e : sg.edges) {
        vertex_set.insert(e.source);
        vertex_set.insert(e.destination);
    }
    for (auto v : vertex_set) lgw.insert_vertex(v);
    for (auto& e : sg.edges)
        lgw.insert_edge(e.source, e.destination, e.weight);

    auto snap = lgw.get_unique_snapshot();
    uint64_t source = *vertex_set.begin();

    std::unordered_map<uint64_t, int64_t> dist;
    std::vector<uint64_t> frontier = {source};
    dist[source] = 0;

    while (!frontier.empty()) {
        std::vector<uint64_t> next;
        for (auto u : frontier) {
            std::vector<uint64_t> nbrs;
            snap->edges(u, nbrs, false);
            for (auto v : nbrs) {
                if (dist.find(v) == dist.end()) {
                    dist[v] = dist[u] + 1;
                    next.push_back(v);
                }
            }
        }
        frontier = next;
    }

    EXPECT_GT(dist.size(), 1u);
}

// ─── Test 5: WCC on LiveGraph ───────────────────────────────────────

TEST_F(IntegrationTest, WCCOnLiveGraph) {
    adapters::livegraph::TieredLiveGraphWrapper lgw(false, true);

    std::set<uint64_t> vertex_set;
    for (auto& e : sg.edges) {
        vertex_set.insert(e.source);
        vertex_set.insert(e.destination);
    }
    for (auto v : vertex_set) lgw.insert_vertex(v);
    for (auto& e : sg.edges)
        lgw.insert_edge(e.source, e.destination, e.weight);

    auto snap = lgw.get_unique_snapshot();

    std::unordered_map<uint64_t, uint64_t> parent;
    for (auto v : vertex_set) parent[v] = v;

    std::function<uint64_t(uint64_t)> find = [&](uint64_t x) -> uint64_t {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };

    for (auto u : vertex_set) {
        std::vector<uint64_t> nbrs;
        snap->edges(u, nbrs, false);
        for (auto v : nbrs) {
            uint64_t ru = find(u), rv = find(v);
            if (ru != rv) parent[ru] = rv;
        }
    }

    std::set<uint64_t> roots;
    for (auto v : vertex_set) roots.insert(find(v));
    EXPECT_LE(roots.size(), 10u);
}

// ─── Test 6: ConfigParser ───────────────────────────────────────────

TEST_F(IntegrationTest, ConfigParsing) {
    std::string path = "/tmp/philemon_integ_test.cfg";
    {
        std::ofstream out(path);
        out << "num_threads=4\nseed=99\nalpha=3\nbeta=2\n"
            << "bfs_source=5\nsssp_source=5\nnum_iterations=10\nwriter_threads=1\n";
    }

    auto& parser = ConfigParser::get_instance();
    parser.parse(path);
    EXPECT_EQ(parser.get_num_threads(), 4);
    EXPECT_EQ(parser.get_seed(), 99);
    std::remove(path.c_str());
}

// ─── Test 7: QueryPhaseTimer ────────────────────────────────────────

TEST_F(IntegrationTest, QueryPhaseTimer) {
    query_driver::QueryPhaseTimer timer;

    timer.begin_phase("load");
    volatile int x = 0;
    for (int i = 0; i < 10000; i++) x += i;
    timer.end_phase();

    timer.begin_phase("query");
    for (int i = 0; i < 10000; i++) x += i;
    timer.end_phase();

    EXPECT_GT(timer.total_us(), 0.0);
}

// ─── Test 8: Cost model tier specs ──────────────────────────────────

TEST_F(IntegrationTest, CostModelTierSpecs) {
    cost_model::TierSpec hbm, gddr, dram;
    hbm.access_latency_ns = 1.2;
    hbm.bandwidth_gbps = 3350;
    hbm.capacity_bytes = 80ULL * 1024 * 1024 * 1024;

    gddr.access_latency_ns = 5.0;
    gddr.bandwidth_gbps = 768;
    gddr.capacity_bytes = 48ULL * 1024 * 1024 * 1024;

    dram.access_latency_ns = 50.0;
    dram.bandwidth_gbps = 80;
    dram.capacity_bytes = 512ULL * 1024 * 1024 * 1024;

    EXPECT_GT(hbm.bytes_per_ns(), gddr.bytes_per_ns());
    EXPECT_GT(gddr.bytes_per_ns(), dram.bytes_per_ns());
}

// ─── Test 9: Debug infrastructure ───────────────────────────────────

TEST_F(IntegrationTest, DebugInfrastructure) {
    debug::set_debug_level(3);

    auto& ring = debug::global_trace();
    ring.record(debug::TraceEvent::ALLOC, 0, 100);

    auto& pc = debug::tier_perf(0);
    pc.read_count.fetch_add(1);

    {
        debug::BreakpointGuard guard("integration_test");
    }

    debug::record_inspection("test_phase", "test message");
    debug::set_debug_level(0);
}

// ─── Test 10: Full pipeline timing ──────────────────────────────────

TEST_F(IntegrationTest, FullPipelineTiming) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: Build graph
    adapters::livegraph::TieredLiveGraphWrapper lgw(false, true);
    std::set<uint64_t> vset;
    for (auto& e : sg.edges) {
        vset.insert(e.source); vset.insert(e.destination);
    }
    for (auto v : vset) lgw.insert_vertex(v);
    for (auto& e : sg.edges)
        lgw.insert_edge(e.source, e.destination, e.weight);

    // Step 2: Build interval index
    index::TemGraph tg;
    tg.load_from_edges(index::CONTAINS_QUERY, sg.to_intervals());

    // Step 3: Run queries
    int total_results = 0;
    for (int q = 0; q < 100; q++) {
        total_results += tg.contains_query(q * 100, q * 100 + 200);
    }

    // Step 4: BFS
    auto snap = lgw.get_unique_snapshot();
    uint64_t source = *vset.begin();
    std::unordered_map<uint64_t, int64_t> dist;
    dist[source] = 0;
    std::vector<uint64_t> frontier = {source};
    while (!frontier.empty()) {
        std::vector<uint64_t> next;
        for (auto u : frontier) {
            std::vector<uint64_t> nbrs;
            snap->edges(u, nbrs, false);
            for (auto v : nbrs) {
                if (dist.find(v) == dist.end()) {
                    dist[v] = dist[u] + 1;
                    next.push_back(v);
                }
            }
        }
        frontier = next;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_LT(ms, 5000.0) << "Pipeline took too long: " << ms << "ms";
    EXPECT_GE(total_results, 0);
}
