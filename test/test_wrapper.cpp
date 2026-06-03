/**
 * test_wrapper.cpp — Wrapper / Reader / Config 单元测试
 *
 * M039: wrapper模块UT覆盖
 *   - TierRouter: route_by_hash distribution
 *   - TierLatencyModel: cost ordering
 *   - TieredLiveGraphWrapper: vertex/edge CRUD + snapshot + batch
 *   - VertexDictionary: insert/find/erase
 *   - weightedEdge: construction/comparison
 *   - edgeStream: basic structure
 *   - ConfigParser: parse config file
 *   - TierStats: accumulate + dump
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>
#include <set>

#include "wrapper/apps/backend_adapters.hpp"
#include "wrapper/apps/livegraph_tiered.hpp"
#include "wrapper/graph_edge.hpp"
#include "wrapper/edge_stream.hpp"
#include "utils/config_parser.hpp"
#include "debug/philemon_debug.hpp"
#include "core/temporal_edge.hpp"

using namespace philemon;

// ═══════════════════════════════════════════════════════════════════
// TierRouter Tests (backend_adapters.hpp)
// ═══════════════════════════════════════════════════════════════════

class TierRouterTest : public ::testing::Test {
protected:
    void SetUp() override { debug::set_debug_level(0); }
};

TEST_F(TierRouterTest, RouteByHashDistribution) {
    adapters::TierRouter router;  // defaults: 15% HBM, 35% GDDR, 50% DRAM

    int counts[3] = {0, 0, 0};
    for (uint64_t i = 0; i < 10000; i++) {
        auto tier = router.route_by_hash(i, i + 1);
        counts[static_cast<int>(tier)]++;
    }

    EXPECT_GT(counts[0], 0) << "HBM got zero edges";
    EXPECT_GT(counts[1], 0) << "GDDR got zero edges";
    EXPECT_GT(counts[2], 0) << "DRAM got zero edges";
}

TEST_F(TierRouterTest, DeterministicRouting) {
    adapters::TierRouter router;
    auto t1 = router.route_by_hash(100, 200);
    auto t2 = router.route_by_hash(100, 200);
    EXPECT_EQ(t1, t2);
}

TEST_F(TierRouterTest, RouteByDegree) {
    adapters::TierRouter router;
    // High degree → HBM
    auto hot = router.route_by_degree(950, 1000);
    EXPECT_EQ(hot, adapters::TierLevel::HBM);

    // Low degree → DRAM
    auto cold = router.route_by_degree(10, 1000);
    EXPECT_EQ(cold, adapters::TierLevel::DRAM);
}

// ═══════════════════════════════════════════════════════════════════
// TierLatencyModel Tests
// ═══════════════════════════════════════════════════════════════════

TEST(TierLatencyModelTest, CostOrdering) {
    using TLM = adapters::TierLatencyModel;
    double hbm  = TLM::edge_cost(adapters::TierLevel::HBM);
    double gddr = TLM::edge_cost(adapters::TierLevel::GDDR);
    double dram = TLM::edge_cost(adapters::TierLevel::DRAM);
    EXPECT_LT(hbm, gddr);
    EXPECT_LT(gddr, dram);
}

// ═══════════════════════════════════════════════════════════════════
// TieredLiveGraphWrapper Tests
// ═══════════════════════════════════════════════════════════════════

class LiveGraphTest : public ::testing::Test {
protected:
    adapters::livegraph::TieredLiveGraphWrapper wrapper;

    LiveGraphTest() : wrapper(false, true) {}

    void SetUp() override {
        debug::set_debug_level(0);
        for (uint64_t i = 0; i < 5; i++) wrapper.insert_vertex(i);
    }
};

TEST_F(LiveGraphTest, InsertVertex) {
    EXPECT_EQ(wrapper.vertex_count(), 5u);
    EXPECT_TRUE(wrapper.has_vertex(0));
    EXPECT_TRUE(wrapper.has_vertex(4));
    EXPECT_FALSE(wrapper.has_vertex(99));
}

TEST_F(LiveGraphTest, InsertVertexDuplicate) {
    bool ok = wrapper.insert_vertex(0);
    EXPECT_FALSE(ok);
    EXPECT_EQ(wrapper.vertex_count(), 5u);
}

TEST_F(LiveGraphTest, InsertEdge) {
    EXPECT_TRUE(wrapper.insert_edge(0, 1, 2.5));
    EXPECT_EQ(wrapper.edge_count(), 1u);
    EXPECT_TRUE(wrapper.has_edge(0, 1));
}

TEST_F(LiveGraphTest, GetWeight) {
    wrapper.insert_edge(0, 1, 3.14);
    double w = wrapper.get_weight(0, 1);
    EXPECT_NEAR(w, 3.14, 1e-9);
}

TEST_F(LiveGraphTest, Degree) {
    wrapper.insert_edge(0, 1);
    wrapper.insert_edge(0, 2);
    wrapper.insert_edge(0, 3);
    uint64_t deg = wrapper.degree(0);
    EXPECT_GE(deg, 3u);
}

TEST_F(LiveGraphTest, RemoveVertex) {
    EXPECT_TRUE(wrapper.remove_vertex(4));
    EXPECT_EQ(wrapper.vertex_count(), 4u);
    EXPECT_FALSE(wrapper.has_vertex(4));
}

TEST_F(LiveGraphTest, RemoveEdge) {
    wrapper.insert_edge(0, 1);
    EXPECT_TRUE(wrapper.remove_edge(0, 1));
    EXPECT_FALSE(wrapper.has_edge(0, 1));
}

TEST_F(LiveGraphTest, GetNeighbors) {
    wrapper.insert_edge(0, 1);
    wrapper.insert_edge(0, 2);
    std::vector<uint64_t> nbrs;
    wrapper.get_neighbors(0, nbrs);
    EXPECT_GE(nbrs.size(), 2u);
}

TEST_F(LiveGraphTest, SnapshotBasic) {
    wrapper.insert_edge(0, 1);
    wrapper.insert_edge(1, 2);
    auto snap = wrapper.get_unique_snapshot();
    EXPECT_EQ(snap->vertex_count(), 5u);
    EXPECT_EQ(snap->edge_count(), 2u);
}

TEST_F(LiveGraphTest, SnapshotEdges) {
    wrapper.insert_edge(0, 1);
    wrapper.insert_edge(0, 2);
    auto snap = wrapper.get_unique_snapshot();
    std::vector<uint64_t> nbrs;
    snap->edges(0, nbrs, false);
    EXPECT_GE(nbrs.size(), 2u);
}

TEST_F(LiveGraphTest, SnapshotCallback) {
    wrapper.insert_edge(0, 1, 1.5);
    wrapper.insert_edge(0, 2, 2.5);
    auto snap = wrapper.get_unique_snapshot();
    double wsum = 0;
    snap->edges(0, [&](uint64_t, double w) { wsum += w; }, false);
    EXPECT_GT(wsum, 0.0);
}

TEST_F(LiveGraphTest, SnapshotClone) {
    wrapper.insert_edge(0, 1);
    auto snap = wrapper.get_shared_snapshot();
    auto clone = snap->clone();
    EXPECT_EQ(clone->vertex_count(), snap->vertex_count());
}

TEST_F(LiveGraphTest, BatchVertexUpdate) {
    std::vector<uint64_t> verts = {10, 11, 12, 13};
    EXPECT_TRUE(wrapper.run_batch_vertex_update(verts, 0, 4));
    EXPECT_EQ(wrapper.vertex_count(), 9u);
}

TEST_F(LiveGraphTest, BatchEdgeUpdate) {
    using TE = adapters::livegraph::TieredEdge;
    std::vector<TE> edges = {
        {0, 1, 1.0, 0}, {1, 2, 1.0, 1}, {2, 3, 1.0, 2},
    };
    EXPECT_TRUE(wrapper.run_batch_edge_update(edges, 0, 3));
    EXPECT_EQ(wrapper.edge_count(), 3u);
}

TEST_F(LiveGraphTest, Clear) {
    wrapper.insert_edge(0, 1);
    wrapper.insert_edge(1, 2);
    wrapper.clear();
    EXPECT_EQ(wrapper.edge_count(), 0u);
}

TEST_F(LiveGraphTest, IsEmpty) {
    adapters::livegraph::TieredLiveGraphWrapper empty_w(false, true);
    EXPECT_TRUE(empty_w.is_empty());
    EXPECT_FALSE(wrapper.is_empty());
}

TEST_F(LiveGraphTest, SelfTestPasses) {
    EXPECT_NO_FATAL_FAILURE(
        adapters::livegraph::wrapper_test::run_livegraph_self_test()
    );
}

// ═══════════════════════════════════════════════════════════════════
// VertexDictionary Tests
// ═══════════════════════════════════════════════════════════════════

TEST(VertexDictionaryTest, InsertFindErase) {
    adapters::livegraph::VertexDictionary dict;
    adapters::livegraph::VertexEntry entry{100, adapters::livegraph::TierTag::HBM, 0};
    EXPECT_TRUE(dict.insert(42, entry));

    adapters::livegraph::VertexEntry out;
    EXPECT_TRUE(dict.find(42, out));
    EXPECT_EQ(out.physical_id, 100u);

    EXPECT_TRUE(dict.erase(42));
    EXPECT_FALSE(dict.find(42, out));
}

TEST(VertexDictionaryTest, DuplicateInsertFails) {
    adapters::livegraph::VertexDictionary dict;
    adapters::livegraph::VertexEntry e{1, adapters::livegraph::TierTag::DRAM, 0};
    EXPECT_TRUE(dict.insert(0, e));
    EXPECT_FALSE(dict.insert(0, e));
}

TEST(VertexDictionaryTest, Size) {
    adapters::livegraph::VertexDictionary dict;
    EXPECT_EQ(dict.size(), 0u);
    adapters::livegraph::VertexEntry e{0, adapters::livegraph::TierTag::DRAM, 0};
    dict.insert(1, e);
    dict.insert(2, e);
    EXPECT_EQ(dict.size(), 2u);
}

// ═══════════════════════════════════════════════════════════════════
// weightedEdge Tests (graph_edge.hpp)
// ═══════════════════════════════════════════════════════════════════

TEST(WeightedEdgeTest, Construction) {
    driver::graph::weightedEdge e;
    e.set_edge(10, 20, 3.5);
    EXPECT_EQ(e.source, 10u);
    EXPECT_EQ(e.destination, 20u);
    EXPECT_NEAR(e.weight, 3.5, 1e-9);
}

TEST(WeightedEdgeTest, Equality) {
    driver::graph::weightedEdge a, b, c;
    a.set_edge(1, 2, 1.0);
    b.set_edge(1, 2, 1.0);
    c.set_edge(1, 3, 1.0);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(WeightedEdgeTest, Ordering) {
    driver::graph::weightedEdge a, b;
    a.set_edge(1, 2, 1.0);
    b.set_edge(2, 1, 1.0);
    EXPECT_TRUE(a < b);
}

// ═══════════════════════════════════════════════════════════════════
// edgeStream Tests
// ═══════════════════════════════════════════════════════════════════

TEST(EdgeStreamTest, BasicOperations) {
    driver::graph::edgeStream stream;
    EXPECT_EQ(stream.get_current_index(), 0);
    stream.reset_index();
    EXPECT_EQ(stream.get_current_index(), 0);
}

// ═══════════════════════════════════════════════════════════════════
// ConfigParser Tests
// ═══════════════════════════════════════════════════════════════════

class ConfigParserTest : public ::testing::Test {
protected:
    std::string tmp_path;

    void SetUp() override {
        debug::set_debug_level(0);
        tmp_path = "/tmp/philemon_test_config.cfg";
        std::ofstream out(tmp_path);
        out << "num_threads=8\n"
            << "seed=42\n"
            << "alpha=10\n"
            << "beta=5\n"
            << "bfs_source=0\n"
            << "sssp_source=0\n"
            << "num_iterations=20\n"
            << "writer_threads=2\n";
        out.close();
    }

    void TearDown() override { std::remove(tmp_path.c_str()); }
};

TEST_F(ConfigParserTest, ParseAndGetValues) {
    auto& parser = ConfigParser::get_instance();
    parser.parse(tmp_path);

    EXPECT_EQ(parser.get_num_threads(), 8);
    EXPECT_EQ(parser.get_seed(), 42);
    EXPECT_EQ(parser.get_alpha(), 10);
    EXPECT_EQ(parser.get_beta(), 5);
    EXPECT_EQ(parser.get_bfs_source(), 0u);
    EXPECT_EQ(parser.get_sssp_source(), 0u);
    EXPECT_EQ(parser.get_num_iterations(), 20);
    EXPECT_EQ(parser.get_writer_threads(), 2);
}

// ═══════════════════════════════════════════════════════════════════
// TierStats Tests
// ═══════════════════════════════════════════════════════════════════

TEST(TierStatsTest, RecordAndAccumulate) {
    adapters::TierStats stats;
    stats.record_read(adapters::TierLevel::HBM, 10);
    stats.record_read(adapters::TierLevel::GDDR, 20);
    stats.record_read(adapters::TierLevel::DRAM, 30);
    stats.record_write(5);

    EXPECT_EQ(stats.edge_reads.load(), 60u);
    EXPECT_EQ(stats.edge_writes.load(), 5u);
}
