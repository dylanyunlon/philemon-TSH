/**
 * m098_upstream_io_experiment.cpp — M098迁移验证实验
 *
 * 测试范围:
 *   1. philemon_timer.hpp — Timer + TimerRegistry + lap
 *   2. philemon_cli_engine.hpp — ConfigEngine解析 + 校验
 *   3. edge_stream_file_io.hpp — FileEdgeStream加载 + 分层
 *   4. philemon_file_readers.hpp — EdgeListFileReader + VertexFileReader
 *   5. philemon_driver_main.hpp — 入口 + 系统信息
 *   6. art_node_iter_impl.hpp — ART迭代器 (编译验证)
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m098_test experiment/m098_upstream_io_experiment.cpp
 * 运行: ./m098_test
 *
 * Milestone: M098
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>

// M098 headers
#include "../src/utils/philemon_timer.hpp"
#include "../src/utils/philemon_cli_engine.hpp"
#include "../src/io/edge_stream_file_io.hpp"
#include "../src/io/philemon_file_readers.hpp"
#include "../src/entry/philemon_driver_main.hpp"
#include "../src/neograph/art/art_node_iter_impl.hpp"

// ─── test infrastructure ────────────────────────────────────────────
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    std::fprintf(stderr, "\n[TEST %d] %s ...\n", tests_run, name); \
} while(0)

#define PASS() do { \
    tests_passed++; \
    std::fprintf(stderr, "[PASS]\n"); \
} while(0)

// ═══════════════════════════════════════════════════════════════════
// Test 1: Timer
// ═══════════════════════════════════════════════════════════════════
void test_timer() {
    TEST("Timer basic elapsed");
    {
        philemon::utils::Timer t;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        double e = t.elapsed();
        assert(e >= 0.04 && e < 0.5);
        PASS();
    }

    TEST("Timer reset + elapsed_and_reset");
    {
        philemon::utils::Timer t;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        double e = t.elapsed_and_reset();
        assert(e >= 0.02);
        double e2 = t.elapsed();
        assert(e2 < e);  // should be near zero after reset
        PASS();
    }

    TEST("Timer lap tracking");
    {
        philemon::utils::Timer t;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        double l1 = t.lap();
        assert(l1 >= 0.01);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        double l2 = t.lap();
        assert(l2 >= 0.01);
        assert(t.lap_count() == 2);
        double acc = t.accumulated_elapsed();
        assert(acc >= 0.03);
        t.dump_laps("test_lap");
        PASS();
    }

    TEST("TimerRegistry global");
    {
        auto& reg = philemon::utils::TimerRegistry::instance();
        auto& t1 = reg.get("test_global_1");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        t1.lap();
        auto& t2 = reg.get("test_global_2");
        t2.lap();
        DUMP_ALL_TIMERS();
        reg.clear();
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 2: ConfigEngine
// ═══════════════════════════════════════════════════════════════════
void test_config_engine() {
    TEST("ConfigEngine parse from file");
    {
        // Create temp config file
        std::ofstream f("/tmp/test_config.cfg");
        f << "# test config\n";
        f << "num_threads = 8\n";
        f << "alpha = 20\n";
        f << "beta = 25\n";
        f << "bfs_source = 42\n";
        f << "delta = 3.0\n";
        f << "damping_factor = 0.90\n";
        f << "num_iterations = 15\n";
        f << "workload_type = sssp\n";
        f << "seed = 123\n";
        f << "num_vertices = 5000\n";
        f << "real_graph = true\n";
        f.close();

        philemon::config::ConfigEngine eng;
        eng.parse("/tmp/test_config.cfg");
        
        assert(eng.get_num_threads() == 8);
        assert(eng.get_alpha() == 20);
        assert(eng.get_beta() == 25);
        assert(eng.get_bfs_source() == 42);
        assert(eng.get_delta() == 3.0);
        assert(eng.get_damping_factor() == 0.90);
        assert(eng.get_num_iterations() == 15);
        assert(eng.get_workload_type() == philemon::config::operationType::SSSP);
        assert(eng.get_seed() == 123);
        assert(eng.get_num_vertices() == 5000);
        assert(eng.get_real_graph() == true);

        eng.dump_config();
        PASS();
    }

    TEST("ConfigEngine validation");
    {
        philemon::config::ConfigEngine eng;
        eng.use_defaults();
        int errors = eng.validate();
        assert(errors == 0);
        PASS();
    }

    TEST("ConfigEngine defaults");
    {
        philemon::config::ConfigEngine eng;
        eng.use_defaults();
        assert(eng.get_num_threads() == 4);
        assert(eng.get_alpha() == 15);
        assert(eng.get_damping_factor() == 0.85);
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 3: FileEdgeStream
// ═══════════════════════════════════════════════════════════════════
void test_edge_stream() {
    TEST("FileEdgeStream in-memory add + stats");
    {
        philemon::io::FileEdgeStream stream;
        for (uint64_t i = 0; i < 1000; i++) {
            stream.add_edge(i, (i + 1) % 1000, 1.0);
        }
        assert(stream.get_size() == 1000);
        auto stats = stream.compute_stats();
        assert(stats.edge_count == 1000);
        assert(stats.unique_vertices == 1000);
        BREAKPOINT_STREAM(stream);
        PASS();
    }

    TEST("FileEdgeStream permute + sort + dedup");
    {
        philemon::io::FileEdgeStream stream;
        // Add with duplicates
        stream.add_edge(1, 2, 1.0);
        stream.add_edge(3, 4, 1.0);
        stream.add_edge(1, 2, 1.0);  // duplicate
        stream.add_edge(5, 6, 1.0);
        stream.add_edge(3, 4, 1.0);  // duplicate
        
        assert(stream.get_size() == 5);
        stream.remove_duplicates();
        assert(stream.get_size() == 3);
        
        stream.permute_stream();
        stream.sort();
        assert(stream.edges()[0].source <= stream.edges()[1].source);
        PASS();
    }

    TEST("FileEdgeStream tier_partition");
    {
        philemon::io::FileEdgeStream stream;
        // Create power-law-ish graph
        for (uint64_t i = 0; i < 100; i++) {
            for (uint64_t j = 0; j < (i < 10 ? 50 : 2); j++) {
                stream.add_edge(i, 100 + j, 1.0);
            }
        }
        stream.tier_partition();
        auto stats = stream.compute_stats();
        assert(stats.tier_counts[0] > 0);  // some DRAM
        // Note: cold tier may be 0 if degree distribution is very skewed
        assert(stats.tier_counts[0] + stats.tier_counts[1] + stats.tier_counts[2] 
               == stats.edge_count);  // all edges assigned
        PASS();
    }

    TEST("FileEdgeStream file load");
    {
        // Create temp edge file
        std::ofstream f("/tmp/test_edges.txt");
        f << "# comment line\n";
        f << "0 1\n";
        f << "1 2\n";
        f << "2 3 0.5\n";
        f << "3 0\n";
        f.close();

        philemon::io::FileEdgeStream stream;
        stream.load_stream("/tmp/test_edges.txt");
        assert(stream.get_size() == 4);
        
        philemon::io::StreamEdge e;
        stream.reset_index();
        bool ok = stream.get_next_edge(e);
        assert(ok);
        assert(e.source == 0 && e.destination == 1);
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 4: FileReaders
// ═══════════════════════════════════════════════════════════════════
void test_file_readers() {
    TEST("EdgeListFileReader basic");
    {
        // Create temp edge file
        std::ofstream f("/tmp/test_reader_edges.txt");
        f << "# header\n";
        f << "10 20\n";
        f << "30 40\n";
        f << "50 60\n";
        f.close();

        auto reader = philemon::io::readers::FileReader::open(
            "/tmp/test_reader_edges.txt", 
            philemon::io::readers::readerType::edgeList, 
            false);
        
        assert(reader != nullptr);
        philemon::io::StreamEdge e;
        assert(reader->read_edge(e));
        assert(e.source == 10 && e.destination == 20);
        assert(reader->read_edge(e));
        assert(e.source == 30 && e.destination == 40);
        assert(reader->read_edge(e));
        assert(e.source == 50 && e.destination == 60);
        assert(!reader->read_edge(e));
        assert(reader->get_read_count() == 3);
        assert(reader->get_skip_count() == 1);  // comment line
        reader->dump_state("edge_reader");
        PASS();
    }

    TEST("VertexFileReader basic");
    {
        std::ofstream f("/tmp/test_vertices.txt");
        f << "100\n";
        f << "200\n";
        f << "# comment\n";
        f << "300\n";
        f.close();

        auto reader = philemon::io::readers::FileReader::open(
            "/tmp/test_vertices.txt",
            philemon::io::readers::readerType::vertexList);
        
        uint64_t v;
        assert(reader->read_vertex(v)); assert(v == 100);
        assert(reader->read_vertex(v)); assert(v == 200);
        assert(reader->read_vertex(v)); assert(v == 300);
        assert(!reader->read_vertex(v));
        assert(reader->get_read_count() == 3);
        PASS();
    }

    TEST("EdgeListFileReader progress callback");
    {
        std::ofstream f("/tmp/test_bulk_edges.txt");
        for (int i = 0; i < 500; i++) {
            f << i << " " << (i + 1) << "\n";
        }
        f.close();

        auto reader = philemon::io::readers::FileReader::open(
            "/tmp/test_bulk_edges.txt",
            philemon::io::readers::readerType::edgeList, false);
        
        int callback_count = 0;
        reader->set_progress_callback([&](uint64_t count, const char* type) {
            callback_count++;
        }, 100);

        philemon::io::StreamEdge e;
        while (reader->read_edge(e)) {}
        assert(reader->get_read_count() == 500);
        assert(callback_count >= 4);  // 100,200,300,400,500
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 5: Driver main (system info print)
// ═══════════════════════════════════════════════════════════════════
void test_driver_main() {
    TEST("Driver main system info");
    {
        philemon::entry::print_system_info();
        PASS();
    }

    TEST("Driver main with config");
    {
        std::ofstream f("/tmp/test_driver.cfg");
        f << "num_threads = 2\n";
        f << "workload_type = bfs\n";
        f << "bfs_source = 0\n";
        f.close();
        
        char* argv[] = {(char*)"test", (char*)"/tmp/test_driver.cfg"};
        int rc = philemon::entry::driver_main(2, argv);
        assert(rc == 0);
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Test 6: ART Node Iterator (compilation + basic semantics)
// ═══════════════════════════════════════════════════════════════════
void test_art_iterator() {
    TEST("ART Node4 iterator");
    {
        using namespace philemon::art;
        
        // Construct a Node4 with 3 children
        ARTNode_4 node4;
        node4.n.type = NODE4;
        node4.n.num_children = 3;
        node4.keys[0] = 'a';
        node4.keys[1] = 'b';
        node4.keys[2] = 'c';
        
        // Create leaf-like children (use tagged pointers)
        uint64_t fake_values[3] = {100, 200, 300};
        node4.children[0] = SET_LEAF(&fake_values[0]);
        node4.children[1] = SET_LEAF(&fake_values[1]);
        node4.children[2] = SET_LEAF(&fake_values[2]);
        
        ARTNodeIterator_4 iter(&node4);
        assert(iter.is_valid());
        
        auto [key0, child0] = iter.get();
        assert(key0 == 'a');
        
        iter.next_without_skip();
        assert(iter.is_valid());
        auto [key1, child1] = iter.get();
        assert(key1 == 'b');
        
        iter.next_without_skip();
        assert(iter.is_valid());
        
        iter.next_without_skip();
        assert(!iter.is_valid());
        
        assert(iter.stats.advance_count == 3);
        iter.stats.dump("node4_test");
        PASS();
    }

    TEST("ART alloc_iterator + destroy");
    {
        using namespace philemon::art;
        
        ARTNode_4 node4;
        node4.n.type = NODE4;
        node4.n.num_children = 0;
        
        ARTNode base;
        base.type = NODE4;
        
        // alloc via base type
        auto* iter = alloc_iterator(&base);
        // Note: this will have undefined node pointer, just testing alloc/destroy path
        destroy_iterator(iter);
        PASS();
    }

    TEST("ART IteratorStats");
    {
        philemon::art::IteratorStats stats;
        stats.advance_count = 42;
        stats.leaf_skip_count = 7;
        stats.node_type_visits[0] = 10;
        stats.node_type_visits[2] = 5;
        stats.dump("test_stats");
        PASS();
    }

    TEST("ART Bitmap basic ops");
    {
        philemon::art::Bitmap<4> bm;
        bm.set(0);
        bm.set(5);
        bm.set(63);
        bm.set(64);
        bm.set(255);
        
        uint64_t first = bm.find_first();
        assert(first == 0);
        
        uint64_t c1 = bm.consume();
        assert(c1 == 0);
        uint64_t c2 = bm.consume();
        assert(c2 == 5);
        
        bm.reset(63);
        // 63 was reset, next should be 64
        uint64_t c3 = bm.consume();
        assert(c3 == 64);
        
        PASS();
    }
}

// ═══════════════════════════════════════════════════════════════════
int main() {
    std::fprintf(stderr, "╔═══════════════════════════════════════════════╗\n");
    std::fprintf(stderr, "║  M098 Upstream IO Migration Experiment       ║\n");
    std::fprintf(stderr, "║  第8位Claude (调度者) — M098                 ║\n");
    std::fprintf(stderr, "╚═══════════════════════════════════════════════╝\n\n");

    test_timer();
    test_config_engine();
    test_edge_stream();
    test_file_readers();
    test_driver_main();
    test_art_iterator();

    std::fprintf(stderr, "\n═══════════════════════════════════════════════\n");
    std::fprintf(stderr, "  Results: %d/%d tests passed\n", tests_passed, tests_run);
    std::fprintf(stderr, "═══════════════════════════════════════════════\n");
    
    if (tests_passed == tests_run) {
        std::printf("M098 ALL TESTS PASSED (%d/%d)\n", tests_passed, tests_run);
        return 0;
    } else {
        std::printf("M098 SOME TESTS FAILED (%d/%d)\n", tests_passed, tests_run);
        return 1;
    }
}
