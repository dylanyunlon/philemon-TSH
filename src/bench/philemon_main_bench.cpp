/**
 * philemon_main_bench.cpp — 完整 benchmark 入口 (移植自 upstream main.cpp)
 *
 * 骨架来源: upstream/rapidstore/main.cpp (202行)
 * 修改 (~25%):
 *   - 用 PhilemonDriver + NeoGraphAdapter 替代 teseo_driver
 *   - 用 ConfigParser 替代 commandLineParser
 *   - 用 Philemon wrapper algorithm 模板替代直接调用
 *   - 增加 per-algorithm debug dump: 结构体快照 + tier 统计
 *   - 增加 startup/shutdown banner
 *   - 增加 全流程 ScopedTimer
 *
 * Milestone: M028
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <memory>

#include "../utils/config_parser_ext.hpp"
#include "../utils/log/philemon_log.hpp"
#include "../utils/timer_utils_ext.hpp"
#include "../wrapper/apps/philemon_neo_adapter.hpp"
#include "../wrapper/philemon_wrapper_ops.hpp"
#include "../driver/philemon_driver_ext.hpp"
#include "../entry/philemon_main_ext.hpp"

// Algorithm wrappers
#include "../wrapper/algorithms/cross_tier_bfs_wrapper.hpp"
#include "../wrapper/algorithms/cross_tier_sssp_wrapper.hpp"
#include "../wrapper/algorithms/cross_tier_pr_wrapper.hpp"
#include "../wrapper/algorithms/cross_tier_wcc_wrapper.hpp"
#include "../wrapper/algorithms/cross_tier_tc_wrapper.hpp"

int main(int argc, char** argv) {
    // ─── Banner ─────────────────────────────────────────────────────
    philemon::entry::print_banner();

    // ─── Config ─────────────────────────────────────────────────────
    auto& cfg = philemon::config::ConfigParser::instance();
    if (argc > 1) {
        cfg.parse(argv[1]);
    } else {
        plog_warn("No config file; use: ./philemon_main_bench config.cfg");
        plog_info("Running with default configuration");
    }
    if (argc > 2) cfg.parse_args(argc, argv);
    cfg.dump_config("MAIN_BENCH");

    // ─── Graph Setup ────────────────────────────────────────────────
    philemon::utils::ScopedTimer total_timer("FULL_BENCH", "MAIN");

    auto graph = std::make_shared<philemon::adapter::NeoGraphAdapter>();

    plog_info("[MAIN] Graph adapter created");

    // ─── Load vertices (from config or synthetic) ───────────────────
    {
        philemon::utils::ScopedTimer vt("VERTEX_LOAD");
        std::string vf = cfg.input_file();
        if (!vf.empty()) {
            // Load from file
            std::ifstream fin(vf);
            uint64_t vid;
            std::vector<uint64_t> verts;
            while (fin >> vid) verts.push_back(vid);
            fin.close();

            wrapper::run_batch_vertex_update(*graph, verts, 0, verts.size());
            plog_info("[MAIN] Loaded %zu vertices from %s", verts.size(), vf.c_str());
        } else {
            // Synthetic
            uint64_t N = 1000;
            std::vector<uint64_t> verts(N);
            for (uint64_t i = 0; i < N; i++) verts[i] = i;
            wrapper::run_batch_vertex_update(*graph, verts, 0, N);
            plog_info("[MAIN] Generated %lu synthetic vertices", N);
        }
    }

    // ─── Load edges ─────────────────────────────────────────────────
    {
        philemon::utils::ScopedTimer et("EDGE_LOAD");
        std::string ef = cfg.get_str_public("edge_file", "");
        if (!ef.empty()) {
            std::ifstream fin(ef);
            uint64_t src, dst;
            uint64_t cnt = 0;
            while (fin >> src >> dst) {
                wrapper::insert_edge(*graph, src, dst, 0.0);
                cnt++;
                if (cnt % 500000 == 0) {
                    plog_info("[MAIN] edges loaded: %lu  RSS=%ld KB",
                              cnt, philemon_get_rss_kb());
                }
            }
            fin.close();
            plog_info("[MAIN] Loaded %lu edges", cnt);
        } else {
            // Synthetic random edges
            uint64_t N = wrapper::vertex_count(*graph);
            std::mt19937 rng(cfg.seed());
            uint64_t target_edges = N * 5;
            for (uint64_t i = 0; i < target_edges; i++) {
                uint64_t s = rng() % N, d = rng() % N;
                if (s != d) wrapper::insert_edge(*graph, s, d, 0.0);
            }
            plog_info("[MAIN] Generated %lu synthetic edges", target_edges);
        }
    }

    // ─── Print graph state ──────────────────────────────────────────
    graph->dump_adapter_state("POST_LOAD");
    wrapper::counters::dump("POST_LOAD");
    wrapper::counters::reset();

    plog_info("[MAIN] Graph: V=%lu  E=%lu",
              wrapper::vertex_count(*graph), wrapper::edge_count(*graph));

    // ─── Run Algorithms ─────────────────────────────────────────────
    int thread_counts[] = {1, 2, 4};
    for (auto num_t : thread_counts) {
        plog_info("\n========== Running with %d threads ==========", num_t);

        auto snapshot = graph->get_shared_snapshot();

        // BFS
        {
            philemon::utils::ScopedTimer st("BFS");
            PhilemonBfsWrapper<philemon::adapter::NeoGraphAdapter,
                              decltype(snapshot)>
                bfs(num_t, cfg.alpha(), cfg.beta(), *graph, snapshot);

            std::vector<std::pair<uint64_t, int64_t>> bfs_result;
            bfs.run_bfs(cfg.bfs_source(), bfs_result);
            bfs.dump_tier_stats();
        }

        // SSSP
        {
            philemon::utils::ScopedTimer st("SSSP");
            PhilemonSsspWrapper<philemon::adapter::NeoGraphAdapter,
                               decltype(snapshot)>
                sssp(num_t, cfg.delta(), *graph, snapshot);

            std::vector<std::pair<uint64_t, double>> sssp_result;
            sssp.run_sssp(cfg.sssp_source(), sssp_result);
            PhilemonSsspWrapper<philemon::adapter::NeoGraphAdapter,
                               decltype(snapshot)>::dump_distance_histogram(sssp_result);
        }

        // PageRank
        {
            philemon::utils::ScopedTimer st("PAGE_RANK");
            PhilemonPrWrapper<philemon::adapter::NeoGraphAdapter,
                             decltype(snapshot)>
                pr(num_t, cfg.num_iterations(), cfg.damping_factor(),
                   *graph, snapshot);

            std::vector<std::pair<uint64_t, double>> pr_result;
            pr.run_page_rank(pr_result);
        }

        // WCC
        {
            philemon::utils::ScopedTimer st("WCC");
            PhilemonWccWrapper<philemon::adapter::NeoGraphAdapter,
                              decltype(snapshot)>
                wcc(num_t, *graph, snapshot);

            std::vector<std::pair<uint64_t, int64_t>> wcc_result;
            wcc.run_wcc(wcc_result);
            PhilemonWccWrapper<philemon::adapter::NeoGraphAdapter,
                              decltype(snapshot)>::dump_component_distribution(wcc_result);
        }

        // TC (standard)
        {
            philemon::utils::ScopedTimer st("TC");
            PhilemonTcWrapper<philemon::adapter::NeoGraphAdapter,
                             decltype(snapshot)>
                tc(*graph, snapshot, false);
            tc.run_tc();
        }

        wrapper::counters::dump(("AFTER_" + std::to_string(num_t) + "_THREADS").c_str());
        wrapper::counters::reset();
        graph->tier_stats().dump(("TIER_" + std::to_string(num_t) + "_THREADS").c_str());
        graph->tier_stats().reset();
    }

    // ─── Shutdown ───────────────────────────────────────────────────
    graph->dump_adapter_state("FINAL");
    philemon::entry::shutdown_report();

    return 0;
}
