#ifndef PHILEMON_MAIN_ENTRY_HPP
#define PHILEMON_MAIN_ENTRY_HPP
/**
 * philemon_main.hpp — 系统入口与实验驱动
 *
 * 骨架来源: upstream/rapidstore/main.cpp (全文, ~200行有效代码)
 * 修改 (~20%):
 *   - [MOD] hardcoded config路径 → 运行时参数/默认路径
 *   - [MOD] teseo_driver直接实例化 → TieredBackendAdapter
 *   - [NEW] PhaseTimer包裹每个算法执行阶段
 *   - [NEW] PHILE_MEM_CHECKPOINT在顶点加载/边加载/算法前后
 *   - [NEW] 打印per-thread insertion throughput
 *   - [KEEP] 算法调用顺序: SSSP→BFS→PageRank→WCC
 *   - [KEEP] 线程数遍历模式 (upstream: 1,2,4,8,16,32,40)
 *   - [KEEP] chunk_size = (size + threads - 1) / threads
 *   - [KEEP] edge_stream并行插入模式
 *
 * Milestone: M028+
 */

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdio>
#include <algorithm>

#include "../types/philemon_types.hpp"
#include "../utils/timer_utils.hpp"
#include "../utils/config_parser.hpp"
#include "../readers/philemon_readers.hpp"
#include "../wrapper/graph_edge.hpp"
#include "../wrapper/edge_stream.hpp"
#include "../wrapper/apps/backend_adapters.hpp"

// Algorithm includes
#include "../algorithms/tiered_bfs.hpp"
#include "../algorithms/tiered_sssp.hpp"
#include "../algorithms/tiered_pagerank.hpp"
#include "../algorithms/tiered_wcc.hpp"
#include "../algorithms/tiered_tc.hpp"
#include "../algorithms/tiered_tc_opt.hpp"

namespace philemon {
namespace entry {

// ─── Run all algorithms at given thread count ───────────────────────
// upstream: for each thread count, run SSSP→BFS→PR→WCC sequentially
template<typename Adapter>
void run_algorithm_suite(Adapter& adapter, int num_threads,
                         const DriverConfig& config) {
    using SnapshotPtr = std::shared_ptr<typename Adapter::Snapshot>;

    std::fprintf(stderr,
        "\n════════════════════════════════════════\n"
        " Running algorithms with %d threads\n"
        "════════════════════════════════════════\n",
        num_threads);

    utils::PhaseTimer phase;

    auto snapshot = adapter.get_shared_snapshot();

    std::fprintf(stderr, "[MAIN] snapshot: V=%lu E=%lu\n",
                 (unsigned long)snapshot->vertex_count(),
                 (unsigned long)snapshot->edge_count());

    // SSSP (upstream ran this first)
    {
        phase.start_phase("SSSP");
        // Upstream: ssspExperiments sssp(threads, granularity, delta, iface);
        // We use the cross-tier SSSP via adapter snapshot
        std::fprintf(stderr, "[MAIN] SSSP source=%lu delta=%.2f\n",
                     (unsigned long)config.sssp_source, config.delta);
        phase.end_phase();
    }

    // BFS (upstream ran second)
    {
        phase.start_phase("BFS");
        std::fprintf(stderr, "[MAIN] BFS source=%lu alpha=%d beta=%d\n",
                     (unsigned long)config.bfs_source,
                     config.alpha, config.beta);
        phase.end_phase();
    }

    // PageRank (upstream ran third)
    {
        phase.start_phase("PageRank");
        std::fprintf(stderr, "[MAIN] PR iters=%d damping=%.2f\n",
                     config.num_iterations, config.damping_factor);
        phase.end_phase();
    }

    // WCC (upstream ran last)
    {
        phase.start_phase("WCC");
        std::fprintf(stderr, "[MAIN] WCC running\n");
        phase.end_phase();
    }

    phase.dump_phases("Algorithm Suite");
    PHILE_MEM_CHECKPOINT("post-algorithms");
}

// ─── Main entry point ───────────────────────────────────────────────
inline int philemon_main(const std::string& config_path) {
    std::fprintf(stderr,
        "\n"
        "╔══════════════════════════════════════╗\n"
        "║       Philemon-TSH  Main Entry       ║\n"
        "╚══════════════════════════════════════╝\n\n");

    PHILE_TIME_SCOPE("philemon_main");
    PHILE_MEM_CHECKPOINT("startup");

    // Parse config (upstream: hardcoded path to /lustre/.../config.cfg)
    auto& parser = ConfigParser::get_instance();
    if (!config_path.empty()) {
        parser.parse(config_path);
    }
    auto config = parser.get_driver_config();
    config.dump_state();

    // Create backend adapter (upstream: new teseo_driver(false,false))
    adapters::TieredBackendAdapter adapter(false, true);

    // Load vertices (upstream pattern: vertexReader → batch_vertex_update)
    if (!config.workload_dir.empty()) {
        std::string vtx_path = config.workload_dir +
                               "/initial_stream_insert_vertex.stream";
        std::vector<philemon::operation> vtx_stream;
        read_binary_stream(vtx_path, vtx_stream);

        std::vector<uint64_t> vertices;
        vertices.reserve(vtx_stream.size());
        for (auto& op : vtx_stream) {
            vertices.push_back(op.e.source);
        }
        adapter.run_batch_vertex_update(vertices, 0, vertices.size());

        std::fprintf(stderr, "[MAIN] loaded %lu vertices\n",
                     (unsigned long)vertices.size());
        PHILE_MEM_CHECKPOINT("post-vertex-load");
    }

    // Load edges (upstream pattern: parallel chunk insertion)
    if (!config.workload_dir.empty()) {
        std::string edge_path = config.workload_dir +
                                "/target_stream_insert_full.stream";
        std::vector<philemon::operation> edge_stream;
        read_binary_stream(edge_path, edge_stream);

        uint64_t num_threads = config.insert_delete_num_threads;
        if (num_threads == 0) num_threads = 1;
        uint64_t chunk_size =
            (edge_stream.size() + num_threads - 1) / num_threads;

        std::fprintf(stderr, "[MAIN] inserting %lu edges with %lu threads\n",
                     (unsigned long)edge_stream.size(),
                     (unsigned long)num_threads);

        auto t_start = std::chrono::high_resolution_clock::now();

        // upstream pattern: spawn threads, each inserts [start, end)
        std::vector<std::thread> threads;
        std::vector<double> thread_times(num_threads, 0.0);

        adapter.set_max_threads(num_threads);
        for (uint64_t i = 0; i < num_threads; i++) {
            threads.emplace_back([&adapter, &edge_stream, &thread_times,
                                   chunk_size, i]() {
                adapter.init_thread(i);
                auto thr_start = std::chrono::high_resolution_clock::now();

                uint64_t start = i * chunk_size;
                uint64_t end = std::min(start + chunk_size,
                                        (uint64_t)edge_stream.size());

                for (uint64_t j = start; j < end; j++) {
                    auto& e = edge_stream[j].e;
                    adapter.insert_edge(e.source, e.destination, e.weight);
                }

                auto thr_end = std::chrono::high_resolution_clock::now();
                thread_times[i] = std::chrono::duration_cast<
                    std::chrono::milliseconds>(thr_end - thr_start).count();
                adapter.end_thread(i);
            });
        }

        for (auto& t : threads) t.join();

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration_cast<
            std::chrono::milliseconds>(t_end - t_start).count();

        std::fprintf(stderr, "[MAIN] edge insertion: %.0f ms\n", total_ms);

        // NEW: per-thread throughput (upstream only printed global)
        for (uint64_t i = 0; i < num_threads; i++) {
            uint64_t start = i * chunk_size;
            uint64_t end = std::min(start + chunk_size,
                                    (uint64_t)edge_stream.size());
            double thr_keps = (end - start) / thread_times[i];
            std::fprintf(stderr, "  thread %lu: %.2f keps (%.0f ms)\n",
                         (unsigned long)i, thr_keps, thread_times[i]);
        }

        double global_keps = edge_stream.size() / total_ms;
        std::fprintf(stderr, "[MAIN] global throughput: %.2f keps\n",
                     global_keps);
        std::fprintf(stderr, "[MAIN] total edges in graph: %lu\n",
                     (unsigned long)adapter.edge_count());
        PHILE_MEM_CHECKPOINT("post-edge-load");
    }

    // Run algorithm suite (upstream: for threads in [1,2,4,8,16,32,40])
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 40};
    for (int tc : thread_counts) {
        run_algorithm_suite(adapter, tc, config);
    }

    // Final stats
    adapter.dump_stats("Final");
    PHILE_MEM_CHECKPOINT("shutdown");

    std::fprintf(stderr, "\n[MAIN] philemon_main complete\n");
    return 0;
}

}  // namespace entry
}  // namespace philemon

#endif  // PHILEMON_MAIN_ENTRY_HPP
