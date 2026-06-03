#ifndef PHILEMON_DRIVER_EXT_HPP
#define PHILEMON_DRIVER_EXT_HPP
/**
 * philemon_driver_ext.hpp — 实验驱动器模板 (跨层级 benchmark 调度)
 *
 * 骨架来源: upstream/rapidstore/wrapper/driver.h (1577行, 前200行核心)
 * 修改 (~20%):
 *   - 保留 Driver<F,S> 模板, initialize_graph, execute 等骨架
 *   - 移除 ittnotify / perf_event Linux 特定依赖
 *   - Barrier 类保留 (upstream arrive_and_wait)
 *   - 增加 dump_driver_state(): 打印图加载后的完整状态
 *   - 增加 per-phase 计时打印 (load_vertices / load_edges / run_algo)
 *   - 增加 memory footprint 打印 (在每个 checkpoint)
 *   - bind_thread_to_core 保留但改为可选 (PHILEMON_PIN_THREADS)
 *
 * Milestone: M028
 */

#include <random>
#include <string>
#include <future>
#include <queue>
#include <vector>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <functional>

#include "../utils/log/philemon_log.hpp"
#include "../utils/timer_utils_ext.hpp"
#include "philemon_wrapper_ops.hpp"

// ─── Thread pinning (optional, from upstream) ───────────────────────
#ifdef PHILEMON_PIN_THREADS
#include <pthread.h>
inline void philemon_bind_thread(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) plog_warn("Thread pin to core %d failed: %d", core_id, rc);
}
#else
inline void philemon_bind_thread(std::thread&, int) {}
#endif

// ─── Barrier (from upstream, unchanged) ─────────────────────────────
class PhilemonBarrier {
public:
    explicit PhilemonBarrier(std::size_t count) : count_(count), waiting_(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        ++waiting_;
        if (waiting_ == count_) {
            waiting_ = 0;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return waiting_ == 0; });
        }
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::size_t count_;
    std::size_t waiting_;
};

// ─── Memory helper (from upstream parseLine/getValue) ────────────────
inline int64_t philemon_get_rss_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return -1;
    int64_t result = -1;
    char line[128];
    while (fgets(line, sizeof(line), file) != nullptr) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            const char* p = line;
            while (*p < '0' || *p > '9') p++;
            result = atol(p);
            break;
        }
    }
    fclose(file);
    return result;
}

// ─── Driver Template ────────────────────────────────────────────────
template <class F, class S>
class PhilemonDriver {
private:
    F& m_method;
    int m_num_threads;
    std::string m_workload_dir;
    std::string m_output_dir;

public:
    PhilemonDriver(F& method, int num_threads,
                   const std::string& workload_dir,
                   const std::string& output_dir)
        : m_method(method), m_num_threads(num_threads),
          m_workload_dir(workload_dir), m_output_dir(output_dir) {}

    ~PhilemonDriver() = default;

    // ─── Graph initialization (from upstream, with debug) ───────────
    void initialize_graph(const std::string& vertex_path,
                          const std::string& edge_path) {
        philemon::utils::Timer timer("DRIVER");

        // Phase 1: load vertices
        plog_info("[DRIVER] Loading vertices from %s", vertex_path.c_str());
        std::ifstream vfile(vertex_path);
        std::vector<uint64_t> vertices;
        uint64_t vid;
        while (vfile >> vid) vertices.push_back(vid);
        vfile.close();

        wrapper::run_batch_vertex_update(m_method, vertices, 0, vertices.size());
        timer.lap("vertices_loaded");

        plog_info("[DRIVER] Loaded %zu vertices  RSS=%ld KB",
                  vertices.size(), philemon_get_rss_kb());

        // Phase 2: load edges
        plog_info("[DRIVER] Loading edges from %s", edge_path.c_str());
        std::ifstream efile(edge_path);
        uint64_t src, dst;
        uint64_t edge_count = 0;

        uint64_t chunk_size = 100000; // batch insert
        std::vector<std::pair<uint64_t, uint64_t>> batch;
        batch.reserve(chunk_size);

        while (efile >> src >> dst) {
            batch.push_back({src, dst});
            edge_count++;

            if (batch.size() >= chunk_size) {
                insert_edge_batch(batch);
                batch.clear();

                // ─── NEW: 每10万条打印进度 ──────────────────────────
                if (edge_count % 1000000 == 0) {
                    plog_info("[DRIVER] edges_loaded=%lu  RSS=%ld KB  elapsed=%.2f s",
                              edge_count, philemon_get_rss_kb(), timer.elapsed());
                }
            }
        }
        if (!batch.empty()) insert_edge_batch(batch);
        efile.close();
        timer.lap("edges_loaded");

        plog_info("[DRIVER] Graph initialized: V=%lu  E=%lu  RSS=%ld KB",
                  (uint64_t)vertices.size(), edge_count, philemon_get_rss_kb());

        // ─── NEW: 打印完整加载状态 ─────────────────────────────────
        dump_driver_state("POST_INIT");
        timer.dump_laps();
    }

    // ─── NEW: 打印驱动器完整状态 ────────────────────────────────────
    void dump_driver_state(const char* label = "") const {
        std::printf("\n╔═══════════════════════════════════════════════╗\n");
        std::printf("║  [DRIVER_STATE] %s\n", label);
        std::printf("║  vertices   = %lu\n", wrapper::vertex_count(const_cast<F&>(m_method)));
        std::printf("║  edges      = %lu\n", wrapper::edge_count(const_cast<F&>(m_method)));
        std::printf("║  threads    = %d\n", m_num_threads);
        std::printf("║  workload   = %s\n", m_workload_dir.c_str());
        std::printf("║  output     = %s\n", m_output_dir.c_str());
        std::printf("║  RSS        = %ld KB\n", philemon_get_rss_kb());
        std::printf("╚═══════════════════════════════════════════════╝\n\n");

        wrapper::counters::dump(label);
    }

private:
    void insert_edge_batch(const std::vector<std::pair<uint64_t, uint64_t>>& batch) {
        for (auto& [s, d] : batch) {
            wrapper::insert_edge(m_method, s, d, 0.0);
        }
    }
};

#endif // PHILEMON_DRIVER_EXT_HPP
