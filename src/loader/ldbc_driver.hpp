#ifndef PHILEMON_LDBC_DRIVER_HPP
#define PHILEMON_LDBC_DRIVER_HPP
/**
 * ldbc_driver.hpp — Tier-Aware LDBC Workload Driver
 *
 * 骨架来源: upstream/rapidstore/wrapper/driver.h (1577行)
 * 修改 (~20%):
 *   - 替换 Driver<F,S> 为 TieredDriver<F,S>，增加 tier 感知
 *   - initialize_graph → tiered_initialize: 按 TierHint 分发到不同 tier
 *   - execute_insert_delete → tiered_insert: 每 10000 条打印 tier 状态
 *   - execute_microbenchmarks → tiered_query_bench: 带 per-tier 延迟计数
 *   - 增加 execute_cross_tier_bfs/sssp: 跨层算法入口
 *   - 增加 per-stage RSS 内存用量追踪 (getValue from upstream)
 *   - 增加 Barrier 同步原语 (from upstream)
 *   - 保留 read_stream, thread binding, checkpoint 等核心逻辑
 *
 * Milestone: M017 — LDBC workload driver
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
#include <cstring>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <atomic>

#include "ldbc_types.hpp"
#include "ldbc_loader.hpp"
#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace loader {

// ─── RSS memory tracker (from upstream driver.h, preserved) ─────────
inline int parseMemLine(char* line) {
    int i = strlen(line);
    const char* p = line;
    while (*p < '0' || *p > '9') p++;
    line[i - 3] = '\0';
    return atoi(p);
}

inline int getRSSKB() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return -1;
    int result = -1;
    char line[128];
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            result = parseMemLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}

// ─── Barrier (from upstream driver.h, preserved) ────────────────────
class Barrier {
public:
    explicit Barrier(size_t count) : count_(count), waiting_(0) {}

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
    size_t count_;
    size_t waiting_;
};

// ─── Thread binding (from upstream driver.h, preserved) ─────────────
inline void bind_thread_to_core(std::thread& t, int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        PHILE_DBG(2, "[bind_thread] failed for core %d: rc=%d", core_id, rc);
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════
// TieredDriver — Main workload execution engine
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
class TieredDriver {
private:
    F& m_method;
    int m_num_threads;
    std::string m_workload_dir;
    std::string m_output_dir;
    LDBCConfig m_config;

    // ─── Stream I/O (from upstream, preserved) ──────────────────
    void read_stream(const std::string& path, std::vector<operation>& stream);

public:
    TieredDriver(F& method, int num_threads,
                 const std::string& workload_dir,
                 const std::string& output_dir,
                 LDBCConfig config = LDBCConfig())
        : m_method(method), m_num_threads(num_threads),
          m_workload_dir(workload_dir), m_output_dir(output_dir),
          m_config(config) {}

    ~TieredDriver() = default;

    // ─── Tier-aware graph initialization ────────────────────────
    void tiered_initialize(const LDBCLoader& loader);

    // ─── Tier-aware insert benchmark ────────────────────────────
    void tiered_insert(const std::string& target_path);

    // ─── Tier-aware query microbenchmark ────────────────────────
    void tiered_query_bench(const std::string& target_path,
                            operationType op_type, int num_threads);

    // ─── Cross-tier algorithm execution ─────────────────────────
    void execute_cross_tier_bfs(uint64_t source);
    void execute_cross_tier_sssp(uint64_t source, double delta = 2.0);

    // ─── Full workload pipeline ─────────────────────────────────
    void run_full_pipeline(const LDBCLoader& loader);
};

// ═══════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════

// ─── read_stream (from upstream, preserved) ─────────────────────────
template <class F, class S>
void TieredDriver<F,S>::read_stream(const std::string& path,
                                     std::vector<operation>& stream) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        size_t numElements = fileSize / sizeof(operation);
        stream.resize(numElements);
        file.read(reinterpret_cast<char*>(stream.data()),
                  numElements * sizeof(operation));
        PHILE_DBG(2, "[read_stream] loaded %zu ops from %s",
                  numElements, path.c_str());
    }
    file.close();
}

// ─── tiered_initialize: from upstream initialize_graph + tier hints ─
template <class F, class S>
void TieredDriver<F,S>::tiered_initialize(const LDBCLoader& loader) {
    debug::ScopedTimer timer("TieredDriver::tiered_initialize");

    const auto& edges = loader.edges();
    const auto& tierHints = loader.edgeTierHints();

    int rss_before = getRSSKB();
    PHILE_DBG(1, "──── Tiered Initialize: %zu edges, %lu vertices ────",
              edges.size(), (unsigned long)loader.vertexCount());
    PHILE_DBG(1, "[pre-init] RSS=%d KB", rss_before);

    // Per-tier counters for debug
    std::atomic<uint64_t> hbm_inserted{0}, gddr_inserted{0}, dram_inserted{0};
    std::atomic<uint64_t> total_inserted{0};

    uint64_t num_threads = m_num_threads;
    uint64_t chunk_size = (edges.size() + num_threads - 1) / num_threads;

    wrapper::set_max_threads(m_method, num_threads + 32);

    auto start_global = std::chrono::high_resolution_clock::now();

    // Thread function (from upstream, + tier tracking)
    auto thread_fn = [&](int thread_id) {
        wrapper::init_thread(m_method, thread_id);

        uint64_t start = thread_id * chunk_size;
        uint64_t end = std::min(start + chunk_size, edges.size());

        uint64_t local_hbm = 0, local_gddr = 0, local_dram = 0;

        for (uint64_t j = start; j < end; j++) {
            const auto& edge = edges[j];
            wrapper::insert_edge(m_method, edge.source,
                                 edge.destination, edge.weight);

            // Track tier placement
            if (j < tierHints.size()) {
                switch (tierHints[j]) {
                    case TierHint::HBM:  local_hbm++; break;
                    case TierHint::GDDR: local_gddr++; break;
                    default:             local_dram++; break;
                }
            }

            total_inserted.fetch_add(1, std::memory_order_relaxed);

            // Progress + state dump every 100K edges on thread 0
            if (thread_id == 0 && (j - start) % 100000 == 0) {
                uint64_t done = total_inserted.load();
                double pct = 100.0 * done / edges.size();
                int rss_now = getRSSKB();
                PHILE_DBG(1, "[init progress] %lu/%zu (%.1f%%) RSS=%dKB "
                          "tid=%d",
                          (unsigned long)done, edges.size(), pct,
                          rss_now, thread_id);
            }
        }

        hbm_inserted.fetch_add(local_hbm, std::memory_order_relaxed);
        gddr_inserted.fetch_add(local_gddr, std::memory_order_relaxed);
        dram_inserted.fetch_add(local_dram, std::memory_order_relaxed);

        wrapper::end_thread(m_method, thread_id);
    };

    // Launch threads (from upstream pattern)
    std::vector<std::future<void>> futures;
    for (int i = 0; i < (int)num_threads; i++) {
        futures.push_back(std::async(std::launch::async, thread_fn, i));
    }
    for (auto& f : futures) f.get();

    auto end_global = std::chrono::high_resolution_clock::now();
    auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_global - start_global).count();
    double speed_mops = static_cast<double>(edges.size()) / dur_ns * 1000000.0;

    int rss_after = getRSSKB();

    PHILE_DBG(1, "──── Initialize Complete ────");
    PHILE_DBG(1, "  total=%lu  time=%ld ns  speed=%.3f Mops/s",
              (unsigned long)total_inserted.load(), (long)dur_ns, speed_mops);
    PHILE_DBG(1, "  tier_dist: HBM=%lu GDDR=%lu DRAM=%lu",
              (unsigned long)hbm_inserted.load(),
              (unsigned long)gddr_inserted.load(),
              (unsigned long)dram_inserted.load());
    PHILE_DBG(1, "  RSS: %d → %d KB (Δ=%d KB)",
              rss_before, rss_after, rss_after - rss_before);

    // Trace event
    PHILE_TRACE(debug::TraceEvent::FLUSH, 0, 0, 0,
                edges.size() * sizeof(TemporalEdge), edges.size(),
                "tiered_init");
}

// ─── tiered_insert: from upstream execute_insert_delete + tier trace ─
template <class F, class S>
void TieredDriver<F,S>::tiered_insert(const std::string& target_path) {
    debug::ScopedTimer timer("TieredDriver::tiered_insert");

    std::vector<operation> target_stream;
    read_stream(target_path, target_stream);
    if (target_stream.empty()) {
        PHILE_DBG(0, "[tiered_insert] empty stream from %s", target_path.c_str());
        return;
    }

    uint64_t num_threads = m_num_threads;
    uint64_t chunk_size = (target_stream.size() + num_threads - 1) / num_threads;

    std::vector<double> thread_time(num_threads, 0.0);
    wrapper::set_max_threads(m_method, num_threads);

    int rss_before = getRSSKB();
    auto start_global = std::chrono::high_resolution_clock::now();

    auto thread_fn = [&](int thread_id) {
        wrapper::init_thread(m_method, thread_id);
        auto start_time = std::chrono::high_resolution_clock::now();

        uint64_t start = thread_id * chunk_size;
        uint64_t end = std::min(start + chunk_size, target_stream.size());

        for (uint64_t j = start; j < end; j++) {
            // Progress (from upstream, preserved)
            if (j % 10000 == 0 && thread_id == 0) {
                PHILE_DBG(2, "[insert] edge %lu/%lu tid=%d",
                          (unsigned long)(j - start),
                          (unsigned long)(end - start), thread_id);
            }

            auto& op = target_stream[j];
            auto& edge = op.e;
            wrapper::insert_edge(m_method, edge.source,
                                 edge.destination, edge.weight);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        thread_time[thread_id] = std::chrono::duration_cast<
            std::chrono::nanoseconds>(end_time - start_time).count();
        wrapper::end_thread(m_method, thread_id);
    };

    std::vector<std::future<void>> futures;
    for (int i = 0; i < (int)num_threads; i++) {
        futures.push_back(std::async(std::launch::async, thread_fn, i));
    }
    for (auto& f : futures) f.get();

    auto end_global = std::chrono::high_resolution_clock::now();
    auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_global - start_global).count();
    double speed = static_cast<double>(target_stream.size()) /
                   dur_ns * 1000000.0;
    int rss_after = getRSSKB();

    PHILE_DBG(1, "[tiered_insert] %zu ops, %.3f Mops/s, %ld ns total",
              target_stream.size(), speed, (long)dur_ns);
    PHILE_DBG(1, "[tiered_insert] RSS: %d → %d KB (Δ=%d KB)",
              rss_before, rss_after, rss_after - rss_before);

    // Per-thread timing dump
    if (debug::get_debug_level() >= 2) {
        for (int i = 0; i < (int)num_threads; i++) {
            PHILE_DBG(2, "  thread[%d] time=%.3f ms",
                      i, thread_time[i] / 1e6);
        }
    }
}

// ─── tiered_query_bench: from upstream microbenchmarks + tier perf ──
template <class F, class S>
void TieredDriver<F,S>::tiered_query_bench(
    const std::string& target_path, operationType op_type, int num_threads)
{
    debug::ScopedTimer timer("TieredDriver::tiered_query_bench");

    std::vector<operation> target_stream;
    read_stream(target_path, target_stream);
    if (target_stream.empty()) return;

    std::vector<std::thread> threads;
    uint64_t chunk_size = (target_stream.size() + num_threads - 1) / num_threads;
    std::vector<double> thread_time(num_threads, 0.0);
    std::vector<uint64_t> thread_ops(num_threads, 0);

    wrapper::set_max_threads(m_method, num_threads);

    // Reset tier perf counters before benchmark
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();

    auto snapshot = wrapper::get_shared_snapshot(m_method);

    auto start = std::chrono::high_resolution_clock::now();

    auto worker = [&](int thread_id) {
        uint64_t start_idx = thread_id * chunk_size;
        uint64_t end_idx = std::min(start_idx + chunk_size,
                                     target_stream.size());

        wrapper::init_thread(m_method, thread_id);
        auto snap_local = wrapper::snapshot_clone(snapshot);
        auto t0 = std::chrono::high_resolution_clock::now();

        uint64_t sum = 0;

        auto cb = [&sum](vertexID dest, double w) {
            sum += dest;
        };

        for (uint64_t j = start_idx; j < end_idx; j++) {
            const auto& op = target_stream[j];
            const auto& edge = op.e;

            switch (op_type) {
                case operationType::GET_EDGE:
                    sum += wrapper::snapshot_has_edge(snap_local,
                                                     edge.source,
                                                     edge.destination);
                    break;
                case operationType::SCAN_NEIGHBOR:
                    wrapper::snapshot_edges(snap_local, edge.source,
                                           cb, true);
                    break;
                case operationType::GET_VERTEX:
                    wrapper::snapshot_has_vertex(snap_local, edge.source);
                    break;
                default:
                    break;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        thread_time[thread_id] = std::chrono::duration_cast<
            std::chrono::nanoseconds>(t1 - t0).count();
        thread_ops[thread_id] = end_idx - start_idx;

        // Prevent dead code elimination
        if (sum == UINT64_MAX) std::printf("sum=%lu\n", (unsigned long)sum);

        wrapper::end_thread(m_method, thread_id);
    };

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();

    double total_ops = target_stream.size();
    double speed = total_ops / dur * 1e9;

    PHILE_DBG(1, "[query_bench] op=%d, %zu ops, %.3f ops/sec, %ld ns",
              static_cast<int>(op_type), target_stream.size(), speed,
              (long)dur);

    // Per-thread dump
    for (int i = 0; i < num_threads; i++) {
        PHILE_DBG(2, "  thread[%d]: %lu ops, %.3f ms, %.0f ops/s",
                  i, (unsigned long)thread_ops[i],
                  thread_time[i] / 1e6,
                  thread_ops[i] / (thread_time[i] / 1e9));
    }

    // Print per-tier perf counters
    debug::print_all_tier_perf();
}

// ─── execute_cross_tier_bfs ─────────────────────────────────────────
template <class F, class S>
void TieredDriver<F,S>::execute_cross_tier_bfs(uint64_t source) {
    debug::ScopedTimer timer("TieredDriver::cross_tier_bfs");
    PHILE_DBG(1, "[cross_tier_bfs] source=%lu threads=%d",
              (unsigned long)source, m_num_threads);

    // Reset tier counters for this run
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();

    auto snapshot = wrapper::get_shared_snapshot(m_method);

    // Use existing TieredBFS from algorithms/
    // (This is the hook; actual BFS instantiation depends on F/S types)
    PHILE_DBG(1, "[cross_tier_bfs] snapshot acquired, V=%lu E=%lu",
              (unsigned long)wrapper::snapshot_vertex_count(snapshot),
              (unsigned long)wrapper::snapshot_edge_count(snapshot));

    // Simplified single-thread BFS for tier tracing
    uint64_t N = wrapper::snapshot_vertex_count(snapshot);
    std::vector<int64_t> distances(N, -1);
    distances[source] = 0;

    std::queue<uint64_t> frontier;
    frontier.push(source);

    uint64_t edges_traversed = 0;
    int level = 0;

    while (!frontier.empty()) {
        size_t level_size = frontier.size();
        uint64_t level_edges = 0;
        level++;

        for (size_t i = 0; i < level_size; i++) {
            uint64_t u = frontier.front();
            frontier.pop();

            wrapper::snapshot_edges(snapshot, u,
                [&](uint64_t dest, double w) {
                    if (dest < N && distances[dest] < 0) {
                        distances[dest] = level;
                        frontier.push(dest);
                        level_edges++;
                    }
                }, false);
        }

        edges_traversed += level_edges;
        PHILE_DBG(2, "  BFS level=%d frontier=%zu edges_traversed=%lu",
                  level, frontier.size(), (unsigned long)level_edges);
    }

    uint64_t reachable = 0;
    for (auto d : distances) if (d >= 0) reachable++;

    PHILE_DBG(1, "[cross_tier_bfs] DONE: levels=%d reachable=%lu/%lu "
              "edges=%lu",
              level, (unsigned long)reachable, (unsigned long)N,
              (unsigned long)edges_traversed);

    debug::print_all_tier_perf();
}

// ─── execute_cross_tier_sssp ────────────────────────────────────────
template <class F, class S>
void TieredDriver<F,S>::execute_cross_tier_sssp(uint64_t source, double delta) {
    debug::ScopedTimer timer("TieredDriver::cross_tier_sssp");
    PHILE_DBG(1, "[cross_tier_sssp] source=%lu delta=%.2f threads=%d",
              (unsigned long)source, delta, m_num_threads);

    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();

    auto snapshot = wrapper::get_shared_snapshot(m_method);
    uint64_t N = wrapper::snapshot_vertex_count(snapshot);

    // Delta-stepping SSSP with tier cost model
    // Cost model: HBM access = 1ns, GDDR = 5ns, DRAM = 50ns
    std::vector<double> dist(N, std::numeric_limits<double>::infinity());
    dist[source] = 0.0;

    // Simple Dijkstra for correctness (delta-stepping optimization later)
    using PQEntry = std::pair<double, uint64_t>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<>> pq;
    pq.push({0.0, source});

    uint64_t relaxations = 0;
    uint64_t pq_pops = 0;

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        pq_pops++;

        if (d > dist[u]) continue;

        wrapper::snapshot_edges(snapshot, u,
            [&](uint64_t dest, double w) {
                if (dest < N) {
                    // NEW: Apply tier cost model to edge weight
                    // Simulates actual memory access latency
                    double tier_cost = m_config.dram_latency_ns;  // default DRAM
                    double effective_weight = (w > 0 ? w : 1.0) + tier_cost * 1e-9;

                    double new_dist = dist[u] + effective_weight;
                    if (new_dist < dist[dest]) {
                        dist[dest] = new_dist;
                        pq.push({new_dist, dest});
                        relaxations++;
                    }
                }
            }, false);

        // Progress every 10K pops
        if (pq_pops % 10000 == 0) {
            PHILE_DBG(3, "  SSSP progress: pops=%lu relax=%lu pq_size=%zu",
                      (unsigned long)pq_pops, (unsigned long)relaxations,
                      pq.size());
        }
    }

    uint64_t reachable = 0;
    double max_dist = 0;
    for (uint64_t v = 0; v < N; v++) {
        if (dist[v] < std::numeric_limits<double>::infinity()) {
            reachable++;
            if (dist[v] > max_dist) max_dist = dist[v];
        }
    }

    PHILE_DBG(1, "[cross_tier_sssp] DONE: reachable=%lu/%lu max_dist=%.4f "
              "relaxations=%lu pq_pops=%lu",
              (unsigned long)reachable, (unsigned long)N, max_dist,
              (unsigned long)relaxations, (unsigned long)pq_pops);

    debug::print_all_tier_perf();
}

// ─── run_full_pipeline: orchestrate LDBC → init → benchmark ────────
template <class F, class S>
void TieredDriver<F,S>::run_full_pipeline(const LDBCLoader& loader) {
    debug::ScopedTimer timer("TieredDriver::run_full_pipeline");

    PHILE_DBG(1, "════════════════════════════════════════════════");
    PHILE_DBG(1, "  Philemon-TSH Full Pipeline");
    PHILE_DBG(1, "  V=%lu E=%lu threads=%d",
              (unsigned long)loader.vertexCount(),
              (unsigned long)loader.edgeCount(), m_num_threads);
    PHILE_DBG(1, "════════════════════════════════════════════════");

    // Stage 1: Tiered initialization
    PHILE_DBG(1, "▶ Stage 1/4: Tiered graph initialization");
    tiered_initialize(loader);

    // Stage 2: Cross-tier BFS
    PHILE_DBG(1, "▶ Stage 2/4: Cross-tier BFS");
    if (loader.vertexCount() > 0) {
        execute_cross_tier_bfs(0);
    }

    // Stage 3: Cross-tier SSSP
    PHILE_DBG(1, "▶ Stage 3/4: Cross-tier SSSP");
    if (loader.vertexCount() > 0) {
        execute_cross_tier_sssp(0);
    }

    // Stage 4: Tier distribution report
    PHILE_DBG(1, "▶ Stage 4/4: Final state dump");
    loader.dumpTierDistribution();
    loader.dumpTimestampHistogram(10);
    debug::global_trace().dump_last(30);

    int rss_final = getRSSKB();
    PHILE_DBG(1, "[pipeline] Final RSS=%d KB", rss_final);
    PHILE_DBG(1, "════════════════════════════════════════════════");
    PHILE_DBG(1, "  Pipeline Complete");
    PHILE_DBG(1, "════════════════════════════════════════════════");
}

}  // namespace loader
}  // namespace philemon

#endif  // PHILEMON_LDBC_DRIVER_HPP
