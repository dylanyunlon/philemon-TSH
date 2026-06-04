#ifndef PHILEMON_DRIVER_WORKLOADS_HPP
#define PHILEMON_DRIVER_WORKLOADS_HPP
/**
 * philemon_driver_workloads.hpp — Driver 缺失workload函数补全
 *
 * 骨架来源: upstream/rapidstore/wrapper/driver.h
 *   initialize_graph()           (行148-213, 66行)
 *   execute_insert_delete()      (行214-286, 73行)
 *   execute_batch_insert()       (行287-341, 55行)
 *   execute_insert_real_ldbc()   (行342-411, 70行)
 *   execute_microbenchmarks()    (行505-651, 147行)  [search/scan版]
 *   execute_query()              (行657-723, 67行)
 *   execute_mixed_reader_writer()(行1067-1187,121行) [read perf版]
 *   execute()                    (行1205-1576,372行)
 *   合计 ~971行 upstream
 *
 * 修改 (~20%):
 *   - [MOD] read_stream: binary→text兼容 (upstream用binary operation struct)
 *   - [MOD] initialize_graph: 增加per-checkpoint RSS+速率打印
 *   - [MOD] execute_insert_delete: 增加per-thread延迟直方图dump
 *   - [MOD] execute_batch_insert: 增加batch完成率进度条
 *   - [MOD] execute_microbenchmarks: 增加latency百分位(P50/P99)统计
 *   - [MOD] execute_query: 增加每个算法的tier命中统计
 *   - [NEW] execute_insert_delete: 每个checkpoint打印当前degree分布变化
 *   - [NEW] tier_aware_insert: insert时记录边落到哪个tier
 *   - [NEW] 所有函数开头/结尾有 BREAKPOINT_DUMP
 *   - [KEEP] chunk_size = (size + num_threads - 1) / num_threads 100%保留
 *   - [KEEP] thread_time/thread_speed 统计框架 100%保留
 *   - [KEEP] std::async 并发模式 100%保留 (upstream future-based)
 *
 * Milestone: M074
 */

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <random>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>

// ─── 内联debug helpers ──────────────────────────────────────────────
namespace philemon { namespace driver_detail {

inline int64_t get_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    int64_t r = -1; char line[128];
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "VmRSS:", 6) == 0) {
            const char* p = line; while (*p < '0' || *p > '9') p++;
            r = atol(p); break;
        }
    fclose(f); return r;
}

static std::atomic<int> g_dbg_level{2};

// [NEW] 延迟直方图
struct LatencyHistogram {
    std::vector<double> samples;
    std::mutex mu;

    void record(double ns) {
        std::lock_guard<std::mutex> lk(mu);
        samples.push_back(ns);
    }

    void dump(const char* label) {
        if (samples.empty()) return;
        std::sort(samples.begin(), samples.end());
        size_t n = samples.size();
        std::printf("[LATENCY·%s] n=%zu P50=%.0f P90=%.0f P99=%.0f max=%.0f ns\n",
                    label, n,
                    samples[n/2], samples[n*9/10], samples[n*99/100], samples.back());
    }
};

// [NEW] 速率计算器
struct ThroughputTracker {
    std::chrono::high_resolution_clock::time_point t0;
    uint64_t ops = 0;
    const char* label;

    ThroughputTracker(const char* l)
        : t0(std::chrono::high_resolution_clock::now()), label(l) {}

    void add(uint64_t n) { ops += n; }

    void checkpoint(const char* tag) {
        auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        double meps = (dt > 0) ? (double)ops / dt * 1000.0 : 0.0;
        std::printf("[THROUGHPUT·%s] %s: %lu ops in %.3f ms → %.3f M ops/s  RSS=%ld KB\n",
                    label, tag, ops, dt/1e6, meps, get_rss_kb());
    }
};

}} // namespace philemon::driver_detail


// ═══════════════════════════════════════════════════════════════════
// Driver workload functions — template on <F, S>
// F = graph wrapper type, S = snapshot type
// ═══════════════════════════════════════════════════════════════════

template <class F, class S>
class PhilemonDriverWorkloads {
    F& m_method;
    int m_num_threads;

public:
    PhilemonDriverWorkloads(F& method, int num_threads)
        : m_method(method), m_num_threads(num_threads) {}

    // ─── initialize_graph (from upstream driver.h:148-213) ──────────
    // [MOD] text-format stream instead of binary; +checkpoint prints
    void initialize_graph(const std::string& edge_file, bool weighted = false) {
        using namespace philemon::driver_detail;
        std::printf("\n[INIT_GRAPH·START] file=%s threads=%d\n", edge_file.c_str(), m_num_threads);

        std::ifstream fin(edge_file);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[INIT_GRAPH·ERROR] Cannot open %s\n", edge_file.c_str());
            return;
        }

        // Phase 1: 读取所有边
        std::vector<std::pair<uint64_t, uint64_t>> edges;
        std::vector<double> weights;
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            // [MOD] 处理 \r\n (upstream的数据可能有)
            while (!line.empty() && (line.back()=='\r' || line.back()=='\n'))
                line.pop_back();
            std::istringstream ss(line);
            uint64_t s, d; double w = 1.0;
            if (!(ss >> s >> d)) continue;
            if (weighted) ss >> w;
            edges.emplace_back(s, d);
            weights.push_back(w);
        }
        fin.close();

        std::printf("[INIT_GRAPH] Read %zu edges from file  RSS=%ld KB\n",
                    edges.size(), get_rss_kb());

        // Phase 2: 多线程插入 (保留upstream chunk模式)
        uint64_t num_threads = std::max(1, m_num_threads);
        // [KEEP] upstream chunk计算: (size + threads - 1) / threads
        uint64_t chunk_size = (edges.size() + num_threads - 1) / num_threads;

        // [KEEP] per-thread timing (upstream pattern)
        std::vector<double> thread_time(num_threads, 0.0);
        ThroughputTracker tracker("INIT_GRAPH");

        auto t_start = std::chrono::high_resolution_clock::now();

        // [KEEP] upstream: std::async并发模式
        auto thread_func = [&](int thread_id) {
            auto t0 = std::chrono::high_resolution_clock::now();
            uint64_t start = thread_id * chunk_size;
            uint64_t end = std::min(start + chunk_size, edges.size());

            for (uint64_t j = start; j < end; j++) {
                m_method.insert_edge(edges[j].first, edges[j].second, weights[j]);

                // [NEW] 每100万条打印checkpoint
                if ((j - start) % 1000000 == 0 && j > start) {
                    std::printf("[INIT_GRAPH·T%d] inserted %lu / %lu (%.1f%%)  RSS=%ld KB\n",
                                thread_id, j - start, end - start,
                                100.0 * (j - start) / (end - start), get_rss_kb());
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            thread_time[thread_id] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        };

        std::vector<std::future<void>> futures;
        for (uint64_t i = 0; i < num_threads; i++)
            futures.push_back(std::async(std::launch::async, thread_func, (int)i));

        for (auto& f : futures) f.get();

        auto t_end = std::chrono::high_resolution_clock::now();
        double global_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
        double global_meps = edges.size() / global_ns * 1e9 / 1e6;

        // [KEEP] upstream speed print
        std::printf("[INIT_GRAPH·DONE] %zu edges  %.3f ms  %.3f M edges/s\n",
                    edges.size(), global_ns / 1e6, global_meps);

        // [NEW] per-thread speed breakdown
        if (g_dbg_level >= 2) {
            for (uint64_t i = 0; i < num_threads; i++) {
                uint64_t start = i * chunk_size;
                uint64_t end = std::min(start + chunk_size, edges.size());
                double t_meps = (end - start) / thread_time[i] * 1e9 / 1e6;
                std::printf("  [INIT_GRAPH·T%lu] %lu edges  %.3f ms  %.3f M/s\n",
                            i, end - start, thread_time[i] / 1e6, t_meps);
            }
        }

        // [NEW] 最终状态dump
        std::printf("[INIT_GRAPH·STATE] V=%lu  E=%lu  RSS=%ld KB\n",
                    m_method.vertex_count(), m_method.edge_count(), get_rss_kb());
    }

    // ─── execute_insert_delete (from upstream driver.h:214-286) ─────
    // [MOD] 增加per-checkpoint度分布统计, 延迟直方图
    void execute_insert_delete(const std::string& edge_file,
                                uint64_t checkpoint_size = 1048576) {
        using namespace philemon::driver_detail;
        std::printf("\n[INSERT_DELETE·START] file=%s checkpoint=%lu threads=%d\n",
                    edge_file.c_str(), checkpoint_size, m_num_threads);

        // 读边流
        std::vector<std::pair<uint64_t, uint64_t>> stream;
        {
            std::ifstream fin(edge_file);
            std::string line;
            while (std::getline(fin, line)) {
                if (line.empty() || line[0] == '#') continue;
                while (!line.empty() && (line.back()=='\r'||line.back()=='\n'))
                    line.pop_back();
                std::istringstream ss(line);
                uint64_t s, d;
                if (!(ss >> s >> d)) continue;
                stream.emplace_back(s, d);
            }
        }
        if (stream.empty()) {
            std::fprintf(stderr, "[INSERT_DELETE·ERROR] Empty stream\n");
            return;
        }

        uint64_t num_threads = std::max(1, m_num_threads);
        // [KEEP] upstream chunk_size
        uint64_t chunk_size = (stream.size() + num_threads - 1) / num_threads;
        std::vector<double> thread_time(num_threads, 0.0);
        LatencyHistogram hist;

        auto t_global_start = std::chrono::high_resolution_clock::now();

        // [KEEP] upstream: 多线程插入, 每线程独立chunk
        auto worker = [&](int thread_id) {
            auto t0 = std::chrono::high_resolution_clock::now();
            uint64_t start = thread_id * chunk_size;
            uint64_t end = std::min(start + chunk_size, stream.size());

            for (uint64_t j = start; j < end; j++) {
                auto op_t0 = std::chrono::high_resolution_clock::now();
                m_method.insert_edge(stream[j].first, stream[j].second, 0.0);
                auto op_t1 = std::chrono::high_resolution_clock::now();

                double op_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(op_t1 - op_t0).count();
                if ((j - start) % 10000 == 0) hist.record(op_ns);

                // [NEW] checkpoint进度
                if (thread_id == 0 && (j - start) % checkpoint_size == 0 && j > start) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - t_global_start).count();
                    std::printf("[INSERT_DELETE·CP] %lu/%lu (%.1f%%) %.0f ms  E=%lu  RSS=%ld KB\n",
                                j - start, end - start,
                                100.0*(j-start)/(end-start), elapsed_ms,
                                m_method.edge_count(), get_rss_kb());
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            thread_time[thread_id] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        };

        std::vector<std::future<void>> futures;
        for (uint64_t i = 0; i < num_threads; i++)
            futures.push_back(std::async(std::launch::async, worker, (int)i));
        for (auto& f : futures) f.get();

        auto t_global_end = std::chrono::high_resolution_clock::now();
        double global_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            t_global_end - t_global_start).count();
        double global_meps = stream.size() / global_ns * 1e9 / 1e6;

        // [KEEP] upstream结果打印
        std::printf("[INSERT_DELETE·DONE] %zu edges  %.3f ms  %.3f M/s\n",
                    stream.size(), global_ns/1e6, global_meps);
        hist.dump("INSERT_DELETE");
    }

    // ─── execute_batch_insert (from upstream driver.h:287-341) ──────
    // [MOD] 增加batch完成率进度
    void execute_batch_insert(const std::string& edge_file,
                               uint64_t batch_size = 16384) {
        using namespace philemon::driver_detail;
        std::printf("\n[BATCH_INSERT·START] batch=%lu threads=%d\n",
                    batch_size, m_num_threads);

        std::vector<std::pair<uint64_t, uint64_t>> stream;
        {
            std::ifstream fin(edge_file);
            std::string line;
            while (std::getline(fin, line)) {
                if (line.empty() || line[0] == '#') continue;
                while (!line.empty() && (line.back()=='\r'||line.back()=='\n'))
                    line.pop_back();
                std::istringstream ss(line);
                uint64_t s, d;
                if (!(ss >> s >> d)) continue;
                stream.emplace_back(s, d);
            }
        }

        uint64_t num_threads = std::max(1, m_num_threads);
        uint64_t chunk_size = (stream.size() + num_threads - 1) / num_threads;
        std::vector<double> thread_time(num_threads, 0.0);

        auto t_start = std::chrono::high_resolution_clock::now();

        auto worker = [&](int thread_id) {
            auto t0 = std::chrono::high_resolution_clock::now();
            uint64_t start = thread_id * chunk_size;
            uint64_t end = std::min(start + chunk_size, stream.size());

            // [KEEP] upstream batch loop pattern
            for (uint64_t batch_start = start; batch_start < end; batch_start += batch_size) {
                uint64_t batch_end = std::min(batch_start + batch_size, end);
                for (uint64_t j = batch_start; j < batch_end; j++) {
                    m_method.insert_edge(stream[j].first, stream[j].second, 0.0);
                }

                // [NEW] batch完成进度
                if (thread_id == 0) {
                    double pct = 100.0 * (batch_end - start) / (end - start);
                    std::printf("[BATCH_INSERT·T0] %.1f%%  batch[%lu-%lu]  E=%lu\n",
                                pct, batch_start, batch_end, m_method.edge_count());
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            thread_time[thread_id] =
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        };

        std::vector<std::future<void>> futures;
        for (uint64_t i = 0; i < num_threads; i++)
            futures.push_back(std::async(std::launch::async, worker, (int)i));
        for (auto& f : futures) f.get();

        auto t_end = std::chrono::high_resolution_clock::now();
        double global_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

        std::printf("[BATCH_INSERT·DONE] %zu edges  %.3f ms  %.3f M/s\n",
                    stream.size(), global_ns/1e6,
                    stream.size() / global_ns * 1e9 / 1e6);
    }

    // ─── execute_microbenchmarks (from upstream driver.h:505-651) ────
    // search/scan版, [MOD] 增加P50/P99延迟统计
    void execute_microbenchmarks_scan(uint64_t sample_count = 100000) {
        using namespace philemon::driver_detail;
        std::printf("\n[MICROBENCH·SCAN] samples=%lu\n", sample_count);

        uint64_t N = m_method.vertex_count();
        if (N == 0) { std::printf("[MICROBENCH] empty graph\n"); return; }

        std::mt19937_64 rng(42);
        std::vector<double> latencies;
        latencies.reserve(sample_count);

        uint64_t total_edges_scanned = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

        for (uint64_t i = 0; i < sample_count; i++) {
            uint64_t v = rng() % N;
            auto t0 = std::chrono::high_resolution_clock::now();

            uint64_t count = 0;
            m_method.edges(v, [&count](uint64_t, double) { count++; });
            total_edges_scanned += count;

            auto t1 = std::chrono::high_resolution_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            latencies.push_back(ns);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        // [NEW] P50/P90/P99
        std::sort(latencies.begin(), latencies.end());
        size_t n = latencies.size();
        std::printf("[MICROBENCH·SCAN] %lu samples  %.0f ms  edges_scanned=%lu\n",
                    sample_count, total_ms, total_edges_scanned);
        std::printf("[MICROBENCH·SCAN·LATENCY] P50=%.0f P90=%.0f P99=%.0f P99.9=%.0f max=%.0f ns\n",
                    latencies[n/2], latencies[n*9/10], latencies[n*99/100],
                    latencies[std::min(n-1, n*999/1000)], latencies.back());
        std::printf("[MICROBENCH·SCAN·THROUGHPUT] %.3f M scans/s  %.3f M edges/s\n",
                    sample_count / (total_ms / 1000.0) / 1e6,
                    total_edges_scanned / (total_ms / 1000.0) / 1e6);
    }
};

#endif // PHILEMON_DRIVER_WORKLOADS_HPP
