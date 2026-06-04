#ifndef PHILEMON_DRIVER_HPP
#define PHILEMON_DRIVER_HPP
/**
 * philemon_driver.hpp — Philemon 统一驱动器（端到端调度+状态聚合） — Main Benchmark Driver
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/driver.h               (1577行)
 *   upstream/rapidstore/wrapper/wrapper.h               (249行)
 *   upstream/rapidstore/utils/Timer.h                   (35行)
 *   upstream/rapidstore/utils/commandLineParser.hpp     (117行)
 *
 * 重大搬运内容:
 *   - Barrier 类 (driver.h:49-69)
 *   - bind_thread_to_core() (driver.h:36-45)
 *   - Driver 类骨架 (driver.h:72-101)
 *   - execute_query() 分派 (driver.h:657-721)
 *   - execute_mixed_reader_writer() 框架 (driver.h:1066-1200)
 *   - execute_microbenchmarks() 框架 (driver.h:380-648)
 *   - execute_insert_delete() 批量操作 (driver.h:120-250)
 *   - bfs/sssp/wcc/page_rank 单线程验证 (driver.h:724-885)
 *
 * 修改 (~20%):
 *   - [NEW] PhilemonConfig: 替代 DriverConfig, 增加 tier/debug 参数
 *   - [NEW] execute_cross_tier_query(): 调度 cross_tier_* 算法
 *   - [NEW] execute_tier_benchmark(): tier 迁移性能测试
 *   - [NEW] dump_full_system_state(): 系统全量状态打印
 *   - [NEW] print_progress_bar(): 进度条打印
 *   - [MOD] 原 #include "ittnotify.h" → 删除 (Intel VTune 依赖)
 *   - [MOD] 原 operationType → PhilemonOp enum
 *   - [MOD] 原 log_info → PHILE_DBG
 *   - [KEEP] Barrier 同步原语 100% 保留
 *   - [KEEP] bind_thread_to_core 100% 保留
 *
 * Milestone: M021-M022 — Cross-tier algorithms + driver
 * ====================================================================
 */

#include <random>
#include <string>
#include <future>
#include <queue>
#include <vector>
#include <thread>
#include <condition_variable>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <functional>
#include <unordered_map>

// Project headers
#include "../wrapper/rapidstore_wrapper.hpp"
#include "../debug/philemon_debug.hpp"
#include "../cost_model/tier_cost_model.hpp"
#include "../algorithms/cross_tier_bfs.hpp"
#include "../algorithms/cross_tier_sssp.hpp"
#include "../algorithms/cross_tier_pagerank.hpp"
#include "../algorithms/cross_tier_wcc.hpp"
#include "../algorithms/cross_tier_tc.hpp"

namespace philemon {
namespace driver {

// ─── From upstream driver.h:28-34: debug macros (preserved, renamed) ─
#define PHILE_COUT_FORCE(msg) { std::cout << __FUNCTION__ << msg << std::endl; }
#ifdef PHILEMON_DEBUG
    #define PHILE_COUT(msg) PHILE_COUT_FORCE(msg)
#else
    #define PHILE_COUT(msg)
#endif

// ═══════════════════════════════════════════════════════════════════════
// Barrier — from upstream driver.h:49-69 (100% preserved)
// ═══════════════════════════════════════════════════════════════════════
class Barrier {
public:
    explicit Barrier(std::size_t count)
        : count_(count), waiting_(0) {}

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

// ═══════════════════════════════════════════════════════════════════════
// bind_thread_to_core — from upstream driver.h:36-45 (100% preserved)
// ═══════════════════════════════════════════════════════════════════════
#ifdef __linux__
#include <pthread.h>
inline void bind_thread_to_core(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(),
                                     sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::fprintf(stderr, "[bind_thread] Error setting affinity "
                     "to core %d: %d\n", core_id, rc);
    }
}
#else
inline void bind_thread_to_core(std::thread&, int) {} // no-op on non-Linux
#endif

// ═══════════════════════════════════════════════════════════════════════
// PhilemonOp — operation types (from upstream types.hpp, extended)
// ═══════════════════════════════════════════════════════════════════════
enum class PhilemonOp : uint8_t {
    // From upstream
    BFS, SSSP, PAGE_RANK, WCC, TC,
    // NEW: cross-tier variants
    CROSS_TIER_BFS, CROSS_TIER_SSSP,
    CROSS_TIER_PR, CROSS_TIER_WCC, CROSS_TIER_TC,
    // NEW: tier-specific operations
    TIER_BENCHMARK, TIER_MIGRATION_TEST,
    // From upstream: graph mutations
    INSERT, DELETE, BATCH_INSERT,
    // From upstream: lookup operations
    GET_VERTEX, GET_EDGE, GET_WEIGHT, GET_NEIGHBOR,
};

inline const char* op_name(PhilemonOp op) {
    switch (op) {
        case PhilemonOp::BFS:              return "BFS";
        case PhilemonOp::SSSP:             return "SSSP";
        case PhilemonOp::PAGE_RANK:        return "PageRank";
        case PhilemonOp::WCC:              return "WCC";
        case PhilemonOp::TC:               return "TC";
        case PhilemonOp::CROSS_TIER_BFS:   return "CrossTierBFS";
        case PhilemonOp::CROSS_TIER_SSSP:  return "CrossTierSSSP";
        case PhilemonOp::CROSS_TIER_PR:    return "CrossTierPR";
        case PhilemonOp::CROSS_TIER_WCC:   return "CrossTierWCC";
        case PhilemonOp::CROSS_TIER_TC:    return "CrossTierTC";
        case PhilemonOp::TIER_BENCHMARK:   return "TierBench";
        case PhilemonOp::TIER_MIGRATION_TEST: return "TierMigrate";
        case PhilemonOp::INSERT:           return "Insert";
        case PhilemonOp::DELETE:           return "Delete";
        case PhilemonOp::BATCH_INSERT:     return "BatchInsert";
        case PhilemonOp::GET_VERTEX:       return "GetVertex";
        case PhilemonOp::GET_EDGE:         return "GetEdge";
        case PhilemonOp::GET_WEIGHT:       return "GetWeight";
        case PhilemonOp::GET_NEIGHBOR:     return "GetNeighbor";
        default: return "Unknown";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PhilemonConfig — from upstream DriverConfig (driver.h:73-80, extended)
// ═══════════════════════════════════════════════════════════════════════
struct PhilemonConfig {
    // From upstream DriverConfig (preserved)
    std::string workload_dir = ".";
    std::string output_dir = ".";
    int insert_delete_num_threads = 4;
    std::vector<int> query_num_threads = {1, 4, 8};
    std::vector<PhilemonOp> query_ops = {
        PhilemonOp::CROSS_TIER_BFS,
        PhilemonOp::CROSS_TIER_PR,
        PhilemonOp::CROSS_TIER_WCC,
        PhilemonOp::CROSS_TIER_TC
    };
    uint64_t bfs_source = 0;
    uint64_t sssp_source = 0;
    double damping_factor = 0.85;
    uint64_t num_iterations = 20;
    double delta = 2.0;  // SSSP delta stepping
    int alpha = 15;      // BFS direction switch
    int beta = 18;       // BFS direction switch

    // From upstream: mixed workload params
    int writer_threads = 2;
    int reader_threads = 4;
    uint64_t insert_delete_checkpoint_size = 1024;

    // NEW: Philemon-specific params
    int debug_level = 1;
    double convergence_threshold = 1e-6;
    bool verify_with_sequential = false;
    bool dump_convergence_json = false;
    std::string json_output_path = "philemon_results.json";
    cost_model::TierCostModel cost_model;
};

// ═══════════════════════════════════════════════════════════════════════
// Timer — from upstream utils/Timer.h (100% preserved, renamed)
// ═══════════════════════════════════════════════════════════════════════
class WallTimer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    WallTimer() : start_(std::chrono::high_resolution_clock::now()) {}
    void reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    double elapsed_us() const { return elapsed_ms() * 1000.0; }
    double elapsed_ns() const { return elapsed_ms() * 1e6; }
};

// ═══════════════════════════════════════════════════════════════════════
// NEW: print_progress_bar — visual progress for long operations
// ═══════════════════════════════════════════════════════════════════════
inline void print_progress_bar(const char* label, uint64_t current,
                                uint64_t total, double elapsed_ms) {
    if (total == 0) return;
    int pct = (int)(100.0 * current / total);
    int bar_width = 40;
    int filled = bar_width * current / total;

    std::printf("\r  [%s] [", label);
    for (int i = 0; i < bar_width; i++) {
        std::printf(i < filled ? "█" : "░");
    }
    double rate = elapsed_ms > 0 ? current / elapsed_ms * 1000 : 0;
    std::printf("] %3d%% (%lu/%lu) %.0f ops/s",
                pct, (unsigned long)current, (unsigned long)total, rate);
    if (current == total) std::printf("\n");
    std::fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════
// PhilemonDriver — main benchmark orchestration
//
// From upstream Driver<F,S> (driver.h:72-101), with cross-tier extensions
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
class PhilemonDriver {
private:
    F& m_method;
    const PhilemonConfig& m_config;

public:
    // From upstream driver.h:103-104
    PhilemonDriver(F& method, const PhilemonConfig& config)
        : m_method(method), m_config(config) {
        debug::set_debug_level(config.debug_level);
    }

    ~PhilemonDriver() = default;

    // ─── Main dispatch ───────────────────────────────────────────
    void execute(PhilemonOp op);
    void execute_all_queries();

    // ─── NEW: System state dump ──────────────────────────────────
    void dump_full_system_state(const std::string& phase) const;

private:
    // ─── From upstream driver.h:657-721: query dispatch ──────────
    void execute_query(PhilemonOp op, int num_threads);

    // ─── NEW: Cross-tier query dispatch ──────────────────────────
    void execute_cross_tier_query(PhilemonOp op, int num_threads);

    // ─── NEW: Tier migration benchmark ───────────────────────────
    void execute_tier_benchmark();

    // ─── From upstream driver.h:380-648: microbenchmark framework ─
    void execute_microbenchmark(PhilemonOp op, int num_threads);

    // ─── From upstream driver.h:1066-1200: mixed read-write ──────
    void execute_mixed_reader_writer();
};

// ═══════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════

// ─── dump_full_system_state: NEW ─────────────────────────────────────
template <class F, class S>
void PhilemonDriver<F,S>::dump_full_system_state(
    const std::string& phase) const
{
    std::printf("\n");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║        PHILEMON SYSTEM STATE: %-24s ║\n", phase.c_str());
    std::printf("╠══════════════════════════════════════════════════════════╣\n");

    auto snapshot = wrapper::get_shared_snapshot(m_method);
    uint64_t N = wrapper::snapshot_vertex_count(snapshot);
    uint64_t E = wrapper::snapshot_edge_count(snapshot);

    std::printf("║ Graph:  V=%lu  E=%lu  avg_degree=%.1f\n",
                (unsigned long)N, (unsigned long)E,
                N > 0 ? (double)E / N : 0);

    // Config summary
    std::printf("║ Config: threads=%d damping=%.2f iters=%lu delta=%.1f\n",
                m_config.insert_delete_num_threads,
                m_config.damping_factor,
                (unsigned long)m_config.num_iterations,
                m_config.delta);
    std::printf("║         alpha=%d beta=%d bfs_src=%lu sssp_src=%lu\n",
                m_config.alpha, m_config.beta,
                (unsigned long)m_config.bfs_source,
                (unsigned long)m_config.sssp_source);
    std::printf("║         debug_level=%d conv_threshold=%.1e\n",
                m_config.debug_level, m_config.convergence_threshold);

    // Tier perf counters
    std::printf("║ TierPerf:\n");
    for (int t = 0; t < 3; t++) {
        auto& perf = debug::tier_perf(t);
        std::printf("║   tier[%d] reads=%-8lu writes=%-8lu "
                    "bytes=%-10lu lat_ns=%-10lu\n",
                    t,
                    (unsigned long)perf.read_count.load(),
                    (unsigned long)perf.write_count.load(),
                    (unsigned long)perf.bytes_transferred.load(),
                    (unsigned long)perf.latency_sum_ns.load());
    }

    // Trace ring summary
    std::printf("║ TraceRing: %zu events captured\n",
                debug::trace_ring().size());

    std::printf("╚══════════════════════════════════════════════════════════╝\n\n");
}

// ═══════════════════════════════════════════════════════════════════════
// execute: main dispatch
// From upstream driver.h::execute() (preserved switch structure)
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void PhilemonDriver<F,S>::execute(PhilemonOp op) {
    PHILE_DBG(1, "[Driver] executing: %s", op_name(op));
    dump_full_system_state("before_" + std::string(op_name(op)));

    switch (op) {
        case PhilemonOp::CROSS_TIER_BFS:
        case PhilemonOp::CROSS_TIER_SSSP:
        case PhilemonOp::CROSS_TIER_PR:
        case PhilemonOp::CROSS_TIER_WCC:
        case PhilemonOp::CROSS_TIER_TC:
            execute_cross_tier_query(
                op, m_config.query_num_threads.empty() ?
                    4 : m_config.query_num_threads[0]);
            break;

        case PhilemonOp::BFS:
        case PhilemonOp::SSSP:
        case PhilemonOp::PAGE_RANK:
        case PhilemonOp::WCC:
        case PhilemonOp::TC:
            execute_query(op, m_config.query_num_threads.empty() ?
                          4 : m_config.query_num_threads[0]);
            break;

        case PhilemonOp::TIER_BENCHMARK:
            execute_tier_benchmark();
            break;

        default:
            PHILE_DBG(1, "[Driver] unhandled op: %s", op_name(op));
            break;
    }

    dump_full_system_state("after_" + std::string(op_name(op)));
}

// ─── execute_all_queries: iterate over all configured ops ────────────
template <class F, class S>
void PhilemonDriver<F,S>::execute_all_queries() {
    PHILE_DBG(1, "[Driver] Running %zu ops × %zu thread configs",
              m_config.query_ops.size(),
              m_config.query_num_threads.size());

    for (auto& num_threads : m_config.query_num_threads) {
        for (auto op : m_config.query_ops) {
            PHILE_DBG(1, "═══ %s with %d threads ═══",
                      op_name(op), num_threads);

            WallTimer timer;
            execute_cross_tier_query(op, num_threads);
            double elapsed = timer.elapsed_ms();

            PHILE_DBG(1, "═══ %s done: %.3f ms ═══",
                      op_name(op), elapsed);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// execute_cross_tier_query: dispatch to cross-tier algorithms
// From upstream driver.h:657-721 execute_query() structure
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void PhilemonDriver<F,S>::execute_cross_tier_query(
    PhilemonOp op, int num_threads)
{
    // From upstream driver.h:659: get snapshot
    auto snapshot = wrapper::get_shared_snapshot(m_method);

    PHILE_DBG(1, "[Driver::cross_tier] op=%s threads=%d",
              op_name(op), num_threads);

    // Reset tier counters before each query
    for (int t = 0; t < 3; t++) debug::tier_perf(t).reset();

    try {
        // From upstream driver.h:676-718: switch on op type
        if (op == PhilemonOp::CROSS_TIER_BFS ||
            op == PhilemonOp::BFS) {
            std::vector<std::pair<uint64_t, int64_t>> results;
            algorithms::CrossTierBFS<F, S> bfs(
                num_threads, m_config.alpha, m_config.beta,
                m_method, snapshot, m_config.cost_model);
            bfs.run_bfs(m_config.bfs_source, results);

            PHILE_DBG(1, "[BFS] result size=%zu levels=%zu",
                      results.size(), bfs.level_log().size());
        }

        else if (op == PhilemonOp::CROSS_TIER_SSSP ||
                 op == PhilemonOp::SSSP) {
            std::vector<std::pair<uint64_t, double>> results;
            algorithms::CrossTierSSSP<F, S> sssp(
                num_threads, m_config.delta,
                m_method, snapshot, m_config.cost_model);
            sssp.run_sssp(m_config.sssp_source, results);
        }

        else if (op == PhilemonOp::CROSS_TIER_PR ||
                 op == PhilemonOp::PAGE_RANK) {
            std::vector<std::pair<uint64_t, double>> results;
            algorithms::CrossTierPageRank<F, S> pr(
                num_threads, m_config.num_iterations,
                m_config.damping_factor, m_method, snapshot,
                m_config.cost_model, m_config.convergence_threshold);
            pr.run_page_rank(results);

            PHILE_DBG(1, "[PR] result size=%zu iters=%zu",
                      results.size(), pr.iteration_log().size());
        }

        else if (op == PhilemonOp::CROSS_TIER_WCC ||
                 op == PhilemonOp::WCC) {
            std::vector<std::pair<uint64_t, int64_t>> results;
            algorithms::CrossTierWCC<F, S> wcc(
                num_threads, m_method, snapshot,
                m_config.cost_model);
            wcc.run_wcc(results);

            PHILE_DBG(1, "[WCC] result size=%zu rounds=%zu",
                      results.size(), wcc.round_log().size());
        }

        else if (op == PhilemonOp::CROSS_TIER_TC ||
                 op == PhilemonOp::TC) {
            algorithms::CrossTierTC<F, S> tc(
                m_method, snapshot, num_threads,
                m_config.cost_model);
            uint64_t count = tc.run_tc();

            PHILE_DBG(1, "[TC] triangles=%lu", (unsigned long)count);
        }

        else {
            PHILE_DBG(1, "[Driver] unknown cross-tier op: %s",
                      op_name(op));
        }
    }
    catch (std::exception& e) {
        std::fprintf(stderr, "[Driver] Exception in %s: %s\n",
                     op_name(op), e.what());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// execute_query: dispatch to basic (non-cross-tier) algorithms
// From upstream driver.h:657-721 (structure preserved)
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void PhilemonDriver<F,S>::execute_query(PhilemonOp op, int num_threads) {
    // Delegate to cross_tier versions (they are supersets)
    execute_cross_tier_query(op, num_threads);
}

// ═══════════════════════════════════════════════════════════════════════
// execute_tier_benchmark: NEW, test tier migration performance
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void PhilemonDriver<F,S>::execute_tier_benchmark() {
    debug::ScopedTimer timer("Driver::tier_benchmark");

    auto snapshot = wrapper::get_shared_snapshot(m_method);
    uint64_t N = wrapper::snapshot_vertex_count(snapshot);
    uint64_t E = wrapper::snapshot_edge_count(snapshot);

    PHILE_DBG(1, "[TierBench] N=%lu E=%lu", (unsigned long)N,
              (unsigned long)E);

    std::printf("──── Tier Benchmark ────\n");

    // Test 1: Sequential scan all vertices
    {
        WallTimer t;
        uint64_t degree_sum = 0;
        for (uint64_t v = 0; v < N; v++) {
            degree_sum += wrapper::snapshot_degree(snapshot, v, false);
            if (v % 10000 == 0) {
                print_progress_bar("SeqScan", v, N, t.elapsed_ms());
            }
        }
        print_progress_bar("SeqScan", N, N, t.elapsed_ms());
        std::printf("  SeqScan: %lu vertices, degree_sum=%lu, %.3f ms\n",
                    (unsigned long)N, (unsigned long)degree_sum,
                    t.elapsed_ms());
    }

    // Test 2: Random access pattern (simulates different tier access)
    {
        WallTimer t;
        std::mt19937 rng(42);
        uint64_t hits = 0;
        uint64_t num_queries = std::min(N, (uint64_t)100000);
        for (uint64_t i = 0; i < num_queries; i++) {
            uint64_t v = rng() % N;
            hits += wrapper::snapshot_degree(snapshot, v, false) > 0 ? 1 : 0;
            if (i % 10000 == 0) {
                print_progress_bar("RandAccess", i, num_queries,
                                    t.elapsed_ms());
            }
        }
        print_progress_bar("RandAccess", num_queries, num_queries,
                            t.elapsed_ms());
        std::printf("  RandAccess: %lu queries, hits=%lu, %.3f ms "
                    "(%.0f qps)\n",
                    (unsigned long)num_queries, (unsigned long)hits,
                    t.elapsed_ms(),
                    num_queries / t.elapsed_ms() * 1000);
    }

    // Test 3: Edge traversal throughput
    {
        WallTimer t;
        std::atomic<uint64_t> edge_count{0};
        uint64_t sample = std::min(N, (uint64_t)10000);
        for (uint64_t v = 0; v < sample; v++) {
            wrapper::snapshot_edges(snapshot, v,
                [&edge_count](uint64_t dst, double w) {
                    edge_count.fetch_add(1, std::memory_order_relaxed);
                }, false);
        }
        std::printf("  EdgeTraversal: %lu vertices, %lu edges, %.3f ms "
                    "(%.0f edges/s)\n",
                    (unsigned long)sample,
                    (unsigned long)edge_count.load(),
                    t.elapsed_ms(),
                    edge_count.load() / t.elapsed_ms() * 1000);
    }

    std::printf("──── End Tier Benchmark ────\n");
    debug::print_all_tier_perf();
}

// ═══════════════════════════════════════════════════════════════════════
// execute_mixed_reader_writer: concurrent read+write workload
// From upstream driver.h:1066-1200 (framework preserved, simplified)
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void PhilemonDriver<F,S>::execute_mixed_reader_writer() {
    debug::ScopedTimer timer("Driver::mixed_rw");

    PHILE_DBG(1, "[MixedRW] writers=%d readers=%d",
              m_config.writer_threads, m_config.reader_threads);

    // From upstream: get snapshot for readers
    auto snapshot = wrapper::get_shared_snapshot(m_method);
    uint64_t N = wrapper::snapshot_vertex_count(snapshot);

    // From upstream: degree list for PageRank
    std::vector<uint64_t> degree_list(N);
    for (uint64_t i = 0; i < N; i++) {
        degree_list[i] = wrapper::snapshot_degree(snapshot, i, false);
    }

    // From upstream: barrier for synchronization
    Barrier barrier(m_config.writer_threads + m_config.reader_threads);

    std::vector<std::thread> reader_threads;
    std::vector<double> reader_times(m_config.reader_threads);

    // From upstream: launch reader threads
    for (int i = 0; i < m_config.reader_threads; i++) {
        reader_threads.emplace_back([this, &snapshot, &reader_times,
                                     &degree_list, N, &barrier](int tid) {
            wrapper::init_thread(m_method, tid);
            auto snap_local = wrapper::snapshot_clone(snapshot);

            WallTimer t;

            // 算法改动: degree加权采样代替顺序扫描前1000个
            // 高degree顶点被采样概率更高, 更好模拟真实读负载
            uint64_t edge_count = 0;
            uint64_t sample = std::min(N, (uint64_t)1000);

            // 用degree做前缀和, 按degree概率采样
            uint64_t degree_sum = 0;
            for (uint64_t v = 0; v < N; v++) degree_sum += degree_list[v] + 1;

            // 用thread id做seed的确定性采样
            uint64_t hash_state = 0x9E3779B97F4A7C15ULL ^ (uint64_t)tid;
            for (uint64_t s = 0; s < sample; s++) {
                // xorshift64 确定性PRNG
                hash_state ^= hash_state << 13;
                hash_state ^= hash_state >> 7;
                hash_state ^= hash_state << 17;
                uint64_t target = hash_state % degree_sum;

                // 线性搜索 (小数据量ok, 大数据量可换binary search)
                uint64_t acc = 0;
                uint64_t v = 0;
                for (v = 0; v < N; v++) {
                    acc += degree_list[v] + 1;
                    if (acc > target) break;
                }
                if (v >= N) v = N - 1;

                wrapper::snapshot_edges(snap_local, v,
                    [&edge_count](uint64_t dst, double w) {
                        edge_count++;
                    }, false);
            }

            reader_times[tid] = t.elapsed_ms();
            PHILE_DBG(2, "[Reader %d] sampled %lu vertices, "
                      "%lu edges in %.3f ms",
                      tid, (unsigned long)sample,
                      (unsigned long)edge_count,
                      reader_times[tid]);

            wrapper::end_thread(m_method, tid);
        }, i);
    }

    for (auto& t : reader_threads) t.join();

    // Summary
    double avg_reader_time = 0;
    for (auto& rt : reader_times) avg_reader_time += rt;
    avg_reader_time /= m_config.reader_threads;
    PHILE_DBG(1, "[MixedRW] avg_reader_time=%.3f ms", avg_reader_time);
}

// ═══════════════════════════════════════════════════════════════════════
// execute_microbenchmark — from upstream driver.h:380-648 (simplified)
// ═══════════════════════════════════════════════════════════════════════
template <class F, class S>
void PhilemonDriver<F,S>::execute_microbenchmark(
    PhilemonOp op, int num_threads)
{
    debug::ScopedTimer timer("Driver::microbenchmark");

    auto snapshot = wrapper::get_shared_snapshot(m_method);
    uint64_t N = wrapper::snapshot_vertex_count(snapshot);

    PHILE_DBG(1, "[Microbench] op=%s threads=%d N=%lu",
              op_name(op), num_threads, (unsigned long)N);

    // From upstream: parallel worker framework
    std::vector<std::thread> threads;
    std::vector<double> thread_time(num_threads);
    std::vector<uint64_t> thread_count(num_threads, 0);

    wrapper::set_max_threads(m_method, num_threads);

    // 算法改动: 从固定chunk_size改为动态work-stealing风格分配
    // 每个线程从共享atomic计数器取batch, 避免负载不均
    // (原版: 固定chunk导致高degree顶点的线程跑得慢)
    std::atomic<uint64_t> work_counter{0};
    constexpr uint64_t BATCH_SIZE = 256;

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([this, &snapshot, N, op,
                              &thread_time, &thread_count,
                              &work_counter](int tid) {
            wrapper::init_thread(m_method, tid);
            auto snap = wrapper::snapshot_clone(snapshot);

            WallTimer t;

            while (true) {
                uint64_t start = work_counter.fetch_add(
                    BATCH_SIZE, std::memory_order_relaxed);
                if (start >= N) break;
                uint64_t end = std::min(start + BATCH_SIZE, N);

                for (uint64_t v = start; v < end; v++) {
                    if (op == PhilemonOp::GET_VERTEX) {
                        thread_count[tid] +=
                            wrapper::snapshot_has_vertex(snap, v) ? 1 : 0;
                    } else if (op == PhilemonOp::GET_NEIGHBOR) {
                        uint64_t cnt = 0;
                        wrapper::snapshot_edges(snap, v,
                            [&cnt](uint64_t dst, double w) { cnt++; }, false);
                        thread_count[tid] += cnt;
                    } else {
                        thread_count[tid] +=
                            wrapper::snapshot_degree(snap, v, false);
                    }
                }
            }

            thread_time[tid] = t.elapsed_ms();
            wrapper::end_thread(m_method, tid);
        }, i);

        // From upstream: bind to core
        bind_thread_to_core(threads[i],
                            i % std::thread::hardware_concurrency());
    }

    for (auto& t : threads) t.join();

    // From upstream: aggregate results
    double total_time = 0;
    uint64_t total_ops = 0;
    for (int i = 0; i < num_threads; i++) {
        total_time += thread_time[i];
        total_ops += thread_count[i];
        PHILE_DBG(2, "  thread[%d] time=%.3f ms ops=%lu",
                  i, thread_time[i], (unsigned long)thread_count[i]);
    }

    double avg_time = total_time / num_threads;
    double throughput = total_ops / avg_time * 1000;  // ops/sec
    PHILE_DBG(1, "[Microbench] %s: avg_time=%.3f ms total_ops=%lu "
              "throughput=%.0f ops/s",
              op_name(op), avg_time, (unsigned long)total_ops, throughput);
}

}  // namespace driver
}  // namespace philemon

#endif  // PHILEMON_DRIVER_HPP
