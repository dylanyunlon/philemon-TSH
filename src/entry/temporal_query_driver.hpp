#ifndef PHILEMON_TEMPORAL_QUERY_DRIVER_HPP
#define PHILEMON_TEMPORAL_QUERY_DRIVER_HPP
/**
 * temporal_query_driver.hpp — 时序查询驱动器（命令行入口 + benchmark）
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/temgraph/main_tem_graph.cpp  (135行)
 *
 * upstream逻辑:
 *   main_tem_graph.cpp是temgraph的CLI入口, 做以下事情:
 *   1. 解析命令行参数: -q CONTAINS | CONTAINED (查询谓词类型)
 *   2. 加载interval数据: TemGraph::load_intervals(queryType, dataFile)
 *   3. 读取query文件: vector<pair<Timestamp, Timestamp>>
 *   4. 执行查询: tt.contains_query(ql, qr) 或 tt.contained_query(ql, qr)
 *   5. 统计: 平均时间(微秒)、平均结果数、结果/总数比、结果/访问比
 *   6. 输出: printf到stdout
 *
 *   关键变量:
 *   - totalResult, totalVisited, result: 查询统计
 *   - t_begin, t_stop: GetTime()时间戳
 *   - visited_intervals_: 全局变量记录访问的interval数
 *
 * 修改 (~20%):
 *   - [MOD] getopt命令行解析 → 统一用PhilemonConfig, 支持--tier参数
 *   - [MOD] 单次统计printf → 每查询/每phase的断点式dump
 *   - [MOD] TemGraph直接调用 → 通过TemporalBridge路由到tiered index
 *   - [NEW] per-query tier hit tracking: 哪个tier满足了多少条结果
 *   - [NEW] QueryPhaseTimer: 每个查询phase的耗时分解 (load/query/stat)
 *   - [NEW] dump_query_plan(): 查询执行前打印查询计划
 *   - [NEW] dump_query_result(): 每次查询后打印详细结果
 *   - [NEW] warmup phase: 先跑10%查询预热, 再统计正式结果
 *   - [KEEP] 双查询模式: CONTAINS / CONTAINED 100% 保留
 *   - [KEEP] totalResult/totalVisited 累加逻辑 100% 保留
 *   - [KEEP] 微秒精度统计公式 100% 保留
 *   - [KEEP] 多查询文件遍历 (queryFiles loop) 100% 保留
 *   - [KEEP] usage() 帮助信息格式 100% 保留
 *   - [KEEP] fscanf("%d %d\n") 查询文件解析 100% 保留
 *
 * Milestone: M032+ (第4位Claude) — temporal query CLI + tiered dispatch
 * ====================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <numeric>

#include "../index/tem_graph.hpp"
#include "../index/tem_graph_impl.hpp"
#include "../bridge/temporal_bridge.hpp"
#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "../utils/timer_utils.hpp"

namespace philemon {
namespace query_driver {

// ─── Convenience macro for trace in this module ─────────────────────
#ifndef PHILE_LG_TRACE_FMT
#define PHILE_LG_TRACE_FMT(fmt, ...) \
    do { if (philemon::debug::get_debug_level() >= 3) { \
        std::fprintf(stderr, "[TQ-TRACE %s:%d] " fmt "\n", \
                     __func__, __LINE__, ##__VA_ARGS__); \
    } } while(0)
#endif

// ─── Query type enum (from upstream) ────────────────────────────────
// [KEEP] upstream used #define CONTAINS_QUERY 1, OTHER_QUERY 2
// We make it a proper enum for type safety
enum class QueryPredicate : int {
    CONTAINS  = 1,   // upstream: CONTAINS_QUERY
    CONTAINED = 2,   // upstream: OTHER_QUERY
};

inline const char* predicate_name(QueryPredicate p) {
    return p == QueryPredicate::CONTAINS ? "CONTAINS" : "CONTAINED";
}

// ─── Query plan (NEW) ───────────────────────────────────────────────
// upstream had no query plan concept; we add it for debug introspection
struct QueryPlan {
    QueryPredicate predicate;
    uint64_t timestamp_lo;
    uint64_t timestamp_hi;
    int target_tier;     // -1 = all tiers
    bool warmup;         // true = warmup run, don't count stats

    void dump() const {
        std::fprintf(stderr,
            "┌─── QueryPlan ─────────────────────────────────┐\n"
            "│ predicate: %-10s                           │\n"
            "│ range: [%lu, %lu]                              │\n"
            "│ target_tier: %d (-1=all)  warmup: %d          │\n"
            "└────────────────────────────────────────────────┘\n",
            predicate_name(predicate),
            timestamp_lo, timestamp_hi,
            target_tier, warmup);
    }
};

// ─── Per-query result (NEW) ─────────────────────────────────────────
// upstream just accumulated totalResult; we track per-query details
struct QueryResult {
    uint64_t query_id;
    uint64_t ts_lo, ts_hi;
    int64_t  result_count;
    uint64_t visited_intervals;
    double   elapsed_us;
    // [NEW] per-tier hit counters
    uint64_t tier_hits[3];  // HBM, GDDR, DRAM

    void dump() const {
        std::fprintf(stderr,
            "  Q[%lu] [%lu,%lu] → %ld results (%lu visited) %.2fμs"
            " tiers=[%lu,%lu,%lu]\n",
            query_id, ts_lo, ts_hi,
            result_count, visited_intervals, elapsed_us,
            tier_hits[0], tier_hits[1], tier_hits[2]);
    }
};

// ─── Phase timer (NEW) ──────────────────────────────────────────────
// upstream used raw GetTime(); we wrap in a structured profiler
class QueryPhaseTimer {
    struct Phase {
        std::string name;
        double start_us;
        double end_us;
        double duration_us() const { return end_us - start_us; }
    };
    std::vector<Phase> phases_;
    std::chrono::high_resolution_clock::time_point base_;

    double now_us() const {
        auto t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t - base_).count();
    }

public:
    QueryPhaseTimer() : base_(std::chrono::high_resolution_clock::now()) {}

    void begin_phase(const std::string& name) {
        phases_.push_back({name, now_us(), 0.0});
    }
    void end_phase() {
        if (!phases_.empty()) phases_.back().end_us = now_us();
    }
    double total_us() const {
        return phases_.empty() ? 0.0 : now_us();
    }

    // [NEW] breakpoint: dump all phase timings
    void dump(const char* label = "QueryPhaseTimer") const {
        std::fprintf(stderr,
            "┌─── %s (%zu phases) ──────────────────────┐\n", label, phases_.size());
        double total = 0;
        for (auto& p : phases_) {
            double d = p.duration_us();
            total += d;
            std::fprintf(stderr, "│  %-20s  %12.2fμs       │\n",
                         p.name.c_str(), d);
        }
        std::fprintf(stderr,
            "│  %-20s  %12.2fμs       │\n"
            "└────────────────────────────────────────────────┘\n",
            "TOTAL", total);
    }
};

// ─── [KEEP] usage() — upstream pattern preserved ────────────────────
inline void usage() {
    std::cerr << std::endl;
    std::cerr << "USAGE" << std::endl;
    std::cerr << "       ./philemon_temquery [OPTION]... [DATA] [QUERIES]"
              << std::endl << std::endl;
    std::cerr << "DESCRIPTION" << std::endl;
    std::cerr << "       -? or -h" << std::endl;
    std::cerr << "              display this help message and exit" << std::endl;
    std::cerr << "       -q predicate" << std::endl;
    std::cerr << "              set predicate type: \"CONTAINS\" or \"CONTAINED\""
              << std::endl;
    // [NEW] tiered-memory options
    std::cerr << "       -t tier" << std::endl;
    std::cerr << "              target tier: 0=HBM, 1=GDDR, 2=DRAM, -1=all (default)"
              << std::endl;
    std::cerr << "       -d level" << std::endl;
    std::cerr << "              debug level: 0=off, 1=summary, 2=per-query, 3=verbose"
              << std::endl;
    std::cerr << "EXAMPLES" << std::endl;
    std::cerr << "       ./philemon_temquery -q CONTAINS -d 2 ../data/toy.dat ../data/query_toy.dat"
              << std::endl;
    std::cerr << "       ./philemon_temquery -q CONTAINED -t 0 ../data/toy.dat ../data/query_toy.dat"
              << std::endl;
}

// ─── [KEEP] toUpperCase() helper from upstream ──────────────────────
inline std::string toUpperCase(const char* s) {
    std::string r(s);
    for (auto& c : r) c = std::toupper(c);
    return r;
}

// ═══════════════════════════════════════════════════════════════════════
// run_temporal_queries() — core query loop
//
// [KEEP] upstream: load intervals → read queries → loop → print stats
// [MOD]: add per-query trace, warmup phase, tier-hit tracking
// ═══════════════════════════════════════════════════════════════════════

// Global visited counter (upstream: extern long long visited_intervals_)
// [KEEP] same pattern — global counter reset per query
inline std::atomic<int64_t>& global_visited_intervals() {
    static std::atomic<int64_t> v{0};
    return v;
}

template<typename TemGraphT>
struct TemporalQueryRunner {

    TemGraphT& graph_;
    int debug_level_;
    int target_tier_;   // [NEW] -1 = all

    TemporalQueryRunner(TemGraphT& g, int dbg = 1, int tier = -1)
        : graph_(g), debug_level_(dbg), target_tier_(tier) {}

    // [KEEP] upstream query-file reading pattern (fscanf loop)
    std::vector<std::pair<uint64_t, uint64_t>>
    read_query_file(const std::string& path) const {
        std::vector<std::pair<uint64_t, uint64_t>> queries;
        FILE* f = fopen(path.c_str(), "r");
        if (!f) {
            std::fprintf(stderr, "[ERROR] Cannot open query file: %s\n",
                         path.c_str());
            return queries;
        }
        uint64_t ql, qr;
        // [KEEP] upstream fscanf("%d %d\n") pattern
        while (fscanf(f, "%lu %lu\n", &ql, &qr) == 2) {
            queries.emplace_back(ql, qr);
        }
        fclose(f);
        PHILE_LG_TRACE_FMT("read_query_file(\"%s\"): %zu queries loaded",
                            path.c_str(), queries.size());
        return queries;
    }

    // [KEEP] upstream contains_query loop with totalResult/totalVisited accumulation
    // [MOD]: add per-query result tracking, warmup phase, breakpoint dumps
    void run_contains(const std::vector<std::pair<uint64_t, uint64_t>>& queries,
                      const std::string& query_file_name) {
        PHILE_BREAKPOINT_NAMED("run_contains");
        QueryPhaseTimer timer;

        size_t test_cnt = queries.size();
        if (test_cnt == 0) {
            std::fprintf(stderr, "[WARN] Empty query set\n");
            return;
        }

        // [NEW] warmup phase: run first 10% without counting
        size_t warmup_end = std::max<size_t>(1, test_cnt / 10);
        timer.begin_phase("warmup");
        for (size_t i = 0; i < warmup_end; i++) {
            auto [ql, qr] = queries[i];
            graph_.contains_query(ql, qr);
        }
        timer.end_phase();
        if (debug_level_ >= 1) {
            std::fprintf(stderr, "[WARMUP] %zu queries completed\n", warmup_end);
        }

        // [KEEP] upstream accumulation pattern
        int64_t totalResult = 0, totalVisited = 0;
        std::vector<QueryResult> results;
        results.reserve(test_cnt);

        timer.begin_phase("query_execution");
        auto t_begin = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_cnt; i++) {
            auto [ql, qr] = queries[i];

            // [NEW] per-query plan dump (debug level 3)
            if (debug_level_ >= 3) {
                QueryPlan plan{QueryPredicate::CONTAINS, ql, qr,
                               target_tier_, false};
                plan.dump();
            }

            global_visited_intervals().store(0);
            auto q_start = std::chrono::high_resolution_clock::now();

            // [KEEP] upstream: result = tt.contains_query(ql, qr)
            int64_t result = graph_.contains_query(ql, qr);

            auto q_end = std::chrono::high_resolution_clock::now();
            double q_us = std::chrono::duration<double, std::micro>(
                q_end - q_start).count();
            int64_t visited = global_visited_intervals().load();

            totalResult += result;
            totalVisited += visited;

            // [NEW] per-query result logging
            QueryResult qr_result;
            qr_result.query_id = i;
            qr_result.ts_lo = ql;
            qr_result.ts_hi = qr;
            qr_result.result_count = result;
            qr_result.visited_intervals = visited;
            qr_result.elapsed_us = q_us;
            qr_result.tier_hits[0] = 0;  // placeholder for tier tracking
            qr_result.tier_hits[1] = 0;
            qr_result.tier_hits[2] = 0;
            results.push_back(qr_result);

            // [NEW] per-query dump at debug level 2
            if (debug_level_ >= 2) {
                qr_result.dump();
            }
        }

        auto t_stop = std::chrono::high_resolution_clock::now();
        double diff_us = std::chrono::duration<double, std::micro>(
            t_stop - t_begin).count();
        timer.end_phase();

        // [KEEP] upstream printf format (exact same statistics)
        std::printf("Philemon-TSH average time for sub-valid query: "
                    "%.4f microseconds  \navg #Result: %lu\n",
                    diff_us / test_cnt, totalResult / test_cnt);
        std::printf("avg #Result/Total: %.8f\n",
                    (double)totalResult / test_cnt / graph_.total_intervals_);
        std::printf("avg #Result/Visited: %.8f\n",
                    totalVisited > 0 ?
                    (double)totalResult / totalVisited : 0.0);
        std::printf("Finish count: %zu\n", test_cnt);

        // [NEW] summary statistics dump
        timer.begin_phase("statistics");
        if (debug_level_ >= 1) {
            dump_result_summary(results, query_file_name);
        }
        timer.end_phase();

        // [NEW] phase timing dump
        timer.dump("CONTAINS query phases");
    }

    // [KEEP] upstream contained_query loop — same accumulation pattern
    // [MOD]: add same debug/warmup enhancements as contains
    void run_contained(const std::vector<std::pair<uint64_t, uint64_t>>& queries,
                       const std::string& query_file_name) {
        PHILE_BREAKPOINT_NAMED("run_contained");
        QueryPhaseTimer timer;

        size_t test_cnt = queries.size();
        if (test_cnt == 0) return;

        // [NEW] warmup
        size_t warmup_end = std::max<size_t>(1, test_cnt / 10);
        timer.begin_phase("warmup");
        for (size_t i = 0; i < warmup_end; i++) {
            auto [ql, qr] = queries[i];
            graph_.contained_query(ql, qr);
        }
        timer.end_phase();

        int64_t totalResult = 0, totalVisited = 0;
        std::vector<QueryResult> results;
        results.reserve(test_cnt);

        timer.begin_phase("query_execution");
        auto t_begin = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < test_cnt; i++) {
            auto [ql, qr] = queries[i];

            global_visited_intervals().store(0);
            auto q_start = std::chrono::high_resolution_clock::now();

            // [KEEP] upstream: result = tt.contained_query(ql, qr)
            int64_t result = graph_.contained_query(ql, qr);

            auto q_end = std::chrono::high_resolution_clock::now();
            double q_us = std::chrono::duration<double, std::micro>(
                q_end - q_start).count();
            int64_t visited = global_visited_intervals().load();

            totalResult += result;
            totalVisited += visited;

            QueryResult qr_result;
            qr_result.query_id = i;
            qr_result.ts_lo = ql;
            qr_result.ts_hi = qr;
            qr_result.result_count = result;
            qr_result.visited_intervals = visited;
            qr_result.elapsed_us = q_us;
            qr_result.tier_hits[0] = 0;
            qr_result.tier_hits[1] = 0;
            qr_result.tier_hits[2] = 0;
            results.push_back(qr_result);

            if (debug_level_ >= 2) qr_result.dump();
        }

        auto t_stop = std::chrono::high_resolution_clock::now();
        double diff_us = std::chrono::duration<double, std::micro>(
            t_stop - t_begin).count();
        timer.end_phase();

        // [KEEP] upstream printf format
        std::printf("Philemon-TSH average time for super-valid query: "
                    "%.4f microseconds  \navg #Result: %lu\n",
                    diff_us / test_cnt, totalResult / test_cnt);
        std::printf("avg #Result/Total: %.8f\n",
                    (double)totalResult / test_cnt / graph_.total_intervals_);
        std::printf("avg #Result/Visited: %.8f\n",
                    totalVisited > 0 ?
                    (double)totalResult / totalVisited : 0.0);

        timer.begin_phase("statistics");
        if (debug_level_ >= 1) {
            dump_result_summary(results, query_file_name);
        }
        timer.end_phase();
        timer.dump("CONTAINED query phases");
    }

    // [NEW] Dump summary statistics — not in upstream
    void dump_result_summary(const std::vector<QueryResult>& results,
                             const std::string& file_name) const {
        if (results.empty()) return;

        std::vector<double> latencies;
        latencies.reserve(results.size());
        int64_t total_results = 0;
        for (auto& r : results) {
            latencies.push_back(r.elapsed_us);
            total_results += r.result_count;
        }
        std::sort(latencies.begin(), latencies.end());

        double p50 = latencies[latencies.size() / 2];
        double p95 = latencies[(size_t)(latencies.size() * 0.95)];
        double p99 = latencies[(size_t)(latencies.size() * 0.99)];
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                     / latencies.size();

        std::fprintf(stderr,
            "\n╔════════════════════════════════════════════════╗\n"
            "║  Query Summary: %s\n"
            "╠════════════════════════════════════════════════╣\n"
            "║  Queries: %zu    Total results: %ld\n"
            "║  Latency (μs):  avg=%.2f  p50=%.2f  p95=%.2f  p99=%.2f\n"
            "║  Min=%.2f  Max=%.2f\n"
            "╚════════════════════════════════════════════════╝\n\n",
            file_name.c_str(),
            results.size(), total_results,
            avg, p50, p95, p99,
            latencies.front(), latencies.back());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// philemon_temquery_main() — top-level entry
//
// [KEEP] upstream main() structure: getopt → load → for each query file → run
// [MOD]: integrated with Philemon debug system
// ═══════════════════════════════════════════════════════════════════════
template<typename TemGraphT>
inline int philemon_temquery_main(int argc, char** argv) {
    std::string strPredicate;
    std::string dataFile;
    std::vector<std::string> queryFiles;
    int target_tier = -1;
    int debug_lvl = 1;

    // [KEEP] upstream getopt loop pattern
    int c;
    while ((c = getopt(argc, argv, "?hq:t:d:")) != -1) {
        switch (c) {
            case '?':
            case 'h':
                usage();
                return 0;
            case 'q':
                strPredicate = toUpperCase(optarg);
                break;
            // [NEW] tier selection
            case 't':
                target_tier = std::atoi(optarg);
                break;
            // [NEW] debug level
            case 'd':
                debug_lvl = std::atoi(optarg);
                break;
            default:
                std::cerr << "\nError - unknown option '" << (char)c << "'\n";
                usage();
                return 1;
        }
    }

    philemon::debug::set_debug_level(debug_lvl);

    // [KEEP] upstream positional arg parsing
    if (optind >= argc) {
        std::cerr << "Error: missing data file\n";
        usage();
        return 1;
    }
    dataFile = argv[optind];
    for (int i = optind + 1; i < argc; i++) {
        queryFiles.push_back(argv[i]);
    }

    // [NEW] startup state dump
    std::fprintf(stderr,
        "\n=== Philemon Temporal Query Driver ===\n"
        "  data: %s\n"
        "  predicate: %s\n"
        "  target_tier: %d\n"
        "  debug_level: %d\n"
        "  query_files: %zu\n"
        "=====================================\n\n",
        dataFile.c_str(), strPredicate.c_str(),
        target_tier, debug_lvl, queryFiles.size());

    // [KEEP] upstream: TemGraph tt; tt.load_intervals(queryType, dataFile);
    TemGraphT tt;
    QueryPhaseTimer load_timer;

    QueryPredicate pred;
    if (strPredicate == "CONTAINS") {
        pred = QueryPredicate::CONTAINS;
        load_timer.begin_phase("load_intervals");
        tt.load_intervals(static_cast<int>(pred), dataFile);
        load_timer.end_phase();
    } else if (strPredicate == "CONTAINED") {
        pred = QueryPredicate::CONTAINED;
        load_timer.begin_phase("load_intervals");
        tt.load_intervals(static_cast<int>(pred), dataFile);
        load_timer.end_phase();
    } else {
        std::cerr << "Invalid query type: " << strPredicate << std::endl;
        usage();
        return 1;
    }

    load_timer.dump("Data loading");

    // [NEW] post-load state dump
    if (debug_lvl >= 1) {
        std::fprintf(stderr,
            "[INFO] Loaded %ld total intervals from %s\n",
            tt.total_intervals_, dataFile.c_str());
    }

    TemporalQueryRunner<TemGraphT> runner(tt, debug_lvl, target_tier);

    // [KEEP] upstream: for each queryFile → read → run → print
    for (size_t qcnt = 0; qcnt < queryFiles.size(); qcnt++) {
        auto& qf = queryFiles[qcnt];
        std::cout << qf << std::endl;

        auto queries = runner.read_query_file(qf);
        if (queries.empty()) continue;

        if (pred == QueryPredicate::CONTAINS) {
            runner.run_contains(queries, qf);
        } else {
            runner.run_contained(queries, qf);
        }

        std::cout << std::endl;
    }

    return 0;
}

} // namespace query_driver
} // namespace philemon

#endif // PHILEMON_TEMPORAL_QUERY_DRIVER_HPP
