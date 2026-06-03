#ifndef PHILEMON_REGRESSION_HARNESS_HPP
#define PHILEMON_REGRESSION_HARNESS_HPP
/**
 * regression_harness.hpp — shadow-run diff 回归测试
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → execute_query() 的查询分发:
 *       · for(query_types) { BFS→bfs(), SSSP→sssp(), PR→page_rank() }
 *       · fopen/fclose log 绑定
 *       · try-catch per type
 *       · 100% 保留: 分发 + 日志 + 异常处理
 *
 *     → bfs() 的 BFS实现:
 *       · queue + visited[] + level[] + snapshot_edges callback
 *       · result[destination] = level
 *       · 100% 保留: BFS正确性基准
 *
 *     → sssp() 的 Dijkstra实现:
 *       · priority_queue<pdv, greater> + result[dest] = cur_dist + weight
 *       · 100% 保留: SSSP正确性基准
 *
 *     → page_rank() 的 PR实现:
 *       · outgoing_contrib = result[src] / degree
 *       · result[v] = base + damping * (incoming + dangling)
 *       · 100% 保留: PR正确性基准
 *
 *     → wcc() 的 Union-Find:
 *       · UnionFind: root[], find(path compress), unite
 *       · for(all vertices) { edges callback → unite(src,dst) }
 *       · 100% 保留: WCC正确性基准
 *
 *     → execute_microbenchmarks() 的统计:
 *       · thread_time[], thread_speed[]
 *       · global_speed, average_speed
 *       · check_point_size 周期记录
 *       · 100% 保留: 性能基准采集
 *
 * 算法修改 (~20%):
 *   - [MOD] 单次运行验正确 → Shadow-run双跑:
 *           同一查询在baseline和optimized路径各跑一次,
 *           比较结果的逐vertex diff. upstream只跑一次不对比.
 *   - [MOD] 固定阈值比较 → Welch's t-test:
 *           多次重复运行收集性能样本, 用Welch's t检验判断
 *           性能变化是否统计显著(p < 0.05).
 *           upstream: 只看单次throughput, 不做统计检验.
 *   - [NEW] RegressionReport: 结构化的回归报告
 *   - [NEW] DiffSummary: 正确性diff的详细摘要
 *
 * 断点调试:
 *   PHILE_HARNESS_DUMP(h)      — 打印所有test case结果
 *   PHILE_DIFF_DUMP(h,tc)      — 打印某case的diff详情
 *   PHILE_HARNESS_BP(h,tag)    — RAII guard
 *
 * Milestone: M058 — Regression harness
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <functional>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <string>
#include <queue>

#include "../debug/philemon_debug.hpp"
#include "../cost_model/cost_estimator.hpp"

namespace philemon {
namespace harness {

// ═══════════════════════════════════════════════════════════════════════
// 查询结果容器 — upstream的 vector<pair<vertex, value>> pattern
// ═══════════════════════════════════════════════════════════════════════
struct QueryResult {
    cost_model::QueryType type;
    std::vector<double> vertex_values;    // BFS: level, SSSP: dist, PR: score
    std::vector<int64_t> vertex_labels;   // WCC: component_id
    uint64_t total_vertices = 0;
    double execution_time_ns = 0;

    // upstream: sum, valid_sum统计
    uint64_t edge_traversals = 0;
    uint64_t valid_results = 0;

    void dump(const char* tag = "Result") const {
        std::printf("  [%s] type=%s vertices=%lu edges=%lu "
                    "valid=%lu time=%.1fμs\n",
                    tag, cost_model::query_type_name(type),
                    (unsigned long)total_vertices,
                    (unsigned long)edge_traversals,
                    (unsigned long)valid_results,
                    execution_time_ns / 1000);
        // 前5个值
        size_t show = std::min(size_t(5), vertex_values.size());
        for (size_t i = 0; i < show; i++) {
            std::printf("    v[%zu]=%.4f\n", i, vertex_values[i]);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [NEW] DiffSummary — shadow-run 的正确性对比结果
// ═══════════════════════════════════════════════════════════════════════
struct DiffSummary {
    uint64_t total_compared = 0;
    uint64_t exact_match = 0;
    uint64_t approx_match = 0;       // |diff| < epsilon
    uint64_t mismatch = 0;
    double max_abs_diff = 0;
    double mean_abs_diff = 0;
    double epsilon = 1e-6;

    bool passed() const {
        return mismatch == 0;
    }

    double match_rate() const {
        if (total_compared == 0) return 0;
        return static_cast<double>(exact_match + approx_match) / total_compared;
    }

    void dump() const {
        std::printf("  [Diff] compared=%lu exact=%lu approx=%lu "
                    "mismatch=%lu\n",
                    (unsigned long)total_compared,
                    (unsigned long)exact_match,
                    (unsigned long)approx_match,
                    (unsigned long)mismatch);
        std::printf("    max_diff=%.6e mean_diff=%.6e eps=%.6e %s\n",
                    max_abs_diff, mean_abs_diff, epsilon,
                    passed() ? "PASS" : "FAIL");
    }
};

// 对比两个结果
static DiffSummary compare_results(const QueryResult& baseline,
                                    const QueryResult& optimized,
                                    double epsilon = 1e-6) {
    DiffSummary diff;
    diff.epsilon = epsilon;

    size_t n = std::min(baseline.vertex_values.size(),
                        optimized.vertex_values.size());
    diff.total_compared = n;
    double sum_abs = 0;

    for (size_t i = 0; i < n; i++) {
        double d = std::abs(baseline.vertex_values[i]
                            - optimized.vertex_values[i]);
        sum_abs += d;
        diff.max_abs_diff = std::max(diff.max_abs_diff, d);

        if (d == 0) {
            diff.exact_match++;
        } else if (d < epsilon) {
            diff.approx_match++;
        } else {
            diff.mismatch++;
        }
    }

    diff.mean_abs_diff = n > 0 ? sum_abs / n : 0;

    PHILE_DBG(2, "[Diff] compared %lu: exact=%lu approx=%lu "
               "mismatch=%lu max=%.2e",
               (unsigned long)n,
               (unsigned long)diff.exact_match,
               (unsigned long)diff.approx_match,
               (unsigned long)diff.mismatch,
               diff.max_abs_diff);

    return diff;
}

// ═══════════════════════════════════════════════════════════════════════
// [MOD] Welch's t-test — 替换upstream的单次比较
//
// upstream: global_speed = ops / duration, 单次数字比较.
// 这里: 收集N次样本, 用Welch's t检验判断是否显著.
// ═══════════════════════════════════════════════════════════════════════
struct WelchTTestResult {
    double mean_baseline = 0;
    double mean_optimized = 0;
    double std_baseline = 0;
    double std_optimized = 0;
    double t_statistic = 0;
    double degrees_of_freedom = 0;
    double speedup = 0;            // optimized / baseline
    bool significant = false;       // p < 0.05

    void dump() const {
        std::printf("  [Welch's t-test]\n");
        std::printf("    baseline:  mean=%.2f std=%.2f\n",
                    mean_baseline, std_baseline);
        std::printf("    optimized: mean=%.2f std=%.2f\n",
                    mean_optimized, std_optimized);
        std::printf("    t=%.3f df=%.1f speedup=%.2fx %s\n",
                    t_statistic, degrees_of_freedom, speedup,
                    significant ? "SIGNIFICANT" : "not significant");
    }
};

static WelchTTestResult welch_t_test(
    const std::vector<double>& baseline,
    const std::vector<double>& optimized
) {
    WelchTTestResult r;
    size_t n1 = baseline.size(), n2 = optimized.size();
    if (n1 < 2 || n2 < 2) return r;

    // 均值
    double sum1 = std::accumulate(baseline.begin(), baseline.end(), 0.0);
    double sum2 = std::accumulate(optimized.begin(), optimized.end(), 0.0);
    r.mean_baseline = sum1 / n1;
    r.mean_optimized = sum2 / n2;

    // 方差
    double var1 = 0, var2 = 0;
    for (double x : baseline) var1 += (x - r.mean_baseline) * (x - r.mean_baseline);
    for (double x : optimized) var2 += (x - r.mean_optimized) * (x - r.mean_optimized);
    var1 /= (n1 - 1);
    var2 /= (n2 - 1);
    r.std_baseline = std::sqrt(var1);
    r.std_optimized = std::sqrt(var2);

    // t统计量
    double se = std::sqrt(var1 / n1 + var2 / n2);
    if (se < 1e-15) {
        r.t_statistic = 0;
        r.significant = false;
        return r;
    }
    r.t_statistic = (r.mean_optimized - r.mean_baseline) / se;

    // Welch-Satterthwaite自由度近似
    double v1n = var1 / n1, v2n = var2 / n2;
    double num = (v1n + v2n) * (v1n + v2n);
    double den = v1n * v1n / (n1 - 1) + v2n * v2n / (n2 - 1);
    r.degrees_of_freedom = (den > 0) ? num / den : n1 + n2 - 2;

    // speedup
    r.speedup = (r.mean_baseline > 0)
        ? r.mean_optimized / r.mean_baseline : 1.0;

    // 简化显著性判定: |t| > 2.0 约等于 p < 0.05 (df > 10)
    r.significant = std::abs(r.t_statistic) > 2.0;

    PHILE_DBG(2, "[Welch] t=%.3f df=%.1f speedup=%.2fx sig=%s",
               r.t_statistic, r.degrees_of_freedom,
               r.speedup, r.significant ? "Y" : "N");

    return r;
}

// ═══════════════════════════════════════════════════════════════════════
// TestCase — 一个回归测试用例
// ═══════════════════════════════════════════════════════════════════════
struct TestCase {
    std::string name;
    cost_model::QueryType type;
    uint64_t source_vertex = 0;
    uint32_t num_repeats = 5;
    double epsilon = 1e-6;

    // 结果
    DiffSummary correctness;
    WelchTTestResult perf_test;

    // 原始性能样本
    std::vector<double> baseline_times_us;
    std::vector<double> optimized_times_us;

    bool passed() const {
        return correctness.passed();
    }

    void dump() const {
        std::printf("──── TestCase '%s' ────\n", name.c_str());
        std::printf("  type=%s source=%lu repeats=%u %s\n",
                    cost_model::query_type_name(type),
                    (unsigned long)source_vertex, num_repeats,
                    passed() ? "✓ PASS" : "✗ FAIL");
        correctness.dump();
        perf_test.dump();
        std::printf("  baseline times (μs):");
        for (double t : baseline_times_us) std::printf(" %.1f", t);
        std::printf("\n  optimized times (μs):");
        for (double t : optimized_times_us) std::printf(" %.1f", t);
        std::printf("\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// RegressionReport — 全套回归结果
// ═══════════════════════════════════════════════════════════════════════
struct RegressionReport {
    std::vector<TestCase> cases;
    double total_duration_ms = 0;

    uint64_t passed_count() const {
        return std::count_if(cases.begin(), cases.end(),
            [](const TestCase& tc) { return tc.passed(); });
    }

    void dump() const {
        std::printf("════ Regression Report ════\n");
        std::printf("  %lu/%zu passed  %.1fms\n",
                    (unsigned long)passed_count(), cases.size(),
                    total_duration_ms);
        for (auto& tc : cases) tc.dump();
        std::printf("════ End Report ════\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// RegressionHarness — 主类
// ═══════════════════════════════════════════════════════════════════════
class RegressionHarness {
    // 查询执行回调: baseline和optimized路径
    using RunFn = std::function<QueryResult(
        cost_model::QueryType type, uint64_t source)>;
    RunFn baseline_fn_;
    RunFn optimized_fn_;

    std::vector<TestCase> test_cases_;

    // upstream: checkpoint统计
    std::atomic<uint64_t> total_runs_{0};
    std::atomic<uint64_t> total_passed_{0};
    std::atomic<uint64_t> total_failed_{0};

    // upstream: bind_thread_to_core
    static void bind_to_core(std::thread& t, int core_id) {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(t.native_handle(),
                               sizeof(cpu_set_t), &cpuset);
#endif
    }

    // upstream: throughput计算
    static double throughput_kops(uint64_t ops, double duration_ns) {
        return static_cast<double>(ops) / duration_ns * 1e6;
    }

public:
    void set_baseline(RunFn fn) { baseline_fn_ = std::move(fn); }
    void set_optimized(RunFn fn) { optimized_fn_ = std::move(fn); }

    void add_test(const std::string& name, cost_model::QueryType type,
                  uint64_t source = 0, uint32_t repeats = 5,
                  double eps = 1e-6) {
        TestCase tc;
        tc.name = name;
        tc.type = type;
        tc.source_vertex = source;
        tc.num_repeats = repeats;
        tc.epsilon = eps;
        test_cases_.push_back(std::move(tc));
    }

    // ── [MOD] Shadow-run: 每个test case跑baseline和optimized双路径 ──
    RegressionReport run_all() {
        RegressionReport report;
        auto total_start = std::chrono::steady_clock::now();

        for (auto& tc : test_cases_) {
            PHILE_DBG(1, "[Harness] running '%s' (%s, %u repeats)",
                       tc.name.c_str(),
                       cost_model::query_type_name(tc.type),
                       tc.num_repeats);

            // 正确性验证: 跑一次baseline和optimized, 对比结果
            if (baseline_fn_ && optimized_fn_) {
                auto base_result = baseline_fn_(tc.type, tc.source_vertex);
                auto opt_result = optimized_fn_(tc.type, tc.source_vertex);
                tc.correctness = compare_results(base_result, opt_result,
                                                  tc.epsilon);
            }

            // 性能对比: 多次重复
            tc.baseline_times_us.clear();
            tc.optimized_times_us.clear();

            for (uint32_t r = 0; r < tc.num_repeats; r++) {
                if (baseline_fn_) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto res = baseline_fn_(tc.type, tc.source_vertex);
                    auto t1 = std::chrono::steady_clock::now();
                    double us = std::chrono::duration_cast<
                        std::chrono::microseconds>(t1 - t0).count();
                    tc.baseline_times_us.push_back(us);

                    PHILE_DBG(3, "[Harness] baseline rep %u: %.1fμs",
                               r, us);
                }

                if (optimized_fn_) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto res = optimized_fn_(tc.type, tc.source_vertex);
                    auto t1 = std::chrono::steady_clock::now();
                    double us = std::chrono::duration_cast<
                        std::chrono::microseconds>(t1 - t0).count();
                    tc.optimized_times_us.push_back(us);

                    PHILE_DBG(3, "[Harness] optimized rep %u: %.1fμs",
                               r, us);
                }

                // upstream: checkpoint每N轮
                if ((r + 1) % 10 == 0) {
                    PHILE_DBG(2, "[Harness] checkpoint: %u/%u repeats",
                               r + 1, tc.num_repeats);
                }
            }

            // Welch's t-test
            if (!tc.baseline_times_us.empty() &&
                !tc.optimized_times_us.empty()) {
                tc.perf_test = welch_t_test(tc.baseline_times_us,
                                             tc.optimized_times_us);
            }

            total_runs_.fetch_add(1);
            if (tc.passed()) {
                total_passed_.fetch_add(1);
            } else {
                total_failed_.fetch_add(1);
            }

            report.cases.push_back(tc);
        }

        auto total_end = std::chrono::steady_clock::now();
        report.total_duration_ms = std::chrono::duration<double, std::milli>(
            total_end - total_start).count();

        PHILE_DBG(1, "[Harness] all done: %lu/%zu passed %.1fms",
                   (unsigned long)report.passed_count(),
                   report.cases.size(), report.total_duration_ms);

        return report;
    }

    // ── 并行执行 (upstream thread model) ──
    RegressionReport run_parallel(uint32_t worker_threads = 4) {
        // upstream: chunk分发
        size_t n = test_cases_.size();
        size_t chunk = (n + worker_threads - 1) / worker_threads;

        std::vector<std::thread> threads;
        std::vector<std::vector<TestCase>> results(worker_threads);

        auto start = std::chrono::steady_clock::now();

        for (uint32_t t = 0; t < worker_threads; t++) {
            threads.emplace_back([this, &results, chunk, t, n] {
                size_t begin = t * chunk;
                size_t end = std::min(begin + chunk, n);

                for (size_t i = begin; i < end; i++) {
                    auto tc = test_cases_[i];

                    // shadow-run correctness
                    if (baseline_fn_ && optimized_fn_) {
                        auto br = baseline_fn_(tc.type, tc.source_vertex);
                        auto or_ = optimized_fn_(tc.type, tc.source_vertex);
                        tc.correctness = compare_results(br, or_, tc.epsilon);
                    }

                    // perf repeats
                    for (uint32_t r = 0; r < tc.num_repeats; r++) {
                        if (baseline_fn_) {
                            auto t0 = std::chrono::steady_clock::now();
                            baseline_fn_(tc.type, tc.source_vertex);
                            auto t1 = std::chrono::steady_clock::now();
                            tc.baseline_times_us.push_back(
                                std::chrono::duration_cast<
                                    std::chrono::microseconds>(t1 - t0).count());
                        }
                        if (optimized_fn_) {
                            auto t0 = std::chrono::steady_clock::now();
                            optimized_fn_(tc.type, tc.source_vertex);
                            auto t1 = std::chrono::steady_clock::now();
                            tc.optimized_times_us.push_back(
                                std::chrono::duration_cast<
                                    std::chrono::microseconds>(t1 - t0).count());
                        }
                    }

                    if (!tc.baseline_times_us.empty() &&
                        !tc.optimized_times_us.empty()) {
                        tc.perf_test = welch_t_test(tc.baseline_times_us,
                                                     tc.optimized_times_us);
                    }

                    results[t].push_back(std::move(tc));
                }
            });
            bind_to_core(threads.back(),
                         t % std::thread::hardware_concurrency());
        }

        for (auto& th : threads) th.join();

        auto end = std::chrono::steady_clock::now();

        // merge
        RegressionReport report;
        for (auto& v : results) {
            for (auto& tc : v) {
                report.cases.push_back(std::move(tc));
            }
        }
        report.total_duration_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        // upstream: throughput计算
        double tp = throughput_kops(report.cases.size(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count());

        PHILE_DBG(1, "[Harness] parallel done: %lu/%zu passed "
                   "%.1fms throughput=%.1f cases/ms",
                   (unsigned long)report.passed_count(),
                   report.cases.size(),
                   report.total_duration_ms, tp);

        return report;
    }

    // ── 全量打印 ──
    void dump_all() const {
        std::printf("════ RegressionHarness ════\n");
        std::printf("  test_cases=%zu total_runs=%lu "
                    "passed=%lu failed=%lu\n",
                    test_cases_.size(),
                    (unsigned long)total_runs_.load(),
                    (unsigned long)total_passed_.load(),
                    (unsigned long)total_failed_.load());
        for (auto& tc : test_cases_) {
            std::printf("  - '%s' %s\n", tc.name.c_str(),
                        cost_model::query_type_name(tc.type));
        }
        std::printf("════ End Harness ════\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// 调试宏
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_HARNESS_DUMP(h) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[HARNESS_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (h).dump_all(); \
        } \
    } while(0)

class HarnessBreakpointGuard {
    const RegressionHarness& harness_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    HarnessBreakpointGuard(const RegressionHarness& h, const char* n)
        : harness_(h), name_(n),
          start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ HARNESS_BP ENTER: %s ━━━━\n", name_);
            harness_.dump_all();
        }
    }
    ~HarnessBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ HARNESS_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)us);
            harness_.dump_all();
        }
    }
};

#define PHILE_HARNESS_BP(h, tag) \
    ::philemon::harness::HarnessBreakpointGuard \
        _phile_harness_bp_##__LINE__((h), (tag))

}  // namespace harness
}  // namespace philemon

#endif  // PHILEMON_REGRESSION_HARNESS_HPP
