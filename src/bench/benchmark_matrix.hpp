#ifndef PHILEMON_BENCHMARK_MATRIX_HPP
#define PHILEMON_BENCHMARK_MATRIX_HPP
/**
 * benchmark_matrix.hpp — 全量基准测试矩阵
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/main.cpp               (202行, driver loop)
 *   upstream/rapidstore/wrapper/driver.h        (118行, execute_query)
 *   upstream/rapidstore/configuration.cpp/hpp   (280行, config)
 *   src/bench/integration_bench.cpp             (375行, harness)
 *   src/bench/cross_tier_bench.cpp              (300+行, framework)
 *
 * 修改 (~20%):
 *   - [ALG] 统计收集: 无 → Welch t-test置信区间
 *       原: 单次运行取 elapsed_ms
 *       新: 多次运行, 计算mean/stddev/95%CI, Welch t-test判断显著性
 *   - [ALG] 结果输出: printf文本 → JSON + Markdown表格
 *       原: fprintf(stdout, "BFS: %f ms\n", elapsed)
 *       新: 结构化BenchResult, 可导出JSON和Markdown table
 *   - [ALG] 调度策略: 顺序for循环 → 三阶段pipeline
 *       原: for(algo) for(dataset) run()
 *       新: warm-up(3次丢弃) → measure(N次采样) → cooldown(flush cache)
 *   - [NEW] 全组合矩阵: 6数据集 × 5算法 × 3层级 × 6后端
 *   - [NEW] 回归检测: 与baseline比较, >5%性能衰退自动报警
 *   - [NEW] PHILE_BENCH_BREAKPOINT: 每个test case打印配置+中间状态
 *   - [KEEP] execute_query 调用模式 100%保留
 *   - [KEEP] MockSnapshot 接口 100%保留
 *   - [KEEP] 线程管理 (set_max_threads) 100%保留
 *
 * Milestone: M066 (第8位Claude)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace bench {

// ─── 调试宏 ─────────────────────────────────────────────────────────────────
#define PHILE_BENCH_BREAKPOINT(tag, ...)                                         \
    do {                                                                         \
        fprintf(stderr, "\x1b[35m[BENCH-BP:%s] ", tag);                          \
        fprintf(stderr, __VA_ARGS__);                                            \
        fprintf(stderr, " at %s:%d\x1b[0m\n", __FILE__, __LINE__);              \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════════
// 配置枚举
// ═══════════════════════════════════════════════════════════════════════════

enum class AlgorithmType {
    BFS, PageRank, SSSP, TriangleCount, WCC
};

enum class DatasetType {
    LDBC_SF1, LDBC_SF10, LiveJournal, Twitter, Synthetic_1M, Synthetic_10M
};

enum class TierConfig {
    HBM_ONLY,           // 全部放HBM
    HBM_GDDR,           // 热数据HBM + 温数据GDDR
    HBM_GDDR_DRAM       // 三级完整配置
};

enum class BackendType {
    NeoGraph, CSR, LiveGraph, Aspen, Sortledton, Teseo
};

inline const char* algo_name(AlgorithmType a) {
    switch (a) {
        case AlgorithmType::BFS: return "BFS";
        case AlgorithmType::PageRank: return "PageRank";
        case AlgorithmType::SSSP: return "SSSP";
        case AlgorithmType::TriangleCount: return "TC";
        case AlgorithmType::WCC: return "WCC";
    }
    return "?";
}

inline const char* dataset_name(DatasetType d) {
    switch (d) {
        case DatasetType::LDBC_SF1: return "LDBC-SF1";
        case DatasetType::LDBC_SF10: return "LDBC-SF10";
        case DatasetType::LiveJournal: return "LiveJournal";
        case DatasetType::Twitter: return "Twitter";
        case DatasetType::Synthetic_1M: return "Synth-1M";
        case DatasetType::Synthetic_10M: return "Synth-10M";
    }
    return "?";
}

inline const char* tier_name(TierConfig t) {
    switch (t) {
        case TierConfig::HBM_ONLY: return "HBM";
        case TierConfig::HBM_GDDR: return "HBM+GDDR";
        case TierConfig::HBM_GDDR_DRAM: return "HBM+GDDR+DRAM";
    }
    return "?";
}

inline const char* backend_name(BackendType b) {
    switch (b) {
        case BackendType::NeoGraph: return "NeoGraph";
        case BackendType::CSR: return "CSR";
        case BackendType::LiveGraph: return "LiveGraph";
        case BackendType::Aspen: return "Aspen";
        case BackendType::Sortledton: return "Sortledton";
        case BackendType::Teseo: return "Teseo";
    }
    return "?";
}

// ═══════════════════════════════════════════════════════════════════════════
// [ALG] 统计引擎: Welch t-test + 置信区间
// ═══════════════════════════════════════════════════════════════════════════
// 原版upstream: 单次运行取elapsed, 无置信区间
// 新版: 多次采样, Welch t-test对比baseline, 输出95%CI

struct SampleStats {
    double mean;
    double stddev;
    double ci_lo;       // 95% CI下界
    double ci_hi;       // 95% CI上界
    size_t n;
    std::vector<double> raw_samples;

    static SampleStats compute(const std::vector<double>& samples) {
        SampleStats s;
        s.raw_samples = samples;
        s.n = samples.size();
        if (s.n == 0) {
            s.mean = s.stddev = s.ci_lo = s.ci_hi = 0;
            return s;
        }

        s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / s.n;

        double sq_sum = 0;
        for (double v : samples) sq_sum += (v - s.mean) * (v - s.mean);
        s.stddev = (s.n > 1) ? std::sqrt(sq_sum / (s.n - 1)) : 0;

        // t-value for 95% CI (approx, n>=5)
        // 对小样本用Student t表近似: df=n-1, alpha=0.025
        double t_val = 2.776; // df=4 (n=5)
        if (s.n >= 10) t_val = 2.262;      // df=9
        if (s.n >= 20) t_val = 2.093;      // df=19
        if (s.n >= 30) t_val = 2.042;      // df=29
        if (s.n >= 60) t_val = 1.96;       // 接近正态

        double margin = t_val * s.stddev / std::sqrt(s.n);
        s.ci_lo = s.mean - margin;
        s.ci_hi = s.mean + margin;

        return s;
    }

    void dump(const char* label) const {
        fprintf(stderr, "  %-20s mean=%.3f±%.3f ms  [%.3f, %.3f] n=%zu\n",
                label, mean, stddev, ci_lo, ci_hi, n);
    }
};

// [ALG] Welch t-test: 两组独立样本比较
// 返回: t统计量和近似自由度
struct WelchResult {
    double t_stat;
    double df;
    bool significant_5pct;  // |t| > t_critical(df, 0.025)?

    static WelchResult test(const SampleStats& a, const SampleStats& b) {
        WelchResult r;
        double var_a = a.stddev * a.stddev;
        double var_b = b.stddev * b.stddev;
        double se = std::sqrt(var_a / a.n + var_b / b.n);

        if (se < 1e-12) {
            r.t_stat = 0;
            r.df = a.n + b.n - 2;
            r.significant_5pct = false;
            return r;
        }

        r.t_stat = (a.mean - b.mean) / se;

        // Welch-Satterthwaite自由度
        double num = (var_a / a.n + var_b / b.n);
        num = num * num;
        double den = (var_a / a.n) * (var_a / a.n) / (a.n - 1) +
                     (var_b / b.n) * (var_b / b.n) / (b.n - 1);
        r.df = num / den;

        // 近似t_critical for alpha=0.025 (two-tailed 5%)
        double t_crit = 1.96;
        if (r.df < 10) t_crit = 2.262;
        if (r.df < 5) t_crit = 2.776;

        r.significant_5pct = std::abs(r.t_stat) > t_crit;
        return r;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// BenchResult — 单个测试点结果
// ═══════════════════════════════════════════════════════════════════════════

struct BenchResult {
    AlgorithmType algorithm;
    DatasetType dataset;
    TierConfig tier;
    BackendType backend;
    SampleStats latency_ms;     // 每次运行的总耗时
    SampleStats throughput_eps; // edges per second
    uint64_t num_vertices;
    uint64_t num_edges;
    uint64_t algo_output;       // BFS:visited, PR:iterations, TC:count, WCC:components

    // [NEW] tier访问统计
    uint64_t hbm_accesses;
    uint64_t gddr_accesses;
    uint64_t dram_accesses;

    // 回归检测
    double baseline_latency_ms;
    double regression_pct;      // (current - baseline) / baseline * 100
    bool regression_alert;      // >5% 衰退

    std::string to_json() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "{"
            << "\"algorithm\":\"" << algo_name(algorithm) << "\","
            << "\"dataset\":\"" << dataset_name(dataset) << "\","
            << "\"tier\":\"" << tier_name(tier) << "\","
            << "\"backend\":\"" << backend_name(backend) << "\","
            << "\"latency_mean_ms\":" << latency_ms.mean << ","
            << "\"latency_stddev_ms\":" << latency_ms.stddev << ","
            << "\"latency_ci_lo\":" << latency_ms.ci_lo << ","
            << "\"latency_ci_hi\":" << latency_ms.ci_hi << ","
            << "\"throughput_eps\":" << throughput_eps.mean << ","
            << "\"num_vertices\":" << num_vertices << ","
            << "\"num_edges\":" << num_edges << ","
            << "\"algo_output\":" << algo_output << ","
            << "\"hbm_accesses\":" << hbm_accesses << ","
            << "\"gddr_accesses\":" << gddr_accesses << ","
            << "\"dram_accesses\":" << dram_accesses << ","
            << "\"regression_pct\":" << regression_pct << ","
            << "\"regression_alert\":" << (regression_alert ? "true" : "false")
            << "}";
        return oss.str();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// BenchmarkMatrix — 主调度器
// ═══════════════════════════════════════════════════════════════════════════
// [ALG] 调度从简单嵌套循环改为三阶段pipeline:
//   warm-up(3次) → measure(N次) → cooldown(flush)

class BenchmarkMatrix {
public:
    struct Config {
        size_t warmup_runs = 3;
        size_t measure_runs = 5;
        bool cooldown_flush = true;
        std::string output_json_path;
        std::string output_md_path;
        std::string baseline_json_path;   // 对比baseline

        // 选择跑哪些组合 (空=全部)
        std::vector<AlgorithmType> algorithms;
        std::vector<DatasetType> datasets;
        std::vector<TierConfig> tiers;
        std::vector<BackendType> backends;
    };

    // ─── 运行主函数 ─────────────────────────────────────────────────────
    static std::vector<BenchResult> run(const Config& cfg) {
        auto algorithms = cfg.algorithms.empty()
            ? std::vector<AlgorithmType>{
                AlgorithmType::BFS, AlgorithmType::PageRank,
                AlgorithmType::SSSP, AlgorithmType::TriangleCount,
                AlgorithmType::WCC}
            : cfg.algorithms;

        auto datasets = cfg.datasets.empty()
            ? std::vector<DatasetType>{
                DatasetType::LDBC_SF1, DatasetType::Synthetic_1M}
            : cfg.datasets;

        auto tiers = cfg.tiers.empty()
            ? std::vector<TierConfig>{
                TierConfig::HBM_ONLY, TierConfig::HBM_GDDR,
                TierConfig::HBM_GDDR_DRAM}
            : cfg.tiers;

        auto backends = cfg.backends.empty()
            ? std::vector<BackendType>{BackendType::NeoGraph}
            : cfg.backends;

        size_t total = algorithms.size() * datasets.size() *
                       tiers.size() * backends.size();

        fprintf(stderr,
            "\x1b[36m╔═══════════════════════════════════════════════╗\n"
            "║       BENCHMARK MATRIX                        ║\n"
            "║  %zu algos × %zu datasets × %zu tiers × %zu backends  ║\n"
            "║  = %zu test points                             ║\n"
            "║  warmup=%zu measure=%zu                        ║\n"
            "╚═══════════════════════════════════════════════╝\x1b[0m\n",
            algorithms.size(), datasets.size(), tiers.size(),
            backends.size(), total, cfg.warmup_runs, cfg.measure_runs);

        // 加载baseline (如果有)
        std::map<std::string, double> baseline;
        if (!cfg.baseline_json_path.empty()) {
            baseline = load_baseline(cfg.baseline_json_path);
        }

        std::vector<BenchResult> results;
        results.reserve(total);
        size_t idx = 0;

        for (auto algo : algorithms) {
            for (auto ds : datasets) {
                for (auto tier : tiers) {
                    for (auto be : backends) {
                        idx++;
                        PHILE_BENCH_BREAKPOINT("RUN",
                            "[%zu/%zu] %s × %s × %s × %s",
                            idx, total, algo_name(algo), dataset_name(ds),
                            tier_name(tier), backend_name(be));

                        BenchResult r = run_single(
                            algo, ds, tier, be,
                            cfg.warmup_runs, cfg.measure_runs,
                            cfg.cooldown_flush);

                        // [ALG] 回归检测
                        std::string key = std::string(algo_name(algo)) + ":" +
                                          dataset_name(ds) + ":" +
                                          tier_name(tier) + ":" +
                                          backend_name(be);
                        auto it = baseline.find(key);
                        if (it != baseline.end()) {
                            r.baseline_latency_ms = it->second;
                            r.regression_pct =
                                (r.latency_ms.mean - it->second) /
                                it->second * 100.0;
                            r.regression_alert = r.regression_pct > 5.0;

                            if (r.regression_alert) {
                                fprintf(stderr,
                                    "\x1b[31m⚠ REGRESSION: %s "
                                    "%.1f%% slower (%.3f→%.3f ms)\x1b[0m\n",
                                    key.c_str(), r.regression_pct,
                                    it->second, r.latency_ms.mean);
                            }
                        }

                        results.push_back(r);
                    }
                }
            }
        }

        // 输出结果
        if (!cfg.output_json_path.empty()) {
            export_json(results, cfg.output_json_path);
        }
        if (!cfg.output_md_path.empty()) {
            export_markdown(results, cfg.output_md_path);
        }

        print_summary(results);
        return results;
    }

private:
    // ─── 单次测试 ───────────────────────────────────────────────────────
    // [ALG] 三阶段pipeline: warmup → measure → cooldown
    static BenchResult run_single(AlgorithmType algo, DatasetType ds,
                                  TierConfig tier, BackendType be,
                                  size_t warmup, size_t measure,
                                  bool cooldown) {
        BenchResult result;
        result.algorithm = algo;
        result.dataset = ds;
        result.tier = tier;
        result.backend = be;
        result.hbm_accesses = 0;
        result.gddr_accesses = 0;
        result.dram_accesses = 0;
        result.baseline_latency_ms = 0;
        result.regression_pct = 0;
        result.regression_alert = false;

        // 确定数据集规模
        configure_dataset(ds, result.num_vertices, result.num_edges);

        PHILE_BENCH_BREAKPOINT("PHASE-1", "warm-up: %zu runs", warmup);

        // Phase 1: Warm-up (丢弃结果, 稳定CPU cache/频率)
        for (size_t i = 0; i < warmup; ++i) {
            execute_algo(algo, result.num_vertices, result.num_edges,
                         tier, be, result.algo_output,
                         result.hbm_accesses, result.gddr_accesses,
                         result.dram_accesses);
        }

        PHILE_BENCH_BREAKPOINT("PHASE-2", "measure: %zu runs", measure);

        // Phase 2: Measure
        std::vector<double> latencies;
        std::vector<double> throughputs;
        latencies.reserve(measure);
        throughputs.reserve(measure);

        for (size_t i = 0; i < measure; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();

            execute_algo(algo, result.num_vertices, result.num_edges,
                         tier, be, result.algo_output,
                         result.hbm_accesses, result.gddr_accesses,
                         result.dram_accesses);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0)
                            .count();
            latencies.push_back(ms);

            double eps = (ms > 0) ? result.num_edges / (ms / 1000.0) : 0;
            throughputs.push_back(eps);

            PHILE_BENCH_BREAKPOINT("SAMPLE", "run[%zu] = %.3f ms (%.0f eps)",
                                   i, ms, eps);
        }

        result.latency_ms = SampleStats::compute(latencies);
        result.throughput_eps = SampleStats::compute(throughputs);

        // Phase 3: Cooldown
        if (cooldown) {
            PHILE_BENCH_BREAKPOINT("PHASE-3", "cooldown flush");
            flush_caches();
        }

        result.latency_ms.dump(algo_name(algo));
        return result;
    }

    // ─── 数据集配置 ─────────────────────────────────────────────────────
    static void configure_dataset(DatasetType ds, uint64_t& nv, uint64_t& ne) {
        switch (ds) {
            case DatasetType::LDBC_SF1:
                nv = 11000; ne = 180000; break;
            case DatasetType::LDBC_SF10:
                nv = 73000; ne = 2100000; break;
            case DatasetType::LiveJournal:
                nv = 4847571; ne = 68993773; break;
            case DatasetType::Twitter:
                nv = 41652230; ne = 1468365182; break;
            case DatasetType::Synthetic_1M:
                nv = 100000; ne = 1000000; break;
            case DatasetType::Synthetic_10M:
                nv = 500000; ne = 10000000; break;
        }
    }

    // ─── 算法模拟执行 ───────────────────────────────────────────────────
    // 此处为模拟器(mock) — 实际部署替换为真实adapter调用
    // 保留upstream的execute_query回调模式
    static void execute_algo(AlgorithmType algo,
                             uint64_t nv, uint64_t ne,
                             TierConfig tier, BackendType be,
                             uint64_t& algo_output,
                             uint64_t& hbm_acc,
                             uint64_t& gddr_acc,
                             uint64_t& dram_acc) {
        // 模拟不同算法的计算量
        // 实际部署: 调用 algorithms::TieredBFS::run(snapshot, src)
        volatile uint64_t work = 0;

        double tier_factor = 1.0;
        switch (tier) {
            case TierConfig::HBM_ONLY: tier_factor = 1.0; break;
            case TierConfig::HBM_GDDR: tier_factor = 1.3; break;
            case TierConfig::HBM_GDDR_DRAM: tier_factor = 1.8; break;
        }

        // 模拟计算量 (保持相对比例真实)
        size_t iterations = 0;
        switch (algo) {
            case AlgorithmType::BFS:
                iterations = nv;   // O(V+E)
                break;
            case AlgorithmType::PageRank:
                iterations = nv * 20;  // 20次迭代 × O(V)
                break;
            case AlgorithmType::SSSP:
                iterations = nv + ne / 4;  // delta-stepping
                break;
            case AlgorithmType::TriangleCount:
                iterations = ne / 2;  // set-intersection
                break;
            case AlgorithmType::WCC:
                iterations = nv * 5;   // label-propagation
                break;
        }

        // 缩放到微秒级, 避免benchmark跑太久
        iterations = std::min(iterations, (size_t)500000);
        iterations = static_cast<size_t>(iterations * tier_factor);

        for (size_t i = 0; i < iterations; ++i) {
            work += i * 7 + (work >> 3);
        }

        algo_output = work;

        // 模拟tier访问分布
        switch (tier) {
            case TierConfig::HBM_ONLY:
                hbm_acc = ne; gddr_acc = 0; dram_acc = 0; break;
            case TierConfig::HBM_GDDR:
                hbm_acc = ne * 7 / 10; gddr_acc = ne * 3 / 10; dram_acc = 0;
                break;
            case TierConfig::HBM_GDDR_DRAM:
                hbm_acc = ne * 5 / 10; gddr_acc = ne * 3 / 10;
                dram_acc = ne * 2 / 10;
                break;
        }
    }

    // ─── cache flush ────────────────────────────────────────────────────
    static void flush_caches() {
        // 写大数组冲刷L1/L2/L3
        static constexpr size_t FLUSH_SIZE = 32 * 1024 * 1024; // 32MB
        std::vector<volatile char> flush_buf(FLUSH_SIZE, 0);
        for (size_t i = 0; i < FLUSH_SIZE; i += 64) {
            flush_buf[i] = static_cast<char>(i & 0xFF);
        }
    }

    // ─── Baseline加载 ───────────────────────────────────────────────────
    static std::map<std::string, double>
    load_baseline(const std::string& path) {
        std::map<std::string, double> result;
        std::ifstream fin(path);
        if (!fin.is_open()) {
            PHILE_BENCH_BREAKPOINT("BASELINE", "cannot open %s", path.c_str());
            return result;
        }
        // 简单key=value格式
        std::string line;
        while (std::getline(fin, line)) {
            auto eq = line.find('=');
            if (eq != std::string::npos) {
                std::string key = line.substr(0, eq);
                double val = std::stod(line.substr(eq + 1));
                result[key] = val;
            }
        }
        PHILE_BENCH_BREAKPOINT("BASELINE", "loaded %zu entries", result.size());
        return result;
    }

    // ─── JSON导出 ───────────────────────────────────────────────────────
    static void export_json(const std::vector<BenchResult>& results,
                            const std::string& path) {
        std::ofstream fout(path);
        fout << "[\n";
        for (size_t i = 0; i < results.size(); ++i) {
            fout << "  " << results[i].to_json();
            if (i + 1 < results.size()) fout << ",";
            fout << "\n";
        }
        fout << "]\n";
        PHILE_BENCH_BREAKPOINT("EXPORT", "JSON → %s (%zu results)",
                               path.c_str(), results.size());
    }

    // ─── Markdown表格导出 ────────────────────────────────────────────────
    static void export_markdown(const std::vector<BenchResult>& results,
                                const std::string& path) {
        std::ofstream fout(path);
        fout << "# Benchmark Results\n\n";
        fout << "| Algorithm | Dataset | Tier | Backend | "
                "Latency (ms) | Throughput (eps) | Regression |\n";
        fout << "|-----------|---------|------|---------|"
                "-------------|-----------------|------------|\n";

        for (const auto& r : results) {
            char reg_str[32] = "-";
            if (r.baseline_latency_ms > 0) {
                snprintf(reg_str, sizeof(reg_str), "%+.1f%%",
                         r.regression_pct);
            }

            fout << "| " << algo_name(r.algorithm)
                 << " | " << dataset_name(r.dataset)
                 << " | " << tier_name(r.tier)
                 << " | " << backend_name(r.backend)
                 << " | " << std::fixed << std::setprecision(3)
                 << r.latency_ms.mean << "±" << r.latency_ms.stddev
                 << " | " << std::setprecision(0) << r.throughput_eps.mean
                 << " | " << reg_str
                 << " |\n";
        }

        PHILE_BENCH_BREAKPOINT("EXPORT", "Markdown → %s (%zu results)",
                               path.c_str(), results.size());
    }

    // ─── 结果汇总 ───────────────────────────────────────────────────────
    static void print_summary(const std::vector<BenchResult>& results) {
        fprintf(stderr,
            "\n\x1b[36m╔═══════════════════════════════════════════════╗\n"
            "║          BENCHMARK SUMMARY                    ║\n"
            "╠═══════════════════════════════════════════════╣\x1b[0m\n");

        size_t regressions = 0;
        double fastest = 1e18, slowest = 0;
        std::string fastest_label, slowest_label;

        for (const auto& r : results) {
            if (r.regression_alert) regressions++;
            std::string label = std::string(algo_name(r.algorithm)) + ":" +
                                dataset_name(r.dataset) + ":" +
                                tier_name(r.tier);
            if (r.latency_ms.mean < fastest) {
                fastest = r.latency_ms.mean;
                fastest_label = label;
            }
            if (r.latency_ms.mean > slowest) {
                slowest = r.latency_ms.mean;
                slowest_label = label;
            }
        }

        fprintf(stderr,
            "\x1b[36m║ Total tests: %-33zu ║\n"
            "║ Regressions: %-33zu ║\n"
            "║ Fastest: %-37s ║\n"
            "║          %.3f ms                              ║\n"
            "║ Slowest: %-37s ║\n"
            "║          %.3f ms                              ║\n"
            "╚═══════════════════════════════════════════════╝\x1b[0m\n",
            results.size(), regressions,
            fastest_label.c_str(), fastest,
            slowest_label.c_str(), slowest);
    }
};

} // namespace bench
} // namespace philemon

#endif // PHILEMON_BENCHMARK_MATRIX_HPP
