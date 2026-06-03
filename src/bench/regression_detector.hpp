#ifndef PHILEMON_REGRESSION_DETECTOR_HPP
#define PHILEMON_REGRESSION_DETECTOR_HPP
/**
 * regression_detector.hpp — 性能回归自动检测
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/benchmark/baseline.cpp  (概念性)
 *   src/bench/benchmark_matrix.hpp              (SampleStats/WelchResult)
 *   通用CI回归检测模式
 *
 * 修改 (~20%):
 *   - [ALG] 检测算法: 简单阈值 → Welch t-test + CUSUM变点检测
 *       原: if (current > baseline * 1.05) alert()
 *       新: 两层检测:
 *         1. Welch t-test: 单次commit的突变检测(p<0.05)
 *         2. CUSUM: 累积和控制图, 检测渐进性能退化
 *            S_n = max(0, S_{n-1} + x_n - mu - k), 当S_n > h时告警
 *            k=allowable slack(0.5σ), h=decision interval(4σ)
 *   - [ALG] 告警: 无 → 三级告警
 *       INFO:     回归<3%, 可能是噪声
 *       WARNING:  回归3-10%, 需要关注
 *       CRITICAL: 回归>10%, 阻塞合并
 *   - [NEW] 历史趋势追踪: 保存最近100个commit的结果
 *   - [NEW] 移动平均平滑: 5-point SMA消除短期波动
 *   - [NEW] PHILE_REG_BREAKPOINT: 每次检测打印详细状态
 *   - [KEEP] benchmark result格式 100%保留
 *
 * Milestone: M069 (第8位Claude)
 */

#include "benchmark_matrix.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace philemon {
namespace bench {

// ─── 调试宏 ─────────────────────────────────────────────────────────────────
#define PHILE_REG_BREAKPOINT(tag, ...)                                           \
    do {                                                                          \
        fprintf(stderr, "\x1b[31m[REG-BP:%s] ", tag);                            \
        fprintf(stderr, __VA_ARGS__);                                            \
        fprintf(stderr, " at %s:%d\x1b[0m\n", __FILE__, __LINE__);              \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════════
// 告警级别
// ═══════════════════════════════════════════════════════════════════════════

enum class AlertLevel {
    OK,
    INFO,       // <3%
    WARNING,    // 3-10%
    CRITICAL    // >10%
};

inline const char* alert_name(AlertLevel level) {
    switch (level) {
        case AlertLevel::OK: return "OK";
        case AlertLevel::INFO: return "INFO";
        case AlertLevel::WARNING: return "WARNING";
        case AlertLevel::CRITICAL: return "CRITICAL";
    }
    return "?";
}

inline const char* alert_color(AlertLevel level) {
    switch (level) {
        case AlertLevel::OK: return "\x1b[32m";
        case AlertLevel::INFO: return "\x1b[36m";
        case AlertLevel::WARNING: return "\x1b[33m";
        case AlertLevel::CRITICAL: return "\x1b[31m";
    }
    return "\x1b[0m";
}

// ═══════════════════════════════════════════════════════════════════════════
// 历史数据点
// ═══════════════════════════════════════════════════════════════════════════

struct HistoryPoint {
    std::string commit_hash;
    std::string timestamp;
    std::string test_name;
    double latency_ms;
    double throughput_eps;
    double memory_mb;
};

// ═══════════════════════════════════════════════════════════════════════════
// [ALG] CUSUM — 累积和控制图变点检测
// ═══════════════════════════════════════════════════════════════════════════
// 原版: 无变点检测
// 新版: Page's CUSUM算法, 检测均值从mu_0偏移到mu_1的变点
//   S_n^+ = max(0, S_{n-1}^+ + (x_n - mu_0 - k))  检测正向偏移(变慢)
//   S_n^- = max(0, S_{n-1}^- - (x_n - mu_0 - k))  检测负向偏移(变快)
//   当 S_n^+ > h 或 S_n^- > h 时, 告警

class CUSUMDetector {
public:
    struct Config {
        double k_factor = 0.5;  // 允许偏移量 (sigma的倍数)
        double h_factor = 4.0;  // 决策阈值 (sigma的倍数)
        size_t min_samples = 5; // 最少需要的历史样本
    };

    struct Result {
        bool change_detected;
        double cusum_pos;       // 正向累积和(变慢)
        double cusum_neg;       // 负向累积和(变快)
        double threshold;       // h值
        int change_point;       // 变点位置 (-1=无)
        std::string direction;  // "slower" / "faster" / "none"

        void dump() const {
            fprintf(stderr,
                "    CUSUM: S+=%7.3f S-=%7.3f h=%7.3f change=%s dir=%s "
                "point=%d\n",
                cusum_pos, cusum_neg, threshold,
                change_detected ? "YES" : "no",
                direction.c_str(), change_point);
        }
    };

    static Result detect(const std::vector<double>& samples, const Config& cfg) {
        Result r;
        r.change_detected = false;
        r.cusum_pos = 0;
        r.cusum_neg = 0;
        r.change_point = -1;
        r.direction = "none";

        if (samples.size() < cfg.min_samples) return r;

        // 用前half作为参考期计算mu和sigma
        size_t ref_size = samples.size() / 2;
        if (ref_size < 3) ref_size = 3;

        double mu = 0;
        for (size_t i = 0; i < ref_size; ++i) mu += samples[i];
        mu /= ref_size;

        double var = 0;
        for (size_t i = 0; i < ref_size; ++i) {
            double d = samples[i] - mu;
            var += d * d;
        }
        double sigma = std::sqrt(var / (ref_size - 1));
        if (sigma < 1e-12) sigma = 1e-12;

        double k = cfg.k_factor * sigma;
        double h = cfg.h_factor * sigma;
        r.threshold = h;

        // 从参考期之后开始检测
        double s_pos = 0, s_neg = 0;
        for (size_t i = ref_size; i < samples.size(); ++i) {
            double x = samples[i];
            s_pos = std::max(0.0, s_pos + (x - mu - k));
            s_neg = std::max(0.0, s_neg - (x - mu + k));

            if (s_pos > h && !r.change_detected) {
                r.change_detected = true;
                r.change_point = static_cast<int>(i);
                r.direction = "slower";
            }
            if (s_neg > h && !r.change_detected) {
                r.change_detected = true;
                r.change_point = static_cast<int>(i);
                r.direction = "faster";
            }
        }

        r.cusum_pos = s_pos;
        r.cusum_neg = s_neg;
        return r;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// [NEW] 移动平均平滑器
// ═══════════════════════════════════════════════════════════════════════════

class MovingAverageSmoother {
public:
    // Simple Moving Average with window
    static std::vector<double> smooth(const std::vector<double>& data,
                                      size_t window = 5) {
        if (data.size() <= window) return data;
        std::vector<double> result;
        result.reserve(data.size() - window + 1);

        double sum = 0;
        for (size_t i = 0; i < window; ++i) sum += data[i];
        result.push_back(sum / window);

        for (size_t i = window; i < data.size(); ++i) {
            sum += data[i] - data[i - window];
            result.push_back(sum / window);
        }
        return result;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// RegressionDetector — 主检测器
// ═══════════════════════════════════════════════════════════════════════════

class RegressionDetector {
public:
    struct Config {
        double info_threshold = 3.0;      // >3% = INFO
        double warning_threshold = 5.0;   // >5% = WARNING (原upstream的5%固定阈值)
        double critical_threshold = 10.0; // >10% = CRITICAL
        size_t measure_runs = 7;
        bool use_cusum = true;
        bool use_smoothing = true;
        size_t smoothing_window = 5;
        CUSUMDetector::Config cusum_cfg;
        std::string history_path;         // 历史数据文件
    };

    struct Detection {
        std::string test_name;
        AlertLevel level;
        double regression_pct;
        double current_mean;
        double baseline_mean;
        bool welch_significant;
        CUSUMDetector::Result cusum_result;
        std::string commit_hash;

        std::string to_json() const {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "{"
                << "\"test\":\"" << test_name << "\","
                << "\"alert\":\"" << alert_name(level) << "\","
                << "\"regression_pct\":" << regression_pct << ","
                << "\"current_mean\":" << current_mean << ","
                << "\"baseline_mean\":" << baseline_mean << ","
                << "\"welch_significant\":" << (welch_significant ? "true" : "false") << ","
                << "\"cusum_change\":" << (cusum_result.change_detected ? "true" : "false") << ","
                << "\"commit\":\"" << commit_hash << "\""
                << "}";
            return oss.str();
        }
    };

    // ─── 主入口: 检测单个测试的回归 ─────────────────────────────────────
    static Detection detect(
            const std::string& test_name,
            const std::vector<double>& current_samples,
            const std::vector<double>& baseline_samples,
            const std::string& commit_hash,
            const Config& cfg) {
        Detection d;
        d.test_name = test_name;
        d.commit_hash = commit_hash;

        SampleStats current = SampleStats::compute(current_samples);
        SampleStats baseline = SampleStats::compute(baseline_samples);

        d.current_mean = current.mean;
        d.baseline_mean = baseline.mean;

        PHILE_REG_BREAKPOINT("DETECT",
            "%s: current=%.3f±%.3f baseline=%.3f±%.3f",
            test_name.c_str(), current.mean, current.stddev,
            baseline.mean, baseline.stddev);

        // 计算回归百分比
        if (baseline.mean > 0) {
            d.regression_pct = (current.mean - baseline.mean) /
                               baseline.mean * 100.0;
        } else {
            d.regression_pct = 0;
        }

        // [ALG] Welch t-test
        WelchResult welch = WelchResult::test(current, baseline);
        d.welch_significant = welch.significant_5pct;

        PHILE_REG_BREAKPOINT("WELCH",
            "t=%.3f df=%.1f significant=%d regression=%.1f%%",
            welch.t_stat, welch.df, welch.significant_5pct,
            d.regression_pct);

        // [ALG] CUSUM变点检测 (合并历史+当前)
        if (cfg.use_cusum) {
            std::vector<double> all_samples;
            all_samples.insert(all_samples.end(),
                               baseline_samples.begin(),
                               baseline_samples.end());
            all_samples.insert(all_samples.end(),
                               current_samples.begin(),
                               current_samples.end());

            // 可选平滑
            auto detect_samples = all_samples;
            if (cfg.use_smoothing && all_samples.size() > cfg.smoothing_window) {
                detect_samples = MovingAverageSmoother::smooth(
                    all_samples, cfg.smoothing_window);
            }

            d.cusum_result = CUSUMDetector::detect(detect_samples, cfg.cusum_cfg);

            PHILE_REG_BREAKPOINT("CUSUM", "change=%d dir=%s point=%d",
                d.cusum_result.change_detected,
                d.cusum_result.direction.c_str(),
                d.cusum_result.change_point);
        }

        // [ALG] 分级告警
        double abs_reg = std::abs(d.regression_pct);
        if (abs_reg >= cfg.critical_threshold && d.welch_significant) {
            d.level = AlertLevel::CRITICAL;
        } else if (abs_reg >= cfg.warning_threshold && d.welch_significant) {
            d.level = AlertLevel::WARNING;
        } else if (abs_reg >= cfg.info_threshold) {
            d.level = AlertLevel::INFO;
        } else if (d.cusum_result.change_detected &&
                   d.cusum_result.direction == "slower") {
            // CUSUM检测到渐进退化, 即使单次t-test不显著
            d.level = AlertLevel::WARNING;
        } else {
            d.level = AlertLevel::OK;
        }

        // 打印检测结果
        fprintf(stderr, "  %s[%s] %s: %+.1f%% (%.3f→%.3f ms)%s\n",
                alert_color(d.level), alert_name(d.level),
                test_name.c_str(), d.regression_pct,
                d.baseline_mean, d.current_mean,
                "\x1b[0m");

        if (d.cusum_result.change_detected) {
            d.cusum_result.dump();
        }

        return d;
    }

    // ─── 批量检测 ───────────────────────────────────────────────────────
    static std::vector<Detection> detect_all(
            const std::map<std::string, std::vector<double>>& current,
            const std::map<std::string, std::vector<double>>& baseline,
            const std::string& commit_hash,
            const Config& cfg) {
        fprintf(stderr,
            "\x1b[36m╔═══════════════════════════════════════════════╗\n"
            "║      REGRESSION DETECTION                     ║\n"
            "║  commit: %.40s  ║\n"
            "║  tests: %-38zu ║\n"
            "╚═══════════════════════════════════════════════╝\x1b[0m\n",
            commit_hash.c_str(), current.size());

        std::vector<Detection> results;

        for (const auto& [name, samples] : current) {
            auto it = baseline.find(name);
            if (it == baseline.end()) {
                PHILE_REG_BREAKPOINT("SKIP", "%s: no baseline",
                                     name.c_str());
                continue;
            }
            results.push_back(
                detect(name, samples, it->second, commit_hash, cfg));
        }

        // 汇总
        size_t ok = 0, info = 0, warn = 0, crit = 0;
        for (const auto& d : results) {
            switch (d.level) {
                case AlertLevel::OK: ok++; break;
                case AlertLevel::INFO: info++; break;
                case AlertLevel::WARNING: warn++; break;
                case AlertLevel::CRITICAL: crit++; break;
            }
        }

        fprintf(stderr,
            "\x1b[36m  Summary: %zu OK, %zu INFO, %zu WARNING, %zu CRITICAL\x1b[0m\n",
            ok, info, warn, crit);

        return results;
    }

    // ─── 历史数据管理 ───────────────────────────────────────────────────

    static void save_history(const std::vector<HistoryPoint>& points,
                             const std::string& path) {
        std::ofstream fout(path, std::ios::app);
        for (const auto& p : points) {
            fout << p.commit_hash << "\t"
                 << p.timestamp << "\t"
                 << p.test_name << "\t"
                 << std::fixed << std::setprecision(4)
                 << p.latency_ms << "\t"
                 << p.throughput_eps << "\t"
                 << p.memory_mb << "\n";
        }
    }

    static std::vector<HistoryPoint> load_history(
            const std::string& path,
            const std::string& test_name_filter = "",
            size_t max_entries = 100) {
        std::vector<HistoryPoint> points;
        std::ifstream fin(path);
        if (!fin.is_open()) return points;

        std::string line;
        while (std::getline(fin, line)) {
            HistoryPoint p;
            std::istringstream iss(line);
            if (!(iss >> p.commit_hash >> p.timestamp >> p.test_name
                      >> p.latency_ms >> p.throughput_eps >> p.memory_mb))
                continue;

            if (!test_name_filter.empty() &&
                p.test_name != test_name_filter)
                continue;

            points.push_back(p);
        }

        // 只保留最近的max_entries条
        if (points.size() > max_entries) {
            points.erase(points.begin(),
                         points.end() - max_entries);
        }
        return points;
    }

    // ─── [NEW] 趋势分析 ────────────────────────────────────────────────
    static void print_trend(const std::vector<HistoryPoint>& history,
                            const std::string& test_name) {
        if (history.size() < 5) {
            fprintf(stderr, "  (insufficient history for trend: %zu points)\n",
                    history.size());
            return;
        }

        std::vector<double> latencies;
        for (const auto& p : history) {
            if (p.test_name == test_name) {
                latencies.push_back(p.latency_ms);
            }
        }

        if (latencies.size() < 5) return;

        // 最近5个点的SMA
        auto smoothed = MovingAverageSmoother::smooth(latencies, 5);

        fprintf(stderr, "\x1b[34m  [TREND: %s] last %zu points:\n    ",
                test_name.c_str(), latencies.size());
        size_t start = (latencies.size() > 10) ? latencies.size() - 10 : 0;
        for (size_t i = start; i < latencies.size(); ++i) {
            fprintf(stderr, "%.2f ", latencies[i]);
        }
        fprintf(stderr, "\n    SMA: ");
        start = (smoothed.size() > 8) ? smoothed.size() - 8 : 0;
        for (size_t i = start; i < smoothed.size(); ++i) {
            fprintf(stderr, "%.2f ", smoothed[i]);
        }
        fprintf(stderr, "\x1b[0m\n");

        // 简单线性回归看趋势方向
        double x_mean = (latencies.size() - 1) / 2.0;
        double y_mean = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                        / latencies.size();
        double num = 0, den = 0;
        for (size_t i = 0; i < latencies.size(); ++i) {
            double xi = static_cast<double>(i) - x_mean;
            double yi = latencies[i] - y_mean;
            num += xi * yi;
            den += xi * xi;
        }
        double slope = (den > 0) ? num / den : 0;

        fprintf(stderr, "    trend slope: %.4f ms/commit %s\n",
                slope, slope > 0.01 ? "(degrading)" :
                       slope < -0.01 ? "(improving)" : "(stable)");
    }
};

} // namespace bench
} // namespace philemon

#endif // PHILEMON_REGRESSION_DETECTOR_HPP
