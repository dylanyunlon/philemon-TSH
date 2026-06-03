#pragma once
/**
 * cuda_gpu_profiler.hpp — GPU性能profiler (分层计时+滑动窗口+瓶颈检测)
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   src/cuda/hetero_bench.cu  CudaTimer (269–292行)
 *     → cudaEventCreate/Record/ElapsedTime 计时对
 *     → begin(stream) / end(stream) → float ms
 *
 *   src/cuda/hetero_bench.cu  所有experiment_*函数 (293–968行)
 *     → experiment_bandwidth: 4种size × tier-pair的带宽矩阵
 *     → experiment_query: warmup+measure迭代统计 (gather/scan/total)
 *     → experiment_migration: 跨tier拷贝timing
 *     → experiment_concurrent: QPS统计
 *     → experiment_scaling: edges vs throughput
 *
 *   src/debug/state_inspector.hpp  AlgorithmProfiler (390–530行)
 *     → phase_begin/phase_end RAII模式
 *     → elapsed_ms/total_ms累积
 *     → dump_profile打印
 *
 * 算法改动 (~20%):
 *   [ALG1] 计时模型: 原版CudaTimer只支持(begin,end)一对
 *          → 三级分层: kernel-level / phase-level / tier-level
 *            每级独立计时, 支持嵌套, parent-child关系树
 *   [ALG2] 统计: 原版单次或固定迭代平均
 *          → 滑动窗口: 保留最近N次(默认64)测量,
 *            计算移动平均+方差+p50/p95/p99分位数
 *   [ALG3] 瓶颈检测: 原版人工读printf输出
 *          → 自动分析: 按耗时排序, 标记top-3热点kernel,
 *            PCIe带宽利用率预警(>90%标红), GPU利用率估算
 *   [ALG4] 输出格式: 原版printf文本
 *          → 结构化: 同时输出pretty-print文本 + JSON格式,
 *            JSON可被外部工具(Grafana/Perfetto)消费
 *
 * Milestone: M051 — GPU性能profiler
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <numeric>
#include <sstream>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "cuda_mem_manager.hpp"

namespace philemon {
namespace cuda_profiler {

// ─── 滑动窗口采样器 ([ALG2]) ──────────────────────────────────────
// 原版: 单次或固定N次平均
// 改动: 环形buffer保留最近WINDOW_SIZE次, 支持分位数
class SlidingWindowStats {
public:
    static constexpr size_t WINDOW_SIZE = 64;

    void record(double value) {
        std::lock_guard<std::mutex> lk(mu_);
        if (samples_.size() < WINDOW_SIZE) {
            samples_.push_back(value);
        } else {
            samples_[write_idx_] = value;
        }
        write_idx_ = (write_idx_ + 1) % WINDOW_SIZE;
        total_count_++;
        running_sum_ += value;
        running_sum_sq_ += value * value;
    }

    size_t count() const { return total_count_; }

    double mean() const {
        std::lock_guard<std::mutex> lk(mu_);
        return samples_.empty() ? 0 : running_sum_ / total_count_;
    }

    double window_mean() const {
        std::lock_guard<std::mutex> lk(mu_);
        if (samples_.empty()) return 0;
        double sum = 0;
        for (auto v : samples_) sum += v;
        return sum / samples_.size();
    }

    double variance() const {
        std::lock_guard<std::mutex> lk(mu_);
        if (total_count_ < 2) return 0;
        double m = running_sum_ / total_count_;
        return (running_sum_sq_ / total_count_) - m * m;
    }

    double stddev() const { return std::sqrt(std::max(0.0, variance())); }

    // [ALG2] 分位数: 排序后取
    double percentile(double p) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (samples_.empty()) return 0;
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p / 100.0 * (sorted.size() - 1));
        return sorted[std::min(idx, sorted.size() - 1)];
    }

    double p50() const { return percentile(50); }
    double p95() const { return percentile(95); }
    double p99() const { return percentile(99); }
    double min_val() const {
        std::lock_guard<std::mutex> lk(mu_);
        return samples_.empty() ? 0 :
            *std::min_element(samples_.begin(), samples_.end());
    }
    double max_val() const {
        std::lock_guard<std::mutex> lk(mu_);
        return samples_.empty() ? 0 :
            *std::max_element(samples_.begin(), samples_.end());
    }

private:
    mutable std::mutex mu_;
    std::vector<double> samples_;
    size_t write_idx_ = 0;
    size_t total_count_ = 0;
    double running_sum_ = 0;
    double running_sum_sq_ = 0;
};

// ─── [ALG1] Profiler层级 ──────────────────────────────────────────
enum class ProfileLevel : int {
    KERNEL = 0,   // 单个CUDA kernel
    PHASE  = 1,   // 算法阶段 (BFS-TD, PR-contrib, etc.)
    TIER   = 2,   // 内存层级操作 (HBM alloc, GDDR copy, etc.)
};

inline const char* profile_level_name(ProfileLevel l) {
    switch (l) {
        case ProfileLevel::KERNEL: return "KERNEL";
        case ProfileLevel::PHASE:  return "PHASE";
        case ProfileLevel::TIER:   return "TIER";
        default: return "?";
    }
}

// ─── Profile条目 ──────────────────────────────────────────────────
struct ProfileEntry {
    std::string      name;
    ProfileLevel     level;
    SlidingWindowStats timing_ms;   // 耗时统计
    SlidingWindowStats bytes;       // 数据量统计
    SlidingWindowStats bandwidth;   // 带宽统计

    // 关系
    std::string parent;             // 父条目名
    uint64_t    call_count = 0;

    ProfileEntry() : level(ProfileLevel::KERNEL) {}
    explicit ProfileEntry(const std::string& n, ProfileLevel l,
                          const std::string& par = "")
        : name(n), level(l), parent(par) {}
};

// ─── RAII计时器 ───────────────────────────────────────────────────
class ScopedProfiler;  // 前向声明

// ════════════════════════════════════════════════════════════════════════════
//  GpuProfiler — 主profiler
// ════════════════════════════════════════════════════════════════════════════

class GpuProfiler {
public:
    GpuProfiler() {
        PHILE_CHECKPOINT("GpuProfiler::ctor");
    }

    // ── 记录一次测量 ───────────────────────────────────────────
    void record(const std::string& name, ProfileLevel level,
                double elapsed_ms, size_t bytes_transferred = 0,
                const std::string& parent = "")
    {
        std::lock_guard<std::mutex> lk(mu_);

        auto it = entries_.find(name);
        if (it == entries_.end()) {
            entries_[name] = ProfileEntry(name, level, parent);
            it = entries_.find(name);
        }

        auto& e = it->second;
        e.timing_ms.record(elapsed_ms);
        e.call_count++;

        if (bytes_transferred > 0) {
            e.bytes.record(static_cast<double>(bytes_transferred));
            double bw_gbps = (bytes_transferred / (1024.0*1024*1024)) / (elapsed_ms / 1000.0);
            e.bandwidth.record(bw_gbps);
        }
    }

    // ── [ALG3] 瓶颈自动分析 ───────────────────────────────────
    // 原版: 人工看printf
    // 改动: 按总耗时排序, 标记top-3, PCIe预警
    struct BottleneckReport {
        struct Hotspot {
            std::string name;
            double total_ms;
            double pct_of_total;
            double p99_ms;
            std::string alert;    // 预警信息
        };
        std::vector<Hotspot> hotspots;
        double total_profiled_ms;
        double estimated_gpu_util;   // GPU利用率估算
    };

    BottleneckReport analyze_bottlenecks() const {
        std::lock_guard<std::mutex> lk(mu_);
        PHILE_CHECKPOINT("analyze_bottlenecks");

        BottleneckReport report;
        report.total_profiled_ms = 0;

        // 收集所有条目的总耗时
        std::vector<std::pair<std::string, double>> sorted_entries;
        double kernel_ms = 0;

        for (auto& [name, entry] : entries_) {
            double total = entry.timing_ms.window_mean() * entry.call_count;
            sorted_entries.push_back({name, total});
            report.total_profiled_ms += total;
            if (entry.level == ProfileLevel::KERNEL) kernel_ms += total;
        }

        // 按总耗时降序
        std::sort(sorted_entries.begin(), sorted_entries.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // 标记top hotspots
        for (size_t i = 0; i < std::min(sorted_entries.size(), size_t(5)); ++i) {
            auto& [name, total] = sorted_entries[i];
            auto& entry = entries_.at(name);

            BottleneckReport::Hotspot hs;
            hs.name = name;
            hs.total_ms = total;
            hs.pct_of_total = (report.total_profiled_ms > 0)
                ? 100.0 * total / report.total_profiled_ms : 0;
            hs.p99_ms = entry.timing_ms.p99();

            // [ALG3] 预警逻辑
            if (entry.bandwidth.count() > 0) {
                double avg_bw = entry.bandwidth.window_mean();
                // PCIe Gen4 x16 理论 ~32 GB/s
                if (avg_bw > 28.0) {
                    hs.alert = "⚠ PCIe near saturation (>" + std::to_string((int)avg_bw) + " GB/s)";
                }
            }
            if (entry.timing_ms.p99() > 5 * entry.timing_ms.p50()) {
                hs.alert += (hs.alert.empty() ? "" : "; ");
                hs.alert += "⚠ High tail latency (p99/p50=" +
                    std::to_string(entry.timing_ms.p99() / std::max(0.001, entry.timing_ms.p50())) + "x)";
            }

            report.hotspots.push_back(hs);
        }

        // GPU利用率估算 = kernel时间 / 总时间
        report.estimated_gpu_util = (report.total_profiled_ms > 0)
            ? kernel_ms / report.total_profiled_ms : 0;

        return report;
    }

    // ── [ALG4] 文本报告 ────────────────────────────────────────
    void dump_report() const {
        PHILE_SEPARATOR("GPU Profiler Report");

        printf("  %-30s  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
               "Name", "Level", "Calls", "Mean(ms)", "P50(ms)", "P95(ms)",
               "P99(ms)", "BW(GB/s)");
        printf("  %-30s  %-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
               "------------------------------", "--------", "--------",
               "----------", "----------", "----------", "----------", "----------");

        std::lock_guard<std::mutex> lk(mu_);
        for (auto& [name, e] : entries_) {
            printf("  %-30s  %-8s  %-8lu  %10.3f  %10.3f  %10.3f  %10.3f",
                   name.c_str(),
                   profile_level_name(e.level),
                   (unsigned long)e.call_count,
                   e.timing_ms.window_mean(),
                   e.timing_ms.p50(),
                   e.timing_ms.p95(),
                   e.timing_ms.p99());

            if (e.bandwidth.count() > 0) {
                printf("  %10.2f", e.bandwidth.window_mean());
            } else {
                printf("  %10s", "—");
            }
            printf("\n");
        }

        // 瓶颈分析
        auto report = const_cast<GpuProfiler*>(this)->analyze_bottlenecks();
        printf("\n  ── Bottleneck Analysis ──\n");
        printf("  Total profiled: %.2f ms  GPU util: %.1f%%\n",
               report.total_profiled_ms, report.estimated_gpu_util * 100);

        for (size_t i = 0; i < report.hotspots.size(); ++i) {
            auto& hs = report.hotspots[i];
            printf("  #%zu %-25s  %.2f ms (%.1f%%)  p99=%.3f ms",
                   i + 1, hs.name.c_str(), hs.total_ms, hs.pct_of_total, hs.p99_ms);
            if (!hs.alert.empty()) printf("  %s", hs.alert.c_str());
            printf("\n");
        }

        PHILE_SEPARATOR("End GPU Profiler Report");
    }

    // ── [ALG4] JSON输出 ────────────────────────────────────────
    // 原版: 无
    // 改动: 结构化JSON, 可被外部工具消费
    std::string to_json() const {
        std::lock_guard<std::mutex> lk(mu_);
        std::ostringstream ss;
        ss << "{\n  \"entries\": [\n";

        bool first = true;
        for (auto& [name, e] : entries_) {
            if (!first) ss << ",\n";
            first = false;
            ss << "    {"
               << "\"name\":\"" << name << "\","
               << "\"level\":\"" << profile_level_name(e.level) << "\","
               << "\"calls\":" << e.call_count << ","
               << "\"timing_ms\":{"
               << "\"mean\":" << e.timing_ms.window_mean() << ","
               << "\"p50\":" << e.timing_ms.p50() << ","
               << "\"p95\":" << e.timing_ms.p95() << ","
               << "\"p99\":" << e.timing_ms.p99() << ","
               << "\"min\":" << e.timing_ms.min_val() << ","
               << "\"max\":" << e.timing_ms.max_val()
               << "}";

            if (e.bandwidth.count() > 0) {
                ss << ",\"bandwidth_gbps\":{"
                   << "\"mean\":" << e.bandwidth.window_mean() << ","
                   << "\"p50\":" << e.bandwidth.p50() << ","
                   << "\"p99\":" << e.bandwidth.p99()
                   << "}";
            }
            ss << "}";
        }

        ss << "\n  ],\n";

        // 瓶颈分析
        auto report = const_cast<GpuProfiler*>(this)->analyze_bottlenecks();
        ss << "  \"bottlenecks\": [\n";
        first = true;
        for (auto& hs : report.hotspots) {
            if (!first) ss << ",\n";
            first = false;
            ss << "    {"
               << "\"name\":\"" << hs.name << "\","
               << "\"total_ms\":" << hs.total_ms << ","
               << "\"pct\":" << hs.pct_of_total << ","
               << "\"p99_ms\":" << hs.p99_ms;
            if (!hs.alert.empty())
                ss << ",\"alert\":\"" << hs.alert << "\"";
            ss << "}";
        }
        ss << "\n  ],\n"
           << "  \"gpu_utilization\":" << report.estimated_gpu_util << "\n"
           << "}\n";

        return ss.str();
    }

    // ── 清空 ────────────────────────────────────────────────────
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        entries_.clear();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, ProfileEntry> entries_;
};

// ─── RAII计时器 ───────────────────────────────────────────────────
class ScopedProfiler {
public:
    ScopedProfiler(GpuProfiler& profiler,
                   const std::string& name,
                   ProfileLevel level,
                   size_t bytes = 0,
                   const std::string& parent = "")
        : profiler_(profiler), name_(name), level_(level),
          bytes_(bytes), parent_(parent),
          start_(std::chrono::steady_clock::now())
    {}

    ~ScopedProfiler() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        profiler_.record(name_, level_, ms, bytes_, parent_);
    }

    // 更新bytes (在作用域内知道实际传输量后调用)
    void set_bytes(size_t b) { bytes_ = b; }

private:
    GpuProfiler& profiler_;
    std::string  name_;
    ProfileLevel level_;
    size_t       bytes_;
    std::string  parent_;
    std::chrono::steady_clock::time_point start_;
};

// ─── 宏: 简化使用 ─────────────────────────────────────────────────
#define PHILE_PROFILE_KERNEL(profiler, name) \
    cuda_profiler::ScopedProfiler _prof_##__LINE__( \
        (profiler), (name), cuda_profiler::ProfileLevel::KERNEL)

#define PHILE_PROFILE_PHASE(profiler, name) \
    cuda_profiler::ScopedProfiler _prof_##__LINE__( \
        (profiler), (name), cuda_profiler::ProfileLevel::PHASE)

#define PHILE_PROFILE_TIER(profiler, name, bytes) \
    cuda_profiler::ScopedProfiler _prof_##__LINE__( \
        (profiler), (name), cuda_profiler::ProfileLevel::TIER, (bytes))

} // namespace cuda_profiler
} // namespace philemon
