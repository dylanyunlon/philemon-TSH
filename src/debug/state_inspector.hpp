#ifndef PHILEMON_STATE_INSPECTOR_HPP
#define PHILEMON_STATE_INSPECTOR_HPP
/**
 * state_inspector.hpp — Runtime State Inspector for Breakpoint-Style Debugging
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *   upstream/rapidstore/utils/log/log.h                               (68行)
 *   upstream/rapidstore/utils/log/log.cpp                             (157行)
 *
 * 修改 (~20%):
 *   - [NEW] BreakpointGuard: RAII 对象, 在作用域入口/出口自动打印状态
 *   - [NEW] DataDumper: 结构化打印任何容器的前N个元素
 *   - [NEW] TierHeatmap: 打印每个 tier 的访问热度
 *   - [NEW] ConvergenceTracker: 收集+序列化迭代算法的收敛数据
 *   - [NEW] AlgorithmProfiler: 统计每个算法阶段的耗时
 *   - [MOD] 原 neo_reader_trace 的 trace_entry → 扩展为 InspectionPoint
 *   - [MOD] 原 log 模块 → 集成到 PHILE_INSPECT 宏
 *   - [KEEP] ring buffer trace 结构 100% 保留
 *   - [KEEP] 原 log 级别概念 100% 保留
 *
 * 使用方法 (像现实开发断点调试):
 *
 *   // 方法1: RAII breakpoint — 自动打印函数入口/出口
 *   void my_function() {
 *       PHILE_BREAKPOINT("my_function");
 *       // ... 函数体 ...
 *   }  // 出作用域时自动打印耗时+状态
 *
 *   // 方法2: 打印容器内容
 *   std::vector<double> scores(1000);
 *   PHILE_DUMP_VEC("scores", scores, 20);  // 打印前20个
 *
 *   // 方法3: 打印 tier 热度图
 *   PHILE_TIER_HEATMAP();
 *
 *   // 方法4: 手动 inspection point
 *   PHILE_INSPECT("before_phase2", "iter=%d N=%lu", iter, N);
 *
 * Milestone: M021-M022 debug enhancement
 * ====================================================================
 */

#include <cstdio>
#include <cstdint>
#include <cstdarg>
#include <chrono>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <iomanip>

#include "philemon_debug.hpp"  // base debug module

namespace philemon {
namespace debug {

// ═══════════════════════════════════════════════════════════════════════
// InspectionPoint — from upstream neo_reader_trace.h TraceEntry (extended)
//
// Original: event + timestamp + 2 uint64 fields
// Extended: + phase name + formatted message + tier perf snapshot
// ═══════════════════════════════════════════════════════════════════════
struct InspectionPoint {
    uint64_t timestamp_ns;
    std::string phase;
    std::string message;
    // Tier perf snapshot at this point
    uint64_t tier_reads[3];
    uint64_t tier_writes[3];
    uint64_t tier_bytes[3];

    static InspectionPoint capture(const std::string& phase,
                                    const std::string& msg) {
        InspectionPoint ip;
        ip.timestamp_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch()).count();
        ip.phase = phase;
        ip.message = msg;
        for (int t = 0; t < 3; t++) {
            ip.tier_reads[t]  = tier_perf(t).read_count.load();
            ip.tier_writes[t] = tier_perf(t).migrate_out_count.load();
            ip.tier_bytes[t]  = tier_perf(t).migrate_out_bytes.load();
        }
        return ip;
    }

    void print() const {
        double ts_ms = timestamp_ns / 1e6;
        std::printf("  📍 [%.3fms] %s: %s\n",
                    ts_ms, phase.c_str(), message.c_str());
        std::printf("     tier_io: R[%lu,%lu,%lu] W[%lu,%lu,%lu] "
                    "B[%lu,%lu,%lu]\n",
                    (unsigned long)tier_reads[0],
                    (unsigned long)tier_reads[1],
                    (unsigned long)tier_reads[2],
                    (unsigned long)tier_writes[0],
                    (unsigned long)tier_writes[1],
                    (unsigned long)tier_writes[2],
                    (unsigned long)tier_bytes[0],
                    (unsigned long)tier_bytes[1],
                    (unsigned long)tier_bytes[2]);
    }
};

// ─── Global inspection log ───────────────────────────────────────────
inline std::vector<InspectionPoint>& inspection_log() {
    static std::vector<InspectionPoint> log;
    return log;
}

inline std::mutex& inspection_mutex() {
    static std::mutex mtx;
    return mtx;
}

inline void record_inspection(const std::string& phase,
                               const std::string& msg) {
    if (get_debug_level() < 1) return;
    std::lock_guard<std::mutex> lock(inspection_mutex());
    inspection_log().push_back(InspectionPoint::capture(phase, msg));
}

inline void print_inspection_log() {
    std::lock_guard<std::mutex> lock(inspection_mutex());
    if (inspection_log().empty()) return;

    std::printf("──── Inspection Log (%zu points) ────\n",
                inspection_log().size());
    for (const auto& ip : inspection_log()) {
        ip.print();
    }
    std::printf("──── End Inspection Log ────\n");
}

inline void clear_inspection_log() {
    std::lock_guard<std::mutex> lock(inspection_mutex());
    inspection_log().clear();
}

// ═══════════════════════════════════════════════════════════════════════
// BreakpointGuard — RAII breakpoint, prints state at scope entry/exit
//
// Usage: PHILE_BREAKPOINT("my_function");
//        → 入口打印 "[ENTER my_function]"
//        → 出口打印 "[EXIT my_function] elapsed=X.XXms"
// ═══════════════════════════════════════════════════════════════════════
class BreakpointGuard {
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    int level_;

public:
    BreakpointGuard(const std::string& name, int level = 1)
        : name_(name), level_(level),
          start_(std::chrono::high_resolution_clock::now()) {
        if (get_debug_level() >= level_) {
            std::printf("  🔵 [ENTER %s]\n", name_.c_str());
            record_inspection("ENTER", name_);
        }
    }

    ~BreakpointGuard() {
        if (get_debug_level() >= level_) {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(
                end - start_).count();
            std::printf("  🟢 [EXIT  %s] elapsed=%.3fms\n",
                        name_.c_str(), ms);

            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s (%.3fms)", name_.c_str(), ms);
            record_inspection("EXIT", buf);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// DataDumper — structured printing of containers and arrays
//
// Prints first N elements of std::vector, unique_ptr<T[]>, raw arrays
// ═══════════════════════════════════════════════════════════════════════
class DataDumper {
public:
    // Dump std::vector<double>
    static void dump(const char* name, const std::vector<double>& v,
                     size_t max_show = 20) {
        if (get_debug_level() < 2) return;
        size_t n = std::min(v.size(), max_show);
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        double sum = 0;
        for (auto x : v) sum += x;

        std::printf("  📊 %s[%zu]: min=%.6f max=%.6f avg=%.6f\n",
                    name, v.size(), mn, mx, sum / v.size());
        std::printf("     [");
        for (size_t i = 0; i < n; i++) {
            std::printf("%.4f%s", v[i], i < n - 1 ? ", " : "");
        }
        if (v.size() > max_show) std::printf(", ...");
        std::printf("]\n");
    }

    // Dump std::vector<uint64_t>
    static void dump(const char* name, const std::vector<uint64_t>& v,
                     size_t max_show = 20) {
        if (get_debug_level() < 2) return;
        size_t n = std::min(v.size(), max_show);
        uint64_t mn = *std::min_element(v.begin(), v.end());
        uint64_t mx = *std::max_element(v.begin(), v.end());

        std::printf("  📊 %s[%zu]: min=%lu max=%lu\n",
                    name, v.size(),
                    (unsigned long)mn, (unsigned long)mx);
        std::printf("     [");
        for (size_t i = 0; i < n; i++) {
            std::printf("%lu%s", (unsigned long)v[i],
                        i < n - 1 ? ", " : "");
        }
        if (v.size() > max_show) std::printf(", ...");
        std::printf("]\n");
    }

    // Dump raw array
    template <typename T>
    static void dump_array(const char* name, const T* arr, size_t size,
                           size_t max_show = 20) {
        if (get_debug_level() < 2) return;
        size_t n = std::min(size, max_show);
        std::printf("  📊 %s[%zu]: ", name, size);
        std::printf("[");
        for (size_t i = 0; i < n; i++) {
            // Use a string stream for generic T formatting
            std::ostringstream oss;
            oss << arr[i];
            std::printf("%s%s", oss.str().c_str(),
                        i < n - 1 ? ", " : "");
        }
        if (size > max_show) std::printf(", ...");
        std::printf("]\n");
    }

    // Dump pair vector (for algorithm results)
    template <typename A, typename B>
    static void dump_pairs(const char* name,
                           const std::vector<std::pair<A, B>>& v,
                           size_t max_show = 10) {
        if (get_debug_level() < 2) return;
        size_t n = std::min(v.size(), max_show);
        std::printf("  📊 %s[%zu]:\n", name, v.size());
        for (size_t i = 0; i < n; i++) {
            std::ostringstream oss;
            oss << "     [" << i << "] first=" << v[i].first
                << " second=" << v[i].second;
            std::printf("%s\n", oss.str().c_str());
        }
        if (v.size() > max_show)
            std::printf("     ... (%zu more)\n", v.size() - max_show);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// TierHeatmap — visual ASCII heatmap of tier access patterns
// ═══════════════════════════════════════════════════════════════════════
class TierHeatmap {
public:
    static void print() {
        if (get_debug_level() < 1) return;

        const char* tier_names[] = {"HBM ", "GDDR", "DRAM"};
        const char* bars = "████████████████████████████████████████";

        // Find max for scaling
        uint64_t max_reads = 1;
        for (int t = 0; t < 3; t++) {
            max_reads = std::max(max_reads, tier_perf(t).read_count.load());
        }

        std::printf("  🔥 Tier Access Heatmap:\n");
        for (int t = 0; t < 3; t++) {
            auto& perf = tier_perf(t);
            uint64_t reads = perf.read_count.load();
            uint64_t writes = perf.migrate_out_count.load();
            uint64_t bytes = perf.migrate_out_bytes.load();
            int bar_len = (int)(40.0 * reads / max_reads);

            std::printf("    %s [", tier_names[t]);
            // Print bar_len characters from bars
            for (int i = 0; i < 40; i++) {
                std::printf(i < bar_len ? "█" : "░");
            }
            std::printf("] R=%-8lu W=%-8lu B=%-10lu\n",
                        (unsigned long)reads,
                        (unsigned long)writes,
                        (unsigned long)bytes);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// ConvergenceTracker — collect convergence data for paper figures
// ═══════════════════════════════════════════════════════════════════════
class ConvergenceTracker {
    struct DataPoint {
        int step;
        double metric;
        double time_ms;
    };

    std::string name_;
    std::vector<DataPoint> data_;

public:
    ConvergenceTracker(const std::string& name) : name_(name) {}

    void record(int step, double metric, double time_ms) {
        data_.push_back({step, metric, time_ms});
    }

    void print() const {
        if (data_.empty()) return;
        std::printf("  📈 Convergence [%s] (%zu points):\n",
                    name_.c_str(), data_.size());
        std::printf("    %-6s %-14s %-10s\n", "Step", "Metric", "Time(ms)");
        for (const auto& d : data_) {
            std::printf("    %-6d %-14.8f %-10.3f\n",
                        d.step, d.metric, d.time_ms);
        }
    }

    // Serialize to JSON-like string (for paper data)
    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\"name\":\"" << name_ << "\",\"data\":[";
        for (size_t i = 0; i < data_.size(); i++) {
            oss << "{\"step\":" << data_[i].step
                << ",\"metric\":" << std::setprecision(10) << data_[i].metric
                << ",\"time_ms\":" << std::setprecision(4) << data_[i].time_ms
                << "}";
            if (i < data_.size() - 1) oss << ",";
        }
        oss << "]}";
        return oss.str();
    }

    size_t size() const { return data_.size(); }
    void clear() { data_.clear(); }
};

// ═══════════════════════════════════════════════════════════════════════
// AlgorithmProfiler — per-phase timing for algorithms
//
// From upstream neo_reader_trace.h (trace entry structure, extended)
// ═══════════════════════════════════════════════════════════════════════
class AlgorithmProfiler {
    struct PhaseRecord {
        std::string name;
        double total_ms;
        uint64_t call_count;
    };

    std::string algo_name_;
    std::unordered_map<std::string, PhaseRecord> phases_;
    std::mutex mtx_;

public:
    AlgorithmProfiler(const std::string& name) : algo_name_(name) {}

    void record_phase(const std::string& phase, double ms) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto& rec = phases_[phase];
        rec.name = phase;
        rec.total_ms += ms;
        rec.call_count++;
    }

    void print() const {
        if (phases_.empty()) return;
        std::printf("  ⏱️  Algorithm Profile [%s]:\n", algo_name_.c_str());
        std::printf("    %-30s %-12s %-10s %-12s\n",
                    "Phase", "Total(ms)", "Calls", "Avg(ms)");
        for (const auto& [name, rec] : phases_) {
            std::printf("    %-30s %-12.3f %-10lu %-12.3f\n",
                        rec.name.c_str(), rec.total_ms,
                        (unsigned long)rec.call_count,
                        rec.call_count > 0 ?
                            rec.total_ms / rec.call_count : 0);
        }
    }

    void clear() { phases_.clear(); }
};

}  // namespace debug
}  // namespace philemon

// ═══════════════════════════════════════════════════════════════════════
// Convenience macros
// ═══════════════════════════════════════════════════════════════════════

// Breakpoint guard — prints entry/exit of scope
#define PHILE_BREAKPOINT(name) \
    philemon::debug::BreakpointGuard _bp_guard_##__LINE__(name)

// Inspection point — formatted message + tier snapshot
#define PHILE_INSPECT(phase, fmt, ...) do { \
    if (philemon::debug::get_debug_level() >= 1) { \
        char _buf[512]; \
        std::snprintf(_buf, sizeof(_buf), fmt, ##__VA_ARGS__); \
        philemon::debug::record_inspection(phase, _buf); \
        std::printf("  📍 [%s] %s\n", phase, _buf); \
    } \
} while(0)

// Dump vector
#define PHILE_DUMP_VEC(name, vec, n) \
    philemon::debug::DataDumper::dump(name, vec, n)

// Dump pair vector
#define PHILE_DUMP_PAIRS(name, vec, n) \
    philemon::debug::DataDumper::dump_pairs(name, vec, n)

// Dump raw array
#define PHILE_DUMP_ARRAY(name, arr, size, n) \
    philemon::debug::DataDumper::dump_array(name, arr, size, n)

// Tier heatmap
#define PHILE_TIER_HEATMAP() \
    philemon::debug::TierHeatmap::print()

// Print full inspection log
#define PHILE_PRINT_INSPECTIONS() \
    philemon::debug::print_inspection_log()

#endif  // PHILEMON_STATE_INSPECTOR_HPP
