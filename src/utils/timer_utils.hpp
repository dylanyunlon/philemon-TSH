#ifndef PHILEMON_TIMER_UTILS_HPP
#define PHILEMON_TIMER_UTILS_HPP
/**
 * timer_utils.hpp — 计时器 + 错误类型 + 日志工具
 *
 * 骨架来源:
 *   upstream/rapidstore/utils/Timer.h           (30行, 100% elapsed逻辑保留)
 *   upstream/rapidstore/utils/error_type.hpp    (30行, 异常类型)
 *   upstream/rapidstore/utils/error_type.cpp    (20行)
 *   upstream/rapidstore/utils/log/log.h         (68行, log_info/log_warn)
 *   upstream/rapidstore/utils/log/log.cpp       (157行, log_set_fp等)
 *
 * 修改 (~20%):
 *   - [MOD] 裸Timer类 → philemon::utils::Timer, 加namespace
 *   - [NEW] ScopedTimer: RAII自动打印构造→析构的耗时
 *   - [NEW] PhaseTimer: 多阶段耗时累积, dump_phases()打印全部阶段
 *   - [NEW] PHILE_TIME_SCOPE(name): 一行加计时断点
 *   - [MOD] log模块: log_info/log_warn → 内联实现, 不依赖全局FILE*
 *   - [KEEP] Timer::elapsed() 精度和接口100%保留
 *   - [KEEP] error_type枚举100%保留
 *
 * Milestone: M027+
 */

#include <chrono>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <vector>
#include <utility>
#include <mutex>
#include <atomic>
#include <cstring>

namespace philemon {
namespace utils {

// ─── Timer (upstream core 100% preserved) ───────────────────────────
class Timer {
public:
    Timer() : beg_(clock_::now()) {}

    void reset() { beg_ = clock_::now(); }

    double elapsed() const {
        return std::chrono::duration_cast<second_>(
            clock_::now() - beg_).count();
    }

    double elapsed_and_reset() {
        double e = std::chrono::duration_cast<second_>(
            clock_::now() - beg_).count();
        beg_ = clock_::now();
        return e;
    }

    // NEW: elapsed in microseconds (for fine-grained profiling)
    double elapsed_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            clock_::now() - beg_).count();
    }

private:
    using clock_  = std::chrono::high_resolution_clock;
    using second_ = std::chrono::duration<double, std::ratio<1>>;
    std::chrono::time_point<clock_> beg_;
};

// ─── ScopedTimer (NEW: RAII 自动打印耗时) ───────────────────────────
class ScopedTimer {
public:
    explicit ScopedTimer(const char* label)
        : label_(label), start_(std::chrono::high_resolution_clock::now()) {
        std::fprintf(stderr, "[TIMER] >>> %s begin\n", label_);
    }

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start_).count() / 1000.0;
        std::fprintf(stderr, "[TIMER] <<< %s end (%.3f ms)\n", label_, ms);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char* label_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// Macro: 一行加计时断点
#define PHILE_TIME_SCOPE(name) \
    ::philemon::utils::ScopedTimer _phile_timer_##__LINE__(name)

// ─── PhaseTimer (NEW: 多阶段累积耗时) ──────────────────────────────
class PhaseTimer {
public:
    void start_phase(const std::string& name) {
        current_phase_ = name;
        phase_start_ = std::chrono::high_resolution_clock::now();
    }

    void end_phase() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(
            end - phase_start_).count() / 1000.0;
        phases_.emplace_back(current_phase_, ms);
        std::fprintf(stderr, "[PHASE] %s: %.3f ms\n",
                     current_phase_.c_str(), ms);
    }

    void dump_phases(const char* header = "Phase Summary") const {
        std::fprintf(stderr, "\n╔══════════════════════════════════════╗\n");
        std::fprintf(stderr, "║ %-36s ║\n", header);
        std::fprintf(stderr, "╠══════════════════════════════════════╣\n");
        double total = 0;
        for (auto& [name, ms] : phases_) {
            std::fprintf(stderr, "║ %-24s %10.3f ms ║\n", name.c_str(), ms);
            total += ms;
        }
        std::fprintf(stderr, "╠══════════════════════════════════════╣\n");
        std::fprintf(stderr, "║ %-24s %10.3f ms ║\n", "TOTAL", total);
        std::fprintf(stderr, "╚══════════════════════════════════════╝\n\n");
    }

    void clear() { phases_.clear(); }

private:
    std::string current_phase_;
    std::chrono::time_point<std::chrono::high_resolution_clock> phase_start_;
    std::vector<std::pair<std::string, double>> phases_;
};

// ─── Error types (upstream 100%) ────────────────────────────────────
enum class ErrorType {
    OK = 0,
    NOT_FOUND,
    DUPLICATE_KEY,
    INVALID_ARGUMENT,
    OUT_OF_MEMORY,
    TIER_FULL,       // NEW
    MIGRATION_FAIL,  // NEW
    INTERNAL_ERROR
};

inline const char* error_name(ErrorType e) {
    switch(e) {
        case ErrorType::OK:               return "OK";
        case ErrorType::NOT_FOUND:        return "NOT_FOUND";
        case ErrorType::DUPLICATE_KEY:    return "DUPLICATE_KEY";
        case ErrorType::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ErrorType::OUT_OF_MEMORY:    return "OUT_OF_MEMORY";
        case ErrorType::TIER_FULL:        return "TIER_FULL";
        case ErrorType::MIGRATION_FAIL:   return "MIGRATION_FAIL";
        case ErrorType::INTERNAL_ERROR:   return "INTERNAL_ERROR";
        default:                          return "UNKNOWN";
    }
}

// ─── Log module (upstream log.h/log.cpp → inline) ───────────────────
// upstream用FILE*全局指针+log_set_fp()。我们改为直接stderr+可选文件。
namespace log {

inline std::atomic<int>& log_verbosity() {
    static std::atomic<int> v{2};  // 0=off, 1=error, 2=info, 3=debug
    return v;
}

inline void set_log_level(int lvl) { log_verbosity().store(lvl); }

inline void log_info(const char* fmt, ...) {
    if (log_verbosity().load() < 2) return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[INFO] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

inline void log_warn(const char* fmt, ...) {
    if (log_verbosity().load() < 1) return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[WARN] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

inline void log_debug(const char* fmt, ...) {
    if (log_verbosity().load() < 3) return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[DBG]  ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

}  // namespace log

// ─── Memory usage (from upstream driver.h) ──────────────────────────
inline int get_vm_rss_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return -1;
    int result = -1;
    char line[128];
    while (fgets(line, 128, file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            const char* p = line;
            while (*p < '0' || *p > '9') p++;
            result = atoi(p);
            break;
        }
    }
    fclose(file);
    return result;
}

// NEW: 打印当前内存使用
#define PHILE_MEM_CHECKPOINT(label) do { \
    int _rss = ::philemon::utils::get_vm_rss_kb(); \
    std::fprintf(stderr, "[MEM] %s: VmRSS = %d KB (%.2f MB)\n", \
                 (label), _rss, _rss / 1024.0); \
} while(0)

}  // namespace utils
}  // namespace philemon

// Backward-compatible macros
#define log_info  ::philemon::utils::log::log_info
#define log_warn  ::philemon::utils::log::log_warn
#define log_debug ::philemon::utils::log::log_debug

#endif  // PHILEMON_TIMER_UTILS_HPP
