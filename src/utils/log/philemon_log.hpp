#ifndef PHILEMON_LOG_HPP
#define PHILEMON_LOG_HPP
/**
 * philemon_log.hpp — 分层日志系统 (header-only)
 *
 * 骨架来源: upstream/rapidstore/utils/log/log.h + log.cpp
 * 修改 (~25%):
 *   - 合并 .h/.cpp 为 header-only (减少编译依赖)
 *   - 原始全局 mutex 保留; 增加 tier_tag 字段注入每条日志
 *   - 增加 log_dump_struct() 宏: 一键打印任意结构体摘要
 *   - 增加 PHILEMON_BREAKPOINT() 宏: 条件断点 + 上下文快照
 *   - 将 Timer 依赖改为 chrono 内联, 不再依赖 Timer.h
 *   - 增加 log_set_tier_tag() 动态切换当前线程的层级标签
 *
 * Milestone: M027 (upstream utility migration)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <mutex>
#include <chrono>
#include <string>

// ─── Log Level Constants (from upstream) ────────────────────────────
enum {
    PHILEMON_LOG_TRACE, PHILEMON_LOG_DEBUG, PHILEMON_LOG_INFO,
    PHILEMON_LOG_WARN,  PHILEMON_LOG_ERROR, PHILEMON_LOG_FATAL
};

// ─── Macro Definitions ─────────────────────────────────────────────
#define __PHILEMON_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define plog_trace(...) philemon::log::log_impl(PHILEMON_LOG_TRACE, __PHILEMON_FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define plog_debug(...) philemon::log::log_impl(PHILEMON_LOG_DEBUG, __PHILEMON_FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define plog_info(...)  philemon::log::log_impl(PHILEMON_LOG_INFO,  __PHILEMON_FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define plog_warn(...)  philemon::log::log_impl(PHILEMON_LOG_WARN,  __PHILEMON_FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define plog_error(...) philemon::log::log_impl(PHILEMON_LOG_ERROR, __PHILEMON_FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define plog_fatal(...) philemon::log::log_impl(PHILEMON_LOG_FATAL, __PHILEMON_FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

// ─── NEW: 一键打印结构体摘要 ───────────────────────────────────────
#define plog_dump_struct(label, obj) do { \
    plog_info("[STRUCT_DUMP] %s => calling dump()", label); \
    (obj).dump(label); \
} while(0)

// ─── NEW: 条件断点宏 ───────────────────────────────────────────────
// 用法: PHILEMON_BREAKPOINT(edge_count > 10000, "edge overflow suspected")
#define PHILEMON_BREAKPOINT(condition, msg) do { \
    if (condition) { \
        plog_warn("[BREAKPOINT] condition=true: %s at %s:%d", msg, __FILE__, __LINE__); \
        plog_warn("[BREAKPOINT] Attach debugger or set env PHILEMON_ABORT_ON_BP=1 to trap"); \
        if (std::getenv("PHILEMON_ABORT_ON_BP")) { std::abort(); } \
    } \
} while(0)

// ─── Backward compatibility with upstream log_info etc. ─────────────
#define log_info  plog_info
#define log_warn  plog_warn
#define log_error plog_error
#define log_debug plog_debug
#define log_trace plog_trace
#define log_fatal plog_fatal

namespace philemon {
namespace log {

// ─── Internal state ─────────────────────────────────────────────────
namespace detail {
    inline std::mutex& global_mutex() {
        static std::mutex mtx;
        return mtx;
    }

    inline auto& start_time() {
        static auto t = std::chrono::high_resolution_clock::now();
        return t;
    }

    // NEW: per-thread tier tag
    inline const char*& tier_tag() {
        static thread_local const char* tag = "GLOBAL";
        return tag;
    }

    inline int& log_level() {
        static int lvl = PHILEMON_LOG_TRACE;
        return lvl;
    }

    inline FILE*& log_fp() {
        static FILE* fp = nullptr;
        return fp;
    }

    inline bool& quiet_mode() {
        static bool q = false;
        return q;
    }

    static const char* level_names[] = {
        "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
    };

    static const char* level_colors[] = {
        "\x1b[94m", "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[35m"
    };
} // namespace detail

// ─── Public API ─────────────────────────────────────────────────────
inline void set_level(int level) { detail::log_level() = level; }
inline void set_fp(FILE* fp)     { detail::log_fp() = fp; }
inline void set_quiet(bool q)    { detail::quiet_mode() = q; }

// NEW: 设置当前线程的存储层级标签
inline void set_tier_tag(const char* tag) { detail::tier_tag() = tag; }

inline void log_impl(int level, const char* file, const char* func,
                      int line, const char* fmt, ...) {
    if (level < detail::log_level()) return;

    using namespace std::chrono;
    double elapsed = duration_cast<duration<double>>(
        high_resolution_clock::now() - detail::start_time()).count();

    std::lock_guard<std::mutex> lock(detail::global_mutex());

    time_t t = time(nullptr);
    struct tm* lt = localtime(&t);

    if (!detail::quiet_mode()) {
        char buf[64];
        buf[strftime(buf, sizeof(buf), "%H:%M:%S", lt)] = '\0';

        // NEW: tier tag injected
        fprintf(stderr,
                "%s %s%-5s\x1b[0m [%s] (%.4fs) \x1b[36m%s()\x1b[0m \x1b[90m%s:%d\x1b[0m ",
                buf, detail::level_colors[level], detail::level_names[level],
                detail::tier_tag(), elapsed, func, file, line);

        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
        fprintf(stderr, "\n");
    }

    if (detail::log_fp()) {
        char buf[32];
        buf[strftime(buf, sizeof(buf), "%H:%M:%S", lt)] = '\0';
        fprintf(detail::log_fp(), "%s %-5s [%s] (%.4fs) %s() %s:%d: ",
                buf, detail::level_names[level],
                detail::tier_tag(), elapsed, func, file, line);

        va_list args;
        va_start(args, fmt);
        vfprintf(detail::log_fp(), fmt, args);
        va_end(args);
        fprintf(detail::log_fp(), "\n");
    }
}

} // namespace log
} // namespace philemon

#endif // PHILEMON_LOG_HPP
