/**
 * philemon_timer.hpp — High-resolution timer utility
 *
 * 骨架来源: upstream/rapidstore/utils/Timer.h (35行)
 * 修改 (~20%):
 *   - [MOD] namespace包裹: philemon::utils
 *   - [NEW] lap()方法: 记录分段计时, 支持多段累计
 *   - [NEW] accumulated_elapsed(): 返回所有lap之和
 *   - [NEW] TimerRegistry: 全局命名计时器注册表, DUMP_ALL_TIMERS()宏
 *   - [NEW] BREAKPOINT_TIMER(name): 在关键路径打印当前计时器状态
 *   - [KEEP] reset(), elapsed(), elapsed_and_reset() 100%保留upstream语义
 *
 * Milestone: M098
 */
#ifndef PHILEMON_TIMER_HPP
#define PHILEMON_TIMER_HPP

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstdio>

namespace philemon {
namespace utils {

// ─── Core Timer (upstream骨架, 修改20%) ─────────────────────────────
class Timer {
public:
    Timer() : beg_(clock_::now()) {}

    void reset() { 
        beg_ = clock_::now(); 
        laps_.clear();          // [NEW] reset清空分段记录
    }

    double elapsed() const {
        return std::chrono::duration_cast<second_>(
            clock_::now() - beg_).count();
    }

    double elapsed_and_reset() {
        double el = std::chrono::duration_cast<second_>(
            clock_::now() - beg_).count();
        beg_ = clock_::now();
        return el;
    }

    // ─── [NEW] 分段计时 ───────────────────────────────────────────
    double lap() {
        auto now = clock_::now();
        double seg = std::chrono::duration_cast<second_>(now - beg_).count();
        laps_.push_back(seg);
        beg_ = now;
        return seg;
    }

    double accumulated_elapsed() const {
        double total = 0.0;
        for (double l : laps_) total += l;
        total += elapsed();  // 加上当前未lap的部分
        return total;
    }

    size_t lap_count() const { return laps_.size(); }

    void dump_laps(const char* label = "Timer") const {
        std::fprintf(stderr, "[TIMER] %s: %zu laps, accumulated=%.6fs\n",
                     label, laps_.size(), accumulated_elapsed());
        for (size_t i = 0; i < laps_.size(); i++) {
            std::fprintf(stderr, "  lap[%zu] = %.6fs\n", i, laps_[i]);
        }
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
    std::vector<double> laps_;    // [NEW] 分段记录
};

// ─── [NEW] 全局计时器注册表 ─────────────────────────────────────────
class TimerRegistry {
public:
    static TimerRegistry& instance() {
        static TimerRegistry reg;
        return reg;
    }

    Timer& get(const std::string& name) {
        std::lock_guard<std::mutex> lk(mtx_);
        return timers_[name];
    }

    void dump_all() {
        std::lock_guard<std::mutex> lk(mtx_);
        std::fprintf(stderr, "\n═══ TIMER REGISTRY DUMP (%zu timers) ═══\n",
                     timers_.size());
        for (auto& [name, t] : timers_) {
            t.dump_laps(name.c_str());
        }
        std::fprintf(stderr, "═══════════════════════════════════════\n\n");
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        timers_.clear();
    }

private:
    TimerRegistry() = default;
    std::mutex mtx_;
    std::unordered_map<std::string, Timer> timers_;
};

// ─── 调试宏 ─────────────────────────────────────────────────────────
#define BREAKPOINT_TIMER(name) do { \
    auto& __t = philemon::utils::TimerRegistry::instance().get(name); \
    std::fprintf(stderr, "[BREAKPOINT_TIMER] %s: elapsed=%.6fs, laps=%zu\n", \
                 name, __t.elapsed(), __t.lap_count()); \
} while(0)

#define DUMP_ALL_TIMERS() \
    philemon::utils::TimerRegistry::instance().dump_all()

} // namespace utils
} // namespace philemon

#endif // PHILEMON_TIMER_HPP
