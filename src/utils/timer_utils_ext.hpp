#ifndef PHILEMON_TIMER_UTILS_EXT_HPP
#define PHILEMON_TIMER_UTILS_EXT_HPP
/**
 * timer_utils_ext.hpp — 高精度计时器（含分层存储层级标注）
 *
 * 骨架来源: upstream/rapidstore/utils/Timer.h (35行)
 * 修改 (~20%):
 *   - 保留 Timer 类核心接口 (reset, elapsed, elapsed_and_reset)
 *   - 增加 tier_label_ 字段：标记计时归属哪个存储层 (DRAM/NVM/SSD)
 *   - 增加 dump_elapsed()：打印当前计时 + 层级标签
 *   - 增加 ScopedTimer RAII 类：自动测量作用域耗时并打印
 *   - 增加 lap() 分段计时功能
 *
 * Milestone: M027 (upstream utility migration)
 */

#include <iostream>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace philemon {
namespace utils {

class Timer {
public:
    Timer(const char* tier_label = "default")
        : beg_(clock_::now()), tier_label_(tier_label) {}

    void reset() { beg_ = clock_::now(); laps_.clear(); }

    double elapsed() const {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

    double elapsed_and_reset() {
        double e = elapsed();
        beg_ = clock_::now();
        return e;
    }

    // ─── NEW: 分段计时 ───────────────────────────────────────────────
    void lap(const char* lap_label = "") {
        double e = elapsed();
        laps_.push_back({std::string(lap_label), e});
    }

    // ─── NEW: 打印当前计时状态 ───────────────────────────────────────
    void dump_elapsed(const char* context = "") const {
        std::printf("[TIMER][%s] %s elapsed=%.6f s\n",
                    tier_label_, context, elapsed());
    }

    // ─── NEW: 打印所有分段计时 ───────────────────────────────────────
    void dump_laps() const {
        std::printf("[TIMER][%s] === Lap Summary (%zu laps) ===\n",
                    tier_label_, laps_.size());
        double prev = 0.0;
        for (size_t i = 0; i < laps_.size(); ++i) {
            double delta = laps_[i].second - prev;
            std::printf("  lap[%zu] %-20s cumulative=%.6f s  delta=%.6f s\n",
                        i, laps_[i].first.c_str(), laps_[i].second, delta);
            prev = laps_[i].second;
        }
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
    const char* tier_label_;
    std::vector<std::pair<std::string, double>> laps_;
};

// ─── NEW: RAII 作用域计时器 ─────────────────────────────────────────
struct ScopedTimer {
    const char* label_;
    Timer timer_;

    ScopedTimer(const char* label, const char* tier = "default")
        : label_(label), timer_(tier) {
        std::printf("[SCOPE_TIMER] >>> ENTER %s\n", label_);
    }

    ~ScopedTimer() {
        std::printf("[SCOPE_TIMER] <<< EXIT  %s  elapsed=%.6f s\n",
                    label_, timer_.elapsed());
    }
};

} // namespace utils
} // namespace philemon

#endif // PHILEMON_TIMER_UTILS_EXT_HPP
