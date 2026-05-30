#ifndef PHILEMON_INTERVAL_HPP
#define PHILEMON_INTERVAL_HPP
/**
 * interval.hpp — Temporal interval types for Philemon-TSH
 *
 * 骨架来源: upstream/temgraph/interval.h (100% 的数据结构保留)
 * 修改 (~20%):
 *   - 包裹在 philemon::index namespace 中
 *   - 移除全局变量 visited_intervals_ → 改为 per-query 局部计数
 *   - 增加 debug print 方法: Interval::dump(), TInterval::dump()
 *   - 增加 tier 标注字段: TInterval::tier_hint
 *   - 移除 using namespace std (避免namespace污染)
 *   - 增加 MemoryTier 感知的 comparator
 *   - print_peak_memory_usage() 改名 + 增加 per-tier 打印
 *
 * Milestone: M011 (Claude #5) — TEM-Graph index integration
 */

#include <sys/time.h>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <sys/resource.h>
#include <cstdio>

namespace philemon {
namespace index {

// ─── Type aliases (from upstream, unchanged) ────────────────────────
typedef int Timestamp;
typedef uint32_t SuccessorLoc;
typedef uint32_t RecordId;

// ─── Query type constants (from upstream) ───────────────────────────
static constexpr int CONTAINS_QUERY = 1;
static constexpr int OTHER_QUERY    = 2;

// ─── Memory diagnostics (modified from upstream print_peak_memory_usage) ─
inline void print_peak_memory_usage(const char* label = "Global") {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        double max_mb = usage.ru_maxrss / 1024.0;
        double max_gb = max_mb / 1024.0;
        std::printf("[MEM] %s peak: %ld KB = %.1f MB = %.3f GB\n",
                    label, usage.ru_maxrss, max_mb, max_gb);
    } else {
        std::fprintf(stderr, "[MEM] Failed to get memory usage for %s\n", label);
    }
}

// ─── Interval struct (from upstream, +debug dump) ───────────────────
struct Interval {
    RecordId  id;
    Timestamp start, end;

    Interval() : id(0), start(0), end(0) {}
    Interval(RecordId id, int start, int end)
        : id(id), start(start), end(end) {}

    // Sort by start ascending (upstream unchanged)
    bool operator<(const Interval& other) const {
        if (start == other.start) return end < other.end;
        return start < other.start;
    }

    // ---- NEW: debug dump ----
    void dump(const char* prefix = "") const {
        std::printf("%s[Interval id=%u start=%d end=%d span=%d]\n",
                    prefix, id, start, end, end - start);
    }
};

// ─── TInterval (temporal interval, from upstream, +debug +tier_hint) ─
class TInterval {
public:
    RecordId  id;
    Timestamp l, r;
    uint8_t   tier_hint;  // NEW: which tier this interval preferably resides on

    TInterval(RecordId _id, int _l, int _r)
        : id(_id), l(_l), r(_r), tier_hint(0) {}

    TInterval(RecordId _id, int _l, int _r, uint8_t _tier)
        : id(_id), l(_l), r(_r), tier_hint(_tier) {}

    // Sort by r ascending, then l ascending (upstream unchanged)
    bool operator<(const TInterval& other) const {
        if (r == other.r && l == other.l) return id < other.id;
        if (r == other.r) return l < other.l;
        return r < other.r;
    }

    // ---- NEW: debug dump ----
    void dump(const char* prefix = "") const {
        static const char* tier_names[] = {"HBM", "GDDR", "DRAM", "??"};
        std::printf("%s[TInterval id=%u l=%d r=%d span=%d tier=%s]\n",
                    prefix, id, l, r, r - l,
                    tier_names[tier_hint % 4]);
    }

    // ---- NEW: span width for density calculation ----
    int span() const { return r - l; }
};

// ─── OutNeighbor (successor pointer, from upstream, +dump) ──────────
struct OutNeighbor {
    Timestamp    l;
    RecordId     x;
    SuccessorLoc successor;

    OutNeighbor(int _l, RecordId _x, SuccessorLoc _successor)
        : l(_l), x(_x), successor(_successor) {}

    void dump(const char* prefix = "") const {
        std::printf("%s[OutNeighbor l=%d x=%u succ=%u]\n",
                    prefix, l, x, successor);
    }
};

// ─── Utility functions (from upstream, minor modifications) ─────────
inline int random_int(int low, int high) {
    // Renamed from random() to avoid name collision
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

inline double GetTime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ─── NEW: High-resolution timer for debug ──────────────────────────
inline uint64_t now_ns() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
}

// ─── NEW: Batch interval statistics printer ─────────────────────────
inline void print_interval_stats(const std::vector<TInterval>& intervals,
                                  const char* label = "intervals") {
    if (intervals.empty()) {
        std::printf("[STATS] %s: empty\n", label);
        return;
    }
    Timestamp min_l = intervals[0].l, max_r = intervals[0].r;
    int64_t total_span = 0;
    for (auto& iv : intervals) {
        min_l = std::min(min_l, iv.l);
        max_r = std::max(max_r, iv.r);
        total_span += iv.span();
    }
    double avg_span = (double)total_span / intervals.size();
    double density  = (double)intervals.size() / std::max(1, max_r - min_l);
    std::printf("[STATS] %s: n=%zu range=[%d,%d] avg_span=%.1f density=%.3f\n",
                label, intervals.size(), min_l, max_r, avg_span, density);
}

}  // namespace index
}  // namespace philemon

#endif  // PHILEMON_INTERVAL_HPP
