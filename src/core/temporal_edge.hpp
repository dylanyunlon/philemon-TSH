/**
 * temporal_edge.hpp — 时间边的基本数据单元
 *
 * 从 temporal_bridge 中抽离，避免 partition_index 对桥接层的循环依赖。
 * 每条边携带源/目标顶点、权重、时间区间 [ts_begin, ts_finish]。
 *
 * 调试支持: dump_edge() 输出单条边全部字段，便于断点检查。
 */

#pragma once

#include <cstdint>
#include <cstdio>

namespace philemon {

struct TemporalEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_begin;     // interval start
    int32_t  ts_finish;       // interval end

    TemporalEdge()
        : source(0), destination(0), weight(0.0), ts_begin(0), ts_finish(0) {}
    TemporalEdge(uint64_t s, uint64_t d, double w, int32_t t0, int32_t t1)
        : source(s), destination(d), weight(w), ts_begin(t0), ts_finish(t1) {}

    // ── 断点调试: 打印当前边的完整状态 ──
    void dump_edge(const char* tag = "") const {
        std::printf("[EDGE-DUMP] %s src=%lu dst=%lu w=%.4f ts=[%d, %d] span=%d\n",
                    tag, (unsigned long)source, (unsigned long)destination,
                    weight, ts_begin, ts_finish, ts_finish - ts_begin);
    }

    bool operator==(const TemporalEdge& o) const {
        return source == o.source && destination == o.destination
            && ts_begin == o.ts_begin && ts_finish == o.ts_finish;
    }

    // 时间跨度
    int32_t span() const { return ts_finish - ts_begin; }

    // Interval containment: does [ts_begin, ts_finish] lie within [lo, hi]?
    // This is TEM-Graph's "contains query" predicate at the edge level.
    bool contained_in(int32_t lo, int32_t hi) const {
        return ts_begin >= lo && ts_finish <= hi;
    }

    // Interval overlap: does [ts_begin, ts_finish] overlap [lo, hi]?
    bool overlaps(int32_t lo, int32_t hi) const {
        return ts_begin <= hi && ts_finish >= lo;
    }

    // Lightweight hash for deduplication or hash-based lookup.
    // Combines vertex ids and timestamps via Knuth's multiplicative hash.
    uint64_t hash() const {
        uint64_t h = source * 2654435761ULL;
        h ^= destination * 40503ULL;
        h ^= static_cast<uint64_t>(ts_begin) << 17;
        h ^= static_cast<uint64_t>(ts_finish) << 31;
        return h;
    }
};

}  // namespace philemon
