/**
 * temporal_edge.hpp — TemporalEdge struct definition
 *
 * Extracted from temporal_bridge.hpp to avoid circular dependencies
 * when partition_index.hpp needs access to edge fields.
 *
 * Milestone: M011 (Claude #5)
 */

#pragma once

#include <cstdint>

namespace philemon {

struct TemporalEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_start;     // interval start
    int32_t  ts_end;       // interval end

    TemporalEdge()
        : source(0), destination(0), weight(0.0), ts_start(0), ts_end(0) {}
    TemporalEdge(uint64_t s, uint64_t d, double w, int32_t t0, int32_t t1)
        : source(s), destination(d), weight(w), ts_start(t0), ts_end(t1) {}
};

}  // namespace philemon
