#ifndef PHILEMON_GRAPH_EDGE_HPP
#define PHILEMON_GRAPH_EDGE_HPP
/**
 * graph_edge.hpp — Weighted edge type for RapidStore compatibility
 *
 * 骨架来源: upstream/rapidstore/graph/edge.hpp (26行)
 * 修改 (~15%):
 *   - 保留 driver::graph namespace (RapidStore算法直接引用)
 *   - 增加 philemon::graph alias namespace
 *   - 增加 dump() debug 方法
 *   - 增加 temporal 字段 (ts_start, ts_end)
 *
 * Milestone: M013 (wrapper type support)
 */

#include <cstdint>
#include <cstdio>

namespace driver {
namespace graph {

class weightedEdge {
public:
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_start = 0;  // NEW: temporal annotation
    int32_t  ts_end   = 0;  // NEW: temporal annotation

    weightedEdge() : source(0), destination(0), weight(0.0) {}

    weightedEdge(uint64_t source, uint64_t destination, double weight)
        : source(source), destination(destination), weight(weight) {}

    weightedEdge(uint64_t source, uint64_t destination)
        : source(source), destination(destination), weight(-1.0) {}

    // NEW: temporal constructor
    weightedEdge(uint64_t source, uint64_t destination, double weight,
                 int32_t ts_start, int32_t ts_end)
        : source(source), destination(destination), weight(weight),
          ts_start(ts_start), ts_end(ts_end) {}

    void set_edge(uint64_t source, uint64_t destination, double weight) {
        this->source = source;
        this->destination = destination;
        this->weight = weight;
    }

    void set_edge(weightedEdge& edge) {
        this->source = edge.source;
        this->destination = edge.destination;
        this->weight = edge.weight;
        this->ts_start = edge.ts_start;
        this->ts_end = edge.ts_end;
    }

    bool operator==(const weightedEdge& rhs) const {
        return source == rhs.source && destination == rhs.destination;
    }
    bool operator!=(const weightedEdge& rhs) const {
        return !(*this == rhs);
    }
    bool operator<(const weightedEdge& rhs) const {
        return source < rhs.source ||
               (source == rhs.source && destination < rhs.destination);
    }

    // NEW: debug dump
    void dump(const char* prefix = "") const {
        std::printf("%s[Edge %lu→%lu w=%.2f ts=[%d,%d]]\n",
                    prefix, (unsigned long)source,
                    (unsigned long)destination, weight,
                    ts_start, ts_end);
    }
    void dump_graph_edge(const char* tag = "") const {
        std::printf("[GRAPH-EDGE] %s src=%lu dst=%lu w=%.4f\n",
                    tag, (unsigned long)source, (unsigned long)destination, weight);
    }
};

}  // namespace graph
}  // namespace driver

// NEW: Philemon alias
namespace philemon {
namespace graph {
    using Edge = driver::graph::weightedEdge;
}  // namespace graph
}  // namespace philemon

#endif  // PHILEMON_GRAPH_EDGE_HPP
