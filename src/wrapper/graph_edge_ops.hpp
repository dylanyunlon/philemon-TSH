#ifndef PHILEMON_GRAPH_EDGE_OPS_HPP
#define PHILEMON_GRAPH_EDGE_OPS_HPP
/**
 * graph_edge_ops.hpp — Edge 实现层（operator、序列化、temporal操作）
 *
 * 骨架来源:
 *   upstream/rapidstore/graph/edge.cpp  (38行)
 *
 * 修改 (~20%):
 *   - [MOD] driver::graph → 保持 (兼容wrapper层)
 *   - [MOD] hpp+cpp合并为header-only
 *   - [NEW] dump_edge(): 打印边的完整状态 (src, dst, weight, ts)
 *   - [NEW] temporal_overlap(): 检测两条边的时间窗口是否重叠
 *   - [NEW] batch_dump(): 一次性打印前N条边
 *   - [KEEP] set_edge 两个重载 100%保留
 *   - [KEEP] operator==, !=, < 100%保留
 *   - [KEEP] 构造函数签名 100%保留
 *
 * graph_edge.hpp 已有类声明和内联实现，
 * 本文件补全 upstream edge.cpp 中的非内联实现 + debug扩展。
 *
 * Milestone: M028
 */

#include "graph_edge.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>

namespace driver {
namespace graph {

// ─── Upstream edge.cpp 实现 (已内联到graph_edge.hpp, 此处补充debug) ──

// [NEW] 打印单条边的完整状态
inline void dump_edge(const weightedEdge& e, const char* label = "") {
    std::printf("[EDGE] %s src=%lu dst=%lu w=%.4f ts=[%d,%d]\n",
                label,
                (unsigned long)e.source,
                (unsigned long)e.destination,
                e.weight, e.ts_start, e.ts_end);
}

// [NEW] 批量打印前N条边
inline void batch_dump_edges(const std::vector<weightedEdge>& edges,
                              size_t max_show = 10,
                              const char* label = "EDGES") {
    size_t n = std::min(edges.size(), max_show);
    std::printf("[%s] showing %lu/%lu edges:\n",
                label, (unsigned long)n, (unsigned long)edges.size());
    for (size_t i = 0; i < n; i++) {
        std::printf("  [%lu] %lu→%lu w=%.3f ts=[%d,%d]\n",
                    (unsigned long)i,
                    (unsigned long)edges[i].source,
                    (unsigned long)edges[i].destination,
                    edges[i].weight,
                    edges[i].ts_start, edges[i].ts_end);
    }
    if (edges.size() > max_show)
        std::printf("  ... (%lu more)\n",
                    (unsigned long)(edges.size() - max_show));
}

// [NEW] 检测两条边的时间窗口是否重叠
inline bool temporal_overlap(const weightedEdge& a,
                              const weightedEdge& b) {
    return a.ts_start <= b.ts_end && b.ts_start <= a.ts_end;
}

// [NEW] 按时间戳排序的比较器
inline bool temporal_less(const weightedEdge& a,
                           const weightedEdge& b) {
    if (a.ts_start != b.ts_start) return a.ts_start < b.ts_start;
    if (a.ts_end != b.ts_end) return a.ts_end < b.ts_end;
    return a < b;  // fallback到upstream的operator<
}

} // namespace graph
} // namespace driver

#endif // PHILEMON_GRAPH_EDGE_OPS_HPP
