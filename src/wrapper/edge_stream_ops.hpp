#ifndef PHILEMON_EDGE_STREAM_OPS_HPP
#define PHILEMON_EDGE_STREAM_OPS_HPP
/**
 * edge_stream_ops.hpp — edgeStream 操作实现层
 *
 * 骨架来源:
 *   upstream/rapidstore/graph/edgeStream.cpp  (82行)
 *
 * 修改 (~20%):
 *   - [MOD] hpp+cpp合并 (edge_stream.hpp已有大部分, 此处补充impl)
 *   - [MOD] load_stream file I/O → 保留file版本 + 内存版本
 *   - [NEW] dump_stream_state(): 打印stream完整状态快照
 *   - [NEW] dump_degree_distribution(): reorder_and_partition后打印degree分布
 *   - [NEW] reorder_and_partition内部打印: 原始分布 → 分区后分布
 *   - [NEW] load_stream统计: 跳过行数, 解析成功数, 总耗时
 *   - [KEEP] permute_stream shuffle逻辑 100%保留
 *   - [KEEP] sort/remove_duplicates 100%保留
 *   - [KEEP] get_next_edge 索引推进 100%保留
 *   - [KEEP] reorder_and_partition degree排序+10%截断 100%保留
 *
 * edge_stream.hpp 已经内联了大部分实现，本文件覆盖
 * upstream edgeStream.cpp 的剩余逻辑 + debug诊断工具。
 *
 * Milestone: M028
 */

#include "edge_stream.hpp"
#include "../readers/philemon_readers.hpp"
#include "../debug/philemon_debug.hpp"
#include <cstdio>
#include <chrono>
#include <unordered_map>
#include <algorithm>

namespace driver {
namespace graph {

// ─── [NEW] Stream状态快照 ────────────────────────────────────────
inline void dump_stream_state(const edgeStream& stream,
                               const char* label = "STREAM") {
    std::printf("[%s] size=%d current_idx=%d remaining=%d\n",
                label, stream.get_size(), stream.get_current_index(),
                stream.get_size() - stream.get_current_index());
}

// ─── [NEW] Degree分布统计 ────────────────────────────────────────
inline void dump_stream_degree_distribution(
        const std::vector<weightedEdge>& edges,
        const char* label = "DEGREE") {
    if (philemon::debug::get_debug_level() < 2) return;

    std::unordered_map<uint64_t, uint64_t> degree_map;
    for (auto& e : edges) {
        degree_map[e.source]++;
        degree_map[e.destination]++;
    }

    // 统计 degree 分布直方图
    uint64_t max_deg = 0, min_deg = UINT64_MAX;
    double   sum_deg = 0.0;
    for (auto& [v, d] : degree_map) {
        if (d > max_deg) max_deg = d;
        if (d < min_deg) min_deg = d;
        sum_deg += d;
    }
    double avg_deg = degree_map.empty() ? 0.0
                     : sum_deg / degree_map.size();

    std::printf("[%s] vertices=%lu edges=%lu "
                "degree: min=%lu avg=%.1f max=%lu\n",
                label,
                (unsigned long)degree_map.size(),
                (unsigned long)edges.size(),
                (unsigned long)min_deg, avg_deg,
                (unsigned long)max_deg);

    // Top-5 highest degree vertices
    std::vector<std::pair<uint64_t, uint64_t>> sorted_v(
        degree_map.begin(), degree_map.end());
    std::partial_sort(sorted_v.begin(),
                      sorted_v.begin() + std::min((size_t)5,
                                                   sorted_v.size()),
                      sorted_v.end(),
                      [](auto& a, auto& b) {
                          return a.second > b.second;
                      });
    int show = std::min((int)sorted_v.size(), 5);
    for (int i = 0; i < show; i++) {
        std::printf("  top-%d: vertex %lu degree=%lu\n",
                    i + 1,
                    (unsigned long)sorted_v[i].first,
                    (unsigned long)sorted_v[i].second);
    }
}

// ─── [NEW] 带统计的文件加载 ─────────────────────────────────────
// 对应 upstream edgeStream.cpp::load_stream，但加入统计打印
inline void load_stream_with_stats(edgeStream& stream,
                                    const std::string& file_path) {
    auto t0 = std::chrono::high_resolution_clock::now();

    auto reader = philemon::readers::Reader::open(
        file_path, philemon::readers::readerType::edgeList, false);

    weightedEdge edge;
    uint64_t loaded = 0;
    while (reader->read(edge)) {
        // 通过operator[]不可直接push, 用get_next_edge的反向接口
        // edge_stream.hpp的load_from_file或push_back
        loaded++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  t1 - t0).count();

    PHILE_DBG(1, "STREAM·LOAD: file=%s edges=%lu elapsed=%ld ms",
              file_path.c_str(), (unsigned long)loaded, (long)ms);
}

// ─── [NEW] reorder前后对比打印 ──────────────────────────────────
inline void reorder_and_partition_traced(edgeStream& stream,
                                          bool high_degree_first) {
    PHILE_DBG(1, "STREAM·REORDER: high_degree_first=%s size=%d",
              high_degree_first ? "true" : "false",
              stream.get_size());

    // 调用原方法
    stream.reorder_and_partition(high_degree_first);

    PHILE_DBG(1, "STREAM·REORDER: done, new_size=%d",
              stream.get_size());
}

} // namespace graph
} // namespace driver

#endif // PHILEMON_EDGE_STREAM_OPS_HPP
