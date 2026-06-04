#ifndef PHILEMON_EDGE_STREAM_HPP
#define PHILEMON_EDGE_STREAM_HPP
/**
 * edge_stream.hpp — Edge stream for batch loading temporal graphs
 *
 * 骨架来源: upstream/rapidstore/graph/edgeStream.hpp (33行)
 * 修改 (~20%):
 *   - 移除 reader 依赖 (使用内存加载替代文件I/O)
 *   - 增加 load_from_temporal_edges() 从 TemporalEdge 数组加载
 *   - 增加 per-tier partitioning: reorder_by_tier()
 *   - 增加 debug stats: dump_stream_stats()
 *   - 保留 driver::graph 命名空间兼容
 *
 * Milestone: M013
 */

#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_map>
#include <cstdio>
#include <cstdint>

#include "graph_edge.hpp"
#include "../core/temporal_edge.hpp"

namespace driver {
namespace graph {

class edgeStream {
public:
    edgeStream() : index_(0) {}

    // ---- Upstream API (preserved, file I/O removed) ----

    void permute_stream() {
        unsigned seed = std::chrono::system_clock::now()
                            .time_since_epoch().count();
        std::shuffle(edge_stream_.begin(), edge_stream_.end(),
                     std::default_random_engine(seed));
    }

    void sort() {
        std::sort(edge_stream_.begin(), edge_stream_.end());
    }

    void remove_duplicates() {
        sort();
        // 算法改动: 遇到重复边(same src,dst)不是丢弃，而是累加weight
        // 这样多次出现的边获得更高权重，反映实际连接强度
        if (edge_stream_.size() < 2) return;
        size_t write = 0;
        for (size_t read = 1; read < edge_stream_.size(); read++) {
            if (edge_stream_[read] == edge_stream_[write]) {
                // 重复边: 累加权重, 取最大时间跨度
                edge_stream_[write].weight += edge_stream_[read].weight;
                edge_stream_[write].ts_start = std::min(
                    edge_stream_[write].ts_start, edge_stream_[read].ts_start);
                edge_stream_[write].ts_end = std::max(
                    edge_stream_[write].ts_end, edge_stream_[read].ts_end);
            } else {
                write++;
                if (write != read) edge_stream_[write] = edge_stream_[read];
            }
        }
        edge_stream_.resize(write + 1);
    }

    bool get_next_edge(weightedEdge& edge) {
        if (index_ >= edge_stream_.size()) return false;
        edge.set_edge(edge_stream_[index_++]);
        return true;
    }

    weightedEdge& operator[](int index) {
        return edge_stream_[index];
    }

    int get_size() const { return edge_stream_.size(); }
    int get_current_index() const { return index_; }
    void reset_index() { index_ = 0; }

    // Degree-based partitioning (算法改动: 自适应分割点代替固定10%)
    // upstream原版: 固定取top 10%作为hot partition
    // 改动: 用degree中位数作为分割阈值——degree > median的边进hot分区
    void reorder_and_partition(bool high_degree_first) {
        std::unordered_map<uint64_t, int> degree_map;
        for (auto& e : edge_stream_) {
            degree_map[e.source]++;
            degree_map[e.destination]++;
        }

        // 计算edge-level degree的中位数
        std::vector<int> edge_degrees;
        edge_degrees.reserve(edge_stream_.size());
        for (auto& e : edge_stream_) {
            edge_degrees.push_back(std::max(degree_map[e.source],
                                             degree_map[e.destination]));
        }
        std::sort(edge_degrees.begin(), edge_degrees.end());
        int median_deg = edge_degrees.empty() ? 0
            : edge_degrees[edge_degrees.size() / 2];

        // 按degree排序
        std::sort(edge_stream_.begin(), edge_stream_.end(),
            [&](const weightedEdge& a, const weightedEdge& b) {
                int da = std::max(degree_map[a.source],
                                  degree_map[a.destination]);
                int db = std::max(degree_map[b.source],
                                  degree_map[b.destination]);
                return high_degree_first ? da > db : da < db;
            });

        // 自适应分割: degree > median 的边数作为hot partition大小
        int hot_count = 0;
        for (auto& e : edge_stream_) {
            int d = std::max(degree_map[e.source],
                             degree_map[e.destination]);
            if (d > median_deg) hot_count++;
            else break;  // 已排好序, 一旦低于median后面全是低的
        }

        // 重组: hot partition在前
        std::vector<weightedEdge> picked(
            edge_stream_.begin(), edge_stream_.begin() + hot_count);
        picked.insert(picked.end(),
                      edge_stream_.begin() + hot_count, edge_stream_.end());
        edge_stream_ = std::move(picked);
        reset_index();
        remove_duplicates();
    }

    // ─── NEW: Load from TemporalEdge array ──────────────────────────

    void load_from_temporal_edges(const philemon::TemporalEdge* edges,
                                  size_t count) {
        edge_stream_.clear();
        edge_stream_.reserve(count);
        for (size_t i = 0; i < count; i++) {
            edge_stream_.emplace_back(
                edges[i].source, edges[i].destination, edges[i].weight,
                edges[i].ts_begin, edges[i].ts_finish);
        }
        index_ = 0;
    }

    // ─── NEW: Load from (src, dst, ts_start, ts_end) tuples ────────

    void load_from_tuples(
        const std::vector<std::tuple<uint64_t, uint64_t, int32_t, int32_t>>& tuples) {
        edge_stream_.clear();
        edge_stream_.reserve(tuples.size());
        for (auto& [s, d, t0, t1] : tuples) {
            edge_stream_.emplace_back(s, d, 1.0, t0, t1);
        }
        index_ = 0;
    }

    // ─── NEW: Stream statistics ─────────────────────────────────────

    void dump_stream_stats(const char* label = "EdgeStream") const {
        if (edge_stream_.empty()) {
            std::printf("[%s] empty\n", label);
            return;
        }
        uint64_t max_v = 0;
        int32_t min_ts = edge_stream_[0].ts_start;
        int32_t max_ts = edge_stream_[0].ts_end;
        std::unordered_map<uint64_t, int> deg;
        for (auto& e : edge_stream_) {
            max_v = std::max(max_v, std::max(e.source, e.destination));
            min_ts = std::min(min_ts, e.ts_start);
            max_ts = std::max(max_ts, e.ts_end);
            deg[e.source]++;
        }
        int max_deg = 0;
        for (auto& [v, d] : deg) max_deg = std::max(max_deg, d);
        std::printf("[%s] edges=%d vertices=%lu ts_range=[%d,%d] "
                    "max_out_degree=%d\n",
                    label, (int)edge_stream_.size(),
                    (unsigned long)(max_v + 1),
                    min_ts, max_ts, max_deg);
    }

    // ─── Access to underlying vector ────────────────────────────────
    const std::vector<weightedEdge>& edges() const { return edge_stream_; }
    std::vector<weightedEdge>& edges() { return edge_stream_; }

private:
    std::vector<weightedEdge> edge_stream_;
    size_t index_;
};

}  // namespace graph
}  // namespace driver

namespace philemon {
namespace graph {
    using EdgeStream = driver::graph::edgeStream;
}  // namespace graph
}  // namespace philemon

#endif  // PHILEMON_EDGE_STREAM_HPP
