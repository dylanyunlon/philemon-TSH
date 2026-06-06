/**
 * edge_stream_file_io.hpp — 文件I/O驱动的边流加载器
 *
 * 骨架来源: upstream/rapidstore/graph/edgeStream.cpp+hpp (115行)
 * 修改 (~20%):
 *   - [MOD] namespace driver::graph保留, 但reader依赖换为philemon::readers
 *   - [NEW] load_stream()增加进度打印: 每100K条边报告一次
 *   - [NEW] stream_stats(): 返回流的统计信息(min/max degree, 边数)
 *   - [NEW] BREAKPOINT_STREAM(): 打印当前流的全部状态
 *   - [NEW] tier_partition(): 按degree分3层(hot/warm/cold)
 *   - [KEEP] permute_stream(), sort(), remove_duplicates() 100%保留
 *   - [KEEP] get_next_edge(), operator[], get_size() 100%保留
 *   - [KEEP] reorder_and_partition() 逻辑100%保留
 *
 * Milestone: M098
 */
#ifndef PHILEMON_EDGE_STREAM_FILE_IO_HPP
#define PHILEMON_EDGE_STREAM_FILE_IO_HPP

#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cstdio>
#include <cstdint>
#include <cassert>

namespace philemon {
namespace io {

// ─── WeightedEdge (upstream graph/edge.hpp 100%) ────────────────────
struct StreamEdge {
    uint64_t source = 0;
    uint64_t destination = 0;
    double weight = 0.0;
    uint8_t tier_hint = 0;   // [NEW] 0=DRAM, 1=CXL, 2=SSD

    void set_edge(uint64_t s, uint64_t d, double w) {
        source = s; destination = d; weight = w;
    }
    void set_edge(const StreamEdge& other) {
        source = other.source; 
        destination = other.destination; 
        weight = other.weight;
        tier_hint = other.tier_hint;
    }

    bool operator<(const StreamEdge& o) const {
        if (source != o.source) return source < o.source;
        if (destination != o.destination) return destination < o.destination;
        return weight < o.weight;
    }
    bool operator==(const StreamEdge& o) const {
        return source == o.source && destination == o.destination;
    }
};

// ─── 流统计 [NEW] ───────────────────────────────────────────────────
struct StreamStats {
    uint64_t edge_count = 0;
    uint64_t unique_vertices = 0;
    uint64_t max_degree = 0;
    uint64_t min_degree = UINT64_MAX;
    double avg_degree = 0.0;
    uint64_t tier_counts[3] = {0, 0, 0};
};

// ─── EdgeStream (upstream骨架, 修改20%) ─────────────────────────────
class FileEdgeStream {
    std::vector<StreamEdge> edge_stream_;
    int index_ = 0;

public:
    FileEdgeStream() = default;

    // ─── upstream load_stream (modified: progress tracking) ─────
    void load_stream(const std::string& file_path) {
        std::ifstream fin(file_path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[STREAM] ERROR: Cannot open %s\n", file_path.c_str());
            return;
        }

        std::string line;
        uint64_t loaded = 0;
        uint64_t skipped = 0;

        std::fprintf(stderr, "[STREAM] Loading edges from: %s\n", file_path.c_str());
        auto t_start = std::chrono::steady_clock::now();

        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#' || line[0] == '%') {
                skipped++;
                continue;
            }

            std::istringstream ss(line);
            uint64_t src, dst;
            double w = 1.0;

            if (!(ss >> src) || !(ss >> dst)) {
                skipped++;
                continue;
            }
            ss >> w;  // weight可选

            StreamEdge e;
            e.set_edge(src, dst, w);
            edge_stream_.push_back(e);
            loaded++;

            // [NEW] 每100K条边打印进度
            if (loaded % 100000 == 0) {
                auto now = std::chrono::steady_clock::now();
                double secs = std::chrono::duration<double>(now - t_start).count();
                std::fprintf(stderr, "[STREAM] %luK edges loaded (%.1f sec, %.0f edges/sec)\n",
                             loaded / 1000, secs, loaded / secs);
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        double total_secs = std::chrono::duration<double>(t_end - t_start).count();
        std::fprintf(stderr, "[STREAM] Done: %lu edges loaded, %lu skipped (%.2f sec)\n",
                     loaded, skipped, total_secs);
    }

    // ─── upstream permute_stream (100%) ─────────────────────────
    void permute_stream() {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(edge_stream_.begin(), edge_stream_.end(), 
                     std::default_random_engine(seed));
        std::fprintf(stderr, "[STREAM] Permuted %zu edges\n", edge_stream_.size());
    }

    // ─── upstream sort (100%) ───────────────────────────────────
    void sort() {
        std::sort(edge_stream_.begin(), edge_stream_.end());
    }

    // ─── upstream remove_duplicates (100%) ──────────────────────
    void remove_duplicates() {
        sort();
        auto before = edge_stream_.size();
        edge_stream_.erase(
            std::unique(edge_stream_.begin(), edge_stream_.end()),
            edge_stream_.end());
        std::fprintf(stderr, "[STREAM] remove_duplicates: %zu → %zu\n",
                     before, edge_stream_.size());
    }

    // ─── upstream get_next_edge (100%) ──────────────────────────
    bool get_next_edge(StreamEdge& edge) {
        if (index_ >= static_cast<int>(edge_stream_.size())) return false;
        edge.set_edge(edge_stream_[index_++]);
        return true;
    }

    StreamEdge& operator[](int idx) {
        return edge_stream_[idx];
    }

    int get_size() const { return static_cast<int>(edge_stream_.size()); }
    int get_current_index() const { return index_; }
    void reset_index() { index_ = 0; }

    // ─── upstream reorder_and_partition (100% + debug) ──────────
    void reorder_and_partition(bool high_degree_partition) {
        std::unordered_map<uint64_t, int> degree_map;
        for (auto& e : edge_stream_) {
            degree_map[e.source]++;
            degree_map[e.destination]++;
        }

        std::sort(edge_stream_.begin(), edge_stream_.end(),
            [&degree_map, high_degree_partition](const StreamEdge& a, const StreamEdge& b) {
                int da = std::max(degree_map[a.source], degree_map[a.destination]);
                int db = std::max(degree_map[b.source], degree_map[b.destination]);
                return high_degree_partition ? da > db : da < db;
            });

        int num_pick = static_cast<int>(edge_stream_.size() * 0.10);
        std::vector<StreamEdge> picked(edge_stream_.begin(), 
                                       edge_stream_.begin() + num_pick);
        picked.insert(picked.end(), 
                      edge_stream_.begin() + num_pick, edge_stream_.end());
        edge_stream_ = std::move(picked);
        reset_index();
        remove_duplicates();

        std::fprintf(stderr, "[STREAM] reorder_and_partition(high=%d): %zu edges remain\n",
                     high_degree_partition, edge_stream_.size());
    }

    // ─── [NEW] tier_partition: 按degree分3层 ────────────────────
    void tier_partition() {
        std::unordered_map<uint64_t, uint64_t> degree_map;
        for (auto& e : edge_stream_) {
            degree_map[e.source]++;
            degree_map[e.destination]++;
        }

        // 计算degree分位点
        std::vector<uint64_t> degrees;
        degrees.reserve(degree_map.size());
        for (auto& [v, d] : degree_map) degrees.push_back(d);
        std::sort(degrees.begin(), degrees.end());
        
        uint64_t p33 = degrees.empty() ? 0 : degrees[degrees.size() / 3];
        uint64_t p66 = degrees.empty() ? 0 : degrees[degrees.size() * 2 / 3];

        uint64_t tier_counts[3] = {0, 0, 0};
        for (auto& e : edge_stream_) {
            uint64_t max_deg = std::max(degree_map[e.source], 
                                         degree_map[e.destination]);
            if (max_deg >= p66) {
                e.tier_hint = 0;  // hot → DRAM
                tier_counts[0]++;
            } else if (max_deg >= p33) {
                e.tier_hint = 1;  // warm → CXL
                tier_counts[1]++;
            } else {
                e.tier_hint = 2;  // cold → SSD
                tier_counts[2]++;
            }
        }

        std::fprintf(stderr, "[STREAM] tier_partition: DRAM=%lu CXL=%lu SSD=%lu (p33=%lu p66=%lu)\n",
                     tier_counts[0], tier_counts[1], tier_counts[2], p33, p66);
    }

    // ─── [NEW] stream_stats ─────────────────────────────────────
    StreamStats compute_stats() const {
        StreamStats st;
        st.edge_count = edge_stream_.size();
        std::unordered_map<uint64_t, uint64_t> deg;
        for (auto& e : edge_stream_) {
            deg[e.source]++;
            deg[e.destination]++;
            st.tier_counts[e.tier_hint]++;
        }
        st.unique_vertices = deg.size();
        for (auto& [v, d] : deg) {
            st.max_degree = std::max(st.max_degree, d);
            st.min_degree = std::min(st.min_degree, d);
        }
        st.avg_degree = st.unique_vertices > 0 
            ? (2.0 * st.edge_count / st.unique_vertices) : 0;
        return st;
    }

    void dump_state(const char* label = "EdgeStream") const {
        auto st = compute_stats();
        std::fprintf(stderr, 
            "[BREAKPOINT] %s: edges=%lu vertices=%lu index=%d "
            "degree=[%lu,%lu,avg=%.1f] tiers=[%lu,%lu,%lu]\n",
            label, st.edge_count, st.unique_vertices, index_,
            st.min_degree, st.max_degree, st.avg_degree,
            st.tier_counts[0], st.tier_counts[1], st.tier_counts[2]);
    }

    // ─── 内存加载接口 (配合合成数据) ────────────────────────────
    void add_edge(uint64_t src, uint64_t dst, double w = 1.0) {
        StreamEdge e;
        e.set_edge(src, dst, w);
        edge_stream_.push_back(e);
    }

    const std::vector<StreamEdge>& edges() const { return edge_stream_; }
    std::vector<StreamEdge>& edges() { return edge_stream_; }
};

#define BREAKPOINT_STREAM(stream) \
    (stream).dump_state(#stream)

} // namespace io
} // namespace philemon

#endif // PHILEMON_EDGE_STREAM_FILE_IO_HPP
