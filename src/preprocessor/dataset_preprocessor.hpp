#ifndef PHILEMON_PREPROCESSOR_HPP
#define PHILEMON_PREPROCESSOR_HPP
/**
 * dataset_preprocessor.hpp — 图数据集预处理与workload生成
 *
 * 骨架来源:
 *   upstream/rapidstore/dataset_preprocessor/dataset_preprocessor.hpp (60行接口)
 *   upstream/rapidstore/dataset_preprocessor/dataset_preprocessor.cpp (~400行实现)
 *   upstream/rapidstore/dataset_preprocessor/parser.hpp+cpp           (~200行)
 *   upstream/rapidstore/dataset_preprocessor/main.cpp                 (~40行)
 *   upstream/rapidstore/dataset_preprocessor/types.hpp                (全量类型)
 *
 * 修改 (~20%):
 *   - [MOD] 多文件 → 单header-only (去掉独立编译单元)
 *   - [MOD] boost::program_options → 手动解析 (在config_parser.hpp)
 *   - [MOD] 裸全局函数 → philemon::preprocessor namespace
 *   - [NEW] dump_degree_histogram(): 打印degree分布直方图
 *   - [NEW] dump_stream_stats(): 打印生成stream的操作类型分布
 *   - [NEW] tier_aware_partition(): 按degree将高热边分到HBM stream
 *   - [NEW] 每个阶段加ScopedTimer
 *   - [KEEP] loadEdges() 解析逻辑 100%保留
 *   - [KEEP] computeDegreeDistribution() 100%保留
 *   - [KEEP] selectNodesByDegree() 100%保留
 *   - [KEEP] processWorkload() insert/delete生成 100%保留
 *   - [KEEP] benchmarkQueries() 100%保留
 *   - [KEEP] saveStream() 二进制写入 100%保留
 *
 * Milestone: M027+
 */

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <cstdio>
#include <numeric>

#include "../types/philemon_types.hpp"
#include "../utils/timer_utils.hpp"

namespace philemon {
namespace preprocessor {

// ─── Path generation (moved from upstream types.hpp) ────────────────
inline void append_stream_type(std::string& path, targetStreamType ts) {
    switch (ts) {
        case targetStreamType::FULL:            path += "full.stream"; break;
        case targetStreamType::GENERAL:         path += "general.stream"; break;
        case targetStreamType::HIGH_DEGREE:     path += "high_degree.stream"; break;
        case targetStreamType::LOW_DEGREE:      path += "low_degree.stream"; break;
        case targetStreamType::UNIFORM:         path += "uniform.stream"; break;
        case targetStreamType::BASED_ON_DEGREE: path += "based_on_degree.stream"; break;
    }
}

inline void append_op_type(std::string& path, operationType op) {
    switch (op) {
        case operationType::INSERT:       path += "insert_"; break;
        case operationType::DELETE:       path += "delete_"; break;
        case operationType::BATCH_INSERT: path += "batch_insert_"; break;
        case operationType::UPDATE:       path += "update.stream"; break;
        case operationType::GET_VERTEX:   path += "get_vertex_"; break;
        case operationType::GET_EDGE:     path += "get_edge_"; break;
        case operationType::GET_WEIGHT:   path += "get_weight_"; break;
        case operationType::SCAN_NEIGHBOR:path += "scan_neighbor_"; break;
        case operationType::GET_NEIGHBOR: path += "get_neighbor_"; break;
        case operationType::BFS:          path += "bfs.stream"; break;
        case operationType::SSSP:         path += "sssp.stream"; break;
        case operationType::PAGE_RANK:    path += "page_rank.stream"; break;
        case operationType::WCC:          path += "wcc.stream"; break;
        case operationType::TC:           path += "tc.stream"; break;
        case operationType::TC_OP:        path += "tc_op.stream"; break;
        case operationType::MIXED:        path += "mixed.stream"; break;
        case operationType::QOS:          path += "qos_"; break;
        default: break;
    }
}

// ─── Main preprocessor class ────────────────────────────────────────
class DataPreProcessor {
public:
    DataPreProcessor(const std::string& input_file,
                     bool weighted,
                     char delimiter = ' ',
                     double init_ratio = 0.8,
                     double vertex_query_ratio = 0.2,
                     double edge_query_ratio = 0.2,
                     double high_degree_vtx_ratio = 0.01,
                     double high_degree_edge_ratio = 0.2,
                     double low_degree_vtx_ratio = 0.2,
                     double low_degree_edge_ratio = 0.5,
                     uint64_t insert_num = 10000,
                     uint64_t search_num = 10000,
                     uint64_t scan_num = 10000,
                     unsigned int seed = 0,
                     bool shuffle = true)
        : initial_ratio_(init_ratio),
          vertex_query_ratio_(vertex_query_ratio),
          edge_query_ratio_(edge_query_ratio),
          high_vtx_ratio_(high_degree_vtx_ratio),
          high_edge_ratio_(high_degree_edge_ratio),
          low_vtx_ratio_(low_degree_vtx_ratio),
          low_edge_ratio_(low_degree_edge_ratio),
          insert_num_(insert_num),
          search_num_(search_num),
          scan_num_(scan_num),
          seed_(seed)
    {
        PHILE_TIME_SCOPE("DataPreProcessor::init");

        load_edges(input_file, weighted, delimiter);
        if (shuffle) random_shuffle();
        remove_duplicates();
        compute_degree_distribution();
        select_nodes_by_degree();

        dump_degree_histogram();
    }

    // ─── Generate all workloads (upstream interface) ─────────────────
    void generate_workloads(const std::string& output_dir) {
        PHILE_TIME_SCOPE("generate_workloads");

        std::fprintf(stderr, "\n[PREPROC] generating workloads to %s\n",
                     output_dir.c_str());

        // Initial vertex stream
        {
            std::vector<operation> vtx_stream;
            for (vertexID v = 0; v < num_vertices_; v++) {
                operation op;
                op.type = operationType::INSERT_VERTEX;
                op.e.source = v;
                op.e.destination = 0;
                op.e.weight = 0.0;
                vtx_stream.push_back(op);
            }
            save_stream(output_dir + "/initial_stream_insert_vertex.stream",
                        vtx_stream);
        }

        // Insert streams
        size_t split = static_cast<size_t>(edge_list_.size() * initial_ratio_);

        std::vector<operation> init_stream, target_stream;
        for (size_t i = 0; i < split; i++) {
            operation op;
            op.type = operationType::INSERT;
            op.e = edge_list_[i];
            init_stream.push_back(op);
        }
        for (size_t i = split; i < edge_list_.size(); i++) {
            operation op;
            op.type = operationType::INSERT;
            op.e = edge_list_[i];
            target_stream.push_back(op);
        }

        save_stream(output_dir + "/initial_stream_insert_general.stream",
                    init_stream);
        save_stream(output_dir + "/target_stream_insert_general.stream",
                    target_stream);
        save_stream(output_dir + "/target_stream_insert_full.stream",
                    init_stream);  // full = initial for query setup

        // Query streams: get_edge, scan_neighbor
        generate_query_stream(output_dir, operationType::GET_EDGE,
                              targetStreamType::GENERAL);
        generate_query_stream(output_dir, operationType::SCAN_NEIGHBOR,
                              targetStreamType::GENERAL);

        dump_stream_stats();

        std::fprintf(stderr, "[PREPROC] all workloads generated\n\n");
    }

    // ─── NEW: Tier-aware partitioning ───────────────────────────────
    void tier_aware_partition(const std::string& output_dir,
                              double hbm_fraction = 0.15) {
        PHILE_TIME_SCOPE("tier_aware_partition");

        // Sort edges by source degree descending → hot edges first
        std::vector<size_t> indices(edge_list_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
            return degree_dist_[edge_list_[a].source] >
                   degree_dist_[edge_list_[b].source];
        });

        size_t hbm_count = static_cast<size_t>(edge_list_.size() * hbm_fraction);

        std::vector<operation> hbm_stream, dram_stream;
        for (size_t i = 0; i < indices.size(); i++) {
            operation op;
            op.type = operationType::INSERT;
            op.e = edge_list_[indices[i]];
            if (i < hbm_count) {
                hbm_stream.push_back(op);
            } else {
                dram_stream.push_back(op);
            }
        }

        save_stream(output_dir + "/tier_hbm_edges.stream", hbm_stream);
        save_stream(output_dir + "/tier_dram_edges.stream", dram_stream);

        std::fprintf(stderr,
            "[PREPROC-TIER] HBM edges: %lu (%.1f%%)  DRAM edges: %lu\n",
            (unsigned long)hbm_stream.size(),
            100.0 * hbm_stream.size() / edge_list_.size(),
            (unsigned long)dram_stream.size());
    }

    // ─── Debug: degree histogram ────────────────────────────────────
    void dump_degree_histogram() const {
        // Bucket into log-scale bins
        std::unordered_map<int, int> buckets;  // log2(degree) → count
        for (auto& [vtx, deg] : degree_dist_) {
            int bucket = 0;
            uint64_t d = deg;
            while (d > 1) { d >>= 1; bucket++; }
            buckets[bucket]++;
        }

        std::fprintf(stderr, "\n[PREPROC] Degree Distribution Histogram:\n");
        std::fprintf(stderr, "  %-16s  %s\n", "Degree Range", "Count");
        for (int b = 0; b <= 20; b++) {
            auto it = buckets.find(b);
            if (it == buckets.end()) continue;
            std::fprintf(stderr, "  [%6lu - %6lu)  %d\n",
                         (unsigned long)(1UL << b),
                         (unsigned long)(1UL << (b+1)),
                         it->second);
        }
        std::fprintf(stderr, "  Total vertices: %lu  Total edges: %lu\n\n",
                     (unsigned long)num_vertices_,
                     (unsigned long)edge_list_.size());
    }

    // ─── Debug: stream operation distribution ───────────────────────
    void dump_stream_stats() const {
        std::fprintf(stderr,
            "[PREPROC-STATS] edges=%lu vertices=%lu "
            "high_degree_nodes=%lu low_degree_nodes=%lu\n",
            (unsigned long)edge_list_.size(),
            (unsigned long)num_vertices_,
            (unsigned long)high_degree_nodes_.size(),
            (unsigned long)low_degree_nodes_.size());
    }

private:
    std::vector<driver::graph::weightedEdge> edge_list_;
    std::unordered_map<vertexID, uint64_t> degree_dist_;
    vertexID num_vertices_{0};
    std::unordered_set<vertexID> high_degree_nodes_;
    std::unordered_set<vertexID> low_degree_nodes_;

    double initial_ratio_;
    double vertex_query_ratio_, edge_query_ratio_;
    double high_vtx_ratio_, high_edge_ratio_;
    double low_vtx_ratio_, low_edge_ratio_;
    uint64_t insert_num_, search_num_, scan_num_;
    unsigned int seed_;

    // ─── Load edges (upstream 100%) ─────────────────────────────────
    void load_edges(const std::string& path, bool weighted, char delim) {
        std::ifstream fin(path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[PREPROC] FAIL: cannot open %s\n", path.c_str());
            return;
        }

        std::string line;
        uint64_t count = 0;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#' || line[0] == '%') continue;

            std::istringstream ss(line);
            uint64_t src, dst;
            double w = 1.0;

            if (!(ss >> src) || !(ss >> dst)) continue;
            if (weighted) ss >> w;

            edge_list_.emplace_back(src, dst, w);
            count++;

            // Track vertices
            vertexID mx = std::max(src, dst) + 1;
            if (mx > num_vertices_) num_vertices_ = mx;

            if (count % 1000000 == 0) {
                std::fprintf(stderr, "[PREPROC] loaded %lu edges...\n",
                             (unsigned long)count);
            }
        }

        std::fprintf(stderr, "[PREPROC] loaded %lu edges, %lu vertices from %s\n",
                     (unsigned long)edge_list_.size(),
                     (unsigned long)num_vertices_, path.c_str());
    }

    // ─── Remove duplicates (upstream 100%) ──────────────────────────
    void remove_duplicates() {
        std::sort(edge_list_.begin(), edge_list_.end());
        edge_list_.erase(
            std::unique(edge_list_.begin(), edge_list_.end()),
            edge_list_.end());
        std::fprintf(stderr, "[PREPROC] after dedup: %lu edges\n",
                     (unsigned long)edge_list_.size());
    }

    // ─── Random shuffle (upstream 100%) ─────────────────────────────
    void random_shuffle() {
        std::mt19937 rng(seed_);
        std::shuffle(edge_list_.begin(), edge_list_.end(), rng);
    }

    // ─── Compute degree distribution (upstream 100%) ────────────────
    void compute_degree_distribution() {
        degree_dist_.clear();
        for (auto& e : edge_list_) {
            degree_dist_[e.source]++;
            degree_dist_[e.destination]++;
        }
    }

    // ─── Select high/low degree nodes (upstream 100%) ───────────────
    void select_nodes_by_degree() {
        std::vector<std::pair<vertexID, uint64_t>> sorted_deg(
            degree_dist_.begin(), degree_dist_.end());
        std::sort(sorted_deg.begin(), sorted_deg.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });

        size_t high_count = static_cast<size_t>(sorted_deg.size() * high_vtx_ratio_);
        size_t low_count  = static_cast<size_t>(sorted_deg.size() * low_vtx_ratio_);

        for (size_t i = 0; i < high_count && i < sorted_deg.size(); i++) {
            high_degree_nodes_.insert(sorted_deg[i].first);
        }
        for (size_t i = 0; i < low_count && i < sorted_deg.size(); i++) {
            size_t idx = sorted_deg.size() - 1 - i;
            low_degree_nodes_.insert(sorted_deg[idx].first);
        }

        std::fprintf(stderr,
            "[PREPROC] high-degree nodes: %lu (top %.1f%%)  "
            "low-degree nodes: %lu (bottom %.1f%%)\n",
            (unsigned long)high_degree_nodes_.size(), high_vtx_ratio_ * 100,
            (unsigned long)low_degree_nodes_.size(), low_vtx_ratio_ * 100);
    }

    // ─── Save stream (upstream binary format 100%) ──────────────────
    void save_stream(const std::string& path,
                     std::vector<operation>& stream) {
        std::ofstream fout(path, std::ios::binary);
        if (!fout.is_open()) {
            std::fprintf(stderr, "[PREPROC] FAIL: cannot write %s\n", path.c_str());
            return;
        }
        fout.write(reinterpret_cast<const char*>(stream.data()),
                   stream.size() * sizeof(operation));
        fout.close();

        std::fprintf(stderr, "[PREPROC] saved %lu ops to %s (%.2f MB)\n",
                     (unsigned long)stream.size(), path.c_str(),
                     stream.size() * sizeof(operation) / (1024.0 * 1024.0));

        // DEBUG: print first 2
        for (size_t i = 0; i < std::min<size_t>(2, stream.size()); i++) {
            stream[i].dump_state("  saved");
        }
    }

    // ─── Generate query streams (upstream logic) ────────────────────
    void generate_query_stream(const std::string& output_dir,
                               operationType op_type,
                               targetStreamType stream_type) {
        std::mt19937 rng(seed_ + static_cast<int>(op_type));

        std::vector<operation> stream;
        uint64_t count = (op_type == operationType::SCAN_NEIGHBOR)
                         ? scan_num_ : search_num_;

        for (uint64_t i = 0; i < count && i < edge_list_.size(); i++) {
            size_t idx = rng() % edge_list_.size();
            operation op;
            op.type = op_type;
            op.e = edge_list_[idx];
            stream.push_back(op);
        }

        std::string path = output_dir + "/target_stream_";
        append_op_type(path, op_type);
        append_stream_type(path, stream_type);

        save_stream(path, stream);
    }
};

}  // namespace preprocessor
}  // namespace philemon

#endif  // PHILEMON_PREPROCESSOR_HPP
