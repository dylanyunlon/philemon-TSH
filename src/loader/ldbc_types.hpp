#ifndef PHILEMON_LDBC_TYPES_HPP
#define PHILEMON_LDBC_TYPES_HPP
/**
 * ldbc_types.hpp — Type definitions for LDBC SNB temporal graph loading
 *
 * 骨架来源: upstream/rapidstore/dataset_preprocessor/types.hpp (284行)
 * 修改 (~20%):
 *   - 增加 timestamp 字段到 weightedEdge → TemporalEdge
 *   - 增加 TierHint enum 用于按 degree/recency 分层
 *   - 增加 PartitionHint 结构体用于自适应阈值校准
 *   - 保留 operationType, targetStreamType, Config 等原始定义
 *   - 增加 PHILE_DBG 状态打印宏
 *
 * Milestone: M017 — LDBC SNB loader types
 */

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace loader {

typedef uint64_t vertexID;
typedef uint8_t label;

// ─── Core edge type (upstream weightedEdge + temporal extension) ─────
// Original upstream: struct weightedEdge { source, destination, weight }
// We add timestamp for temporal graph support.
struct TemporalEdge {
    vertexID source;
    vertexID destination;
    double   weight;
    uint64_t timestamp;    // NEW: LDBC temporal edge arrival time

    TemporalEdge() : source(0), destination(0), weight(0.0), timestamp(0) {}
    TemporalEdge(uint64_t src, uint64_t dst, double w)
        : source(src), destination(dst), weight(w), timestamp(0) {}
    TemporalEdge(uint64_t src, uint64_t dst, double w, uint64_t ts)
        : source(src), destination(dst), weight(w), timestamp(ts) {}
    TemporalEdge(uint64_t src, uint64_t dst)
        : source(src), destination(dst), weight(0.0), timestamp(0) {}

    // ─── Debug: print edge state ─────────────────────────────────
    void dump(const char* tag = "edge") const {
        PHILE_DBG(3, "[%s] src=%lu dst=%lu w=%.4f ts=%lu",
                  tag, (unsigned long)source, (unsigned long)destination,
                  weight, (unsigned long)timestamp);
    }
};

// ─── Tier hint (NEW: which memory tier an edge should initially land in) ─
enum class TierHint : uint8_t {
    HBM   = 0,   // Hot: high-degree + recent
    GDDR  = 1,   // Warm: medium-degree or semi-recent
    DRAM  = 2,   // Cold: low-degree + old
    AUTO  = 3,   // Let the system decide
};

inline const char* tier_hint_name(TierHint h) {
    switch (h) {
        case TierHint::HBM:  return "HBM";
        case TierHint::GDDR: return "GDDR";
        case TierHint::DRAM: return "DRAM";
        case TierHint::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

// ─── Partition hint for adaptive threshold calibration (NEW) ─────────
struct PartitionHint {
    double   density_threshold;     // edges/vertex ratio for splitting
    uint64_t min_partition_edges;   // minimum edges per partition
    uint64_t max_partition_edges;   // maximum edges per partition
    double   hot_fraction;          // fraction considered "hot" (→ HBM)
    double   warm_fraction;         // fraction considered "warm" (→ GDDR)

    PartitionHint()
        : density_threshold(10.0),
          min_partition_edges(1024),
          max_partition_edges(1048576),
          hot_fraction(0.1),
          warm_fraction(0.3) {}

    // ─── Debug: print all thresholds ─────────────────────────────
    void dump() const {
        PHILE_DBG(1, "[PartitionHint] density_thresh=%.2f min_edges=%lu "
                  "max_edges=%lu hot=%.2f warm=%.2f",
                  density_threshold,
                  (unsigned long)min_partition_edges,
                  (unsigned long)max_partition_edges,
                  hot_fraction, warm_fraction);
    }
};

// ─── Operation types (from upstream, preserved) ─────────────────────
// All enum values kept; we add TEMPORAL_QUERY and TIER_MIGRATE.
enum class operationType {
    // write operations (upstream)
    INSERT,
    DELETE,
    INSERT_VERTEX,
    UPDATE,
    // read operations (upstream)
    GET_VERTEX,
    GET_EDGE,
    GET_WEIGHT,
    SCAN_NEIGHBOR,
    GET_NEIGHBOR,
    PHYSICAL2LOGICAL,
    LOGICAL2PHYSICAL,
    // analytic operations (upstream)
    BFS,
    PAGE_RANK,
    SSSP,
    TC,
    TC_OP,
    WCC,

    QUERY,
    MIXED,
    QOS,
    CONCURRENT,
    BATCH_INSERT,

    // NEW: Philemon-TSH tier-aware operations
    TEMPORAL_QUERY,    // time-range query across tiers
    TIER_MIGRATE,      // explicit tier migration
    CROSS_TIER_BFS,    // BFS with tier-aware frontier
    CROSS_TIER_SSSP,   // SSSP with tier cost model
};

struct operation {
    operationType type;
    TemporalEdge  e;
};

// ─── Target stream type (from upstream, preserved) ──────────────────
enum class targetStreamType {
    FULL,
    GENERAL,
    HIGH_DEGREE,
    LOW_DEGREE,
    UNIFORM,
    BASED_ON_DEGREE,
    // NEW: tier-based stream types
    HOT_TIER,          // operations targeting HBM-resident data
    WARM_TIER,         // operations targeting GDDR-resident data
    COLD_TIER,         // operations targeting DRAM-resident data
};

// ─── Stream reader (from upstream, preserved, +temporal extension) ───
inline void read_temporal_stream(const std::string& path,
                                  std::vector<operation>& stream) {
    std::ifstream file(path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        size_t numElements = fileSize / sizeof(operation);
        stream.resize(numElements);
        file.read(reinterpret_cast<char*>(stream.data()),
                  numElements * sizeof(operation));

        PHILE_DBG(1, "[read_temporal_stream] loaded %zu ops from %s",
                  numElements, path.c_str());
    } else {
        PHILE_DBG(0, "[read_temporal_stream] ERROR: cannot open %s",
                  path.c_str());
    }
    file.close();
}

// ─── Config (from upstream, + tier cost model parameters) ───────────
struct LDBCConfig {
    // Upstream fields preserved
    double   timestamp_rate;
    int      seed;
    uint64_t num_search{1000000};

    // NEW: tier cost model (HBM=1ns, GDDR=5ns, DRAM=50ns)
    double   hbm_latency_ns{1.0};
    double   gddr_latency_ns{5.0};
    double   dram_latency_ns{50.0};
    double   hbm_bandwidth_gbps{3352.0};   // H100 HBM3e
    double   gddr_bandwidth_gbps{768.0};   // A6000 GDDR6X
    double   dram_bandwidth_gbps{204.8};   // DDR5-3200 dual-channel

    // Capacity limits (bytes)
    uint64_t hbm_capacity{80ULL * 1024 * 1024 * 1024};    // 80 GB
    uint64_t gddr_capacity{48ULL * 1024 * 1024 * 1024};   // 48 GB
    uint64_t dram_capacity{512ULL * 1024 * 1024 * 1024};  // 512 GB

    LDBCConfig() : timestamp_rate(0.0), seed(42) {}
    LDBCConfig(double ts_rate) : timestamp_rate(ts_rate), seed(42) {}

    // ─── Debug: print full config ────────────────────────────────
    void dump() const {
        PHILE_DBG(1, "──── LDBCConfig ────");
        PHILE_DBG(1, "  ts_rate=%.2f seed=%d num_search=%lu",
                  timestamp_rate, seed, (unsigned long)num_search);
        PHILE_DBG(1, "  HBM:  lat=%.1fns bw=%.1fGB/s cap=%luGB",
                  hbm_latency_ns, hbm_bandwidth_gbps,
                  (unsigned long)(hbm_capacity / (1024ULL*1024*1024)));
        PHILE_DBG(1, "  GDDR: lat=%.1fns bw=%.1fGB/s cap=%luGB",
                  gddr_latency_ns, gddr_bandwidth_gbps,
                  (unsigned long)(gddr_capacity / (1024ULL*1024*1024)));
        PHILE_DBG(1, "  DRAM: lat=%.1fns bw=%.1fGB/s cap=%luGB",
                  dram_latency_ns, dram_bandwidth_gbps,
                  (unsigned long)(dram_capacity / (1024ULL*1024*1024)));
        PHILE_DBG(1, "──── End Config ────");
    }
};

// ─── Degree distribution stats (for breakpoint inspection) ──────────
struct DegreeStats {
    uint64_t num_vertices;
    uint64_t num_edges;
    uint64_t max_degree;
    double   avg_degree;
    uint64_t median_degree;
    uint64_t p90_degree;     // 90th percentile
    uint64_t p99_degree;     // 99th percentile
    uint64_t high_degree_count;  // vertices with degree > threshold
    uint64_t low_degree_count;

    void dump() const {
        std::printf("──── DegreeStats ────\n");
        std::printf("  V=%lu  E=%lu  max_deg=%lu  avg=%.2f\n",
                    (unsigned long)num_vertices, (unsigned long)num_edges,
                    (unsigned long)max_degree, avg_degree);
        std::printf("  median=%lu  p90=%lu  p99=%lu\n",
                    (unsigned long)median_degree,
                    (unsigned long)p90_degree, (unsigned long)p99_degree);
        std::printf("  high_deg_vertices=%lu  low_deg_vertices=%lu\n",
                    (unsigned long)high_degree_count,
                    (unsigned long)low_degree_count);
        std::printf("──── End Stats ────\n");
    }
};

// ─── Path generation (from upstream, preserved) ─────────────────────
inline void generate_path_type(std::string& path, operationType type) {
    switch (type) {
        case operationType::INSERT:         path += "insert_"; break;
        case operationType::BATCH_INSERT:   path += "batch_insert_"; break;
        case operationType::DELETE:         path += "delete_"; break;
        case operationType::UPDATE:         path += "update.stream"; break;
        case operationType::GET_VERTEX:     path += "get_vertex_"; break;
        case operationType::GET_EDGE:       path += "get_edge_"; break;
        case operationType::GET_WEIGHT:     path += "get_weight_"; break;
        case operationType::SCAN_NEIGHBOR:  path += "scan_neighbor_"; break;
        case operationType::GET_NEIGHBOR:   path += "get_neighbor_"; break;
        case operationType::BFS:            path += "bfs.stream"; break;
        case operationType::SSSP:           path += "sssp.stream"; break;
        case operationType::PAGE_RANK:      path += "page_rank.stream"; break;
        case operationType::WCC:            path += "wcc.stream"; break;
        case operationType::TC:             path += "tc.stream"; break;
        case operationType::TC_OP:          path += "tc_op.stream"; break;
        case operationType::MIXED:          path += "mixed.stream"; break;
        // NEW tier operations
        case operationType::TEMPORAL_QUERY: path += "temporal_query.stream"; break;
        case operationType::CROSS_TIER_BFS: path += "cross_tier_bfs.stream"; break;
        case operationType::CROSS_TIER_SSSP:path += "cross_tier_sssp.stream"; break;
        default: throw std::runtime_error("Invalid operation type\n");
    }
}

}  // namespace loader
}  // namespace philemon

#endif  // PHILEMON_LDBC_TYPES_HPP
