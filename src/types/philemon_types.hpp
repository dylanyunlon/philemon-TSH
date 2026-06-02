#ifndef PHILEMON_TYPES_HPP
#define PHILEMON_TYPES_HPP
/**
 * philemon_types.hpp — 全局类型系统
 *
 * 骨架来源: upstream/rapidstore/types/types.hpp (全文)
 *           upstream/rapidstore/dataset_preprocessor/types.hpp (扩展DriverConfig)
 * 修改 (~20%):
 *   - [MOD] 包裹进 philemon:: namespace (upstream用裸全局typedef)
 *   - [MOD] concurrent_workload → TieredWorkload, 增加tier_hint字段
 *   - [NEW] 每个struct加 dump_state() 方法: 打印当前全部字段到stderr
 *   - [NEW] PHILE_DUMP_OP(op) 宏: 一行打印operation的类型+边
 *   - [NEW] DriverConfig增加tier相关字段(hbm_ratio, prefetch_depth)
 *   - [KEEP] operationType enum 100%保留全部枚举值
 *   - [KEEP] operation struct 布局与upstream二进制兼容
 *   - [KEEP] Config/EdgeDriverConfig/DriverConfig字段100%保留
 *   - [KEEP] Iterator模板100%保留
 *   - [MOVE] generate_path_type/generate_path_ts → preprocessor模块
 *
 * Milestone: M027+ (第3位Claude)
 */

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

#include "../wrapper/graph_edge.hpp"

namespace philemon {

// ─── Core typedefs (upstream verbatim) ──────────────────────────────
using vertexID = uint64_t;
using label    = uint8_t;

// ─── Operation types (upstream 100% preserved) ──────────────────────
enum class operationType {
    INSERT,
    DELETE,
    INSERT_VERTEX,
    UPDATE,
    BATCH_INSERT,          // from preprocessor types
    GET_VERTEX,
    GET_EDGE,
    GET_WEIGHT,
    SCAN_NEIGHBOR,
    GET_NEIGHBOR,
    PHYSICAL2LOGICAL,
    LOGICAL2PHYSICAL,
    BFS,
    PAGE_RANK,
    SSSP,
    TC,
    TC_OP,
    WCC,
    QUERY,
    MIXED,
    QOS,
    CONCURRENT             // from preprocessor types
};

inline const char* op_name(operationType t) {
    switch(t) {
        case operationType::INSERT:          return "INSERT";
        case operationType::DELETE:          return "DELETE";
        case operationType::INSERT_VERTEX:   return "INSERT_VERTEX";
        case operationType::UPDATE:          return "UPDATE";
        case operationType::BATCH_INSERT:    return "BATCH_INSERT";
        case operationType::GET_VERTEX:      return "GET_VERTEX";
        case operationType::GET_EDGE:        return "GET_EDGE";
        case operationType::GET_WEIGHT:      return "GET_WEIGHT";
        case operationType::SCAN_NEIGHBOR:   return "SCAN_NEIGHBOR";
        case operationType::GET_NEIGHBOR:    return "GET_NEIGHBOR";
        case operationType::BFS:             return "BFS";
        case operationType::PAGE_RANK:       return "PAGE_RANK";
        case operationType::SSSP:            return "SSSP";
        case operationType::TC:              return "TC";
        case operationType::TC_OP:           return "TC_OP";
        case operationType::WCC:             return "WCC";
        case operationType::QUERY:           return "QUERY";
        case operationType::MIXED:           return "MIXED";
        case operationType::QOS:             return "QOS";
        case operationType::CONCURRENT:      return "CONCURRENT";
        default:                             return "UNKNOWN";
    }
}

// ─── operation struct (binary-compatible with upstream) ──────────────
struct operation {
    operationType type;
    driver::graph::weightedEdge e;

    // NEW: debug dump
    void dump_state(const char* tag = "") const {
        std::fprintf(stderr, "[OP-DUMP] %s type=%s src=%lu dst=%lu w=%.4f\n",
                     tag, op_name(type),
                     (unsigned long)e.source,
                     (unsigned long)e.destination, e.weight);
    }
};

// ─── Debug macro: 打印operation一行 ────────────────────────────────
#define PHILE_DUMP_OP(op) do { \
    (op).dump_state(__func__); \
} while(0)

// ─── Stream types (upstream 100% preserved) ─────────────────────────
enum class targetStreamType {
    FULL,
    GENERAL,
    HIGH_DEGREE,
    LOW_DEGREE,
    UNIFORM,
    BASED_ON_DEGREE
};

inline const char* stream_type_name(targetStreamType t) {
    switch(t) {
        case targetStreamType::FULL:            return "FULL";
        case targetStreamType::GENERAL:         return "GENERAL";
        case targetStreamType::HIGH_DEGREE:     return "HIGH_DEGREE";
        case targetStreamType::LOW_DEGREE:      return "LOW_DEGREE";
        case targetStreamType::UNIFORM:         return "UNIFORM";
        case targetStreamType::BASED_ON_DEGREE: return "BASED_ON_DEGREE";
        default:                                return "UNKNOWN";
    }
}

// ─── TieredWorkload (upstream: concurrent_workload + tier_hint) ─────
struct TieredWorkload {
    operationType workload_type;
    targetStreamType target_stream_type;
    int num_threads;
    int tier_hint = -1;  // NEW: -1=auto, 0=HBM, 1=GDDR, 2=DRAM

    void dump_state(const char* tag = "") const {
        std::fprintf(stderr, "[WORKLOAD] %s type=%s stream=%s threads=%d tier_hint=%d\n",
                     tag, op_name(workload_type),
                     stream_type_name(target_stream_type),
                     num_threads, tier_hint);
    }
};

// ─── Config (upstream 100%) ─────────────────────────────────────────
struct Config {
    double timestamp_rate;
    int seed;
    uint64_t num_search{1000000};
    bool test_version_chain{false};
    bool enable_bloom_filter{false};

    Config(double ts_rate) : timestamp_rate(ts_rate), seed(0) {}
    Config() : timestamp_rate(0.0), seed(0) {}

    void dump_state() const {
        std::fprintf(stderr, "[CONFIG] ts_rate=%.4f seed=%d num_search=%lu\n",
                     timestamp_rate, seed, (unsigned long)num_search);
    }
};

// ─── EdgeDriverConfig (upstream 100% + tier fields) ─────────────────
struct EdgeDriverConfig {
    std::string workload_dir;
    std::string output_dir;
    operationType workload_type{operationType::INSERT};
    targetStreamType target_stream_type{targetStreamType::FULL};
    std::vector<operationType> mb_operation_types;
    std::vector<targetStreamType> mb_ts_types;

    double initial_graph_rate{0.8};
    double version_rate{0.8};
    double timestamp_rate{0.8};

    std::vector<int> element_sizes;
    uint64_t neighbor_size{1024};
    uint64_t num_of_vertices{1024};
    bool is_shuffle{false};
    int seed{0};

    uint64_t num_search{1000000};
    uint64_t num_scan{1000000};
    int repeat_times{0};

    bool test_version_chain{false};
    int version_chain_length{0};
    bool is_real_graph{false};

    // NEW: tier parameters
    double hbm_capacity_ratio{0.15};
    int    prefetch_lookahead{4};

    void dump_state() const {
        std::fprintf(stderr,
            "[EDGE-DRIVER-CFG] dir=%s out=%s type=%s vertices=%lu "
            "hbm_ratio=%.2f prefetch=%d\n",
            workload_dir.c_str(), output_dir.c_str(),
            op_name(workload_type),
            (unsigned long)num_of_vertices,
            hbm_capacity_ratio, prefetch_lookahead);
    }
};

// ─── DriverConfig (upstream 100% + tier fields) ─────────────────────
struct DriverConfig {
    std::string workload_dir;
    std::string output_dir;
    operationType workload_type{operationType::INSERT};
    targetStreamType target_stream_type{targetStreamType::FULL};

    // insert/delete
    uint64_t insert_delete_checkpoint_size{10000};
    int insert_delete_num_threads{1};
    uint64_t insert_batch_size{1};

    // update
    uint64_t update_checkpoint_size{10000};
    int update_num_threads{1};
    int update_repeat_times{10};

    // microbenchmark
    int repeat_times{0};
    uint64_t mb_checkpoint_size{10000};
    std::vector<int> microbenchmark_num_threads;
    std::vector<operationType> mb_operation_types;
    std::vector<targetStreamType> mb_ts_types;

    // query
    std::vector<int> query_num_threads;
    std::vector<operationType> query_operation_types;

    // algorithm params (upstream 100%)
    int alpha{15};
    int beta{18};
    uint64_t bfs_source{0};
    double delta{2.0};
    uint64_t sssp_source{0};
    int num_iterations{10};
    double damping_factor{0.85};

    // mixed
    int writer_threads{16};
    int reader_threads{16};

    // qos
    int num_threads_search{8};
    int num_threads_scan{20};

    // concurrent workloads
    std::vector<TieredWorkload> concurrent_workloads;

    // NEW: tiered memory config
    double hbm_capacity_gb{4.0};
    double gddr_capacity_gb{16.0};
    int    migration_batch_size{1024};
    bool   enable_tier_trace{true};

    void dump_state() const {
        std::fprintf(stderr,
            "[DRIVER-CFG] dir=%s threads=%d bfs_src=%lu sssp_src=%lu "
            "alpha=%d beta=%d delta=%.2f "
            "hbm=%.1fGB gddr=%.1fGB mig_batch=%d\n",
            workload_dir.c_str(), insert_delete_num_threads,
            (unsigned long)bfs_source, (unsigned long)sssp_source,
            alpha, beta, delta,
            hbm_capacity_gb, gddr_capacity_gb, migration_batch_size);
    }
};

// ─── Binary stream I/O (upstream 100%) ──────────────────────────────
inline void read_binary_stream(const std::string& path,
                               std::vector<operation>& stream) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[STREAM-READ] FAIL: cannot open %s\n", path.c_str());
        return;
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t n = file_size / sizeof(operation);
    stream.resize(n);
    file.read(reinterpret_cast<char*>(stream.data()), n * sizeof(operation));
    file.close();

    // DEBUG: print first 3 and last 1 operations
    std::fprintf(stderr, "[STREAM-READ] loaded %lu ops from %s\n",
                 (unsigned long)n, path.c_str());
    for (size_t i = 0; i < std::min<size_t>(3, n); i++) {
        stream[i].dump_state("  head");
    }
    if (n > 3) {
        stream[n-1].dump_state("  tail");
    }
}

// ─── Iterator (upstream 100%) ───────────────────────────────────────
template <typename T>
struct Iterator {
    T iterator;
    Iterator(const T& it) : iterator(it) {}

    bool is_valid() const { return iterator.valid(); }

    Iterator& operator++() {
        ++iterator;
        return *this;
    }

    uint64_t operator*() { return iterator->dest; }
};

}  // namespace philemon

// Backward-compatible global typedefs for upstream code
using vertexID = philemon::vertexID;

#endif  // PHILEMON_TYPES_HPP
