#ifndef PHILEMON_PREPROCESSOR_TYPES_HPP
#define PHILEMON_PREPROCESSOR_TYPES_HPP
/**
 * philemon_preprocessor_types.hpp — 预处理器类型定义
 *
 * 骨架来源: upstream/rapidstore/dataset_preprocessor/types.hpp (284行)
 * 修改 (~20%):
 *   - 保留 weightedEdge, operationType, operation, targetStreamType 等
 *   - 保留 Config, EdgeDriverConfig, DriverConfig 结构
 *   - 增加 tier_hint 字段到 operation 结构
 *   - 增加 dump() 方法到每个 Config 结构
 *   - read_stream() 增加加载进度打印和字节数校验
 *   - 增加 concurrent_workload 的 dump
 *
 * Milestone: M028
 */

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdint>

typedef uint64_t vertexID;
typedef uint8_t label;

// ─── From upstream, with tier_hint added ────────────────────────────
struct preprocessor_weightedEdge {
    vertexID source;
    vertexID destination;
    double weight;
    uint8_t tier_hint = 0; // NEW: 0=auto, 1=DRAM, 2=NVM, 3=SSD

    preprocessor_weightedEdge() : source(0), destination(0), weight(0.0) {}
    preprocessor_weightedEdge(uint64_t s, uint64_t d, double w)
        : source(s), destination(d), weight(w) {}
    preprocessor_weightedEdge(uint64_t s, uint64_t d)
        : source(s), destination(d), weight(0.0) {}

    // ─── NEW: debug ─────────────────────────────────────────────────
    void dump(const char* label = "") const {
        std::printf("[EDGE] %s src=%lu dst=%lu w=%.3f tier=%d\n",
                    label, source, destination, weight, tier_hint);
    }
};

// ─── Operation Type (from upstream, with additions) ─────────────────
enum class preprocessor_operationType {
    INSERT, DELETE, INSERT_VERTEX, UPDATE,
    GET_VERTEX, GET_EDGE, GET_WEIGHT,
    SCAN_NEIGHBOR, GET_NEIGHBOR,
    PHYSICAL2LOGICAL, LOGICAL2PHYSICAL,
    BFS, PAGE_RANK, SSSP, TC, TC_OP, WCC,
    QUERY, MIXED, QOS, CONCURRENT, BATCH_INSERT,
    // ─── NEW ────────────────────────────────────────────────────────
    TIER_MIGRATE,      // cross-tier data migration
    TEMPORAL_QUERY     // time-range query
};

inline const char* op_type_name(preprocessor_operationType t) {
    switch (t) {
        case preprocessor_operationType::INSERT:         return "INSERT";
        case preprocessor_operationType::DELETE:         return "DELETE";
        case preprocessor_operationType::INSERT_VERTEX:  return "INSERT_VERTEX";
        case preprocessor_operationType::BFS:            return "BFS";
        case preprocessor_operationType::PAGE_RANK:      return "PAGE_RANK";
        case preprocessor_operationType::SSSP:           return "SSSP";
        case preprocessor_operationType::TC:             return "TC";
        case preprocessor_operationType::WCC:            return "WCC";
        case preprocessor_operationType::TIER_MIGRATE:   return "TIER_MIGRATE";
        case preprocessor_operationType::TEMPORAL_QUERY: return "TEMPORAL_QUERY";
        default: return "OTHER";
    }
}

struct preprocessor_operation {
    preprocessor_operationType type;
    preprocessor_weightedEdge e;
    uint8_t tier_hint = 0; // NEW

    void dump(const char* label = "") const {
        std::printf("[OP] %s type=%s ", label, op_type_name(type));
        e.dump("");
    }
};

enum class preprocessor_targetStreamType {
    FULL, GENERAL, HIGH_DEGREE, LOW_DEGREE, UNIFORM, BASED_ON_DEGREE
};

// ─── Stream I/O (from upstream, with debug) ─────────────────────────
inline void read_preprocessor_stream(
        const std::string& stream_path,
        std::vector<preprocessor_operation>& stream) {
    std::ifstream file(stream_path, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[STREAM] ERROR: cannot open %s\n", stream_path.c_str());
        return;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t elem_size = sizeof(preprocessor_operation);
    size_t num_elements = file_size / elem_size;

    // ─── NEW: 校验字节对齐 ──────────────────────────────────────────
    if (file_size % elem_size != 0) {
        std::fprintf(stderr, "[STREAM] WARN: file size %zu not aligned to "
                    "operation size %zu (remainder=%zu)\n",
                    file_size, elem_size, file_size % elem_size);
    }

    stream.resize(num_elements);
    file.read(reinterpret_cast<char*>(stream.data()), num_elements * elem_size);
    file.close();

    // ─── NEW: 加载摘要 ──────────────────────────────────────────────
    std::printf("[STREAM] Loaded %zu operations (%.2f MB) from %s\n",
                num_elements, file_size / (1024.0 * 1024.0), stream_path.c_str());

    // Print type distribution
    size_t counts[32] = {};
    for (auto& op : stream) {
        int idx = static_cast<int>(op.type);
        if (idx >= 0 && idx < 32) counts[idx]++;
    }
    std::printf("[STREAM] Type distribution:\n");
    for (int i = 0; i < 32; i++) {
        if (counts[i] > 0)
            std::printf("  %s: %zu\n", op_type_name(static_cast<preprocessor_operationType>(i)), counts[i]);
    }
}

// ─── Config structures (from upstream, with dump) ───────────────────
struct PreprocessorConfig {
    double timestamp_rate = 0.0;
    int seed = 42;
    uint64_t num_search = 1000000;
    bool test_version_chain = false;
    bool enable_bloom_filter = false;

    void dump(const char* label = "Config") const {
        std::printf("[%s] ts_rate=%.2f seed=%d search=%lu version_chain=%d bloom=%d\n",
                    label, timestamp_rate, seed, num_search,
                    test_version_chain, enable_bloom_filter);
    }
};

struct PreprocessorEdgeDriverConfig {
    std::string workload_dir;
    std::string output_dir;
    preprocessor_operationType workload_type = preprocessor_operationType::INSERT;
    preprocessor_targetStreamType target_stream_type = preprocessor_targetStreamType::FULL;

    double timestamp_rate = 0.8;
    double initial_graph_rate = 0.8;
    std::vector<int> element_sizes;
    int seed = 42;
    uint64_t num_search = 1000000;
    uint64_t num_scan = 1000000;
    int repeat_times = 0;
    bool test_version_chain = false;
    bool is_real_graph = false;

    void dump(const char* label = "EdgeDriverConfig") const {
        std::printf("[%s] workload=%s  output=%s  type=%s  "
                    "ts_rate=%.2f  init_rate=%.2f  seed=%d  repeat=%d\n",
                    label, workload_dir.c_str(), output_dir.c_str(),
                    op_type_name(workload_type),
                    timestamp_rate, initial_graph_rate, seed, repeat_times);
    }
};

struct PreprocessorDriverConfig {
    std::string workload_dir;
    std::string output_dir;
    preprocessor_operationType workload_type = preprocessor_operationType::INSERT;
    preprocessor_targetStreamType target_stream_type = preprocessor_targetStreamType::FULL;

    uint64_t insert_delete_checkpoint_size = 10000;
    int insert_delete_num_threads = 1;
    int insert_batch_size = 100000;

    int repeat_times = 0;
    uint64_t mb_checkpoint_size = 10000;
    std::vector<int> query_num_threads;

    // Algorithm params (from upstream)
    int alpha = 15;
    int beta = 18;
    uint64_t bfs_source = 0;
    double delta = 2.0;
    uint64_t sssp_source = 0;
    int num_iterations = 10;
    double damping_factor = 0.85;
    int writer_threads = 1;
    int reader_threads = 1;

    void dump(const char* label = "DriverConfig") const {
        std::printf("[%s] workload=%s  output=%s  threads=%d  "
                    "batch=%d  checkpoint=%lu\n",
                    label, workload_dir.c_str(), output_dir.c_str(),
                    insert_delete_num_threads, insert_batch_size,
                    insert_delete_checkpoint_size);
        std::printf("  BFS: alpha=%d beta=%d source=%lu\n", alpha, beta, bfs_source);
        std::printf("  SSSP: delta=%.2f source=%lu\n", delta, sssp_source);
        std::printf("  PR: iters=%d damping=%.2f\n", num_iterations, damping_factor);
    }
};

// ─── Concurrent workload (from upstream) ────────────────────────────
struct preprocessor_concurrent_workload {
    preprocessor_operationType workload_type;
    preprocessor_targetStreamType target_stream_type;
    int num_threads;

    void dump() const {
        std::printf("[CONCURRENT] type=%s threads=%d\n",
                    op_type_name(workload_type), num_threads);
    }
};

#endif // PHILEMON_PREPROCESSOR_TYPES_HPP
