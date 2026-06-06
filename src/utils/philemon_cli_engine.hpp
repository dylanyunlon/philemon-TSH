/**
 * philemon_cli_engine.hpp — 配置解析引擎
 *
 * 骨架来源: upstream/rapidstore/utils/commandLineParser.hpp+cpp (700行)
 * 修改 (~20%):
 *   - [MOD] 移除boost::program_options依赖 → 纯std::ifstream解析
 *   - [MOD] Singleton → 可实例化+全局便利接口
 *   - [NEW] dump_config(): 打印全部已解析参数的当前值
 *   - [NEW] BREAKPOINT_CONFIG(): 在解析关键路径打印状态
 *   - [NEW] validate(): 参数范围校验 + 错误报告
 *   - [KEEP] 所有upstream字段名100%保留 (alpha, beta, delta, bfs_source等)
 *   - [KEEP] get_driver_config()/get_exp_config() 返回结构体
 *   - [KEEP] operationType/targetStreamType枚举100%保留
 *
 * Milestone: M098
 */
#ifndef PHILEMON_CLI_ENGINE_HPP
#define PHILEMON_CLI_ENGINE_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <cassert>

namespace philemon {
namespace config {

// ─── 操作类型枚举 (upstream 100%) ───────────────────────────────────
enum class operationType {
    INSERT, DELETE, UPDATE,
    GET_VERTEX, GET_WEIGHT, GET_EDGE,
    SCAN_NEIGHBOR, GET_NEIGHBOR,
    BFS, SSSP, PAGE_RANK, WCC, TC, TC_OP,
    QUERY, MIXED, QOS
};

enum class targetStreamType {
    FULL, GENERAL, HIGH_DEGREE, LOW_DEGREE, UNIFORM, BASED_ON_DEGREE
};

// ─── 驱动配置结构体 (upstream保留) ──────────────────────────────────
struct DriverConfig {
    std::string workload_dir;
    std::string output_dir;
    int num_threads = 4;
    int seed = 42;
    operationType workload_type = operationType::BFS;
    targetStreamType target_stream_type = targetStreamType::FULL;

    uint64_t insert_delete_checkpoint_size = 10000;
    int insert_delete_num_threads = 1;
    int repeat_times = 0;
    uint64_t mb_checkpoint_size = 10000;
    std::vector<int> microbenchmark_num_threads;
    std::vector<int> query_num_threads;

    int alpha = 15;
    int beta = 18;
    uint64_t bfs_source = 0;
    double delta = 2.0;
    uint64_t sssp_source = 0;
    int num_iterations = 10;
    double damping_factor = 0.85;

    int writer_threads = 16;
    int reader_threads = 16;
    int num_threads_search = 8;
    int num_threads_scan = 20;
};

struct EdgeDriverConfig {
    std::vector<int> element_sizes;
    uint64_t num_vertices = 1000;
    double initial_graph_rate = 0.8;
    double timestamp_rate = 0.8;
    uint64_t num_search = 1000000;
    uint64_t num_scan = 1000000;
    int neighbor_test_repeat_times = 0;
    bool test_version_chain = false;
    bool is_real_graph = false;
};

struct Config {
    DriverConfig driver;
    EdgeDriverConfig edge;
};

// ─── 辅助: 字符串→枚举 (upstream 100%) ─────────────────────────────
inline operationType parse_op_type(const std::string& s) {
    if (s == "insert")    return operationType::INSERT;
    if (s == "delete")    return operationType::DELETE;
    if (s == "update")    return operationType::UPDATE;
    if (s == "bfs")       return operationType::BFS;
    if (s == "sssp")      return operationType::SSSP;
    if (s == "pr")        return operationType::PAGE_RANK;
    if (s == "wcc")       return operationType::WCC;
    if (s == "tc")        return operationType::TC;
    if (s == "tc_op")     return operationType::TC_OP;
    if (s == "query")     return operationType::QUERY;
    if (s == "mixed")     return operationType::MIXED;
    if (s == "qos")       return operationType::QOS;
    if (s == "get_vertex") return operationType::GET_VERTEX;
    if (s == "get_weight") return operationType::GET_WEIGHT;
    if (s == "get_edge")   return operationType::GET_EDGE;
    if (s == "scan_neighbor") return operationType::SCAN_NEIGHBOR;
    if (s == "get_neighbor")  return operationType::GET_NEIGHBOR;
    if (s == "micro_benchmark") return operationType::GET_VERTEX;
    std::fprintf(stderr, "[WARN] Unknown op type '%s', defaulting to BFS\n", s.c_str());
    return operationType::BFS;
}

inline targetStreamType parse_ts_type(const std::string& s) {
    if (s == "full")           return targetStreamType::FULL;
    if (s == "general")        return targetStreamType::GENERAL;
    if (s == "high_degree")    return targetStreamType::HIGH_DEGREE;
    if (s == "low_degree")     return targetStreamType::LOW_DEGREE;
    if (s == "uniform")        return targetStreamType::UNIFORM;
    if (s == "based_on_degree") return targetStreamType::BASED_ON_DEGREE;
    std::fprintf(stderr, "[WARN] Unknown stream type '%s', defaulting to FULL\n", s.c_str());
    return targetStreamType::FULL;
}

inline const char* op_type_name(operationType t) {
    switch(t) {
        case operationType::INSERT: return "INSERT";
        case operationType::DELETE: return "DELETE";
        case operationType::UPDATE: return "UPDATE";
        case operationType::BFS:    return "BFS";
        case operationType::SSSP:   return "SSSP";
        case operationType::PAGE_RANK: return "PAGE_RANK";
        case operationType::WCC:    return "WCC";
        case operationType::TC:     return "TC";
        case operationType::TC_OP:  return "TC_OP";
        case operationType::QUERY:  return "QUERY";
        case operationType::MIXED:  return "MIXED";
        case operationType::QOS:    return "QOS";
        default: return "UNKNOWN";
    }
}

// ─── 配置引擎 (upstream commandLineParser重写) ─────────────────────
class ConfigEngine {
    std::unordered_map<std::string, std::string> kv_;
    Config cfg_;
    bool parsed_ = false;
    int parse_errors_ = 0;

public:
    ConfigEngine() = default;

    // ─── upstream parse()等价 — 不用boost ────────────────────────
    void parse(const std::string& config_path) {
        std::ifstream fin(config_path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[CONFIG] Cannot open config file: %s\n", 
                         config_path.c_str());
            use_defaults();
            return;
        }
        
        std::string line;
        int line_no = 0;
        while (std::getline(fin, line)) {
            line_no++;
            // 去掉注释和空行
            auto comment_pos = line.find('#');
            if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
            
            // 去掉首尾空白
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            size_t end = line.find_last_not_of(" \t\r\n");
            if (end != std::string::npos) line = line.substr(0, end + 1);
            if (line.empty()) continue;

            // 解析 key=value 或 key = value
            auto eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                std::fprintf(stderr, "[CONFIG] Line %d: no '=' found: %s\n",
                             line_no, line.c_str());
                parse_errors_++;
                continue;
            }

            std::string key = line.substr(0, eq_pos);
            std::string val = line.substr(eq_pos + 1);
            
            // trim
            auto trim = [](std::string& s) {
                size_t a = s.find_first_not_of(" \t");
                size_t b = s.find_last_not_of(" \t");
                s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
            };
            trim(key);
            trim(val);
            
            kv_[key] = val;
        }

        // [BREAKPOINT] 解析完成后打印所有键值对
        std::fprintf(stderr, "[CONFIG] Parsed %zu keys from %s (%d errors)\n",
                     kv_.size(), config_path.c_str(), parse_errors_);
        
        apply_to_config();
        parsed_ = true;
    }

    void use_defaults() {
        cfg_ = Config{};
        parsed_ = true;
        std::fprintf(stderr, "[CONFIG] Using default configuration\n");
    }

    // ─── 字段提取 (upstream接口100%保留) ────────────────────────
    std::string get_workload_dir() const { return cfg_.driver.workload_dir; }
    std::string get_output_dir() const { return cfg_.driver.output_dir; }
    int get_num_threads() const { return cfg_.driver.num_threads; }
    int get_seed() const { return cfg_.driver.seed; }
    operationType get_workload_type() const { return cfg_.driver.workload_type; }
    targetStreamType get_target_stream_type() const { return cfg_.driver.target_stream_type; }
    uint64_t get_insert_delete_checkpoint_size() const { return cfg_.driver.insert_delete_checkpoint_size; }
    int get_insert_delete_num_threads() const { return cfg_.driver.insert_delete_num_threads; }
    int get_repeat_times() const { return cfg_.driver.repeat_times; }
    uint64_t get_mb_checkpoint_size() const { return cfg_.driver.mb_checkpoint_size; }
    std::vector<int> get_microbenchmark_num_threads() const { return cfg_.driver.microbenchmark_num_threads; }
    std::vector<int> get_query_num_threads() const { return cfg_.driver.query_num_threads; }
    int get_alpha() const { return cfg_.driver.alpha; }
    int get_beta() const { return cfg_.driver.beta; }
    uint64_t get_bfs_source() const { return cfg_.driver.bfs_source; }
    double get_delta() const { return cfg_.driver.delta; }
    uint64_t get_sssp_source() const { return cfg_.driver.sssp_source; }
    int get_num_iterations() const { return cfg_.driver.num_iterations; }
    double get_damping_factor() const { return cfg_.driver.damping_factor; }
    int get_writer_threads() const { return cfg_.driver.writer_threads; }
    int get_reader_threads() const { return cfg_.driver.reader_threads; }
    int get_num_threads_search() const { return cfg_.driver.num_threads_search; }
    int get_num_threads_scan() const { return cfg_.driver.num_threads_scan; }
    uint64_t get_num_vertices() const { return cfg_.edge.num_vertices; }
    bool get_real_graph() const { return cfg_.edge.is_real_graph; }
    double get_initial_graph_rate() const { return cfg_.edge.initial_graph_rate; }
    double get_timestamp_rate() const { return cfg_.edge.timestamp_rate; }
    uint64_t get_num_search() const { return cfg_.edge.num_search; }
    uint64_t get_num_scan() const { return cfg_.edge.num_scan; }
    int get_neighbor_test_repeat_times() const { return cfg_.edge.neighbor_test_repeat_times; }
    bool get_test_version_chain() const { return cfg_.edge.test_version_chain; }

    DriverConfig get_driver_config() const { return cfg_.driver; }
    EdgeDriverConfig get_edge_driver_config() const { return cfg_.edge; }
    Config get_exp_config() const { return cfg_; }

    // ─── [NEW] 调试: 打印全部配置 ───────────────────────────────
    void dump_config() const {
        std::fprintf(stderr, "\n╔═══ CONFIG ENGINE STATE DUMP ═══╗\n");
        std::fprintf(stderr, "║ parsed: %s\n", parsed_ ? "yes" : "no");
        std::fprintf(stderr, "║ raw KV pairs: %zu\n", kv_.size());
        for (auto& [k, v] : kv_) {
            std::fprintf(stderr, "║   %-30s = %s\n", k.c_str(), v.c_str());
        }
        std::fprintf(stderr, "║─── Resolved Driver Config ───\n");
        std::fprintf(stderr, "║   threads=%d seed=%d\n", cfg_.driver.num_threads, cfg_.driver.seed);
        std::fprintf(stderr, "║   workload=%s\n", op_type_name(cfg_.driver.workload_type));
        std::fprintf(stderr, "║   alpha=%d beta=%d bfs_source=%lu\n", 
                     cfg_.driver.alpha, cfg_.driver.beta, cfg_.driver.bfs_source);
        std::fprintf(stderr, "║   delta=%.2f sssp_source=%lu\n", cfg_.driver.delta, cfg_.driver.sssp_source);
        std::fprintf(stderr, "║   num_iterations=%d damping_factor=%.3f\n",
                     cfg_.driver.num_iterations, cfg_.driver.damping_factor);
        std::fprintf(stderr, "║─── Edge Driver Config ───\n");
        std::fprintf(stderr, "║   num_vertices=%lu real_graph=%d\n",
                     cfg_.edge.num_vertices, cfg_.edge.is_real_graph);
        std::fprintf(stderr, "║   initial_graph_rate=%.2f timestamp_rate=%.2f\n",
                     cfg_.edge.initial_graph_rate, cfg_.edge.timestamp_rate);
        std::fprintf(stderr, "╚═══════════════════════════════╝\n\n");
    }

    // ─── [NEW] 校验 ─────────────────────────────────────────────
    int validate() const {
        int errors = 0;
        if (cfg_.driver.num_threads <= 0) {
            std::fprintf(stderr, "[VALIDATE] num_threads must be > 0\n");
            errors++;
        }
        if (cfg_.driver.alpha <= 0 || cfg_.driver.beta <= 0) {
            std::fprintf(stderr, "[VALIDATE] alpha/beta must be > 0\n");
            errors++;
        }
        if (cfg_.driver.damping_factor < 0 || cfg_.driver.damping_factor > 1) {
            std::fprintf(stderr, "[VALIDATE] damping_factor must be in [0,1]\n");
            errors++;
        }
        if (cfg_.driver.delta <= 0) {
            std::fprintf(stderr, "[VALIDATE] delta must be > 0\n");
            errors++;
        }
        if (cfg_.driver.num_iterations <= 0) {
            std::fprintf(stderr, "[VALIDATE] num_iterations must be > 0\n");
            errors++;
        }
        return errors;
    }

    // ─── Singleton便利接口 (upstream兼容) ────────────────────────
    static ConfigEngine& get_instance() {
        static ConfigEngine inst;
        return inst;
    }

private:
    std::string get_str(const std::string& key, const std::string& def = "") const {
        auto it = kv_.find(key);
        return it != kv_.end() ? it->second : def;
    }
    int get_int(const std::string& key, int def = 0) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        try { return std::stoi(it->second); } catch(...) { return def; }
    }
    uint64_t get_u64(const std::string& key, uint64_t def = 0) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        try { return std::stoull(it->second); } catch(...) { return def; }
    }
    double get_dbl(const std::string& key, double def = 0) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        try { return std::stod(it->second); } catch(...) { return def; }
    }
    bool get_bool(const std::string& key, bool def = false) const {
        auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        return (it->second == "true" || it->second == "1" || it->second == "yes");
    }
    std::vector<int> get_int_vec(const std::string& key) const {
        std::vector<int> result;
        auto it = kv_.find(key);
        if (it == kv_.end()) return result;
        std::istringstream iss(it->second);
        int v;
        while (iss >> v) result.push_back(v);
        return result;
    }

    void apply_to_config() {
        auto& d = cfg_.driver;
        auto& e = cfg_.edge;

        d.workload_dir   = get_str("workload_dir", ".");
        d.output_dir     = get_str("output_dir", "./results");
        d.num_threads    = get_int("num_threads", 4);
        d.seed           = get_int("seed", 42);
        d.workload_type  = parse_op_type(get_str("workload_type", "bfs"));
        d.target_stream_type = parse_ts_type(get_str("target_stream_type", "full"));

        d.insert_delete_checkpoint_size = get_u64("insert_delete_checkpoint_size", 10000);
        d.insert_delete_num_threads     = get_int("insert_delete_num_threads", 1);
        d.repeat_times  = get_int("mb_repeat_times", 0);
        d.mb_checkpoint_size = get_u64("mb_checkpoint_size", 10000);
        d.microbenchmark_num_threads = get_int_vec("microbenchmark_num_threads");
        d.query_num_threads = get_int_vec("query_num_threads");

        d.alpha      = get_int("alpha", 15);
        d.beta       = get_int("beta", 18);
        d.bfs_source = get_u64("bfs_source", 0);
        d.delta      = get_dbl("delta", 2.0);
        d.sssp_source = get_u64("sssp_source", 0);
        d.num_iterations = get_int("num_iterations", 10);
        d.damping_factor = get_dbl("damping_factor", 0.85);

        d.writer_threads = get_int("writer_threads", 16);
        d.reader_threads = get_int("reader_threads", 16);
        d.num_threads_search = get_int("num_threads_search", 8);
        d.num_threads_scan   = get_int("num_threads_scan", 20);

        e.num_vertices = get_u64("num_vertices", 1000);
        e.is_real_graph = get_bool("real_graph", false);
        e.initial_graph_rate = get_dbl("initial_graph_rate", 0.8);
        e.timestamp_rate     = get_dbl("timestamp_rate", 0.8);
        e.num_search = get_u64("num_search", 1000000);
        e.num_scan   = get_u64("num_scan", 1000000);
        e.neighbor_test_repeat_times = get_int("neighbor_test_repeat_times", 0);
        e.test_version_chain = get_bool("test_version_chain", false);
        e.element_sizes = get_int_vec("element_sizes");

        // [BREAKPOINT] 解析后自动校验
        int errs = validate();
        if (errs > 0) {
            std::fprintf(stderr, "[CONFIG] %d validation errors found\n", errs);
        }
    }
};

#define BREAKPOINT_CONFIG() do { \
    philemon::config::ConfigEngine::get_instance().dump_config(); \
} while(0)

} // namespace config
} // namespace philemon

#endif // PHILEMON_CLI_ENGINE_HPP
