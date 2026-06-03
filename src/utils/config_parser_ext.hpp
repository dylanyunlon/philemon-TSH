#ifndef PHILEMON_CONFIG_PARSER_EXT_HPP
#define PHILEMON_CONFIG_PARSER_EXT_HPP
/**
 * config_parser_ext.hpp — 配置文件解析器 (从 upstream 移植, 去 boost 依赖)
 *
 * 骨架来源: upstream/rapidstore/utils/commandLineParser.hpp (117行)
 *           upstream/rapidstore/dataset_preprocessor/parser.hpp (59行)
 * 修改 (~25%):
 *   - 移除 boost::program_options 依赖 → 纯标准库 key=value 解析
 *   - 合并 commandLineParser 和 preprocessor Parser 为统一接口
 *   - 增加 tier_config 字段: dram_capacity_mb, nvm_capacity_mb, ssd_path
 *   - 增加 dump_config() 打印完整配置快照 (调试核心)
 *   - 增加 validate() 配置一致性校验
 *   - 保留 singleton 模式和所有 upstream getter
 *
 * Milestone: M027
 */

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "../types/philemon_types.hpp"

namespace philemon {
namespace config {

class ConfigParser {
public:
    static ConfigParser& instance() {
        static ConfigParser inst;
        return inst;
    }

    // ─── Parse a key=value config file (replaces boost) ─────────────
    bool parse(const std::string& config_path) {
        std::ifstream handle(config_path);
        if (!handle.is_open()) {
            std::fprintf(stderr, "[CONFIG] ERROR: cannot open %s\n", config_path.c_str());
            return false;
        }

        std::string line;
        int line_no = 0;
        while (std::getline(handle, line)) {
            line_no++;
            // strip comments and whitespace
            auto pos = line.find('#');
            if (pos != std::string::npos) line.erase(pos);
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty()) continue;

            auto eq = line.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "[CONFIG] WARN: skip malformed line %d: %s\n",
                             line_no, line.c_str());
                continue;
            }

            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t"));
            kv_[key] = val;
        }
        handle.close();

        // hydrate fields from key-value store
        hydrate();
        parsed_ = true;

        std::printf("[CONFIG] Parsed %zu entries from %s\n", kv_.size(), config_path.c_str());
        return true;
    }

    // ─── Parse command-line args (replaces preprocessor Parser) ──────
    bool parse_args(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            auto eq = arg.find('=');
            if (eq != std::string::npos) {
                kv_[arg.substr(0, eq)] = arg.substr(eq + 1);
            } else if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
                std::string key = arg.substr(2);
                if (i + 1 < argc) {
                    kv_[key] = std::string(argv[++i]);
                }
            }
        }
        hydrate();
        parsed_ = true;
        return true;
    }

    // ─── Upstream-compatible getters ────────────────────────────────
    std::string workload_dir() const   { return get_str("workload_dir", "."); }
    std::string output_dir() const     { return get_str("output_dir", "./output"); }
    std::string input_file() const     { return get_str("input_file", ""); }
    std::string vertex_file() const    { return get_str("vertex_file", ""); }
    std::string edge_file() const      { return get_str("edge_file", ""); }
    int  num_threads() const           { return get_int("num_threads", 1); }
    int  seed() const                  { return get_int("seed", 42); }
    bool is_weighted() const           { return get_int("is_weighted", 0) != 0; }
    char delimiter() const             { return get_str("delimiter", " ")[0]; }

    // BFS parameters
    int  alpha() const                 { return get_int("alpha", 15); }
    int  beta() const                  { return get_int("beta", 18); }
    uint64_t bfs_source() const        { return get_u64("bfs_source", 0); }

    // SSSP parameters
    double delta() const               { return get_dbl("delta", 2.0); }
    uint64_t sssp_source() const       { return get_u64("sssp_source", 0); }

    // PageRank parameters
    int  num_iterations() const        { return get_int("num_iterations", 10); }
    double damping_factor() const      { return get_dbl("damping_factor", 0.85); }

    // Graph ratios
    double initial_graph_ratio() const { return get_dbl("initial_graph_ratio", 0.8); }
    double timestamp_rate() const      { return get_dbl("timestamp_rate", 0.8); }

    // ─── NEW: Tier-specific config ──────────────────────────────────
    uint64_t dram_capacity_mb() const  { return get_u64("dram_capacity_mb", 4096); }
    uint64_t nvm_capacity_mb() const   { return get_u64("nvm_capacity_mb", 16384); }
    std::string ssd_path() const       { return get_str("ssd_path", "/tmp/philemon_ssd"); }
    double hot_threshold() const       { return get_dbl("hot_threshold", 0.8); }
    double cold_threshold() const      { return get_dbl("cold_threshold", 0.2); }
    int eviction_batch() const         { return get_int("eviction_batch", 1024); }

    // ─── NEW: Debug 一键打印完整配置 ────────────────────────────────
    void dump_config(const char* label = "CURRENT") const {
        std::printf("\n╔══════════════════════════════════════════════╗\n");
        std::printf("║  [CONFIG DUMP] %s                            \n", label);
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("║  workload_dir     = %s\n", workload_dir().c_str());
        std::printf("║  output_dir       = %s\n", output_dir().c_str());
        std::printf("║  input_file       = %s\n", input_file().c_str());
        std::printf("║  num_threads      = %d\n", num_threads());
        std::printf("║  seed             = %d\n", seed());
        std::printf("║  ── BFS ──\n");
        std::printf("║  alpha=%d  beta=%d  source=%lu\n", alpha(), beta(), bfs_source());
        std::printf("║  ── SSSP ──\n");
        std::printf("║  delta=%.2f  source=%lu\n", delta(), sssp_source());
        std::printf("║  ── PageRank ──\n");
        std::printf("║  iterations=%d  damping=%.2f\n", num_iterations(), damping_factor());
        std::printf("║  ── Tier Config ──\n");
        std::printf("║  DRAM=%lu MB  NVM=%lu MB  SSD=%s\n",
                    dram_capacity_mb(), nvm_capacity_mb(), ssd_path().c_str());
        std::printf("║  hot_threshold=%.2f  cold_threshold=%.2f  eviction_batch=%d\n",
                    hot_threshold(), cold_threshold(), eviction_batch());
        std::printf("║  ── Raw KV (%zu entries) ──\n", kv_.size());
        for (auto& [k, v] : kv_) {
            std::printf("║    %s = %s\n", k.c_str(), v.c_str());
        }
        std::printf("╚══════════════════════════════════════════════╝\n\n");
    }

    // ─── NEW: 配置校验 ──────────────────────────────────────────────
    bool validate() const {
        bool ok = true;
        if (num_threads() <= 0) {
            std::fprintf(stderr, "[CONFIG] ERROR: num_threads must be > 0\n"); ok = false;
        }
        if (dram_capacity_mb() == 0) {
            std::fprintf(stderr, "[CONFIG] ERROR: dram_capacity_mb must be > 0\n"); ok = false;
        }
        if (damping_factor() <= 0 || damping_factor() >= 1.0) {
            std::fprintf(stderr, "[CONFIG] WARN: damping_factor=%.2f (unusual)\n", damping_factor());
        }
        if (ok) std::printf("[CONFIG] Validation passed\n");
        return ok;
    }

private:
    ConfigParser() = default;
    ConfigParser(const ConfigParser&) = delete;
    ConfigParser& operator=(const ConfigParser&) = delete;

    bool parsed_ = false;
    std::unordered_map<std::string, std::string> kv_;

    void hydrate() { /* kv_ is the source of truth; getters read from it */ }

    std::string get_str(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k); return it != kv_.end() ? it->second : def;
    }
    int get_int(const std::string& k, int def) const {
        auto it = kv_.find(k); return it != kv_.end() ? std::atoi(it->second.c_str()) : def;
    }
    double get_dbl(const std::string& k, double def) const {
        auto it = kv_.find(k); return it != kv_.end() ? std::atof(it->second.c_str()) : def;
    }
    uint64_t get_u64(const std::string& k, uint64_t def) const {
        auto it = kv_.find(k); return it != kv_.end() ? std::strtoull(it->second.c_str(), nullptr, 10) : def;
    }
};

} // namespace config
} // namespace philemon

#endif // PHILEMON_CONFIG_PARSER_EXT_HPP
