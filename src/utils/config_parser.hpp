#ifndef PHILEMON_CONFIG_PARSER_HPP
#define PHILEMON_CONFIG_PARSER_HPP
/**
 * config_parser.hpp — 配置文件解析器
 *
 * 骨架来源:
 *   upstream/rapidstore/utils/commandLineParser.hpp (120行)
 *   upstream/rapidstore/utils/commandLineParser.cpp (180行)
 *
 * 修改 (~20%):
 *   - [MOD] boost::program_options → 手动 key=value INI 解析 (去外部依赖)
 *   - [MOD] commandLineParser → philemon::ConfigParser (namespace化)
 *   - [NEW] dump_parsed_config(): 解析完后全量打印所有字段到stderr
 *   - [NEW] tier相关配置项: hbm_capacity, gddr_capacity, migration_batch
 *   - [KEEP] get_*() accessor方法签名100%保留
 *   - [KEEP] DriverConfig/EdgeDriverConfig组装逻辑100%保留
 *   - [KEEP] 默认值100%保留(alpha=15, beta=18, delta=2.0等)
 *
 * Milestone: M027+
 */

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <cstdio>
#include <algorithm>
#include <cstdlib>

#include "../types/philemon_types.hpp"

namespace philemon {

class ConfigParser {
public:
    static ConfigParser& get_instance() {
        static ConfigParser inst;
        return inst;
    }

    // Parse config file (key=value format, one per line)
    void parse(const std::string& config_path) {
        std::fprintf(stderr, "\n[CONFIG-PARSER] loading: %s\n", config_path.c_str());
        std::ifstream fin(config_path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[CONFIG-PARSER] FAIL: cannot open %s\n",
                         config_path.c_str());
            return;
        }

        std::string line;
        int line_num = 0;
        while (std::getline(fin, line)) {
            line_num++;
            // strip comments and whitespace
            auto comment_pos = line.find('#');
            if (comment_pos != std::string::npos)
                line = line.substr(0, comment_pos);

            // trim
            while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
                line.erase(line.begin());
            while (!line.empty() && (line.back() == ' ' || line.back() == '\t'))
                line.pop_back();

            if (line.empty()) continue;

            auto eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                std::fprintf(stderr, "[CONFIG-PARSER] skip line %d (no '='): %s\n",
                             line_num, line.c_str());
                continue;
            }

            std::string key = line.substr(0, eq_pos);
            std::string val = line.substr(eq_pos + 1);

            // trim key/val
            while (!key.empty() && key.back() == ' ') key.pop_back();
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());

            kv_[key] = val;
        }

        // Apply to fields (upstream field mapping)
        workload_dir = get_str("workload_dir", "");
        output_dir   = get_str("output_dir", "");
        num_threads  = get_int("num_threads", 1);
        seed         = get_int("seed", 42);

        insert_delete_checkpoint_size = get_uint64("insert_delete_checkpoint_size", 10000);
        insert_delete_num_threads = get_int("insert_delete_num_threads", 1);
        insert_batch_size = get_uint64("insert_batch_size", 1);

        update_num_threads   = get_int("update_num_threads", 1);
        update_repeat_times  = get_int("update_repeat_times", 10);

        repeat_times = get_int("repeat_times", 0);
        mb_checkpoint_size = get_uint64("mb_checkpoint_size", 10000);

        alpha      = get_int("alpha", 15);
        beta       = get_int("beta", 18);
        bfs_source = get_uint64("bfs_source", 0);
        delta      = get_double("delta", 2.0);
        sssp_source = get_uint64("sssp_source", 0);
        num_iterations  = get_int("num_iterations", 10);
        damping_factor  = get_double("damping_factor", 0.85);

        writer_threads = get_int("writer_threads", 16);
        reader_threads = get_int("reader_threads", 16);

        num_threads_search = get_int("num_threads_search", 8);
        num_threads_scan   = get_int("num_threads_scan", 20);

        // NEW tier configs
        hbm_capacity_gb  = get_double("hbm_capacity_gb", 4.0);
        gddr_capacity_gb = get_double("gddr_capacity_gb", 16.0);
        migration_batch  = get_int("migration_batch_size", 1024);
        enable_tier_trace = get_int("enable_tier_trace", 1) != 0;

        // Parse thread lists
        microbenchmark_num_threads = parse_int_list("microbenchmark_num_threads");
        query_num_threads = parse_int_list("query_num_threads");

        std::fprintf(stderr, "[CONFIG-PARSER] parsed %lu keys\n",
                     (unsigned long)kv_.size());

        dump_parsed_config();
    }

    // ─── Accessors (upstream 100%) ──────────────────────────────────
    std::string get_workload_dir() const { return workload_dir; }
    std::string get_output_dir()   const { return output_dir; }
    int  get_num_threads()   const { return num_threads; }
    int  get_seed()          const { return seed; }
    int  get_alpha()         const { return alpha; }
    int  get_beta()          const { return beta; }
    uint64_t get_bfs_source()   const { return bfs_source; }
    double   get_delta()        const { return delta; }
    uint64_t get_sssp_source()  const { return sssp_source; }
    int    get_num_iterations()    const { return num_iterations; }
    double get_damping_factor()    const { return damping_factor; }
    int    get_writer_threads()    const { return writer_threads; }
    int    get_reader_threads()    const { return reader_threads; }
    int    get_num_threads_search() const { return num_threads_search; }
    int    get_num_threads_scan()   const { return num_threads_scan; }

    std::vector<int> get_microbenchmark_num_threads() const {
        return microbenchmark_num_threads;
    }
    std::vector<int> get_query_num_threads() const {
        return query_num_threads;
    }

    // Build DriverConfig (upstream assembly logic 100% preserved)
    DriverConfig get_driver_config() const {
        DriverConfig cfg;
        cfg.workload_dir = workload_dir;
        cfg.output_dir   = output_dir;
        cfg.insert_delete_checkpoint_size = insert_delete_checkpoint_size;
        cfg.insert_delete_num_threads = insert_delete_num_threads;
        cfg.insert_batch_size = insert_batch_size;
        cfg.update_num_threads = update_num_threads;
        cfg.update_repeat_times = update_repeat_times;
        cfg.repeat_times = repeat_times;
        cfg.mb_checkpoint_size = mb_checkpoint_size;
        cfg.microbenchmark_num_threads = microbenchmark_num_threads;
        cfg.query_num_threads = query_num_threads;
        cfg.alpha = alpha;
        cfg.beta  = beta;
        cfg.bfs_source = bfs_source;
        cfg.delta = delta;
        cfg.sssp_source = sssp_source;
        cfg.num_iterations = num_iterations;
        cfg.damping_factor = damping_factor;
        cfg.writer_threads = writer_threads;
        cfg.reader_threads = reader_threads;
        cfg.num_threads_search = num_threads_search;
        cfg.num_threads_scan = num_threads_scan;

        // NEW tier
        cfg.hbm_capacity_gb = hbm_capacity_gb;
        cfg.gddr_capacity_gb = gddr_capacity_gb;
        cfg.migration_batch_size = migration_batch;
        cfg.enable_tier_trace = enable_tier_trace;

        return cfg;
    }

    // ─── Debug: 全量打印 ────────────────────────────────────────────
    void dump_parsed_config() const {
        std::fprintf(stderr, "\n╔══════════ Parsed Configuration ══════════╗\n");
        std::fprintf(stderr, "║ workload_dir = %-26s ║\n", workload_dir.c_str());
        std::fprintf(stderr, "║ output_dir   = %-26s ║\n", output_dir.c_str());
        std::fprintf(stderr, "║ threads=%d  seed=%d                       ║\n",
                     num_threads, seed);
        std::fprintf(stderr, "║ alpha=%d beta=%d bfs_src=%lu              ║\n",
                     alpha, beta, (unsigned long)bfs_source);
        std::fprintf(stderr, "║ delta=%.2f sssp_src=%lu                   ║\n",
                     delta, (unsigned long)sssp_source);
        std::fprintf(stderr, "║ PR: iters=%d damping=%.2f                 ║\n",
                     num_iterations, damping_factor);
        std::fprintf(stderr, "║ writer_t=%d reader_t=%d                   ║\n",
                     writer_threads, reader_threads);
        std::fprintf(stderr, "║ HBM=%.1fGB GDDR=%.1fGB mig_batch=%d      ║\n",
                     hbm_capacity_gb, gddr_capacity_gb, migration_batch);
        std::fprintf(stderr, "║ tier_trace=%s                             ║\n",
                     enable_tier_trace ? "ON" : "OFF");
        std::fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");
    }

private:
    ConfigParser() = default;
    ConfigParser(const ConfigParser&) = delete;
    ConfigParser& operator=(const ConfigParser&) = delete;

    std::unordered_map<std::string, std::string> kv_;

    // Fields (upstream 100%)
    std::string workload_dir, output_dir;
    int num_threads{1}, seed{42};
    uint64_t insert_delete_checkpoint_size{10000};
    int insert_delete_num_threads{1};
    uint64_t insert_batch_size{1};
    int update_num_threads{1}, update_repeat_times{10};
    int repeat_times{0};
    uint64_t mb_checkpoint_size{10000};
    std::vector<int> microbenchmark_num_threads;
    std::vector<int> query_num_threads;
    int alpha{15}, beta{18};
    uint64_t bfs_source{0}, sssp_source{0};
    double delta{2.0}, damping_factor{0.85};
    int num_iterations{10};
    int writer_threads{16}, reader_threads{16};
    int num_threads_search{8}, num_threads_scan{20};

    // NEW
    double hbm_capacity_gb{4.0}, gddr_capacity_gb{16.0};
    int migration_batch{1024};
    bool enable_tier_trace{true};

    // Helpers
    std::string get_str(const std::string& key, const std::string& def) const {
        auto it = kv_.find(key);
        return it != kv_.end() ? it->second : def;
    }
    int get_int(const std::string& key, int def) const {
        auto it = kv_.find(key);
        return it != kv_.end() ? std::atoi(it->second.c_str()) : def;
    }
    uint64_t get_uint64(const std::string& key, uint64_t def) const {
        auto it = kv_.find(key);
        return it != kv_.end() ? std::strtoull(it->second.c_str(), nullptr, 10) : def;
    }
    double get_double(const std::string& key, double def) const {
        auto it = kv_.find(key);
        return it != kv_.end() ? std::atof(it->second.c_str()) : def;
    }
    std::vector<int> parse_int_list(const std::string& key) const {
        std::vector<int> result;
        auto it = kv_.find(key);
        if (it == kv_.end()) return result;
        std::istringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) result.push_back(std::atoi(tok.c_str()));
        }
        return result;
    }
};

}  // namespace philemon

#endif  // PHILEMON_CONFIG_PARSER_HPP
