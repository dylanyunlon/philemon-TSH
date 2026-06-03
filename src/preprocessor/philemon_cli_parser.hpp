#ifndef PHILEMON_CLI_PARSER_HPP
#define PHILEMON_CLI_PARSER_HPP
/**
 * philemon_cli_parser.hpp — 命令行参数解析器 (去 boost 版)
 *
 * 骨架来源: upstream/rapidstore/dataset_preprocessor/parser.hpp (59行)
 *           upstream/rapidstore/dataset_preprocessor/parser.cpp (156行)
 * 修改 (~25%):
 *   - 移除 boost::program_options → 纯 STL 解析 (getopt 风格)
 *   - 保留 singleton + 所有 getter 签名
 *   - 增加 dump_parsed() 打印解析结果
 *   - 增加 tier 相关参数: --dram-mb, --nvm-mb, --ssd-path
 *   - 增加 usage() 帮助信息打印
 *
 * Milestone: M028
 */

#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace philemon {
namespace cli {

class CliParser {
public:
    static CliParser& instance() {
        static CliParser inst;
        return inst;
    }

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg(argv[i]);

            if (arg == "-h" || arg == "--help") {
                usage(); std::exit(0);
            }

            // --key=value format
            auto eq = arg.find('=');
            if (eq != std::string::npos && arg.substr(0, 2) == "--") {
                kv_[arg.substr(2, eq - 2)] = arg.substr(eq + 1);
                continue;
            }

            // --key value format
            if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
                std::string key = arg.substr(2);
                if (i + 1 < argc) {
                    kv_[key] = std::string(argv[++i]);
                }
                continue;
            }

            // -k value format (short)
            if (arg.size() == 2 && arg[0] == '-') {
                char k = arg[1];
                if (i + 1 < argc) {
                    switch (k) {
                        case 'i': kv_["input_file"] = argv[++i]; break;
                        case 'o': kv_["output_dir"] = argv[++i]; break;
                        case 's': kv_["seed"] = argv[++i]; break;
                        case 'w': kv_["is_weighted"] = argv[++i]; break;
                        case 'd': kv_["delimiter"] = argv[++i]; break;
                        default:
                            std::fprintf(stderr, "[CLI] Unknown flag: -%c\n", k);
                    }
                }
            }
        }
        parsed_ = true;
    }

    // ─── Getters (from upstream Parser) ─────────────────────────────
    std::string input_file() const       { return get("input_file", ""); }
    std::string input_file_static() const{ return get("input_file_static", ""); }
    std::string input_file_dynamic() const{ return get("input_file_dynamic", ""); }
    std::string output_dir() const       { return get("output_dir", "./output"); }
    bool is_weighted() const             { return get("is_weighted", "0") != "0"; }
    bool is_shuffle() const              { return get("is_shuffle", "1") != "0"; }
    char delimiter() const               { return get("delimiter", " ")[0]; }
    double initial_graph_ratio() const   { return getd("initial_graph_ratio", 0.8); }
    double vertex_query_ratio() const    { return getd("vertex_query_ratio", 0.2); }
    double edge_query_ratio() const      { return getd("edge_query_ratio", 0.2); }
    double high_degree_vertex_ratio() const { return getd("high_degree_vertex_ratio", 0.01); }
    double high_degree_edge_ratio() const   { return getd("high_degree_edge_ratio", 0.2); }
    double low_degree_vertex_ratio() const  { return getd("low_degree_vertex_ratio", 0.2); }
    double low_degree_edge_ratio() const    { return getd("low_degree_edge_ratio", 0.5); }
    uint64_t insert_num() const          { return getu("insert_num", 10000); }
    uint64_t search_num() const          { return getu("search_num", 10000); }
    uint64_t scan_num() const            { return getu("scan_num", 10000); }
    unsigned int seed() const            { return (unsigned)geti("seed", 0); }

    // ─── NEW: Tier params ───────────────────────────────────────────
    uint64_t dram_mb() const             { return getu("dram_mb", 4096); }
    uint64_t nvm_mb() const              { return getu("nvm_mb", 16384); }
    std::string ssd_path() const         { return get("ssd_path", "/tmp/philemon"); }

    // ─── NEW: dump ──────────────────────────────────────────────────
    void dump_parsed(const char* label = "CLI") const {
        std::printf("[%s] Parsed %zu arguments:\n", label, kv_.size());
        for (auto& [k, v] : kv_) {
            std::printf("  %-30s = %s\n", k.c_str(), v.c_str());
        }
    }

    void usage() const {
        std::printf(
            "Usage: philemon [OPTIONS]\n"
            "  -i, --input_file FILE       Input edge list file\n"
            "  -o, --output_dir DIR        Output directory\n"
            "  -s, --seed N                Random seed\n"
            "  -w, --is_weighted 0|1       Weighted graph\n"
            "  -d, --delimiter CHAR        Field delimiter\n"
            "  --initial_graph_ratio F     Initial graph ratio (default 0.8)\n"
            "  --dram_mb N                 DRAM capacity in MB (default 4096)\n"
            "  --nvm_mb N                  NVM capacity in MB (default 16384)\n"
            "  --ssd_path PATH             SSD storage path\n"
            "  -h, --help                  Show this help\n"
        );
    }

private:
    CliParser() = default;
    CliParser(const CliParser&) = delete;
    bool parsed_ = false;
    std::unordered_map<std::string, std::string> kv_;

    std::string get(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k); return it != kv_.end() ? it->second : def;
    }
    int geti(const std::string& k, int def) const {
        auto it = kv_.find(k); return it != kv_.end() ? std::atoi(it->second.c_str()) : def;
    }
    double getd(const std::string& k, double def) const {
        auto it = kv_.find(k); return it != kv_.end() ? std::atof(it->second.c_str()) : def;
    }
    uint64_t getu(const std::string& k, uint64_t def) const {
        auto it = kv_.find(k); return it != kv_.end() ? std::strtoull(it->second.c_str(), nullptr, 10) : def;
    }
};

} // namespace cli
} // namespace philemon

#endif // PHILEMON_CLI_PARSER_HPP
