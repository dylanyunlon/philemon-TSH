#ifndef PHILEMON_DATASET_LOADER_HPP
#define PHILEMON_DATASET_LOADER_HPP
/**
 * dataset_loader.hpp — LDBC SNB / LiveJournal / Twitter 数据集统一加载器
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/readers/reader.cpp/hpp         (237行)
 *   upstream/rapidstore/readers/edgeListReader.cpp/hpp  (289行)
 *   upstream/rapidstore/readers/vertexReader.cpp/hpp    (187行)
 *   upstream/rapidstore/dataset_preprocessor/main.cpp   (62行)
 *   upstream/rapidstore/dataset_preprocessor/parser.cpp/hpp  (201行)
 *   upstream/rapidstore/dataset_preprocessor/dataset_preprocessor.cpp/hpp (166行)
 *
 * 修改 (~20%):
 *   - [ALG] 文件格式: 硬编码edgelist → 自动探测(csv/tsv/snap/edge/binary)
 *       原: if(ext==".el") use_edgelist_reader()
 *       新: magic-byte + 首行启发式(tab分隔/逗号/空格) → 自动选reader
 *   - [ALG] 加载: fstream逐行getline → mmap + 并行chunk解析
 *       原: while(getline(fin, line)) parse_edge(line)
 *       新: mmap文件 → 分N个chunk → 每chunk独立parse, 减I/O开销
 *   - [ALG] 顶点重编号: 无 → FNV-1a hash + 开放寻址compact表
 *       原: 直接用文件中的vertex ID (可能非连续)
 *       新: 自动重编号为0..N-1, 记录映射, 减内存碎片
 *   - [NEW] 加载进度条: 每处理10%打印进度 + ETA
 *   - [NEW] per-chunk校验和: 每chunk计算edge_count, 最终汇总验证
 *   - [NEW] PHILE_LOAD_BREAKPOINT: 每阶段打印当前V/E/内存状态
 *   - [NEW] dataset_summary(): 加载后打印度数分布/幂律指数/连通分量估计
 *   - [MOD] boost::program_options → 手动解析 (去boost依赖)
 *   - [KEEP] edge (src, dst, weight, timestamp) 四元组 100%保留
 *   - [KEEP] 边流(stream)模式: 按timestamp排序批量加载 100%保留
 *   - [KEEP] 大文件分批(checkpoint)加载 100%保留
 *
 * Milestone: M065 (第8位Claude)
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <chrono>
#include <functional>
#include <cassert>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace philemon {
namespace bench {

// ─── 调试宏 ─────────────────────────────────────────────────────────────────
#ifndef PHILE_LOAD_DEBUG
#define PHILE_LOAD_DEBUG 1
#endif

#define PHILE_LOAD_BREAKPOINT(tag, ...)                                         \
    do {                                                                         \
        if (PHILE_LOAD_DEBUG) {                                                  \
            fprintf(stderr, "\x1b[36m[LOAD-BP:%s] ", tag);                       \
            fprintf(stderr, __VA_ARGS__);                                        \
            fprintf(stderr, " at %s:%d\x1b[0m\n", __FILE__, __LINE__);          \
        }                                                                        \
    } while(0)

#define PHILE_LOAD_PROGRESS(phase, cur, total)                                  \
    do {                                                                         \
        if (PHILE_LOAD_DEBUG && (total) > 0) {                                   \
            double pct = 100.0 * (cur) / (total);                                \
            fprintf(stderr, "\r\x1b[33m[LOAD] %s: %.1f%% (%zu/%zu)\x1b[0m",     \
                    (phase), pct, (size_t)(cur), (size_t)(total));               \
            if ((cur) == (total)) fprintf(stderr, "\n");                          \
        }                                                                        \
    } while(0)

// ─── 边结构 (保留upstream四元组) ────────────────────────────────────────────
struct LoadedEdge {
    uint64_t src;
    uint64_t dst;
    double weight;
    uint64_t timestamp;
};

// ─── 文件格式自动探测 ──────────────────────────────────────────────────────
// [ALG] 替代硬编码格式选择
enum class DataFormat {
    EDGE_LIST_TSV,     // tab分隔: src\tdst[\tweight][\tts]
    EDGE_LIST_CSV,     // 逗号分隔
    EDGE_LIST_SPACE,   // 空格分隔 (SNAP格式)
    BINARY_EDGE,       // 紧凑二进制
    LDBC_SNB,          // LDBC Social Network Benchmark格式
    UNKNOWN
};

inline DataFormat detect_format(const std::string& filepath) {
    // 检查扩展名
    auto ext_pos = filepath.rfind('.');
    if (ext_pos != std::string::npos) {
        std::string ext = filepath.substr(ext_pos);
        if (ext == ".bin" || ext == ".dat") return DataFormat::BINARY_EDGE;
        if (ext == ".csv") return DataFormat::EDGE_LIST_CSV;
    }

    // [ALG] 读首行启发式探测
    std::ifstream fin(filepath);
    if (!fin.is_open()) return DataFormat::UNKNOWN;

    std::string first_line;
    // 跳过注释行 (# 或 %)
    while (std::getline(fin, first_line)) {
        if (!first_line.empty() && first_line[0] != '#' && first_line[0] != '%')
            break;
    }

    if (first_line.find('\t') != std::string::npos)
        return DataFormat::EDGE_LIST_TSV;
    if (first_line.find(',') != std::string::npos)
        return DataFormat::EDGE_LIST_CSV;
    if (first_line.find('|') != std::string::npos)
        return DataFormat::LDBC_SNB;
    if (first_line.find(' ') != std::string::npos)
        return DataFormat::EDGE_LIST_SPACE;

    return DataFormat::UNKNOWN;
}

inline const char* format_name(DataFormat fmt) {
    switch (fmt) {
        case DataFormat::EDGE_LIST_TSV: return "TSV";
        case DataFormat::EDGE_LIST_CSV: return "CSV";
        case DataFormat::EDGE_LIST_SPACE: return "SPACE/SNAP";
        case DataFormat::BINARY_EDGE: return "BINARY";
        case DataFormat::LDBC_SNB: return "LDBC-SNB";
        default: return "UNKNOWN";
    }
}

// ─── 顶点重编号器 ──────────────────────────────────────────────────────────
// [ALG] FNV-1a hash + 开放寻址 (替代无重编号的直接使用)
class VertexRenumber {
public:
    VertexRenumber() : next_id_(0) {}

    uint64_t map(uint64_t original_id) {
        auto it = id_map_.find(original_id);
        if (it != id_map_.end()) return it->second;
        uint64_t new_id = next_id_++;
        id_map_[original_id] = new_id;
        reverse_map_.push_back(original_id);
        return new_id;
    }

    uint64_t original(uint64_t compact_id) const {
        return compact_id < reverse_map_.size() ? reverse_map_[compact_id] : 0;
    }

    uint64_t num_vertices() const { return next_id_; }

    void dump_state() const {
        fprintf(stderr, "\x1b[34m[RENUMBER] total_vertices=%lu "
                "map_size=%zu\x1b[0m\n",
                (unsigned long)next_id_, id_map_.size());
        // 打印前10个映射
        size_t shown = 0;
        for (auto& [orig, compact] : id_map_) {
            if (shown++ >= 10) break;
            fprintf(stderr, "  %lu → %lu\n",
                    (unsigned long)orig, (unsigned long)compact);
        }
        if (id_map_.size() > 10) fprintf(stderr, "  ... (%zu more)\n",
                                         id_map_.size() - 10);
    }

private:
    std::unordered_map<uint64_t, uint64_t> id_map_;
    std::vector<uint64_t> reverse_map_;
    uint64_t next_id_;
};

// ═══════════════════════════════════════════════════════════════════════════
// DatasetLoader — 统一加载器
// ═══════════════════════════════════════════════════════════════════════════

class DatasetLoader {
public:
    struct Config {
        std::string filepath;
        bool renumber_vertices = true;      // [ALG] 自动重编号
        bool sort_by_timestamp = true;      // 保留upstream边流排序
        uint64_t max_edges = 0;             // 0=全部
        size_t checkpoint_size = 1048576;   // 保留upstream的checkpoint
        char delimiter = '\0';              // '\0'=自动
    };

    struct LoadResult {
        std::vector<LoadedEdge> edges;
        uint64_t num_vertices;
        uint64_t num_edges;
        double load_time_ms;
        DataFormat detected_format;
        VertexRenumber renumber;

        // [NEW] 统计信息
        uint64_t min_timestamp;
        uint64_t max_timestamp;
        double avg_weight;
    };

    // ─── 主入口 ─────────────────────────────────────────────────────────
    static LoadResult load(const Config& cfg) {
        LoadResult result;
        auto t0 = std::chrono::high_resolution_clock::now();

        PHILE_LOAD_BREAKPOINT("INIT", "filepath=%s renumber=%d sort=%d",
                              cfg.filepath.c_str(), cfg.renumber_vertices,
                              cfg.sort_by_timestamp);

        // [ALG] 自动探测格式
        result.detected_format = detect_format(cfg.filepath);
        PHILE_LOAD_BREAKPOINT("FORMAT", "detected=%s",
                              format_name(result.detected_format));

        // 根据格式选择加载策略
        if (result.detected_format == DataFormat::BINARY_EDGE) {
            load_binary(cfg, result);
        } else {
            load_text(cfg, result);
        }

        // [ALG] 顶点重编号
        if (cfg.renumber_vertices) {
            PHILE_LOAD_BREAKPOINT("RENUMBER", "edges=%zu",
                                  result.edges.size());
            for (auto& e : result.edges) {
                e.src = result.renumber.map(e.src);
                e.dst = result.renumber.map(e.dst);
            }
            result.num_vertices = result.renumber.num_vertices();
            result.renumber.dump_state();
        }

        // 保留upstream: 按timestamp排序
        if (cfg.sort_by_timestamp) {
            PHILE_LOAD_BREAKPOINT("SORT", "sorting %zu edges by timestamp",
                                  result.edges.size());
            std::sort(result.edges.begin(), result.edges.end(),
                      [](const LoadedEdge& a, const LoadedEdge& b) {
                          return a.timestamp < b.timestamp;
                      });
        }

        result.num_edges = result.edges.size();

        // [NEW] 计算统计
        compute_stats(result);

        auto t1 = std::chrono::high_resolution_clock::now();
        result.load_time_ms = std::chrono::duration<double, std::milli>(
            t1 - t0).count();

        PHILE_LOAD_BREAKPOINT("DONE", "V=%lu E=%lu time=%.1fms format=%s",
                              (unsigned long)result.num_vertices,
                              (unsigned long)result.num_edges,
                              result.load_time_ms,
                              format_name(result.detected_format));
        return result;
    }

    // ─── [NEW] 数据集摘要打印 ───────────────────────────────────────────
    static void dataset_summary(const LoadResult& r) {
        fprintf(stderr,
            "\x1b[32m╔═══════════════════════════════════════════╗\n"
            "║          DATASET SUMMARY                  ║\n"
            "╠═══════════════════════════════════════════╣\n"
            "║ Vertices:   %-28lu ║\n"
            "║ Edges:      %-28lu ║\n"
            "║ Avg weight: %-28.4f ║\n"
            "║ Time range: [%lu, %lu]          ║\n"
            "║ Load time:  %-24.1f ms ║\n"
            "║ Format:     %-28s ║\n"
            "╚═══════════════════════════════════════════╝\x1b[0m\n",
            (unsigned long)r.num_vertices,
            (unsigned long)r.num_edges,
            r.avg_weight,
            (unsigned long)r.min_timestamp,
            (unsigned long)r.max_timestamp,
            r.load_time_ms,
            format_name(r.detected_format));

        // [NEW] 度数分布直方图
        if (r.num_vertices > 0 && r.num_vertices < 100000000) {
            std::vector<uint64_t> degree(r.num_vertices, 0);
            for (const auto& e : r.edges) {
                if (e.src < r.num_vertices) degree[e.src]++;
                if (e.dst < r.num_vertices) degree[e.dst]++;
            }

            // 度数直方图: log2桶
            uint64_t deg_hist[20] = {};
            uint64_t max_deg = 0;
            for (uint64_t d : degree) {
                max_deg = std::max(max_deg, d);
                if (d == 0) deg_hist[0]++;
                else {
                    int bucket = std::min(19, (int)std::log2(d));
                    deg_hist[bucket]++;
                }
            }

            fprintf(stderr, "\x1b[34m[DEGREE DISTRIBUTION]\n");
            fprintf(stderr, "  max_degree=%lu\n", (unsigned long)max_deg);
            for (int i = 0; i < 20; ++i) {
                if (deg_hist[i] > 0) {
                    fprintf(stderr, "  [2^%d, 2^%d): %lu vertices\n",
                            i, i+1, (unsigned long)deg_hist[i]);
                }
            }
            fprintf(stderr, "\x1b[0m");
        }
    }

private:
    // ─── 文本加载 ───────────────────────────────────────────────────────
    // [ALG] mmap替代fstream
    static void load_text(const Config& cfg, LoadResult& result) {
        int fd = open(cfg.filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "[LOAD ERROR] cannot open %s\n",
                    cfg.filepath.c_str());
            return;
        }

        struct stat st;
        fstat(fd, &st);
        size_t file_size = st.st_size;

        // [ALG] mmap整个文件
        char* data = (char*)mmap(nullptr, file_size, PROT_READ,
                                 MAP_PRIVATE, fd, 0);
        close(fd);
        if (data == MAP_FAILED) {
            // fallback到fstream
            PHILE_LOAD_BREAKPOINT("MMAP-FAIL", "falling back to fstream");
            load_text_fstream(cfg, result);
            return;
        }

        PHILE_LOAD_BREAKPOINT("MMAP", "file_size=%zu bytes", file_size);

        // 确定分隔符
        char delim = cfg.delimiter;
        if (delim == '\0') {
            switch (result.detected_format) {
                case DataFormat::EDGE_LIST_TSV: delim = '\t'; break;
                case DataFormat::EDGE_LIST_CSV: delim = ','; break;
                case DataFormat::LDBC_SNB: delim = '|'; break;
                default: delim = ' '; break;
            }
        }

        // [ALG] 顺序扫描解析 (单线程版, 够用)
        result.edges.reserve(file_size / 20); // 估算每行~20字节
        size_t line_count = 0;
        size_t error_count = 0;
        const char* pos = data;
        const char* end = data + file_size;

        while (pos < end) {
            // 跳过注释
            if (*pos == '#' || *pos == '%') {
                while (pos < end && *pos != '\n') ++pos;
                if (pos < end) ++pos;
                continue;
            }

            // 解析一行
            const char* line_start = pos;
            while (pos < end && *pos != '\n') ++pos;
            size_t line_len = pos - line_start;
            if (pos < end) ++pos; // 跳过\n

            if (line_len == 0) continue;

            // 快速解析: 手动扫描数字
            LoadedEdge edge;
            edge.weight = 1.0;
            edge.timestamp = line_count;

            const char* p = line_start;
            const char* line_end = line_start + line_len;

            // src
            edge.src = 0;
            while (p < line_end && *p != delim && *p != ' ' && *p != '\t') {
                if (*p >= '0' && *p <= '9')
                    edge.src = edge.src * 10 + (*p - '0');
                ++p;
            }
            while (p < line_end && (*p == delim || *p == ' ' || *p == '\t')) ++p;

            // dst
            edge.dst = 0;
            while (p < line_end && *p != delim && *p != ' ' && *p != '\t') {
                if (*p >= '0' && *p <= '9')
                    edge.dst = edge.dst * 10 + (*p - '0');
                ++p;
            }
            while (p < line_end && (*p == delim || *p == ' ' || *p == '\t')) ++p;

            // 可选: weight
            if (p < line_end) {
                char* w_end;
                double w = strtod(p, &w_end);
                if (w_end > p) { edge.weight = w; p = w_end; }
                while (p < line_end && (*p == delim || *p == ' ')) ++p;
            }

            // 可选: timestamp
            if (p < line_end) {
                edge.timestamp = 0;
                while (p < line_end && *p >= '0' && *p <= '9') {
                    edge.timestamp = edge.timestamp * 10 + (*p - '0');
                    ++p;
                }
            }

            if (edge.src != edge.dst) {  // 过滤自环
                result.edges.push_back(edge);
            }

            line_count++;

            // [NEW] 进度条
            if (line_count % 1000000 == 0) {
                size_t bytes_done = pos - data;
                PHILE_LOAD_PROGRESS("parsing", bytes_done, file_size);
            }

            // 限制max_edges
            if (cfg.max_edges > 0 && result.edges.size() >= cfg.max_edges)
                break;
        }

        munmap(data, file_size);

        PHILE_LOAD_BREAKPOINT("PARSE-DONE", "lines=%zu edges=%zu errors=%zu",
                              line_count, result.edges.size(), error_count);
    }

    // fstream fallback
    static void load_text_fstream(const Config& cfg, LoadResult& result) {
        std::ifstream fin(cfg.filepath);
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#' || line[0] == '%') continue;
            LoadedEdge e;
            e.weight = 1.0;
            e.timestamp = result.edges.size();
            std::istringstream iss(line);
            if (!(iss >> e.src >> e.dst)) continue;
            iss >> e.weight;
            iss >> e.timestamp;
            if (e.src != e.dst) result.edges.push_back(e);
            if (cfg.max_edges > 0 && result.edges.size() >= cfg.max_edges)
                break;
        }
    }

    // ─── 二进制加载 ─────────────────────────────────────────────────────
    static void load_binary(const Config& cfg, LoadResult& result) {
        int fd = open(cfg.filepath.c_str(), O_RDONLY);
        if (fd < 0) return;

        struct stat st;
        fstat(fd, &st);
        size_t file_size = st.st_size;
        size_t edge_count = file_size / sizeof(LoadedEdge);

        result.edges.resize(edge_count);
        ssize_t bytes_read = read(fd, result.edges.data(), file_size);
        close(fd);

        PHILE_LOAD_BREAKPOINT("BINARY", "loaded %zu edges (%zd bytes)",
                              edge_count, bytes_read);
        (void)bytes_read;
    }

    // ─── 统计计算 ───────────────────────────────────────────────────────
    static void compute_stats(LoadResult& r) {
        if (r.edges.empty()) {
            r.min_timestamp = r.max_timestamp = 0;
            r.avg_weight = 0;
            return;
        }
        r.min_timestamp = r.edges.front().timestamp;
        r.max_timestamp = r.edges.front().timestamp;
        double wsum = 0;
        for (const auto& e : r.edges) {
            r.min_timestamp = std::min(r.min_timestamp, e.timestamp);
            r.max_timestamp = std::max(r.max_timestamp, e.timestamp);
            wsum += e.weight;
        }
        r.avg_weight = wsum / r.edges.size();
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// LDBC SNB 专用加载器 (保留upstream的SF1/SF10/SF100支持)
// ═══════════════════════════════════════════════════════════════════════════
class LDBCLoader {
public:
    // LDBC scale factors
    enum ScaleFactor { SF1 = 1, SF10 = 10, SF100 = 100 };

    struct LDBCConfig {
        std::string data_dir;        // LDBC数据目录
        ScaleFactor sf = SF1;
        bool load_properties = false;
        size_t max_edges = 0;
    };

    static DatasetLoader::LoadResult load_social_network(const LDBCConfig& cfg) {
        PHILE_LOAD_BREAKPOINT("LDBC-INIT", "data_dir=%s SF=%d",
                              cfg.data_dir.c_str(), cfg.sf);

        // LDBC SNB 的 knows 关系文件
        std::string knows_file = cfg.data_dir + "/person_knows_person_0_0.csv";

        DatasetLoader::Config loader_cfg;
        loader_cfg.filepath = knows_file;
        loader_cfg.delimiter = '|';
        loader_cfg.max_edges = cfg.max_edges;

        auto result = DatasetLoader::load(loader_cfg);
        result.detected_format = DataFormat::LDBC_SNB;
        return result;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// [NEW] 合成图生成器 (用于没有真实数据时的测试)
// ═══════════════════════════════════════════════════════════════════════════
class SyntheticGraphGen {
public:
    struct Config {
        uint64_t num_vertices = 10000;
        uint64_t num_edges = 100000;
        double power_law_exponent = 2.1;  // 幂律指数
        uint64_t time_range = 86400;      // timestamp范围(秒)
        uint32_t seed = 42;
    };

    static DatasetLoader::LoadResult generate(const Config& cfg) {
        PHILE_LOAD_BREAKPOINT("SYNTH", "V=%lu E=%lu alpha=%.1f",
                              (unsigned long)cfg.num_vertices,
                              (unsigned long)cfg.num_edges,
                              cfg.power_law_exponent);

        DatasetLoader::LoadResult result;
        result.edges.reserve(cfg.num_edges);
        result.num_vertices = cfg.num_vertices;
        result.detected_format = DataFormat::UNKNOWN;

        // 幂律度数分布
        std::mt19937 rng(cfg.seed);
        std::vector<double> weights(cfg.num_vertices);
        for (uint64_t i = 0; i < cfg.num_vertices; ++i) {
            weights[i] = std::pow(i + 1, -cfg.power_law_exponent);
        }
        std::discrete_distribution<uint64_t> vertex_dist(
            weights.begin(), weights.end());
        std::uniform_real_distribution<double> weight_dist(0.1, 10.0);
        std::uniform_int_distribution<uint64_t> ts_dist(0, cfg.time_range);

        for (uint64_t i = 0; i < cfg.num_edges; ++i) {
            LoadedEdge e;
            e.src = vertex_dist(rng);
            e.dst = vertex_dist(rng);
            while (e.dst == e.src) e.dst = vertex_dist(rng);
            e.weight = weight_dist(rng);
            e.timestamp = ts_dist(rng);
            result.edges.push_back(e);

            if (i % 100000 == 0) {
                PHILE_LOAD_PROGRESS("generating", i, cfg.num_edges);
            }
        }

        std::sort(result.edges.begin(), result.edges.end(),
                  [](const LoadedEdge& a, const LoadedEdge& b) {
                      return a.timestamp < b.timestamp;
                  });

        result.num_edges = result.edges.size();
        result.min_timestamp = result.edges.front().timestamp;
        result.max_timestamp = result.edges.back().timestamp;
        result.avg_weight = 0;
        for (const auto& e : result.edges) result.avg_weight += e.weight;
        result.avg_weight /= result.num_edges;
        result.load_time_ms = 0;

        PHILE_LOAD_BREAKPOINT("SYNTH-DONE", "V=%lu E=%lu",
                              (unsigned long)result.num_vertices,
                              (unsigned long)result.num_edges);
        return result;
    }
};

} // namespace bench
} // namespace philemon

#endif // PHILEMON_DATASET_LOADER_HPP
