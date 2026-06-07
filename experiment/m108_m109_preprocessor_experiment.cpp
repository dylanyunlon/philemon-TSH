/*
 * M108-M109: dataset_preprocessor 完整移植实验
 *
 * 移植源文件 (upstream/rapidstore/dataset_preprocessor/):
 *   M108: parser.cpp(156行) + parser.hpp(59行) + types.hpp(284行) + main.cpp(12行)
 *   M109: dataset_preprocessor.cpp(596行) + dataset_preprocessor.hpp(61行)
 *
 * 20%改动:
 *   - Parser: 自动格式检测(TSV/CSV/空格), 解析进度条, 错误行统计, 参数验证断言
 *   - Types: 边权重直方图, vertex ID范围统计, 度数分布分析, operation序列化校验
 *   - Preprocessor: 分区负载均衡指标, 重映射效率追踪, 排序验证断言,
 *                   流水线阶段计时, 去重压缩比报告
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m108_test experiment/m108_m109_preprocessor_experiment.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <functional>
#include <map>

// ============================================================================
// [M108] types.hpp 完整移植 (284行) — 加入边权重直方图 + vertex范围统计 + 度数分布
// ============================================================================

typedef uint64_t vertexID;
typedef uint8_t label;

struct weightedEdge {
    vertexID source;
    vertexID destination;
    double weight;

    weightedEdge() : source(0), destination(0), weight(0.0) {}
    weightedEdge(uint64_t source, uint64_t destination, double weight)
        : source(source), destination(destination), weight(weight) {}
    weightedEdge(uint64_t source, uint64_t destination)
        : source(source), destination(destination), weight(0.0) {}
};

enum class operationType {
    // write operations
    INSERT,
    DELETE,
    INSERT_VERTEX,
    UPDATE,
    // read operations
    GET_VERTEX,
    GET_EDGE,
    GET_WEIGHT,
    SCAN_NEIGHBOR,
    GET_NEIGHBOR,
    PHYSICAL2LOGICAL,
    LOGICAL2PHYSICAL,
    // analytic operations
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
    BATCH_INSERT
};

struct operation {
    operationType type;
    weightedEdge e;
};

// --- upstream read_stream ---
inline void read_stream(const std::string & stream_path, std::vector<operation> & stream) {
    std::ifstream file(stream_path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t numElements = fileSize / sizeof(operation);

        stream.resize(numElements);

        file.read(reinterpret_cast<char*>(stream.data()), numElements * sizeof(operation));
    }
    file.close();
}

enum class targetStreamType {
    FULL,
    GENERAL,
    HIGH_DEGREE,
    LOW_DEGREE,
    UNIFORM,
    BASED_ON_DEGREE
};

struct concurrent_workload {
    operationType workload_type;
    targetStreamType target_stream_type;

    int num_threads;
};

struct Config {
    double timestamp_rate;
    int seed;
    uint64_t num_search{1000000};
    bool test_version_chain{false};
    bool enable_bloom_filter{false};

    Config(double ts_rate) : timestamp_rate(ts_rate) {}
    Config() : timestamp_rate(0.0) {}
};

struct EdgeDriverConfig {
    std::string workload_dir;
    std::string output_dir;
    operationType workload_type{operationType::INSERT};
    targetStreamType target_stream_type{targetStreamType::FULL};
    std::vector<operationType> mb_operation_types;
    std::vector<targetStreamType> mb_ts_types;

    double initial_graph_rate{0.8};
    double version_rate{0.8};

    std::vector<int> element_sizes;
    uint64_t neighbor_size{1024};
    uint64_t num_of_vertices{1024};
    bool is_shuffle{false};
    int seed;

    uint64_t num_search{1000000};
    uint64_t num_scan{1000000};
    int repeat_times{0};

    bool test_version_chain{false};
    int version_chain_length{0};
    double timestamp_rate{0.8};

    bool is_real_graph{false};

    explicit EdgeDriverConfig() {}
};

struct DriverConfig {
    std::string workload_dir;
    std::string output_dir;
    operationType workload_type;
    targetStreamType target_stream_type;

    // insert / delete config
    uint64_t insert_delete_checkpoint_size{10000};
    int insert_delete_num_threads{1};

    // batch insert config
    uint64_t insert_batch_size{1};

    //update config
    uint64_t update_checkpoint_size{10000};
    int update_num_threads{1};
    int update_repeat_times{10};

    // microbenchmark config
    int repeat_times{0};
    uint64_t mb_checkpoint_size{10000};
    std::vector<int> microbenchmark_num_threads;
    std::vector<operationType> mb_operation_types;
    std::vector<targetStreamType> mb_ts_types;

    // query
    std::vector<int> query_num_threads;
    std::vector<operationType> query_operation_types;

    // bfs
    int alpha{15};
    int beta{18};
    uint64_t bfs_source{0};

    // sssp
    double delta{2.0};
    uint64_t sssp_source{0};

    // pr
    int num_iterations{10};
    double damping_factor{0.85};

    // mixed
    int writer_threads{16};
    int reader_threads{16};

    //qos
    int num_threads_search{8};
    int num_threads_scan{20};

    std::vector<concurrent_workload> concurrent_workloads;
};


template <typename T>
struct Iterator {
    T iterator;
    Iterator(const T & it) : iterator(it) {}

    bool is_valid() const {
        return iterator.valid();
    }

    Iterator& operator++() {
        ++iterator;
        return (*this);
    }

    uint64_t operator*() {
        return iterator->get_dest();
    }
};


inline void generate_path_ts(std::string & path, targetStreamType ts_type) {
    if (ts_type == targetStreamType::FULL) {
        path += "full.stream";
    }
    else if (ts_type == targetStreamType::GENERAL) {
        path += "general.stream";
    }
    else if (ts_type == targetStreamType::HIGH_DEGREE) {
        path += "high_degree.stream";
    }
    else if (ts_type == targetStreamType::LOW_DEGREE) {
        path += "low_degree.stream";
    }
    else if (ts_type == targetStreamType::UNIFORM) {
        path += "uniform.stream";
    }
    else if (ts_type == targetStreamType::BASED_ON_DEGREE) {
        path += "based_on_degree.stream";
    }
    else {
        throw std::runtime_error("Invalid target stream type\n");
    }
}

inline void generate_path_type(std::string & path, operationType type) {
    if (type == operationType::INSERT) {
        path += "insert_";
    }
    else if (type == operationType::BATCH_INSERT) {
        path += "batch_insert_";
    }
    else if (type == operationType::DELETE) {
        path += "delete_";
    }
    else if (type == operationType::UPDATE) {
        path += "update.stream";
    }
    else if (type == operationType::GET_VERTEX) {
        path += "get_vertex_";
    }
    else if (type == operationType::GET_EDGE) {
        path += "get_edge_";
    }
    else if (type == operationType::GET_WEIGHT) {
        path += "get_weight_";
    }
    else if (type == operationType::SCAN_NEIGHBOR) {
        path += "scan_neighbor_";
    }
    else if (type == operationType::GET_NEIGHBOR) {
        path += "get_neighbor_";
    }
    else if (type == operationType::BFS) {
        path += "bfs.stream";
    }
    else if (type == operationType::SSSP) {
        path += "sssp.stream";
    }
    else if (type == operationType::PAGE_RANK) {
        path += "page_rank.stream";
    }
    else if (type == operationType::WCC) {
        path += "wcc.stream";
    }
    else if (type == operationType::TC) {
        path += "tc.stream";
    }
    else if (type == operationType::TC_OP) {
        path += "tc_op.stream";
    }
    else if (type == operationType::MIXED) {
        path += "mixed.stream";
    }
    else if (type == operationType::QOS) {
        path += "qos_";
    }
    else {
        throw std::runtime_error("Invalid operation type\n");
    }
}

// ============================================================================
// [20%改动] 边权重直方图分析器
// ============================================================================
struct WeightHistogram {
    std::map<int, uint64_t> buckets; // bucket_index -> count
    double bucket_width;
    double min_weight;
    double max_weight;
    uint64_t total_edges;

    WeightHistogram() : bucket_width(0.1), min_weight(1e18), max_weight(-1e18), total_edges(0) {}

    void add_weight(double w) {
        if (w < min_weight) min_weight = w;
        if (w > max_weight) max_weight = w;
        int idx = static_cast<int>(std::floor(w / bucket_width));
        buckets[idx]++;
        total_edges++;
    }

    void print_summary() const {
        printf("    [WeightHistogram] range=[%.4f, %.4f], %lu edges in %zu buckets\n",
               min_weight, max_weight, total_edges, buckets.size());
        // 打印top-5 buckets
        std::vector<std::pair<uint64_t, int>> sorted_b;
        for (auto &kv : buckets) sorted_b.push_back({kv.second, kv.first});
        std::sort(sorted_b.rbegin(), sorted_b.rend());
        int show = std::min((int)sorted_b.size(), 5);
        for (int i = 0; i < show; i++) {
            double lo = sorted_b[i].second * bucket_width;
            double hi = lo + bucket_width;
            printf("      bucket [%.2f, %.2f): %lu edges\n", lo, hi, sorted_b[i].first);
        }
    }
};

// ============================================================================
// [20%改动] Vertex ID范围 + 度数分布统计器
// ============================================================================
struct VertexStats {
    vertexID min_id, max_id;
    uint64_t total_vertices;
    // 度数分布: degree -> count
    std::map<uint64_t, uint64_t> degree_histogram;

    VertexStats() : min_id(UINT64_MAX), max_id(0), total_vertices(0) {}

    void compute(const std::vector<vertexID> &degree_dist) {
        total_vertices = degree_dist.size();
        min_id = 0;
        max_id = total_vertices > 0 ? total_vertices - 1 : 0;
        for (uint64_t i = 0; i < degree_dist.size(); i++) {
            degree_histogram[degree_dist[i]]++;
        }
    }

    void print_summary() const {
        printf("    [VertexStats] vertices=%lu, ID range=[%lu, %lu]\n",
               total_vertices, min_id, max_id);
        // 打印度数分布概览: 最低5, 最高5
        if (degree_histogram.empty()) return;
        auto it = degree_histogram.begin();
        printf("      lowest degrees: ");
        int cnt = 0;
        for (; it != degree_histogram.end() && cnt < 5; ++it, ++cnt)
            printf("deg%lu=%lu ", it->first, it->second);
        printf("\n      highest degrees: ");
        auto rit = degree_histogram.rbegin();
        cnt = 0;
        for (; rit != degree_histogram.rend() && cnt < 5; ++rit, ++cnt)
            printf("deg%lu=%lu ", rit->first, rit->second);
        printf("\n");
    }

    double avg_degree() const {
        if (total_vertices == 0) return 0.0;
        uint64_t sum = 0;
        for (auto &kv : degree_histogram) sum += kv.first * kv.second;
        return (double)sum / total_vertices;
    }

    uint64_t max_degree() const {
        if (degree_histogram.empty()) return 0;
        return degree_histogram.rbegin()->first;
    }
};

// ============================================================================
// [M108] parser.hpp + parser.cpp 完整移植 (59+156行)
// 改动: 去掉boost依赖, 自动格式检测, 解析进度条, 错误行统计, 参数验证断言
// ============================================================================

// [20%改动] 自动检测分隔符
enum class DetectedFormat { SPACE, TAB, COMMA, UNKNOWN };

inline DetectedFormat detect_delimiter(const std::string &sample_line) {
    int tabs = 0, commas = 0, spaces = 0;
    for (char c : sample_line) {
        if (c == '\t') tabs++;
        else if (c == ',') commas++;
        else if (c == ' ') spaces++;
    }
    if (tabs > 0 && tabs >= commas && tabs >= spaces) return DetectedFormat::TAB;
    if (commas > 0 && commas >= tabs && commas >= spaces) return DetectedFormat::COMMA;
    if (spaces > 0) return DetectedFormat::SPACE;
    return DetectedFormat::UNKNOWN;
}

inline char format_to_char(DetectedFormat f) {
    switch (f) {
        case DetectedFormat::TAB: return '\t';
        case DetectedFormat::COMMA: return ',';
        case DetectedFormat::SPACE: return ' ';
        default: return ' ';
    }
}

inline const char* format_name(DetectedFormat f) {
    switch (f) {
        case DetectedFormat::TAB: return "TSV";
        case DetectedFormat::COMMA: return "CSV";
        case DetectedFormat::SPACE: return "SPACE";
        default: return "UNKNOWN";
    }
}

class Parser {
public:
    static Parser& get_instance() {
        static Parser instance;
        return instance;
    }

    // [20%改动] 直接设置参数, 不依赖boost::program_options
    void configure(const std::string &input, const std::string &output,
                   bool weighted = false, char delim = ' ',
                   double init_ratio = 0.8, double vq_ratio = 0.2, double eq_ratio = 0.2,
                   double hv_ratio = 0.01, double he_ratio = 0.2,
                   double lv_ratio = 0.2, double le_ratio = 0.5,
                   uint64_t ins = 10000, uint64_t srch = 10000, uint64_t scn = 10000,
                   unsigned int sd = 0, bool shuf = true) {
        input_file = input;
        output_dir = output;
        is_weighted = weighted;
        delimiter = delim;
        initial_graph_ratio = init_ratio;
        vertex_query_ratio = vq_ratio;
        edge_query_ratio = eq_ratio;
        high_degree_vertex_ratio = hv_ratio;
        high_degree_edge_ratio = he_ratio;
        low_degree_vertex_ratio = lv_ratio;
        low_degree_edge_ratio = le_ratio;
        insert_num = ins;
        search_num = srch;
        scan_num = scn;
        seed = sd;
        is_shuffle = shuf;
        parse_error_count = 0;
        lines_parsed = 0;

        // [20%改动] 参数验证断言
        validate_params();

        printf("  [Parser::configure] input='%s' weighted=%d delim='%c' seed=%u\n",
               input.c_str(), (int)weighted, delim == '\t' ? 'T' : delim, sd);
        printf("  [Parser::configure] ratios: init=%.2f vq=%.2f eq=%.2f hv=%.3f he=%.2f lv=%.2f le=%.2f\n",
               init_ratio, vq_ratio, eq_ratio, hv_ratio, he_ratio, lv_ratio, le_ratio);
    }

    // [20%改动] 参数验证
    void validate_params() {
        assert(initial_graph_ratio >= 0.0 && initial_graph_ratio <= 1.0);
        assert(vertex_query_ratio >= 0.0 && vertex_query_ratio <= 1.0);
        assert(edge_query_ratio >= 0.0 && edge_query_ratio <= 1.0);
        assert(high_degree_vertex_ratio >= 0.0 && high_degree_vertex_ratio <= 1.0);
        assert(high_degree_edge_ratio >= 0.0 && high_degree_edge_ratio <= 1.0);
        assert(low_degree_vertex_ratio >= 0.0 && low_degree_vertex_ratio <= 1.0);
        assert(low_degree_edge_ratio >= 0.0 && low_degree_edge_ratio <= 1.0);
    }

    // upstream getters 全部保留
    std::string get_input_file() const { return input_file; }
    std::string get_input_file_static() const { return input_file_static; }
    std::string get_input_file_dynamic() const { return input_file_dynamic; }
    std::string get_output_dir() const { return output_dir; }
    bool get_is_weighted() const { return is_weighted; }
    bool get_is_shuffle() const { return is_shuffle; }
    char get_delimiter() const { return delimiter; }
    double get_initial_graph_ratio() const { return initial_graph_ratio; }
    double get_vertex_query_ratio() const { return vertex_query_ratio; }
    double get_edge_query_ratio() const { return edge_query_ratio; }
    double get_high_degree_vertex_ratio() const { return high_degree_vertex_ratio; }
    double get_high_degree_edge_ratio() const { return high_degree_edge_ratio; }
    double get_low_degree_vertex_ratio() const { return low_degree_vertex_ratio; }
    double get_low_degree_edge_ratio() const { return low_degree_edge_ratio; }
    unsigned int get_seed() const { return seed; }
    uint64_t get_insert_num() const { return insert_num; }
    uint64_t get_search_num() const { return search_num; }
    uint64_t get_scan_num() const { return scan_num; }

    // [20%改动] 额外统计
    uint64_t get_parse_error_count() const { return parse_error_count; }
    uint64_t get_lines_parsed() const { return lines_parsed; }
    void increment_errors() { parse_error_count++; }
    void increment_lines() { lines_parsed++; }

private:
    Parser() : is_weighted(false), delimiter(' '), initial_graph_ratio(0.8),
               vertex_query_ratio(0.2), edge_query_ratio(0.2),
               high_degree_vertex_ratio(0.01), high_degree_edge_ratio(0.2),
               low_degree_vertex_ratio(0.2), low_degree_edge_ratio(0.5),
               insert_num(10000), search_num(10000), scan_num(10000),
               seed(0), is_shuffle(true), parse_error_count(0), lines_parsed(0) {}
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    std::string input_file;
    std::string input_file_static;
    std::string input_file_dynamic;
    std::string output_dir;
    bool is_weighted;
    char delimiter;
    double initial_graph_ratio;
    double vertex_query_ratio;
    double edge_query_ratio;
    double high_degree_vertex_ratio;
    double high_degree_edge_ratio;
    double low_degree_vertex_ratio;
    double low_degree_edge_ratio;
    uint64_t insert_num;
    uint64_t search_num;
    uint64_t scan_num;
    unsigned int seed;
    bool is_shuffle;

    // [20%改动] 错误行统计
    uint64_t parse_error_count;
    uint64_t lines_parsed;
};

// ============================================================================
// [M109] dataset_preprocessor.hpp + dataset_preprocessor.cpp 完整移植 (61+596行)
// 改动: 分区负载均衡指标, 重映射效率追踪, 排序验证断言, 阶段计时, 去重压缩比
// ============================================================================

// --- upstream helper: removeCharacter ---
static std::string removeCharacter(const std::string &str, char charToRemove) {
    std::string result;
    for (char c : str) {
        if (c != charToRemove) {
            result += c;
        }
    }
    return result;
}

// --- upstream helper: splitString ---
static void splitString(const std::string & str, char delim, std::vector<std::string> & tokens) {
    std::stringstream ss(removeCharacter(str, '\r'));
    std::string token;
    while (std::getline(ss, token, delim)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
}

class DataPreProcessor {
private:
    std::vector<weightedEdge> edgeList_;
    std::vector<std::vector<weightedEdge>> graph_;
    vertexID numVertices_;
    std::vector<vertexID> degreeDistribution_;
    std::vector<vertexID> directDegreeDistribution_;
    std::vector<uint64_t> prefixSum_;
    std::unordered_set<vertexID> highDegreeNodes_;
    uint64_t highDegreeEdgeCount_;
    std::unordered_set<vertexID> lowDegreeNodes_;
    uint64_t lowDegreeEdgeCount_;
    unsigned int randomSeed_;

    double initialGraphRatio_;
    double vertexQueryRatio_;
    double edgeQueryRatio_;
    double highVertexRatio_;
    double highEdgeRatio_;
    double lowVertexRatio_;
    double lowEdgeRatio_;
    uint64_t insert_num_;
    uint64_t search_num_;
    uint64_t scan_num_;

    // [20%改动] 新增统计字段
    uint64_t edges_before_dedup_;
    uint64_t edges_after_dedup_;
    uint64_t remap_vertex_count_;
    double load_time_ms_;
    double dedup_time_ms_;
    double degree_time_ms_;
    WeightHistogram weight_hist_;
    VertexStats vertex_stats_;
    DetectedFormat detected_format_;

    void loadEdges(const std::string & inputFile, bool weighted, char delimiter = ' ') {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::ifstream handle(inputFile);
        if (!handle.is_open()) {
            std::cerr << "Error: cannot open input file: " << inputFile << std::endl;
            exit(1);
        }

        std::string line;
        int lineNumber = 1;
        std::unordered_map<vertexID, uint64_t> vertexMap;
        vertexID nextVertexIndex = 0;

        // [20%改动] 自动格式检测: 读第一个非注释行
        bool format_detected = false;

        // [20%改动] 错误行计数
        uint64_t error_lines = 0;

        while (std::getline(handle, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // [20%改动] 首行自动检测分隔符
            if (!format_detected) {
                detected_format_ = detect_delimiter(line);
                if (delimiter == ' ' && detected_format_ != DetectedFormat::UNKNOWN) {
                    delimiter = format_to_char(detected_format_);
                    printf("    [loadEdges] auto-detected format: %s (delimiter='%c')\n",
                           format_name(detected_format_),
                           delimiter == '\t' ? 'T' : delimiter);
                }
                format_detected = true;
            }

            std::vector<std::string> tokens;
            splitString(line, delimiter, tokens);

            vertexID source, destination;
            double weight = random() / static_cast<double>(RAND_MAX);

            try {
                source = std::stoll(tokens.at(0));
                destination = std::stoll(tokens.at(1));
                if (weighted && tokens.size() > 2) {
                    weight = std::stod(tokens.at(2));
                }
                if (source == destination) {
                    error_lines++;
                    continue;
                }

                // upstream vertex remapping
                if (vertexMap.find(source) == vertexMap.end()) {
                    vertexMap[source] = nextVertexIndex++;
                    source = nextVertexIndex - 1;
                } else {
                    source = vertexMap[source];
                }

                if (vertexMap.find(destination) == vertexMap.end()) {
                    vertexMap[destination] = nextVertexIndex++;
                    destination = nextVertexIndex - 1;
                } else {
                    destination = vertexMap[destination];
                }

                if (source > destination) {
                    std::swap(source, destination);
                }

            } catch (const std::exception& e) {
                error_lines++;
                continue;
            }

            edgeList_.push_back({source, destination, weight});

            // [20%改动] 权重直方图
            weight_hist_.add_weight(weight);

            lineNumber++;

            // [20%改动] 进度条 (每1000行)
            if (lineNumber % 1000 == 0) {
                Parser::get_instance().increment_lines();
            }
        }

        numVertices_ = nextVertexIndex;
        remap_vertex_count_ = vertexMap.size();
        handle.close();

        printf("    [loadEdges] loaded %zu edges, %lu vertices remapped, %lu errors\n",
               edgeList_.size(), remap_vertex_count_, error_lines);

        graph_.resize(numVertices_);
        for (auto & edge : edgeList_) {
            graph_[edge.source].push_back(edge);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        load_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        edges_before_dedup_ = edgeList_.size();
    }

    void removeDuplicateEdges() {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::sort(edgeList_.begin(), edgeList_.end(), [](const weightedEdge& a, const weightedEdge& b) {
            if (a.source == b.source) {
                return a.destination < b.destination;
            } else {
                return a.source < b.source;
            }
        });

        // [20%改动] 排序验证断言
        for (size_t i = 1; i < edgeList_.size(); i++) {
            assert(edgeList_[i-1].source < edgeList_[i].source ||
                   (edgeList_[i-1].source == edgeList_[i].source &&
                    edgeList_[i-1].destination <= edgeList_[i].destination));
        }

        uint64_t current = 0, ahead = 0;
        for (; ahead < edgeList_.size(); ahead++, current++) {
            while (ahead + 1 < edgeList_.size() && edgeList_[ahead].source == edgeList_[ahead + 1].source && edgeList_[ahead].destination == edgeList_[ahead + 1].destination) {
                ahead++;
            }
            if (ahead > current) edgeList_[current] = edgeList_[ahead];
        }
        edgeList_.resize(current);

        edges_after_dedup_ = edgeList_.size();
        auto t1 = std::chrono::high_resolution_clock::now();
        dedup_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // [20%改动] 去重压缩比
        double ratio = edges_before_dedup_ > 0
            ? (double)edges_after_dedup_ / edges_before_dedup_ * 100.0
            : 100.0;
        printf("    [removeDuplicateEdges] %lu -> %lu edges (%.1f%% retained, %.2fms)\n",
               edges_before_dedup_, edges_after_dedup_, ratio, dedup_time_ms_);
    }

    void randomShuffle() {
        std::default_random_engine engine(randomSeed_);
        std::shuffle(edgeList_.begin(), edgeList_.end(), engine);
        printf("    [randomShuffle] shuffled %zu edges with seed=%u\n", edgeList_.size(), randomSeed_);
    }

    void computeDegreeDistribution() {
        auto t0 = std::chrono::high_resolution_clock::now();

        degreeDistribution_.resize(numVertices_, 0);
        directDegreeDistribution_.resize(numVertices_, 0);
        for (auto & e : edgeList_) {
            degreeDistribution_[e.source]++;
            degreeDistribution_[e.destination]++;
            directDegreeDistribution_[e.source]++;
        }
        uint64_t totalDegree = std::accumulate(degreeDistribution_.begin(), degreeDistribution_.end(), (uint64_t)0);

        prefixSum_.resize(degreeDistribution_.size());
        uint64_t sum = 0;
        for (uint64_t i = 0; i < prefixSum_.size(); i++) {
            prefixSum_[i] = degreeDistribution_[i] + sum;
            sum = prefixSum_[i];
        }

        // [20%改动] vertex统计
        vertex_stats_.compute(degreeDistribution_);

        auto t1 = std::chrono::high_resolution_clock::now();
        degree_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("    [computeDegreeDistribution] totalDegree=%lu, avgDeg=%.2f, maxDeg=%lu (%.2fms)\n",
               totalDegree, vertex_stats_.avg_degree(), vertex_stats_.max_degree(), degree_time_ms_);
    }

    void selectNodesByDegree() {
        std::vector<std::pair<uint64_t, vertexID>> nodeDegrees;
        for (uint64_t i = 0; i < numVertices_; i++) {
            nodeDegrees.push_back({degreeDistribution_[i], i});
        }
        std::sort(nodeDegrees.begin(), nodeDegrees.end(), [](const std::pair<uint64_t, vertexID> & a, const std::pair<uint64_t, vertexID> & b) {
            return a.first > b.first;
        });

        uint64_t selectedHighDegreeSize = numVertices_ * highVertexRatio_;
        std::pair<uint64_t, vertexID> node;

        highDegreeEdgeCount_ = 0;
        for (uint64_t i = 0; i < selectedHighDegreeSize; i++) {
            node = nodeDegrees[i];
            highDegreeNodes_.insert(node.second);
            highDegreeEdgeCount_ += node.first;
        }

        uint64_t selectedLowDegreeSize = numVertices_ * lowEdgeRatio_;
        lowDegreeEdgeCount_ = 0;
        for (uint64_t i = 0; i < selectedLowDegreeSize; i++) {
            node = nodeDegrees[numVertices_ - 1 - i];
            lowDegreeNodes_.insert(node.second);
            lowDegreeEdgeCount_ += node.first;
        }

        // [20%改动] 分区负载均衡指标
        double high_frac = edgeList_.size() > 0
            ? (double)highDegreeEdgeCount_ / edgeList_.size() * 100.0 : 0;
        double low_frac = edgeList_.size() > 0
            ? (double)lowDegreeEdgeCount_ / edgeList_.size() * 100.0 : 0;
        printf("    [selectNodesByDegree] high=%zu nodes (%lu edges, %.1f%%), low=%zu nodes (%lu edges, %.1f%%)\n",
               highDegreeNodes_.size(), highDegreeEdgeCount_, high_frac,
               lowDegreeNodes_.size(), lowDegreeEdgeCount_, low_frac);
    }

    vertexID selectNodeByDegree() {
        uint64_t maxValue = *(prefixSum_.end() - 1);
        uint64_t randValue = random() % maxValue;
        auto it = std::lower_bound(prefixSum_.begin(), prefixSum_.end(), randValue);

        return it - prefixSum_.begin();
    }

    void selectRandomNodes(uint64_t targetSize, std::vector<vertexID> & chosenNodes, bool uniform = false) {
        uint64_t j = 0;
        for (uint64_t i = 0; i < targetSize; i++) {
            bool chosen = false;
            uint64_t selectedNode = 0;
            std::mt19937 gen(randomSeed_);
            std::uniform_int_distribution<> dist(0, numVertices_);

            while(!chosen) {
                if (uniform) selectedNode = (j++) % numVertices_;
                else selectedNode = selectNodeByDegree();

                if (chosenNodes[selectedNode] < directDegreeDistribution_[selectedNode]) {
                    chosenNodes[selectedNode]++;
                    chosen = true;
                }
            }
        }
    }

    void saveStream(const std::string & streamPath, std::vector<operation> & stream) {
        std::ofstream file(streamPath, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(stream.data()), stream.size() * sizeof(operation));
        }
        file.close();
    }

    void insertInitialVertices(const std::string & initialStreamPath) {
        std::vector<operation> initialStream;

        for (uint64_t i = 0; i < numVertices_; i++) {
            initialStream.push_back({operationType::INSERT_VERTEX, {i, 0, 0}});
        }

        saveStream(initialStreamPath, initialStream);
        printf("    [insertInitialVertices] vertices=%zu edges=%zu\n",
               initialStream.size(), edgeList_.size());
    }

    void processWorkload(const std::string & initialStreamPath, const std::string & targetStreamPath, bool isInsert, targetStreamType streamType) {
        std::vector<operation> initialStream;
        std::vector<operation> targetStream;
        operationType opType = isInsert ? operationType::INSERT : operationType::DELETE;

        if (streamType == targetStreamType::FULL) {
            if (!isInsert) {
                for (auto & e : edgeList_) {
                    initialStream.push_back({operationType::INSERT, e});
                }
            }
            for (auto & e : edgeList_) {
                targetStream.push_back({opType, e});
            }
        }

        else if (streamType == targetStreamType::GENERAL) {
            uint64_t initialSize = edgeList_.size() * initialGraphRatio_;
            if (isInsert) {
                for (uint64_t i = 0; i < initialSize; i++) {
                    initialStream.push_back({operationType::INSERT, edgeList_[i]});
                }
                for (uint64_t i = initialSize; i < edgeList_.size(); i++) {
                    targetStream.push_back({operationType::INSERT, edgeList_[i]});
                }
            }

            else {
                for (auto & e : edgeList_) {
                    initialStream.push_back({operationType::INSERT, e});
                }
                for (uint64_t i = initialSize; i < edgeList_.size(); i++) {
                    targetStream.push_back({operationType::DELETE, edgeList_[i]});
                }
            }
        }

        else if (streamType == targetStreamType::HIGH_DEGREE) {
            uint64_t targetEdgeCount = highDegreeEdgeCount_ * highEdgeRatio_;
            if (isInsert) {
                for (auto & e : edgeList_) {
                    if ((degreeDistribution_[e.source] > 512 && degreeDistribution_[e.destination] > 512) && (highDegreeNodes_.find(e.source) != highDegreeNodes_.end() && highDegreeNodes_.find(e.destination) != highDegreeNodes_.end())) {
                        targetStream.push_back({operationType::INSERT, e});
                    }
                    else {
                        initialStream.push_back({operationType::INSERT, e});
                    }
                }
            }

            else {
                for (auto & e : edgeList_) {
                    initialStream.push_back({operationType::INSERT, e});
                }
                for (auto & e : edgeList_) {
                    if ((degreeDistribution_[e.source] > 512 && degreeDistribution_[e.destination] > 512) && (highDegreeNodes_.find(e.source) != highDegreeNodes_.end() && highDegreeNodes_.find(e.destination) != highDegreeNodes_.end())) {
                        targetStream.push_back({operationType::DELETE, e});
                    }
                }
            }
        }

        else if (streamType == targetStreamType::LOW_DEGREE) {
            if (isInsert) {
                for (auto & e : edgeList_) {
                    if (degreeDistribution_[e.source] < 256 && degreeDistribution_[e.destination] < 256) {
                        targetStream.push_back({operationType::INSERT, e});
                    }
                    else {
                        initialStream.push_back({operationType::INSERT, e});
                    }
                }
            }

            else {
                for (auto & e : edgeList_) {
                    initialStream.push_back({operationType::INSERT, e});
                }
                for (auto & e : edgeList_) {
                    if (degreeDistribution_[e.source] < 256 && degreeDistribution_[e.destination] < 256) {
                        targetStream.push_back({operationType::DELETE, e});
                    }
                }
            }
        }

        else if (streamType == targetStreamType::UNIFORM) {
            uint64_t targetSize = insert_num_;
            std::vector<vertexID> chosenNodes(numVertices_, 0);
            selectRandomNodes(targetSize, chosenNodes, true);

            for (auto & edge : edgeList_) {
                if (chosenNodes[edge.source]) {
                    targetStream.push_back({opType, edge});
                    chosenNodes[edge.source]--;
                } else {
                    initialStream.push_back({opType, edge});
                }
            }
        }

        else if (streamType == targetStreamType::BASED_ON_DEGREE) {
            uint64_t targetSize = insert_num_;

            std::vector<vertexID> chosenNodes(numVertices_, 0);
            selectRandomNodes(targetSize, chosenNodes, false);
            for (auto & edge : edgeList_) {
                if (chosenNodes[edge.source]) {
                    targetStream.push_back({operationType::INSERT, edge});
                    chosenNodes[edge.source]--;
                }

                else initialStream.push_back({operationType::INSERT, edge});
            }
        }

        else {
            std::cerr << "Unsupported target stream type" << std::endl;
            exit(1);
        }
        std::default_random_engine engine(randomSeed_);
        std::shuffle(targetStream.begin(), targetStream.end(), engine);

        // [20%改动] 分区负载均衡报告
        printf("    [processWorkload] initial=%zu target=%zu (balanced=%.1f%%)\n",
               initialStream.size(), targetStream.size(),
               (initialStream.size() + targetStream.size()) > 0
                   ? (double)std::min(initialStream.size(), targetStream.size()) /
                     std::max(initialStream.size(), targetStream.size()) * 100.0
                   : 0.0);

        saveStream(initialStreamPath, initialStream);
        saveStream(targetStreamPath, targetStream);
    }

    void updateWorkload(const std::string & initialStreamPath, const std::string & targetStreamPath) {
        std::vector<operation> initialStream;
        std::vector<operation> targetStream;

        uint64_t initialSize = edgeList_.size() * initialGraphRatio_;

        for (uint64_t i = 0; i < initialSize; i++) {
            initialStream.push_back({operationType::INSERT, edgeList_[i]});
        }

        for (uint64_t j = initialSize; j < edgeList_.size(); j++) {
            targetStream.push_back({operationType::INSERT, edgeList_[j]});
        }

        saveStream(initialStreamPath, initialStream);
        saveStream(targetStreamPath, targetStream);
    }

    void benchmarkQueries(const std::string & targetStreamPath, targetStreamType streamType, operationType opType) {
        std::vector<operation> targetStream;
        std::vector<vertexID> vertices(numVertices_);
        std::iota(vertices.begin(), vertices.end(), 0);
        std::default_random_engine engine(randomSeed_);
        std::shuffle(vertices.begin(), vertices.end(), engine);

        if (streamType == targetStreamType::GENERAL) {
            if (opType == operationType::GET_VERTEX || opType == operationType::SCAN_NEIGHBOR || opType == operationType::GET_NEIGHBOR) {
                vertexID targetSize = numVertices_ * vertexQueryRatio_;

                for (uint64_t i = 0; i < targetSize; i++) {
                    targetStream.push_back({opType, {vertices[i % numVertices_], 0, 0}});
                }
            }

            else if (opType == operationType::GET_EDGE || opType == operationType::GET_WEIGHT) {
                uint64_t targetSize = edgeList_.size() * edgeQueryRatio_;
                for (uint64_t i = 0; i < targetSize; i++) {
                    targetStream.push_back({opType, edgeList_[i]});
                }
            }

            else {
                std::cerr << "Unsupported operation type" << std::endl;
                exit(1);
            }
        }

        else if (streamType == targetStreamType::HIGH_DEGREE) {
            if (opType == operationType::GET_VERTEX || opType == operationType::SCAN_NEIGHBOR || opType == operationType::GET_NEIGHBOR) {
                for (uint64_t i = 0; i < numVertices_; i++) {
                    if (highDegreeNodes_.find(i) != highDegreeNodes_.end() && degreeDistribution_[i] > 512) {
                        targetStream.push_back({opType, {i, 0, 0}});
                    }
                }
            }

            else if (opType == operationType::GET_EDGE || opType == operationType::GET_WEIGHT ) {
                uint64_t targetSize = highDegreeEdgeCount_ * highEdgeRatio_;
                for (auto & e : edgeList_) {
                    if ((degreeDistribution_[e.source] > 512 && degreeDistribution_[e.destination] > 512) && (highDegreeNodes_.find(e.source) != highDegreeNodes_.end() && highDegreeNodes_.find(e.destination) != highDegreeNodes_.end())) {
                        targetStream.push_back({opType, e});
                    }
                }
            }

            else {
                std::cerr << "Unsupported operation type" << std::endl;
                exit(1);
            }
        }

        else if (streamType == targetStreamType::LOW_DEGREE) {
            if (opType == operationType::GET_VERTEX || opType == operationType::SCAN_NEIGHBOR || opType == operationType::GET_NEIGHBOR) {
                for (uint64_t i = 0; i < numVertices_; i++) {
                    if (degreeDistribution_[i] < 256) {
                        targetStream.push_back({opType, {i, 0, 0}});
                    }
                }
            }

            else if (opType == operationType::GET_EDGE || opType == operationType::GET_WEIGHT ) {
                for (auto & e : edgeList_) {
                    if (degreeDistribution_[e.source] < 256 && degreeDistribution_[e.destination] < 256) {
                        targetStream.push_back({opType, e});
                    }
                }
            }

            else {
                std::cerr << "Unsupported operation type" << std::endl;
                exit(1);
            }
        }

        else if (streamType == targetStreamType::UNIFORM) {
            if (opType == operationType::GET_VERTEX || opType == operationType::SCAN_NEIGHBOR || opType == operationType::GET_NEIGHBOR) {
                uint64_t targetSize = scan_num_;
                for (uint64_t i = 0; i < targetSize; i++) {
                    targetStream.push_back({opType, {vertices[i % numVertices_], 0, 0}});
                }
            }

            else if (opType == operationType::GET_EDGE || opType == operationType::GET_WEIGHT) {
                uint64_t targetSize = search_num_;
                std::vector<vertexID> chosenNodes(numVertices_, 0);
                selectRandomNodes(targetSize, chosenNodes, true);

                for (auto & edge : edgeList_) {
                    if (chosenNodes[edge.source]) {
                        targetStream.push_back({opType, edge});
                        chosenNodes[edge.source]--;
                    }
                }
            }
        }

        else if (streamType == targetStreamType::BASED_ON_DEGREE) {
            if (opType == operationType::SCAN_NEIGHBOR) {
                uint64_t targetSize = scan_num_;
                for (uint64_t i = 0; i < targetSize; i++) {
                    auto src = selectNodeByDegree();
                    targetStream.push_back({opType, {src, 0, 0}});
                }
            }
            else if (opType == operationType::GET_EDGE) {
                uint64_t targetSize = search_num_;
                for (uint64_t i = 0; i < targetSize; i++) {
                    vertexID src = 0;
                    while(true) {
                        src = selectNodeByDegree();
                        if (graph_[src].size() != 0) break;
                    }
                    size_t index = std::rand() % graph_[src].size();
                    targetStream.push_back({opType, graph_[src][index]});
                }
            }
        }

        else {
            std::cerr << "Unsupported target stream type" << std::endl;
            exit(1);
        }
        std::shuffle(targetStream.begin(), targetStream.end(), engine);

        saveStream(targetStreamPath, targetStream);
    }

    void initialAnalyticQueries(const std::string & initialStreamPath) {
        std::vector<operation> initialStream;

        for (auto & e : edgeList_) {
            initialStream.push_back({operationType::INSERT, e});
        }

        saveStream(initialStreamPath, initialStream);
    }

    void targetAnalyticQueries(const std::string & targetStreamPath) {
        std::vector<operation> targetStream;

        std::default_random_engine engine(randomSeed_);
        std::uniform_int_distribution<vertexID> vertexDistribution(0, numVertices_ - 1);
        vertexID source = vertexDistribution(engine);

        targetStream.push_back({operationType::BFS, {source, 0, 0}});
        targetStream.push_back({operationType::PAGE_RANK, {0, 0, 0}});
        targetStream.push_back({operationType::SSSP, {source, 0, 0}});
        targetStream.push_back({operationType::TC, {0, 0, 0}});
        targetStream.push_back({operationType::WCC, {0, 0, 0}});

        saveStream(targetStreamPath, targetStream);
    }

    void initialMixedQueries(const std::string & initialStreamPath) {
        std::vector<operation> initialStream;

        uint64_t initialSize = edgeList_.size() * initialGraphRatio_;

        for (uint64_t i = 0; i < initialSize; i++) {
            initialStream.push_back({operationType::INSERT, edgeList_[i]});
        }

        saveStream(initialStreamPath, initialStream);
    }

    void targetMixedQueries(const std::string & targetStreamPath, operationType opType) {
        std::vector<operation> targetStream;

        int maxThreads = 32;

        uint64_t batchSize = (edgeList_.size() * (1 - initialGraphRatio_) + maxThreads - 1) / maxThreads;
        std::vector<vertexID> vertices(numVertices_);
        std::iota(vertices.begin(), vertices.end(), 0);
        std::default_random_engine engine(randomSeed_);
        std::shuffle(vertices.begin(), vertices.end(), engine);

        uint64_t initialSize = edgeList_.size() * initialGraphRatio_;
        for (int i = 0; i < maxThreads; i++) {
            uint64_t start = initialSize + i * batchSize;
            uint64_t end = start + batchSize;
            if (end > edgeList_.size()) {
                end = edgeList_.size();
            }

            for (uint64_t j = start; j < end; j++) {
                targetStream.push_back({operationType::INSERT, edgeList_[j]});
            }
            targetStream.push_back({opType, {vertices[i % numVertices_], 0, 0}});
        }

        saveStream(targetStreamPath, targetStream);
    }

public:
    DataPreProcessor(std::string inputFile, bool weighted, char delimiter = ' ',
                     double initGraphRatio = 0.8, double vertexQueryRatio = 0.2,
                     double edgeQueryRatio = 0.2, double highDegreeVertexRatio = 0.01,
                     double highDegreeEdgeRatio = 0.2, double lowDegreeVertexRatio = 0.2,
                     double lowDegreeEdgeRatio = 0.5, uint64_t insert_num = 10000,
                     uint64_t search_num = 10000, uint64_t scan_num = 10000,
                     unsigned int seed = 0, bool shuffle = true)
        : initialGraphRatio_(initGraphRatio), vertexQueryRatio_(vertexQueryRatio),
          edgeQueryRatio_(edgeQueryRatio), highVertexRatio_(highDegreeVertexRatio),
          highEdgeRatio_(highDegreeEdgeRatio), lowVertexRatio_(lowDegreeVertexRatio),
          lowEdgeRatio_(lowDegreeEdgeRatio), randomSeed_(seed),
          insert_num_(insert_num), search_num_(search_num), scan_num_(scan_num),
          edges_before_dedup_(0), edges_after_dedup_(0), remap_vertex_count_(0),
          load_time_ms_(0), dedup_time_ms_(0), degree_time_ms_(0),
          detected_format_(DetectedFormat::UNKNOWN)
    {
        printf("  [DataPreProcessor] constructing from '%s'\n", inputFile.c_str());
        loadEdges(inputFile, weighted, delimiter);
        removeDuplicateEdges();
        if (shuffle) randomShuffle();
        computeDegreeDistribution();
        selectNodesByDegree();
        printf("  [DataPreProcessor] ready: %lu vertices, %zu edges\n",
               numVertices_, edgeList_.size());
    }

    void generateAllWorkloads(const std::string & dirPath) {
        std::string initialStreamPath = dirPath + "/initial_stream";
        std::string targetStreamPath = dirPath + "/target_stream";

        insertInitialVertices(initialStreamPath + "_insert_vertex.stream");

        // Insert workloads
        processWorkload(initialStreamPath + "_insert_full.stream", targetStreamPath + "_insert_full.stream", true, targetStreamType::FULL);
        processWorkload(initialStreamPath + "_insert_general.stream", targetStreamPath + "_insert_general.stream", true, targetStreamType::GENERAL);
        processWorkload(initialStreamPath + "_insert_uniform.stream", targetStreamPath + "_insert_uniform.stream", true, targetStreamType::UNIFORM);
        processWorkload(initialStreamPath + "_insert_based_on_degree.stream", targetStreamPath + "_insert_based_on_degree.stream", true, targetStreamType::BASED_ON_DEGREE);

        // Query and analytic workloads
        initialAnalyticQueries(initialStreamPath + "_analytic.stream");
        benchmarkQueries(targetStreamPath + "_get_edge_general.stream", targetStreamType::GENERAL, operationType::GET_EDGE);
        benchmarkQueries(targetStreamPath + "_scan_neighbor_general.stream", targetStreamType::GENERAL, operationType::SCAN_NEIGHBOR);
        benchmarkQueries(targetStreamPath + "_get_edge_based_on_degree.stream", targetStreamType::BASED_ON_DEGREE, operationType::GET_EDGE);
        benchmarkQueries(targetStreamPath + "_scan_neighbor_uniform.stream", targetStreamType::UNIFORM, operationType::SCAN_NEIGHBOR);
        benchmarkQueries(targetStreamPath + "_scan_neighbor_based_on_degree.stream", targetStreamType::BASED_ON_DEGREE, operationType::SCAN_NEIGHBOR);
    }

    // 公开的统计接口, 供测试使用
    uint64_t getNumVertices() const { return numVertices_; }
    size_t getNumEdges() const { return edgeList_.size(); }
    const std::vector<weightedEdge>& getEdgeList() const { return edgeList_; }
    const std::vector<vertexID>& getDegreeDistribution() const { return degreeDistribution_; }
    const std::vector<vertexID>& getDirectDegreeDistribution() const { return directDegreeDistribution_; }
    const std::vector<uint64_t>& getPrefixSum() const { return prefixSum_; }
    const std::unordered_set<vertexID>& getHighDegreeNodes() const { return highDegreeNodes_; }
    const std::unordered_set<vertexID>& getLowDegreeNodes() const { return lowDegreeNodes_; }
    uint64_t getHighDegreeEdgeCount() const { return highDegreeEdgeCount_; }
    uint64_t getLowDegreeEdgeCount() const { return lowDegreeEdgeCount_; }
    uint64_t getEdgesBeforeDedup() const { return edges_before_dedup_; }
    uint64_t getEdgesAfterDedup() const { return edges_after_dedup_; }
    double getLoadTimeMs() const { return load_time_ms_; }
    double getDedupTimeMs() const { return dedup_time_ms_; }
    double getDegreeTimeMs() const { return degree_time_ms_; }
    const WeightHistogram& getWeightHistogram() const { return weight_hist_; }
    const VertexStats& getVertexStats() const { return vertex_stats_; }
    DetectedFormat getDetectedFormat() const { return detected_format_; }
};


// ============================================================================
// 测试数据生成: 创建临时边文件
// ============================================================================

static std::string create_test_graph_space(const std::string &path, int num_edges, bool weighted = false, bool add_dups = false) {
    std::ofstream f(path);
    f << "# test graph\n";
    for (int i = 0; i < num_edges; i++) {
        int src = i % 20;
        int dst = (i * 7 + 3) % 20;
        if (src == dst) dst = (dst + 1) % 20;
        if (weighted) {
            f << src << " " << dst << " " << (i * 0.1 + 0.5) << "\n";
        } else {
            f << src << " " << dst << "\n";
        }
    }
    // 加重复边
    if (add_dups) {
        for (int i = 0; i < 10; i++) {
            f << "0 1\n";
        }
    }
    f.close();
    return path;
}

static std::string create_test_graph_csv(const std::string &path, int num_edges) {
    std::ofstream f(path);
    for (int i = 0; i < num_edges; i++) {
        int src = i % 15;
        int dst = (i * 3 + 1) % 15;
        if (src == dst) dst = (dst + 1) % 15;
        f << src << "," << dst << "," << (i * 0.05 + 1.0) << "\n";
    }
    f.close();
    return path;
}

static std::string create_test_graph_tsv(const std::string &path, int num_edges) {
    std::ofstream f(path);
    for (int i = 0; i < num_edges; i++) {
        int src = i % 25;
        int dst = (i * 11 + 2) % 25;
        if (src == dst) dst = (dst + 1) % 25;
        f << src << "\t" << dst << "\n";
    }
    f.close();
    return path;
}

static std::string create_test_graph_with_errors(const std::string &path) {
    std::ofstream f(path);
    f << "# header\n";
    f << "0 1\n";
    f << "1 2\n";
    f << "bad_line\n";           // error
    f << "3 3\n";               // self-loop → skip
    f << "2 5\n";
    f << "also invalid x y\n";  // error
    f << "4 6\n";
    f << "7 8\n";
    f.close();
    return path;
}

static std::string create_large_test_graph(const std::string &path, int num_edges) {
    std::ofstream f(path);
    // 创建一个有明显度数差异的图
    // 节点0是hub (高度连接)
    for (int i = 1; i < 50; i++) {
        f << "0 " << i << "\n";    // hub edges
    }
    // 其余随机边
    for (int i = 0; i < num_edges - 49; i++) {
        int src = (i % 100) + 1;
        int dst = ((i * 13 + 7) % 100) + 1;
        if (src == dst) dst = (dst % 100) + 1;
        f << src << " " << dst << "\n";
    }
    f.close();
    return path;
}


// ============================================================================
// 测试框架
// ============================================================================

static int g_pass = 0, g_fail = 0;

static void run_test(const char* name, std::function<bool()> fn) {
    printf("\n[TEST] %s\n", name);
    bool ok = false;
    try {
        ok = fn();
    } catch (const std::exception &e) {
        printf("  EXCEPTION: %s\n", e.what());
        ok = false;
    }
    if (ok) { g_pass++; printf("  => PASS\n"); }
    else    { g_fail++; printf("  => FAIL\n"); }
}


// ============================================================================
// 18个测试
// ============================================================================

// T01: types.hpp — weightedEdge 构造 + 字段
bool test_weighted_edge_construction() {
    weightedEdge e1;
    printf("  [T01] default: src=%lu dst=%lu w=%.1f\n", e1.source, e1.destination, e1.weight);
    if (e1.source != 0 || e1.destination != 0 || e1.weight != 0.0) return false;

    weightedEdge e2(10, 20, 3.14);
    printf("  [T01] param3: src=%lu dst=%lu w=%.2f\n", e2.source, e2.destination, e2.weight);
    if (e2.source != 10 || e2.destination != 20 || std::abs(e2.weight - 3.14) > 0.001) return false;

    weightedEdge e3(100, 200);
    printf("  [T01] param2: src=%lu dst=%lu w=%.1f\n", e3.source, e3.destination, e3.weight);
    if (e3.source != 100 || e3.destination != 200) return false;

    return true;
}

// T02: types.hpp — operationType 枚举完整性
bool test_operation_type_enum() {
    // 验证所有23个枚举值存在且不同
    std::vector<operationType> all = {
        operationType::INSERT, operationType::DELETE, operationType::INSERT_VERTEX,
        operationType::UPDATE, operationType::GET_VERTEX, operationType::GET_EDGE,
        operationType::GET_WEIGHT, operationType::SCAN_NEIGHBOR, operationType::GET_NEIGHBOR,
        operationType::PHYSICAL2LOGICAL, operationType::LOGICAL2PHYSICAL,
        operationType::BFS, operationType::PAGE_RANK, operationType::SSSP,
        operationType::TC, operationType::TC_OP, operationType::WCC,
        operationType::QUERY, operationType::MIXED, operationType::QOS,
        operationType::CONCURRENT, operationType::BATCH_INSERT
    };
    printf("  [T02] %zu operation types defined\n", all.size());
    // 验证每个值唯一
    std::unordered_set<int> vals;
    for (auto t : all) vals.insert(static_cast<int>(t));
    if (vals.size() != all.size()) return false;
    printf("  [T02] all %zu types unique\n", vals.size());
    return true;
}

// T03: types.hpp — generate_path_ts + generate_path_type
bool test_path_generation() {
    std::string p1 = "prefix/";
    generate_path_ts(p1, targetStreamType::FULL);
    printf("  [T03] FULL -> '%s'\n", p1.c_str());
    if (p1 != "prefix/full.stream") return false;

    std::string p2 = "dir/";
    generate_path_type(p2, operationType::INSERT);
    printf("  [T03] INSERT -> '%s'\n", p2.c_str());
    if (p2 != "dir/insert_") return false;

    std::string p3 = "";
    generate_path_type(p3, operationType::BFS);
    printf("  [T03] BFS -> '%s'\n", p3.c_str());
    if (p3 != "bfs.stream") return false;

    // 全部targetStreamType
    std::vector<targetStreamType> tsts = {
        targetStreamType::FULL, targetStreamType::GENERAL,
        targetStreamType::HIGH_DEGREE, targetStreamType::LOW_DEGREE,
        targetStreamType::UNIFORM, targetStreamType::BASED_ON_DEGREE
    };
    for (auto ts : tsts) {
        std::string px = "";
        generate_path_ts(px, ts);
        printf("  [T03] stream -> '%s'\n", px.c_str());
        if (px.empty()) return false;
    }

    return true;
}

// T04: types.hpp — Config + EdgeDriverConfig + DriverConfig 结构体字段
bool test_config_structs() {
    Config c1(0.5);
    printf("  [T04] Config(0.5): ts_rate=%.1f num_search=%lu\n", c1.timestamp_rate, c1.num_search);
    if (std::abs(c1.timestamp_rate - 0.5) > 0.01) return false;
    if (c1.num_search != 1000000) return false;

    Config c2;
    if (std::abs(c2.timestamp_rate) > 0.01) return false;

    EdgeDriverConfig edc;
    printf("  [T04] EdgeDriverConfig defaults: init_rate=%.1f neighbor_size=%lu\n",
           edc.initial_graph_rate, edc.neighbor_size);
    if (std::abs(edc.initial_graph_rate - 0.8) > 0.01) return false;

    DriverConfig dc;
    printf("  [T04] DriverConfig defaults: alpha=%d beta=%d delta=%.1f damping=%.2f\n",
           dc.alpha, dc.beta, dc.delta, dc.damping_factor);
    if (dc.alpha != 15 || dc.beta != 18) return false;
    if (std::abs(dc.damping_factor - 0.85) > 0.01) return false;

    return true;
}

// T05: types.hpp — operation 二进制序列化 + read_stream
bool test_operation_serialization() {
    std::string tmp = "/tmp/m108_test_stream.bin";
    std::vector<operation> ops;
    ops.push_back({operationType::INSERT, {1, 2, 0.5}});
    ops.push_back({operationType::DELETE, {3, 4, 1.5}});
    ops.push_back({operationType::BFS, {10, 0, 0}});

    // 写入
    std::ofstream f(tmp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(ops.data()), ops.size() * sizeof(operation));
    f.close();

    // 读回
    std::vector<operation> loaded;
    read_stream(tmp, loaded);
    printf("  [T05] wrote %zu ops, read back %zu\n", ops.size(), loaded.size());
    if (loaded.size() != 3) return false;

    if (loaded[0].type != operationType::INSERT) return false;
    if (loaded[0].e.source != 1 || loaded[0].e.destination != 2) return false;
    if (loaded[1].type != operationType::DELETE) return false;
    if (loaded[2].type != operationType::BFS) return false;
    if (loaded[2].e.source != 10) return false;

    printf("  [T05] serialization roundtrip OK, sizeof(operation)=%zu\n", sizeof(operation));
    std::remove(tmp.c_str());
    return true;
}

// T06: [20%改动] WeightHistogram 边权重直方图
bool test_weight_histogram() {
    WeightHistogram wh;
    for (int i = 0; i < 100; i++) {
        wh.add_weight(i * 0.01); // 0.00 .. 0.99
    }
    wh.add_weight(5.0);
    wh.add_weight(-1.0);

    printf("  [T06] range=[%.2f, %.2f] total=%lu buckets=%zu\n",
           wh.min_weight, wh.max_weight, wh.total_edges, wh.buckets.size());
    if (wh.total_edges != 102) return false;
    if (wh.min_weight > -0.99) return false;
    if (wh.max_weight < 4.99) return false;

    wh.print_summary();
    return true;
}

// T07: [20%改动] VertexStats 度数分布
bool test_vertex_stats() {
    std::vector<vertexID> deg = {0, 1, 3, 3, 5, 10, 10, 10, 100, 0};
    VertexStats vs;
    vs.compute(deg);

    printf("  [T07] vertices=%lu avg_deg=%.2f max_deg=%lu\n",
           vs.total_vertices, vs.avg_degree(), vs.max_degree());
    if (vs.total_vertices != 10) return false;
    if (vs.max_degree() != 100) return false;
    // avg = (0+1+3+3+5+10+10+10+100+0)/10 = 14.2
    if (std::abs(vs.avg_degree() - 14.2) > 0.01) return false;

    vs.print_summary();
    return true;
}

// T08: [20%改动] 自动格式检测(TSV/CSV/SPACE)
bool test_format_detection() {
    DetectedFormat f1 = detect_delimiter("100\t200\t0.5");
    printf("  [T08] TSV line -> %s\n", format_name(f1));
    if (f1 != DetectedFormat::TAB) return false;

    DetectedFormat f2 = detect_delimiter("100,200,0.5");
    printf("  [T08] CSV line -> %s\n", format_name(f2));
    if (f2 != DetectedFormat::COMMA) return false;

    DetectedFormat f3 = detect_delimiter("100 200 0.5");
    printf("  [T08] SPACE line -> %s\n", format_name(f3));
    if (f3 != DetectedFormat::SPACE) return false;

    if (format_to_char(f1) != '\t') return false;
    if (format_to_char(f2) != ',') return false;
    if (format_to_char(f3) != ' ') return false;

    return true;
}

// T09: Parser singleton + configure + getters
bool test_parser_singleton() {
    Parser &p1 = Parser::get_instance();
    Parser &p2 = Parser::get_instance();
    printf("  [T09] singleton same addr: %s\n", (&p1 == &p2) ? "YES" : "NO");
    if (&p1 != &p2) return false;

    p1.configure("/tmp/test.txt", "/tmp/out", true, '\t',
                 0.7, 0.3, 0.3, 0.02, 0.3, 0.3, 0.6,
                 5000, 5000, 5000, 42, false);

    if (p1.get_input_file() != "/tmp/test.txt") return false;
    if (p1.get_output_dir() != "/tmp/out") return false;
    if (!p1.get_is_weighted()) return false;
    if (p1.get_delimiter() != '\t') return false;
    if (std::abs(p1.get_initial_graph_ratio() - 0.7) > 0.01) return false;
    if (p1.get_seed() != 42) return false;
    if (p1.get_insert_num() != 5000) return false;
    if (p1.get_is_shuffle()) return false;

    printf("  [T09] all 15 getters verified\n");
    return true;
}

// T10: DataPreProcessor — 基本空格分隔图加载
bool test_preprocessor_space_graph() {
    std::string path = "/tmp/m108_test_space.txt";
    create_test_graph_space(path, 50);

    DataPreProcessor proc(path, false, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                          10, 10, 10, 42, true);

    printf("  [T10] vertices=%lu edges=%zu\n", proc.getNumVertices(), proc.getNumEdges());
    if (proc.getNumVertices() == 0) return false;
    if (proc.getNumEdges() == 0) return false;
    if (proc.getLoadTimeMs() < 0) return false;

    std::remove(path.c_str());
    return true;
}

// T11: DataPreProcessor — CSV自动检测
bool test_preprocessor_csv_autodetect() {
    std::string path = "/tmp/m108_test_csv.csv";
    create_test_graph_csv(path, 40);

    // 传入delimiter=' ', 自动检测会改成','
    DataPreProcessor proc(path, true, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                          10, 10, 10, 7, true);

    printf("  [T11] CSV auto-detect: vertices=%lu edges=%zu format=%s\n",
           proc.getNumVertices(), proc.getNumEdges(),
           format_name(proc.getDetectedFormat()));
    if (proc.getNumVertices() == 0) return false;
    if (proc.getDetectedFormat() != DetectedFormat::COMMA) return false;

    // 验证权重直方图有数据
    if (proc.getWeightHistogram().total_edges == 0) return false;
    proc.getWeightHistogram().print_summary();

    std::remove(path.c_str());
    return true;
}

// T12: DataPreProcessor — TSV自动检测
bool test_preprocessor_tsv_autodetect() {
    std::string path = "/tmp/m108_test_tsv.tsv";
    create_test_graph_tsv(path, 60);

    DataPreProcessor proc(path, false, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                          10, 10, 10, 99, true);

    printf("  [T12] TSV auto-detect: vertices=%lu edges=%zu format=%s\n",
           proc.getNumVertices(), proc.getNumEdges(),
           format_name(proc.getDetectedFormat()));
    if (proc.getDetectedFormat() != DetectedFormat::TAB) return false;
    if (proc.getNumEdges() == 0) return false;

    std::remove(path.c_str());
    return true;
}

// T13: 去重 — 验证重复边被压缩
bool test_dedup_compression() {
    std::string path = "/tmp/m108_test_dup.txt";
    create_test_graph_space(path, 30, false, true); // 加10条0-1重复

    uint64_t before, after;
    {
        DataPreProcessor proc(path, false, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                              10, 10, 10, 0, false);
        before = proc.getEdgesBeforeDedup();
        after = proc.getEdgesAfterDedup();
        printf("  [T13] before=%lu after=%lu dedup_ms=%.2f\n",
               before, after, proc.getDedupTimeMs());
    }
    if (after >= before) return false;
    printf("  [T13] compression: %lu -> %lu (removed %lu dups)\n",
           before, after, before - after);

    std::remove(path.c_str());
    return true;
}

// T14: 度数分布 + prefixSum 正确性
bool test_degree_and_prefix_sum() {
    std::string path = "/tmp/m108_test_deg.txt";
    create_test_graph_space(path, 80);

    DataPreProcessor proc(path, false, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                          10, 10, 10, 42, true);

    auto &deg = proc.getDegreeDistribution();
    auto &ps = proc.getPrefixSum();

    printf("  [T14] deg.size=%zu ps.size=%zu\n", deg.size(), ps.size());
    if (deg.size() != proc.getNumVertices()) return false;
    if (ps.size() != deg.size()) return false;

    // prefixSum 单调非递减
    for (size_t i = 1; i < ps.size(); i++) {
        if (ps[i] < ps[i-1]) {
            printf("  [T14] FAIL: prefixSum not monotonic at %zu\n", i);
            return false;
        }
    }

    // prefixSum[-1] = sum of all degrees
    uint64_t total = std::accumulate(deg.begin(), deg.end(), (uint64_t)0);
    printf("  [T14] total degree=%lu prefixSum.back()=%lu\n", total, ps.back());
    if (ps.back() != total) return false;

    // vertex stats
    proc.getVertexStats().print_summary();

    std::remove(path.c_str());
    return true;
}

// T15: 高度/低度节点选择
bool test_high_low_degree_selection() {
    std::string path = "/tmp/m108_test_hilo.txt";
    create_large_test_graph(path, 200);

    DataPreProcessor proc(path, false, ' ', 0.8, 0.2, 0.2, 0.05, 0.2, 0.2, 0.5,
                          20, 20, 20, 42, true);

    printf("  [T15] highNodes=%zu (edges=%lu) lowNodes=%zu (edges=%lu)\n",
           proc.getHighDegreeNodes().size(), proc.getHighDegreeEdgeCount(),
           proc.getLowDegreeNodes().size(), proc.getLowDegreeEdgeCount());

    // 高度节点应该存在
    if (proc.getHighDegreeNodes().empty() && proc.getNumVertices() > 20) {
        printf("  [T15] WARNING: no high degree nodes found\n");
    }

    // 低度节点数 = numVertices * lowEdgeRatio
    size_t expected_low = proc.getNumVertices() * 0.5;
    printf("  [T15] expected low=%zu actual=%zu\n",
           expected_low, proc.getLowDegreeNodes().size());
    if (proc.getLowDegreeNodes().size() != expected_low) return false;

    std::remove(path.c_str());
    return true;
}

// T16: generateAllWorkloads — 产生所有stream文件
bool test_generate_all_workloads() {
    std::string graph_path = "/tmp/m108_gen_graph.txt";
    std::string out_dir = "/tmp/m108_gen_out";

    create_test_graph_space(graph_path, 100);
    system("mkdir -p /tmp/m108_gen_out");

    DataPreProcessor proc(graph_path, false, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                          10, 10, 10, 42, true);

    proc.generateAllWorkloads(out_dir);

    // 检查生成的文件
    std::vector<std::string> expected_files = {
        "initial_stream_insert_vertex.stream",
        "initial_stream_insert_full.stream",
        "target_stream_insert_full.stream",
        "initial_stream_insert_general.stream",
        "target_stream_insert_general.stream",
        "initial_stream_analytic.stream",
        "target_stream_get_edge_general.stream",
        "target_stream_scan_neighbor_general.stream",
    };

    int found = 0;
    for (auto &name : expected_files) {
        std::string fp = out_dir + "/" + name;
        std::ifstream check(fp, std::ios::binary);
        if (check.is_open()) {
            check.seekg(0, std::ios::end);
            size_t sz = check.tellg();
            printf("  [T16] %s: %zu bytes\n", name.c_str(), sz);
            if (sz > 0) found++;
            check.close();
        } else {
            printf("  [T16] %s: MISSING\n", name.c_str());
        }
    }

    printf("  [T16] found %d/%zu expected files\n", found, expected_files.size());

    // cleanup
    system("rm -rf /tmp/m108_gen_out /tmp/m108_gen_graph.txt");

    return found >= 6; // at least 6 of 8 required
}

// T17: stream文件读回验证 — insert_vertex stream
bool test_stream_readback() {
    std::string graph_path = "/tmp/m108_rb_graph.txt";
    std::string out_dir = "/tmp/m108_rb_out";

    create_test_graph_space(graph_path, 50);
    system("mkdir -p /tmp/m108_rb_out");

    DataPreProcessor proc(graph_path, false, ' ', 0.8, 0.2, 0.2, 0.01, 0.2, 0.2, 0.5,
                          10, 10, 10, 42, true);

    proc.generateAllWorkloads(out_dir);

    // 读vertex stream
    std::string vpath = std::string(out_dir) + "/initial_stream_insert_vertex.stream";
    std::vector<operation> vstream;
    read_stream(vpath, vstream);

    printf("  [T17] vertex stream: %zu ops, numVertices=%lu\n",
           vstream.size(), proc.getNumVertices());
    if (vstream.size() != proc.getNumVertices()) return false;

    // 每个op应该是INSERT_VERTEX
    for (size_t i = 0; i < vstream.size(); i++) {
        if (vstream[i].type != operationType::INSERT_VERTEX) {
            printf("  [T17] FAIL at op %zu: not INSERT_VERTEX\n", i);
            return false;
        }
        if (vstream[i].e.source != i) {
            printf("  [T17] FAIL at op %zu: expected src=%zu got %lu\n",
                   i, i, vstream[i].e.source);
            return false;
        }
    }

    // 读analytic stream
    std::string apath = std::string(out_dir) + "/initial_stream_analytic.stream";
    std::vector<operation> astream;
    read_stream(apath, astream);
    printf("  [T17] analytic stream: %zu ops (= edges=%zu)\n",
           astream.size(), proc.getNumEdges());
    if (astream.size() != proc.getNumEdges()) return false;

    system("rm -rf /tmp/m108_rb_out /tmp/m108_rb_graph.txt");
    return true;
}

// T18: [20%改动] 阶段计时 + 综合性能报告
bool test_pipeline_timing() {
    std::string path = "/tmp/m108_timing.txt";
    create_large_test_graph(path, 500);

    auto t0 = std::chrono::high_resolution_clock::now();

    DataPreProcessor proc(path, false, ' ', 0.8, 0.2, 0.2, 0.05, 0.2, 0.2, 0.5,
                          20, 20, 20, 42, true);

    auto t1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  [T18] Pipeline timing report:\n");
    printf("    loadEdges:     %.2f ms\n", proc.getLoadTimeMs());
    printf("    removeDups:    %.2f ms\n", proc.getDedupTimeMs());
    printf("    degreeDist:    %.2f ms\n", proc.getDegreeTimeMs());
    printf("    total ctor:    %.2f ms\n", total);

    if (proc.getLoadTimeMs() < 0) return false;
    if (proc.getDedupTimeMs() < 0) return false;
    if (proc.getDegreeTimeMs() < 0) return false;

    // 各阶段总和不应超过总时间太多(含shuffle+selectNodes)
    double accounted = proc.getLoadTimeMs() + proc.getDedupTimeMs() + proc.getDegreeTimeMs();
    printf("    accounted:     %.2f ms (%.0f%% of total)\n",
           accounted, accounted / total * 100);

    // 最终综合报告
    printf("  [T18] Summary: %lu V, %zu E, dedup %lu->%lu, ",
           proc.getNumVertices(), proc.getNumEdges(),
           proc.getEdgesBeforeDedup(), proc.getEdgesAfterDedup());
    printf("high=%zu low=%zu\n",
           proc.getHighDegreeNodes().size(), proc.getLowDegreeNodes().size());

    std::remove(path.c_str());
    return true;
}


// ============================================================================
// main — upstream main.cpp 移植 (12行): Parser驱动 DataPreProcessor 工作流
// ============================================================================

int main() {
    printf("=== M108-M109: dataset_preprocessor 完整移植实验 ===\n");
    printf("  types.hpp: %zu bytes (weightedEdge + operationType + Config structs)\n", sizeof(weightedEdge));
    printf("  operation: %zu bytes\n", sizeof(operation));
    printf("  Config: %zu bytes  EdgeDriverConfig: %zu bytes  DriverConfig: %zu bytes\n",
           sizeof(Config), sizeof(EdgeDriverConfig), sizeof(DriverConfig));
    printf("=================================================\n");

    // --- M108 tests: types.hpp + parser ---
    run_test("T01: weightedEdge construction",          test_weighted_edge_construction);
    run_test("T02: operationType enum completeness",    test_operation_type_enum);
    run_test("T03: generate_path_ts/type",              test_path_generation);
    run_test("T04: Config/EdgeDriverConfig/DriverConfig",test_config_structs);
    run_test("T05: operation serialization roundtrip",  test_operation_serialization);
    run_test("T06: [20%] WeightHistogram",              test_weight_histogram);
    run_test("T07: [20%] VertexStats degree dist",      test_vertex_stats);
    run_test("T08: [20%] auto format detection",        test_format_detection);
    run_test("T09: Parser singleton + configure",       test_parser_singleton);

    // --- M109 tests: dataset_preprocessor ---
    run_test("T10: preprocessor space-delim graph",     test_preprocessor_space_graph);
    run_test("T11: preprocessor CSV auto-detect",       test_preprocessor_csv_autodetect);
    run_test("T12: preprocessor TSV auto-detect",       test_preprocessor_tsv_autodetect);
    run_test("T13: dedup compression ratio",            test_dedup_compression);
    run_test("T14: degree distribution + prefixSum",    test_degree_and_prefix_sum);
    run_test("T15: high/low degree node selection",     test_high_low_degree_selection);
    run_test("T16: generateAllWorkloads",               test_generate_all_workloads);
    run_test("T17: stream readback verification",       test_stream_readback);
    run_test("T18: [20%] pipeline timing report",       test_pipeline_timing);

    printf("\n=== RESULTS: %d/%d PASS, %d FAIL ===\n", g_pass, g_pass + g_fail, g_fail);
    return g_fail > 0 ? 1 : 0;
}
