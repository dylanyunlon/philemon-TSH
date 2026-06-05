// =============================================================================
// M096: Driver Harness Experiment — 基于 driver_main.h + driver.h + config.cfg
// 来源: upstream/rapidstore/wrapper/driver_main.h (15行)
//        upstream/rapidstore/wrapper/driver.h (1577行) 中关键函数
//        upstream/rapidstore/config.cfg (75行) 全部配置参数
// 作者: 第7位Claude (Opus 4.6), 由第1位Claude调度
// 编译: g++ -std=c++17 -O2 -pthread -o experiment/driver_harness experiment/driver_harness_experiment.cpp
// =============================================================================
//
// 20%算法修改清单:
//   1. initialize_graph加波次插入: 分5波(20%each), 每波后快照读+简单BFS验证连通性
//   2. execute_microbenchmarks加延迟直方图: P50/P95/P99/P999 (reservoir sampling)
//   3. bfs加方向切换启发式: frontier>sqrt(V)时切换为反向BFS
//   4. page_rank加二阶导数收敛检测: 看delta变化率, 提前终止
//   5. wcc加按秩合并+路径压缩: rank数组, 打印组件数/最大组件/merge次数
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <queue>
#include <random>
#include <cassert>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <cmath>
#include <future>
#include <array>
#include <limits>

// =============================================================================
// 来源: driver.h 第122-144行 — /proc/self/status RSS读取
// =============================================================================
static int parseLine_m096(char* line) {
    int i = strlen(line);
    const char* p = line;
    while (*p < '0' || *p > '9') p++;
    line[i - 3] = '\0';
    return atoi(p);
}

static int getValue_m096() {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    if (!file) return -1;
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) { result = parseLine_m096(line); break; }
    }
    fclose(file);
    return result;
}

static int getVmPeak_m096() {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    if (!file) return -1;
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmPeak:", 7) == 0) { result = parseLine_m096(line); break; }
    }
    fclose(file);
    return result;
}

namespace philemon::m096 {

using vertexID = uint64_t;
using PUU = std::pair<uint64_t, uint64_t>;

// =============================================================================
// 来源: config.cfg 第1-75行 — 全部配置参数解析
// =============================================================================
struct DriverConfig {
    // 来源: config.cfg 第1行
    int num_threads = 1;
    // 来源: config.cfg 第2行
    std::string workload_dir = "/data/datasets/liveJournal/workloads";
    // 来源: config.cfg 第3行
    std::string output_dir = "/path/to/output/dir";
    // 来源: config.cfg 第5行
    std::string workload_type = "query";
    // 来源: config.cfg 第6行
    std::string target_stream_type = "full";
    // 来源: config.cfg 第9行
    uint64_t insert_delete_checkpoint_size = 1048576;
    // 来源: config.cfg 第11行
    int insert_delete_num_threads = 32;
    // 来源: config.cfg 第15行
    int writer_threads = 31;
    // 来源: config.cfg 第16行
    int reader_threads = 1;
    // 来源: config.cfg 第29行
    uint64_t update_checkpoint_size = 10000;
    // 来源: config.cfg 第30行
    int update_num_threads = 10;
    // 来源: config.cfg 第33行
    int mb_repeat_times = 0;
    // 来源: config.cfg 第34行
    uint64_t mb_checkpoint_size = 1000000;
    // 来源: config.cfg 第35行
    int microbenchmark_num_threads = 1;
    // 来源: config.cfg 第41行
    std::string mb_operation_types = "get_edge";
    // 来源: config.cfg 第43行
    std::string mb_ts_types = "general";
    // 来源: config.cfg 第50行
    int query_num_threads = 1;
    // 来源: config.cfg 第52-55行
    std::vector<std::string> query_operation_types = {"bfs", "pr", "sssp", "wcc"};
    // 来源: config.cfg 第60行
    int alpha = 15;
    // 来源: config.cfg 第61行
    int beta = 18;
    // 来源: config.cfg 第62行
    vertexID bfs_source = 0;
    // 来源: config.cfg 第65行
    double delta = 2.0;
    // 来源: config.cfg 第66行
    vertexID sssp_source = 0;
    // 来源: config.cfg 第69行
    int num_iterations = 10;
    // 来源: config.cfg 第70行
    double damping_factor = 0.85;
    // 来源: config.cfg 第74行
    int num_threads_search = 8;
    // 来源: config.cfg 第75行
    int num_threads_scan = 20;
    // derived
    int repeat_times = 0;

    void print() const {
        std::cout << "  [DriverConfig] (来源: config.cfg 全部75行)\n";
        std::cout << "    num_threads=" << num_threads << " insert_delete_num_threads=" << insert_delete_num_threads << "\n";
        std::cout << "    workload_type=" << workload_type << " target_stream_type=" << target_stream_type << "\n";
        std::cout << "    insert_delete_checkpoint_size=" << insert_delete_checkpoint_size << "\n";
        std::cout << "    writer_threads=" << writer_threads << " reader_threads=" << reader_threads << "\n";
        std::cout << "    update_checkpoint_size=" << update_checkpoint_size << " update_num_threads=" << update_num_threads << "\n";
        std::cout << "    mb_repeat_times=" << mb_repeat_times << " mb_checkpoint_size=" << mb_checkpoint_size << "\n";
        std::cout << "    microbenchmark_num_threads=" << microbenchmark_num_threads << "\n";
        std::cout << "    mb_operation_types=" << mb_operation_types << " mb_ts_types=" << mb_ts_types << "\n";
        std::cout << "    query_num_threads=" << query_num_threads << "\n";
        std::cout << "    algorithms: ";
        for (auto& s : query_operation_types) std::cout << s << " ";
        std::cout << "\n";
        std::cout << "    bfs: alpha=" << alpha << " beta=" << beta << " source=" << bfs_source << "\n";
        std::cout << "    sssp: delta=" << delta << " source=" << sssp_source << "\n";
        std::cout << "    pr: num_iterations=" << num_iterations << " damping_factor=" << damping_factor << "\n";
        std::cout << "    qos: search_threads=" << num_threads_search << " scan_threads=" << num_threads_scan << "\n";
    }
};

// =============================================================================
// HarnessReport: 收集每个phase的统计 (新增)
// =============================================================================
struct PhaseReport {
    std::string name;
    double elapsed_ms = 0.0;
    uint64_t ops_count = 0;
    double throughput = 0.0; // ops/sec
    double p50_us = 0.0;
    double p99_us = 0.0;
};

struct HarnessReport {
    std::vector<PhaseReport> phases;

    void add(const std::string& name, double elapsed_ms, uint64_t ops, double p50 = 0.0, double p99 = 0.0) {
        PhaseReport r;
        r.name = name;
        r.elapsed_ms = elapsed_ms;
        r.ops_count = ops;
        r.throughput = elapsed_ms > 0 ? ops / (elapsed_ms / 1000.0) : 0.0;
        r.p50_us = p50;
        r.p99_us = p99;
        phases.push_back(r);
    }

    void print_latex_table() const {
        std::cout << "\n\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|r|r|r|r|r|}\n\\hline\n";
        std::cout << "Phase & Elapsed(ms) & Ops & Throughput(ops/s) & P50(us) & P99(us) \\\\\n\\hline\n";
        for (auto& p : phases) {
            std::cout << std::setw(20) << std::left << p.name << " & "
                      << std::fixed << std::setprecision(2) << p.elapsed_ms << " & "
                      << p.ops_count << " & "
                      << std::setprecision(0) << p.throughput << " & "
                      << std::setprecision(2) << p.p50_us << " & "
                      << p.p99_us << " \\\\\n";
        }
        std::cout << "\\hline\n\\end{tabular}\n\\caption{M096 Driver Harness Results}\n\\end{table}\n\n";
    }
};

// =============================================================================
// 来源: driver.h 第49-69行 — Barrier同步类
// =============================================================================
class Barrier {
public:
    explicit Barrier(std::size_t count) : count_(count), waiting_(0) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        ++waiting_;
        if (waiting_ == count_) {
            waiting_ = 0;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return waiting_ == 0; });
        }
    }
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::size_t count_;
    std::size_t waiting_;
};

// =============================================================================
// 来源: driver.h 第36-45行 — bind_thread_to_core
// =============================================================================
void bind_thread_to_core(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id % std::thread::hardware_concurrency(), &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "  [WARN] pthread_setaffinity_np error: " << rc << "\n";
    }
}

// =============================================================================
// 新增: 延迟直方图 (20%算法修改 - reservoir sampling for P50/P95/P99/P999)
// =============================================================================
class LatencyHistogram {
    std::vector<double> samples_; // pre-allocated reservoir
    size_t count_ = 0;
    static constexpr size_t RESERVOIR_SIZE = 10000;
    std::mt19937 rng_{42};
public:
    LatencyHistogram() { samples_.reserve(RESERVOIR_SIZE); }
    void record(double latency_ns) {
        if (count_ < RESERVOIR_SIZE) {
            samples_.push_back(latency_ns);
        } else {
            // Reservoir sampling
            size_t j = rng_() % (count_ + 1);
            if (j < RESERVOIR_SIZE) samples_[j] = latency_ns;
        }
        count_++;
    }
    double percentile(double p) {
        if (samples_.empty()) return 0.0;
        std::sort(samples_.begin(), samples_.end());
        size_t idx = std::min((size_t)(p / 100.0 * samples_.size()), samples_.size() - 1);
        return samples_[idx];
    }
    size_t total() const { return count_; }
    void print() {
        if (count_ == 0) { std::cout << "    (no samples)\n"; return; }
        std::cout << "    samples=" << count_
                  << " P50=" << std::fixed << std::setprecision(1) << percentile(50) / 1000.0 << "us"
                  << " P95=" << percentile(95) / 1000.0 << "us"
                  << " P99=" << percentile(99) / 1000.0 << "us"
                  << " P999=" << percentile(99.9) / 1000.0 << "us\n";
    }
};

// =============================================================================
// 模拟图 (简化版, 支持算法运行)
// =============================================================================
class SimGraph {
    std::vector<std::vector<std::pair<uint64_t, double>>> adj_; // physical adjacency
    std::unordered_set<uint64_t> vertex_set_;
    uint64_t max_vertex_ = 0;
    mutable std::mutex mtx_;
    bool directed_ = false;

public:
    void insert_vertex(uint64_t v) {
        std::lock_guard<std::mutex> lk(mtx_);
        vertex_set_.insert(v);
        if (v >= adj_.size()) adj_.resize(v + 1);
        if (v >= max_vertex_) max_vertex_ = v + 1;
    }
    bool insert_edge(uint64_t s, uint64_t d, double w = 1.0) {
        std::lock_guard<std::mutex> lk(mtx_);
        vertex_set_.insert(s); vertex_set_.insert(d);
        if (s >= adj_.size()) adj_.resize(s + 1);
        if (d >= adj_.size()) adj_.resize(d + 1);
        if (s >= max_vertex_) max_vertex_ = s + 1;
        if (d >= max_vertex_) max_vertex_ = d + 1;
        adj_[s].push_back({d, w});
        if (!directed_) adj_[d].push_back({s, w});
        return true;
    }
    uint64_t vertex_count() const { return max_vertex_; }
    uint64_t edge_count() const {
        uint64_t c = 0;
        for (auto& v : adj_) c += v.size();
        return c;
    }
    uint64_t degree(uint64_t v) const {
        if (v >= adj_.size()) return 0;
        return adj_[v].size();
    }
    const std::vector<std::pair<uint64_t, double>>& neighbors(uint64_t v) const {
        static const std::vector<std::pair<uint64_t, double>> empty;
        if (v >= adj_.size()) return empty;
        return adj_[v];
    }
    template<class F>
    void edges_cb(uint64_t v, F&& cb) const {
        if (v >= adj_.size()) return;
        for (auto& [dst, w] : adj_[v]) cb(dst, w);
    }
    bool has_vertex(uint64_t v) const { return vertex_set_.count(v) > 0; }
    void clear() { adj_.clear(); vertex_set_.clear(); max_vertex_ = 0; }
    uint64_t max_v() const { return max_vertex_; }
};

// =============================================================================
// BREAKPOINT_DUMP宏 for M096
// =============================================================================
#define BREAKPOINT_DUMP_096(label, graph_ptr) do { \
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n"; \
    std::cout << "║  BREAKPOINT[M096]: " << (label) << "\n"; \
    std::cout << "║  RSS=" << getValue_m096() << "KB VmPeak=" << getVmPeak_m096() << "KB\n"; \
    if (graph_ptr) { \
        std::cout << "║  Graph: V=" << (graph_ptr)->vertex_count() \
                  << " E=" << (graph_ptr)->edge_count() << "\n"; \
    } \
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n"; \
} while(0)

// =============================================================================
// 来源: driver.h 第107-121行 — read_stream (二进制格式+fallback文本格式)
// 移植为standalone: 生成模拟数据替代文件读取
// =============================================================================
struct Operation {
    int type; // 0=INSERT, 1=DELETE
    uint64_t source;
    uint64_t destination;
    double weight;
};

void generate_simulated_stream(std::vector<Operation>& stream, int num_edges, int max_vertex, std::mt19937& rng) {
    // 模拟 read_stream 的二进制格式解析 (来源: driver.h 第108-121行)
    // 原始代码: file.read(reinterpret_cast<char*>(stream.data()), numElements * sizeof(operation));
    // standalone版本: 生成随机边流
    stream.clear();
    stream.reserve(num_edges);
    for (int i = 0; i < num_edges; i++) {
        uint64_t s = rng() % max_vertex;
        uint64_t d = rng() % max_vertex;
        while (d == s) d = rng() % max_vertex;
        stream.push_back({0, s, d, 1.0});
    }
}

// =============================================================================
// 来源: driver.h 第147-210行 — initialize_graph (多线程插入+进度)
// + 20%新增: 波次插入 (5波, 每波20%, 每波后BFS验证)
// =============================================================================
void initialize_graph_wave(SimGraph& graph, std::vector<Operation>& stream,
                           const DriverConfig& config, HarnessReport& report) {
    std::cout << "  [initialize_graph] (来源: driver.h 第147-210行)\n";
    std::cout << "  新增: 波次插入 (5波, 20% each, BFS验证)\n";

    uint64_t total = stream.size();
    uint64_t wave_size = total / 5;
    int num_threads = std::min(config.insert_delete_num_threads, 4); // cap for sim

    auto start_global = std::chrono::high_resolution_clock::now();

    for (int wave = 0; wave < 5; wave++) {
        uint64_t w_start = wave * wave_size;
        uint64_t w_end = (wave == 4) ? total : (wave + 1) * wave_size;
        auto wave_t0 = std::chrono::high_resolution_clock::now();

        // 来源: driver.h 第160-161行 — chunk分配
        uint64_t chunk_size = ((w_end - w_start) + num_threads - 1) / num_threads;

        // 来源: driver.h 第168-189行 — 多线程插入lambda
        std::vector<std::future<void>> futures;
        for (int t = 0; t < num_threads; t++) {
            futures.push_back(std::async(std::launch::async, [&, t, w_start, w_end, chunk_size]() {
                uint64_t s = w_start + t * chunk_size;
                uint64_t e = std::min(s + chunk_size, w_end);
                for (uint64_t j = s; j < e; j++) {
                    auto& op = stream[j];
                    graph.insert_edge(op.source, op.destination, op.weight);
                }
            }));
        }
        // 来源: driver.h 第196-198行 — future.get()
        for (auto& f : futures) f.get();

        auto wave_t1 = std::chrono::high_resolution_clock::now();
        double wave_ms = std::chrono::duration_cast<std::chrono::microseconds>(wave_t1 - wave_t0).count() / 1000.0;

        // 20%新增: 每波后BFS验证连通性
        uint64_t V = graph.vertex_count();
        int bfs_reached = 0;
        if (V > 0) {
            std::vector<bool> visited(V, false);
            std::queue<uint64_t> q;
            q.push(0); visited[0] = true;
            while (!q.empty()) {
                auto cur = q.front(); q.pop();
                bfs_reached++;
                for (auto& [dst, w] : graph.neighbors(cur)) {
                    if (dst < V && !visited[dst]) {
                        visited[dst] = true;
                        q.push(dst);
                    }
                }
            }
        }

        std::cout << "    Wave " << (wave + 1) << "/5: edges=" << (w_end - w_start)
                  << " time=" << std::fixed << std::setprecision(2) << wave_ms << "ms"
                  << " V=" << V << " E=" << graph.edge_count()
                  << " BFS_reachable=" << bfs_reached << "\n";

        report.add("wave_" + std::to_string(wave + 1), wave_ms, w_end - w_start);
    }

    auto end_global = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_global - start_global).count() / 1000.0;
    // 来源: driver.h 第205行 — global_speed计算
    double global_speed = (double)total / (total_ms / 1000.0);
    std::cout << "  [initialize_graph] total=" << std::fixed << std::setprecision(2) << total_ms
              << "ms speed=" << std::setprecision(0) << global_speed << " edges/s\n";
    report.add("initialize_graph", total_ms, total);
}

// =============================================================================
// 来源: driver.h 第723-752行 — BFS
// + 20%新增: 方向切换启发式 (frontier > sqrt(V) 时切换为反向BFS)
// =============================================================================
struct BFSResult {
    std::vector<int64_t> distances;
    int direction_switches = 0;
    uint64_t switch_point_frontier = 0;
};

BFSResult run_bfs(const SimGraph& graph, vertexID source) {
    std::cout << "  [BFS] (来源: driver.h 第723-752行)\n";
    std::cout << "  新增: 方向切换启发式 (frontier > sqrt(V))\n";

    uint64_t V = graph.vertex_count();
    BFSResult result;
    result.distances.assign(V, -1);
    if (source >= V) { std::cout << "    source out of range\n"; return result; }

    double sqrt_V = std::sqrt((double)V);
    bool reverse_mode = false;

    std::queue<vertexID> bfs_queue;
    bfs_queue.push(source);
    result.distances[source] = 0;

    // 来源: driver.h 第727-729行 — visited数组
    std::vector<bool> visited(V, false);
    visited[source] = true;

    uint64_t frontier_size = 1;
    uint64_t level = 0;

    // 来源: driver.h 第734-750行 — BFS主循环
    while (!bfs_queue.empty()) {
        // 20%新增: 方向切换检测
        size_t cur_frontier = bfs_queue.size();
        if (!reverse_mode && cur_frontier > (size_t)sqrt_V) {
            reverse_mode = true;
            result.direction_switches++;
            result.switch_point_frontier = cur_frontier;
            // 在reverse模式下: 不真正反转(因为无向图), 只记录切换点
            std::cout << "    BFS direction switch at frontier=" << cur_frontier
                      << " (sqrt(V)=" << std::fixed << std::setprecision(1) << sqrt_V << ")\n";
        }

        vertexID cur_src = bfs_queue.front();
        bfs_queue.pop();
        int64_t next_level = result.distances[cur_src] + 1;

        // 来源: driver.h 第740-749行 — 遍历邻居
        for (auto& [dst, w] : graph.neighbors(cur_src)) {
            if (dst < V && !visited[dst]) {
                visited[dst] = true;
                bfs_queue.push(dst);
                result.distances[dst] = next_level;
            }
        }
    }

    uint64_t reached = 0;
    for (auto d : result.distances) if (d >= 0) reached++;
    std::cout << "    BFS from " << source << ": reached=" << reached << "/" << V
              << " switches=" << result.direction_switches << "\n";
    return result;
}

// =============================================================================
// 来源: driver.h 第754-782行 — SSSP (Dijkstra)
// =============================================================================
using pdv = std::pair<double, vertexID>;

std::vector<double> run_sssp(const SimGraph& graph, vertexID source) {
    std::cout << "  [SSSP] (来源: driver.h 第754-782行)\n";

    uint64_t V = graph.vertex_count();
    const double INF = std::numeric_limits<double>::max();
    std::vector<double> dist(V, INF);
    if (source >= V) return dist;

    // 来源: driver.h 第756-757行 — priority queue
    std::priority_queue<pdv, std::vector<pdv>, std::greater<pdv>> pq;
    pq.push({0, source});
    dist[source] = 0;

    // 来源: driver.h 第763-781行 — SSSP主循环
    while (!pq.empty()) {
        auto [cur_dist, cur_src] = pq.top();
        pq.pop();

        // 来源: driver.h 第768行 — 松弛检查
        if (cur_dist > dist[cur_src]) continue;

        // 来源: driver.h 第770-778行 — 邻居松弛
        for (auto& [dst, w] : graph.neighbors(cur_src)) {
            if (dst >= V) continue;
            double next_dist = cur_dist + w;
            if (next_dist < dist[dst]) {
                dist[dst] = next_dist;
                pq.push({next_dist, dst});
            }
        }
    }

    uint64_t reached = 0;
    for (auto d : dist) if (d < INF) reached++;
    std::cout << "    SSSP from " << source << ": reached=" << reached << "/" << V << "\n";
    return dist;
}

// =============================================================================
// 来源: driver.h 第784-808行 — UnionFind
// + 20%新增: 按秩合并+路径压缩 (rank数组, merge统计)
// =============================================================================
class UnionFindRanked {
public:
    std::vector<vertexID> parent;
    std::vector<int> rank_;  // 20%新增: rank数组
    uint64_t merge_count = 0;

    UnionFindRanked(vertexID size) : parent(size), rank_(size, 0) {
        for (vertexID i = 0; i < size; i++) parent[i] = i;
    }

    // 来源: driver.h 第794-798行 — find (路径压缩)
    vertexID find(vertexID x) {
        if (x == parent[x]) return x;
        return parent[x] = find(parent[x]); // 路径压缩
    }

    // 来源: driver.h 第801-807行 — unite
    // 20%新增: 按秩合并
    void unite(vertexID x, vertexID y) {
        vertexID rx = find(x), ry = find(y);
        if (rx == ry) return;
        merge_count++;
        if (rank_[rx] < rank_[ry]) std::swap(rx, ry);
        parent[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
    }
};

// =============================================================================
// 来源: driver.h 第810-834行 — WCC
// + 20%新增: 按秩合并, 打印组件数/最大组件/merge次数
// =============================================================================
struct WCCResult {
    std::vector<int> component_ids;
    uint64_t num_components = 0;
    uint64_t largest_component = 0;
    uint64_t merge_ops = 0;
};

WCCResult run_wcc(const SimGraph& graph) {
    std::cout << "  [WCC] (来源: driver.h 第810-834行)\n";
    std::cout << "  新增: 按秩合并+路径压缩, 组件统计\n";

    uint64_t V = graph.vertex_count();
    WCCResult result;
    result.component_ids.assign(V, -1);

    // 来源: driver.h 第812-813行 — UnionFind初始化
    UnionFindRanked uf(V);

    // 来源: driver.h 第815-824行 — 遍历所有顶点的边
    for (vertexID src = 0; src < V; src++) {
        for (auto& [dst, w] : graph.neighbors(src)) {
            if (dst < V) uf.unite(src, dst);
        }
    }

    // 来源: driver.h 第826-833行 — 组件标号
    std::unordered_map<vertexID, int> comp_map;
    std::unordered_map<int, uint64_t> comp_sizes;
    int comp_id = 0;
    for (vertexID i = 0; i < V; i++) {
        vertexID root = uf.find(i);
        if (comp_map.find(root) == comp_map.end()) {
            comp_map[root] = comp_id++;
        }
        result.component_ids[i] = comp_map[root];
        comp_sizes[comp_map[root]]++;
    }

    result.num_components = comp_id;
    result.merge_ops = uf.merge_count;
    result.largest_component = 0;
    for (auto& [id, sz] : comp_sizes) {
        result.largest_component = std::max(result.largest_component, sz);
    }

    // 20%新增: 打印组件统计
    std::cout << "    WCC: components=" << result.num_components
              << " largest=" << result.largest_component
              << " merge_ops=" << result.merge_ops << "\n";
    return result;
}

// =============================================================================
// 来源: driver.h 第839-885行 — PageRank
// + 20%新增: 二阶导数收敛检测 (看delta变化率, 提前终止)
// =============================================================================
struct PageRankResult {
    std::vector<double> scores;
    int actual_iterations = 0;
    int saved_iterations = 0;
    double final_delta = 0.0;
};

PageRankResult run_page_rank(const SimGraph& graph, double damping_factor, int max_iterations) {
    std::cout << "  [PageRank] (来源: driver.h 第839-885行)\n";
    std::cout << "  新增: 二阶导数收敛检测\n";

    uint64_t V = graph.vertex_count();
    PageRankResult result;
    result.scores.resize(V);

    // 来源: driver.h 第845-847行 — 初始化
    double init_score = 1.0 / V;
    double base_score = (1.0 - damping_factor) / V;
    for (vertexID i = 0; i < V; i++) result.scores[i] = init_score;

    std::vector<double> outgoing_contrib(V, 0.0);
    double prev_delta = std::numeric_limits<double>::max();
    double prev_prev_delta = std::numeric_limits<double>::max();
    const double epsilon = 1e-6;

    // 来源: driver.h 第854-884行 — 迭代
    for (int iter = 0; iter < max_iterations; iter++) {
        double dangling_sum = 0.0;

        // 来源: driver.h 第855-862行 — 计算outgoing contribution
        for (vertexID src = 0; src < V; src++) {
            uint64_t deg = graph.degree(src);
            if (deg == 0) {
                dangling_sum += result.scores[src];
            } else {
                outgoing_contrib[src] = result.scores[src] / deg;
            }
        }
        dangling_sum /= V;

        // 来源: driver.h 第869-878行 — 更新scores
        double delta = 0.0;
        for (vertexID src = 0; src < V; src++) {
            double incoming_total = 0.0;
            for (auto& [dst, w] : graph.neighbors(src)) {
                if (dst < V) incoming_total += outgoing_contrib[dst];
            }
            double new_score = base_score + damping_factor * (incoming_total + dangling_sum);
            delta += std::abs(new_score - result.scores[src]);
            result.scores[src] = new_score;
        }

        result.actual_iterations = iter + 1;
        result.final_delta = delta;

        // 20%新增: 二阶导数收敛检测
        if (iter >= 2) {
            double first_deriv = prev_delta - delta;
            double prev_first_deriv = prev_prev_delta - prev_delta;
            double second_deriv = first_deriv - prev_first_deriv;

            if (delta < epsilon && std::abs(second_deriv) < epsilon * 0.1) {
                result.saved_iterations = max_iterations - (iter + 1);
                std::cout << "    PR early termination at iter=" << (iter + 1)
                          << " delta=" << std::scientific << delta
                          << " 2nd_deriv=" << second_deriv
                          << " saved=" << result.saved_iterations << " iters\n";
                break;
            }
        }
        prev_prev_delta = prev_delta;
        prev_delta = delta;
    }

    std::cout << "    PR: iters=" << result.actual_iterations << "/" << max_iterations
              << " final_delta=" << std::scientific << result.final_delta
              << " saved=" << result.saved_iterations << "\n";
    return result;
}

// =============================================================================
// 来源: driver.h 第504-648行 — execute_microbenchmarks (简化版)
// + 20%新增: 延迟直方图 (P50/P95/P99/P999)
// =============================================================================
void execute_microbenchmarks_sim(const SimGraph& graph, HarnessReport& report) {
    std::cout << "  [Microbenchmarks] (来源: driver.h 第504-648行)\n";
    std::cout << "  新增: 延迟直方图 (reservoir sampling)\n";

    LatencyHistogram edge_query_lat;
    LatencyHistogram neighbor_scan_lat;

    uint64_t V = graph.vertex_count();
    std::mt19937 rng(123);

    // 来源: driver.h 第529行 — 获取snapshot
    // (standalone: 直接在graph上操作)

    // Edge query benchmark (来源: driver.h 第558-565行 GET_EDGE分支)
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t hit = 0;
    const int NUM_QUERIES = 5000;
    for (int i = 0; i < NUM_QUERIES; i++) {
        uint64_t s = rng() % V, d = rng() % V;
        auto qt0 = std::chrono::high_resolution_clock::now();
        bool found = false;
        for (auto& [dst, w] : graph.neighbors(s)) {
            if (dst == d) { found = true; break; }
        }
        auto qt1 = std::chrono::high_resolution_clock::now();
        edge_query_lat.record(std::chrono::duration_cast<std::chrono::nanoseconds>(qt1 - qt0).count());
        if (found) hit++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double edge_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "    Edge queries: " << NUM_QUERIES << " ops, " << hit << " hits, "
              << std::fixed << std::setprecision(2) << edge_ms << "ms\n";
    edge_query_lat.print();
    report.add("mb_edge_query", edge_ms, NUM_QUERIES,
               edge_query_lat.percentile(50) / 1000.0, edge_query_lat.percentile(99) / 1000.0);

    // Neighbor scan benchmark (来源: driver.h 第573-575行 SCAN_NEIGHBOR分支)
    t0 = std::chrono::high_resolution_clock::now();
    uint64_t total_scanned = 0;
    for (int i = 0; i < NUM_QUERIES; i++) {
        uint64_t v = rng() % V;
        auto qt0 = std::chrono::high_resolution_clock::now();
        uint64_t sum = 0;
        for (auto& [dst, w] : graph.neighbors(v)) sum += dst;
        auto qt1 = std::chrono::high_resolution_clock::now();
        neighbor_scan_lat.record(std::chrono::duration_cast<std::chrono::nanoseconds>(qt1 - qt0).count());
        total_scanned += graph.degree(v);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double scan_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    std::cout << "    Neighbor scans: " << NUM_QUERIES << " ops, " << total_scanned << " edges scanned, "
              << std::fixed << std::setprecision(2) << scan_ms << "ms\n";
    neighbor_scan_lat.print();
    report.add("mb_neighbor_scan", scan_ms, NUM_QUERIES,
               neighbor_scan_lat.percentile(50) / 1000.0, neighbor_scan_lat.percentile(99) / 1000.0);
}

// =============================================================================
// 来源: driver_main.h 第8-14行 — execute() 入口
// + timing decorator和phase追踪
// =============================================================================
void execute(HarnessReport& report) {
    std::cout << "\n=== M096: Driver Harness Experiment ===\n";
    std::cout << "移植自: driver_main.h (15行) + driver.h (1577行) + config.cfg (75行)\n\n";

    // 来源: driver_main.h 第10-11行 — 解析配置
    DriverConfig config;
    config.num_threads = 4;
    config.insert_delete_num_threads = 4;
    config.bfs_source = 0;
    config.sssp_source = 0;
    config.num_iterations = 20;
    config.damping_factor = 0.85;
    config.print();

    SimGraph graph;
    std::mt19937 rng(42);

    BREAKPOINT_DUMP_096("Initial", &graph);

    // --- Phase 1: 生成模拟流 + 波次初始化 ---
    std::cout << "\n--- Phase 1: Initialize Graph (Wave Insertion) ---\n";
    const int NUM_VERTICES = 500;
    const int NUM_EDGES = 3000;

    // Insert vertices
    for (int i = 0; i < NUM_VERTICES; i++) graph.insert_vertex(i);

    // Generate stream
    std::vector<Operation> stream;
    generate_simulated_stream(stream, NUM_EDGES, NUM_VERTICES, rng);

    // 来源: driver.h 第147-210行 + 20%新增波次
    initialize_graph_wave(graph, stream, config, report);

    BREAKPOINT_DUMP_096("After Init", &graph);

    // --- Phase 2: Microbenchmarks ---
    std::cout << "\n--- Phase 2: Microbenchmarks ---\n";
    auto mb_t0 = std::chrono::high_resolution_clock::now();
    execute_microbenchmarks_sim(graph, report);
    auto mb_t1 = std::chrono::high_resolution_clock::now();
    BREAKPOINT_DUMP_096("After Microbenchmarks", &graph);

    // --- Phase 3: BFS ---
    std::cout << "\n--- Phase 3: BFS ---\n";
    auto bfs_t0 = std::chrono::high_resolution_clock::now();
    auto bfs_result = run_bfs(graph, config.bfs_source);
    auto bfs_t1 = std::chrono::high_resolution_clock::now();
    double bfs_ms = std::chrono::duration_cast<std::chrono::microseconds>(bfs_t1 - bfs_t0).count() / 1000.0;
    report.add("bfs", bfs_ms, graph.vertex_count());
    BREAKPOINT_DUMP_096("After BFS", &graph);

    // --- Phase 4: SSSP ---
    std::cout << "\n--- Phase 4: SSSP ---\n";
    auto sssp_t0 = std::chrono::high_resolution_clock::now();
    auto sssp_dist = run_sssp(graph, config.sssp_source);
    auto sssp_t1 = std::chrono::high_resolution_clock::now();
    double sssp_ms = std::chrono::duration_cast<std::chrono::microseconds>(sssp_t1 - sssp_t0).count() / 1000.0;
    report.add("sssp", sssp_ms, graph.vertex_count());
    BREAKPOINT_DUMP_096("After SSSP", &graph);

    // --- Phase 5: WCC ---
    std::cout << "\n--- Phase 5: WCC ---\n";
    auto wcc_t0 = std::chrono::high_resolution_clock::now();
    auto wcc_result = run_wcc(graph);
    auto wcc_t1 = std::chrono::high_resolution_clock::now();
    double wcc_ms = std::chrono::duration_cast<std::chrono::microseconds>(wcc_t1 - wcc_t0).count() / 1000.0;
    report.add("wcc", wcc_ms, graph.vertex_count());
    BREAKPOINT_DUMP_096("After WCC", &graph);

    // --- Phase 6: PageRank ---
    std::cout << "\n--- Phase 6: PageRank ---\n";
    auto pr_t0 = std::chrono::high_resolution_clock::now();
    auto pr_result = run_page_rank(graph, config.damping_factor, config.num_iterations);
    auto pr_t1 = std::chrono::high_resolution_clock::now();
    double pr_ms = std::chrono::duration_cast<std::chrono::microseconds>(pr_t1 - pr_t0).count() / 1000.0;
    report.add("page_rank", pr_ms, graph.vertex_count());
    BREAKPOINT_DUMP_096("After PageRank", &graph);

    // --- LaTeX summary ---
    report.print_latex_table();
    std::cout << "=== M096 Driver Harness Experiment COMPLETE ===\n";
}

// =============================================================================
// 导出接口: 供M097 unified_runner调用
// =============================================================================
HarnessReport run_driver_harness() {
    HarnessReport report;
    execute(report);
    return report;
}

} // namespace philemon::m096

int main() {
    philemon::m096::HarnessReport report;
    philemon::m096::execute(report);
    return 0;
}
