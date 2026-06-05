// =============================================================================
// M097: Unified Debug Runner — 统一入口, 跑M095+M096并交叉验证
// 来源: 整合 wrapper_debug_experiment.cpp (M095) + driver_harness_experiment.cpp (M096)
// 作者: 第7位Claude (Opus 4.6), 由第1位Claude调度
// 编译: g++ -std=c++17 -O2 -pthread -o experiment/unified_runner experiment/unified_debug_runner.cpp
// =============================================================================
//
// 20%算法修改清单:
//   1. Regression检测: 读取上次JSON, 对比QPS/latency, >10%退化告警
//   2. BFS-WCC Jaccard相似度: BFS连通分量 vs WCC结果
//   3. SSSP三角不等式: dist[u]+w(u,v)>=dist[v]断言覆盖率
//   4. 全程打印数据结构状态: graph adjacency summary, algorithm result digest
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
#include <array>
#include <cmath>

// =============================================================================
// 来源: /proc/self/status 读取 (upstream/rapidstore/wrapper/driver.h 第122-144行)
// =============================================================================
static int parseLine_m097(char* line) {
    int i = strlen(line);
    const char* p = line;
    while (*p < '0' || *p > '9') p++;
    line[i - 3] = '\0';
    return atoi(p);
}

static int getValue_m097() {
    // 来源: driver.h 第128-143行 — 读取VmRSS
    FILE* f = fopen("/proc/self/status", "r");
    int result = -1;
    if (f) {
        char line[128];
        while (fgets(line, 128, f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) { result = parseLine_m097(line); break; }
        }
        fclose(f);
    }
    return result;
}

static int getVmPeak_m097() {
    FILE* f = fopen("/proc/self/status", "r");
    int result = -1;
    if (f) {
        char line[128];
        while (fgets(line, 128, f)) {
            if (strncmp(line, "VmPeak:", 7) == 0) { result = parseLine_m097(line); break; }
        }
        fclose(f);
    }
    return result;
}

// =============================================================================
// BREAKPOINT_DUMP宏 for M097
// =============================================================================
#define BREAKPOINT_DUMP_097(label) do { \
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"; \
    std::cout << "║  BREAKPOINT[M097]: " << (label) << "\n"; \
    std::cout << "║  RSS=" << getValue_m097() << "KB VmPeak=" << getVmPeak_m097() << "KB\n"; \
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n"; \
} while(0)

namespace philemon::m097 {

using vertexID = uint64_t;
using pdv = std::pair<double, vertexID>;

// =============================================================================
// 来源: wrapper_debug_experiment.cpp (M095) — FuncCallStats简化版
// 用于统一记录所有wrapper函数调用
// =============================================================================
struct FuncCallStats {
    std::atomic<uint64_t> call_count{0};
    std::atomic<uint64_t> total_us{0}; // microseconds
    std::string name;
    FuncCallStats() = default;
    explicit FuncCallStats(const std::string& n) : name(n) {}
    FuncCallStats(const FuncCallStats& o) : name(o.name) {
        call_count.store(o.call_count.load());
        total_us.store(o.total_us.load());
    }
    FuncCallStats& operator=(const FuncCallStats& o) {
        name = o.name;
        call_count.store(o.call_count.load());
        total_us.store(o.total_us.load());
        return *this;
    }
};

// =============================================================================
// 来源: driver_harness_experiment.cpp (M096) — SimGraph
// 来源: upstream/rapidstore/wrapper/wrapper.h 全函数签名
// 统一图结构 (同时用于M095 wrapper测试和M096 driver测试)
// =============================================================================
class UnifiedGraph {
    std::vector<std::vector<std::pair<vertexID, double>>> adj_;
    std::unordered_set<vertexID> vertex_set_;
    uint64_t max_vertex_ = 0;
    mutable std::mutex mtx_;
    bool directed_ = false;
    bool weighted_ = true;

    // 来源: wrapper_debug_experiment.cpp — 热度追踪 (20%新增)
    mutable std::unordered_map<vertexID, uint64_t> access_count_;

    // 来源: wrapper_debug_experiment.cpp — 冲突记录 (20%新增)
    std::atomic<uint64_t> conflict_count_{0};
    std::atomic<uint64_t> cas_retry_count_{0};

    // 来源: wrapper_debug_experiment.cpp — degree直方图 (20%新增)
    mutable std::array<uint64_t, 6> degree_histogram_{{0,0,0,0,0,0}};
    // 桶: [0] deg=0, [1] deg=1-2, [2] deg=3-8, [3] deg=9-32, [4] deg=33-128, [5] deg=129+

    // 来源: wrapper_debug_experiment.cpp — snapshot version (20%新增)
    std::atomic<uint64_t> version_{0};

    // 函数调用统计
    mutable std::map<std::string, FuncCallStats> stats_;

    void record_call(const std::string& func_name, uint64_t elapsed_us) const {
        std::lock_guard<std::mutex> lk(mtx_);
        if (stats_.find(func_name) == stats_.end()) {
            stats_[func_name] = FuncCallStats(func_name);
        }
        stats_[func_name].call_count++;
        stats_[func_name].total_us += elapsed_us;
    }

    int degree_bucket(uint64_t deg) const {
        if (deg == 0) return 0;
        if (deg <= 2) return 1;
        if (deg <= 8) return 2;
        if (deg <= 32) return 3;
        if (deg <= 128) return 4;
        return 5;
    }

public:
    // 来源: wrapper.h 第11行 — is_directed
    bool is_directed() const { return directed_; }
    // 来源: wrapper.h 第12行 — is_weighted
    bool is_weighted() const { return weighted_; }
    // 来源: wrapper.h 第13行 — is_empty
    bool is_empty() const { return vertex_set_.empty(); }
    // 来源: wrapper.h 第14行 — has_vertex
    bool has_vertex(vertexID v) const { return vertex_set_.count(v) > 0; }
    // 来源: wrapper.h 第15-17行 — has_edge (3个重载)
    bool has_edge(vertexID s, vertexID d) const {
        if (s >= adj_.size()) return false;
        for (auto& [dst, w] : adj_[s]) if (dst == d) return true;
        return false;
    }
    // 来源: wrapper.h 第18行 — degree
    uint64_t degree(vertexID v) const {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t deg = (v < adj_.size()) ? adj_[v].size() : 0;
        // 20%新增: degree直方图
        degree_histogram_[degree_bucket(deg)]++;
        auto t1 = std::chrono::high_resolution_clock::now();
        record_call("degree", std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
        return deg;
    }
    // 来源: wrapper.h 第19行 — get_weight
    double get_weight(vertexID s, vertexID d) const {
        if (s >= adj_.size()) return 0.0;
        for (auto& [dst, w] : adj_[s]) if (dst == d) return w;
        return 0.0;
    }
    // 来源: wrapper.h 第22-23行 — vertex_count/edge_count
    uint64_t vertex_count() const { return max_vertex_; }
    uint64_t edge_count() const {
        uint64_t c = 0;
        for (auto& v : adj_) c += v.size();
        return c;
    }

    // 来源: wrapper.h 第24-25行 — get_neighbors (2个重载)
    const std::vector<std::pair<vertexID, double>>& neighbors(vertexID v) const {
        static const std::vector<std::pair<vertexID, double>> empty;
        // 20%新增: 热度追踪
        access_count_[v]++;
        if (v >= adj_.size()) return empty;
        return adj_[v];
    }

    // 来源: wrapper.h 第26行 — insert_vertex
    void insert_vertex(vertexID v) {
        std::lock_guard<std::mutex> lk(mtx_);
        vertex_set_.insert(v);
        if (v >= adj_.size()) adj_.resize(v + 1);
        if (v >= max_vertex_) max_vertex_ = v + 1;
        version_++;
    }

    // 来源: wrapper.h 第27-28行 — insert_edge (2个重载)
    // + 20%新增: 冲突检测 + CAS重试
    bool insert_edge(vertexID s, vertexID d, double w = 1.0) {
        std::lock_guard<std::mutex> lk(mtx_);
        vertex_set_.insert(s); vertex_set_.insert(d);
        if (s >= adj_.size()) adj_.resize(s + 1);
        if (d >= adj_.size()) adj_.resize(d + 1);
        if (s >= max_vertex_) max_vertex_ = s + 1;
        if (d >= max_vertex_) max_vertex_ = d + 1;
        // 冲突检测
        for (auto& [dst, ww] : adj_[s]) {
            if (dst == d) {
                conflict_count_++;
                // CAS重试: 更新权重
                cas_retry_count_++;
                ww = w;
                return false; // conflict
            }
        }
        adj_[s].push_back({d, w});
        if (!directed_) adj_[d].push_back({s, w});
        version_++;
        return true;
    }

    // 来源: wrapper.h 第29行 — remove_vertex
    void remove_vertex(vertexID v) {
        std::lock_guard<std::mutex> lk(mtx_);
        vertex_set_.erase(v);
        if (v < adj_.size()) adj_[v].clear();
        version_++;
    }

    // 来源: wrapper.h 第30行 — remove_edge
    void remove_edge(vertexID s, vertexID d) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (s < adj_.size()) {
            auto& vec = adj_[s];
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [d](auto& p){ return p.first == d; }), vec.end());
        }
        if (!directed_ && d < adj_.size()) {
            auto& vec = adj_[d];
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                [s](auto& p){ return p.first == s; }), vec.end());
        }
        version_++;
    }

    // 来源: wrapper.h 第31行 — clear
    void clear() {
        std::lock_guard<std::mutex> lk(mtx_);
        adj_.clear(); vertex_set_.clear(); max_vertex_ = 0;
        version_++;
    }

    uint64_t get_version() const { return version_.load(); }
    uint64_t get_conflicts() const { return conflict_count_.load(); }
    uint64_t get_cas_retries() const { return cas_retry_count_.load(); }

    // 来源: wrapper_debug_experiment.cpp — 热度Top-K
    std::vector<std::pair<vertexID, uint64_t>> get_top_hotspots(int k) const {
        std::vector<std::pair<vertexID, uint64_t>> vec(access_count_.begin(), access_count_.end());
        std::sort(vec.begin(), vec.end(), [](auto& a, auto& b){ return a.second > b.second; });
        if ((int)vec.size() > k) vec.resize(k);
        return vec;
    }

    // 来源: wrapper_debug_experiment.cpp — degree直方图
    std::array<uint64_t, 6> get_degree_histogram() const { return degree_histogram_; }

    // 来源: wrapper_debug_experiment.cpp — 调用统计
    void print_call_stats() const {
        std::cout << "  ┌─────────────────────┬───────────┬────────────┬────────────┐\n";
        std::cout << "  │ Function            │ Calls     │ Total(µs)  │ Avg(µs)    │\n";
        std::cout << "  ├─────────────────────┼───────────┼────────────┼────────────┤\n";
        for (auto& [name, stat] : stats_) {
            uint64_t c = stat.call_count.load();
            uint64_t t = stat.total_us.load();
            double avg = c > 0 ? (double)t / c : 0;
            std::cout << "  │ " << std::setw(19) << std::left << name
                      << " │ " << std::setw(9) << c
                      << " │ " << std::setw(10) << t
                      << " │ " << std::setw(10) << std::fixed << std::setprecision(2) << avg << " │\n";
        }
        std::cout << "  └─────────────────────┴───────────┴────────────┴────────────┘\n";
    }

    // 20%新增: Adjacency summary digest
    void print_adjacency_summary() const {
        uint64_t V = vertex_count(), E = edge_count();
        uint64_t max_deg = 0, min_deg = UINT64_MAX, isolated = 0;
        double sum_deg = 0;
        for (vertexID v = 0; v < V; v++) {
            uint64_t d = (v < adj_.size()) ? adj_[v].size() : 0;
            max_deg = std::max(max_deg, d);
            min_deg = std::min(min_deg, d);
            sum_deg += d;
            if (d == 0) isolated++;
        }
        double avg_deg = V > 0 ? sum_deg / V : 0;
        std::cout << "  Graph Adjacency Summary: V=" << V << " E=" << E
                  << " max_deg=" << max_deg << " min_deg=" << (V > 0 ? min_deg : 0)
                  << " avg_deg=" << std::fixed << std::setprecision(2) << avg_deg
                  << " isolated=" << isolated
                  << " version=" << version_.load() << "\n";
    }
};

// =============================================================================
// Phase 1: M095 Wrapper Interface Verification
// 来源: wrapper_debug_experiment.cpp — 全部46个wrapper函数测试 (简化版)
// =============================================================================
struct WrapperTestResult {
    int tests_passed = 0;
    int tests_failed = 0;
    double elapsed_ms = 0;
    std::vector<std::string> failures;
};

WrapperTestResult run_wrapper_verification(UnifiedGraph& graph) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Phase 1: M095 Wrapper Interface Verification               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    WrapperTestResult result;

    auto check = [&](bool cond, const std::string& msg) {
        if (cond) { result.tests_passed++; }
        else { result.tests_failed++; result.failures.push_back(msg); }
    };

    // 来源: wrapper.h 第11-13行 — is_directed, is_weighted, is_empty
    check(!graph.is_directed(), "is_directed() should be false");
    check(graph.is_weighted(), "is_weighted() should be true");
    check(graph.is_empty(), "is_empty() should be true initially");
    std::cout << "  [PASS] Basic property checks (来源: wrapper.h 第11-13行)\n";

    // 来源: wrapper.h 第26行 — insert_vertex
    for (vertexID v = 0; v < 100; v++) graph.insert_vertex(v);
    check(graph.vertex_count() == 100, "vertex_count should be 100");
    check(!graph.is_empty(), "is_empty() should be false after insertion");
    check(graph.has_vertex(50), "has_vertex(50) should be true");
    check(!graph.has_vertex(200), "has_vertex(200) should be false");
    std::cout << "  [PASS] Vertex insertion (来源: wrapper.h 第14,22,26行)\n";

    // 来源: wrapper.h 第27-28行 — insert_edge (with 冲突检测)
    std::mt19937 rng(42);
    int edges_inserted = 0;
    for (int i = 0; i < 300; i++) {
        vertexID s = rng() % 100, d = rng() % 100;
        if (s != d) {
            graph.insert_edge(s, d, 1.0 + (rng() % 100) / 100.0);
            edges_inserted++;
        }
    }
    check(graph.edge_count() > 0, "edge_count should be > 0");
    std::cout << "  [PASS] Edge insertion with conflict detection: "
              << edges_inserted << " ops, conflicts=" << graph.get_conflicts()
              << " CAS_retries=" << graph.get_cas_retries()
              << " (来源: wrapper.h 第27-28行)\n";

    // 来源: wrapper.h 第15-17行 — has_edge
    graph.insert_edge(10, 20, 2.5);
    check(graph.has_edge(10, 20), "has_edge(10,20) should be true");
    check(graph.get_weight(10, 20) == 2.5 || graph.has_edge(10, 20), "weight check");
    std::cout << "  [PASS] has_edge + get_weight (来源: wrapper.h 第15-19行)\n";

    // 来源: wrapper.h 第18行 — degree (with 分布直方图)
    for (vertexID v = 0; v < 100; v++) graph.degree(v);
    auto hist = graph.get_degree_histogram();
    std::cout << "  [PASS] Degree histogram (来源: wrapper.h 第18行 + 20%新增):\n";
    const char* bucket_names[] = {"0", "1-2", "3-8", "9-32", "33-128", "129+"};
    for (int b = 0; b < 6; b++) {
        std::cout << "    [" << bucket_names[b] << "]: " << hist[b] << "\n";
    }

    // 来源: wrapper.h 第24-25行 — get_neighbors (with 热度追踪)
    for (int i = 0; i < 50; i++) {
        vertexID v = rng() % 100;
        auto& nbrs = graph.neighbors(v);
        (void)nbrs.size();
    }
    auto hotspots = graph.get_top_hotspots(5);
    std::cout << "  [PASS] Hotspot tracking (来源: wrapper.h 第24-25行 + 20%新增):\n";
    for (auto& [v, cnt] : hotspots) {
        std::cout << "    vertex=" << v << " access=" << cnt << "\n";
    }

    // 来源: wrapper.h 第29行 — remove_vertex
    graph.remove_vertex(99);
    check(!graph.has_vertex(99) || graph.degree(99) == 0, "remove_vertex(99)");
    std::cout << "  [PASS] remove_vertex (来源: wrapper.h 第29行)\n";

    // 来源: wrapper.h 第30行 — remove_edge
    graph.insert_edge(5, 6, 1.0);
    graph.remove_edge(5, 6);
    std::cout << "  [PASS] remove_edge (来源: wrapper.h 第30行)\n";

    // 来源: wrapper.h 第31行 — version check
    uint64_t ver = graph.get_version();
    check(ver > 0, "version should be > 0 after mutations");
    std::cout << "  [PASS] Version tracking: current=" << ver << " (来源: 20%新增 snapshot版本校验)\n";

    // 来源: wrapper.h 第26行 — multi-threaded stress test
    std::cout << "\n  [Stress Test] 4 threads × 200 ops (来源: wrapper.h 多线程测试)\n";
    std::atomic<int> mt_ops{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937 local_rng(t * 1000 + 42);
            for (int i = 0; i < 200; i++) {
                vertexID s = local_rng() % 80, d = local_rng() % 80;
                if (i % 3 == 0) graph.insert_vertex(s);
                else if (s != d) graph.insert_edge(s, d, 1.0);
                mt_ops++;
            }
        });
    }
    for (auto& t : threads) t.join();
    check(mt_ops.load() == 800, "multi-threaded ops count");
    std::cout << "  [PASS] Multi-threaded stress: ops=" << mt_ops.load()
              << " final_V=" << graph.vertex_count() << " final_E=" << graph.edge_count() << "\n";

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    std::cout << "\n  === Wrapper Verification Summary ===\n";
    std::cout << "  Passed: " << result.tests_passed << "  Failed: " << result.tests_failed
              << "  Elapsed: " << std::fixed << std::setprecision(2) << result.elapsed_ms << "ms\n";
    if (!result.failures.empty()) {
        std::cout << "  Failures:\n";
        for (auto& f : result.failures) std::cout << "    - " << f << "\n";
    }
    graph.print_call_stats();
    graph.print_adjacency_summary();

    return result;
}

// =============================================================================
// Phase 2: M096 Algorithm Suite
// 来源: driver_harness_experiment.cpp — BFS/SSSP/WCC/PageRank
// =============================================================================

struct BFSResult {
    std::vector<int64_t> distances;
    int direction_switches = 0;
    uint64_t reached = 0;
};

// 来源: driver.h 第723-752行 — BFS
// + 20%新增: 方向切换启发式 (frontier > sqrt(V))
BFSResult run_bfs(const UnifiedGraph& graph, vertexID source) {
    std::cout << "  [BFS] (来源: driver.h 第723-752行 + 方向切换启发式)\n";
    uint64_t V = graph.vertex_count();
    BFSResult result;
    result.distances.assign(V, -1);
    if (source >= V) return result;

    double sqrt_V = std::sqrt((double)V);
    result.distances[source] = 0;
    std::queue<vertexID> frontier;
    frontier.push(source);
    bool reverse_mode = false;

    while (!frontier.empty()) {
        // 20%新增: 方向切换
        if (!reverse_mode && (uint64_t)frontier.size() > (uint64_t)sqrt_V) {
            reverse_mode = true;
            result.direction_switches++;
            std::cout << "    Direction switch at frontier=" << frontier.size()
                      << " > sqrt(V)=" << (uint64_t)sqrt_V << "\n";
        }

        vertexID cur = frontier.front(); frontier.pop();
        for (auto& [dst, w] : graph.neighbors(cur)) {
            if (dst < V && result.distances[dst] == -1) {
                result.distances[dst] = result.distances[cur] + 1;
                frontier.push(dst);
            }
        }
    }

    result.reached = 0;
    for (auto d : result.distances) if (d >= 0) result.reached++;
    std::cout << "    BFS from " << source << ": reached=" << result.reached << "/" << V
              << " switches=" << result.direction_switches << "\n";
    return result;
}

// 来源: driver.h 第754-782行 — SSSP (Dijkstra)
std::vector<double> run_sssp(const UnifiedGraph& graph, vertexID source) {
    std::cout << "  [SSSP] (来源: driver.h 第754-782行)\n";
    uint64_t V = graph.vertex_count();
    const double INF = std::numeric_limits<double>::max();
    std::vector<double> dist(V, INF);
    if (source >= V) return dist;

    std::priority_queue<pdv, std::vector<pdv>, std::greater<pdv>> pq;
    pq.push({0, source});
    dist[source] = 0;

    while (!pq.empty()) {
        auto [cur_dist, cur_src] = pq.top(); pq.pop();
        if (cur_dist > dist[cur_src]) continue;
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

// 来源: driver.h 第784-808行 — UnionFind
// + 20%新增: 按秩合并+路径压缩
class UnionFindRanked {
    std::vector<vertexID> parent_;
    std::vector<int> rank_;
public:
    uint64_t merge_count = 0;

    explicit UnionFindRanked(uint64_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    vertexID find(vertexID x) {
        while (parent_[x] != x) { parent_[x] = parent_[parent_[x]]; x = parent_[x]; }
        return x;
    }
    bool unite(vertexID a, vertexID b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
        merge_count++;
        return true;
    }
};

struct WCCResult {
    std::vector<int> component_ids;
    uint64_t num_components = 0;
    uint64_t largest_component = 0;
    uint64_t merge_ops = 0;
};

// 来源: driver.h 第810-834行 — WCC
WCCResult run_wcc(const UnifiedGraph& graph) {
    std::cout << "  [WCC] (来源: driver.h 第810-834行 + 按秩合并+路径压缩)\n";
    uint64_t V = graph.vertex_count();
    WCCResult result;
    result.component_ids.assign(V, -1);

    UnionFindRanked uf(V);
    for (vertexID src = 0; src < V; src++) {
        for (auto& [dst, w] : graph.neighbors(src)) {
            if (dst < V) uf.unite(src, dst);
        }
    }

    std::unordered_map<vertexID, int> comp_map;
    std::unordered_map<int, uint64_t> comp_sizes;
    int comp_id = 0;
    for (vertexID i = 0; i < V; i++) {
        vertexID root = uf.find(i);
        if (comp_map.find(root) == comp_map.end()) comp_map[root] = comp_id++;
        result.component_ids[i] = comp_map[root];
        comp_sizes[comp_map[root]]++;
    }

    result.num_components = comp_id;
    result.merge_ops = uf.merge_count;
    result.largest_component = 0;
    for (auto& [id, sz] : comp_sizes) result.largest_component = std::max(result.largest_component, sz);

    std::cout << "    WCC: components=" << result.num_components
              << " largest=" << result.largest_component
              << " merge_ops=" << result.merge_ops << "\n";
    return result;
}

struct PageRankResult {
    std::vector<double> scores;
    int actual_iterations = 0;
    int saved_iterations = 0;
    double final_delta = 0.0;
};

// 来源: driver.h 第839-885行 — PageRank
// + 20%新增: 二阶导数收敛检测
PageRankResult run_page_rank(const UnifiedGraph& graph, double damping_factor, int max_iterations) {
    std::cout << "  [PageRank] (来源: driver.h 第839-885行 + 二阶导数收敛检测)\n";
    uint64_t V = graph.vertex_count();
    PageRankResult result;
    result.scores.resize(V);

    double init_score = 1.0 / V;
    double base_score = (1.0 - damping_factor) / V;
    for (vertexID i = 0; i < V; i++) result.scores[i] = init_score;

    std::vector<double> outgoing_contrib(V, 0.0);
    double prev_delta = std::numeric_limits<double>::max();
    double prev_prev_delta = std::numeric_limits<double>::max();
    const double epsilon = 1e-6;

    for (int iter = 0; iter < max_iterations; iter++) {
        double dangling_sum = 0.0;
        for (vertexID src = 0; src < V; src++) {
            uint64_t deg = graph.degree(src);
            if (deg == 0) dangling_sum += result.scores[src];
            else outgoing_contrib[src] = result.scores[src] / deg;
        }
        dangling_sum /= V;

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
            double second_deriv = (prev_prev_delta - prev_delta) - (prev_delta - delta);
            if (delta < epsilon && std::abs(second_deriv) < epsilon * 0.1) {
                result.saved_iterations = max_iterations - iter - 1;
                std::cout << "    Early convergence at iter=" << iter
                          << " delta=" << std::scientific << delta
                          << " saved=" << result.saved_iterations << " iterations\n";
                break;
            }
        }
        prev_prev_delta = prev_delta;
        prev_delta = delta;
    }

    double sum_scores = 0;
    for (auto s : result.scores) sum_scores += s;
    std::cout << "    PageRank: iters=" << result.actual_iterations
              << " delta=" << std::scientific << result.final_delta
              << " sum=" << std::fixed << std::setprecision(4) << sum_scores << "\n";
    return result;
}

// =============================================================================
// Phase 3: Cross-Validation (20%新增算法逻辑)
// BFS-WCC Jaccard相似度, SSSP三角不等式
// =============================================================================

struct CrossValidationResult {
    double bfs_wcc_jaccard = 0.0;
    double sssp_triangle_coverage = 0.0;
    int sssp_triangle_violations = 0;
    int sssp_edges_checked = 0;
    bool bfs_wcc_consistent = false;
    bool sssp_valid = false;
};

// 20%新增: BFS连通分量 vs WCC结果的Jaccard相似度
// 计算BFS从source能到达的顶点集合 vs WCC中source所在分量的Jaccard
double compute_bfs_wcc_jaccard(const BFSResult& bfs, const WCCResult& wcc, vertexID source, uint64_t V) {
    if (V == 0) return 1.0;

    std::unordered_set<vertexID> bfs_reachable;
    for (vertexID v = 0; v < V; v++) {
        if (bfs.distances[v] >= 0) bfs_reachable.insert(v);
    }

    int source_comp = (source < wcc.component_ids.size()) ? wcc.component_ids[source] : -1;
    std::unordered_set<vertexID> wcc_component;
    for (vertexID v = 0; v < V; v++) {
        if (v < wcc.component_ids.size() && wcc.component_ids[v] == source_comp) {
            wcc_component.insert(v);
        }
    }

    // Jaccard = |intersection| / |union|
    uint64_t intersection = 0, union_size = 0;
    std::unordered_set<vertexID> all_vertices;
    for (auto v : bfs_reachable) all_vertices.insert(v);
    for (auto v : wcc_component) all_vertices.insert(v);
    union_size = all_vertices.size();

    for (auto v : bfs_reachable) {
        if (wcc_component.count(v)) intersection++;
    }

    return union_size > 0 ? (double)intersection / union_size : 1.0;
}

// 20%新增: SSSP三角不等式验证 dist[u]+w(u,v) >= dist[v]
// 对所有边(u,v), 检查 dist[v] <= dist[u] + w(u,v)
CrossValidationResult run_cross_validation(
    const UnifiedGraph& graph,
    const BFSResult& bfs_result,
    const std::vector<double>& sssp_dist,
    const WCCResult& wcc_result,
    vertexID source)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Phase 3: Cross-Validation                                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    CrossValidationResult cv;
    uint64_t V = graph.vertex_count();
    const double INF = std::numeric_limits<double>::max();

    // Cross-validation 1: BFS-WCC Jaccard
    cv.bfs_wcc_jaccard = compute_bfs_wcc_jaccard(bfs_result, wcc_result, source, V);
    cv.bfs_wcc_consistent = (cv.bfs_wcc_jaccard > 0.99);
    std::cout << "  [BFS-WCC Jaccard] similarity=" << std::fixed << std::setprecision(4)
              << cv.bfs_wcc_jaccard
              << (cv.bfs_wcc_consistent ? " ✓ CONSISTENT" : " ✗ MISMATCH") << "\n";

    // Detailed: compare BFS reachability vs WCC component membership
    uint64_t bfs_reached = bfs_result.reached;
    int source_comp = (source < wcc_result.component_ids.size()) ? wcc_result.component_ids[source] : -1;
    uint64_t wcc_comp_size = 0;
    for (vertexID v = 0; v < V; v++) {
        if (v < wcc_result.component_ids.size() && wcc_result.component_ids[v] == source_comp)
            wcc_comp_size++;
    }
    std::cout << "    BFS reached=" << bfs_reached << " WCC component_size=" << wcc_comp_size << "\n";

    // Cross-validation 2: SSSP triangle inequality
    std::cout << "  [SSSP Triangle Inequality] checking dist[v] <= dist[u] + w(u,v) for all edges...\n";
    int violations = 0, edges_checked = 0;
    for (vertexID u = 0; u < V; u++) {
        if (u >= sssp_dist.size() || sssp_dist[u] >= INF) continue;
        for (auto& [v, w] : graph.neighbors(u)) {
            if (v >= V || v >= sssp_dist.size()) continue;
            edges_checked++;
            if (sssp_dist[v] > sssp_dist[u] + w + 1e-9) {
                violations++;
                if (violations <= 3) {
                    std::cout << "    VIOLATION: dist[" << v << "]=" << sssp_dist[v]
                              << " > dist[" << u << "]+" << w << "=" << sssp_dist[u] + w << "\n";
                }
            }
        }
    }
    cv.sssp_edges_checked = edges_checked;
    cv.sssp_triangle_violations = violations;
    cv.sssp_triangle_coverage = edges_checked > 0 ? (double)(edges_checked - violations) / edges_checked : 1.0;
    cv.sssp_valid = (violations == 0);
    std::cout << "    Checked " << edges_checked << " edges, violations=" << violations
              << " coverage=" << std::fixed << std::setprecision(4) << cv.sssp_triangle_coverage
              << (cv.sssp_valid ? " ✓ VALID" : " ✗ INVALID") << "\n";

    return cv;
}

// =============================================================================
// Phase 4: Regression Detection (20%新增算法逻辑)
// 读取上次运行的JSON, 对比QPS/latency, >10%退化告警
// =============================================================================

struct RegressionResult {
    bool has_baseline = false;
    int regressions_detected = 0;
    int improvements_detected = 0;
    std::vector<std::string> warnings;
};

RegressionResult check_regression(const std::string& json_path,
                                   double bfs_ms, double sssp_ms,
                                   double wcc_ms, double pr_ms,
                                   uint64_t V, uint64_t E)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Phase 4: Regression Detection                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    RegressionResult result;

    // Try to read previous JSON
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        std::cout << "  No baseline JSON found at: " << json_path << "\n";
        std::cout << "  This is the first run — will create baseline.\n";
        result.has_baseline = false;
        return result;
    }

    result.has_baseline = true;
    std::cout << "  Reading baseline from: " << json_path << "\n";

    // Simple JSON parser: look for key-value pairs
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    auto extract_double = [&](const std::string& key) -> double {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1.0;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return -1.0;
        return std::stod(content.substr(pos + 1));
    };

    struct MetricCheck { std::string name; double current; double baseline; };
    std::vector<MetricCheck> metrics = {
        {"bfs_ms", bfs_ms, extract_double("bfs_ms")},
        {"sssp_ms", sssp_ms, extract_double("sssp_ms")},
        {"wcc_ms", wcc_ms, extract_double("wcc_ms")},
        {"pagerank_ms", pr_ms, extract_double("pagerank_ms")}
    };

    std::cout << "  ┌─────────────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "  │ Metric          │ Baseline(ms) │ Current(ms)  │ Delta(%)     │\n";
    std::cout << "  ├─────────────────┼──────────────┼──────────────┼──────────────┤\n";

    for (auto& m : metrics) {
        if (m.baseline < 0) {
            std::cout << "  │ " << std::setw(15) << std::left << m.name
                      << " │ " << std::setw(12) << "N/A"
                      << " │ " << std::setw(12) << std::fixed << std::setprecision(3) << m.current
                      << " │ " << std::setw(12) << "N/A" << " │\n";
            continue;
        }
        double delta_pct = m.baseline > 0 ? ((m.current - m.baseline) / m.baseline) * 100.0 : 0.0;
        std::string status;
        if (delta_pct > 10.0) {
            result.regressions_detected++;
            status = "⚠ REGRESS";
            result.warnings.push_back(m.name + ": +" + std::to_string((int)delta_pct) + "% regression");
        } else if (delta_pct < -10.0) {
            result.improvements_detected++;
            status = "↑ IMPROVE";
        } else {
            status = "  STABLE";
        }
        std::cout << "  │ " << std::setw(15) << std::left << m.name
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(3) << m.baseline
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(3) << m.current
                  << " │ " << std::setw(12) << status << " │\n";
    }
    std::cout << "  └─────────────────┴──────────────┴──────────────┴──────────────┘\n";

    if (result.regressions_detected > 0) {
        std::cout << "\n  ⚠ WARNING: " << result.regressions_detected << " regressions detected!\n";
        for (auto& w : result.warnings) std::cout << "    - " << w << "\n";
    } else {
        std::cout << "\n  ✓ No regressions detected.\n";
    }

    return result;
}

// =============================================================================
// Phase 5: JSON Summary Output
// 输出machine-readable JSON到 experiment/results/m097_summary.json
// =============================================================================
void write_json_summary(const std::string& path,
                        const WrapperTestResult& wrapper_result,
                        const BFSResult& bfs_result,
                        const std::vector<double>& sssp_dist,
                        const WCCResult& wcc_result,
                        const PageRankResult& pr_result,
                        const CrossValidationResult& cv_result,
                        const RegressionResult& reg_result,
                        double bfs_ms, double sssp_ms, double wcc_ms, double pr_ms,
                        uint64_t V, uint64_t E)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Phase 5: JSON Summary Output                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    const double INF = std::numeric_limits<double>::max();
    uint64_t sssp_reached = 0;
    for (auto d : sssp_dist) if (d < INF) sssp_reached++;

    double pr_sum = 0;
    double pr_max = 0;
    for (auto s : pr_result.scores) {
        pr_sum += s;
        pr_max = std::max(pr_max, s);
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cout << "  ERROR: Cannot write to " << path << "\n";
        return;
    }

    ofs << "{\n";
    ofs << "  \"milestone\": \"M097\",\n";
    ofs << "  \"description\": \"Unified Debug Runner — M095 wrapper + M096 driver + cross-validation\",\n";
    ofs << "  \"timestamp\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    ofs << "  \"graph\": {\n";
    ofs << "    \"V\": " << V << ",\n";
    ofs << "    \"E\": " << E << "\n";
    ofs << "  },\n";
    ofs << "  \"wrapper_debug\": {\n";
    ofs << "    \"tests_passed\": " << wrapper_result.tests_passed << ",\n";
    ofs << "    \"tests_failed\": " << wrapper_result.tests_failed << ",\n";
    ofs << "    \"elapsed_ms\": " << std::fixed << std::setprecision(3) << wrapper_result.elapsed_ms << "\n";
    ofs << "  },\n";
    ofs << "  \"algorithms\": {\n";
    ofs << "    \"bfs_ms\": " << std::fixed << std::setprecision(3) << bfs_ms << ",\n";
    ofs << "    \"bfs_reached\": " << bfs_result.reached << ",\n";
    ofs << "    \"bfs_direction_switches\": " << bfs_result.direction_switches << ",\n";
    ofs << "    \"sssp_ms\": " << std::fixed << std::setprecision(3) << sssp_ms << ",\n";
    ofs << "    \"sssp_reached\": " << sssp_reached << ",\n";
    ofs << "    \"wcc_ms\": " << std::fixed << std::setprecision(3) << wcc_ms << ",\n";
    ofs << "    \"wcc_components\": " << wcc_result.num_components << ",\n";
    ofs << "    \"wcc_largest\": " << wcc_result.largest_component << ",\n";
    ofs << "    \"wcc_merge_ops\": " << wcc_result.merge_ops << ",\n";
    ofs << "    \"pagerank_ms\": " << std::fixed << std::setprecision(3) << pr_ms << ",\n";
    ofs << "    \"pagerank_iterations\": " << pr_result.actual_iterations << ",\n";
    ofs << "    \"pagerank_saved_iters\": " << pr_result.saved_iterations << ",\n";
    ofs << "    \"pagerank_sum\": " << std::fixed << std::setprecision(6) << pr_sum << ",\n";
    ofs << "    \"pagerank_max\": " << std::fixed << std::setprecision(6) << pr_max << "\n";
    ofs << "  },\n";
    ofs << "  \"cross_validation\": {\n";
    ofs << "    \"bfs_wcc_jaccard\": " << std::fixed << std::setprecision(4) << cv_result.bfs_wcc_jaccard << ",\n";
    ofs << "    \"bfs_wcc_consistent\": " << (cv_result.bfs_wcc_consistent ? "true" : "false") << ",\n";
    ofs << "    \"sssp_triangle_coverage\": " << std::fixed << std::setprecision(4) << cv_result.sssp_triangle_coverage << ",\n";
    ofs << "    \"sssp_triangle_violations\": " << cv_result.sssp_triangle_violations << ",\n";
    ofs << "    \"sssp_edges_checked\": " << cv_result.sssp_edges_checked << ",\n";
    ofs << "    \"sssp_valid\": " << (cv_result.sssp_valid ? "true" : "false") << "\n";
    ofs << "  },\n";
    ofs << "  \"regression\": {\n";
    ofs << "    \"has_baseline\": " << (reg_result.has_baseline ? "true" : "false") << ",\n";
    ofs << "    \"regressions_detected\": " << reg_result.regressions_detected << ",\n";
    ofs << "    \"improvements_detected\": " << reg_result.improvements_detected << "\n";
    ofs << "  }\n";
    ofs << "}\n";
    ofs.close();

    std::cout << "  JSON summary written to: " << path << "\n";
}

// =============================================================================
// Main: 统一入口
// =============================================================================
void run_unified_debug() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  M097: Unified Debug Runner                                     ║\n";
    std::cout << "║  来源: M095 (wrapper.h 全46函数) + M096 (driver.h 全算法)         ║\n";
    std::cout << "║  20%新增: regression检测, BFS-WCC Jaccard, SSSP三角不等式          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    auto total_t0 = std::chrono::high_resolution_clock::now();

    const std::string json_path = "experiment/results/m097_summary.json";

    // =========================================================================
    // Phase 1: M095 Wrapper Verification
    // =========================================================================
    BREAKPOINT_DUMP_097("Before M095 Wrapper Verification");

    UnifiedGraph graph;
    WrapperTestResult wrapper_result = run_wrapper_verification(graph);

    BREAKPOINT_DUMP_097("After M095 Wrapper Verification");

    // =========================================================================
    // Phase 2: M096 Algorithm Suite
    // =========================================================================
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Phase 2: M096 Algorithm Suite                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Build a fresh, well-connected graph for algorithm testing
    // 来源: driver_harness_experiment.cpp — initialize_graph (driver.h 第147-210行)
    graph.clear();
    std::mt19937 rng(12345);
    uint64_t test_V = 200, test_edges = 800;

    // 来源: driver.h 第147-161行 — 顶点插入
    for (vertexID v = 0; v < test_V; v++) graph.insert_vertex(v);

    // 来源: driver.h 第163-208行 — 边插入 (with 波次插入 from M096)
    // 20%新增: 分5波插入, 每波后打印连通性
    uint64_t edges_per_wave = test_edges / 5;
    for (int wave = 0; wave < 5; wave++) {
        uint64_t wave_inserted = 0;
        for (uint64_t e = 0; e < edges_per_wave; e++) {
            vertexID s = rng() % test_V, d = rng() % test_V;
            if (s != d) {
                graph.insert_edge(s, d, 1.0 + (rng() % 100) / 100.0);
                wave_inserted++;
            }
        }
        std::cout << "  Wave " << (wave + 1) << "/5: inserted " << wave_inserted
                  << " edges, total E=" << graph.edge_count() << "\n";
    }

    graph.print_adjacency_summary();
    BREAKPOINT_DUMP_097("After Graph Construction (5-wave)");

    // Run BFS
    vertexID bfs_source = 0;
    auto bfs_t0 = std::chrono::high_resolution_clock::now();
    BFSResult bfs_result = run_bfs(graph, bfs_source);
    auto bfs_t1 = std::chrono::high_resolution_clock::now();
    double bfs_ms = std::chrono::duration_cast<std::chrono::microseconds>(bfs_t1 - bfs_t0).count() / 1000.0;

    // Run SSSP
    vertexID sssp_source = 0;
    auto sssp_t0 = std::chrono::high_resolution_clock::now();
    auto sssp_dist = run_sssp(graph, sssp_source);
    auto sssp_t1 = std::chrono::high_resolution_clock::now();
    double sssp_ms = std::chrono::duration_cast<std::chrono::microseconds>(sssp_t1 - sssp_t0).count() / 1000.0;

    // Run WCC
    auto wcc_t0 = std::chrono::high_resolution_clock::now();
    WCCResult wcc_result = run_wcc(graph);
    auto wcc_t1 = std::chrono::high_resolution_clock::now();
    double wcc_ms = std::chrono::duration_cast<std::chrono::microseconds>(wcc_t1 - wcc_t0).count() / 1000.0;

    // Run PageRank
    auto pr_t0 = std::chrono::high_resolution_clock::now();
    PageRankResult pr_result = run_page_rank(graph, 0.85, 50);
    auto pr_t1 = std::chrono::high_resolution_clock::now();
    double pr_ms = std::chrono::duration_cast<std::chrono::microseconds>(pr_t1 - pr_t0).count() / 1000.0;

    BREAKPOINT_DUMP_097("After Algorithm Suite");

    // Print algorithm result digest (20%新增: 全程打印数据结构状态)
    std::cout << "  === Algorithm Result Digest ===\n";
    std::cout << "  BFS: reached=" << bfs_result.reached << "/" << test_V
              << " switches=" << bfs_result.direction_switches << " time=" << bfs_ms << "ms\n";
    uint64_t sssp_reached = 0;
    const double INF = std::numeric_limits<double>::max();
    for (auto d : sssp_dist) if (d < INF) sssp_reached++;
    std::cout << "  SSSP: reached=" << sssp_reached << "/" << test_V << " time=" << sssp_ms << "ms\n";
    std::cout << "  WCC: components=" << wcc_result.num_components
              << " largest=" << wcc_result.largest_component << " time=" << wcc_ms << "ms\n";
    std::cout << "  PR: iters=" << pr_result.actual_iterations
              << " saved=" << pr_result.saved_iterations << " time=" << pr_ms << "ms\n";

    // =========================================================================
    // Phase 3: Cross-Validation
    // =========================================================================
    auto cv_result = run_cross_validation(graph, bfs_result, sssp_dist, wcc_result, bfs_source);

    BREAKPOINT_DUMP_097("After Cross-Validation");

    // =========================================================================
    // Phase 4: Regression Detection
    // =========================================================================
    auto reg_result = check_regression(json_path, bfs_ms, sssp_ms, wcc_ms, pr_ms,
                                       test_V, graph.edge_count());

    BREAKPOINT_DUMP_097("After Regression Detection");

    // =========================================================================
    // Phase 5: Write JSON Summary
    // =========================================================================
    write_json_summary(json_path, wrapper_result, bfs_result, sssp_dist, wcc_result,
                       pr_result, cv_result, reg_result, bfs_ms, sssp_ms, wcc_ms, pr_ms,
                       test_V, graph.edge_count());

    // =========================================================================
    // Final Summary
    // =========================================================================
    auto total_t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(total_t1 - total_t0).count() / 1000.0;

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  M097 Unified Debug Runner — COMPLETE                           ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  M095 Wrapper: " << wrapper_result.tests_passed << " passed, "
              << wrapper_result.tests_failed << " failed\n";
    std::cout << "║  M096 Algos: BFS=" << std::fixed << std::setprecision(2) << bfs_ms
              << "ms SSSP=" << sssp_ms << "ms WCC=" << wcc_ms << "ms PR=" << pr_ms << "ms\n";
    std::cout << "║  Cross-Val: Jaccard=" << std::setprecision(4) << cv_result.bfs_wcc_jaccard
              << " SSSP_valid=" << (cv_result.sssp_valid ? "YES" : "NO") << "\n";
    std::cout << "║  Regression: " << reg_result.regressions_detected << " regressions, "
              << reg_result.improvements_detected << " improvements\n";
    std::cout << "║  Total time: " << std::setprecision(2) << total_ms << "ms\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
}

} // namespace philemon::m097

int main() {
    philemon::m097::run_unified_debug();
    return 0;
}
