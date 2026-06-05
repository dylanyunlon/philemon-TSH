/**
 * walking_experiment.cpp — LLM4Walking 全流程合成数据实验入口
 *
 * 策略: mv upstream/philemon-TSH 骨架 + 20% 动态修改 + 断点调试密布
 *
 * 来源文件 (upstream/rapidstore/) → 移植后统一编译:
 *   graph/edge.cpp          (38行)  → WeightedEdge         [MOD: 加tier_id+访问时间戳]
 *   graph/edgeStream.cpp    (82行)  → TieredEdgeStream      [MOD: load时打印进度+校验和]
 *   readers/reader.cpp      (17行)  → ReaderFactory          [KEEP ~100%]
 *   readers/edgeListReader  (76行)  → EdgeListReader         [MOD: 带计数+CRC校验]
 *   readers/vertexReader    (59行)  → VertexReader           [MOD: 带计数的read]
 *   algorithms/BFS.cpp      (302行) → TieredBFS              [MOD: tier感知+自适应阈值]
 *   algorithms/pageRank.cpp (159行) → TieredPageRank         [MOD: 收敛监控+二阶导数]
 *   algorithms/SSSP.cpp     (175行) → TieredSSSP             [MOD: bin分布+松弛率追踪]
 *   algorithms/WCC.cpp      (137行) → TieredWCC              [MOD: 组件合并追踪+联通率]
 *   utils/commandLineParser (700行) → ConfigEngine           [MOD: 简化为直接解析]
 *   types/types.hpp         (150行) → 统一类型               [MOD: 加walking扩展]
 *   main.cpp                (202行) → 本文件main()           [MOD: tier循环+debug]
 *
 * 修改摘要 (每个模块 ~20% 变更):
 *   1. namespace: driver::* → walking::experiment::*
 *   2. 接口: m_interface/m_snapshot → 模板化FakeGraph/FakeSnapshot
 *   3. debug: 每个关键循环加 BREAKPOINT_DUMP() + INSPECT_STATE()
 *   4. tier: 算法内部加 tier_hit_count[3] 追踪 DRAM/CXL/SSD 访问
 *   5. 收敛: PageRank加delta收敛追踪, SSSP加bin分布, WCC加组件大小
 *   6. [+20% NEW] 每阶段自动断言 invariant 检查
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -o walking_experiment walking_experiment.cpp
 *
 * Run:
 *   ./walking_experiment [config.cfg]   # 或无参数用合成数据
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <numeric>
#include <cmath>
#include <sys/resource.h>
#include <unistd.h>

// ═══════════════════════════════════════════════════════════════════
// §0  全局调试基础设施 (贯穿所有模块)
// ═══════════════════════════════════════════════════════════════════

namespace walking { namespace experiment {

// 调试级别: 0=静默 1=摘要 2=逐步 3=逐边
static int g_debug_level = 2;

// 三层内存 tier 枚举
enum TierID : uint8_t { TIER_DRAM = 0, TIER_CXL = 1, TIER_SSD = 2, TIER_COUNT = 3 };
static const char* tier_name(TierID t) {
    static const char* names[] = {"DRAM", "CXL", "SSD"};
    return (t < TIER_COUNT) ? names[t] : "???";
}

// per-thread tier 访问计数器 (用于断点dump)
struct TierAccessCounters {
    std::atomic<uint64_t> reads[TIER_COUNT]  = {};
    std::atomic<uint64_t> writes[TIER_COUNT] = {};
    uint64_t total_reads()  const { uint64_t s=0; for(int i=0;i<TIER_COUNT;i++) s+=reads[i].load(); return s; }
    uint64_t total_writes() const { uint64_t s=0; for(int i=0;i<TIER_COUNT;i++) s+=writes[i].load(); return s; }
    void reset() { for(int i=0;i<TIER_COUNT;i++){reads[i]=0;writes[i]=0;} }
    void dump(const char* label) {
        std::printf("[TIER·ACCESS] %s — reads: DRAM=%lu CXL=%lu SSD=%lu | writes: DRAM=%lu CXL=%lu SSD=%lu\n",
                    label,
                    reads[0].load(), reads[1].load(), reads[2].load(),
                    writes[0].load(), writes[1].load(), writes[2].load());
    }
};
static TierAccessCounters g_tier_counters;

// 通用定时器
struct ScopedTimer {
    const char* label;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {
        if (g_debug_level >= 1) std::printf("[TIMER·START] %s\n", label);
    }
    ~ScopedTimer() {
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::printf("[TIMER·END]   %s → %ld ms\n", label, dt);
    }
};

// RSS 内存查询 (KB)
static long get_rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  // Linux: KB
}

// 分隔线
static void print_separator(const char* tag) {
    std::printf("\n════════════════════════════════════════════════════════════\n");
    std::printf("  %s\n", tag);
    std::printf("════════════════════════════════════════════════════════════\n\n");
}

// 断点宏: 打印完整系统状态
#define BREAKPOINT_DUMP(tag, ...) do { \
    if (walking::experiment::g_debug_level >= 2) { \
        std::printf("[BP·%s] ", tag); \
        std::printf(__VA_ARGS__); \
        std::printf(" | RSS=%ld KB\n", walking::experiment::get_rss_kb()); \
    } \
} while(0)

#define PROGRESS_PRINT(phase, cur, total) do { \
    if (walking::experiment::g_debug_level >= 1 && (cur) % std::max(1UL,(total)/20) == 0) { \
        std::printf("[PROGRESS] %s: %lu / %lu (%.1f%%)\n", \
                    phase, (unsigned long)(cur), (unsigned long)(total), \
                    (total) > 0 ? 100.0*(cur)/(total) : 0.0); \
    } \
} while(0)


// ═══════════════════════════════════════════════════════════════════
// §1  数据结构 (mv from upstream/rapidstore/graph/ + types/)
// ═══════════════════════════════════════════════════════════════════
//
// [MOD] 对比 upstream edge.hpp: 增加 tier_hint 字段 (4字节)
// [MOD] 对比 upstream types.hpp: operationType 精简为实验需要的子集

struct WeightedEdge {
    uint64_t source      = 0;
    uint64_t destination = 0;
    double   weight      = 0.0;
    TierID   tier_hint   = TIER_DRAM;  // [MOD +20%] tier placement hint

    WeightedEdge() = default;
    WeightedEdge(uint64_t s, uint64_t d, double w = 1.0)
        : source(s), destination(d), weight(w), tier_hint(TIER_DRAM) {}

    bool operator==(const WeightedEdge& o) const { return source==o.source && destination==o.destination; }
    bool operator<(const WeightedEdge& o) const {
        return source < o.source || (source == o.source && destination < o.destination);
    }
};

// Edge stream (mv from edgeStream.cpp, +debug进度)
class TieredEdgeStream {
    std::vector<WeightedEdge> edges_;
    size_t idx_ = 0;
public:
    // [MOD] 从文件加载, 每10万条打印一次进度
    void load_from_file(const std::string& path) {
        std::ifstream fin(path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[EdgeStream·ERROR] Cannot open: %s\n", path.c_str());
            return;
        }
        std::string line;
        uint64_t count = 0;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            uint64_t s, d; double w = 1.0;
            if (!(ss >> s >> d)) continue;
            ss >> w;  // optional weight
            edges_.emplace_back(s, d, w);
            count++;
            PROGRESS_PRINT("EDGE_LOAD", count, 0UL);  // 不知道总数,每10万条打印
        }
        std::printf("[EdgeStream] Loaded %lu edges from %s\n", count, path.c_str());
    }

    // 合成数据生成 (替代文件加载)
    void generate_synthetic(uint64_t num_vertices, uint64_t num_edges, uint64_t seed = 42) {
        ScopedTimer t("SYNTHETIC_EDGE_GEN");
        std::mt19937_64 rng(seed);
        edges_.reserve(num_edges);
        for (uint64_t i = 0; i < num_edges; i++) {
            uint64_t s = rng() % num_vertices;
            uint64_t d = rng() % num_vertices;
            if (s == d) { d = (d + 1) % num_vertices; }
            double w = 1.0 + (rng() % 100) / 10.0;

            // [MOD +20%] 给边分配 tier hint: 80% DRAM, 15% CXL, 5% SSD
            TierID th = TIER_DRAM;
            uint64_t r = rng() % 100;
            if (r >= 95) th = TIER_SSD;
            else if (r >= 80) th = TIER_CXL;

            WeightedEdge e(s, d, w);
            e.tier_hint = th;
            edges_.push_back(e);

            if (g_debug_level >= 2 && i % (num_edges/5+1) == 0)
                BREAKPOINT_DUMP("EDGE_GEN", "i=%lu src=%lu dst=%lu tier=%s",
                                i, s, d, tier_name(th));
        }
        // [DEBUG] 打印 tier 分布统计
        uint64_t tc[3] = {};
        for (auto& e : edges_) tc[e.tier_hint]++;
        std::printf("[EdgeStream·STATS] tier distribution: DRAM=%lu CXL=%lu SSD=%lu\n",
                    tc[0], tc[1], tc[2]);
    }

    void permute(uint64_t seed = 0) {
        std::shuffle(edges_.begin(), edges_.end(), std::mt19937_64(seed));
    }

    WeightedEdge& operator[](size_t i) { return edges_[i]; }
    const WeightedEdge& operator[](size_t i) const { return edges_[i]; }
    size_t size() const { return edges_.size(); }
    const std::vector<WeightedEdge>& data() const { return edges_; }
};


// ═══════════════════════════════════════════════════════════════════
// §2  FakeGraph — 模拟 upstream 的 graphalyticsInterface + snapshot
// ═══════════════════════════════════════════════════════════════════
//
// upstream 用 m_interface→get_shared_snapshot()→edges(v, callback)
// 我们用邻接表 + tier标注模拟，不依赖任何外部库

class FakeSnapshot;

class FakeGraph {
    uint64_t num_v_ = 0;
    // adjacency: adj_[src] = vector of (dst, weight, tier)
    std::vector<std::vector<std::tuple<uint64_t, double, TierID>>> adj_;
    std::mutex mu_;
public:
    void init(uint64_t num_vertices) {
        num_v_ = num_vertices;
        adj_.resize(num_vertices);
    }

    void insert_edge(uint64_t src, uint64_t dst, double w, TierID tier = TIER_DRAM) {
        if (src >= num_v_ || dst >= num_v_) return;
        std::lock_guard<std::mutex> lk(mu_);
        adj_[src].emplace_back(dst, w, tier);
        g_tier_counters.writes[tier]++;
    }

    void batch_insert(const TieredEdgeStream& stream, size_t begin, size_t end) {
        for (size_t i = begin; i < end && i < stream.size(); i++) {
            auto& e = stream[i];
            insert_edge(e.source, e.destination, e.weight, e.tier_hint);
        }
    }

    uint64_t vertex_count() const { return num_v_; }
    uint64_t edge_count() const {
        uint64_t c = 0;
        for (auto& a : adj_) c += a.size();
        return c;
    }

    uint64_t degree(uint64_t v) const {
        return (v < num_v_) ? adj_[v].size() : 0;
    }

    // snapshot-style edge iteration with tier tracking
    template <typename Callback>
    void edges(uint64_t v, Callback&& cb) const {
        if (v >= num_v_) return;
        for (auto& [dst, w, tier] : adj_[v]) {
            g_tier_counters.reads[tier]++;
            cb(dst, w);
        }
    }

    // tier 分布统计 (断点用)
    void dump_tier_distribution(const char* label) const {
        uint64_t counts[3] = {};
        for (auto& a : adj_)
            for (auto& [d,w,t] : a) counts[t]++;
        std::printf("[GRAPH·TIER] %s — edges by tier: DRAM=%lu CXL=%lu SSD=%lu total=%lu\n",
                    label, counts[0], counts[1], counts[2],
                    counts[0]+counts[1]+counts[2]);
    }

    // degree distribution histogram (断点用)
    void dump_degree_histogram(const char* label) const {
        if (g_debug_level < 2) return;
        uint64_t d0=0, d1_10=0, d11_100=0, d101_1k=0, d_big=0;
        for (uint64_t v = 0; v < num_v_; v++) {
            uint64_t d = adj_[v].size();
            if (d == 0) d0++;
            else if (d <= 10) d1_10++;
            else if (d <= 100) d11_100++;
            else if (d <= 1000) d101_1k++;
            else d_big++;
        }
        std::printf("[GRAPH·DEGREE] %s — isolated=%lu 1-10=%lu 11-100=%lu 101-1k=%lu 1k+=%lu\n",
                    label, d0, d1_10, d11_100, d101_1k, d_big);
    }
};


// ═══════════════════════════════════════════════════════════════════
// §3  BFS (mv from upstream/rapidstore/algorithms/BFS.cpp, 302行)
// ═══════════════════════════════════════════════════════════════════
//
// [KEEP] direction-optimized BFS 的 TD↔BU 切换逻辑 100% 保留
// [KEEP] 负数编码degree的init_distances技巧 100% 保留
// [KEEP] TDStep的CAS更新 100% 保留
// [MOD]  namespace: driver::algorithm → philemon::experiment
// [MOD]  m_interface/m_snapshot → FakeGraph& 引用
// [MOD]  gapbs::pvector → std::vector (功能等价)
// [MOD]  每层BFS后打印: 层号、frontier大小、已访问比例、tier命中分布
// [NEW]  TD→BU切换时打印决策参量 (scout_count vs threshold)
// [NEW]  全BFS结束后打印距离分布直方图

class TieredBFS {
    const FakeGraph& graph_;
    int num_threads_;
    int alpha_, beta_;
    std::mutex mu_;

    // [NEW] per-run tier hit 统计
    uint64_t tier_edge_hits_[TIER_COUNT] = {};

public:
    TieredBFS(const FakeGraph& g, int threads, int alpha = 15, int beta = 18)
        : graph_(g), num_threads_(threads), alpha_(alpha), beta_(beta) {}

    std::vector<int64_t> run(uint64_t source) {
        ScopedTimer timer("BFS");
        const uint64_t N = graph_.vertex_count();
        std::vector<int64_t> dist(N);

        // [KEEP] init_distances: 负数编码out-degree (upstream技巧)
        {
            ScopedTimer t("BFS·init_distances");
            for (uint64_t v = 0; v < N; v++) {
                uint64_t deg = graph_.degree(v);
                dist[v] = (deg != 0) ? -(int64_t)deg : -1;
            }
            // [NEW] debug: 打印初始距离分布
            int64_t zero_deg = 0, pos_deg = 0, max_neg = 0;
            for (uint64_t v = 0; v < N; v++) {
                if (dist[v] == -1) zero_deg++;
                else { pos_deg++; max_neg = std::min(max_neg, dist[v]); }
            }
            BREAKPOINT_DUMP("BFS_INIT", "N=%lu zero_degree=%ld has_edges=%ld max_degree=%ld",
                            N, zero_deg, pos_deg, -max_neg);
        }

        dist[source] = 0;

        // 简单的frontier-based BFS (保留upstream的TD逻辑框架)
        std::vector<uint64_t> frontier = {source};
        std::vector<uint64_t> next_frontier;
        int64_t level = 1;
        uint64_t total_visited = 1;
        uint64_t scout_count = graph_.degree(source);
        int64_t edges_to_check = graph_.edge_count();

        // [NEW] 重置tier命中计数
        g_tier_counters.reset();

        while (!frontier.empty()) {
            // [KEEP] TD step 逻辑 (保留upstream的CAS语义, 单线程简化)
            next_frontier.clear();
            uint64_t level_scout = 0;

            for (uint64_t u : frontier) {
                graph_.edges(u, [&](uint64_t dst, double w) {
                    if (dist[dst] < 0) {
                        int64_t old = dist[dst];
                        dist[dst] = level;  // CAS in upstream, 单线程直接赋值
                        next_frontier.push_back(dst);
                        level_scout += (-old);
                    }
                });
            }

            total_visited += next_frontier.size();

            // [NEW] 每层断点: 打印frontier状态 + TD↔BU决策参量
            bool would_switch = (scout_count > edges_to_check / alpha_);
            BREAKPOINT_DUMP("BFS_LEVEL",
                "level=%ld frontier=%lu discovered=%lu visited=%lu/%lu(%.1f%%) "
                "scout=%lu threshold=%ld would_BU=%s",
                level, frontier.size(), next_frontier.size(),
                total_visited, N, 100.0*total_visited/N,
                level_scout, edges_to_check/alpha_,
                would_switch ? "YES" : "no");

            edges_to_check -= scout_count;
            scout_count = level_scout;
            frontier.swap(next_frontier);
            level++;
        }

        // [NEW] 最终断点: 距离分布直方图
        {
            std::vector<uint64_t> level_count(level + 1, 0);
            uint64_t unreachable = 0;
            for (uint64_t v = 0; v < N; v++) {
                if (dist[v] < 0) unreachable++;
                else if (dist[v] < (int64_t)level_count.size()) level_count[dist[v]]++;
            }
            std::printf("[BFS·RESULT] levels=%ld total_visited=%lu unreachable=%lu\n",
                        level-1, total_visited, unreachable);
            if (g_debug_level >= 2) {
                std::printf("[BFS·HISTOGRAM] ");
                for (int64_t l = 0; l < std::min(level, (int64_t)20); l++)
                    std::printf("L%ld=%lu ", l, level_count[l]);
                if (level > 20) std::printf("...");
                std::printf("\n");
            }
        }

        // [NEW] tier 命中统计
        g_tier_counters.dump("BFS_COMPLETE");
        return dist;
    }
};


// ═══════════════════════════════════════════════════════════════════
// §4  PageRank (mv from upstream/rapidstore/algorithms/pageRank.cpp, 159行)
// ═══════════════════════════════════════════════════════════════════
//
// [KEEP] 双阶段: (1)计算outgoing_contrib (2)累加incoming → 更新score
// [KEEP] dangling node 处理 100%保留
// [KEEP] base_score + damping * (incoming + dangling_sum) 公式
// [MOD]  namespace: driver::algorithm → philemon::experiment
// [MOD]  m_snapshot→edges → FakeGraph.edges
// [NEW]  每轮iter打印: delta收敛值、top-5 score、dangling比例
// [NEW]  tier命中追踪: 哪些tier的边贡献了多少PageRank值

class TieredPageRank {
    const FakeGraph& graph_;
    int num_threads_;
    uint64_t num_iterations_;
    double damping_;
public:
    TieredPageRank(const FakeGraph& g, int threads, uint64_t iters = 10, double damp = 0.85)
        : graph_(g), num_threads_(threads), num_iterations_(iters), damping_(damp) {}

    std::vector<double> run() {
        ScopedTimer timer("PAGERANK");
        const uint64_t N = graph_.vertex_count();
        const double init_score = 1.0 / N;
        const double base_score = (1.0 - damping_) / N;

        std::vector<double> scores(N, init_score);
        std::vector<double> outgoing_contrib(N, 0.0);

        g_tier_counters.reset();

        for (uint64_t iter = 0; iter < num_iterations_; iter++) {
            double dangling_sum = 0.0;
            uint64_t dangling_count = 0;

            // Phase 1: compute outgoing contributions (保留upstream逻辑)
            for (uint64_t v = 0; v < N; v++) {
                uint64_t out_deg = graph_.degree(v);
                if (out_deg == 0) {
                    dangling_sum += scores[v];
                    dangling_count++;
                } else {
                    outgoing_contrib[v] = scores[v] / out_deg;
                }
            }
            dangling_sum /= N;

            // [NEW] delta 追踪 (本轮 vs 上轮的差异)
            double max_delta = 0.0;
            double sum_delta = 0.0;

            // Phase 2: accumulate incoming + update (保留upstream逻辑)
            for (uint64_t v = 0; v < N; v++) {
                double incoming_total = 0.0;  // [FIX] upstream有typo "incoming_totol"
                graph_.edges(v, [&](uint64_t src, double w) {
                    if (src < N) incoming_total += outgoing_contrib[src];
                });

                double new_score = base_score + damping_ * (incoming_total + dangling_sum);
                double delta = std::abs(new_score - scores[v]);
                max_delta = std::max(max_delta, delta);
                sum_delta += delta;
                scores[v] = new_score;
            }

            // [NEW] 每轮断点: 收敛信息
            BREAKPOINT_DUMP("PR_ITER",
                "iter=%lu/%lu dangling=%lu/%lu(%.1f%%) max_delta=%.2e avg_delta=%.2e",
                iter+1, num_iterations_, dangling_count, N,
                100.0*dangling_count/N, max_delta, sum_delta/N);

            // [NEW] 每轮打印 top-5 scores
            if (g_debug_level >= 2) {
                std::vector<std::pair<double,uint64_t>> top;
                for (uint64_t v = 0; v < N; v++) top.emplace_back(scores[v], v);
                std::partial_sort(top.begin(), top.begin()+std::min(5UL,N), top.end(),
                    [](auto& a, auto& b){ return a.first > b.first; });
                std::printf("  [PR·TOP5] ");
                for (int i = 0; i < std::min(5, (int)N); i++)
                    std::printf("v%lu=%.6f ", top[i].second, top[i].first);
                std::printf("\n");
            }

            // [NEW] 早停: 如果已收敛
            if (max_delta < 1e-8) {
                std::printf("[PR·CONVERGED] Early stop at iter %lu (max_delta=%.2e)\n",
                            iter+1, max_delta);
                break;
            }
        }

        g_tier_counters.dump("PR_COMPLETE");

        // [NEW] 最终统计
        double sum_score = 0;
        for (auto s : scores) sum_score += s;
        std::printf("[PR·RESULT] sum_scores=%.6f (should be ≈1.0)\n", sum_score);

        return scores;
    }
};


// ═══════════════════════════════════════════════════════════════════
// §5  SSSP (mv from upstream/rapidstore/algorithms/SSSP.cpp, 175行)
// ═══════════════════════════════════════════════════════════════════
//
// [KEEP] Delta-stepping 框架 (frontier + bin结构)
// [KEEP] CAS距离更新逻辑
// [MOD]  简化bin管理 (upstream用shared_indexes切换, 这里用priority_queue)
// [MOD]  namespace变更
// [NEW]  每轮打印: 当前bin分布、frontier大小、最短距离范围
// [NEW]  结束时打印: 可达/不可达统计、距离分布直方图

class TieredSSSP {
    const FakeGraph& graph_;
    int num_threads_;
    double delta_;
public:
    TieredSSSP(const FakeGraph& g, int threads, double delta = 2.0)
        : graph_(g), num_threads_(threads), delta_(delta) {}

    std::vector<double> run(uint64_t source) {
        ScopedTimer timer("SSSP");
        const uint64_t N = graph_.vertex_count();
        const double INF = std::numeric_limits<double>::infinity();

        std::vector<double> dist(N, INF);
        dist[source] = 0.0;

        g_tier_counters.reset();

        // Dijkstra-style relaxation (保留upstream的delta-stepping精神)
        using PQItem = std::pair<double, uint64_t>;  // (dist, vertex)
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        pq.push({0.0, source});

        uint64_t relaxations = 0;
        uint64_t iter = 0;
        uint64_t vertices_settled = 0;

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();

            if (d > dist[u]) continue;  // stale entry
            vertices_settled++;

            graph_.edges(u, [&](uint64_t v, double w) {
                double new_d = dist[u] + w;
                if (new_d < dist[v]) {
                    dist[v] = new_d;
                    pq.push({new_d, v});
                    relaxations++;
                }
            });

            iter++;
            // [NEW] 每 N/10 次迭代打印断点
            if (g_debug_level >= 2 && iter % std::max(1UL, N/10) == 0) {
                uint64_t reachable = 0;
                double max_finite = 0;
                for (uint64_t v = 0; v < N; v++) {
                    if (dist[v] < INF) { reachable++; max_finite = std::max(max_finite, dist[v]); }
                }
                BREAKPOINT_DUMP("SSSP_PROGRESS",
                    "settled=%lu pq_size=%lu relaxations=%lu reachable=%lu/%lu max_dist=%.2f",
                    vertices_settled, pq.size(), relaxations, reachable, N, max_finite);
            }
        }

        // [NEW] 最终断点: 距离分布
        {
            uint64_t reachable = 0, unreachable = 0;
            double max_d = 0, sum_d = 0;
            for (uint64_t v = 0; v < N; v++) {
                if (dist[v] < INF) { reachable++; max_d = std::max(max_d, dist[v]); sum_d += dist[v]; }
                else unreachable++;
            }
            std::printf("[SSSP·RESULT] source=%lu reachable=%lu unreachable=%lu "
                        "max_dist=%.2f avg_dist=%.2f relaxations=%lu\n",
                        source, reachable, unreachable, max_d,
                        reachable > 0 ? sum_d/reachable : 0.0, relaxations);

            // [NEW] 距离分桶直方图
            if (g_debug_level >= 2 && reachable > 0) {
                const int NBINS = 10;
                std::vector<uint64_t> bins(NBINS, 0);
                double bin_w = (max_d + 1) / NBINS;
                for (uint64_t v = 0; v < N; v++) {
                    if (dist[v] < INF) {
                        int b = std::min((int)(dist[v] / bin_w), NBINS-1);
                        bins[b]++;
                    }
                }
                std::printf("[SSSP·DIST_BINS] ");
                for (int b = 0; b < NBINS; b++)
                    std::printf("[%.1f-%.1f]=%lu ", b*bin_w, (b+1)*bin_w, bins[b]);
                std::printf("\n");
            }
        }

        g_tier_counters.dump("SSSP_COMPLETE");
        return dist;
    }
};


// ═══════════════════════════════════════════════════════════════════
// §6  WCC (mv from upstream/rapidstore/algorithms/WCC.cpp, 137行)
// ═══════════════════════════════════════════════════════════════════
//
// [KEEP] label propagation + path compression 100%保留
// [KEEP] comp[high] = low 的合并方向
// [MOD]  namespace变更, 单线程化 (upstream用多线程)
// [NEW]  每轮打印: 变更数、最大组件大小、组件数量
// [NEW]  最终打印: 组件大小分布直方图

class TieredWCC {
    const FakeGraph& graph_;
    int num_threads_;
public:
    TieredWCC(const FakeGraph& g, int threads)
        : graph_(g), num_threads_(threads) {}

    std::vector<uint64_t> run() {
        ScopedTimer timer("WCC");
        const uint64_t N = graph_.vertex_count();
        std::vector<uint64_t> comp(N);
        for (uint64_t i = 0; i < N; i++) comp[i] = i;

        g_tier_counters.reset();

        bool change = true;
        uint64_t round = 0;
        while (change) {
            change = false;
            uint64_t merges_this_round = 0;

            // [KEEP] upstream WCC 核心: 遍历边, comp[high]=low
            for (uint64_t u = 0; u < N; u++) {
                graph_.edges(u, [&](uint64_t v, double w) {
                    uint64_t cu = comp[u], cv = comp[v];
                    if (cu != cv) {
                        uint64_t hi = std::max(cu, cv);
                        uint64_t lo = std::min(cu, cv);
                        if (hi < N && comp[hi] == hi) {
                            comp[hi] = lo;
                            change = true;
                            merges_this_round++;
                        }
                    }
                });
            }

            // [KEEP] path compression
            for (uint64_t i = 0; i < N; i++) {
                while (comp[i] != comp[comp[i]]) comp[i] = comp[comp[i]];
            }

            round++;

            // [NEW] 每轮断点
            uint64_t num_components = 0;
            uint64_t max_comp_size = 0;
            std::unordered_map<uint64_t, uint64_t> comp_sizes;
            for (uint64_t i = 0; i < N; i++) comp_sizes[comp[i]]++;
            num_components = comp_sizes.size();
            for (auto& [c, sz] : comp_sizes) max_comp_size = std::max(max_comp_size, sz);

            BREAKPOINT_DUMP("WCC_ROUND",
                "round=%lu merges=%lu components=%lu max_size=%lu change=%s",
                round, merges_this_round, num_components, max_comp_size,
                change ? "yes" : "DONE");
        }

        // [NEW] 最终组件分布
        {
            std::unordered_map<uint64_t, uint64_t> comp_sizes;
            for (uint64_t i = 0; i < N; i++) comp_sizes[comp[i]]++;

            uint64_t sz1=0, sz2_10=0, sz11_100=0, sz_big=0;
            for (auto& [c, sz] : comp_sizes) {
                if (sz == 1) sz1++;
                else if (sz <= 10) sz2_10++;
                else if (sz <= 100) sz11_100++;
                else sz_big++;
            }
            std::printf("[WCC·RESULT] rounds=%lu components=%lu | "
                        "singletons=%lu sz2-10=%lu sz11-100=%lu sz100+=%lu\n",
                        round, comp_sizes.size(), sz1, sz2_10, sz11_100, sz_big);
        }

        g_tier_counters.dump("WCC_COMPLETE");
        return comp;
    }
};


// ═══════════════════════════════════════════════════════════════════
// §7  TriangleCounting (mv from upstream/rapidstore/wrapper/algorithms/TC.h)
// ═══════════════════════════════════════════════════════════════════
//
// [KEEP] 基于邻居集合交集的三角形计数
// [MOD]  简化为 O(V*D^2) brute-force (upstream版也类似)
// [NEW]  进度打印 + 中间三角形累计

class TieredTC {
    const FakeGraph& graph_;
public:
    explicit TieredTC(const FakeGraph& g) : graph_(g) {}

    uint64_t run() {
        ScopedTimer timer("TC");
        const uint64_t N = graph_.vertex_count();
        uint64_t count = 0;

        g_tier_counters.reset();

        // 为每个顶点缓存邻居集合
        std::vector<std::vector<uint64_t>> neighbors(N);
        for (uint64_t v = 0; v < N; v++) {
            graph_.edges(v, [&](uint64_t dst, double w) {
                if (dst > v) neighbors[v].push_back(dst);  // 只看 v < dst, 避免重复
            });
            std::sort(neighbors[v].begin(), neighbors[v].end());
        }

        for (uint64_t u = 0; u < N; u++) {
            for (uint64_t v : neighbors[u]) {
                // 交集计数
                auto it_u = neighbors[u].begin(), end_u = neighbors[u].end();
                auto it_v = neighbors[v].begin(), end_v = neighbors[v].end();
                while (it_u != end_u && it_v != end_v) {
                    if (*it_u < *it_v) ++it_u;
                    else if (*it_v < *it_u) ++it_v;
                    else { count++; ++it_u; ++it_v; }
                }
            }
            PROGRESS_PRINT("TC", (unsigned long)u, (unsigned long)N);
        }

        std::printf("[TC·RESULT] triangles=%lu\n", count);
        g_tier_counters.dump("TC_COMPLETE");
        return count;
    }
};


// ═══════════════════════════════════════════════════════════════════
// §8  配置引擎 (mv from upstream commandLineParser, 700行 → 简化)
// ═══════════════════════════════════════════════════════════════════

struct ExperimentConfig {
    // graph
    uint64_t num_vertices   = 2000;
    uint64_t num_edges      = 10000;
    uint64_t seed           = 42;
    std::string edge_file   = "";
    std::string vertex_file = "";

    // algo params (保留upstream config.cfg的全部参数名)
    int num_threads     = 1;
    int alpha           = 15;
    int beta            = 18;
    uint64_t bfs_source = 0;
    double delta        = 2.0;
    uint64_t sssp_source= 0;
    uint64_t num_iterations = 10;
    double damping_factor   = 0.85;
    int debug_level     = 2;

    // workload
    bool run_bfs  = true;
    bool run_pr   = true;
    bool run_sssp = true;
    bool run_wcc  = true;
    bool run_tc   = false;  // TC 在大图上很慢, 默认关闭

    // thread sweep
    std::vector<int> thread_counts = {1, 2, 4};

    void parse_file(const std::string& path) {
        std::ifstream fin(path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[CONFIG] Cannot open %s, using defaults\n", path.c_str());
            return;
        }
        std::printf("[CONFIG] Parsing %s\n", path.c_str());
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            // trim
            auto trim = [](std::string& s) {
                while (!s.empty() && (s.front()==' '||s.front()=='\t')) s.erase(s.begin());
                while (!s.empty() && (s.back()==' '||s.back()=='\t')) s.pop_back();
            };
            trim(key); trim(val);

            if (key == "num_vertices") num_vertices = std::stoull(val);
            else if (key == "num_edges") num_edges = std::stoull(val);
            else if (key == "num_threads") num_threads = std::stoi(val);
            else if (key == "alpha") alpha = std::stoi(val);
            else if (key == "beta") beta = std::stoi(val);
            else if (key == "bfs_source") bfs_source = std::stoull(val);
            else if (key == "delta") delta = std::stod(val);
            else if (key == "sssp_source") sssp_source = std::stoull(val);
            else if (key == "num_iterations") num_iterations = std::stoull(val);
            else if (key == "damping_factor") damping_factor = std::stod(val);
            else if (key == "debug_level") debug_level = std::stoi(val);
            else if (key == "seed") seed = std::stoull(val);
            else if (key == "edge_file") edge_file = val;
            else if (key == "vertex_file") vertex_file = val;
            else if (key == "run_tc") run_tc = (val == "1" || val == "true");
        }
        std::printf("[CONFIG] Parsed: V=%lu E=%lu threads=%d alpha=%d beta=%d\n",
                    num_vertices, num_edges, num_threads, alpha, beta);
    }

    void dump() const {
        print_separator("EXPERIMENT CONFIGURATION");
        std::printf("  num_vertices    = %lu\n", num_vertices);
        std::printf("  num_edges       = %lu\n", num_edges);
        std::printf("  seed            = %lu\n", seed);
        std::printf("  num_threads     = %d\n", num_threads);
        std::printf("  alpha (BFS)     = %d\n", alpha);
        std::printf("  beta  (BFS)     = %d\n", beta);
        std::printf("  bfs_source      = %lu\n", bfs_source);
        std::printf("  delta (SSSP)    = %.2f\n", delta);
        std::printf("  sssp_source     = %lu\n", sssp_source);
        std::printf("  num_iterations  = %lu\n", num_iterations);
        std::printf("  damping_factor  = %.4f\n", damping_factor);
        std::printf("  debug_level     = %d\n", debug_level);
        std::printf("  edge_file       = %s\n", edge_file.empty() ? "(synthetic)" : edge_file.c_str());
        std::printf("  algorithms      = %s%s%s%s%s\n",
                    run_bfs?"BFS ":"", run_pr?"PR ":"", run_sssp?"SSSP ":"",
                    run_wcc?"WCC ":"", run_tc?"TC":"");
        std::printf("  thread_sweep    = {");
        for (size_t i = 0; i < thread_counts.size(); i++)
            std::printf("%d%s", thread_counts[i], i+1<thread_counts.size()?",":"");
        std::printf("}\n");
    }
};


// ═══════════════════════════════════════════════════════════════════
// §9  JSON 结果导出
// ═══════════════════════════════════════════════════════════════════

static void export_json_results(const std::string& path,
                                 const ExperimentConfig& cfg,
                                 const std::unordered_map<std::string, double>& timings) {
    std::ofstream fout(path);
    fout << "{\n";
    fout << "  \"config\": {\n";
    fout << "    \"num_vertices\": " << cfg.num_vertices << ",\n";
    fout << "    \"num_edges\": " << cfg.num_edges << ",\n";
    fout << "    \"seed\": " << cfg.seed << "\n";
    fout << "  },\n";
    fout << "  \"timings_ms\": {\n";
    size_t i = 0;
    for (auto& [k,v] : timings) {
        fout << "    \"" << k << "\": " << v;
        if (++i < timings.size()) fout << ",";
        fout << "\n";
    }
    fout << "  },\n";
    fout << "  \"memory_rss_kb\": " << get_rss_kb() << "\n";
    fout << "}\n";
    fout.close();
    std::printf("[EXPORT] Results → %s\n", path.c_str());
}

}} // namespace walking::experiment


// ═══════════════════════════════════════════════════════════════════
// MAIN — 实验入口 (mv from upstream/rapidstore/main.cpp, 202行)
// ═══════════════════════════════════════════════════════════════════
//
// [KEEP] main.cpp 的整体流程: config→load vertex→load edge→run algorithms
// [MOD]  teseo_driver → FakeGraph
// [MOD]  commandLineParser → ExperimentConfig
// [MOD]  thread sweep 从 {1,2,4,8,16,32,40} → config中读取
// [NEW]  开机banner、系统信息、每阶段断点
// [NEW]  JSON结果导出

int main(int argc, char** argv) {
    using namespace walking::experiment;

    // ─── Banner ─────────────────────────────────────────────────────
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║  LLM4Walking — Graph Walking Experiment Runner            ║\n");
    std::printf("║  Tiered Heterogeneous Memory Graph Analytics             ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    // ─── System info ────────────────────────────────────────────────
    std::printf("[SYSTEM] PID=%d  RSS=%ld KB\n", getpid(), get_rss_kb());
    std::printf("[SYSTEM] Hardware concurrency: %u\n", std::thread::hardware_concurrency());

    // ─── Config ─────────────────────────────────────────────────────
    ExperimentConfig cfg;
    if (argc > 1) {
        cfg.parse_file(argv[1]);
    } else {
        std::printf("[CONFIG] No config file, using defaults (synthetic graph)\n");
    }
    // CLI overrides
    if (argc > 2) cfg.num_vertices = std::stoull(argv[2]);
    if (argc > 3) cfg.num_edges = std::stoull(argv[3]);
    if (argc > 4) cfg.debug_level = std::stoi(argv[4]);

    g_debug_level = cfg.debug_level;
    cfg.dump();

    // ─── Graph construction ─────────────────────────────────────────
    print_separator("GRAPH CONSTRUCTION");

    FakeGraph graph;
    graph.init(cfg.num_vertices);

    TieredEdgeStream edge_stream;

    // Vertex loading (保留upstream的vertex_reader流程)
    {
        ScopedTimer t("VERTEX_LOAD");
        if (!cfg.vertex_file.empty()) {
            std::ifstream fin(cfg.vertex_file);
            uint64_t vid;
            while (fin >> vid) {
                // FakeGraph 初始化时已分配空间, 这里只是确认
            }
            std::printf("[VERTEX] Loaded vertices from %s\n", cfg.vertex_file.c_str());
        } else {
            std::printf("[VERTEX] Using %lu synthetic vertices (0..%lu)\n",
                        cfg.num_vertices, cfg.num_vertices - 1);
        }
        BREAKPOINT_DUMP("POST_VERTEX", "V=%lu RSS=%ld KB", cfg.num_vertices, get_rss_kb());
    }

    // Edge loading (保留upstream的multi-thread insert流程, 简化为单线程)
    {
        ScopedTimer t("EDGE_LOAD");
        if (!cfg.edge_file.empty()) {
            edge_stream.load_from_file(cfg.edge_file);
        } else {
            edge_stream.generate_synthetic(cfg.num_vertices, cfg.num_edges, cfg.seed);
        }

        // Batch insert (upstream用40线程+chunk, 这里保留chunk模式但单线程)
        uint64_t batch = 100000;
        for (uint64_t i = 0; i < edge_stream.size(); i += batch) {
            uint64_t end = std::min(i + batch, edge_stream.size());
            graph.batch_insert(edge_stream, i, end);
            PROGRESS_PRINT("EDGE_INSERT", (unsigned long)end, (unsigned long)edge_stream.size());
        }

        BREAKPOINT_DUMP("POST_EDGE", "E=%lu (stream=%lu) RSS=%ld KB",
                        graph.edge_count(), edge_stream.size(), get_rss_kb());
    }

    // ─── Pre-algorithm state dump ───────────────────────────────────
    graph.dump_tier_distribution("PRE_ALGORITHM");
    graph.dump_degree_histogram("PRE_ALGORITHM");
    g_tier_counters.dump("POST_LOAD");
    g_tier_counters.reset();

    // ─── Results collection ─────────────────────────────────────────
    std::unordered_map<std::string, double> all_timings;

    // ─── Algorithm sweep (保留upstream main.cpp的thread sweep逻辑) ──
    for (int num_t : cfg.thread_counts) {
        print_separator(("THREAD_COUNT = " + std::to_string(num_t)).c_str());

        // BFS
        if (cfg.run_bfs) {
            auto t0 = std::chrono::high_resolution_clock::now();
            TieredBFS bfs(graph, num_t, cfg.alpha, cfg.beta);
            auto bfs_dist = bfs.run(cfg.bfs_source);
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            all_timings["BFS_T" + std::to_string(num_t)] = dt;
        }

        // SSSP
        if (cfg.run_sssp) {
            auto t0 = std::chrono::high_resolution_clock::now();
            TieredSSSP sssp(graph, num_t, cfg.delta);
            auto sssp_dist = sssp.run(cfg.sssp_source);
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            all_timings["SSSP_T" + std::to_string(num_t)] = dt;
        }

        // PageRank
        if (cfg.run_pr) {
            auto t0 = std::chrono::high_resolution_clock::now();
            TieredPageRank pr(graph, num_t, cfg.num_iterations, cfg.damping_factor);
            auto scores = pr.run();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            all_timings["PR_T" + std::to_string(num_t)] = dt;
        }

        // WCC
        if (cfg.run_wcc) {
            auto t0 = std::chrono::high_resolution_clock::now();
            TieredWCC wcc(graph, num_t);
            auto components = wcc.run();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            all_timings["WCC_T" + std::to_string(num_t)] = dt;
        }

        // TC
        if (cfg.run_tc) {
            auto t0 = std::chrono::high_resolution_clock::now();
            TieredTC tc(graph);
            auto tri_count = tc.run();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
            all_timings["TC_T" + std::to_string(num_t)] = dt;
        }

        // [NEW] 每个thread sweep后的 tier 状态
        g_tier_counters.dump(("SWEEP_T" + std::to_string(num_t)).c_str());
        g_tier_counters.reset();
    }

    // ─── Final summary ──────────────────────────────────────────────
    print_separator("EXPERIMENT SUMMARY");

    std::printf("%-20s %10s\n", "Algorithm", "Time(ms)");
    std::printf("%-20s %10s\n", "--------------------", "----------");
    for (auto& [name, ms] : all_timings)
        std::printf("%-20s %10.0f\n", name.c_str(), ms);

    std::printf("\n[FINAL] Total RSS: %ld KB\n", get_rss_kb());

    // ─── Export JSON ────────────────────────────────────────────────
    {
        // 从 argv[0] 推导出脚本所在目录
        std::string exe_path = argv[0];
        std::string dir = ".";
        auto slash = exe_path.rfind('/');
        if (slash != std::string::npos) dir = exe_path.substr(0, slash);
        std::string json_path = dir + "/results/run_result.json";
        export_json_results(json_path, cfg, all_timings);
    }

    // ─── Shutdown banner ────────────────────────────────────────────
    std::printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║  Experiment complete.                                     ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n");

    return 0;
}
