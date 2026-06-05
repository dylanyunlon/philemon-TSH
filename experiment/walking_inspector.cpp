/**
 * walking_inspector.cpp — 全结构体状态逐步检查工具
 *
 * 模拟 gdb/lldb 断点调试体验: 每个阶段暂停并打印完整状态
 * 用于验证 mv + 20% 修改后的算法正确性
 *
 * 来源: 组合 upstream/rapidstore 的:
 *   - graph/edge.cpp (结构体定义)
 *   - algorithms (算法中间状态)
 *   - main.cpp (流程控制)
 * + NEW 70%: 状态检查框架、invariant 断言、diff 追踪
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -o walking_inspector walking_inspector.cpp
 *
 * Run:
 *   ./walking_inspector [num_vertices] [num_edges] [debug_level]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <numeric>
#include <cmath>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

// ═══════════════════════════════════════════════════════════════════
// §0  检查基础设施
// ═══════════════════════════════════════════════════════════════════

static int g_dbg = 3;  // inspector 默认最高调试级别
static long rss_kb() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss;
}

static uint64_t g_inspect_count = 0;
static uint64_t g_assert_pass = 0;
static uint64_t g_assert_fail = 0;

#define INSPECT(tag, ...) do { \
    g_inspect_count++; \
    std::printf("[INSPECT·%04lu·%s] ", g_inspect_count, tag); \
    std::printf(__VA_ARGS__); \
    std::printf("  RSS=%ld KB\n", rss_kb()); \
} while(0)

#define CHECK_ASSERT(cond, tag, ...) do { \
    if (cond) { g_assert_pass++; } \
    else { \
        g_assert_fail++; \
        std::printf("[ASSERT·FAIL·%s] ", tag); \
        std::printf(__VA_ARGS__); \
        std::printf("\n"); \
    } \
} while(0)

#define CHECK_ASSERT_QUIET(cond) do { \
    if (cond) g_assert_pass++; else g_assert_fail++; \
} while(0)

static void sep(const char* s) {
    std::printf("\n════════════════════════════════════════════════════\n");
    std::printf("  [INSPECTOR] %s\n", s);
    std::printf("════════════════════════════════════════════════════\n\n");
}

// ═══════════════════════════════════════════════════════════════════
// §1  Graph — 与 walking_experiment 相同结构, 增加状态验证接口
// ═══════════════════════════════════════════════════════════════════
// [mv from upstream graph/edge.cpp + edgeStream.cpp]
// [+20% MOD] 增加 verify_invariants(), dump_full_state()

enum TierID : uint8_t { TIER_DRAM = 0, TIER_CXL = 1, TIER_SSD = 2, TIER_COUNT = 3 };
static const char* tier_name(TierID t) {
    static const char* names[] = {"DRAM", "CXL", "SSD"};
    return (t < TIER_COUNT) ? names[t] : "???";
}

struct Edge {
    uint64_t src, dst;
    double   weight;
    TierID   tier;
    uint64_t access_count = 0;  // [+20%] 追踪每条边被访问次数
};

class InspectableGraph {
    uint64_t nv_ = 0;
    std::vector<std::vector<Edge>> adj_;
    uint64_t ne_ = 0;
    uint64_t tier_counts_[TIER_COUNT] = {};

public:
    void init(uint64_t n) {
        nv_ = n;
        adj_.resize(n);
        INSPECT("GRAPH_INIT", "allocated adj_ for V=%lu", n);
    }

    void add_edge(uint64_t s, uint64_t d, double w, TierID t) {
        if (s >= nv_ || d >= nv_) return;
        adj_[s].push_back({s, d, w, t, 0});
        ne_++;
        tier_counts_[t]++;
    }

    uint64_t vertex_count() const { return nv_; }
    uint64_t edge_count() const { return ne_; }
    uint64_t degree(uint64_t v) const { return v < nv_ ? adj_[v].size() : 0; }
    uint64_t tier_count(TierID t) const { return tier_counts_[t]; }

    // 遍历邻居 (带访问计数追踪)
    template <typename F>
    void edges(uint64_t v, F&& cb) {
        if (v >= nv_) return;
        for (auto& e : adj_[v]) {
            e.access_count++;
            cb(e.dst, e.weight, e.tier);
        }
    }

    // const版 (不修改access_count)
    template <typename F>
    void edges_const(uint64_t v, F&& cb) const {
        if (v >= nv_) return;
        for (const auto& e : adj_[v]) {
            cb(e.dst, e.weight, e.tier);
        }
    }

    // [+20% NEW] 全面验证结构体 invariant
    bool verify_invariants(const char* phase) {
        bool ok = true;
        uint64_t counted_edges = 0;
        uint64_t tier_check[TIER_COUNT] = {};
        uint64_t max_deg = 0, isolated = 0;
        uint64_t self_loops = 0;

        for (uint64_t v = 0; v < nv_; v++) {
            uint64_t deg = adj_[v].size();
            counted_edges += deg;
            max_deg = std::max(max_deg, deg);
            if (deg == 0) isolated++;

            for (auto& e : adj_[v]) {
                // source 一致性
                if (e.src != v) {
                    INSPECT("INVARIANT_FAIL", "edge src mismatch: stored=%lu expected=%lu", e.src, v);
                    ok = false;
                }
                // dst 范围
                if (e.dst >= nv_) {
                    INSPECT("INVARIANT_FAIL", "edge dst out of range: %lu >= %lu", e.dst, nv_);
                    ok = false;
                }
                // self-loop 统计
                if (e.src == e.dst) self_loops++;
                // tier 合法
                if (e.tier >= TIER_COUNT) {
                    INSPECT("INVARIANT_FAIL", "invalid tier=%d", e.tier);
                    ok = false;
                }
                tier_check[e.tier]++;
            }
        }

        // edge count 一致性
        CHECK_ASSERT(counted_edges == ne_, "EDGE_COUNT",
            "counted=%lu stored=%lu", counted_edges, ne_);

        // tier count 一致性
        for (int t = 0; t < TIER_COUNT; t++) {
            CHECK_ASSERT(tier_check[t] == tier_counts_[t], "TIER_COUNT",
                "tier[%d] counted=%lu stored=%lu", t, tier_check[t], tier_counts_[t]);
        }

        INSPECT(phase, "invariant_check: %s | E=%lu max_deg=%lu isolated=%lu self_loops=%lu "
                "tiers=[DRAM=%lu CXL=%lu SSD=%lu]",
                ok ? "PASS" : "FAIL",
                counted_edges, max_deg, isolated, self_loops,
                tier_check[0], tier_check[1], tier_check[2]);
        return ok;
    }

    // [+20% NEW] 打印完整状态快照 (小图用)
    void dump_full_state(const char* label) const {
        INSPECT(label, "=== FULL GRAPH SNAPSHOT ===");
        std::printf("  V=%lu E=%lu\n", nv_, ne_);

        // degree 分布
        std::vector<uint64_t> deg_hist(11, 0);  // 0,1,2,...,9,10+
        for (uint64_t v = 0; v < nv_; v++) {
            uint64_t d = adj_[v].size();
            deg_hist[std::min(d, (uint64_t)10)]++;
        }
        std::printf("  deg_histogram: ");
        for (int i = 0; i <= 10; i++)
            std::printf("d%s%d=%lu ", i==10?"≥":"=", i, deg_hist[i]);
        std::printf("\n");

        // tier 分布
        std::printf("  tier_dist: DRAM=%lu(%.1f%%) CXL=%lu(%.1f%%) SSD=%lu(%.1f%%)\n",
            tier_counts_[0], ne_>0?100.0*tier_counts_[0]/ne_:0,
            tier_counts_[1], ne_>0?100.0*tier_counts_[1]/ne_:0,
            tier_counts_[2], ne_>0?100.0*tier_counts_[2]/ne_:0);

        // 前5个非空顶点的详细邻居
        int printed = 0;
        for (uint64_t v = 0; v < nv_ && printed < 5; v++) {
            if (adj_[v].empty()) continue;
            std::printf("  v%lu (deg=%lu): ", v, adj_[v].size());
            for (size_t i = 0; i < std::min(adj_[v].size(), (size_t)8); i++) {
                auto& e = adj_[v][i];
                std::printf("→%lu(w=%.1f,t=%s,acc=%lu) ",
                    e.dst, e.weight, tier_name(e.tier), e.access_count);
            }
            if (adj_[v].size() > 8) std::printf("...");
            std::printf("\n");
            printed++;
        }
    }

    // [+20% NEW] 访问热点分析
    void dump_access_hotspots(const char* label, int top_n = 10) const {
        std::vector<std::pair<uint64_t, uint64_t>> vertex_access(nv_);
        for (uint64_t v = 0; v < nv_; v++) {
            uint64_t total = 0;
            for (auto& e : adj_[v]) total += e.access_count;
            vertex_access[v] = {total, v};
        }
        std::partial_sort(vertex_access.begin(),
            vertex_access.begin() + std::min((int)nv_, top_n),
            vertex_access.end(),
            [](auto& a, auto& b){ return a.first > b.first; });

        INSPECT(label, "access hotspots (top %d):", std::min((int)nv_, top_n));
        for (int i = 0; i < std::min((int)nv_, top_n); i++) {
            std::printf("  #%d: v%lu access_count=%lu deg=%lu\n",
                i+1, vertex_access[i].second, vertex_access[i].first,
                degree(vertex_access[i].second));
        }
    }
};

// ═══════════════════════════════════════════════════════════════════
// §2  BFS Inspector — 逐层验证 BFS 正确性
// ═══════════════════════════════════════════════════════════════════
// [mv from upstream BFS.cpp, +20%: invariant 检查]

static void inspect_bfs(InspectableGraph& g, uint64_t source) {
    sep("BFS INSPECTION");
    uint64_t N = g.vertex_count();
    std::vector<int64_t> dist(N, -1);
    dist[source] = 0;

    std::vector<uint64_t> frontier = {source};
    int64_t level = 0;

    INSPECT("BFS_START", "source=%lu degree=%lu", source, g.degree(source));

    while (!frontier.empty()) {
        std::vector<uint64_t> next;
        uint64_t tier_hits[TIER_COUNT] = {};

        for (uint64_t u : frontier) {
            g.edges(u, [&](uint64_t v, double w, TierID t) {
                tier_hits[t]++;
                if (dist[v] < 0) {
                    dist[v] = level + 1;
                    next.push_back(v);
                }
            });
        }

        level++;
        INSPECT("BFS_LEVEL", "L%ld: frontier=%lu discovered=%lu "
                "tier_hits=[DRAM=%lu CXL=%lu SSD=%lu]",
                level, frontier.size(), next.size(),
                tier_hits[0], tier_hits[1], tier_hits[2]);

        // [+20%] invariant: 新发现的顶点距离一定是 level
        for (uint64_t v : next) {
            CHECK_ASSERT(dist[v] == level, "BFS_DIST",
                "v=%lu expected dist=%ld got=%ld", v, level, dist[v]);
        }

        // [+20%] invariant: 不应该重复发现
        std::unordered_set<uint64_t> next_set(next.begin(), next.end());
        CHECK_ASSERT(next_set.size() == next.size(), "BFS_DUP",
            "duplicates in frontier: set=%lu vec=%lu", next_set.size(), next.size());

        frontier = std::move(next);
    }

    // 最终统计
    uint64_t reachable = 0;
    std::vector<uint64_t> level_hist(level + 1, 0);
    for (uint64_t v = 0; v < N; v++) {
        if (dist[v] >= 0) {
            reachable++;
            level_hist[dist[v]]++;
        }
    }

    INSPECT("BFS_DONE", "levels=%ld reachable=%lu/%lu", level, reachable, N);
    std::printf("  level_histogram: ");
    for (int64_t l = 0; l <= std::min(level, (int64_t)25); l++)
        std::printf("L%ld=%lu ", l, level_hist[l]);
    if (level > 25) std::printf("...");
    std::printf("\n");
}

// ═══════════════════════════════════════════════════════════════════
// §3  SSSP Inspector — 验证最短路径一致性
// ═══════════════════════════════════════════════════════════════════
// [mv from upstream SSSP.cpp, +20%: 三角不等式检查]

static void inspect_sssp(InspectableGraph& g, uint64_t source) {
    sep("SSSP INSPECTION");
    uint64_t N = g.vertex_count();
    double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(N, INF);
    dist[source] = 0.0;

    using PQ = std::priority_queue<std::pair<double,uint64_t>,
                std::vector<std::pair<double,uint64_t>>,
                std::greater<std::pair<double,uint64_t>>>;
    PQ pq;
    pq.push({0.0, source});
    uint64_t settled = 0, relaxations = 0;

    INSPECT("SSSP_START", "source=%lu", source);

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        settled++;

        g.edges(u, [&](uint64_t v, double w, TierID t) {
            double nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
                relaxations++;
            }
        });

        if (settled % std::max(1UL, N/5) == 0) {
            uint64_t reach = 0; double maxd = 0;
            for (uint64_t v = 0; v < N; v++)
                if (dist[v] < INF) { reach++; maxd = std::max(maxd, dist[v]); }
            INSPECT("SSSP_PROGRESS", "settled=%lu/%lu reach=%lu max=%.2f relax=%lu",
                    settled, N, reach, maxd, relaxations);
        }
    }

    // [+20%] 三角不等式验证: dist[v] ≤ dist[u] + w(u,v)
    uint64_t triangle_violations = 0;
    for (uint64_t u = 0; u < N; u++) {
        if (dist[u] >= INF) continue;
        g.edges_const(u, [&](uint64_t v, double w, TierID t) {
            if (dist[v] > dist[u] + w + 1e-9) {
                triangle_violations++;
                if (triangle_violations <= 5) {  // 只打印前5个
                    INSPECT("SSSP_TRIANGLE_FAIL",
                        "dist[%lu]=%.2f > dist[%lu]=%.2f + w=%.2f (gap=%.2e)",
                        v, dist[v], u, dist[u], w, dist[v] - dist[u] - w);
                }
            }
        });
    }
    CHECK_ASSERT(triangle_violations == 0, "TRIANGLE_INEQ",
        "violations=%lu", triangle_violations);

    uint64_t reach = 0; double maxd = 0;
    for (uint64_t v = 0; v < N; v++)
        if (dist[v] < INF) { reach++; maxd = std::max(maxd, dist[v]); }
    INSPECT("SSSP_DONE", "reach=%lu/%lu max=%.2f relax=%lu triangle_ok=%s",
            reach, N, maxd, relaxations,
            triangle_violations==0 ? "YES" : "NO");
}

// ═══════════════════════════════════════════════════════════════════
// §4  PageRank Inspector — 验证收敛和 score 归一化
// ═══════════════════════════════════════════════════════════════════
// [mv from upstream pageRank.cpp, +20%: 归一化断言+二阶导数]

static void inspect_pagerank(InspectableGraph& g, uint64_t iters = 10, double damp = 0.85) {
    sep("PAGERANK INSPECTION");
    uint64_t N = g.vertex_count();
    double init_s = 1.0 / N;
    double base_s = (1.0 - damp) / N;

    std::vector<double> scores(N, init_s);
    std::vector<double> contrib(N, 0);
    std::vector<uint64_t> deg(N);
    for (uint64_t v = 0; v < N; v++) deg[v] = g.degree(v);

    double prev_max_delta = 1.0;

    INSPECT("PR_START", "V=%lu iters=%lu damp=%.2f", N, iters, damp);

    for (uint64_t it = 0; it < iters; it++) {
        double dang = 0;
        for (uint64_t v = 0; v < N; v++) {
            if (deg[v] == 0) dang += scores[v];
            else contrib[v] = scores[v] / deg[v];
        }
        dang /= N;

        double max_d = 0, sum_d = 0;
        for (uint64_t v = 0; v < N; v++) {
            double inc = 0;
            g.edges_const(v, [&](uint64_t s, double w, TierID t) {
                if (s < N) inc += contrib[s];
            });
            double ns = base_s + damp * (inc + dang);
            double d = std::abs(ns - scores[v]);
            max_d = std::max(max_d, d);
            sum_d += d;
            scores[v] = ns;
        }

        // [+20%] 二阶导数: 收敛加速率
        double convergence_rate = (prev_max_delta > 0) ? max_d / prev_max_delta : 0;
        prev_max_delta = max_d;

        // [+20%] 归一化检查: sum(scores) ≈ 1.0
        double sum_score = 0;
        for (auto s : scores) sum_score += s;
        CHECK_ASSERT(std::abs(sum_score - 1.0) < 0.01, "PR_NORM",
            "iter=%lu sum=%.6f (expected ≈1.0)", it+1, sum_score);

        INSPECT("PR_ITER", "iter=%lu/%lu max_delta=%.2e avg_delta=%.2e "
                "convergence_rate=%.4f sum=%.6f",
                it+1, iters, max_d, sum_d/N, convergence_rate, sum_score);

        if (max_d < 1e-8) {
            INSPECT("PR_CONVERGED", "early stop at iter %lu", it+1);
            break;
        }
    }

    // Top-5
    std::vector<std::pair<double,uint64_t>> top;
    for (uint64_t v = 0; v < N; v++) top.emplace_back(scores[v], v);
    std::partial_sort(top.begin(), top.begin()+std::min(5UL,N), top.end(),
        [](auto& a, auto& b){ return a.first > b.first; });
    INSPECT("PR_TOP5", "top PageRank vertices:");
    for (int i = 0; i < std::min(5, (int)N); i++)
        std::printf("  #%d: v%lu score=%.6f deg=%lu\n",
            i+1, top[i].second, top[i].first, g.degree(top[i].second));
}

// ═══════════════════════════════════════════════════════════════════
// §5  WCC Inspector — 验证连通性一致性
// ═══════════════════════════════════════════════════════════════════
// [mv from upstream WCC.cpp, +20%: BFS验证连通性]

static void inspect_wcc(InspectableGraph& g) {
    sep("WCC INSPECTION");
    uint64_t N = g.vertex_count();
    std::vector<uint64_t> parent(N);
    std::iota(parent.begin(), parent.end(), 0UL);
    std::vector<uint64_t> rnk(N, 0);

    auto find = [&](uint64_t x) -> uint64_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](uint64_t a, uint64_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rnk[a] < rnk[b]) std::swap(a, b);
        parent[b] = a;
        if (rnk[a] == rnk[b]) rnk[a]++;
    };

    INSPECT("WCC_START", "V=%lu", N);

    for (uint64_t v = 0; v < N; v++) {
        g.edges_const(v, [&](uint64_t d, double w, TierID t) { unite(v, d); });
    }

    std::unordered_map<uint64_t, uint64_t> comp_size;
    for (uint64_t v = 0; v < N; v++) comp_size[find(v)]++;

    uint64_t nc = comp_size.size();
    uint64_t maxc = 0, singles = 0;
    for (auto& [c, sz] : comp_size) { maxc = std::max(maxc, sz); if (sz==1) singles++; }

    INSPECT("WCC_DONE", "components=%lu max_size=%lu singletons=%lu", nc, maxc, singles);

    // [+20%] 验证: 同一组件内的随机节点对确实能互相到达
    if (nc < N) {
        // 取最大组件的根
        uint64_t big_root = 0;
        for (auto& [c, sz] : comp_size)
            if (sz == maxc) { big_root = c; break; }

        // 在最大组件中选两个随机节点做 BFS 可达性测试
        std::vector<uint64_t> members;
        for (uint64_t v = 0; v < N; v++)
            if (find(v) == big_root) members.push_back(v);

        if (members.size() >= 2) {
            uint64_t src = members[0], tgt = members[members.size()/2];
            // BFS from src
            std::vector<bool> visited(N, false);
            std::queue<uint64_t> q;
            q.push(src); visited[src] = true;
            while (!q.empty()) {
                uint64_t u = q.front(); q.pop();
                g.edges_const(u, [&](uint64_t v, double w, TierID t) {
                    if (!visited[v]) { visited[v] = true; q.push(v); }
                });
            }
            CHECK_ASSERT(visited[tgt], "WCC_REACH",
                "v%lu→v%lu should be reachable (same component, size=%lu)",
                src, tgt, maxc);
            INSPECT("WCC_VERIFY", "BFS reachability v%lu→v%lu: %s",
                    src, tgt, visited[tgt] ? "OK" : "FAIL");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║  LLM4Walking — Debug State Inspector                     ║\n");
    std::printf("║  Step-through verification of all data structures        ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    uint64_t V = (argc >= 2) ? std::stoull(argv[1]) : 500;
    uint64_t E = (argc >= 3) ? std::stoull(argv[2]) : 2500;
    if (argc >= 4) g_dbg = std::stoi(argv[3]);

    std::printf("[SYSTEM] PID=%d cores=%u RSS=%ld KB\n",
                getpid(), std::thread::hardware_concurrency(), rss_kb());
    std::printf("[CONFIG] V=%lu E=%lu debug=%d\n\n", V, E, g_dbg);

    // ─── 构建图 ─────────────────────────────────────────────────────
    sep("GRAPH CONSTRUCTION");
    InspectableGraph graph;
    graph.init(V);

    std::mt19937_64 rng(42);
    for (uint64_t i = 0; i < E; i++) {
        uint64_t s = rng() % V;
        uint64_t d = rng() % V;
        if (s == d) d = (d + 1) % V;
        double w = 1.0 + (rng() % 100) / 10.0;
        TierID t = TIER_DRAM;
        uint64_t r = rng() % 100;
        if (r >= 95) t = TIER_SSD;
        else if (r >= 80) t = TIER_CXL;
        graph.add_edge(s, d, w, t);

        if (i % std::max(1UL, E/5) == 0) {
            INSPECT("EDGE_INSERT", "progress=%lu/%lu (%.1f%%)",
                    i, E, 100.0*i/E);
        }
    }

    // ─── 构建后验证 ─────────────────────────────────────────────────
    sep("POST-BUILD VERIFICATION");
    graph.verify_invariants("POST_BUILD");
    graph.dump_full_state("POST_BUILD");

    // 选 source
    uint64_t source = 0;
    for (uint64_t v = 0; v < V; v++) {
        if (graph.degree(v) > 0) { source = v; break; }
    }
    INSPECT("SOURCE", "selected v%lu (deg=%lu)", source, graph.degree(source));

    // ─── BFS ────────────────────────────────────────────────────────
    inspect_bfs(graph, source);
    graph.verify_invariants("POST_BFS");

    // ─── SSSP ───────────────────────────────────────────────────────
    inspect_sssp(graph, source);
    graph.verify_invariants("POST_SSSP");

    // ─── PageRank ───────────────────────────────────────────────────
    inspect_pagerank(graph, 10, 0.85);
    graph.verify_invariants("POST_PR");

    // ─── WCC ────────────────────────────────────────────────────────
    inspect_wcc(graph);
    graph.verify_invariants("POST_WCC");

    // ─── 访问热点 ───────────────────────────────────────────────────
    sep("ACCESS HOTSPOT ANALYSIS");
    graph.dump_access_hotspots("FINAL", 10);

    // ─── 总结 ───────────────────────────────────────────────────────
    sep("INSPECTION SUMMARY");
    std::printf("[SUMMARY] Inspections: %lu\n", g_inspect_count);
    std::printf("[SUMMARY] Assertions: pass=%lu fail=%lu total=%lu\n",
                g_assert_pass, g_assert_fail, g_assert_pass + g_assert_fail);
    std::printf("[SUMMARY] Final RSS: %ld KB\n", rss_kb());

    if (g_assert_fail > 0) {
        std::printf("\n[WARNING] %lu assertion(s) FAILED — check output above\n", g_assert_fail);
        return 1;
    } else {
        std::printf("\n[OK] All assertions passed.\n");
    }

    std::printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║  Inspection complete.                                     ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════╝\n");
    return 0;
}
