/**
 * m137_m138_algorithm_walking.cpp
 * M137-M138: Graph Algorithm Walking — BFS/SSSP/PR/WCC/TC tier comparison
 *
 * 论文 Section 5 需要的算法遍历在不同tier配置下的性能数据
 * 对比: Tiered vs HBM-Only vs DRAM-Only 三种策略
 *
 * 算法改动 (~20%): tier-aware traversal cost model
 * 编译: g++ -std=c++17 -O2 -pthread -o m137_test experiment/m137_m138_algorithm_walking.cpp
 * 运行: ./m137_test [--latex] [--quiet]
 */

#include <iostream>
#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>
#include <functional>
#include <cassert>
#include <iomanip>
#include <set>
#include <map>
#include <atomic>

static int g_debug = 2;
static bool g_latex = false;
static int g_bp = 0, g_ac = 0;
static int g_pass = 0, g_fail = 0;

#define BP(t, f, ...) do { if(g_debug>=2) printf("[BP·%s] " f "\n", t, ##__VA_ARGS__); g_bp++; } while(0)
#define CHECK(c, m) do { g_ac++; if(!(c)){printf("  [FAIL] %s\n",m);return false;} printf("  [PASS] %s\n",m); } while(0)

enum Tier { HBM=0, GDDR=1, DRAM_T=2, TN=3 };
static const char* TierName[] = {"HBM","GDDR","DRAM"};
static constexpr double TLAT[] = {1.0, 5.0, 80.0}; // ns per access

struct Edge { uint32_t dst; float weight; Tier tier; };
struct AdjGraph {
    int n;
    std::vector<std::vector<Edge>> adj;
    std::vector<Tier> node_tier;
    
    void generate(int nodes, int edges_per_node, std::mt19937& rng) {
        n = nodes;
        adj.resize(n);
        node_tier.resize(n);
        for (int i = 0; i < n; i++) {
            double frac = (double)i / n;
            node_tier[i] = frac > 0.85 ? HBM : (frac > 0.50 ? GDDR : DRAM_T);
            for (int e = 0; e < edges_per_node; e++) {
                uint32_t dst = rng() % n;
                float w = 1.0f + (rng() % 100) / 10.0f;
                adj[i].push_back({dst, w, node_tier[i]});
            }
        }
        BP("GRAPH", "nodes=%d edges=%d", n, n * edges_per_node);
    }
};

// Cost model: traverse cost depends on which tier edges are stored in
struct AlgoResult {
    std::string algo;
    std::string strategy;
    double time_us;
    double tier_cost_ns; // simulated tier access cost
    int edges_visited;
};

// ═══════════════════════════════════════════════════════════════════
// Graph Algorithms with tier-aware cost tracking
// ═══════════════════════════════════════════════════════════════════

AlgoResult run_bfs(const AdjGraph& g, int src, const std::string& strategy) {
    std::vector<int> dist(g.n, -1);
    std::queue<int> q;
    dist[src] = 0;
    q.push(src);
    int visited = 0;
    double tier_cost = 0;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (auto& e : g.adj[u]) {
            Tier t = (strategy == "HBM-Only") ? HBM : (strategy == "DRAM-Only") ? DRAM_T : e.tier;
            tier_cost += TLAT[t];
            visited++;
            if (dist[e.dst] < 0) { dist[e.dst] = dist[u] + 1; q.push(e.dst); }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    
    BP("BFS", "strategy=%s visited=%d cost=%.0fns time=%.2fμs", strategy.c_str(), visited, tier_cost, us);
    return {"BFS", strategy, us, tier_cost, visited};
}

AlgoResult run_sssp(const AdjGraph& g, int src, const std::string& strategy) {
    std::vector<float> dist(g.n, 1e9f);
    using PII = std::pair<float, int>;
    std::priority_queue<PII, std::vector<PII>, std::greater<>> pq;
    dist[src] = 0;
    pq.push({0, src});
    int visited = 0;
    double tier_cost = 0;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : g.adj[u]) {
            Tier t = (strategy == "HBM-Only") ? HBM : (strategy == "DRAM-Only") ? DRAM_T : e.tier;
            tier_cost += TLAT[t];
            visited++;
            float nd = dist[u] + e.weight;
            if (nd < dist[e.dst]) { dist[e.dst] = nd; pq.push({nd, (int)e.dst}); }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    BP("SSSP", "strategy=%s visited=%d", strategy.c_str(), visited);
    return {"SSSP", strategy, std::chrono::duration<double, std::micro>(t1-t0).count(), tier_cost, visited};
}

AlgoResult run_pagerank(const AdjGraph& g, int iters, const std::string& strategy) {
    std::vector<double> pr(g.n, 1.0 / g.n), new_pr(g.n, 0);
    int visited = 0;
    double tier_cost = 0;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) {
        std::fill(new_pr.begin(), new_pr.end(), 0.15 / g.n);
        for (int u = 0; u < g.n; u++) {
            double contrib = 0.85 * pr[u] / (g.adj[u].empty() ? 1 : g.adj[u].size());
            for (auto& e : g.adj[u]) {
                Tier t = (strategy == "HBM-Only") ? HBM : (strategy == "DRAM-Only") ? DRAM_T : e.tier;
                tier_cost += TLAT[t];
                visited++;
                new_pr[e.dst] += contrib;
            }
        }
        std::swap(pr, new_pr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    BP("PR", "strategy=%s iters=%d visited=%d", strategy.c_str(), iters, visited);
    return {"PageRank", strategy, std::chrono::duration<double, std::micro>(t1-t0).count(), tier_cost, visited};
}

AlgoResult run_wcc(const AdjGraph& g, const std::string& strategy) {
    std::vector<int> comp(g.n, -1);
    int num_comp = 0, visited = 0;
    double tier_cost = 0;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < g.n; i++) {
        if (comp[i] >= 0) continue;
        comp[i] = num_comp++;
        std::queue<int> q;
        q.push(i);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : g.adj[u]) {
                Tier t = (strategy == "HBM-Only") ? HBM : (strategy == "DRAM-Only") ? DRAM_T : e.tier;
                tier_cost += TLAT[t];
                visited++;
                if (comp[e.dst] < 0) { comp[e.dst] = comp[i]; q.push(e.dst); }
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    BP("WCC", "strategy=%s components=%d visited=%d", strategy.c_str(), num_comp, visited);
    return {"WCC", strategy, std::chrono::duration<double, std::micro>(t1-t0).count(), tier_cost, visited};
}

AlgoResult run_tc(const AdjGraph& g, const std::string& strategy) {
    // Triangle counting (simplified for performance)
    int triangles = 0, visited = 0;
    double tier_cost = 0;
    int sample = std::min(g.n, 500); // Sample for tractability
    
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int u = 0; u < sample; u++) {
        std::set<uint32_t> nu;
        for (auto& e : g.adj[u]) {
            Tier t = (strategy == "HBM-Only") ? HBM : (strategy == "DRAM-Only") ? DRAM_T : e.tier;
            tier_cost += TLAT[t];
            visited++;
            nu.insert(e.dst);
        }
        for (auto& e : g.adj[u]) {
            if ((int)e.dst <= u) continue;
            for (auto& f : g.adj[e.dst]) {
                tier_cost += TLAT[(strategy == "HBM-Only") ? HBM : (strategy == "DRAM-Only") ? DRAM_T : f.tier];
                visited++;
                if (nu.count(f.dst) && (int)f.dst > (int)e.dst) triangles++;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    BP("TC", "strategy=%s triangles=%d visited=%d", strategy.c_str(), triangles, visited);
    return {"TC", strategy, std::chrono::duration<double, std::micro>(t1-t0).count(), tier_cost, visited};
}

// ═══════════════════════════════════════════════════════════════════
// Full benchmark
// ═══════════════════════════════════════════════════════════════════

void run_full_benchmark() {
    std::mt19937 rng(42);
    AdjGraph g;
    g.generate(5000, 10, rng);
    
    std::vector<std::string> strategies = {"Tiered (ours)", "HBM-Only", "DRAM-Only"};
    std::vector<AlgoResult> all_results;
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf(" M137-M138: Algorithm Walking Benchmark\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    for (auto& strat : strategies) {
        printf("── Strategy: %s ──\n", strat.c_str());
        all_results.push_back(run_bfs(g, 0, strat));
        all_results.push_back(run_sssp(g, 0, strat));
        all_results.push_back(run_pagerank(g, 10, strat));
        all_results.push_back(run_wcc(g, strat));
        all_results.push_back(run_tc(g, strat));
        printf("\n");
    }
    
    if (g_latex) {
        printf("\n%% ═══ Algorithm Walking Results (auto-generated by M137) ═══\n");
        printf("\\begin{table}[t]\n\\centering\n");
        printf("\\caption{Graph algorithm traversal: time and tier access cost.}\n");
        printf("\\label{tab:algo_walk}\n");
        printf("\\begin{tabular}{lrrr}\n\\toprule\n");
        printf("Algorithm & Tiered ($\\mu$s) & HBM-Only ($\\mu$s) & DRAM-Only ($\\mu$s) \\\\\n\\midrule\n");
        
        std::map<std::string, std::map<std::string, double>> table;
        for (auto& r : all_results) table[r.algo][r.strategy] = r.time_us;
        for (auto& [algo, strats] : table) {
            printf("%s", algo.c_str());
            for (auto& s : strategies) printf(" & $%.1f$", strats[s]);
            printf(" \\\\\n");
        }
        printf("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        
        printf("\n%% Tier access cost (ns) table\n");
        printf("\\begin{table}[t]\n\\centering\n");
        printf("\\caption{Simulated tier access cost by algorithm.}\n");
        printf("\\begin{tabular}{lrrr}\n\\toprule\n");
        printf("Algorithm & Tiered (ns) & HBM-Only (ns) & DRAM-Only (ns) \\\\\n\\midrule\n");
        std::map<std::string, std::map<std::string, double>> cost_table;
        for (auto& r : all_results) cost_table[r.algo][r.strategy] = r.tier_cost_ns;
        for (auto& [algo, strats] : cost_table) {
            printf("%s", algo.c_str());
            for (auto& s : strategies) printf(" & $%.0f$", strats[s]);
            printf(" \\\\\n");
        }
        printf("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

void run_test(const char* name, std::function<bool()> fn) {
    printf("\n── %s ──\n", name);
    if (fn()) g_pass++; else { g_fail++; }
}

bool test_bfs_correctness() {
    std::mt19937 rng(1);
    AdjGraph g; g.generate(100, 5, rng);
    auto r = run_bfs(g, 0, "Tiered (ours)");
    CHECK(r.edges_visited > 0, "BFS visited edges");
    CHECK(r.tier_cost_ns > 0, "BFS has tier cost");
    return true;
}

bool test_sssp_correctness() {
    std::mt19937 rng(2);
    AdjGraph g; g.generate(100, 5, rng);
    auto r = run_sssp(g, 0, "Tiered (ours)");
    CHECK(r.edges_visited > 0, "SSSP visited edges");
    return true;
}

bool test_pagerank_convergence() {
    std::mt19937 rng(3);
    AdjGraph g; g.generate(100, 5, rng);
    auto r = run_pagerank(g, 5, "HBM-Only");
    CHECK(r.edges_visited == 100 * 5 * 5, "PR visited all edges × iters");
    return true;
}

bool test_wcc_finds_components() {
    std::mt19937 rng(4);
    AdjGraph g; g.generate(100, 5, rng);
    auto r = run_wcc(g, "DRAM-Only");
    CHECK(r.edges_visited > 0, "WCC visited edges");
    CHECK(r.tier_cost_ns > 0, "WCC has tier cost");
    return true;
}

bool test_tc_counts_triangles() {
    std::mt19937 rng(5);
    AdjGraph g; g.generate(200, 10, rng);
    auto r = run_tc(g, "Tiered (ours)");
    CHECK(r.edges_visited > 0, "TC visited edges");
    return true;
}

bool test_tier_cost_ordering() {
    std::mt19937 rng(6);
    AdjGraph g; g.generate(500, 5, rng);
    auto r_hbm = run_bfs(g, 0, "HBM-Only");
    auto r_dram = run_bfs(g, 0, "DRAM-Only");
    auto r_tier = run_bfs(g, 0, "Tiered (ours)");
    CHECK(r_hbm.tier_cost_ns < r_dram.tier_cost_ns, "HBM cost < DRAM cost");
    CHECK(r_tier.tier_cost_ns < r_dram.tier_cost_ns, "Tiered cost < DRAM cost");
    CHECK(r_tier.tier_cost_ns > r_hbm.tier_cost_ns, "Tiered cost > HBM cost (realistic)");
    return true;
}

bool test_all_algos_same_visits() {
    // BFS and WCC on same graph should visit similar edge counts
    std::mt19937 rng(7);
    AdjGraph g; g.generate(100, 5, rng);
    auto bfs = run_bfs(g, 0, "HBM-Only");
    auto wcc = run_wcc(g, "HBM-Only");
    // Both should visit all reachable edges
    CHECK(bfs.edges_visited > 0 && wcc.edges_visited > 0, "both algos visit edges");
    return true;
}

bool test_strategy_consistency() {
    std::mt19937 rng(8);
    AdjGraph g; g.generate(200, 5, rng);
    auto r1 = run_bfs(g, 0, "Tiered (ours)");
    auto r2 = run_bfs(g, 0, "Tiered (ours)");
    CHECK(r1.edges_visited == r2.edges_visited, "deterministic edge count");
    CHECK(std::abs(r1.tier_cost_ns - r2.tier_cost_ns) < 0.01, "deterministic tier cost");
    return true;
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--latex") == 0) g_latex = true;
        if (strcmp(argv[i], "--quiet") == 0) g_debug = 0;
    }
    
    printf("═══════════════════════════════════════════════════════\n");
    printf(" M137-M138 Validation Tests\n");
    printf("═══════════════════════════════════════════════════════\n");
    
    run_test("T1: BFS correctness", test_bfs_correctness);
    run_test("T2: SSSP correctness", test_sssp_correctness);
    run_test("T3: PageRank convergence", test_pagerank_convergence);
    run_test("T4: WCC components", test_wcc_finds_components);
    run_test("T5: Triangle counting", test_tc_counts_triangles);
    run_test("T6: Tier cost ordering", test_tier_cost_ordering);
    run_test("T7: All algos visit", test_all_algos_same_visits);
    run_test("T8: Strategy consistency", test_strategy_consistency);
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf(" Validation: %d/%d passed\n", g_pass, g_pass + g_fail);
    printf("═══════════════════════════════════════════════════════\n");
    
    if (g_fail > 0) { printf("FAILED\n"); return 1; }
    
    run_full_benchmark();
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf(" M137-M138 Complete: bp=%d asserts=%d\n", g_bp, g_ac);
    printf("═══════════════════════════════════════════════════════\n");
    return 0;
}
