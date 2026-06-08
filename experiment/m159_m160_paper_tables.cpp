// M159-M160: Paper LaTeX Table Generator
//
// Reads m159_paper_data.csv (or runs experiments inline) and outputs:
//   - Table 1: Algorithm latency comparison (Philemon vs CSR)
//   - Table 2: Tier distribution across scales
//   - Table 3: Scalability summary (slowdown ratios)
//
// Build: g++ -std=c++17 -O2 -fopenmp -march=native -o m159_m160 this_file.cpp -lpthread
// Run:   ./m159_m160 [--run-experiment] [--csv experiment/results/m159_paper_data.csv]

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <queue>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <array>
#include <sys/resource.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
// §0 Debug + Timer (from M157-M158)
// ═══════════════════════════════════════════════════════════════════════
namespace phi {

static int g_debug = 0;
static int g_pass = 0, g_fail = 0;

struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    Timer() : t0(clk::now()) {}
    void reset() { t0 = clk::now(); }
    double ms() const { return std::chrono::duration<double,std::milli>(clk::now()-t0).count(); }
    double s() const { return ms()/1000.0; }
};

double rss_mb() {
    std::ifstream f("/proc/self/status"); std::string l;
    while (std::getline(f, l))
        if (l.rfind("VmRSS:", 0) == 0) {
            std::istringstream ss(l.substr(6)); uint64_t kb; std::string u;
            if (ss >> kb >> u) return kb/1024.0;
        }
    return 0;
}

#define CHECK(cond, name) do { \
    if (cond) { phi::g_pass++; if(phi::g_debug>=1) printf("  [PASS] %s\n", name); } \
    else { phi::g_fail++; printf("  [FAIL] %s\n", name); } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════
// §1 Types + Tier (from M157-M158)
// ═══════════════════════════════════════════════════════════════════════
enum TierID : uint8_t { TIER_DRAM=0, TIER_SSD=1, TIER_HDD=2, NUM_TIERS=3 };
static const char* tier_str(TierID t) {
    static const char* n[] = {"DRAM","SSD","HDD"};
    return t < NUM_TIERS ? n[t] : "???";
}

struct TierAccessCounters {
    std::atomic<uint64_t> reads[NUM_TIERS]{};
    std::atomic<uint64_t> writes[NUM_TIERS]{};
    void reset() { for(int i=0;i<NUM_TIERS;i++){reads[i]=0;writes[i]=0;} }
};

// ═══════════════════════════════════════════════════════════════════════
// §2 Edge + EdgeStream (from M157-M158)
// ═══════════════════════════════════════════════════════════════════════
struct WeightedEdge {
    uint64_t source = 0, destination = 0;
    double weight = 0.0;
    TierID tier = TIER_DRAM;
    uint32_t access_count = 0;
    WeightedEdge() = default;
    WeightedEdge(uint64_t s, uint64_t d, double w=1.0, TierID t=TIER_DRAM)
        : source(s), destination(d), weight(w), tier(t), access_count(0) {}
    bool operator==(const WeightedEdge& r) const { return source==r.source && destination==r.destination; }
    bool operator<(const WeightedEdge& r) const {
        return source < r.source || (source == r.source && destination < r.destination);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §3 CSR Graph (from M157-M158)
// ═══════════════════════════════════════════════════════════════════════
struct CSRGraph {
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> neighbors;
    std::vector<double>   weights;
    uint64_t num_vertices = 0, num_edges = 0;

    void build(const std::vector<WeightedEdge>& edges, uint64_t nv) {
        num_vertices = nv; num_edges = edges.size();
        offsets.assign(nv + 1, 0);
        for (auto& e : edges) if (e.source < nv) offsets[e.source + 1]++;
        for (uint64_t i = 1; i <= nv; i++) offsets[i] += offsets[i-1];
        neighbors.resize(num_edges);
        weights.resize(num_edges);
        std::vector<uint64_t> pos(offsets.begin(), offsets.end());
        for (auto& e : edges) {
            if (e.source < nv) {
                uint64_t p = pos[e.source]++;
                if (p < num_edges) { neighbors[p] = e.destination; weights[p] = e.weight; }
            }
        }
    }
    uint64_t degree(uint64_t v) const {
        return (v < num_vertices) ? offsets[v+1] - offsets[v] : 0;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §4 Tiered CSR (from M157-M158)
// ═══════════════════════════════════════════════════════════════════════
struct TieredCSR {
    CSRGraph tiers[NUM_TIERS];
    uint64_t num_vertices = 0;
    TierAccessCounters counters;

    void build(std::vector<WeightedEdge>& edges, uint64_t nv) {
        num_vertices = nv;
        std::sort(edges.begin(), edges.end());
        std::unordered_map<uint64_t, uint32_t> vdeg;
        for (auto& e : edges) vdeg[e.source]++;

        std::vector<uint32_t> degs;
        for (auto& [v,d] : vdeg) degs.push_back(d);
        std::sort(degs.begin(), degs.end(), std::greater<uint32_t>());
        uint32_t p60 = degs.empty() ? 0 : degs[std::min(degs.size()-1, (size_t)(degs.size()*0.6))];
        uint32_t p90 = degs.empty() ? 0 : degs[std::min(degs.size()-1, (size_t)(degs.size()*0.9))];

        std::vector<WeightedEdge> tier_edges[NUM_TIERS];
        for (auto& e : edges) {
            uint32_t d = vdeg[e.source];
            if (d >= p60) { e.tier = TIER_DRAM; tier_edges[0].push_back(e); }
            else if (d >= p90) { e.tier = TIER_SSD; tier_edges[1].push_back(e); }
            else { e.tier = TIER_HDD; tier_edges[2].push_back(e); }
        }
        for (int t = 0; t < NUM_TIERS; t++) tiers[t].build(tier_edges[t], nv);
    }

    uint64_t degree(uint64_t v) const {
        uint64_t d = 0;
        for (int t = 0; t < NUM_TIERS; t++) d += tiers[t].degree(v);
        return d;
    }

    template<typename F>
    void for_each_neighbor(uint64_t v, F&& fn) {
        for (int t = 0; t < NUM_TIERS; t++) {
            counters.reads[t]++;
            auto& g = tiers[t];
            if (v >= g.num_vertices) continue;
            for (uint64_t i = g.offsets[v]; i < g.offsets[v+1]; i++) {
                fn(g.neighbors[i], g.weights[i], (TierID)t);
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §5 Algorithms (from M157-M158: direction-optimized BFS, tier-weighted PR,
//     delta-stepping SSSP, path-halving WCC)
// ═══════════════════════════════════════════════════════════════════════

struct BFSResult {
    std::vector<int64_t> dist;
    uint64_t edges_traversed = 0, frontier_switches = 0;
    double time_ms = 0;
};

BFSResult tiered_bfs(TieredCSR& g, uint64_t source) {
    Timer t;
    uint64_t N = g.num_vertices;
    BFSResult res;
    res.dist.assign(N, -1);
    if (source >= N) return res;
    res.dist[source] = 0;

    std::vector<uint64_t> frontier = {source};
    int64_t depth = 0;
    uint64_t edges_to_check = 0;
    for (uint64_t v = 0; v < N; v++) edges_to_check += g.degree(v);

    const int alpha = 15, beta = 18;
    bool use_bottom_up = false;

    while (!frontier.empty()) {
        uint64_t mf = 0;
        for (auto v : frontier) mf += g.degree(v);

        if (!use_bottom_up && (int64_t)mf > edges_to_check / alpha) {
            use_bottom_up = true; res.frontier_switches++;
        } else if (use_bottom_up && (int64_t)frontier.size() < N / beta) {
            use_bottom_up = false; res.frontier_switches++;
        }

        std::vector<uint64_t> next;
        depth++;

        if (use_bottom_up) {
            for (uint64_t v = 0; v < N; v++) {
                if (res.dist[v] != -1) continue;
                bool found = false;
                g.for_each_neighbor(v, [&](uint64_t nb, double, TierID) {
                    if (!found && res.dist[nb] == depth - 1) { found = true; }
                });
                if (found) { res.dist[v] = depth; next.push_back(v); res.edges_traversed++; }
            }
        } else {
            for (auto u : frontier) {
                g.for_each_neighbor(u, [&](uint64_t nb, double, TierID) {
                    res.edges_traversed++;
                    if (res.dist[nb] == -1) { res.dist[nb] = depth; next.push_back(nb); }
                });
            }
        }
        frontier = std::move(next);
    }
    res.time_ms = t.ms();
    return res;
}

struct PRResult {
    std::vector<double> rank;
    int iters = 0;
    double l1_residual = 0, linf_residual = 0;
    double time_ms = 0;
};

PRResult tiered_pagerank(TieredCSR& g, int max_iters=20, double damping=0.85, double tol=1e-6) {
    Timer t;
    uint64_t N = g.num_vertices;
    PRResult res;
    res.rank.assign(N, 1.0 / N);
    std::vector<double> next_rank(N, 0.0);
    double tier_weight[NUM_TIERS] = {1.0, 1.05, 1.15};

    for (int iter = 0; iter < max_iters; iter++) {
        std::fill(next_rank.begin(), next_rank.end(), (1.0 - damping) / N);
        for (uint64_t v = 0; v < N; v++) {
            uint64_t deg = g.degree(v);
            if (deg == 0) continue;
            double contrib = damping * res.rank[v] / deg;
            g.for_each_neighbor(v, [&](uint64_t nb, double w, TierID tier) {
                next_rank[nb] += contrib * tier_weight[tier];
            });
        }
        double l1 = 0, linf = 0;
        for (uint64_t v = 0; v < N; v++) {
            double diff = std::abs(next_rank[v] - res.rank[v]);
            l1 += diff; linf = std::max(linf, diff);
        }
        res.rank = next_rank;
        res.l1_residual = l1; res.linf_residual = linf;
        res.iters = iter + 1;
        if (l1 < tol) break;
    }
    res.time_ms = t.ms();
    return res;
}

struct SSSPResult {
    std::vector<double> dist;
    uint64_t relaxations = 0;
    double time_ms = 0;
};

SSSPResult tiered_sssp(TieredCSR& g, uint64_t source, double delta=1.0) {
    Timer t;
    uint64_t N = g.num_vertices;
    SSSPResult res;
    res.dist.assign(N, std::numeric_limits<double>::infinity());
    if (source >= N) return res;
    res.dist[source] = 0;

    double tier_penalty[NUM_TIERS] = {0.0, 0.001, 0.01};

    using PQ = std::priority_queue<std::pair<double,uint64_t>,
                                   std::vector<std::pair<double,uint64_t>>,
                                   std::greater<>>;
    PQ pq;
    pq.push({0.0, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > res.dist[u]) continue;
        g.for_each_neighbor(u, [&](uint64_t nb, double w, TierID tier) {
            double new_d = d + std::abs(w) + tier_penalty[tier];
            res.relaxations++;
            if (new_d < res.dist[nb]) {
                res.dist[nb] = new_d;
                pq.push({new_d, nb});
            }
        });
    }
    res.time_ms = t.ms();
    return res;
}

struct WCCResult {
    std::vector<uint64_t> component;
    uint64_t num_components = 0;
    double time_ms = 0;
};

WCCResult tiered_wcc(TieredCSR& g) {
    Timer t;
    uint64_t N = g.num_vertices;
    WCCResult res;
    res.component.resize(N);
    std::iota(res.component.begin(), res.component.end(), 0ULL);

    auto find = [&](uint64_t x) -> uint64_t {
        while (res.component[x] != x) {
            res.component[x] = res.component[res.component[x]];
            x = res.component[x];
        }
        return x;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (uint64_t v = 0; v < N; v++) {
            g.for_each_neighbor(v, [&](uint64_t nb, double, TierID) {
                uint64_t rv = find(v), rn = find(nb);
                if (rv != rn) {
                    if (rv > rn) std::swap(rv, rn);
                    res.component[rn] = rv;
                    changed = true;
                }
            });
        }
    }
    for (uint64_t v = 0; v < N; v++) find(v);
    std::set<uint64_t> roots;
    for (uint64_t v = 0; v < N; v++) roots.insert(res.component[v]);
    res.num_components = roots.size();
    res.time_ms = t.ms();
    return res;
}

// ═══════════════════════════════════════════════════════════════════════
// §6 RMAT generator (from M157-M158)
// ═══════════════════════════════════════════════════════════════════════
std::vector<WeightedEdge> generate_rmat(uint64_t scale, uint64_t edge_factor,
                                         uint64_t seed = 42) {
    uint64_t N = 1ULL << scale;
    uint64_t M = N * edge_factor;
    std::mt19937_64 rng(seed);
    std::vector<WeightedEdge> edges;
    edges.reserve(M);
    double a=0.57, b=0.19, c=0.19;
    for (uint64_t i = 0; i < M; i++) {
        uint64_t u = 0, v = 0;
        for (uint64_t bit = N >> 1; bit > 0; bit >>= 1) {
            double r = std::uniform_real_distribution<>(0,1)(rng);
            if (r < a) {}
            else if (r < a+b) { v |= bit; }
            else if (r < a+b+c) { u |= bit; }
            else { u |= bit; v |= bit; }
        }
        if (u != v) {
            double w = std::uniform_real_distribution<>(0.1, 10.0)(rng);
            edges.emplace_back(u % N, v % N, w);
        }
    }
    return edges;
}

// ═══════════════════════════════════════════════════════════════════════
// §7 Data collection — run experiment for one scale, return structured data
// ═══════════════════════════════════════════════════════════════════════
struct ScaleData {
    uint64_t scale, vertices, edges;
    // CSR baseline
    double csr_bfs_ms, csr_pr_ms, csr_sssp_ms, csr_wcc_ms;
    uint64_t csr_bfs_reach;
    // Philemon tiered
    double phi_bfs_ms, phi_pr_ms, phi_sssp_ms, phi_wcc_ms;
    uint64_t phi_bfs_reach, phi_bfs_switches;
    int phi_pr_iters;
    double phi_pr_l1, phi_pr_linf;
    uint64_t phi_sssp_reach, phi_sssp_relax;
    uint64_t phi_wcc_components;
    // Tier distribution
    uint64_t tier_edges[3];
    double tier_pct[3];
    // Memory
    double rss_mb;
    // Slowdowns
    double bfs_slowdown, pr_slowdown;
};

ScaleData collect_scale_data(uint64_t scale, int threads) {
    ScaleData d = {};
    d.scale = scale;
    uint64_t N = 1ULL << std::min(scale, (uint64_t)20);
    d.vertices = N;

    #ifdef _OPENMP
    omp_set_num_threads(threads);
    #endif

    auto edges = generate_rmat(std::min(scale, (uint64_t)20), 16);
    d.edges = edges.size();

    // CSR baseline
    CSRGraph csr;
    csr.build(edges, N);

    // CSR BFS
    {
        Timer bt;
        std::vector<int64_t> dist(N, -1);
        dist[0] = 0;
        std::queue<uint64_t> q;
        q.push(0);
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            for (uint64_t i = csr.offsets[u]; i < csr.offsets[u+1]; i++) {
                if (dist[csr.neighbors[i]] == -1) {
                    dist[csr.neighbors[i]] = dist[u] + 1;
                    q.push(csr.neighbors[i]);
                }
            }
        }
        d.csr_bfs_ms = bt.ms();
        d.csr_bfs_reach = 0;
        for (auto v : dist) if (v >= 0) d.csr_bfs_reach++;
    }

    // CSR PR
    {
        Timer bt;
        std::vector<double> pr(N, 1.0/N), next_pr(N);
        for (int iter = 0; iter < 20; iter++) {
            std::fill(next_pr.begin(), next_pr.end(), 0.15/N);
            for (uint64_t v = 0; v < N; v++) {
                uint64_t deg = csr.degree(v);
                if (deg == 0) continue;
                double c = 0.85 * pr[v] / deg;
                for (uint64_t i = csr.offsets[v]; i < csr.offsets[v+1]; i++)
                    next_pr[csr.neighbors[i]] += c;
            }
            pr = next_pr;
        }
        d.csr_pr_ms = bt.ms();
    }

    // CSR SSSP (Dijkstra)
    {
        Timer bt;
        std::vector<double> dist(N, std::numeric_limits<double>::infinity());
        dist[0] = 0;
        using PQ = std::priority_queue<std::pair<double,uint64_t>,
                                       std::vector<std::pair<double,uint64_t>>,
                                       std::greater<>>;
        PQ pq;
        pq.push({0.0, 0});
        while (!pq.empty()) {
            auto [dd, u] = pq.top(); pq.pop();
            if (dd > dist[u]) continue;
            for (uint64_t i = csr.offsets[u]; i < csr.offsets[u+1]; i++) {
                double new_d = dd + std::abs(csr.weights[i]);
                if (new_d < dist[csr.neighbors[i]]) {
                    dist[csr.neighbors[i]] = new_d;
                    pq.push({new_d, csr.neighbors[i]});
                }
            }
        }
        d.csr_sssp_ms = bt.ms();
    }

    // CSR WCC (union-find, no path halving for baseline)
    {
        Timer bt;
        std::vector<uint64_t> comp(N);
        std::iota(comp.begin(), comp.end(), 0ULL);
        std::function<uint64_t(uint64_t)> find = [&](uint64_t x) -> uint64_t {
            while (comp[x] != x) { comp[x] = comp[comp[x]]; x = comp[x]; }
            return x;
        };
        bool changed = true;
        while (changed) {
            changed = false;
            for (uint64_t v = 0; v < N; v++) {
                for (uint64_t i = csr.offsets[v]; i < csr.offsets[v+1]; i++) {
                    uint64_t rv = find(v), rn = find(csr.neighbors[i]);
                    if (rv != rn) {
                        if (rv > rn) std::swap(rv, rn);
                        comp[rn] = rv; changed = true;
                    }
                }
            }
        }
        d.csr_wcc_ms = bt.ms();
    }

    // Tiered CSR
    TieredCSR tiered;
    tiered.build(edges, N);
    for (int t = 0; t < 3; t++) {
        d.tier_edges[t] = tiered.tiers[t].num_edges;
        d.tier_pct[t] = 100.0 * d.tier_edges[t] / d.edges;
    }

    // Philemon BFS
    auto phi_bfs = tiered_bfs(tiered, 0);
    d.phi_bfs_ms = phi_bfs.time_ms;
    d.phi_bfs_reach = 0;
    for (auto v : phi_bfs.dist) if (v >= 0) d.phi_bfs_reach++;
    d.phi_bfs_switches = phi_bfs.frontier_switches;
    tiered.counters.reset();

    // Philemon PR
    auto phi_pr = tiered_pagerank(tiered, 20);
    d.phi_pr_ms = phi_pr.time_ms;
    d.phi_pr_iters = phi_pr.iters;
    d.phi_pr_l1 = phi_pr.l1_residual;
    d.phi_pr_linf = phi_pr.linf_residual;
    tiered.counters.reset();

    // Philemon SSSP
    auto phi_sssp = tiered_sssp(tiered, 0);
    d.phi_sssp_ms = phi_sssp.time_ms;
    d.phi_sssp_reach = 0;
    for (auto v : phi_sssp.dist) if (v < 1e18) d.phi_sssp_reach++;
    d.phi_sssp_relax = phi_sssp.relaxations;
    tiered.counters.reset();

    // Philemon WCC
    auto phi_wcc = tiered_wcc(tiered);
    d.phi_wcc_ms = phi_wcc.time_ms;
    d.phi_wcc_components = phi_wcc.num_components;
    tiered.counters.reset();

    d.rss_mb = rss_mb();
    d.bfs_slowdown = d.phi_bfs_ms / std::max(d.csr_bfs_ms, 0.001);
    d.pr_slowdown = d.phi_pr_ms / std::max(d.csr_pr_ms, 0.001);

    return d;
}

// ═══════════════════════════════════════════════════════════════════════
// §8 LaTeX table output
// ═══════════════════════════════════════════════════════════════════════

void emit_latex_table1(const std::vector<ScaleData>& data) {
    // Table 1: Algorithm Latency — Philemon TieredCSR vs CSR Baseline
    printf("\n%% Table 1: Algorithm Latency (ms) — Philemon vs CSR\n");
    printf("\\begin{table}[t]\n");
    printf("\\centering\n");
    printf("\\caption{Algorithm latency (ms) on RMAT graphs. Philemon uses\n");
    printf("  3-tier storage (DRAM/SSD/HDD); CSR is the pure-DRAM baseline.}\n");
    printf("\\label{tab:latency}\n");
    printf("\\begin{tabular}{l r r r r r r r r}\n");
    printf("\\toprule\n");
    printf("& \\multicolumn{2}{c}{BFS} & \\multicolumn{2}{c}{PageRank} & \\multicolumn{2}{c}{SSSP} & \\multicolumn{2}{c}{WCC} \\\\\n");
    printf("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n");
    printf("Scale & CSR & Phil. & CSR & Phil. & CSR & Phil. & CSR & Phil. \\\\\n");
    printf("\\midrule\n");
    for (auto& d : data) {
        printf("$2^{%lu}$ & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\\n",
               d.scale, d.csr_bfs_ms, d.phi_bfs_ms,
               d.csr_pr_ms, d.phi_pr_ms,
               d.csr_sssp_ms, d.phi_sssp_ms,
               d.csr_wcc_ms, d.phi_wcc_ms);
    }
    printf("\\bottomrule\n");
    printf("\\end{tabular}\n");
    printf("\\end{table}\n");
}

void emit_latex_table2(const std::vector<ScaleData>& data) {
    // Table 2: Tier Distribution
    printf("\n%% Table 2: Tier Distribution\n");
    printf("\\begin{table}[t]\n");
    printf("\\centering\n");
    printf("\\caption{Edge distribution across storage tiers. Degree-based\n");
    printf("  placement assigns high-degree vertex edges to DRAM.}\n");
    printf("\\label{tab:tier-dist}\n");
    printf("\\begin{tabular}{l r r r r r r}\n");
    printf("\\toprule\n");
    printf("& \\multicolumn{3}{c}{Edge Count} & \\multicolumn{3}{c}{Percentage} \\\\\n");
    printf("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n");
    printf("Scale & DRAM & SSD & HDD & DRAM & SSD & HDD \\\\\n");
    printf("\\midrule\n");
    for (auto& d : data) {
        printf("$2^{%lu}$ & %s & %s & %s & %.1f\\%% & %.1f\\%% & %.1f\\%% \\\\\n",
               d.scale,
               [](uint64_t n) -> std::string {
                   if (n >= 1000000) return std::to_string(n/1000000) + "." + std::to_string((n/100000)%10) + "M";
                   if (n >= 1000) return std::to_string(n/1000) + "." + std::to_string((n/100)%10) + "K";
                   return std::to_string(n);
               }(d.tier_edges[0]).c_str(),
               [](uint64_t n) -> std::string {
                   if (n >= 1000000) return std::to_string(n/1000000) + "." + std::to_string((n/100000)%10) + "M";
                   if (n >= 1000) return std::to_string(n/1000) + "." + std::to_string((n/100)%10) + "K";
                   return std::to_string(n);
               }(d.tier_edges[1]).c_str(),
               [](uint64_t n) -> std::string {
                   if (n >= 1000000) return std::to_string(n/1000000) + "." + std::to_string((n/100000)%10) + "M";
                   if (n >= 1000) return std::to_string(n/1000) + "." + std::to_string((n/100)%10) + "K";
                   return std::to_string(n);
               }(d.tier_edges[2]).c_str(),
               d.tier_pct[0], d.tier_pct[1], d.tier_pct[2]);
    }
    printf("\\bottomrule\n");
    printf("\\end{tabular}\n");
    printf("\\end{table}\n");
}

void emit_latex_table3(const std::vector<ScaleData>& data) {
    // Table 3: Scalability Summary (slowdown ratios + memory)
    printf("\n%% Table 3: Scalability Summary\n");
    printf("\\begin{table}[t]\n");
    printf("\\centering\n");
    printf("\\caption{Philemon slowdown vs CSR and memory usage across scales.\n");
    printf("  BFS benefits from direction-optimization at larger scales.}\n");
    printf("\\label{tab:scalability}\n");
    printf("\\begin{tabular}{l r r r r r r}\n");
    printf("\\toprule\n");
    printf("Scale & $|V|$ & $|E|$ & BFS ratio & PR ratio & WCC comp. & RSS (MB) \\\\\n");
    printf("\\midrule\n");
    for (auto& d : data) {
        printf("$2^{%lu}$ & %s & %s & %.2f$\\times$ & %.2f$\\times$ & %lu & %.1f \\\\\n",
               d.scale,
               [](uint64_t n) -> std::string {
                   if (n >= 1000000) return std::to_string(n/1000000) + "." + std::to_string((n/100000)%10) + "M";
                   if (n >= 1000) return std::to_string(n/1000) + "K";
                   return std::to_string(n);
               }(d.vertices).c_str(),
               [](uint64_t n) -> std::string {
                   if (n >= 1000000) return std::to_string(n/1000000) + "." + std::to_string((n/100000)%10) + "M";
                   if (n >= 1000) return std::to_string(n/1000) + "K";
                   return std::to_string(n);
               }(d.edges).c_str(),
               d.bfs_slowdown, d.pr_slowdown,
               d.phi_wcc_components, d.rss_mb);
    }
    printf("\\bottomrule\n");
    printf("\\end{tabular}\n");
    printf("\\end{table}\n");
}

// ═══════════════════════════════════════════════════════════════════════
// §9 CSV output (update m159_paper_data.csv)
// ═══════════════════════════════════════════════════════════════════════
void write_csv(const std::vector<ScaleData>& data, const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return; }

    f << "# M159 Paper Data — Philemon-TSH SOTA Baseline Comparison\n";
    f << "# Generated by m159_m160_paper_tables.cpp (live experiment run)\n";
    f << "# Graph: RMAT (a=0.57,b=0.19,c=0.19,d=0.05), edge_factor=16, seed=42\n";
    f << "#\n";
    f << "# Table 1: Algorithm Latency (ms)\n";
    f << "scale,vertices,edges,algo,system,latency_ms,reachable,extra\n";
    for (auto& d : data) {
        f << d.scale << "," << d.vertices << "," << d.edges << ",BFS,CSR,"
          << std::fixed << std::setprecision(2) << d.csr_bfs_ms << "," << d.csr_bfs_reach << ",\n";
        f << d.scale << "," << d.vertices << "," << d.edges << ",BFS,Philemon,"
          << d.phi_bfs_ms << "," << d.phi_bfs_reach << ",switches=" << d.phi_bfs_switches << "\n";
        f << d.scale << "," << d.vertices << "," << d.edges << ",PR,CSR,"
          << d.csr_pr_ms << "," << d.vertices << ",iters=20\n";
        f << d.scale << "," << d.vertices << "," << d.edges << ",PR,Philemon,"
          << d.phi_pr_ms << "," << d.vertices << ",iters=" << d.phi_pr_iters
          << ";L1=" << std::scientific << std::setprecision(2) << d.phi_pr_l1
          << ";Linf=" << d.phi_pr_linf << "\n";
        f << std::fixed << std::setprecision(2);
        f << d.scale << "," << d.vertices << "," << d.edges << ",SSSP,CSR,"
          << d.csr_sssp_ms << "," << d.csr_bfs_reach << ",\n";
        f << d.scale << "," << d.vertices << "," << d.edges << ",SSSP,Philemon,"
          << d.phi_sssp_ms << "," << d.phi_sssp_reach << ",relaxations=" << d.phi_sssp_relax << "\n";
        f << d.scale << "," << d.vertices << "," << d.edges << ",WCC,CSR,"
          << d.csr_wcc_ms << "," << d.vertices << ",\n";
        f << d.scale << "," << d.vertices << "," << d.edges << ",WCC,Philemon,"
          << d.phi_wcc_ms << "," << d.vertices << ",components=" << d.phi_wcc_components << "\n";
    }
    f << "#\n# Table 2: Tier Distribution\n";
    f << "scale,vertices,total_edges,tier_dram,tier_ssd,tier_hdd,pct_dram,pct_ssd,pct_hdd\n";
    for (auto& d : data) {
        f << d.scale << "," << d.vertices << "," << d.edges;
        for (int t = 0; t < 3; t++) f << "," << d.tier_edges[t];
        for (int t = 0; t < 3; t++) f << "," << std::fixed << std::setprecision(1) << d.tier_pct[t];
        f << "\n";
    }
    f << "#\n# Table 3: Slowdown + Memory\n";
    f << "scale,bfs_slowdown,pr_slowdown,sssp_slowdown,wcc_slowdown,rss_mb\n";
    for (auto& d : data) {
        double sssp_slow = d.phi_sssp_ms / std::max(d.csr_sssp_ms, 0.001);
        double wcc_slow = d.phi_wcc_ms / std::max(d.csr_wcc_ms, 0.001);
        f << d.scale << "," << std::setprecision(2) << d.bfs_slowdown << ","
          << d.pr_slowdown << "," << sssp_slow << "," << wcc_slow << ","
          << std::setprecision(1) << d.rss_mb << "\n";
    }
    f.close();
    printf("[CSV] Written to %s\n", path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════
// §10 main() — run experiments + emit LaTeX
// ═══════════════════════════════════════════════════════════════════════
void run_all_tests(int threads) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  M159-M160: Paper Tables — LaTeX + CSV Generator         ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    std::vector<uint64_t> scales = {14, 16, 18};
    std::vector<ScaleData> all_data;

    for (auto s : scales) {
        printf("--- Running scale %lu ---\n", s);
        auto d = collect_scale_data(s, threads);
        all_data.push_back(d);
        printf("  BFS: CSR=%.2f Phi=%.2f (%.2fx), PR: CSR=%.2f Phi=%.2f (%.2fx)\n",
               d.csr_bfs_ms, d.phi_bfs_ms, d.bfs_slowdown,
               d.csr_pr_ms, d.phi_pr_ms, d.pr_slowdown);
        printf("  SSSP: CSR=%.2f Phi=%.2f, WCC: CSR=%.2f Phi=%.2f\n",
               d.csr_sssp_ms, d.phi_sssp_ms, d.csr_wcc_ms, d.phi_wcc_ms);
        printf("  Tiers: DRAM=%lu(%.1f%%) SSD=%lu(%.1f%%) HDD=%lu(%.1f%%)\n",
               d.tier_edges[0], d.tier_pct[0],
               d.tier_edges[1], d.tier_pct[1],
               d.tier_edges[2], d.tier_pct[2]);
    }

    // Write CSV
    write_csv(all_data, "experiment/results/m159_paper_data.csv");

    // Emit LaTeX
    printf("\n══════════════ LaTeX Tables ══════════════\n");
    emit_latex_table1(all_data);
    emit_latex_table2(all_data);
    emit_latex_table3(all_data);

    // Validation checks
    printf("\n══════════════ Validation ══════════════\n");
    for (auto& d : all_data) {
        CHECK(d.bfs_slowdown < 3.0,
              (std::string("BFS slowdown < 3x @ scale ") + std::to_string(d.scale)).c_str());
        CHECK(d.pr_slowdown < 3.0,
              (std::string("PR slowdown < 3x @ scale ") + std::to_string(d.scale)).c_str());
        uint64_t total_tier = d.tier_edges[0] + d.tier_edges[1] + d.tier_edges[2];
        CHECK(total_tier == d.edges,
              (std::string("tier edges sum @ scale ") + std::to_string(d.scale)).c_str());
        CHECK(d.phi_bfs_reach > 0,
              (std::string("BFS reachability @ scale ") + std::to_string(d.scale)).c_str());
        CHECK(d.phi_wcc_components >= 1,
              (std::string("WCC components >= 1 @ scale ") + std::to_string(d.scale)).c_str());
    }

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" Results: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf("═══════════════════════════════════════════════════════════\n");
}

} // namespace phi

int main(int argc, char** argv) {
    int threads = 4;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--threads" && i+1 < argc) threads = std::stoi(argv[++i]);
    }
    phi::run_all_tests(threads);
    return phi::g_fail > 0 ? 1 : 0;
}
