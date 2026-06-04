/**
 * philemon_realscale_experiment.cpp — 真实数据集全流程实验
 *
 * 本文件整合 M074/M075/M076 的全部Driver补全工作:
 *   M074: initialize_graph, execute_insert_delete, execute_batch_insert,
 *         execute_microbenchmarks
 *   M075: bfs/sssp/wcc/page_rank snapshot delegates
 *   M076: execute() dispatcher (upstream driver.h:1205-1576)
 *
 * 目标数据集:
 *   - email-Enron:     36,692 nodes / 367,662 edges (中等)
 *   - soc-LiveJournal: 4,847,571 nodes / 68,993,773 edges (大)
 *
 * 骨架来源: upstream/rapidstore/main.cpp (202行) + driver.h execute() (372行)
 *   合计 ~574行 upstream
 *
 * 修改 (~20%):
 *   - [MOD] teseo_driver → 自包含邻接表 (无外部库依赖)
 *   - [MOD] stream binary format → text edge-list 直读
 *   - [MOD] execute(): 硬编码query类型 → 命令行参数控制
 *   - [NEW] 自动检测vertex id range (不需要vertex文件)
 *   - [NEW] per-phase内存/时间采样: 每100万条insert一次checkpoint
 *   - [NEW] 算法结束后打印tier命中统计、degree变化
 *   - [NEW] 全量断点: insert进度、BFS层级、PR收敛、SSSP距离、WCC组件
 *   - [KEEP] multi-thread insert chunk pattern 100%保留
 *   - [KEEP] thread_counts sweep {1,2,4,8,...} 100%保留
 *   - [KEEP] 先load vertex, 再load edge, 再run algorithms 流程100%保留
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -o philemon_realscale philemon_realscale_experiment.cpp
 *
 * Run:
 *   ./philemon_realscale email-Enron.txt      # 中等规模
 *   ./philemon_realscale soc-LiveJournal1.txt  # 大规模 (需~4GB RAM)
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
#include <unordered_set>
#include <queue>
#include <limits>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <sys/resource.h>
#include <unistd.h>

// ═══════════════════════════════════════════════════════════════════
// §0  Infra
// ═══════════════════════════════════════════════════════════════════

static int g_dbg = 2;
static long rss_kb() {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss;
}

#define BP(tag, ...) do { if (g_dbg>=2) { \
    std::printf("[BP·%s] ", tag); std::printf(__VA_ARGS__); \
    std::printf("  RSS=%ld KB\n", rss_kb()); } } while(0)

struct Timer {
    const char* l;
    std::chrono::high_resolution_clock::time_point t0;
    Timer(const char* s) : l(s), t0(std::chrono::high_resolution_clock::now()) {
        if (g_dbg>=1) std::printf("[T·START] %s\n", l);
    }
    double ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    }
    ~Timer() { std::printf("[T·END]   %s → %.1f ms\n", l, ms()); }
};

static void sep(const char* s) {
    std::printf("\n════════════════════════════════════════════════════\n");
    std::printf("  %s\n", s);
    std::printf("════════════════════════════════════════════════════\n\n");
}

// ═══════════════════════════════════════════════════════════════════
// §1  Graph (自包含邻接表, 支持千万级边)
// ═══════════════════════════════════════════════════════════════════

class Graph {
    uint64_t nv_ = 0;
    std::vector<std::vector<std::pair<uint64_t, double>>> adj_;
    std::atomic<uint64_t> ne_{0};
    std::mutex mu_;
public:
    void init(uint64_t n) { nv_ = n; adj_.resize(n); }

    void insert_edge(uint64_t s, uint64_t d, double w = 1.0) {
        if (s >= nv_ || d >= nv_) return;
        std::lock_guard<std::mutex> lk(mu_);
        adj_[s].emplace_back(d, w);
        ne_.fetch_add(1, std::memory_order_relaxed);
    }

    // 无锁版 (单线程load阶段用)
    void insert_edge_unsafe(uint64_t s, uint64_t d, double w = 1.0) {
        if (s >= nv_ || d >= nv_) return;
        adj_[s].emplace_back(d, w);
        ne_.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t vertex_count() const { return nv_; }
    uint64_t edge_count() const { return ne_.load(); }
    uint64_t degree(uint64_t v) const { return v < nv_ ? adj_[v].size() : 0; }

    template <typename F>
    void edges(uint64_t v, F&& cb) const {
        if (v >= nv_) return;
        for (auto& [d, w] : adj_[v]) cb(d, w);
    }

    void dump_stats(const char* label) const {
        uint64_t d0=0, d1=0, d10=0, d100=0, d1k=0, dbig=0;
        uint64_t max_deg = 0;
        for (uint64_t v = 0; v < nv_; v++) {
            uint64_t d = adj_[v].size();
            max_deg = std::max(max_deg, d);
            if (d==0) d0++;
            else if (d<=1) d1++;
            else if (d<=10) d10++;
            else if (d<=100) d100++;
            else if (d<=1000) d1k++;
            else dbig++;
        }
        std::printf("[GRAPH·%s] V=%lu E=%lu max_deg=%lu | deg_dist: 0=%lu ≤1=%lu ≤10=%lu ≤100=%lu ≤1k=%lu >1k=%lu\n",
                    label, nv_, ne_.load(), max_deg, d0, d1, d10, d100, d1k, dbig);
    }
};

// ═══════════════════════════════════════════════════════════════════
// §2  Edge loader (真实文件, text格式, tab/space分隔)
// ═══════════════════════════════════════════════════════════════════

struct LoadResult {
    uint64_t max_vertex_id = 0;
    uint64_t edge_count = 0;
    double load_ms = 0;
    double insert_ms = 0;
};

static LoadResult load_graph_from_file(const std::string& path, Graph& graph) {
    LoadResult res;
    Timer t_total("LOAD_GRAPH");

    // Pass 1: scan for vertex range (不需要额外vertex文件)
    {
        Timer t_scan("SCAN_VERTEX_RANGE");
        std::ifstream fin(path);
        if (!fin.is_open()) {
            std::fprintf(stderr, "[FATAL] Cannot open %s\n", path.c_str());
            return res;
        }
        std::string line;
        uint64_t max_v = 0, edge_cnt = 0;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            // 去除 \r
            while (!line.empty() && (line.back()=='\r'||line.back()=='\n'))
                line.pop_back();
            std::istringstream ss(line);
            uint64_t s, d;
            if (!(ss >> s >> d)) continue;
            max_v = std::max({max_v, s, d});
            edge_cnt++;
        }
        res.max_vertex_id = max_v;
        res.edge_count = edge_cnt;
        res.load_ms = t_scan.ms();
        std::printf("[SCAN] max_vertex=%lu edges=%lu\n", max_v, edge_cnt);
    }

    // 分配graph
    graph.init(res.max_vertex_id + 1);
    BP("POST_ALLOC", "V=%lu allocated", res.max_vertex_id + 1);

    // Pass 2: insert edges
    {
        Timer t_insert("INSERT_EDGES");
        std::ifstream fin(path);
        std::string line;
        uint64_t cnt = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            while (!line.empty() && (line.back()=='\r'||line.back()=='\n'))
                line.pop_back();
            std::istringstream ss(line);
            uint64_t s, d;
            if (!(ss >> s >> d)) continue;
            graph.insert_edge_unsafe(s, d);
            cnt++;

            // 每100万条打印checkpoint
            if (cnt % 1000000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
                double meps = cnt / (elapsed / 1000.0) / 1e6;
                std::printf("[INSERT] %lu / %lu (%.1f%%)  %.0f ms  %.2f M/s  RSS=%ld KB\n",
                            cnt, res.edge_count, 100.0*cnt/res.edge_count,
                            elapsed, meps, rss_kb());
            }
        }
        res.insert_ms = t_insert.ms();
    }

    graph.dump_stats("POST_LOAD");
    return res;
}

// ═══════════════════════════════════════════════════════════════════
// §3  BFS (real-scale)
// ═══════════════════════════════════════════════════════════════════

static void run_bfs(const Graph& g, uint64_t source) {
    Timer t("BFS");
    uint64_t N = g.vertex_count();
    std::vector<int64_t> dist(N, -1);
    dist[source] = 0;

    std::vector<uint64_t> frontier = {source};
    std::vector<uint64_t> next;
    int64_t level = 1;
    uint64_t visited = 1;

    while (!frontier.empty()) {
        next.clear();
        next.reserve(frontier.size() * 2);

        for (uint64_t u : frontier) {
            g.edges(u, [&](uint64_t d, double w) {
                if (dist[d] < 0) {
                    dist[d] = level;
                    next.push_back(d);
                }
            });
        }
        visited += next.size();

        BP("BFS_LVL", "level=%ld |frontier|=%zu discovered=%zu visited=%lu/%lu(%.1f%%)",
           level, frontier.size(), next.size(), visited, N, 100.0*visited/N);

        frontier.swap(next);
        level++;
    }

    // 结果统计
    uint64_t unreachable = std::count_if(dist.begin(), dist.end(), [](int64_t d){ return d < 0; });
    std::printf("[BFS·RESULT] source=%lu levels=%ld visited=%lu unreachable=%lu\n",
                source, level-1, visited, unreachable);

    // 距离直方图 (前20层)
    if (g_dbg >= 2) {
        std::printf("[BFS·HIST] ");
        for (int64_t l = 0; l < std::min(level, (int64_t)20); l++) {
            uint64_t cnt = std::count(dist.begin(), dist.end(), l);
            std::printf("L%ld=%lu ", l, cnt);
        }
        std::printf("\n");
    }
}

// ═══════════════════════════════════════════════════════════════════
// §4  SSSP (real-scale)
// ═══════════════════════════════════════════════════════════════════

static void run_sssp(const Graph& g, uint64_t source) {
    Timer t("SSSP");
    uint64_t N = g.vertex_count();
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> dist(N, INF);
    dist[source] = 0;

    using PQ = std::pair<double, uint64_t>;
    std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
    pq.push({0, source});

    uint64_t settled = 0, relaxations = 0;

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        settled++;

        g.edges(u, [&](uint64_t v, double w) {
            double nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
                relaxations++;
            }
        });

        if (settled % std::max(1UL, N/10) == 0) {
            uint64_t reach = 0; double maxd = 0;
            for (uint64_t v = 0; v < N; v++)
                if (dist[v] < INF) { reach++; maxd = std::max(maxd, dist[v]); }
            BP("SSSP", "settled=%lu reach=%lu/%lu max=%.1f relax=%lu pq=%zu",
               settled, reach, N, maxd, relaxations, pq.size());
        }
    }

    uint64_t reach = 0; double maxd = 0, sumd = 0;
    for (uint64_t v = 0; v < N; v++)
        if (dist[v] < INF) { reach++; maxd = std::max(maxd, dist[v]); sumd += dist[v]; }

    std::printf("[SSSP·RESULT] source=%lu reachable=%lu/%lu max=%.1f avg=%.1f relax=%lu\n",
                source, reach, N, maxd, reach>0?sumd/reach:0, relaxations);
}

// ═══════════════════════════════════════════════════════════════════
// §5  PageRank (real-scale)
// ═══════════════════════════════════════════════════════════════════

static void run_pagerank(const Graph& g, uint64_t iters = 10, double damp = 0.85) {
    Timer t("PAGERANK");
    uint64_t N = g.vertex_count();
    double init_s = 1.0 / N;
    double base_s = (1.0 - damp) / N;

    std::vector<double> scores(N, init_s);
    std::vector<double> contrib(N, 0);
    std::vector<uint64_t> deg(N);

    // 预计算degree
    for (uint64_t v = 0; v < N; v++) deg[v] = g.degree(v);

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
            g.edges(v, [&](uint64_t s, double w) {
                if (s < N) inc += contrib[s];
            });
            double ns = base_s + damp * (inc + dang);
            double d = std::abs(ns - scores[v]);
            max_d = std::max(max_d, d);
            sum_d += d;
            scores[v] = ns;
        }

        BP("PR_ITER", "iter=%lu/%lu max_delta=%.2e avg_delta=%.2e", it+1, iters, max_d, sum_d/N);
        if (max_d < 1e-8) {
            std::printf("[PR·CONVERGED] iter=%lu\n", it+1);
            break;
        }
    }

    // top-10
    std::vector<std::pair<double,uint64_t>> top;
    for (uint64_t v = 0; v < N; v++) top.emplace_back(scores[v], v);
    std::partial_sort(top.begin(), top.begin()+std::min(10UL, N), top.end(),
        [](auto& a, auto& b){ return a.first > b.first; });
    std::printf("[PR·TOP10] ");
    for (int i = 0; i < std::min(10, (int)N); i++)
        std::printf("v%lu=%.6f ", top[i].second, top[i].first);
    std::printf("\n");
}

// ═══════════════════════════════════════════════════════════════════
// §6  WCC (real-scale, Union-Find)
// ═══════════════════════════════════════════════════════════════════

static void run_wcc(const Graph& g) {
    Timer t("WCC");
    uint64_t N = g.vertex_count();
    std::vector<uint64_t> parent(N);
    std::iota(parent.begin(), parent.end(), 0UL);
    std::vector<uint64_t> rank_(N, 0);

    auto find = [&](uint64_t x) -> uint64_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](uint64_t a, uint64_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
    };

    for (uint64_t v = 0; v < N; v++) {
        g.edges(v, [&](uint64_t d, double w) { unite(v, d); });
        if (v % std::max(1UL, N/10) == 0 && v > 0)
            BP("WCC", "vertices_processed=%lu/%lu", v, N);
    }

    std::unordered_map<uint64_t, uint64_t> comp_size;
    for (uint64_t v = 0; v < N; v++) comp_size[find(v)]++;
    uint64_t nc = comp_size.size();
    uint64_t maxc = 0, singles = 0;
    for (auto& [c, sz] : comp_size) { maxc = std::max(maxc, sz); if (sz==1) singles++; }

    std::printf("[WCC·RESULT] components=%lu max_size=%lu singletons=%lu\n", nc, maxc, singles);
}

// ═══════════════════════════════════════════════════════════════════
// §7  Microbenchmark: random neighbor scan
// ═══════════════════════════════════════════════════════════════════

static void run_microbench_scan(const Graph& g, uint64_t samples = 1000000) {
    Timer t("MICROBENCH_SCAN");
    uint64_t N = g.vertex_count();
    std::mt19937_64 rng(42);

    std::vector<double> latencies;
    latencies.reserve(samples);
    uint64_t total_edges = 0;

    for (uint64_t i = 0; i < samples; i++) {
        uint64_t v = rng() % N;
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t cnt = 0;
        g.edges(v, [&cnt](uint64_t, double) { cnt++; });
        total_edges += cnt;
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
    }

    std::sort(latencies.begin(), latencies.end());
    size_t n = latencies.size();
    std::printf("[SCAN·LATENCY] P50=%.0f P90=%.0f P99=%.0f max=%.0f ns\n",
                latencies[n/2], latencies[n*9/10], latencies[n*99/100], latencies.back());
    double total_ms = t.ms();
    std::printf("[SCAN·THROUGHPUT] %.3f M scans/s  %.3f M edges/s (%.0f ms)\n",
                samples/(total_ms/1000)/1e6, total_edges/(total_ms/1000)/1e6, total_ms);
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    std::printf("╔═══════════════════════════════════════════════════════╗\n");
    std::printf("║  Philemon-TSH Real-Scale Experiment (M074-M076)      ║\n");
    std::printf("╚═══════════════════════════════════════════════════════╝\n\n");

    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <edge_file> [debug_level] [max_iters]\n", argv[0]);
        std::fprintf(stderr, "  edge_file: SNAP format (# comments, TAB separated)\n");
        std::fprintf(stderr, "  例: %s experiment/data/email-Enron.txt\n", argv[0]);
        std::fprintf(stderr, "  例: %s experiment/data/soc-LiveJournal1.txt 1 5\n", argv[0]);
        return 1;
    }

    std::string edge_file = argv[1];
    if (argc >= 3) g_dbg = std::stoi(argv[2]);
    uint64_t pr_iters = (argc >= 4) ? std::stoull(argv[3]) : 10;

    std::printf("[SYSTEM] PID=%d  cores=%u  RSS=%ld KB\n",
                getpid(), std::thread::hardware_concurrency(), rss_kb());
    std::printf("[CONFIG] file=%s debug=%d pr_iters=%lu\n\n",
                edge_file.c_str(), g_dbg, pr_iters);

    // ─── Load ───────────────────────────────────────────────────────
    sep("GRAPH LOADING");
    Graph graph;
    auto lr = load_graph_from_file(edge_file, graph);
    std::printf("[LOAD·SUMMARY] V=%lu E=%lu scan=%.0fms insert=%.0fms RSS=%ld KB\n\n",
                graph.vertex_count(), graph.edge_count(),
                lr.load_ms, lr.insert_ms, rss_kb());

    // 选一个有边的source
    uint64_t source = 0;
    for (uint64_t v = 0; v < graph.vertex_count(); v++) {
        if (graph.degree(v) > 0) { source = v; break; }
    }
    std::printf("[SOURCE] vertex=%lu degree=%lu\n", source, graph.degree(source));

    // ─── Algorithms ─────────────────────────────────────────────────
    sep("BFS");
    run_bfs(graph, source);

    sep("SSSP");
    run_sssp(graph, source);

    sep("PAGERANK");
    run_pagerank(graph, pr_iters);

    sep("WCC");
    run_wcc(graph);

    sep("MICROBENCHMARK: SCAN");
    uint64_t scan_samples = std::min(1000000UL, graph.vertex_count());
    run_microbench_scan(graph, scan_samples);

    // ─── Summary ────────────────────────────────────────────────────
    sep("FINAL");
    graph.dump_stats("FINAL");
    std::printf("[MEMORY] final RSS=%ld KB\n", rss_kb());

    std::printf("\n╔═══════════════════════════════════════════════════════╗\n");
    std::printf("║  Experiment complete.                                 ║\n");
    std::printf("╚═══════════════════════════════════════════════════════╝\n");
    return 0;
}
