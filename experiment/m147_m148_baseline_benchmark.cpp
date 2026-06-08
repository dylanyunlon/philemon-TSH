// ═══════════════════════════════════════════════════════════════════════════════
// M147-M148: Baseline microbenchmark — insert / scan / search
//
// Purpose: produce reproducible CSR baseline numbers (MEPS for insert/scan,
// ns/op for search, RSS in MB) on RMAT scale20 avg_degree=32. This is the
// reference point against which philemon-TSH tiered storage will be compared
// (RQ1, RQ4 in CLAUDE_DEVELOPMENT_PLAN.md).
//
// vs M145-M146:
//   - Drops 4 graph algorithms (BFS/PR/SSSP/WCC), keeps RMAT generator + CSR.
//   - Switches std::vector<bool> visited[] → std::vector<char> to fix the
//     concurrent-write race (vector<bool> is a bit-packed proxy; adjacent
//     bits share a byte, so two threads writing visited[i]=true and
//     visited[i^1]=true race on the same byte even without overlap).
//   - Adds three parallel microbenchmarks instead of full algorithms.
//
// Build: g++ -std=c++17 -O2 -fopenmp -o m147_m148 m147_m148_baseline_benchmark.cpp
// Run:   ./m147_m148 [--scale 20] [--avg-deg 32] [--threads N] [--search-ops 1000000]
// ═══════════════════════════════════════════════════════════════════════════════

#include <atomic>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <omp.h>
#include <parallel/algorithm>

// ─── Timer (lifted from M145-M146 utils::Timer) ─────────────────────────────
namespace philemon { namespace utils {
class Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point m_start;
public:
    Timer() : m_start(clk::now()) {}
    void reset() { m_start = clk::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clk::now() - m_start).count();
    }
    double elapsed_s() const { return elapsed_ms() / 1000.0; }
};
}} // philemon::utils

// ─── Memory probe — read VmRSS from /proc/self/status ───────────────────────
namespace philemon { namespace utils {
double current_rss_mb() {
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return -1.0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream ss(line.substr(6));
            uint64_t kb; std::string unit;
            if (ss >> kb >> unit) return (double)kb / 1024.0;
        }
    }
    return -1.0;
}
double peak_rss_mb() {
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return -1.0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream ss(line.substr(6));
            uint64_t kb; std::string unit;
            if (ss >> kb >> unit) return (double)kb / 1024.0;
        }
    }
    return -1.0;
}
}} // philemon::utils

// ═══════════════════════════════════════════════════════════════════════════════
// RMAT generator — same a/b/c/d coefficients as M145, parallel via OpenMP
// Returns deduplicated edge list (u, v). Edges are directed; we leave it to
// the benchmark to symmetrize if needed.
// ═══════════════════════════════════════════════════════════════════════════════
namespace philemon { namespace graph {

struct EdgeList {
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    uint64_t num_vertices = 0;
};

EdgeList generate_rmat(int scale, int avg_degree, int num_threads, int seed = 42) {
    const uint64_t V = 1ULL << scale;
    const uint64_t target_E = V * (uint64_t)avg_degree;

    std::printf("[RMAT] scale=%d V=2^%d=%lu target_E=%lu avg_deg=%d threads=%d\n",
                scale, scale, (unsigned long)V, (unsigned long)target_E,
                avg_degree, num_threads);

    uint64_t logN = 0;
    { uint64_t n = 1; while (n < V) { n <<= 1; logN++; } }
    const double a = 0.57, b = 0.19, c = 0.19;
    // d = 1 - a - b - c = 0.05

    const uint64_t chunk = (target_E + num_threads - 1) / num_threads;
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> thread_edges(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        uint64_t start = (uint64_t)tid * chunk;
        uint64_t end = std::min(start + chunk, target_E);
        std::mt19937_64 rng((uint64_t)seed + (uint64_t)tid * 1000003ULL);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        thread_edges[tid].reserve(end - start);

        for (uint64_t e = start; e < end; e++) {
            uint64_t u = 0, v = 0;
            for (uint64_t d = logN; d > 0; d--) {
                double r = dist(rng);
                uint64_t bit = 1ULL << (d - 1);
                if (r < a) {}
                else if (r < a + b) { v |= bit; }
                else if (r < a + b + c) { u |= bit; }
                else { u |= bit; v |= bit; }
            }
            u %= V; v %= V;
            if (u != v) thread_edges[tid].push_back({u, v});
        }
    }

    EdgeList el;
    el.num_vertices = V;
    uint64_t total = 0;
    for (auto& te : thread_edges) total += te.size();
    el.edges.reserve(total);
    for (auto& te : thread_edges) {
        el.edges.insert(el.edges.end(), te.begin(), te.end());
        te.clear(); te.shrink_to_fit();
    }

    __gnu_parallel::sort(el.edges.begin(), el.edges.end());
    el.edges.erase(std::unique(el.edges.begin(), el.edges.end()), el.edges.end());

    std::printf("[RMAT] generated %lu unique directed edges (avg_deg=%.2f)\n",
                (unsigned long)el.edges.size(),
                (double)el.edges.size() / (double)V);
    return el;
}

}} // philemon::graph

// ═══════════════════════════════════════════════════════════════════════════════
// CSR — built in parallel from the edge list. Insert benchmark measures this
// build phase (edges/sec). The neighbor arrays per vertex are sorted so we can
// run binary-search lookups for the search benchmark.
// ═══════════════════════════════════════════════════════════════════════════════
namespace philemon { namespace graph {

struct CSR {
    uint64_t num_vertices = 0;
    uint64_t num_edges = 0;
    std::vector<uint64_t> offsets;        // size V+1
    std::vector<uint64_t> destinations;   // size num_edges

    size_t memory_bytes() const {
        return offsets.capacity() * sizeof(uint64_t)
             + destinations.capacity() * sizeof(uint64_t);
    }
    uint64_t degree(uint64_t v) const { return offsets[v + 1] - offsets[v]; }
};

// Build CSR from sorted edge list, in parallel. Returns elapsed seconds.
double build_csr_parallel(const EdgeList& el, CSR& csr, int num_threads) {
    utils::Timer t;
    const uint64_t V = el.num_vertices;
    const uint64_t E = el.edges.size();
    csr.num_vertices = V;
    csr.num_edges = E;

    // 1) Count degrees in parallel using atomic adds.
    csr.offsets.assign(V + 1, 0);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (uint64_t i = 0; i < E; i++) {
        #pragma omp atomic
        csr.offsets[el.edges[i].first + 1]++;
    }

    // 2) Prefix sum (sequential — V+1 adds, cheap relative to the rest).
    for (uint64_t i = 1; i <= V; i++) csr.offsets[i] += csr.offsets[i - 1];

    // 3) Fill destinations using per-vertex atomic write cursors.
    csr.destinations.assign(E, 0);
    std::vector<std::atomic<uint64_t>> pos(V);
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (uint64_t i = 0; i < V; i++) pos[i].store(csr.offsets[i]);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (uint64_t i = 0; i < E; i++) {
        uint64_t u = el.edges[i].first;
        uint64_t idx = pos[u].fetch_add(1, std::memory_order_relaxed);
        csr.destinations[idx] = el.edges[i].second;
    }

    // 4) Sort each vertex's neighbor list so binary search works.
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 256)
    for (uint64_t v = 0; v < V; v++) {
        auto begin = csr.destinations.begin() + csr.offsets[v];
        auto end = csr.destinations.begin() + csr.offsets[v + 1];
        std::sort(begin, end);
    }

    return t.elapsed_s();
}

}} // philemon::graph

// ═══════════════════════════════════════════════════════════════════════════════
// Microbenchmarks
// ═══════════════════════════════════════════════════════════════════════════════
namespace philemon { namespace bench {

struct InsertResult {
    double elapsed_s = 0;
    double meps = 0;
    uint64_t edges_inserted = 0;
};

struct ScanResult {
    double elapsed_s = 0;
    double meps = 0;
    uint64_t edges_scanned = 0;
    uint64_t checksum = 0;        // prevent dead-code elimination
    uint64_t vertices_reached = 0;
};

struct SearchResult {
    double elapsed_s = 0;
    double ns_per_op = 0;
    uint64_t ops = 0;
    uint64_t hits = 0;
};

// ── INSERT ────────────────────────────────────────────────────────────────────
// Measure end-to-end CSR build (degree count → prefix sum → fill → sort).
// MEPS = edges / seconds / 1e6.
InsertResult bench_insert(const graph::EdgeList& el, graph::CSR& csr, int nthreads) {
    InsertResult r;
    r.elapsed_s = graph::build_csr_parallel(el, csr, nthreads);
    r.edges_inserted = el.edges.size();
    r.meps = (double)r.edges_inserted / r.elapsed_s / 1e6;
    return r;
}

// ── SCAN ──────────────────────────────────────────────────────────────────────
// BFS-like full-frontier expansion from a single source, but using the
// reachability traversal pattern only — no levels, no early termination, no
// direction switching. Every reachable edge is visited exactly once.
//
// This is where the M145 bug lived. M145 used `std::vector<bool> visited` and
// flipped bits from multiple OpenMP threads in the top-down step. Because
// vector<bool> packs 8 elements into a byte, two threads writing visited[i]
// and visited[i+1] race on the same byte, which can drop updates or corrupt
// neighboring entries. We use `std::vector<char>` (one byte per element) and
// upgrade reads/writes to atomic_ref-style semantics via std::atomic on a
// reinterpreted pointer — for char that's a single byte store which is safe.
//
// To stay simple and portable, we treat the visited array as char and rely on
// the fact that disjoint-byte writes do not race. The CAS guard ensures only
// one thread claims each vertex.
ScanResult bench_scan(const graph::CSR& csr, uint64_t source, int nthreads) {
    ScanResult r;
    const uint64_t V = csr.num_vertices;

    // FIX: vector<char>, not vector<bool>. One byte per vertex, no shared bytes.
    std::vector<char> visited(V, 0);
    visited[source] = 1;

    std::vector<uint64_t> frontier;
    frontier.reserve(1024);
    frontier.push_back(source);

    uint64_t checksum = 0;
    uint64_t edges_scanned = 0;
    uint64_t vertices_reached = 1;

    utils::Timer t;

    while (!frontier.empty()) {
        std::vector<uint64_t> next_frontier;
        std::mutex mtx;
        std::atomic<uint64_t> local_edges{0};
        std::atomic<uint64_t> local_checksum{0};

        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<uint64_t> local_next;
            local_next.reserve(256);
            uint64_t tcheck = 0;
            uint64_t tedges = 0;

            #pragma omp for schedule(dynamic, 64) nowait
            for (size_t i = 0; i < frontier.size(); i++) {
                uint64_t u = frontier[i];
                uint64_t beg = csr.offsets[u];
                uint64_t end = csr.offsets[u + 1];
                tedges += (end - beg);
                for (uint64_t j = beg; j < end; j++) {
                    uint64_t w = csr.destinations[j];
                    tcheck += w;
                    // CAS-style claim using __sync builtin on char.
                    // bool_compare_and_swap on a byte is atomic and won't
                    // corrupt neighbors (each byte stands alone).
                    if (visited[w] == 0) {
                        char expected = 0;
                        if (__atomic_compare_exchange_n(
                                &visited[w], &expected, (char)1,
                                false,
                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                            local_next.push_back(w);
                        }
                    }
                }
            }

            local_edges.fetch_add(tedges, std::memory_order_relaxed);
            local_checksum.fetch_add(tcheck, std::memory_order_relaxed);

            std::lock_guard<std::mutex> lk(mtx);
            next_frontier.insert(next_frontier.end(),
                                 local_next.begin(), local_next.end());
        }

        edges_scanned += local_edges.load();
        checksum += local_checksum.load();
        vertices_reached += next_frontier.size();
        frontier = std::move(next_frontier);
    }

    r.elapsed_s = t.elapsed_s();
    r.edges_scanned = edges_scanned;
    r.checksum = checksum;
    r.vertices_reached = vertices_reached;
    r.meps = r.elapsed_s > 0
           ? (double)edges_scanned / r.elapsed_s / 1e6
           : 0.0;
    return r;
}

// ── SEARCH ────────────────────────────────────────────────────────────────────
// Point lookup: for each (u, v) query, binary search v in csr.destinations[
// offsets[u] .. offsets[u+1] ). Queries are drawn 50/50 from real edges (hits)
// and random pairs (mostly misses). Reported as average ns/op.
SearchResult bench_search(const graph::CSR& csr,
                          const graph::EdgeList& el,
                          uint64_t num_ops,
                          int nthreads,
                          int seed = 1337) {
    SearchResult r;
    r.ops = num_ops;
    const uint64_t E = el.edges.size();
    const uint64_t V = csr.num_vertices;

    // Pre-generate queries so the timing loop is pure search work.
    std::vector<std::pair<uint64_t, uint64_t>> queries(num_ops);
    {
        std::mt19937_64 rng((uint64_t)seed);
        std::uniform_int_distribution<uint64_t> edge_pick(0, E - 1);
        std::uniform_int_distribution<uint64_t> vtx_pick(0, V - 1);
        for (uint64_t i = 0; i < num_ops; i++) {
            if ((i & 1) == 0) {
                // Hit query — pull a real edge.
                queries[i] = el.edges[edge_pick(rng)];
            } else {
                // Random query — likely a miss.
                queries[i] = {vtx_pick(rng), vtx_pick(rng)};
            }
        }
    }

    uint64_t total_hits = 0;

    utils::Timer t;
    #pragma omp parallel for num_threads(nthreads) schedule(static) reduction(+:total_hits)
    for (uint64_t i = 0; i < num_ops; i++) {
        uint64_t u = queries[i].first;
        uint64_t v = queries[i].second;
        if (u >= V) continue;
        auto begin = csr.destinations.begin() + csr.offsets[u];
        auto end = csr.destinations.begin() + csr.offsets[u + 1];
        if (std::binary_search(begin, end, v)) total_hits++;
    }
    r.hits = total_hits;
    r.elapsed_s = t.elapsed_s();
    r.ns_per_op = r.elapsed_s * 1e9 / (double)num_ops;
    return r;
}

}} // philemon::bench

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    int scale = 20;
    int avg_degree = 32;
    int threads = omp_get_max_threads();
    uint64_t search_ops = 1'000'000;
    int seed = 42;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next_int = [&](long long defv) -> long long {
            if (i + 1 < argc) return std::atoll(argv[++i]);
            return defv;
        };
        if      (a == "--scale")       scale = (int)next_int(scale);
        else if (a == "--avg-deg")     avg_degree = (int)next_int(avg_degree);
        else if (a == "--threads")     threads = (int)next_int(threads);
        else if (a == "--search-ops")  search_ops = (uint64_t)next_int(search_ops);
        else if (a == "--seed")        seed = (int)next_int(seed);
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--scale N] [--avg-deg N] [--threads N] "
                        "[--search-ops N] [--seed N]\n", argv[0]);
            return 0;
        }
    }

    omp_set_num_threads(threads);

    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf(" M147-M148: Baseline microbenchmark — insert/scan/search\n");
    std::printf(" RMAT scale=%d (V=%lu)  avg_degree=%d  threads=%d\n",
                scale, (1UL << scale), avg_degree, threads);
    std::printf(" Search ops=%lu  seed=%d\n",
                (unsigned long)search_ops, seed);
    std::printf("═══════════════════════════════════════════════════════\n");

    double rss_before_gen = philemon::utils::current_rss_mb();

    // 1) Generate RMAT edges.
    auto el = philemon::graph::generate_rmat(scale, avg_degree, threads, seed);

    double rss_after_gen = philemon::utils::current_rss_mb();

    // 2) INSERT — build the CSR.
    philemon::graph::CSR csr;
    auto ins = philemon::bench::bench_insert(el, csr, threads);
    double rss_after_csr = philemon::utils::current_rss_mb();
    double csr_bytes_mb = (double)csr.memory_bytes() / (1024.0 * 1024.0);

    // We no longer need the raw edge list for SCAN; keep it for SEARCH queries.
    std::printf("\n── INSERT ──\n");
    std::printf("  edges inserted : %lu\n",   (unsigned long)ins.edges_inserted);
    std::printf("  elapsed (s)    : %.3f\n",  ins.elapsed_s);
    std::printf("  throughput     : %.2f MEPS\n", ins.meps);

    // 3) SCAN — BFS-style traversal from vertex 0 (the typical RMAT hub).
    auto scn = philemon::bench::bench_scan(csr, 0, threads);
    std::printf("\n── SCAN ──\n");
    std::printf("  source vertex      : 0\n");
    std::printf("  vertices reached   : %lu / %lu  (%.1f%%)\n",
                (unsigned long)scn.vertices_reached,
                (unsigned long)csr.num_vertices,
                100.0 * (double)scn.vertices_reached / (double)csr.num_vertices);
    std::printf("  edges scanned      : %lu\n",   (unsigned long)scn.edges_scanned);
    std::printf("  elapsed (s)        : %.3f\n",  scn.elapsed_s);
    std::printf("  throughput         : %.2f MEPS\n", scn.meps);
    std::printf("  checksum           : %lu (guard against DCE)\n",
                (unsigned long)scn.checksum);

    // 4) SEARCH — binary-search point lookups.
    auto srch = philemon::bench::bench_search(csr, el, search_ops, threads, seed + 7);
    std::printf("\n── SEARCH ──\n");
    std::printf("  ops              : %lu\n",   (unsigned long)srch.ops);
    std::printf("  hits             : %lu  (%.1f%%)\n",
                (unsigned long)srch.hits,
                100.0 * (double)srch.hits / (double)srch.ops);
    std::printf("  elapsed (s)      : %.3f\n",  srch.elapsed_s);
    std::printf("  latency          : %.1f ns/op (aggregate, %d-way parallel)\n",
                srch.ns_per_op, threads);
    std::printf("  latency (1T eq.) : %.1f ns/op\n",
                srch.ns_per_op * (double)threads);

    // 5) MEMORY
    double rss_now = philemon::utils::current_rss_mb();
    double rss_peak = philemon::utils::peak_rss_mb();
    std::printf("\n── MEMORY ──\n");
    std::printf("  RSS before RMAT  : %8.1f MB\n", rss_before_gen);
    std::printf("  RSS after RMAT   : %8.1f MB\n", rss_after_gen);
    std::printf("  RSS after CSR    : %8.1f MB\n", rss_after_csr);
    std::printf("  RSS current      : %8.1f MB\n", rss_now);
    std::printf("  RSS peak (VmHWM) : %8.1f MB\n", rss_peak);
    std::printf("  CSR struct size  : %8.1f MB (offsets+destinations only)\n",
                csr_bytes_mb);

    // 6) Summary line — pipe-friendly for log scrapers in run_ags1_ci.sh
    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf("SUMMARY scale=%d avg_deg=%d threads=%d "
                "insert_meps=%.2f scan_meps=%.2f search_ns=%.1f "
                "memory_mb=%.1f peak_mb=%.1f\n",
                scale, avg_degree, threads,
                ins.meps, scn.meps, srch.ns_per_op,
                rss_now, rss_peak);
    std::printf("═══════════════════════════════════════════════════════\n");

    // Basic sanity gates so CI can fail loudly if something is wrong.
    int fails = 0;
    if (ins.edges_inserted == 0) { std::fprintf(stderr, "FAIL: zero edges\n"); fails++; }
    if (scn.vertices_reached < 2) { std::fprintf(stderr, "FAIL: scan reached <2 vertices\n"); fails++; }
    if (srch.hits == 0) { std::fprintf(stderr, "FAIL: search hits=0\n"); fails++; }
    if (rss_peak <= 0) { std::fprintf(stderr, "WARN: VmHWM unreadable (non-Linux?)\n"); }

    return fails > 0 ? 1 : 0;
}
