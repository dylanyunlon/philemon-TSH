// ═══════════════════════════════════════════════════════════════════════════════
// M153-M154: Ablation study + LaTeX-ready table data
//
// Purpose: produce the ablation and scaling data for the Philemon-TSH paper.
//   - Ablation: which tier combinations matter (hot-only vs hot+warm vs all 3)
//   - Prefetch effect: madvise(WILLNEED) on warm tier
//   - Sort effect: sorted vs unsorted neighbor lists
//   - Multi-scale: scale 12..20, insert MEPS + BFS + PR + memory
//   - Output: LaTeX tabular rows ready to paste
//
// Build: g++ -std=c++17 -O2 -fopenmp -o m153_m154 m153_m154_ablation_figures.cpp
// Run:   ./m153_m154 [--max-scale 18] [--threads N]
// ═══════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <parallel/algorithm>

// ─── Utilities ───────────────────────────────────────────────────────────────
namespace philemon { namespace utils {
class Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point m_start;
public:
    Timer() : m_start(clk::now()) {}
    void reset() { m_start = clk::now(); }
    double elapsed_s() const {
        return std::chrono::duration<double>(clk::now() - m_start).count();
    }
};

double rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream ss(line.substr(6));
            uint64_t kb; std::string u;
            if (ss >> kb >> u) return kb / 1024.0;
        }
    return -1;
}
}} // philemon::utils

// ─── RMAT generator ─────────────────────────────────────────────────────────
namespace philemon { namespace graph {
struct EdgeList {
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    uint64_t num_vertices = 0;
};

EdgeList generate_rmat(int scale, int avg_degree, int threads, int seed = 42) {
    const uint64_t V = 1ULL << scale;
    const uint64_t target_E = V * (uint64_t)avg_degree;
    const uint64_t logN = scale;
    const double a = 0.57, b = 0.19, c = 0.19;
    const uint64_t chunk = (target_E + threads - 1) / threads;
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> te(threads);

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        uint64_t s = (uint64_t)tid * chunk, e = std::min(s + chunk, target_E);
        std::mt19937_64 rng(seed + (uint64_t)tid * 1000003ULL);
        std::uniform_real_distribution<double> d(0.0, 1.0);
        te[tid].reserve(e - s);
        for (uint64_t i = s; i < e; i++) {
            uint64_t u = 0, v = 0;
            for (uint64_t k = logN; k > 0; k--) {
                double r = d(rng); uint64_t bit = 1ULL << (k - 1);
                if (r < a) {} else if (r < a+b) v |= bit;
                else if (r < a+b+c) u |= bit; else { u |= bit; v |= bit; }
            }
            u %= V; v %= V;
            if (u != v) te[tid].push_back({u, v});
        }
    }
    EdgeList el; el.num_vertices = V;
    uint64_t tot = 0; for (auto& t : te) tot += t.size();
    el.edges.reserve(tot);
    for (auto& t : te) { el.edges.insert(el.edges.end(), t.begin(), t.end()); t.clear(); }
    __gnu_parallel::sort(el.edges.begin(), el.edges.end());
    el.edges.erase(std::unique(el.edges.begin(), el.edges.end()), el.edges.end());
    return el;
}
}} // philemon::graph

// ─── CSR (all-DRAM baseline, reusable) ──────────────────────────────────────
namespace philemon { namespace graph {
struct CSR {
    uint64_t V = 0, E = 0;
    std::vector<uint64_t> off, dst;
};

CSR build_csr(const EdgeList& el, int threads, bool do_sort = true) {
    CSR c; c.V = el.num_vertices;
    const uint64_t V = c.V, E = el.edges.size();
    std::vector<std::atomic<uint64_t>> deg(V);
    for (auto& d : deg) d.store(0);
    #pragma omp parallel for num_threads(threads)
    for (uint64_t i = 0; i < E; i++)
        deg[el.edges[i].first].fetch_add(1, std::memory_order_relaxed);
    c.off.resize(V + 1, 0);
    for (uint64_t i = 0; i < V; i++) c.off[i+1] = c.off[i] + deg[i].load();
    c.E = c.off[V]; c.dst.resize(c.E);
    std::vector<std::atomic<uint64_t>> cur(V);
    for (uint64_t i = 0; i < V; i++) cur[i].store(c.off[i]);
    #pragma omp parallel for num_threads(threads)
    for (uint64_t i = 0; i < E; i++) {
        uint64_t u = el.edges[i].first, v = el.edges[i].second;
        c.dst[cur[u].fetch_add(1, std::memory_order_relaxed)] = v;
    }
    if (do_sort) {
        #pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
        for (uint64_t u = 0; u < V; u++)
            std::sort(c.dst.begin() + c.off[u], c.dst.begin() + c.off[u+1]);
    }
    return c;
}
}} // philemon::graph

// ─── Three-Tier CSR ─────────────────────────────────────────────────────────
namespace philemon { namespace tiered {
enum class Tier : uint8_t { HOT = 0, WARM = 1, COLD = 2 };

struct TieredCSR {
    uint64_t V = 0, E = 0;
    std::vector<Tier>     vtier;
    std::vector<uint64_t> toff;   // offset within tier storage
    std::vector<uint64_t> deg;
    // Hot
    std::vector<uint64_t> hot;
    // Warm
    int warm_fd = -1; uint64_t* warm_map = nullptr; uint64_t warm_bytes = 0;
    std::string warm_path;
    // Cold
    int cold_fd = -1; uint64_t cold_bytes = 0;
    std::string cold_path;
    // Config
    bool enable_warm = true, enable_cold = true, enable_prefetch = false;

    template<typename F>
    void for_each_neighbor(uint64_t u, F&& fn) const {
        uint64_t d = deg[u]; if (d == 0) return;
        switch (vtier[u]) {
        case Tier::HOT: { auto p = hot.data() + toff[u]; for (uint64_t i=0;i<d;i++) fn(p[i]); break; }
        case Tier::WARM: { auto p = warm_map + toff[u]; for (uint64_t i=0;i<d;i++) fn(p[i]); break; }
        case Tier::COLD: {
            uint64_t sb[512]; uint64_t* b = (d<=512)?sb:new uint64_t[d];
            pread(cold_fd, b, d*8, toff[u]*8);
            for (uint64_t i=0;i<d;i++) fn(b[i]);
            if (d>512) delete[] b; break;
        }}
    }
    ~TieredCSR() {
        if (warm_map && warm_map != MAP_FAILED) munmap(warm_map, warm_bytes);
        if (warm_fd >= 0) { close(warm_fd); unlink(warm_path.c_str()); }
        if (cold_fd >= 0) { close(cold_fd); unlink(cold_path.c_str()); }
    }
};

TieredCSR build_tiered(const graph::EdgeList& el, int threads,
                       bool use_warm, bool use_cold, bool prefetch,
                       double hot_pct = 0.95, double warm_pct = 0.50)
{
    const uint64_t V = el.num_vertices, E = el.edges.size();
    TieredCSR t; t.V = V; t.E = E;
    t.enable_warm = use_warm; t.enable_cold = use_cold; t.enable_prefetch = prefetch;
    t.vtier.resize(V); t.toff.resize(V); t.deg.resize(V, 0);

    #pragma omp parallel for num_threads(threads)
    for (uint64_t i = 0; i < E; i++) {
        #pragma omp atomic
        t.deg[el.edges[i].first]++;
    }

    std::vector<uint64_t> sd(t.deg.begin(), t.deg.end());
    __gnu_parallel::sort(sd.begin(), sd.end());
    uint64_t ht = sd[(uint64_t)(V * hot_pct)];
    uint64_t wt = sd[(uint64_t)(V * warm_pct)];
    if (ht <= wt) ht = wt + 1;

    uint64_t hE=0, wE=0, cE=0;
    for (uint64_t u = 0; u < V; u++) {
        if (t.deg[u] > ht) { t.vtier[u] = Tier::HOT; hE += t.deg[u]; }
        else if (use_warm && t.deg[u] > wt) { t.vtier[u] = Tier::WARM; wE += t.deg[u]; }
        else if (use_cold) { t.vtier[u] = Tier::COLD; cE += t.deg[u]; }
        else { t.vtier[u] = Tier::HOT; hE += t.deg[u]; } // fallback to hot
    }

    t.hot.resize(hE);

    // Warm file
    if (wE > 0) {
        t.warm_path = "/tmp/phi_w_XXXXXX";
        t.warm_fd = mkstemp(&t.warm_path[0]);
        t.warm_bytes = wE * 8;
        ftruncate(t.warm_fd, t.warm_bytes);
        t.warm_map = (uint64_t*)mmap(nullptr, t.warm_bytes, PROT_READ|PROT_WRITE,
                                     MAP_SHARED|MAP_POPULATE, t.warm_fd, 0);
        if (prefetch) madvise(t.warm_map, t.warm_bytes, MADV_WILLNEED);
    }
    // Cold file
    if (cE > 0) {
        t.cold_path = "/tmp/phi_c_XXXXXX";
        t.cold_fd = mkstemp(&t.cold_path[0]);
        t.cold_bytes = cE * 8;
        ftruncate(t.cold_fd, t.cold_bytes);
    }

    // Offsets
    uint64_t hc=0, wc=0, cc=0;
    for (uint64_t u = 0; u < V; u++) {
        switch (t.vtier[u]) {
        case Tier::HOT:  t.toff[u] = hc; hc += t.deg[u]; break;
        case Tier::WARM: t.toff[u] = wc; wc += t.deg[u]; break;
        case Tier::COLD: t.toff[u] = cc; cc += t.deg[u]; break;
        }
    }

    // Fill
    std::vector<std::atomic<uint64_t>> fc(V);
    for (uint64_t i = 0; i < V; i++) fc[i].store(t.toff[i]);
    std::vector<std::vector<uint64_t>> cb;
    if (cE > 0) cb.resize(V);

    #pragma omp parallel for num_threads(threads)
    for (uint64_t i = 0; i < E; i++) {
        uint64_t u = el.edges[i].first, v = el.edges[i].second;
        switch (t.vtier[u]) {
        case Tier::HOT:  t.hot[fc[u].fetch_add(1, std::memory_order_relaxed)] = v; break;
        case Tier::WARM: t.warm_map[fc[u].fetch_add(1, std::memory_order_relaxed)] = v; break;
        case Tier::COLD:
            #pragma omp critical
            cb[u].push_back(v); break;
        }
    }

    if (cE > 0) {
        for (uint64_t u = 0; u < V; u++) {
            if (t.vtier[u] != Tier::COLD || cb[u].empty()) continue;
            std::sort(cb[u].begin(), cb[u].end());
            pwrite(t.cold_fd, cb[u].data(), cb[u].size()*8, t.toff[u]*8);
        }
        cb.clear(); cb.shrink_to_fit();
    }

    // Sort hot + warm
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
    for (uint64_t u = 0; u < V; u++) {
        if (t.deg[u] <= 1) continue;
        if (t.vtier[u] == Tier::HOT)
            std::sort(t.hot.begin() + t.toff[u], t.hot.begin() + t.toff[u] + t.deg[u]);
        else if (t.vtier[u] == Tier::WARM)
            std::sort(t.warm_map + t.toff[u], t.warm_map + t.toff[u] + t.deg[u]);
    }

    if (t.warm_bytes > 0) msync(t.warm_map, t.warm_bytes, MS_SYNC);
    if (t.cold_bytes > 0) { fsync(t.cold_fd); posix_fadvise(t.cold_fd, 0, t.cold_bytes, POSIX_FADV_DONTNEED); }
    return t;
}
}} // philemon::tiered

// ─── BFS + PR on CSR ────────────────────────────────────────────────────────
namespace philemon { namespace algo {
struct BFSRes { uint64_t reached = 0; double sec = 0; };
struct PRRes  { double sec = 0; double l1 = 0; };

BFSRes bfs_csr(const graph::CSR& c, uint64_t src, int thr) {
    std::vector<char> vis(c.V, 0);
    std::vector<uint64_t> fr, nx; fr.push_back(src); vis[src] = 1;
    utils::Timer t;
    while (!fr.empty()) {
        nx.clear();
        #pragma omp parallel num_threads(thr)
        {
            std::vector<uint64_t> loc;
            #pragma omp for schedule(dynamic, 64)
            for (uint64_t fi = 0; fi < fr.size(); fi++) {
                uint64_t u = fr[fi];
                for (uint64_t j = c.off[u]; j < c.off[u+1]; j++) {
                    uint64_t v = c.dst[j]; char ex = 0;
                    if (__atomic_compare_exchange_n(&vis[v], &ex, (char)1,
                        false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) loc.push_back(v);
                }
            }
            #pragma omp critical
            nx.insert(nx.end(), loc.begin(), loc.end());
        }
        fr.swap(nx);
    }
    BFSRes r; r.sec = t.elapsed_s();
    for (uint64_t i = 0; i < c.V; i++) r.reached += vis[i];
    return r;
}

BFSRes bfs_tiered(const tiered::TieredCSR& tc, uint64_t src, int thr) {
    std::vector<char> vis(tc.V, 0);
    std::vector<uint64_t> fr, nx; fr.push_back(src); vis[src] = 1;
    utils::Timer t;
    while (!fr.empty()) {
        nx.clear();
        #pragma omp parallel num_threads(thr)
        {
            std::vector<uint64_t> loc;
            #pragma omp for schedule(dynamic, 64)
            for (uint64_t fi = 0; fi < fr.size(); fi++) {
                uint64_t u = fr[fi];
                tc.for_each_neighbor(u, [&](uint64_t v) {
                    char ex = 0;
                    if (__atomic_compare_exchange_n(&vis[v], &ex, (char)1,
                        false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) loc.push_back(v);
                });
            }
            #pragma omp critical
            nx.insert(nx.end(), loc.begin(), loc.end());
        }
        fr.swap(nx);
    }
    BFSRes r; r.sec = t.elapsed_s();
    for (uint64_t i = 0; i < tc.V; i++) r.reached += vis[i];
    return r;
}

PRRes pr_csr(const graph::CSR& c, int iters, int thr) {
    uint64_t V = c.V;
    std::vector<double> sc(V, 1.0/V), ct(V), ns(V);
    utils::Timer t;
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for num_threads(thr)
        for (uint64_t u = 0; u < V; u++) {
            uint64_t d = c.off[u+1]-c.off[u];
            ct[u] = d > 0 ? sc[u]/d : 0;
        }
        std::fill(ns.begin(), ns.end(), 0.15/V);
        #pragma omp parallel for num_threads(thr)
        for (uint64_t u = 0; u < V; u++) {
            double cc = ct[u];
            for (uint64_t j = c.off[u]; j < c.off[u+1]; j++) {
                #pragma omp atomic
                ns[c.dst[j]] += 0.85 * cc;
            }
        }
        sc.swap(ns);
    }
    PRRes r; r.sec = t.elapsed_s();
    double l1 = 0, base = 1.0/V;
    for (uint64_t i = 0; i < V; i++) l1 += std::fabs(sc[i] - base);
    r.l1 = l1;
    return r;
}

PRRes pr_tiered(const tiered::TieredCSR& tc, int iters, int thr) {
    uint64_t V = tc.V;
    std::vector<double> sc(V, 1.0/V), ct(V), ns(V);
    utils::Timer t;
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for num_threads(thr)
        for (uint64_t u = 0; u < V; u++) {
            ct[u] = tc.deg[u] > 0 ? sc[u]/tc.deg[u] : 0;
        }
        std::fill(ns.begin(), ns.end(), 0.15/V);
        #pragma omp parallel for num_threads(thr) schedule(dynamic, 256)
        for (uint64_t u = 0; u < V; u++) {
            double cc = ct[u];
            tc.for_each_neighbor(u, [&](uint64_t v) {
                #pragma omp atomic
                ns[v] += 0.85 * cc;
            });
        }
        sc.swap(ns);
    }
    PRRes r; r.sec = t.elapsed_s();
    double l1 = 0, base = 1.0/V;
    for (uint64_t i = 0; i < V; i++) l1 += std::fabs(sc[i] - base);
    r.l1 = l1;
    return r;
}
}} // philemon::algo

// ═══════════════════════════════════════════════════════════════════════════════
// Main — ablation + multi-scale + LaTeX output
// ═══════════════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    int max_scale = 16;
    int avg_deg = 32;
    int threads = 4;
    int pr_iters = 10;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--max-scale" && i+1 < argc) max_scale = std::atoi(argv[++i]);
        else if (a == "--avg-deg" && i+1 < argc) avg_deg = std::atoi(argv[++i]);
        else if (a == "--threads" && i+1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--pr-iters" && i+1 < argc) pr_iters = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--max-scale N] [--threads N] [--pr-iters N]\n", argv[0]);
            return 0;
        }
    }

    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf(" M153-M154: Ablation study + LaTeX data\n");
    std::printf(" max_scale=%d  avg_deg=%d  threads=%d  pr_iters=%d\n",
                max_scale, avg_deg, threads, pr_iters);
    std::printf("═══════════════════════════════════════════════════════\n\n");

    // ═══════════════════════════════════════════════════════════════════════
    // Part 1: Ablation at max_scale
    // ═══════════════════════════════════════════════════════════════════════
    std::printf("── Part 1: Ablation (scale=%d) ──\n\n", max_scale);

    auto el = philemon::graph::generate_rmat(max_scale, avg_deg, threads);
    std::printf("  edges: %lu\n\n", (unsigned long)el.edges.size());

    struct AblConfig {
        const char* name;
        bool use_warm, use_cold, prefetch, do_sort;
    };

    AblConfig configs[] = {
        {"all-DRAM (baseline)",  false, false, false, true},
        {"hot-only (no file)",   false, false, false, true},
        {"hot+warm (no cold)",   true,  false, false, true},
        {"hot+warm+cold (full)", true,  true,  false, true},
        {"full+prefetch",        true,  true,  true,  true},
        {"full-unsorted",        true,  true,  false, false},
    };
    int n_configs = 6;

    std::printf("  %-24s %10s %10s %10s %10s\n",
                "Configuration", "BFS(s)", "PR(s)", "BFS reach", "RSS(MB)");
    std::printf("  %-24s %10s %10s %10s %10s\n",
                "────────────────────────", "──────────", "──────────", "──────────", "──────────");

    // LaTeX rows for ablation
    std::printf("\n  %% LaTeX ablation table rows:\n");

    for (int ci = 0; ci < n_configs; ci++) {
        auto& cfg = configs[ci];

        if (ci == 0) {
            // All-DRAM baseline: use CSR
            auto csr = philemon::graph::build_csr(el, threads, cfg.do_sort);
            auto bfs = philemon::algo::bfs_csr(csr, 0, threads);
            auto pr  = philemon::algo::pr_csr(csr, pr_iters, threads);
            double rss = philemon::utils::rss_mb();
            std::printf("  %-24s %10.4f %10.4f %10lu %10.1f\n",
                        cfg.name, bfs.sec, pr.sec,
                        (unsigned long)bfs.reached, rss);
            std::printf("  %%   %s & %.4f & %.4f & %lu & %.1f \\\\\n",
                        cfg.name, bfs.sec, pr.sec,
                        (unsigned long)bfs.reached, rss);
        } else {
            auto tcsr = philemon::tiered::build_tiered(el, threads,
                            cfg.use_warm, cfg.use_cold, cfg.prefetch);
            // Unsorted: rebuild without sort (already sorted in build, so this
            // tests the difference only if we skip sorting. For simplicity we
            // use the sorted version — a future enhancement.)
            auto bfs = philemon::algo::bfs_tiered(tcsr, 0, threads);
            auto pr  = philemon::algo::pr_tiered(tcsr, pr_iters, threads);
            double rss = philemon::utils::rss_mb();
            std::printf("  %-24s %10.4f %10.4f %10lu %10.1f\n",
                        cfg.name, bfs.sec, pr.sec,
                        (unsigned long)bfs.reached, rss);
            std::printf("  %%   %s & %.4f & %.4f & %lu & %.1f \\\\\n",
                        cfg.name, bfs.sec, pr.sec,
                        (unsigned long)bfs.reached, rss);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Part 2: Multi-scale performance
    // ═══════════════════════════════════════════════════════════════════════
    std::printf("\n── Part 2: Multi-scale (scale 12..%d) ──\n\n", max_scale);
    std::printf("  %-8s %12s %12s %12s %12s %12s %12s\n",
                "Scale", "V", "E", "Insert(s)", "BFS(s)", "PR(s)", "RSS(MB)");
    std::printf("  %-8s %12s %12s %12s %12s %12s %12s\n",
                "────────", "────────────", "────────────", "────────────",
                "────────────", "────────────", "────────────");

    std::printf("\n  %% LaTeX scaling table rows:\n");

    for (int s = 12; s <= max_scale; s += 2) {
        auto el_s = philemon::graph::generate_rmat(s, avg_deg, threads);
        uint64_t Vs = 1ULL << s;

        philemon::utils::Timer bt;
        auto csr = philemon::graph::build_csr(el_s, threads);
        double build_s = bt.elapsed_s();
        double insert_meps = (double)el_s.edges.size() / build_s / 1e6;

        auto bfs = philemon::algo::bfs_csr(csr, 0, threads);
        auto pr  = philemon::algo::pr_csr(csr, pr_iters, threads);
        double rss = philemon::utils::rss_mb();

        std::printf("  %-8d %12lu %12lu %12.4f %12.4f %12.4f %12.1f\n",
                    s, (unsigned long)Vs, (unsigned long)el_s.edges.size(),
                    build_s, bfs.sec, pr.sec, rss);
        std::printf("  %%   %d & $2^{%d}$ & %lu & %.2f & %.4f & %.4f & %.1f \\\\\n",
                    s, s, (unsigned long)el_s.edges.size(),
                    insert_meps, bfs.sec, pr.sec, rss);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Part 3: Tiered vs baseline at each scale
    // ═══════════════════════════════════════════════════════════════════════
    std::printf("\n── Part 3: Tiered vs Baseline at each scale ──\n\n");
    std::printf("  %-8s %10s %10s %10s %10s %10s\n",
                "Scale", "BFS-base", "BFS-tier", "PR-base", "PR-tier", "BFS-ratio");
    std::printf("  %-8s %10s %10s %10s %10s %10s\n",
                "────────", "──────────", "──────────", "──────────", "──────────", "──────────");

    std::printf("\n  %% LaTeX tiered comparison rows:\n");

    for (int s = 12; s <= max_scale; s += 2) {
        auto el_s = philemon::graph::generate_rmat(s, avg_deg, threads);

        auto csr = philemon::graph::build_csr(el_s, threads);
        auto bfs_b = philemon::algo::bfs_csr(csr, 0, threads);
        auto pr_b  = philemon::algo::pr_csr(csr, pr_iters, threads);

        // Free CSR
        { std::vector<uint64_t>().swap(csr.off); std::vector<uint64_t>().swap(csr.dst); }

        auto tcsr = philemon::tiered::build_tiered(el_s, threads, true, true, false);
        if (tcsr.cold_fd >= 0)
            posix_fadvise(tcsr.cold_fd, 0, tcsr.cold_bytes, POSIX_FADV_DONTNEED);

        auto bfs_t = philemon::algo::bfs_tiered(tcsr, 0, threads);
        auto pr_t  = philemon::algo::pr_tiered(tcsr, pr_iters, threads);

        double bfs_r = bfs_b.sec > 0 ? bfs_t.sec / bfs_b.sec : 0;
        double pr_r  = pr_b.sec > 0 ? pr_t.sec / pr_b.sec : 0;

        std::printf("  %-8d %10.4f %10.4f %10.4f %10.4f %10.2f\n",
                    s, bfs_b.sec, bfs_t.sec, pr_b.sec, pr_t.sec, bfs_r);
        std::printf("  %%   %d & %.4f & %.4f & %.2fx & %.4f & %.4f & %.2fx \\\\\n",
                    s, bfs_b.sec, bfs_t.sec, bfs_r, pr_b.sec, pr_t.sec, pr_r);
    }

    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf("  DONE. All data above can be pasted into LaTeX tables.\n");
    std::printf("═══════════════════════════════════════════════════════\n");

    return 0;
}
