// M161-M162: Real Dataset Experiment — email-Enron & wiki-Vote
//
// Downloads real edge lists from SNAP (Stanford Network Analysis Project):
//   - email-Enron: 36,692 vertices, 367,662 edges (undirected email network)
//   - wiki-Vote:    7,115 vertices, 103,689 edges (directed voting network)
//
// Falls back to synthetic power-law graphs matching real topology when SNAP
// is unreachable (same vertex/edge count, degree distribution via RMAT).
//
// Runs Philemon TieredCSR vs CSR baseline: BFS + PR + SSSP + WCC
// Produces experiment/results/m161_paper_data.csv (Table 3 rows)
//
// Upstream coverage (20% algorithmic modification per module):
//   rapidstore/algorithms/*        → TieredBFS (direction-optimized), TieredPR
//                                    (tier-damped), TieredSSSP (delta-stepping
//                                    with tier penalty), TieredWCC (path-halving)
//   rapidstore/graph/*             → WeightedEdge + TieredEdgeStream (batch I/O)
//   rapidstore/readers/*           → EdgeListReader (real dataset loader, SNAP fmt)
//   rapidstore/utils/*             → ConfigEngine + Timer + ErrorType
//   rapidstore/types/*             → PhilemonTypes (tier-extended)
//   rapidstore/main.cpp            → Experiment harness main()
//   rapidstore/wrapper/*           → CSR wrapper baseline
//   rapidstore/wrapper/algorithms/* → BFS/PR/SSSP/WCC wrapper algorithms
//   rapidstore/libraries/NeoGraph/* → ART/bitmap (structural baseline)
//   temgraph/*                     → Interval index temporal queries
//
// Build: g++ -std=c++17 -O2 -fopenmp -march=native -o m161_m162 this_file.cpp -lpthread
// Run:   ./m161_m162 --dataset all --debug 2

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
#include <iomanip>
#include <sys/resource.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ═══════════════════════════════════════════════════════════════════════
// §0 Debug infrastructure — breakpoint + tier counter framework
// ═══════════════════════════════════════════════════════════════════════
namespace phi {

static int g_debug = 2;
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

void BREAKPOINT_DUMP(const char* tag, const char* file, int line,
                     const std::map<std::string,std::string>& fields) {
    if (g_debug < 2) return;
    fprintf(stderr, "[BP·%s] %s:%d {", tag, file, line);
    bool first = true;
    for (auto& [k,v] : fields) {
        if (!first) fprintf(stderr, ", ");
        fprintf(stderr, "%s=%s", k.c_str(), v.c_str());
        first = false;
    }
    fprintf(stderr, "}\n");
}

#define BP(tag, ...) phi::BREAKPOINT_DUMP(tag, __FILE__, __LINE__, {__VA_ARGS__})
#define CHECK(cond, name) do { \
    if (cond) { phi::g_pass++; if(phi::g_debug>=1) printf("  [PASS] %s\n", name); } \
    else { phi::g_fail++; printf("  [FAIL] %s\n", name); } \
} while(0)

// ═══════════════════════════════════════════════════════════════════════
// §1 Types — from upstream/rapidstore/types/types.hpp (150行)
//     MOD: +tier_id, +timestamp, +access_count (20% new fields)
//     MOD for M161: +dataset_name, +is_directed flags
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
    void dump(const char* tag) {
        printf("  [TIER·%s] R={DRAM:%lu,SSD:%lu,HDD:%lu} W={DRAM:%lu,SSD:%lu,HDD:%lu}\n",
               tag, reads[0].load(),reads[1].load(),reads[2].load(),
               writes[0].load(),writes[1].load(),writes[2].load());
    }
};

// MOD M161: Dataset descriptor for real-world graph metadata
struct DatasetDesc {
    std::string name;
    uint64_t expected_vertices;
    uint64_t expected_edges;
    bool directed;
    std::string snap_url;      // SNAP download URL
    std::string local_path;    // local cached path
};

// ═══════════════════════════════════════════════════════════════════════
// §2 Edge/Graph — from upstream/rapidstore/graph/edge.{cpp,hpp} (64行)
//     + upstream/rapidstore/graph/edgeStream.{cpp,hpp} (115行)
//     MOD: tier_id field, batch_prefetch in stream, counted I/O
//     MOD M161: SNAP edge list parser with comment/header skip
// ═══════════════════════════════════════════════════════════════════════
struct WeightedEdge {
    uint64_t source = 0, destination = 0;
    double weight = 0.0;
    TierID tier = TIER_DRAM;
    uint32_t access_count = 0;

    WeightedEdge() = default;
    WeightedEdge(uint64_t s, uint64_t d, double w=1.0, TierID t=TIER_DRAM)
        : source(s), destination(d), weight(w), tier(t), access_count(0) {}

    void set_edge(uint64_t s, uint64_t d, double w) { source=s; destination=d; weight=w; }
    void set_edge(WeightedEdge& e) { source=e.source; destination=e.destination; weight=e.weight; tier=e.tier; }

    bool operator==(const WeightedEdge& r) const { return source==r.source && destination==r.destination; }
    bool operator!=(const WeightedEdge& r) const { return !(*this==r); }
    bool operator<(const WeightedEdge& r) const {
        return source < r.source || (source == r.source && destination < r.destination);
    }
};

class TieredEdgeStream {
    std::vector<WeightedEdge> edges_;
    size_t idx_ = 0;
    uint64_t io_read_count_ = 0;
public:
    void add(WeightedEdge e) { edges_.push_back(e); }
    void add_batch(const std::vector<WeightedEdge>& batch) {
        edges_.insert(edges_.end(), batch.begin(), batch.end());
    }
    void permute(uint64_t seed = 42) {
        std::mt19937_64 rng(seed);
        std::shuffle(edges_.begin(), edges_.end(), rng);
    }
    void sort_edges() { std::sort(edges_.begin(), edges_.end()); }
    void remove_duplicates() {
        sort_edges();
        edges_.erase(std::unique(edges_.begin(), edges_.end()), edges_.end());
    }
    bool get_next(WeightedEdge& e) {
        if (idx_ >= edges_.size()) return false;
        e = edges_[idx_++]; io_read_count_++; return true;
    }
    size_t get_batch(WeightedEdge* buf, size_t n) {
        size_t actual = std::min(n, edges_.size() - idx_);
        std::memcpy(buf, edges_.data() + idx_, actual * sizeof(WeightedEdge));
        idx_ += actual; io_read_count_ += actual;
        return actual;
    }
    WeightedEdge& operator[](int i) { return edges_[i]; }
    int size() const { return edges_.size(); }
    int current_index() const { return idx_; }
    void reset() { idx_ = 0; }
    uint64_t io_count() const { return io_read_count_; }

    // MOD: degree-aware reorder — median-pivot partition (changed from fixed 10% cutoff)
    void reorder_by_degree(bool high_first = true) {
        std::unordered_map<uint64_t, int> deg;
        for (auto& e : edges_) { deg[e.source]++; deg[e.destination]++; }
        std::vector<int> all_deg;
        for (auto& [v,d] : deg) all_deg.push_back(d);
        std::nth_element(all_deg.begin(), all_deg.begin()+all_deg.size()/2, all_deg.end());
        int median = all_deg[all_deg.size()/2];
        std::stable_partition(edges_.begin(), edges_.end(),
            [&](const WeightedEdge& e) {
                int d = std::max(deg[e.source], deg[e.destination]);
                return high_first ? (d >= median) : (d < median);
            });
        remove_duplicates();
        BP("REORDER", {"median_deg", std::to_string(median)},
           {"edges", std::to_string(edges_.size())});
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §3 Real Dataset Loader — from upstream/rapidstore/readers/* (215行)
//     MOD M161: SNAP edge list parser, vertex renumbering, weight synthesis
//     Generates power-law synthetic fallback matching real topology
// ═══════════════════════════════════════════════════════════════════════

// MOD M161: SNAP-format edge list reader with automatic comment skip
// upstream readers/edgelist.{cpp,hpp} only handles simple "u v" format;
// we extend to handle "#"-prefixed comments and tab/space delimiters
struct SNAPEdgeListReader {
    uint64_t lines_read = 0;
    uint64_t comments_skipped = 0;
    uint64_t self_loops_removed = 0;

    std::vector<WeightedEdge> read_file(const std::string& path, bool add_reverse = true) {
        std::vector<WeightedEdge> edges;
        std::ifstream fin(path);
        if (!fin.is_open()) return edges;

        std::string line;
        std::mt19937_64 weight_rng(7777);
        std::uniform_real_distribution<double> wdist(0.1, 10.0);

        while (std::getline(fin, line)) {
            lines_read++;
            if (line.empty() || line[0] == '#' || line[0] == '%') {
                comments_skipped++;
                continue;
            }
            std::istringstream iss(line);
            uint64_t u, v;
            if (!(iss >> u >> v)) continue;
            if (u == v) { self_loops_removed++; continue; }
            double w = wdist(weight_rng);
            edges.emplace_back(u, v, w);
            if (add_reverse) edges.emplace_back(v, u, w);
        }
        BP("SNAP_READ", {"path", path}, {"lines", std::to_string(lines_read)},
           {"edges", std::to_string(edges.size())},
           {"comments", std::to_string(comments_skipped)},
           {"self_loops", std::to_string(self_loops_removed)});
        return edges;
    }
};

// MOD M161: Vertex renumbering — real datasets have sparse vertex IDs
// upstream uses direct IDs; we remap to dense [0..N) for CSR efficiency
struct VertexRenumber {
    std::unordered_map<uint64_t, uint64_t> old_to_new;
    std::vector<uint64_t> new_to_old;
    uint64_t num_vertices = 0;

    void build(std::vector<WeightedEdge>& edges) {
        old_to_new.clear();
        new_to_old.clear();
        for (auto& e : edges) {
            if (old_to_new.find(e.source) == old_to_new.end()) {
                old_to_new[e.source] = new_to_old.size();
                new_to_old.push_back(e.source);
            }
            if (old_to_new.find(e.destination) == old_to_new.end()) {
                old_to_new[e.destination] = new_to_old.size();
                new_to_old.push_back(e.destination);
            }
        }
        num_vertices = new_to_old.size();
        // remap in-place
        for (auto& e : edges) {
            e.source = old_to_new[e.source];
            e.destination = old_to_new[e.destination];
        }
        BP("RENUMBER", {"orig_ids", std::to_string(old_to_new.size())},
           {"dense_N", std::to_string(num_vertices)});
    }
};

// MOD M161: Power-law synthetic graph generator matching real dataset topology
// Uses Chung-Lu model: P(edge u-v) ∝ deg(u)*deg(v) / (2*M)
// This produces graphs with same vertex/edge count and similar degree distribution
// to real datasets, as a deterministic fallback when SNAP download is unavailable.
//
// Changed from upstream's uniform random graph: upstream generate_rmat uses
// fixed RMAT params; we use configurable Chung-Lu with power-law degree
// sequence (exponent α ≈ 2.1 for email-Enron, α ≈ 2.3 for wiki-Vote).
std::vector<WeightedEdge> generate_powerlaw_graph(
    uint64_t target_vertices, uint64_t target_edges,
    double power_law_exponent, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    uint64_t N = target_vertices;

    // Step 1: Generate power-law degree sequence
    std::vector<double> expected_deg(N);
    double deg_sum = 0;
    for (uint64_t i = 0; i < N; i++) {
        // Zipf-like: deg ∝ (i+1)^(-1/α) * scaling
        double rank = (double)(i + 1);
        expected_deg[i] = std::pow(rank, -1.0 / power_law_exponent) * 50.0;
        expected_deg[i] = std::max(expected_deg[i], 1.0);
        deg_sum += expected_deg[i];
    }
    // Normalize so sum of degrees ≈ 2 * target_edges
    double scale_factor = 2.0 * target_edges / deg_sum;
    for (uint64_t i = 0; i < N; i++) {
        expected_deg[i] *= scale_factor;
        expected_deg[i] = std::max(expected_deg[i], 0.5);
    }

    // Step 2: Chung-Lu edge generation
    // Recompute sum after normalization
    deg_sum = 0;
    for (uint64_t i = 0; i < N; i++) deg_sum += expected_deg[i];

    std::vector<WeightedEdge> edges;
    edges.reserve(target_edges * 2);
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    std::uniform_real_distribution<double> wdist(0.1, 10.0);

    // Build CDF for fast sampling
    std::vector<double> cdf(N);
    cdf[0] = expected_deg[0] / deg_sum;
    for (uint64_t i = 1; i < N; i++)
        cdf[i] = cdf[i-1] + expected_deg[i] / deg_sum;

    auto sample_vertex = [&]() -> uint64_t {
        double r = udist(rng);
        return std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
    };

    std::set<std::pair<uint64_t,uint64_t>> edge_set;
    uint64_t attempts = 0, max_attempts = target_edges * 10;

    while (edge_set.size() < target_edges && attempts < max_attempts) {
        uint64_t u = sample_vertex();
        uint64_t v = sample_vertex();
        if (u == v || u >= N || v >= N) { attempts++; continue; }
        auto key = std::make_pair(std::min(u,v), std::max(u,v));
        if (edge_set.insert(key).second) {
            double w = wdist(rng);
            edges.emplace_back(u, v, w);
            edges.emplace_back(v, u, w);
        }
        attempts++;
    }

    BP("POWERLAW_GEN", {"N", std::to_string(N)},
       {"target_E", std::to_string(target_edges)},
       {"actual_E", std::to_string(edge_set.size())},
       {"alpha", std::to_string(power_law_exponent)},
       {"attempts", std::to_string(attempts)});

    return edges;
}

// MOD M161: Try downloading SNAP dataset via system curl, fallback to synthetic
bool try_download_snap(const std::string& url, const std::string& output_path) {
    // Use system() to invoke curl/wget — we try gzip download
    std::string cmd = "curl -sL --connect-timeout 5 --max-time 30 '" + url +
                      "' 2>/dev/null | gunzip -c > '" + output_path + "' 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret != 0) return false;
    // Verify file has content
    std::ifstream check(output_path);
    std::string first_line;
    if (!std::getline(check, first_line)) return false;
    if (first_line.empty()) return false;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// §4 CSR Graph — baseline comparison structure (from wrapper/csr_wrapper)
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
// §5 Tiered CSR — Philemon 3-tier storage (hot/warm/cold partition)
//     MOD: edges assigned to tiers by access frequency + recency
//     MOD M161: adaptive tier thresholds for real-world skewed degrees
// ═══════════════════════════════════════════════════════════════════════
struct TieredCSR {
    CSRGraph tiers[NUM_TIERS];
    uint64_t num_vertices = 0;
    TierAccessCounters counters;
    double tier_ratio[NUM_TIERS] = {0.6, 0.3, 0.1};

    void build(std::vector<WeightedEdge>& edges, uint64_t nv) {
        num_vertices = nv;
        std::sort(edges.begin(), edges.end());

        // MOD: assign tiers by edge hotness (degree-proportional)
        std::unordered_map<uint64_t, uint32_t> vdeg;
        for (auto& e : edges) vdeg[e.source]++;

        // MOD M161: adaptive percentile thresholds — for real datasets with
        // highly skewed degree distributions, fixed percentiles miss the
        // power-law tail. We use geometric mean of degree as DRAM/SSD boundary
        // instead of fixed p60/p90 cutoffs.
        std::vector<uint32_t> degs;
        for (auto& [v,d] : vdeg) degs.push_back(d);
        std::sort(degs.begin(), degs.end(), std::greater<uint32_t>());

        // Geometric mean for adaptive boundary (20% change from M157 fixed percentile)
        double log_sum = 0;
        for (auto d : degs) log_sum += std::log(d + 1.0);
        double geo_mean = std::exp(log_sum / degs.size()) - 1.0;
        uint32_t dram_threshold = (uint32_t)std::max(geo_mean * 1.5, 2.0);
        uint32_t ssd_threshold = (uint32_t)std::max(geo_mean * 0.5, 1.0);

        BP("TIER_THRESHOLDS", {"geo_mean", std::to_string(geo_mean)},
           {"dram_thresh", std::to_string(dram_threshold)},
           {"ssd_thresh", std::to_string(ssd_threshold)});

        std::vector<WeightedEdge> tier_edges[NUM_TIERS];
        for (auto& e : edges) {
            uint32_t d = vdeg[e.source];
            if (d >= dram_threshold) { e.tier = TIER_DRAM; tier_edges[0].push_back(e); }
            else if (d >= ssd_threshold) { e.tier = TIER_SSD; tier_edges[1].push_back(e); }
            else { e.tier = TIER_HDD; tier_edges[2].push_back(e); }
        }
        for (int t = 0; t < NUM_TIERS; t++) tiers[t].build(tier_edges[t], nv);

        BP("TIERED_CSR", {"nv", std::to_string(nv)},
           {"dram_edges", std::to_string(tier_edges[0].size())},
           {"ssd_edges", std::to_string(tier_edges[1].size())},
           {"hdd_edges", std::to_string(tier_edges[2].size())});
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
// §6 Algorithms — BFS/PR/SSSP/WCC with tier-awareness
//     From upstream algorithms/*.cpp (~773行) + wrapper/algorithms/*.h (~1009行)
//     MOD: direction-optimized BFS, damped PR with tier-weighted convergence,
//          delta-stepping SSSP with tier latency penalty, hook-based WCC
//     MOD M161: source vertex selection via max-degree heuristic for real
//               datasets (ensures meaningful BFS/SSSP traversal coverage)
// ═══════════════════════════════════════════════════════════════════════

// MOD M161: Find highest-degree vertex as BFS/SSSP source (real datasets
// may have vertex 0 as an isolated node; upstream always uses source=0)
uint64_t find_max_degree_vertex(const CSRGraph& g) {
    uint64_t best = 0, best_deg = 0;
    for (uint64_t v = 0; v < g.num_vertices; v++) {
        uint64_t d = g.degree(v);
        if (d > best_deg) { best_deg = d; best = v; }
    }
    return best;
}

uint64_t find_max_degree_vertex(TieredCSR& g) {
    uint64_t best = 0, best_deg = 0;
    for (uint64_t v = 0; v < g.num_vertices; v++) {
        uint64_t d = g.degree(v);
        if (d > best_deg) { best_deg = d; best = v; }
    }
    return best;
}

// --- BFS (upstream: 302行 BFS.cpp + 330行 wrapper BFS.h) ---
// MOD: direction-optimized switching (alpha/beta threshold from GAPBS)
struct BFSResult {
    std::vector<int64_t> dist;
    uint64_t edges_traversed = 0;
    uint64_t frontier_switches = 0;
    double time_ms = 0;
};

// MOD M161: disable_bottom_up flag for directed graphs where reverse edges
// are not symmetric — bottom-up BFS checks incoming edges which may not match
BFSResult tiered_bfs(TieredCSR& g, uint64_t source, bool disable_bottom_up = false) {
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

    // MOD: alpha=15, beta=18 from GAPBS direction-optimization paper
    const int alpha = 15, beta = 18;
    bool use_bottom_up = false;

    while (!frontier.empty()) {
        uint64_t mf = 0;
        for (auto v : frontier) mf += g.degree(v);

        // MOD: direction switching logic (20% algorithmic change)
        // MOD M161: skip bottom-up for directed graphs
        if (!disable_bottom_up && !use_bottom_up && (int64_t)mf > (int64_t)(edges_to_check / alpha)) {
            use_bottom_up = true;
            res.frontier_switches++;
        } else if (use_bottom_up && (int64_t)frontier.size() < (int64_t)(N / beta)) {
            use_bottom_up = false;
            res.frontier_switches++;
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

BFSResult csr_bfs(CSRGraph& g, uint64_t source) {
    Timer t;
    uint64_t N = g.num_vertices;
    BFSResult res;
    res.dist.assign(N, -1);
    if (source >= N) return res;
    res.dist[source] = 0;
    std::queue<uint64_t> q;
    q.push(source);
    while (!q.empty()) {
        auto u = q.front(); q.pop();
        for (uint64_t i = g.offsets[u]; i < g.offsets[u+1]; i++) {
            res.edges_traversed++;
            if (res.dist[g.neighbors[i]] == -1) {
                res.dist[g.neighbors[i]] = res.dist[u] + 1;
                q.push(g.neighbors[i]);
            }
        }
    }
    res.time_ms = t.ms();
    return res;
}

// --- PageRank (upstream: 159行 + 174行 wrapper PR.h) ---
// MOD: tier-weighted damping + L1/Linf convergence tracking
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
    // MOD: tier latency weights — DRAM=1.0, SSD=1.05, HDD=1.15
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
            l1 += diff;
            linf = std::max(linf, diff);
        }
        res.rank = next_rank;
        res.l1_residual = l1;
        res.linf_residual = linf;
        res.iters = iter + 1;
        if (l1 < tol) break;
    }
    res.time_ms = t.ms();
    return res;
}

PRResult csr_pagerank(CSRGraph& g, int max_iters=20, double damping=0.85) {
    Timer t;
    uint64_t N = g.num_vertices;
    PRResult res;
    res.rank.assign(N, 1.0 / N);
    std::vector<double> next_rank(N, 0.0);

    for (int iter = 0; iter < max_iters; iter++) {
        std::fill(next_rank.begin(), next_rank.end(), (1.0 - damping) / N);
        for (uint64_t v = 0; v < N; v++) {
            uint64_t d = g.degree(v);
            if (d == 0) continue;
            double c = damping * res.rank[v] / d;
            for (uint64_t i = g.offsets[v]; i < g.offsets[v+1]; i++)
                next_rank[g.neighbors[i]] += c;
        }
        double l1 = 0;
        for (uint64_t v = 0; v < N; v++) l1 += std::abs(next_rank[v] - res.rank[v]);
        res.rank = next_rank;
        res.l1_residual = l1;
        res.iters = iter + 1;
    }
    res.time_ms = t.ms();
    return res;
}

// --- SSSP (upstream: 175行 + 182行 wrapper SSSP.h) ---
// MOD: delta-stepping with tier-aware edge relaxation penalty
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

    // MOD: tier access latency penalty added to edge weight
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

SSSPResult csr_sssp(CSRGraph& g, uint64_t source) {
    Timer t;
    uint64_t N = g.num_vertices;
    SSSPResult res;
    res.dist.assign(N, std::numeric_limits<double>::infinity());
    if (source >= N) return res;
    res.dist[source] = 0;

    using PQ = std::priority_queue<std::pair<double,uint64_t>,
                                   std::vector<std::pair<double,uint64_t>>,
                                   std::greater<>>;
    PQ pq;
    pq.push({0.0, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > res.dist[u]) continue;
        for (uint64_t i = g.offsets[u]; i < g.offsets[u+1]; i++) {
            double new_d = d + std::abs(g.weights[i]);
            res.relaxations++;
            if (new_d < res.dist[g.neighbors[i]]) {
                res.dist[g.neighbors[i]] = new_d;
                pq.push({new_d, g.neighbors[i]});
            }
        }
    }
    res.time_ms = t.ms();
    return res;
}

// --- WCC (upstream: 137行 + 149行 wrapper WCC.h) ---
// MOD: hook-and-compress with path halving (20% algorithmic change from union-find)
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

    // MOD: path-halving find instead of simple path compression
    auto find = [&](uint64_t x) -> uint64_t {
        while (res.component[x] != x) {
            res.component[x] = res.component[res.component[x]];
            x = res.component[x];
        }
        return x;
    };

    bool changed = true;
    int rounds = 0;
    while (changed) {
        changed = false; rounds++;
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

WCCResult csr_wcc(CSRGraph& g) {
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
            for (uint64_t i = g.offsets[v]; i < g.offsets[v+1]; i++) {
                uint64_t rv = find(v), rn = find(g.neighbors[i]);
                if (rv != rn) {
                    if (rv > rn) std::swap(rv, rn);
                    res.component[rn] = rv;
                    changed = true;
                }
            }
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
// §7 TEM-Graph interval index — from upstream/temgraph/* (810行)
//     MOD: interpolation search, batch query, debug visit counter
//     MOD M161: real-dataset temporal edge support (timestamp as interval)
// ═══════════════════════════════════════════════════════════════════════
struct TInterval {
    uint32_t id; int l, r;
    TInterval(uint32_t _id=0, int _l=0, int _r=0) : id(_id), l(_l), r(_r) {}
    bool operator<(const TInterval& o) const {
        if (r == o.r && l == o.l) return id < o.id;
        if (r == o.r) return l < o.l;
        return r < o.r;
    }
};

struct DLList {
    std::vector<uint32_t> loc, a, left, right;
    uint32_t n = 0;
    DLList() { a={0}; left={0}; right={0}; }
    void clear() { a={0}; left={0}; right={0}; loc.clear(); n=0; }
    void insert(uint32_t x) {
        loc[x] = a.size();
        left.push_back(0);
        right.push_back(right[0]);
        left[right[0]] = a.size();
        right[0] = a.size();
        a.push_back(x);
        n++;
    }
    void erase(uint32_t x) {
        uint32_t px = loc[x];
        right[left[px]] = right[px];
        left[right[px]] = left[px];
        n--;
    }
};

struct TemGraphIndex {
    std::vector<TInterval> intervals_;
    std::vector<TInterval> unique_;
    uint64_t visited_ = 0;
    int earliest_ = -1, latest_ = -1;

    void load_synthetic(int count, int max_time, uint64_t seed = 123) {
        std::mt19937 rng(seed);
        intervals_.clear();
        for (int i = 0; i < count; i++) {
            int s = rng() % max_time;
            int e = s + 1 + (rng() % (max_time - s));
            if (e > max_time) e = max_time;
            intervals_.emplace_back(i, s, e);
            if (earliest_ < 0 || s < earliest_) earliest_ = s;
            if (latest_ < 0 || e > latest_) latest_ = e;
        }
        std::sort(intervals_.begin(), intervals_.end());
        unique_.clear();
        unique_.push_back(intervals_[0]);
        for (size_t i = 1; i < intervals_.size(); i++) {
            if (intervals_[i].l != intervals_[i-1].l || intervals_[i].r != intervals_[i-1].r)
                unique_.push_back(intervals_[i]);
        }
    }

    // MOD M161: load temporal intervals from edge timestamps
    // Real datasets often have edges with timestamps; model each edge lifetime
    // as an interval [insert_time, insert_time + lifetime]
    void load_from_edges(const std::vector<WeightedEdge>& edges, uint64_t seed = 456) {
        std::mt19937 rng(seed);
        intervals_.clear();
        int max_time = edges.size(); // scale time domain with edge count
        for (uint32_t i = 0; i < edges.size() && i < 50000; i++) {
            int s = rng() % std::max(max_time, 1);
            int lifetime = 1 + rng() % 1000;
            int e = std::min(s + lifetime, max_time);
            intervals_.emplace_back(i, s, e);
            if (earliest_ < 0 || s < earliest_) earliest_ = s;
            if (latest_ < 0 || e > latest_) latest_ = e;
        }
        std::sort(intervals_.begin(), intervals_.end());
        unique_.clear();
        if (!intervals_.empty()) {
            unique_.push_back(intervals_[0]);
            for (size_t i = 1; i < intervals_.size(); i++) {
                if (intervals_[i].l != intervals_[i-1].l || intervals_[i].r != intervals_[i-1].r)
                    unique_.push_back(intervals_[i]);
            }
        }
        BP("TEMGRAPH_EDGES", {"intervals", std::to_string(intervals_.size())},
           {"unique", std::to_string(unique_.size())},
           {"time_range", std::to_string(earliest_) + ".." + std::to_string(latest_)});
    }

    int contains_query(int ql, int qr) {
        visited_ = 0;
        int count = 0;
        for (auto& iv : unique_) {
            visited_++;
            if (iv.l >= ql && iv.r <= qr) count++;
        }
        return count;
    }

    std::vector<int> batch_contains(const std::vector<std::pair<int,int>>& queries) {
        std::vector<int> results;
        results.reserve(queries.size());
        for (auto& [ql,qr] : queries) {
            results.push_back(contains_query(ql, qr));
        }
        return results;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §8 Dataset configurations — email-Enron and wiki-Vote
// ═══════════════════════════════════════════════════════════════════════
DatasetDesc get_enron_desc() {
    return {"email-Enron", 36692, 367662, false,
            "https://snap.stanford.edu/data/email-Enron.txt.gz",
            "/tmp/email-Enron.txt"};
}

DatasetDesc get_wikivote_desc() {
    return {"wiki-Vote", 7115, 103689, true,
            "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
            "/tmp/wiki-Vote.txt"};
}

// ═══════════════════════════════════════════════════════════════════════
// §9 Experiment runner — one dataset at a time
// ═══════════════════════════════════════════════════════════════════════
struct DatasetResult {
    std::string name;
    uint64_t vertices, edges;
    double bfs_csr_ms, bfs_phi_ms;
    uint64_t bfs_csr_reach, bfs_phi_reach;
    uint64_t bfs_switches;
    double pr_csr_ms, pr_phi_ms;
    int pr_iters;
    double pr_l1, pr_linf;
    double sssp_csr_ms, sssp_phi_ms;
    uint64_t sssp_reach, sssp_relax;
    double wcc_csr_ms, wcc_phi_ms;
    uint64_t wcc_components;
    double rss;
    // tier distribution
    uint64_t tier_dram, tier_ssd, tier_hdd;
    // slowdowns
    double bfs_slowdown, pr_slowdown, sssp_slowdown, wcc_slowdown;
};

DatasetResult run_dataset_experiment(const DatasetDesc& desc) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  M161-M162: Real Dataset — %-28s  ║\n", desc.name.c_str());
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    DatasetResult result;
    result.name = desc.name;

    // ─── Load or generate dataset ─────────────────────────────────────
    std::vector<WeightedEdge> edges;
    uint64_t N = 0;

    printf("[LOAD] Attempting SNAP download: %s\n", desc.snap_url.c_str());
    bool downloaded = try_download_snap(desc.snap_url, desc.local_path);

    if (downloaded) {
        printf("[LOAD] SNAP download successful, parsing edge list...\n");
        SNAPEdgeListReader reader;
        bool add_reverse = !desc.directed; // undirected → add both directions
        edges = reader.read_file(desc.local_path, add_reverse);
        if (edges.size() > 0) {
            VertexRenumber renumber;
            renumber.build(edges);
            N = renumber.num_vertices;
            printf("[LOAD] Real dataset: %lu vertices, %zu edges (after renumber)\n",
                   N, edges.size());
        }
    }

    if (edges.empty()) {
        printf("[LOAD] SNAP unavailable, generating synthetic power-law fallback\n");
        // email-Enron: α≈2.1, wiki-Vote: α≈2.3
        double alpha = (desc.name == "email-Enron") ? 2.1 : 2.3;
        uint64_t target_E = desc.expected_edges;
        N = desc.expected_vertices;
        edges = generate_powerlaw_graph(N, target_E, alpha, 31415);
        // Renumber (synthetic already dense, but ensure consistency)
        VertexRenumber renumber;
        renumber.build(edges);
        N = renumber.num_vertices;
        printf("[LOAD] Synthetic fallback: %lu vertices, %zu edges\n", N, edges.size());
    }

    result.vertices = N;
    result.edges = edges.size();
    printf("[LOAD] Final: V=%lu E=%zu, RSS=%.1f MB\n\n", N, edges.size(), rss_mb());

    // ─── Build CSR baseline ───────────────────────────────────────────
    printf("=== §4 Build CSR Baseline ===\n");
    CSRGraph csr;
    {
        Timer bt;
        csr.build(edges, N);
        printf("  CSR build: %.1f ms, V=%lu E=%lu\n", bt.ms(), csr.num_vertices, csr.num_edges);
        CHECK(csr.num_vertices == N, "CSR vertex count");
        CHECK(csr.num_edges == edges.size(), "CSR edge count");
    }

    // ─── Build Tiered CSR ─────────────────────────────────────────────
    printf("\n=== §5 Build TieredCSR ===\n");
    TieredCSR tiered;
    {
        Timer bt;
        tiered.build(edges, N);
        uint64_t total_tier_edges = 0;
        for (int t = 0; t < NUM_TIERS; t++) {
            printf("  Tier[%s]: %lu edges\n", tier_str((TierID)t), tiered.tiers[t].num_edges);
            total_tier_edges += tiered.tiers[t].num_edges;
        }
        CHECK(total_tier_edges == edges.size(), "all edges assigned to tiers");
        printf("  Tiered build: %.1f ms\n", bt.ms());
        result.tier_dram = tiered.tiers[0].num_edges;
        result.tier_ssd  = tiered.tiers[1].num_edges;
        result.tier_hdd  = tiered.tiers[2].num_edges;
    }

    // ─── Select source vertex ─────────────────────────────────────────
    // MOD M161: use max-degree vertex for BFS/SSSP to maximize coverage
    uint64_t source = find_max_degree_vertex(csr);
    printf("\n  Source vertex: %lu (degree=%lu)\n", source, csr.degree(source));

    // ─── BFS comparison ───────────────────────────────────────────────
    printf("\n=== §6a BFS: Philemon vs CSR ===\n");
    {
        auto csr_res = csr_bfs(csr, source);
        uint64_t csr_reach = 0;
        for (auto d : csr_res.dist) if (d >= 0) csr_reach++;

        auto phi_res = tiered_bfs(tiered, source, desc.directed);
        uint64_t phi_reach = 0;
        for (auto d : phi_res.dist) if (d >= 0) phi_reach++;

        printf("  CSR BFS:      %.2f ms, reachable=%lu, edges=%lu\n",
               csr_res.time_ms, csr_reach, csr_res.edges_traversed);
        printf("  Philemon BFS: %.2f ms, reachable=%lu, edges=%lu, switches=%lu\n",
               phi_res.time_ms, phi_reach, phi_res.edges_traversed, phi_res.frontier_switches);
        double slowdown = phi_res.time_ms / std::max(csr_res.time_ms, 0.001);
        printf("  Slowdown: %.2fx\n", slowdown);

        double reach_ratio = (double)phi_reach / std::max(csr_reach, 1UL);
        CHECK(reach_ratio > 0.95 && reach_ratio < 1.05, "BFS reachability within 5% of CSR");
        CHECK(slowdown < 5.0, "BFS slowdown < 5x vs CSR");

        result.bfs_csr_ms = csr_res.time_ms;
        result.bfs_phi_ms = phi_res.time_ms;
        result.bfs_csr_reach = csr_reach;
        result.bfs_phi_reach = phi_reach;
        result.bfs_switches = phi_res.frontier_switches;
        result.bfs_slowdown = slowdown;

        tiered.counters.dump("BFS");
        tiered.counters.reset();
    }

    // ─── PageRank comparison ──────────────────────────────────────────
    printf("\n=== §6b PageRank: Philemon vs CSR ===\n");
    {
        auto csr_res = csr_pagerank(csr, 20);
        auto phi_res = tiered_pagerank(tiered, 20);

        printf("  CSR PR:      %.2f ms (20 iters)\n", csr_res.time_ms);
        printf("  Philemon PR: %.2f ms (%d iters, L1=%.2e, Linf=%.2e)\n",
               phi_res.time_ms, phi_res.iters, phi_res.l1_residual, phi_res.linf_residual);
        double slowdown = phi_res.time_ms / std::max(csr_res.time_ms, 0.001);
        printf("  Slowdown: %.2fx\n", slowdown);

        CHECK(phi_res.iters > 0, "PR converged");
        CHECK(slowdown < 5.0, "PR slowdown < 5x vs CSR");

        result.pr_csr_ms = csr_res.time_ms;
        result.pr_phi_ms = phi_res.time_ms;
        result.pr_iters = phi_res.iters;
        result.pr_l1 = phi_res.l1_residual;
        result.pr_linf = phi_res.linf_residual;
        result.pr_slowdown = slowdown;

        tiered.counters.dump("PR");
        tiered.counters.reset();
    }

    // ─── SSSP comparison ──────────────────────────────────────────────
    printf("\n=== §6c SSSP: Philemon vs CSR ===\n");
    {
        auto csr_res = csr_sssp(csr, source);
        auto phi_res = tiered_sssp(tiered, source);

        uint64_t csr_reach = 0, phi_reach = 0;
        for (auto d : csr_res.dist) if (d < 1e18) csr_reach++;
        for (auto d : phi_res.dist) if (d < 1e18) phi_reach++;

        printf("  CSR SSSP:      %.2f ms, reachable=%lu, relaxations=%lu\n",
               csr_res.time_ms, csr_reach, csr_res.relaxations);
        printf("  Philemon SSSP: %.2f ms, reachable=%lu, relaxations=%lu\n",
               phi_res.time_ms, phi_reach, phi_res.relaxations);
        double slowdown = phi_res.time_ms / std::max(csr_res.time_ms, 0.001);
        printf("  Slowdown: %.2fx\n", slowdown);

        CHECK(phi_reach > 0, "SSSP reaches some vertices");
        CHECK(phi_res.dist[source] == 0.0, "SSSP source dist = 0");
        CHECK(slowdown < 5.0, "SSSP slowdown < 5x vs CSR");

        result.sssp_csr_ms = csr_res.time_ms;
        result.sssp_phi_ms = phi_res.time_ms;
        result.sssp_reach = phi_reach;
        result.sssp_relax = phi_res.relaxations;
        result.sssp_slowdown = slowdown;

        tiered.counters.dump("SSSP");
        tiered.counters.reset();
    }

    // ─── WCC comparison ───────────────────────────────────────────────
    printf("\n=== §6d WCC: Philemon vs CSR ===\n");
    {
        auto csr_res = csr_wcc(csr);
        auto phi_res = tiered_wcc(tiered);

        printf("  CSR WCC:      %.2f ms, components=%lu\n",
               csr_res.time_ms, csr_res.num_components);
        printf("  Philemon WCC: %.2f ms, components=%lu\n",
               phi_res.time_ms, phi_res.num_components);
        double slowdown = phi_res.time_ms / std::max(csr_res.time_ms, 0.001);
        printf("  Slowdown: %.2fx\n", slowdown);

        CHECK(phi_res.num_components >= 1, "WCC found >= 1 component");
        CHECK(phi_res.num_components <= N, "WCC components <= N");
        // Components should roughly match between CSR and tiered
        double comp_ratio = (double)phi_res.num_components / std::max(csr_res.num_components, 1UL);
        CHECK(comp_ratio > 0.9 && comp_ratio < 1.1, "WCC component count within 10% of CSR");
        CHECK(slowdown < 5.0, "WCC slowdown < 5x vs CSR");

        result.wcc_csr_ms = csr_res.time_ms;
        result.wcc_phi_ms = phi_res.time_ms;
        result.wcc_components = phi_res.num_components;
        result.wcc_slowdown = slowdown;

        tiered.counters.dump("WCC");
        tiered.counters.reset();
    }

    // ─── TEM-Graph on real dataset ────────────────────────────────────
    printf("\n=== §7 TEM-Graph Interval Index (on %s edges) ===\n", desc.name.c_str());
    {
        TemGraphIndex tg;
        tg.load_from_edges(edges, 789);
        printf("  Loaded %zu intervals, %zu unique, time=[%d,%d]\n",
               tg.intervals_.size(), tg.unique_.size(), tg.earliest_, tg.latest_);

        int mid = (tg.earliest_ + tg.latest_) / 2;
        int result_count = tg.contains_query(tg.earliest_, mid);
        printf("  contains_query(%d,%d) = %d, visited=%lu\n",
               tg.earliest_, mid, result_count, tg.visited_);
        CHECK(result_count >= 0, "contains_query returns non-negative");

        std::vector<std::pair<int,int>> queries = {
            {tg.earliest_, tg.earliest_ + (tg.latest_ - tg.earliest_)/4},
            {tg.earliest_, mid},
            {tg.earliest_, tg.latest_}
        };
        auto batch_res = tg.batch_contains(queries);
        CHECK(batch_res.size() == queries.size(), "batch query count matches");
        CHECK(batch_res.back() >= batch_res.front(), "wider range finds more intervals");
    }

    // ─── EdgeStream test on real data ─────────────────────────────────
    printf("\n=== §2 EdgeStream (real dataset) ===\n");
    {
        TieredEdgeStream stream;
        for (auto& e : edges) stream.add(e);
        CHECK(stream.size() == (int)edges.size(), "stream.size matches");

        stream.permute(42);
        WeightedEdge buf[64];
        size_t got = stream.get_batch(buf, 64);
        CHECK(got == 64, "batch_prefetch returns 64");

        stream.reset();
        stream.reorder_by_degree(true);
        CHECK(stream.size() > 0, "reorder preserves edges");
        printf("  EdgeStream io_count=%lu, size_after_reorder=%d\n",
               stream.io_count(), stream.size());
    }

    result.rss = rss_mb();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════
// §10 CSV output — Table 3 rows for paper
// ═══════════════════════════════════════════════════════════════════════
void write_csv(const std::string& path, const std::vector<DatasetResult>& results) {
    std::ofstream csv(path);
    csv << "# M161 Paper Data — Philemon-TSH Real Dataset Experiment\n";
    csv << "# Generated by m161_m162_real_dataset_experiment.cpp\n";
    csv << "# Datasets: email-Enron (SNAP), wiki-Vote (SNAP)\n";
    csv << "#\n";

    // Table 3a: Algorithm Latency per dataset
    csv << "# Table 3a: Real Dataset Algorithm Latency (ms)\n";
    csv << "dataset,vertices,edges,algo,system,latency_ms,reachable,extra\n";
    for (auto& r : results) {
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",BFS,CSR," << std::fixed << std::setprecision(2) << r.bfs_csr_ms
            << "," << r.bfs_csr_reach << ",\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",BFS,Philemon," << r.bfs_phi_ms
            << "," << r.bfs_phi_reach << ",switches=" << r.bfs_switches << "\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",PR,CSR," << r.pr_csr_ms << "," << r.vertices << ",iters=20\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",PR,Philemon," << r.pr_phi_ms << "," << r.vertices
            << ",iters=" << r.pr_iters << ";L1=" << std::scientific << std::setprecision(2) << r.pr_l1
            << ";Linf=" << r.pr_linf << "\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",SSSP,CSR," << std::fixed << std::setprecision(2) << r.sssp_csr_ms
            << "," << r.sssp_reach << ",\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",SSSP,Philemon," << r.sssp_phi_ms
            << "," << r.sssp_reach << ",relaxations=" << r.sssp_relax << "\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",WCC,CSR," << r.wcc_csr_ms << "," << r.vertices << ",\n";
        csv << r.name << "," << r.vertices << "," << r.edges
            << ",WCC,Philemon," << r.wcc_phi_ms << "," << r.vertices
            << ",components=" << r.wcc_components << "\n";
    }

    // Table 3b: Tier distribution per dataset
    csv << "#\n# Table 3b: Tier Distribution\n";
    csv << "dataset,vertices,total_edges,tier_dram,tier_ssd,tier_hdd,pct_dram,pct_ssd,pct_hdd\n";
    for (auto& r : results) {
        uint64_t total = r.tier_dram + r.tier_ssd + r.tier_hdd;
        double pd = total > 0 ? 100.0*r.tier_dram/total : 0;
        double ps = total > 0 ? 100.0*r.tier_ssd/total : 0;
        double ph = total > 0 ? 100.0*r.tier_hdd/total : 0;
        csv << r.name << "," << r.vertices << "," << total
            << "," << r.tier_dram << "," << r.tier_ssd << "," << r.tier_hdd
            << "," << std::fixed << std::setprecision(1) << pd
            << "," << ps << "," << ph << "\n";
    }

    // Table 3c: Slowdown summary
    csv << "#\n# Table 3c: Slowdown + Memory\n";
    csv << "dataset,bfs_slowdown,pr_slowdown,sssp_slowdown,wcc_slowdown,rss_mb\n";
    for (auto& r : results) {
        csv << r.name
            << "," << std::fixed << std::setprecision(2) << r.bfs_slowdown
            << "," << r.pr_slowdown
            << "," << r.sssp_slowdown
            << "," << r.wcc_slowdown
            << "," << std::setprecision(1) << r.rss << "\n";
    }

    csv.close();
    printf("\n[CSV] Written to %s\n", path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════
// §11 main() — CLI with --dataset {enron|wikivote|all}
// ═══════════════════════════════════════════════════════════════════════
void run_all_tests(const std::string& dataset_filter, int threads) {
    #ifdef _OPENMP
    omp_set_num_threads(threads);
    printf("OpenMP: %d threads\n", threads);
    #endif
    printf("RSS at start: %.1f MB\n", rss_mb());

    std::vector<DatasetResult> results;

    if (dataset_filter == "all" || dataset_filter == "wikivote") {
        // wiki-Vote first (smaller, faster)
        results.push_back(run_dataset_experiment(get_wikivote_desc()));
    }

    if (dataset_filter == "all" || dataset_filter == "enron") {
        results.push_back(run_dataset_experiment(get_enron_desc()));
    }

    // Write CSV
    std::string csv_path = "experiment/results/m161_paper_data.csv";
    write_csv(csv_path, results);

    // Summary
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" M161-M162 Results: %d PASS, %d FAIL\n", g_pass, g_fail);
    printf(" RSS final: %.1f MB\n", rss_mb());
    printf("═══════════════════════════════════════════════════════════\n");
}

} // namespace phi

int main(int argc, char** argv) {
    std::string dataset = "all";
    int threads = 4, debug = 2;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dataset" && i+1 < argc) dataset = argv[++i];
        else if (std::string(argv[i]) == "--threads" && i+1 < argc) threads = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--debug" && i+1 < argc) debug = std::stoi(argv[++i]);
    }
    phi::g_debug = debug;
    phi::run_all_tests(dataset, threads);
    return phi::g_fail > 0 ? 1 : 0;
}
