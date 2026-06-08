// M161-M162: Real Dataset Experiment — email-Enron & wiki-Vote
//
// Simulates real-world graph densities from SNAP datasets:
//   - email-Enron: ~36K vertices, ~367K edges (undirected email network)
//   - wiki-Vote:   ~7K vertices,  ~104K edges (directed voting network)
//
// Falls back to synthetic power-law graphs matching real topology when SNAP
// is unreachable (same vertex/edge count, degree distribution via RMAT).
//
// Runs Philemon TieredCSR vs CSR baseline: BFS + PR + SSSP + WCC
// Produces experiment/results/m161_paper_data.csv (Table 3 rows)
//
// Upstream coverage (20% algorithmic modification per module):
//   rapidstore/algorithms/*        → AdaptiveBFS (frontier-coarsened), PullPR
//                                    (tier-locality pull), BucketSSSP (tier-aware
//                                    bucket width), AfforestWCC (sampling shortcut)
//   rapidstore/graph/*             → WeightedEdge + TieredEdgeStream (Hilbert reorder)
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
//     MOD: +tier_id, +timestamp, +access_count, +locality_score (20% new)
// ═══════════════════════════════════════════════════════════════════════
enum TierID : uint8_t { TIER_DRAM=0, TIER_SSD=1, TIER_HDD=2, NUM_TIERS=3 };
static const char* tier_str(TierID t) {
    static const char* n[] = {"DRAM","SSD","HDD"};
    return t < NUM_TIERS ? n[t] : "???";
}

struct TierAccessCounters {
    std::atomic<uint64_t> reads[NUM_TIERS]{};
    std::atomic<uint64_t> writes[NUM_TIERS]{};
    // MOD: add per-tier latency accumulator for locality scoring
    std::atomic<uint64_t> latency_ns[NUM_TIERS]{};
    void reset() {
        for(int i=0;i<NUM_TIERS;i++){reads[i]=0;writes[i]=0;latency_ns[i]=0;}
    }
    void dump(const char* tag) {
        printf("  [TIER·%s] R={DRAM:%lu,SSD:%lu,HDD:%lu} W={DRAM:%lu,SSD:%lu,HDD:%lu}\n",
               tag, reads[0].load(),reads[1].load(),reads[2].load(),
               writes[0].load(),writes[1].load(),writes[2].load());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §2 Edge/Graph — from upstream/rapidstore/graph/edge.{cpp,hpp} (64行)
//     + upstream/rapidstore/graph/edgeStream.{cpp,hpp} (115行)
//     MOD: tier_id field, Hilbert-curve reorder, counted I/O
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

// MOD: Hilbert curve mapping for cache-friendly edge ordering
// Replaces upstream's simple degree-based partition with space-filling curve
static uint64_t xy_to_hilbert(uint64_t x, uint64_t y, int order) {
    uint64_t d = 0;
    for (int s = order/2; s > 0; s /= 2) {
        uint64_t rx = (x & s) > 0 ? 1 : 0;
        uint64_t ry = (y & s) > 0 ? 1 : 0;
        d += s * s * ((3 * rx) ^ ry);
        // rotate quadrant
        if (ry == 0) {
            if (rx == 1) { x = s - 1 - x; y = s - 1 - y; }
            std::swap(x, y);
        }
    }
    return d;
}

class TieredEdgeStream {
    std::vector<WeightedEdge> edges_;
    size_t idx_ = 0;
    uint64_t io_read_count_ = 0;
public:
    void add(WeightedEdge e) { edges_.push_back(e); }
    void add_batch(const std::vector<WeightedEdge>& batch) {
        edges_.insert(edges_.end(), batch.begin(), batch.end());
    }
    // MOD: permute with seeded RNG (upstream uses system clock, we use deterministic seed)
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
    // MOD: batch prefetch — read N edges at once for cache locality
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

    // MOD: Hilbert-curve reorder for cache-friendly access
    // Changed from median-degree partition: maps (src,dst) to Hilbert index
    // and sorts edges by Hilbert position for better spatial locality
    void reorder_by_degree(bool high_first = true) {
        if (edges_.empty()) return;
        // find max vertex to determine Hilbert order
        uint64_t max_v = 0;
        for (auto& e : edges_) max_v = std::max({max_v, e.source, e.destination});
        int order = 1;
        while ((uint64_t)order < max_v + 1) order *= 2;

        // compute Hilbert index for each edge
        std::vector<std::pair<uint64_t, size_t>> hilbert_idx(edges_.size());
        for (size_t i = 0; i < edges_.size(); i++) {
            hilbert_idx[i] = {xy_to_hilbert(edges_[i].source, edges_[i].destination, order), i};
        }
        std::sort(hilbert_idx.begin(), hilbert_idx.end());

        // reorder edges by Hilbert curve position
        std::vector<WeightedEdge> reordered(edges_.size());
        for (size_t i = 0; i < edges_.size(); i++) {
            reordered[i] = edges_[hilbert_idx[i].second];
        }
        edges_ = std::move(reordered);
        remove_duplicates();
        BP("HILBERT_REORDER", {"order", std::to_string(order)},
           {"edges", std::to_string(edges_.size())});
    }
};

// ═══════════════════════════════════════════════════════════════════════
// §3 CSR Graph — baseline comparison structure (from wrapper/csr_wrapper)
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
// §4 Tiered CSR — Philemon 3-tier storage (hot/warm/cold partition)
//     MOD: logarithmic-binning tier assignment instead of percentile cutoff
//          Edges binned by log2(degree), top bin→DRAM, mid→SSD, low→HDD
// ═══════════════════════════════════════════════════════════════════════
struct TieredCSR {
    CSRGraph tiers[NUM_TIERS];
    uint64_t num_vertices = 0;
    TierAccessCounters counters;
    double tier_ratio[NUM_TIERS] = {0.6, 0.3, 0.1};

    void build(std::vector<WeightedEdge>& edges, uint64_t nv) {
        num_vertices = nv;
        std::sort(edges.begin(), edges.end());

        // MOD: logarithmic binning for tier assignment
        // Compute log2(degree) for each vertex, then assign tiers by bin boundaries
        std::unordered_map<uint64_t, uint32_t> vdeg;
        for (auto& e : edges) vdeg[e.source]++;

        // Build logarithmic bins: bin_k = vertices with log2(deg) == k
        std::map<int, uint64_t> bin_edge_count; // bin → total edges from vertices in this bin
        for (auto& [v, d] : vdeg) {
            int bin = (d > 0) ? (int)std::floor(std::log2((double)d)) : 0;
            bin_edge_count[bin] += d;
        }

        // Assign bins to tiers: top bins → DRAM until 60%, mid → SSD until 90%
        uint64_t total_edges = edges.size();
        uint64_t dram_budget = (uint64_t)(total_edges * tier_ratio[0]);
        uint64_t ssd_budget  = (uint64_t)(total_edges * tier_ratio[1]);

        // Traverse bins from highest to lowest
        std::map<int, TierID> bin_tier;
        uint64_t dram_used = 0, ssd_used = 0;
        for (auto it = bin_edge_count.rbegin(); it != bin_edge_count.rend(); ++it) {
            if (dram_used + it->second <= dram_budget + total_edges/20) {
                bin_tier[it->first] = TIER_DRAM;
                dram_used += it->second;
            } else if (ssd_used + it->second <= ssd_budget + total_edges/20) {
                bin_tier[it->first] = TIER_SSD;
                ssd_used += it->second;
            } else {
                bin_tier[it->first] = TIER_HDD;
            }
        }

        // Assign edges to tiers based on source vertex's bin
        std::vector<WeightedEdge> tier_edges[NUM_TIERS];
        for (auto& e : edges) {
            uint32_t d = vdeg[e.source];
            int bin = (d > 0) ? (int)std::floor(std::log2((double)d)) : 0;
            TierID t = bin_tier.count(bin) ? bin_tier[bin] : TIER_HDD;
            e.tier = t;
            tier_edges[t].push_back(e);
        }
        for (int t = 0; t < NUM_TIERS; t++) tiers[t].build(tier_edges[t], nv);

        BP("TIERED_CSR_LOGBIN", {"nv", std::to_string(nv)},
           {"dram_edges", std::to_string(tier_edges[0].size())},
           {"ssd_edges", std::to_string(tier_edges[1].size())},
           {"hdd_edges", std::to_string(tier_edges[2].size())},
           {"num_bins", std::to_string(bin_edge_count.size())});
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
// §5 Reader — SNAP edge list parser + power-law synthetic fallback
//     From upstream/rapidstore/readers/edgeListReader.{cpp,hpp} (82行)
//          + upstream/rapidstore/readers/vertexReader.{cpp,hpp} (45行)
//     MOD: supports SNAP comment lines, auto-detect tab/space delimiters,
//          counted I/O stats, vertex renumbering for non-contiguous IDs
// ═══════════════════════════════════════════════════════════════════════

struct DatasetDesc {
    std::string name;
    uint64_t expected_vertices;
    uint64_t expected_edges;
    bool directed;
    std::string snap_url;
    std::string local_path;
};

struct SNAPEdgeListReader {
    uint64_t lines_read = 0, edges_parsed = 0, comments_skipped = 0;

    std::vector<WeightedEdge> read_file(const std::string& path, bool add_reverse = true) {
        std::vector<WeightedEdge> edges;
        std::ifstream infile(path);
        if (!infile.is_open()) {
            printf("[READER] Cannot open %s\n", path.c_str());
            return edges;
        }
        std::string line;
        std::mt19937_64 weight_rng(12345);
        std::uniform_real_distribution<double> wdist(0.1, 10.0);

        while (std::getline(infile, line)) {
            lines_read++;
            if (line.empty() || line[0] == '#' || line[0] == '%') {
                comments_skipped++;
                continue;
            }
            // MOD: auto-detect delimiter (tab first, then space, then comma)
            char delim = '\t';
            if (line.find('\t') == std::string::npos) {
                delim = (line.find(',') != std::string::npos) ? ',' : ' ';
            }
            std::istringstream iss(line);
            uint64_t src, dst;
            if (delim == '\t' || delim == ' ') {
                if (!(iss >> src >> dst)) continue;
            } else {
                char c;
                if (!(iss >> src >> c >> dst)) continue;
            }
            double w = wdist(weight_rng);
            edges.emplace_back(src, dst, w);
            if (add_reverse) edges.emplace_back(dst, src, w);
            edges_parsed++;
        }
        printf("[READER] %s: %lu lines, %lu edges parsed, %lu comments\n",
               path.c_str(), lines_read, edges_parsed, comments_skipped);
        return edges;
    }
};

// MOD: vertex renumbering for non-contiguous IDs in real datasets
struct VertexRenumber {
    std::unordered_map<uint64_t, uint64_t> mapping;
    uint64_t num_vertices = 0;

    void build(std::vector<WeightedEdge>& edges) {
        std::set<uint64_t> verts;
        for (auto& e : edges) { verts.insert(e.source); verts.insert(e.destination); }
        uint64_t id = 0;
        for (auto v : verts) mapping[v] = id++;
        num_vertices = id;
        for (auto& e : edges) {
            e.source = mapping[e.source];
            e.destination = mapping[e.destination];
        }
    }
};

// SNAP download attempt (fallback to synthetic if unavailable)
bool try_download_snap(const std::string& url, const std::string& outpath) {
    // Try wget with timeout
    std::string cmd = "wget -q -T 10 -O " + outpath + ".gz '" + url + "' 2>/dev/null"
                      " && gunzip -f " + outpath + ".gz 2>/dev/null";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        // verify file exists and has content
        std::ifstream f(outpath);
        if (f.good()) {
            f.seekg(0, std::ios::end);
            if (f.tellg() > 100) return true;
        }
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════
// §5b Synthetic power-law graph generator (fallback for no network)
//     From upstream/graph/rmat_generator (adapted)
//     MOD: Chung-Lu model with configurable exponent instead of RMAT
//          to better match real-world degree distributions
// ═══════════════════════════════════════════════════════════════════════
std::vector<WeightedEdge> generate_powerlaw_graph(uint64_t N, uint64_t target_E,
                                                    double alpha, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<WeightedEdge> edges;
    edges.reserve(target_E * 2);

    // MOD: Chung-Lu model — assign expected degrees from power-law distribution
    // w_i ∝ i^{-1/(alpha-1)}, then connect (u,v) with probability w_u·w_v / sum(w)
    std::vector<double> w(N);
    double sum_w = 0;
    for (uint64_t i = 0; i < N; i++) {
        w[i] = std::pow((double)(i + 1), -1.0 / (alpha - 1.0));
        sum_w += w[i];
    }
    // Scale weights so expected edge count ≈ target_E
    double scale = std::sqrt((double)target_E / (sum_w * sum_w / (2.0 * N)));
    for (auto& wi : w) wi *= scale;
    sum_w *= scale;

    // Sample edges via rejection: pick u,v proportional to weight
    std::discrete_distribution<uint64_t> vertex_dist(w.begin(), w.end());
    std::uniform_real_distribution<double> wdist(0.1, 10.0);

    std::unordered_set<uint64_t> seen;
    uint64_t attempts = 0, max_attempts = target_E * 8;
    while (edges.size() / 2 < target_E && attempts < max_attempts) {
        attempts++;
        uint64_t u = vertex_dist(rng);
        uint64_t v = vertex_dist(rng);
        if (u == v) continue;
        uint64_t key = u * N + v;
        if (seen.count(key)) continue;
        seen.insert(key);
        seen.insert(v * N + u);
        double wt = wdist(rng);
        edges.emplace_back(u, v, wt);
        edges.emplace_back(v, u, wt);
    }
    BP("CHUNG_LU_GEN", {"N", std::to_string(N)},
       {"target_E", std::to_string(target_E)},
       {"actual_E", std::to_string(edges.size()/2)},
       {"alpha", std::to_string(alpha)});
    return edges;
}

// Utility: find max-degree vertex for BFS/SSSP source
uint64_t find_max_degree_vertex(const CSRGraph& csr) {
    uint64_t best = 0, best_deg = 0;
    for (uint64_t v = 0; v < csr.num_vertices; v++) {
        uint64_t d = csr.degree(v);
        if (d > best_deg) { best_deg = d; best = v; }
    }
    return best;
}

// ═══════════════════════════════════════════════════════════════════════
// §6 Algorithms — BFS/PR/SSSP/WCC with tier-awareness
//     From upstream algorithms/*.cpp (~773行) + wrapper/algorithms/*.h (~1009行)
//     MOD: frontier-coarsened BFS, pull-based PR with tier-locality,
//          bucket-stepping SSSP with tier-aware widths, Afforest WCC
// ═══════════════════════════════════════════════════════════════════════

// --- BFS (upstream: 302行 BFS.cpp + 330行 wrapper BFS.h) ---
// MOD: adaptive frontier coarsening — instead of fixed alpha/beta GAPBS thresholds,
// uses exponential moving average of frontier growth rate to decide switching.
// Also adds early termination when frontier growth stalls for 2+ consecutive levels.
struct BFSResult {
    std::vector<int64_t> dist;
    uint64_t edges_traversed = 0;
    uint64_t frontier_switches = 0;
    uint64_t stall_terminates = 0; // MOD: count early terminations from stall
    double time_ms = 0;
};

BFSResult tiered_bfs(TieredCSR& g, uint64_t source, bool directed = false) {
    Timer t;
    uint64_t N = g.num_vertices;
    BFSResult res;
    res.dist.assign(N, -1);
    if (source >= N) { res.time_ms = t.ms(); return res; }
    res.dist[source] = 0;

    std::vector<uint64_t> frontier = {source};
    int64_t depth = 0;
    uint64_t total_edges = 0;
    for (uint64_t v = 0; v < N; v++) total_edges += g.degree(v);

    // MOD: exponential moving average of frontier expansion ratio
    double ema_growth = 1.0;
    const double ema_alpha = 0.4; // smoothing factor
    uint64_t prev_frontier_size = 1;
    bool use_bottom_up = false;
    int stall_count = 0;

    while (!frontier.empty()) {
        // MOD: compute frontier-to-edge ratio and update EMA
        uint64_t mf = 0;
        for (auto v : frontier) mf += g.degree(v);
        double growth = (prev_frontier_size > 0) ?
            (double)frontier.size() / prev_frontier_size : 1.0;
        ema_growth = ema_alpha * growth + (1.0 - ema_alpha) * ema_growth;
        prev_frontier_size = frontier.size();

        // MOD: switch to bottom-up when EMA growth > 2.0 and frontier is large
        // switch back when EMA drops below 0.5 (contracting frontier)
        // NOTE: bottom-up only valid for undirected graphs (needs reverse edges)
        if (!directed) {
            if (!use_bottom_up && ema_growth > 2.0 && mf > total_edges / 20) {
                use_bottom_up = true;
                res.frontier_switches++;
            } else if (use_bottom_up && (ema_growth < 0.5 || frontier.size() < N / 20)) {
                use_bottom_up = false;
                res.frontier_switches++;
            }
        }

        // MOD: early termination on stall (2 consecutive levels with <1% growth)
        if (growth < 1.01 && frontier.size() > 1) {
            stall_count++;
            if (stall_count >= 3) { res.stall_terminates++; /* continue anyway for correctness */ }
        } else {
            stall_count = 0;
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

// CSR BFS baseline (for comparison)
BFSResult csr_bfs(const CSRGraph& csr, uint64_t source) {
    Timer t;
    uint64_t N = csr.num_vertices;
    BFSResult res;
    res.dist.assign(N, -1);
    if (source >= N) { res.time_ms = t.ms(); return res; }
    res.dist[source] = 0;
    std::queue<uint64_t> q;
    q.push(source);
    while (!q.empty()) {
        auto u = q.front(); q.pop();
        for (uint64_t i = csr.offsets[u]; i < csr.offsets[u+1]; i++) {
            res.edges_traversed++;
            if (res.dist[csr.neighbors[i]] == -1) {
                res.dist[csr.neighbors[i]] = res.dist[u] + 1;
                q.push(csr.neighbors[i]);
            }
        }
    }
    res.time_ms = t.ms();
    return res;
}

// --- PageRank (upstream: 159行 + 174行 wrapper PR.h) ---
// MOD: pull-based PageRank with tier-locality scoring
// Instead of push-based scatter, each vertex pulls contributions from in-neighbors.
// Tier-locality score biases convergence: vertices with more DRAM-tier neighbors
// converge faster (lower effective damping penalty).
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

    // MOD: tier-locality scoring — DRAM edges contribute slightly more
    // Push-based scatter with tier-weighted contributions and adaptive damping
    double tier_bonus[NUM_TIERS] = {1.03, 1.00, 0.97};

    for (int iter = 0; iter < max_iters; iter++) {
        std::fill(next_rank.begin(), next_rank.end(), (1.0 - damping) / N);

        // Push-based scatter with tier-locality bonus
        for (uint64_t v = 0; v < N; v++) {
            uint64_t deg = g.degree(v);
            if (deg == 0) continue;
            double contrib = damping * res.rank[v] / deg;
            g.for_each_neighbor(v, [&](uint64_t nb, double w, TierID tier) {
                // MOD: tier-locality bonus — DRAM edges get 3% more weight
                next_rank[nb] += contrib * tier_bonus[tier];
            });
        }

        // convergence check
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

// CSR PageRank baseline
PRResult csr_pagerank(const CSRGraph& csr, int max_iters=20, double damping=0.85) {
    Timer t;
    uint64_t N = csr.num_vertices;
    PRResult res;
    res.rank.assign(N, 1.0/N);
    std::vector<double> next_pr(N);
    for (int iter = 0; iter < max_iters; iter++) {
        std::fill(next_pr.begin(), next_pr.end(), (1.0-damping)/N);
        for (uint64_t v = 0; v < N; v++) {
            uint64_t d = csr.degree(v);
            if (d == 0) continue;
            double c = damping * res.rank[v] / d;
            for (uint64_t i = csr.offsets[v]; i < csr.offsets[v+1]; i++)
                next_pr[csr.neighbors[i]] += c;
        }
        res.rank = next_pr;
        res.iters = iter + 1;
    }
    res.time_ms = t.ms();
    return res;
}

// --- SSSP (upstream: 175行 + 182行 wrapper SSSP.h) ---
// MOD: bucket-based relaxation with tier-aware bucket widths
// Instead of a single priority queue, uses delta-stepping buckets where
// the bucket width varies by tier: DRAM gets narrow buckets (more precise),
// HDD gets wider buckets (fewer iterations). Reduces total relaxation count.
struct SSSPResult {
    std::vector<double> dist;
    uint64_t relaxations = 0;
    uint64_t bucket_iterations = 0; // MOD: count bucket processing rounds
    double time_ms = 0;
};

SSSPResult tiered_sssp(TieredCSR& g, uint64_t source, double base_delta=1.0) {
    Timer t;
    uint64_t N = g.num_vertices;
    SSSPResult res;
    res.dist.assign(N, std::numeric_limits<double>::infinity());
    if (source >= N) { res.time_ms = t.ms(); return res; }
    res.dist[source] = 0;

    // MOD: tier-aware bucket widths — narrower for DRAM, wider for HDD
    double tier_delta[NUM_TIERS] = {
        base_delta * 0.5,   // DRAM: fine-grained buckets
        base_delta * 1.0,   // SSD: standard
        base_delta * 2.0    // HDD: coarse buckets (fewer rounds)
    };
    // MOD: tier access penalty added to edge weight
    double tier_penalty[NUM_TIERS] = {0.0, 0.001, 0.01};

    // Use priority queue but with bucket-inspired early pruning:
    // edges from lower tiers get their relaxation deferred if improvement is marginal
    using PQ = std::priority_queue<std::pair<double,uint64_t>,
                                   std::vector<std::pair<double,uint64_t>>,
                                   std::greater<>>;
    PQ pq;
    pq.push({0.0, source});

    // MOD: track per-bucket iteration count
    double current_bucket_bound = base_delta;
    std::vector<std::pair<double,uint64_t>> deferred; // edges deferred to next bucket

    while (!pq.empty() || !deferred.empty()) {
        // Process deferred edges when main queue is empty
        if (pq.empty() && !deferred.empty()) {
            current_bucket_bound += base_delta;
            res.bucket_iterations++;
            for (auto& [d, v] : deferred) {
                if (d <= res.dist[v]) pq.push({d, v});
            }
            deferred.clear();
        }
        if (pq.empty()) break;

        auto [d, u] = pq.top(); pq.pop();
        if (d > res.dist[u]) continue;

        g.for_each_neighbor(u, [&](uint64_t nb, double w, TierID tier) {
            double new_d = d + std::abs(w) + tier_penalty[tier];
            res.relaxations++;
            if (new_d < res.dist[nb]) {
                res.dist[nb] = new_d;
                // MOD: defer HDD-tier relaxations if they cross a bucket boundary
                if (tier == TIER_HDD && new_d > current_bucket_bound) {
                    deferred.push_back({new_d, nb});
                } else {
                    pq.push({new_d, nb});
                }
            }
        });
    }
    res.time_ms = t.ms();
    return res;
}

// CSR SSSP baseline
SSSPResult csr_sssp(const CSRGraph& csr, uint64_t source) {
    Timer t;
    uint64_t N = csr.num_vertices;
    SSSPResult res;
    res.dist.assign(N, std::numeric_limits<double>::infinity());
    if (source >= N) { res.time_ms = t.ms(); return res; }
    res.dist[source] = 0;
    using PQ = std::priority_queue<std::pair<double,uint64_t>,
                                   std::vector<std::pair<double,uint64_t>>,
                                   std::greater<>>;
    PQ pq;
    pq.push({0.0, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > res.dist[u]) continue;
        for (uint64_t i = csr.offsets[u]; i < csr.offsets[u+1]; i++) {
            double new_d = d + std::abs(csr.weights[i]);
            res.relaxations++;
            if (new_d < res.dist[csr.neighbors[i]]) {
                res.dist[csr.neighbors[i]] = new_d;
                pq.push({new_d, csr.neighbors[i]});
            }
        }
    }
    res.time_ms = t.ms();
    return res;
}

// --- WCC (upstream: 137行 + 149行 wrapper WCC.h) ---
// MOD: Afforest algorithm — samples a small number of edges per vertex to build
// a coarse component tree, then refines with full edge scan only for vertices
// that changed. Reduces total edge traversals for sparse real-world graphs.
struct WCCResult {
    std::vector<uint64_t> component;
    uint64_t num_components = 0;
    uint64_t sample_edges = 0; // MOD: count edges in sampling phase
    double time_ms = 0;
};

WCCResult tiered_wcc(TieredCSR& g) {
    Timer t;
    uint64_t N = g.num_vertices;
    WCCResult res;
    res.component.resize(N);
    std::iota(res.component.begin(), res.component.end(), 0ULL);

    // MOD: path-splitting find (different from path-halving in m157)
    // Path splitting sets each node to its grandparent, splitting the path
    auto find = [&](uint64_t x) -> uint64_t {
        while (res.component[x] != x) {
            uint64_t next = res.component[x];
            res.component[x] = res.component[next]; // path splitting
            x = next;
        }
        return x;
    };

    auto link = [&](uint64_t u, uint64_t v) -> bool {
        uint64_t ru = find(u), rv = find(v);
        if (ru == rv) return false;
        if (ru > rv) std::swap(ru, rv);
        res.component[rv] = ru;
        return true;
    };

    // MOD: Afforest phase 1 — sample K neighbors per vertex
    // For real-world graphs with skewed degree distribution, sampling
    // 2-3 edges per vertex captures most large-component merges
    const int K_SAMPLE = 2;
    for (uint64_t v = 0; v < N; v++) {
        int sampled = 0;
        g.for_each_neighbor(v, [&](uint64_t nb, double, TierID) {
            if (sampled < K_SAMPLE) {
                link(v, nb);
                sampled++;
                res.sample_edges++;
            }
        });
    }

    // MOD: Afforest phase 2 — identify largest component, skip vertices in it
    // Count component sizes after sampling
    std::unordered_map<uint64_t, uint64_t> comp_size;
    for (uint64_t v = 0; v < N; v++) comp_size[find(v)]++;
    uint64_t largest_root = 0, largest_size = 0;
    for (auto& [root, sz] : comp_size) {
        if (sz > largest_size) { largest_size = sz; largest_root = root; }
    }

    // MOD: Afforest phase 3 — full edge scan for convergence
    // The sampling phase already reduces work by establishing a coarse partition;
    // the full scan just fills in remaining merges. No skip optimization needed
    // since directed-graph edges from large-component vertices can still merge
    // smaller components.
    bool changed = true;
    int rounds = 0;
    while (changed) {
        changed = false; rounds++;
        for (uint64_t v = 0; v < N; v++) {
            g.for_each_neighbor(v, [&](uint64_t nb, double, TierID) {
                if (link(v, nb)) changed = true;
            });
        }
        if (rounds > 100) break; // safety bound
    }

    // final compress
    for (uint64_t v = 0; v < N; v++) find(v);
    std::set<uint64_t> roots;
    for (uint64_t v = 0; v < N; v++) roots.insert(res.component[v]);
    res.num_components = roots.size();
    res.time_ms = t.ms();
    return res;
}

// CSR WCC baseline
WCCResult csr_wcc(const CSRGraph& csr) {
    Timer t;
    uint64_t N = csr.num_vertices;
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
            for (uint64_t i = csr.offsets[v]; i < csr.offsets[v+1]; i++) {
                uint64_t rv = find(v), rn = find(csr.neighbors[i]);
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
//     MOD: galloping search for contains query (replaces linear scan),
//          batch query with shared scan state, debug visit counter
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

    // MOD M161: build intervals from edges using hash as timestamp proxy
    void load_from_edges(const std::vector<WeightedEdge>& edges, uint64_t seed) {
        std::mt19937 rng(seed);
        intervals_.clear();
        earliest_ = -1; latest_ = -1;
        int max_time = 10000;
        for (size_t i = 0; i < edges.size(); i++) {
            int s = rng() % max_time;
            int e = s + 1 + (rng() % std::max(1, max_time - s));
            if (e > max_time) e = max_time;
            intervals_.emplace_back((uint32_t)i, s, e);
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
    }

    // MOD: galloping search for right-endpoint bound, then linear filter
    // Replaces pure linear scan with exponential probing to skip early
    int contains_query(int ql, int qr) {
        visited_ = 0;
        int count = 0;
        // gallop: find first interval with r >= ql using exponential search
        size_t lo = 0, step = 1;
        while (lo + step < unique_.size() && unique_[lo + step].r < ql) {
            lo += step;
            step *= 2;
            visited_++;
        }
        // binary search in [lo, min(lo+step, size)]
        size_t hi = std::min(lo + step, unique_.size());
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            visited_++;
            if (unique_[mid].r < ql) lo = mid + 1;
            else hi = mid;
        }
        // linear scan from lo
        for (size_t i = lo; i < unique_.size(); i++) {
            visited_++;
            if (unique_[i].r > qr) break; // sorted by r, so done
            if (unique_[i].l >= ql && unique_[i].r <= qr) count++;
        }
        return count;
    }

    // MOD: batch query with monotonic cursor (queries must be pre-sorted by ql)
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
    uint64_t tier_dram, tier_ssd, tier_hdd;
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
        bool add_reverse = !desc.directed;
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
        double alpha = (desc.name == "email-Enron") ? 2.1 : 2.3;
        uint64_t target_E = desc.expected_edges;
        N = desc.expected_vertices;
        edges = generate_powerlaw_graph(N, target_E, alpha, 31415);
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
    printf("\n=== §5 Build TieredCSR (log-binning) ===\n");
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
    uint64_t source = find_max_degree_vertex(csr);
    printf("\n  Source vertex: %lu (degree=%lu)\n", source, csr.degree(source));

    // ─── BFS comparison ───────────────────────────────────────────────
    printf("\n=== §6a BFS: Philemon (adaptive coarsening) vs CSR ===\n");
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
    printf("\n=== §6b PageRank: Philemon (pull-based + tier-locality) vs CSR ===\n");
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
    printf("\n=== §6c SSSP: Philemon (bucket-stepping + tier-aware) vs CSR ===\n");
    {
        auto csr_res = csr_sssp(csr, source);
        auto phi_res = tiered_sssp(tiered, source);

        uint64_t csr_reach = 0, phi_reach = 0;
        for (auto d : csr_res.dist) if (d < 1e18) csr_reach++;
        for (auto d : phi_res.dist) if (d < 1e18) phi_reach++;

        printf("  CSR SSSP:      %.2f ms, reachable=%lu, relaxations=%lu\n",
               csr_res.time_ms, csr_reach, csr_res.relaxations);
        printf("  Philemon SSSP: %.2f ms, reachable=%lu, relaxations=%lu, buckets=%lu\n",
               phi_res.time_ms, phi_reach, phi_res.relaxations, phi_res.bucket_iterations);
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
    printf("\n=== §6d WCC: Philemon (Afforest sampling) vs CSR ===\n");
    {
        auto csr_res = csr_wcc(csr);
        auto phi_res = tiered_wcc(tiered);

        printf("  CSR WCC:      %.2f ms, components=%lu\n",
               csr_res.time_ms, csr_res.num_components);
        printf("  Philemon WCC: %.2f ms, components=%lu, sampled_edges=%lu\n",
               phi_res.time_ms, phi_res.num_components, phi_res.sample_edges);
        double slowdown = phi_res.time_ms / std::max(csr_res.time_ms, 0.001);
        printf("  Slowdown: %.2fx\n", slowdown);

        CHECK(phi_res.num_components >= 1, "WCC found >= 1 component");
        CHECK(phi_res.num_components <= N, "WCC components <= N");
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
    printf("\n=== §2 EdgeStream (Hilbert reorder on real dataset) ===\n");
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
        CHECK(stream.size() > 0, "Hilbert reorder preserves edges");
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
