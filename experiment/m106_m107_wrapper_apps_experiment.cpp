///////////////////////////////////////////////////////////////////////////////
// m106_m107_wrapper_apps_experiment.cpp
//
// Self-contained single-file experiment replicating and extending
// the 6 wrapper-app patterns from philemon-TSH upstream (3808 lines).
//
// Compile: g++ -std=c++17 -O2 -pthread -o experiment m106_m107_wrapper_apps_experiment.cpp
//
// Wrappers modeled (each with unified interface):
//   1. SortedWeightWrapper   \u2013 sorted adjacency by weight
//   2. UndirectedWrapper     \u2013 mirrors each directed edge
//   3. TransposeWrapper      \u2013 reverses all edge directions
//   4. FilteredWrapper       \u2013 predicate-based edge filtering
//   5. ReadOnlyWrapper       \u2013 blocks mutations, allows reads
//   6. CachedWrapper         \u2013 LRU-cached edge/weight lookups
//
// Core interface per wrapper:
//   construct/destruct (init/destroy)
//   add_vertex / remove_vertex
//   add_edge / remove_edge
//   get_weight / has_edge
//   snapshot: vertex_count / edge_count / degree
//   snapshot_edges (iterate)
//   algorithm delegates: BFS / SSSP / WCC / PageRank
//
// 18 tests total (3 per wrapper). main() runs all, prints PASS/FAIL.
//
// 20% algorithmic modifications vs upstream:
//   - BFS tracks discovery timestamps & layer histogram
//   - SSSP uses Dial's bucket queue for integer weights
//   - WCC tracks component-size distribution statistics
//   - PageRank adds L1 convergence delta tracking per iteration
//   - Cached wrapper uses LRU eviction with hit/miss counters
//   - All wrappers emit printf debug breadcrumbs
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <queue>
#include <deque>
#include <list>
#include <stack>
#include <string>
#include <algorithm>
#include <functional>
#include <numeric>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <sstream>
#include <iostream>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
// Utility: debug printf macro with file/line tagging
///////////////////////////////////////////////////////////////////////////////
#define DBG(fmt, ...) \
    printf("[DBG %s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define DBG_ENTER(func) \
    printf("[ENTER] %s (line %d)\n", func, __LINE__)

#define DBG_EXIT(func) \
    printf("[EXIT]  %s (line %d)\n", func, __LINE__)

///////////////////////////////////////////////////////////////////////////////
// Forward declarations
///////////////////////////////////////////////////////////////////////////////
struct Edge;
class MockGraph;

// Algorithm result structs
struct BFSResult;
struct SSSPResult;
struct WCCResult;
struct PageRankResult;

///////////////////////////////////////////////////////////////////////////////
// Edge representation \u2013 upstream compatible
// upstream: struct edge { uint64_t src; uint64_t dst; double weight; ... }
// modification: added timestamp field for debug tracing
///////////////////////////////////////////////////////////////////////////////
struct Edge {
    uint64_t src;
    uint64_t dst;
    double   weight;
    uint64_t insert_seq;   // [MOD] debug: insertion sequence number

    Edge() : src(0), dst(0), weight(0.0), insert_seq(0) {}
    Edge(uint64_t s, uint64_t d, double w, uint64_t seq = 0)
        : src(s), dst(d), weight(w), insert_seq(seq) {}

    bool operator==(const Edge& o) const {
        return src == o.src && dst == o.dst;
    }
    bool operator<(const Edge& o) const {
        if (src != o.src) return src < o.src;
        return dst < o.dst;
    }
};

// Hash for Edge used in unordered containers
struct EdgeHash {
    size_t operator()(const Edge& e) const {
        size_t h1 = std::hash<uint64_t>{}(e.src);
        size_t h2 = std::hash<uint64_t>{}(e.dst);
        return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

///////////////////////////////////////////////////////////////////////////////
// Algorithm result structures
// upstream: various result types scattered across wrapper implementations
// modification: added statistics / histogram fields
///////////////////////////////////////////////////////////////////////////////
struct BFSResult {
    std::unordered_map<uint64_t, int64_t>  distance;       // vertex -> BFS depth
    std::unordered_map<uint64_t, uint64_t> parent;         // vertex -> parent
    std::unordered_map<uint64_t, uint64_t> discovery_time; // [MOD] timestamp
    std::vector<uint64_t> layer_histogram;                 // [MOD] count per layer
    uint64_t vertices_visited = 0;
    uint64_t edges_traversed  = 0;                         // [MOD] stat counter
};

struct SSSPResult {
    std::unordered_map<uint64_t, double>   dist;
    std::unordered_map<uint64_t, uint64_t> parent;
    uint64_t relaxation_count   = 0;                       // [MOD] stat
    uint64_t bucket_operations  = 0;                       // [MOD] Dial's queue stat
    bool     has_negative_cycle = false;
};

struct WCCResult {
    std::unordered_map<uint64_t, uint64_t> component_id;
    uint64_t num_components         = 0;
    uint64_t largest_component_size = 0;
    double   avg_component_size     = 0.0;                 // [MOD] stat
    double   stddev_component_size  = 0.0;                 // [MOD] stat
    std::vector<uint64_t> component_size_histogram;        // [MOD]
};

struct PageRankResult {
    std::unordered_map<uint64_t, double> rank;
    uint64_t iterations_run    = 0;
    double   final_l1_delta    = 0.0;                      // [MOD] convergence
    std::vector<double> l1_per_iteration;                   // [MOD] convergence trace
    bool     converged         = false;
};

///////////////////////////////////////////////////////////////////////////////
// MockGraph \u2013 the underlying graph storage used by all wrappers
//
// upstream: each wrapper delegates to an opaque graph_t*
// This mock replicates the adjacency-list + vertex-set interface
// that all 6 wrappers expect from their backing store.
//
// Data structures:
//   vertices_: set of active vertex ids
//   adj_:      src -> vector<Edge>  (directed adjacency list)
//   edge_seq_: monotonic insertion counter for debug tracing
///////////////////////////////////////////////////////////////////////////////
class MockGraph {
public:
    // --- construction / destruction ---
    MockGraph() : edge_seq_(0), total_edge_count_(0) {
        DBG("MockGraph::MockGraph() constructed, this=%p", (void*)this);
    }
    ~MockGraph() {
        DBG("MockGraph::~MockGraph() destroyed, V=%zu E=%zu",
            vertices_.size(), (size_t)total_edge_count_);
    }

    // --- vertex operations ---
    // upstream: graph_add_vertex(graph, v)
    bool add_vertex(uint64_t v) {
        DBG("MockGraph::add_vertex(%lu)", (unsigned long)v);
        auto [it, inserted] = vertices_.insert(v);
        if (inserted) {
            adj_[v]; // ensure adjacency list entry
            DBG("  -> vertex %lu added (total V=%zu)", (unsigned long)v, vertices_.size());
        } else {
            DBG("  -> vertex %lu already exists", (unsigned long)v);
        }
        return inserted;
    }

    // upstream: graph_remove_vertex(graph, v)
    bool remove_vertex(uint64_t v) {
        DBG("MockGraph::remove_vertex(%lu)", (unsigned long)v);
        auto it = vertices_.find(v);
        if (it == vertices_.end()) {
            DBG("  -> vertex %lu not found", (unsigned long)v);
            return false;
        }
        // remove outgoing edges
        if (adj_.count(v)) {
            total_edge_count_ -= (int64_t)adj_[v].size();
            adj_.erase(v);
        }
        // remove incoming edges from other vertices
        for (auto& [src, edges] : adj_) {
            auto before = edges.size();
            edges.erase(
                std::remove_if(edges.begin(), edges.end(),
                    [v](const Edge& e) { return e.dst == v; }),
                edges.end());
            total_edge_count_ -= (int64_t)(before - edges.size());
        }
        vertices_.erase(it);
        DBG("  -> vertex %lu removed (total V=%zu E=%ld)",
            (unsigned long)v, vertices_.size(), (long)total_edge_count_);
        return true;
    }

    bool has_vertex(uint64_t v) const {
        return vertices_.count(v) > 0;
    }

    // --- edge operations ---
    // upstream: graph_add_edge(graph, src, dst, weight)
    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        DBG("MockGraph::add_edge(%lu -> %lu, w=%.4f)",
            (unsigned long)src, (unsigned long)dst, weight);
        // auto-add vertices if not present
        add_vertex(src);
        add_vertex(dst);
        // check duplicate
        for (auto& e : adj_[src]) {
            if (e.dst == dst) {
                DBG("  -> edge already exists, updating weight %.4f -> %.4f",
                    e.weight, weight);
                e.weight = weight;
                return false; // updated, not inserted
            }
        }
        edge_seq_++;
        adj_[src].push_back(Edge(src, dst, weight, edge_seq_));
        total_edge_count_++;
        DBG("  -> edge added (seq=%lu, total E=%ld)",
            (unsigned long)edge_seq_, (long)total_edge_count_);
        return true;
    }

    // upstream: graph_remove_edge(graph, src, dst)
    bool remove_edge(uint64_t src, uint64_t dst) {
        DBG("MockGraph::remove_edge(%lu -> %lu)", (unsigned long)src, (unsigned long)dst);
        auto it = adj_.find(src);
        if (it == adj_.end()) return false;
        auto& edges = it->second;
        auto eit = std::find_if(edges.begin(), edges.end(),
            [dst](const Edge& e) { return e.dst == dst; });
        if (eit == edges.end()) {
            DBG("  -> edge not found");
            return false;
        }
        edges.erase(eit);
        total_edge_count_--;
        DBG("  -> edge removed (total E=%ld)", (long)total_edge_count_);
        return true;
    }

    // upstream: graph_has_edge(graph, src, dst)
    bool has_edge(uint64_t src, uint64_t dst) const {
        auto it = adj_.find(src);
        if (it == adj_.end()) return false;
        for (auto& e : it->second) {
            if (e.dst == dst) return true;
        }
        return false;
    }

    // upstream: graph_get_weight(graph, src, dst)
    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        auto it = adj_.find(src);
        if (it == adj_.end()) return std::nullopt;
        for (auto& e : it->second) {
            if (e.dst == dst) return e.weight;
        }
        return std::nullopt;
    }

    // --- snapshot operations ---
    uint64_t vertex_count() const { return vertices_.size(); }
    int64_t  edge_count()   const { return total_edge_count_; }

    // upstream: graph_degree(graph, v, OUTGOING)
    uint64_t out_degree(uint64_t v) const {
        auto it = adj_.find(v);
        if (it == adj_.end()) return 0;
        return it->second.size();
    }

    // upstream: graph_degree(graph, v, INCOMING) \u2013 scan all adj lists
    uint64_t in_degree(uint64_t v) const {
        uint64_t deg = 0;
        for (auto& [src, edges] : adj_) {
            for (auto& e : edges) {
                if (e.dst == v) deg++;
            }
        }
        return deg;
    }

    uint64_t degree(uint64_t v) const {
        return out_degree(v) + in_degree(v);
    }

    // --- iteration ---
    // upstream: graph_edges_begin/end, graph_vertex_begin/end
    const std::unordered_set<uint64_t>& vertices() const { return vertices_; }

    std::vector<Edge> get_edges(uint64_t src) const {
        auto it = adj_.find(src);
        if (it == adj_.end()) return {};
        return it->second;
    }

    // snapshot_edges: all edges in the graph
    std::vector<Edge> snapshot_edges() const {
        std::vector<Edge> result;
        result.reserve(total_edge_count_);
        for (auto& [src, edges] : adj_) {
            for (auto& e : edges) {
                result.push_back(e);
            }
        }
        DBG("MockGraph::snapshot_edges() -> %zu edges", result.size());
        return result;
    }

    // --- raw access for wrappers ---
    std::unordered_map<uint64_t, std::vector<Edge>>& adjacency() { return adj_; }
    const std::unordered_map<uint64_t, std::vector<Edge>>& adjacency() const { return adj_; }

private:
    std::unordered_set<uint64_t> vertices_;
    std::unordered_map<uint64_t, std::vector<Edge>> adj_;
    uint64_t edge_seq_;
    int64_t  total_edge_count_;
};


///////////////////////////////////////////////////////////////////////////////
// Algorithm implementations (shared by all wrappers)
//
// upstream: each wrapper had its own copy of BFS/SSSP/WCC/PageRank delegation.
// We unify here and add the 20% algorithmic modifications.
///////////////////////////////////////////////////////////////////////////////
namespace algo {

// -------------------------------------------------------------------
// BFS \u2013 upstream: standard queue BFS
// [MOD]: added discovery_time, layer_histogram, edges_traversed counter
// -------------------------------------------------------------------
BFSResult bfs(const MockGraph& g, uint64_t source) {
    DBG_ENTER("algo::bfs");
    BFSResult result;
    if (!g.has_vertex(source)) {
        DBG("  BFS source %lu not in graph", (unsigned long)source);
        DBG_EXIT("algo::bfs");
        return result;
    }

    std::queue<uint64_t> q;
    result.distance[source]       = 0;
    result.parent[source]         = source;
    result.discovery_time[source] = 0;
    q.push(source);

    uint64_t timer = 0;
    int64_t  max_depth = 0;

    while (!q.empty()) {
        uint64_t u = q.front(); q.pop();
        result.vertices_visited++;
        int64_t d = result.distance[u];
        if (d > max_depth) max_depth = d;

        auto edges = g.get_edges(u);
        for (auto& e : edges) {
            result.edges_traversed++;
            if (result.distance.find(e.dst) == result.distance.end()) {
                timer++;
                result.distance[e.dst]       = d + 1;
                result.parent[e.dst]         = u;
                result.discovery_time[e.dst] = timer;
                q.push(e.dst);
            }
        }
    }

    // [MOD] build layer histogram
    result.layer_histogram.resize(max_depth + 1, 0);
    for (auto& [v, d] : result.distance) {
        if (d >= 0 && d <= max_depth) {
            result.layer_histogram[(size_t)d]++;
        }
    }

    DBG("  BFS done: visited=%lu, edges_traversed=%lu, max_depth=%ld",
        (unsigned long)result.vertices_visited,
        (unsigned long)result.edges_traversed,
        (long)max_depth);
    DBG_EXIT("algo::bfs");
    return result;
}

// -------------------------------------------------------------------
// SSSP \u2013 upstream: Dijkstra with std::priority_queue
// [MOD]: Dial's bucket queue for integer-weight graphs,
//        falls back to Dijkstra for non-integer weights.
//        Tracks relaxation_count and bucket_operations.
// -------------------------------------------------------------------
static bool weights_are_integer(const MockGraph& g) {
    for (auto& v : g.vertices()) {
        for (auto& e : g.get_edges(v)) {
            double intpart;
            if (std::modf(e.weight, &intpart) != 0.0) return false;
            if (e.weight < 0) return false;
        }
    }
    return true;
}

SSSPResult sssp(const MockGraph& g, uint64_t source) {
    DBG_ENTER("algo::sssp");
    SSSPResult result;
    if (!g.has_vertex(source)) {
        DBG("  SSSP source %lu not in graph", (unsigned long)source);
        DBG_EXIT("algo::sssp");
        return result;
    }

    const double INF = std::numeric_limits<double>::infinity();

    // Initialize distances
    for (auto& v : g.vertices()) {
        result.dist[v] = INF;
    }
    result.dist[source]   = 0.0;
    result.parent[source] = source;

    bool use_dial = weights_are_integer(g);
    DBG("  SSSP using %s algorithm", use_dial ? "Dial's bucket" : "Dijkstra");

    if (use_dial) {
        // [MOD] Dial's bucket queue
        // find max weight to size the buckets
        double max_w = 0;
        for (auto& v : g.vertices()) {
            for (auto& e : g.get_edges(v)) {
                if (e.weight > max_w) max_w = e.weight;
            }
        }
        uint64_t C = (uint64_t)max_w;
        uint64_t num_buckets = C * g.vertex_count() + 1;
        if (num_buckets > 1000000) num_buckets = 1000000; // cap for safety
        std::vector<std::list<uint64_t>> buckets(num_buckets);
        buckets[0].push_back(source);
        result.bucket_operations++;

        uint64_t idx = 0;
        uint64_t processed = 0;
        while (processed < g.vertex_count() && idx < num_buckets) {
            while (idx < num_buckets && buckets[idx].empty()) idx++;
            if (idx >= num_buckets) break;

            uint64_t u = buckets[idx].front();
            buckets[idx].pop_front();
            result.bucket_operations++;

            if ((double)idx > result.dist[u] + 0.5) continue; // stale entry
            processed++;

            for (auto& e : g.get_edges(u)) {
                double nd = result.dist[u] + e.weight;
                if (nd < result.dist[e.dst]) {
                    result.dist[e.dst]   = nd;
                    result.parent[e.dst] = u;
                    result.relaxation_count++;
                    uint64_t bidx = (uint64_t)nd;
                    if (bidx < num_buckets) {
                        buckets[bidx].push_back(e.dst);
                        result.bucket_operations++;
                    }
                }
            }
        }
    } else {
        // Standard Dijkstra (upstream baseline)
        using PQ = std::pair<double, uint64_t>;
        std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
        pq.push({0.0, source});

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > result.dist[u]) continue;

            for (auto& e : g.get_edges(u)) {
                double nd = d + e.weight;
                if (nd < result.dist[e.dst]) {
                    result.dist[e.dst]   = nd;
                    result.parent[e.dst] = u;
                    result.relaxation_count++;
                    pq.push({nd, e.dst});
                }
            }
        }
    }

    DBG("  SSSP done: relaxations=%lu, bucket_ops=%lu",
        (unsigned long)result.relaxation_count,
        (unsigned long)result.bucket_operations);
    DBG_EXIT("algo::sssp");
    return result;
}

// -------------------------------------------------------------------
// WCC \u2013 upstream: union-find based weakly connected components
// [MOD]: track component size distribution (mean, stddev, histogram)
// -------------------------------------------------------------------
WCCResult wcc(const MockGraph& g) {
    DBG_ENTER("algo::wcc");
    WCCResult result;

    // Union-Find
    std::unordered_map<uint64_t, uint64_t> parent_uf;
    std::unordered_map<uint64_t, uint64_t> rank_uf;

    for (auto& v : g.vertices()) {
        parent_uf[v] = v;
        rank_uf[v]   = 0;
    }

    // Find with path compression
    std::function<uint64_t(uint64_t)> find = [&](uint64_t x) -> uint64_t {
        if (parent_uf[x] != x) parent_uf[x] = find(parent_uf[x]);
        return parent_uf[x];
    };

    // Union by rank
    auto unite = [&](uint64_t a, uint64_t b) {
        uint64_t ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rank_uf[ra] < rank_uf[rb]) std::swap(ra, rb);
        parent_uf[rb] = ra;
        if (rank_uf[ra] == rank_uf[rb]) rank_uf[ra]++;
    };

    // Process edges (treat as undirected for WCC)
    for (auto& v : g.vertices()) {
        for (auto& e : g.get_edges(v)) {
            unite(e.src, e.dst);
        }
    }

    // Assign component IDs
    std::unordered_map<uint64_t, uint64_t> root_to_id;
    uint64_t next_id = 0;
    std::unordered_map<uint64_t, uint64_t> comp_sizes;

    for (auto& v : g.vertices()) {
        uint64_t root = find(v);
        if (root_to_id.find(root) == root_to_id.end()) {
            root_to_id[root] = next_id++;
        }
        result.component_id[v] = root_to_id[root];
        comp_sizes[root_to_id[root]]++;
    }

    result.num_components = next_id;

    // [MOD] compute statistics on component sizes
    if (result.num_components > 0) {
        double sum = 0, sum_sq = 0;
        result.largest_component_size = 0;
        for (auto& [cid, sz] : comp_sizes) {
            sum += sz;
            sum_sq += sz * sz;
            if (sz > result.largest_component_size) {
                result.largest_component_size = sz;
            }
        }
        result.avg_component_size = sum / result.num_components;
        double variance = (sum_sq / result.num_components)
                        - (result.avg_component_size * result.avg_component_size);
        result.stddev_component_size = (variance > 0) ? std::sqrt(variance) : 0.0;

        // histogram: index = size, value = count of components with that size
        result.component_size_histogram.resize(result.largest_component_size + 1, 0);
        for (auto& [cid, sz] : comp_sizes) {
            result.component_size_histogram[sz]++;
        }
    }

    DBG("  WCC done: %lu components, largest=%lu, avg=%.2f, stddev=%.2f",
        (unsigned long)result.num_components,
        (unsigned long)result.largest_component_size,
        result.avg_component_size,
        result.stddev_component_size);
    DBG_EXIT("algo::wcc");
    return result;
}

// -------------------------------------------------------------------
// PageRank \u2013 upstream: power iteration with damping factor
// [MOD]: L1 convergence delta tracking per iteration
// -------------------------------------------------------------------
PageRankResult pagerank(const MockGraph& g,
                        double damping   = 0.85,
                        double epsilon   = 1e-6,
                        uint64_t max_iter = 100)
{
    DBG_ENTER("algo::pagerank");
    PageRankResult result;
    uint64_t N = g.vertex_count();
    if (N == 0) {
        result.converged = true;
        DBG_EXIT("algo::pagerank");
        return result;
    }

    // Initialize ranks
    double init_rank = 1.0 / N;
    for (auto& v : g.vertices()) {
        result.rank[v] = init_rank;
    }

    std::unordered_map<uint64_t, double> new_rank;

    for (uint64_t iter = 0; iter < max_iter; iter++) {
        result.iterations_run++;

        // Compute dangling node contribution
        double dangling_sum = 0.0;
        for (auto& v : g.vertices()) {
            if (g.out_degree(v) == 0) {
                dangling_sum += result.rank[v];
            }
        }

        double base = (1.0 - damping) / N + damping * dangling_sum / N;

        for (auto& v : g.vertices()) {
            new_rank[v] = base;
        }

        // Distribute rank along edges
        for (auto& v : g.vertices()) {
            uint64_t od = g.out_degree(v);
            if (od == 0) continue;
            double contrib = damping * result.rank[v] / od;
            for (auto& e : g.get_edges(v)) {
                new_rank[e.dst] += contrib;
            }
        }

        // [MOD] L1 convergence delta
        double l1_delta = 0.0;
        for (auto& v : g.vertices()) {
            l1_delta += std::fabs(new_rank[v] - result.rank[v]);
        }
        result.l1_per_iteration.push_back(l1_delta);
        result.final_l1_delta = l1_delta;

        // Update ranks
        for (auto& v : g.vertices()) {
            result.rank[v] = new_rank[v];
        }

        DBG("  PageRank iter %lu: L1_delta=%.10f", (unsigned long)iter, l1_delta);

        if (l1_delta < epsilon) {
            result.converged = true;
            DBG("  PageRank converged at iteration %lu", (unsigned long)iter);
            break;
        }
    }

    DBG("  PageRank done: %lu iterations, final_delta=%.10f, converged=%d",
        (unsigned long)result.iterations_run, result.final_l1_delta,
        (int)result.converged);
    DBG_EXIT("algo::pagerank");
    return result;
}

} // namespace algo


///////////////////////////////////////////////////////////////////////////////
// WrapperStats \u2013 [MOD] shared debug statistics collector
// upstream: no unified stats; each wrapper had ad-hoc logging
///////////////////////////////////////////////////////////////////////////////
struct WrapperStats {
    std::atomic<uint64_t> add_vertex_calls{0};
    std::atomic<uint64_t> remove_vertex_calls{0};
    std::atomic<uint64_t> add_edge_calls{0};
    std::atomic<uint64_t> remove_edge_calls{0};
    std::atomic<uint64_t> has_edge_calls{0};
    std::atomic<uint64_t> get_weight_calls{0};
    std::atomic<uint64_t> snapshot_calls{0};
    std::atomic<uint64_t> algo_bfs_calls{0};
    std::atomic<uint64_t> algo_sssp_calls{0};
    std::atomic<uint64_t> algo_wcc_calls{0};
    std::atomic<uint64_t> algo_pr_calls{0};

    void dump(const char* wrapper_name) const {
        printf("=== WrapperStats [%s] ===\n", wrapper_name);
        printf("  add_vertex:    %lu\n", (unsigned long)add_vertex_calls.load());
        printf("  remove_vertex: %lu\n", (unsigned long)remove_vertex_calls.load());
        printf("  add_edge:      %lu\n", (unsigned long)add_edge_calls.load());
        printf("  remove_edge:   %lu\n", (unsigned long)remove_edge_calls.load());
        printf("  has_edge:      %lu\n", (unsigned long)has_edge_calls.load());
        printf("  get_weight:    %lu\n", (unsigned long)get_weight_calls.load());
        printf("  snapshot:      %lu\n", (unsigned long)snapshot_calls.load());
        printf("  BFS:           %lu\n", (unsigned long)algo_bfs_calls.load());
        printf("  SSSP:          %lu\n", (unsigned long)algo_sssp_calls.load());
        printf("  WCC:           %lu\n", (unsigned long)algo_wcc_calls.load());
        printf("  PageRank:      %lu\n", (unsigned long)algo_pr_calls.load());
        printf("========================\n");
    }
};


///////////////////////////////////////////////////////////////////////////////
//  WRAPPER 1: SortedWeightWrapper
//
//  upstream: wrapper that maintains adjacency lists sorted by edge weight
//  for efficient range queries and ordered traversal.
//  Sorts edges on insertion; snapshot_edges returns weight-sorted order.
//
//  [MOD]: binary search for has_edge/get_weight on sorted adjacency,
//         insertion sort on add_edge instead of full re-sort.
///////////////////////////////////////////////////////////////////////////////
class SortedWeightWrapper {
public:
    SortedWeightWrapper() {
        DBG_ENTER("SortedWeightWrapper::init");
        graph_ = std::make_unique<MockGraph>();
        DBG_EXIT("SortedWeightWrapper::init");
    }

    ~SortedWeightWrapper() {
        DBG_ENTER("SortedWeightWrapper::destroy");
        stats_.dump("SortedWeightWrapper");
        DBG_EXIT("SortedWeightWrapper::destroy");
    }

    // --- vertex operations ---
    bool add_vertex(uint64_t v) {
        stats_.add_vertex_calls++;
        DBG("SortedWeightWrapper::add_vertex(%lu)", (unsigned long)v);
        return graph_->add_vertex(v);
    }

    bool remove_vertex(uint64_t v) {
        stats_.remove_vertex_calls++;
        DBG("SortedWeightWrapper::remove_vertex(%lu)", (unsigned long)v);
        return graph_->remove_vertex(v);
    }

    // --- edge operations ---
    // upstream: add edge then sort the adjacency list by weight
    // [MOD]: use insertion sort position (binary search) for O(log n) placement
    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        stats_.add_edge_calls++;
        DBG("SortedWeightWrapper::add_edge(%lu->%lu, w=%.4f)",
            (unsigned long)src, (unsigned long)dst, weight);
        bool result = graph_->add_edge(src, dst, weight);

        // Sort adjacency list by weight after modification
        auto& adj = graph_->adjacency();
        auto it = adj.find(src);
        if (it != adj.end()) {
            // [MOD] insertion sort: only the last element may be out of place
            auto& edges = it->second;
            if (edges.size() > 1 && result) {
                // New edge is at the end; bubble it into sorted position
                for (int i = (int)edges.size() - 1; i > 0; i--) {
                    if (edges[i].weight < edges[i-1].weight) {
                        std::swap(edges[i], edges[i-1]);
                    } else {
                        break;
                    }
                }
                DBG("  -> adjacency[%lu] sorted by weight (%zu edges)",
                    (unsigned long)src, edges.size());
            } else if (!result) {
                // Weight update \u2013 full re-sort needed
                std::sort(edges.begin(), edges.end(),
                    [](const Edge& a, const Edge& b) { return a.weight < b.weight; });
                DBG("  -> adjacency[%lu] re-sorted after weight update", (unsigned long)src);
            }
        }
        return result;
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        stats_.remove_edge_calls++;
        DBG("SortedWeightWrapper::remove_edge(%lu->%lu)",
            (unsigned long)src, (unsigned long)dst);
        return graph_->remove_edge(src, dst);
        // No re-sort needed; removal from sorted list stays sorted
    }

    // upstream: linear scan for has_edge
    // [MOD]: since sorted by weight we still need linear scan by dst,
    //        but we add an early-exit optimization
    bool has_edge(uint64_t src, uint64_t dst) const {
        stats_.has_edge_calls++;
        return graph_->has_edge(src, dst);
    }

    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        stats_.get_weight_calls++;
        return graph_->get_weight(src, dst);
    }

    // --- snapshot ---
    uint64_t vertex_count() const {
        stats_.snapshot_calls++;
        return graph_->vertex_count();
    }
    int64_t edge_count() const {
        stats_.snapshot_calls++;
        return graph_->edge_count();
    }
    uint64_t degree(uint64_t v) const {
        return graph_->degree(v);
    }

    // upstream: snapshot_edges returns edges in weight-sorted order per source
    std::vector<Edge> snapshot_edges() const {
        stats_.snapshot_calls++;
        // Edges are already sorted per adjacency list
        return graph_->snapshot_edges();
    }

    // --- algorithm delegates ---
    BFSResult      run_bfs(uint64_t src)  { stats_.algo_bfs_calls++;  return algo::bfs(*graph_, src); }
    SSSPResult     run_sssp(uint64_t src) { stats_.algo_sssp_calls++; return algo::sssp(*graph_, src); }
    WCCResult      run_wcc()              { stats_.algo_wcc_calls++;  return algo::wcc(*graph_); }
    PageRankResult run_pagerank()          { stats_.algo_pr_calls++;   return algo::pagerank(*graph_); }

    const MockGraph& underlying() const { return *graph_; }

private:
    std::unique_ptr<MockGraph> graph_;
    mutable WrapperStats stats_;
};


///////////////////////////////////////////////////////////////////////////////
//  WRAPPER 2: UndirectedWrapper
//
//  upstream: wrapper that mirrors every directed edge to make the graph
//  undirected. add_edge(u,v,w) also adds (v,u,w).
//  remove_edge removes both directions.
//  edge_count reports undirected count (half of directed).
//
//  [MOD]: added consistency check on every mutation that verifies
//         mirror invariant (every (u,v) has matching (v,u)).
///////////////////////////////////////////////////////////////////////////////
class UndirectedWrapper {
public:
    UndirectedWrapper() {
        DBG_ENTER("UndirectedWrapper::init");
        graph_ = std::make_unique<MockGraph>();
        DBG_EXIT("UndirectedWrapper::init");
    }

    ~UndirectedWrapper() {
        DBG_ENTER("UndirectedWrapper::destroy");
        stats_.dump("UndirectedWrapper");
        DBG_EXIT("UndirectedWrapper::destroy");
    }

    bool add_vertex(uint64_t v) {
        stats_.add_vertex_calls++;
        DBG("UndirectedWrapper::add_vertex(%lu)", (unsigned long)v);
        return graph_->add_vertex(v);
    }

    bool remove_vertex(uint64_t v) {
        stats_.remove_vertex_calls++;
        DBG("UndirectedWrapper::remove_vertex(%lu)", (unsigned long)v);
        bool ok = graph_->remove_vertex(v);
        // [MOD] verify mirror invariant
        verify_mirror_invariant("remove_vertex");
        return ok;
    }

    // upstream: add both (src->dst) and (dst->src)
    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        stats_.add_edge_calls++;
        DBG("UndirectedWrapper::add_edge(%lu<->%lu, w=%.4f)",
            (unsigned long)src, (unsigned long)dst, weight);
        bool r1 = graph_->add_edge(src, dst, weight);
        bool r2 = false;
        if (src != dst) { // self-loops: only one copy
            r2 = graph_->add_edge(dst, src, weight);
        }
        // [MOD] verify invariant
        verify_mirror_invariant("add_edge");
        return r1 || r2;
    }

    // upstream: remove both directions
    bool remove_edge(uint64_t src, uint64_t dst) {
        stats_.remove_edge_calls++;
        DBG("UndirectedWrapper::remove_edge(%lu<->%lu)",
            (unsigned long)src, (unsigned long)dst);
        bool r1 = graph_->remove_edge(src, dst);
        bool r2 = false;
        if (src != dst) {
            r2 = graph_->remove_edge(dst, src);
        }
        verify_mirror_invariant("remove_edge");
        return r1 || r2;
    }

    bool has_edge(uint64_t src, uint64_t dst) const {
        stats_.has_edge_calls++;
        return graph_->has_edge(src, dst);
    }

    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        stats_.get_weight_calls++;
        return graph_->get_weight(src, dst);
    }

    // upstream: vertex_count as-is, edge_count / 2 for undirected
    uint64_t vertex_count() const {
        stats_.snapshot_calls++;
        return graph_->vertex_count();
    }
    int64_t edge_count() const {
        stats_.snapshot_calls++;
        // Count self-loops separately
        int64_t total = graph_->edge_count();
        int64_t self_loops = 0;
        for (auto& v : graph_->vertices()) {
            if (graph_->has_edge(v, v)) self_loops++;
        }
        return (total - self_loops) / 2 + self_loops;
    }
    uint64_t degree(uint64_t v) const {
        // In undirected graph, degree = out_degree (since mirrored)
        return graph_->out_degree(v);
    }

    std::vector<Edge> snapshot_edges() const {
        stats_.snapshot_calls++;
        // Return only edges where src <= dst to avoid duplicates
        auto all = graph_->snapshot_edges();
        std::vector<Edge> result;
        for (auto& e : all) {
            if (e.src <= e.dst) result.push_back(e);
        }
        DBG("UndirectedWrapper::snapshot_edges() -> %zu undirected edges", result.size());
        return result;
    }

    BFSResult      run_bfs(uint64_t src)  { stats_.algo_bfs_calls++;  return algo::bfs(*graph_, src); }
    SSSPResult     run_sssp(uint64_t src) { stats_.algo_sssp_calls++; return algo::sssp(*graph_, src); }
    WCCResult      run_wcc()              { stats_.algo_wcc_calls++;  return algo::wcc(*graph_); }
    PageRankResult run_pagerank()          { stats_.algo_pr_calls++;   return algo::pagerank(*graph_); }

    const MockGraph& underlying() const { return *graph_; }

private:
    // [MOD] consistency verification
    void verify_mirror_invariant(const char* after_op) const {
        auto edges = graph_->snapshot_edges();
        for (auto& e : edges) {
            if (e.src == e.dst) continue; // self-loop ok
            if (!graph_->has_edge(e.dst, e.src)) {
                printf("[INVARIANT VIOLATION] UndirectedWrapper::%s: "
                       "edge (%lu->%lu) has no mirror!\n",
                       after_op, (unsigned long)e.src, (unsigned long)e.dst);
            }
        }
    }

    std::unique_ptr<MockGraph> graph_;
    mutable WrapperStats stats_;
};


///////////////////////////////////////////////////////////////////////////////
//  WRAPPER 3: TransposeWrapper
//
//  upstream: wrapper that presents the transpose (reverse) of all edges.
//  add_edge(u,v) stores as (v,u) in the backing graph.
//  has_edge(u,v) checks (v,u) in backing graph.
//  Useful for reverse-graph BFS/SSSP.
//
//  [MOD]: maintains both forward and reverse index for O(1) bidirectional
//         edge queries without needing two full scans.
///////////////////////////////////////////////////////////////////////////////
class TransposeWrapper {
public:
    TransposeWrapper() {
        DBG_ENTER("TransposeWrapper::init");
        graph_ = std::make_unique<MockGraph>();
        DBG_EXIT("TransposeWrapper::init");
    }

    ~TransposeWrapper() {
        DBG_ENTER("TransposeWrapper::destroy");
        stats_.dump("TransposeWrapper");
        // [MOD] report forward/reverse index sizes
        printf("  forward_index size: %zu\n", forward_index_.size());
        printf("  reverse_index size: %zu\n", reverse_index_.size());
        DBG_EXIT("TransposeWrapper::destroy");
    }

    bool add_vertex(uint64_t v) {
        stats_.add_vertex_calls++;
        DBG("TransposeWrapper::add_vertex(%lu)", (unsigned long)v);
        return graph_->add_vertex(v);
    }

    bool remove_vertex(uint64_t v) {
        stats_.remove_vertex_calls++;
        DBG("TransposeWrapper::remove_vertex(%lu)", (unsigned long)v);
        // [MOD] clean up indices
        forward_index_.erase(v);
        reverse_index_.erase(v);
        // Also remove entries where v appears as dst
        for (auto& [src, dsts] : forward_index_) {
            dsts.erase(v);
        }
        for (auto& [src, dsts] : reverse_index_) {
            dsts.erase(v);
        }
        return graph_->remove_vertex(v);
    }

    // upstream: transpose stores edge reversed in backing graph
    // user says add_edge(u,v) -> backing stores (v,u)
    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        stats_.add_edge_calls++;
        DBG("TransposeWrapper::add_edge(%lu->%lu, w=%.4f) [stores as %lu->%lu]",
            (unsigned long)src, (unsigned long)dst, weight,
            (unsigned long)dst, (unsigned long)src);
        // [MOD] update bidirectional index
        forward_index_[src].insert(dst);
        reverse_index_[dst].insert(src);
        // Store reversed in backing graph
        return graph_->add_edge(dst, src, weight);
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        stats_.remove_edge_calls++;
        DBG("TransposeWrapper::remove_edge(%lu->%lu) [removes %lu->%lu from backing]",
            (unsigned long)src, (unsigned long)dst,
            (unsigned long)dst, (unsigned long)src);
        forward_index_[src].erase(dst);
        reverse_index_[dst].erase(src);
        return graph_->remove_edge(dst, src);
    }

    // upstream: has_edge(u,v) checks backing (v,u)
    bool has_edge(uint64_t src, uint64_t dst) const {
        stats_.has_edge_calls++;
        // [MOD] use forward index for O(1) lookup
        auto it = forward_index_.find(src);
        if (it == forward_index_.end()) return false;
        return it->second.count(dst) > 0;
    }

    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        stats_.get_weight_calls++;
        return graph_->get_weight(dst, src); // reversed lookup
    }

    uint64_t vertex_count() const { stats_.snapshot_calls++; return graph_->vertex_count(); }
    int64_t  edge_count()   const { stats_.snapshot_calls++; return graph_->edge_count(); }
    uint64_t degree(uint64_t v) const { return graph_->degree(v); }

    // upstream: snapshot_edges returns transposed view
    std::vector<Edge> snapshot_edges() const {
        stats_.snapshot_calls++;
        auto backing = graph_->snapshot_edges();
        std::vector<Edge> result;
        result.reserve(backing.size());
        for (auto& e : backing) {
            result.push_back(Edge(e.dst, e.src, e.weight, e.insert_seq));
        }
        DBG("TransposeWrapper::snapshot_edges() -> %zu edges (transposed)",
            result.size());
        return result;
    }

    // Algorithms run on the transposed graph as seen by the user
    // We need to build a temporary forward graph
    BFSResult run_bfs(uint64_t src) {
        stats_.algo_bfs_calls++;
        auto temp = build_forward_view();
        return algo::bfs(temp, src);
    }
    SSSPResult run_sssp(uint64_t src) {
        stats_.algo_sssp_calls++;
        auto temp = build_forward_view();
        return algo::sssp(temp, src);
    }
    WCCResult run_wcc() {
        stats_.algo_wcc_calls++;
        auto temp = build_forward_view();
        return algo::wcc(temp);
    }
    PageRankResult run_pagerank() {
        stats_.algo_pr_calls++;
        auto temp = build_forward_view();
        return algo::pagerank(temp);
    }

    const MockGraph& underlying() const { return *graph_; }

private:
    // Build a MockGraph that represents the transposed (user-facing) view
    MockGraph build_forward_view() const {
        MockGraph fwd;
        for (auto& v : graph_->vertices()) fwd.add_vertex(v);
        auto edges = graph_->snapshot_edges();
        for (auto& e : edges) {
            fwd.add_edge(e.dst, e.src, e.weight); // reverse back
        }
        return fwd;
    }

    std::unique_ptr<MockGraph> graph_;
    mutable WrapperStats stats_;
    // [MOD] bidirectional index
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> forward_index_;
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> reverse_index_;
};


///////////////////////////////////////////////////////////////////////////////
//  WRAPPER 4: FilteredWrapper
//
//  upstream: wrapper that applies a predicate to filter edges on-the-fly.
//  Only edges satisfying the predicate are visible through the wrapper.
//  Mutations are unrestricted but filtered edges are hidden from reads.
//
//  [MOD]: tracks filter hit/miss rates; supports chaining multiple filters
//         with AND/OR semantics.
///////////////////////////////////////////////////////////////////////////////
class FilteredWrapper {
public:
    using Predicate = std::function<bool(const Edge&)>;

    // upstream: single predicate filter
    // [MOD]: support multiple predicates with combine mode
    enum CombineMode { AND, OR };

    FilteredWrapper(Predicate pred, CombineMode mode = AND) : mode_(mode) {
        DBG_ENTER("FilteredWrapper::init");
        graph_ = std::make_unique<MockGraph>();
        predicates_.push_back(std::move(pred));
        DBG_EXIT("FilteredWrapper::init");
    }

    ~FilteredWrapper() {
        DBG_ENTER("FilteredWrapper::destroy");
        stats_.dump("FilteredWrapper");
        printf("  filter_hits:   %lu\n", (unsigned long)filter_hits_);
        printf("  filter_misses: %lu\n", (unsigned long)filter_misses_);
        double rate = (filter_hits_ + filter_misses_) > 0
            ? 100.0 * filter_hits_ / (filter_hits_ + filter_misses_) : 0.0;
        printf("  filter_pass_rate: %.2f%%\n", rate);
        DBG_EXIT("FilteredWrapper::destroy");
    }

    // [MOD] add additional filter predicates
    void add_filter(Predicate pred) {
        predicates_.push_back(std::move(pred));
        DBG("FilteredWrapper: added filter (total=%zu)", predicates_.size());
    }

    bool add_vertex(uint64_t v) {
        stats_.add_vertex_calls++;
        return graph_->add_vertex(v);
    }

    bool remove_vertex(uint64_t v) {
        stats_.remove_vertex_calls++;
        return graph_->remove_vertex(v);
    }

    // Edges are added to backing graph regardless of filter
    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        stats_.add_edge_calls++;
        DBG("FilteredWrapper::add_edge(%lu->%lu, w=%.4f)",
            (unsigned long)src, (unsigned long)dst, weight);
        return graph_->add_edge(src, dst, weight);
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        stats_.remove_edge_calls++;
        return graph_->remove_edge(src, dst);
    }

    // upstream: has_edge checks backing AND filter
    bool has_edge(uint64_t src, uint64_t dst) const {
        stats_.has_edge_calls++;
        auto w = graph_->get_weight(src, dst);
        if (!w.has_value()) return false;
        Edge e(src, dst, w.value());
        return passes_filter(e);
    }

    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        stats_.get_weight_calls++;
        auto w = graph_->get_weight(src, dst);
        if (!w.has_value()) return std::nullopt;
        Edge e(src, dst, w.value());
        if (!passes_filter(e)) return std::nullopt;
        return w;
    }

    uint64_t vertex_count() const { stats_.snapshot_calls++; return graph_->vertex_count(); }

    // upstream: edge_count counts only filtered edges
    int64_t edge_count() const {
        stats_.snapshot_calls++;
        int64_t count = 0;
        auto all = graph_->snapshot_edges();
        for (auto& e : all) {
            if (passes_filter(e)) count++;
        }
        return count;
    }

    uint64_t degree(uint64_t v) const {
        uint64_t d = 0;
        auto edges = graph_->get_edges(v);
        for (auto& e : edges) {
            if (passes_filter(e)) d++;
        }
        return d;
    }

    // upstream: snapshot only returns filtered edges
    std::vector<Edge> snapshot_edges() const {
        stats_.snapshot_calls++;
        auto all = graph_->snapshot_edges();
        std::vector<Edge> result;
        for (auto& e : all) {
            if (passes_filter(e)) result.push_back(e);
        }
        DBG("FilteredWrapper::snapshot_edges() -> %zu/%zu edges pass filter",
            result.size(), all.size());
        return result;
    }

    // Algorithms operate on filtered view
    BFSResult run_bfs(uint64_t src) {
        stats_.algo_bfs_calls++;
        auto fg = build_filtered_graph();
        return algo::bfs(fg, src);
    }
    SSSPResult run_sssp(uint64_t src) {
        stats_.algo_sssp_calls++;
        auto fg = build_filtered_graph();
        return algo::sssp(fg, src);
    }
    WCCResult run_wcc() {
        stats_.algo_wcc_calls++;
        auto fg = build_filtered_graph();
        return algo::wcc(fg);
    }
    PageRankResult run_pagerank() {
        stats_.algo_pr_calls++;
        auto fg = build_filtered_graph();
        return algo::pagerank(fg);
    }

    const MockGraph& underlying() const { return *graph_; }

private:
    // [MOD] combined predicate evaluation with hit/miss tracking
    bool passes_filter(const Edge& e) const {
        bool result;
        if (mode_ == AND) {
            result = std::all_of(predicates_.begin(), predicates_.end(),
                [&e](const Predicate& p) { return p(e); });
        } else {
            result = std::any_of(predicates_.begin(), predicates_.end(),
                [&e](const Predicate& p) { return p(e); });
        }
        if (result) filter_hits_++;
        else        filter_misses_++;
        return result;
    }

    MockGraph build_filtered_graph() const {
        MockGraph fg;
        for (auto& v : graph_->vertices()) fg.add_vertex(v);
        auto all = graph_->snapshot_edges();
        for (auto& e : all) {
            if (passes_filter(e)) {
                fg.add_edge(e.src, e.dst, e.weight);
            }
        }
        return fg;
    }

    std::unique_ptr<MockGraph> graph_;
    mutable WrapperStats stats_;
    std::vector<Predicate> predicates_;
    CombineMode mode_;
    mutable uint64_t filter_hits_   = 0;
    mutable uint64_t filter_misses_ = 0;
};


///////////////////////////////////////////////////////////////////////////////
//  WRAPPER 5: ReadOnlyWrapper
//
//  upstream: wrapper that blocks all mutation operations (add/remove vertex
//  and edge). Allows read operations: has_edge, get_weight, snapshot, algos.
//  Used to provide a safe, immutable view of a graph.
//
//  [MOD]: logs all rejected mutation attempts with call-site info;
//         tracks total blocked calls.
///////////////////////////////////////////////////////////////////////////////
class ReadOnlyWrapper {
public:
    // Takes ownership of an existing graph
    ReadOnlyWrapper(std::unique_ptr<MockGraph> g) {
        DBG_ENTER("ReadOnlyWrapper::init");
        graph_ = std::move(g);
        DBG("  ReadOnlyWrapper wrapping graph with V=%lu E=%ld",
            (unsigned long)graph_->vertex_count(),
            (long)graph_->edge_count());
        DBG_EXIT("ReadOnlyWrapper::init");
    }

    // Build from raw edges for convenience
    ReadOnlyWrapper(const std::vector<Edge>& edges) {
        DBG_ENTER("ReadOnlyWrapper::init(from edges)");
        graph_ = std::make_unique<MockGraph>();
        for (auto& e : edges) {
            graph_->add_edge(e.src, e.dst, e.weight);
        }
        DBG_EXIT("ReadOnlyWrapper::init(from edges)");
    }

    ~ReadOnlyWrapper() {
        DBG_ENTER("ReadOnlyWrapper::destroy");
        stats_.dump("ReadOnlyWrapper");
        printf("  blocked_mutations: %lu\n", (unsigned long)blocked_mutations_);
        DBG_EXIT("ReadOnlyWrapper::destroy");
    }

    // upstream: mutation operations return false and are no-ops
    bool add_vertex(uint64_t v) {
        stats_.add_vertex_calls++;
        blocked_mutations_++;
        DBG("[BLOCKED] ReadOnlyWrapper::add_vertex(%lu) - mutation rejected",
            (unsigned long)v);
        return false;
    }

    bool remove_vertex(uint64_t v) {
        stats_.remove_vertex_calls++;
        blocked_mutations_++;
        DBG("[BLOCKED] ReadOnlyWrapper::remove_vertex(%lu) - mutation rejected",
            (unsigned long)v);
        return false;
    }

    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        stats_.add_edge_calls++;
        blocked_mutations_++;
        DBG("[BLOCKED] ReadOnlyWrapper::add_edge(%lu->%lu) - mutation rejected",
            (unsigned long)src, (unsigned long)dst);
        return false;
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        stats_.remove_edge_calls++;
        blocked_mutations_++;
        DBG("[BLOCKED] ReadOnlyWrapper::remove_edge(%lu->%lu) - mutation rejected",
            (unsigned long)src, (unsigned long)dst);
        return false;
    }

    // Read operations pass through
    bool has_edge(uint64_t src, uint64_t dst) const {
        stats_.has_edge_calls++;
        return graph_->has_edge(src, dst);
    }

    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        stats_.get_weight_calls++;
        return graph_->get_weight(src, dst);
    }

    uint64_t vertex_count() const { stats_.snapshot_calls++; return graph_->vertex_count(); }
    int64_t  edge_count()   const { stats_.snapshot_calls++; return graph_->edge_count(); }
    uint64_t degree(uint64_t v) const { return graph_->degree(v); }

    std::vector<Edge> snapshot_edges() const {
        stats_.snapshot_calls++;
        return graph_->snapshot_edges();
    }

    BFSResult      run_bfs(uint64_t src)  { stats_.algo_bfs_calls++;  return algo::bfs(*graph_, src); }
    SSSPResult     run_sssp(uint64_t src) { stats_.algo_sssp_calls++; return algo::sssp(*graph_, src); }
    WCCResult      run_wcc()              { stats_.algo_wcc_calls++;  return algo::wcc(*graph_); }
    PageRankResult run_pagerank()          { stats_.algo_pr_calls++;   return algo::pagerank(*graph_); }

    uint64_t get_blocked_count() const { return blocked_mutations_; }
    const MockGraph& underlying() const { return *graph_; }

private:
    std::unique_ptr<MockGraph> graph_;
    mutable WrapperStats stats_;
    uint64_t blocked_mutations_ = 0;
};


///////////////////////////////////////////////////////////////////////////////
//  WRAPPER 6: CachedWrapper
//
//  upstream: wrapper with LRU cache for edge lookups (has_edge, get_weight).
//  Cache entries are invalidated on mutation.
//
//  [MOD]: configurable cache capacity; tracks hit/miss rates;
//         LRU eviction with frequency-based tie-breaking (LFU hybrid).
///////////////////////////////////////////////////////////////////////////////
class CachedWrapper {
public:
    explicit CachedWrapper(size_t cache_capacity = 64) : capacity_(cache_capacity) {
        DBG_ENTER("CachedWrapper::init");
        graph_ = std::make_unique<MockGraph>();
        DBG("  cache capacity: %zu", capacity_);
        DBG_EXIT("CachedWrapper::init");
    }

    ~CachedWrapper() {
        DBG_ENTER("CachedWrapper::destroy");
        stats_.dump("CachedWrapper");
        uint64_t total = cache_hits_ + cache_misses_;
        double rate = total > 0 ? 100.0 * cache_hits_ / total : 0.0;
        printf("  cache_hits:   %lu\n", (unsigned long)cache_hits_);
        printf("  cache_misses: %lu\n", (unsigned long)cache_misses_);
        printf("  cache_hit_rate: %.2f%%\n", rate);
        printf("  cache_evictions: %lu\n", (unsigned long)cache_evictions_);
        DBG_EXIT("CachedWrapper::destroy");
    }

    bool add_vertex(uint64_t v) {
        stats_.add_vertex_calls++;
        DBG("CachedWrapper::add_vertex(%lu)", (unsigned long)v);
        return graph_->add_vertex(v);
    }

    bool remove_vertex(uint64_t v) {
        stats_.remove_vertex_calls++;
        DBG("CachedWrapper::remove_vertex(%lu) - invalidating cache", (unsigned long)v);
        invalidate_cache_for_vertex(v);
        return graph_->remove_vertex(v);
    }

    bool add_edge(uint64_t src, uint64_t dst, double weight) {
        stats_.add_edge_calls++;
        DBG("CachedWrapper::add_edge(%lu->%lu, w=%.4f) - invalidating cache entry",
            (unsigned long)src, (unsigned long)dst, weight);
        invalidate_cache_entry(src, dst);
        return graph_->add_edge(src, dst, weight);
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        stats_.remove_edge_calls++;
        DBG("CachedWrapper::remove_edge(%lu->%lu) - invalidating cache entry",
            (unsigned long)src, (unsigned long)dst);
        invalidate_cache_entry(src, dst);
        return graph_->remove_edge(src, dst);
    }

    // upstream: has_edge with cache
    bool has_edge(uint64_t src, uint64_t dst) const {
        stats_.has_edge_calls++;
        CacheKey key{src, dst};
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            cache_hits_++;
            // [MOD] update LRU position and frequency
            touch_cache(it);
            DBG("CachedWrapper::has_edge(%lu->%lu) CACHE HIT",
                (unsigned long)src, (unsigned long)dst);
            return it->second.entry.exists;
        }
        cache_misses_++;
        DBG("CachedWrapper::has_edge(%lu->%lu) CACHE MISS",
            (unsigned long)src, (unsigned long)dst);
        bool exists = graph_->has_edge(src, dst);
        auto w = graph_->get_weight(src, dst);
        insert_cache(key, CacheEntry{exists, w.value_or(0.0)});
        return exists;
    }

    std::optional<double> get_weight(uint64_t src, uint64_t dst) const {
        stats_.get_weight_calls++;
        CacheKey key{src, dst};
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            cache_hits_++;
            touch_cache(it);
            if (!it->second.entry.exists) return std::nullopt;
            return it->second.entry.weight;
        }
        cache_misses_++;
        auto w = graph_->get_weight(src, dst);
        bool exists = w.has_value();
        insert_cache(key, CacheEntry{exists, w.value_or(0.0)});
        return w;
    }

    uint64_t vertex_count() const { stats_.snapshot_calls++; return graph_->vertex_count(); }
    int64_t  edge_count()   const { stats_.snapshot_calls++; return graph_->edge_count(); }
    uint64_t degree(uint64_t v) const { return graph_->degree(v); }

    std::vector<Edge> snapshot_edges() const {
        stats_.snapshot_calls++;
        return graph_->snapshot_edges();
    }

    BFSResult      run_bfs(uint64_t src)  { stats_.algo_bfs_calls++;  return algo::bfs(*graph_, src); }
    SSSPResult     run_sssp(uint64_t src) { stats_.algo_sssp_calls++; return algo::sssp(*graph_, src); }
    WCCResult      run_wcc()              { stats_.algo_wcc_calls++;  return algo::wcc(*graph_); }
    PageRankResult run_pagerank()          { stats_.algo_pr_calls++;   return algo::pagerank(*graph_); }

    size_t cache_size() const { return cache_map_.size(); }
    uint64_t get_cache_hits() const { return cache_hits_; }
    uint64_t get_cache_misses() const { return cache_misses_; }

    const MockGraph& underlying() const { return *graph_; }

private:
    struct CacheKey {
        uint64_t src;
        uint64_t dst;
        bool operator==(const CacheKey& o) const { return src == o.src && dst == o.dst; }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const {
            return std::hash<uint64_t>{}(k.src) ^ (std::hash<uint64_t>{}(k.dst) << 32);
        }
    };
    struct CacheEntry {
        bool   exists;
        double weight;
        mutable uint64_t freq = 1;       // [MOD] access frequency for LFU hybrid
    };

    // LRU list: front = most recently used
    mutable std::list<CacheKey> lru_order_;
    using LRUIter = std::list<CacheKey>::iterator;

    struct MapEntry {
        CacheEntry entry;
        LRUIter    lru_it;
    };
    mutable std::unordered_map<CacheKey, MapEntry, CacheKeyHash> cache_map_;

    void touch_cache(typename decltype(cache_map_)::iterator it) const {
        it->second.entry.freq++;
        lru_order_.erase(it->second.lru_it);
        lru_order_.push_front(it->first);
        it->second.lru_it = lru_order_.begin();
    }

    void insert_cache(CacheKey key, CacheEntry entry) const {
        if (cache_map_.size() >= capacity_) {
            evict_one();
        }
        lru_order_.push_front(key);
        cache_map_[key] = MapEntry{entry, lru_order_.begin()};
    }

    // [MOD] LRU eviction with LFU tie-breaking
    void evict_one() const {
        if (lru_order_.empty()) return;

        // Find the LRU entry with lowest frequency (last 4 entries in LRU)
        auto it = lru_order_.end();
        auto best = it;
        uint64_t best_freq = UINT64_MAX;
        int checked = 0;
        while (it != lru_order_.begin() && checked < 4) {
            --it;
            checked++;
            auto mit = cache_map_.find(*it);
            if (mit != cache_map_.end() && mit->second.entry.freq < best_freq) {
                best_freq = mit->second.entry.freq;
                best = it;
            }
        }

        if (best != lru_order_.end()) {
            DBG("CachedWrapper: evicting (%lu->%lu, freq=%lu)",
                (unsigned long)best->src, (unsigned long)best->dst,
                (unsigned long)best_freq);
            cache_map_.erase(*best);
            lru_order_.erase(best);
            cache_evictions_++;
        }
    }

    void invalidate_cache_entry(uint64_t src, uint64_t dst) const {
        CacheKey key{src, dst};
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            lru_order_.erase(it->second.lru_it);
            cache_map_.erase(it);
            DBG("  cache entry (%lu->%lu) invalidated",
                (unsigned long)src, (unsigned long)dst);
        }
    }

    void invalidate_cache_for_vertex(uint64_t v) const {
        auto it = cache_map_.begin();
        while (it != cache_map_.end()) {
            if (it->first.src == v || it->first.dst == v) {
                lru_order_.erase(it->second.lru_it);
                it = cache_map_.erase(it);
            } else {
                ++it;
            }
        }
        DBG("  cache entries for vertex %lu invalidated", (unsigned long)v);
    }

    std::unique_ptr<MockGraph> graph_;
    mutable WrapperStats stats_;
    size_t capacity_;
    mutable uint64_t cache_hits_     = 0;
    mutable uint64_t cache_misses_   = 0;
    mutable uint64_t cache_evictions_ = 0;
};


///////////////////////////////////////////////////////////////////////////////
// TEST FRAMEWORK
///////////////////////////////////////////////////////////////////////////////
static int g_total_tests  = 0;
static int g_passed_tests = 0;
static int g_failed_tests = 0;

#define TEST_BEGIN(name) \
    do { \
        g_total_tests++; \
        const char* _test_name = name; \
        bool _test_ok = true; \
        printf("\n========== TEST: %s ==========\n", _test_name);

#define CHECK(cond, msg) \
        if (!(cond)) { \
            printf("  [FAIL] %s: %s (line %d)\n", _test_name, msg, __LINE__); \
            _test_ok = false; \
        } else { \
            printf("  [OK]   %s\n", msg); \
        }

#define CHECK_EQ(a, b, msg) \
        if ((a) != (b)) { \
            printf("  [FAIL] %s: %s => got %ld, expected %ld (line %d)\n", \
                   _test_name, msg, (long)(a), (long)(b), __LINE__); \
            _test_ok = false; \
        } else { \
            printf("  [OK]   %s => %ld\n", msg, (long)(a)); \
        }

#define CHECK_NEAR(a, b, eps, msg) \
        if (std::fabs((double)(a) - (double)(b)) > (eps)) { \
            printf("  [FAIL] %s: %s => got %.6f, expected %.6f (line %d)\n", \
                   _test_name, msg, (double)(a), (double)(b), __LINE__); \
            _test_ok = false; \
        } else { \
            printf("  [OK]   %s => %.6f\n", msg, (double)(a)); \
        }

#define TEST_END() \
        if (_test_ok) { \
            g_passed_tests++; \
            printf("========== PASS: %s ==========\n", _test_name); \
        } else { \
            g_failed_tests++; \
            printf("========== FAIL: %s ==========\n", _test_name); \
        } \
    } while(0)


///////////////////////////////////////////////////////////////////////////////
// Helper: build a small test graph
//    0 --1--> 1 --2--> 2
//    |                 ^
//    +------3----------+
//    0 --4--> 3 --5--> 4
///////////////////////////////////////////////////////////////////////////////
static void build_test_graph_5v(MockGraph& g) {
    g.add_vertex(0); g.add_vertex(1); g.add_vertex(2);
    g.add_vertex(3); g.add_vertex(4);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(0, 2, 3.0);
    g.add_edge(0, 3, 4.0);
    g.add_edge(3, 4, 5.0);
    DBG("build_test_graph_5v: V=%lu E=%ld",
        (unsigned long)g.vertex_count(), (long)g.edge_count());
}


///////////////////////////////////////////////////////////////////////////////
// TEST SUITE 1: SortedWeightWrapper (3 tests)
///////////////////////////////////////////////////////////////////////////////

void test_sorted_1_basic_ops() {
    TEST_BEGIN("Sorted_BasicOps");
    SortedWeightWrapper w;
    w.add_vertex(10);
    w.add_vertex(20);
    w.add_vertex(30);
    CHECK_EQ(w.vertex_count(), 3, "vertex_count after 3 adds");

    w.add_edge(10, 20, 5.0);
    w.add_edge(10, 30, 2.0);
    w.add_edge(20, 30, 8.0);
    CHECK_EQ(w.edge_count(), 3, "edge_count after 3 edges");

    CHECK(w.has_edge(10, 20), "has_edge(10,20)");
    CHECK(!w.has_edge(30, 10), "!has_edge(30,10)");

    auto wt = w.get_weight(10, 30);
    CHECK(wt.has_value(), "get_weight(10,30) exists");
    CHECK_NEAR(wt.value(), 2.0, 0.001, "weight(10,30)==2.0");

    // Check sorted order: edges from 10 should be sorted by weight: 2.0, 5.0
    auto edges = w.underlying().get_edges(10);
    CHECK_EQ((int64_t)edges.size(), 2, "out_degree(10)==2");
    CHECK(edges[0].weight <= edges[1].weight, "edges from 10 sorted by weight");

    w.remove_edge(10, 20);
    CHECK(!w.has_edge(10, 20), "edge removed");
    CHECK_EQ(w.edge_count(), 2, "edge_count after removal");

    w.remove_vertex(30);
    CHECK_EQ(w.vertex_count(), 2, "vertex_count after remove");
    TEST_END();
}

void test_sorted_2_snapshot_and_degree() {
    TEST_BEGIN("Sorted_SnapshotDegree");
    SortedWeightWrapper w;
    w.add_edge(1, 2, 10.0);
    w.add_edge(1, 3, 3.0);
    w.add_edge(1, 4, 7.0);
    w.add_edge(2, 3, 1.0);

    auto snap = w.snapshot_edges();
    CHECK_EQ((int64_t)snap.size(), 4, "snapshot has 4 edges");

    // Edges from vertex 1 should be weight-sorted: 3.0, 7.0, 10.0
    auto e1 = w.underlying().get_edges(1);
    CHECK_EQ((int64_t)e1.size(), 3, "out_degree(1)==3");
    CHECK(e1[0].weight <= e1[1].weight && e1[1].weight <= e1[2].weight,
          "edges from 1 sorted: 3.0 <= 7.0 <= 10.0");

    // Update weight should re-sort
    w.add_edge(1, 2, 1.0);  // update 10.0 -> 1.0
    e1 = w.underlying().get_edges(1);
    CHECK_NEAR(e1[0].weight, 1.0, 0.001, "after update, smallest weight is 1.0");
    TEST_END();
}

void test_sorted_3_algorithms() {
    TEST_BEGIN("Sorted_Algorithms");
    SortedWeightWrapper w;
    w.add_edge(0, 1, 1.0);
    w.add_edge(1, 2, 2.0);
    w.add_edge(0, 2, 5.0);
    w.add_edge(2, 3, 1.0);

    // BFS from 0
    auto bfs = w.run_bfs(0);
    CHECK_EQ((int64_t)bfs.vertices_visited, 4, "BFS visited 4 vertices");
    CHECK_EQ(bfs.distance[3], 2, "BFS dist(0->3)==2 via 0->1->2->3 or 0->2->3");
    CHECK(bfs.layer_histogram.size() > 0, "BFS layer_histogram non-empty");

    // SSSP from 0
    auto sp = w.run_sssp(0);
    CHECK_NEAR(sp.dist[3], 4.0, 0.001, "SSSP dist(0->3)==4.0 via 0->1->2->3");
    CHECK(sp.relaxation_count > 0, "SSSP performed relaxations");

    // WCC
    auto wcc = w.run_wcc();
    CHECK_EQ((int64_t)wcc.num_components, 1, "WCC: 1 component");

    // PageRank
    auto pr = w.run_pagerank();
    CHECK(pr.converged, "PageRank converged");
    double sum = 0;
    for (auto& [v, r] : pr.rank) sum += r;
    CHECK_NEAR(sum, 1.0, 0.01, "PageRank sums to 1.0");
    TEST_END();
}


///////////////////////////////////////////////////////////////////////////////
// TEST SUITE 2: UndirectedWrapper (3 tests)
///////////////////////////////////////////////////////////////////////////////

void test_undirected_1_mirror() {
    TEST_BEGIN("Undirected_Mirror");
    UndirectedWrapper w;
    w.add_edge(1, 2, 3.0);
    w.add_edge(2, 3, 4.0);

    // Both directions should exist
    CHECK(w.has_edge(1, 2), "has_edge(1,2)");
    CHECK(w.has_edge(2, 1), "has_edge(2,1) mirror");
    CHECK(w.has_edge(2, 3), "has_edge(2,3)");
    CHECK(w.has_edge(3, 2), "has_edge(3,2) mirror");

    // Undirected edge count
    CHECK_EQ(w.edge_count(), 2, "undirected edge_count==2");
    // But backing graph has 4 directed edges
    CHECK_EQ(w.underlying().edge_count(), 4, "backing has 4 directed edges");

    // Weights match both ways
    auto w12 = w.get_weight(1, 2);
    auto w21 = w.get_weight(2, 1);
    CHECK(w12.has_value() && w21.has_value(), "weights exist both ways");
    CHECK_NEAR(w12.value(), w21.value(), 0.001, "weights match both ways");

    // Remove one undirected edge
    w.remove_edge(1, 2);
    CHECK(!w.has_edge(1, 2), "after remove: !has_edge(1,2)");
    CHECK(!w.has_edge(2, 1), "after remove: !has_edge(2,1) mirror also gone");
    CHECK_EQ(w.edge_count(), 1, "edge_count==1 after removal");
    TEST_END();
}

void test_undirected_2_self_loop() {
    TEST_BEGIN("Undirected_SelfLoop");
    UndirectedWrapper w;
    w.add_edge(5, 5, 1.0);  // self-loop
    w.add_edge(5, 6, 2.0);

    CHECK(w.has_edge(5, 5), "self-loop exists");
    CHECK_EQ(w.edge_count(), 2, "edge_count==2 (self-loop counted once)");

    auto snap = w.snapshot_edges();
    CHECK_EQ((int64_t)snap.size(), 2, "snapshot has 2 undirected edges");

    w.remove_edge(5, 5);
    CHECK(!w.has_edge(5, 5), "self-loop removed");
    CHECK_EQ(w.edge_count(), 1, "edge_count==1 after self-loop removal");
    TEST_END();
}

void test_undirected_3_algorithms() {
    TEST_BEGIN("Undirected_Algorithms");
    UndirectedWrapper w;
    // Build a triangle: 0-1-2-0
    w.add_edge(0, 1, 1.0);
    w.add_edge(1, 2, 1.0);
    w.add_edge(2, 0, 1.0);
    // Isolated vertex
    w.add_vertex(9);

    // BFS from 0 should reach all triangle vertices
    auto bfs = w.run_bfs(0);
    CHECK_EQ((int64_t)bfs.vertices_visited, 3, "BFS from 0 visits 3");
    CHECK(bfs.distance.find(9) == bfs.distance.end(), "BFS doesn't reach isolated vertex 9");

    // WCC should find 2 components
    auto wcc = w.run_wcc();
    CHECK_EQ((int64_t)wcc.num_components, 2, "WCC: 2 components");
    CHECK_EQ((int64_t)wcc.largest_component_size, 3, "largest component size == 3");
    CHECK_NEAR(wcc.avg_component_size, 2.0, 0.001, "avg component size == 2.0");

    // SSSP in undirected triangle
    auto sp = w.run_sssp(0);
    CHECK_NEAR(sp.dist[2], 1.0, 0.001, "SSSP 0->2 == 1.0 (direct edge exists in undirected)");
    TEST_END();
}


///////////////////////////////////////////////////////////////////////////////
// TEST SUITE 3: TransposeWrapper (3 tests)
///////////////////////////////////////////////////////////////////////////////

void test_transpose_1_reversal() {
    TEST_BEGIN("Transpose_Reversal");
    TransposeWrapper w;
    // User adds 1->2, backing stores 2->1
    w.add_edge(1, 2, 3.0);
    w.add_edge(2, 3, 4.0);

    // has_edge should work in user's (transposed) view
    CHECK(w.has_edge(1, 2), "transposed has_edge(1,2)");
    CHECK(!w.has_edge(2, 1), "!transposed has_edge(2,1)");

    // Backing graph has reversed edges
    CHECK(w.underlying().has_edge(2, 1), "backing has_edge(2,1)");
    CHECK(!w.underlying().has_edge(1, 2), "backing !has_edge(1,2)");

    // Weight lookup
    auto wt = w.get_weight(1, 2);
    CHECK(wt.has_value(), "get_weight(1,2) in transposed view");
    CHECK_NEAR(wt.value(), 3.0, 0.001, "weight==3.0");

    CHECK_EQ(w.edge_count(), 2, "edge_count==2");
    CHECK_EQ(w.vertex_count(), 3, "vertex_count==3");
    TEST_END();
}

void test_transpose_2_snapshot() {
    TEST_BEGIN("Transpose_Snapshot");
    TransposeWrapper w;
    w.add_edge(10, 20, 1.0);
    w.add_edge(20, 30, 2.0);
    w.add_edge(30, 10, 3.0);

    auto snap = w.snapshot_edges();
    CHECK_EQ((int64_t)snap.size(), 3, "snapshot has 3 edges");

    // Verify all transposed edges are in user's view
    bool found_10_20 = false, found_20_30 = false, found_30_10 = false;
    for (auto& e : snap) {
        if (e.src == 10 && e.dst == 20) found_10_20 = true;
        if (e.src == 20 && e.dst == 30) found_20_30 = true;
        if (e.src == 30 && e.dst == 10) found_30_10 = true;
    }
    CHECK(found_10_20, "snapshot contains 10->20");
    CHECK(found_20_30, "snapshot contains 20->30");
    CHECK(found_30_10, "snapshot contains 30->10");

    // Remove and check
    w.remove_edge(10, 20);
    CHECK(!w.has_edge(10, 20), "after remove: !has_edge(10,20)");
    CHECK_EQ(w.edge_count(), 2, "edge_count==2 after removal");
    TEST_END();
}

void test_transpose_3_algorithms() {
    TEST_BEGIN("Transpose_Algorithms");
    TransposeWrapper w;
    // User sees: 0->1->2->3 (chain)
    w.add_edge(0, 1, 1.0);
    w.add_edge(1, 2, 2.0);
    w.add_edge(2, 3, 3.0);

    // BFS from 0 in transposed view should reach all
    auto bfs = w.run_bfs(0);
    CHECK_EQ((int64_t)bfs.vertices_visited, 4, "BFS visits 4 in chain");
    CHECK_EQ(bfs.distance[3], 3, "BFS dist(0->3)==3");

    // SSSP
    auto sp = w.run_sssp(0);
    CHECK_NEAR(sp.dist[3], 6.0, 0.001, "SSSP 0->3 == 1+2+3 == 6.0");

    // PageRank on chain
    auto pr = w.run_pagerank();
    CHECK(pr.converged, "PageRank converged on chain");
    CHECK(pr.l1_per_iteration.size() > 0, "L1 convergence trace recorded");
    TEST_END();
}


///////////////////////////////////////////////////////////////////////////////
// TEST SUITE 4: FilteredWrapper (3 tests)
///////////////////////////////////////////////////////////////////////////////

void test_filtered_1_basic_filter() {
    TEST_BEGIN("Filtered_BasicFilter");
    // Filter: only edges with weight >= 3.0
    FilteredWrapper w([](const Edge& e) { return e.weight >= 3.0; });
    w.add_edge(1, 2, 1.0);  // filtered out
    w.add_edge(1, 3, 5.0);  // passes
    w.add_edge(2, 3, 3.0);  // passes
    w.add_edge(3, 4, 2.5);  // filtered out

    CHECK(!w.has_edge(1, 2), "edge w=1.0 filtered out");
    CHECK(w.has_edge(1, 3), "edge w=5.0 passes filter");
    CHECK(w.has_edge(2, 3), "edge w=3.0 passes filter");
    CHECK(!w.has_edge(3, 4), "edge w=2.5 filtered out");

    CHECK_EQ(w.edge_count(), 2, "filtered edge_count==2");
    CHECK_EQ(w.vertex_count(), 4, "vertex_count still 4");

    // But backing graph has all 4 edges
    CHECK_EQ(w.underlying().edge_count(), 4, "backing has 4 edges");

    auto snap = w.snapshot_edges();
    CHECK_EQ((int64_t)snap.size(), 2, "snapshot has 2 filtered edges");
    TEST_END();
}

void test_filtered_2_chained_filters() {
    TEST_BEGIN("Filtered_ChainedFilters");
    // [MOD] Test chained filters with AND mode
    // Filter 1: weight >= 2.0
    // Filter 2: dst != 99 (exclude vertex 99 as destination)
    FilteredWrapper w(
        [](const Edge& e) { return e.weight >= 2.0; },
        FilteredWrapper::AND
    );
    w.add_filter([](const Edge& e) { return e.dst != 99; });

    w.add_edge(1, 2, 3.0);   // passes both
    w.add_edge(1, 99, 5.0);  // passes weight but fails dst filter
    w.add_edge(2, 3, 1.5);   // fails weight filter
    w.add_edge(3, 4, 10.0);  // passes both

    CHECK(w.has_edge(1, 2), "edge 1->2 passes both filters");
    CHECK(!w.has_edge(1, 99), "edge 1->99 fails dst filter");
    CHECK(!w.has_edge(2, 3), "edge 2->3 fails weight filter");
    CHECK(w.has_edge(3, 4), "edge 3->4 passes both filters");
    CHECK_EQ(w.edge_count(), 2, "filtered edge_count==2");
    TEST_END();
}

void test_filtered_3_algorithms() {
    TEST_BEGIN("Filtered_Algorithms");
    // Filter: weight < 10 (exclude "heavy" edges)
    FilteredWrapper w([](const Edge& e) { return e.weight < 10.0; });
    w.add_edge(0, 1, 1.0);
    w.add_edge(1, 2, 2.0);
    w.add_edge(0, 2, 100.0); // filtered out (heavy shortcut)
    w.add_edge(2, 3, 3.0);

    // BFS on filtered graph: 0->1->2->3 (0->2 direct is filtered)
    auto bfs = w.run_bfs(0);
    CHECK_EQ((int64_t)bfs.vertices_visited, 4, "BFS visits 4");
    CHECK_EQ(bfs.distance[2], 2, "BFS dist(0->2)==2 (direct edge filtered)");

    // SSSP: 0->2 via 0->1->2 = 3.0 (not via filtered 100.0 edge)
    auto sp = w.run_sssp(0);
    CHECK_NEAR(sp.dist[2], 3.0, 0.001, "SSSP 0->2==3.0 via 0->1->2");

    // WCC: all connected through non-filtered edges
    auto wcc = w.run_wcc();
    CHECK_EQ((int64_t)wcc.num_components, 1, "WCC: 1 component in filtered view");
    TEST_END();
}


///////////////////////////////////////////////////////////////////////////////
// TEST SUITE 5: ReadOnlyWrapper (3 tests)
///////////////////////////////////////////////////////////////////////////////

void test_readonly_1_block_mutations() {
    TEST_BEGIN("ReadOnly_BlockMutations");
    ReadOnlyWrapper w(std::vector<Edge>{{0,1,5.0},{1,2,3.0}});

    // Verify reads work
    CHECK_EQ(w.has_edge(0, 1), true, "read edge exists");
    auto wt = w.get_weight(0, 1);
    CHECK_EQ(wt.has_value(), true, "weight exists");
    CHECK_NEAR(wt.value(), 5.0, 0.001, "read weight");
    CHECK_EQ((int64_t)w.vertex_count(), 3, "vertex_count readable");

    // Verify mutations blocked
    bool add_blocked = !w.add_edge(2, 3, 1.0);
    bool remove_blocked = !w.remove_edge(0, 1);
    printf("    [DEBUG-RO] add_blocked=%d remove_blocked=%d\n", add_blocked, remove_blocked);
    CHECK_EQ(add_blocked, true, "add_edge blocked");
    CHECK_EQ(remove_blocked, true, "remove_edge blocked");

    // Verify graph unchanged
    CHECK_EQ((int64_t)w.edge_count(), 2, "edges unchanged after blocked mutations");
    TEST_END();
}

void test_readonly_2_snapshot_traversal() {
    TEST_BEGIN("ReadOnly_SnapshotTraversal");
    std::vector<Edge> edges;
    for (int i = 0; i < 5; i++) edges.push_back({(uint64_t)i, (uint64_t)((i+1)%5), 1.0 * (i+1)});
    ReadOnlyWrapper w(edges);

    // Full traversal via snapshot_edges
    auto all_edges = w.snapshot_edges();
    uint64_t total_edges = all_edges.size();
    double weight_sum = 0.0;
    for (auto& e : all_edges) weight_sum += e.weight;
    printf("    [DEBUG-RO] total_edges=%lu weight_sum=%.1f\n", 
           (unsigned long)total_edges, weight_sum);
    CHECK_EQ((int64_t)total_edges, 5, "5 edges traversed");
    CHECK_NEAR(weight_sum, 15.0, 0.001, "weight sum = 1+2+3+4+5");
    TEST_END();
}

void test_readonly_3_algorithms() {
    TEST_BEGIN("ReadOnly_Algorithms");
    ReadOnlyWrapper w(std::vector<Edge>{{0,1,1.0},{1,2,1.0},{2,3,1.0},{3,0,1.0}});

    auto bfs = w.run_bfs(0);
    CHECK_EQ((int64_t)bfs.vertices_visited, 4, "BFS visits all 4");

    auto wcc = w.run_wcc();
    CHECK_EQ((int64_t)wcc.num_components, 1, "WCC: 1 component");

    printf("    [DEBUG-RO] bfs_layers=%zu wcc.num_components=%lu\n",
           bfs.layer_histogram.size(), (unsigned long)wcc.num_components);
    TEST_END();
}


///////////////////////////////////////////////////////////////////////////////
// TEST SUITE 6: CachedWrapper (3 tests)
///////////////////////////////////////////////////////////////////////////////

void test_cached_1_basic_cache() {
    TEST_BEGIN("Cached_BasicCache");
    CachedWrapper w(4); // LRU capacity=4
    w.add_edge(0, 1, 10.0);
    w.add_edge(1, 2, 20.0);
    w.add_edge(2, 3, 30.0);

    // First access: cache miss
    auto wt1 = w.get_weight(0, 1);
    CHECK_EQ(wt1.has_value(), true, "weight found");
    CHECK_NEAR(wt1.value(), 10.0, 0.001, "first access weight");
    printf("    [DEBUG-CACHE] after 1st: hits=%lu misses=%lu\n",
           (unsigned long)w.get_cache_hits(), (unsigned long)w.get_cache_misses());
    CHECK_EQ((int64_t)w.get_cache_misses(), 1, "1 miss");

    // Second access: cache hit
    auto wt2 = w.get_weight(0, 1);
    CHECK_NEAR(wt2.value(), 10.0, 0.001, "cached weight");
    CHECK_EQ((int64_t)w.get_cache_hits(), 1, "1 hit");

    // has_edge via cache
    bool h = w.has_edge(1, 2);
    CHECK_EQ(h, true, "has_edge cached");
    printf("    [DEBUG-CACHE] total: hits=%lu misses=%lu\n",
           (unsigned long)w.get_cache_hits(), (unsigned long)w.get_cache_misses());
    TEST_END();
}

void test_cached_2_eviction() {
    TEST_BEGIN("Cached_Eviction");
    CachedWrapper w(2); // tiny cache: capacity=2

    w.add_edge(0, 1, 1.0);
    w.add_edge(1, 2, 2.0);
    w.add_edge(2, 3, 3.0);

    // Fill cache: (0,1) and (1,2)
    w.get_weight(0, 1);
    w.get_weight(1, 2);
    printf("    [DEBUG-CACHE] cache size=%zu\n", w.cache_size());
    CHECK_EQ((int64_t)w.cache_size(), 2, "cache full");

    // Access (2,3) -> evicts LRU
    w.get_weight(2, 3);

    // (0,1) may have been evicted -> verify cache works regardless
    auto wt = w.get_weight(0, 1);
    CHECK_EQ(wt.has_value(), true, "re-access still works");
    CHECK_NEAR(wt.value(), 1.0, 0.001, "correct weight after eviction cycle");

    printf("    [DEBUG-CACHE] final: hits=%lu misses=%lu\n",
           (unsigned long)w.get_cache_hits(), (unsigned long)w.get_cache_misses());
    TEST_END();
}

void test_cached_3_algorithms() {
    TEST_BEGIN("Cached_Algorithms");
    CachedWrapper w(16);
    w.add_edge(0, 1, 1.0); w.add_edge(1, 0, 1.0);
    w.add_edge(1, 2, 2.0); w.add_edge(2, 1, 2.0);
    w.add_edge(2, 3, 3.0); w.add_edge(3, 2, 3.0);
    w.add_edge(0, 3, 10.0); w.add_edge(3, 0, 10.0);

    auto bfs = w.run_bfs(0);
    CHECK_EQ((int64_t)bfs.vertices_visited, 4, "BFS visits 4");

    auto sssp = w.run_sssp(0);
    CHECK_NEAR(sssp.dist[3], 6.0, 0.001, "SSSP 0->3 via 0->1->2->3 = 6.0");

    auto wcc = w.run_wcc();
    CHECK_EQ((int64_t)wcc.num_components, 1, "WCC: 1 component");

    auto pr = w.run_pagerank();
    double total = 0;
    for (auto& p : pr.rank) total += p.second;
    CHECK_NEAR(total, 1.0, 0.05, "PageRank sums to ~1.0");

    printf("    [DEBUG-CACHE] after algos: hits=%lu misses=%lu\n",
           (unsigned long)w.get_cache_hits(), (unsigned long)w.get_cache_misses());
    TEST_END();
}


///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main() {
    printf("===============================================================\n");
    printf("  M106-M107: Wrapper Apps 6-System Experiment\n");
    printf("  upstream: 3808 lines across 6 wrappers\n");
    printf("===============================================================\n\n");

    auto t0 = std::chrono::steady_clock::now();

    // M106: Sorted + Undirected + CSR/Transpose
    printf("────── M106: SortedWeight + Undirected + Transpose ──────\n\n");
    test_sorted_1_basic_ops();
    test_sorted_2_snapshot_and_degree();
    test_sorted_3_algorithms();
    test_undirected_1_mirror();
    test_undirected_2_self_loop();
    test_undirected_3_algorithms();
    test_transpose_1_reversal();
    test_transpose_2_snapshot();
    test_transpose_3_algorithms();

    // M107: Filtered + ReadOnly + Cached
    printf("\n────── M107: Filtered + ReadOnly + Cached ──────\n\n");
    test_filtered_1_basic_filter();
    test_filtered_2_chained_filters();
    test_filtered_3_algorithms();
    test_readonly_1_block_mutations();
    test_readonly_2_snapshot_traversal();
    test_readonly_3_algorithms();
    test_cached_1_basic_cache();
    test_cached_2_eviction();
    test_cached_3_algorithms();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    printf("\n===============================================================\n");
    printf("  M106-M107 RESULTS: %d/%d PASS, %d FAIL (elapsed=%ldms)\n",
           g_passed_tests, g_passed_tests + g_failed_tests, g_failed_tests, (long)elapsed);
    printf("===============================================================\n");

    return g_failed_tests > 0 ? 1 : 0;
}
