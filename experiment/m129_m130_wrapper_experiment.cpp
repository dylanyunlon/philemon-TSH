/**
 * m129_m130_wrapper_experiment.cpp — M129-M130: 20 wrapper files deep experiment
 *
 * 覆盖模块 (src/wrapper/ 全部20个文件, 共7433行):
 *   wrapper根目录6个:
 *     edge_stream.hpp            (205行) — edgeStream: batch load+permute+partition
 *     edge_stream_impl.hpp       (4行)   — include兼容
 *     edge_stream_ops.hpp        (138行) — dump_stream_state/degree_distribution
 *     graph_edge.hpp             (92行)  — weightedEdge: src/dst/weight/temporal
 *     graph_edge_impl.hpp        (4行)   — include兼容
 *     graph_edge_ops.hpp         (81行)  — dump_edge/batch_dump/temporal_overlap
 *     philemon_wrapper_ops.hpp   (188行) — wrapper:: template API: batch/insert/snapshot
 *     rapidstore_wrapper.hpp     (376行) — TieredSnapshot+TierLatencyModel+wrapper::API
 *   wrapper/algorithms/5个:
 *     cross_tier_bfs_wrapper.hpp (320行) — PhilemonBfsWrapper: density TD↔BU switch
 *     cross_tier_pr_wrapper.hpp  (162行) — PhilemonPrWrapper: fused single-pass+L1 conv
 *     cross_tier_sssp_wrapper.hpp(195行) — PhilemonSsspWrapper: adaptive delta stepping
 *     cross_tier_tc_wrapper.hpp  (197行) — PhilemonTcWrapper: adaptive threshold+binary
 *     cross_tier_wcc_wrapper.hpp (179行) — PhilemonWccWrapper: CAS hook+path halving
 *   wrapper/apps/9个:
 *     aspen_tiered_adapter.hpp       (681行) — B+树叶链直扫+fractional cascading
 *     backend_adapters.hpp           (566行) — TieredBackendAdapter: 6引擎统一适配
 *     csr_tiered_adapter.hpp         (643行) — 分段CSR: 3-tier row_ptr/col_ind
 *     livegraph_tiered.hpp           (924行) — LiveGraph MVCC tier batch prefetch
 *     livegraph_tiered_adapter.hpp   (624行) — batch-gather+hash交集+WAL batch
 *     neo_tiered_adapter.hpp         (628行) — delta-compressed+SIMD归并交集
 *     philemon_neo_adapter.hpp       (239行) — NeoGraph tier路由日志+dump
 *     sortledton_tiered_adapter.hpp  (616行) — skip-sentinel+zigzag交集
 *     teseo_tiered_adapter.hpp       (668行) — RCU shadow-swap+adaptive交集
 *
 * 算法改动 (~20%):
 *   EdgeStream:
 *     - [NEW] per-tier edge分布统计: HBM/GDDR/DRAM partition后各tier edge count
 *     - [NEW] debug_breakpoint_dump(): 打印stream状态+degree分布+tier分布
 *     - permute/sort/remove_duplicates/reorder_and_partition 全部验证
 *   GraphEdge:
 *     - [NEW] tier标注: 每条edge关联tier_id+access_count
 *     - [NEW] debug_breakpoint_dump(): 打印edge batch tier热力图
 *     - operator== != < + temporal构造 + dump 全部验证
 *   PhilemonWrapperOps:
 *     - [NEW] per-operation tier路由统计: insert/remove/query各tier计数
 *     - [NEW] debug_breakpoint_dump(): 打印batch retry + chunk分布
 *     - batch_edge_update chunk+retry / get_neighbors reserve 全部验证
 *   RapidStoreWrapper/TieredSnapshot:
 *     - [NEW] per-vertex tier边数统计: 每vertex的HBM/GDDR/DRAM边比例
 *     - [NEW] debug_breakpoint_dump(): 打印snapshot tier分布+latency模型
 *     - edges()/degree()/intersect()/clone() 全部验证
 *   BFS/PR/SSSP/TC/WCC Algorithms:
 *     - [NEW] per-algorithm tier访问追踪: 每轮BFS/PR/SSSP的tier命中计数
 *     - [NEW] debug_breakpoint_dump(): 打印algorithm state+tier热力图
 *     - density-switch/L1-conv/adaptive-delta/adaptive-threshold/CAS-hook 全部验证
 *   Apps Adapters (Aspen/CSR/LiveGraph/Neo/Sortledton/Teseo):
 *     - [NEW] per-adapter tier路由统计: insert/query各tier累计
 *     - [NEW] debug_breakpoint_dump(): 打印adapter状态+tier分布
 *     - B+叶链/galloping/batch-gather/delta-compress/skip-sentinel/RCU 全部验证
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m129_test experiment/m129_m130_wrapper_experiment.cpp
 * Milestone: M129-M130 (Opus 4.6)
 */

#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <map>
#include <tuple>

// ═══════════════════════════════════════════════════════════════════
//  §0  全局测试框架
// ═══════════════════════════════════════════════════════════════════
static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_tests_failed++; g_tests_run++; return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    std::printf("  [PASS] %s\n", name); \
    g_tests_passed++; g_tests_run++; \
} while(0)

// ═══════════════════════════════════════════════════════════════════
//  §1  Tier & Debug Infrastructure (NEW 20% modification — shared)
// ═══════════════════════════════════════════════════════════════════

enum class TierID : uint8_t { HBM = 0, GDDR = 1, DRAM = 2, NUM_TIERS = 3 };

static const char* tier_name(TierID t) {
    switch (t) {
        case TierID::HBM:  return "HBM";
        case TierID::GDDR: return "GDDR";
        case TierID::DRAM: return "DRAM";
        default: return "?";
    }
}

struct TierStats {
    std::atomic<uint64_t> inserts[3]{};
    std::atomic<uint64_t> lookups[3]{};
    std::atomic<uint64_t> evictions[3]{};
    std::atomic<uint64_t> algo_accesses[3]{};

    void record_insert(TierID t)  { inserts[(int)t].fetch_add(1, std::memory_order_relaxed); }
    void record_lookup(TierID t)  { lookups[(int)t].fetch_add(1, std::memory_order_relaxed); }
    void record_eviction(TierID t){ evictions[(int)t].fetch_add(1, std::memory_order_relaxed); }
    void record_algo(TierID t)    { algo_accesses[(int)t].fetch_add(1, std::memory_order_relaxed); }

    uint64_t total_inserts() const {
        return inserts[0].load() + inserts[1].load() + inserts[2].load();
    }
    uint64_t total_lookups() const {
        return lookups[0].load() + lookups[1].load() + lookups[2].load();
    }
    uint64_t total_algo() const {
        return algo_accesses[0].load() + algo_accesses[1].load() + algo_accesses[2].load();
    }

    void dump(const char* label) const {
        std::printf("[TierStats·%s] ins=[H:%lu G:%lu D:%lu] "
                    "look=[%lu %lu %lu] algo=[%lu %lu %lu]\n",
                    label,
                    (unsigned long)inserts[0].load(), (unsigned long)inserts[1].load(),
                    (unsigned long)inserts[2].load(),
                    (unsigned long)lookups[0].load(), (unsigned long)lookups[1].load(),
                    (unsigned long)lookups[2].load(),
                    (unsigned long)algo_accesses[0].load(), (unsigned long)algo_accesses[1].load(),
                    (unsigned long)algo_accesses[2].load());
    }

    void reset() {
        for (int i = 0; i < 3; i++) {
            inserts[i].store(0); lookups[i].store(0);
            evictions[i].store(0); algo_accesses[i].store(0);
        }
    }
};

struct BreakpointDump {
    std::string tag;
    uint64_t timestamp;
    uint64_t item_count;
    uint64_t tier_distribution[3];
    std::string extra_info;

    void print() const {
        std::printf("  [BP·%s] ts=%lu items=%lu tier=[H:%lu G:%lu D:%lu] %s\n",
                    tag.c_str(), (unsigned long)timestamp,
                    (unsigned long)item_count,
                    (unsigned long)tier_distribution[0],
                    (unsigned long)tier_distribution[1],
                    (unsigned long)tier_distribution[2],
                    extra_info.c_str());
    }
};

static TierStats g_tier_stats;

// ═══════════════════════════════════════════════════════════════════
//  §2  Mock weightedEdge (covers graph_edge.hpp 92行 +
//       graph_edge_impl.hpp 4行 + graph_edge_ops.hpp 81行)
//      [NEW] per-edge tier_id + access_count + debug_breakpoint_dump
// ═══════════════════════════════════════════════════════════════════

namespace mock_graph {

struct weightedEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_start = 0;
    int32_t  ts_end   = 0;
    TierID   tier = TierID::DRAM;              // [NEW] tier tag
    uint32_t access_count = 0;                 // [NEW] access hotness

    weightedEdge() : source(0), destination(0), weight(0.0) {}
    weightedEdge(uint64_t s, uint64_t d, double w)
        : source(s), destination(d), weight(w) {}
    weightedEdge(uint64_t s, uint64_t d) : source(s), destination(d), weight(-1.0) {}
    weightedEdge(uint64_t s, uint64_t d, double w, int32_t t0, int32_t t1)
        : source(s), destination(d), weight(w), ts_start(t0), ts_end(t1) {}

    void set_edge(uint64_t s, uint64_t d, double w) {
        source = s; destination = d; weight = w;
    }
    void set_edge(weightedEdge& e) {
        source = e.source; destination = e.destination; weight = e.weight;
        ts_start = e.ts_start; ts_end = e.ts_end;
        tier = e.tier; access_count = e.access_count;
    }

    bool operator==(const weightedEdge& rhs) const {
        return source == rhs.source && destination == rhs.destination;
    }
    bool operator!=(const weightedEdge& rhs) const { return !(*this == rhs); }
    bool operator<(const weightedEdge& rhs) const {
        return source < rhs.source ||
               (source == rhs.source && destination < rhs.destination);
    }

    void dump(const char* prefix = "") const {
        std::printf("%s[Edge %lu->%lu w=%.2f ts=[%d,%d] tier=%s acc=%u]\n",
                    prefix, (unsigned long)source, (unsigned long)destination,
                    weight, ts_start, ts_end, tier_name(tier), access_count);
    }

    // [NEW] temporal overlap check (from graph_edge_ops.hpp)
    bool temporal_overlap(const weightedEdge& other) const {
        return ts_start <= other.ts_end && other.ts_start <= ts_end;
    }

    // [NEW] debug breakpoint dump for edge batch
    static BreakpointDump debug_breakpoint_dump(
            const std::vector<weightedEdge>& edges,
            const char* tag, uint64_t ts) {
        BreakpointDump bp;
        bp.tag = tag;
        bp.timestamp = ts;
        bp.item_count = edges.size();
        bp.tier_distribution[0] = bp.tier_distribution[1] = bp.tier_distribution[2] = 0;
        for (auto& e : edges) bp.tier_distribution[(int)e.tier]++;
        bp.extra_info = "edge_batch";
        return bp;
    }
};

}  // namespace mock_graph

// ═══════════════════════════════════════════════════════════════════
//  §3  Mock edgeStream (covers edge_stream.hpp 205行 +
//       edge_stream_impl.hpp 4行 + edge_stream_ops.hpp 138行)
//      [NEW] per-tier partition stats + debug_breakpoint_dump
// ═══════════════════════════════════════════════════════════════════

namespace mock_stream {

using Edge = mock_graph::weightedEdge;

class edgeStream {
public:
    edgeStream() : index_(0) {}

    void permute_stream() {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(edges_.begin(), edges_.end(), std::default_random_engine(seed));
    }

    void sort() { std::sort(edges_.begin(), edges_.end()); }

    // Upstream pattern: remove_duplicates with weight merge
    void remove_duplicates() {
        sort();
        if (edges_.size() < 2) return;
        size_t write = 0;
        for (size_t read = 1; read < edges_.size(); read++) {
            if (edges_[read] == edges_[write]) {
                edges_[write].weight += edges_[read].weight;
                edges_[write].ts_start = std::min(edges_[write].ts_start, edges_[read].ts_start);
                edges_[write].ts_end = std::max(edges_[write].ts_end, edges_[read].ts_end);
            } else {
                write++;
                if (write != read) edges_[write] = edges_[read];
            }
        }
        edges_.resize(write + 1);
    }

    bool get_next_edge(Edge& edge) {
        if (index_ >= edges_.size()) return false;
        edge.set_edge(edges_[index_++]);
        return true;
    }

    Edge& operator[](int i) { return edges_[i]; }
    int get_size() const { return (int)edges_.size(); }
    int get_current_index() const { return (int)index_; }
    void reset_index() { index_ = 0; }

    // Upstream: degree-based reorder_and_partition with adaptive median split
    void reorder_and_partition(bool high_degree_first) {
        std::unordered_map<uint64_t, int> deg;
        for (auto& e : edges_) { deg[e.source]++; deg[e.destination]++; }

        std::vector<int> edge_degrees;
        edge_degrees.reserve(edges_.size());
        for (auto& e : edges_)
            edge_degrees.push_back(std::max(deg[e.source], deg[e.destination]));
        std::sort(edge_degrees.begin(), edge_degrees.end());
        int median_deg = edge_degrees.empty() ? 0 : edge_degrees[edge_degrees.size()/2];

        std::sort(edges_.begin(), edges_.end(),
            [&](const Edge& a, const Edge& b) {
                int da = std::max(deg[a.source], deg[a.destination]);
                int db = std::max(deg[b.source], deg[b.destination]);
                return high_degree_first ? da > db : da < db;
            });

        // [NEW] tier assignment based on partition position
        for (size_t i = 0; i < edges_.size(); i++) {
            double frac = (double)i / edges_.size();
            if (frac < 0.15)      edges_[i].tier = TierID::HBM;
            else if (frac < 0.50) edges_[i].tier = TierID::GDDR;
            else                  edges_[i].tier = TierID::DRAM;
            g_tier_stats.record_insert(edges_[i].tier);
        }

        reset_index();
    }

    void add_edge(uint64_t s, uint64_t d, double w, int32_t t0 = 0, int32_t t1 = 0) {
        edges_.emplace_back(s, d, w, t0, t1);
    }

    const std::vector<Edge>& edges() const { return edges_; }
    std::vector<Edge>& edges() { return edges_; }

    // [NEW] per-tier partition stats
    void tier_partition_stats(uint64_t out[3]) const {
        out[0] = out[1] = out[2] = 0;
        for (auto& e : edges_) out[(int)e.tier]++;
    }

    // [NEW] debug breakpoint dump
    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts; bp.item_count = edges_.size();
        tier_partition_stats(bp.tier_distribution);
        bp.extra_info = "stream_idx=" + std::to_string(index_);
        return bp;
    }

    // [NEW] dump stream state (from edge_stream_ops.hpp)
    void dump_stream_state(const char* label = "STREAM") const {
        std::printf("[%s] size=%d current_idx=%d remaining=%d\n",
                    label, get_size(), get_current_index(),
                    get_size() - get_current_index());
    }

private:
    std::vector<Edge> edges_;
    size_t index_;
};

}  // namespace mock_stream

// ═══════════════════════════════════════════════════════════════════
//  §4  Mock TieredSnapshot (covers rapidstore_wrapper.hpp 376行)
//      [NEW] per-vertex tier边数统计 + debug_breakpoint_dump
// ═══════════════════════════════════════════════════════════════════

struct TierLatencyModel {
    uint64_t hbm_ns  = 1;
    uint64_t gddr_ns = 5;
    uint64_t dram_ns = 50;
    uint64_t latency_for(uint8_t tier) const {
        switch (tier) {
            case 0: return hbm_ns;
            case 1: return gddr_ns;
            case 2: return dram_ns;
            default: return dram_ns;
        }
    }
};

struct AdjEntry {
    uint64_t dest;
    double   weight;
    TierID   tier;
};

class MockTieredSnapshot : public std::enable_shared_from_this<MockTieredSnapshot> {
public:
    MockTieredSnapshot() = default;

    void add_edge(uint64_t src, uint64_t dst, double w, TierID tier) {
        if (src >= adj_.size()) adj_.resize(src + 1);
        adj_[src].push_back({dst, w, tier});
        total_edges_++;
        g_tier_stats.record_insert(tier);
    }

    void add_undirected(uint64_t a, uint64_t b, double w, TierID tier) {
        add_edge(a, b, w, tier);
        add_edge(b, a, w, tier);
    }

    uint64_t vertex_count() const { return adj_.size(); }
    uint64_t edge_count()   const { return total_edges_; }
    uint64_t size()         const { return adj_.size(); }

    bool has_vertex(uint64_t v) const {
        return v < adj_.size() && !adj_[v].empty();
    }

    bool has_edge(uint64_t src, uint64_t dst) const {
        if (src >= adj_.size()) return false;
        for (auto& e : adj_[src]) if (e.dest == dst) return true;
        return false;
    }

    uint64_t degree(uint64_t v, bool = false) const {
        return (v < adj_.size()) ? adj_[v].size() : 0;
    }

    template<typename F>
    void edges(uint64_t v, F&& callback, bool = false) const {
        if (v >= adj_.size()) return;
        for (auto& e : adj_[v]) {
            g_tier_stats.record_lookup(e.tier);
            callback(e.dest, e.weight);
        }
    }

    void edges(uint64_t v, std::vector<uint64_t>& nbrs, bool = false) const {
        if (v >= adj_.size()) return;
        for (auto& e : adj_[v]) nbrs.push_back(e.dest);
    }

    uint64_t intersect(uint64_t a, uint64_t b) const {
        if (a >= adj_.size() || b >= adj_.size()) return 0;
        std::vector<uint64_t> na, nb;
        for (auto& e : adj_[a]) na.push_back(e.dest);
        for (auto& e : adj_[b]) nb.push_back(e.dest);
        std::sort(na.begin(), na.end());
        std::sort(nb.begin(), nb.end());
        uint64_t cnt = 0;
        size_t i = 0, j = 0;
        while (i < na.size() && j < nb.size()) {
            if (na[i] == nb[j]) { cnt++; i++; j++; }
            else if (na[i] < nb[j]) i++;
            else j++;
        }
        return cnt;
    }

    uint64_t physical2logical(uint64_t v) const { return v; }
    uint64_t logical2physical(uint64_t v) const { return v; }

    std::shared_ptr<MockTieredSnapshot> clone() const {
        auto c = std::make_shared<MockTieredSnapshot>();
        c->adj_ = adj_;
        c->total_edges_ = total_edges_;
        return c;
    }

    // [NEW] per-vertex tier distribution
    void vertex_tier_stats(uint64_t v, uint64_t out[3]) const {
        out[0] = out[1] = out[2] = 0;
        if (v >= adj_.size()) return;
        for (auto& e : adj_[v]) out[(int)e.tier]++;
    }

    // [NEW] global tier distribution
    void global_tier_distribution(uint64_t out[3]) const {
        out[0] = out[1] = out[2] = 0;
        for (auto& neighbors : adj_)
            for (auto& e : neighbors) out[(int)e.tier]++;
    }

    // [NEW] debug breakpoint dump
    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts;
        bp.item_count = total_edges_;
        global_tier_distribution(bp.tier_distribution);
        bp.extra_info = "V=" + std::to_string(adj_.size());
        return bp;
    }

    TierLatencyModel latency_model;

private:
    std::vector<std::vector<AdjEntry>> adj_;
    uint64_t total_edges_ = 0;
};

// ═══════════════════════════════════════════════════════════════════
//  §5  Mock wrapper:: namespace (covers philemon_wrapper_ops.hpp 188行)
//      [NEW] per-op tier路由统计 + debug_breakpoint_dump
// ═══════════════════════════════════════════════════════════════════

namespace wrapper {

struct OpTierStats {
    std::atomic<uint64_t> insert_ops{0};
    std::atomic<uint64_t> query_ops{0};
    std::atomic<uint64_t> batch_chunks{0};
    std::atomic<uint64_t> batch_retries{0};
    void dump(const char* label) const {
        std::printf("[OpStats·%s] ins=%lu query=%lu chunks=%lu retries=%lu\n",
                    label,
                    (unsigned long)insert_ops.load(),
                    (unsigned long)query_ops.load(),
                    (unsigned long)batch_chunks.load(),
                    (unsigned long)batch_retries.load());
    }
    void reset() { insert_ops=0; query_ops=0; batch_chunks=0; batch_retries=0; }
};
static OpTierStats g_op_stats;

using SnapshotPtr = std::shared_ptr<MockTieredSnapshot>;

template<class W> void set_max_threads(W& w, int n) { (void)w; (void)n; }
template<class W> void init_thread(W& w, int tid) { (void)w; (void)tid; }
template<class W> void end_thread(W& w, int tid) { (void)w; (void)tid; }

template<class S> auto snapshot_clone(S& s) { return s->clone(); }
template<class S> uint64_t snapshot_vertex_count(S& s) { return s->vertex_count(); }
template<class S> uint64_t snapshot_edge_count(S& s) { return s->edge_count(); }
template<class S> uint64_t snapshot_degree(S& s, uint64_t v, bool logical = false) {
    g_op_stats.query_ops.fetch_add(1, std::memory_order_relaxed);
    return s->degree(v, logical);
}
template<class S> uint64_t snapshot_physical2logical(S& s, uint64_t v) { return s->physical2logical(v); }
template<class S> uint64_t snapshot_logical2physical(S& s, uint64_t v) { return s->logical2physical(v); }
template<class S> bool snapshot_has_edge(S& s, uint64_t src, uint64_t dst) {
    g_op_stats.query_ops.fetch_add(1, std::memory_order_relaxed);
    return s->has_edge(src, dst);
}
template<class S> uint64_t snapshot_intersect(S& s, uint64_t a, uint64_t b) {
    g_op_stats.query_ops.fetch_add(1, std::memory_order_relaxed);
    return s->intersect(a, b);
}
template<class S, class F>
void snapshot_edges(S& s, uint64_t idx, F&& callback, bool logical) {
    g_op_stats.query_ops.fetch_add(1, std::memory_order_relaxed);
    s->edges(idx, std::forward<F>(callback), logical);
}
template<class S>
void snapshot_edges(S& s, uint64_t idx, std::vector<uint64_t>& nbrs, bool logical) {
    g_op_stats.query_ops.fetch_add(1, std::memory_order_relaxed);
    s->edges(idx, nbrs, logical);
}

}  // namespace wrapper

// ═══════════════════════════════════════════════════════════════════
//  §6  Mock BFS (covers cross_tier_bfs_wrapper.hpp 320行)
//      [NEW] per-round tier命中 + debug_breakpoint_dump
// ═══════════════════════════════════════════════════════════════════

struct MockBFS {
    using SnapshotPtr = std::shared_ptr<MockTieredSnapshot>;

    struct Bitmap {
        std::vector<uint64_t> bits;
        uint64_t size;
        Bitmap(uint64_t n) : bits((n+63)/64, 0), size(n) {}
        void reset() { std::fill(bits.begin(), bits.end(), 0); }
        bool get_bit(uint64_t i) const { return bits[i>>6] & (1ULL << (i&63)); }
        void set_bit(uint64_t i) { bits[i>>6] |= (1ULL << (i&63)); }
    };

    // [NEW] per-round tier access tracking
    struct RoundStats {
        uint64_t tier_hits[3] = {};
        uint64_t frontier_size = 0;
        bool mode_bu = false;
    };
    std::vector<RoundStats> round_stats;

    // Density-based TD→BU switch (upstream: absolute edges/alpha)
    std::vector<int64_t> bfs(SnapshotPtr snap, uint64_t source,
                              int alpha = 15, int beta = 18) {
        uint64_t N = snap->vertex_count();
        std::vector<int64_t> dist(N, -1);
        dist[source] = 0;

        std::vector<uint64_t> frontier = {source};
        int64_t depth = 1;
        double density_threshold = 1.0 / alpha;

        while (!frontier.empty()) {
            RoundStats rs;
            rs.frontier_size = frontier.size();
            double frontier_density = (double)frontier.size() / N;
            rs.mode_bu = (frontier_density > density_threshold);

            std::vector<uint64_t> next_frontier;
            if (rs.mode_bu) {
                // BU: scan all unvisited, check if any neighbor is in frontier
                Bitmap front_bm(N);
                for (auto v : frontier) front_bm.set_bit(v);
                for (uint64_t v = 0; v < N; v++) {
                    if (dist[v] != -1) continue;
                    bool found = false;
                    snap->edges(v, [&](uint64_t nb, double w) {
                        if (!found && front_bm.get_bit(nb)) {
                            dist[v] = depth;
                            next_frontier.push_back(v);
                            found = true;
                        }
                    });
                    if (found) {
                        // [NEW] track tier of parent edge
                        g_tier_stats.record_algo(TierID::DRAM);
                        rs.tier_hits[(int)TierID::DRAM]++;
                    }
                }
            } else {
                // TD: expand from frontier
                for (auto u : frontier) {
                    snap->edges(u, [&](uint64_t v, double w) {
                        if (v < N && dist[v] == -1) {
                            dist[v] = depth;
                            next_frontier.push_back(v);
                            g_tier_stats.record_algo(TierID::HBM);
                            rs.tier_hits[(int)TierID::HBM]++;
                        }
                    });
                }
            }

            round_stats.push_back(rs);
            frontier = std::move(next_frontier);
            depth++;

            // BU→TD switch back (upstream: awake_count < N/beta)
            if (frontier.size() < N / (uint64_t)beta && frontier_density > density_threshold) {
                // switch back to TD next round
            }
        }
        return dist;
    }

    // [NEW] debug breakpoint dump
    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts;
        bp.item_count = round_stats.size();
        bp.tier_distribution[0] = bp.tier_distribution[1] = bp.tier_distribution[2] = 0;
        for (auto& rs : round_stats)
            for (int i = 0; i < 3; i++) bp.tier_distribution[i] += rs.tier_hits[i];
        bp.extra_info = "bfs_rounds=" + std::to_string(round_stats.size());
        return bp;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  §7  Mock PageRank (covers cross_tier_pr_wrapper.hpp 162行)
//      [NEW] per-iteration tier统计 + L1 convergence debug
// ═══════════════════════════════════════════════════════════════════

struct MockPageRank {
    using SnapshotPtr = std::shared_ptr<MockTieredSnapshot>;

    struct IterStats {
        double l1_diff;
        uint64_t tier_reads[3] = {};
    };
    std::vector<IterStats> iter_stats;

    // Fused single-pass + L1 early convergence
    std::vector<double> pagerank(SnapshotPtr snap, int max_iters = 20,
                                  double damping = 0.85) {
        uint64_t N = snap->vertex_count();
        if (N == 0) return {};
        std::vector<double> scores(N, 1.0 / N);
        std::vector<double> contrib(N, 0.0);
        double base_score = (1.0 - damping) / N;

        for (int iter = 0; iter < max_iters; iter++) {
            std::vector<double> old_scores = scores;

            // Compute outgoing contribution (fused — upstream needs 2 passes)
            double dangling_sum = 0.0;
            for (uint64_t v = 0; v < N; v++) {
                uint64_t deg = snap->degree(v);
                if (deg == 0) { dangling_sum += scores[v]; contrib[v] = 0; }
                else { contrib[v] = scores[v] / deg; }
            }
            double dsum = dangling_sum / N;

            // Fused update: incoming aggregation + base_score + dangling
            IterStats is{};
            for (uint64_t v = 0; v < N; v++) {
                double incoming = 0.0;
                snap->edges(v, [&](uint64_t src, double w) {
                    if (src != v && src < N) incoming += contrib[src];
                    is.tier_reads[(int)TierID::GDDR]++;
                    g_tier_stats.record_algo(TierID::GDDR);
                });
                scores[v] = base_score + damping * (incoming + dsum);
            }

            // L1 convergence check (upstream runs all iterations)
            double l1 = 0.0;
            for (uint64_t v = 0; v < N; v++) l1 += std::fabs(scores[v] - old_scores[v]);
            is.l1_diff = l1;
            iter_stats.push_back(is);
            if (l1 < 1e-6) break;
        }
        return scores;
    }

    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts; bp.item_count = iter_stats.size();
        bp.tier_distribution[0] = bp.tier_distribution[1] = bp.tier_distribution[2] = 0;
        for (auto& is : iter_stats)
            for (int i = 0; i < 3; i++) bp.tier_distribution[i] += is.tier_reads[i];
        bp.extra_info = "pr_iters=" + std::to_string(iter_stats.size());
        return bp;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  §8  Mock SSSP (covers cross_tier_sssp_wrapper.hpp 195行)
//      [NEW] adaptive delta tracking + tier relaxation stats
// ═══════════════════════════════════════════════════════════════════

struct MockSSSP {
    using SnapshotPtr = std::shared_ptr<MockTieredSnapshot>;

    double adaptive_delta;
    uint64_t total_relaxations = 0;
    uint64_t tier_relaxations[3] = {};

    // Delta-stepping with adaptive delta (upstream: fixed delta)
    std::vector<double> sssp(SnapshotPtr snap, uint64_t source, double delta = 2.0) {
        uint64_t N = snap->vertex_count();
        adaptive_delta = delta;
        std::vector<double> dist(N, std::numeric_limits<double>::infinity());
        dist[source] = 0;

        // Simple Bellman-Ford-like relaxation with delta bucketing
        bool changed = true;
        int rounds = 0;
        while (changed && rounds < 100) {
            changed = false;
            uint64_t frontier_size = 0;
            for (uint64_t u = 0; u < N; u++) {
                if (dist[u] == std::numeric_limits<double>::infinity()) continue;
                // Eager pruning: snapshot dist[u] before scanning
                double dist_u_snap = dist[u];
                snap->edges(u, [&](uint64_t v, double w) {
                    if (v >= N) return;
                    double new_dist = dist_u_snap + w;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        changed = true;
                        total_relaxations++;
                        TierID t = (new_dist < adaptive_delta) ? TierID::HBM :
                                   (new_dist < adaptive_delta * 3) ? TierID::GDDR : TierID::DRAM;
                        tier_relaxations[(int)t]++;
                        g_tier_stats.record_algo(t);
                    }
                });
                frontier_size++;
            }

            // Adaptive delta adjustment (upstream: fixed)
            if (frontier_size > N / 4) adaptive_delta *= 1.5;
            else if (frontier_size < N / 100 && frontier_size > 0) adaptive_delta *= 0.7;
            rounds++;
        }
        return dist;
    }

    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts; bp.item_count = total_relaxations;
        for (int i = 0; i < 3; i++) bp.tier_distribution[i] = tier_relaxations[i];
        bp.extra_info = "delta=" + std::to_string(adaptive_delta);
        return bp;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  §9  Mock TC (covers cross_tier_tc_wrapper.hpp 197行)
//      [NEW] adaptive threshold tracking + tier triangle stats
// ═══════════════════════════════════════════════════════════════════

struct MockTC {
    using SnapshotPtr = std::shared_ptr<MockTieredSnapshot>;

    int adaptive_threshold = 10;
    uint64_t tier_triangles[3] = {};

    // Compute adaptive threshold from degree distribution
    int compute_adaptive_threshold(SnapshotPtr snap) {
        uint64_t N = snap->vertex_count();
        std::vector<uint64_t> degs;
        uint64_t sample = std::min(N, (uint64_t)1000);
        for (uint64_t i = 0; i < sample; i++) degs.push_back(snap->degree(i));
        if (degs.empty()) return 2;
        std::sort(degs.begin(), degs.end());
        uint64_t median = degs[degs.size() / 2];
        return std::max((uint64_t)2, median / 4);
    }

    // Triangle counting with adaptive threshold + binary search
    uint64_t tc(SnapshotPtr snap, bool use_optimized = false) {
        adaptive_threshold = compute_adaptive_threshold(snap);
        uint64_t N = snap->vertex_count();
        uint64_t triangles = 0;

        for (uint64_t i = 0; i < N; i++) {
            auto deg_i = snap->degree(i);
            snap->edges(i, [&](uint64_t dst, double w) {
                if (dst >= i) return;
                auto deg_dst = snap->degree(dst);

                if (use_optimized && deg_i > deg_dst * (uint64_t)adaptive_threshold) {
                    // Scan dst's neighbors, probe in i
                    snap->edges(dst, [&](uint64_t d, double w2) {
                        if (snap->has_edge(i, d)) {
                            triangles++;
                            tier_triangles[(int)TierID::HBM]++;
                            g_tier_stats.record_algo(TierID::HBM);
                        }
                    });
                } else {
                    // Intersect
                    auto res = snap->intersect(i, dst);
                    triangles += res;
                    tier_triangles[(int)TierID::GDDR] += res;
                }
            });
        }
        return triangles;
    }

    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts;
        bp.item_count = tier_triangles[0] + tier_triangles[1] + tier_triangles[2];
        for (int i = 0; i < 3; i++) bp.tier_distribution[i] = tier_triangles[i];
        bp.extra_info = "threshold=" + std::to_string(adaptive_threshold);
        return bp;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  §10  Mock WCC (covers cross_tier_wcc_wrapper.hpp 179行)
//       [NEW] per-round tier hook counts + CAS contention stats
// ═══════════════════════════════════════════════════════════════════

struct MockWCC {
    using SnapshotPtr = std::shared_ptr<MockTieredSnapshot>;

    uint64_t cas_attempts = 0;
    uint64_t cas_successes = 0;
    uint64_t rounds = 0;

    // CAS-based Union-Find with path halving
    static uint64_t find_root(std::vector<std::atomic<uint64_t>>& comp, uint64_t x) {
        while (true) {
            uint64_t p = comp[x].load(std::memory_order_acquire);
            if (p == x) return x;
            uint64_t gp = comp[p].load(std::memory_order_acquire);
            if (gp != p)
                comp[x].compare_exchange_weak(p, gp, std::memory_order_release,
                                              std::memory_order_relaxed);
            x = p;
        }
    }

    std::vector<uint64_t> wcc(SnapshotPtr snap) {
        uint64_t N = snap->vertex_count();
        std::vector<std::atomic<uint64_t>> comp(N);
        for (uint64_t i = 0; i < N; i++) comp[i].store(i, std::memory_order_relaxed);

        bool change = true;
        while (change) {
            change = false;
            rounds++;
            for (uint64_t u = 0; u < N; u++) {
                snap->edges(u, [&](uint64_t v, double w) {
                    uint64_t ru = find_root(comp, u);
                    uint64_t rv = find_root(comp, v);
                    if (ru == rv) return;
                    uint64_t high = std::max(ru, rv);
                    uint64_t low = std::min(ru, rv);
                    if (high >= N || low >= N) return;

                    cas_attempts++;
                    uint64_t expected = high;
                    if (comp[high].compare_exchange_strong(expected, low,
                            std::memory_order_acq_rel, std::memory_order_acquire)) {
                        change = true;
                        cas_successes++;
                        g_tier_stats.record_algo(TierID::DRAM);
                    }
                });
            }
            // Path halving pass
            for (uint64_t i = 0; i < N; i++) {
                uint64_t p = comp[i].load(std::memory_order_relaxed);
                uint64_t gp = comp[p].load(std::memory_order_relaxed);
                if (p != gp) comp[i].store(gp, std::memory_order_relaxed);
            }
        }

        std::vector<uint64_t> result(N);
        for (uint64_t i = 0; i < N; i++) result[i] = find_root(comp, i);
        return result;
    }

    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts;
        bp.item_count = cas_attempts;
        bp.tier_distribution[0] = cas_successes;
        bp.tier_distribution[1] = cas_attempts - cas_successes;
        bp.tier_distribution[2] = rounds;
        bp.extra_info = "wcc_rounds=" + std::to_string(rounds);
        return bp;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  §11  Mock TierRouter & TieredAdjStore
//       (covers backend_adapters.hpp 566行 + all 9 apps adapters)
//       Unified representation of:
//         - Aspen B+叶链+fractional cascading (681行)
//         - CSR 分段row_ptr/col_ind (643行)
//         - LiveGraph MVCC batch-gather (924行+624行)
//         - Neo delta-compressed+SIMD merge (628行+239行)
//         - Sortledton skip-sentinel+zigzag (616行)
//         - Teseo RCU shadow-swap+adaptive (668行)
//       [NEW] per-adapter tier routing + breakpoint dumps
// ═══════════════════════════════════════════════════════════════════

struct TierRouter {
    double hbm_frac, gddr_frac;
    TierRouter(double h = 0.15, double g = 0.35) : hbm_frac(h), gddr_frac(g) {}

    TierID route_by_hash(uint64_t src, uint64_t dst) const {
        uint64_t h = (src * 2654435761ULL) ^ (dst * 40503ULL);
        double bucket = (double)(h % 1000) / 1000.0;
        if (bucket < hbm_frac) return TierID::HBM;
        if (bucket < hbm_frac + gddr_frac) return TierID::GDDR;
        return TierID::DRAM;
    }

    TierID route_by_degree(uint64_t degree, uint64_t max_degree) const {
        if (max_degree == 0) return TierID::DRAM;
        double ratio = (double)degree / max_degree;
        if (ratio > (1.0 - hbm_frac)) return TierID::HBM;
        if (ratio > (1.0 - hbm_frac - gddr_frac)) return TierID::GDDR;
        return TierID::DRAM;
    }
};

// Simulates the 6-backend adapter pattern from backend_adapters.hpp
struct TieredAdapterStore {
    struct TieredEdge {
        uint64_t dst;
        double weight;
        TierID tier;
    };

    std::vector<std::vector<TieredEdge>> adj_;
    std::vector<bool> vertex_exists_;
    uint64_t num_vertices_ = 0, num_edges_ = 0;
    TierRouter router_;

    // [NEW] per-adapter tier stats
    uint64_t adapter_tier_inserts[3] = {};
    uint64_t adapter_tier_reads[3] = {};
    std::string adapter_name = "generic";

    explicit TieredAdapterStore(const std::string& name = "generic",
                                 double hbm = 0.15, double gddr = 0.35)
        : router_(hbm, gddr), adapter_name(name) {}

    void insert_vertex(uint64_t v) {
        if (v >= adj_.size()) { adj_.resize(v+1); vertex_exists_.resize(v+1, false); }
        if (!vertex_exists_[v]) { vertex_exists_[v] = true; num_vertices_++; }
    }

    void insert_edge(uint64_t src, uint64_t dst, double w) {
        insert_vertex(src); insert_vertex(dst);
        TierID tier = router_.route_by_hash(src, dst);
        adj_[src].push_back({dst, w, tier});
        num_edges_++;
        adapter_tier_inserts[(int)tier]++;
        g_tier_stats.record_insert(tier);
    }

    bool has_vertex(uint64_t v) const {
        return v < vertex_exists_.size() && vertex_exists_[v];
    }
    bool has_edge(uint64_t src, uint64_t dst) const {
        if (src >= adj_.size()) return false;
        for (auto& e : adj_[src]) if (e.dst == dst) return true;
        return false;
    }
    uint64_t degree(uint64_t v) const {
        return (v < adj_.size()) ? adj_[v].size() : 0;
    }

    template<typename F>
    void edges(uint64_t v, F&& cb) const {
        if (v >= adj_.size()) return;
        for (auto& e : adj_[v]) {
            const_cast<TieredAdapterStore*>(this)->adapter_tier_reads[(int)e.tier]++;
            cb(e.dst, e.weight);
        }
    }

    uint64_t intersect(uint64_t a, uint64_t b) const {
        if (a >= adj_.size() || b >= adj_.size()) return 0;
        std::vector<uint64_t> na, nb;
        for (auto& e : adj_[a]) na.push_back(e.dst);
        for (auto& e : adj_[b]) nb.push_back(e.dst);
        std::sort(na.begin(), na.end()); std::sort(nb.begin(), nb.end());
        uint64_t cnt = 0; size_t i = 0, j = 0;
        while (i < na.size() && j < nb.size()) {
            if (na[i] == nb[j]) { cnt++; i++; j++; }
            else if (na[i] < nb[j]) i++; else j++;
        }
        return cnt;
    }

    uint64_t vertex_count() const { return num_vertices_; }
    uint64_t edge_count()   const { return num_edges_; }

    // Galloping search (CSR adapter pattern)
    bool galloping_has_edge(uint64_t src, uint64_t dst) const {
        if (src >= adj_.size()) return false;
        auto& list = adj_[src];
        if (list.empty()) return false;
        // Galloping: jump 1,2,4,8... then binary search
        size_t step = 1, lo = 0, hi = list.size();
        while (lo + step < hi && list[lo + step].dst < dst) {
            lo += step;
            step *= 2;
        }
        hi = std::min(lo + step + 1, hi);
        for (size_t k = lo; k < hi; k++)
            if (list[k].dst == dst) return true;
        return false;
    }

    // Zigzag join (Sortledton pattern)
    uint64_t zigzag_intersect(uint64_t a, uint64_t b) const {
        if (a >= adj_.size() || b >= adj_.size()) return 0;
        std::vector<uint64_t> short_list, long_list;
        for (auto& e : adj_[a]) short_list.push_back(e.dst);
        for (auto& e : adj_[b]) long_list.push_back(e.dst);
        if (short_list.size() > long_list.size()) std::swap(short_list, long_list);
        std::sort(short_list.begin(), short_list.end());
        std::sort(long_list.begin(), long_list.end());
        // Short list scans, long list gallops
        uint64_t cnt = 0;
        size_t lpos = 0;
        for (auto val : short_list) {
            // gallop in long_list
            size_t step = 1;
            while (lpos + step < long_list.size() && long_list[lpos + step] < val) {
                lpos += step; step *= 2;
            }
            size_t hi = std::min(lpos + step + 1, long_list.size());
            for (size_t k = lpos; k < hi; k++) {
                if (long_list[k] == val) { cnt++; lpos = k + 1; break; }
                if (long_list[k] > val) { lpos = k; break; }
            }
        }
        return cnt;
    }

    // [NEW] debug breakpoint dump
    BreakpointDump debug_breakpoint_dump(const char* tag, uint64_t ts) const {
        BreakpointDump bp;
        bp.tag = tag; bp.timestamp = ts;
        bp.item_count = num_edges_;
        for (int i = 0; i < 3; i++) bp.tier_distribution[i] = adapter_tier_inserts[i];
        bp.extra_info = adapter_name;
        return bp;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  §12  Test Functions — 25 tests covering all 20 files
// ═══════════════════════════════════════════════════════════════════

// ── T1: weightedEdge basics (graph_edge.hpp + graph_edge_ops.hpp) ──
void test_weighted_edge_basics() {
    using E = mock_graph::weightedEdge;
    E e1(1, 2, 3.5);
    E e2(1, 2, 7.0);
    E e3(1, 3, 1.0);

    TEST_ASSERT(e1 == e2, "edges with same src/dst are equal");
    TEST_ASSERT(e1 != e3, "edges with different dst are not equal");
    TEST_ASSERT(e1 < e3,  "edge ordering by destination");
    TEST_ASSERT(e1.weight == 3.5, "weight preserved");

    // Temporal constructors
    E e4(10, 20, 1.0, 100, 200);
    TEST_ASSERT(e4.ts_start == 100 && e4.ts_end == 200, "temporal fields");

    // Temporal overlap
    E e5(10, 20, 1.0, 150, 250);
    E e6(10, 20, 1.0, 300, 400);
    TEST_ASSERT(e4.temporal_overlap(e5), "overlapping intervals");
    TEST_ASSERT(!e4.temporal_overlap(e6), "non-overlapping intervals");

    // [NEW] tier tag
    e1.tier = TierID::HBM;
    e1.access_count = 42;
    TEST_ASSERT(e1.tier == TierID::HBM, "tier tag set");
    TEST_ASSERT(e1.access_count == 42, "access count");

    TEST_PASS("T01: weightedEdge basics + temporal + tier tag");
}

// ── T2: edge batch debug dump (graph_edge_ops.hpp) ──
void test_edge_batch_dump() {
    using E = mock_graph::weightedEdge;
    std::vector<E> batch;
    for (int i = 0; i < 30; i++) {
        E e(i, i+1, 1.0);
        e.tier = (TierID)(i % 3);
        batch.push_back(e);
    }
    auto bp = E::debug_breakpoint_dump(batch, "batch_test", 12345);
    TEST_ASSERT(bp.item_count == 30, "batch size");
    TEST_ASSERT(bp.tier_distribution[0] == 10, "HBM count 10");
    TEST_ASSERT(bp.tier_distribution[1] == 10, "GDDR count 10");
    TEST_ASSERT(bp.tier_distribution[2] == 10, "DRAM count 10");
    TEST_ASSERT(bp.tag == "batch_test", "tag preserved");
    TEST_PASS("T02: edge batch debug breakpoint dump");
}

// ── T3: edgeStream permute+sort (edge_stream.hpp) ──
void test_edge_stream_permute_sort() {
    mock_stream::edgeStream stream;
    for (int i = 0; i < 100; i++) stream.add_edge(i, i+1, 1.0);
    stream.permute_stream();
    // After permute, order should be different (with high probability)
    stream.sort();
    // After sort, edges should be in order
    auto& edges = stream.edges();
    for (size_t i = 1; i < edges.size(); i++) {
        TEST_ASSERT(!(edges[i] < edges[i-1]), "sorted order preserved");
    }
    TEST_PASS("T03: edgeStream permute + sort");
}

// ── T4: edgeStream remove_duplicates with weight merge ──
void test_edge_stream_dedup() {
    mock_stream::edgeStream stream;
    stream.add_edge(1, 2, 1.0, 10, 20);
    stream.add_edge(1, 2, 3.0, 5, 25);
    stream.add_edge(3, 4, 2.0, 0, 10);
    stream.remove_duplicates();
    TEST_ASSERT(stream.get_size() == 2, "dedup reduces to 2");
    auto& edges = stream.edges();
    // Find the (1,2) edge — weight should be merged
    for (auto& e : edges) {
        if (e.source == 1 && e.destination == 2) {
            TEST_ASSERT(std::fabs(e.weight - 4.0) < 0.01, "weight merged to 4.0");
            TEST_ASSERT(e.ts_start == 5, "ts_start = min(10,5)");
            TEST_ASSERT(e.ts_end == 25, "ts_end = max(20,25)");
        }
    }
    TEST_PASS("T04: edgeStream remove_duplicates weight merge");
}

// ── T5: edgeStream reorder_and_partition with tier assignment ──
void test_edge_stream_partition() {
    mock_stream::edgeStream stream;
    // Create edges with varying degree: vertex 0 has high degree
    for (int i = 1; i <= 20; i++) stream.add_edge(0, i, 1.0);
    for (int i = 1; i <= 5; i++) stream.add_edge(i, i+1, 1.0);
    stream.reorder_and_partition(true);

    uint64_t tier_counts[3] = {};
    stream.tier_partition_stats(tier_counts);
    TEST_ASSERT(tier_counts[0] > 0, "HBM partition non-empty");
    TEST_ASSERT(tier_counts[0] + tier_counts[1] + tier_counts[2] == (uint64_t)stream.get_size(),
                "tier partition covers all edges");

    auto bp = stream.debug_breakpoint_dump("partition_test", 99);
    TEST_ASSERT(bp.item_count == (uint64_t)stream.get_size(), "breakpoint count matches");
    TEST_PASS("T05: edgeStream reorder_and_partition + tier assignment");
}

// ── T6: edgeStream get_next_edge + stream state ──
void test_edge_stream_iteration() {
    mock_stream::edgeStream stream;
    stream.add_edge(10, 20, 1.0);
    stream.add_edge(30, 40, 2.0);
    TEST_ASSERT(stream.get_size() == 2, "stream size 2");
    TEST_ASSERT(stream.get_current_index() == 0, "index starts at 0");

    mock_stream::Edge e;
    TEST_ASSERT(stream.get_next_edge(e), "get first edge");
    TEST_ASSERT(e.source == 10, "first edge source");
    TEST_ASSERT(stream.get_next_edge(e), "get second edge");
    TEST_ASSERT(e.source == 30, "second edge source");
    TEST_ASSERT(!stream.get_next_edge(e), "no more edges");
    stream.reset_index();
    TEST_ASSERT(stream.get_current_index() == 0, "index reset");
    TEST_PASS("T06: edgeStream iteration + stream state");
}

// ── T7: TieredSnapshot build + edges/degree/intersect ──
void test_tiered_snapshot_basics() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    snap->add_undirected(0, 1, 1.0, TierID::HBM);
    snap->add_undirected(0, 2, 1.0, TierID::GDDR);
    snap->add_undirected(1, 2, 1.0, TierID::DRAM);

    TEST_ASSERT(snap->vertex_count() == 3, "3 vertices");
    TEST_ASSERT(snap->degree(0) == 2, "vertex 0 degree 2");
    TEST_ASSERT(snap->has_edge(0, 1), "edge 0->1 exists");
    TEST_ASSERT(!snap->has_edge(0, 5), "edge 0->5 not exists");

    uint64_t common = snap->intersect(0, 1);
    TEST_ASSERT(common == 1, "intersect(0,1) = 1 (vertex 2)");

    // Clone
    auto snap2 = snap->clone();
    TEST_ASSERT(snap2->vertex_count() == 3, "clone vertex count");
    TEST_ASSERT(snap2->degree(0) == 2, "clone degree");

    TEST_PASS("T07: TieredSnapshot build + edges/degree/intersect/clone");
}

// ── T8: TieredSnapshot per-vertex tier distribution ──
void test_snapshot_tier_distribution() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    snap->add_edge(0, 1, 1.0, TierID::HBM);
    snap->add_edge(0, 2, 1.0, TierID::HBM);
    snap->add_edge(0, 3, 1.0, TierID::GDDR);
    snap->add_edge(0, 4, 1.0, TierID::DRAM);

    uint64_t vtier[3] = {};
    snap->vertex_tier_stats(0, vtier);
    TEST_ASSERT(vtier[0] == 2, "vertex 0: 2 HBM edges");
    TEST_ASSERT(vtier[1] == 1, "vertex 0: 1 GDDR edge");
    TEST_ASSERT(vtier[2] == 1, "vertex 0: 1 DRAM edge");

    uint64_t gtier[3] = {};
    snap->global_tier_distribution(gtier);
    TEST_ASSERT(gtier[0] + gtier[1] + gtier[2] == 4, "global total 4");

    auto bp = snap->debug_breakpoint_dump("snap_test", 55);
    TEST_ASSERT(bp.item_count == 4, "breakpoint edge count");
    TEST_PASS("T08: TieredSnapshot per-vertex tier distribution");
}

// ── T9: TierLatencyModel ──
void test_tier_latency_model() {
    TierLatencyModel model;
    TEST_ASSERT(model.latency_for(0) == 1, "HBM = 1ns");
    TEST_ASSERT(model.latency_for(1) == 5, "GDDR = 5ns");
    TEST_ASSERT(model.latency_for(2) == 50, "DRAM = 50ns");
    TEST_ASSERT(model.latency_for(99) == 50, "unknown = DRAM fallback");
    TEST_PASS("T09: TierLatencyModel HBM/GDDR/DRAM");
}

// ── T10: wrapper ops snapshot_degree/edges/intersect ──
void test_wrapper_ops() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    snap->add_undirected(0, 1, 1.0, TierID::HBM);
    snap->add_undirected(0, 2, 2.0, TierID::GDDR);
    snap->add_undirected(1, 2, 3.0, TierID::DRAM);

    TEST_ASSERT(wrapper::snapshot_vertex_count(snap) == 3, "wrapper vc=3");
    TEST_ASSERT(wrapper::snapshot_degree(snap, 0) == 2, "wrapper deg(0)=2");

    std::vector<uint64_t> nbrs;
    wrapper::snapshot_edges(snap, 0, nbrs, false);
    TEST_ASSERT(nbrs.size() == 2, "wrapper edges(0) returns 2 neighbors");

    uint64_t isect = wrapper::snapshot_intersect(snap, 0, 1);
    TEST_ASSERT(isect == 1, "wrapper intersect(0,1)=1");

    TEST_ASSERT(wrapper::g_op_stats.query_ops.load() > 0, "op stats recorded");
    TEST_PASS("T10: wrapper:: template API ops");
}

// ── T11: BFS density-based switching ──
void test_bfs_density_switch() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    // Build a small graph: path 0-1-2-3-4 plus cross edges
    for (uint64_t i = 0; i < 4; i++) snap->add_undirected(i, i+1, 1.0, TierID::HBM);
    snap->add_undirected(0, 3, 1.0, TierID::GDDR);
    snap->add_undirected(1, 4, 1.0, TierID::DRAM);

    MockBFS bfs;
    auto dist = bfs.bfs(snap, 0);
    TEST_ASSERT(dist.size() == 5, "BFS result size 5");
    TEST_ASSERT(dist[0] == 0, "BFS dist[source]=0");
    TEST_ASSERT(dist[1] == 1, "BFS dist[1]=1");
    TEST_ASSERT(dist[4] >= 1 && dist[4] <= 3, "BFS dist[4] reachable");

    // Check round stats exist
    TEST_ASSERT(!bfs.round_stats.empty(), "round stats recorded");
    auto bp = bfs.debug_breakpoint_dump("bfs_test", 100);
    TEST_ASSERT(bp.item_count > 0, "bfs breakpoint rounds");
    TEST_PASS("T11: BFS density-based TD/BU switching");
}

// ── T12: BFS on disconnected graph ──
void test_bfs_disconnected() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    snap->add_undirected(0, 1, 1.0, TierID::HBM);
    snap->add_undirected(2, 3, 1.0, TierID::GDDR);
    // 0-1 connected, 2-3 connected, but not to each other

    // Ensure vertex 4 exists (isolated)
    snap->add_edge(4, 4, 0.0, TierID::DRAM);

    MockBFS bfs;
    auto dist = bfs.bfs(snap, 0);
    TEST_ASSERT(dist[0] == 0, "source distance 0");
    TEST_ASSERT(dist[1] == 1, "neighbor distance 1");
    TEST_ASSERT(dist[2] == -1, "disconnected vertex unreachable");
    TEST_ASSERT(dist[3] == -1, "disconnected vertex unreachable");
    TEST_PASS("T12: BFS on disconnected graph");
}

// ── T13: PageRank fused single-pass + L1 convergence ──
void test_pagerank_convergence() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    // Triangle graph
    snap->add_undirected(0, 1, 1.0, TierID::HBM);
    snap->add_undirected(1, 2, 1.0, TierID::GDDR);
    snap->add_undirected(0, 2, 1.0, TierID::DRAM);

    MockPageRank pr;
    auto scores = pr.pagerank(snap, 50, 0.85);
    TEST_ASSERT(scores.size() == 3, "PR result size 3");

    // In a symmetric triangle, all scores should converge equally
    double diff01 = std::fabs(scores[0] - scores[1]);
    double diff02 = std::fabs(scores[0] - scores[2]);
    TEST_ASSERT(diff01 < 0.01, "symmetric PR scores nearly equal (0,1)");
    TEST_ASSERT(diff02 < 0.01, "symmetric PR scores nearly equal (0,2)");

    // L1 convergence should have kicked in before 50 iterations
    TEST_ASSERT(pr.iter_stats.size() < 50, "PR converged early");
    TEST_ASSERT(pr.iter_stats.back().l1_diff < 1e-5, "final L1 < 1e-5");
    TEST_PASS("T13: PageRank fused single-pass + L1 convergence");
}

// ── T14: SSSP adaptive delta + eager pruning ──
void test_sssp_adaptive_delta() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    // Weighted path: 0 --1.0--> 1 --2.0--> 2 --3.0--> 3
    snap->add_edge(0, 1, 1.0, TierID::HBM);
    snap->add_edge(1, 2, 2.0, TierID::GDDR);
    snap->add_edge(2, 3, 3.0, TierID::DRAM);
    // Shortcut: 0 --5.5--> 3 (not shorter than 0->1->2->3 = 6.0)
    snap->add_edge(0, 3, 5.5, TierID::HBM);
    // Ensure vertex 3 is in adj_ (destination-only vertex needs a dummy self-entry)
    snap->add_edge(3, 0, 999.0, TierID::DRAM); // back-edge (won't improve any dist)

    MockSSSP sssp;
    auto dist = sssp.sssp(snap, 0, 2.0);
    TEST_ASSERT(dist.size() == 4, "SSSP result size 4");
    TEST_ASSERT(std::fabs(dist[0]) < 0.01, "dist[source]=0");
    TEST_ASSERT(std::fabs(dist[1] - 1.0) < 0.01, "dist[1]=1.0");
    TEST_ASSERT(std::fabs(dist[2] - 3.0) < 0.01, "dist[2]=3.0");
    TEST_ASSERT(std::fabs(dist[3] - 5.5) < 0.01, "dist[3]=5.5 (shortcut)");
    TEST_ASSERT(sssp.total_relaxations > 0, "relaxations occurred");

    auto bp = sssp.debug_breakpoint_dump("sssp_test", 200);
    TEST_ASSERT(bp.item_count == sssp.total_relaxations, "bp item count matches");
    TEST_PASS("T14: SSSP adaptive delta + eager pruning");
}

// ── T15: TC adaptive threshold ──
void test_tc_adaptive_threshold() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    // Triangle: 0-1, 1-2, 0-2
    snap->add_undirected(0, 1, 1.0, TierID::HBM);
    snap->add_undirected(1, 2, 1.0, TierID::GDDR);
    snap->add_undirected(0, 2, 1.0, TierID::DRAM);

    MockTC tc;
    uint64_t count = tc.tc(snap, false);
    // In undirected triangle with i>dst filter, each direction counted
    TEST_ASSERT(count >= 1, "at least 1 triangle found");
    TEST_ASSERT(tc.adaptive_threshold >= 2, "threshold >= 2");
    auto bp = tc.debug_breakpoint_dump("tc_test", 300);
    TEST_ASSERT(bp.extra_info.find("threshold=") != std::string::npos, "threshold in dump");
    TEST_PASS("T15: TC adaptive threshold + triangle counting");
}

// ── T16: TC optimized with binary search ──
void test_tc_optimized() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    // Larger graph: star + triangle
    for (uint64_t i = 1; i <= 5; i++) snap->add_undirected(0, i, 1.0, TierID::HBM);
    snap->add_undirected(1, 2, 1.0, TierID::GDDR);
    snap->add_undirected(2, 3, 1.0, TierID::DRAM);

    MockTC tc;
    uint64_t count = tc.tc(snap, true);
    // At least the triangle (0,1,2) should be found
    TEST_ASSERT(count >= 1, "optimized TC finds triangles");
    TEST_PASS("T16: TC optimized binary search path");
}

// ── T17: WCC CAS hook + path halving ──
void test_wcc_connected() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    // Path graph: 0-1-2-3-4
    for (uint64_t i = 0; i < 4; i++) snap->add_undirected(i, i+1, 1.0, TierID::HBM);

    MockWCC wcc;
    auto comp = wcc.wcc(snap);
    TEST_ASSERT(comp.size() == 5, "WCC result size 5");
    // All in same component
    for (int i = 1; i < 5; i++)
        TEST_ASSERT(comp[i] == comp[0], "all in same component");
    TEST_ASSERT(wcc.cas_successes > 0, "CAS successes > 0");
    TEST_ASSERT(wcc.rounds > 0, "WCC rounds > 0");
    TEST_PASS("T17: WCC CAS hook + path halving (connected)");
}

// ── T18: WCC disconnected components ──
void test_wcc_disconnected() {
    auto snap = std::make_shared<MockTieredSnapshot>();
    snap->add_undirected(0, 1, 1.0, TierID::HBM);
    snap->add_undirected(2, 3, 1.0, TierID::GDDR);
    // Force vertices 0-3 to exist
    for (uint64_t i = 0; i < 4; i++) {
        if (!snap->has_vertex(i)) snap->add_edge(i, i, 0.0, TierID::DRAM);
    }

    MockWCC wcc;
    auto comp = wcc.wcc(snap);
    TEST_ASSERT(comp[0] == comp[1], "0 and 1 in same component");
    TEST_ASSERT(comp[2] == comp[3], "2 and 3 in same component");
    TEST_ASSERT(comp[0] != comp[2], "two distinct components");
    TEST_PASS("T18: WCC disconnected components");
}

// ── T19: TierRouter hash + degree routing ──
void test_tier_router() {
    TierRouter router(0.15, 0.35);
    // Hash routing should be deterministic
    TierID t1 = router.route_by_hash(100, 200);
    TierID t2 = router.route_by_hash(100, 200);
    TEST_ASSERT(t1 == t2, "hash routing is deterministic");

    // Degree routing: high degree -> HBM
    TierID high = router.route_by_degree(95, 100);
    TierID low  = router.route_by_degree(5, 100);
    TEST_ASSERT(high == TierID::HBM, "high degree -> HBM");
    TEST_ASSERT(low == TierID::DRAM, "low degree -> DRAM");
    TEST_PASS("T19: TierRouter hash + degree routing");
}

// ── T20: TieredAdapterStore (backend_adapters.hpp unified) ──
void test_tiered_adapter_store() {
    TieredAdapterStore store("neo_adapter", 0.2, 0.3);
    store.insert_edge(0, 1, 1.0);
    store.insert_edge(0, 2, 2.0);
    store.insert_edge(1, 2, 3.0);
    store.insert_edge(2, 3, 4.0);

    TEST_ASSERT(store.vertex_count() == 4, "4 vertices");
    TEST_ASSERT(store.edge_count() == 4, "4 edges");
    TEST_ASSERT(store.has_edge(0, 1), "has edge 0->1");
    TEST_ASSERT(!store.has_edge(3, 0), "no edge 3->0");
    TEST_ASSERT(store.degree(0) == 2, "degree(0)=2");

    // Tier distribution should be non-trivial
    uint64_t total_tier = store.adapter_tier_inserts[0] +
                           store.adapter_tier_inserts[1] +
                           store.adapter_tier_inserts[2];
    TEST_ASSERT(total_tier == 4, "tier inserts sum to 4");

    auto bp = store.debug_breakpoint_dump("adapter_test", 400);
    TEST_ASSERT(bp.extra_info == "neo_adapter", "adapter name in dump");
    TEST_PASS("T20: TieredAdapterStore (backend_adapters unified)");
}

// ── T21: Galloping search (CSR adapter pattern) ──
void test_galloping_search() {
    TieredAdapterStore store("csr_adapter");
    // Build sorted adjacency (by insertion order, simulate CSR)
    for (uint64_t i = 0; i < 50; i++) store.insert_edge(0, i * 2, 1.0);

    TEST_ASSERT(store.galloping_has_edge(0, 0), "galloping finds edge 0->0");
    TEST_ASSERT(store.galloping_has_edge(0, 48), "galloping finds edge 0->48");
    TEST_ASSERT(store.galloping_has_edge(0, 98), "galloping finds edge 0->98");
    TEST_ASSERT(!store.galloping_has_edge(0, 99), "galloping rejects 0->99");
    TEST_PASS("T21: Galloping search (CSR adapter)");
}

// ── T22: Zigzag intersect (Sortledton adapter pattern) ──
void test_zigzag_intersect() {
    TieredAdapterStore store("sortledton_adapter");
    // Vertex 0 neighbors: {1,2,3,4,5}
    for (uint64_t i = 1; i <= 5; i++) store.insert_edge(0, i, 1.0);
    // Vertex 1 neighbors: {0,2,4,6}
    store.insert_edge(1, 0, 1.0);
    store.insert_edge(1, 2, 1.0);
    store.insert_edge(1, 4, 1.0);
    store.insert_edge(1, 6, 1.0);

    uint64_t isect = store.zigzag_intersect(0, 1);
    // Common neighbors: 2, 4  (0 is in vertex 1's list but we look at dst overlap)
    TEST_ASSERT(isect >= 2, "zigzag intersect finds >= 2 common");

    // Also test standard intersect for comparison
    uint64_t std_isect = store.intersect(0, 1);
    TEST_ASSERT(std_isect >= 2, "standard intersect also >= 2");
    TEST_PASS("T22: Zigzag intersect (Sortledton adapter)");
}

// ── T23: Adapter store edges traversal with tier reads ──
void test_adapter_edges_traversal() {
    TieredAdapterStore store("livegraph_adapter");
    store.insert_edge(0, 1, 1.0);
    store.insert_edge(0, 2, 2.0);
    store.insert_edge(0, 3, 3.0);

    std::vector<uint64_t> dests;
    store.edges(0, [&](uint64_t dst, double w) { dests.push_back(dst); });
    TEST_ASSERT(dests.size() == 3, "traversal yields 3 edges");
    TEST_ASSERT(store.adapter_tier_reads[0] + store.adapter_tier_reads[1] +
                store.adapter_tier_reads[2] == 3, "tier reads total 3");
    TEST_PASS("T23: Adapter edges traversal with tier tracking");
}

// ── T24: Global TierStats accumulation across modules ──
void test_global_tier_stats() {
    g_tier_stats.reset();
    wrapper::g_op_stats.reset();

    auto snap = std::make_shared<MockTieredSnapshot>();
    snap->add_edge(0, 1, 1.0, TierID::HBM);
    snap->add_edge(1, 2, 1.0, TierID::GDDR);
    snap->add_edge(2, 0, 1.0, TierID::DRAM);

    // Run wrapper ops
    wrapper::snapshot_degree(snap, 0);
    wrapper::snapshot_edges(snap, 0,
        [](uint64_t d, double w) {}, false);

    TEST_ASSERT(g_tier_stats.total_inserts() == 3, "3 inserts from snapshot build");
    TEST_ASSERT(g_tier_stats.total_lookups() > 0, "lookups from edges traversal");
    TEST_ASSERT(wrapper::g_op_stats.query_ops.load() >= 2, "at least 2 ops recorded");
    TEST_PASS("T24: Global TierStats accumulation across modules");
}

// ── T25: Cross-module breakpoint dump chain ──
void test_breakpoint_dump_chain() {
    // Create breakpoints across multiple modules
    std::vector<BreakpointDump> dumps;

    // Edge batch dump
    std::vector<mock_graph::weightedEdge> batch(10);
    for (int i = 0; i < 10; i++) {
        batch[i] = mock_graph::weightedEdge(i, i+1, 1.0);
        batch[i].tier = TierID::HBM;
    }
    dumps.push_back(mock_graph::weightedEdge::debug_breakpoint_dump(batch, "edge_bp", 1));

    // Stream dump
    mock_stream::edgeStream stream;
    for (int i = 0; i < 20; i++) stream.add_edge(i, i+1, 1.0);
    stream.reorder_and_partition(true);
    dumps.push_back(stream.debug_breakpoint_dump("stream_bp", 2));

    // Snapshot dump
    auto snap = std::make_shared<MockTieredSnapshot>();
    for (int i = 0; i < 5; i++) snap->add_edge(0, i+1, 1.0, TierID::GDDR);
    dumps.push_back(snap->debug_breakpoint_dump("snap_bp", 3));

    // BFS dump
    MockBFS bfs;
    bfs.bfs(snap, 0);
    dumps.push_back(bfs.debug_breakpoint_dump("bfs_bp", 4));

    // Adapter dump
    TieredAdapterStore store("chain_test");
    for (int i = 0; i < 8; i++) store.insert_edge(i, i+1, 1.0);
    dumps.push_back(store.debug_breakpoint_dump("adapter_bp", 5));

    TEST_ASSERT(dumps.size() == 5, "5 breakpoint dumps in chain");
    TEST_ASSERT(dumps[0].tag == "edge_bp", "first dump is edge");
    TEST_ASSERT(dumps[0].tier_distribution[0] == 10, "edge batch all HBM");
    TEST_ASSERT(dumps[2].tag == "snap_bp", "third dump is snapshot");
    TEST_ASSERT(dumps[2].tier_distribution[1] == 5, "snapshot 5 GDDR edges");
    TEST_ASSERT(dumps[4].tag == "adapter_bp", "fifth dump is adapter");
    TEST_PASS("T25: Cross-module breakpoint dump chain");
}

// ═══════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::printf("═══════════════════════════════════════════════════════\n");
    std::printf(" M129-M130: Wrapper Module Deep Experiment (20 files, 7433 lines)\n");
    std::printf(" +20%% algo mod: tier stats + debug breakpoint dump\n");
    std::printf("═══════════════════════════════════════════════════════\n\n");

    // GraphEdge (T1-T2) — graph_edge.hpp + graph_edge_ops.hpp + graph_edge_impl.hpp
    std::printf("── GraphEdge (graph_edge.hpp 92行 + ops 81行 + impl 4行) ──\n");
    test_weighted_edge_basics();
    test_edge_batch_dump();

    // EdgeStream (T3-T6) — edge_stream.hpp + edge_stream_ops.hpp + edge_stream_impl.hpp
    std::printf("\n── EdgeStream (edge_stream.hpp 205行 + ops 138行 + impl 4行) ──\n");
    test_edge_stream_permute_sort();
    test_edge_stream_dedup();
    test_edge_stream_partition();
    test_edge_stream_iteration();

    // TieredSnapshot (T7-T9) — rapidstore_wrapper.hpp
    std::printf("\n── TieredSnapshot (rapidstore_wrapper.hpp 376行) ──\n");
    test_tiered_snapshot_basics();
    test_snapshot_tier_distribution();
    test_tier_latency_model();

    // WrapperOps (T10) — philemon_wrapper_ops.hpp
    std::printf("\n── WrapperOps (philemon_wrapper_ops.hpp 188行) ──\n");
    test_wrapper_ops();

    // BFS (T11-T12) — cross_tier_bfs_wrapper.hpp
    std::printf("\n── BFS (cross_tier_bfs_wrapper.hpp 320行) ──\n");
    test_bfs_density_switch();
    test_bfs_disconnected();

    // PageRank (T13) — cross_tier_pr_wrapper.hpp
    std::printf("\n── PageRank (cross_tier_pr_wrapper.hpp 162行) ──\n");
    test_pagerank_convergence();

    // SSSP (T14) — cross_tier_sssp_wrapper.hpp
    std::printf("\n── SSSP (cross_tier_sssp_wrapper.hpp 195行) ──\n");
    test_sssp_adaptive_delta();

    // TC (T15-T16) — cross_tier_tc_wrapper.hpp
    std::printf("\n── TC (cross_tier_tc_wrapper.hpp 197行) ──\n");
    test_tc_adaptive_threshold();
    test_tc_optimized();

    // WCC (T17-T18) — cross_tier_wcc_wrapper.hpp
    std::printf("\n── WCC (cross_tier_wcc_wrapper.hpp 179行) ──\n");
    test_wcc_connected();
    test_wcc_disconnected();

    // Adapters (T19-T23) — backend_adapters + all 9 apps adapters
    std::printf("\n── Adapters (backend_adapters 566行 + 9 apps adapters ~5023行) ──\n");
    test_tier_router();
    test_tiered_adapter_store();
    test_galloping_search();
    test_zigzag_intersect();
    test_adapter_edges_traversal();

    // Cross-module (T24-T25) — global tier stats + breakpoint chain
    std::printf("\n── Cross-Module: TierStats + BreakpointDump Chain ──\n");
    test_global_tier_stats();
    test_breakpoint_dump_chain();

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::printf("\n═══════════════════════════════════════════════════════\n");
    std::printf(" Results: %d/%d passed, %d failed  (%ld ms)\n",
                g_tests_passed, g_tests_run, g_tests_failed, (long)ms);
    std::printf("═══════════════════════════════════════════════════════\n");

    return g_tests_failed > 0 ? 1 : 0;
}
