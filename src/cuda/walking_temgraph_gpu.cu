/**
 * walking_temgraph_gpu.cu — TemGraph GPU 时序查询: CSR化 + range query + successor walk
 *
 * mv来源与算法改动对照:
 *
 *   tem_graph_impl.hpp (913行)
 *     KEEP: interval排序 + successor链遍历
 *     KEEP: contains(v, t1, t2) / contained(v, t1, t2) 查询逻辑
 *     KEEP: successor_link 链遍历 (CPU baseline)
 *     MOD:  successor链 linked list → CSR格式 (row_ptr/col_idx/timestamps)
 *     MOD:  contains → GPU kernel: 每thread一个查询, CSR内二分查找
 *     MOD:  successor walk → GPU kernel: 多起点并行遍历
 *
 *   walking_gpu_tree.cu (§1-§4 pattern)
 *     KEEP: Timer, INSPECT, CHK, sep, rss_kb debug基础设施
 *     KEEP: WALKING_CUDA dispatch宏, GPU_CHECK
 *     KEEP: kern_ prefix命名, blockDim/gridDim pattern
 *
 * Milestones:
 *   M083: TemGraph successor链CSR化 (~500行)
 *   M084: GPU temporal range query kernel (~400行)
 *   M085: successor walk batch kernel (~400行)
 *
 * Build:
 *   GPU:  nvcc -std=c++17 -O2 -arch=sm_86 -DWALKING_CUDA=1 -o walking_temgraph_gpu walking_temgraph_gpu.cu
 *   CPU:  g++ -std=c++17 -O2 -pthread -DWALKING_CUDA=0 -x c++ -o walking_temgraph_gpu walking_temgraph_gpu.cu
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <functional>
#include <sys/resource.h>
#include <unistd.h>

// ════════════════════════════════════════════════════════════════
// CUDA / CPU compile-mode dispatch
// ════════════════════════════════════════════════════════════════
#ifndef WALKING_CUDA
  #ifdef __CUDACC__
    #define WALKING_CUDA 1
  #else
    #define WALKING_CUDA 0
  #endif
#endif

#if WALKING_CUDA
  #include <cuda_runtime.h>
  #define GPU_CHECK(call) do { \
      cudaError_t e = (call); \
      if (e != cudaSuccess) { \
          fprintf(stderr, "[CUDA·FATAL] %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
          exit(1); } } while(0)
#else
  #define GPU_CHECK(call) ((void)0)
  enum { cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2 };
  inline void* _fake_alloc(size_t n) { return malloc(n); }
  #define cudaMalloc(p,n) (*(p)=_fake_alloc(n),(void)0)
  #define cudaFree(p) free(p)
  #define cudaMemcpy(d,s,n,k) memcpy(d,s,n)
  #define cudaMemset(p,v,n) memset(p,v,n)
  #define cudaDeviceSynchronize() ((void)0)
#endif

// ════════════════════════════════════════════════════════════════
// Debug infra  (same pattern as walking_gpu_tree.cu)
// ════════════════════════════════════════════════════════════════
static int g_dbg = 2;
static long rss_kb() { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss; }
static uint64_t g_insp = 0, g_pass = 0, g_fail = 0;

#define INSPECT(tag, ...) do { g_insp++; \
    std::printf("[INSPECT·%04lu·%s] ", (unsigned long)g_insp, tag); \
    std::printf(__VA_ARGS__); std::printf("  RSS=%ldKB\n", rss_kb()); } while(0)

#define CHK(cond, tag, ...) do { if(cond){g_pass++;} else { g_fail++; \
    std::printf("[FAIL·%s] ", tag); std::printf(__VA_ARGS__); std::printf("\n"); }} while(0)

struct Timer {
    const char* l; std::chrono::high_resolution_clock::time_point t0;
    Timer(const char* s) : l(s), t0(std::chrono::high_resolution_clock::now()) {
        if (g_dbg >= 1) std::printf("[T·START] %s\n", l);
    }
    double ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;
    }
    ~Timer() { std::printf("[T·END]   %s → %.2f ms\n", l, ms()); }
};

static void sep(const char* s) {
    std::printf("\n════════════════════════════════════════════════════\n  %s\n"
                "════════════════════════════════════════════════════\n\n", s);
}


// ════════════════════════════════════════════════════════════════════════
//  M083 §1: Temporal edge representation + CPU baseline
//  mv: tem_graph_impl.hpp TemGraphCore edge_list, successor_link
//  [KEEP] edge struct with (src, dst, timestamp)
//  [KEEP] successor traversal logic (CPU linked list baseline)
//  [MOD]  linked list → CSR conversion
// ════════════════════════════════════════════════════════════════════════

namespace walking {
namespace temgraph {

// ── Temporal edge: directed edge with timestamp ──
// [KEEP from upstream] matches TemGraphCore::Edge
struct TemEdge {
    uint32_t src;
    uint32_t dst;
    int64_t  timestamp;     // temporal coordinate
};

// ── Linked-list successor representation (CPU baseline) ──
// [KEEP from upstream] successor_link: for each vertex, a linked chain
// of outgoing edges sorted by timestamp
struct SuccessorNode {
    uint32_t dst;
    int64_t  timestamp;
    int32_t  next;          // index into pool (-1 = end)
};

struct LinkedSuccessor {
    std::vector<SuccessorNode> pool;
    std::vector<int32_t>       head;    // head[vertex] → index into pool
    uint32_t                   n_verts;

    LinkedSuccessor() : n_verts(0) {}

    // Build linked-list successor from edge list
    // [KEEP] upstream pattern: sort edges by (src, timestamp desc),
    // then build chain per vertex
    void build(const std::vector<TemEdge>& edges, uint32_t num_verts) {
        Timer t("LinkedSuccessor::build");
        n_verts = num_verts;
        head.assign(num_verts, -1);
        pool.clear();
        pool.reserve(edges.size());

        // Sort edges by (src, timestamp) ascending
        std::vector<size_t> order(edges.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (edges[a].src != edges[b].src) return edges[a].src < edges[b].src;
            return edges[a].timestamp < edges[b].timestamp;
        });

        // Build per-vertex linked chains (insert at head → reverse order)
        // This gives newest-first traversal order
        for (int i = (int)order.size() - 1; i >= 0; --i) {
            const TemEdge& e = edges[order[i]];
            int32_t idx = (int32_t)pool.size();
            pool.push_back({e.dst, e.timestamp, head[e.src]});
            head[e.src] = idx;
        }

        INSPECT("LINKED_SUCC", "verts=%u edges=%lu pool=%lu",
                n_verts, edges.size(), pool.size());
    }

    // [KEEP] successor walk: from vertex v, follow chain collecting edges
    // in [t1, t2] window
    int contains_cpu(uint32_t v, int64_t t1, int64_t t2,
                     std::vector<uint32_t>* out_dsts = nullptr) const {
        if (v >= n_verts) return 0;
        int count = 0;
        int32_t cur = head[v];
        while (cur >= 0) {
            const SuccessorNode& sn = pool[cur];
            if (sn.timestamp >= t1 && sn.timestamp <= t2) {
                count++;
                if (out_dsts) out_dsts->push_back(sn.dst);
            }
            // [KEEP] early termination: chain sorted ascending,
            // once past t2 we stop
            if (sn.timestamp > t2) break;
            cur = sn.next;
        }
        return count;
    }

    // [KEEP] successor walk: from start, follow successor chain max_hops steps
    // At each step, pick the first edge in time window (or any edge if unconstrained)
    void walk_cpu(uint32_t start, int max_hops,
                  std::vector<uint32_t>& path,
                  int64_t t_start = INT64_MIN,
                  int64_t t_end = INT64_MAX) const {
        path.clear();
        path.push_back(start);
        uint32_t cur = start;
        for (int step = 0; step < max_hops; ++step) {
            if (cur >= n_verts) break;
            int32_t idx = head[cur];
            uint32_t next_v = UINT32_MAX;
            while (idx >= 0) {
                const SuccessorNode& sn = pool[idx];
                if (sn.timestamp >= t_start && sn.timestamp <= t_end) {
                    next_v = sn.dst;
                    break;  // take first valid successor
                }
                if (sn.timestamp > t_end) break;
                idx = sn.next;
            }
            if (next_v == UINT32_MAX) break;
            path.push_back(next_v);
            cur = next_v;
        }
    }
};


// ════════════════════════════════════════════════════════════════════════
//  M083 §2: FlatSuccessorCSR — GPU-portable CSR representation
//  [MOD] successor链 linked list → CSR格式
//  row_ptr[v] .. row_ptr[v+1]: edges out of vertex v, sorted by timestamp
//  col_idx[i]: destination vertex of edge i
//  timestamps[i]: timestamp of edge i
// ════════════════════════════════════════════════════════════════════════

struct FlatSuccessorCSR {
    uint32_t  n_verts;
    uint32_t  n_edges;
    std::vector<uint32_t> row_ptr;      // size n_verts+1
    std::vector<uint32_t> col_idx;      // size n_edges
    std::vector<int64_t>  timestamps;   // size n_edges

    FlatSuccessorCSR() : n_verts(0), n_edges(0) {}

    // [NEW] build_csr: from edge list, construct CSR with per-vertex
    // timestamp-sorted adjacency
    void build(const std::vector<TemEdge>& edges, uint32_t num_verts) {
        Timer t("FlatSuccessorCSR::build");
        n_verts = num_verts;
        n_edges = (uint32_t)edges.size();

        // Step 1: count edges per vertex
        std::vector<uint32_t> degree(num_verts, 0);
        for (const auto& e : edges) {
            if (e.src < num_verts) degree[e.src]++;
        }

        INSPECT("CSR_DEGREE", "max_deg=%u min_deg=%u total_edges=%u",
                *std::max_element(degree.begin(), degree.end()),
                *std::min_element(degree.begin(), degree.end()),
                n_edges);

        // Step 2: prefix sum → row_ptr
        row_ptr.resize(num_verts + 1);
        row_ptr[0] = 0;
        for (uint32_t v = 0; v < num_verts; ++v) {
            row_ptr[v + 1] = row_ptr[v] + degree[v];
        }

        CHK(row_ptr[num_verts] == n_edges, "CSR_ROWPTR",
            "row_ptr[end]=%u != n_edges=%u", row_ptr[num_verts], n_edges);

        if (g_dbg >= 2) {
            // Print first 10 row_ptr entries
            std::printf("[CSR·ROWPTR] first 10:");
            for (uint32_t v = 0; v < std::min(num_verts + 1, 10u); ++v) {
                std::printf(" [%u]=%u", v, row_ptr[v]);
            }
            std::printf("\n");
        }

        // Step 3: scatter edges into CSR arrays
        col_idx.resize(n_edges);
        timestamps.resize(n_edges);
        std::vector<uint32_t> offset(num_verts, 0);    // write cursor per vertex

        // Sort edges by (src, timestamp) for deterministic ordering
        std::vector<size_t> order(edges.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (edges[a].src != edges[b].src) return edges[a].src < edges[b].src;
            return edges[a].timestamp < edges[b].timestamp;
        });

        for (size_t oi = 0; oi < order.size(); ++oi) {
            const TemEdge& e = edges[order[oi]];
            uint32_t v = e.src;
            uint32_t pos = row_ptr[v] + offset[v];
            col_idx[pos] = e.dst;
            timestamps[pos] = e.timestamp;
            offset[v]++;
        }

        // Step 4: verify per-vertex timestamp sorting
        uint32_t unsorted_count = 0;
        for (uint32_t v = 0; v < num_verts; ++v) {
            for (uint32_t j = row_ptr[v] + 1; j < row_ptr[v + 1]; ++j) {
                if (timestamps[j] < timestamps[j - 1]) unsorted_count++;
            }
        }
        CHK(unsorted_count == 0, "CSR_SORTED",
            "unsorted_pairs=%u (should be 0)", unsorted_count);

        // Step 5: edge count statistics
        uint32_t isolated = 0, max_deg = 0;
        double avg_deg = 0.0;
        for (uint32_t v = 0; v < num_verts; ++v) {
            uint32_t d = row_ptr[v + 1] - row_ptr[v];
            if (d == 0) isolated++;
            max_deg = std::max(max_deg, d);
        }
        avg_deg = n_edges > 0 ? (double)n_edges / num_verts : 0.0;

        INSPECT("CSR_BUILT", "verts=%u edges=%u isolated=%u max_deg=%u avg_deg=%.2f",
                n_verts, n_edges, isolated, max_deg, avg_deg);

        // Memory footprint
        size_t mem = row_ptr.size() * sizeof(uint32_t)
                   + col_idx.size() * sizeof(uint32_t)
                   + timestamps.size() * sizeof(int64_t);
        INSPECT("CSR_MEM", "%.2f MB (row_ptr=%.2f col_idx=%.2f ts=%.2f)",
                mem / (1024.0 * 1024.0),
                row_ptr.size() * sizeof(uint32_t) / (1024.0 * 1024.0),
                col_idx.size() * sizeof(uint32_t) / (1024.0 * 1024.0),
                timestamps.size() * sizeof(int64_t) / (1024.0 * 1024.0));
    }

    // [NEW] CPU range query on CSR: count edges of vertex v with timestamp in [t1,t2]
    // Uses binary search on the sorted timestamp array within row_ptr[v]..row_ptr[v+1]
    int range_query_cpu(uint32_t v, int64_t t1, int64_t t2) const {
        if (v >= n_verts) return 0;
        uint32_t lo = row_ptr[v], hi = row_ptr[v + 1];
        if (lo == hi) return 0;

        // lower_bound for t1
        uint32_t lb = lo, rb = hi;
        while (lb < rb) {
            uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] < t1) lb = mid + 1;
            else rb = mid;
        }
        uint32_t start = lb;

        // upper_bound for t2
        lb = start; rb = hi;
        while (lb < rb) {
            uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] <= t2) lb = mid + 1;
            else rb = mid;
        }
        uint32_t end = lb;

        return (int)(end - start);
    }

    // [NEW] CPU range query returning matched destinations
    int range_query_cpu_full(uint32_t v, int64_t t1, int64_t t2,
                             std::vector<uint32_t>* out_dsts = nullptr,
                             uint32_t* out_first_pos = nullptr) const {
        if (v >= n_verts) return 0;
        uint32_t lo = row_ptr[v], hi = row_ptr[v + 1];
        if (lo == hi) return 0;

        // lower_bound for t1
        uint32_t lb = lo, rb = hi;
        while (lb < rb) {
            uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] < t1) lb = mid + 1;
            else rb = mid;
        }
        uint32_t start = lb;

        // upper_bound for t2
        lb = start; rb = hi;
        while (lb < rb) {
            uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] <= t2) lb = mid + 1;
            else rb = mid;
        }
        uint32_t end = lb;

        if (out_first_pos) *out_first_pos = start;
        if (out_dsts) {
            for (uint32_t i = start; i < end; ++i) {
                out_dsts->push_back(col_idx[i]);
            }
        }
        return (int)(end - start);
    }

    // [NEW] CPU successor walk on CSR: from start, follow first valid edge
    // in time window at each step
    void walk_cpu(uint32_t start, int max_hops,
                  std::vector<uint32_t>& path,
                  int64_t t_start = INT64_MIN,
                  int64_t t_end = INT64_MAX) const {
        path.clear();
        path.push_back(start);
        uint32_t cur = start;
        for (int step = 0; step < max_hops; ++step) {
            if (cur >= n_verts) break;
            uint32_t lo = row_ptr[cur], hi = row_ptr[cur + 1];
            if (lo == hi) break;

            // Binary search: lower_bound for t_start
            uint32_t lb = lo, rb = hi;
            while (lb < rb) {
                uint32_t mid = lb + (rb - lb) / 2;
                if (timestamps[mid] < t_start) lb = mid + 1;
                else rb = mid;
            }

            // Find first edge within [t_start, t_end]
            uint32_t next_v = UINT32_MAX;
            for (uint32_t j = lb; j < hi; ++j) {
                if (timestamps[j] > t_end) break;
                if (timestamps[j] >= t_start) {
                    next_v = col_idx[j];
                    break;
                }
            }
            if (next_v == UINT32_MAX || next_v >= n_verts) break;
            path.push_back(next_v);
            cur = next_v;
        }
    }

    // [NEW] dump first N vertices of CSR for debugging
    void dump(const char* tag, int max_verts = 10) const {
        std::printf("[CSR·%s] n_verts=%u n_edges=%u\n", tag, n_verts, n_edges);
        for (uint32_t v = 0; v < std::min(n_verts, (uint32_t)max_verts); ++v) {
            uint32_t lo = row_ptr[v], hi = row_ptr[v + 1];
            std::printf("  v=%u deg=%u:", v, hi - lo);
            for (uint32_t j = lo; j < std::min(lo + 5u, hi); ++j) {
                std::printf(" →%u@%ld", col_idx[j], (long)timestamps[j]);
            }
            if (hi - lo > 5) std::printf(" ...");
            std::printf("\n");
        }
    }
};


// ════════════════════════════════════════════════════════════════════════
//  M083 §3: FlatTemGraph — unified GPU-portable temporal graph struct
//  [NEW] Wraps CSR data in a format suitable for GPU memcpy
// ════════════════════════════════════════════════════════════════════════

struct FlatTemGraph {
    uint32_t  n_verts;
    uint32_t  n_edges;

    // Host arrays (owned)
    std::vector<uint32_t> h_row_ptr;
    std::vector<uint32_t> h_col_idx;
    std::vector<int64_t>  h_timestamps;

    // Device pointers (managed by alloc_gpu / free_gpu)
    uint32_t* d_row_ptr     = nullptr;
    uint32_t* d_col_idx     = nullptr;
    int64_t*  d_timestamps  = nullptr;

    FlatTemGraph() : n_verts(0), n_edges(0) {}

    // Build from CSR
    void build_from_csr(const FlatSuccessorCSR& csr) {
        n_verts = csr.n_verts;
        n_edges = csr.n_edges;
        h_row_ptr = csr.row_ptr;
        h_col_idx = csr.col_idx;
        h_timestamps = csr.timestamps;
        INSPECT("FLATTEM", "built from CSR: verts=%u edges=%u", n_verts, n_edges);
    }

    // Build from raw edges
    void build(const std::vector<TemEdge>& edges, uint32_t num_verts) {
        FlatSuccessorCSR csr;
        csr.build(edges, num_verts);
        build_from_csr(csr);
    }

    void alloc_gpu() {
        Timer t("FlatTemGraph::alloc_gpu");
        size_t rp_bytes = h_row_ptr.size() * sizeof(uint32_t);
        size_t ci_bytes = h_col_idx.size() * sizeof(uint32_t);
        size_t ts_bytes = h_timestamps.size() * sizeof(int64_t);

        GPU_CHECK(cudaMalloc(&d_row_ptr, rp_bytes));
        GPU_CHECK(cudaMalloc(&d_col_idx, ci_bytes));
        GPU_CHECK(cudaMalloc(&d_timestamps, ts_bytes));

        GPU_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), rp_bytes, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), ci_bytes, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_timestamps, h_timestamps.data(), ts_bytes, cudaMemcpyHostToDevice));

        INSPECT("GPU_ALLOC", "row_ptr=%.2fMB col_idx=%.2fMB timestamps=%.2fMB total=%.2fMB",
                rp_bytes / (1024.0 * 1024.0), ci_bytes / (1024.0 * 1024.0),
                ts_bytes / (1024.0 * 1024.0),
                (rp_bytes + ci_bytes + ts_bytes) / (1024.0 * 1024.0));
    }

    void free_gpu() {
        if (d_row_ptr)    { GPU_CHECK(cudaFree(d_row_ptr));    d_row_ptr = nullptr; }
        if (d_col_idx)    { GPU_CHECK(cudaFree(d_col_idx));    d_col_idx = nullptr; }
        if (d_timestamps) { GPU_CHECK(cudaFree(d_timestamps)); d_timestamps = nullptr; }
    }

    ~FlatTemGraph() { free_gpu(); }
};


// ════════════════════════════════════════════════════════════════════════
//  M083 §4: Random temporal graph generator (for experiments)
// ════════════════════════════════════════════════════════════════════════

static std::vector<TemEdge> gen_temporal_graph(
    uint32_t n_verts, uint32_t n_edges,
    int64_t t_min, int64_t t_max, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> vdist(0, n_verts - 1);
    std::uniform_int_distribution<int64_t>  tdist(t_min, t_max);

    std::vector<TemEdge> edges(n_edges);
    for (uint32_t i = 0; i < n_edges; ++i) {
        edges[i].src = vdist(rng);
        edges[i].dst = vdist(rng);
        edges[i].timestamp = tdist(rng);
    }

    INSPECT("GEN_GRAPH", "verts=%u edges=%u t_range=[%ld,%ld] seed=%lu",
            n_verts, n_edges, (long)t_min, (long)t_max, (unsigned long)seed);
    return edges;
}


// ════════════════════════════════════════════════════════════════════════
//  M084 §1: GPU temporal range query kernel
//  mv: tem_graph_impl.hpp contains_query → GPU parallel
//  [KEEP] binary search on sorted timestamps within CSR row
//  [MOD]  per-thread: one (vertex, t1, t2) query
// ════════════════════════════════════════════════════════════════════════

// Query struct for GPU transfer
struct TemporalRangeQuery {
    uint32_t vertex;
    int64_t  t1;
    int64_t  t2;
};

#if WALKING_CUDA

// ── kern_temporal_range_query: one thread per query ──
// Each thread does binary search within CSR[row_ptr[v] .. row_ptr[v+1]]
// to find edges with timestamp in [t1, t2]
// [KEEP] binary search logic from upstream contains_query
// [MOD]  CSR array access instead of linked list traversal
__global__ void kern_temporal_range_query(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ col_idx,
    const int64_t*  __restrict__ timestamps,
    uint32_t                     n_verts,
    const uint32_t* __restrict__ q_vertices,
    const int64_t*  __restrict__ q_t1,
    const int64_t*  __restrict__ q_t2,
    int32_t*        __restrict__ counts,
    uint32_t*       __restrict__ first_match,
    uint64_t                     n_queries)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= n_queries) return;

    uint32_t v = q_vertices[tid];
    int64_t  t1 = q_t1[tid];
    int64_t  t2 = q_t2[tid];

    if (v >= n_verts) {
        counts[tid] = 0;
        first_match[tid] = UINT32_MAX;
        return;
    }

    uint32_t lo = row_ptr[v];
    uint32_t hi = row_ptr[v + 1];

    if (lo == hi) {
        counts[tid] = 0;
        first_match[tid] = UINT32_MAX;
        return;
    }

    // lower_bound: first timestamp >= t1
    uint32_t lb = lo, rb = hi;
    while (lb < rb) {
        uint32_t mid = lb + (rb - lb) / 2;
        if (timestamps[mid] < t1) lb = mid + 1;
        else rb = mid;
    }
    uint32_t start = lb;

    // upper_bound: first timestamp > t2
    lb = start; rb = hi;
    while (lb < rb) {
        uint32_t mid = lb + (rb - lb) / 2;
        if (timestamps[mid] <= t2) lb = mid + 1;
        else rb = mid;
    }
    uint32_t end = lb;

    counts[tid] = (int32_t)(end - start);
    first_match[tid] = (start < end) ? start : UINT32_MAX;
}

// ── kern_temporal_range_batch: batch version with SoA queries ──
// Identical logic but processes queries from a packed TemporalRangeQuery array
// [NEW] batch variant for more compact transfer
__global__ void kern_temporal_range_batch(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ col_idx,
    const int64_t*  __restrict__ timestamps,
    uint32_t                     n_verts,
    const uint32_t* __restrict__ batch_vertices,
    const int64_t*  __restrict__ batch_t1,
    const int64_t*  __restrict__ batch_t2,
    int32_t*        __restrict__ batch_counts,
    uint32_t*       __restrict__ batch_first,
    uint64_t                     n_queries)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= n_queries) return;

    uint32_t v  = batch_vertices[tid];
    int64_t  t1 = batch_t1[tid];
    int64_t  t2 = batch_t2[tid];

    if (v >= n_verts) {
        batch_counts[tid] = 0;
        batch_first[tid]  = UINT32_MAX;
        return;
    }

    uint32_t lo = row_ptr[v];
    uint32_t hi = row_ptr[v + 1];

    if (lo == hi) {
        batch_counts[tid] = 0;
        batch_first[tid]  = UINT32_MAX;
        return;
    }

    // lower_bound for t1
    uint32_t lb = lo, rb = hi;
    while (lb < rb) {
        uint32_t mid = lb + (rb - lb) / 2;
        if (timestamps[mid] < t1) lb = mid + 1;
        else rb = mid;
    }
    uint32_t start = lb;

    // upper_bound for t2
    lb = start; rb = hi;
    while (lb < rb) {
        uint32_t mid = lb + (rb - lb) / 2;
        if (timestamps[mid] <= t2) lb = mid + 1;
        else rb = mid;
    }
    uint32_t end = lb;

    batch_counts[tid] = (int32_t)(end - start);
    batch_first[tid]  = (start < end) ? start : UINT32_MAX;
}

#else
// CPU stubs for compilation without CUDA
static void kern_temporal_range_query(
    const uint32_t* row_ptr, const uint32_t* col_idx, const int64_t* timestamps,
    uint32_t n_verts,
    const uint32_t* q_vertices, const int64_t* q_t1, const int64_t* q_t2,
    int32_t* counts, uint32_t* first_match, uint64_t n_queries)
{
    for (uint64_t tid = 0; tid < n_queries; ++tid) {
        uint32_t v = q_vertices[tid];
        int64_t t1 = q_t1[tid], t2 = q_t2[tid];
        if (v >= n_verts) { counts[tid] = 0; first_match[tid] = UINT32_MAX; continue; }

        uint32_t lo = row_ptr[v], hi = row_ptr[v + 1];
        if (lo == hi) { counts[tid] = 0; first_match[tid] = UINT32_MAX; continue; }

        uint32_t lb = lo, rb = hi;
        while (lb < rb) { uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] < t1) lb = mid + 1; else rb = mid; }
        uint32_t start = lb;

        lb = start; rb = hi;
        while (lb < rb) { uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] <= t2) lb = mid + 1; else rb = mid; }
        uint32_t end = lb;

        counts[tid] = (int32_t)(end - start);
        first_match[tid] = (start < end) ? start : UINT32_MAX;
    }
}

static void kern_temporal_range_batch(
    const uint32_t* row_ptr, const uint32_t* col_idx, const int64_t* timestamps,
    uint32_t n_verts,
    const uint32_t* batch_vertices, const int64_t* batch_t1, const int64_t* batch_t2,
    int32_t* batch_counts, uint32_t* batch_first, uint64_t n_queries)
{
    kern_temporal_range_query(row_ptr, col_idx, timestamps, n_verts,
        batch_vertices, batch_t1, batch_t2, batch_counts, batch_first, n_queries);
}
#endif


// ════════════════════════════════════════════════════════════════════════
//  M085 §1: GPU successor walk kernels
//  mv: tem_graph_impl.hpp successor_link traversal
//  [KEEP] walk logic: from start, at each step pick first valid successor
//  [MOD]  single-start serial → multi-start GPU parallel
// ════════════════════════════════════════════════════════════════════════

#if WALKING_CUDA

// ── kern_successor_walk: unconstrained walk ──
// Each thread: from starts[tid], follow first outgoing edge at each step
// Record path into paths[tid * max_hops + step]
// [MOD] linked list traversal → CSR first-neighbor access
__global__ void kern_successor_walk(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ col_idx,
    const int64_t*  __restrict__ timestamps,
    uint32_t                     n_verts,
    const uint32_t* __restrict__ starts,
    uint32_t*       __restrict__ paths,         // [n_starts * max_hops]
    uint32_t*       __restrict__ path_lengths,   // [n_starts]
    uint32_t                     n_starts,
    uint32_t                     max_hops)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_starts) return;

    uint32_t cur = starts[tid];
    uint32_t len = 0;
    uint32_t base = tid * max_hops;

    // Initialize path to UINT32_MAX (invalid)
    for (uint32_t s = 0; s < max_hops; ++s) {
        paths[base + s] = UINT32_MAX;
    }

    for (uint32_t step = 0; step < max_hops; ++step) {
        if (cur >= n_verts) break;

        uint32_t lo = row_ptr[cur];
        uint32_t hi = row_ptr[cur + 1];
        if (lo == hi) break;   // no outgoing edges → stop

        // Pick first outgoing edge (unconstrained)
        uint32_t next_v = col_idx[lo];
        if (next_v >= n_verts) break;

        paths[base + step] = next_v;
        len++;
        cur = next_v;
    }
    path_lengths[tid] = len;
}

// ── kern_successor_walk_timed: time-constrained walk ──
// [NEW] Only follow edges with timestamp in [t_start, t_end]
// Binary search at each step to find first valid edge
__global__ void kern_successor_walk_timed(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ col_idx,
    const int64_t*  __restrict__ timestamps,
    uint32_t                     n_verts,
    const uint32_t* __restrict__ starts,
    uint32_t*       __restrict__ paths,          // [n_starts * max_hops]
    uint32_t*       __restrict__ path_lengths,    // [n_starts]
    int64_t*        __restrict__ path_timestamps, // [n_starts * max_hops]
    uint32_t                     n_starts,
    uint32_t                     max_hops,
    int64_t                      t_start,
    int64_t                      t_end)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_starts) return;

    uint32_t cur = starts[tid];
    uint32_t len = 0;
    uint32_t base = tid * max_hops;

    for (uint32_t s = 0; s < max_hops; ++s) {
        paths[base + s] = UINT32_MAX;
        path_timestamps[base + s] = INT64_MIN;
    }

    for (uint32_t step = 0; step < max_hops; ++step) {
        if (cur >= n_verts) break;

        uint32_t lo = row_ptr[cur];
        uint32_t hi = row_ptr[cur + 1];
        if (lo == hi) break;

        // Binary search: lower_bound for t_start
        uint32_t lb = lo, rb = hi;
        while (lb < rb) {
            uint32_t mid = lb + (rb - lb) / 2;
            if (timestamps[mid] < t_start) lb = mid + 1;
            else rb = mid;
        }

        // Find first edge in [t_start, t_end]
        uint32_t next_v = UINT32_MAX;
        int64_t  next_ts = INT64_MIN;
        for (uint32_t j = lb; j < hi; ++j) {
            if (timestamps[j] > t_end) break;
            if (timestamps[j] >= t_start) {
                next_v = col_idx[j];
                next_ts = timestamps[j];
                break;
            }
        }

        if (next_v == UINT32_MAX || next_v >= n_verts) break;

        paths[base + step] = next_v;
        path_timestamps[base + step] = next_ts;
        len++;
        cur = next_v;
    }
    path_lengths[tid] = len;
}

#else
// CPU stubs
static void kern_successor_walk(
    const uint32_t* row_ptr, const uint32_t* col_idx, const int64_t* timestamps,
    uint32_t n_verts, const uint32_t* starts,
    uint32_t* paths, uint32_t* path_lengths,
    uint32_t n_starts, uint32_t max_hops)
{
    for (uint32_t tid = 0; tid < n_starts; ++tid) {
        uint32_t cur = starts[tid];
        uint32_t len = 0;
        uint32_t base = tid * max_hops;
        for (uint32_t s = 0; s < max_hops; ++s) paths[base + s] = UINT32_MAX;

        for (uint32_t step = 0; step < max_hops; ++step) {
            if (cur >= n_verts) break;
            uint32_t lo = row_ptr[cur], hi = row_ptr[cur + 1];
            if (lo == hi) break;
            uint32_t next_v = col_idx[lo];
            if (next_v >= n_verts) break;
            paths[base + step] = next_v;
            len++;
            cur = next_v;
        }
        path_lengths[tid] = len;
    }
}

static void kern_successor_walk_timed(
    const uint32_t* row_ptr, const uint32_t* col_idx, const int64_t* timestamps,
    uint32_t n_verts, const uint32_t* starts,
    uint32_t* paths, uint32_t* path_lengths, int64_t* path_timestamps,
    uint32_t n_starts, uint32_t max_hops, int64_t t_start, int64_t t_end)
{
    for (uint32_t tid = 0; tid < n_starts; ++tid) {
        uint32_t cur = starts[tid];
        uint32_t len = 0;
        uint32_t base = tid * max_hops;
        for (uint32_t s = 0; s < max_hops; ++s) {
            paths[base + s] = UINT32_MAX;
            path_timestamps[base + s] = INT64_MIN;
        }

        for (uint32_t step = 0; step < max_hops; ++step) {
            if (cur >= n_verts) break;
            uint32_t lo = row_ptr[cur], hi = row_ptr[cur + 1];
            if (lo == hi) break;

            uint32_t lb = lo, rb = hi;
            while (lb < rb) { uint32_t mid = lb + (rb - lb) / 2;
                if (timestamps[mid] < t_start) lb = mid + 1; else rb = mid; }

            uint32_t next_v = UINT32_MAX;
            int64_t next_ts = INT64_MIN;
            for (uint32_t j = lb; j < hi; ++j) {
                if (timestamps[j] > t_end) break;
                if (timestamps[j] >= t_start) { next_v = col_idx[j]; next_ts = timestamps[j]; break; }
            }
            if (next_v == UINT32_MAX || next_v >= n_verts) break;
            paths[base + step] = next_v;
            path_timestamps[base + step] = next_ts;
            len++;
            cur = next_v;
        }
        path_lengths[tid] = len;
    }
}
#endif


// ════════════════════════════════════════════════════════════════════════
//  M083 §5: exp_csr_build — CSR construction correctness test
// ════════════════════════════════════════════════════════════════════════

static void exp_csr_build() {
    sep("M083: exp_csr_build — CSR Construction Correctness");

    // ── Test 1: Small deterministic graph ──
    {
        Timer t("csr_small_deterministic");
        std::vector<TemEdge> edges = {
            {0, 1, 100}, {0, 2, 200}, {0, 1, 150},
            {1, 2, 300}, {1, 0, 250},
            {2, 0, 400}, {2, 1, 350}, {2, 0, 500},
        };
        uint32_t nv = 3;
        FlatSuccessorCSR csr;
        csr.build(edges, nv);
        csr.dump("SMALL", 10);

        // Verify: vertex 0 has 3 edges, vertex 1 has 2, vertex 2 has 3
        CHK(csr.row_ptr[1] - csr.row_ptr[0] == 3, "CSR_DEG_V0",
            "expected 3, got %u", csr.row_ptr[1] - csr.row_ptr[0]);
        CHK(csr.row_ptr[2] - csr.row_ptr[1] == 2, "CSR_DEG_V1",
            "expected 2, got %u", csr.row_ptr[2] - csr.row_ptr[1]);
        CHK(csr.row_ptr[3] - csr.row_ptr[2] == 3, "CSR_DEG_V2",
            "expected 3, got %u", csr.row_ptr[3] - csr.row_ptr[2]);

        // Verify timestamps sorted per vertex
        for (uint32_t v = 0; v < nv; ++v) {
            for (uint32_t j = csr.row_ptr[v] + 1; j < csr.row_ptr[v + 1]; ++j) {
                CHK(csr.timestamps[j] >= csr.timestamps[j - 1], "CSR_SORT",
                    "v=%u pos=%u: %ld < %ld", v, j,
                    (long)csr.timestamps[j], (long)csr.timestamps[j - 1]);
            }
        }

        // Verify range query on vertex 0: [100, 200] should yield 3 edges
        int cnt = csr.range_query_cpu(0, 100, 200);
        CHK(cnt == 3, "CSR_RANGE_V0", "expected 3, got %d", cnt);

        // Range query vertex 2: [350, 400] → 2 edges
        cnt = csr.range_query_cpu(2, 350, 400);
        CHK(cnt == 2, "CSR_RANGE_V2", "expected 2, got %d", cnt);

        INSPECT("SMALL_DONE", "all CSR small tests passed");
    }

    // ── Test 2: Medium random graph (1K vertices, 10K edges) ──
    {
        Timer t("csr_medium_1K");
        auto edges = gen_temporal_graph(1000, 10000, 0, 1000000, 42);
        FlatSuccessorCSR csr;
        csr.build(edges, 1000);

        // Verify every edge can be found in CSR
        uint32_t found = 0, missing = 0;
        for (const auto& e : edges) {
            bool ok = false;
            for (uint32_t j = csr.row_ptr[e.src]; j < csr.row_ptr[e.src + 1]; ++j) {
                if (csr.col_idx[j] == e.dst && csr.timestamps[j] == e.timestamp) {
                    ok = true; break;
                }
            }
            if (ok) found++; else missing++;
        }
        CHK(missing == 0, "CSR_1K_COMPLETE",
            "found=%u missing=%u", found, missing);
        INSPECT("1K_DONE", "found=%u/%lu edges in CSR", found, edges.size());
    }

    // ── Test 3: Large random graph (10K vertices, 100K edges) ──
    {
        Timer t("csr_large_10K");
        auto edges = gen_temporal_graph(10000, 100000, 0, 10000000, 123);
        FlatSuccessorCSR csr;
        csr.build(edges, 10000);

        // Spot-check 1000 random edges
        std::mt19937_64 rng(999);
        uint32_t spot_ok = 0;
        for (int i = 0; i < 1000; ++i) {
            size_t idx = rng() % edges.size();
            const auto& e = edges[idx];
            bool found = false;
            for (uint32_t j = csr.row_ptr[e.src]; j < csr.row_ptr[e.src + 1]; ++j) {
                if (csr.col_idx[j] == e.dst && csr.timestamps[j] == e.timestamp) {
                    found = true; break;
                }
            }
            if (found) spot_ok++;
        }
        CHK(spot_ok == 1000, "CSR_10K_SPOT",
            "spot_ok=%u/1000", spot_ok);
        INSPECT("10K_DONE", "spot_check=%u/1000", spot_ok);
    }

    // ── Test 4: CSR vs LinkedSuccessor cross-validation ──
    {
        Timer t("csr_vs_linked_crossval");
        uint32_t nv = 500, ne = 5000;
        auto edges = gen_temporal_graph(nv, ne, 100, 100000, 777);

        FlatSuccessorCSR csr;
        csr.build(edges, nv);

        LinkedSuccessor linked;
        linked.build(edges, nv);

        // Cross-validate range queries
        std::mt19937_64 rng(888);
        uint32_t match = 0, mismatch = 0;
        for (int q = 0; q < 500; ++q) {
            uint32_t v = rng() % nv;
            int64_t t1 = 100 + (int64_t)(rng() % 50000);
            int64_t t2 = t1 + (int64_t)(rng() % 30000);

            int csr_cnt = csr.range_query_cpu(v, t1, t2);
            int lnk_cnt = linked.contains_cpu(v, t1, t2);

            if (csr_cnt == lnk_cnt) match++;
            else {
                mismatch++;
                if (mismatch <= 5) {
                    std::printf("[XVAL·MISMATCH] v=%u t=[%ld,%ld] csr=%d linked=%d\n",
                                v, (long)t1, (long)t2, csr_cnt, lnk_cnt);
                }
            }
        }
        CHK(mismatch == 0, "CSR_VS_LINKED",
            "match=%u mismatch=%u", match, mismatch);
        INSPECT("CROSSVAL", "match=%u mismatch=%u", match, mismatch);
    }

    // ── Test 5: FlatTemGraph GPU allocation round-trip ──
    {
        Timer t("flat_temgraph_roundtrip");
        auto edges = gen_temporal_graph(1000, 5000, 0, 500000, 555);
        FlatTemGraph ftg;
        ftg.build(edges, 1000);
        ftg.alloc_gpu();

        // Read back from GPU and verify
        std::vector<uint32_t> rb_row_ptr(ftg.h_row_ptr.size());
        std::vector<uint32_t> rb_col_idx(ftg.h_col_idx.size());
        std::vector<int64_t>  rb_timestamps(ftg.h_timestamps.size());

        GPU_CHECK(cudaMemcpy(rb_row_ptr.data(), ftg.d_row_ptr,
                  rb_row_ptr.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(rb_col_idx.data(), ftg.d_col_idx,
                  rb_col_idx.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(rb_timestamps.data(), ftg.d_timestamps,
                  rb_timestamps.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));

        bool rp_ok = (rb_row_ptr == ftg.h_row_ptr);
        bool ci_ok = (rb_col_idx == ftg.h_col_idx);
        bool ts_ok = (rb_timestamps == ftg.h_timestamps);
        CHK(rp_ok, "GPU_RT_ROWPTR", "row_ptr mismatch after round-trip");
        CHK(ci_ok, "GPU_RT_COLIDX", "col_idx mismatch after round-trip");
        CHK(ts_ok, "GPU_RT_TIMESTAMPS", "timestamps mismatch after round-trip");
        INSPECT("GPU_ROUNDTRIP", "row_ptr=%s col_idx=%s timestamps=%s",
                rp_ok ? "OK" : "FAIL", ci_ok ? "OK" : "FAIL",
                ts_ok ? "OK" : "FAIL");

        ftg.free_gpu();
    }

    // ── Test 6: Edge cases ──
    {
        Timer t("csr_edge_cases");
        // Empty graph
        FlatSuccessorCSR csr_empty;
        csr_empty.build({}, 10);
        CHK(csr_empty.n_edges == 0, "CSR_EMPTY", "expected 0 edges");
        CHK(csr_empty.row_ptr.size() == 11, "CSR_EMPTY_RP", "expected 11 row_ptr entries");
        for (uint32_t v = 0; v < 10; ++v) {
            CHK(csr_empty.row_ptr[v] == 0, "CSR_EMPTY_RP_V",
                "v=%u row_ptr=%u", v, csr_empty.row_ptr[v]);
        }

        // Single vertex, many self-loops
        std::vector<TemEdge> self_edges;
        for (int i = 0; i < 100; ++i) {
            self_edges.push_back({0, 0, (int64_t)i * 10});
        }
        FlatSuccessorCSR csr_self;
        csr_self.build(self_edges, 1);
        CHK(csr_self.row_ptr[1] - csr_self.row_ptr[0] == 100, "CSR_SELF",
            "expected 100, got %u", csr_self.row_ptr[1] - csr_self.row_ptr[0]);
        int cnt = csr_self.range_query_cpu(0, 0, 990);
        CHK(cnt == 100, "CSR_SELF_RANGE", "expected 100, got %d", cnt);

        INSPECT("EDGE_CASES", "empty and self-loop tests passed");
    }

    sep("M083: exp_csr_build — COMPLETE");
}


// ════════════════════════════════════════════════════════════════════════
//  M084 §2: exp_temporal_range — GPU vs CPU range query comparison
// ════════════════════════════════════════════════════════════════════════

static void exp_temporal_range() {
    sep("M084: exp_temporal_range — GPU Temporal Range Query");

    // ── Generate temporal graph: 10K vertices, 100K edges ──
    const uint32_t NV = 10000, NE = 100000;
    const int64_t T_MIN = 0, T_MAX = 10000000;
    auto edges = gen_temporal_graph(NV, NE, T_MIN, T_MAX, 2024);

    FlatTemGraph ftg;
    ftg.build(edges, NV);
    ftg.alloc_gpu();

    // Also keep a CPU-side CSR for baseline
    FlatSuccessorCSR csr_cpu;
    csr_cpu.build(edges, NV);

    // ── Generate 10K random range queries ──
    const uint64_t NQ = 10000;
    std::mt19937_64 rng(3141);
    std::uniform_int_distribution<uint32_t> vdist(0, NV - 1);
    std::uniform_int_distribution<int64_t>  tdist(T_MIN, T_MAX);

    std::vector<uint32_t> q_verts(NQ);
    std::vector<int64_t>  q_t1(NQ), q_t2(NQ);
    for (uint64_t i = 0; i < NQ; ++i) {
        q_verts[i] = vdist(rng);
        int64_t a = tdist(rng), b = tdist(rng);
        q_t1[i] = std::min(a, b);
        q_t2[i] = std::max(a, b);
    }

    INSPECT("QUERIES", "generated %lu queries, vertex_range=[0,%u) t_range=[%ld,%ld]",
            (unsigned long)NQ, NV, (long)T_MIN, (long)T_MAX);

    // ── CPU baseline ──
    std::vector<int32_t> cpu_counts(NQ);
    std::vector<uint32_t> cpu_first(NQ);
    {
        Timer t("cpu_range_baseline");
        for (uint64_t i = 0; i < NQ; ++i) {
            uint32_t first_pos = UINT32_MAX;
            cpu_counts[i] = csr_cpu.range_query_cpu_full(
                q_verts[i], q_t1[i], q_t2[i], nullptr, &first_pos);
            cpu_first[i] = first_pos;
        }
    }

    // CPU stats
    int64_t total_matches = 0;
    uint32_t zero_count = 0;
    int32_t max_match = 0;
    for (uint64_t i = 0; i < NQ; ++i) {
        total_matches += cpu_counts[i];
        if (cpu_counts[i] == 0) zero_count++;
        max_match = std::max(max_match, cpu_counts[i]);
    }
    INSPECT("CPU_STATS", "total_matches=%ld zero_queries=%u max_match=%d avg_match=%.2f",
            (long)total_matches, zero_count, max_match,
            (double)total_matches / NQ);

    // ── GPU execution ──
    std::vector<int32_t>  gpu_counts(NQ);
    std::vector<uint32_t> gpu_first(NQ);
    {
        Timer t("gpu_range_query");

        // Allocate query arrays on GPU
        uint32_t *d_q_verts;
        int64_t  *d_q_t1, *d_q_t2;
        int32_t  *d_counts;
        uint32_t *d_first;

        GPU_CHECK(cudaMalloc(&d_q_verts, NQ * sizeof(uint32_t)));
        GPU_CHECK(cudaMalloc(&d_q_t1,    NQ * sizeof(int64_t)));
        GPU_CHECK(cudaMalloc(&d_q_t2,    NQ * sizeof(int64_t)));
        GPU_CHECK(cudaMalloc(&d_counts,  NQ * sizeof(int32_t)));
        GPU_CHECK(cudaMalloc(&d_first,   NQ * sizeof(uint32_t)));

        GPU_CHECK(cudaMemcpy(d_q_verts, q_verts.data(), NQ * sizeof(uint32_t), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_q_t1,    q_t1.data(),    NQ * sizeof(int64_t),  cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_q_t2,    q_t2.data(),    NQ * sizeof(int64_t),  cudaMemcpyHostToDevice));

        uint32_t block = 256;
        uint32_t grid  = (uint32_t)((NQ + block - 1) / block);

        INSPECT("GPU_LAUNCH", "kern_temporal_range_query grid=%u block=%u queries=%lu",
                grid, block, (unsigned long)NQ);

#if WALKING_CUDA
        kern_temporal_range_query<<<grid, block>>>(
            ftg.d_row_ptr, ftg.d_col_idx, ftg.d_timestamps, NV,
            d_q_verts, d_q_t1, d_q_t2, d_counts, d_first, NQ);
        GPU_CHECK(cudaDeviceSynchronize());
#else
        kern_temporal_range_query(
            ftg.h_row_ptr.data(), ftg.h_col_idx.data(), ftg.h_timestamps.data(), NV,
            q_verts.data(), q_t1.data(), q_t2.data(),
            gpu_counts.data(), gpu_first.data(), NQ);
#endif

        GPU_CHECK(cudaMemcpy(gpu_counts.data(), d_counts, NQ * sizeof(int32_t),  cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(gpu_first.data(),  d_first,  NQ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        GPU_CHECK(cudaFree(d_q_verts));
        GPU_CHECK(cudaFree(d_q_t1));
        GPU_CHECK(cudaFree(d_q_t2));
        GPU_CHECK(cudaFree(d_counts));
        GPU_CHECK(cudaFree(d_first));
    }

    // ── Verify GPU == CPU ──
    uint32_t count_match = 0, count_mismatch = 0;
    uint32_t first_match_ok = 0, first_mismatch = 0;
    for (uint64_t i = 0; i < NQ; ++i) {
        if (gpu_counts[i] == cpu_counts[i]) count_match++;
        else {
            count_mismatch++;
            if (count_mismatch <= 10) {
                std::printf("[RANGE·MISMATCH] q=%lu v=%u t=[%ld,%ld] gpu=%d cpu=%d\n",
                            (unsigned long)i, q_verts[i], (long)q_t1[i],
                            (long)q_t2[i], gpu_counts[i], cpu_counts[i]);
            }
        }
        if (gpu_first[i] == cpu_first[i]) first_match_ok++;
        else first_mismatch++;
    }
    CHK(count_mismatch == 0, "RANGE_GPU_CPU",
        "count match=%u mismatch=%u", count_match, count_mismatch);
    CHK(first_mismatch == 0, "RANGE_FIRST_GPU_CPU",
        "first_match ok=%u mismatch=%u", first_match_ok, first_mismatch);
    INSPECT("RANGE_VERIFY", "count: match=%u mismatch=%u  first: ok=%u mismatch=%u",
            count_match, count_mismatch, first_match_ok, first_mismatch);

    // ── Batch kernel test ──
    {
        Timer t("gpu_range_batch");

        std::vector<int32_t>  batch_counts(NQ);
        std::vector<uint32_t> batch_first(NQ);

        uint32_t *d_bv; int64_t *d_bt1, *d_bt2;
        int32_t *d_bc; uint32_t *d_bf;

        GPU_CHECK(cudaMalloc(&d_bv,  NQ * sizeof(uint32_t)));
        GPU_CHECK(cudaMalloc(&d_bt1, NQ * sizeof(int64_t)));
        GPU_CHECK(cudaMalloc(&d_bt2, NQ * sizeof(int64_t)));
        GPU_CHECK(cudaMalloc(&d_bc,  NQ * sizeof(int32_t)));
        GPU_CHECK(cudaMalloc(&d_bf,  NQ * sizeof(uint32_t)));

        GPU_CHECK(cudaMemcpy(d_bv,  q_verts.data(), NQ * sizeof(uint32_t), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_bt1, q_t1.data(),    NQ * sizeof(int64_t),  cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_bt2, q_t2.data(),    NQ * sizeof(int64_t),  cudaMemcpyHostToDevice));

        uint32_t block = 256;
        uint32_t grid  = (uint32_t)((NQ + block - 1) / block);

        INSPECT("BATCH_LAUNCH", "kern_temporal_range_batch grid=%u block=%u", grid, block);

#if WALKING_CUDA
        kern_temporal_range_batch<<<grid, block>>>(
            ftg.d_row_ptr, ftg.d_col_idx, ftg.d_timestamps, NV,
            d_bv, d_bt1, d_bt2, d_bc, d_bf, NQ);
        GPU_CHECK(cudaDeviceSynchronize());
#else
        kern_temporal_range_batch(
            ftg.h_row_ptr.data(), ftg.h_col_idx.data(), ftg.h_timestamps.data(), NV,
            q_verts.data(), q_t1.data(), q_t2.data(),
            batch_counts.data(), batch_first.data(), NQ);
#endif

        GPU_CHECK(cudaMemcpy(batch_counts.data(), d_bc, NQ * sizeof(int32_t),  cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(batch_first.data(),  d_bf, NQ * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Verify batch == single-query
        uint32_t batch_ok = 0;
        for (uint64_t i = 0; i < NQ; ++i) {
            if (batch_counts[i] == cpu_counts[i]) batch_ok++;
        }
        CHK(batch_ok == NQ, "BATCH_MATCH",
            "batch_ok=%u/%lu", batch_ok, (unsigned long)NQ);

        GPU_CHECK(cudaFree(d_bv));
        GPU_CHECK(cudaFree(d_bt1));
        GPU_CHECK(cudaFree(d_bt2));
        GPU_CHECK(cudaFree(d_bc));
        GPU_CHECK(cudaFree(d_bf));
    }

    // ── Query selectivity analysis ──
    {
        // Bucket queries by window width and analyze hit rates
        struct Bucket { const char* name; int64_t max_width; int count; int64_t total_hits; };
        Bucket buckets[] = {
            {"narrow (<1K)",    1000,        0, 0},
            {"medium (1K-100K)", 100000,     0, 0},
            {"wide (100K-1M)",  1000000,     0, 0},
            {"huge (>1M)",      INT64_MAX,   0, 0},
        };
        for (uint64_t i = 0; i < NQ; ++i) {
            int64_t w = q_t2[i] - q_t1[i];
            for (auto& b : buckets) {
                if (w <= b.max_width) {
                    b.count++;
                    b.total_hits += cpu_counts[i];
                    break;
                }
            }
        }
        std::printf("\n[SELECTIVITY TABLE]\n");
        std::printf("  %-20s  %8s  %12s  %10s\n", "Window", "Queries", "Total Hits", "Avg Hits");
        std::printf("  %-20s  %8s  %12s  %10s\n", "------", "-------", "----------", "--------");
        for (auto& b : buckets) {
            if (b.count > 0) {
                std::printf("  %-20s  %8d  %12ld  %10.2f\n",
                            b.name, b.count, (long)b.total_hits,
                            (double)b.total_hits / b.count);
            }
        }
        std::printf("\n");
    }

    ftg.free_gpu();
    sep("M084: exp_temporal_range — COMPLETE");
}


// ════════════════════════════════════════════════════════════════════════
//  M085 §2: exp_successor_walk — GPU vs CPU successor walk comparison
// ════════════════════════════════════════════════════════════════════════

static void exp_successor_walk() {
    sep("M085: exp_successor_walk — GPU Successor Walk");

    // ── Generate temporal graph ──
    // Use a graph where vertices have moderate out-degree so walks are non-trivial
    const uint32_t NV = 5000, NE = 50000;
    const int64_t T_MIN = 0, T_MAX = 1000000;
    auto edges = gen_temporal_graph(NV, NE, T_MIN, T_MAX, 6789);

    FlatTemGraph ftg;
    ftg.build(edges, NV);
    ftg.alloc_gpu();

    FlatSuccessorCSR csr_cpu;
    csr_cpu.build(edges, NV);

    // ── Generate random start vertices ──
    const uint32_t N_STARTS = 100;
    std::mt19937_64 rng(1111);
    std::vector<uint32_t> starts(N_STARTS);

    // Pick vertices with non-zero out-degree for more interesting walks
    std::vector<uint32_t> active_verts;
    for (uint32_t v = 0; v < NV; ++v) {
        if (csr_cpu.row_ptr[v + 1] - csr_cpu.row_ptr[v] > 0) {
            active_verts.push_back(v);
        }
    }
    INSPECT("ACTIVE_VERTS", "verts_with_edges=%lu / %u",
            active_verts.size(), NV);

    for (uint32_t i = 0; i < N_STARTS; ++i) {
        starts[i] = active_verts[rng() % active_verts.size()];
    }

    // ── Test multiple max_hops values ──
    uint32_t hop_configs[] = {4, 8, 16};

    for (uint32_t max_hops : hop_configs) {
        std::printf("\n── max_hops = %u ──\n", max_hops);

        // ── CPU baseline: unconstrained walk ──
        std::vector<std::vector<uint32_t>> cpu_paths(N_STARTS);
        {
            Timer t("cpu_walk_unconstrained");
            for (uint32_t i = 0; i < N_STARTS; ++i) {
                csr_cpu.walk_cpu(starts[i], max_hops, cpu_paths[i]);
            }
        }

        // CPU path length stats
        uint32_t cpu_total_len = 0, cpu_max_len = 0, cpu_zero_len = 0;
        for (uint32_t i = 0; i < N_STARTS; ++i) {
            uint32_t plen = (uint32_t)cpu_paths[i].size() - 1;  // exclude start
            cpu_total_len += plen;
            cpu_max_len = std::max(cpu_max_len, plen);
            if (plen == 0) cpu_zero_len++;
        }
        INSPECT("CPU_WALK", "max_hops=%u starts=%u avg_len=%.2f max_len=%u zero=%u",
                max_hops, N_STARTS,
                (double)cpu_total_len / N_STARTS, cpu_max_len, cpu_zero_len);

        // ── GPU unconstrained walk ──
        std::vector<uint32_t> gpu_paths_flat(N_STARTS * max_hops, UINT32_MAX);
        std::vector<uint32_t> gpu_path_lengths(N_STARTS, 0);
        {
            Timer t("gpu_walk_unconstrained");

            uint32_t *d_starts, *d_paths, *d_lengths;
            GPU_CHECK(cudaMalloc(&d_starts,  N_STARTS * sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_paths,   N_STARTS * max_hops * sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_lengths, N_STARTS * sizeof(uint32_t)));

            GPU_CHECK(cudaMemcpy(d_starts, starts.data(),
                      N_STARTS * sizeof(uint32_t), cudaMemcpyHostToDevice));

            uint32_t block = 128;
            uint32_t grid  = (N_STARTS + block - 1) / block;

            INSPECT("WALK_LAUNCH", "kern_successor_walk grid=%u block=%u starts=%u hops=%u",
                    grid, block, N_STARTS, max_hops);

#if WALKING_CUDA
            kern_successor_walk<<<grid, block>>>(
                ftg.d_row_ptr, ftg.d_col_idx, ftg.d_timestamps, NV,
                d_starts, d_paths, d_lengths, N_STARTS, max_hops);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            kern_successor_walk(
                ftg.h_row_ptr.data(), ftg.h_col_idx.data(), ftg.h_timestamps.data(), NV,
                starts.data(), gpu_paths_flat.data(), gpu_path_lengths.data(),
                N_STARTS, max_hops);
#endif

            GPU_CHECK(cudaMemcpy(gpu_paths_flat.data(), d_paths,
                      N_STARTS * max_hops * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(gpu_path_lengths.data(), d_lengths,
                      N_STARTS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            GPU_CHECK(cudaFree(d_starts));
            GPU_CHECK(cudaFree(d_paths));
            GPU_CHECK(cudaFree(d_lengths));
        }

        // ── Verify GPU == CPU paths ──
        uint32_t path_match = 0, path_mismatch = 0;
        uint32_t len_match = 0, len_mismatch = 0;
        for (uint32_t i = 0; i < N_STARTS; ++i) {
            uint32_t cpu_len = (uint32_t)cpu_paths[i].size() - 1;
            uint32_t gpu_len = gpu_path_lengths[i];

            if (cpu_len == gpu_len) {
                len_match++;
                // Compare path vertices
                bool all_ok = true;
                for (uint32_t s = 0; s < gpu_len; ++s) {
                    uint32_t gpu_v = gpu_paths_flat[i * max_hops + s];
                    uint32_t cpu_v = cpu_paths[i][s + 1]; // +1 because cpu_paths includes start
                    if (gpu_v != cpu_v) {
                        all_ok = false;
                        if (path_mismatch < 5) {
                            std::printf("[WALK·PATH_MISMATCH] start=%u step=%u gpu=%u cpu=%u\n",
                                        starts[i], s, gpu_v, cpu_v);
                        }
                        break;
                    }
                }
                if (all_ok) path_match++;
                else path_mismatch++;
            } else {
                len_mismatch++;
                if (len_mismatch <= 5) {
                    std::printf("[WALK·LEN_MISMATCH] start=%u cpu_len=%u gpu_len=%u\n",
                                starts[i], cpu_len, gpu_len);
                }
            }
        }
        CHK(len_mismatch == 0, "WALK_LEN",
            "hops=%u len_match=%u len_mismatch=%u",
            max_hops, len_match, len_mismatch);
        CHK(path_mismatch == 0, "WALK_PATH",
            "hops=%u path_match=%u path_mismatch=%u",
            max_hops, path_match, path_mismatch);
        INSPECT("WALK_VERIFY", "hops=%u len: ok=%u fail=%u  path: ok=%u fail=%u",
                max_hops, len_match, len_mismatch, path_match, path_mismatch);

        // ── Print sample walks ──
        if (g_dbg >= 2) {
            std::printf("[WALK·SAMPLES] first 5 walks (max_hops=%u):\n", max_hops);
            for (uint32_t i = 0; i < std::min(N_STARTS, 5u); ++i) {
                std::printf("  start=%u len=%u: %u",
                            starts[i], gpu_path_lengths[i], starts[i]);
                for (uint32_t s = 0; s < gpu_path_lengths[i]; ++s) {
                    std::printf(" → %u", gpu_paths_flat[i * max_hops + s]);
                }
                std::printf("\n");
            }
        }
    }

    // ── Time-constrained walk ──
    {
        std::printf("\n── Time-Constrained Walk ──\n");

        const uint32_t max_hops = 8;
        const int64_t t_start = T_MAX / 4;         // window: [25%, 75%] of time range
        const int64_t t_end   = 3 * T_MAX / 4;

        INSPECT("TIMED_WALK_PARAMS", "t_start=%ld t_end=%ld window=%ld",
                (long)t_start, (long)t_end, (long)(t_end - t_start));

        // CPU baseline: timed walk
        std::vector<std::vector<uint32_t>> cpu_timed_paths(N_STARTS);
        {
            Timer t("cpu_walk_timed");
            for (uint32_t i = 0; i < N_STARTS; ++i) {
                csr_cpu.walk_cpu(starts[i], max_hops, cpu_timed_paths[i],
                                 t_start, t_end);
            }
        }

        // GPU timed walk
        std::vector<uint32_t> gpu_tpaths(N_STARTS * max_hops, UINT32_MAX);
        std::vector<uint32_t> gpu_tlens(N_STARTS, 0);
        std::vector<int64_t>  gpu_tts(N_STARTS * max_hops, INT64_MIN);
        {
            Timer t("gpu_walk_timed");

            uint32_t *d_starts, *d_paths, *d_lengths;
            int64_t  *d_path_ts;
            GPU_CHECK(cudaMalloc(&d_starts,  N_STARTS * sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_paths,   N_STARTS * max_hops * sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_lengths, N_STARTS * sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_path_ts, N_STARTS * max_hops * sizeof(int64_t)));

            GPU_CHECK(cudaMemcpy(d_starts, starts.data(),
                      N_STARTS * sizeof(uint32_t), cudaMemcpyHostToDevice));

            uint32_t block = 128;
            uint32_t grid  = (N_STARTS + block - 1) / block;

            INSPECT("TIMED_LAUNCH", "kern_successor_walk_timed grid=%u block=%u",
                    grid, block);

#if WALKING_CUDA
            kern_successor_walk_timed<<<grid, block>>>(
                ftg.d_row_ptr, ftg.d_col_idx, ftg.d_timestamps, NV,
                d_starts, d_paths, d_lengths, d_path_ts,
                N_STARTS, max_hops, t_start, t_end);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            kern_successor_walk_timed(
                ftg.h_row_ptr.data(), ftg.h_col_idx.data(), ftg.h_timestamps.data(), NV,
                starts.data(), gpu_tpaths.data(), gpu_tlens.data(), gpu_tts.data(),
                N_STARTS, max_hops, t_start, t_end);
#endif

            GPU_CHECK(cudaMemcpy(gpu_tpaths.data(), d_paths,
                      N_STARTS * max_hops * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(gpu_tlens.data(), d_lengths,
                      N_STARTS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(gpu_tts.data(), d_path_ts,
                      N_STARTS * max_hops * sizeof(int64_t), cudaMemcpyDeviceToHost));

            GPU_CHECK(cudaFree(d_starts));
            GPU_CHECK(cudaFree(d_paths));
            GPU_CHECK(cudaFree(d_lengths));
            GPU_CHECK(cudaFree(d_path_ts));
        }

        // Verify timed walk GPU == CPU
        uint32_t timed_len_ok = 0, timed_path_ok = 0;
        uint32_t timed_len_fail = 0, timed_path_fail = 0;
        for (uint32_t i = 0; i < N_STARTS; ++i) {
            uint32_t cpu_len = (uint32_t)cpu_timed_paths[i].size() - 1;
            uint32_t gpu_len = gpu_tlens[i];

            if (cpu_len == gpu_len) {
                timed_len_ok++;
                bool ok = true;
                for (uint32_t s = 0; s < gpu_len; ++s) {
                    if (gpu_tpaths[i * max_hops + s] != cpu_timed_paths[i][s + 1]) {
                        ok = false;
                        if (timed_path_fail < 5) {
                            std::printf("[TIMED·PATH_MISMATCH] start=%u step=%u "
                                        "gpu=%u cpu=%u\n",
                                        starts[i], s,
                                        gpu_tpaths[i * max_hops + s],
                                        cpu_timed_paths[i][s + 1]);
                        }
                        break;
                    }
                }
                if (ok) timed_path_ok++;
                else timed_path_fail++;
            } else {
                timed_len_fail++;
                if (timed_len_fail <= 5) {
                    std::printf("[TIMED·LEN_MISMATCH] start=%u cpu=%u gpu=%u\n",
                                starts[i], cpu_len, gpu_len);
                }
            }
        }
        CHK(timed_len_fail == 0, "TIMED_LEN",
            "ok=%u fail=%u", timed_len_ok, timed_len_fail);
        CHK(timed_path_fail == 0, "TIMED_PATH",
            "ok=%u fail=%u", timed_path_ok, timed_path_fail);
        INSPECT("TIMED_VERIFY", "len: ok=%u fail=%u  path: ok=%u fail=%u",
                timed_len_ok, timed_len_fail, timed_path_ok, timed_path_fail);

        // Verify all timestamps in paths are within [t_start, t_end]
        uint32_t ts_violations = 0;
        for (uint32_t i = 0; i < N_STARTS; ++i) {
            for (uint32_t s = 0; s < gpu_tlens[i]; ++s) {
                int64_t ts = gpu_tts[i * max_hops + s];
                if (ts < t_start || ts > t_end) {
                    ts_violations++;
                    if (ts_violations <= 5) {
                        std::printf("[TIMED·TS_VIOLATION] start=%u step=%u ts=%ld "
                                    "window=[%ld,%ld]\n",
                                    starts[i], s, (long)ts, (long)t_start, (long)t_end);
                    }
                }
            }
        }
        CHK(ts_violations == 0, "TIMED_TS_BOUNDS",
            "violations=%u", ts_violations);

        // Timed walk length distribution
        uint32_t len_hist[9] = {};    // 0..8
        for (uint32_t i = 0; i < N_STARTS; ++i) {
            uint32_t l = std::min(gpu_tlens[i], (uint32_t)8);
            len_hist[l]++;
        }
        std::printf("\n[TIMED WALK LENGTH DISTRIBUTION] (max_hops=%u)\n", max_hops);
        for (uint32_t l = 0; l <= max_hops; ++l) {
            std::printf("  len=%u: %u starts", l, len_hist[l]);
            if (N_STARTS > 0) {
                std::printf(" (%.1f%%)", 100.0 * len_hist[l] / N_STARTS);
            }
            std::printf("\n");
        }

        // Print sample timed walks
        if (g_dbg >= 2) {
            std::printf("[TIMED·SAMPLES] first 5 timed walks:\n");
            for (uint32_t i = 0; i < std::min(N_STARTS, 5u); ++i) {
                std::printf("  start=%u len=%u:", starts[i], gpu_tlens[i]);
                std::printf(" %u", starts[i]);
                for (uint32_t s = 0; s < gpu_tlens[i]; ++s) {
                    std::printf(" →%u@%ld", gpu_tpaths[i * max_hops + s],
                                (long)gpu_tts[i * max_hops + s]);
                }
                std::printf("\n");
            }
        }
    }

    // ── Path length distribution across all hop configs ──
    {
        std::printf("\n[PATH LENGTH SUMMARY TABLE]\n");
        std::printf("  %-10s  %8s  %8s  %8s  %8s\n",
                    "max_hops", "avg_len", "max_len", "zero_ct", "full_ct");
        std::printf("  %-10s  %8s  %8s  %8s  %8s\n",
                    "--------", "-------", "-------", "-------", "-------");

        for (uint32_t max_hops : {4u, 8u, 16u}) {
            std::vector<std::vector<uint32_t>> paths(N_STARTS);
            uint32_t total = 0, maxl = 0, zero = 0, full = 0;
            for (uint32_t i = 0; i < N_STARTS; ++i) {
                csr_cpu.walk_cpu(starts[i], max_hops, paths[i]);
                uint32_t l = (uint32_t)paths[i].size() - 1;
                total += l;
                maxl = std::max(maxl, l);
                if (l == 0) zero++;
                if (l == max_hops) full++;
            }
            std::printf("  %-10u  %8.2f  %8u  %8u  %8u\n",
                        max_hops, (double)total / N_STARTS, maxl, zero, full);
        }
        std::printf("\n");
    }

    ftg.free_gpu();
    sep("M085: exp_successor_walk — COMPLETE");
}


}  // namespace temgraph
}  // namespace walking


// ════════════════════════════════════════════════════════════════════════
//  main
// ════════════════════════════════════════════════════════════════════════

using namespace walking::temgraph;

int main() {
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  walking_temgraph_gpu — M083/M084/M085                  ║\n");
    std::printf("║  TemGraph GPU Temporal Queries                          ║\n");
    std::printf("║  Milestones: CSR Build · Range Query · Successor Walk   ║\n");
#if WALKING_CUDA
    std::printf("║  Mode: GPU (CUDA)                                       ║\n");
#else
    std::printf("║  Mode: CPU (no CUDA)                                    ║\n");
#endif
    std::printf("╚══════════════════════════════════════════════════════════╝\n\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    exp_csr_build();          // M083: CSR construction + validation
    exp_temporal_range();     // M084: GPU temporal range query
    exp_successor_walk();     // M085: GPU successor walk

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    sep("SUMMARY");
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║  RESULTS                                                ║\n");
    std::printf("╠══════════════════════════════════════════════════════════╣\n");
    std::printf("║  Inspections : %6lu                                    ║\n", (unsigned long)g_insp);
    std::printf("║  CHK pass    : %6lu                                    ║\n", (unsigned long)g_pass);
    std::printf("║  CHK fail    : %6lu                                    ║\n", (unsigned long)g_fail);
    std::printf("║  Total time  : %8.2f ms                               ║\n", total_ms);
    std::printf("║  Peak RSS    : %6ld KB                                 ║\n", rss_kb());
    std::printf("╠══════════════════════════════════════════════════════════╣\n");
    if (g_fail == 0) {
        std::printf("║  STATUS: ALL PASS ✓                                     ║\n");
    } else {
        std::printf("║  STATUS: %lu FAILURES — see [FAIL·*] above              ║\n",
                    (unsigned long)g_fail);
    }
    std::printf("╚══════════════════════════════════════════════════════════╝\n");

    return (g_fail == 0) ? 0 : 1;
}
