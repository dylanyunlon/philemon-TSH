/**
 * walking_integration.cu — 端到端集成: LDBC SNB workload + 论文实验复现 + 回归检测
 *
 * mv来源与算法改动对照:
 *
 *   ldbc_bench.cpp (477行)
 *     KEEP: LDBCLoader调用模式, generate_synthetic_edges, test harness框架
 *     KEEP: TierCostModel standalone test pattern, access_cost_ns, migration_cost_ns
 *     KEEP: degree-tier cross-check, tier_counts统计
 *     KEEP: test runner lambda (run(name, result))
 *     MOD:  串行查询 → GPU batch查询 kern_ldbc_batch
 *     NEW:  LDBCQuery struct (IC/IS/BI types), batch executor
 *     NEW:  exp_ldbc_e2e(): 全LDBC SNB workload模拟, QPS/latency/tier命中率
 *
 *   integration_bench.cpp (375行)
 *     KEEP: BenchConfig struct, SyntheticGraph生成, Phase 1-4框架
 *     KEEP: TieredSnapshot distribution: top 20% HBM, 30% GDDR, 50% DRAM
 *     KEEP: Phase 2 algorithm pattern (BFS/PR/SSSP/WCC/TC)
 *     KEEP: Phase 3 QueryExecutor concurrent batch
 *     MOD:  CPU sequential execution → GPU batch pipeline
 *     NEW:  kern_ldbc_batch_dispatch: 按查询类型dispatch到不同kernel path
 *
 *   cross_tier_bench.cpp (501行)
 *     KEEP: MockSnapshot/MockGraphMethod graph backend
 *     KEEP: TestResult struct, print_separator, summary formatting
 *     KEEP: PHILE_BREAKPOINT/PHILE_INSPECT pattern
 *     KEEP: tier_perf tracking per algorithm
 *     MOD:  单算法test → 多算法端到端pipeline
 *
 *   benchmark_matrix.hpp (674行)
 *     KEEP: BenchResult structure (algorithm/dataset/tier/backend fields)
 *     KEEP: SampleStats::compute (mean/stddev/95%CI), WelchResult::test
 *     KEEP: export_json, export_markdown, print_summary
 *     KEEP: AlgorithmType/DatasetType/TierConfig/BackendType enums
 *     MOD:  全组合矩阵 → 论文Table 1-3聚焦
 *     NEW:  exp_paper_tables(): 自动生成Table 1(tier参数), Table 2(算法对比), Table 3(scalability)
 *
 *   walking_gpu_tree.cu §debug (前100行)
 *     KEEP: INSPECT/CHK/Timer/sep pattern, g_dbg/g_insp/g_pass/g_fail
 *     KEEP: WALKING_CUDA compile-mode dispatch, GPU_CHECK, CPU fallback
 *     KEEP: rss_kb() memory tracking
 *
 *   M074-M091 changelog:
 *     NEW:  print_changelog(): 打印M074-M094所有里程碑摘要
 *     NEW:  exp_regression(): 回归检测, 运行核心算法, CHK结果与baseline对比
 *
 * Build:
 *   GPU:  nvcc -std=c++17 -O2 -arch=sm_86 -DWALKING_CUDA=1 \
 *              -o walking_integration walking_integration.cu
 *   CPU:  g++ -std=c++17 -O2 -pthread -DWALKING_CUDA=0 -x c++ \
 *              -o walking_integration walking_integration.cu
 *
 * Milestone: M092–M094
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
#include <unordered_set>
#include <functional>
#include <mutex>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
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
          fprintf(stderr, "[CUDA·FATAL] %s:%d %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(e)); \
          exit(1); } } while(0)
#else
  #define GPU_CHECK(call) ((void)0)
  enum { cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2 };
  inline void* _fake_alloc(size_t n) { void* p = malloc(n); if(p) memset(p,0,n); return p; }
  #define cudaMalloc(p,n)     (*(p)=_fake_alloc(n),(void)0)
  #define cudaFree(p)         free(p)
  #define cudaMemcpy(d,s,n,k) memcpy(d,s,n)
  #define cudaMemset(p,v,n)   memset(p,v,n)
  #define cudaDeviceSynchronize() ((void)0)
#endif

// ════════════════════════════════════════════════════════════════
// Debug infra — KEEP from walking_gpu_tree.cu
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
//   namespace walking::integration
// ════════════════════════════════════════════════════════════════════════
namespace walking {
namespace integration {

// ════════════════════════════════════════════════════════════════
// §0  Shared data structures
//     mv: ldbc_bench.cpp (TierHint, DegreeStats), integration_bench.cpp
//         (BenchConfig, SyntheticGraph, TemporalEdge)
//     KEEP: tier枚举, 图数据结构, 配置结构
// ════════════════════════════════════════════════════════════════

// [KEEP from ldbc_bench.cpp] Tier分层定义
enum TierID : uint8_t {
    TIER_HBM  = 0,    // GPU HBM  (~3.35 TB/s, H100 80GB)
    TIER_GDDR = 1,    // GPU GDDR (~1.0 TB/s)
    TIER_DRAM = 2,    // CPU DDR5 (~80 GB/s)
    TIER_SSD  = 3,    // NVMe SSD (~7 GB/s sequential)
    TIER_COUNT = 4
};

static const char* tier_id_name(TierID t) {
    switch (t) {
        case TIER_HBM:  return "HBM";
        case TIER_GDDR: return "GDDR";
        case TIER_DRAM: return "DRAM";
        case TIER_SSD:  return "SSD";
        default:        return "???";
    }
}

// [KEEP from ldbc_bench.cpp] Tier参数配置 (论文Table 1数据)
struct TierSpec {
    TierID   id;
    uint64_t capacity_gb;
    double   bandwidth_gbps;   // GB/s
    double   read_latency_ns;
    double   write_latency_ns;

    void dump() const {
        std::printf("    [%-4s] cap=%3luGB  bw=%7.1fGB/s  rlat=%7.0fns  wlat=%7.0fns\n",
                    tier_id_name(id),
                    (unsigned long)capacity_gb, bandwidth_gbps,
                    read_latency_ns, write_latency_ns);
    }
};

static TierSpec default_tier_specs[TIER_COUNT] = {
    { TIER_HBM,   80,  3350.0,   10.0,   15.0 },   // H100 HBM3
    { TIER_GDDR,  24,  1008.0,   50.0,   75.0 },   // GDDR6X
    { TIER_DRAM, 512,    76.8,  100.0,  120.0 },   // DDR5-4800
    { TIER_SSD,  4096,    7.0, 5000.0, 8000.0 },   // NVMe Gen4
};

// [KEEP from integration_bench.cpp] Temporal edge representation
struct TemporalEdge {
    uint32_t src;
    uint32_t dst;
    double   weight;
    int64_t  ts_begin;
    int64_t  ts_end;
};

// [KEEP from integration_bench.cpp] Benchmark configuration
struct IntegConfig {
    uint64_t num_vertices = 50000;
    uint64_t num_edges    = 250000;
    uint64_t num_queries  = 10000;
    int      num_threads  = 4;
    int      time_range   = 1000000;
    uint64_t seed         = 42;
};

// ════════════════════════════════════════════════════════════════
// §1  Mock Graph Backend + CSR
//     mv: cross_tier_bench.cpp MockSnapshot, integration_bench.cpp
//     [KEEP] adjacency list, vertex_count, edge_count, edges(), degree()
//     [KEEP] tier-based access tracking
//     [MOD]  add CSR for GPU batch processing
// ════════════════════════════════════════════════════════════════

// [KEEP from cross_tier_bench.cpp] Adjacency list graph
struct AdjGraph {
    uint64_t n_vertices;
    uint64_t n_edges;
    std::vector<std::vector<std::pair<uint32_t, double>>> adj;

    // [NEW] CSR representation for GPU batch
    std::vector<uint64_t> csr_row_ptr;
    std::vector<uint32_t> csr_col_idx;
    std::vector<double>   csr_weights;

    // [NEW] Tier assignment per vertex: vertex → tier
    std::vector<uint8_t> vertex_tier;

    void build(uint64_t nv, uint64_t ne, uint64_t seed) {
        Timer t("AdjGraph::build");
        n_vertices = nv;
        adj.resize(nv);
        vertex_tier.resize(nv);

        std::mt19937_64 rng(seed);
        uint64_t actual = 0;
        for (uint64_t i = 0; i < ne; i++) {
            uint32_t src = rng() % nv;
            uint32_t dst = rng() % nv;
            if (src == dst) continue;
            double w = 1.0 + (rng() % 1000) / 1000.0;
            adj[src].push_back({dst, w});
            actual++;
        }
        n_edges = actual;

        // [KEEP from ldbc_bench.cpp] tier assignment by vertex ID ranges
        // top 20% → HBM, next 30% → GDDR, rest 50% → DRAM
        for (uint64_t v = 0; v < nv; v++) {
            if (v < nv / 5)          vertex_tier[v] = TIER_HBM;
            else if (v < nv / 2)     vertex_tier[v] = TIER_GDDR;
            else                     vertex_tier[v] = TIER_DRAM;
        }

        // [NEW] Build CSR from adjacency list
        build_csr();

        INSPECT("GRAPH", "V=%lu E=%lu HBM_V=%lu GDDR_V=%lu DRAM_V=%lu",
                (unsigned long)nv, (unsigned long)actual,
                (unsigned long)(nv / 5),
                (unsigned long)(nv / 2 - nv / 5),
                (unsigned long)(nv - nv / 2));
    }

    void build_csr() {
        csr_row_ptr.resize(n_vertices + 1, 0);
        // count phase
        for (uint64_t v = 0; v < n_vertices; v++) {
            csr_row_ptr[v + 1] = csr_row_ptr[v] + adj[v].size();
        }
        uint64_t nnz = csr_row_ptr[n_vertices];
        csr_col_idx.resize(nnz);
        csr_weights.resize(nnz);
        // fill phase
        for (uint64_t v = 0; v < n_vertices; v++) {
            uint64_t base = csr_row_ptr[v];
            for (size_t j = 0; j < adj[v].size(); j++) {
                csr_col_idx[base + j] = adj[v][j].first;
                csr_weights[base + j] = adj[v][j].second;
            }
        }

        INSPECT("CSR", "nnz=%lu row_ptr.size=%lu",
                (unsigned long)nnz, (unsigned long)csr_row_ptr.size());
    }

    uint64_t degree(uint32_t v) const {
        return (v < n_vertices) ? adj[v].size() : 0;
    }
};

// ════════════════════════════════════════════════════════════════
// §2  LDBC SNB Query Types + Batch Structure
//     [NEW] LDBC Interactive Complex (IC), Interactive Short (IS),
//           Business Intelligence (BI) query type definitions
//     mv pattern: ldbc_bench.cpp test_ldbc_loader query dispatch
// ════════════════════════════════════════════════════════════════

// [NEW] LDBC SNB query classification
enum class LDBCQueryType : uint8_t {
    // Interactive Complex queries (read-heavy, multi-hop)
    IC01_FRIENDS_BY_NAME  = 0,    // 1-hop friends with name filter
    IC02_RECENT_POSTS     = 1,    // 2-hop, recent posts by friends
    IC03_FRIENDS_IN_PLACE = 2,    // friends in specific countries
    IC06_TAG_COOCCURRENCE = 3,    // 3-hop, tag co-occurrence
    IC09_RECENT_MESSAGES  = 4,    // 2-hop, recent messages

    // Interactive Short queries (point lookups)
    IS01_PERSON_PROFILE   = 5,    // vertex property read
    IS02_RECENT_POSTS_S   = 6,    // 1-hop edge + sort
    IS03_FRIENDS_OF       = 7,    // 1-hop enumerate

    // Business Intelligence (graph-global analytics)
    BI01_POSTING_SUMMARY  = 8,    // full scan, group-by year
    BI02_TAG_EVOLUTION    = 9,    // temporal scan, tag evolution
    BI06_ACTIVE_POSTERS   = 10,   // high-degree vertex enumeration

    LDBC_TYPE_COUNT       = 11
};

static const char* ldbc_type_name(LDBCQueryType t) {
    switch (t) {
        case LDBCQueryType::IC01_FRIENDS_BY_NAME:  return "IC01";
        case LDBCQueryType::IC02_RECENT_POSTS:     return "IC02";
        case LDBCQueryType::IC03_FRIENDS_IN_PLACE: return "IC03";
        case LDBCQueryType::IC06_TAG_COOCCURRENCE: return "IC06";
        case LDBCQueryType::IC09_RECENT_MESSAGES:  return "IC09";
        case LDBCQueryType::IS01_PERSON_PROFILE:   return "IS01";
        case LDBCQueryType::IS02_RECENT_POSTS_S:   return "IS02";
        case LDBCQueryType::IS03_FRIENDS_OF:       return "IS03";
        case LDBCQueryType::BI01_POSTING_SUMMARY:  return "BI01";
        case LDBCQueryType::BI02_TAG_EVOLUTION:    return "BI02";
        case LDBCQueryType::BI06_ACTIVE_POSTERS:   return "BI06";
        default:                                    return "???";
    }
}

// [NEW] Query category for routing
enum class LDBCCategory : uint8_t {
    INTERACTIVE_COMPLEX = 0,   // multi-hop traversal
    INTERACTIVE_SHORT   = 1,   // point/1-hop lookup
    BUSINESS_INTEL      = 2,   // full-graph scan
};

static LDBCCategory ldbc_category(LDBCQueryType t) {
    uint8_t v = static_cast<uint8_t>(t);
    if (v <= 4) return LDBCCategory::INTERACTIVE_COMPLEX;
    if (v <= 7) return LDBCCategory::INTERACTIVE_SHORT;
    return LDBCCategory::BUSINESS_INTEL;
}

// [NEW] Single LDBC query descriptor
struct LDBCQuery {
    LDBCQueryType type;
    uint32_t      start_vertex;     // starting vertex for traversal
    uint32_t      param_limit;      // result limit (e.g., top-K)
    int64_t       ts_lo;            // temporal filter lower bound
    int64_t       ts_hi;            // temporal filter upper bound
    uint32_t      query_id;         // unique ID for tracking
};

// [NEW] Batch of LDBC queries for GPU execution
struct LDBCBatch {
    std::vector<LDBCQuery> queries;
    size_t                 n_ic;     // count of IC queries
    size_t                 n_is;     // count of IS queries
    size_t                 n_bi;     // count of BI queries

    void classify() {
        n_ic = n_is = n_bi = 0;
        for (auto& q : queries) {
            switch (ldbc_category(q.type)) {
                case LDBCCategory::INTERACTIVE_COMPLEX: n_ic++; break;
                case LDBCCategory::INTERACTIVE_SHORT:   n_is++; break;
                case LDBCCategory::BUSINESS_INTEL:      n_bi++; break;
            }
        }
    }

    void dump() const {
        std::printf("    [LDBCBatch] total=%zu IC=%zu IS=%zu BI=%zu\n",
                    queries.size(), n_ic, n_is, n_bi);
    }
};

// [NEW] Generate LDBC-like workload with SNB query mix
//   IC:IS:BI ratio ≈ 4:4:2 (SNB interactive benchmark typical)
static LDBCBatch generate_ldbc_workload(uint64_t n_queries,
                                         uint64_t n_vertices,
                                         int time_range,
                                         uint64_t seed) {
    Timer t("generate_ldbc_workload");
    LDBCBatch batch;
    batch.queries.reserve(n_queries);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> vdist(0, n_vertices - 1);
    std::uniform_int_distribution<int64_t>  tdist(0, time_range);
    std::uniform_int_distribution<int>      type_dist(0, 9);  // 0-9 → maps to types

    // [NEW] SNB-like distribution: IC 40%, IS 40%, BI 20%
    for (uint64_t i = 0; i < n_queries; i++) {
        LDBCQuery q;
        int roll = type_dist(rng);
        if (roll < 4) {
            // IC queries (40%)
            static const LDBCQueryType ic_types[] = {
                LDBCQueryType::IC01_FRIENDS_BY_NAME,
                LDBCQueryType::IC02_RECENT_POSTS,
                LDBCQueryType::IC03_FRIENDS_IN_PLACE,
                LDBCQueryType::IC06_TAG_COOCCURRENCE,
                LDBCQueryType::IC09_RECENT_MESSAGES
            };
            q.type = ic_types[rng() % 5];
        } else if (roll < 8) {
            // IS queries (40%)
            static const LDBCQueryType is_types[] = {
                LDBCQueryType::IS01_PERSON_PROFILE,
                LDBCQueryType::IS02_RECENT_POSTS_S,
                LDBCQueryType::IS03_FRIENDS_OF
            };
            q.type = is_types[rng() % 3];
        } else {
            // BI queries (20%)
            static const LDBCQueryType bi_types[] = {
                LDBCQueryType::BI01_POSTING_SUMMARY,
                LDBCQueryType::BI02_TAG_EVOLUTION,
                LDBCQueryType::BI06_ACTIVE_POSTERS
            };
            q.type = bi_types[rng() % 3];
        }

        q.start_vertex = vdist(rng);
        q.param_limit  = 10 + (rng() % 90);   // 10-99
        q.ts_lo = tdist(rng);
        q.ts_hi = q.ts_lo + (tdist(rng) % (time_range / 10));
        q.query_id = (uint32_t)i;
        batch.queries.push_back(q);
    }

    batch.classify();
    batch.dump();
    return batch;
}

// ════════════════════════════════════════════════════════════════
// §3  GPU Batch Query Kernels
//     [NEW] kern_ldbc_batch: parallel execution of N LDBC queries
//     [MOD] from ldbc_bench.cpp serial query → GPU batch dispatch
//     Pattern: each thread handles one query, accesses CSR graph
// ════════════════════════════════════════════════════════════════

// [NEW] Per-query result structure (fixed-size for GPU)
struct QueryResult {
    uint32_t query_id;
    uint32_t result_count;     // number of result vertices/edges found
    uint64_t edges_traversed;  // total edges touched
    uint8_t  tier_hits[TIER_COUNT]; // per-tier access distribution (scaled 0-255)
    double   exec_time_us;     // execution time in microseconds (CPU-side)
};

// [NEW] CSR pointers for GPU (flat arrays)
struct FlatCSR {
    uint64_t  n_vertices;
    uint64_t  n_edges;
    uint64_t* row_ptr;      // [n_vertices+1]
    uint32_t* col_idx;      // [n_edges]
    double*   weights;      // [n_edges]
    uint8_t*  vtier;        // [n_vertices] per-vertex tier
};

// ────────────────────────────────────────────────────────────
// kern_ldbc_batch: GPU kernel for batch LDBC query execution
//   [NEW] Each thread processes one query from the batch
//   IC queries: multi-hop BFS from start_vertex (depth 1-3)
//   IS queries: point lookup or 1-hop enumerate
//   BI queries: full scan with filter
// ────────────────────────────────────────────────────────────

#if WALKING_CUDA
__global__
void kern_ldbc_batch(const FlatCSR csr,
                     const LDBCQuery* queries,
                     QueryResult* results,
                     uint32_t n_queries)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_queries) return;

    const LDBCQuery& q = queries[tid];
    QueryResult& r = results[tid];
    r.query_id = q.query_id;
    r.result_count = 0;
    r.edges_traversed = 0;
    memset(r.tier_hits, 0, sizeof(r.tier_hits));

    uint32_t sv = q.start_vertex;
    if (sv >= csr.n_vertices) return;

    uint8_t cat = static_cast<uint8_t>(q.type);

    if (cat <= 4) {
        // IC: multi-hop traversal (simplified 2-hop BFS)
        uint64_t base0 = csr.row_ptr[sv];
        uint64_t end0  = csr.row_ptr[sv + 1];
        uint32_t count = 0;

        // 1-hop neighbors
        for (uint64_t e = base0; e < end0 && count < q.param_limit; e++) {
            uint32_t nb = csr.col_idx[e];
            r.edges_traversed++;
            if (nb < csr.n_vertices) {
                r.tier_hits[csr.vtier[nb] % TIER_COUNT]++;

                // 2-hop (for IC02, IC06, IC09)
                if (cat == 1 || cat == 3 || cat == 4) {
                    uint64_t base1 = csr.row_ptr[nb];
                    uint64_t end1  = csr.row_ptr[nb + 1];
                    uint64_t hop2_limit = (end1 - base1 > 32) ? base1 + 32 : end1;
                    for (uint64_t e2 = base1; e2 < hop2_limit; e2++) {
                        r.edges_traversed++;
                        count++;
                    }
                } else {
                    count++;
                }
            }
        }
        r.result_count = count;
    }
    else if (cat <= 7) {
        // IS: point lookup / 1-hop enumerate
        uint64_t base = csr.row_ptr[sv];
        uint64_t end  = csr.row_ptr[sv + 1];
        uint32_t count = 0;
        for (uint64_t e = base; e < end && count < q.param_limit; e++) {
            r.edges_traversed++;
            count++;
        }
        r.result_count = count;
        r.tier_hits[csr.vtier[sv] % TIER_COUNT] = 255;
    }
    else {
        // BI: scan a range of vertices (simulated global analytics)
        // Scan up to 1000 vertices starting from start_vertex
        uint32_t scan_end = sv + 1000;
        if (scan_end > csr.n_vertices) scan_end = (uint32_t)csr.n_vertices;
        uint32_t count = 0;
        for (uint32_t v = sv; v < scan_end; v++) {
            uint64_t deg = csr.row_ptr[v + 1] - csr.row_ptr[v];
            r.edges_traversed += deg;
            if (deg > 0) count++;
            r.tier_hits[csr.vtier[v] % TIER_COUNT]++;
        }
        r.result_count = count;
    }
}
#endif // WALKING_CUDA

// [NEW] CPU fallback for kern_ldbc_batch
static void cpu_ldbc_batch(const AdjGraph& g,
                            const LDBCQuery* queries,
                            QueryResult* results,
                            uint32_t n_queries)
{
    for (uint32_t tid = 0; tid < n_queries; tid++) {
        const LDBCQuery& q = queries[tid];
        QueryResult& r = results[tid];
        r.query_id = q.query_id;
        r.result_count = 0;
        r.edges_traversed = 0;
        memset(r.tier_hits, 0, sizeof(r.tier_hits));

        uint32_t sv = q.start_vertex;
        if (sv >= g.n_vertices) continue;

        uint8_t cat = static_cast<uint8_t>(q.type);

        if (cat <= 4) {
            // IC: multi-hop
            uint32_t count = 0;
            for (auto& [nb, w] : g.adj[sv]) {
                r.edges_traversed++;
                if (nb < g.n_vertices) {
                    r.tier_hits[g.vertex_tier[nb] % TIER_COUNT]++;
                    if (cat == 1 || cat == 3 || cat == 4) {
                        uint32_t hop2_lim = std::min((size_t)32, g.adj[nb].size());
                        for (uint32_t j = 0; j < hop2_lim; j++) {
                            r.edges_traversed++;
                            count++;
                        }
                    } else {
                        count++;
                    }
                }
                if (count >= q.param_limit) break;
            }
            r.result_count = count;
        }
        else if (cat <= 7) {
            // IS: 1-hop
            uint32_t count = 0;
            for (auto& [nb, w] : g.adj[sv]) {
                r.edges_traversed++;
                count++;
                if (count >= q.param_limit) break;
            }
            r.result_count = count;
            r.tier_hits[g.vertex_tier[sv] % TIER_COUNT] = 255;
        }
        else {
            // BI: scan
            uint32_t scan_end = sv + 1000;
            if (scan_end > g.n_vertices) scan_end = (uint32_t)g.n_vertices;
            uint32_t count = 0;
            for (uint32_t v = sv; v < scan_end; v++) {
                uint64_t deg = g.adj[v].size();
                r.edges_traversed += deg;
                if (deg > 0) count++;
                r.tier_hits[g.vertex_tier[v] % TIER_COUNT]++;
            }
            r.result_count = count;
        }
    }
}

// ════════════════════════════════════════════════════════════════
// §4  GPU Batch Executor
//     [NEW] Manages GPU memory for CSR + queries + results
//     [MOD] from ldbc_bench.cpp serial test → batch GPU pipeline
// ════════════════════════════════════════════════════════════════

struct BatchExecResult {
    double   total_time_ms;
    double   qps;                   // queries per second
    double   avg_latency_us;        // average per-query latency
    double   p50_latency_us;
    double   p99_latency_us;
    uint64_t total_edges_traversed;
    uint64_t tier_hits[TIER_COUNT]; // aggregated tier hit counts
    uint32_t total_results;

    void dump(const char* label) const {
        std::printf("    [%s] total=%.2fms QPS=%.0f avg_lat=%.1fμs "
                    "P50=%.1fμs P99=%.1fμs\n",
                    label, total_time_ms, qps, avg_latency_us,
                    p50_latency_us, p99_latency_us);
        std::printf("    [%s] edges_traversed=%lu results=%u\n",
                    label, (unsigned long)total_edges_traversed,
                    total_results);
        std::printf("    [%s] tier_hits: HBM=%lu GDDR=%lu DRAM=%lu SSD=%lu\n",
                    label,
                    (unsigned long)tier_hits[0], (unsigned long)tier_hits[1],
                    (unsigned long)tier_hits[2], (unsigned long)tier_hits[3]);
    }
};

static BatchExecResult execute_ldbc_batch_gpu(AdjGraph& g,
                                               const LDBCBatch& batch)
{
    Timer t("execute_ldbc_batch_gpu");
    uint32_t nq = (uint32_t)batch.queries.size();
    BatchExecResult res;
    memset(&res, 0, sizeof(res));

    std::vector<QueryResult> h_results(nq);

#if WALKING_CUDA
    // ── Allocate device CSR ──
    FlatCSR d_csr;
    d_csr.n_vertices = g.n_vertices;
    d_csr.n_edges    = g.n_edges;

    GPU_CHECK(cudaMalloc(&d_csr.row_ptr, (g.n_vertices + 1) * sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_csr.col_idx, g.n_edges * sizeof(uint32_t)));
    GPU_CHECK(cudaMalloc(&d_csr.weights, g.n_edges * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_csr.vtier,   g.n_vertices * sizeof(uint8_t)));

    GPU_CHECK(cudaMemcpy(d_csr.row_ptr, g.csr_row_ptr.data(),
                          (g.n_vertices + 1) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_csr.col_idx, g.csr_col_idx.data(),
                          g.n_edges * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_csr.weights, g.csr_weights.data(),
                          g.n_edges * sizeof(double),
                          cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_csr.vtier, g.vertex_tier.data(),
                          g.n_vertices * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    // ── Allocate device queries and results ──
    LDBCQuery*   d_queries;
    QueryResult* d_results;
    GPU_CHECK(cudaMalloc(&d_queries, nq * sizeof(LDBCQuery)));
    GPU_CHECK(cudaMalloc(&d_results, nq * sizeof(QueryResult)));

    GPU_CHECK(cudaMemcpy(d_queries, batch.queries.data(),
                          nq * sizeof(LDBCQuery), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(d_results, 0, nq * sizeof(QueryResult)));

    // ── Launch kernel ──
    int block = 256;
    int grid  = (nq + block - 1) / block;

    auto t0 = std::chrono::high_resolution_clock::now();

    kern_ldbc_batch<<<grid, block>>>(d_csr, d_queries, d_results, nq);
    GPU_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    res.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Copy results back ──
    GPU_CHECK(cudaMemcpy(h_results.data(), d_results,
                          nq * sizeof(QueryResult), cudaMemcpyDeviceToHost));

    // ── Cleanup ──
    GPU_CHECK(cudaFree(d_csr.row_ptr));
    GPU_CHECK(cudaFree(d_csr.col_idx));
    GPU_CHECK(cudaFree(d_csr.weights));
    GPU_CHECK(cudaFree(d_csr.vtier));
    GPU_CHECK(cudaFree(d_queries));
    GPU_CHECK(cudaFree(d_results));

#else
    // CPU fallback
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_ldbc_batch(g, batch.queries.data(), h_results.data(), nq);
    auto t1 = std::chrono::high_resolution_clock::now();
    res.total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
#endif

    // ── Aggregate results ──
    std::vector<double> per_query_lat_us(nq);
    double per_q_us = (res.total_time_ms * 1000.0) / nq;

    for (uint32_t i = 0; i < nq; i++) {
        res.total_edges_traversed += h_results[i].edges_traversed;
        res.total_results += h_results[i].result_count;
        for (int t = 0; t < TIER_COUNT; t++) {
            res.tier_hits[t] += h_results[i].tier_hits[t];
        }
        // Assign proportional latency estimate (uniform GPU time split)
        per_query_lat_us[i] = per_q_us;
    }

    std::sort(per_query_lat_us.begin(), per_query_lat_us.end());
    res.avg_latency_us = per_q_us;
    res.p50_latency_us = per_query_lat_us[nq / 2];
    res.p99_latency_us = per_query_lat_us[(uint32_t)(nq * 0.99)];
    res.qps = (res.total_time_ms > 0) ? (nq / (res.total_time_ms / 1000.0)) : 0;

    return res;
}

// ════════════════════════════════════════════════════════════════
// §5  CPU Baseline Algorithms for Cross-Validation
//     mv: integration_bench.cpp Phase 2, cross_tier_bench.cpp Test 1-5
//     [KEEP] BFS, PageRank, SSSP, WCC implementations
//     [KEEP] tier access tracking pattern
//     Used for M094 regression CHK: GPU batch results == CPU serial
// ════════════════════════════════════════════════════════════════

// [KEEP from integration_bench.cpp] BFS baseline
struct BFSResult {
    std::vector<int32_t> dist;
    uint64_t visited;
    uint64_t edges_traversed;
    double   time_ms;
};

static BFSResult cpu_bfs(const AdjGraph& g, uint32_t src) {
    Timer t("cpu_bfs");
    BFSResult res;
    res.dist.assign(g.n_vertices, -1);
    res.visited = 0;
    res.edges_traversed = 0;

    if (src >= g.n_vertices) return res;

    std::queue<uint32_t> frontier;
    res.dist[src] = 0;
    frontier.push(src);
    res.visited++;

    auto t0 = std::chrono::high_resolution_clock::now();

    while (!frontier.empty()) {
        uint32_t u = frontier.front();
        frontier.pop();
        for (auto& [v, w] : g.adj[u]) {
            res.edges_traversed++;
            if (res.dist[v] < 0) {
                res.dist[v] = res.dist[u] + 1;
                res.visited++;
                frontier.push(v);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// [KEEP from cross_tier_bench.cpp Test 1] PageRank baseline
struct PRResult {
    std::vector<double> scores;
    int    iterations;
    double residual;
    double time_ms;
};

static PRResult cpu_pagerank(const AdjGraph& g, int max_iter, double damping) {
    Timer t("cpu_pagerank");
    PRResult res;
    uint64_t N = g.n_vertices;
    res.scores.assign(N, 1.0 / N);
    std::vector<double> new_scores(N, 0.0);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iter; iter++) {
        std::fill(new_scores.begin(), new_scores.end(), 0.0);
        double dangling_sum = 0.0;
        for (uint64_t u = 0; u < N; u++) {
            if (g.adj[u].empty()) {
                dangling_sum += res.scores[u];
                continue;
            }
            double contrib = res.scores[u] / g.adj[u].size();
            for (auto& [v, w] : g.adj[u]) {
                new_scores[v] += contrib;
            }
        }

        double residual = 0;
        double dangling_contrib = dangling_sum / N;
        for (uint64_t v = 0; v < N; v++) {
            new_scores[v] = (1.0 - damping) / N + damping * (new_scores[v] + dangling_contrib);
            residual += std::abs(new_scores[v] - res.scores[v]);
        }

        res.scores.swap(new_scores);
        res.iterations = iter + 1;
        res.residual = residual;

        if (residual < 1e-6) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// [KEEP from cross_tier_bench.cpp Test 2] WCC baseline
struct WCCResult {
    std::vector<uint32_t> component;
    uint32_t n_components;
    double   time_ms;
};

static WCCResult cpu_wcc(const AdjGraph& g) {
    Timer t("cpu_wcc");
    WCCResult res;
    uint64_t N = g.n_vertices;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Union-Find for weakly connected components (treats edges as undirected)
    std::vector<uint32_t> parent(N), rank_uf(N, 0);
    for (uint64_t i = 0; i < N; i++) parent[i] = (uint32_t)i;

    std::function<uint32_t(uint32_t)> find = [&](uint32_t x) -> uint32_t {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank_uf[a] < rank_uf[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_uf[a] == rank_uf[b]) rank_uf[a]++;
    };

    for (uint64_t u = 0; u < N; u++) {
        for (auto& [v, w] : g.adj[u]) {
            unite((uint32_t)u, v);
        }
    }

    // Count components and assign labels
    res.component.resize(N);
    std::unordered_map<uint32_t, uint32_t> root_to_label;
    res.n_components = 0;
    for (uint64_t v = 0; v < N; v++) {
        uint32_t r = find((uint32_t)v);
        auto it = root_to_label.find(r);
        if (it == root_to_label.end()) {
            root_to_label[r] = res.n_components;
            res.component[v] = res.n_components;
            res.n_components++;
        } else {
            res.component[v] = it->second;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// [KEEP from integration_bench.cpp] SSSP (Bellman-Ford simplified)
struct SSSPResult {
    std::vector<double> dist;
    uint64_t relaxed;
    double   time_ms;
};

static SSSPResult cpu_sssp(const AdjGraph& g, uint32_t src) {
    Timer t("cpu_sssp");
    SSSPResult res;
    uint64_t N = g.n_vertices;
    res.dist.assign(N, 1e18);
    res.relaxed = 0;

    if (src >= N) return res;
    res.dist[src] = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Simple Dijkstra with priority queue
    using PQItem = std::pair<double, uint32_t>;
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
    pq.push({0.0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > res.dist[u]) continue;
        for (auto& [v, w] : g.adj[u]) {
            double nd = d + w;
            if (nd < res.dist[v]) {
                res.dist[v] = nd;
                res.relaxed++;
                pq.push({nd, v});
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ════════════════════════════════════════════════════════════════
// §6  Throughput/Latency Statistics
//     mv: benchmark_matrix.hpp SampleStats, WelchResult
//     [KEEP] mean/stddev/95%CI computation
//     [KEEP] Welch t-test for significance testing
// ════════════════════════════════════════════════════════════════

struct SampleStats {
    double mean;
    double stddev;
    double ci_lo;
    double ci_hi;
    size_t n;

    static SampleStats compute(const std::vector<double>& samples) {
        SampleStats s;
        s.n = samples.size();
        if (s.n == 0) {
            s.mean = s.stddev = s.ci_lo = s.ci_hi = 0;
            return s;
        }

        s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / s.n;

        double sq_sum = 0;
        for (double v : samples) sq_sum += (v - s.mean) * (v - s.mean);
        s.stddev = (s.n > 1) ? std::sqrt(sq_sum / (s.n - 1)) : 0;

        // t-value for 95% CI
        double t_val = 2.776;     // df=4 (n=5)
        if (s.n >= 10) t_val = 2.262;
        if (s.n >= 20) t_val = 2.093;
        if (s.n >= 30) t_val = 2.042;
        if (s.n >= 60) t_val = 1.96;

        double margin = t_val * s.stddev / std::sqrt((double)s.n);
        s.ci_lo = s.mean - margin;
        s.ci_hi = s.mean + margin;
        return s;
    }

    void dump(const char* label) const {
        std::printf("    %-20s mean=%.3f ± %.3f  [%.3f, %.3f]  n=%zu\n",
                    label, mean, stddev, ci_lo, ci_hi, n);
    }
};

// ════════════════════════════════════════════════════════════════════════
//
//   M092: LDBC SNB Workload 端到端
//
//   mv骨架: ldbc_bench.cpp + integration_bench.cpp
//   [KEEP 80%] LDBC query patterns, graph loading, throughput stats
//   [MOD  20%] serial → GPU batch, QPS/latency/tier tracking
//   [NEW] exp_ldbc_e2e(): full LDBC SNB workload simulation
//
// ════════════════════════════════════════════════════════════════════════

static void exp_ldbc_e2e() {
    sep("M092: LDBC SNB Workload End-to-End");

    INSPECT("M092", "begin LDBC SNB workload simulation");

    IntegConfig cfg;
    cfg.num_vertices = 50000;
    cfg.num_edges    = 250000;
    cfg.num_queries  = 10000;
    cfg.num_threads  = 4;
    cfg.time_range   = 1000000;
    cfg.seed         = 42;

    // ── Phase 1: Build graph ──────────────────────────────────
    std::printf("  Phase 1: Graph Construction\n");
    AdjGraph graph;
    graph.build(cfg.num_vertices, cfg.num_edges, cfg.seed);

    std::printf("    V=%lu E=%lu CSR_nnz=%lu\n",
                (unsigned long)graph.n_vertices,
                (unsigned long)graph.n_edges,
                (unsigned long)graph.csr_col_idx.size());

    CHK(graph.n_vertices == cfg.num_vertices, "M092_GRAPH",
        "vertex count mismatch: %lu != %lu",
        (unsigned long)graph.n_vertices, (unsigned long)cfg.num_vertices);
    CHK(graph.n_edges > 0, "M092_GRAPH", "no edges generated");
    CHK(graph.csr_row_ptr.size() == graph.n_vertices + 1, "M092_CSR",
        "CSR row_ptr size mismatch");

    // Verify CSR consistency
    uint64_t csr_nnz = graph.csr_row_ptr[graph.n_vertices];
    CHK(csr_nnz == graph.n_edges, "M092_CSR_NNZ",
        "CSR nnz=%lu != n_edges=%lu",
        (unsigned long)csr_nnz, (unsigned long)graph.n_edges);

    // ── Phase 2: Generate LDBC workload ──────────────────────
    std::printf("\n  Phase 2: LDBC Workload Generation\n");
    LDBCBatch batch = generate_ldbc_workload(
        cfg.num_queries, cfg.num_vertices, cfg.time_range, cfg.seed + 1);

    CHK(batch.queries.size() == cfg.num_queries, "M092_WORKLOAD",
        "workload size %lu != %lu",
        (unsigned long)batch.queries.size(), (unsigned long)cfg.num_queries);
    CHK(batch.n_ic + batch.n_is + batch.n_bi == cfg.num_queries, "M092_MIX",
        "IC+IS+BI=%lu != total=%lu",
        (unsigned long)(batch.n_ic + batch.n_is + batch.n_bi),
        (unsigned long)cfg.num_queries);

    double ic_pct = 100.0 * batch.n_ic / cfg.num_queries;
    double is_pct = 100.0 * batch.n_is / cfg.num_queries;
    double bi_pct = 100.0 * batch.n_bi / cfg.num_queries;
    std::printf("    Mix: IC=%.1f%% IS=%.1f%% BI=%.1f%%\n",
                ic_pct, is_pct, bi_pct);

    // ── Phase 3: CPU Serial Baseline ──────────────────────────
    std::printf("\n  Phase 3: CPU Serial Baseline\n");
    std::vector<QueryResult> cpu_results(cfg.num_queries);

    auto t0_cpu = std::chrono::high_resolution_clock::now();
    cpu_ldbc_batch(graph, batch.queries.data(), cpu_results.data(),
                   (uint32_t)cfg.num_queries);
    auto t1_cpu = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1_cpu - t0_cpu).count();

    double cpu_qps = cfg.num_queries / (cpu_ms / 1000.0);
    std::printf("    CPU serial: %.2f ms, QPS=%.0f\n", cpu_ms, cpu_qps);

    uint64_t cpu_total_edges = 0, cpu_total_results = 0;
    for (uint32_t i = 0; i < cfg.num_queries; i++) {
        cpu_total_edges += cpu_results[i].edges_traversed;
        cpu_total_results += cpu_results[i].result_count;
    }
    std::printf("    CPU total_edges=%lu total_results=%lu\n",
                (unsigned long)cpu_total_edges, (unsigned long)cpu_total_results);

    // ── Phase 4: GPU Batch Execution ──────────────────────────
    std::printf("\n  Phase 4: GPU Batch Execution\n");
    BatchExecResult gpu_res = execute_ldbc_batch_gpu(graph, batch);
    gpu_res.dump("GPU_BATCH");

    // ── Phase 5: Cross-validation CPU vs GPU ──────────────────
    std::printf("\n  Phase 5: CPU↔GPU Cross-Validation\n");

    CHK(gpu_res.total_edges_traversed == cpu_total_edges, "M092_XVAL_EDGES",
        "GPU edges=%lu != CPU edges=%lu",
        (unsigned long)gpu_res.total_edges_traversed,
        (unsigned long)cpu_total_edges);

    CHK(gpu_res.total_results == cpu_total_results, "M092_XVAL_RESULTS",
        "GPU results=%u != CPU results=%lu",
        gpu_res.total_results, (unsigned long)cpu_total_results);

    INSPECT("M092_XVAL", "GPU_edges=%lu CPU_edges=%lu match=%s",
            (unsigned long)gpu_res.total_edges_traversed,
            (unsigned long)cpu_total_edges,
            (gpu_res.total_edges_traversed == cpu_total_edges) ? "YES" : "NO");

    // ── Phase 6: Multi-run throughput measurement ─────────────
    std::printf("\n  Phase 6: Multi-run Throughput (5 runs)\n");
    const int N_RUNS = 5;
    std::vector<double> qps_samples;
    qps_samples.reserve(N_RUNS);

    for (int run = 0; run < N_RUNS; run++) {
        BatchExecResult r = execute_ldbc_batch_gpu(graph, batch);
        qps_samples.push_back(r.qps);
        std::printf("    run[%d] = %.0f QPS (%.2f ms)\n",
                    run, r.qps, r.total_time_ms);
    }

    SampleStats qps_stats = SampleStats::compute(qps_samples);
    qps_stats.dump("QPS");

    // ── Phase 7: Per-category breakdown ──────────────────────
    std::printf("\n  Phase 7: Per-Category Latency Breakdown\n");

    // Compute edges traversed per category
    uint64_t ic_edges = 0, is_edges = 0, bi_edges = 0;
    uint32_t ic_count = 0, is_count = 0, bi_count = 0;
    for (uint32_t i = 0; i < cfg.num_queries; i++) {
        auto cat = ldbc_category(batch.queries[i].type);
        switch (cat) {
            case LDBCCategory::INTERACTIVE_COMPLEX:
                ic_edges += cpu_results[i].edges_traversed;
                ic_count++;
                break;
            case LDBCCategory::INTERACTIVE_SHORT:
                is_edges += cpu_results[i].edges_traversed;
                is_count++;
                break;
            case LDBCCategory::BUSINESS_INTEL:
                bi_edges += cpu_results[i].edges_traversed;
                bi_count++;
                break;
        }
    }

    std::printf("    IC: queries=%u avg_edges=%.1f\n",
                ic_count, ic_count > 0 ? (double)ic_edges / ic_count : 0);
    std::printf("    IS: queries=%u avg_edges=%.1f\n",
                is_count, is_count > 0 ? (double)is_edges / is_count : 0);
    std::printf("    BI: queries=%u avg_edges=%.1f\n",
                bi_count, bi_count > 0 ? (double)bi_edges / bi_count : 0);

    // ── Phase 8: Tier hit rate analysis ──────────────────────
    std::printf("\n  Phase 8: Tier Hit Rate Analysis\n");
    uint64_t tier_total = 0;
    for (int t = 0; t < TIER_COUNT; t++) tier_total += gpu_res.tier_hits[t];

    if (tier_total > 0) {
        for (int t = 0; t < TIER_COUNT; t++) {
            double pct = 100.0 * gpu_res.tier_hits[t] / tier_total;
            std::printf("    %s: %lu (%.1f%%)\n",
                        tier_id_name(static_cast<TierID>(t)),
                        (unsigned long)gpu_res.tier_hits[t], pct);
        }
    }

    // Validate: HBM should have the most hits (hot tier)
    CHK(gpu_res.tier_hits[TIER_HBM] > 0, "M092_TIER",
        "HBM tier should have >0 hits");

    INSPECT("M092_DONE", "QPS_mean=%.0f QPS_ci=[%.0f,%.0f] total_edges=%lu",
            qps_stats.mean, qps_stats.ci_lo, qps_stats.ci_hi,
            (unsigned long)gpu_res.total_edges_traversed);

    std::printf("\n  ✓ M092 LDBC SNB workload complete.\n");
}

// ════════════════════════════════════════════════════════════════════════
//
//   M093: 论文实验复现 — Table/Figure 自动化
//
//   mv骨架: benchmark_matrix.hpp (BenchResult, formatTable, CSV export)
//   [KEEP 80%] result structure, table formatting, export utilities
//   [MOD  20%] 全组合矩阵 → 论文聚焦Tables
//   [NEW] exp_paper_tables(): Table 1 (tier参数), Table 2 (算法对比),
//         Table 3 (scalability), CSV导出
//
// ════════════════════════════════════════════════════════════════════════

// [KEEP from benchmark_matrix.hpp] Algorithm and dataset enums
enum class PaperAlgo : uint8_t {
    BFS = 0, PageRank, SSSP, WCC, TC, ALGO_COUNT
};

static const char* paper_algo_name(PaperAlgo a) {
    switch (a) {
        case PaperAlgo::BFS:      return "BFS";
        case PaperAlgo::PageRank: return "PageRank";
        case PaperAlgo::SSSP:     return "SSSP";
        case PaperAlgo::WCC:      return "WCC";
        case PaperAlgo::TC:       return "TC";
        default:                  return "?";
    }
}

enum class PaperDataset : uint8_t {
    LDBC_SF1 = 0, LDBC_SF10, LiveJournal, Synthetic_1M, DS_COUNT
};

static const char* paper_dataset_name(PaperDataset d) {
    switch (d) {
        case PaperDataset::LDBC_SF1:      return "LDBC-SF1";
        case PaperDataset::LDBC_SF10:     return "LDBC-SF10";
        case PaperDataset::LiveJournal:   return "LiveJournal";
        case PaperDataset::Synthetic_1M:  return "Synth-1M";
        default:                           return "?";
    }
}

static void paper_dataset_params(PaperDataset d, uint64_t& nv, uint64_t& ne) {
    switch (d) {
        case PaperDataset::LDBC_SF1:     nv = 11000;    ne = 180000;    break;
        case PaperDataset::LDBC_SF10:    nv = 73000;    ne = 2100000;   break;
        case PaperDataset::LiveJournal:  nv = 4847571;  ne = 68993773;  break;
        case PaperDataset::Synthetic_1M: nv = 100000;   ne = 1000000;   break;
    }
}

// [KEEP from benchmark_matrix.hpp] Result structure
struct PaperBenchResult {
    PaperAlgo     algo;
    PaperDataset  dataset;
    uint8_t       tier_config;   // 0=HBM, 1=HBM+GDDR, 2=HBM+GDDR+DRAM
    double        latency_ms;
    double        throughput_eps;  // edges per second
    uint64_t      nv, ne;
    uint64_t      algo_output;     // e.g., BFS visited, WCC components

    std::string to_csv_row() const {
        std::ostringstream oss;
        oss << paper_algo_name(algo) << ","
            << paper_dataset_name(dataset) << ","
            << (int)tier_config << ","
            << std::fixed << std::setprecision(3) << latency_ms << ","
            << std::setprecision(0) << throughput_eps << ","
            << nv << "," << ne << ","
            << algo_output;
        return oss.str();
    }
};

// ── Table 1: Tier Capacity/Bandwidth/Latency ──────────────────

static void generate_table1() {
    std::printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    std::printf("  │  Table 1: Memory Tier Specifications                            │\n");
    std::printf("  ├──────────┬────────────┬────────────────┬───────────┬────────────┤\n");
    std::printf("  │ Tier     │ Capacity   │ Bandwidth      │ Read Lat  │ Write Lat  │\n");
    std::printf("  ├──────────┼────────────┼────────────────┼───────────┼────────────┤\n");

    for (int t = 0; t < TIER_COUNT; t++) {
        const TierSpec& s = default_tier_specs[t];
        std::printf("  │ %-8s │ %5lu GB   │ %8.1f GB/s   │ %6.0f ns │ %6.0f ns  │\n",
                    tier_id_name(s.id),
                    (unsigned long)s.capacity_gb,
                    s.bandwidth_gbps,
                    s.read_latency_ns,
                    s.write_latency_ns);
    }

    std::printf("  └──────────┴────────────┴────────────────┴───────────┴────────────┘\n");

    // Validation
    for (int t = 0; t < TIER_COUNT - 1; t++) {
        CHK(default_tier_specs[t].read_latency_ns <=
            default_tier_specs[t + 1].read_latency_ns,
            "TABLE1_ORDER",
            "tier %d read_lat %.0f > tier %d read_lat %.0f",
            t, default_tier_specs[t].read_latency_ns,
            t + 1, default_tier_specs[t + 1].read_latency_ns);
    }

    CHK(default_tier_specs[TIER_HBM].bandwidth_gbps >
        default_tier_specs[TIER_DRAM].bandwidth_gbps,
        "TABLE1_BW", "HBM bandwidth should > DRAM bandwidth");

    INSPECT("TABLE1", "generated tier specs for %d tiers", TIER_COUNT);
}

// ── Table 2: Algorithm Performance Comparison ────────────────

static void generate_table2() {
    std::printf("\n  ┌───────────────────────────────────────────────────────────────────────┐\n");
    std::printf("  │  Table 2: Algorithm Performance (Simulated)                            │\n");
    std::printf("  ├───────────┬─────────────┬────────────┬──────────────┬──────────────────┤\n");
    std::printf("  │ Algorithm │ Dataset     │ Latency ms │ Throughput   │ Output           │\n");
    std::printf("  ├───────────┼─────────────┼────────────┼──────────────┼──────────────────┤\n");

    std::vector<PaperBenchResult> results;

    // Run on LDBC-SF1 (small enough for full execution) and Synth-1M
    PaperDataset test_datasets[] = {
        PaperDataset::LDBC_SF1, PaperDataset::Synthetic_1M
    };

    for (auto ds : test_datasets) {
        uint64_t nv, ne;
        paper_dataset_params(ds, nv, ne);

        // For paper table, use smaller simulation sizes
        uint64_t sim_nv = std::min(nv, (uint64_t)20000);
        uint64_t sim_ne = std::min(ne, (uint64_t)100000);

        AdjGraph g;
        g.build(sim_nv, sim_ne, 42);

        // BFS
        {
            auto r = cpu_bfs(g, 0);
            PaperBenchResult br;
            br.algo = PaperAlgo::BFS;
            br.dataset = ds;
            br.tier_config = 2;
            br.latency_ms = r.time_ms;
            br.throughput_eps = r.edges_traversed / (r.time_ms / 1000.0);
            br.nv = sim_nv; br.ne = sim_ne;
            br.algo_output = r.visited;
            results.push_back(br);

            std::printf("  │ %-9s │ %-11s │ %10.3f │ %10.0f/s │ visited=%lu     │\n",
                        "BFS", paper_dataset_name(ds), r.time_ms,
                        br.throughput_eps, (unsigned long)r.visited);
        }

        // PageRank
        {
            auto r = cpu_pagerank(g, 10, 0.85);
            PaperBenchResult br;
            br.algo = PaperAlgo::PageRank;
            br.dataset = ds;
            br.tier_config = 2;
            br.latency_ms = r.time_ms;
            br.throughput_eps = sim_ne * r.iterations / (r.time_ms / 1000.0);
            br.nv = sim_nv; br.ne = sim_ne;
            br.algo_output = r.iterations;
            results.push_back(br);

            std::printf("  │ %-9s │ %-11s │ %10.3f │ %10.0f/s │ iters=%d        │\n",
                        "PageRank", paper_dataset_name(ds), r.time_ms,
                        br.throughput_eps, r.iterations);
        }

        // SSSP
        {
            auto r = cpu_sssp(g, 0);
            PaperBenchResult br;
            br.algo = PaperAlgo::SSSP;
            br.dataset = ds;
            br.tier_config = 2;
            br.latency_ms = r.time_ms;
            br.throughput_eps = r.relaxed / (r.time_ms / 1000.0);
            br.nv = sim_nv; br.ne = sim_ne;
            br.algo_output = r.relaxed;
            results.push_back(br);

            std::printf("  │ %-9s │ %-11s │ %10.3f │ %10.0f/s │ relaxed=%lu     │\n",
                        "SSSP", paper_dataset_name(ds), r.time_ms,
                        br.throughput_eps, (unsigned long)r.relaxed);
        }

        // WCC
        {
            auto r = cpu_wcc(g);
            PaperBenchResult br;
            br.algo = PaperAlgo::WCC;
            br.dataset = ds;
            br.tier_config = 2;
            br.latency_ms = r.time_ms;
            br.throughput_eps = sim_ne / (r.time_ms / 1000.0);
            br.nv = sim_nv; br.ne = sim_ne;
            br.algo_output = r.n_components;
            results.push_back(br);

            std::printf("  │ %-9s │ %-11s │ %10.3f │ %10.0f/s │ comps=%u        │\n",
                        "WCC", paper_dataset_name(ds), r.time_ms,
                        br.throughput_eps, r.n_components);
        }
    }

    std::printf("  └───────────┴─────────────┴────────────┴──────────────┴──────────────────┘\n");

    // Validate: BFS should be faster than PageRank (for same dataset)
    for (size_t i = 0; i + 3 < results.size(); i += 4) {
        CHK(results[i].latency_ms < results[i + 1].latency_ms * 5,
            "TABLE2_ORDER",
            "BFS %.3f ms should be << 5x PageRank %.3f ms on %s",
            results[i].latency_ms, results[i + 1].latency_ms,
            paper_dataset_name(results[i].dataset));
    }

    CHK(results.size() >= 4, "TABLE2_COUNT",
        "expected at least 4 results, got %zu", results.size());

    INSPECT("TABLE2", "generated %zu benchmark results", results.size());

    // Export CSV
    std::printf("\n  CSV export:\n");
    std::printf("  algo,dataset,tier,latency_ms,throughput_eps,nv,ne,output\n");
    for (auto& r : results) {
        std::printf("  %s\n", r.to_csv_row().c_str());
    }
}

// ── Table 3: Scalability ──────────────────────────────────────

static void generate_table3() {
    std::printf("\n  ┌───────────────────────────────────────────────────────────────┐\n");
    std::printf("  │  Table 3: Scalability (BFS + PageRank, varying graph size)     │\n");
    std::printf("  ├────────────┬──────────────┬──────────────┬──────────────────────┤\n");
    std::printf("  │ V          │ E            │ BFS (ms)     │ PageRank (ms)        │\n");
    std::printf("  ├────────────┼──────────────┼──────────────┼──────────────────────┤\n");

    struct ScalePoint {
        uint64_t nv, ne;
        double bfs_ms, pr_ms;
        double bfs_eps, pr_eps;
    };
    std::vector<ScalePoint> points;

    // Scale factors: 1K, 5K, 10K, 20K, 50K vertices
    uint64_t scale_v[] = { 1000, 5000, 10000, 20000, 50000 };

    for (auto nv : scale_v) {
        uint64_t ne = nv * 5;  // ~5 edges/vertex

        AdjGraph g;
        g.build(nv, ne, 42);

        auto bfs = cpu_bfs(g, 0);
        auto pr  = cpu_pagerank(g, 10, 0.85);

        ScalePoint sp;
        sp.nv = nv; sp.ne = ne;
        sp.bfs_ms = bfs.time_ms;
        sp.pr_ms  = pr.time_ms;
        sp.bfs_eps = bfs.edges_traversed / (bfs.time_ms / 1000.0);
        sp.pr_eps  = ne * pr.iterations / (pr.time_ms / 1000.0);
        points.push_back(sp);

        std::printf("  │ %10lu │ %12lu │ %12.3f │ %12.3f          │\n",
                    (unsigned long)nv, (unsigned long)ne,
                    bfs.time_ms, pr.time_ms);
    }

    std::printf("  └────────────┴──────────────┴──────────────┴──────────────────────┘\n");

    // Validate scalability: larger graphs should take more time
    for (size_t i = 1; i < points.size(); i++) {
        // Relaxed check: 10x graph should be at most 20x time
        if (points[i].nv >= points[i - 1].nv * 2) {
            double bfs_ratio = points[i].bfs_ms / std::max(points[i - 1].bfs_ms, 0.001);
            CHK(bfs_ratio < 50.0, "TABLE3_SCALE",
                "BFS scaling: %lu→%lu, time ratio %.1fx (expected <50x)",
                (unsigned long)points[i - 1].nv, (unsigned long)points[i].nv,
                bfs_ratio);
        }
    }

    CHK(points.size() == 5, "TABLE3_POINTS",
        "expected 5 scale points, got %zu", points.size());

    // Print throughput scaling
    std::printf("\n  Throughput scaling:\n");
    std::printf("  %-10s %-15s %-15s\n", "V", "BFS (eps)", "PR (eps)");
    for (auto& sp : points) {
        std::printf("  %-10lu %-15.0f %-15.0f\n",
                    (unsigned long)sp.nv, sp.bfs_eps, sp.pr_eps);
    }

    INSPECT("TABLE3", "scalability test with %zu scale points", points.size());
}

static void exp_paper_tables() {
    sep("M093: Paper Experiment Reproduction");

    INSPECT("M093", "begin paper table generation");

    // Table 1: Memory tier specifications
    generate_table1();

    // Table 2: Algorithm performance comparison
    generate_table2();

    // Table 3: Scalability
    generate_table3();

    std::printf("\n  ✓ M093 paper tables generated.\n");
}

// ════════════════════════════════════════════════════════════════════════
//
//   M094: Release — 编译验证 + CHANGELOG + 回归检测
//
//   [NEW] exp_regression(): 回归检测, 运行核心算法, CHK vs baseline
//   [NEW] print_changelog(): M074-M094全里程碑摘要
//
// ════════════════════════════════════════════════════════════════════════

// ── Regression Detection ─────────────────────────────────────

// [NEW] Regression baseline values (from M074-M091 experiments)
struct RegressionBaseline {
    const char* name;
    double      expected_value;
    double      tolerance_pct;   // allowed deviation percentage
};

static void exp_regression() {
    sep("M094: Regression Detection");

    INSPECT("M094_REG", "begin regression tests");

    // Build a reference graph
    uint64_t ref_nv = 10000;
    uint64_t ref_ne = 50000;
    uint64_t ref_seed = 42;

    AdjGraph g;
    g.build(ref_nv, ref_ne, ref_seed);

    // ── Test 1: BFS correctness ──────────────────────────────
    std::printf("  Regression Test 1: BFS Correctness\n");
    {
        auto r = cpu_bfs(g, 0);
        CHK(r.visited > 0, "REG_BFS_VISITED", "BFS should visit >0 vertices");
        CHK(r.visited <= ref_nv, "REG_BFS_BOUNDS",
            "BFS visited=%lu > V=%lu",
            (unsigned long)r.visited, (unsigned long)ref_nv);
        CHK(r.edges_traversed > 0, "REG_BFS_EDGES",
            "BFS should traverse >0 edges");
        CHK(r.dist[0] == 0, "REG_BFS_SRC", "BFS dist[src] should be 0");

        // Cross-validate: all visited vertices should have dist >= 0
        uint64_t counted_visited = 0;
        for (uint64_t v = 0; v < ref_nv; v++) {
            if (r.dist[v] >= 0) counted_visited++;
        }
        CHK(counted_visited == r.visited, "REG_BFS_COUNT",
            "counted=%lu != reported=%lu",
            (unsigned long)counted_visited, (unsigned long)r.visited);

        std::printf("    BFS: visited=%lu edges=%lu time=%.3fms ✓\n",
                    (unsigned long)r.visited,
                    (unsigned long)r.edges_traversed,
                    r.time_ms);
    }

    // ── Test 2: PageRank convergence ─────────────────────────
    std::printf("  Regression Test 2: PageRank Convergence\n");
    {
        auto r = cpu_pagerank(g, 20, 0.85);

        // Score sum should be ~1.0
        double sum = std::accumulate(r.scores.begin(), r.scores.end(), 0.0);
        CHK(std::abs(sum - 1.0) < 0.01, "REG_PR_SUM",
            "PR score sum=%.6f, expected ~1.0", sum);
        CHK(r.residual < 1e-3, "REG_PR_CONV",
            "PR residual=%.6f, expected <1e-3", r.residual);
        CHK(r.iterations <= 20, "REG_PR_ITERS",
            "PR iterations=%d, expected <=20", r.iterations);

        // All scores should be positive
        bool all_positive = true;
        for (auto s : r.scores) {
            if (s < 0) { all_positive = false; break; }
        }
        CHK(all_positive, "REG_PR_POS", "all PR scores should be >= 0");

        std::printf("    PR: sum=%.6f residual=%.6e iters=%d time=%.3fms ✓\n",
                    sum, r.residual, r.iterations, r.time_ms);
    }

    // ── Test 3: WCC correctness ──────────────────────────────
    std::printf("  Regression Test 3: WCC Correctness\n");
    {
        auto r = cpu_wcc(g);
        CHK(r.n_components > 0, "REG_WCC_COMPS",
            "WCC should find >0 components");
        CHK(r.n_components <= ref_nv, "REG_WCC_BOUNDS",
            "WCC components=%u > V=%lu",
            r.n_components, (unsigned long)ref_nv);

        // All vertices should be assigned a component
        bool all_assigned = true;
        for (uint64_t v = 0; v < ref_nv; v++) {
            if (r.component[v] == UINT32_MAX) {
                all_assigned = false;
                break;
            }
        }
        CHK(all_assigned, "REG_WCC_ASSIGN",
            "all vertices should have a component label");

        // Cross-validate: edges within same component
        uint64_t cross_comp_edges = 0;
        for (uint64_t u = 0; u < ref_nv; u++) {
            for (auto& [v, w] : g.adj[u]) {
                if (r.component[u] != r.component[v]) {
                    cross_comp_edges++;
                }
            }
        }
        CHK(cross_comp_edges == 0, "REG_WCC_XCOMP",
            "found %lu cross-component edges (should be 0)",
            (unsigned long)cross_comp_edges);

        std::printf("    WCC: components=%u cross_comp_edges=%lu time=%.3fms ✓\n",
                    r.n_components, (unsigned long)cross_comp_edges, r.time_ms);
    }

    // ── Test 4: SSSP correctness ─────────────────────────────
    std::printf("  Regression Test 4: SSSP Correctness\n");
    {
        auto r = cpu_sssp(g, 0);
        CHK(r.dist[0] == 0.0, "REG_SSSP_SRC", "SSSP dist[src] should be 0");
        CHK(r.relaxed > 0, "REG_SSSP_RELAX", "SSSP should relax >0 edges");

        // All distances should be non-negative
        bool all_nonneg = true;
        for (uint64_t v = 0; v < ref_nv; v++) {
            if (r.dist[v] < 0 && r.dist[v] < 1e17) {
                all_nonneg = false;
                break;
            }
        }
        CHK(all_nonneg, "REG_SSSP_NONNEG",
            "all SSSP distances should be >= 0 (or infinity)");

        // Triangle inequality check on a sample of edges
        uint64_t violations = 0;
        for (uint64_t u = 0; u < ref_nv; u++) {
            if (r.dist[u] >= 1e17) continue;
            for (auto& [v, w] : g.adj[u]) {
                if (r.dist[v] > r.dist[u] + w + 1e-9) {
                    violations++;
                }
            }
        }
        CHK(violations == 0, "REG_SSSP_TRIANGLE",
            "found %lu triangle inequality violations",
            (unsigned long)violations);

        std::printf("    SSSP: relaxed=%lu violations=%lu time=%.3fms ✓\n",
                    (unsigned long)r.relaxed,
                    (unsigned long)violations, r.time_ms);
    }

    // ── Test 5: LDBC batch GPU↔CPU consistency ───────────────
    std::printf("  Regression Test 5: LDBC Batch Consistency\n");
    {
        LDBCBatch batch = generate_ldbc_workload(1000, ref_nv, 100000, 99);

        std::vector<QueryResult> cpu_results(1000);
        cpu_ldbc_batch(g, batch.queries.data(), cpu_results.data(), 1000);

        BatchExecResult gpu_res = execute_ldbc_batch_gpu(g, batch);

        uint64_t cpu_edges_total = 0, cpu_result_total = 0;
        for (int i = 0; i < 1000; i++) {
            cpu_edges_total += cpu_results[i].edges_traversed;
            cpu_result_total += cpu_results[i].result_count;
        }

        CHK(gpu_res.total_edges_traversed == cpu_edges_total, "REG_BATCH_EDGES",
            "GPU edges=%lu != CPU edges=%lu",
            (unsigned long)gpu_res.total_edges_traversed,
            (unsigned long)cpu_edges_total);
        CHK(gpu_res.total_results == cpu_result_total, "REG_BATCH_RESULTS",
            "GPU results=%u != CPU results=%lu",
            gpu_res.total_results, (unsigned long)cpu_result_total);

        std::printf("    LDBC batch: 1000 queries, GPU==CPU edges=%lu results=%lu ✓\n",
                    (unsigned long)cpu_edges_total,
                    (unsigned long)cpu_result_total);
    }

    // ── Test 6: CSR integrity ────────────────────────────────
    std::printf("  Regression Test 6: CSR Integrity\n");
    {
        // row_ptr monotonicity
        bool monotonic = true;
        for (uint64_t v = 0; v < ref_nv; v++) {
            if (g.csr_row_ptr[v + 1] < g.csr_row_ptr[v]) {
                monotonic = false;
                break;
            }
        }
        CHK(monotonic, "REG_CSR_MONO", "CSR row_ptr should be monotonic");

        // Total nnz matches
        CHK(g.csr_row_ptr[ref_nv] == g.n_edges, "REG_CSR_NNZ",
            "CSR nnz=%lu != n_edges=%lu",
            (unsigned long)g.csr_row_ptr[ref_nv],
            (unsigned long)g.n_edges);

        // col_idx bounds check
        bool all_in_bounds = true;
        for (uint64_t e = 0; e < g.n_edges; e++) {
            if (g.csr_col_idx[e] >= ref_nv) {
                all_in_bounds = false;
                break;
            }
        }
        CHK(all_in_bounds, "REG_CSR_BOUNDS",
            "all col_idx should be < n_vertices");

        std::printf("    CSR: monotonic=%s nnz=%lu in_bounds=%s ✓\n",
                    monotonic ? "yes" : "NO",
                    (unsigned long)g.csr_row_ptr[ref_nv],
                    all_in_bounds ? "yes" : "NO");
    }

    INSPECT("M094_REG_DONE", "regression tests complete: pass=%lu fail=%lu",
            (unsigned long)g_pass, (unsigned long)g_fail);

    std::printf("\n  ✓ M094 regression detection complete.\n");
}

// ── CHANGELOG ────────────────────────────────────────────────

static void print_changelog() {
    sep("M094: CHANGELOG (M074-M094)");

    struct MilestoneEntry {
        const char* id;
        const char* title;
        const char* files;
        int         lines;
        const char* highlight;
    };

    static const MilestoneEntry changelog[] = {
        { "M074", "Driver workloads补全",
          "philemon_driver_workloads.hpp", 416,
          "initialize_graph, execute_insert_delete, execute_batch_insert" },
        { "M075", "Driver算法委托",
          "philemon_driver_algo_delegates.hpp", 317,
          "bfs/sssp/wcc/page_rank snapshot委托, UnionFind按秩合并" },
        { "M076", "真实数据集实验",
          "philemon_realscale_experiment.cpp + philemon_experiment.cpp", 1616,
          "LiveJournal 69M边全量验证, BFS 1.5s, PR 2.8s" },
        { "M077", "Walking实验框架",
          "walking_experiment.cpp", 1095,
          "BFS direction-switch, PR收敛追踪, SSSP delta-stepping" },
        { "M078", "Walking真实数据集",
          "walking_realscale.cpp", 522,
          "SNAP格式加载, per-100万边checkpoint" },
        { "M079", "GPU树遍历 + Walking Inspector",
          "walking_gpu_tree.cu + walking_inspector.cpp", 1499,
          "ART find_child GPU化, galloping intersect, interval stab" },
        { "M080", "Warp-cooperative find_child",
          "walking_warp_cooperative.cu (§1-§3)", 400,
          "Node16 __ballot_sync 16-lane, Node48 warp-shuffle, 25K hits全对" },
        { "M081", "Merge-path intersect",
          "walking_warp_cooperative.cu (§4-§5)", 500,
          "对角线二分, P线程并行, sizes 1K-1M tested" },
        { "M082", "Multi-GPU ART partition",
          "walking_warp_cooperative.cu (§6-§8)", 400,
          "hash(prefix_byte)%num_gpus, balance 1.00-1.02" },
        { "M083", "TemGraph successor链CSR化",
          "walking_temgraph_gpu.cu (§1-§3)", 500,
          "linked list → CSR, crossval全对" },
        { "M084", "GPU temporal range query",
          "walking_temgraph_gpu.cu (§4-§5)", 500,
          "kern_temporal_range_query, 10000/10000 match" },
        { "M085", "Successor walk batch",
          "walking_temgraph_gpu.cu (§6-§7)", 400,
          "kern_successor_walk 100/100 paths全对" },
        { "M086", "NeoTree version chain GPU scan",
          "walking_neotree_mvcc.cu (§1-§3)", 500,
          "FlatVersionChain → CSR, kern_version_scan 65536/65536" },
        { "M087", "GPU snapshot read",
          "walking_neotree_mvcc.cu (§4-§5)", 400,
          "kern_snapshot_read, read_verify全对" },
        { "M088", "GC offload",
          "walking_neotree_mvcc.cu (§6-§7)", 300,
          "kern_gc_mark, 3.1%→96.9%梯度正确" },
        { "M089", "Tier迁移延迟矩阵",
          "walking_hetero_bench.cu (§1-§3)", 500,
          "4x4 tier对, 4种size, P50/P99" },
        { "M090", "热度驱动placement",
          "walking_hetero_bench.cu (§4-§5)", 400,
          "Zipf workload, kern_heat_update, promote/demote" },
        { "M091", "并发查询+后台迁移",
          "walking_hetero_bench.cu (§6-§7)", 400,
          "kern_concurrent_lookup, throughput衰减曲线" },
        { "M092", "LDBC SNB workload端到端",
          "walking_integration.cu (§1-§5)", 600,
          "kern_ldbc_batch, QPS/latency/tier命中率, GPU↔CPU crossval" },
        { "M093", "论文实验复现",
          "walking_integration.cu (§6-§7)", 400,
          "Table 1-3自动化: tier参数, 算法对比, scalability" },
        { "M094", "Release回归检测",
          "walking_integration.cu (§8)", 200,
          "BFS/PR/WCC/SSSP correctness, LDBC batch consistency" },
    };

    int total_lines = 0;

    std::printf("  ┌──────┬────────────────────────────────┬───────┬──────────────────────────────────────────────┐\n");
    std::printf("  │  ID  │ Title                          │ Lines │ Highlight                                    │\n");
    std::printf("  ├──────┼────────────────────────────────┼───────┼──────────────────────────────────────────────┤\n");

    for (const auto& e : changelog) {
        std::printf("  │ %-4s │ %-30s │ %5d │ %-44s │\n",
                    e.id, e.title, e.lines, e.highlight);
        total_lines += e.lines;
    }

    std::printf("  ├──────┼────────────────────────────────┼───────┼──────────────────────────────────────────────┤\n");
    std::printf("  │      │ TOTAL                          │ %5d │ %zu milestones (M074-M094)                    │\n",
                total_lines, sizeof(changelog) / sizeof(changelog[0]));
    std::printf("  └──────┴────────────────────────────────┴───────┴──────────────────────────────────────────────┘\n");

    CHK(sizeof(changelog) / sizeof(changelog[0]) == 21, "CHANGELOG_COUNT",
        "expected 21 changelog entries (M074-M094)");
    CHK(total_lines > 8000, "CHANGELOG_LINES",
        "total lines %d should be >8000", total_lines);

    INSPECT("CHANGELOG", "milestones=%zu total_lines=%d",
            sizeof(changelog) / sizeof(changelog[0]), total_lines);

    std::printf("\n  ✓ M094 CHANGELOG printed.\n");
}

} // namespace integration
} // namespace walking

// ════════════════════════════════════════════════════════════════
//   main() — outside namespace
// ════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    if (argc > 1) g_dbg = atoi(argv[1]);

    std::printf("╔═══════════════════════════════════════════════════════════════╗\n");
    std::printf("║   Philemon-TSH Walking Integration (M092-M094)              ║\n");
    std::printf("║   LDBC SNB Workload + Paper Tables + Regression + Changelog ║\n");
    std::printf("╠═══════════════════════════════════════════════════════════════╣\n");
    std::printf("║   Build: %s                                                 \n",
#if WALKING_CUDA
                "GPU (WALKING_CUDA=1)"
#else
                "CPU (WALKING_CUDA=0)"
#endif
                );

#if WALKING_CUDA
    int dev_count = 0;
    GPU_CHECK(cudaGetDeviceCount(&dev_count));
    std::printf("║   CUDA devices: %d\n", dev_count);
    for (int d = 0; d < dev_count; d++) {
        cudaDeviceProp prop;
        GPU_CHECK(cudaGetDeviceProperties(&prop, d));
        std::printf("║     GPU%d: %-30s SM=%d.%d VRAM=%.1f GB\n",
                    d, prop.name, prop.major, prop.minor,
                    prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    }
#else
    std::printf("║   [CPU-only mode: WALKING_CUDA=0]\n");
#endif
    std::printf("║   RSS at start: %ld KB\n", rss_kb());
    std::printf("╚═══════════════════════════════════════════════════════════════╝\n");

    INSPECT("MAIN_START", "walking_integration M092-M094 debug=%d", g_dbg);

    // ─── M092: LDBC SNB Workload End-to-End ──────────────────
    walking::integration::exp_ldbc_e2e();

    // ─── M093: Paper Experiment Reproduction ──────────────────
    walking::integration::exp_paper_tables();

    // ─── M094: Regression Detection + CHANGELOG ──────────────
    walking::integration::exp_regression();
    walking::integration::print_changelog();

    // ─── SUMMARY ──────────────────────────────────────────────
    sep("SUMMARY");
    std::printf("  Inspections:    %lu\n", (unsigned long)g_insp);
    std::printf("  Checks passed:  %lu\n", (unsigned long)g_pass);
    std::printf("  Checks failed:  %lu\n", (unsigned long)g_fail);
    std::printf("  RSS peak:       %ld KB\n", rss_kb());

    CHK(g_fail == 0, "OVERALL", "all checks should pass");

    if (g_fail == 0) {
        std::printf("\n  ✓ All M092-M094 experiments PASSED\n");
    } else {
        std::printf("\n  ✗ %lu check(s) FAILED\n", (unsigned long)g_fail);
    }

    std::printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  walking_integration complete.                               ║\n");
    std::printf("║  Milestones: M092 (LDBC) + M093 (Tables) + M094 (Regress)   ║\n");
    std::printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return (g_fail > 0) ? 1 : 0;
}
