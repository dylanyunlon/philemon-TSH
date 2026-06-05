/**
 * walking_neotree_mvcc.cu — GPU-parallel MVCC version chain operations
 *
 * mv来源与算法改动对照:
 *
 *   neo_tree_version_impl.hpp (2345行)
 *     KEEP: Version结构体 txn_id, begin_ts, end_ts, value, next指针
 *     KEEP: find_version(txn_id, read_ts): 遍历version chain找对应timestamp的版本
 *     KEEP: VersionedValue基本CRUD: install_version, is_visible
 *     KEEP: gc_copied/gc_ref 过期版本判定: end_ts < min_active_ts
 *     MOD:  链表version chain → flat CSR:
 *           FlatVersionChain: row_ptr[N+1], versions[M]
 *           build_flat() 按key排序构建
 *     MOD:  CPU串行遍历 → kern_version_scan 每thread处理一个查询
 *     MOD:  CPU串行GC → kern_gc_mark GPU标记过期版本
 *
 *   neo_snapshot.hpp (59行 + 180行)
 *     KEEP: snapshot_edges遍历, is_visible(version, snapshot_ts)
 *     MOD:  CPU串行 → kern_snapshot_scan 每thread负责一个key范围
 *     MOD:  kern_snapshot_edge_filter 每thread处理一条边可见性
 *
 *   neo_tree.hpp (127行 + 446行)
 *     KEEP: edges() 模板迭代, find_version 版本链遍历
 *     KEEP: Timer, INSPECT, CHK, sep, rss_kb debug基础设施
 *     KEEP: WALKING_CUDA dispatch宏, GPU_CHECK
 *
 * Build:
 *   GPU:  nvcc -std=c++17 -O2 -arch=sm_86 -DWALKING_CUDA=1 -o walking_neotree_mvcc walking_neotree_mvcc.cu
 *   CPU:  g++ -std=c++17 -O2 -pthread -DWALKING_CUDA=0 -x c++ -o walking_neotree_mvcc walking_neotree_mvcc.cu
 *
 * Milestones: M086 (version chain GPU scan), M087 (snapshot read), M088 (GC offload)
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
          fprintf(stderr, "[CUDA·FATAL] %s:%d %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(e)); \
          exit(1); } } while(0)
#else
  #define GPU_CHECK(call) ((void)0)
  enum { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
  template<typename T> inline int _fake_cuda_malloc(T** p, size_t n) {
      *p = (T*)malloc(n); memset(*p, 0, n); return 0;
  }
  #define cudaMalloc(p,n) _fake_cuda_malloc(p, n)
  #define cudaFree(p) free(p)
  #define cudaMemcpy(d,s,n,k) memcpy(d,s,n)
  #define cudaMemset(p,v,n) memset(p,v,n)
  #define cudaDeviceSynchronize() ((void)0)
#endif

// ════════════════════════════════════════════════════════════════
// Debug infra (upstream: INSPECT/CHK/Timer/sep/rss_kb)
// ════════════════════════════════════════════════════════════════
static int g_dbg = 2;
static long rss_kb() {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss;
}
static uint64_t g_insp = 0, g_pass = 0, g_fail = 0;

#define INSPECT(tag, ...) do { g_insp++; \
    std::printf("[INSPECT·%04lu·%s] ", (unsigned long)g_insp, tag); \
    std::printf(__VA_ARGS__); std::printf("  RSS=%ldKB\n", rss_kb()); } while(0)

#define CHK(cond, tag, ...) do { if(cond){g_pass++;} else { g_fail++; \
    std::printf("[FAIL·%s] ", tag); std::printf(__VA_ARGS__); std::printf("\n"); }} while(0)

struct Timer {
    const char* l;
    std::chrono::high_resolution_clock::time_point t0;
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


// ════════════════════════════════════════════════════════════════
//   namespace walking::mvcc
// ════════════════════════════════════════════════════════════════
namespace walking {
namespace mvcc {

// ════════════════════════════════════════════════════════════════
// §1  Core version structures
//     mv: neo_tree_version_impl.hpp — Version struct, VersionedValue
// ════════════════════════════════════════════════════════════════

// [KEEP from upstream] Each version of a key stores txn_id, begin_ts, end_ts, value
// [MOD] 32-byte aligned for GPU coalesced access
struct alignas(32) FlatVersion {
    uint64_t txn_id;      // transaction that created this version
    uint64_t begin_ts;    // visibility start timestamp
    uint64_t end_ts;      // visibility end timestamp (UINT64_MAX = still active)
    int64_t  value;       // the versioned payload

    // [KEEP from upstream] is_visible(read_ts): begin_ts <= read_ts < end_ts
    bool is_visible(uint64_t read_ts) const {
        return begin_ts <= read_ts && read_ts < end_ts;
    }
};

static_assert(sizeof(FlatVersion) == 32, "FlatVersion must be 32 bytes");

// ─── Timestamp sentinel ───
static constexpr uint64_t TS_INF = UINT64_MAX;
// ─── Result sentinel for "version not found" ───
static constexpr int64_t  NOT_FOUND = INT64_MIN;

// ════════════════════════════════════════════════════════════════
// §2  FlatVersionChain — CSR-style version storage
//     mv: neo_tree_version_impl.hpp — linked list version chain → flat array
// ════════════════════════════════════════════════════════════════
//
// [KEEP from upstream] Each key has a chain of versions ordered by begin_ts descending
//   (newest first). find_version scans from newest to oldest, returns first
//   where begin_ts <= read_ts < end_ts.
//
// [MOD +20%] Linked list → CSR flat array:
//   row_ptr[key]: start offset into versions[] for this key
//   row_ptr[key+1] - row_ptr[key]: number of versions for this key
//   Versions within each key are sorted by begin_ts descending (newest first)
//   This enables GPU coalesced reads and eliminates pointer chasing.

struct FlatVersionChain {
    std::vector<uint64_t>     row_ptr;      // [num_keys + 1]
    std::vector<FlatVersion>  versions;     // [total_versions]
    uint64_t                  num_keys;
    uint64_t                  total_versions;

    FlatVersionChain() : num_keys(0), total_versions(0) {}

    // ── CPU baseline: find visible version for (key, read_ts) ──
    // [KEEP from upstream] Linear scan through version chain
    int64_t find_version_cpu(uint64_t key, uint64_t read_ts) const {
        if (key >= num_keys) return NOT_FOUND;
        uint64_t lo = row_ptr[key];
        uint64_t hi = row_ptr[key + 1];
        // [KEEP] scan newest→oldest, return first visible
        for (uint64_t i = lo; i < hi; i++) {
            if (versions[i].is_visible(read_ts)) {
                return versions[i].value;
            }
        }
        return NOT_FOUND;
    }

    // ── Build from per-key version lists ──
    // [MOD] Converts map<key, vector<Version>> into CSR
    void build_flat(const std::vector<std::vector<FlatVersion>>& per_key) {
        num_keys = per_key.size();
        row_ptr.resize(num_keys + 1);
        row_ptr[0] = 0;
        total_versions = 0;
        for (uint64_t k = 0; k < num_keys; k++) {
            total_versions += per_key[k].size();
            row_ptr[k + 1] = total_versions;
        }
        versions.resize(total_versions);
        for (uint64_t k = 0; k < num_keys; k++) {
            uint64_t off = row_ptr[k];
            for (uint64_t v = 0; v < per_key[k].size(); v++) {
                versions[off + v] = per_key[k][v];
            }
        }
    }

    void dump_stats(const char* tag) const {
        double avg_chain = num_keys > 0 ? (double)total_versions / num_keys : 0.0;
        INSPECT(tag, "keys=%lu versions=%lu avg_chain=%.2f",
                (unsigned long)num_keys, (unsigned long)total_versions, avg_chain);
    }
};


// ════════════════════════════════════════════════════════════════
// §3  Query structures for GPU
// ════════════════════════════════════════════════════════════════

struct VersionQuery {
    uint64_t key;
    uint64_t read_ts;
};

struct VersionResult {
    int64_t  value;       // NOT_FOUND if invisible
    uint64_t found_ts;    // begin_ts of matched version, or 0
};


// ════════════════════════════════════════════════════════════════
// §4  Edge structure for snapshot filtering (M087)
// ════════════════════════════════════════════════════════════════

// [KEEP from upstream] Edge with versioned timestamps
struct VersionedEdge {
    uint64_t src;
    uint64_t dest;
    uint64_t begin_ts;
    uint64_t end_ts;
    int64_t  weight;
};


// ════════════════════════════════════════════════════════════════
// §5  M086 — GPU kernels: version chain scan
//     mv: neo_tree_version_impl.hpp find_version
// ════════════════════════════════════════════════════════════════

#if WALKING_CUDA

// ── 5.1 kern_version_scan ──
// Each thread processes one query (key, read_ts).
// Scans row_ptr[key]..row_ptr[key+1] range linearly, finds first visible version.
// [MOD vs upstream] Pointer chasing → flat array range scan
// [KEEP] Visibility predicate: begin_ts <= read_ts < end_ts
__global__ void kern_version_scan(
    const uint64_t*     __restrict__ d_row_ptr,
    const FlatVersion*  __restrict__ d_versions,
    const VersionQuery* __restrict__ d_queries,
    VersionResult*      __restrict__ d_results,
    uint64_t num_queries,
    uint64_t num_keys)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    uint64_t key     = d_queries[tid].key;
    uint64_t read_ts = d_queries[tid].read_ts;

    // Out-of-range key → NOT_FOUND
    if (key >= num_keys) {
        d_results[tid].value    = NOT_FOUND;
        d_results[tid].found_ts = 0;
        return;
    }

    uint64_t lo = d_row_ptr[key];
    uint64_t hi = d_row_ptr[key + 1];

    // [KEEP] Linear scan newest→oldest (upstream find_version semantics)
    for (uint64_t i = lo; i < hi; i++) {
        FlatVersion ver = d_versions[i];  // coalesced 32-byte read
        if (ver.begin_ts <= read_ts && read_ts < ver.end_ts) {
            d_results[tid].value    = ver.value;
            d_results[tid].found_ts = ver.begin_ts;
            return;
        }
    }

    d_results[tid].value    = NOT_FOUND;
    d_results[tid].found_ts = 0;
}

// ── 5.2 kern_version_scan_warp: warp-cooperative scan for long chains ──
// [MOD +20%] For chains longer than 32, each warp cooperatively scans
// 32 versions per step, uses __ballot_sync to find earliest visible.
// This matches the warp-cooperative pattern from walking_gpu_tree §4.
__global__ void kern_version_scan_warp(
    const uint64_t*     __restrict__ d_row_ptr,
    const FlatVersion*  __restrict__ d_versions,
    const VersionQuery* __restrict__ d_queries,
    VersionResult*      __restrict__ d_results,
    uint64_t num_queries,
    uint64_t num_keys)
{
    uint32_t warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane     = threadIdx.x & 31;
    if (warp_id >= num_queries) return;

    uint64_t key     = d_queries[warp_id].key;
    uint64_t read_ts = d_queries[warp_id].read_ts;

    if (key >= num_keys) {
        if (lane == 0) {
            d_results[warp_id].value    = NOT_FOUND;
            d_results[warp_id].found_ts = 0;
        }
        return;
    }

    uint64_t lo = d_row_ptr[key];
    uint64_t hi = d_row_ptr[key + 1];

    // Warp-cooperative: each lane checks one version per step
    int64_t  found_val = NOT_FOUND;
    uint64_t found_ts  = 0;

    for (uint64_t base = lo; base < hi; base += 32) {
        uint64_t idx = base + lane;
        bool visible = false;
        FlatVersion ver;
        ver.begin_ts = 0; ver.end_ts = 0; ver.value = 0;

        if (idx < hi) {
            ver = d_versions[idx];
            visible = (ver.begin_ts <= read_ts && read_ts < ver.end_ts);
        }

        // __ballot: which lanes found a visible version?
        unsigned mask = __ballot_sync(0xFFFFFFFF, visible);
        if (mask != 0) {
            // Lowest set bit = earliest version in this batch
            // (versions sorted newest-first, so lowest lane = newest visible)
            int winner = __ffs(mask) - 1;
            if (lane == winner) {
                found_val = ver.value;
                found_ts  = ver.begin_ts;
            }
            // Broadcast from winner to lane 0
            found_val = __shfl_sync(0xFFFFFFFF, found_val, winner);
            found_ts  = __shfl_sync(0xFFFFFFFF, found_ts, winner);
            break;  // found; stop scanning
        }
    }

    if (lane == 0) {
        d_results[warp_id].value    = found_val;
        d_results[warp_id].found_ts = found_ts;
    }
}

#endif  // WALKING_CUDA


// ════════════════════════════════════════════════════════════════
// §6  CPU fallback kernels for version scan
// ════════════════════════════════════════════════════════════════

static void cpu_version_scan(
    const uint64_t*     row_ptr,
    const FlatVersion*  versions,
    const VersionQuery* queries,
    VersionResult*      results,
    uint64_t num_queries,
    uint64_t num_keys)
{
    for (uint64_t tid = 0; tid < num_queries; tid++) {
        uint64_t key     = queries[tid].key;
        uint64_t read_ts = queries[tid].read_ts;

        if (key >= num_keys) {
            results[tid].value    = NOT_FOUND;
            results[tid].found_ts = 0;
            continue;
        }

        uint64_t lo = row_ptr[key];
        uint64_t hi = row_ptr[key + 1];
        bool found = false;

        for (uint64_t i = lo; i < hi; i++) {
            if (versions[i].begin_ts <= read_ts && read_ts < versions[i].end_ts) {
                results[tid].value    = versions[i].value;
                results[tid].found_ts = versions[i].begin_ts;
                found = true;
                break;
            }
        }
        if (!found) {
            results[tid].value    = NOT_FOUND;
            results[tid].found_ts = 0;
        }
    }
}


// ════════════════════════════════════════════════════════════════
// §7  Data generation helpers
// ════════════════════════════════════════════════════════════════

// [NEW] Generate version chains for N keys, each with K versions
// Versions have non-overlapping [begin_ts, end_ts) intervals per key
// to ensure at most one version is visible for any read_ts.
static FlatVersionChain generate_version_data(
    uint64_t num_keys, uint64_t versions_per_key, std::mt19937_64& rng)
{
    std::vector<std::vector<FlatVersion>> per_key(num_keys);
    uint64_t txn_counter = 1;

    for (uint64_t k = 0; k < num_keys; k++) {
        // Generate K non-overlapping intervals sorted by begin_ts descending
        // Use sequential timestamps: [0, gap), [gap, 2*gap), ...
        uint64_t gap = 100;  // time span per version
        std::vector<FlatVersion> chain(versions_per_key);

        for (uint64_t v = 0; v < versions_per_key; v++) {
            uint64_t begin = v * gap;
            uint64_t end   = (v == versions_per_key - 1) ? TS_INF : (v + 1) * gap;
            chain[v].txn_id   = txn_counter++;
            chain[v].begin_ts = begin;
            chain[v].end_ts   = end;
            chain[v].value    = (int64_t)(k * 1000 + v);  // deterministic value
        }

        // Sort newest first (descending begin_ts) — upstream semantics
        std::sort(chain.begin(), chain.end(),
            [](const FlatVersion& a, const FlatVersion& b) {
                return a.begin_ts > b.begin_ts;
            });

        per_key[k] = std::move(chain);
    }

    FlatVersionChain fvc;
    fvc.build_flat(per_key);
    return fvc;
}

// [NEW] Generate random queries against the version chain
static std::vector<VersionQuery> generate_queries(
    uint64_t num_queries, uint64_t num_keys, uint64_t max_ts,
    std::mt19937_64& rng)
{
    std::uniform_int_distribution<uint64_t> key_dist(0, num_keys - 1);
    std::uniform_int_distribution<uint64_t> ts_dist(0, max_ts);
    std::vector<VersionQuery> queries(num_queries);
    for (uint64_t i = 0; i < num_queries; i++) {
        queries[i].key     = key_dist(rng);
        queries[i].read_ts = ts_dist(rng);
    }
    return queries;
}


// ════════════════════════════════════════════════════════════════
// §8  M086 — exp_version_chain: version chain GPU scan experiment
// ════════════════════════════════════════════════════════════════

static void exp_version_chain() {
    sep("M086 · VERSION CHAIN GPU SCAN");

    // ─── Parameters ───
    const uint64_t NUM_KEYS          = 4096;
    const uint64_t VERSIONS_PER_KEY  = 16;
    const uint64_t NUM_QUERIES       = 65536;
    const uint64_t MAX_TS            = VERSIONS_PER_KEY * 100 + 50;
    std::mt19937_64 rng(86086);

    // ─── 8.1 Build flat version chain ───
    FlatVersionChain fvc;
    {
        Timer t("M086·build_flat_version_chain");
        fvc = generate_version_data(NUM_KEYS, VERSIONS_PER_KEY, rng);
    }
    fvc.dump_stats("M086·FVC");
    CHK(fvc.num_keys == NUM_KEYS, "M086·keys", "expected %lu got %lu",
        (unsigned long)NUM_KEYS, (unsigned long)fvc.num_keys);
    CHK(fvc.total_versions == NUM_KEYS * VERSIONS_PER_KEY, "M086·total",
        "expected %lu got %lu",
        (unsigned long)(NUM_KEYS * VERSIONS_PER_KEY),
        (unsigned long)fvc.total_versions);

    // Verify CSR structure
    CHK(fvc.row_ptr[0] == 0, "M086·row_ptr_start", "row_ptr[0]=%lu",
        (unsigned long)fvc.row_ptr[0]);
    CHK(fvc.row_ptr[NUM_KEYS] == fvc.total_versions, "M086·row_ptr_end",
        "row_ptr[end]=%lu total=%lu",
        (unsigned long)fvc.row_ptr[NUM_KEYS],
        (unsigned long)fvc.total_versions);

    // ─── 8.2 Generate queries ───
    auto queries = generate_queries(NUM_QUERIES, NUM_KEYS, MAX_TS, rng);
    INSPECT("M086·queries", "num=%lu keys=%lu max_ts=%lu",
            (unsigned long)NUM_QUERIES, (unsigned long)NUM_KEYS,
            (unsigned long)MAX_TS);

    // Inspect first few queries
    for (int i = 0; i < 5 && i < (int)NUM_QUERIES; i++) {
        INSPECT("M086·query_sample", "q[%d] key=%lu read_ts=%lu",
                i, (unsigned long)queries[i].key,
                (unsigned long)queries[i].read_ts);
    }

    // ─── 8.3 CPU baseline ───
    std::vector<VersionResult> cpu_results(NUM_QUERIES);
    {
        Timer t("M086·cpu_version_scan");
        cpu_version_scan(
            fvc.row_ptr.data(), fvc.versions.data(),
            queries.data(), cpu_results.data(),
            NUM_QUERIES, NUM_KEYS);
    }

    // Count hits/misses
    uint64_t cpu_hits = 0, cpu_misses = 0;
    for (uint64_t i = 0; i < NUM_QUERIES; i++) {
        if (cpu_results[i].value != NOT_FOUND) cpu_hits++;
        else cpu_misses++;
    }
    INSPECT("M086·cpu_results", "hits=%lu misses=%lu hit_rate=%.3f",
            (unsigned long)cpu_hits, (unsigned long)cpu_misses,
            (double)cpu_hits / NUM_QUERIES);

    // Inspect first few results
    for (int i = 0; i < 5 && i < (int)NUM_QUERIES; i++) {
        uint64_t lo = fvc.row_ptr[queries[i].key];
        uint64_t hi = fvc.row_ptr[queries[i].key + 1];
        INSPECT("M086·cpu_result_sample",
                "q[%d] key=%lu ts=%lu → val=%ld found_ts=%lu chain_lo=%lu chain_hi=%lu",
                i, (unsigned long)queries[i].key,
                (unsigned long)queries[i].read_ts,
                (long)cpu_results[i].value,
                (unsigned long)cpu_results[i].found_ts,
                (unsigned long)lo, (unsigned long)hi);
    }

    // ─── 8.4 Cross-verify CPU with direct lookup ───
    {
        Timer t("M086·cpu_cross_verify");
        uint64_t match_ok = 0, match_fail = 0;
        for (uint64_t i = 0; i < NUM_QUERIES; i++) {
            int64_t expected = fvc.find_version_cpu(queries[i].key, queries[i].read_ts);
            if (cpu_results[i].value == expected) match_ok++;
            else {
                match_fail++;
                if (match_fail <= 5) {
                    INSPECT("M086·mismatch", "q[%lu] key=%lu ts=%lu cpu_scan=%ld direct=%ld",
                            (unsigned long)i, (unsigned long)queries[i].key,
                            (unsigned long)queries[i].read_ts,
                            (long)cpu_results[i].value, (long)expected);
                }
            }
        }
        CHK(match_fail == 0, "M086·cpu_cross",
            "match_ok=%lu match_fail=%lu", (unsigned long)match_ok, (unsigned long)match_fail);
        INSPECT("M086·cpu_cross", "ok=%lu fail=%lu",
                (unsigned long)match_ok, (unsigned long)match_fail);
    }

    // ─── 8.5 GPU scan ───
    std::vector<VersionResult> gpu_results(NUM_QUERIES);
    {
        Timer t("M086·gpu_version_scan");

        uint64_t*     d_row_ptr   = nullptr;
        FlatVersion*  d_versions  = nullptr;
        VersionQuery* d_queries   = nullptr;
        VersionResult* d_results  = nullptr;

        size_t sz_rp  = (NUM_KEYS + 1) * sizeof(uint64_t);
        size_t sz_ver = fvc.total_versions * sizeof(FlatVersion);
        size_t sz_q   = NUM_QUERIES * sizeof(VersionQuery);
        size_t sz_r   = NUM_QUERIES * sizeof(VersionResult);

        INSPECT("M086·gpu_alloc", "row_ptr=%luB versions=%luB queries=%luB results=%luB total=%luB",
                (unsigned long)sz_rp, (unsigned long)sz_ver,
                (unsigned long)sz_q, (unsigned long)sz_r,
                (unsigned long)(sz_rp + sz_ver + sz_q + sz_r));

        cudaMalloc(&d_row_ptr, sz_rp);
        cudaMalloc(&d_versions, sz_ver);
        cudaMalloc(&d_queries, sz_q);
        cudaMalloc(&d_results, sz_r);
        GPU_CHECK(cudaMemcpy(d_row_ptr, fvc.row_ptr.data(), sz_rp, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_versions, fvc.versions.data(), sz_ver, cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(d_queries, queries.data(), sz_q, cudaMemcpyHostToDevice));

#if WALKING_CUDA
        int threads = 256;
        int blocks  = (NUM_QUERIES + threads - 1) / threads;
        INSPECT("M086·gpu_launch", "blocks=%d threads=%d total_threads=%d queries=%lu",
                blocks, threads, blocks * threads, (unsigned long)NUM_QUERIES);
        kern_version_scan<<<blocks, threads>>>(
            d_row_ptr, d_versions, d_queries, d_results,
            NUM_QUERIES, NUM_KEYS);
        GPU_CHECK(cudaDeviceSynchronize());
#else
        cpu_version_scan(
            (const uint64_t*)d_row_ptr, (const FlatVersion*)d_versions,
            (const VersionQuery*)d_queries, (VersionResult*)d_results,
            NUM_QUERIES, NUM_KEYS);
#endif
        GPU_CHECK(cudaMemcpy(gpu_results.data(), d_results, sz_r, cudaMemcpyDeviceToHost));

        cudaFree(d_row_ptr);
        cudaFree(d_versions);
        cudaFree(d_queries);
        cudaFree(d_results);
    }

    // ─── 8.6 Verify GPU vs CPU ───
    {
        Timer t("M086·verify_gpu_vs_cpu");
        uint64_t ok = 0, mismatch = 0;
        for (uint64_t i = 0; i < NUM_QUERIES; i++) {
            if (gpu_results[i].value == cpu_results[i].value) {
                ok++;
            } else {
                mismatch++;
                if (mismatch <= 5) {
                    INSPECT("M086·gpu_mismatch",
                            "q[%lu] key=%lu ts=%lu gpu_val=%ld cpu_val=%ld",
                            (unsigned long)i, (unsigned long)queries[i].key,
                            (unsigned long)queries[i].read_ts,
                            (long)gpu_results[i].value,
                            (long)cpu_results[i].value);
                }
            }
        }
        CHK(mismatch == 0, "M086·gpu_vs_cpu",
            "ok=%lu mismatch=%lu", (unsigned long)ok, (unsigned long)mismatch);
        INSPECT("M086·gpu_vs_cpu", "ok=%lu mismatch=%lu",
                (unsigned long)ok, (unsigned long)mismatch);
    }

    // ─── 8.7 Warp-cooperative scan (long chains) ───
    {
        // Generate data with longer chains for warp-cooperative test
        const uint64_t LONG_KEYS   = 1024;
        const uint64_t LONG_VPK    = 64;  // 64 versions per key → benefits from warp scan
        const uint64_t LONG_QUERIES = 16384;
        std::mt19937_64 rng2(860862);

        auto long_fvc     = generate_version_data(LONG_KEYS, LONG_VPK, rng2);
        auto long_queries = generate_queries(LONG_QUERIES, LONG_KEYS, LONG_VPK * 100 + 50, rng2);

        long_fvc.dump_stats("M086·LONG_FVC");

        // CPU baseline
        std::vector<VersionResult> long_cpu(LONG_QUERIES);
        {
            Timer t("M086·long_cpu_scan");
            cpu_version_scan(
                long_fvc.row_ptr.data(), long_fvc.versions.data(),
                long_queries.data(), long_cpu.data(),
                LONG_QUERIES, LONG_KEYS);
        }

        // GPU warp scan
        std::vector<VersionResult> long_gpu(LONG_QUERIES);
        {
            Timer t("M086·long_gpu_warp_scan");
            uint64_t*      d_rp = nullptr;
            FlatVersion*   d_vr = nullptr;
            VersionQuery*  d_q  = nullptr;
            VersionResult* d_r  = nullptr;

            size_t s1 = (LONG_KEYS + 1) * sizeof(uint64_t);
            size_t s2 = long_fvc.total_versions * sizeof(FlatVersion);
            size_t s3 = LONG_QUERIES * sizeof(VersionQuery);
            size_t s4 = LONG_QUERIES * sizeof(VersionResult);

            cudaMalloc(&d_rp, s1);
            cudaMalloc(&d_vr, s2);
            cudaMalloc(&d_q, s3);
            cudaMalloc(&d_r, s4);
            GPU_CHECK(cudaMemcpy(d_rp, long_fvc.row_ptr.data(), s1, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_vr, long_fvc.versions.data(), s2, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_q, long_queries.data(), s3, cudaMemcpyHostToDevice));

#if WALKING_CUDA
            // Warp-cooperative: 32 threads per query
            int threads = 256;  // 8 warps per block
            int warps_per_block = threads / 32;
            int blocks = ((int)LONG_QUERIES + warps_per_block - 1) / warps_per_block;
            INSPECT("M086·warp_launch", "blocks=%d threads=%d queries=%lu vpk=%lu",
                    blocks, threads, (unsigned long)LONG_QUERIES, (unsigned long)LONG_VPK);
            kern_version_scan_warp<<<blocks, threads>>>(
                d_rp, d_vr, d_q, d_r, LONG_QUERIES, LONG_KEYS);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            cpu_version_scan(
                (const uint64_t*)d_rp, (const FlatVersion*)d_vr,
                (const VersionQuery*)d_q, (VersionResult*)d_r,
                LONG_QUERIES, LONG_KEYS);
#endif
            GPU_CHECK(cudaMemcpy(long_gpu.data(), d_r, s4, cudaMemcpyDeviceToHost));
            cudaFree(d_rp); cudaFree(d_vr); cudaFree(d_q); cudaFree(d_r);
        }

        // Verify warp results
        uint64_t warp_ok = 0, warp_fail = 0;
        for (uint64_t i = 0; i < LONG_QUERIES; i++) {
            if (long_gpu[i].value == long_cpu[i].value) warp_ok++;
            else {
                warp_fail++;
                if (warp_fail <= 3) {
                    INSPECT("M086·warp_mismatch",
                            "q[%lu] key=%lu ts=%lu gpu=%ld cpu=%ld",
                            (unsigned long)i,
                            (unsigned long)long_queries[i].key,
                            (unsigned long)long_queries[i].read_ts,
                            (long)long_gpu[i].value, (long)long_cpu[i].value);
                }
            }
        }
        CHK(warp_fail == 0, "M086·warp_scan",
            "ok=%lu fail=%lu", (unsigned long)warp_ok, (unsigned long)warp_fail);
        INSPECT("M086·warp_vs_cpu", "ok=%lu fail=%lu",
                (unsigned long)warp_ok, (unsigned long)warp_fail);
    }

    // ─── 8.8 Edge cases ───
    {
        Timer t("M086·edge_cases");

        // Empty key → no versions
        FlatVersionChain empty_fvc;
        empty_fvc.num_keys = 0;
        empty_fvc.total_versions = 0;
        empty_fvc.row_ptr = {0};
        int64_t r = empty_fvc.find_version_cpu(0, 100);
        CHK(r == NOT_FOUND, "M086·empty", "expected NOT_FOUND got %ld", (long)r);
        INSPECT("M086·edge_empty", "result=%ld (NOT_FOUND=%ld)", (long)r, (long)NOT_FOUND);

        // Key out of range
        r = fvc.find_version_cpu(NUM_KEYS + 999, 50);
        CHK(r == NOT_FOUND, "M086·oor_key", "expected NOT_FOUND got %ld", (long)r);
        INSPECT("M086·edge_oor", "key=%lu result=%ld",
                (unsigned long)(NUM_KEYS + 999), (long)r);

        // Timestamp before any version (ts=0, first version starts at 0 → visible)
        r = fvc.find_version_cpu(0, 0);
        // version 0 has begin_ts=0, should be visible
        // (value = key*1000 + version_index for that begin_ts)
        INSPECT("M086·edge_ts0", "key=0 ts=0 result=%ld", (long)r);
        CHK(r != NOT_FOUND, "M086·ts0_visible", "ts=0 should see version 0");

        // Single-version key
        std::vector<std::vector<FlatVersion>> single_key(1);
        single_key[0] = {{42, 10, TS_INF, 999}};
        FlatVersionChain sfvc;
        sfvc.build_flat(single_key);
        int64_t r1 = sfvc.find_version_cpu(0, 5);   // before begin
        int64_t r2 = sfvc.find_version_cpu(0, 10);  // at begin
        int64_t r3 = sfvc.find_version_cpu(0, 500); // well within
        CHK(r1 == NOT_FOUND, "M086·single_before", "ts=5 got %ld", (long)r1);
        CHK(r2 == 999, "M086·single_at", "ts=10 got %ld", (long)r2);
        CHK(r3 == 999, "M086·single_within", "ts=500 got %ld", (long)r3);
        INSPECT("M086·edge_single", "before=%ld at=%ld within=%ld",
                (long)r1, (long)r2, (long)r3);
    }

    INSPECT("M086·DONE", "pass=%lu fail=%lu", (unsigned long)g_pass, (unsigned long)g_fail);
}


// ════════════════════════════════════════════════════════════════
// §9  M087 — GPU snapshot read kernels
//     mv: neo_snapshot.hpp snapshot_edges + neo_tree.hpp edges()
// ════════════════════════════════════════════════════════════════

// ── 9.1 kern_snapshot_scan ──
// Each thread scans one key's version chain for visibility at snapshot_ts.
// Outputs: d_visible[key] = value if visible, NOT_FOUND otherwise
// [MOD vs upstream] serial traversal → parallel per-key

#if WALKING_CUDA

__global__ void kern_snapshot_scan(
    const uint64_t*    __restrict__ d_row_ptr,
    const FlatVersion* __restrict__ d_versions,
    uint64_t snapshot_ts,
    int64_t* __restrict__ d_visible_vals,
    uint8_t* __restrict__ d_visible_flags,
    uint64_t num_keys)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    uint64_t lo = d_row_ptr[tid];
    uint64_t hi = d_row_ptr[tid + 1];

    // [KEEP] scan for first visible version
    for (uint64_t i = lo; i < hi; i++) {
        FlatVersion ver = d_versions[i];
        if (ver.begin_ts <= snapshot_ts && snapshot_ts < ver.end_ts) {
            d_visible_vals[tid]  = ver.value;
            d_visible_flags[tid] = 1;
            return;
        }
    }
    d_visible_vals[tid]  = NOT_FOUND;
    d_visible_flags[tid] = 0;
}

// ── 9.2 kern_snapshot_edge_filter ──
// Each thread processes one edge, checks if both src and dest versions
// are visible at snapshot_ts. Outputs visibility bitmap.
// [MOD] upstream serial edges() → parallel per-edge GPU filter
__global__ void kern_snapshot_edge_filter(
    const VersionedEdge* __restrict__ d_edges,
    const uint64_t*      __restrict__ d_row_ptr,
    const FlatVersion*   __restrict__ d_versions,
    uint64_t snapshot_ts,
    uint8_t* __restrict__ d_edge_visible,
    uint64_t num_edges,
    uint64_t num_keys)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    VersionedEdge edge = d_edges[tid];

    // Check edge's own temporal validity
    bool edge_ok = (edge.begin_ts <= snapshot_ts && snapshot_ts < edge.end_ts);
    if (!edge_ok) {
        d_edge_visible[tid] = 0;
        return;
    }

    // Check src vertex visibility
    bool src_ok = false;
    if (edge.src < num_keys) {
        uint64_t lo = d_row_ptr[edge.src];
        uint64_t hi = d_row_ptr[edge.src + 1];
        for (uint64_t i = lo; i < hi; i++) {
            FlatVersion ver = d_versions[i];
            if (ver.begin_ts <= snapshot_ts && snapshot_ts < ver.end_ts) {
                src_ok = true;
                break;
            }
        }
    }

    // Check dest vertex visibility
    bool dst_ok = false;
    if (edge.dest < num_keys) {
        uint64_t lo = d_row_ptr[edge.dest];
        uint64_t hi = d_row_ptr[edge.dest + 1];
        for (uint64_t i = lo; i < hi; i++) {
            FlatVersion ver = d_versions[i];
            if (ver.begin_ts <= snapshot_ts && snapshot_ts < ver.end_ts) {
                dst_ok = true;
                break;
            }
        }
    }

    d_edge_visible[tid] = (src_ok && dst_ok) ? 1 : 0;
}

#endif  // WALKING_CUDA

// ── CPU fallback for snapshot scan ──
static void cpu_snapshot_scan(
    const uint64_t*    row_ptr,
    const FlatVersion* versions,
    uint64_t snapshot_ts,
    int64_t* visible_vals,
    uint8_t* visible_flags,
    uint64_t num_keys)
{
    for (uint64_t k = 0; k < num_keys; k++) {
        uint64_t lo = row_ptr[k];
        uint64_t hi = row_ptr[k + 1];
        bool found = false;
        for (uint64_t i = lo; i < hi; i++) {
            if (versions[i].begin_ts <= snapshot_ts && snapshot_ts < versions[i].end_ts) {
                visible_vals[k]  = versions[i].value;
                visible_flags[k] = 1;
                found = true;
                break;
            }
        }
        if (!found) {
            visible_vals[k]  = NOT_FOUND;
            visible_flags[k] = 0;
        }
    }
}

// ── CPU fallback for edge filter ──
static void cpu_snapshot_edge_filter(
    const VersionedEdge* edges,
    const uint64_t*      row_ptr,
    const FlatVersion*   versions,
    uint64_t snapshot_ts,
    uint8_t* edge_visible,
    uint64_t num_edges,
    uint64_t num_keys)
{
    for (uint64_t e = 0; e < num_edges; e++) {
        const auto& edge = edges[e];
        bool edge_ok = (edge.begin_ts <= snapshot_ts && snapshot_ts < edge.end_ts);
        if (!edge_ok) { edge_visible[e] = 0; continue; }

        bool src_ok = false;
        if (edge.src < num_keys) {
            uint64_t lo = row_ptr[edge.src];
            uint64_t hi = row_ptr[edge.src + 1];
            for (uint64_t i = lo; i < hi; i++) {
                if (versions[i].begin_ts <= snapshot_ts && snapshot_ts < versions[i].end_ts) {
                    src_ok = true; break;
                }
            }
        }
        bool dst_ok = false;
        if (edge.dest < num_keys) {
            uint64_t lo = row_ptr[edge.dest];
            uint64_t hi = row_ptr[edge.dest + 1];
            for (uint64_t i = lo; i < hi; i++) {
                if (versions[i].begin_ts <= snapshot_ts && snapshot_ts < versions[i].end_ts) {
                    dst_ok = true; break;
                }
            }
        }
        edge_visible[e] = (src_ok && dst_ok) ? 1 : 0;
    }
}


// ════════════════════════════════════════════════════════════════
// §10  M087 — exp_snapshot_read
// ════════════════════════════════════════════════════════════════

static void exp_snapshot_read() {
    sep("M087 · GPU SNAPSHOT READ");

    const uint64_t NUM_KEYS         = 2048;
    const uint64_t VERSIONS_PER_KEY = 8;
    const uint64_t NUM_EDGES        = 32768;
    std::mt19937_64 rng(87087);

    // ─── 10.1 Build version chain ───
    FlatVersionChain fvc;
    {
        Timer t("M087·build");
        fvc = generate_version_data(NUM_KEYS, VERSIONS_PER_KEY, rng);
    }
    fvc.dump_stats("M087·FVC");

    // ─── 10.2 Generate temporal edges ───
    std::vector<VersionedEdge> edges(NUM_EDGES);
    {
        std::uniform_int_distribution<uint64_t> key_dist(0, NUM_KEYS - 1);
        std::uniform_int_distribution<uint64_t> ts_dist(0, VERSIONS_PER_KEY * 100 - 1);
        for (uint64_t i = 0; i < NUM_EDGES; i++) {
            edges[i].src      = key_dist(rng);
            edges[i].dest     = key_dist(rng);
            uint64_t t1       = ts_dist(rng);
            uint64_t t2       = ts_dist(rng);
            if (t1 > t2) std::swap(t1, t2);
            edges[i].begin_ts = t1;
            edges[i].end_ts   = (t2 == t1) ? t1 + 50 : t2;
            edges[i].weight   = (int64_t)(i * 7 + 3);
        }
    }
    INSPECT("M087·edges", "num=%lu", (unsigned long)NUM_EDGES);

    // Inspect sample edges
    for (int i = 0; i < 5; i++) {
        INSPECT("M087·edge_sample", "e[%d] src=%lu dest=%lu ts=[%lu,%lu) w=%ld",
                i, (unsigned long)edges[i].src, (unsigned long)edges[i].dest,
                (unsigned long)edges[i].begin_ts, (unsigned long)edges[i].end_ts,
                (long)edges[i].weight);
    }

    // ─── 10.3 Test multiple snapshot timestamps ───
    uint64_t snapshot_timestamps[] = {0, 50, 150, 350, 700, VERSIONS_PER_KEY * 100 - 1};
    int num_snapshots = sizeof(snapshot_timestamps) / sizeof(snapshot_timestamps[0]);

    for (int s = 0; s < num_snapshots; s++) {
        uint64_t snap_ts = snapshot_timestamps[s];
        INSPECT("M087·snapshot", "ts=%lu", (unsigned long)snap_ts);

        // ─── CPU snapshot scan ───
        std::vector<int64_t> cpu_vis_vals(NUM_KEYS);
        std::vector<uint8_t> cpu_vis_flags(NUM_KEYS);
        {
            Timer t("M087·cpu_snap_scan");
            cpu_snapshot_scan(
                fvc.row_ptr.data(), fvc.versions.data(), snap_ts,
                cpu_vis_vals.data(), cpu_vis_flags.data(), NUM_KEYS);
        }
        uint64_t cpu_visible_count = 0;
        for (uint64_t k = 0; k < NUM_KEYS; k++) {
            if (cpu_vis_flags[k]) cpu_visible_count++;
        }
        INSPECT("M087·cpu_snap", "ts=%lu visible_keys=%lu/%lu",
                (unsigned long)snap_ts, (unsigned long)cpu_visible_count,
                (unsigned long)NUM_KEYS);

        // ─── GPU snapshot scan ───
        std::vector<int64_t> gpu_vis_vals(NUM_KEYS);
        std::vector<uint8_t> gpu_vis_flags(NUM_KEYS);
        {
            Timer t("M087·gpu_snap_scan");
            uint64_t*    d_rp = nullptr;
            FlatVersion* d_vr = nullptr;
            int64_t*     d_vv = nullptr;
            uint8_t*     d_vf = nullptr;

            cudaMalloc(&d_rp, (NUM_KEYS + 1) * sizeof(uint64_t));
            cudaMalloc(&d_vr, fvc.total_versions * sizeof(FlatVersion));
            cudaMalloc(&d_vv, NUM_KEYS * sizeof(int64_t));
            cudaMalloc(&d_vf, NUM_KEYS * sizeof(uint8_t));
            GPU_CHECK(cudaMemcpy(d_rp, fvc.row_ptr.data(),
                                  (NUM_KEYS + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_vr, fvc.versions.data(),
                                  fvc.total_versions * sizeof(FlatVersion), cudaMemcpyHostToDevice));

#if WALKING_CUDA
            int threads = 256;
            int blocks  = (NUM_KEYS + threads - 1) / threads;
            kern_snapshot_scan<<<blocks, threads>>>(
                d_rp, d_vr, snap_ts, d_vv, d_vf, NUM_KEYS);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            cpu_snapshot_scan(
                (const uint64_t*)d_rp, (const FlatVersion*)d_vr, snap_ts,
                (int64_t*)d_vv, (uint8_t*)d_vf, NUM_KEYS);
#endif
            GPU_CHECK(cudaMemcpy(gpu_vis_vals.data(), d_vv,
                                  NUM_KEYS * sizeof(int64_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(gpu_vis_flags.data(), d_vf,
                                  NUM_KEYS * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            cudaFree(d_rp); cudaFree(d_vr); cudaFree(d_vv); cudaFree(d_vf);
        }

        // Verify snapshot scan
        uint64_t snap_ok = 0, snap_fail = 0;
        for (uint64_t k = 0; k < NUM_KEYS; k++) {
            if (gpu_vis_flags[k] == cpu_vis_flags[k] &&
                (gpu_vis_flags[k] == 0 || gpu_vis_vals[k] == cpu_vis_vals[k])) {
                snap_ok++;
            } else {
                snap_fail++;
                if (snap_fail <= 3) {
                    INSPECT("M087·snap_mismatch",
                            "key=%lu ts=%lu gpu_f=%u cpu_f=%u gpu_v=%ld cpu_v=%ld",
                            (unsigned long)k, (unsigned long)snap_ts,
                            gpu_vis_flags[k], cpu_vis_flags[k],
                            (long)gpu_vis_vals[k], (long)cpu_vis_vals[k]);
                }
            }
        }
        CHK(snap_fail == 0, "M087·snap_verify",
            "ts=%lu ok=%lu fail=%lu",
            (unsigned long)snap_ts, (unsigned long)snap_ok, (unsigned long)snap_fail);

        // ─── CPU edge filter ───
        std::vector<uint8_t> cpu_edge_vis(NUM_EDGES);
        {
            Timer t("M087·cpu_edge_filter");
            cpu_snapshot_edge_filter(
                edges.data(), fvc.row_ptr.data(), fvc.versions.data(),
                snap_ts, cpu_edge_vis.data(), NUM_EDGES, NUM_KEYS);
        }
        uint64_t cpu_vis_edges = 0;
        for (uint64_t e = 0; e < NUM_EDGES; e++) {
            if (cpu_edge_vis[e]) cpu_vis_edges++;
        }
        INSPECT("M087·cpu_edge", "ts=%lu visible_edges=%lu/%lu",
                (unsigned long)snap_ts, (unsigned long)cpu_vis_edges,
                (unsigned long)NUM_EDGES);

        // ─── GPU edge filter ───
        std::vector<uint8_t> gpu_edge_vis(NUM_EDGES);
        {
            Timer t("M087·gpu_edge_filter");
            VersionedEdge* d_edges = nullptr;
            uint64_t*      d_rp   = nullptr;
            FlatVersion*   d_vr   = nullptr;
            uint8_t*       d_ev   = nullptr;

            cudaMalloc(&d_edges, NUM_EDGES * sizeof(VersionedEdge));
            cudaMalloc(&d_rp, (NUM_KEYS + 1) * sizeof(uint64_t));
            cudaMalloc(&d_vr, fvc.total_versions * sizeof(FlatVersion));
            cudaMalloc(&d_ev, NUM_EDGES * sizeof(uint8_t));
            GPU_CHECK(cudaMemcpy(d_edges, edges.data(),
                                  NUM_EDGES * sizeof(VersionedEdge), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_rp, fvc.row_ptr.data(),
                                  (NUM_KEYS + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_vr, fvc.versions.data(),
                                  fvc.total_versions * sizeof(FlatVersion), cudaMemcpyHostToDevice));

#if WALKING_CUDA
            int threads = 256;
            int blocks  = (NUM_EDGES + threads - 1) / threads;
            INSPECT("M087·edge_launch", "blocks=%d threads=%d edges=%lu",
                    blocks, threads, (unsigned long)NUM_EDGES);
            kern_snapshot_edge_filter<<<blocks, threads>>>(
                d_edges, d_rp, d_vr, snap_ts, d_ev, NUM_EDGES, NUM_KEYS);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            cpu_snapshot_edge_filter(
                (const VersionedEdge*)d_edges,
                (const uint64_t*)d_rp, (const FlatVersion*)d_vr,
                snap_ts, (uint8_t*)d_ev, NUM_EDGES, NUM_KEYS);
#endif
            GPU_CHECK(cudaMemcpy(gpu_edge_vis.data(), d_ev,
                                  NUM_EDGES * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            cudaFree(d_edges); cudaFree(d_rp); cudaFree(d_vr); cudaFree(d_ev);
        }

        // Verify edge filter
        uint64_t edge_ok = 0, edge_fail = 0;
        for (uint64_t e = 0; e < NUM_EDGES; e++) {
            if (gpu_edge_vis[e] == cpu_edge_vis[e]) edge_ok++;
            else {
                edge_fail++;
                if (edge_fail <= 3) {
                    INSPECT("M087·edge_mismatch",
                            "e[%lu] src=%lu dest=%lu ts=%lu gpu=%u cpu=%u",
                            (unsigned long)e, (unsigned long)edges[e].src,
                            (unsigned long)edges[e].dest, (unsigned long)snap_ts,
                            gpu_edge_vis[e], cpu_edge_vis[e]);
                }
            }
        }
        CHK(edge_fail == 0, "M087·edge_verify",
            "ts=%lu ok=%lu fail=%lu",
            (unsigned long)snap_ts, (unsigned long)edge_ok, (unsigned long)edge_fail);
    }

    // ─── 10.4 Edge cases: empty graph, single snapshot ───
    {
        Timer t("M087·edge_cases");

        // Empty graph
        FlatVersionChain empty_fvc;
        empty_fvc.num_keys = 0;
        empty_fvc.total_versions = 0;
        empty_fvc.row_ptr = {0};

        std::vector<int64_t> ev(1, 0);
        std::vector<uint8_t> ef(1, 99);
        cpu_snapshot_scan(empty_fvc.row_ptr.data(), nullptr, 50,
                          ev.data(), ef.data(), 0);
        INSPECT("M087·edge_empty", "empty_graph scanned ok");

        // Snapshot at ts=0: should see version with begin_ts=0
        std::vector<int64_t> v0_vals(NUM_KEYS);
        std::vector<uint8_t> v0_flags(NUM_KEYS);
        cpu_snapshot_scan(fvc.row_ptr.data(), fvc.versions.data(), 0,
                          v0_vals.data(), v0_flags.data(), NUM_KEYS);
        uint64_t ts0_visible = 0;
        for (uint64_t k = 0; k < NUM_KEYS; k++) {
            if (v0_flags[k]) ts0_visible++;
        }
        INSPECT("M087·edge_ts0", "visible=%lu/%lu",
                (unsigned long)ts0_visible, (unsigned long)NUM_KEYS);
        CHK(ts0_visible == NUM_KEYS, "M087·ts0_all_visible",
            "all keys should be visible at ts=0, got %lu",
            (unsigned long)ts0_visible);
    }

    INSPECT("M087·DONE", "pass=%lu fail=%lu", (unsigned long)g_pass, (unsigned long)g_fail);
}


// ════════════════════════════════════════════════════════════════
// §11  M088 — GC mark kernel
//      mv: neo_tree_version_impl.hpp gc_copied/gc_ref
// ════════════════════════════════════════════════════════════════

// [KEEP from upstream] GC predicate: version is expired if end_ts < min_active_ts
// [MOD] CPU recursive GC → GPU parallel mark + CPU sequential compact

#if WALKING_CUDA

// ── 11.1 kern_gc_mark ──
// Each thread processes one version entry.
// Marks gc_bitmap[i]=1 if end_ts < min_active_ts (reclaimable).
__global__ void kern_gc_mark(
    const FlatVersion* __restrict__ d_versions,
    uint8_t*           __restrict__ d_gc_bitmap,
    uint64_t min_active_ts,
    uint64_t total_versions)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_versions) return;

    FlatVersion ver = d_versions[tid];
    // [KEEP from upstream] Expired: end_ts <= min_active_ts
    // A version whose end_ts <= min_active_ts can never be visible
    // to any currently active transaction
    d_gc_bitmap[tid] = (ver.end_ts <= min_active_ts) ? 1 : 0;
}

// ── 11.2 kern_gc_count_per_key ──
// Each thread processes one key, counts how many of its versions are retained.
// Used to rebuild row_ptr after compaction.
__global__ void kern_gc_count_per_key(
    const uint64_t* __restrict__ d_row_ptr,
    const uint8_t*  __restrict__ d_gc_bitmap,
    uint64_t* __restrict__ d_retained_count,
    uint64_t num_keys)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    uint64_t lo = d_row_ptr[tid];
    uint64_t hi = d_row_ptr[tid + 1];
    uint64_t count = 0;
    for (uint64_t i = lo; i < hi; i++) {
        if (d_gc_bitmap[i] == 0) count++;  // not marked → retained
    }
    d_retained_count[tid] = count;
}

#endif  // WALKING_CUDA

// ── CPU fallback for gc_mark ──
static void cpu_gc_mark(
    const FlatVersion* versions,
    uint8_t* gc_bitmap,
    uint64_t min_active_ts,
    uint64_t total_versions)
{
    for (uint64_t i = 0; i < total_versions; i++) {
        gc_bitmap[i] = (versions[i].end_ts <= min_active_ts) ? 1 : 0;
    }
}

// ── CPU compaction: rebuild version chain after GC ──
// [NEW] Compact versions array and rebuild row_ptr based on gc_bitmap
static FlatVersionChain gc_compact_cpu(
    const FlatVersionChain& fvc,
    const uint8_t* gc_bitmap)
{
    FlatVersionChain result;
    result.num_keys = fvc.num_keys;
    result.row_ptr.resize(fvc.num_keys + 1);
    result.versions.reserve(fvc.total_versions);  // upper bound

    uint64_t write_pos = 0;
    for (uint64_t k = 0; k < fvc.num_keys; k++) {
        result.row_ptr[k] = write_pos;
        uint64_t lo = fvc.row_ptr[k];
        uint64_t hi = fvc.row_ptr[k + 1];
        for (uint64_t i = lo; i < hi; i++) {
            if (gc_bitmap[i] == 0) {  // not marked → keep
                result.versions.push_back(fvc.versions[i]);
                write_pos++;
            }
        }
    }
    result.row_ptr[fvc.num_keys] = write_pos;
    result.total_versions = write_pos;
    return result;
}


// ════════════════════════════════════════════════════════════════
// §12  M088 — exp_gc_offload
// ════════════════════════════════════════════════════════════════

static void exp_gc_offload() {
    sep("M088 · GC OFFLOAD");

    const uint64_t NUM_KEYS         = 4096;
    const uint64_t VERSIONS_PER_KEY = 32;  // many versions → lots to GC
    std::mt19937_64 rng(88088);

    // ─── 12.1 Build version chain with many expired versions ───
    FlatVersionChain fvc;
    {
        Timer t("M088·build");
        fvc = generate_version_data(NUM_KEYS, VERSIONS_PER_KEY, rng);
    }
    fvc.dump_stats("M088·FVC");

    // ─── 12.2 Test multiple min_active_ts thresholds ───
    // Versions have begin_ts: 0, 100, 200, ..., 3100
    // end_ts: 100, 200, ..., 3100, TS_INF
    // A min_active_ts of X should mark all versions with end_ts <= X
    uint64_t gc_thresholds[] = {0, 100, 500, 1500, 2500, 3100};
    int num_thresholds = sizeof(gc_thresholds) / sizeof(gc_thresholds[0]);

    for (int t_idx = 0; t_idx < num_thresholds; t_idx++) {
        uint64_t min_active = gc_thresholds[t_idx];
        INSPECT("M088·gc_threshold", "min_active_ts=%lu", (unsigned long)min_active);

        // ─── CPU GC mark ───
        std::vector<uint8_t> cpu_bitmap(fvc.total_versions);
        uint64_t cpu_marked = 0;
        {
            Timer t("M088·cpu_gc_mark");
            cpu_gc_mark(fvc.versions.data(), cpu_bitmap.data(),
                        min_active, fvc.total_versions);
            for (uint64_t i = 0; i < fvc.total_versions; i++) {
                if (cpu_bitmap[i]) cpu_marked++;
            }
        }
        uint64_t cpu_retained = fvc.total_versions - cpu_marked;
        INSPECT("M088·cpu_gc", "marked=%lu retained=%lu total=%lu ratio=%.3f",
                (unsigned long)cpu_marked, (unsigned long)cpu_retained,
                (unsigned long)fvc.total_versions,
                (double)cpu_marked / fvc.total_versions);

        // Inspect per-key breakdown for first few keys
        for (int k = 0; k < 3 && k < (int)NUM_KEYS; k++) {
            uint64_t lo = fvc.row_ptr[k];
            uint64_t hi = fvc.row_ptr[k + 1];
            uint64_t km = 0;
            for (uint64_t i = lo; i < hi; i++) {
                if (cpu_bitmap[i]) km++;
            }
            INSPECT("M088·cpu_gc_key", "key=%d versions=%lu marked=%lu retained=%lu",
                    k, (unsigned long)(hi - lo), (unsigned long)km,
                    (unsigned long)(hi - lo - km));
        }

        // ─── GPU GC mark ───
        std::vector<uint8_t> gpu_bitmap(fvc.total_versions);
        {
            Timer t("M088·gpu_gc_mark");
            FlatVersion* d_versions = nullptr;
            uint8_t*     d_bitmap   = nullptr;

            size_t sz_v = fvc.total_versions * sizeof(FlatVersion);
            size_t sz_b = fvc.total_versions * sizeof(uint8_t);

            cudaMalloc(&d_versions, sz_v);
            cudaMalloc(&d_bitmap, sz_b);
            GPU_CHECK(cudaMemcpy(d_versions, fvc.versions.data(), sz_v, cudaMemcpyHostToDevice));

#if WALKING_CUDA
            int threads = 256;
            int blocks  = (fvc.total_versions + threads - 1) / threads;
            INSPECT("M088·gc_launch", "blocks=%d threads=%d versions=%lu",
                    blocks, threads, (unsigned long)fvc.total_versions);
            kern_gc_mark<<<blocks, threads>>>(
                d_versions, d_bitmap, min_active, fvc.total_versions);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            cpu_gc_mark((const FlatVersion*)d_versions, (uint8_t*)d_bitmap,
                        min_active, fvc.total_versions);
#endif
            GPU_CHECK(cudaMemcpy(gpu_bitmap.data(), d_bitmap, sz_b, cudaMemcpyDeviceToHost));
            cudaFree(d_versions);
            cudaFree(d_bitmap);
        }

        // Verify GPU vs CPU bitmap
        uint64_t bm_ok = 0, bm_fail = 0;
        for (uint64_t i = 0; i < fvc.total_versions; i++) {
            if (gpu_bitmap[i] == cpu_bitmap[i]) bm_ok++;
            else {
                bm_fail++;
                if (bm_fail <= 5) {
                    INSPECT("M088·bm_mismatch",
                            "v[%lu] txn=%lu begin=%lu end=%lu min_active=%lu gpu=%u cpu=%u",
                            (unsigned long)i,
                            (unsigned long)fvc.versions[i].txn_id,
                            (unsigned long)fvc.versions[i].begin_ts,
                            (unsigned long)fvc.versions[i].end_ts,
                            (unsigned long)min_active,
                            gpu_bitmap[i], cpu_bitmap[i]);
                }
            }
        }
        CHK(bm_fail == 0, "M088·bitmap_verify",
            "min_active=%lu ok=%lu fail=%lu",
            (unsigned long)min_active, (unsigned long)bm_ok, (unsigned long)bm_fail);

        // ─── GPU per-key retained count ───
        std::vector<uint64_t> gpu_retained_count(NUM_KEYS);
        {
            Timer t("M088·gpu_per_key_count");
            uint64_t* d_rp  = nullptr;
            uint8_t*  d_bm  = nullptr;
            uint64_t* d_cnt = nullptr;

            cudaMalloc(&d_rp, (NUM_KEYS + 1) * sizeof(uint64_t));
            cudaMalloc(&d_bm, fvc.total_versions * sizeof(uint8_t));
            cudaMalloc(&d_cnt, NUM_KEYS * sizeof(uint64_t));
            GPU_CHECK(cudaMemcpy(d_rp, fvc.row_ptr.data(),
                                  (NUM_KEYS + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_bm, gpu_bitmap.data(),
                                  fvc.total_versions * sizeof(uint8_t), cudaMemcpyHostToDevice));

#if WALKING_CUDA
            int threads = 256;
            int blocks  = (NUM_KEYS + threads - 1) / threads;
            kern_gc_count_per_key<<<blocks, threads>>>(d_rp, d_bm, d_cnt, NUM_KEYS);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            // CPU fallback
            for (uint64_t k = 0; k < NUM_KEYS; k++) {
                uint64_t lo = fvc.row_ptr[k];
                uint64_t hi = fvc.row_ptr[k + 1];
                uint64_t cnt = 0;
                for (uint64_t i = lo; i < hi; i++) {
                    if (gpu_bitmap[i] == 0) cnt++;
                }
                ((uint64_t*)d_cnt)[k] = cnt;
            }
#endif
            GPU_CHECK(cudaMemcpy(gpu_retained_count.data(), d_cnt,
                                  NUM_KEYS * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            cudaFree(d_rp); cudaFree(d_bm); cudaFree(d_cnt);
        }

        // Verify per-key counts
        uint64_t cnt_ok = 0, cnt_fail = 0;
        for (uint64_t k = 0; k < NUM_KEYS; k++) {
            uint64_t lo = fvc.row_ptr[k];
            uint64_t hi = fvc.row_ptr[k + 1];
            uint64_t expected = 0;
            for (uint64_t i = lo; i < hi; i++) {
                if (cpu_bitmap[i] == 0) expected++;
            }
            if (gpu_retained_count[k] == expected) cnt_ok++;
            else {
                cnt_fail++;
                if (cnt_fail <= 3) {
                    INSPECT("M088·cnt_mismatch", "key=%lu gpu=%lu expected=%lu",
                            (unsigned long)k, (unsigned long)gpu_retained_count[k],
                            (unsigned long)expected);
                }
            }
        }
        CHK(cnt_fail == 0, "M088·per_key_count",
            "ok=%lu fail=%lu", (unsigned long)cnt_ok, (unsigned long)cnt_fail);

        // ─── CPU compaction ───
        FlatVersionChain compacted;
        {
            Timer t("M088·compact");
            compacted = gc_compact_cpu(fvc, cpu_bitmap.data());
        }
        INSPECT("M088·compacted", "before=%lu after=%lu saved=%lu (%.1f%%)",
                (unsigned long)fvc.total_versions,
                (unsigned long)compacted.total_versions,
                (unsigned long)(fvc.total_versions - compacted.total_versions),
                100.0 * (fvc.total_versions - compacted.total_versions) / fvc.total_versions);

        // Verify compacted chain correctness: all retained versions should be the same
        CHK(compacted.total_versions == cpu_retained, "M088·compact_count",
            "compacted=%lu expected=%lu",
            (unsigned long)compacted.total_versions, (unsigned long)cpu_retained);
        CHK(compacted.num_keys == NUM_KEYS, "M088·compact_keys",
            "keys=%lu expected=%lu",
            (unsigned long)compacted.num_keys, (unsigned long)NUM_KEYS);

        // Verify no expired versions remain in compacted chain
        uint64_t stale_count = 0;
        for (uint64_t i = 0; i < compacted.total_versions; i++) {
            if (compacted.versions[i].end_ts <= min_active) stale_count++;
        }
        CHK(stale_count == 0, "M088·no_stale",
            "stale_count=%lu after compaction", (unsigned long)stale_count);
        INSPECT("M088·verify_no_stale", "stale_after_gc=%lu", (unsigned long)stale_count);

        // Verify that compacted chain still produces correct reads
        // Pick a few read_ts values and compare results
        uint64_t verify_ts_list[] = {50, 250, 1000, 2000};
        for (uint64_t vts : verify_ts_list) {
            if (vts > VERSIONS_PER_KEY * 100) continue;
            uint64_t read_match = 0, read_fail = 0;
            for (uint64_t k = 0; k < NUM_KEYS; k++) {
                int64_t orig = fvc.find_version_cpu(k, vts);
                int64_t comp = compacted.find_version_cpu(k, vts);
                // If the original version was expired (end_ts <= min_active) and
                // the read_ts falls within its [begin, end), the compacted chain
                // won't have it. That's correct GC behavior: only active txns matter.
                // We only check that for read_ts >= min_active, results match.
                if (vts >= min_active) {
                    if (orig == comp) read_match++;
                    else {
                        read_fail++;
                        if (read_fail <= 2) {
                            INSPECT("M088·read_mismatch",
                                    "key=%lu ts=%lu min_active=%lu orig=%ld comp=%ld",
                                    (unsigned long)k, (unsigned long)vts,
                                    (unsigned long)min_active,
                                    (long)orig, (long)comp);
                        }
                    }
                } else {
                    read_match++;  // don't check reads below min_active
                }
            }
            INSPECT("M088·read_verify", "ts=%lu min_active=%lu match=%lu fail=%lu",
                    (unsigned long)vts, (unsigned long)min_active,
                    (unsigned long)read_match, (unsigned long)read_fail);
        }
    }

    // ─── 12.3 Edge cases ───
    {
        Timer t("M088·edge_cases");

        // GC with min_active=0 → nothing should be marked
        // (no version has end_ts <= 0 since all end_ts >= 100)
        std::vector<uint8_t> zero_bm(fvc.total_versions);
        cpu_gc_mark(fvc.versions.data(), zero_bm.data(), 0, fvc.total_versions);
        uint64_t zero_marked = 0;
        for (uint64_t i = 0; i < fvc.total_versions; i++) {
            if (zero_bm[i]) zero_marked++;
        }
        CHK(zero_marked == 0, "M088·gc_zero", "min_active=0 should mark nothing, got %lu",
            (unsigned long)zero_marked);
        INSPECT("M088·edge_gc_zero", "min_active=0 marked=%lu", (unsigned long)zero_marked);

        // GC with min_active=TS_INF → all non-INF versions should be marked
        std::vector<uint8_t> inf_bm(fvc.total_versions);
        cpu_gc_mark(fvc.versions.data(), inf_bm.data(), TS_INF, fvc.total_versions);
        uint64_t inf_marked = 0;
        for (uint64_t i = 0; i < fvc.total_versions; i++) {
            if (inf_bm[i]) inf_marked++;
        }
        // All versions should be marked (end_ts <= TS_INF is always true)
        CHK(inf_marked == fvc.total_versions, "M088·gc_inf",
            "min_active=INF should mark all, got %lu/%lu",
            (unsigned long)inf_marked, (unsigned long)fvc.total_versions);
        INSPECT("M088·edge_gc_inf", "min_active=INF marked=%lu/%lu",
                (unsigned long)inf_marked, (unsigned long)fvc.total_versions);

        // Empty version chain GC
        FlatVersionChain empty_fvc;
        empty_fvc.num_keys = 0;
        empty_fvc.total_versions = 0;
        empty_fvc.row_ptr = {0};
        auto empty_compacted = gc_compact_cpu(empty_fvc, nullptr);
        CHK(empty_compacted.total_versions == 0, "M088·gc_empty", "empty gc ok");
        INSPECT("M088·edge_gc_empty", "compacted=%lu", (unsigned long)empty_compacted.total_versions);

        // Single version, not expired
        std::vector<std::vector<FlatVersion>> single_key(1);
        single_key[0] = {{1, 100, TS_INF, 42}};
        FlatVersionChain single_fvc;
        single_fvc.build_flat(single_key);
        std::vector<uint8_t> single_bm(1);
        cpu_gc_mark(single_fvc.versions.data(), single_bm.data(), 50, 1);
        CHK(single_bm[0] == 0, "M088·single_keep",
            "version with end_ts=INF should not be marked at min_active=50");
        INSPECT("M088·edge_single", "end_ts=INF min_active=50 marked=%u", single_bm[0]);

        // Single version, expired
        single_key[0] = {{1, 100, 200, 42}};
        single_fvc.build_flat(single_key);
        cpu_gc_mark(single_fvc.versions.data(), single_bm.data(), 300, 1);
        CHK(single_bm[0] == 1, "M088·single_expire",
            "version with end_ts=200 should be marked at min_active=300");
        INSPECT("M088·edge_single_exp", "end_ts=200 min_active=300 marked=%u", single_bm[0]);
    }

    // ─── 12.4 Throughput measurement ───
    {
        Timer t("M088·throughput");
        // Large-scale GC mark benchmark
        const uint64_t BENCH_KEYS = 8192;
        const uint64_t BENCH_VPK  = 64;
        std::mt19937_64 bench_rng(880884);

        auto bench_fvc = generate_version_data(BENCH_KEYS, BENCH_VPK, bench_rng);
        bench_fvc.dump_stats("M088·BENCH");

        std::vector<uint8_t> bench_bm(bench_fvc.total_versions);
        uint64_t bench_min_active = BENCH_VPK * 50;  // mid-point

        // CPU throughput
        double cpu_ms;
        {
            Timer t2("M088·bench_cpu");
            cpu_gc_mark(bench_fvc.versions.data(), bench_bm.data(),
                        bench_min_active, bench_fvc.total_versions);
            cpu_ms = t2.ms();
        }
        uint64_t bench_cpu_marked = 0;
        for (uint64_t i = 0; i < bench_fvc.total_versions; i++) {
            if (bench_bm[i]) bench_cpu_marked++;
        }

        // GPU throughput
        std::vector<uint8_t> bench_gpu_bm(bench_fvc.total_versions);
        double gpu_ms;
        {
            Timer t2("M088·bench_gpu");
            FlatVersion* d_v = nullptr;
            uint8_t*     d_b = nullptr;
            cudaMalloc(&d_v, bench_fvc.total_versions * sizeof(FlatVersion));
            cudaMalloc(&d_b, bench_fvc.total_versions * sizeof(uint8_t));
            GPU_CHECK(cudaMemcpy(d_v, bench_fvc.versions.data(),
                                  bench_fvc.total_versions * sizeof(FlatVersion),
                                  cudaMemcpyHostToDevice));
#if WALKING_CUDA
            int threads = 256;
            int blocks = (bench_fvc.total_versions + threads - 1) / threads;
            kern_gc_mark<<<blocks, threads>>>(d_v, d_b, bench_min_active,
                                               bench_fvc.total_versions);
            GPU_CHECK(cudaDeviceSynchronize());
#else
            cpu_gc_mark((const FlatVersion*)d_v, (uint8_t*)d_b,
                        bench_min_active, bench_fvc.total_versions);
#endif
            GPU_CHECK(cudaMemcpy(bench_gpu_bm.data(), d_b,
                                  bench_fvc.total_versions * sizeof(uint8_t),
                                  cudaMemcpyDeviceToHost));
            gpu_ms = t2.ms();
            cudaFree(d_v); cudaFree(d_b);
        }

        // Verify consistency
        uint64_t bench_ok = 0;
        for (uint64_t i = 0; i < bench_fvc.total_versions; i++) {
            if (bench_gpu_bm[i] == bench_bm[i]) bench_ok++;
        }
        CHK(bench_ok == bench_fvc.total_versions, "M088·bench_verify",
            "ok=%lu total=%lu", (unsigned long)bench_ok,
            (unsigned long)bench_fvc.total_versions);

        double cpu_throughput = bench_fvc.total_versions / (cpu_ms / 1000.0);
        double gpu_throughput = bench_fvc.total_versions / (gpu_ms / 1000.0);
        INSPECT("M088·throughput_result",
                "versions=%lu marked=%lu cpu=%.0f ver/s (%.2fms) gpu=%.0f ver/s (%.2fms)",
                (unsigned long)bench_fvc.total_versions,
                (unsigned long)bench_cpu_marked,
                cpu_throughput, cpu_ms,
                gpu_throughput, gpu_ms);
    }

    INSPECT("M088·DONE", "pass=%lu fail=%lu", (unsigned long)g_pass, (unsigned long)g_fail);
}


}  // namespace mvcc
}  // namespace walking


// ════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════

using namespace walking::mvcc;

int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  walking_neotree_mvcc — M086-M088 NeoTree GPU MVCC  ║\n");
    std::printf("║  WALKING_CUDA=%d                                     ║\n", WALKING_CUDA);
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    exp_version_chain();   // M086: version chain GPU scan
    exp_snapshot_read();   // M087: GPU snapshot read
    exp_gc_offload();      // M088: GC offload

    // ═══ SUMMARY ═══
    sep("SUMMARY");
    std::printf("Total INSPECT: %lu\n", (unsigned long)g_insp);
    std::printf("Total PASS:    %lu\n", (unsigned long)g_pass);
    std::printf("Total FAIL:    %lu\n", (unsigned long)g_fail);
    std::printf("RSS:           %ld KB\n", rss_kb());

    if (g_fail == 0) {
        std::printf("\n✓ ALL CHECKS PASSED — M086-M088 complete\n");
    } else {
        std::printf("\n✗ %lu CHECKS FAILED\n", (unsigned long)g_fail);
    }
    return (g_fail == 0) ? 0 : 1;
}
