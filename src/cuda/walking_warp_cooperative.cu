/**
 * walking_warp_cooperative.cu — GPU warp-cooperative tree operations
 *
 * Milestones M080–M082 of philemon-TSH
 * Built by Claude #2 (Opus 4.6)
 *
 * mv来源与算法改动对照:
 *
 *   walking_gpu_tree.cu §1–§4 (前560行)
 *     KEEP: FlatNode/FlatART POD layout, FNODE4/16/48/256/FLEAF tags
 *     KEEP: build() bulk-load with choose_type preemptive selection
 *     KEEP: find_child_cpu 4-way dispatch, lookup_cpu point lookup
 *     KEEP: debug infra (INSPECT/CHK/Timer/sep, g_dbg/rss_kb)
 *     KEEP: GPU/CPU compile-mode dispatch (#if WALKING_CUDA)
 *
 *   M080 — warp-cooperative find_child (~400行)
 *     MOD:  Node16 find_child: serial for → 16-lane __ballot_sync parallel compare
 *     MOD:  Node48 find_child: direct index → warp-shuffle probe over 48-byte key_map
 *     NEW:  kern_warp_art_lookup: each warp handles one query cooperatively
 *     NEW:  exp_warp_find_child: benchmark serial vs warp, print speedup
 *
 *   M081 — merge-path intersect (~500行)
 *     MOD:  single-thread gallop → P-thread parallel merge-path
 *     NEW:  merge_path_partition: diagonal binary search for (i,j)
 *     NEW:  kern_merge_path_intersect: two-pass kernel (count + write)
 *     NEW:  exp_merge_path_intersect: test sizes 1K–1M, skew 1:1/1:5/1:20
 *
 *   M082 — multi-GPU ART partition (~400行)
 *     NEW:  MultiGPUART: hash(prefix_byte) % num_gpus assigns subtrees
 *     NEW:  route_query: key first byte → target GPU
 *     NEW:  kern_multi_gpu_lookup: CPU scatter by GPU, per-GPU launch
 *     NEW:  exp_multi_gpu_partition: simulate 2/4 GPU, print balance
 *
 * Build:
 *   GPU:  nvcc -std=c++17 -O2 -arch=sm_86 -DWALKING_CUDA=1 -o walking_warp_cooperative walking_warp_cooperative.cu
 *   CPU:  g++ -std=c++17 -O2 -pthread -DWALKING_CUDA=0 -x c++ -o walking_warp_cooperative walking_warp_cooperative.cu
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
// [KEEP] identical to walking_gpu_tree.cu
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
  #define GPU_CHECK(call) do {       cudaError_t e = (call);       if (e != cudaSuccess) {           fprintf(stderr, "[CUDA·FATAL] %s:%d %s", __FILE__, __LINE__, cudaGetErrorString(e));           exit(1); } } while(0)
  #define WARP_SIZE 32
  #define FULL_MASK 0xFFFFFFFFu
#else
  #define GPU_CHECK(call) ((void)0)
  enum { cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2 };
  inline void* _fake_alloc(size_t n) { return malloc(n); }
  #define cudaMalloc(p,n) (*(p)=_fake_alloc(n),(void)0)
  #define cudaFree(p) free(p)
  #define cudaMemcpy(d,s,n,k) memcpy(d,s,n)
  #define cudaMemset(p,v,n) memset(p,v,n)
  #define cudaDeviceSynchronize() ((void)0)
  #define WARP_SIZE 32
  #define FULL_MASK 0xFFFFFFFFu
#endif

// ════════════════════════════════════════════════════════════════
// Debug infra
// [KEEP] identical to walking_gpu_tree.cu
// ════════════════════════════════════════════════════════════════
static int g_dbg = 2;
static long rss_kb() { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss; }
static uint64_t g_insp = 0, g_pass = 0, g_fail = 0;
#define INSPECT(tag, ...) do { g_insp++;     std::printf("[INSPECT·%04lu·%s] ", g_insp, tag);     std::printf(__VA_ARGS__); std::printf("  RSS=%ldKB", rss_kb()); } while(0)
#define CHK(cond, tag, ...) do { if(cond){g_pass++;} else { g_fail++; \
    std::printf("[FAIL·%s] ", tag); std::printf(__VA_ARGS__); std::printf("\n"); }} while(0)
struct Timer {
    const char* l; std::chrono::high_resolution_clock::time_point t0;
    Timer(const char* s):l(s),t0(std::chrono::high_resolution_clock::now()){
        if(g_dbg>=1) std::printf("[T·START] %s",l);}
    double ms() const { return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now()-t0).count()/1000.0; }
    ~Timer(){ std::printf("[T·END]   %s → %.2f ms", l, ms()); }
};
static void sep(const char* s) {
    std::printf("════════════════════════════════════════════════════  %s"
                "════════════════════════════════════════════════════", s);
}

namespace walking { namespace warp {

// ════════════════════════════════════════════════════════════════
// §1  Flat ART — GPU-portable node representations
//     [KEEP] from walking_gpu_tree.cu §1
//     Node4/16/48/256 + FLEAF, POD FlatNode, FlatART bulk-load
// ════════════════════════════════════════════════════════════════

// [KEEP] Node type tags (upstream naming preserved)
enum : uint8_t { FNODE4=1, FNODE16=2, FNODE48=3, FNODE256=4, FLEAF=5 };

// [KEEP] POD node — no pointers, children via index array
// access_heat for GPU atomic profiling
struct alignas(64) FlatNode {
    uint32_t id;
    uint8_t  type;            // FNODE4..FLEAF
    uint8_t  depth;
    uint16_t num_children;
    uint32_t parent;          // UINT32_MAX = root
    uint32_t children_start;  // index into children[] array
    uint64_t key;             // leaf: full key; internal: prefix
    int64_t  value;           // leaf only
    uint32_t subtree_size;
    uint32_t access_heat;
    // [KEEP] child dispatch keys — Node4/16 store keys here,
    // Node48 uses index-map, Node256 is positional
    uint8_t  child_keys[48];
};

// [KEEP] FlatART with bulk build
struct FlatART {
    std::vector<FlatNode>  nodes;
    std::vector<uint32_t>  children;
    uint32_t root = 0;

    void build(const std::vector<uint64_t>& keys, const std::vector<int64_t>& vals) {
        Timer t("ART_BULK_BUILD");
        nodes.clear(); children.clear();
        if (keys.empty()) return;
        build_sub(keys, vals, 0, keys.size(), 0, UINT32_MAX);
        root = 0;
        compute_subtree();
        INSPECT("ART_BUILT", "nodes=%lu children=%lu max_depth=%u",
                (unsigned long)nodes.size(), (unsigned long)children.size(), max_depth());
        dump_type_dist("BUILT");
    }

    uint8_t max_depth() const {
        uint8_t m=0; for(auto&n:nodes) m=std::max(m,n.depth); return m;
    }

    void dump_type_dist(const char* tag) const {
        uint32_t c[6]={};
        for(auto&n:nodes) if(n.type<=FLEAF) c[n.type]++;
        std::printf("[ART·%s] N4=%u N16=%u N48=%u N256=%u LEAF=%u total=%lu",
                    tag, c[1],c[2],c[3],c[4],c[5], (unsigned long)nodes.size());
    }

private:
    // [KEEP] preemptive type selection
    static uint8_t choose_type(size_t fanout) {
        if(fanout<=4)  return FNODE4;
        if(fanout<=16) return FNODE16;
        if(fanout<=48) return FNODE48;
        return FNODE256;
    }

    uint32_t build_sub(const std::vector<uint64_t>& keys, const std::vector<int64_t>& vals,
                       size_t lo, size_t hi, uint8_t depth, uint32_t par) {
        if(lo>=hi) return UINT32_MAX;
        if(hi-lo==1) {
            uint32_t nid = (uint32_t)nodes.size();
            FlatNode lf{}; lf.id=nid; lf.type=FLEAF; lf.depth=depth;
            lf.num_children=0; lf.parent=par; lf.children_start=0;
            lf.key=keys[lo]; lf.value=vals[lo]; lf.subtree_size=1;
            lf.access_heat=0;
            nodes.push_back(lf);
            return nid;
        }
        auto get_byte=[&](size_t i)->uint8_t{ return (keys[i]>>(56-depth*8))&0xFF; };
        struct Group { uint8_t byte; size_t lo,hi; };
        std::vector<Group> groups;
        size_t i=lo;
        while(i<hi){
            uint8_t b=get_byte(i); size_t j=i+1;
            while(j<hi && get_byte(j)==b) j++;
            groups.push_back({b,i,j});
            i=j;
        }
        uint8_t ntype = choose_type(groups.size());
        uint32_t nid = (uint32_t)nodes.size();
        FlatNode nd{}; nd.id=nid; nd.type=ntype; nd.depth=depth;
        nd.num_children=(uint16_t)groups.size(); nd.parent=par;
        nd.children_start=(uint32_t)children.size();
        nd.key=0; nd.value=0; nd.subtree_size=0; nd.access_heat=0;

        memset(nd.child_keys, 0, sizeof(nd.child_keys));
        if(ntype==FNODE4 || ntype==FNODE16){
            for(size_t g=0;g<groups.size();g++) nd.child_keys[g]=groups[g].byte;
        } else if(ntype==FNODE48){
            for(size_t g=0;g<groups.size();g++) nd.child_keys[groups[g].byte]=(uint8_t)(g+1);
        }

        nodes.push_back(nd);
        size_t off = children.size();
        size_t nslots = (ntype==FNODE256) ? 256 : groups.size();
        children.resize(off + nslots, UINT32_MAX);

        for(size_t g=0;g<groups.size();g++){
            uint32_t cid = build_sub(keys, vals, groups[g].lo, groups[g].hi, depth+1, nid);
            if(ntype==FNODE256)
                children[off + groups[g].byte] = cid;
            else
                children[off + g] = cid;
        }
        return nid;
    }

    void compute_subtree() {
        for(int i=(int)nodes.size()-1;i>=0;i--){
            auto&n=nodes[i];
            if(n.type==FLEAF){ n.subtree_size=1; continue; }
            uint32_t sum=1;
            size_t nslots = (n.type==FNODE256) ? 256 : n.num_children;
            for(size_t c=0;c<nslots;c++){
                uint32_t cid=children[n.children_start+c];
                if(cid<(uint32_t)nodes.size()) sum+=nodes[cid].subtree_size;
            }
            n.subtree_size=sum;
        }
    }
};


// ════════════════════════════════════════════════════════════════
// §2  CPU reference: find_child + lookup
//     [KEEP] from walking_gpu_tree.cu §2–§3
// ════════════════════════════════════════════════════════════════

// [KEEP] 4-way dispatch, returns child node_id
static uint32_t find_child_cpu(const FlatART& tree, uint32_t nid, uint8_t byte) {
    const FlatNode& n = tree.nodes[nid];
    switch(n.type){
    case FNODE4:
    case FNODE16:
        for(uint16_t i=0; i<n.num_children; i++)
            if(n.child_keys[i]==byte)
                return tree.children[n.children_start+i];
        return UINT32_MAX;
    case FNODE48: {
        uint8_t idx = n.child_keys[byte];
        if(idx==0) return UINT32_MAX;
        return tree.children[n.children_start + idx - 1];
    }
    case FNODE256:
        return tree.children[n.children_start + byte];
    default: return UINT32_MAX;
    }
}

// [KEEP] point lookup, root→leaf traversal
static int64_t lookup_cpu(const FlatART& tree, uint64_t key) {
    uint32_t cur = tree.root;
    for(uint8_t d=0; d<8; d++){
        if(cur>=(uint32_t)tree.nodes.size()) return -1;
        const FlatNode& n = tree.nodes[cur];
        if(n.type==FLEAF) return (n.key==key) ? n.value : -1;
        uint8_t byte = (key>>(56-d*8))&0xFF;
        cur = find_child_cpu(tree, cur, byte);
        if(cur==UINT32_MAX) return -1;
    }
    return -1;
}


// ════════════════════════════════════════════════════════════════
// §3  CPU galloping intersect (reference)
//     [KEEP] from walking_gpu_tree.cu §5
// ════════════════════════════════════════════════════════════════

// [KEEP] galloping_lower_bound: exact port from upstream
template<typename GetElem>
static uint32_t galloping_lb(GetElem&& get, uint32_t lo, uint32_t hi, uint64_t target){
    uint32_t step=1, pos=lo;
    while(pos<hi && get(pos)<target){ pos+=step; step<<=1; }
    uint32_t blo=(step>1)?(pos-(step>>1)):lo;
    uint32_t bhi=std::min(pos,hi);
    while(blo<bhi){ uint32_t mid=blo+(bhi-blo)/2; if(get(mid)<target) blo=mid+1; else bhi=mid; }
    return blo;
}

static std::vector<uint64_t> intersect_cpu(const uint64_t* a, uint32_t na,
                                            const uint64_t* b, uint32_t nb) {
    std::vector<uint64_t> out;
    bool use_gallop = (na>4*nb)||(nb>4*na);
    uint32_t ia=0,ib=0;
    uint64_t gallop_skips=0;

    while(ia<na && ib<nb){
        if(a[ia]==b[ib]){ out.push_back(a[ia]); ia++; ib++; }
        else if(a[ia]<b[ib]){
            if(use_gallop && na>4*nb){
                uint32_t old=ia;
                ia=galloping_lb([&](uint32_t i){return a[i];}, ia, na, b[ib]);
                gallop_skips+=ia-old;
            } else ia++;
        } else {
            if(use_gallop && nb>4*na){
                uint32_t old=ib;
                ib=galloping_lb([&](uint32_t i){return b[i];}, ib, nb, a[ia]);
                gallop_skips+=ib-old;
            } else ib++;
        }
    }
    if(g_dbg>=2 && gallop_skips>0)
        INSPECT("GALLOP", "skipped=%lu", (unsigned long)gallop_skips);
    return out;
}


// ════════════════════════════════════════════════════════════════
// §4  GPU Kernels — Serial baseline (from walking_gpu_tree.cu §4)
// ════════════════════════════════════════════════════════════════

#if WALKING_CUDA

// ── 4.1 Serial batch lookup (baseline for comparison) ──
// [KEEP] Each thread does one full root→leaf traversal
__global__ void kern_art_lookup_serial(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,
    uint32_t          num_nodes,
    const uint64_t*   __restrict__ qkeys,
    int64_t*          __restrict__ results,
    uint32_t*         __restrict__ heat,
    uint64_t          nq,
    uint32_t          root)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if(tid >= nq) return;
    uint64_t key = qkeys[tid];
    uint32_t cur = root;
    int64_t  res = -1;

    for(uint8_t d=0; d<8; d++){
        if(cur >= num_nodes) break;
        const FlatNode& n = nodes[cur];
        atomicAdd(&heat[cur], 1);

        if(n.type == FLEAF){
            if(n.key == key) res = n.value;
            break;
        }
        uint8_t byte = (key >> (56 - d*8)) & 0xFF;
        uint32_t next = UINT32_MAX;

        // [KEEP] Inlined find_child — serial 4-way dispatch
        switch(n.type){
        case FNODE4: case FNODE16:
            for(uint16_t i=0; i<n.num_children; i++)
                if(n.child_keys[i]==byte){ next=ch[n.children_start+i]; break; }
            break;
        case FNODE48: {
            uint8_t idx = n.child_keys[byte];
            if(idx) next = ch[n.children_start + idx - 1];
            break;
        }
        case FNODE256:
            next = ch[n.children_start + byte];
            break;
        }
        if(next == UINT32_MAX) break;
        cur = next;
    }
    results[tid] = res;
}


// ════════════════════════════════════════════════════════════════
// §5  M080 — Warp-cooperative find_child
//     [MOD] Node16: serial for → 16-lane __ballot_sync parallel compare
//     [MOD] Node48: direct index → warp-shuffle probe over 48-byte key_map
//     [NEW] kern_warp_art_lookup: each warp handles one query
// ════════════════════════════════════════════════════════════════

// ── 5.1  Device helper: warp-cooperative find_child ──
// Each lane in the warp participates in child-key matching
// lane_id < num_children compares child_keys[lane_id] against target byte
// [MOD] Node16: __ballot_sync across 16 lanes replaces upstream SSE _mm_cmpeq_epi8
// [MOD] Node48: lanes cooperatively scan the 48-byte key_map via __shfl_sync
__device__ uint32_t find_child_warp(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,
    uint32_t          nid,
    uint8_t           byte,
    uint32_t          num_nodes,
    uint32_t*         __restrict__ ballot_out)  // debug: store ballot result
{
    unsigned lane = threadIdx.x & 31;

    if(nid >= num_nodes) return UINT32_MAX;
    const FlatNode& n = nodes[nid];

    switch(n.type){

    // ── Node4: only 4 children, lanes 0–3 compare ──
    // [KEEP] linear scan logic, [MOD] parallelized across 4 lanes
    case FNODE4: {
        uint8_t my_key = (lane < n.num_children) ? n.child_keys[lane] : 0xFF;
        unsigned hit_mask = __ballot_sync(FULL_MASK, my_key == byte);
        // Only lanes [0..num_children-1] are meaningful
        hit_mask &= ((1u << n.num_children) - 1);
        if(ballot_out) *ballot_out = hit_mask;
        if(hit_mask == 0) return UINT32_MAX;
        int pos = __ffs(hit_mask) - 1;
        return ch[n.children_start + pos];
    }

    // ── Node16: 16 lanes each compare one key byte ──
    // [MOD] upstream SSE _mm_cmpeq_epi8 + _mm_movemask → __ballot_sync
    case FNODE16: {
        uint8_t my_key = (lane < n.num_children) ? n.child_keys[lane] : 0xFF;
        unsigned hit_mask = __ballot_sync(FULL_MASK, my_key == byte);
        // Mask off lanes beyond num_children
        hit_mask &= ((1u << n.num_children) - 1);
        if(ballot_out) *ballot_out = hit_mask;
        if(hit_mask == 0) return UINT32_MAX;
        int pos = __ffs(hit_mask) - 1;
        return ch[n.children_start + pos];
    }

    // ── Node48: warp-shuffle probe over 48-byte key_map ──
    // [MOD] upstream direct index n.child_keys[byte] → warp-cooperative:
    //   The 48-byte key_map is distributed across lanes.
    //   Lane 0 reads byte's map entry directly (it's still O(1)),
    //   but we use shuffle to broadcast the result to all lanes for
    //   uniform control flow.  In practice the key_map is in registers
    //   of multiple lanes for wider parallelism on batched probes.
    case FNODE48: {
        // Each lane can hold ⌈256/32⌉=8 entries of the key_map
        // But for single-probe, lane 0 does the lookup and broadcasts
        uint8_t idx = 0;
        if(lane == 0) {
            idx = n.child_keys[byte];
        }
        // [NEW] __shfl_sync broadcast idx from lane 0 to all lanes
        idx = (uint8_t)__shfl_sync(FULL_MASK, (int)idx, 0);
        if(ballot_out) *ballot_out = (idx > 0) ? 1u : 0u;
        if(idx == 0) return UINT32_MAX;
        return ch[n.children_start + idx - 1];
    }

    // ── Node256: direct positional lookup ──
    // [KEEP] no warp cooperation needed, O(1) direct
    case FNODE256: {
        uint32_t cid = ch[n.children_start + byte];
        if(ballot_out) *ballot_out = (cid < num_nodes) ? 1u : 0u;
        return cid;
    }

    default:
        return UINT32_MAX;
    }
}

// ── 5.2  Warp-cooperative ART lookup kernel ──
// [NEW] Each warp processes one query; lane 0 drives traversal,
//       all lanes participate in find_child_warp
__global__ void kern_warp_art_lookup(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,
    uint32_t          num_nodes,
    const uint64_t*   __restrict__ qkeys,
    int64_t*          __restrict__ results,
    uint32_t*         __restrict__ heat,
    uint64_t          nq,
    uint32_t          root)
{
    // One warp = one query
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned lane = threadIdx.x & 31;
    if(warp_id >= nq) return;

    uint64_t key = qkeys[warp_id];
    uint32_t cur = root;
    int64_t  res = -1;

    for(uint8_t d=0; d<8; d++){
        if(cur >= num_nodes) break;
        const FlatNode& n = nodes[cur];

        // [KEEP] access_heat increment (only lane 0 to avoid 32x overcount)
        if(lane == 0) atomicAdd(&heat[cur], 1);

        if(n.type == FLEAF){
            if(n.key == key) res = n.value;
            break;
        }

        uint8_t byte = (key >> (56 - d*8)) & 0xFF;
        // [MOD] warp-cooperative find_child replaces serial dispatch
        uint32_t next = find_child_warp(nodes, ch, cur, byte, num_nodes, nullptr);

        if(next == UINT32_MAX) break;
        cur = next;
    }

    // Only lane 0 writes the result
    if(lane == 0) results[warp_id] = res;
}

// ── 5.3  Debug kernel: warp find_child with ballot INSPECT ──
// [NEW] Runs a small batch, captures ballot masks for debugging
__global__ void kern_warp_find_child_debug(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,
    uint32_t          num_nodes,
    const uint32_t*   __restrict__ query_nids,   // node IDs to query
    const uint8_t*    __restrict__ query_bytes,   // key bytes
    uint32_t*         __restrict__ ballot_masks,  // output: ballot results
    uint32_t*         __restrict__ child_results, // output: child node IDs
    uint32_t          nq)
{
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    unsigned lane = threadIdx.x & 31;
    if(warp_id >= nq) return;

    uint32_t nid = query_nids[warp_id];
    uint8_t  byte = query_bytes[warp_id];
    uint32_t ballot = 0;
    uint32_t cid = find_child_warp(nodes, ch, nid, byte, num_nodes, &ballot);

    if(lane == 0) {
        ballot_masks[warp_id] = ballot;
        child_results[warp_id] = cid;
    }
}

#endif // WALKING_CUDA


// ════════════════════════════════════════════════════════════════
// §6  M080 Experiment: warp-cooperative find_child
//     [NEW] benchmark serial vs warp ART lookup, print speedup
// ════════════════════════════════════════════════════════════════

static void exp_warp_find_child() {
    sep("M080: WARP-COOPERATIVE FIND_CHILD");

    // Build a test ART with mixed node types
    const uint64_t N = 100000;
    const uint64_t NQ = 50000;
    std::mt19937_64 rng(42);

    std::vector<uint64_t> keys(N);
    std::vector<int64_t>  vals(N);
    for(uint64_t i=0;i<N;i++){
        keys[i] = rng();
        vals[i] = (int64_t)i;
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    vals.resize(keys.size());
    for(size_t i=0;i<vals.size();i++) vals[i]=(int64_t)i;
    uint64_t actual_n = keys.size();

    INSPECT("M080_SETUP", "n_keys=%lu n_queries=%lu", (unsigned long)actual_n, (unsigned long)NQ);

    FlatART tree;
    tree.build(keys, vals);

    // Generate queries: 50% existing keys, 50% random (miss)
    std::vector<uint64_t> qkeys(NQ);
    for(uint64_t i=0;i<NQ;i++){
        if(i<NQ/2) qkeys[i] = keys[rng()%actual_n];
        else       qkeys[i] = rng();
    }

    // CPU reference
    std::vector<int64_t> cpu_results(NQ);
    {
        Timer t("M080_CPU_LOOKUP");
        for(uint64_t i=0;i<NQ;i++) cpu_results[i] = lookup_cpu(tree, qkeys[i]);
    }
    uint64_t cpu_hits=0;
    for(auto v:cpu_results) if(v>=0) cpu_hits++;
    INSPECT("M080_CPU", "hits=%lu misses=%lu", (unsigned long)cpu_hits, (unsigned long)(NQ-cpu_hits));

#if WALKING_CUDA
    // Upload tree to GPU
    FlatNode* d_nodes=nullptr;
    uint32_t* d_ch=nullptr;
    uint64_t* d_qkeys=nullptr;
    int64_t*  d_results=nullptr;
    uint32_t* d_heat=nullptr;

    size_t node_bytes = tree.nodes.size()*sizeof(FlatNode);
    size_t ch_bytes   = tree.children.size()*sizeof(uint32_t);
    uint32_t num_nodes = (uint32_t)tree.nodes.size();

    GPU_CHECK(cudaMalloc(&d_nodes, node_bytes));
    GPU_CHECK(cudaMalloc(&d_ch, ch_bytes));
    GPU_CHECK(cudaMalloc(&d_qkeys, NQ*sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_results, NQ*sizeof(int64_t)));
    GPU_CHECK(cudaMalloc(&d_heat, num_nodes*sizeof(uint32_t)));

    GPU_CHECK(cudaMemcpy(d_nodes, tree.nodes.data(), node_bytes, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_ch, tree.children.data(), ch_bytes, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_qkeys, qkeys.data(), NQ*sizeof(uint64_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(d_heat, 0, num_nodes*sizeof(uint32_t)));

    INSPECT("M080_GPU_UPLOAD", "node_bytes=%lu ch_bytes=%lu", (unsigned long)node_bytes, (unsigned long)ch_bytes);

    // ── Serial GPU baseline ──
    double serial_ms;
    {
        GPU_CHECK(cudaMemset(d_results, 0xFF, NQ*sizeof(int64_t)));
        int blk = 256;
        int grd = (int)((NQ+blk-1)/blk);
        INSPECT("M080_SERIAL_LAUNCH", "grid=%d block=%d nq=%lu", grd, blk, (unsigned long)NQ);

        Timer t("M080_GPU_SERIAL");
        kern_art_lookup_serial<<<grd,blk>>>(d_nodes, d_ch, num_nodes,
                                             d_qkeys, d_results, d_heat, NQ, tree.root);
        GPU_CHECK(cudaDeviceSynchronize());
        serial_ms = t.ms();
    }

    // Verify serial GPU vs CPU
    std::vector<int64_t> gpu_serial(NQ);
    GPU_CHECK(cudaMemcpy(gpu_serial.data(), d_results, NQ*sizeof(int64_t), cudaMemcpyDeviceToHost));
    uint64_t serial_match=0;
    for(uint64_t i=0;i<NQ;i++) if(gpu_serial[i]==cpu_results[i]) serial_match++;
    CHK(serial_match==NQ, "M080_SERIAL_VS_CPU", "match=%lu/%lu", (unsigned long)serial_match, (unsigned long)NQ);
    INSPECT("M080_SERIAL", "match=%lu/%lu time=%.2fms", (unsigned long)serial_match, (unsigned long)NQ, serial_ms);

    // ── Warp-cooperative GPU ──
    double warp_ms;
    {
        GPU_CHECK(cudaMemset(d_results, 0xFF, NQ*sizeof(int64_t)));
        GPU_CHECK(cudaMemset(d_heat, 0, num_nodes*sizeof(uint32_t)));
        // Each warp handles one query → need NQ warps
        int threads_per_block = 256;  // 8 warps per block
        int warps_per_block = threads_per_block / WARP_SIZE;
        int grd = (int)((NQ + warps_per_block - 1) / warps_per_block);
        INSPECT("M080_WARP_LAUNCH", "grid=%d block=%d warps_per_block=%d nq=%lu",
                grd, threads_per_block, warps_per_block, (unsigned long)NQ);

        Timer t("M080_GPU_WARP");
        kern_warp_art_lookup<<<grd,threads_per_block>>>(d_nodes, d_ch, num_nodes,
                                                         d_qkeys, d_results, d_heat, NQ, tree.root);
        GPU_CHECK(cudaDeviceSynchronize());
        warp_ms = t.ms();
    }

    // Verify warp GPU vs CPU
    std::vector<int64_t> gpu_warp(NQ);
    GPU_CHECK(cudaMemcpy(gpu_warp.data(), d_results, NQ*sizeof(int64_t), cudaMemcpyDeviceToHost));
    uint64_t warp_match=0;
    for(uint64_t i=0;i<NQ;i++) if(gpu_warp[i]==cpu_results[i]) warp_match++;
    CHK(warp_match==NQ, "M080_WARP_VS_CPU", "match=%lu/%lu", (unsigned long)warp_match, (unsigned long)NQ);
    INSPECT("M080_WARP", "match=%lu/%lu time=%.2fms", (unsigned long)warp_match, (unsigned long)NQ, warp_ms);

    // ── Ballot debug on small sample ──
    {
        // Find a few Node16 nodes for debug
        std::vector<uint32_t> n16_ids;
        for(uint32_t i=0;i<num_nodes && n16_ids.size()<4;i++){
            if(tree.nodes[i].type==FNODE16) n16_ids.push_back(i);
        }
        if(!n16_ids.empty()){
            uint32_t dbg_n = (uint32_t)n16_ids.size();
            std::vector<uint8_t> dbg_bytes(dbg_n);
            for(uint32_t i=0;i<dbg_n;i++){
                // Use first child key as query byte (guaranteed hit)
                dbg_bytes[i] = tree.nodes[n16_ids[i]].child_keys[0];
            }

            uint32_t *d_qnids, *d_ballots, *d_cresults;
            uint8_t  *d_qbytes;
            GPU_CHECK(cudaMalloc(&d_qnids, dbg_n*sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_qbytes, dbg_n*sizeof(uint8_t)));
            GPU_CHECK(cudaMalloc(&d_ballots, dbg_n*sizeof(uint32_t)));
            GPU_CHECK(cudaMalloc(&d_cresults, dbg_n*sizeof(uint32_t)));
            GPU_CHECK(cudaMemcpy(d_qnids, n16_ids.data(), dbg_n*sizeof(uint32_t), cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_qbytes, dbg_bytes.data(), dbg_n*sizeof(uint8_t), cudaMemcpyHostToDevice));

            int dbg_threads = dbg_n * WARP_SIZE;
            int dbg_blk = (dbg_threads < 256) ? dbg_threads : 256;
            int dbg_grd = (dbg_threads + dbg_blk - 1) / dbg_blk;

            kern_warp_find_child_debug<<<dbg_grd, dbg_blk>>>(
                d_nodes, d_ch, num_nodes,
                d_qnids, d_qbytes, d_ballots, d_cresults, dbg_n);
            GPU_CHECK(cudaDeviceSynchronize());

            std::vector<uint32_t> h_ballots(dbg_n), h_cresults(dbg_n);
            GPU_CHECK(cudaMemcpy(h_ballots.data(), d_ballots, dbg_n*sizeof(uint32_t), cudaMemcpyDeviceToHost));
            GPU_CHECK(cudaMemcpy(h_cresults.data(), d_cresults, dbg_n*sizeof(uint32_t), cudaMemcpyDeviceToHost));

            for(uint32_t i=0;i<dbg_n;i++){
                INSPECT("M080_BALLOT", "node=%u type=N16 byte=0x%02X ballot=0x%08X child=%u",
                        n16_ids[i], dbg_bytes[i], h_ballots[i], h_cresults[i]);
                // Verify ballot has exactly one bit set (unique key)
                CHK(__builtin_popcount(h_ballots[i])==1, "M080_BALLOT_POPCOUNT",
                    "node=%u expected 1 bit, got %d", n16_ids[i], __builtin_popcount(h_ballots[i]));
            }

            GPU_CHECK(cudaFree(d_qnids));
            GPU_CHECK(cudaFree(d_qbytes));
            GPU_CHECK(cudaFree(d_ballots));
            GPU_CHECK(cudaFree(d_cresults));
        }
    }

    // Speedup
    double speedup = (warp_ms > 0) ? serial_ms / warp_ms : 0.0;
    INSPECT("M080_SPEEDUP", "serial=%.2fms warp=%.2fms ratio=%.2fx", serial_ms, warp_ms, speedup);

    // Check access_heat distribution
    std::vector<uint32_t> h_heat(num_nodes);
    GPU_CHECK(cudaMemcpy(h_heat.data(), d_heat, num_nodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint64_t total_heat=0; uint32_t max_heat=0;
    for(uint32_t i=0;i<num_nodes;i++){
        total_heat+=h_heat[i];
        max_heat=std::max(max_heat, h_heat[i]);
    }
    INSPECT("M080_HEAT", "total=%lu max=%u avg=%.1f",
            (unsigned long)total_heat, max_heat, (double)total_heat/num_nodes);

    GPU_CHECK(cudaFree(d_nodes));
    GPU_CHECK(cudaFree(d_ch));
    GPU_CHECK(cudaFree(d_qkeys));
    GPU_CHECK(cudaFree(d_results));
    GPU_CHECK(cudaFree(d_heat));
#else
    INSPECT("M080_SKIP", "WALKING_CUDA=0, GPU experiments skipped");
    // CPU-only: verify lookup correctness
    uint64_t cpu_correct = 0;
    for(uint64_t i=0;i<NQ/2;i++){
        int64_t r = lookup_cpu(tree, qkeys[i]);
        if(r >= 0) cpu_correct++;
    }
    INSPECT("M080_CPU_VERIFY", "correct_hits=%lu/%lu", (unsigned long)cpu_correct, (unsigned long)(NQ/2));
#endif
}


// ════════════════════════════════════════════════════════════════
// §7  M081 — Merge-path intersect
//     [MOD] single-thread gallop → P-thread parallel merge-path
//     [NEW] merge_path_partition: diagonal binary search
//     [NEW] kern_merge_path_intersect: two-pass (count + write)
// ════════════════════════════════════════════════════════════════

// ── 7.1  CPU merge-path partition (reference + verification) ──
// [NEW] Given sorted A[na] and B[nb], and diagonal d (0-based),
//       find (i,j) such that i+j=d, A[i-1]<=B[j], B[j-1]<=A[i]
// This is the merge-path partition point on diagonal d.
static void merge_path_partition_cpu(
    const uint64_t* a, uint32_t na,
    const uint64_t* b, uint32_t nb,
    uint32_t diag,
    uint32_t* out_i, uint32_t* out_j)
{
    // Binary search on the diagonal
    // i ranges from max(0, diag-nb) to min(diag, na)
    uint32_t lo = (diag > nb) ? diag - nb : 0;
    uint32_t hi = (diag < na) ? diag : na;

    while(lo < hi){
        uint32_t mid = lo + (hi - lo) / 2;
        uint32_t j_mid = diag - mid;
        // Compare A[mid] vs B[j_mid - 1]
        // We want A[mid] >= B[j_mid-1] (merge-path condition)
        if(j_mid > 0 && a[mid] < b[j_mid - 1]){
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *out_i = lo;
    *out_j = diag - lo;
}

// [NEW] CPU merge-path intersect for verification
static std::vector<uint64_t> intersect_merge_path_cpu(
    const uint64_t* a, uint32_t na,
    const uint64_t* b, uint32_t nb,
    uint32_t num_partitions)
{
    uint32_t total = na + nb;
    uint32_t partition_size = (total + num_partitions - 1) / num_partitions;

    // Compute partition points
    std::vector<uint32_t> pi(num_partitions + 1), pj(num_partitions + 1);
    for(uint32_t p = 0; p <= num_partitions; p++){
        uint32_t diag = std::min(p * partition_size, total);
        merge_path_partition_cpu(a, na, b, nb, diag, &pi[p], &pj[p]);
    }

    if(g_dbg >= 2){
        for(uint32_t p = 0; p <= num_partitions; p++){
            INSPECT("MPATH_PART", "p=%u diag=%u → i=%u j=%u",
                    p, std::min(p*partition_size, total), pi[p], pj[p]);
        }
    }

    // Each partition: linear merge within its (i,j) range
    std::vector<uint64_t> result;
    for(uint32_t p = 0; p < num_partitions; p++){
        uint32_t ia = pi[p], ib = pj[p];
        uint32_t ia_end = pi[p+1], ib_end = pj[p+1];

        while(ia < ia_end && ib < ib_end){
            if(a[ia] == b[ib]){
                result.push_back(a[ia]);
                ia++; ib++;
            } else if(a[ia] < b[ib]){
                ia++;
            } else {
                ib++;
            }
        }
    }
    return result;
}


#if WALKING_CUDA

// ── 7.2  GPU merge-path partition (device function) ──
// [NEW] Exact GPU port of the diagonal binary search
__device__ void merge_path_partition_gpu(
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb,
    uint32_t diag,
    uint32_t* out_i, uint32_t* out_j)
{
    uint32_t lo = (diag > nb) ? diag - nb : 0;
    uint32_t hi = (diag < na) ? diag : na;

    while(lo < hi){
        uint32_t mid = lo + (hi - lo) / 2;
        uint32_t j_mid = diag - mid;
        if(j_mid > 0 && a[mid] < b[j_mid - 1]){
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *out_i = lo;
    *out_j = diag - lo;
}

// ── 7.3  Pass 1: count matches per partition ──
// [NEW] Each thread handles one partition, counts intersections
__global__ void kern_merge_path_count(
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb,
    uint32_t        partition_size,
    uint32_t        num_partitions,
    uint32_t*       __restrict__ counts)   // per-partition match count
{
    uint32_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if(pid >= num_partitions) return;

    uint32_t total = na + nb;
    uint32_t diag_lo = pid * partition_size;
    uint32_t diag_hi = min((pid + 1) * partition_size, total);

    // Find partition boundaries
    uint32_t ia, ib, ia_end, ib_end;
    merge_path_partition_gpu(a, na, b, nb, diag_lo, &ia, &ib);
    merge_path_partition_gpu(a, na, b, nb, diag_hi, &ia_end, &ib_end);

    // Count matches in this partition
    uint32_t cnt = 0;
    while(ia < ia_end && ib < ib_end){
        if(a[ia] == b[ib]){
            cnt++; ia++; ib++;
        } else if(a[ia] < b[ib]){
            ia++;
        } else {
            ib++;
        }
    }
    counts[pid] = cnt;
}

// ── 7.4  Pass 2: write matches per partition ──
// [NEW] Each thread writes its matches at prefix-sum offset
__global__ void kern_merge_path_write(
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb,
    uint32_t        partition_size,
    uint32_t        num_partitions,
    const uint32_t* __restrict__ offsets,  // exclusive prefix sum of counts
    uint64_t*       __restrict__ out)
{
    uint32_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if(pid >= num_partitions) return;

    uint32_t total = na + nb;
    uint32_t diag_lo = pid * partition_size;
    uint32_t diag_hi = min((pid + 1) * partition_size, total);

    uint32_t ia, ib, ia_end, ib_end;
    merge_path_partition_gpu(a, na, b, nb, diag_lo, &ia, &ib);
    merge_path_partition_gpu(a, na, b, nb, diag_hi, &ia_end, &ib_end);

    uint32_t write_pos = offsets[pid];
    while(ia < ia_end && ib < ib_end){
        if(a[ia] == b[ib]){
            out[write_pos++] = a[ia];
            ia++; ib++;
        } else if(a[ia] < b[ib]){
            ia++;
        } else {
            ib++;
        }
    }
}

// ── 7.5  Prefix sum kernel (simple single-block) ──
// [NEW] Exclusive prefix sum for partition counts → offsets
__global__ void kern_prefix_sum(
    const uint32_t* __restrict__ counts,
    uint32_t*       __restrict__ offsets,
    uint32_t        n,
    uint32_t*       __restrict__ total_out)
{
    // Single-thread prefix sum (sufficient for moderate partition counts)
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t sum = 0;
    for(uint32_t i = 0; i < n; i++){
        offsets[i] = sum;
        sum += counts[i];
    }
    *total_out = sum;
}

// [KEEP] Single-thread gallop kernel (baseline from walking_gpu_tree.cu)
__global__ void kern_gallop_intersect(
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb,
    uint64_t*       __restrict__ out,
    uint32_t*       __restrict__ out_count)
{
    if(threadIdx.x!=0 || blockIdx.x!=0) return;

    bool use_gallop = (na > 4*nb) || (nb > 4*na);
    uint32_t ia=0, ib=0, cnt=0;

    while(ia<na && ib<nb){
        if(a[ia]==b[ib]){
            out[cnt++]=a[ia]; ia++; ib++;
        } else if(a[ia]<b[ib]){
            if(use_gallop && na>4*nb){
                uint32_t step=1, pos=ia;
                uint64_t target=b[ib];
                while(pos<na && a[pos]<target){ pos+=step; step<<=1; }
                uint32_t blo=(step>1)?(pos-(step>>1)):ia;
                uint32_t bhi=(pos<na)?pos:na;
                while(blo<bhi){
                    uint32_t mid=blo+(bhi-blo)/2;
                    if(a[mid]<target) blo=mid+1; else bhi=mid;
                }
                ia=blo;
            } else { ia++; }
        } else {
            if(use_gallop && nb>4*na){
                uint32_t step=1, pos=ib;
                uint64_t target=a[ia];
                while(pos<nb && b[pos]<target){ pos+=step; step<<=1; }
                uint32_t blo=(step>1)?(pos-(step>>1)):ib;
                uint32_t bhi=(pos<nb)?pos:nb;
                while(blo<bhi){
                    uint32_t mid=blo+(bhi-blo)/2;
                    if(b[mid]<target) blo=mid+1; else bhi=mid;
                }
                ib=blo;
            } else { ib++; }
        }
    }
    *out_count = cnt;
}

#endif // WALKING_CUDA


// ── 7.6  GPU merge-path driver: allocate, launch two passes, download ──
#if WALKING_CUDA
static std::vector<uint64_t> intersect_merge_path_gpu(
    const uint64_t* h_a, uint32_t na,
    const uint64_t* h_b, uint32_t nb,
    uint32_t num_partitions)
{
    INSPECT("M081_GPU_MPATH", "na=%u nb=%u partitions=%u", na, nb, num_partitions);

    uint64_t* d_a=nullptr; uint64_t* d_b=nullptr;
    GPU_CHECK(cudaMalloc(&d_a, na*sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_b, nb*sizeof(uint64_t)));
    GPU_CHECK(cudaMemcpy(d_a, h_a, na*sizeof(uint64_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_b, h_b, nb*sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint32_t total = na + nb;
    uint32_t partition_size = (total + num_partitions - 1) / num_partitions;

    INSPECT("M081_MPATH_PARAMS", "total=%u partition_size=%u", total, partition_size);

    // Allocate counts, offsets, total
    uint32_t* d_counts=nullptr; uint32_t* d_offsets=nullptr; uint32_t* d_total=nullptr;
    GPU_CHECK(cudaMalloc(&d_counts, num_partitions*sizeof(uint32_t)));
    GPU_CHECK(cudaMalloc(&d_offsets, num_partitions*sizeof(uint32_t)));
    GPU_CHECK(cudaMalloc(&d_total, sizeof(uint32_t)));

    // Pass 1: count
    {
        int blk = 256;
        int grd = (num_partitions + blk - 1) / blk;
        INSPECT("M081_PASS1", "grid=%d block=%d", grd, blk);
        kern_merge_path_count<<<grd, blk>>>(d_a, na, d_b, nb, partition_size, num_partitions, d_counts);
        GPU_CHECK(cudaDeviceSynchronize());
    }

    // Prefix sum
    kern_prefix_sum<<<1,1>>>(d_counts, d_offsets, num_partitions, d_total);
    GPU_CHECK(cudaDeviceSynchronize());

    uint32_t h_total = 0;
    GPU_CHECK(cudaMemcpy(&h_total, d_total, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    INSPECT("M081_TOTAL_MATCHES", "total=%u", h_total);

    // Allocate output
    uint64_t* d_out=nullptr;
    if(h_total > 0){
        GPU_CHECK(cudaMalloc(&d_out, h_total*sizeof(uint64_t)));
    } else {
        // Allocate dummy
        GPU_CHECK(cudaMalloc(&d_out, sizeof(uint64_t)));
    }

    // Debug: dump per-partition counts
    if(g_dbg >= 2 && num_partitions <= 64){
        std::vector<uint32_t> h_counts(num_partitions), h_offsets(num_partitions);
        GPU_CHECK(cudaMemcpy(h_counts.data(), d_counts, num_partitions*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(h_offsets.data(), d_offsets, num_partitions*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for(uint32_t p=0; p<std::min(num_partitions, 8u); p++){
            INSPECT("M081_PART_COUNT", "p=%u count=%u offset=%u", p, h_counts[p], h_offsets[p]);
        }
    }

    // Pass 2: write
    {
        int blk = 256;
        int grd = (num_partitions + blk - 1) / blk;
        INSPECT("M081_PASS2", "grid=%d block=%d", grd, blk);
        kern_merge_path_write<<<grd, blk>>>(d_a, na, d_b, nb, partition_size, num_partitions, d_offsets, d_out);
        GPU_CHECK(cudaDeviceSynchronize());
    }

    // Download result
    std::vector<uint64_t> result(h_total);
    if(h_total > 0){
        GPU_CHECK(cudaMemcpy(result.data(), d_out, h_total*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    }

    GPU_CHECK(cudaFree(d_a));
    GPU_CHECK(cudaFree(d_b));
    GPU_CHECK(cudaFree(d_counts));
    GPU_CHECK(cudaFree(d_offsets));
    GPU_CHECK(cudaFree(d_total));
    GPU_CHECK(cudaFree(d_out));

    return result;
}

// [NEW] GPU gallop intersect driver (for comparison)
static std::vector<uint64_t> intersect_gallop_gpu(
    const uint64_t* h_a, uint32_t na,
    const uint64_t* h_b, uint32_t nb)
{
    uint64_t *d_a=nullptr, *d_b=nullptr, *d_out=nullptr;
    uint32_t *d_cnt=nullptr;
    uint32_t max_out = std::min(na, nb);

    GPU_CHECK(cudaMalloc(&d_a, na*sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_b, nb*sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_out, max_out*sizeof(uint64_t)));
    GPU_CHECK(cudaMalloc(&d_cnt, sizeof(uint32_t)));
    GPU_CHECK(cudaMemcpy(d_a, h_a, na*sizeof(uint64_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_b, h_b, nb*sizeof(uint64_t), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemset(d_cnt, 0, sizeof(uint32_t)));

    kern_gallop_intersect<<<1,1>>>(d_a, na, d_b, nb, d_out, d_cnt);
    GPU_CHECK(cudaDeviceSynchronize());

    uint32_t h_cnt=0;
    GPU_CHECK(cudaMemcpy(&h_cnt, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> result(h_cnt);
    if(h_cnt>0)
        GPU_CHECK(cudaMemcpy(result.data(), d_out, h_cnt*sizeof(uint64_t), cudaMemcpyDeviceToHost));

    GPU_CHECK(cudaFree(d_a));
    GPU_CHECK(cudaFree(d_b));
    GPU_CHECK(cudaFree(d_out));
    GPU_CHECK(cudaFree(d_cnt));
    return result;
}
#endif // WALKING_CUDA


// ── 7.7  Experiment driver ──
// [NEW] Test different sizes and skew ratios

struct IntersectTestCase {
    const char* label;
    uint32_t na;
    uint32_t nb;
};

static void exp_merge_path_intersect() {
    sep("M081: MERGE-PATH INTERSECT");

    IntersectTestCase cases[] = {
        {"1K_1:1",    1000,    1000},
        {"1K_1:5",    1000,    5000},
        {"1K_1:20",   1000,   20000},
        {"10K_1:1",  10000,   10000},
        {"10K_1:5",  10000,   50000},
        {"100K_1:1", 100000, 100000},
        {"100K_1:5", 100000, 500000},
        {"1M_1:1",  1000000, 1000000},
    };

    std::mt19937_64 rng(1337);

    for(auto& tc : cases) {
        std::printf("── %s (na=%u nb=%u) ──", tc.label, tc.na, tc.nb);

        // Generate sorted arrays with ~10% overlap
        uint64_t range = (uint64_t)std::max(tc.na, tc.nb) * 5;
        std::vector<uint64_t> a(tc.na), b(tc.nb);
        for(uint32_t i=0;i<tc.na;i++) a[i] = rng() % range;
        for(uint32_t i=0;i<tc.nb;i++) b[i] = rng() % range;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        tc.na = (uint32_t)a.size();
        tc.nb = (uint32_t)b.size();

        INSPECT("M081_ARRAYS", "%s actual_na=%u actual_nb=%u", tc.label, tc.na, tc.nb);

        // CPU gallop reference
        std::vector<uint64_t> cpu_gallop;
        double cpu_gallop_ms;
        {
            Timer t("CPU_GALLOP");
            cpu_gallop = intersect_cpu(a.data(), tc.na, b.data(), tc.nb);
            cpu_gallop_ms = t.ms();
        }
        INSPECT("M081_CPU_GALLOP", "%s hits=%lu time=%.2fms",
                tc.label, (unsigned long)cpu_gallop.size(), cpu_gallop_ms);

        // CPU merge-path reference
        uint32_t num_parts = 32;
        std::vector<uint64_t> cpu_mpath;
        double cpu_mpath_ms;
        {
            Timer t("CPU_MERGE_PATH");
            cpu_mpath = intersect_merge_path_cpu(a.data(), tc.na, b.data(), tc.nb, num_parts);
            cpu_mpath_ms = t.ms();
        }
        INSPECT("M081_CPU_MPATH", "%s hits=%lu time=%.2fms partitions=%u",
                tc.label, (unsigned long)cpu_mpath.size(), cpu_mpath_ms, num_parts);

        // Verify CPU merge-path == CPU gallop
        CHK(cpu_mpath.size() == cpu_gallop.size(), "M081_MPATH_SIZE",
            "%s mpath=%lu gallop=%lu", tc.label,
            (unsigned long)cpu_mpath.size(), (unsigned long)cpu_gallop.size());
        if(cpu_mpath.size() == cpu_gallop.size()){
            bool all_match = true;
            for(size_t i=0;i<cpu_mpath.size();i++){
                if(cpu_mpath[i] != cpu_gallop[i]){ all_match=false; break; }
            }
            CHK(all_match, "M081_MPATH_VALUES", "%s value mismatch", tc.label);
        }

#if WALKING_CUDA
        // GPU gallop (baseline)
        double gpu_gallop_ms;
        std::vector<uint64_t> gpu_gallop;
        {
            Timer t("GPU_GALLOP");
            gpu_gallop = intersect_gallop_gpu(a.data(), tc.na, b.data(), tc.nb);
            gpu_gallop_ms = t.ms();
        }
        CHK(gpu_gallop.size() == cpu_gallop.size(), "M081_GPU_GALLOP_SIZE",
            "%s gpu=%lu cpu=%lu", tc.label,
            (unsigned long)gpu_gallop.size(), (unsigned long)cpu_gallop.size());
        INSPECT("M081_GPU_GALLOP", "%s hits=%lu time=%.2fms",
                tc.label, (unsigned long)gpu_gallop.size(), gpu_gallop_ms);

        // GPU merge-path
        // Scale partitions with data size
        uint32_t gpu_parts = std::max(32u, (tc.na + tc.nb) / 1024);
        gpu_parts = std::min(gpu_parts, 4096u);
        double gpu_mpath_ms;
        std::vector<uint64_t> gpu_mpath;
        {
            Timer t("GPU_MERGE_PATH");
            gpu_mpath = intersect_merge_path_gpu(a.data(), tc.na, b.data(), tc.nb, gpu_parts);
            gpu_mpath_ms = t.ms();
        }
        CHK(gpu_mpath.size() == cpu_gallop.size(), "M081_GPU_MPATH_SIZE",
            "%s gpu_mpath=%lu cpu=%lu", tc.label,
            (unsigned long)gpu_mpath.size(), (unsigned long)cpu_gallop.size());
        if(gpu_mpath.size() == cpu_gallop.size()){
            bool all_match = true;
            for(size_t i=0;i<gpu_mpath.size();i++){
                if(gpu_mpath[i] != cpu_gallop[i]){ all_match=false; break; }
            }
            CHK(all_match, "M081_GPU_MPATH_VALUES", "%s value mismatch", tc.label);
        }
        INSPECT("M081_GPU_MPATH", "%s hits=%lu time=%.2fms partitions=%u",
                tc.label, (unsigned long)gpu_mpath.size(), gpu_mpath_ms, gpu_parts);

        double speedup_vs_gallop = (gpu_mpath_ms > 0) ? gpu_gallop_ms / gpu_mpath_ms : 0.0;
        INSPECT("M081_SPEEDUP", "%s gallop=%.2fms mpath=%.2fms ratio=%.2fx",
                tc.label, gpu_gallop_ms, gpu_mpath_ms, speedup_vs_gallop);
#endif // WALKING_CUDA
    }
}


// ════════════════════════════════════════════════════════════════
// §8  M082 — Multi-GPU ART partition
//     [NEW] hash(prefix_byte) % num_gpus assigns subtrees
//     [NEW] Simulated multi-GPU (single-device with separate trees)
// ════════════════════════════════════════════════════════════════

// ── 8.1  MultiGPUART structure ──
// [NEW] Partitions a FlatART into num_gpus sub-trees by first key byte
struct MultiGPUART {
    uint32_t num_gpus;
    std::vector<FlatART> sub_trees;           // one per simulated GPU
    std::vector<std::vector<uint64_t>> sub_keys;
    std::vector<std::vector<int64_t>>  sub_vals;

    // [NEW] route_query: key's first byte → target GPU
    static uint32_t route_query(uint64_t key, uint32_t num_gpus) {
        uint8_t first_byte = (key >> 56) & 0xFF;
        // [MOD] Simple modular hash; could use more sophisticated mapping
        return first_byte % num_gpus;
    }

    // [NEW] Build by partitioning keys across GPUs
    void build(const std::vector<uint64_t>& keys, const std::vector<int64_t>& vals,
               uint32_t n_gpus) {
        Timer t("MULTI_GPU_BUILD");
        num_gpus = n_gpus;
        sub_keys.resize(num_gpus);
        sub_vals.resize(num_gpus);
        sub_trees.resize(num_gpus);

        // Clear
        for(uint32_t g=0; g<num_gpus; g++){
            sub_keys[g].clear();
            sub_vals[g].clear();
        }

        // Scatter keys to GPUs
        for(size_t i=0; i<keys.size(); i++){
            uint32_t gpu = route_query(keys[i], num_gpus);
            sub_keys[gpu].push_back(keys[i]);
            sub_vals[gpu].push_back(vals[i]);
            if(g_dbg >= 3 && i < 10){
                INSPECT("M082_ROUTE", "key=0x%016lX byte=0x%02X → gpu=%u",
                        (unsigned long)keys[i], (unsigned)(keys[i]>>56)&0xFF, gpu);
            }
        }

        // Build sub-trees
        for(uint32_t g=0; g<num_gpus; g++){
            // Sort within partition (keys may interleave across partitions)
            std::vector<size_t> idx(sub_keys[g].size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
                return sub_keys[g][a] < sub_keys[g][b];
            });
            std::vector<uint64_t> sk(sub_keys[g].size());
            std::vector<int64_t>  sv(sub_vals[g].size());
            for(size_t i=0;i<idx.size();i++){
                sk[i] = sub_keys[g][idx[i]];
                sv[i] = sub_vals[g][idx[i]];
            }
            // Deduplicate
            auto it = std::unique(sk.begin(), sk.end());
            size_t unique_n = it - sk.begin();
            sk.resize(unique_n);
            sv.resize(unique_n);
            sub_keys[g] = sk;
            sub_vals[g] = sv;

            INSPECT("M082_PARTITION", "gpu=%u keys=%lu",
                    g, (unsigned long)sub_keys[g].size());
            if(!sub_keys[g].empty()){
                sub_trees[g].build(sub_keys[g], sub_vals[g]);
            }
        }
    }

    // [NEW] CPU lookup across partitions
    int64_t lookup(uint64_t key) const {
        uint32_t gpu = route_query(key, num_gpus);
        if(sub_trees[gpu].nodes.empty()) return -1;
        return lookup_cpu(sub_trees[gpu], key);
    }
};


// ── 8.2  GPU multi-GPU lookup kernel ──
// [NEW] kern_multi_gpu_lookup: one tree on device at a time (simulated)
//       In production, each sub-tree lives on a separate physical GPU
#if WALKING_CUDA
__global__ void kern_multi_gpu_lookup(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,
    uint32_t          num_nodes,
    const uint64_t*   __restrict__ qkeys,
    int64_t*          __restrict__ results,
    uint64_t          nq,
    uint32_t          root)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if(tid >= nq) return;
    uint64_t key = qkeys[tid];
    uint32_t cur = root;
    int64_t  res = -1;

    for(uint8_t d=0; d<8; d++){
        if(cur >= num_nodes) break;
        const FlatNode& n = nodes[cur];
        if(n.type == FLEAF){
            if(n.key == key) res = n.value;
            break;
        }
        uint8_t byte = (key >> (56 - d*8)) & 0xFF;
        uint32_t next = UINT32_MAX;
        switch(n.type){
        case FNODE4: case FNODE16:
            for(uint16_t i=0; i<n.num_children; i++)
                if(n.child_keys[i]==byte){ next=ch[n.children_start+i]; break; }
            break;
        case FNODE48: {
            uint8_t idx = n.child_keys[byte];
            if(idx) next = ch[n.children_start + idx - 1];
            break;
        }
        case FNODE256:
            next = ch[n.children_start + byte];
            break;
        }
        if(next == UINT32_MAX) break;
        cur = next;
    }
    results[tid] = res;
}
#endif


// ── 8.3  Experiment driver ──
// [NEW] Simulate 1/2/4 GPUs, test partition balance and correctness
static void exp_multi_gpu_partition() {
    sep("M082: MULTI-GPU ART PARTITION");

    const uint64_t N = 200000;
    const uint64_t NQ = 50000;
    std::mt19937_64 rng(7777);

    std::vector<uint64_t> keys(N);
    std::vector<int64_t>  vals(N);
    for(uint64_t i=0;i<N;i++){
        keys[i] = rng();
        vals[i] = (int64_t)i;
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    vals.resize(keys.size());
    for(size_t i=0;i<vals.size();i++) vals[i]=(int64_t)i;
    uint64_t actual_n = keys.size();

    INSPECT("M082_SETUP", "n_keys=%lu n_queries=%lu", (unsigned long)actual_n, (unsigned long)NQ);

    // Full tree for reference
    FlatART full_tree;
    full_tree.build(keys, vals);

    // Generate queries: 60% hits, 40% misses
    std::vector<uint64_t> qkeys(NQ);
    for(uint64_t i=0;i<NQ;i++){
        if(i < NQ*3/5) qkeys[i] = keys[rng() % actual_n];
        else           qkeys[i] = rng();
    }

    // CPU reference on full tree
    std::vector<int64_t> cpu_ref(NQ);
    {
        Timer t("M082_CPU_FULL");
        for(uint64_t i=0;i<NQ;i++) cpu_ref[i] = lookup_cpu(full_tree, qkeys[i]);
    }
    uint64_t ref_hits=0;
    for(auto v:cpu_ref) if(v>=0) ref_hits++;
    INSPECT("M082_CPU_REF", "hits=%lu misses=%lu", (unsigned long)ref_hits, (unsigned long)(NQ-ref_hits));

    // Test with 1, 2, 4 simulated GPUs
    uint32_t gpu_configs[] = {1, 2, 4};
    for(uint32_t ng : gpu_configs){
        std::printf("── %u GPU(s) ──", ng);

        MultiGPUART mgpu;
        mgpu.build(keys, vals, ng);

        // Partition balance stats
        uint64_t total_keys = 0;
        uint64_t max_part = 0, min_part = UINT64_MAX;
        for(uint32_t g=0; g<ng; g++){
            uint64_t pk = mgpu.sub_keys[g].size();
            total_keys += pk;
            max_part = std::max(max_part, pk);
            min_part = std::min(min_part, pk);
            INSPECT("M082_BALANCE", "gpu=%u/%u keys=%lu nodes=%lu",
                    g, ng, (unsigned long)pk, (unsigned long)mgpu.sub_trees[g].nodes.size());
        }
        double balance_ratio = (min_part > 0) ? (double)max_part / min_part : 999.0;
        INSPECT("M082_BALANCE_RATIO", "gpus=%u max=%lu min=%lu ratio=%.2f",
                ng, (unsigned long)max_part, (unsigned long)min_part, balance_ratio);

        // CPU multi-GPU lookup
        std::vector<int64_t> mgpu_results(NQ);
        double mgpu_cpu_ms;
        {
            Timer t("M082_MGPU_CPU");
            for(uint64_t i=0;i<NQ;i++){
                mgpu_results[i] = mgpu.lookup(qkeys[i]);
            }
            mgpu_cpu_ms = t.ms();
        }

        // Verify vs full tree
        uint64_t match=0;
        for(uint64_t i=0;i<NQ;i++){
            if(mgpu_results[i] == cpu_ref[i]) match++;
        }
        CHK(match==NQ, "M082_MGPU_VS_FULL", "gpus=%u match=%lu/%lu",
            ng, (unsigned long)match, (unsigned long)NQ);
        INSPECT("M082_CPU_MGPU", "gpus=%u match=%lu/%lu time=%.2fms",
                ng, (unsigned long)match, (unsigned long)NQ, mgpu_cpu_ms);

#if WALKING_CUDA
        // GPU multi-GPU lookup: scatter queries by GPU, launch per-GPU
        double gpu_total_ms = 0;
        std::vector<int64_t> gpu_mgpu_results(NQ, -1);

        // Scatter queries
        std::vector<std::vector<uint64_t>> gpu_qkeys(ng);
        std::vector<std::vector<uint64_t>> gpu_qidx(ng);  // original index
        for(uint64_t i=0;i<NQ;i++){
            uint32_t target_gpu = MultiGPUART::route_query(qkeys[i], ng);
            gpu_qkeys[target_gpu].push_back(qkeys[i]);
            gpu_qidx[target_gpu].push_back(i);
            if(g_dbg>=3 && i<5){
                INSPECT("M082_SCATTER", "query[%lu] key=0x%016lX → gpu=%u",
                        (unsigned long)i, (unsigned long)qkeys[i], target_gpu);
            }
        }

        for(uint32_t g=0; g<ng; g++){
            uint64_t nq_g = gpu_qkeys[g].size();
            INSPECT("M082_GPU_LAUNCH", "gpu=%u/%u queries=%lu nodes=%lu",
                    g, ng, (unsigned long)nq_g, (unsigned long)mgpu.sub_trees[g].nodes.size());
            if(nq_g == 0 || mgpu.sub_trees[g].nodes.empty()) continue;

            FlatNode* d_nodes=nullptr;
            uint32_t* d_ch=nullptr;
            uint64_t* d_qk=nullptr;
            int64_t*  d_res=nullptr;

            uint32_t num_nodes_g = (uint32_t)mgpu.sub_trees[g].nodes.size();
            size_t node_bytes = num_nodes_g * sizeof(FlatNode);
            size_t ch_bytes = mgpu.sub_trees[g].children.size() * sizeof(uint32_t);

            GPU_CHECK(cudaMalloc(&d_nodes, node_bytes));
            GPU_CHECK(cudaMalloc(&d_ch, ch_bytes));
            GPU_CHECK(cudaMalloc(&d_qk, nq_g*sizeof(uint64_t)));
            GPU_CHECK(cudaMalloc(&d_res, nq_g*sizeof(int64_t)));

            GPU_CHECK(cudaMemcpy(d_nodes, mgpu.sub_trees[g].nodes.data(), node_bytes, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_ch, mgpu.sub_trees[g].children.data(), ch_bytes, cudaMemcpyHostToDevice));
            GPU_CHECK(cudaMemcpy(d_qk, gpu_qkeys[g].data(), nq_g*sizeof(uint64_t), cudaMemcpyHostToDevice));

            Timer tg("M082_GPU_KERN");
            uint32_t blk = 256;
            uint32_t grid = (uint32_t)((nq_g + blk - 1) / blk);
            INSPECT("M082_KERN_LAUNCH", "gpu=%u grid=%u blk=%u nq=%lu root=%u",
                    g, grid, blk, (unsigned long)nq_g, mgpu.sub_trees[g].root);

            kern_multi_gpu_lookup<<<grid, blk>>>(
                d_nodes, d_ch, num_nodes_g, d_qk, d_res, nq_g, mgpu.sub_trees[g].root);
            cudaDeviceSynchronize();
            gpu_total_ms += tg.ms();

            // Copy results back
            std::vector<int64_t> local_res(nq_g);
            GPU_CHECK(cudaMemcpy(local_res.data(), d_res, nq_g*sizeof(int64_t), cudaMemcpyDeviceToHost));

            // Scatter results back to original indices
            for(uint64_t q=0; q<nq_g; q++){
                gpu_mgpu_results[gpu_qidx[g][q]] = local_res[q];
            }

            cudaFree(d_nodes); cudaFree(d_ch); cudaFree(d_qk); cudaFree(d_res);
        }

        // Verify GPU multi-GPU vs CPU reference
        uint64_t gpu_match=0;
        for(uint64_t i=0;i<NQ;i++){
            if(gpu_mgpu_results[i] == cpu_ref[i]) gpu_match++;
        }
        CHK(gpu_match==NQ, "M082_GPU_MGPU_VS_CPU", "gpus=%u match=%lu/%lu",
            ng, (unsigned long)gpu_match, (unsigned long)NQ);
        INSPECT("M082_GPU_MGPU", "gpus=%u match=%lu/%lu total_gpu_time=%.2fms",
                ng, (unsigned long)gpu_match, (unsigned long)NQ, gpu_total_ms);
#endif
    }
}


} // namespace warp
} // namespace walking


// ════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    using namespace walking::warp;
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║  LLM4Walking — Warp-Cooperative GPU Tree (M080-M082)    ║\n");
#if WALKING_CUDA
    std::printf("║  Mode: CUDA GPU                                         ║\n");
#else
    std::printf("║  Mode: CPU fallback                                     ║\n");
#endif
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    if(argc >= 2) g_dbg = std::stoi(argv[1]);

    std::printf("[SYS] PID=%d cores=%u RSS=%ldKB\n",
                getpid(), std::thread::hardware_concurrency(), rss_kb());
    std::printf("[CFG] debug_level=%d\n\n", g_dbg);

#if WALKING_CUDA
    int dc=0; cudaGetDeviceCount(&dc);
    for(int i=0;i<dc;i++){
        cudaDeviceProp p; cudaGetDeviceProperties(&p,i);
        std::printf("[GPU %d] %s SM=%d.%d mem=%.0fMB SMs=%d\n",
                    i,p.name,p.major,p.minor,p.totalGlobalMem/1048576.0,p.multiProcessorCount);
    }
    std::printf("\n");
#endif

    // ── M080: warp-cooperative find_child ──
    exp_warp_find_child();

    // ── M081: merge-path intersect ──
    exp_merge_path_intersect();

    // ── M082: multi-GPU ART partition ──
    exp_multi_gpu_partition();

    // ── Summary ──
    sep("SUMMARY");
    std::printf("[SUMMARY] inspections=%lu pass=%lu fail=%lu\n", g_insp, g_pass, g_fail);
    std::printf("[SUMMARY] RSS=%ldKB\n", rss_kb());
    if(g_fail > 0) std::printf("[WARN] %lu assertion(s) FAILED\n", g_fail);
    else std::printf("[OK] all passed\n");
    return g_fail > 0 ? 1 : 0;
}

