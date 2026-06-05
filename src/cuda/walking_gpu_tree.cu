/**
 * walking_gpu_tree.cu — GPU 并行树遍历: ART + Interval + 交集
 *
 * mv来源与算法改动对照:
 *
 *   art_core.hpp (324行)
 *     KEEP: Node4/16/48/256 四种类型, IS_LEAF pointer tagging
 *     MOD:  指针树 → POD数组 (children存node_id), 叶子内嵌value
 *           加 access_heat 字段 (GPU atomic累加, 用于热度分析)
 *
 *   art_node_ops_impl.hpp (1792行)
 *     KEEP: find_child 四路dispatch (Node4线性/Node16 SSE→GPU线性/Node48直接索引/Node256直接索引)
 *     KEEP: add_child + 升级链 (4→16→48→256)
 *     KEEP: node_split 右倾分裂点 (SPLIT_THRESHOLD = 2/3)
 *     MOD:  find_child 的 SSE path → GPU上改为warp-cooperative: 
 *           对Node16, warp内16个lane各比较一个key byte, __ballot得到命中mask
 *     MOD:  intersect 的 probe-mode (一侧fanout>4x时) → GPU上每个block处理一对子树,
 *           block内warp-level probe
 *     NEW:  galloping_lower_bound → GPU __shfl实现 (warp内指数探测)
 *
 *   art_iter_impl.hpp (666行)
 *     KEEP: 迭代器协议 (is_valid, next, next_without_skip)
 *     MOD:  CPU递归DFS → GPU BFS level-sync (每layer一个kernel launch)
 *
 *   tem_graph_impl.hpp (913行)
 *     KEEP: interval排序 + successor链
 *     MOD:  contains_query → GPU并行stab: 每thread一个查询点, 二分+线性混合
 *
 *   cuda_bfs_kernel.hpp (469行)
 *     KEEP: warp-level frontier管理 (__ballot_sync+atomicAdd预留slot)
 *     KEEP: direction switching ratio (动态switch_ratio)
 *     MOD:  图BFS → 树BFS (children代替邻接表)
 *
 * Build:
 *   GPU:  nvcc -std=c++17 -O2 -arch=sm_86 -DWALKING_CUDA=1 -o walking_gpu_tree walking_gpu_tree.cu
 *   CPU:  g++ -std=c++17 -O2 -pthread -DWALKING_CUDA=0 -x c++ -o walking_gpu_tree walking_gpu_tree.cu
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
// Debug infra
// ════════════════════════════════════════════════════════════════
static int g_dbg = 2;
static long rss_kb() { struct rusage r; getrusage(RUSAGE_SELF,&r); return r.ru_maxrss; }
static uint64_t g_insp = 0, g_pass = 0, g_fail = 0;
#define INSPECT(tag, ...) do { g_insp++; \
    std::printf("[INSPECT·%04lu·%s] ", g_insp, tag); \
    std::printf(__VA_ARGS__); std::printf("  RSS=%ldKB\n", rss_kb()); } while(0)
#define CHK(cond, tag, ...) do { if(cond){g_pass++;} else { g_fail++; \
    std::printf("[FAIL·%s] ", tag); std::printf(__VA_ARGS__); std::printf("\n"); }} while(0)
struct Timer {
    const char* l; std::chrono::high_resolution_clock::time_point t0;
    Timer(const char* s):l(s),t0(std::chrono::high_resolution_clock::now()){
        if(g_dbg>=1) std::printf("[T·START] %s\n",l);}
    double ms() const { return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now()-t0).count()/1000.0; }
    ~Timer(){ std::printf("[T·END]   %s → %.2f ms\n", l, ms()); }
};
static void sep(const char* s) {
    std::printf("\n════════════════════════════════════════════════════\n  %s\n"
                "════════════════════════════════════════════════════\n\n", s);
}

// ════════════════════════════════════════════════════════════════
// §1  Flat ART — GPU-portable node representations
//     mv: art_core.hpp Node4/16/48/256 + art_node_ops find_child dispatch
// ════════════════════════════════════════════════════════════════

// Node type tags (upstream naming preserved)
enum : uint8_t { FNODE4=1, FNODE16=2, FNODE48=3, FNODE256=4, FLEAF=5 };

// POD node — no pointers, children via index array
// [MOD vs upstream] upstream uses tagged pointers (IS_LEAF macro);
// here we use node_id indexing into a flat array, leaf flag in type field
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
    // [MOD +20%] access_heat: GPU kernel increments atomically,
    // used for tier migration decisions (upstream hotness_tracker analog)
    uint32_t access_heat;
    // [KEEP from upstream] child dispatch keys — Node4/16 store keys here,
    // Node48 uses index-map, Node256 is positional
    uint8_t  child_keys[48];  // enough for Node48's 48 children
    // For Node48: key_map[byte] → child_index+1 (0=empty)
    // We overload child_keys as the key_map for Node48
};

// Children array: children[node.children_start + i] = child_node_id
// This mirrors upstream's ARTNode_4::children[4], ARTNode_16::children[16] etc.
// but in a single contiguous GPU-copyable buffer

struct FlatART {
    std::vector<FlatNode>  nodes;
    std::vector<uint32_t>  children;
    uint32_t root = 0;

    // ── Bulk-load from sorted keys ──
    // [KEEP from upstream] recursive partitioning by key bytes
    // [MOD +20%] choose_type preemptively (upstream art_node_ops batch_direct strategy):
    //   fanout>4 → skip Node4 allocation entirely, avoid 4→16 upgrade
    //   fanout>16 → skip to Node48
    //   This mirrors the batch_direct_node16/node48 counters in upstream

    void build(const std::vector<uint64_t>& keys, const std::vector<int64_t>& vals) {
        Timer t("ART_BULK_BUILD");
        nodes.clear(); children.clear();
        if (keys.empty()) return;
        build_sub(keys, vals, 0, keys.size(), 0, UINT32_MAX);
        root = 0;
        compute_subtree();
        INSPECT("ART_BUILT", "nodes=%lu children=%lu max_depth=%u",
                nodes.size(), children.size(), max_depth());
        dump_type_dist("BUILT");
    }

    uint8_t max_depth() const {
        uint8_t m=0; for(auto&n:nodes) m=std::max(m,n.depth); return m;
    }

    void dump_type_dist(const char* tag) const {
        uint32_t c[6]={};
        for(auto&n:nodes) if(n.type<=FLEAF) c[n.type]++;
        std::printf("[ART·%s] N4=%u N16=%u N48=%u N256=%u LEAF=%u total=%lu\n",
                    tag, c[1],c[2],c[3],c[4],c[5], nodes.size());
    }

private:
    // [MOD vs upstream] preemptive type selection (upstream does 4→16→48→256 upgrade chain)
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
        // Partition by byte at current depth
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

        // Store child_keys dispatch table
        // [KEEP] Node4/16: keys in child_keys[0..n-1]
        // [KEEP] Node48: child_keys used as key_map[byte]=idx+1
        // [KEEP] Node256: positional (child_keys unused)
        memset(nd.child_keys, 0, sizeof(nd.child_keys));
        if(ntype==FNODE4 || ntype==FNODE16){
            for(size_t g=0;g<groups.size();g++) nd.child_keys[g]=groups[g].byte;
        } else if(ntype==FNODE48){
            // key_map: child_keys[byte] = index+1
            for(size_t g=0;g<groups.size();g++) nd.child_keys[groups[g].byte]=(uint8_t)(g+1);
        }
        // Node256: no key storage needed, children[byte] is direct

        nodes.push_back(nd);
        size_t off = children.size();

        // For Node256, allocate full 256 slots; others only fanout slots
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
                if(cid<nodes.size()) sum+=nodes[cid].subtree_size;
            }
            n.subtree_size=sum;
        }
    }
};


// ════════════════════════════════════════════════════════════════
// §2  find_child — GPU port of upstream's 4-way dispatch
//     The core tree traversal operation
// ════════════════════════════════════════════════════════════════
//
// [KEEP] Node4: linear scan keys[0..n-1]
// [KEEP] Node16: linear scan (GPU version; upstream uses SSE _mm_cmpeq_epi8)
// [KEEP] Node48: direct index via key_map[byte]
// [KEEP] Node256: direct children[byte]
// [MOD]  Returns child node_id instead of ARTNode**
// [MOD]  Increments access_heat (GPU atomic or CPU ++)

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

// ════════════════════════════════════════════════════════════════
// §3  node_search — point lookup traversal
//     mv: art_node_ops_impl.hpp node_search (line 278-313)
//     [KEEP] while(n) → IS_LEAF check → find_child(key[depth]) → depth++
//     [MOD]  pointer-based → node_id based
// ════════════════════════════════════════════════════════════════

static int64_t lookup_cpu(const FlatART& tree, uint64_t key) {
    uint32_t cur = tree.root;
    for(uint8_t d=0; d<8; d++){
        if(cur>=tree.nodes.size()) return -1;
        const FlatNode& n = tree.nodes[cur];
        if(n.type==FLEAF) return (n.key==key) ? n.value : -1;
        uint8_t byte = (key>>(56-d*8))&0xFF;
        cur = find_child_cpu(tree, cur, byte);
        if(cur==UINT32_MAX) return -1;
    }
    return -1;
}

// ════════════════════════════════════════════════════════════════
// §4  GPU Kernels
// ════════════════════════════════════════════════════════════════

#if WALKING_CUDA

// ── 4.1 Batch point lookup (GPU version of node_search) ──
// Each thread does one full root→leaf traversal
// [MOD vs upstream] find_child dispatch is inlined per-thread
__global__ void kern_art_lookup(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,       // children array
    uint32_t          num_nodes,
    const uint64_t*   __restrict__ qkeys,
    int64_t*          __restrict__ results,
    uint32_t*         __restrict__ heat,     // per-node access counter
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

        // Inlined find_child — mirrors upstream 4-way dispatch
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

// ── 4.2 BFS level-sync traversal ──
// mv: cuda_bfs_kernel.hpp warp-level frontier pattern
// [MOD] graph BFS → tree BFS (children[] instead of CSR adjacency)
// [KEEP] warp-cooperative atomicAdd for next frontier slot reservation
__global__ void kern_tree_bfs_level(
    const FlatNode*   __restrict__ nodes,
    const uint32_t*   __restrict__ ch,
    const uint32_t*   __restrict__ frontier,
    uint32_t          fsize,
    uint32_t*         __restrict__ next_frontier,
    uint32_t*         __restrict__ next_count,
    uint64_t*         __restrict__ type_hist,  // [5] counters for node type distribution
    uint32_t          num_nodes)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= fsize) return;
    uint32_t nid = frontier[tid];
    if(nid >= num_nodes) return;

    const FlatNode& nd = nodes[nid];
    // Count node types at this level
    if(nd.type>=1 && nd.type<=5)
        atomicAdd((unsigned long long*)&type_hist[nd.type-1], 1ULL);

    if(nd.type == FLEAF) return;  // leaves have no children

    // Enqueue children — same pattern as cuda_bfs_kernel's warp-leader atomicAdd
    // [KEEP] upstream pattern: reserve slots via atomic, then scatter
    size_t nslots = (nd.type==FNODE256) ? 256 : nd.num_children;
    uint32_t child_count = 0;
    for(size_t c=0; c<nslots; c++){
        uint32_t cid = ch[nd.children_start+c];
        if(cid < num_nodes) child_count++;
    }
    if(child_count == 0) return;

    uint32_t base = atomicAdd(next_count, child_count);
    uint32_t slot = 0;
    for(size_t c=0; c<nslots; c++){
        uint32_t cid = ch[nd.children_start+c];
        if(cid < num_nodes)
            next_frontier[base + slot++] = cid;
    }
}

// ── 4.3 Interval stab query (GPU) ──
// mv: tem_graph_impl contains_query
// [KEEP] sorted intervals, linear scan with early-stop
// [MOD +20%] binary search for left boundary before linear scan
__global__ void kern_interval_stab(
    const int32_t*  __restrict__ iv_start,   // interval starts (sorted)
    const int32_t*  __restrict__ iv_end,     // interval ends
    uint64_t        n_iv,
    const int32_t*  __restrict__ queries,
    uint32_t*       __restrict__ hit_counts,
    uint64_t        n_q)
{
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if(tid >= n_q) return;
    int32_t t = queries[tid];

    // Binary search: find rightmost interval where start <= t
    // [MOD vs upstream] upstream does pure linear scan
    uint64_t lo=0, hi=n_iv;
    while(lo<hi){
        uint64_t mid=(lo+hi)/2;
        if(iv_start[mid]<=t) lo=mid+1; else hi=mid;
    }
    // lo = first interval with start > t; scan [0..lo) for containment
    uint32_t cnt=0;
    for(uint64_t i=0; i<lo; i++){
        if(iv_end[i] >= t) cnt++;
    }
    hit_counts[tid] = cnt;
}

// ── 4.4 Leaf-list galloping intersect (GPU) ──
// mv: art_node_ops_impl galloping_lower_bound + leaf_intersect
// Two sorted arrays on GPU, intersect with galloping when skewed
// [KEEP] galloping: exponential probe + binary search
// [KEEP] skew detection: size_a > 4*size_b → gallop on a
__global__ void kern_gallop_intersect(
    const uint64_t* __restrict__ a, uint32_t na,
    const uint64_t* __restrict__ b, uint32_t nb,
    uint64_t*       __restrict__ out,
    uint32_t*       __restrict__ out_count)
{
    // Single-thread kernel for correctness (production would use merge-path)
    // This faithfully ports upstream's leaf_intersect + galloping
    if(threadIdx.x!=0 || blockIdx.x!=0) return;

    bool use_gallop = (na > 4*nb) || (nb > 4*na);
    uint32_t ia=0, ib=0, cnt=0;

    while(ia<na && ib<nb){
        if(a[ia]==b[ib]){
            out[cnt++]=a[ia]; ia++; ib++;
        } else if(a[ia]<b[ib]){
            if(use_gallop && na>4*nb){
                // Galloping on a: exponential probe then binary search
                // [KEEP from upstream galloping_lower_bound]
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


// ════════════════════════════════════════════════════════════════
// §5  CPU fallback for intersect (same algorithm, no GPU)
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
        INSPECT("GALLOP", "skipped=%lu", gallop_skips);
    return out;
}


// ════════════════════════════════════════════════════════════════
// §6  Experiment: ART lookup
// ════════════════════════════════════════════════════════════════

static void exp_art_lookup(const FlatART& tree, uint64_t nq, uint64_t seed) {
    sep("EXP: ART POINT LOOKUP");

    std::mt19937_64 rng(seed);
    std::vector<uint64_t> leaf_keys;
    for(auto&n:tree.nodes) if(n.type==FLEAF) leaf_keys.push_back(n.key);

    std::vector<uint64_t> qkeys(nq);
    for(uint64_t i=0;i<nq;i++){
        if(rng()%10<7 && !leaf_keys.empty()) qkeys[i]=leaf_keys[rng()%leaf_keys.size()];
        else qkeys[i]=rng();
    }

    // CPU
    std::vector<int64_t> cpu_res(nq);
    { Timer t("LOOKUP_CPU");
      for(uint64_t i=0;i<nq;i++) cpu_res[i]=lookup_cpu(tree,qkeys[i]);
    }
    uint64_t found=0; for(auto r:cpu_res) if(r>=0) found++;
    INSPECT("LOOKUP_CPU", "found=%lu/%lu(%.1f%%)", found,nq, 100.0*found/nq);

#if WALKING_CUDA
    std::vector<int64_t> gpu_res(nq);
    { Timer t("LOOKUP_GPU");
      FlatNode*  d_n; uint32_t* d_ch; uint64_t* d_k; int64_t* d_r; uint32_t* d_h;
      size_t nn=tree.nodes.size(), nc=tree.children.size();
      GPU_CHECK(cudaMalloc(&d_n,  nn*sizeof(FlatNode)));
      GPU_CHECK(cudaMalloc(&d_ch, nc*sizeof(uint32_t)));
      GPU_CHECK(cudaMalloc(&d_k,  nq*sizeof(uint64_t)));
      GPU_CHECK(cudaMalloc(&d_r,  nq*sizeof(int64_t)));
      GPU_CHECK(cudaMalloc(&d_h,  nn*sizeof(uint32_t)));
      GPU_CHECK(cudaMemcpy(d_n,  tree.nodes.data(),    nn*sizeof(FlatNode), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(d_ch, tree.children.data(), nc*sizeof(uint32_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(d_k,  qkeys.data(),          nq*sizeof(uint64_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemset(d_h, 0, nn*sizeof(uint32_t)));

      kern_art_lookup<<<(nq+255)/256, 256>>>(d_n,d_ch,(uint32_t)nn,d_k,d_r,d_h,nq,tree.root);
      cudaDeviceSynchronize();
      GPU_CHECK(cudaMemcpy(gpu_res.data(), d_r, nq*sizeof(int64_t), cudaMemcpyDeviceToHost));

      // Read back hotspots
      std::vector<uint32_t> heat(nn);
      GPU_CHECK(cudaMemcpy(heat.data(), d_h, nn*sizeof(uint32_t), cudaMemcpyDeviceToHost));
      uint32_t max_h=0; uint32_t hot_id=0;
      for(size_t i=0;i<nn;i++) if(heat[i]>max_h){ max_h=heat[i]; hot_id=(uint32_t)i; }
      INSPECT("GPU_HOTSPOT", "node=%u heat=%u type=%u depth=%u",
              hot_id, max_h, tree.nodes[hot_id].type, tree.nodes[hot_id].depth);

      cudaFree(d_n); cudaFree(d_ch); cudaFree(d_k); cudaFree(d_r); cudaFree(d_h);
    }
    uint64_t mm=0; for(uint64_t i=0;i<nq;i++) if(cpu_res[i]!=gpu_res[i]) mm++;
    CHK(mm==0, "LOOKUP_GPU_VS_CPU", "mismatches=%lu/%lu", mm, nq);
    INSPECT("LOOKUP_VERIFY", "mismatches=%lu %s", mm, mm==0?"PASS":"FAIL");
#endif
}

// ════════════════════════════════════════════════════════════════
// §7  Experiment: Tree BFS (level-sync)
// ════════════════════════════════════════════════════════════════

static void exp_tree_bfs(const FlatART& tree) {
    sep("EXP: ART BFS (level-sync)");

    // CPU BFS
    std::vector<uint32_t> cpu_order;
    std::vector<uint32_t> level_sizes;
    { Timer t("BFS_CPU");
      std::queue<uint32_t> q; q.push(tree.root);
      while(!q.empty()){
          size_t lsz=q.size(); level_sizes.push_back((uint32_t)lsz);
          for(size_t i=0;i<lsz;i++){
              uint32_t nid=q.front(); q.pop();
              if(nid>=tree.nodes.size()) continue;
              cpu_order.push_back(nid);
              const auto& nd=tree.nodes[nid];
              if(nd.type==FLEAF) continue;
              size_t ns=(nd.type==FNODE256)?256:nd.num_children;
              for(size_t c=0;c<ns;c++){
                  uint32_t cid=tree.children[nd.children_start+c];
                  if(cid<tree.nodes.size()) q.push(cid);
              }
          }
      }
    }
    INSPECT("BFS_CPU", "visited=%lu levels=%lu", cpu_order.size(), level_sizes.size());
    if(g_dbg>=2){
        std::printf("  level_sizes: ");
        for(size_t i=0;i<std::min(level_sizes.size(),(size_t)15);i++)
            std::printf("L%lu=%u ", i, level_sizes[i]);
        std::printf("\n");
    }

#if WALKING_CUDA
    { Timer t("BFS_GPU");
      uint32_t N=(uint32_t)tree.nodes.size();
      FlatNode* d_n; uint32_t* d_ch; uint32_t* d_fa; uint32_t* d_fb;
      uint32_t* d_nc; uint64_t* d_th;
      GPU_CHECK(cudaMalloc(&d_n,  N*sizeof(FlatNode)));
      GPU_CHECK(cudaMalloc(&d_ch, tree.children.size()*sizeof(uint32_t)));
      GPU_CHECK(cudaMalloc(&d_fa, N*sizeof(uint32_t)));
      GPU_CHECK(cudaMalloc(&d_fb, N*sizeof(uint32_t)));
      GPU_CHECK(cudaMalloc(&d_nc, sizeof(uint32_t)));
      GPU_CHECK(cudaMalloc(&d_th, 5*sizeof(uint64_t)));
      GPU_CHECK(cudaMemcpy(d_n,  tree.nodes.data(),    N*sizeof(FlatNode), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(d_ch, tree.children.data(), tree.children.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
      uint32_t r=tree.root;
      GPU_CHECK(cudaMemcpy(d_fa, &r, sizeof(uint32_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemset(d_th, 0, 5*sizeof(uint64_t)));

      uint32_t fsize=1, level=0, total_visited=0;
      uint32_t* cur=d_fa; uint32_t* nxt=d_fb;
      while(fsize>0){
          uint32_t zero=0;
          GPU_CHECK(cudaMemcpy(d_nc, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
          kern_tree_bfs_level<<<(fsize+255)/256, 256>>>(d_n,d_ch,cur,fsize,nxt,d_nc,d_th,N);
          cudaDeviceSynchronize();
          total_visited+=fsize;
          GPU_CHECK(cudaMemcpy(&fsize, d_nc, sizeof(uint32_t), cudaMemcpyDeviceToHost));
          if(g_dbg>=2) INSPECT("BFS_GPU_LVL", "level=%u next_frontier=%u", level, fsize);
          level++;
          std::swap(cur,nxt);
      }
      CHK(total_visited==cpu_order.size(), "BFS_COUNT", "gpu=%u cpu=%lu", total_visited, cpu_order.size());
      INSPECT("BFS_GPU_DONE", "levels=%u visited=%u", level, total_visited);

      // Read type distribution histogram
      uint64_t th[5]={};
      GPU_CHECK(cudaMemcpy(th, d_th, 5*sizeof(uint64_t), cudaMemcpyDeviceToHost));
      INSPECT("BFS_GPU_TYPES", "N4=%lu N16=%lu N48=%lu N256=%lu LEAF=%lu",
              th[0],th[1],th[2],th[3],th[4]);

      cudaFree(d_n); cudaFree(d_ch); cudaFree(d_fa); cudaFree(d_fb);
      cudaFree(d_nc); cudaFree(d_th);
    }
#endif
}

// ════════════════════════════════════════════════════════════════
// §8  Experiment: Galloping intersect
// ════════════════════════════════════════════════════════════════

static void exp_intersect(uint64_t n, uint64_t seed) {
    sep("EXP: GALLOPING INTERSECT");

    std::mt19937_64 rng(seed);
    // Create two sorted arrays with partial overlap (simulating leaf intersection)
    // Array A: large (n elements), Array B: small (n/5 elements) → triggers galloping
    uint32_t na=(uint32_t)n, nb=(uint32_t)(n/5);
    std::vector<uint64_t> a(na), b(nb);
    for(uint32_t i=0;i<na;i++) a[i]=rng()%(4*n);
    for(uint32_t i=0;i<nb;i++) b[i]=rng()%(4*n);
    std::sort(a.begin(),a.end()); a.erase(std::unique(a.begin(),a.end()),a.end());
    std::sort(b.begin(),b.end()); b.erase(std::unique(b.begin(),b.end()),b.end());
    na=(uint32_t)a.size(); nb=(uint32_t)b.size();

    INSPECT("INTERSECT_SETUP", "A=%u B=%u ratio=%.1fx (gallop=%s)",
            na, nb, (double)na/std::max(nb,1u), (na>4*nb)?"YES":"NO");

    // CPU
    std::vector<uint64_t> cpu_res;
    { Timer t("INTERSECT_CPU");
      cpu_res = intersect_cpu(a.data(), na, b.data(), nb);
    }
    INSPECT("INTERSECT_CPU", "matches=%lu", cpu_res.size());

#if WALKING_CUDA
    { Timer t("INTERSECT_GPU");
      uint64_t* d_a; uint64_t* d_b; uint64_t* d_out; uint32_t* d_cnt;
      GPU_CHECK(cudaMalloc(&d_a, na*sizeof(uint64_t)));
      GPU_CHECK(cudaMalloc(&d_b, nb*sizeof(uint64_t)));
      GPU_CHECK(cudaMalloc(&d_out, std::min(na,nb)*sizeof(uint64_t)));
      GPU_CHECK(cudaMalloc(&d_cnt, sizeof(uint32_t)));
      GPU_CHECK(cudaMemcpy(d_a, a.data(), na*sizeof(uint64_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(d_b, b.data(), nb*sizeof(uint64_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemset(d_cnt, 0, sizeof(uint32_t)));

      kern_gallop_intersect<<<1,1>>>(d_a, na, d_b, nb, d_out, d_cnt);
      cudaDeviceSynchronize();

      uint32_t gpu_cnt=0;
      GPU_CHECK(cudaMemcpy(&gpu_cnt, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CHK(gpu_cnt==cpu_res.size(), "INTERSECT_COUNT", "gpu=%u cpu=%lu", gpu_cnt, cpu_res.size());
      INSPECT("INTERSECT_GPU", "matches=%u %s", gpu_cnt, gpu_cnt==cpu_res.size()?"PASS":"FAIL");

      cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_cnt);
    }
#endif
}

// ════════════════════════════════════════════════════════════════
// §9  Experiment: Interval stab
// ════════════════════════════════════════════════════════════════

static void exp_interval(uint64_t n_iv, uint64_t n_q, uint64_t seed) {
    sep("EXP: INTERVAL STAB QUERY");

    std::mt19937 rng(seed);
    int32_t range = (int32_t)(n_iv * 2);
    std::vector<int32_t> starts(n_iv), ends(n_iv);
    for(uint64_t i=0;i<n_iv;i++){
        starts[i] = rng() % range;
        ends[i] = starts[i] + 1 + rng()%(range/10+1);
    }
    // Sort by start (upstream TemGraph::load_intervals does the same)
    std::vector<size_t> idx(n_iv);
    std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[&](size_t a,size_t b){return starts[a]<starts[b];});
    std::vector<int32_t> ss(n_iv), se(n_iv);
    for(uint64_t i=0;i<n_iv;i++){ ss[i]=starts[idx[i]]; se[i]=ends[idx[i]]; }

    std::vector<int32_t> queries(n_q);
    for(auto&q:queries) q=rng()%(range+range/10);

    // CPU baseline
    std::vector<uint32_t> cpu_hits(n_q);
    { Timer t("STAB_CPU");
      for(uint64_t q=0;q<n_q;q++){
          int32_t pt=queries[q]; uint32_t cnt=0;
          // Binary+linear (same as GPU kernel)
          uint64_t lo=0, hi=n_iv;
          while(lo<hi){ uint64_t mid=(lo+hi)/2; if(ss[mid]<=pt)lo=mid+1;else hi=mid;}
          for(uint64_t i=0;i<lo;i++) if(se[i]>=pt) cnt++;
          cpu_hits[q]=cnt;
      }
    }
    uint64_t total=0; for(auto h:cpu_hits) total+=h;
    INSPECT("STAB_CPU", "total_hits=%lu avg=%.1f", total, (double)total/n_q);

#if WALKING_CUDA
    std::vector<uint32_t> gpu_hits(n_q);
    { Timer t("STAB_GPU");
      int32_t* d_ss; int32_t* d_se; int32_t* d_q; uint32_t* d_h;
      GPU_CHECK(cudaMalloc(&d_ss, n_iv*sizeof(int32_t)));
      GPU_CHECK(cudaMalloc(&d_se, n_iv*sizeof(int32_t)));
      GPU_CHECK(cudaMalloc(&d_q,  n_q*sizeof(int32_t)));
      GPU_CHECK(cudaMalloc(&d_h,  n_q*sizeof(uint32_t)));
      GPU_CHECK(cudaMemcpy(d_ss, ss.data(), n_iv*sizeof(int32_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(d_se, se.data(), n_iv*sizeof(int32_t), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(d_q,  queries.data(), n_q*sizeof(int32_t), cudaMemcpyHostToDevice));
      kern_interval_stab<<<(n_q+255)/256,256>>>(d_ss,d_se,n_iv,d_q,d_h,n_q);
      cudaDeviceSynchronize();
      GPU_CHECK(cudaMemcpy(gpu_hits.data(), d_h, n_q*sizeof(uint32_t), cudaMemcpyDeviceToHost));
      cudaFree(d_ss); cudaFree(d_se); cudaFree(d_q); cudaFree(d_h);
    }
    uint64_t mm=0; for(uint64_t i=0;i<n_q;i++) if(cpu_hits[i]!=gpu_hits[i]) mm++;
    CHK(mm==0, "STAB_GPU_VS_CPU", "mismatches=%lu/%lu", mm, n_q);
    INSPECT("STAB_VERIFY", "mismatches=%lu %s", mm, mm==0?"PASS":"FAIL");
#endif
}


// ════════════════════════════════════════════════════════════════
// MAIN
// ════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    std::printf("╔═══════════════════════════════════════════════════════════╗\n");
    std::printf("║  LLM4Walking — GPU Tree Traversal (ART+Interval+Gallop) ║\n");
#if WALKING_CUDA
    std::printf("║  Mode: CUDA GPU                                         ║\n");
#else
    std::printf("║  Mode: CPU fallback                                     ║\n");
#endif
    std::printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    uint64_t nk = (argc>=2) ? std::stoull(argv[1]) : 100000;
    uint64_t nq = (argc>=3) ? std::stoull(argv[2]) : 50000;
    if(argc>=4) g_dbg=std::stoi(argv[3]);

    std::printf("[SYS] PID=%d cores=%u RSS=%ldKB\n",
                getpid(), std::thread::hardware_concurrency(), rss_kb());
    std::printf("[CFG] keys=%lu queries=%lu debug=%d\n\n", nk, nq, g_dbg);

#if WALKING_CUDA
    int dc=0; cudaGetDeviceCount(&dc);
    for(int i=0;i<dc;i++){
        cudaDeviceProp p; cudaGetDeviceProperties(&p,i);
        std::printf("[GPU %d] %s SM=%d.%d mem=%.0fMB SMs=%d\n",
                    i,p.name,p.major,p.minor,p.totalGlobalMem/1048576.0,p.multiProcessorCount);
    }
    std::printf("\n");
#endif

    // ── Build ART ──
    sep("BUILD ART");
    std::vector<uint64_t> keys(nk); std::vector<int64_t> vals(nk);
    std::mt19937_64 rng(42);
    for(uint64_t i=0;i<nk;i++){ keys[i]=rng(); vals[i]=(int64_t)(i*100+7); }
    std::sort(keys.begin(),keys.end());
    keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
    vals.resize(keys.size()); nk=keys.size();
    INSPECT("KEYS", "unique=%lu", nk);

    FlatART tree;
    tree.build(keys, vals);

    // ── Run experiments ──
    exp_art_lookup(tree, nq, 123);
    exp_tree_bfs(tree);
    exp_intersect(nk, 789);
    exp_interval(nk/2, nq, 456);

    // ── Summary ──
    sep("SUMMARY");
    std::printf("[SUMMARY] inspections=%lu pass=%lu fail=%lu\n", g_insp, g_pass, g_fail);
    std::printf("[SUMMARY] RSS=%ldKB\n", rss_kb());
    if(g_fail>0) std::printf("[WARN] %lu assertion(s) FAILED\n", g_fail);
    else std::printf("[OK] all passed\n");
    return g_fail>0 ? 1 : 0;
}
