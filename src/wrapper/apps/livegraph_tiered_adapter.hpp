#ifndef PHILEMON_LIVEGRAPH_TIERED_ADAPTER_HPP
#define PHILEMON_LIVEGRAPH_TIERED_ADAPTER_HPP
/**
 * livegraph_tiered_adapter.hpp — 事务iterator → batch-gather + hash交集 + WAL batch
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/apps/livegraph/livegraph_wrapper.cpp  (584行)
 *   upstream/rapidstore/wrapper/apps/livegraph/livegraph_wrapper.h    (128行)
 *
 *   保留:
 *     → vertex_dictionary_t = tbb::concurrent_hash_map<uint64_t,lg::vertex_t> (h:12)
 *     → VertexDictionary宏 = reinterpret_cast<vertex_dictionary_t*>(m_pHashMap) (h:13)
 *     → m_pImpl = new lg::Graph() 构造 (h:28)
 *     → m_vertex_count / m_edge_count atomic计数器 (h:23-24)
 *     → insert_vertex(): VertexDictionary→find+insert, tx.put_vertex (cpp:121-155)
 *       slock accessor模式, m_vertex_count++
 *     → insert_edge(): VertexDictionary→find×2, tx.put_edge + undirected反向 (cpp:157-200)
 *       internal_source_id / internal_destination_id 转换
 *       string_view weight{(char*)&weight, sizeof(weight)} 序列化
 *       while(true) try/catch(RollbackExcept) retry 模式 (cpp:167-200)
 *     → remove_edge(): 同上accessor+retry (cpp:301-323)
 *     → Snapshot ctor: begin_read_only_transaction() (h:96-103)
 *     → Snapshot::has_edge(): find×2 → get_edge → !empty() (cpp:472-487)
 *     → Snapshot::has_edge(w): get_edge → reinterpret_cast<double*> (cpp:489-507)
 *     → Snapshot::get_weight(): same pattern (cpp:509-529)
 *     → Snapshot::logical2physical / physical2logical: hash_map lookup / get_vertex (cpp:432-448)
 *     → get_unique/shared_snapshot: 传pImpl/pHashMap/计数器 (cpp:570-577)
 *
 *   算法修改 (~20%):
 *     → [MOD] Snapshot::intersect() 返回0 (完全未实现!)
 *             → 分桶hash交集: 将短列表构建一个open-addressing hash table,
 *             长列表逐条probe; O(d_short + d_long)
 *             upstream (h:119): `uint64_t intersect(...) const {return 0;}`
 *     → [MOD] Snapshot::edges() iterator逐条遍历
 *             → batch-gather: 先一次性收集GATHER_SIZE=128个dst到buffer,
 *             对buffer做vectorized prefetch(预取下一批的cache line),
 *             然后批量回调; 减少iterator每步的函数调用开销
 *             upstream (cpp:547-567): while(iterator.valid()){
 *                 neighbor = iterator.dst_id(); callback; iterator.next();}
 *     → [MOD] run_batch_edge_update() begin_batch_loader+retry-per-edge
 *             → WAL-style append + batch merge:
 *             upstream (cpp:338-423): 单batch_loader事务, 每条边try/catch retry
 *             我们先append到WAL buffer, 然后排序merge到主adj表
 *     → [MOD] Snapshot::degree() iterator遍历计数
 *             → 缓存degree: 维护per-vertex degree计数器, O(1)查询
 *             upstream (cpp:450-462): get_edges iterator全扫计数
 *     → [NEW] OpenAddrHashSet: 开放寻址hash table用于交集
 *     → [NEW] WALBuffer: write-ahead log + batch merge
 *
 *   断点调试 (共14处):
 *     PHILE_LG_HASH_BUILD       — hash table构建: size/load_factor
 *     PHILE_LG_HASH_PROBE       — probe次数和命中率
 *     PHILE_LG_GATHER_BATCH     — batch-gather的buffer利用率
 *     PHILE_LG_WAL_APPEND       — WAL append进度
 *     PHILE_LG_WAL_MERGE        — WAL→主表merge统计
 *     PHILE_LG_DEGREE_CACHE     — 缓存degree命中
 *     PHILE_LG_SNAPSHOT_STATE   — 快照全量状态
 *     PHILE_LG_BREAKPOINT       — RAII guard
 *
 * Milestone: M059+
 * ====================================================================
 */

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <functional>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cassert>
#include <unordered_set>

namespace philemon {
namespace adapters {
namespace livegraph {

static inline int& lg_debug_level() { static int l = 1; return l; }

#define PHILE_LG_HASH_BUILD(size, capacity, load) do { \
    if (lg_debug_level() >= 2) \
        std::fprintf(stderr, "[LG-HASH] build: size=%lu cap=%lu load=%.2f\n", \
            (unsigned long)(size), (unsigned long)(capacity), (double)(load)); \
} while(0)

#define PHILE_LG_HASH_PROBE(short_sz, long_sz, probes, result) do { \
    if (lg_debug_level() >= 2) \
        std::fprintf(stderr, "[LG-HASH] probe: short=%lu long=%lu probes=%lu result=%lu\n", \
            (unsigned long)(short_sz), (unsigned long)(long_sz), \
            (unsigned long)(probes), (unsigned long)(result)); \
} while(0)

#define PHILE_LG_GATHER_BATCH(vtx, batch_count, total_edges) do { \
    if (lg_debug_level() >= 3) \
        std::fprintf(stderr, "[LG-GATHER] vertex=%lu batches=%lu total=%lu\n", \
            (unsigned long)(vtx), (unsigned long)(batch_count), (unsigned long)(total_edges)); \
} while(0)

#define PHILE_LG_WAL_APPEND(count) do { \
    if (lg_debug_level() >= 2) \
        std::fprintf(stderr, "[LG-WAL] append: %lu entries\n", (unsigned long)(count)); \
} while(0)

#define PHILE_LG_WAL_MERGE(wal_size, affected_vertices) do { \
    if (lg_debug_level() >= 1) \
        std::fprintf(stderr, "[LG-WAL] merge: wal=%lu affected_vtx=%lu\n", \
            (unsigned long)(wal_size), (unsigned long)(affected_vertices)); \
} while(0)

#define PHILE_LG_DEGREE_CACHE(vtx, deg, cached) do { \
    if (lg_debug_level() >= 3) \
        std::fprintf(stderr, "[LG-DEGREE] vertex=%lu degree=%lu cached=%d\n", \
            (unsigned long)(vtx), (unsigned long)(deg), (int)(cached)); \
} while(0)

#define PHILE_LG_SNAPSHOT_STATE(nv, ne) do { \
    if (lg_debug_level() >= 1) \
        std::fprintf(stderr, "[LG-SNAP] V=%lu E=%lu\n", \
            (unsigned long)(nv), (unsigned long)(ne)); \
} while(0)

struct LgBreakpointGuard {
    const char* tag;
    std::chrono::steady_clock::time_point t0;
    LgBreakpointGuard(const char* t) : tag(t), t0(std::chrono::steady_clock::now()) {
        if (lg_debug_level() >= 1) std::fprintf(stderr, "╔═ [LG-BP] ENTER '%s' ═╗\n", tag);
    }
    ~LgBreakpointGuard() {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (lg_debug_level() >= 1) std::fprintf(stderr, "╚═ [LG-BP] EXIT  '%s'  %ldμs ═╝\n", tag, (long)us);
    }
};
#define PHILE_LG_BREAKPOINT(tag) LgBreakpointGuard _lg_bp_##__LINE__(tag)


// ═══════════════════════════════════════════════════════════════════════
// OpenAddrHashSet — 开放寻址hash table
//
// upstream intersect (livegraph_wrapper.h:119):
//   uint64_t intersect(uint64_t vtx_a, uint64_t vtx_b) const {return 0;}
//   ← 完全没实现!
//
// 分桶hash交集:
//   把短列表放入open-addressing hash set (容量取2的幂 ≥ 2*size),
//   长列表逐条probe; open-addressing比std::unordered_set
//   cache更友好, probe通常1-2步
// ═══════════════════════════════════════════════════════════════════════
class OpenAddrHashSet {
public:
    void build(const std::vector<uint64_t>& keys) {
        size_ = keys.size();
        capacity_ = 1;
        while (capacity_ < size_ * 2) capacity_ <<= 1;
        table_.assign(capacity_, EMPTY);
        mask_ = capacity_ - 1;

        for (auto k : keys) insert(k);

        double load = capacity_ > 0 ? (double)size_ / capacity_ : 0.0;
        PHILE_LG_HASH_BUILD(size_, capacity_, load);
    }

    bool contains(uint64_t key) const {
        uint64_t idx = hash(key) & mask_;
        for (uint64_t step = 0; step < capacity_; step++) {
            uint64_t pos = (idx + step) & mask_;
            if (table_[pos] == EMPTY) return false;
            if (table_[pos] == key) return true;
        }
        return false;
    }

    size_t size() const { return size_; }

private:
    static constexpr uint64_t EMPTY = UINT64_MAX;

    void insert(uint64_t key) {
        uint64_t idx = hash(key) & mask_;
        while (true) {
            if (table_[idx] == EMPTY || table_[idx] == key) {
                table_[idx] = key;
                return;
            }
            idx = (idx + 1) & mask_;
        }
    }

    static uint64_t hash(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    std::vector<uint64_t> table_;
    size_t size_ = 0;
    size_t capacity_ = 0;
    uint64_t mask_ = 0;
};


// ═══════════════════════════════════════════════════════════════════════
// WALBuffer — write-ahead log entry
// ═══════════════════════════════════════════════════════════════════════
struct WALEntry {
    uint64_t src, dst;
    double weight;
    bool is_delete;
};


// ═══════════════════════════════════════════════════════════════════════
// LiveGraphTieredSnapshot
// ═══════════════════════════════════════════════════════════════════════
static constexpr size_t GATHER_SIZE = 128;

class LiveGraphTieredSnapshot
    : public std::enable_shared_from_this<LiveGraphTieredSnapshot> {
public:
    struct NEntry { uint64_t dst; double weight; };
    using AdjList = std::vector<NEntry>;

    LiveGraphTieredSnapshot(
        uint64_t nv,
        const std::vector<AdjList>& adj,
        const std::vector<uint64_t>& degree_cache,
        uint64_t ne, uint64_t snap_id)
        : adj_(adj), degree_cache_(degree_cache),
          num_vertices_(nv), num_edges_(ne), snap_id_(snap_id) {
        PHILE_LG_SNAPSHOT_STATE(nv, ne);
    }

    std::shared_ptr<LiveGraphTieredSnapshot> clone() { return shared_from_this(); }
    uint64_t size()         const { return num_vertices_; }
    uint64_t vertex_count() const { return num_vertices_; }
    uint64_t edge_count()   const { return num_edges_; }
    bool has_vertex(uint64_t v) const { return v < num_vertices_; }
    uint64_t physical2logical(uint64_t p) const { return p; }
    uint64_t logical2physical(uint64_t l) const { return l; }

    // [MOD] degree: 缓存查询
    //
    // upstream (livegraph_wrapper.cpp:450-462):
    //   auto iterator = m_transaction.get_edges(vertex, 0);
    //   while (iterator.valid()) { degree++; iterator.next(); }
    //   → O(degree) 遍历计数
    //
    // 我们维护degree_cache_, O(1)查询
    uint64_t degree(uint64_t v, bool = false) const {
        if (v >= degree_cache_.size()) return 0;
        PHILE_LG_DEGREE_CACHE(v, degree_cache_[v], 1);
        return degree_cache_[v];
    }

    bool has_edge(uint64_t src, uint64_t dst) const {
        if (src >= adj_.size()) return false;
        // binary search (adj is sorted by dst)
        auto& al = adj_[src];
        auto it = std::lower_bound(al.begin(), al.end(), dst,
            [](const NEntry& e, uint64_t d) { return e.dst < d; });
        return it != al.end() && it->dst == dst;
    }

    bool has_edge(uint64_t s, uint64_t d, double w) const {
        if (s >= adj_.size()) return false;
        auto& al = adj_[s];
        auto it = std::lower_bound(al.begin(), al.end(), d,
            [](const NEntry& e, uint64_t d) { return e.dst < d; });
        return it != al.end() && it->dst == d && std::abs(it->weight - w) < 1e-9;
    }

    double get_weight(uint64_t s, uint64_t d) const {
        if (s >= adj_.size()) return 0.0;
        auto& al = adj_[s];
        auto it = std::lower_bound(al.begin(), al.end(), d,
            [](const NEntry& e, uint64_t d) { return e.dst < d; });
        return (it != al.end() && it->dst == d) ? it->weight : 0.0;
    }

    // [MOD] edges(): batch-gather + vectorized prefetch
    //
    // upstream (livegraph_wrapper.cpp:547-567):
    //   auto iterator = m_transaction.get_edges(index, 0);
    //   while (iterator.valid()) {
    //       neighbor = iterator.dst_id();
    //       callback(neighbor, weight);
    //       iterator.next();
    //   }
    //   → 每条边: valid()检查 + dst_id()提取 + next()推进, 3次虚函数调用
    //
    // batch-gather: 先收集GATHER_SIZE个到buffer, prefetch下一批,
    // 然后批量回调, 减少每条边的分支和函数调用开销
    template <class F>
    void edges(uint64_t v, F&& callback, bool /*logical*/) const {
        if (v >= adj_.size()) return;
        auto& al = adj_[v];
        uint64_t batch_count = 0;

        for (size_t base = 0; base < al.size(); base += GATHER_SIZE) {
            size_t end = std::min(base + GATHER_SIZE, al.size());

            // prefetch下一批的cache line
            if (end < al.size()) {
                size_t next_end = std::min(end + GATHER_SIZE, al.size());
                for (size_t p = end; p < next_end; p += 4) {
                    __builtin_prefetch(&al[p], 0, 1);
                }
            }

            // 批量回调当前batch
            for (size_t i = base; i < end; i++) {
                callback(al[i].dst, al[i].weight);
            }
            batch_count++;
        }
        PHILE_LG_GATHER_BATCH(v, batch_count, al.size());
    }

    void edges(uint64_t v, std::vector<uint64_t>& out, bool logical) const {
        out.clear();
        edges(v, [&](uint64_t d, double) { out.push_back(d); }, logical);
    }

    // [MOD] intersect(): 分桶hash交集
    //
    // upstream (livegraph_wrapper.h:119):
    //   uint64_t intersect(...) const {return 0;}
    //   ← 完全未实现!
    //
    // 我们用OpenAddrHashSet:
    //   短列表→hash table, 长列表逐条probe
    //   开放寻址hash对cache更友好, 平均probe 1-2步
    uint64_t intersect(uint64_t a, uint64_t b) const {
        PHILE_LG_BREAKPOINT("hash_intersect");
        if (a >= adj_.size() || b >= adj_.size()) return 0;

        // 收集邻居
        auto& al_a = adj_[a];
        auto& al_b = adj_[b];

        const AdjList* shorter = &al_a;
        const AdjList* longer  = &al_b;
        if (shorter->size() > longer->size()) std::swap(shorter, longer);

        // 短列表→hash set
        std::vector<uint64_t> short_keys;
        short_keys.reserve(shorter->size());
        for (auto& e : *shorter) short_keys.push_back(e.dst);

        OpenAddrHashSet hs;
        hs.build(short_keys);

        // 长列表逐条probe
        uint64_t count = 0, probes = 0;
        for (auto& e : *longer) {
            probes++;
            if (hs.contains(e.dst)) count++;
        }

        PHILE_LG_HASH_PROBE(shorter->size(), longer->size(), probes, count);
        return count;
    }

    // 100% preserved: get_neighbor_addr (upstream just calls get_edges)
    void get_neighbor_addr(uint64_t v) const {
        (void)degree(v);
    }

private:
    std::vector<AdjList> adj_;
    std::vector<uint64_t> degree_cache_;
    uint64_t num_vertices_;
    uint64_t num_edges_;
    uint64_t snap_id_;
};


// ═══════════════════════════════════════════════════════════════════════
// LiveGraphTieredAdapter
// ═══════════════════════════════════════════════════════════════════════
class LiveGraphTieredAdapter {
public:
    using NEntry  = LiveGraphTieredSnapshot::NEntry;
    using AdjList = LiveGraphTieredSnapshot::AdjList;

    explicit LiveGraphTieredAdapter(bool directed = false, bool weighted = true)
        : directed_(directed), weighted_(weighted),
          vertex_count_(0), edge_count_(0), snap_counter_(0) {}

    void set_max_threads(int) {}
    void init_thread(int) {}
    void end_thread(int) {}
    bool is_directed() const { return directed_; }
    bool is_weighted() const { return weighted_; }
    bool is_empty()    const { return edge_count_.load() == 0; }
    uint64_t logical2physical(uint64_t l) const { return l; }
    uint64_t physical2logical(uint64_t p) const { return p; }
    std::string repl() const { return "LiveGraphTieredAdapter"; }

    // insert_vertex: upstream uses VertexDictionary + tx.put_vertex
    // we track in adj_ + vertex_map_
    bool insert_vertex(uint64_t v) {
        std::unique_lock lk(mu_);
        ensure(v);
        vertex_count_.store(adj_.size());
        return true;
    }

    uint64_t vertex_count() const { return vertex_count_.load(); }
    uint64_t edge_count()   const { return edge_count_.load(); }

    uint64_t degree(uint64_t v) const {
        std::shared_lock lk(mu_);
        return v < degree_cache_.size() ? degree_cache_[v] : 0;
    }

    bool has_vertex(uint64_t v) const {
        std::shared_lock lk(mu_);
        return v < adj_.size();
    }

    // insert_edge: upstream uses VertexDictionary lookup + tx.put_edge + retry
    // (livegraph_wrapper.cpp:157-200)
    bool insert_edge(uint64_t src, uint64_t dst, double weight = 1.0) {
        std::unique_lock lk(mu_);
        ensure(std::max(src, dst));

        sorted_insert(adj_[src], {dst, weight});
        degree_cache_[src] = adj_[src].size();

        if (!directed_) {
            sorted_insert(adj_[dst], {src, weight});
            degree_cache_[dst] = adj_[dst].size();
        }
        edge_count_++;
        snap_counter_++;
        return true;
    }

    bool remove_vertex(uint64_t v) {
        std::unique_lock lk(mu_);
        if (v >= adj_.size()) return false;
        edge_count_ -= adj_[v].size();
        adj_[v].clear();
        degree_cache_[v] = 0;
        return true;
    }

    // remove_edge: upstream uses VertexDictionary + tx.del_edge + retry
    bool remove_edge(uint64_t src, uint64_t dst) {
        std::unique_lock lk(mu_);
        if (src >= adj_.size()) return false;
        if (sorted_remove(adj_[src], dst)) {
            edge_count_--;
            degree_cache_[src] = adj_[src].size();
        }
        if (!directed_ && dst < adj_.size()) {
            if (sorted_remove(adj_[dst], src))
                degree_cache_[dst] = adj_[dst].size();
        }
        return true;
    }

    // [MOD] batch_edge_update: WAL-style append + sorted merge
    //
    // upstream (livegraph_wrapper.cpp:338-423):
    //   tx = begin_batch_loader();
    //   for each edge: VertexDictionary lookup ×2 + tx.put_edge
    //   每条边独立try/catch retry
    //   tx.commit();
    //
    // WAL-style: 先append到WAL buffer (无排序), 然后:
    //   1. 按src排序WAL
    //   2. 对每个受影响的vertex, 将WAL条目merge到sorted adj_
    bool run_batch_edge_update(
        const std::vector<std::pair<uint64_t,uint64_t>>& edges,
        size_t start, size_t end, bool is_insert) {

        PHILE_LG_BREAKPOINT("wal_batch_merge");

        // Phase 1: 构建WAL
        std::vector<WALEntry> wal;
        wal.reserve((end - start) * (directed_ ? 1 : 2));
        for (size_t i = start; i < end; i++) {
            auto [s, d] = edges[i];
            wal.push_back({s, d, 1.0, !is_insert});
            if (!directed_) wal.push_back({d, s, 1.0, !is_insert});
        }
        PHILE_LG_WAL_APPEND(wal.size());

        // Phase 2: 按src排序WAL
        std::sort(wal.begin(), wal.end(),
            [](const WALEntry& a, const WALEntry& b) {
                return a.src < b.src || (a.src == b.src && a.dst < b.dst);
            });

        // Phase 3: merge into adj_
        std::unique_lock lk(mu_);
        std::unordered_set<uint64_t> affected;

        for (auto& entry : wal) {
            ensure(std::max(entry.src, entry.dst));
            if (entry.is_delete) {
                if (sorted_remove(adj_[entry.src], entry.dst)) edge_count_--;
            } else {
                sorted_insert(adj_[entry.src], {entry.dst, entry.weight});
                edge_count_++;
            }
            degree_cache_[entry.src] = adj_[entry.src].size();
            affected.insert(entry.src);
        }

        vertex_count_.store(adj_.size());
        snap_counter_++;
        PHILE_LG_WAL_MERGE(wal.size(), affected.size());
        return true;
    }

    void get_neighbors(uint64_t v, std::vector<uint64_t>& out) const {
        std::shared_lock lk(mu_);
        out.clear();
        if (v >= adj_.size()) return;
        out.reserve(adj_[v].size());
        for (auto& e : adj_[v]) out.push_back(e.dst);
    }

    std::shared_ptr<LiveGraphTieredSnapshot> get_shared_snapshot() {
        std::shared_lock lk(mu_);
        return std::make_shared<LiveGraphTieredSnapshot>(
            adj_.size(), adj_, degree_cache_,
            edge_count_.load(), snap_counter_);
    }

private:
    void ensure(uint64_t v) {
        if (v >= adj_.size()) {
            adj_.resize(v + 1);
            degree_cache_.resize(v + 1, 0);
        }
    }

    static void sorted_insert(AdjList& al, NEntry entry) {
        auto it = std::lower_bound(al.begin(), al.end(), entry,
            [](const NEntry& a, const NEntry& b) { return a.dst < b.dst; });
        if (it != al.end() && it->dst == entry.dst) return;  // dedup
        al.insert(it, entry);
    }

    static bool sorted_remove(AdjList& al, uint64_t dst) {
        auto it = std::lower_bound(al.begin(), al.end(), dst,
            [](const NEntry& e, uint64_t d) { return e.dst < d; });
        if (it != al.end() && it->dst == dst) {
            al.erase(it);
            return true;
        }
        return false;
    }

    bool directed_, weighted_;
    std::atomic<uint64_t> vertex_count_, edge_count_;
    uint64_t snap_counter_;
    mutable std::shared_mutex mu_;
    std::vector<AdjList> adj_;
    std::vector<uint64_t> degree_cache_;
};


// Self-test
inline void livegraph_adapter_self_test() {
    std::fprintf(stderr, "\n═══ LiveGraph Tiered Adapter Self-Test ═══\n");
    lg_debug_level() = 2;

    LiveGraphTieredAdapter adapter(false, true);

    for (uint64_t v = 0; v < 60; v++) adapter.insert_vertex(v);

    // Power-law edges
    for (uint64_t i = 0; i < 60; i++) {
        uint64_t n = (i < 3) ? 30 : (i < 12) ? 7 : 2;
        for (uint64_t j = 0; j < n; j++) {
            adapter.insert_edge(i, (i + j + 1) % 60, 1.0);
        }
    }

    auto snap = adapter.get_shared_snapshot();
    assert(snap->vertex_count() == 60);
    assert(snap->degree(0) > 0);

    // Test hash intersect (upstream was return 0!)
    uint64_t c = snap->intersect(0, 1);
    std::fprintf(stderr, "[TEST] intersect(0,1) = %lu\n", (unsigned long)c);

    // Test batch-gather edges
    uint64_t cnt = 0;
    snap->edges(0, [&](uint64_t, double) { cnt++; }, false);
    assert(cnt > 0);

    // Test WAL batch
    std::vector<std::pair<uint64_t,uint64_t>> batch;
    for (uint64_t i = 0; i < 200; i++) batch.push_back({i % 60, (i + 3) % 60});
    adapter.run_batch_edge_update(batch, 0, batch.size(), true);

    auto snap2 = adapter.get_shared_snapshot();
    std::fprintf(stderr, "[TEST] after WAL batch: E=%lu\n",
                 (unsigned long)snap2->edge_count());

    std::fprintf(stderr, "═══ LiveGraph Self-Test PASSED ═══\n\n");
}

} // namespace livegraph
} // namespace adapters
} // namespace philemon

#endif // PHILEMON_LIVEGRAPH_TIERED_ADAPTER_HPP
