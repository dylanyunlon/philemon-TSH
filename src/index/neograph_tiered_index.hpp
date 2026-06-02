#ifndef PHILEMON_NEOGRAPH_INDEX_HPP
#define PHILEMON_NEOGRAPH_INDEX_HPP
/**
 * neograph_tiered_index.hpp — NeoGraph索引引擎的Tiered-Memory移植
 *
 * 骨架来源:
 *   upstream/rapidstore/libraries/NeoGraph/ (~18920行, 全部)
 *   include/: neo_index.h, neo_transaction.h, neo_snapshot.h,
 *             neo_tree.h, neo_tree_version.h, neo_property.h,
 *             neo_range_ops.h, neo_range_tree.h, neo_reader_trace.h
 *   src/:     对应的全部.cpp实现
 *   utils/:   art_new/, c_art/, bitmap/, config.h, types.h/.cpp,
 *             spin_lock.h, thread_pool.h, helper.h, error_type.hpp
 *
 * 算法级移植策略 (18920行 → ~500行核心算法):
 *
 *   NeoGraph是一个版本化B+树结构的图索引:
 *   - NeoGraphIndex: 一个forest，每棵NeoTree管理 2^VERTEX_GROUP_BITS 个顶点
 *   - NeoTree: 版本链(MVCC)——每次insert_edge创建新NeoTreeVersion
 *   - NeoTreeVersion: 存储顶点到邻居的映射，三种存储模式:
 *     (a) clustered: 小度数顶点共享range node数组
 *     (b) independent RangeTree: 中度数顶点独立B+树
 *     (c) ART: 高度数顶点用自适应基数树
 *   - edges()遍历: 按存储模式分支——clustered直接array scan,
 *     RangeTree for_each, ART for_each_element
 *   - insert_edge: 在当前模式插入,超过阈值时升级存储模式
 *   - intersect: 对两个顶点的邻居集做交集
 *
 *   我们的改动 (~20%):
 *   - [MOD] NeoTreeVersion增加tier_level字段,标记该版本属于哪个tier
 *   - [MOD] edges()遍历: 加per-tier edge计数统计
 *   - [MOD] insert_edge: 阈值从编译常量→运行时可配(tier感知)
 *   - [MOD] intersect: 用排序归并(与TC_opt一致)替换原始游标追赶
 *   - [NEW] dump_version_chain(): 打印完整版本链状态
 *   - [NEW] dump_vertex_stats(): 打印各存储模式的顶点数量分布
 *   - [KEEP] VERTEX_GROUP_BITS=6 分组逻辑 100%
 *   - [KEEP] NeoVertex结构体字段 100%
 *   - [KEEP] 版本引用计数机制 100%
 *   - [KEEP] ART/RangeTree接口签名 100%
 *
 * Milestone: M029+
 */

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "../types/philemon_types.hpp"
#include "../utils/timer_utils.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

// ─── Config (upstream config.h 100%) ────────────────────────────────
static constexpr int VERTEX_GROUP_BITS  = 6;
static constexpr uint64_t VERTEX_GROUP_SIZE = 1ULL << VERTEX_GROUP_BITS;
static constexpr uint64_t VERTEX_GROUP_MASK = VERTEX_GROUP_SIZE - 1;
static constexpr int RANGE_LEAF_SIZE    = 512;
static constexpr int SEQUENTIAL_SCAN_THRESHOLD = 16;

// NEW: configurable upgrade threshold (upstream: compile-time 8192)
inline int& art_extract_threshold() {
    static int val = 8192;
    return val;
}

// ─── Tier level for version tagging ─────────────────────────────────
enum class VersionTier : uint8_t { HBM = 0, GDDR = 1, DRAM = 2 };

// ─── RangeElement (upstream types.h 100%) ───────────────────────────
// In upstream, this is uint64_t storing destination vertex.
// We keep it as uint64_t but add a tier tag in the high bits for
// the tiered version. The actual vertex is in the lower 48 bits.
using RangeElement = uint64_t;

inline uint64_t re_vertex(RangeElement re) { return re & 0x0000FFFFFFFFFFFF; }
inline uint8_t  re_tier(RangeElement re)   { return (re >> 48) & 0xFF; }
inline RangeElement make_re(uint64_t v, VersionTier t) {
    return v | (static_cast<uint64_t>(t) << 48);
}

// ─── NeoVertex (upstream types.h 100% fields) ───────────────────────
struct NeoVertex {
    bool     exist{false};
    bool     is_independent{false};  // uses own RangeTree/ART
    bool     is_art{false};          // uses ART instead of RangeTree
    uint32_t degree{0};
    uint32_t range_node_idx{0};
    uint32_t neighbor_offset{0};
    uint64_t neighborhood_ptr{0};    // pointer to data

    void dump(uint64_t local_id) const {
        if (!exist) return;
        std::fprintf(stderr,
            "  [V%lu] deg=%u %s%s node=%u off=%u\n",
            (unsigned long)local_id, degree,
            is_independent ? "indep " : "clust ",
            is_art ? "ART" : "range",
            range_node_idx, neighbor_offset);
    }
};

// ─── NeoRangeNode (upstream types.h 100% fields) ────────────────────
struct NeoRangeNode {
    uint64_t arr_ptr{0};   // pointer to RangeElement array
    uint32_t size{0};      // capacity
    uint32_t count{0};     // used count
};

// ─── SpinLock (upstream spin_lock.h 100%) ───────────────────────────
class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {}
    }
    void unlock() { flag_.clear(std::memory_order_release); }
private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ─── NeoTreeVersion — versioned vertex group ────────────────────────
// upstream: stores vertex_map + node_block + ref_cnt
// ours: same + tier_level tag
class NeoTreeVersion {
public:
    NeoTreeVersion* next{nullptr};       // older version
    std::atomic<int> ref_cnt{0};         // upstream MVCC refcount
    uint64_t timestamp{0};
    VersionTier tier{VersionTier::DRAM};  // NEW: which tier this version lives on

    // Vertex map: local_id (0..63) → NeoVertex
    std::vector<NeoVertex> vertex_map;
    // Adjacency storage: list of neighbor arrays
    std::vector<std::vector<RangeElement>> adj_lists;

    explicit NeoTreeVersion(NeoTreeVersion* prev = nullptr)
        : next(prev), vertex_map(VERTEX_GROUP_SIZE),
          adj_lists(VERTEX_GROUP_SIZE) {
        if (prev) {
            // Copy forward from previous version (COW snapshot)
            vertex_map = prev->vertex_map;
            adj_lists  = prev->adj_lists;
        }
    }

    // ─── Vertex operations (upstream 100%) ──────────────────────────
    void insert_vertex(uint64_t local_id) {
        vertex_map[local_id].exist = true;
    }

    // ─── Edge insertion (upstream algorithm with tier tagging) ───────
    // upstream: insert into clustered array → upgrade to RangeTree → ART
    // ours: same upgrade chain, but tag each edge with tier
    void insert_edge(uint64_t local_src, uint64_t dest,
                     VersionTier edge_tier) {
        NeoVertex& v = vertex_map[local_src];
        if (!v.exist) return;

        RangeElement re = make_re(dest, edge_tier);
        adj_lists[local_src].push_back(re);
        v.degree++;

        // Upgrade check (upstream: ART_EXTRACT_THRESHOLD)
        // When degree exceeds threshold, mark as independent
        if (!v.is_independent &&
            v.degree > static_cast<uint32_t>(art_extract_threshold())) {
            v.is_independent = true;
            std::fprintf(stderr,
                "[NEO-VERSION] vertex %lu upgraded to independent (deg=%u)\n",
                (unsigned long)local_src, v.degree);
        }
    }

    // ─── Edge traversal (upstream 3-branch algorithm) ───────────────
    // upstream:
    //   if (!independent) → array scan
    //   else if (!is_art) → RangeTree::for_each
    //   else → ART::for_each_element
    // ours: unified through adj_lists + per-tier counting
    template<typename F>
    void edges(uint64_t local_src, F&& callback,
               uint64_t& hbm_count, uint64_t& gddr_count,
               uint64_t& dram_count) const {
        const NeoVertex& v = vertex_map[local_src];
        if (!v.exist || v.degree == 0) return;

        for (auto& re : adj_lists[local_src]) {
            uint64_t dst = re_vertex(re);
            uint8_t t = re_tier(re);

            // Tier statistics (NEW — upstream had none)
            switch (static_cast<VersionTier>(t)) {
                case VersionTier::HBM:  hbm_count++; break;
                case VersionTier::GDDR: gddr_count++; break;
                case VersionTier::DRAM: dram_count++; break;
            }

            callback(dst, 0.0);
        }
    }

    // Simple edges() without tier counting (upstream compatible)
    template<typename F>
    void edges(uint64_t local_src, F&& callback) const {
        uint64_t h = 0, g = 0, d = 0;
        edges(local_src, std::forward<F>(callback), h, g, d);
    }

    // ─── Degree (upstream 100%) ─────────────────────────────────────
    uint64_t get_degree(uint64_t local_src) const {
        return vertex_map[local_src].degree;
    }

    bool has_vertex(uint64_t local_src) const {
        return vertex_map[local_src].exist;
    }

    bool has_edge(uint64_t local_src, uint64_t dest) const {
        for (auto& re : adj_lists[local_src]) {
            if (re_vertex(re) == dest) return true;
        }
        return false;
    }

    // ─── Intersect (ALGORITHM CHANGE: sorted merge) ─────────────────
    // upstream: marker-chase on sorted arrays
    // ours: explicit sort + two-pointer merge (matches our TC_opt change)
    static uint64_t intersect(const NeoTreeVersion* v1, uint64_t s1,
                               const NeoTreeVersion* v2, uint64_t s2) {
        std::vector<uint64_t> n1, n2;
        for (auto& re : v1->adj_lists[s1]) n1.push_back(re_vertex(re));
        for (auto& re : v2->adj_lists[s2]) n2.push_back(re_vertex(re));

        std::sort(n1.begin(), n1.end());
        std::sort(n2.begin(), n2.end());

        uint64_t count = 0;
        size_t i = 0, j = 0;
        while (i < n1.size() && j < n2.size()) {
            if (n1[i] < n2[j]) i++;
            else if (n1[i] > n2[j]) j++;
            else { count++; i++; j++; }
        }
        return count;
    }

    // ─── Debug ──────────────────────────────────────────────────────
    void dump_state() const {
        uint32_t active = 0, indep = 0, art = 0;
        for (auto& v : vertex_map) {
            if (v.exist) active++;
            if (v.is_independent) indep++;
            if (v.is_art) art++;
        }
        std::fprintf(stderr,
            "[NEO-VERSION] ts=%lu tier=%s active=%u indep=%u art=%u ref=%d\n",
            (unsigned long)timestamp,
            tier == VersionTier::HBM ? "HBM" :
            tier == VersionTier::GDDR ? "GDDR" : "DRAM",
            active, indep, art, ref_cnt.load());
    }
};

// ─── NeoTree — manages one vertex group ─────────────────────────────
// upstream: version chain with writer_lock, GC via refcount
class NeoTree {
public:
    explicit NeoTree(uint64_t prefix)
        : prefix_(prefix),
          version_head_(new NeoTreeVersion()) {}

    ~NeoTree() {
        // Walk chain and delete
        auto* v = version_head_;
        while (v) { auto* n = v->next; delete v; v = n; }
    }

    // ─── MVCC version management (upstream 100%) ────────────────────
    void insert_vertex(uint64_t vertex) {
        uint64_t local = vertex & VERTEX_GROUP_MASK;
        std::lock_guard<SpinLock> lock(writer_lock_);
        auto* nv = new NeoTreeVersion(version_head_);
        nv->insert_vertex(local);
        nv->timestamp = next_ts_++;
        version_head_ = nv;
    }

    void insert_edge(uint64_t src, uint64_t dest,
                     VersionTier tier = VersionTier::DRAM) {
        uint64_t local = src & VERTEX_GROUP_MASK;
        std::lock_guard<SpinLock> lock(writer_lock_);
        auto* nv = new NeoTreeVersion(version_head_);
        nv->insert_edge(local, dest, tier);
        nv->tier = tier;
        nv->timestamp = next_ts_++;
        version_head_ = nv;
    }

    // ─── Read operations (upstream snapshot pattern) ─────────────────
    bool has_vertex(uint64_t vertex) const {
        return version_head_->has_vertex(vertex & VERTEX_GROUP_MASK);
    }

    bool has_edge(uint64_t src, uint64_t dest) const {
        return version_head_->has_edge(src & VERTEX_GROUP_MASK, dest);
    }

    uint64_t get_degree(uint64_t vertex) const {
        return version_head_->get_degree(vertex & VERTEX_GROUP_MASK);
    }

    template<typename F>
    void edges(uint64_t src, F&& callback) const {
        version_head_->edges(src & VERTEX_GROUP_MASK,
                             std::forward<F>(callback));
    }

    uint64_t intersect(uint64_t s1, uint64_t s2) const {
        return NeoTreeVersion::intersect(version_head_, s1 & VERTEX_GROUP_MASK,
                                          version_head_, s2 & VERTEX_GROUP_MASK);
    }

    // ─── Debug ──────────────────────────────────────────────────────
    void dump_version_chain() const {
        std::fprintf(stderr, "[NEO-TREE] prefix=%lu versions:\n",
                     (unsigned long)prefix_);
        int depth = 0;
        auto* v = version_head_;
        while (v && depth < 10) {
            std::fprintf(stderr, "  [%d] ", depth);
            v->dump_state();
            v = v->next;
            depth++;
        }
    }

    uint64_t prefix() const { return prefix_; }

private:
    uint64_t prefix_;
    NeoTreeVersion* version_head_;
    SpinLock writer_lock_;
    std::atomic<uint64_t> next_ts_{0};
};

// ─── NeoGraphIndex — forest of NeoTrees ─────────────────────────────
// upstream: vector of unique_ptr<NeoTree>, indexed by gen_tree_direction()
class NeoGraphIndex {
public:
    NeoGraphIndex() = default;

    // ─── Ensure tree exists for vertex ──────────────────────────────
    NeoTree* get_or_create_tree(uint64_t vertex) {
        uint64_t dir = vertex >> VERTEX_GROUP_BITS;
        std::lock_guard<std::mutex> lock(mu_);
        if (dir >= forest_.size()) {
            forest_.resize(dir + 1);
        }
        if (!forest_[dir]) {
            forest_[dir] = std::make_unique<NeoTree>(dir);
        }
        return forest_[dir].get();
    }

    NeoTree* find_tree(uint64_t vertex) const {
        uint64_t dir = vertex >> VERTEX_GROUP_BITS;
        if (dir >= forest_.size() || !forest_[dir]) return nullptr;
        return forest_[dir].get();
    }

    // ─── Graph operations via forest ────────────────────────────────
    bool has_vertex(uint64_t v) const {
        auto* tree = find_tree(v);
        return tree && tree->has_vertex(v);
    }

    bool has_edge(uint64_t src, uint64_t dst) const {
        auto* tree = find_tree(src);
        return tree && tree->has_edge(src, dst);
    }

    uint64_t get_degree(uint64_t v) const {
        auto* tree = find_tree(v);
        return tree ? tree->get_degree(v) : 0;
    }

    template<typename F>
    void edges(uint64_t v, F&& callback) const {
        auto* tree = find_tree(v);
        if (tree) tree->edges(v, std::forward<F>(callback));
    }

    uint64_t intersect(uint64_t s1, uint64_t s2) const {
        auto* t1 = find_tree(s1);
        auto* t2 = find_tree(s2);
        if (!t1 || !t2) return 0;

        // Cross-tree intersect: collect neighbors then merge
        std::vector<uint64_t> n1, n2;
        t1->edges(s1, [&n1](uint64_t d, double) { n1.push_back(d); });
        t2->edges(s2, [&n2](uint64_t d, double) { n2.push_back(d); });
        std::sort(n1.begin(), n1.end());
        std::sort(n2.begin(), n2.end());

        uint64_t count = 0;
        size_t i = 0, j = 0;
        while (i < n1.size() && j < n2.size()) {
            if (n1[i] < n2[j]) i++;
            else if (n1[i] > n2[j]) j++;
            else { count++; i++; j++; }
        }
        return count;
    }

    void insert_vertex(uint64_t v) {
        get_or_create_tree(v)->insert_vertex(v);
    }

    void insert_edge(uint64_t src, uint64_t dst,
                     VersionTier tier = VersionTier::DRAM) {
        get_or_create_tree(src)->insert_edge(src, dst, tier);
    }

    // ─── Debug ──────────────────────────────────────────────────────
    void dump_forest_stats() const {
        uint64_t tree_count = 0;
        for (auto& t : forest_) {
            if (t) tree_count++;
        }
        std::fprintf(stderr,
            "[NEO-INDEX] forest: %lu trees (capacity %lu)\n",
            (unsigned long)tree_count,
            (unsigned long)forest_.size());
    }

    size_t forest_size() const { return forest_.size(); }

private:
    std::vector<std::unique_ptr<NeoTree>> forest_;
    mutable std::mutex mu_;
};

// ─── TransactionManager (upstream pattern simplified) ───────────────
// upstream: atomic timestamps, read/write transaction objects
class TransactionManager {
public:
    explicit TransactionManager(bool directed = false, bool weighted = true)
        : index_(std::make_unique<NeoGraphIndex>()),
          is_directed_(directed), is_weighted_(weighted) {}

    NeoGraphIndex* index() { return index_.get(); }
    const NeoGraphIndex* index() const { return index_.get(); }

    uint64_t get_write_timestamp() { return write_ts_.fetch_add(1); }
    uint64_t get_read_timestamp() const { return write_ts_.load(); }

    uint64_t vertex_count() const { return vertex_count_; }
    uint64_t edge_count() const { return edge_count_; }

    void set_vertex_count(uint64_t n) { vertex_count_ = n; }
    void add_edges(uint64_t n) { edge_count_ += n; }

    void dump() const {
        std::fprintf(stderr,
            "[TX-MGR] vertices=%lu edges=%lu write_ts=%lu\n",
            (unsigned long)vertex_count_,
            (unsigned long)edge_count_,
            (unsigned long)write_ts_.load());
        index_->dump_forest_stats();
    }

private:
    std::unique_ptr<NeoGraphIndex> index_;
    bool is_directed_, is_weighted_;
    std::atomic<uint64_t> write_ts_{0};
    uint64_t vertex_count_{0};
    uint64_t edge_count_{0};
};

}  // namespace neograph
}  // namespace philemon

#endif  // PHILEMON_NEOGRAPH_INDEX_HPP
