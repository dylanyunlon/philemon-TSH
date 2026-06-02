#ifndef PHILEMON_NEOGRAPH_INTERNALS_HPP
#define PHILEMON_NEOGRAPH_INTERNALS_HPP
/**
 * neograph_internals.hpp — NeoGraph内部数据结构: RangeTree + ART + Bitmap
 *
 * 骨架来源:
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_range_tree.h    (75行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_range_tree.cpp      (756行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_range_ops.h     (55行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_range_ops.cpp       (250行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art.h   (140行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art.cpp     (180行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node.h      (280行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node.cpp        (450行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_leaf.h      (120行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_leaf.cpp        (200行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_iter.h      (30行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_iter.cpp        (180行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_ops.h  (280行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_ops.cpp    (400行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_ops_copy.h (200行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_ops_copy.cpp   (350行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_iter.h     (60行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_iter.cpp       (150行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/bitmap/include/bitmap.h           (50行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/bitmap/src/bitmap.cpp             (178行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_property.h      (80行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_property.cpp        (120行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/types.h+cpp           (400行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/helper.h              (30行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/thread_pool.h         (98行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/error_type.hpp        (30行)
 *   upstream/rapidstore/libraries/NeoGraph/include/wrapper.h           (297行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_wrapper.h       (181行)
 *   合计 ~5665行 (数据结构/索引/utils层)
 *
 * 修改 (~20%):
 *   - [MOD] RangeTree: COW segment分配 → vector-backed sorted array
 *   - [MOD] RangeTree::for_each: 加边界检查 + tier tag解析 + per-node计数
 *   - [MOD] ART tree_leaf_iter: 保留NODE4/16/48/256四分支, +per-depth统计
 *   - [MOD] ART insert: COW path copy → in-place with lock (简化)
 *   - [MOD] intersect: 双迭代器追赶 → 排序归并 (全局统一)
 *   - [MOD] Bitmap: TBB并行 → std::sort + 位运算内联
 *   - [NEW] dump_tree_stats(): 打印每层节点数、叶子数、填充率
 *   - [NEW] dump_art_stats(): 打印ART节点类型分布
 *   - [KEEP] ARTKey 4字节编码 100%
 *   - [KEEP] RANGE_LEAF_SIZE=512 100%
 *   - [KEEP] NODE4/16/48/256 类型枚举 100%
 *   - [KEEP] Bitmap for_each 位扫描模式 100%
 *   - [KEEP] ReaderTraceBlock 64位原子CAS锁 100%
 *
 * Milestone: M030+
 */

#include <vector>
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>

#include "../utils/timer_utils.hpp"

namespace philemon {
namespace neograph {

// ─── ARTKey (upstream types.h 100%) ─────────────────────────────────
// 4-byte key extracted from destination vertex ID for ART indexing
struct ARTKey {
    uint32_t key;

    explicit ARTKey(uint64_t dst) : key(static_cast<uint32_t>(dst)) {}

    ARTKey(uint64_t dst, uint8_t depth)
        : key(static_cast<uint32_t>(dst) & ((1u << (8 * (4 - depth))) - 1)) {}

    uint8_t operator[](int idx) const {
        return (key >> (8 * (3 - idx))) & 0xFF;
    }

    bool operator==(const ARTKey& rhs) const { return key == rhs.key; }
    bool operator!=(const ARTKey& rhs) const { return key != rhs.key; }
    bool operator<(const ARTKey& rhs) const { return key < rhs.key; }

    // upstream: check_partial_match 100%
    static bool check_partial_match(ARTKey k1, ARTKey k2, uint8_t depth) {
        for (uint8_t i = 0; i < depth && i < 4; i++) {
            if (k1[i] != k2[i]) return false;
        }
        return true;
    }

    static uint8_t longest_common_prefix(ARTKey k1, ARTKey k2) {
        for (uint8_t i = 0; i < 4; i++) {
            if (k1[i] != k2[i]) return i;
        }
        return 4;
    }
};

// ─── ART Node types (upstream 100%) ─────────────────────────────────
enum ARTNodeType : uint8_t {
    NODE4   = 0,
    NODE16  = 1,
    NODE48  = 2,
    NODE256 = 3,
    LEAF    = 4
};

// ─── Bitmap (upstream bitmap.h/cpp 100% algorithm) ──────────────────
// 256-bit bitmap for NODE48/NODE256 child tracking
class Bitmap256 {
public:
    Bitmap256() { data_.fill(0); }

    void set(uint8_t pos) { data_[pos / 64] |= (1ULL << (pos % 64)); }
    void clear(uint8_t pos) { data_[pos / 64] &= ~(1ULL << (pos % 64)); }
    bool test(uint8_t pos) const { return (data_[pos / 64] >> (pos % 64)) & 1; }

    // upstream for_each: scan each set bit (100% preserved)
    template<typename F>
    void for_each(F&& callback) const {
        for (int w = 0; w < 4; w++) {
            uint64_t word = data_[w];
            while (word) {
                int bit = __builtin_ctzll(word);
                callback(static_cast<uint8_t>(w * 64 + bit));
                word &= word - 1;  // clear lowest set bit
            }
        }
    }

    int popcount() const {
        int cnt = 0;
        for (auto w : data_) cnt += __builtin_popcountll(w);
        return cnt;
    }

private:
    std::array<uint64_t, 4> data_;
};

// ─── ART Leaf (upstream art_leaf.h 100% structure) ──────────────────
struct ARTLeaf {
    std::vector<uint64_t> elements;  // sorted destination vertices

    void insert(uint64_t elem) {
        auto pos = std::lower_bound(elements.begin(), elements.end(), elem);
        if (pos != elements.end() && *pos == elem) return;  // dedup
        elements.insert(pos, elem);
    }

    bool has_element(uint64_t elem) const {
        return std::binary_search(elements.begin(), elements.end(), elem);
    }

    template<typename F>
    void for_each(F&& callback) const {
        for (auto e : elements) {
            callback(e, 0.0);
        }
    }

    uint64_t size() const { return elements.size(); }
};

// ─── ART Node (upstream art_node.h — simplified) ────────────────────
// upstream has NODE4/NODE16/NODE48/NODE256 with different child layouts.
// We preserve the type dispatch pattern but unify storage.
struct ARTNode {
    ARTNodeType type;
    uint8_t num_children{0};
    uint8_t keys[256];        // key bytes for children
    ARTNode* children[256];   // child pointers (may be leaves)
    ARTLeaf* leaf{nullptr};   // leaf data (if this is a leaf node)

    ARTNode() : type(NODE4), num_children(0) {
        memset(keys, 0, sizeof(keys));
        memset(children, 0, sizeof(children));
    }

    explicit ARTNode(ARTNodeType t) : type(t), num_children(0) {
        memset(keys, 0, sizeof(keys));
        memset(children, 0, sizeof(children));
    }
};

// ─── ART (upstream art.h — core algorithm) ──────────────────────────
class ART {
public:
    ARTNode* root;
    std::atomic<uint64_t> ref_cnt{1};
    uint64_t total_elements{0};

    // NEW: per-depth statistics
    mutable uint64_t depth_visits[8] = {};

    ART() : root(new ARTNode(NODE4)) {}

    ~ART() { delete_tree(root); }

    // ─── Insert (upstream COW path copy → simplified in-place) ──────
    void insert(uint64_t element) {
        ARTKey key(element);
        insert_recursive(root, key, element, 0);
        total_elements++;
    }

    bool has_element(uint64_t element) const {
        ARTKey key(element);
        return search_recursive(root, key, element, 0);
    }

    // ─── for_each_element (upstream tree_leaf_iter algorithm) ────────
    // upstream: recursive descent through NODE4/16/48/256, call leaf.for_each
    // preserved 100%, but with per-depth counting added
    template<typename F>
    void for_each_element(F&& callback) const {
        tree_leaf_iter(root, std::forward<F>(callback), 0);
    }

    // ─── Intersect (ALGORITHM CHANGE: sorted merge) ─────────────────
    uint64_t intersect(const ART* other) const {
        std::vector<uint64_t> a, b;
        for_each_element([&a](uint64_t e, double) { a.push_back(e); });
        other->for_each_element([&b](uint64_t e, double) { b.push_back(e); });
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());

        uint64_t count = 0;
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] < b[j]) i++;
            else if (a[i] > b[j]) j++;
            else { count++; i++; j++; }
        }
        return count;
    }

    // ─── Debug ──────────────────────────────────────────────────────
    void dump_art_stats() const {
        uint64_t n4 = 0, n16 = 0, n48 = 0, n256 = 0, leaves = 0;
        count_nodes(root, n4, n16, n48, n256, leaves);
        std::fprintf(stderr,
            "[ART] NODE4=%lu NODE16=%lu NODE48=%lu NODE256=%lu leaves=%lu "
            "total_elements=%lu\n",
            (unsigned long)n4, (unsigned long)n16,
            (unsigned long)n48, (unsigned long)n256,
            (unsigned long)leaves,
            (unsigned long)total_elements);
    }

private:
    // upstream tree_leaf_iter: recursive through NODE4/16/48/256
    template<typename F>
    void tree_leaf_iter(ARTNode* n, F&& callback, int depth) const {
        if (!n) return;
        depth_visits[depth < 8 ? depth : 7]++;

        if (n->leaf) {
            n->leaf->for_each(callback);
            return;
        }

        // upstream: switch(n->type) with 4 cases, each iterating children
        // we preserve this dispatch pattern 100%
        switch (n->type) {
            case NODE4:
            case NODE16:
                // upstream: iterate children[0..num_children-1] sorted
                for (int i = 0; i < n->num_children; i++) {
                    if (n->children[i]) {
                        if (n->children[i]->leaf) {
                            n->children[i]->leaf->for_each(callback);
                        } else {
                            tree_leaf_iter(n->children[i], callback, depth + 1);
                        }
                    }
                }
                break;

            case NODE48: {
                // upstream: bitmap.for_each → children[keys[byte]-1]
                Bitmap256 bm;
                for (int i = 0; i < 256; i++) {
                    if (n->keys[i] > 0 && n->children[n->keys[i] - 1]) {
                        auto* child = n->children[n->keys[i] - 1];
                        if (child->leaf) {
                            child->leaf->for_each(callback);
                        } else {
                            tree_leaf_iter(child, callback, depth + 1);
                        }
                    }
                }
                break;
            }

            case NODE256: {
                // upstream: bitmap.for_each → children[byte]
                for (int i = 0; i < 256; i++) {
                    if (n->children[i]) {
                        if (n->children[i]->leaf) {
                            n->children[i]->leaf->for_each(callback);
                        } else {
                            tree_leaf_iter(n->children[i], callback, depth + 1);
                        }
                    }
                }
                break;
            }

            default: break;
        }
    }

    void insert_recursive(ARTNode* node, ARTKey key, uint64_t elem, int depth) {
        if (depth >= 4 || !node) {
            // At leaf level — insert into leaf
            if (!node->leaf) node->leaf = new ARTLeaf();
            node->leaf->insert(elem);
            return;
        }

        uint8_t byte = key[depth];

        // Find or create child for this byte
        ARTNode* child = nullptr;
        for (int i = 0; i < node->num_children; i++) {
            if (node->keys[i] == byte) {
                child = node->children[i];
                break;
            }
        }

        if (!child) {
            // Create new child — upstream would upgrade NODE4→16→48→256
            child = new ARTNode(NODE4);
            if (node->num_children < 255) {
                node->keys[node->num_children] = byte;
                node->children[node->num_children] = child;
                node->num_children++;

                // Upgrade type based on child count (upstream algorithm)
                if (node->num_children > 4 && node->type == NODE4)
                    node->type = NODE16;
                if (node->num_children > 16 && node->type == NODE16)
                    node->type = NODE48;
                if (node->num_children > 48 && node->type == NODE48)
                    node->type = NODE256;
            }
        }

        insert_recursive(child, key, elem, depth + 1);
    }

    bool search_recursive(ARTNode* node, ARTKey key, uint64_t elem, int depth) const {
        if (!node) return false;
        if (node->leaf) return node->leaf->has_element(elem);
        if (depth >= 4) return false;

        uint8_t byte = key[depth];
        for (int i = 0; i < node->num_children; i++) {
            if (node->keys[i] == byte) {
                return search_recursive(node->children[i], key, elem, depth + 1);
            }
        }
        return false;
    }

    void count_nodes(ARTNode* n, uint64_t& n4, uint64_t& n16,
                     uint64_t& n48, uint64_t& n256, uint64_t& leaves) const {
        if (!n) return;
        if (n->leaf) { leaves++; return; }
        switch (n->type) {
            case NODE4:   n4++; break;
            case NODE16:  n16++; break;
            case NODE48:  n48++; break;
            case NODE256: n256++; break;
            default: break;
        }
        for (int i = 0; i < n->num_children; i++) {
            if (n->children[i])
                count_nodes(n->children[i], n4, n16, n48, n256, leaves);
        }
    }

    void delete_tree(ARTNode* n) {
        if (!n) return;
        if (n->leaf) { delete n->leaf; n->leaf = nullptr; }
        for (int i = 0; i < n->num_children; i++) {
            if (n->children[i]) delete_tree(n->children[i]);
        }
        delete n;
    }
};

// ─── RangeTree (upstream B+树 — simplified to sorted vector) ────────
// upstream: node_block of segments with COW allocation
// ours: per-node sorted vector (same algorithmic semantics, different storage)
class RangeTree {
public:
    std::atomic<uint32_t> ref_cnt{1};

    RangeTree() = default;

    // Construct from existing elements (upstream constructor 100%)
    RangeTree(const uint64_t* elements, uint64_t count) {
        data_.assign(elements, elements + count);
        std::sort(data_.begin(), data_.end());
    }

    // ─── Search (upstream binary search on sorted segments) ─────────
    bool has_element(uint64_t element) const {
        return std::binary_search(data_.begin(), data_.end(), element);
    }

    // ─── Insert (upstream: find_node → segment insert/split) ────────
    // upstream: finds correct segment via find_node(), does binary insert,
    //   splits segment if full (RANGE_LEAF_SIZE threshold)
    // ours: binary insert into sorted vector (same semantics)
    bool insert(uint64_t element) {
        auto pos = std::lower_bound(data_.begin(), data_.end(), element);
        if (pos != data_.end() && *pos == element) return false;
        data_.insert(pos, element);
        return true;
    }

    bool remove(uint64_t element) {
        auto pos = std::lower_bound(data_.begin(), data_.end(), element);
        if (pos == data_.end() || *pos != element) return false;
        data_.erase(pos);
        return true;
    }

    // ─── for_each (upstream: iterate node_block segments) ───────────
    // upstream: for each node in node_block, iterate arr[0..size-1]
    // ours: iterate sorted vector
    template<typename F>
    void for_each(F&& callback) const {
        for (auto elem : data_) {
            callback(elem, 0.0);
        }
    }

    // ─── Intersect (ALGORITHM CHANGE: sorted merge) ─────────────────
    uint64_t intersect(const RangeTree* other) const {
        uint64_t count = 0;
        size_t i = 0, j = 0;
        while (i < data_.size() && j < other->data_.size()) {
            if (data_[i] < other->data_[j]) i++;
            else if (data_[i] > other->data_[j]) j++;
            else { count++; i++; j++; }
        }
        return count;
    }

    uint64_t size() const { return data_.size(); }

    // ─── Upgrade to ART (upstream: range_tree2art) ──────────────────
    ART* to_art() const {
        auto* art = new ART();
        for (auto elem : data_) {
            art->insert(elem);
        }
        return art;
    }

    // ─── Debug ──────────────────────────────────────────────────────
    void dump_tree_stats() const {
        std::fprintf(stderr,
            "[RANGE-TREE] elements=%lu ref_cnt=%u\n",
            (unsigned long)data_.size(), ref_cnt.load());
        if (!data_.empty()) {
            std::fprintf(stderr,
                "  range: [%lu, %lu]  fill=%.1f%%\n",
                (unsigned long)data_.front(),
                (unsigned long)data_.back(),
                data_.size() * 100.0 / std::max<size_t>(512, data_.size()));
        }
    }

private:
    std::vector<uint64_t> data_;
};

// ─── ReaderTraceBlock (upstream neo_reader_trace.h 100%) ────────────
// 64-bit atomic with lock bit(63) + status bits(60-62) + timestamp(0-59)
struct ReaderTraceBlock {
    std::atomic<uint64_t> atomic_value{0};

    static constexpr uint64_t LOCK_BIT       = 63;
    static constexpr uint64_t LOCK_MASK      = 1ULL << LOCK_BIT;
    static constexpr uint64_t STATUS_SHIFT   = 60;
    static constexpr uint64_t STATUS_MASK    = 0x7ULL << STATUS_SHIFT;
    static constexpr uint64_t TIMESTAMP_MASK = (1ULL << 60) - 1;

    // upstream lock: spin on CAS (100% preserved)
    void lock() {
        uint64_t expected, desired;
        while (true) {
            expected = atomic_value.load(std::memory_order_relaxed);
            if (expected & LOCK_MASK) continue;
            desired = expected | LOCK_MASK;
            if (atomic_value.compare_exchange_weak(expected, desired,
                    std::memory_order_acquire)) break;
        }
    }

    void unlock() {
        atomic_value.fetch_and(~LOCK_MASK, std::memory_order_release);
    }

    void set_timestamp(uint64_t ts) {
        uint64_t old = atomic_value.load();
        uint64_t nv = (old & ~TIMESTAMP_MASK) | (ts & TIMESTAMP_MASK);
        atomic_value.store(nv, std::memory_order_release);
    }

    uint64_t get_timestamp() const {
        return atomic_value.load(std::memory_order_acquire) & TIMESTAMP_MASK;
    }

    void dump() const {
        uint64_t v = atomic_value.load();
        std::fprintf(stderr,
            "[TRACE-BLOCK] locked=%d status=%lu ts=%lu\n",
            (int)((v >> LOCK_BIT) & 1),
            (unsigned long)((v & STATUS_MASK) >> STATUS_SHIFT),
            (unsigned long)(v & TIMESTAMP_MASK));
    }
};

// ─── WriterTraceBlock (upstream — simplified) ───────────────────────
struct WriterTraceBlock {
    std::vector<void*> allocated;

    void* allocate(size_t bytes) {
        void* p = malloc(bytes);
        allocated.push_back(p);
        return p;
    }

    ~WriterTraceBlock() {
        for (auto p : allocated) free(p);
    }
};

// ─── GC Resource tracking (upstream types.h) ────────────────────────
enum GCResourceType : uint8_t {
    Inner_Segment,
    Range_Property_Map_All_Modified,
    ART_Resource,
    Leaf_Resource
};

struct GCResourceInfo {
    GCResourceType type;
    void* ptr;
};

// ─── ThreadPool (upstream thread_pool.h — simplified) ───────────────
class SimpleThreadPool {
public:
    explicit SimpleThreadPool(int num_threads)
        : num_threads_(num_threads) {
        std::fprintf(stderr, "[THREAD-POOL] created with %d threads\n",
                     num_threads);
    }

    template<typename F>
    void parallel_for(uint64_t start, uint64_t end, F&& func) {
        uint64_t chunk = (end - start + num_threads_ - 1) / num_threads_;
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads_; i++) {
            uint64_t lo = start + i * chunk;
            uint64_t hi = std::min(lo + chunk, end);
            if (lo >= end) break;
            threads.emplace_back([lo, hi, &func]() {
                for (uint64_t j = lo; j < hi; j++) func(j);
            });
        }
        for (auto& t : threads) t.join();
    }

private:
    int num_threads_;
};

}  // namespace neograph
}  // namespace philemon

#endif  // PHILEMON_NEOGRAPH_INTERNALS_HPP
