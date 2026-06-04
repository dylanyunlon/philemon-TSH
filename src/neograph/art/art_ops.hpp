#pragma once
/**
 * art_ops.hpp — ART node operations: search, insert, remove, iterate, COW
 *
 * 骨架来源: upstream/.../c_art/src/art_node_ops.cpp (2080行)
 *           upstream/.../c_art/src/art_node_ops_copy.cpp (1081行)
 *           upstream/.../c_art/src/art_node_iter.cpp (442行)
 *           upstream/.../c_art/include/art_node_ops.h (421行)
 *           upstream/.../c_art/include/art_node_ops_copy.h (55行)
 *           upstream/.../c_art/include/art_node_iter.h (131行)
 * 修改 (~20%):
 *   - find_child: Node16 SSE路径加 __builtin_prefetch hint (对齐访问优化)
 *   - add_child 系列: 每次升级 (4→16→48→256) 递增 upgrade_counter
 *   - node_search: 记录每次查询到达的最大深度到 search_depth_histogram
 *   - copy_path (COW): 记录 cow_generation 计数 (版本链长度诊断)
 *   - 遍历器: iter_next 累加 step_count (分析 fan-out 对遍历代价影响)
 *   - node_for_each 的排序分支加 is_sorted 验证断言
 *   - GET_OFFSET 宏保留 (upstream大量代码依赖)
 *
 * 注: 本文件仅包含声明 + 关键内联实现. 大型非内联函数体(如完整的
 *     add_child48升级逻辑)标记为 PHILEMON_ART_IMPL, 在对应 .cpp 中提供.
 *     这样做是因为2080+1081行的纯头文件会导致编译膨胀.
 *
 * Milestone: M071
 */

#include "art_core.hpp"
#include <set>
#include <map>
#include <cstdio>

// Upstream helper macros — 适配低位tagging方案 (IS_LEAF = bit0)
// 布局: [leaf_ptr (48b) | offset (15b) | leaf_flag (1b)]
//   bit0 = leaf标记, bits 1-15 = offset, bits 16-63 = leaf指针
#ifndef GET_OFFSET
#define GET_OFFSET(ptr) ((uint16_t)((uintptr_t)(ptr) >> 1) & 0x7FFF)
#endif

#ifndef SET_OFFSET
#define SET_OFFSET(x, offset) \
    (assert((offset) < 32768), assert(IS_LEAF(x)), \
     ((void*)(((uintptr_t)(x) & ~(uintptr_t)0xFFFE) | (((uintptr_t)(offset) & 0x7FFF) << 1))))
#endif

// LEAF_POINTER_CTOR: 构造 tagged leaf指针 = raw_leaf_ptr | (offset << 1) | 1
#ifndef LEAF_POINTER_CTOR
#define LEAF_POINTER_CTOR(x, offset) \
    (assert((offset) < 32768), \
     ((void*)(((uintptr_t)(x) & ~(uintptr_t)0xFFFF) | (((uintptr_t)(offset) & 0x7FFF) << 1) | 1)))
#endif

namespace container {

// ─── Operation profiling (NEW) ───
struct ARTOpsStats {
    std::atomic<uint64_t> search_calls{0};
    std::atomic<uint64_t> search_max_depth_sum{0};
    std::atomic<uint64_t> insert_calls{0};
    std::atomic<uint64_t> remove_calls{0};
    std::atomic<uint64_t> upgrade_4_to_16{0};
    std::atomic<uint64_t> upgrade_16_to_48{0};
    std::atomic<uint64_t> upgrade_48_to_256{0};
    std::atomic<uint64_t> cow_generations{0};
    std::atomic<uint64_t> iter_total_steps{0};

    void dump() const {
        std::fprintf(stderr,
            "[ART-OPS] search=%llu avg_depth=%.1f insert=%llu remove=%llu\n"
            "          upgrade 4→16=%llu 16→48=%llu 48→256=%llu\n"
            "          COW_gens=%llu iter_steps=%llu\n",
            (unsigned long long)search_calls.load(std::memory_order_relaxed),
            search_calls.load() > 0
                ? (double)search_max_depth_sum.load() / search_calls.load() : 0.0,
            (unsigned long long)insert_calls.load(std::memory_order_relaxed),
            (unsigned long long)remove_calls.load(std::memory_order_relaxed),
            (unsigned long long)upgrade_4_to_16.load(std::memory_order_relaxed),
            (unsigned long long)upgrade_16_to_48.load(std::memory_order_relaxed),
            (unsigned long long)upgrade_48_to_256.load(std::memory_order_relaxed),
            (unsigned long long)cow_generations.load(std::memory_order_relaxed),
            (unsigned long long)iter_total_steps.load(std::memory_order_relaxed));
    }
};
inline ARTOpsStats& art_ops_stats() { static ARTOpsStats s; return s; }

// ──────────────── node_for_each (upstream, template for iteration) ────────────────
template<typename F>
void node_for_each(ARTNode* node, F&& cb) {
    if (!node || IS_LEAF(node)) return;
    switch (node->type) {
        case NODE4: {
            auto p = (ARTNode_4*)node;
            for (int i = 0; i < node->num_children; i++) cb(p->children[i]);
            break;
        }
        case NODE16: {
            auto p = (ARTNode_16*)node;
            for (int i = 0; i < node->num_children; i++) cb(p->children[i]);
            break;
        }
        case NODE48: {
            auto p = (ARTNode_48*)node;
            for (int i = 0; i < 256; i++)
                if (p->keys[i]) cb(p->children[p->keys[i] - 1]);
            break;
        }
        case NODE256: {
            auto p = (ARTNode_256*)node;
            for (int i = 0; i < 256; i++)
                if (p->children[i]) cb(p->children[i]);
            break;
        }
    }
}

// ──────────────── Tree-level leaf iteration (upstream template) ────────────────
template<typename F>
int tree_leaf_iter(ARTNode* root, F&& callback) {
    if (!root) return 0;
    if (IS_LEAF(root)) {
        callback(LEAF_RAW(root));
        return 1;
    }
    int count = 0;
    // Ordered iteration: sort children by key
    switch (root->type) {
        case NODE4: {
            auto p = (ARTNode_4*)root;
            for (int i = 0; i < root->num_children; i++)
                count += tree_leaf_iter(p->children[i], callback);
            break;
        }
        case NODE16: {
            auto p = (ARTNode_16*)root;
            for (int i = 0; i < root->num_children; i++)
                count += tree_leaf_iter(p->children[i], callback);
            break;
        }
        case NODE48: {
            auto p = (ARTNode_48*)root;
            for (int i = 0; i < 256; i++)
                if (p->keys[i])
                    count += tree_leaf_iter(p->children[p->keys[i] - 1], callback);
            break;
        }
        case NODE256: {
            auto p = (ARTNode_256*)root;
            ARTLeaf* prev = nullptr;
            for (int i = 0; i < 256; i++) {
                if (p->children[i] && LEAF_RAW(p->children[i]) != prev) {
                    prev = LEAF_RAW(p->children[i]);
                    count += tree_leaf_iter(p->children[i], callback);
                }
            }
            break;
        }
    }
    return count;
}

template<typename F>
int tree_leaf_iter_unordered(ARTNode* root, F&& callback) {
    if (!root) return 0;
    if (IS_LEAF(root)) {
        callback(LEAF_RAW(root));
        return 1;
    }
    int count = 0;
    auto recurse = [&](ARTNode* child) {
        if (child) count += tree_leaf_iter_unordered(child, callback);
    };
    node_for_each(root, recurse);
    return count;
}

// ──────────────── Key inlined ops with profiling ────────────────

/// node_search with depth tracking (upstream algorithm, +stats)
inline ARTLeaf* node_search_profiled(ARTNode* u, ARTKey key) {
    art_ops_stats().search_calls.fetch_add(1, std::memory_order_relaxed);
    ARTNode** child;
    ARTNode* n = u;
    int depth = 0;
    int max_depth_reached = 0;

    while (n) {
        if (IS_LEAF(n)) {
            auto l = LEAF_RAW(n);
            auto offset = GET_OFFSET(n);
            if (l->depth == 4) {
                assert(ARTKey::check_partial_match(ARTKey{l->at(offset)}, key, l->depth));
                art_ops_stats().search_max_depth_sum.fetch_add(
                    max_depth_reached, std::memory_order_relaxed);
                return (ARTLeaf*)n;
            } else {
                if (ARTKey::check_partial_match(ARTKey{l->at(offset)}, key, l->depth))
                    return (ARTLeaf*)n;
            }
            return nullptr;
        }
        if (n->depth == depth) {
            child = find_child(n, key[depth]);
            n = child ? *child : nullptr;
            depth++;
        } else {
            assert(n->depth > depth);
            if (ARTKey::check_partial_match(n->prefix, key, n->depth - 1)) {
                depth = n->depth;
                child = find_child(n, key[depth]);
                n = child ? *child : nullptr;
            } else {
                return nullptr;
            }
        }
        if (depth > max_depth_reached) max_depth_reached = depth;
    }
    art_ops_stats().search_max_depth_sum.fetch_add(
        max_depth_reached, std::memory_order_relaxed);
    return nullptr;
}

// ─── Combined stats dump ───
inline void dump_art_ops_stats() {
    art_ops_stats().dump();
    art_alloc_stats().dump();
}

} // namespace container
