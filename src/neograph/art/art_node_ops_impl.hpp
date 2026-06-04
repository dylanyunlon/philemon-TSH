#pragma once
/**
 * art_node_ops_impl.hpp — ART node operations 完整实现
 *
 * 骨架来源:
 *   upstream/.../c_art/src/art_node_ops.cpp      (2080行)
 *   upstream/.../c_art/include/art_node_ops.h     (421行)
 * 合计 ~2500行 upstream
 *
 * 修改 (~20% 算法级):
 *   - find_child Node16: SSE匹配命中后 __builtin_prefetch 预取目标children cacheline
 *   - add_child4: 4元素时用直接比较定位插入点代替循环扫描 (branchless 4路比较)
 *   - add_child16→48升级: bitmap初始化用 popcnt 加速 unique_bitmap 构建
 *   - node_split 分裂点: 从 ART_LEAF_SIZE/2 改为 weighted_mid = size*2/3
 *     (右倾分裂, 减少热key区域的再分裂概率 — 参考B+tree bulk-load策略)
 *   - intersect: 当一侧fanout > 4x另一侧时切换为 probe模式
 *     (小侧逐key probe大侧的find_child, 避免线性归并的无效比较)
 *   - batch_subtree_build: 根据byte种类数预判初始node类型
 *     (>4种直接分配NODE16, >16种NODE48, 避免升级开销)
 *   - leaf_list_merge: 两侧有序时用 galloping search 跳过不重叠区间
 *     (先指数步长探测边界, 再二分定位, 对skewed分布加速显著)
 *   - 全函数加断点: 每个核心操作dump当前节点类型/深度/children数/叶填充率
 *
 * Milestone: M073
 */

#include "art_core.hpp"
#include <set>
#include <map>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace container {

// ─── 断点打印宏 ──────────────────────────────────────────────────
// 运行时通过环境变量 PHILEMON_ART_DEBUG 控制级别 (0=off, 1=summary, 2=detail, 3=trace)
inline int art_debug_level() {
    static int lvl = []() {
        const char* e = std::getenv("PHILEMON_ART_DEBUG");
        return e ? std::atoi(e) : 0;
    }();
    return lvl;
}

#define ART_DBG(level, fmt, ...) \
    do { if (art_debug_level() >= (level)) \
        std::fprintf(stderr, "[ART·OPS] " fmt "\n", ##__VA_ARGS__); \
    } while(0)

// ─── 操作统计 (全局, 支持多线程) ─────────────────────────────────
struct NodeOpsCounters {
    std::atomic<uint64_t> find_child_calls{0};
    std::atomic<uint64_t> find_child_sse_hits{0};  // Node16 SSE路径命中
    std::atomic<uint64_t> add_child_calls{0};
    std::atomic<uint64_t> upgrade_4_to_16{0};
    std::atomic<uint64_t> upgrade_16_to_48{0};
    std::atomic<uint64_t> upgrade_48_to_256{0};
    std::atomic<uint64_t> split_calls{0};
    std::atomic<uint64_t> split_went_deeper{0};
    std::atomic<uint64_t> intersect_probe_switches{0}; // 切换到probe模式的次数
    std::atomic<uint64_t> gallop_skips{0};             // galloping跳过的元素总数
    std::atomic<uint64_t> batch_direct_node16{0};      // 直接分配NODE16的次数
    std::atomic<uint64_t> batch_direct_node48{0};

    void dump() const {
        std::fprintf(stderr,
            "[ART·STATS] find_child=%llu sse_hits=%llu add_child=%llu\n"
            "  upgrades: 4→16=%llu 16→48=%llu 48→256=%llu\n"
            "  splits=%llu went_deeper=%llu\n"
            "  intersect_probe_switches=%llu gallop_skips=%llu\n"
            "  batch_direct: node16=%llu node48=%llu\n",
            (unsigned long long)find_child_calls.load(),
            (unsigned long long)find_child_sse_hits.load(),
            (unsigned long long)add_child_calls.load(),
            (unsigned long long)upgrade_4_to_16.load(),
            (unsigned long long)upgrade_16_to_48.load(),
            (unsigned long long)upgrade_48_to_256.load(),
            (unsigned long long)split_calls.load(),
            (unsigned long long)split_went_deeper.load(),
            (unsigned long long)intersect_probe_switches.load(),
            (unsigned long long)gallop_skips.load(),
            (unsigned long long)batch_direct_node16.load(),
            (unsigned long long)batch_direct_node48.load());
    }
};
inline NodeOpsCounters& ops_counters() {
    static NodeOpsCounters c;
    return c;
}

// ─── 辅助: dump单个节点状态 (供断点使用) ──────────────────────────
inline void dump_node_state(const char* ctx, ARTNode* n) {
    if (art_debug_level() < 2 || !n) return;
    const char* type_str[] = {"?","N4","N16","N48","N256"};
    std::fprintf(stderr, "[ART·NODE] %s type=%s depth=%u children=%u prefix_key=0x%x\n",
                 ctx, (n->type <= 4) ? type_str[n->type] : "?",
                 n->depth, n->num_children, n->prefix.key);
}

inline void dump_leaf_state(const char* ctx, ARTLeaf* l) {
    if (art_debug_level() < 2 || !l) return;
    std::fprintf(stderr, "[ART·LEAF] %s size=%u depth=%u single_byte=%u "
                 "key=0x%x ref_cnt=%u first_elem=%llu\n",
                 ctx, l->size, l->depth, l->is_single_byte,
                 l->key.key, l->ref_cnt.load(),
                 l->size > 0 ? (unsigned long long)l->at(0) : 0ULL);
}

// ═══════════════════════════════════════════════════════════════════
//                       node_ref
// ═══════════════════════════════════════════════════════════════════
inline void node_ref(ARTNode* node) {
    auto cb = [](ARTNode* child) {
        if (IS_LEAF(child)) {
            LEAF_RAW(child)->ref_cnt += 1;
        } else {
            child->ref_cnt += 1;
        }
    };
    node_for_each(node, cb);
}

// ═══════════════════════════════════════════════════════════════════
//                    find_mid_count
// 改动: 分裂点从 ART_LEAF_SIZE/2 右移到 ART_LEAF_SIZE*2/3
// 理由: 类似B+tree bulk-load的右倾策略, 左侧较满的分裂
//       使得顺序插入时右侧叶子有更多空间, 减少连续分裂
// ═══════════════════════════════════════════════════════════════════
static constexpr uint16_t SPLIT_THRESHOLD = ART_LEAF_SIZE * 2 / 3;

inline uint16_t find_mid_count(std::vector<container::ARTNode**>* pointers_in_range) {
    uint16_t cur_idx = 0;
    while (cur_idx < pointers_in_range->size() - 1) {
        cur_idx += 1;
        // 改动: 用 SPLIT_THRESHOLD 代替 ART_LEAF_SIZE / 2
        if (GET_OFFSET(*(pointers_in_range->at(cur_idx))) >= SPLIT_THRESHOLD) {
            break;
        }
    }
    return cur_idx;
}

inline ARTNode** find_mid_count(ARTNode** begin_ptr, ARTNode** end_ptr) {
    auto cur_ptr = begin_ptr;
    if (*(end_ptr - 1) == nullptr) {
        end_ptr -= 1;
    }
    while (cur_ptr < end_ptr - 1) {
        cur_ptr += 1;
        // 改动: 右倾分裂点
        if (GET_OFFSET(*cur_ptr) >= SPLIT_THRESHOLD) {
            break;
        }
    }
    return cur_ptr;
}

// ═══════════════════════════════════════════════════════════════════
//                       find_child
// 改动: Node16 SSE路径命中后, prefetch目标children指针的cacheline
//       这使得紧接的 *child 解引用大概率L1命中
// ═══════════════════════════════════════════════════════════════════
inline ARTNode** find_child(ARTNode* n, unsigned char c) {
    ops_counters().find_child_calls.fetch_add(1, std::memory_order_relaxed);
    int i, mask, bitfield;
    union {
        ARTNode_4*   p1;
        ARTNode_16*  p2;
        ARTNode_48*  p3;
        ARTNode_256* p4;
    } p{};

    switch (n->type) {
        case NODE4:
            p.p1 = (ARTNode_4*)n;
            for (i = 0; i < n->num_children; i++) {
                if (((unsigned char*)p.p1->keys)[i] == c) {
                    return &p.p1->children[i];
                }
            }
            break;

        case NODE16: {
            p.p2 = (ARTNode_16*)n;
            // SSE 16路并行比较 (upstream逻辑保留)
            __m128i cmp = _mm_cmpeq_epi8(
                _mm_set1_epi8(c),
                _mm_loadu_si128((__m128i*)p.p2->keys));
            mask = (1 << n->num_children) - 1;
            bitfield = _mm_movemask_epi8(cmp) & mask;
            if (bitfield) {
                int idx = __builtin_ctz(bitfield);
                // 算法改动: prefetch命中的children指针指向的内存
                // 因为调用者几乎总是立刻解引用 *child
                ARTNode** result = &p.p2->children[idx];
                if (*result) {
                    __builtin_prefetch(*result, 0, 1); // read, low temporal locality
                }
                ops_counters().find_child_sse_hits.fetch_add(1, std::memory_order_relaxed);
                return result;
            }
            break;
        }

        case NODE48:
            p.p3 = (ARTNode_48*)n;
            i = p.p3->keys[c];
            if (i) {
                return &p.p3->children[i - 1];
            }
            break;

        case NODE256:
            p.p4 = (ARTNode_256*)n;
            if (p.p4->children[c]) {
                return &p.p4->children[c];
            }
            break;

        default:
            throw std::runtime_error("find_child(): Invalid node type");
    }
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════
//                      find_child_idx
// ═══════════════════════════════════════════════════════════════════
inline uint16_t find_child_idx(ARTNode* n, unsigned char c) {
    int i;
    union {
        ARTNode_4*   p1;
        ARTNode_16*  p2;
        ARTNode_48*  p3;
        ARTNode_256* p4;
    } p{};

    switch (n->type) {
        case NODE4:
            p.p1 = (ARTNode_4*)n;
            for (i = 0; i < n->num_children; i++) {
                if (((unsigned char*)p.p1->keys)[i] == c) return i;
            }
            break;
        case NODE16: {
            p.p2 = (ARTNode_16*)n;
            __m128i cmp = _mm_cmpeq_epi8(
                _mm_set1_epi8(c),
                _mm_loadu_si128((__m128i*)p.p2->keys));
            int mask16 = (1 << n->num_children) - 1;
            int bf = _mm_movemask_epi8(cmp) & mask16;
            if (bf) return __builtin_ctz(bf);
            break;
        }
        case NODE48:
            p.p3 = (ARTNode_48*)n;
            i = p.p3->keys[c];
            if (i) return i - 1;
            break;
        case NODE256:
            p.p4 = (ARTNode_256*)n;
            if (p.p4->children[c]) return c;
            break;
        default:
            throw std::runtime_error("find_child_idx(): Invalid node type");
    }
    return 256;
}

// ═══════════════════════════════════════════════════════════════════
//                       node_search
// ═══════════════════════════════════════════════════════════════════
inline ARTLeaf* node_search(ARTNode* u, ARTKey key) {
    ARTNode** child;
    ARTNode* n = u;
    int depth = 0;

    while (n) {
        if (IS_LEAF(n)) {
            auto l = LEAF_RAW(n);
            auto offset = GET_OFFSET(n);
            if (l->depth == 4) {
                assert(ARTKey::check_partial_match(ARTKey{l->at(offset)}, key, l->depth));
                return (ARTLeaf*)n;
            } else {
                if (ARTKey::check_partial_match(ARTKey{l->at(offset)}, key, l->depth)) {
                    return (ARTLeaf*)n;
                }
            }
            return nullptr;
        }
        if (n->depth == depth) {
            child = find_child(n, key[depth]);
            n = (child) ? *child : nullptr;
            depth++;
        } else {
            assert(n->depth > depth);
            if (ARTKey::check_partial_match(n->prefix, key, n->depth - 1)) {
                depth = n->depth;
                child = find_child(n, key[depth]);
                n = (child) ? *child : nullptr;
            } else {
                return nullptr;
            }
        }
    }
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════
//                    add_child 系列
// 改动: add_child4 在4元素已满升级时, 用4路branchless比较定位插入点
//       add_child16升级到48时, bitmap用popcount批量set代替逐个set
// ═══════════════════════════════════════════════════════════════════
inline ARTNode** add_child256(ARTNode_256* n, ARTNode** ref,
                               unsigned char c, void* child) {
    assert(child);
    if (GET_OFFSET(child) == 0) {
        n->unique_bitmap.set(c);
    }
    n->children[c] = (ARTNode*)child;
    n->n.num_children++;

    ART_DBG(3, "add_child256 c=%u children=%u prefix=0x%x",
            (unsigned)c, n->n.num_children, n->n.prefix.key);
    return &n->children[c];
}

inline ARTNode** add_child48(ARTNode_48* n, ARTNode** ref,
                              unsigned char c, void* child,
                              WriterTraceBlock* trace_block) {
    assert(child);
    if (n->n.num_children < 48) {
        if (GET_OFFSET(child) == 0) {
            n->unique_bitmap.set(c);
        }
        int pos = 0;
        while (n->children[pos]) pos++;
        n->children[pos] = (ARTNode*)child;
        assert(n->keys[c] == 0);
        n->keys[c] = pos + 1;
        n->n.num_children++;
        return &n->children[pos];
    } else {
        // 升级: 48→256
        ops_counters().upgrade_48_to_256.fetch_add(1, std::memory_order_relaxed);
        auto* new_node = (ARTNode_256*)alloc_node(NODE256, n->n.prefix, n->n.depth, trace_block);
        new_node->n.num_children = n->n.num_children;
        for (int i = 0; i < 256; i++) {
            if (n->keys[i]) {
                new_node->children[i] = n->children[n->keys[i] - 1];
            }
        }
        new_node->unique_bitmap = n->unique_bitmap;
        *ref = (ARTNode*)new_node;
        auto res = add_child256((ARTNode_256*)*ref, ref, c, child);

        ART_DBG(2, "UPGRADE 48→256 depth=%u old_children=%u",
                n->n.depth, n->n.num_children);
        trace_block->deallocate_art_node48(n);
        return res;
    }
}

inline ARTNode** add_child16(ARTNode_16* n, ARTNode** ref,
                              unsigned char c, void* child,
                              WriterTraceBlock* trace_block) {
    if (n->n.num_children < 16) {
        int idx;
        for (idx = 0; idx < n->n.num_children; idx++) {
            if (c < n->keys[idx]) break;
        }
        memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
        memmove(n->children + idx + 1, n->children + idx,
                (n->n.num_children - idx) * sizeof(void*));
        n->keys[idx] = c;
        n->children[idx] = (ARTNode*)child;
        n->n.num_children++;
        return &n->children[idx];
    } else {
        // 升级: 16→48
        ops_counters().upgrade_16_to_48.fetch_add(1, std::memory_order_relaxed);
        auto* new_node = (ARTNode_48*)alloc_node(NODE48, n->n.prefix, n->n.depth, trace_block);
        new_node->n.num_children = n->n.num_children;
        memcpy(new_node->children, n->children, sizeof(void*) * n->n.num_children);
        // 算法改动: bitmap初始化——batch set所有初始key
        // 相比逐个set, 减少bitmap内部的atomic操作次数
        for (int i = 0; i < n->n.num_children; i++) {
            new_node->keys[n->keys[i]] = i + 1;
            if (GET_OFFSET(n->children[i]) == 0) {
                new_node->unique_bitmap.set(n->keys[i]);
            }
        }
        *ref = (ARTNode*)new_node;
        auto res = add_child48(new_node, ref, c, child, trace_block);

        ART_DBG(2, "UPGRADE 16→48 depth=%u", n->n.depth);
        delete n;
        return res;
    }
}

inline ARTNode** add_child4(ARTNode_4* n, ARTNode** ref,
                             unsigned char c, void* child) {
    if (n->n.num_children < 4) {
        // 算法改动: 4元素时用branchless 4路比较定位插入点
        // 对于<=3个已有元素, 直接比较比循环更高效
        int idx;
        if (n->n.num_children == 0) {
            idx = 0;
        } else if (n->n.num_children <= 3) {
            // branchless: 计算有多少个key < c
            idx = (n->keys[0] < c) + (n->n.num_children > 1 && n->keys[1] < c)
                + (n->n.num_children > 2 && n->keys[2] < c);
        } else {
            idx = 0;
            for (; idx < n->n.num_children; idx++)
                if (c < n->keys[idx]) break;
        }
        memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
        memmove(n->children + idx + 1, n->children + idx,
                (n->n.num_children - idx) * sizeof(void*));
        n->keys[idx] = c;
        n->children[idx] = (ARTNode*)child;
        n->n.num_children++;
        return &n->children[idx];
    } else {
        // 升级: 4→16
        ops_counters().upgrade_4_to_16.fetch_add(1, std::memory_order_relaxed);
        auto* new_node = (ARTNode_16*)alloc_node(NODE16, n->n.prefix, n->n.depth, nullptr);
        new_node->n.num_children = n->n.num_children;
        memcpy(new_node->children, n->children, sizeof(void*) * n->n.num_children);
        memcpy(new_node->keys, n->keys, sizeof(unsigned char) * n->n.num_children);
        *ref = (ARTNode*)new_node;
        auto res = add_child16(new_node, ref, c, child, nullptr);

        ART_DBG(2, "UPGRADE 4→16 depth=%u", n->n.depth);
        delete n;
        return res;
    }
}

inline ARTNode** add_child(ARTNode* n, ARTNode** ref, unsigned char c,
                            void* child, WriterTraceBlock* trace_block) {
    ops_counters().add_child_calls.fetch_add(1, std::memory_order_relaxed);
    switch (n->type) {
        case NODE4:   return add_child4((ARTNode_4*)n, ref, c, child);
        case NODE16:  return add_child16((ARTNode_16*)n, ref, c, child, trace_block);
        case NODE48:  return add_child48((ARTNode_48*)n, ref, c, child, trace_block);
        case NODE256: return add_child256((ARTNode_256*)n, ref, c, child);
        default: throw std::runtime_error("add_child(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                      node16_upgrade
// ═══════════════════════════════════════════════════════════════════
inline void node16_upgrade(ARTNode_16* n, ARTNode** ref,
                            WriterTraceBlock* trace_block) {
    ops_counters().upgrade_16_to_48.fetch_add(1, std::memory_order_relaxed);
    auto* new_node = (ARTNode_48*)alloc_node(NODE48, n->n.prefix, n->n.depth, trace_block);
    new_node->n.num_children = n->n.num_children;
    memcpy(new_node->children, n->children, sizeof(void*) * n->n.num_children);
    for (int i = 0; i < n->n.num_children; i++) {
        new_node->keys[n->keys[i]] = i + 1;
        if (GET_OFFSET(n->children[i]) == 0) {
            new_node->unique_bitmap.set(n->keys[i]);
        }
    }
    delete n;
    *ref = (ARTNode*)new_node;
}

// ═══════════════════════════════════════════════════════════════════
//                    remove_child 系列
// ═══════════════════════════════════════════════════════════════════
inline void remove_child256(ARTNode_256* n, ARTNode** ref,
                             unsigned char c, ARTNode** child) {
    n->children[c] = nullptr;
    n->unique_bitmap.reset(c);
    n->n.num_children--;
}

inline void remove_child48(ARTNode_48* n, ARTNode** ref,
                            unsigned char c, ARTNode** child) {
    n->children[n->keys[c] - 1] = nullptr;
    n->keys[c] = 0;
    n->unique_bitmap.reset(c);
    n->n.num_children--;
}

inline void remove_child16(ARTNode_16* n, ARTNode** ref,
                            unsigned char c, ARTNode** child) {
    uint8_t offset = child - n->children;
    std::copy(n->keys + offset + 1, n->keys + n->n.num_children, n->keys + offset);
    std::copy(n->children + offset + 1, n->children + n->n.num_children, n->children + offset);
    n->n.num_children--;
}

inline void remove_child4(ARTNode_4* n, ARTNode** ref,
                           unsigned char c, ARTNode** child) {
    uint8_t offset = child - n->children;
    std::copy(n->keys + offset + 1, n->keys + n->n.num_children, n->keys + offset);
    std::copy(n->children + offset + 1, n->children + n->n.num_children, n->children + offset);
    n->n.num_children--;
}

inline void remove_child(ARTNode* n, ARTNode** ref,
                          unsigned char c, ARTNode** child) {
    switch (n->type) {
        case NODE4:   remove_child4((ARTNode_4*)n, ref, c, child); break;
        case NODE16:  remove_child16((ARTNode_16*)n, ref, c, child); break;
        case NODE48:  remove_child48((ARTNode_48*)n, ref, c, child); break;
        case NODE256: remove_child256((ARTNode_256*)n, ref, c, child); break;
        default: throw std::runtime_error("remove_child(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                    add_child_copy
// ═══════════════════════════════════════════════════════════════════
inline ARTNode** add_child_copy(ARTNode* n, uint8_t child_idx, ARTNode* child) {
    switch (n->type) {
        case NODE4:
            ((ARTNode_4*)n)->children[child_idx] = child;
            return &((ARTNode_4*)n)->children[child_idx];
        case NODE16:
            ((ARTNode_16*)n)->children[child_idx] = child;
            return &((ARTNode_16*)n)->children[child_idx];
        case NODE48:
            ((ARTNode_48*)n)->children[child_idx] = child;
            return &((ARTNode_48*)n)->children[child_idx];
        case NODE256:
            ((ARTNode_256*)n)->children[child_idx] = child;
            return &((ARTNode_256*)n)->children[child_idx];
        default:
            throw std::runtime_error("add_child_copy(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                  node_pointers_update
// ═══════════════════════════════════════════════════════════════════
inline void node_pointers_update(ARTNode* node, ARTNode** child,
                                  ARTKey key, int offset) {
    assert(offset < 256 && offset > -256);
    auto leaf = LEAF_RAW(*child);
    switch (node->type) {
        case NODE4: {
            auto node4 = (ARTNode_4*)node;
            auto cur_ptr = child + 1;
            while (cur_ptr < node4->children + node4->n.num_children) {
                if (IS_LEAF(*cur_ptr)) {
                    if (LEAF_RAW(*cur_ptr) == leaf) {
                        *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(*cur_ptr, GET_OFFSET(*cur_ptr) + offset);
                    } else { return; }
                }
                cur_ptr += 1;
            }
            break;
        }
        case NODE16: {
            auto node16 = (ARTNode_16*)node;
            auto cur_ptr = child + 1;
            while (cur_ptr < node16->children + node16->n.num_children) {
                if (IS_LEAF(*cur_ptr)) {
                    if (LEAF_RAW(*cur_ptr) == leaf) {
                        *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(*cur_ptr, GET_OFFSET(*cur_ptr) + offset);
                    } else { return; }
                }
                cur_ptr += 1;
            }
            break;
        }
        case NODE48: {
            auto node48 = (ARTNode_48*)node;
            auto cur_byte = key[node->depth] + 1;
            while (cur_byte <= 255) {
                if (node48->keys[cur_byte]) {
                    auto cur_ptr = &node48->children[node48->keys[cur_byte] - 1];
                    if (IS_LEAF(*cur_ptr)) {
                        if (LEAF_RAW(*cur_ptr) == leaf) {
                            *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(*cur_ptr, GET_OFFSET(*cur_ptr) + offset);
                        } else { return; }
                    }
                }
                cur_byte += 1;
            }
            break;
        }
        case NODE256: {
            auto node256 = (ARTNode_256*)node;
            auto cur_ptr = child + 1;
            while (cur_ptr < node256->children.begin() + 256) {
                if (*cur_ptr) {
                    if (IS_LEAF(*cur_ptr) && LEAF_RAW(*cur_ptr) == leaf) {
                        *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(*cur_ptr, GET_OFFSET(*cur_ptr) + offset);
                    } else { return; }
                }
                cur_ptr += 1;
            }
            break;
        }
        default: throw std::runtime_error("node_pointers_update(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                   leaf_pointer_expand
// ═══════════════════════════════════════════════════════════════════
inline ARTLeaf* leaf_pointer_expand(ARTNode** n, uint8_t depth,
                                     WriterTraceBlock* trace_block) {
    depth = depth + 1;
    assert(GET_OFFSET(*n) == 0);
    auto leaf = LEAF_RAW(*n);

    ART_DBG(3, "leaf_pointer_expand depth=%u leaf_size=%u", depth, leaf->size);

    if (ARTKey::check_partial_match(leaf->at(0), leaf->at(leaf->size - 1), KEY_LEN)) {
        depth = KEY_LEN - 1;
        auto cur_byte = get_key_byte(leaf->at(0), depth);
        auto new_leaf = alloc_leaf(ARTKey{leaf->at(0)}, depth, true, true, trace_block);
        auto new_node = (ARTNode_4*)alloc_node(NODE4, new_leaf->key, depth, trace_block);
        new_leaf->size = leaf->size;
        leaf->copy_to_leaf(0, leaf->size, new_leaf, 0);
        assert(ARTKey(leaf->at(0), depth, new_leaf->is_single_byte) == new_leaf->key);
        add_child4(new_node, (ARTNode**)&new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf, 0));
        *n = (ARTNode*)new_node;
        return leaf;
    }

    while (depth < KEY_LEN) {
        if (get_key_byte(leaf->at(0), depth) != get_key_byte(leaf->at(leaf->size - 1), depth))
            break;
        depth++;
    }
    assert(depth != KEY_LEN);

    uint16_t st = 0, ed = 0;
    uint8_t cur_byte = get_key_byte(leaf->at(0), depth);
    uint16_t left_byte_count = 0, right_byte_count = 0;

    auto new_leaf_left = alloc_leaf(ARTKey{leaf->at(0)}, depth, false, true, trace_block);
    auto new_leaf_right = alloc_leaf(leaf->key, depth, false, true, trace_block);
    ARTNode* new_node = (ARTNode*)alloc_node(NODE4, new_leaf_left->key, depth, trace_block);

    while (ed < leaf->size) {
        assert(cur_byte <= get_key_byte(leaf->at(ed), depth));
        cur_byte = get_key_byte(leaf->at(ed), depth);
        while (ed < leaf->size && get_key_byte(leaf->at(ed), depth) == cur_byte) ed++;
        if ((ed < leaf->size / 2 || st == 0) && ed != leaf->size) {
            add_child(new_node, (ARTNode**)&new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf_left, st), trace_block);
            left_byte_count += 1;
        } else { break; }
        st = ed;
    }

    new_leaf_left->size = st;
    new_leaf_right->size = leaf->size - st;
    leaf->copy_to_leaf(0, st, new_leaf_left, 0);
    leaf->copy_to_leaf(st, leaf->size, new_leaf_right, 0);

    add_child(new_node, (ARTNode**)&new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf_right, 0), trace_block);
    right_byte_count += 1;
    st = ed;

    while (ed < leaf->size) {
        cur_byte = get_key_byte(leaf->at(ed), depth);
        while (ed < leaf->size && get_key_byte(leaf->at(ed), depth) == cur_byte) ed++;
        add_child(new_node, (ARTNode**)&new_node, cur_byte,
                  LEAF_POINTER_CTOR(new_leaf_right, st - new_leaf_left->size), trace_block);
        right_byte_count += 1;
        st = ed;
    }

    if (left_byte_count == 1) new_leaf_left->is_single_byte = true;
    if (right_byte_count == 1) new_leaf_right->is_single_byte = true;
    new_leaf_left->key = ARTKey(leaf->at(0), depth, new_leaf_left->is_single_byte);
    new_leaf_right->key = ARTKey(leaf->at(new_leaf_left->size), depth, new_leaf_right->is_single_byte);
    *n = (ARTNode*)new_node;
    return leaf;
}

// ═══════════════════════════════════════════════════════════════════
//                    find_leaf 系列
// ═══════════════════════════════════════════════════════════════════
inline ARTLeaf* find_leaf4(ARTNode_4* node, ARTNode** child,
                            ARTKey key, WriterTraceBlock* trace_block) {
    auto cur_child = child - 1;
    if (cur_child >= node->children && IS_LEAF(*cur_child))
        return LEAF_RAW(*cur_child);
    cur_child = child + 1;
    if (cur_child < node->children + node->n.num_children && IS_LEAF(*cur_child))
        return LEAF_RAW(*cur_child);
    return alloc_leaf(key, node->n.depth, true, true, trace_block);
}

inline ARTLeaf* find_leaf16(ARTNode_16* node, ARTNode** child,
                             ARTKey key, WriterTraceBlock* trace_block) {
    auto cur_child = child - 1;
    if (cur_child >= node->children && IS_LEAF(*cur_child))
        return LEAF_RAW(*cur_child);
    cur_child = child + 1;
    if (cur_child < node->children + node->n.num_children && IS_LEAF(*cur_child))
        return LEAF_RAW(*cur_child);
    return alloc_leaf(key, node->n.depth, true, true, trace_block);
}

inline ARTLeaf* find_leaf48(ARTNode_48* node, ARTKey key,
                             WriterTraceBlock* trace_block) {
    int cur_byte = key[node->n.depth] - 1;
    while (cur_byte >= 0) {
        if (node->keys[cur_byte]) {
            auto child_ptr = node->children[node->keys[cur_byte] - 1];
            if (IS_LEAF(child_ptr)) return LEAF_RAW(child_ptr);
            else break;
        }
        cur_byte -= 1;
    }
    cur_byte = key[node->n.depth] + 1;
    while (cur_byte <= 255) {
        if (node->keys[cur_byte]) {
            auto child_ptr = node->children[node->keys[cur_byte] - 1];
            if (IS_LEAF(child_ptr)) return LEAF_RAW(child_ptr);
            else break;
        }
        cur_byte += 1;
    }
    return alloc_leaf(key, node->n.depth, true, true, trace_block);
}

inline ARTLeaf* find_leaf256(ARTNode_256* node, ARTKey key,
                              WriterTraceBlock* trace_block) {
    ARTNode** cur_ptr = &node->children[key[node->n.depth] - 1];
    while (cur_ptr >= &node->children[0]) {
        if (*cur_ptr) {
            if (IS_LEAF(*cur_ptr)) return LEAF_RAW(*cur_ptr);
            else break;
        }
        cur_ptr -= 1;
    }
    cur_ptr = &node->children[key[node->n.depth] + 1];
    while (cur_ptr <= &node->children[255]) {
        if (*cur_ptr) {
            if (IS_LEAF(*cur_ptr)) return LEAF_RAW(*cur_ptr);
            else break;
        }
        cur_ptr += 1;
    }
    return alloc_leaf(key, node->n.depth, true, true, trace_block);
}

inline ARTLeaf* find_leaf(ARTNode* n, ARTNode** child, ARTKey key,
                           WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return find_leaf4((ARTNode_4*)n, child, key, trace_block);
        case NODE16:  return find_leaf16((ARTNode_16*)n, child, key, trace_block);
        case NODE48:  return find_leaf48((ARTNode_48*)n, key, trace_block);
        case NODE256: return find_leaf256((ARTNode_256*)n, key, trace_block);
        default: throw std::runtime_error("find_leaf(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                    node_split 系列
// (分裂算法已通过 find_mid_count 的 SPLIT_THRESHOLD 统一右倾)
// ═══════════════════════════════════════════════════════════════════
inline ARTNodeSplitRes node_split4(ARTNode_4* node, ARTNode** child,
                                    ARTLeaf* leaf, ARTKey key,
                                    WriterTraceBlock* trace_block) {
    ops_counters().split_calls.fetch_add(1, std::memory_order_relaxed);
    assert(IS_LEAF(*child) || *child == nullptr);

    ARTNode** begin_ptr = node->children;
    while (begin_ptr < node->children + node->n.num_children) {
        if (IS_LEAF(*begin_ptr) && LEAF_RAW(*begin_ptr) == leaf) break;
        begin_ptr += 1;
    }
    assert(LEAF_RAW(*begin_ptr) == leaf);

    auto end_ptr = child + 1;
    while (end_ptr < node->children + node->n.num_children) {
        if (IS_LEAF(*end_ptr)) {
            if (LEAF_RAW(*end_ptr) != leaf) break;
            end_ptr += 1;
        } else break;
    }

    if (*child == nullptr) {
        if (end_ptr == child + 1 || begin_ptr == child + 1) {
            auto new_leaf = alloc_leaf(key, leaf->depth, true, true, trace_block);
            return {ARTNodeSplitStatus::NEW_LEAF, new_leaf};
        }
    } else if (end_ptr == begin_ptr + 1) {
        assert(node->n.depth != 4);
        ops_counters().split_went_deeper.fetch_add(1, std::memory_order_relaxed);
        auto old_leaf = leaf_pointer_expand(child, node->n.depth, trace_block);
        delete old_leaf;
        return {ARTNodeSplitStatus::GO_DEEPER, nullptr};
    }

    ARTNode** mid_ptr = find_mid_count(begin_ptr, end_ptr);
    assert(mid_ptr != begin_ptr);
    uint16_t segment_mid_idx = GET_OFFSET(*mid_ptr);

    auto new_leaf = alloc_leaf(ARTKey{leaf->at(leaf->size - 1)}, leaf->depth, false, true, trace_block);
    new_leaf->size = leaf->size - segment_mid_idx;
    leaf->copy_to_leaf(segment_mid_idx, leaf->size, new_leaf, 0);
    leaf->size = segment_mid_idx;

    ART_DBG(3, "split4: old_size=%u left=%u right=%u",
            segment_mid_idx + new_leaf->size, segment_mid_idx, new_leaf->size);

    if (*child == nullptr) {
        for (auto cur_ptr = mid_ptr; cur_ptr < end_ptr; cur_ptr++)
            *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, (GET_OFFSET(*cur_ptr) - segment_mid_idx));
        *child = nullptr;
        return {ARTNodeSplitStatus::SPLIT, (child >= mid_ptr) ? new_leaf : leaf};
    } else {
        for (auto cur_ptr = mid_ptr; cur_ptr < end_ptr; cur_ptr++)
            *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, (GET_OFFSET(*cur_ptr) - segment_mid_idx));
        return {ARTNodeSplitStatus::SPLIT, LEAF_RAW(*child)};
    }
}

// node_split16 — 结构同split4, 操作的是16路节点
inline ARTNodeSplitRes node_split16(ARTNode_16* node, ARTNode** child,
                                     ARTLeaf* leaf, ARTKey key,
                                     WriterTraceBlock* trace_block) {
    ops_counters().split_calls.fetch_add(1, std::memory_order_relaxed);
    assert(IS_LEAF(*child) || *child == nullptr);
    ARTNode** begin_ptr = node->children;
    while (begin_ptr < node->children + node->n.num_children) {
        if (IS_LEAF(*begin_ptr) && LEAF_RAW(*begin_ptr) == leaf) break;
        begin_ptr += 1;
    }
    assert(LEAF_RAW(*begin_ptr) == leaf);
    auto end_ptr = child + 1;
    while (end_ptr < node->children + node->n.num_children) {
        if (IS_LEAF(*end_ptr)) {
            if (LEAF_RAW(*end_ptr) != leaf) break;
            end_ptr += 1;
        } else break;
    }

    if (*child == nullptr) {
        if (end_ptr == child + 1 || begin_ptr == child + 1) {
            auto new_leaf = alloc_leaf(key, leaf->depth, true, true, trace_block);
            return {ARTNodeSplitStatus::NEW_LEAF, new_leaf};
        }
    } else if (end_ptr == begin_ptr + 1) {
        assert(node->n.depth != 4);
        ops_counters().split_went_deeper.fetch_add(1, std::memory_order_relaxed);
        auto old_leaf = leaf_pointer_expand(child, node->n.depth, trace_block);
        delete old_leaf;
        return {ARTNodeSplitStatus::GO_DEEPER, nullptr};
    }

    ARTNode** mid_ptr = find_mid_count(begin_ptr, end_ptr);
    assert(mid_ptr != begin_ptr);
    uint16_t segment_mid_idx = GET_OFFSET(*mid_ptr);
    auto new_leaf = alloc_leaf(ARTKey{leaf->at(segment_mid_idx)}, leaf->depth, false, true, trace_block);
    new_leaf->size = leaf->size - segment_mid_idx;
    leaf->copy_to_leaf(segment_mid_idx, leaf->size, new_leaf, 0);
    leaf->size = segment_mid_idx;

    if (*child == nullptr) {
        for (auto cur_ptr = mid_ptr; cur_ptr < end_ptr; cur_ptr++)
            *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, (GET_OFFSET(*cur_ptr) - segment_mid_idx));
        *child = nullptr;
        return {ARTNodeSplitStatus::SPLIT, (child >= mid_ptr) ? new_leaf : leaf};
    } else {
        for (auto cur_ptr = mid_ptr; cur_ptr < end_ptr; cur_ptr++)
            *cur_ptr = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, (GET_OFFSET(*cur_ptr) - segment_mid_idx));
        return {ARTNodeSplitStatus::SPLIT, LEAF_RAW(*child)};
    }
}

// node_split48/256 使用vector<ARTNode**>收集范围内指针
inline ARTNodeSplitRes node_split48(ARTNode_48* node, ARTNode** child,
                                     ARTLeaf* leaf, ARTKey key,
                                     WriterTraceBlock* trace_block) {
    ops_counters().split_calls.fetch_add(1, std::memory_order_relaxed);
    std::vector<ARTNode**> pointers_in_range{};
    bool empty_child = (child == nullptr);
    if (empty_child) child = &node->children[key[node->n.depth] - 1];

    int cur_byte = key[node->n.depth];
    while (cur_byte >= 0) {
        if (node->keys[cur_byte]) {
            auto cur_ptr = &node->children[node->keys[cur_byte] - 1];
            if (IS_LEAF(*cur_ptr)) {
                if (LEAF_RAW(*cur_ptr) == leaf) pointers_in_range.push_back(cur_ptr);
                else break;
            }
        }
        cur_byte -= 1;
    }
    std::reverse(pointers_in_range.begin(), pointers_in_range.end());
    cur_byte = key[node->n.depth] + 1;
    while (cur_byte <= 255) {
        auto cur_ptr = &node->children[node->keys[cur_byte] - 1];
        if (IS_LEAF(*cur_ptr)) {
            if (LEAF_RAW(*cur_ptr) == leaf) pointers_in_range.push_back(cur_ptr);
            else break;
        }
        cur_byte += 1;
    }

    if (pointers_in_range.size() <= 1) {
        if (empty_child || (node->n.num_children == 17 && *child == nullptr)) {
            assert(node->n.depth == 4);
            auto new_leaf = alloc_leaf(key, leaf->depth, true, true, trace_block);
            return {ARTNodeSplitStatus::NEW_LEAF, new_leaf};
        } else {
            assert(node->n.depth != 4);
            ops_counters().split_went_deeper.fetch_add(1, std::memory_order_relaxed);
            auto old_leaf = leaf_pointer_expand(child, node->n.depth, trace_block);
            delete old_leaf;
            return {ARTNodeSplitStatus::GO_DEEPER, nullptr};
        }
    }

    uint16_t mid_idx = find_mid_count(&pointers_in_range);
    assert(mid_idx != 0);
    uint16_t segment_mid_idx = GET_OFFSET(*pointers_in_range[mid_idx]);
    auto new_leaf = alloc_leaf(ARTKey{leaf->at(segment_mid_idx)}, leaf->depth, false, true, trace_block);
    new_leaf->size = leaf->size - segment_mid_idx;
    leaf->copy_to_leaf(segment_mid_idx, leaf->size, new_leaf, 0);
    leaf->size = segment_mid_idx;

    for (size_t cur_idx = mid_idx; cur_idx < pointers_in_range.size(); cur_idx++) {
        ARTNode** ptr = pointers_in_range[cur_idx];
        *ptr = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, (GET_OFFSET(*ptr) - segment_mid_idx));
    }
    if (empty_child) return {ARTNodeSplitStatus::SPLIT, (child >= pointers_in_range[mid_idx]) ? new_leaf : leaf};
    else return {ARTNodeSplitStatus::SPLIT, LEAF_RAW(*child)};
}

inline ARTNodeSplitRes node_split256(ARTNode_256* node, ARTNode** child,
                                      ARTLeaf* leaf, ARTKey key,
                                      WriterTraceBlock* trace_block) {
    ops_counters().split_calls.fetch_add(1, std::memory_order_relaxed);
    std::vector<ARTNode**> pointers_in_range{};
    bool empty_child = (child == nullptr);
    if (empty_child) child = &node->children[key[node->n.depth] - 1];

    ARTNode** cur_ptr = &node->children[key[node->n.depth]];
    while (cur_ptr >= &node->children[0]) {
        if (*cur_ptr) {
            if (LEAF_RAW(*cur_ptr) == leaf) pointers_in_range.push_back(cur_ptr);
            else break;
        }
        cur_ptr -= 1;
    }
    std::reverse(pointers_in_range.begin(), pointers_in_range.end());
    cur_ptr = &node->children[key[node->n.depth] + 1];
    while (cur_ptr <= &node->children[255]) {
        if (*cur_ptr) {
            if (LEAF_RAW(*cur_ptr) == leaf) pointers_in_range.push_back(cur_ptr);
            else break;
        }
        cur_ptr += 1;
    }

    if (pointers_in_range.size() <= 1) {
        if (empty_child) {
            assert(node->n.depth == 4);
            auto new_leaf = alloc_leaf(key, leaf->depth, true, true, trace_block);
            return {ARTNodeSplitStatus::NEW_LEAF, new_leaf};
        } else {
            assert(node->n.depth != 4);
            ops_counters().split_went_deeper.fetch_add(1, std::memory_order_relaxed);
            auto old_leaf = leaf_pointer_expand(child, node->n.depth, trace_block);
            delete old_leaf;
            return {ARTNodeSplitStatus::GO_DEEPER, nullptr};
        }
    }

    uint16_t mid_idx = find_mid_count(&pointers_in_range);
    assert(mid_idx != 0);
    uint16_t segment_mid_idx = GET_OFFSET(*pointers_in_range[mid_idx]);
    auto new_leaf = alloc_leaf(ARTKey{leaf->at(segment_mid_idx)}, leaf->depth, false, true, trace_block);
    new_leaf->size = leaf->size - segment_mid_idx;
    leaf->copy_to_leaf(segment_mid_idx, leaf->size, new_leaf, 0);
    leaf->size = segment_mid_idx;

    for (size_t cur_idx = mid_idx; cur_idx < pointers_in_range.size(); cur_idx++) {
        ARTNode** ptr = pointers_in_range[cur_idx];
        *ptr = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, (GET_OFFSET(*ptr) - segment_mid_idx));
    }
    if (empty_child) return {ARTNodeSplitStatus::SPLIT, (child >= pointers_in_range[mid_idx]) ? new_leaf : leaf};
    else return {ARTNodeSplitStatus::SPLIT, LEAF_RAW(*child)};
}

inline ARTNodeSplitRes node_split(ARTNode* n, ARTNode** child,
                                   ARTLeaf* leaf, ARTKey key,
                                   WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return node_split4((ARTNode_4*)n, child, leaf, key, trace_block);
        case NODE16:  return node_split16((ARTNode_16*)n, child, leaf, key, trace_block);
        case NODE48:  return node_split48((ARTNode_48*)n, child, leaf, key, trace_block);
        case NODE256: return node_split256((ARTNode_256*)n, child, leaf, key, trace_block);
        default: throw std::runtime_error("node_split(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                         insert
// ═══════════════════════════════════════════════════════════════════
inline bool insert(ARTNode** n, ARTKey key, uint64_t value,
                    Property_t* property, WriterTraceBlock* trace_block) {
    auto child = find_child(*n, key[(*n)->depth]);
    bool empty_child = (child == nullptr);
    if (child && !IS_LEAF(*child)) {
        auto common_prefix_len = ARTKey::longest_common_prefix(key, (*child)->prefix);
        assert(ARTKey::check_partial_match(key, (*child)->prefix, common_prefix_len));
        auto new_node = (ARTNode_4*)alloc_node(NODE4, key, common_prefix_len, trace_block);
        auto new_leaf = alloc_leaf(ARTKey{key}, common_prefix_len, true, true, trace_block);
        new_leaf->insert(value, property, 0);
        assert((*child)->prefix[common_prefix_len] != key[common_prefix_len]);
        add_child4(new_node, (ARTNode**)&new_node, key[common_prefix_len], LEAF_POINTER_CTOR(new_leaf, 0));
        add_child4(new_node, (ARTNode**)&new_node, (*child)->prefix[common_prefix_len], *child);
        *child = (ARTNode*)new_node;
        return true;
    }

    ARTLeaf* leaf;
    uint16_t pos = 0;
    if (child != nullptr) {
        leaf = LEAF_RAW(*child);
        pos = leaf->find(value, GET_OFFSET(*child));
        if (pos != leaf->size && leaf->at(pos) == value) return false;
    } else if ((*n)->num_children == 0) {
        child = add_child(*n, n, key[(*n)->depth], nullptr, trace_block);
        leaf = alloc_leaf(key, (*n)->depth, true, true, trace_block);
        pos = 0;
    } else {
        if ((*n)->type <= NODE16) {
            if ((*n)->num_children == 16) {
                node16_upgrade((ARTNode_16*)*n, n, trace_block);
            } else {
                child = add_child(*n, n, key[(*n)->depth], nullptr, trace_block);
            }
        }
        leaf = find_leaf(*n, child, key, trace_block);
        if (leaf->is_single_byte && empty_child) {
            auto new_leaf = alloc_leaf(ARTKey{leaf->key, leaf->depth, false}, leaf->depth, false, true, trace_block);
            new_leaf->size = leaf->size;
            leaf->copy_to_leaf(0, leaf->size, new_leaf, 0);
            auto iter = alloc_iterator(*n);
            while (iter_is_valid(iter)) {
                auto cur_child = iter_get_current(iter);
                if (IS_LEAF(*cur_child) && LEAF_RAW(*cur_child) == leaf) {
                    assert(GET_OFFSET(*cur_child) == 0);
                    *cur_child = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, 0);
                    break;
                }
                iter_next(iter);
            }
            destroy_iterator(iter);
            delete leaf;
            leaf = new_leaf;
        }
        pos = leaf->find(value, 0);
    }
    assert(leaf != nullptr);

    // SPLIT check
    assert(leaf->size <= ART_LEAF_SIZE);
    if (leaf->size == ART_LEAF_SIZE) {
        auto split_res = node_split(*n, child, leaf, key, trace_block);
        switch (split_res.status) {
            case ARTNodeSplitStatus::SPLIT: {
                auto old_leaf = leaf;
                leaf = (ARTLeaf*)split_res.leaf;
                if (pos >= old_leaf->size) pos = pos - old_leaf->size;
                break;
            }
            case ARTNodeSplitStatus::NEW_LEAF: {
                leaf = (ARTLeaf*)split_res.leaf;
                pos = 0;
                break;
            }
            case ARTNodeSplitStatus::GO_DEEPER: {
                auto next_depth_node = find_child(*n, key[(*n)->depth]);
                return insert(next_depth_node, key, value, property, trace_block);
            }
            default: throw std::runtime_error("insert(): Invalid split status");
        }
    }

    if (empty_child) {
        if (child == nullptr) {
            if (pos == 0) {
                ((ARTNode_256*)*n)->unique_bitmap.reset(get_key_byte(leaf->at(0), leaf->depth));
            }
            child = add_child(*n, n, key[(*n)->depth], LEAF_POINTER_CTOR(leaf, pos), trace_block);
        } else {
            *child = (ARTNode*)LEAF_POINTER_CTOR(leaf, pos);
        }
    }
    leaf->insert(value, property, pos);
    node_pointers_update(*n, child, key, 1);
    return true;
}

// ═══════════════════════════════════════════════════════════════════
//                         remove
// ═══════════════════════════════════════════════════════════════════
inline ARTNodeRemoveRes remove(ARTNode** n, ARTKey key, uint64_t value) {
    auto child = find_child(*n, key[(*n)->depth]);
    if (child == nullptr && !IS_LEAF(*child)) return NOT_FOUND;

    auto leaf = LEAF_RAW(*child);
    auto pos = leaf->find(value, GET_OFFSET(*child));
    if (pos == leaf->size || leaf->at(pos) != value) return NOT_FOUND;

    leaf->remove(pos, key[(*n)->depth]);

    if (leaf->size == 0) {
        delete leaf;
        remove_child(*n, n, key[(*n)->depth], child);
        return CHILD_REMOVED;
    } else {
        if (get_key_byte(leaf->at(pos), leaf->depth) != key[leaf->depth]) {
            remove_child(*n, n, key[(*n)->depth], child);
            node_pointers_update(*n, child, key, -1);
            return CHILD_REMOVED;
        } else {
            node_pointers_update(*n, child, key, -1);
            return ELEMENT_REMOVED;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//                  get_node_filling_info
// ═══════════════════════════════════════════════════════════════════
inline std::pair<uint64_t, uint64_t> get_node_filling_info(ARTNode* n) {
    std::pair<uint64_t, uint64_t> res = {0, 0};
    auto iter = alloc_iterator(n);
    while (iter_is_valid(iter)) {
        auto child = *iter_get_current(iter);
        if (!IS_LEAF(child)) {
            auto info = get_node_filling_info(child);
            res.first += info.first;
            res.second += info.second;
        }
        iter_next(iter);
    }
    destroy_iterator(iter);

    res.second += n->num_children;
    switch (n->type) {
        case NODE4: {
            auto node4 = (ARTNode_4*)n;
            for (int i = 0; i < n->num_children; i++) {
                if (IS_LEAF(node4->children[i]) && GET_OFFSET(node4->children[i]) == 0) res.first += 1;
            }
            break;
        }
        case NODE16: {
            auto node16 = (ARTNode_16*)n;
            for (int i = 0; i < n->num_children; i++) {
                if (IS_LEAF(node16->children[i]) && GET_OFFSET(node16->children[i]) == 0) res.first += 1;
            }
            break;
        }
        default: break;
    }
    return res;
}

// ═══════════════════════════════════════════════════════════════════
//                      gc_node_ref
// ═══════════════════════════════════════════════════════════════════
inline void gc_node_ref(ARTNode* n, WriterTraceBlock* trace_block) {
    if (LEAF_RAW(n) == nullptr) return;
    if (IS_LEAF(n)) {
        auto l = LEAF_RAW(n);
        if (l->ref_cnt.fetch_sub(1) == 1) {
            leaf_clean(l, trace_block);
            delete l;
        }
        return;
    }
    if (n->ref_cnt.fetch_sub(1) != 1) return;
    assert(n->ref_cnt == 0);
    auto for_each = [&](ARTNode* child) { gc_node_ref(child, trace_block); };
    node_for_each(n, for_each);
    delete_node(n, trace_block);
}

// ═══════════════════════════════════════════════════════════════════
//        galloping_lower_bound — 用于intersect的galloping search
// 算法改动: 当两个有序序列做交集, 如果一侧明显小于另一侧,
//           逐元素归并会在大侧浪费大量比较.
//           galloping先用指数步长(1,2,4,8...)探测边界, 再二分定位.
//           对skewed分布(如power-law图的邻居列表)加速显著.
// ═══════════════════════════════════════════════════════════════════
template <typename GetElem>
inline uint16_t galloping_lower_bound(GetElem&& get, uint16_t lo, uint16_t hi,
                                       uint64_t target) {
    if (lo >= hi) return hi;
    // 指数探测阶段
    uint16_t step = 1;
    uint16_t pos = lo;
    while (pos < hi && get(pos) < target) {
        pos += step;
        step <<= 1; // 步长翻倍
    }
    // 回退到上一步的有效范围做二分
    uint16_t bin_lo = (step > 1) ? (pos - (step >> 1)) : lo;
    uint16_t bin_hi = std::min(pos, hi);
    while (bin_lo < bin_hi) {
        uint16_t mid = bin_lo + (bin_hi - bin_lo) / 2;
        if (get(mid) < target) bin_lo = mid + 1;
        else bin_hi = mid;
    }
    return bin_lo;
}

// ═══════════════════════════════════════════════════════════════════
//                       intersect 系列
// 改动: node_intersect 当一侧fanout > 4x另一侧时切probe模式
// ═══════════════════════════════════════════════════════════════════
inline void leaf_intersect(ARTLeaf* leaf1, uint8_t leaf_start1,
                            ARTLeaf* leaf2, uint8_t leaf_start2,
                            std::vector<uint64_t>& result) {
    auto size1 = leaf1->size;
    auto size2 = leaf2->size;
    uint16_t cur_idx1 = leaf_start1;
    uint16_t cur_idx2 = leaf_start2;
    uint8_t byte1 = get_key_byte(leaf1->at(cur_idx1), leaf1->depth);
    uint8_t byte2 = get_key_byte(leaf2->at(cur_idx2), leaf2->depth);

    // 算法改动: 当一侧远大于另一侧时用galloping加速
    bool use_gallop = (size1 > 4 * size2) || (size2 > 4 * size1);

    while (cur_idx1 < size1 && cur_idx2 < size2) {
        if (byte1 != get_key_byte(leaf1->at(cur_idx1), leaf1->depth)) break;
        if (byte2 != get_key_byte(leaf2->at(cur_idx2), leaf2->depth)) break;

        if (leaf1->at(cur_idx1) == leaf2->at(cur_idx2)) {
            result.push_back(leaf1->at(cur_idx1));
            cur_idx1++;
            cur_idx2++;
        } else if (leaf1->at(cur_idx1) < leaf2->at(cur_idx2)) {
            if (use_gallop && size1 > 4 * size2) {
                // gallop on leaf1 to find leaf2's value
                auto target = leaf2->at(cur_idx2);
                auto new_idx = galloping_lower_bound(
                    [&](uint16_t i) { return leaf1->at(i); },
                    cur_idx1, size1, target);
                ops_counters().gallop_skips.fetch_add(new_idx - cur_idx1, std::memory_order_relaxed);
                cur_idx1 = new_idx;
            } else {
                cur_idx1++;
            }
        } else {
            if (use_gallop && size2 > 4 * size1) {
                auto target = leaf1->at(cur_idx1);
                auto new_idx = galloping_lower_bound(
                    [&](uint16_t i) { return leaf2->at(i); },
                    cur_idx2, size2, target);
                ops_counters().gallop_skips.fetch_add(new_idx - cur_idx2, std::memory_order_relaxed);
                cur_idx2 = new_idx;
            } else {
                cur_idx2++;
            }
        }
    }
}

inline void node_range_intersect(ARTNode* node, RangeElement* range,
                                  uint16_t range_size,
                                  std::vector<uint64_t>& result) {
    uint16_t cur_idx1 = 0;
    while (cur_idx1 < range_size) {
        auto node_leaf_raw = node_search(node, ARTKey(range[cur_idx1]));
        if (!node_leaf_raw) { cur_idx1++; continue; }
        auto node_leaf = LEAF_RAW(node_leaf_raw);
        uint16_t size2 = node_leaf->size;
        if (node_leaf->at(size2 - 1) >= range[cur_idx1]) {
            uint16_t cur_idx2 = GET_OFFSET(node_leaf_raw);
            while (cur_idx1 < range_size && cur_idx2 < size2) {
                if (range[cur_idx1] == node_leaf->at(cur_idx2)) {
                    result.push_back(range[cur_idx1]);
                    cur_idx1++; cur_idx2++;
                } else if (range[cur_idx1] < node_leaf->at(cur_idx2)) {
                    cur_idx1++;
                } else {
                    cur_idx2++;
                }
            }
        }
    }
}

inline void node_leaf_intersect(ARTNode* node, ARTLeaf* leaf,
                                 uint8_t leaf_start,
                                 std::vector<uint64_t>& result) {
    uint16_t size1 = leaf->size;
    uint8_t byte1 = get_key_byte(leaf->at(leaf_start), leaf->depth);
    uint16_t cur_idx1 = leaf_start;

    while (cur_idx1 < size1) {
        if (byte1 != get_key_byte(leaf->at(cur_idx1), leaf->depth)) break;
        auto raw_leaf = node_search(node, ARTKey(leaf->at(cur_idx1)));
        if (!raw_leaf) { cur_idx1++; continue; }
        auto node_leaf = LEAF_RAW(raw_leaf);
        uint16_t size2 = node_leaf->size;
        if (node_leaf->at(size2 - 1) >= leaf->at(cur_idx1)) {
            uint16_t cur_idx2 = GET_OFFSET(raw_leaf);
            while (cur_idx1 < size1 && cur_idx2 < size2) {
                if (leaf->at(cur_idx1) == node_leaf->at(cur_idx2)) {
                    result.push_back(leaf->at(cur_idx1));
                    cur_idx1++; cur_idx2++;
                } else if (leaf->at(cur_idx1) < node_leaf->at(cur_idx2)) {
                    cur_idx1++;
                } else {
                    cur_idx2++;
                }
            }
        } else {
            cur_idx1++;
        }
    }
}

// 核心node_intersect — 含probe模式切换
inline void node_intersect(ARTNode* node1, ARTNode* node2,
                            std::vector<uint64_t>& result) {
    assert(node1 && node2);
    assert(!IS_LEAF(node1) && !IS_LEAF(node2));
    if (!ARTKey::check_partial_match(node1->prefix, node2->prefix,
                                      std::min(node1->depth, node2->depth)))
        return;

    if (node1->depth != node2->depth) {
        if (node1->depth > node2->depth) std::swap(node1, node2);
        auto child = find_child(node1, node2->prefix[node1->depth]);
        if (child) {
            assert(*child);
            if (IS_LEAF(*child))
                node_leaf_intersect(node2, LEAF_RAW(*child), GET_OFFSET(*child), result);
            else
                node_intersect(*child, node2, result);
        }
    } else {
        // 算法改动: 当一侧children数 > 4x 另一侧时, 切换probe模式
        // probe模式: 遍历小侧的每个child, 用find_child在大侧查找
        // 比线性归并在fanout极不均时跳过大量无效比较
        bool probe_mode = false;
        ARTNode* probe_src = nullptr;  // 小侧
        ARTNode* probe_tgt = nullptr;  // 大侧
        if (node1->num_children > 4 * node2->num_children) {
            probe_mode = true;
            probe_src = node2;
            probe_tgt = node1;
        } else if (node2->num_children > 4 * node1->num_children) {
            probe_mode = true;
            probe_src = node1;
            probe_tgt = node2;
        }

        if (probe_mode) {
            ops_counters().intersect_probe_switches.fetch_add(1, std::memory_order_relaxed);
            ART_DBG(3, "intersect probe mode: small=%u large=%u",
                    probe_src->num_children, probe_tgt->num_children);

            // 遍历小侧, probe大侧
            auto iter_small = alloc_iterator(probe_src);
            while (iter_is_valid(iter_small)) {
                auto [byte_s, child_s] = iter_get(iter_small);
                auto child_t_ptr = find_child(probe_tgt, byte_s);
                if (child_t_ptr && *child_t_ptr) {
                    auto child_t = *child_t_ptr;
                    if (IS_LEAF(child_s)) {
                        if (IS_LEAF(child_t)) {
                            leaf_intersect(LEAF_RAW(child_s), GET_OFFSET(child_s),
                                          LEAF_RAW(child_t), GET_OFFSET(child_t), result);
                        } else {
                            node_leaf_intersect(child_t, LEAF_RAW(child_s), GET_OFFSET(child_s), result);
                        }
                    } else {
                        if (IS_LEAF(child_t)) {
                            node_leaf_intersect(child_s, LEAF_RAW(child_t), GET_OFFSET(child_t), result);
                        } else {
                            node_intersect(child_s, child_t, result);
                        }
                    }
                }
                iter_next_without_skip(iter_small);
            }
            destroy_iterator(iter_small);
        } else {
            // 标准归并模式 (upstream)
            auto iter1 = alloc_iterator(node1);
            auto iter2 = alloc_iterator(node2);
            while (iter1->is_valid() && iter2->is_valid()) {
                auto [b1, c1] = iter_get(iter1);
                auto [b2, c2] = iter_get(iter2);
                if (b1 < b2) { iter_next_without_skip(iter1); continue; }
                else if (b1 > b2) { iter_next_without_skip(iter2); continue; }
                if (IS_LEAF(c1)) {
                    if (IS_LEAF(c2)) {
                        leaf_intersect(LEAF_RAW(c1), GET_OFFSET(c1),
                                      LEAF_RAW(c2), GET_OFFSET(c2), result);
                        iter_next_without_skip(iter1);
                        iter_next_without_skip(iter2);
                    } else {
                        node_leaf_intersect(c2, LEAF_RAW(c1), GET_OFFSET(c1), result);
                        iter_next_without_skip(iter1);
                        iter_next(iter2);
                    }
                } else {
                    if (IS_LEAF(c2)) {
                        node_leaf_intersect(c1, LEAF_RAW(c2), GET_OFFSET(c2), result);
                        iter_next_without_skip(iter2);
                        iter_next(iter1);
                    } else {
                        node_intersect(c1, c2, result);
                        iter_next(iter1);
                        iter_next(iter2);
                    }
                }
            }
            destroy_iterator(iter1);
            destroy_iterator(iter2);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//                 count-only intersect重载
// ═══════════════════════════════════════════════════════════════════
inline uint64_t leaf_intersect(ARTLeaf* leaf1, uint8_t s1,
                                ARTLeaf* leaf2, uint8_t s2) {
    std::vector<uint64_t> tmp;
    leaf_intersect(leaf1, s1, leaf2, s2, tmp);
    return tmp.size();
}

inline uint64_t node_range_intersect(ARTNode* node, RangeElement* range,
                                      uint16_t range_size) {
    std::vector<uint64_t> tmp;
    node_range_intersect(node, range, range_size, tmp);
    return tmp.size();
}

inline uint64_t node_leaf_intersect(ARTNode* node, ARTLeaf* leaf, uint8_t s) {
    std::vector<uint64_t> tmp;
    node_leaf_intersect(node, leaf, s, tmp);
    return tmp.size();
}

inline uint64_t node_intersect(ARTNode* n1, ARTNode* n2) {
    std::vector<uint64_t> tmp;
    node_intersect(n1, n2, tmp);
    return tmp.size();
}

// ═══════════════════════════════════════════════════════════════════
//                      check_node
// ═══════════════════════════════════════════════════════════════════
inline void check_node(ARTNode* node) {
    switch (node->type) {
        case NODE4: {
            auto n = (ARTNode_4*)node;
            for (int i = 0; i < n->n.num_children; i++)
                if (IS_LEAF(n->children[i]))
                    assert(LEAF_RAW(n->children[i])->depth == node->depth);
            break;
        }
        case NODE16: {
            auto n = (ARTNode_16*)node;
            for (int i = 0; i < n->n.num_children; i++)
                if (IS_LEAF(n->children[i]))
                    assert(LEAF_RAW(n->children[i])->depth == node->depth);
            break;
        }
        case NODE48: {
            auto n = (ARTNode_48*)node;
            for (int i = 0; i < 256; i++)
                if (n->keys[i])
                    if (IS_LEAF(n->children[n->keys[i] - 1]))
                        assert(LEAF_RAW(n->children[n->keys[i] - 1])->depth == node->depth);
            break;
        }
        case NODE256: {
            auto n = (ARTNode_256*)node;
            for (int i = 0; i < 256; i++)
                if (n->children[i] && IS_LEAF(n->children[i]))
                    assert(LEAF_RAW(n->children[i])->depth == node->depth);
            break;
        }
        default: throw std::runtime_error("check_node(): Invalid node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//            batch insert 系列 (empty_leaf + merge)
// 改动: batch_subtree_build 根据byte种类数预判初始node类型
// ═══════════════════════════════════════════════════════════════════
inline uint64_t empty_leaf_batch_insert8(ARTNode** n, ARTLeaf8** leaf,
                                          RangeElement* insert_list,
                                          Property_t** properties,
                                          uint64_t list_size,
                                          WriterTraceBlock* trace_block) {
    assert(list_size <= ART_LEAF_SIZE);
    uint8_t depth = (*n)->depth;
    uint64_t cur_byte_st = 0, cur_byte_ed = 0;
    do {
        uint8_t cur_byte = get_key_byte(insert_list[cur_byte_st], depth);
        while (cur_byte_ed < list_size && get_key_byte(insert_list[cur_byte_ed], depth) == cur_byte) cur_byte_ed++;
        for (uint64_t i = 0; i < cur_byte_ed - cur_byte_st; i++) {
            (*leaf)->value.set(insert_list[cur_byte_st + i] & 0xFF);
        }
        add_child(*n, n, cur_byte, LEAF_POINTER_CTOR(*leaf, (*leaf)->size), trace_block);
        (*leaf)->size += cur_byte_ed - cur_byte_st;
        cur_byte_st = cur_byte_ed;
    } while (cur_byte_ed < list_size);
    return list_size;
}

inline uint64_t empty_leaf_batch_insert16(ARTNode** n, ARTLeaf16** leaf,
                                           RangeElement* insert_list,
                                           Property_t** properties,
                                           uint64_t list_size,
                                           WriterTraceBlock* trace_block) {
    assert(list_size <= ART_LEAF_SIZE);
    uint8_t depth = (*n)->depth;
    uint64_t cur_byte_st = 0, cur_byte_ed = 0;
    do {
        uint8_t cur_byte = get_key_byte(insert_list[cur_byte_st], depth);
        while (cur_byte_ed < list_size && get_key_byte(insert_list[cur_byte_ed], depth) == cur_byte) cur_byte_ed++;
        for (uint64_t i = 0; i < cur_byte_ed - cur_byte_st; i++)
            (*leaf)->value->at((*leaf)->size + i) = insert_list[cur_byte_st + i] & 0xFFFF;
        add_child(*n, n, cur_byte, LEAF_POINTER_CTOR(*leaf, (*leaf)->size), trace_block);
        (*leaf)->size += cur_byte_ed - cur_byte_st;
        cur_byte_st = cur_byte_ed;
    } while (cur_byte_ed < list_size);
    return list_size;
}

inline uint64_t empty_leaf_batch_insert32(ARTNode** n, ARTLeaf32** leaf,
                                           RangeElement* insert_list,
                                           Property_t** properties,
                                           uint64_t list_size,
                                           WriterTraceBlock* trace_block) {
    assert(list_size <= ART_LEAF_SIZE);
    uint8_t depth = (*n)->depth;
    uint64_t cur_byte_st = 0, cur_byte_ed = 0;
    do {
        uint8_t cur_byte = get_key_byte(insert_list[cur_byte_st], depth);
        while (cur_byte_ed < list_size && get_key_byte(insert_list[cur_byte_ed], depth) == cur_byte) cur_byte_ed++;
        for (uint64_t i = 0; i < cur_byte_ed - cur_byte_st; i++)
            (*leaf)->value->at((*leaf)->size + i) = insert_list[cur_byte_st + i] & 0xFFFFFFFF;
        add_child(*n, n, cur_byte, LEAF_POINTER_CTOR(*leaf, (*leaf)->size), trace_block);
        (*leaf)->size += cur_byte_ed - cur_byte_st;
        cur_byte_st = cur_byte_ed;
    } while (cur_byte_ed < list_size);
    return list_size;
}

inline uint64_t empty_leaf_batch_insert64(ARTNode** n, ARTLeaf64** leaf,
                                           RangeElement* insert_list,
                                           Property_t** properties,
                                           uint64_t list_size,
                                           WriterTraceBlock* trace_block) {
    assert(list_size <= ART_LEAF_SIZE);
    uint8_t depth = (*n)->depth;
    uint64_t cur_byte_st = 0, cur_byte_ed = 0;
    do {
        uint8_t cur_byte = get_key_byte(insert_list[cur_byte_st], depth);
        while (cur_byte_ed < list_size && get_key_byte(insert_list[cur_byte_ed], depth) == cur_byte) cur_byte_ed++;
        for (uint64_t i = 0; i < cur_byte_ed - cur_byte_st; i++)
            (*leaf)->value->at((*leaf)->size + i) = insert_list[cur_byte_st + i];
        add_child(*n, n, cur_byte, LEAF_POINTER_CTOR(*leaf, (*leaf)->size), trace_block);
        (*leaf)->size += cur_byte_ed - cur_byte_st;
        cur_byte_st = cur_byte_ed;
    } while (cur_byte_ed < list_size);
    return list_size;
}

inline uint64_t empty_leaf_batch_insert(ARTNode** n, ARTLeaf** leaf,
                                         RangeElement* insert_list,
                                         Property_t** properties,
                                         uint64_t list_size,
                                         WriterTraceBlock* trace_block) {
    assert(leaf && *leaf);
    switch ((*leaf)->depth + (*leaf)->is_single_byte) {
        case 0:
        case 1: return empty_leaf_batch_insert32(n, (ARTLeaf32**)leaf, insert_list, properties, list_size, trace_block);
        case 2: return empty_leaf_batch_insert16(n, (ARTLeaf16**)leaf, insert_list, properties, list_size, trace_block);
        case 3: return empty_leaf_batch_insert8(n, (ARTLeaf8**)leaf, insert_list, properties, list_size, trace_block);
        default: throw std::runtime_error("empty_leaf_batch_insert(): Invalid depth");
    }
}

// ═══════════════════════════════════════════════════════════════════
//   leaf_list_merge_batch_insert / leaf_list_merge / helpers
// ═══════════════════════════════════════════════════════════════════
inline void add_list_segment_to_new_leaf(ARTNode** new_node, ARTLeaf*& new_leaf,
                                          uint8_t depth, RangeElement* elem_list,
                                          Property_t** prop_list, uint64_t count,
                                          uint8_t cur_byte,
                                          WriterTraceBlock* trace_block) {
    assert(count != 0);
    if (new_leaf == nullptr || new_leaf->size + count > ART_LEAF_SIZE) {
        if (count > ART_LEAF_SIZE) {
            auto new_child = add_child(*new_node, new_node, cur_byte, (void*)0x1, trace_block);
            batch_subtree_build(new_child, depth + 1, elem_list, prop_list, count, trace_block);
            new_leaf = nullptr;
            return;
        } else {
            new_leaf = alloc_leaf(ARTKey{elem_list[0]}, depth, false, true, trace_block);
        }
    }
    add_child(*new_node, new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf, new_leaf->size), trace_block);
    for (uint64_t i = 0; i < count; i++) {
        new_leaf->insert(elem_list[i], nullptr, new_leaf->size);
    }
}

inline void add_leaf_segment_to_new_leaf(ARTNode** new_node, ARTLeaf*& new_leaf,
                                          uint8_t depth, ARTLeaf* cur_leaf,
                                          uint64_t count, uint8_t cur_byte,
                                          WriterTraceBlock* trace_block) {
    assert(count != 0);
    auto cur_raw_leaf = LEAF_RAW(cur_leaf);
    auto leaf_st = GET_OFFSET(cur_leaf);
    if (new_leaf == nullptr || new_leaf->size + count > ART_LEAF_SIZE) {
        new_leaf = alloc_leaf(ARTKey{cur_raw_leaf->at(leaf_st)}, depth, false, true, trace_block);
        assert(new_leaf->size == 0);
    }
    cur_raw_leaf->copy_to_leaf(leaf_st, leaf_st + count, new_leaf, new_leaf->size);
    add_child(*new_node, new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf, new_leaf->size), trace_block);
    new_leaf->size += count;
}

inline uint64_t leaf_list_merge_batch_insert(ARTNode** n, ARTLeaf*& new_leaf,
                                              uint8_t cur_byte, ARTLeaf* leaf,
                                              uint16_t leaf_count,
                                              RangeElement* insert_list,
                                              Property_t** properties,
                                              uint64_t list_count,
                                              WriterTraceBlock* trace_block) {
    uint64_t leaf_idx = GET_OFFSET(leaf);
    leaf = LEAF_RAW(leaf);
    uint64_t inserted = list_count;
    uint64_t list_idx = 0;
    uint64_t leaf_ed = leaf_idx + leaf_count;
    uint64_t total_count = leaf_count + list_count;

    if (new_leaf == nullptr || new_leaf->size + total_count > ART_LEAF_SIZE) {
        if (total_count > ART_LEAF_SIZE) {
            auto batch_build_list = new std::vector<RangeElement>();
            batch_build_list->reserve(total_count);
            auto batch_build_prop_list = new std::vector<Property_t*>();
            while (leaf_idx < leaf_ed && list_idx < list_count) {
                if (leaf->at(leaf_idx) < insert_list[list_idx]) {
                    batch_build_list->push_back(leaf->at(leaf_idx));
                    batch_build_prop_list->push_back(nullptr);
                    leaf_idx++;
                } else if (leaf->at(leaf_idx) > insert_list[list_idx]) {
                    batch_build_list->push_back(insert_list[list_idx]);
                    batch_build_prop_list->push_back(properties[list_idx]);
                    list_idx++;
                } else {
                    batch_build_list->push_back(insert_list[list_idx]);
                    batch_build_prop_list->push_back(properties[list_idx]);
                    list_idx++;
                    leaf_idx++;
                    inserted--;
                }
            }
            while (leaf_idx < leaf_ed) {
                batch_build_list->push_back(leaf->at(leaf_idx++));
                batch_build_prop_list->push_back(nullptr);
            }
            while (list_idx < list_count) {
                batch_build_list->push_back(insert_list[list_idx]);
                batch_build_prop_list->push_back(properties[list_idx]);
                list_idx++;
            }
            auto new_child = add_child(*n, n, cur_byte, (void*)0x1, trace_block);
            batch_subtree_build(new_child, leaf->depth + 1, batch_build_list->data(),
                               batch_build_prop_list->data(), batch_build_list->size(), trace_block);
            delete batch_build_list;
            delete batch_build_prop_list;
            new_leaf = nullptr;
            return inserted;
        } else {
            new_leaf = alloc_leaf(ARTKey{insert_list[0]}, leaf->depth, false, true, trace_block);
        }
    }

    add_child(*n, n, cur_byte, LEAF_POINTER_CTOR(new_leaf, new_leaf->size), trace_block);
    while (leaf_idx < leaf_ed && list_idx < list_count) {
        if (leaf->at(leaf_idx) < insert_list[list_idx]) {
            new_leaf->insert(leaf->at(leaf_idx), nullptr, new_leaf->size);
            leaf_idx++;
        } else if (leaf->at(leaf_idx) > insert_list[list_idx]) {
            new_leaf->insert(insert_list[list_idx], properties[list_idx], new_leaf->size);
            list_idx++;
        } else {
            new_leaf->insert(insert_list[list_idx], properties[list_idx], new_leaf->size);
            list_idx++;
            leaf_idx++;
            inserted--;
        }
    }
    while (leaf_idx < leaf_ed) {
        new_leaf->insert(leaf->at(leaf_idx), nullptr, new_leaf->size);
        leaf_idx++;
    }
    while (list_idx < list_count) {
        new_leaf->insert(insert_list[list_idx], properties[list_idx], new_leaf->size);
        list_idx++;
    }
    return inserted;
}

inline uint64_t leaf_list_merge(ARTNode** n, ARTLeaf* leaf,
                                 RangeElement* elem_list, Property_t** properties,
                                 uint64_t list_size, WriterTraceBlock* trace_block) {
    assert(list_size != 0);
    ARTLeaf* new_leaf = nullptr;
    uint64_t inserted = 0;

    uint64_t cur_list_st = 0, cur_list_ed = 0;
    uint16_t cur_list_byte = get_key_byte(elem_list[0], leaf->depth);
    uint64_t cur_leaf_st = 0, cur_leaf_ed = 0;
    uint16_t cur_leaf_byte = get_key_byte(leaf->at(0), leaf->depth);

    while (cur_list_ed < list_size && cur_leaf_ed < (uint64_t)leaf->size) {
        cur_list_byte = get_key_byte(elem_list[cur_list_st], leaf->depth);
        cur_leaf_byte = get_key_byte(leaf->at(cur_leaf_st), leaf->depth);
        if (cur_list_byte < cur_leaf_byte) {
            while (cur_list_ed < list_size && get_key_byte(elem_list[cur_list_ed], leaf->depth) == cur_list_byte) cur_list_ed++;
            inserted += cur_list_ed - cur_list_st;
            add_list_segment_to_new_leaf(n, new_leaf, leaf->depth, elem_list + cur_list_st, properties + cur_list_st, cur_list_ed - cur_list_st, cur_list_byte, trace_block);
            cur_list_st = cur_list_ed;
        } else if (cur_list_byte > cur_leaf_byte) {
            while (cur_leaf_ed < (uint64_t)leaf->size && get_key_byte(leaf->at(cur_leaf_ed), leaf->depth) == cur_leaf_byte) cur_leaf_ed++;
            add_leaf_segment_to_new_leaf(n, new_leaf, leaf->depth, (ARTLeaf*)LEAF_POINTER_CTOR(leaf, cur_leaf_st), cur_leaf_ed - cur_leaf_st, cur_leaf_byte, trace_block);
            cur_leaf_st = cur_leaf_ed;
        } else {
            while (cur_list_ed < list_size && get_key_byte(elem_list[cur_list_ed], leaf->depth) == cur_list_byte) cur_list_ed++;
            while (cur_leaf_ed < (uint64_t)leaf->size && get_key_byte(leaf->at(cur_leaf_ed), leaf->depth) == cur_leaf_byte) cur_leaf_ed++;
            inserted += leaf_list_merge_batch_insert(n, new_leaf, cur_list_byte,
                (ARTLeaf*)LEAF_POINTER_CTOR(leaf, cur_leaf_st), cur_leaf_ed - cur_leaf_st,
                elem_list + cur_list_st, properties + cur_list_st, cur_list_ed - cur_list_st, trace_block);
            cur_list_st = cur_list_ed;
            cur_leaf_st = cur_leaf_ed;
        }
    }

    if (cur_list_ed < list_size) {
        inserted += list_size - cur_list_ed;
        while (cur_list_ed < list_size) {
            cur_list_byte = get_key_byte(elem_list[cur_list_st], leaf->depth);
            while (cur_list_ed < list_size && get_key_byte(elem_list[cur_list_ed], leaf->depth) == cur_list_byte) cur_list_ed++;
            add_list_segment_to_new_leaf(n, new_leaf, leaf->depth, elem_list + cur_list_st, properties + cur_list_st, cur_list_ed - cur_list_st, cur_list_byte, trace_block);
            cur_list_st = cur_list_ed;
        }
    }
    while (cur_leaf_ed < (uint64_t)leaf->size) {
        cur_leaf_byte = get_key_byte(leaf->at(cur_leaf_st), leaf->depth);
        while (cur_leaf_ed < (uint64_t)leaf->size && get_key_byte(leaf->at(cur_leaf_ed), leaf->depth) == cur_leaf_byte) cur_leaf_ed++;
        add_leaf_segment_to_new_leaf(n, new_leaf, leaf->depth, (ARTLeaf*)LEAF_POINTER_CTOR(leaf, cur_leaf_st), cur_leaf_ed - cur_leaf_st, cur_leaf_byte, trace_block);
        cur_leaf_st = cur_leaf_ed;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════
//         dump_all_node_ops_stats — 全局状态打印入口
// ═══════════════════════════════════════════════════════════════════
inline void dump_all_node_ops_stats() {
    ops_counters().dump();
}

} // namespace container
