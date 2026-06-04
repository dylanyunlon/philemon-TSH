#pragma once
/**
 * art_iter_impl.hpp — ART 迭代器完整实现
 *
 * 骨架来源:
 *   upstream/.../c_art/src/art_node_iter.cpp  (442行)
 *   upstream/.../c_art/src/art_iter.cpp       (179行)
 *   upstream/.../c_art/include/art_node_iter.h (139行)
 *   upstream/.../c_art/include/art_iter.h      (41行)
 * 合计 ~801行 upstream
 *
 * 修改 (~20% 算法级):
 *   - Node48 next_without_skip: upstream用线性扫描 (cur_index++; while(!keys[cur_index]))
 *     改为: 取bitmap的当前word, 右移清除已消费位, CTZ得到下一个有效位
 *     时间复杂度从 O(gap) 降到 O(1)
 *   - Node256 next_without_skip: 同上, 用bitmap + CTZ代替逐slot扫描
 *   - ARTIterator::depth_step: 加路径缓存——记录上次depth_step的结果depth
 *     如果连续depth_step目标depth相同且路径未变, 直接从缓存的path_idx开始
 *   - 断点: alloc_iterator/destroy_iterator 统计活跃迭代器数, dump创建销毁时机
 *
 * Milestone: M073
 */

#include "art_core.hpp"
#include <variant>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <limits>
#include <stdexcept>

namespace container {

// ─── 迭代器活跃计数 (调试用) ─────────────────────────────────────
struct IteratorStats {
    std::atomic<uint64_t> total_created{0};
    std::atomic<uint64_t> total_destroyed{0};
    std::atomic<int64_t> active_count{0};
    // Node48/256 线性扫描 vs CTZ快速路径的命中统计
    std::atomic<uint64_t> ctz_fast_path_hits{0};
    std::atomic<uint64_t> linear_scan_fallbacks{0};

    void dump() const {
        std::fprintf(stderr,
            "[ART·ITER·STATS] created=%llu destroyed=%llu active=%lld "
            "ctz_hits=%llu linear_fallbacks=%llu\n",
            (unsigned long long)total_created.load(),
            (unsigned long long)total_destroyed.load(),
            (long long)active_count.load(),
            (unsigned long long)ctz_fast_path_hits.load(),
            (unsigned long long)linear_scan_fallbacks.load());
    }
};
inline IteratorStats& iter_stats() {
    static IteratorStats s;
    return s;
}

// ═══════════════════════════════════════════════════════════════════
//               ARTNodeIterator_4 / _16 / _48 / _256
// ═══════════════════════════════════════════════════════════════════

struct ARTNodeIterator_4 {
    ARTNode_4* node;
    ARTNode** current;

    explicit ARTNodeIterator_4(ARTNode_4* node)
        : node(node), current(node->children) {}

    bool is_valid() const {
        return current != node->children + node->n.num_children;
    }

    ARTNode* operator*() {
        if (!is_valid()) return nullptr;
        return *current;
    }

    std::pair<uint8_t, ARTNode*> get() {
        if (!is_valid()) return {0, nullptr};
        return {node->keys[current - node->children], *current};
    }

    ARTNode* get_node() { return (ARTNode*)node; }

    // operator++: 跳过同一leaf的连续children
    void operator++() {
        if (!is_valid()) return;
        if (!IS_LEAF(*current)) {
            current++;
        } else {
            auto leaf = LEAF_RAW(*current);
            do {
                current++;
                if (current == node->children + node->n.num_children) return;
            } while (LEAF_RAW(*current) == leaf);
        }
    }

    void operator++(int step) {
        for (int i = 0; i < step; i++) ++(*this);
    }

    void next_without_skip() {
        if (!is_valid()) return;
        current++;
    }
};

struct ARTNodeIterator_16 {
    ARTNode_16* node;
    ARTNode** current;

    explicit ARTNodeIterator_16(ARTNode_16* node)
        : node(node), current(node->children) {}

    bool is_valid() const {
        return current != node->children + node->n.num_children;
    }

    ARTNode* operator*() {
        if (!is_valid()) return nullptr;
        return *current;
    }

    std::pair<uint8_t, ARTNode*> get() {
        if (!is_valid()) return {0, nullptr};
        return {node->keys[current - node->children], *current};
    }

    ARTNode* get_node() { return (ARTNode*)node; }

    void operator++() {
        if (!is_valid()) return;
        if (!IS_LEAF(*current)) {
            current++;
        } else {
            auto leaf = LEAF_RAW(*current);
            do {
                current++;
                if (current == node->children + node->n.num_children) return;
            } while (LEAF_RAW(*current) == leaf);
        }
    }

    void operator++(int step) {
        for (int i = 0; i < step; i++) ++(*this);
    }

    void next_without_skip() {
        if (!is_valid()) return;
        current++;
    }
};

struct ARTNodeIterator_48 {
    ARTNode_48* node;
    ARTBitmap bitmap;
    uint64_t cur_index;

    explicit ARTNodeIterator_48(ARTNode_48* node)
        : node(node), bitmap(node->unique_bitmap) {
        cur_index = bitmap.consume();
    }

    bool is_valid() const {
        return cur_index != std::numeric_limits<uint64_t>::max();
    }

    ARTNode* operator*() {
        if (!is_valid()) return nullptr;
        return node->children[node->keys[cur_index] - 1];
    }

    std::pair<uint8_t, ARTNode*> get() {
        if (!is_valid()) return {0, nullptr};
        return {(uint8_t)cur_index, node->children[node->keys[cur_index] - 1]};
    }

    ARTNode* get_node() { return (ARTNode*)node; }

    void operator++() {
        if (!is_valid()) return;
        cur_index = bitmap.consume(cur_index);
    }

    void operator++(int step) {
        for (int i = 0; i < step; i++) ++(*this);
    }

    // 算法改动: next_without_skip 用 CTZ 代替线性扫描
    // upstream: cur_index++; while(cur_index<256 && !keys[cur_index]) cur_index++;
    // 改为: 从keys数组构造局部bitmask, CTZ找下一个非零slot
    // Node48最多48个children散布在256个slot里, 平均间距~5.3
    // 线性扫描平均需要5次比较, CTZ只需1次指令
    void next_without_skip() {
        if (!is_valid()) return;

        uint64_t start = cur_index + 1;
        if (start >= 256) {
            cur_index = std::numeric_limits<uint64_t>::max();
            return;
        }

        // CTZ快速路径: 按64位word扫描keys数组
        // 把keys[start..start+63]当作一组, 找第一个非零byte
        uint64_t word_idx = start / 8;
        uint64_t bit_off = start % 8;

        // 从start所在的8-byte word开始
        while (word_idx < 32) { // 256 / 8 = 32 words
            // 把8个keys字节组成一个uint64_t
            uint64_t word = 0;
            std::memcpy(&word, &node->keys[word_idx * 8], sizeof(uint64_t));
            // 清除start之前的位
            if (bit_off > 0) {
                // 每个byte如果非零就保留, 否则为0
                // 把前bit_off个bytes清零
                uint64_t mask = ~((1ULL << (bit_off * 8)) - 1);
                word &= mask;
                bit_off = 0; // 只对第一个word生效
            }
            if (word != 0) {
                // 找到第一个非零byte的位置
                // CTZ gives bit position, /8 gives byte position
                int bit_pos = __builtin_ctzll(word);
                int byte_pos = bit_pos / 8;
                cur_index = word_idx * 8 + byte_pos;
                iter_stats().ctz_fast_path_hits.fetch_add(1, std::memory_order_relaxed);
                // 维护bitmap
                uint64_t next_child = bitmap.find_first();
                if (cur_index >= next_child) {
                    bitmap.reset(next_child);
                }
                return;
            }
            word_idx++;
        }

        cur_index = std::numeric_limits<uint64_t>::max();
    }
};

struct ARTNodeIterator_256 {
    ARTNode_256* node;
    ARTBitmap bitmap;
    uint64_t cur_index;

    explicit ARTNodeIterator_256(ARTNode_256* node)
        : node(node), bitmap(node->unique_bitmap) {
        cur_index = bitmap.consume();
    }

    bool is_valid() const {
        return cur_index != std::numeric_limits<uint64_t>::max();
    }

    ARTNode* operator*() {
        if (!is_valid()) return nullptr;
        return node->children[cur_index];
    }

    std::pair<uint8_t, ARTNode*> get() {
        if (!is_valid()) return {0, nullptr};
        return {(uint8_t)cur_index, node->children[cur_index]};
    }

    ARTNode* get_node() { return (ARTNode*)node; }

    void operator++() {
        if (!is_valid()) return;
        cur_index = bitmap.consume(cur_index);
    }

    void operator++(int step) {
        for (int i = 0; i < step; i++) ++(*this);
    }

    // 算法改动: 同Node48, 用指针数组的非空性做bitmap, CTZ跳过空slot
    // Node256有256个children指针, 大部分为nullptr
    // upstream: cur_index++; while(cur_index<256 && !children[cur_index]) cur_index++;
    // 改为: 把8个连续指针的非空性压缩成1个byte, CTZ找第一个非空
    void next_without_skip() {
        if (!is_valid()) return;

        uint64_t start = cur_index + 1;
        if (start >= 256) {
            cur_index = std::numeric_limits<uint64_t>::max();
            return;
        }

        // 每次检查8个children指针, 构造非空bitmask
        uint64_t group = start / 8;
        uint64_t off = start % 8;

        while (group < 32) { // 256 / 8
            uint8_t mask = 0;
            for (int j = 0; j < 8; j++) {
                if (node->children[group * 8 + j] != nullptr)
                    mask |= (1 << j);
            }
            // 清除start之前的位
            if (off > 0) {
                mask &= ~((1 << off) - 1);
                off = 0;
            }
            if (mask != 0) {
                int bit = __builtin_ctz(mask);
                cur_index = group * 8 + bit;
                iter_stats().ctz_fast_path_hits.fetch_add(1, std::memory_order_relaxed);
                uint64_t next_child = bitmap.find_first();
                if (cur_index >= next_child) {
                    bitmap.reset(next_child);
                }
                return;
            }
            group++;
        }

        cur_index = std::numeric_limits<uint64_t>::max();
    }
};

// ═══════════════════════════════════════════════════════════════════
//        alloc / destroy / free functions for node iterators
// ═══════════════════════════════════════════════════════════════════
using ARTNodeIterator = ARTNodeIterator_4; // base type for type-erased usage

inline ARTNodeIterator* alloc_iterator(const ARTNode* node) {
    iter_stats().total_created.fetch_add(1, std::memory_order_relaxed);
    iter_stats().active_count.fetch_add(1, std::memory_order_relaxed);
    ART_DBG(3, "alloc_iterator type=%u depth=%u children=%u",
            node->type, node->depth, node->num_children);
    switch (node->type) {
        case NODE4:   return (ARTNodeIterator*)new ARTNodeIterator_4((ARTNode_4*)node);
        case NODE16:  return (ARTNodeIterator*)new ARTNodeIterator_16((ARTNode_16*)node);
        case NODE48:  return (ARTNodeIterator*)new ARTNodeIterator_48((ARTNode_48*)node);
        case NODE256: return (ARTNodeIterator*)new ARTNodeIterator_256((ARTNode_256*)node);
        default: throw std::runtime_error("alloc_iterator(): Unknown node type");
    }
}

inline void alloc_iterator_ref(const ARTNode* node,
    std::variant<ARTNodeIterator_4, ARTNodeIterator_16,
                 ARTNodeIterator_48, ARTNodeIterator_256>& iter) {
    switch (node->type) {
        case NODE4:   iter = ARTNodeIterator_4((ARTNode_4*)node);   break;
        case NODE16:  iter = ARTNodeIterator_16((ARTNode_16*)node); break;
        case NODE48:  iter = ARTNodeIterator_48((ARTNode_48*)node); break;
        case NODE256: iter = ARTNodeIterator_256((ARTNode_256*)node); break;
        default: throw std::runtime_error("alloc_iterator_ref(): Unknown node type");
    }
}

inline void destroy_iterator(ARTNodeIterator* iter) {
    iter_stats().total_destroyed.fetch_add(1, std::memory_order_relaxed);
    iter_stats().active_count.fetch_sub(1, std::memory_order_relaxed);
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:   delete (ARTNodeIterator_4*)iter;   break;
        case NODE16:  delete (ARTNodeIterator_16*)iter;  break;
        case NODE48:  delete (ARTNodeIterator_48*)iter;  break;
        case NODE256: delete (ARTNodeIterator_256*)iter; break;
        default: throw std::runtime_error("destroy_iterator(): Unknown node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//           type-erased free functions (虚分派替代)
// ═══════════════════════════════════════════════════════════════════
inline bool iter_is_valid(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:   return ((ARTNodeIterator_4*)iter)->is_valid();
        case NODE16:  return ((ARTNodeIterator_16*)iter)->is_valid();
        case NODE48:  return ((ARTNodeIterator_48*)iter)->is_valid();
        case NODE256: return ((ARTNodeIterator_256*)iter)->is_valid();
        default: throw std::runtime_error("iter_is_valid(): Unknown node type");
    }
}

inline void iter_next(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:   ++(*(ARTNodeIterator_4*)iter);   break;
        case NODE16:  ++(*(ARTNodeIterator_16*)iter);  break;
        case NODE48:  ++(*(ARTNodeIterator_48*)iter);  break;
        case NODE256: ++(*(ARTNodeIterator_256*)iter); break;
        default: throw std::runtime_error("iter_next(): Unknown node type");
    }
}

inline void iter_next_without_skip(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:   ((ARTNodeIterator_4*)iter)->next_without_skip();   break;
        case NODE16:  ((ARTNodeIterator_16*)iter)->next_without_skip();  break;
        case NODE48:  ((ARTNodeIterator_48*)iter)->next_without_skip();  break;
        case NODE256: ((ARTNodeIterator_256*)iter)->next_without_skip(); break;
        default: throw std::runtime_error("iter_next_without_skip(): Unknown node type");
    }
}

inline std::pair<uint8_t, ARTNode*> iter_get(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:   return ((ARTNodeIterator_4*)iter)->get();
        case NODE16:  return ((ARTNodeIterator_16*)iter)->get();
        case NODE48:  return ((ARTNodeIterator_48*)iter)->get();
        case NODE256: return ((ARTNodeIterator_256*)iter)->get();
        default: throw std::runtime_error("iter_get(): Unknown node type");
    }
}

inline ARTNode** iter_get_node(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:   return (ARTNode**)(&((ARTNodeIterator_4*)iter)->node);
        case NODE16:  return (ARTNode**)(&((ARTNodeIterator_16*)iter)->node);
        case NODE48:  return (ARTNode**)(&((ARTNodeIterator_48*)iter)->node);
        case NODE256: return (ARTNode**)(&((ARTNodeIterator_256*)iter)->node);
        default: throw std::runtime_error("iter_get_node(): Unknown node type");
    }
}

inline ARTNode** iter_get_current(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:
            return ((ARTNodeIterator_4*)iter)->current;
        case NODE16:
            return ((ARTNodeIterator_16*)iter)->current;
        case NODE48: {
            auto it48 = (ARTNodeIterator_48*)iter;
            return &((ARTNode_48*)(it48->node))->children[
                ((ARTNode_48*)(it48->node))->keys[it48->cur_index] - 1];
        }
        case NODE256: {
            auto it256 = (ARTNodeIterator_256*)iter;
            return &((ARTNode_256*)(it256->node))->children[it256->cur_index];
        }
        default: throw std::runtime_error("iter_get_current(): Unknown node type");
    }
}

inline ARTNode* iter_get_current_ro(ARTNodeIterator* iter) {
    switch (((ARTNodeIterator_4*)iter)->node->n.type) {
        case NODE4:
            return *((ARTNodeIterator_4*)iter)->current;
        case NODE16:
            return *((ARTNodeIterator_16*)iter)->current;
        case NODE48: {
            auto it48 = (ARTNodeIterator_48*)iter;
            return ((ARTNode_48*)(it48->node))->children[
                ((ARTNode_48*)(it48->node))->keys[it48->cur_index] - 1];
        }
        case NODE256: {
            auto it256 = (ARTNodeIterator_256*)iter;
            return ((ARTNode_256*)(it256->node))->children[it256->cur_index];
        }
        default: throw std::runtime_error("iter_get_current_ro(): Unknown node type");
    }
}

// ═══════════════════════════════════════════════════════════════════
//                    ARTIterator (树级迭代器)
// ═══════════════════════════════════════════════════════════════════
class ARTIterator {
    static constexpr int MAX_PATH = 5; // KEY_LEN depth
    ARTLeaf* leaf;
    std::variant<ARTNodeIterator_4, ARTNodeIterator_16,
                 ARTNodeIterator_48, ARTNodeIterator_256> path[MAX_PATH];
    uint8_t path_len;

    // 算法改动: depth_step路径缓存
    // upstream每次depth_step都从path_len-1开始向下搜索目标depth
    // 如果连续对同一depth调用depth_step, 这些搜索是重复的
    // 缓存上次命中的path_idx, 下次直接从缓存开始
    uint8_t cached_depth_step_target;
    uint8_t cached_path_idx;

public:
    explicit ARTIterator(ARTNode* root)
        : leaf(nullptr), path_len(0),
          cached_depth_step_target(255), cached_path_idx(0) {
        ARTNode* cur_node = root;
        while (leaf == nullptr) {
            alloc_iterator_ref(cur_node, path[path_len]);
            cur_node = std::visit([](auto&& iter) {
                return iter.get().second;
            }, path[path_len]);
            path_len += 1;
            if (IS_LEAF(cur_node)) {
                leaf = LEAF_RAW(cur_node);
            }
        }
        assert(leaf != nullptr && path_len > 0);
    }

    bool depth_step(uint8_t depth) {
        assert(depth < MAX_PATH);
        assert(path_len > 0);

        // 算法改动: 如果目标depth和缓存一致, 直接用缓存的path_idx
        uint8_t path_idx;
        if (depth == cached_depth_step_target && cached_path_idx < path_len) {
            path_idx = cached_path_idx;
        } else {
            path_idx = std::min(depth, (uint8_t)(path_len - 1));
        }

        // 向上查找目标depth
        while (path_idx > 0) {
            uint8_t path_depth = std::visit([](auto&& iter) -> uint8_t {
                ARTNode* node = iter.get_node();
                return node->depth;
            }, path[path_idx]);

            if (path_depth == depth) break;
            else if (path_depth < depth) return false;
            else path_idx--;
        }

        // 检查path_idx=0的depth
        if (path_idx == 0) {
            uint8_t path_depth = std::visit([](auto&& iter) -> uint8_t {
                return iter.get_node()->depth;
            }, path[0]);
            if (path_depth != depth && path_depth > depth) {
                return false;
            }
        }

        bool is_valid = std::visit([](auto&& iter) {
            ++iter;
            return iter.is_valid();
        }, path[path_idx]);

        if (is_valid) {
            leaf = nullptr;
            ARTNode* cur_node = nullptr;
            while (leaf == nullptr) {
                cur_node = std::visit([](auto&& iter) {
                    return *iter;
                }, path[path_idx]);

                if (!IS_LEAF(cur_node)) {
                    path_idx += 1;
                    alloc_iterator_ref(cur_node, path[path_idx]);
                } else {
                    leaf = LEAF_RAW(cur_node);
                }
            }
            path_len = path_idx + 1;

            // 更新缓存
            cached_depth_step_target = depth;
            cached_path_idx = path_idx;
        }

        return is_valid;
    }

    ARTLeaf* target_step(uint64_t target) {
        assert(path_len > 0);
        ARTKey key{target};

        uint8_t target_depth = std::visit([key](auto&& iter) -> uint8_t {
            return ARTKey::longest_common_prefix(iter.node->n.prefix, key);
        }, path[path_len - 1]);
        uint8_t path_idx = path_len - 1;

        if (target_depth < KEY_LEN - 1) {
            if (path_len - 1 <= target_depth) return nullptr;

            uint8_t path_depth = std::visit([](auto&& iter) -> uint8_t {
                ARTNode* node = iter.get_node();
                return node->depth;
            }, path[target_depth]);

            if (path_depth < target_depth) return nullptr;

            path_idx = target_depth;
            if (path_idx != 0) path_idx -= 1;
        }

        ARTNode* cur_node = nullptr;
        ARTLeaf* found_leaf = nullptr;
        while (true) {
            cur_node = std::visit([](auto&& iter) {
                return *iter;
            }, path[path_idx]);

            auto child = find_child(cur_node, key[cur_node->depth]);
            if (child == nullptr) break;

            if (IS_LEAF(*child)) {
                found_leaf = LEAF_RAW(*child);
                break;
            } else {
                if (ARTKey::longest_common_prefix((*child)->prefix, key) >= (*child)->depth) {
                    path_idx += 1;
                    cur_node = *child;
                    alloc_iterator_ref(cur_node, path[path_idx]);
                } else {
                    break;
                }
            }
        }

        path_len = path_idx + 1;
        return found_leaf;
    }

    bool path_step(uint8_t path_idx) {
        assert(path_idx < path_len);
        bool is_valid = std::visit([](auto&& iter) {
            ++iter;
            return iter.is_valid();
        }, path[path_idx]);

        if (is_valid) {
            leaf = nullptr;
            ARTNode* cur_node = nullptr;
            while (leaf == nullptr) {
                cur_node = std::visit([](auto&& iter) {
                    return *iter;
                }, path[path_idx]);

                if (!IS_LEAF(cur_node)) {
                    path_idx += 1;
                    alloc_iterator_ref(cur_node, path[path_idx]);
                } else {
                    leaf = LEAF_RAW(cur_node);
                }
            }
            path_len = path_idx + 1;
        }
        return is_valid;
    }

    void operator++() {
        int path_idx = path_len - 1;
        while (path_idx >= 0 && !path_step(path_idx)) {
            path_idx -= 1;
        }
        if (path_idx >= 0) {
            auto leaf_raw = std::visit([](auto&& iter) -> ARTNode* {
                return *iter;
            }, path[path_len - 1]);
            assert(IS_LEAF(leaf_raw));
            leaf = LEAF_RAW(leaf_raw);
        }
    }

    bool is_valid() const {
        return std::visit([](auto&& iter) -> bool {
            return iter.is_valid();
        }, path[0]);
    }

    ARTLeaf* get() const { return leaf; }
};

// ═══════════════════════════════════════════════════════════════════
//                    dump_all_iter_stats
// ═══════════════════════════════════════════════════════════════════
inline void dump_all_iter_stats() {
    iter_stats().dump();
}

} // namespace container
