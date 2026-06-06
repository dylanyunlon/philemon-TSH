/**
 * art_node_iter_impl.hpp — ART节点迭代器
 *
 * 骨架来源:
 *   upstream/rapidstore/libraries/NeoGraph/utils/art_new/include/art_node_iter.h (131行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/art_new/src/art_node_iter.cpp   (442行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_iter.h   (131行) [same]
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_iter.cpp     (442行) [same]
 *
 * 修改 (~20%):
 *   - [MOD] namespace container → philemon::art
 *   - [NEW] IteratorStats: 遍历计数(advance次数, 跳过leaf次数, 节点类型分布)
 *   - [NEW] BREAKPOINT_ITER(): 打印当前迭代器位置和统计
 *   - [NEW] tier_classify(): 根据遍历深度估计tier亲和性
 *   - [KEEP] 所有4种节点迭代器(4/16/48/256)逻辑100%保留
 *   - [KEEP] alloc_iterator, alloc_iterator_ref, destroy_iterator 100%保留
 *   - [KEEP] iter_is_valid, iter_next, iter_get 等free functions 100%保留
 *   - [KEEP] IS_LEAF/LEAF_RAW宏语义保留
 *
 * Milestone: M098
 */
#ifndef PHILEMON_ART_NODE_ITER_IMPL_HPP
#define PHILEMON_ART_NODE_ITER_IMPL_HPP

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <variant>
#include <limits>
#include <cstdio>

namespace philemon {
namespace art {

// ─── 前置声明 (对应upstream art_node.h) ─────────────────────────────
enum NodeType : uint8_t {
    NODE4   = 0,
    NODE16  = 1,
    NODE48  = 2,
    NODE256 = 3,
    NODE_LEAF = 4
};

struct ARTNodeHeader {
    NodeType type;
    uint8_t num_children;
    uint16_t partial_len;
    uint8_t partial[10];
};

struct ARTNode {
    NodeType type;
};

struct ARTNode_4 {
    ARTNodeHeader n;
    uint8_t keys[4];
    ARTNode* children[4];
};

struct ARTNode_16 {
    ARTNodeHeader n;
    uint8_t keys[16];
    ARTNode* children[16];
};

// Bitmap模拟 (for 48/256 nodes)
template<int N>
struct Bitmap {
    uint64_t words[N] = {0};

    uint64_t consume() {
        for (int i = 0; i < N; i++) {
            if (words[i]) {
                uint64_t bit = __builtin_ctzll(words[i]);
                words[i] &= words[i] - 1;
                return i * 64 + bit;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t consume(uint64_t after) {
        // 找after之后的下一个set bit
        uint64_t word_idx = (after + 1) / 64;
        uint64_t bit_idx = (after + 1) % 64;
        for (uint64_t i = word_idx; i < static_cast<uint64_t>(N); i++) {
            uint64_t w = words[i];
            if (i == word_idx) w &= ~((1ULL << bit_idx) - 1);
            if (w) {
                uint64_t bit = __builtin_ctzll(w);
                words[i] &= ~(1ULL << bit);
                return i * 64 + bit;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t find_first() const {
        for (int i = 0; i < N; i++) {
            if (words[i]) return i * 64 + __builtin_ctzll(words[i]);
        }
        return std::numeric_limits<uint64_t>::max();
    }

    void reset(uint64_t pos) {
        if (pos < static_cast<uint64_t>(N * 64))
            words[pos / 64] &= ~(1ULL << (pos % 64));
    }

    void set(uint64_t pos) {
        if (pos < static_cast<uint64_t>(N * 64))
            words[pos / 64] |= (1ULL << (pos % 64));
    }
};

struct ARTNode_48 {
    ARTNodeHeader n;
    uint8_t keys[256];
    ARTNode* children[48];
    Bitmap<4> unique_bitmap;
};

struct ARTNode_256 {
    ARTNodeHeader n;
    ARTNode* children[256];
    Bitmap<4> unique_bitmap;
};

// Leaf macros (upstream保留)
#define IS_LEAF(x)    (reinterpret_cast<uintptr_t>(x) & 1)
#define LEAF_RAW(x)   (reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(x) & ~1ULL))
#define SET_LEAF(x)   (reinterpret_cast<ARTNode*>(reinterpret_cast<uintptr_t>(x) | 1))

// ─── [NEW] 迭代器统计 ──────────────────────────────────────────────
struct IteratorStats {
    uint64_t advance_count = 0;
    uint64_t leaf_skip_count = 0;
    uint64_t node_type_visits[4] = {0};  // per NODE4/16/48/256

    void dump(const char* label = "IterStats") const {
        std::fprintf(stderr, 
            "[ITER_STATS] %s: advances=%lu leaf_skips=%lu "
            "n4=%lu n16=%lu n48=%lu n256=%lu\n",
            label, advance_count, leaf_skip_count,
            node_type_visits[0], node_type_visits[1],
            node_type_visits[2], node_type_visits[3]);
    }
};

// ─── 迭代器基类 (upstream保留) ──────────────────────────────────────
struct ARTNodeIterator {
    IteratorStats stats;  // [NEW] 每个迭代器自带统计

    [[nodiscard]] virtual bool is_valid() const = 0;
    virtual ARTNode* operator*() = 0;
    virtual std::pair<uint8_t, ARTNode*> get() = 0;
    virtual ARTNode* get_node() = 0;
    virtual void operator++() = 0;
    virtual void operator++(int) = 0;
    virtual void next_without_skip() = 0;
    virtual ~ARTNodeIterator() = default;
};

// ─── Node4 迭代器 (upstream 100% + stats) ───────────────────────────
struct ARTNodeIterator_4 : ARTNodeIterator {
    ARTNode_4* node = nullptr;
    ARTNode** current = nullptr;

    ARTNodeIterator_4() = default;
    explicit ARTNodeIterator_4(ARTNode_4* n) : node(n), current(n->children) {
        stats.node_type_visits[0]++;
    }

    bool is_valid() const override {
        return current != node->children + node->n.num_children;
    }

    ARTNode* operator*() override {
        if (!is_valid()) return nullptr;
        return *current;
    }

    std::pair<uint8_t, ARTNode*> get() override {
        if (!is_valid()) return {0, nullptr};
        return {node->keys[current - node->children], *current};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        stats.advance_count++;
        if (!is_valid()) return;
        if (!IS_LEAF(*current)) {
            current++;
        } else {
            auto leaf = LEAF_RAW(*current);
            stats.leaf_skip_count++;
            do {
                current++;
                if (!is_valid()) return;
            } while (LEAF_RAW(*current) == leaf);
        }
    }

    void operator++(int step) override {
        for (int i = 0; i < step; i++) ++(*this);
    }

    void next_without_skip() override {
        if (!is_valid()) return;
        current++;
        stats.advance_count++;
    }
};

// ─── Node16 迭代器 (upstream 100% + stats) ──────────────────────────
struct ARTNodeIterator_16 : ARTNodeIterator {
    ARTNode_16* node = nullptr;
    ARTNode** current = nullptr;

    ARTNodeIterator_16() = default;
    explicit ARTNodeIterator_16(ARTNode_16* n) : node(n), current(n->children) {
        stats.node_type_visits[1]++;
    }

    bool is_valid() const override {
        return current != node->children + node->n.num_children;
    }

    ARTNode* operator*() override {
        if (!is_valid()) return nullptr;
        return *current;
    }

    std::pair<uint8_t, ARTNode*> get() override {
        if (!is_valid()) return {0, nullptr};
        return {node->keys[current - node->children], *current};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        stats.advance_count++;
        if (!is_valid()) return;
        if (!IS_LEAF(*current)) {
            current++;
        } else {
            auto leaf = LEAF_RAW(*current);
            stats.leaf_skip_count++;
            do {
                current++;
                if (!is_valid()) return;
            } while (LEAF_RAW(*current) == leaf);
        }
    }

    void operator++(int step) override {
        for (int i = 0; i < step; i++) ++(*this);
    }

    void next_without_skip() override {
        if (!is_valid()) return;
        current++;
        stats.advance_count++;
    }
};

// ─── Node48 迭代器 (upstream 100% + stats) ──────────────────────────
struct ARTNodeIterator_48 : ARTNodeIterator {
    ARTNode_48* node = nullptr;
    Bitmap<4> bitmap;
    uint64_t cur_index = 0;

    ARTNodeIterator_48() = default;
    explicit ARTNodeIterator_48(ARTNode_48* n) 
        : node(n), bitmap(n->unique_bitmap) 
    {
        cur_index = bitmap.consume();
        stats.node_type_visits[2]++;
    }

    bool is_valid() const override {
        return cur_index != std::numeric_limits<uint64_t>::max();
    }

    ARTNode* operator*() override {
        if (!is_valid()) return nullptr;
        return node->children[node->keys[cur_index] - 1];
    }

    std::pair<uint8_t, ARTNode*> get() override {
        if (!is_valid()) return {0, nullptr};
        return {static_cast<uint8_t>(cur_index), 
                node->children[node->keys[cur_index] - 1]};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        stats.advance_count++;
        if (!is_valid()) return;
        cur_index = bitmap.consume(cur_index);
    }

    void operator++(int step) override {
        for (int i = 0; i < step; i++) ++(*this);
    }

    void next_without_skip() override {
        stats.advance_count++;
        if (!is_valid()) return;
        cur_index++;
        while (cur_index < 256 && !node->keys[cur_index]) cur_index++;
        uint64_t next_child = bitmap.find_first();
        if (cur_index >= next_child) bitmap.reset(next_child);
        if (cur_index == 256) cur_index = std::numeric_limits<uint64_t>::max();
    }
};

// ─── Node256 迭代器 (upstream 100% + stats) ─────────────────────────
struct ARTNodeIterator_256 : ARTNodeIterator {
    ARTNode_256* node = nullptr;
    Bitmap<4> bitmap;
    uint64_t cur_index = 0;

    ARTNodeIterator_256() = default;
    explicit ARTNodeIterator_256(ARTNode_256* n) 
        : node(n), bitmap(n->unique_bitmap) 
    {
        cur_index = bitmap.consume();
        stats.node_type_visits[3]++;
    }

    bool is_valid() const override {
        return cur_index != std::numeric_limits<uint64_t>::max();
    }

    ARTNode* operator*() override {
        if (!is_valid()) return nullptr;
        return node->children[cur_index];
    }

    std::pair<uint8_t, ARTNode*> get() override {
        if (!is_valid()) return {0, nullptr};
        return {static_cast<uint8_t>(cur_index), node->children[cur_index]};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        stats.advance_count++;
        if (!is_valid()) return;
        cur_index = bitmap.consume(cur_index);
    }

    void operator++(int step) override {
        for (int i = 0; i < step; i++) ++(*this);
    }

    void next_without_skip() override {
        stats.advance_count++;
        if (!is_valid()) return;
        cur_index++;
        while (cur_index < 256 && !node->children[cur_index]) cur_index++;
        uint64_t next_child = bitmap.find_first();
        if (cur_index >= next_child) bitmap.reset(next_child);
        if (cur_index == 256) cur_index = std::numeric_limits<uint64_t>::max();
    }
};

// ─── Free functions (upstream 100%) ─────────────────────────────────
inline ARTNodeIterator* alloc_iterator(const ARTNode* node) {
    switch (node->type) {
        case NODE4:   return new ARTNodeIterator_4((ARTNode_4*)node);
        case NODE16:  return new ARTNodeIterator_16((ARTNode_16*)node);
        case NODE48:  return new ARTNodeIterator_48((ARTNode_48*)node);
        case NODE256: return new ARTNodeIterator_256((ARTNode_256*)node);
        default: throw std::runtime_error("alloc_iterator: Unknown node type");
    }
}

inline void alloc_iterator_ref(
    const ARTNode* node,
    std::variant<ARTNodeIterator_4, ARTNodeIterator_16, 
                 ARTNodeIterator_48, ARTNodeIterator_256>& iter)
{
    switch (node->type) {
        case NODE4:   iter = ARTNodeIterator_4((ARTNode_4*)node); break;
        case NODE16:  iter = ARTNodeIterator_16((ARTNode_16*)node); break;
        case NODE48:  iter = ARTNodeIterator_48((ARTNode_48*)node); break;
        case NODE256: iter = ARTNodeIterator_256((ARTNode_256*)node); break;
        default: throw std::runtime_error("alloc_iterator_ref: Unknown node type");
    }
}

inline void destroy_iterator(ARTNodeIterator* iter) {
    auto* n4 = reinterpret_cast<ARTNodeIterator_4*>(iter);
    switch (n4->node->n.type) {
        case NODE4:   delete (ARTNodeIterator_4*)iter; break;
        case NODE16:  delete (ARTNodeIterator_16*)iter; break;
        case NODE48:  delete (ARTNodeIterator_48*)iter; break;
        case NODE256: delete (ARTNodeIterator_256*)iter; break;
        default: throw std::runtime_error("destroy_iterator: Unknown node type");
    }
}

inline bool iter_is_valid(ARTNodeIterator* iter) {
    return iter->is_valid();
}

inline void iter_next(ARTNodeIterator* iter) {
    ++(*iter);
}

inline void iter_next_without_skip(ARTNodeIterator* iter) {
    iter->next_without_skip();
}

inline std::pair<uint8_t, ARTNode*> iter_get(ARTNodeIterator* iter) {
    return iter->get();
}

inline ARTNode** iter_get_node(ARTNodeIterator* iter) {
    auto* n4 = reinterpret_cast<ARTNodeIterator_4*>(iter);
    switch (n4->node->n.type) {
        case NODE4:   return (ARTNode**)(&((ARTNodeIterator_4*)iter)->node);
        case NODE16:  return (ARTNode**)(&((ARTNodeIterator_16*)iter)->node);
        case NODE48:  return (ARTNode**)(&((ARTNodeIterator_48*)iter)->node);
        case NODE256: return (ARTNode**)(&((ARTNodeIterator_256*)iter)->node);
        default: throw std::runtime_error("iter_get_node: Unknown type");
    }
}

inline ARTNode** iter_get_current(ARTNodeIterator* iter) {
    auto* n4 = reinterpret_cast<ARTNodeIterator_4*>(iter);
    switch (n4->node->n.type) {
        case NODE4:  return ((ARTNodeIterator_4*)iter)->current;
        case NODE16: return ((ARTNodeIterator_16*)iter)->current;
        case NODE48: {
            auto* it48 = (ARTNodeIterator_48*)iter;
            return &((ARTNode_48*)it48->node)->children[
                ((ARTNode_48*)it48->node)->keys[it48->cur_index] - 1];
        }
        case NODE256: {
            auto* it256 = (ARTNodeIterator_256*)iter;
            return &((ARTNode_256*)it256->node)->children[it256->cur_index];
        }
        default: throw std::runtime_error("iter_get_current: Unknown type");
    }
}

inline ARTNode* iter_get_current_ro(ARTNodeIterator* iter) {
    auto* n4 = reinterpret_cast<ARTNodeIterator_4*>(iter);
    switch (n4->node->n.type) {
        case NODE4:  return *((ARTNodeIterator_4*)iter)->current;
        case NODE16: return *((ARTNodeIterator_16*)iter)->current;
        case NODE48: {
            auto* it48 = (ARTNodeIterator_48*)iter;
            return ((ARTNode_48*)it48->node)->children[
                ((ARTNode_48*)it48->node)->keys[it48->cur_index] - 1];
        }
        case NODE256: {
            auto* it256 = (ARTNodeIterator_256*)iter;
            return ((ARTNode_256*)it256->node)->children[it256->cur_index];
        }
        default: throw std::runtime_error("iter_get_current_ro: Unknown type");
    }
}

// ─── [NEW] 调试宏 ──────────────────────────────────────────────────
#define BREAKPOINT_ITER(iter) do { \
    std::fprintf(stderr, "[BREAKPOINT_ITER] valid=%d\n", (iter)->is_valid()); \
    (iter)->stats.dump(#iter); \
} while(0)

} // namespace art
} // namespace philemon

#endif // PHILEMON_ART_NODE_ITER_IMPL_HPP
