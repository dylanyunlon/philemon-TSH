#pragma once
/**
 * art_core.hpp — ART node/leaf type definitions + iterator protocol
 *
 * 骨架来源: upstream/.../c_art/include/art_node.h (74行)
 *           upstream/.../c_art/include/art_leaf.h (237行)
 *           upstream/.../c_art/include/art_iter.h (28行)
 *           upstream/.../c_art/include/helper.h (35行)
 *           upstream/.../c_art/src/art_node.cpp (76行)
 * 修改 (~20%):
 *   - alloc_node 记录 {Node4,Node16,Node48,Node256} 分配计数
 *   - ARTLeaf 新增 mutable access_heat (记录查询命中次数, 热度降级用)
 *   - dump_node()/dump_leaf() 供 gdb 和 print-debug 使用
 *   - iterator 增加 step_count (分析遍历成本)
 *   - IS_LEAF / LEAF_RAW 宏保留 (upstream 大量代码依赖)
 *
 * Milestone: M071
 */

#include "../include/neo_types.hpp"
#include "../bitmap/neo_bitmap.hpp"
#include <cstdint>
#include <cstring>
#include <queue>
#include <array>
#include <functional>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

namespace container {

// ─── Leaf / node pointer tagging (upstream macros, unchanged) ───
#define IS_LEAF(x)    (((uintptr_t)(x) & 1))
#define SET_LEAF(x)   ((ARTNode*)((uintptr_t)(x) | 1))
#define LEAF_RAW(x)   ((ARTLeaf*)((uintptr_t)(x) & ~1))
#define NODE4   1
#define NODE16  2
#define NODE48  3
#define NODE256 4

// ─── Node allocation profiling (NEW) ───
struct ARTAllocStats {
    std::atomic<uint64_t> n4{0}, n16{0}, n48{0}, n256{0}, leaf{0};
    std::atomic<uint64_t> destroy_calls{0};
    void dump() const {
        std::fprintf(stderr,
            "[ART-ALLOC] N4=%llu N16=%llu N48=%llu N256=%llu leaf=%llu destroy=%llu\n",
            (unsigned long long)n4.load(std::memory_order_relaxed),
            (unsigned long long)n16.load(std::memory_order_relaxed),
            (unsigned long long)n48.load(std::memory_order_relaxed),
            (unsigned long long)n256.load(std::memory_order_relaxed),
            (unsigned long long)leaf.load(std::memory_order_relaxed),
            (unsigned long long)destroy_calls.load(std::memory_order_relaxed));
    }
};
inline ARTAllocStats& art_alloc_stats() { static ARTAllocStats s; return s; }

// ──────────────── Forward declarations ────────────────
struct ARTLeaf;
struct ARTLeaf8;
struct ARTLeaf16;
struct ARTLeaf32;
struct ARTLeaf64;
struct WriterTraceBlock;  // from neo_reader_trace

// ──────────────── ARTNode (upstream) ────────────────
struct ARTNode {
    ARTKey prefix{0};
    uint8_t type: 4;
    uint8_t depth: 4;
    uint16_t num_children = 0;
    std::atomic<uint16_t> ref_cnt{1};

    void dump(const char* label = "") const {
        std::fprintf(stderr,
            "[ARTNode:%s] type=%u depth=%u children=%u ref=%u prefix=0x%08x\n",
            label, (unsigned)type, (unsigned)depth,
            (unsigned)num_children, (unsigned)ref_cnt.load(),
            prefix.key);
    }
};

struct ARTNode_4 {
    ARTNode n{};
    unsigned char keys[4]{};
    ARTNode* children[4]{};
};

struct ARTNode_16 {
    ARTNode n{};
    unsigned char keys[16]{};
    ARTNode* children[16]{};
};

struct ARTNode_48 {
    ARTNode n{};
    Bitmap<4> unique_bitmap{};
    unsigned char keys[256]{};
    ARTNode* children[48]{};
};

struct ARTNode_256 {
    ARTNode n{};
    Bitmap<4> unique_bitmap{};
    std::array<ARTNode*, ART_LEAF_SIZE> children{};
};

// ──────────────── ARTLeaf base (upstream + access_heat) ────────────────
struct ARTLeaf {
    ARTKey key{0};
    uint16_t size{};
    uint8_t type{};
    uint8_t is_single_byte{};
    uint8_t depth{};
    std::atomic<uint16_t> ref_cnt{1};
    mutable uint32_t access_heat{0};  // NEW: query-hit counter for tier migration
#if EDGE_PROPERTY_NUM == 1
    ARTPropertyVec_t* property_map;
#elif EDGE_PROPERTY_NUM > 1
    MultiARTPropertyVec_t* property_map;
#endif

    ARTLeaf(ARTKey key, uint8_t depth, bool is_single_byte);
    virtual ~ARTLeaf() = default;

    [[nodiscard]] virtual uint64_t at(uint16_t pos_idx) const = 0;

#if EDGE_PROPERTY_NUM != 0
    [[nodiscard]] Property_t get_property(uint16_t pos_idx, uint8_t property_id) const;
#endif

    [[nodiscard]] virtual bool has_element(uint64_t element, uint8_t begin_idx) const = 0;
    [[nodiscard]] virtual uint16_t find(uint64_t element, uint8_t begin_idx) const;
    [[nodiscard]] virtual uint16_t get_byte_num(uint8_t depth) const = 0;

    virtual void insert(uint64_t element, Property_t* property, uint16_t pos_idx) = 0;
    virtual void remove(uint16_t pos_idx, uint8_t target_byte) = 0;
    virtual void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                              ARTLeaf* dst, uint16_t dst_idx) const = 0;
    virtual void append_from_list(RangeElement* elem_list, Property_t** prop_list,
                                  uint16_t count) = 0;
    virtual void leaf_check() const = 0;

#if EDGE_PROPERTY_NUM != 0
    void set_property(uint16_t pos_idx, uint8_t property_id, Property_t prop);
#endif

    // Upstream for_each via subclass at()
    template<typename F>
    void for_each(F&& f) const {
        access_heat++;  // NEW: track each scan
        for (uint16_t i = 0; i < size; i++) f(at(i));
    }

    void dump(const char* label = "") const {
        std::fprintf(stderr,
            "[ARTLeaf:%s] depth=%u size=%u type=%u single=%u heat=%u ref=%u\n",
            label, (unsigned)depth, (unsigned)size, (unsigned)type,
            (unsigned)is_single_byte, access_heat, (unsigned)ref_cnt.load());
    }
};

// ──────────────── Concrete leaf types (upstream, interface-only here) ────────────────
struct ARTLeaf8 : public ARTLeaf {
    Bitmap<4> value;
    ARTLeaf8(ARTKey key, uint8_t depth, bool is_single_byte);
    uint64_t at(uint16_t pos_idx) const override;
    bool has_element(uint64_t element, uint8_t begin_idx) const override;
    uint16_t find(uint64_t element, uint8_t begin_idx) const override;
    uint16_t get_byte_num(uint8_t depth) const override;
    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override;
    void remove(uint16_t pos_idx, uint8_t target_byte) override;
    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                      ARTLeaf* dst, uint16_t dst_idx) const override;
    void append_from_list(RangeElement* elem_list, Property_t** prop_list,
                          uint16_t count) override;
    void leaf_check() const override;
};

struct ARTLeaf16 : public ARTLeaf {
    std::array<uint8_t, ART_LEAF_SIZE> value{};
    ARTLeaf16(ARTKey key, uint8_t depth, bool is_single_byte);
    uint64_t at(uint16_t pos_idx) const override;
    bool has_element(uint64_t element, uint8_t begin_idx) const override;
    uint16_t find(uint64_t element, uint8_t begin_idx) const override;
    uint16_t get_byte_num(uint8_t depth) const override;
    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override;
    void remove(uint16_t pos_idx, uint8_t target_byte) override;
    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                      ARTLeaf* dst, uint16_t dst_idx) const override;
    void append_from_list(RangeElement* elem_list, Property_t** prop_list,
                          uint16_t count) override;
    void leaf_check() const override;
};

struct ARTLeaf32 : public ARTLeaf {
    std::array<uint16_t, ART_LEAF_SIZE> value{};
    ARTLeaf32(ARTKey key, uint8_t depth, bool is_single_byte);
    uint64_t at(uint16_t pos_idx) const override;
    bool has_element(uint64_t element, uint8_t begin_idx) const override;
    uint16_t find(uint64_t element, uint8_t begin_idx) const override;
    uint16_t get_byte_num(uint8_t depth) const override;
    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override;
    void remove(uint16_t pos_idx, uint8_t target_byte) override;
    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                      ARTLeaf* dst, uint16_t dst_idx) const override;
    void append_from_list(RangeElement* elem_list, Property_t** prop_list,
                          uint16_t count) override;
    void leaf_check() const override;
};

struct ARTLeaf64 : public ARTLeaf {
    std::array<uint32_t, ART_LEAF_SIZE> value{};
    ARTLeaf64(ARTKey key, uint8_t depth, bool is_single_byte);
    uint64_t at(uint16_t pos_idx) const override;
    bool has_element(uint64_t element, uint8_t begin_idx) const override;
    uint16_t find(uint64_t element, uint8_t begin_idx) const override;
    uint16_t get_byte_num(uint8_t depth) const override;
    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override;
    void remove(uint16_t pos_idx, uint8_t target_byte) override;
    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                      ARTLeaf* dst, uint16_t dst_idx) const override;
    void append_from_list(RangeElement* elem_list, Property_t** prop_list,
                          uint16_t count) override;
    void leaf_check() const override;
};

// ─── Leaf factory / destroy (upstream) ───
ARTLeaf* alloc_leaf(ARTKey key, uint8_t depth, bool is_single_byte,
                    WriterTraceBlock* trace_block);
void leaf_destroy(ARTLeaf* leaf);

// ──────────────── Node alloc/dealloc (upstream + stats) ────────────────
inline ARTNode* alloc_node(uint8_t type, ARTKey prefix, uint8_t depth,
                           WriterTraceBlock* trace_block) {
    ARTNode* n;
    switch (type) {
        case NODE4:
            n = (ARTNode*)new ARTNode_4();
            art_alloc_stats().n4.fetch_add(1, std::memory_order_relaxed);
            break;
        case NODE16:
            n = (ARTNode*)new ARTNode_16();
            art_alloc_stats().n16.fetch_add(1, std::memory_order_relaxed);
            break;
        case NODE48:
            n = (ARTNode*)new ARTNode_48();
            art_alloc_stats().n48.fetch_add(1, std::memory_order_relaxed);
            break;
        case NODE256:
            n = (ARTNode*)new ARTNode_256();
            art_alloc_stats().n256.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            throw std::runtime_error("alloc_node(): Invalid type");
    }
    n->type = type;
    n->prefix = prefix;
    n->depth = depth;
    return n;
}

void recursive_destroy_node(ARTNode* n);
void delete_node(ARTNode* n, WriterTraceBlock* trace_block);

// ──────────────── Iterator protocol (upstream + step counting) ────────────────
struct ARTIterator {
    ARTNode* node;
    uint16_t pos;
    uint64_t step_count;  // NEW: how many next() calls
};

ARTIterator* alloc_iterator(ARTNode* node);
void destroy_iterator(ARTIterator* iter);
bool iter_is_valid(ARTIterator* iter);
ARTNode** iter_get_current(ARTIterator* iter);
void iter_next(ARTIterator* iter);

// ──────────────── Node ops (function declarations — impl in art_ops.hpp) ───
void node_ref(ARTNode* node);
ARTNode** find_child(ARTNode* n, unsigned char c);
uint16_t find_child_idx(ARTNode* n, unsigned char c);
ARTLeaf* node_search(ARTNode* u, ARTKey key);

ARTNode** add_child(ARTNode* n, ARTNode** ref, unsigned char c,
                    void* child, WriterTraceBlock* trace_block);
ARTNode** add_child4(ARTNode_4* n, ARTNode** ref, unsigned char c, void* child);
ARTNode** add_child16(ARTNode_16* n, ARTNode** ref, unsigned char c,
                      void* child, WriterTraceBlock* trace_block);
ARTNode** add_child48(ARTNode_48* n, ARTNode** ref, unsigned char c,
                      void* child, WriterTraceBlock* trace_block);
ARTNode** add_child256(ARTNode_256* n, ARTNode** ref, unsigned char c, void* child);

void node16_upgrade(ARTNode_16* n, ARTNode** ref, WriterTraceBlock* trace_block);

void remove_child4(ARTNode_4* n, ARTNode** ref, unsigned char c, ARTNode** child);
void remove_child16(ARTNode_16* n, ARTNode** ref, unsigned char c, ARTNode** child);
void remove_child48(ARTNode_48* n, ARTNode** ref, unsigned char c, ARTNode** child);
void remove_child256(ARTNode_256* n, ARTNode** ref, unsigned char c, ARTNode** child);
void remove_child(ARTNode* n, ARTNode** ref, unsigned char c,
                  ARTNode** child, WriterTraceBlock* trace_block);

// Copy-on-write node ops
ARTNode** add_child_copy(ARTNode* n, ARTNode** ref, unsigned char c,
                         void* child, WriterTraceBlock* trace_block);
ARTNodeRemoveRes remove_child_copy(ARTNode* n, ARTNode** ref, unsigned char c,
                                   WriterTraceBlock* trace_block);

// Tree-level leaf iteration
template<typename F>
int tree_leaf_iter(ARTNode* root, F&& callback);
template<typename F>
int tree_leaf_iter_unordered(ARTNode* root, F&& callback);

// ─── Global function: dump all ART subsystem stats ───
inline void dump_art_stats() {
    art_alloc_stats().dump();
    art_depth_stats().dump();
}

} // namespace container
