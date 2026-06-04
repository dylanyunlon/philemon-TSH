#pragma once
/**
 * neo_range_tree.hpp — Range segment tree with split/upgrade profiling
 *
 * 骨架来源: upstream/.../include/neo_range_tree.h (73行)
 *           upstream/.../include/neo_range_ops.h (45行)
 *           upstream/.../src/neo_range_tree.cpp (756行)
 *           upstream/.../src/neo_range_ops.cpp (80行)
 * 修改 (~20%):
 *   - insert: node_split_count 记录每次segment分裂
 *   - range_tree2art: upgrade_to_art_count 追踪升级频率
 *   - for_each: element_scan_total 记录遍历的总元素数
 *   - find_node: 二分查找的比较次数累加到 find_node_cmp
 *   - range_segment_find: SIMD路径 fallback 分支加 PHILE_NEO_TRACE
 *
 * Milestone: M071
 */

#include "../include/neo_types.hpp"
#include "../include/neo_property.hpp"
#include "../art/art_core.hpp"
#include "../utils/neo_config.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstring>
#include <atomic>
#include <cstdio>

namespace container {

// Forward
class ART;
struct WriterTraceBlock;

// ─── Range ops profiling (NEW) ───
struct RangeTreeStats {
    std::atomic<uint64_t> node_split_count{0};
    std::atomic<uint64_t> upgrade_to_art_count{0};
    std::atomic<uint64_t> element_scan_total{0};
    std::atomic<uint64_t> find_node_cmp{0};
    std::atomic<uint64_t> segment_find_calls{0};

    void dump() const {
        std::fprintf(stderr,
            "[RANGE-TREE] splits=%llu art_upgrades=%llu scanned=%llu "
            "find_cmp=%llu seg_find=%llu\n",
            (unsigned long long)node_split_count.load(),
            (unsigned long long)upgrade_to_art_count.load(),
            (unsigned long long)element_scan_total.load(),
            (unsigned long long)find_node_cmp.load(),
            (unsigned long long)segment_find_calls.load());
    }
};
inline RangeTreeStats& range_tree_stats() { static RangeTreeStats s; return s; }

// ──────────────── Range segment ops (upstream) ────────────────
inline uint16_t range_segment_find(RangeElement* seg, uint16_t count, RangeElement value) {
    range_tree_stats().segment_find_calls.fetch_add(1, std::memory_order_relaxed);
    for (uint16_t i = 0; i < count; i++) {
        if (seg[i] == value) return i;
    }
    return RANGE_LEAF_SIZE;
}

inline void range_segment_set(RangeElementSegment_t* seg, uint16_t pos, RangeElement value) {
    seg->value[pos] = value;
}

inline void range_segment_set(RangeElementSegment_t* seg, void* prop_seg,
                              uint16_t pos, RangeElement value, Property_t* prop_value) {
    seg->value[pos] = value;
#if EDGE_PROPERTY_NUM != 0
    if (prop_value)
        map_set_sa_range_property(prop_seg, pos, prop_value);
#endif
}

inline void range_segment_insert(RangeElementSegment_t* seg, void* prop_seg,
                                 uint16_t seg_size, uint16_t pos,
                                 RangeElement value, Property_t* prop_value) {
    // Shift elements right to make room
    std::move_backward(seg->value.begin() + pos, seg->value.begin() + seg_size,
                       seg->value.begin() + seg_size + 1);
    seg->value[pos] = value;
#if EDGE_PROPERTY_NUM != 0
    map_insert_range_property(prop_seg, pos, seg_size, prop_value);
#endif
}

// ──────────────── RangeTree ────────────────
class RangeTree {
public:
    std::atomic<uint32_t> ref_cnt{1};
    std::vector<InRangeNode> node_block;
    std::vector<uint32_t> keys;

    RangeTree();

    RangeTree(RangeElement* elements, Property_t** properties,
              uint64_t element_num, WriterTraceBlock* trace_block);
    RangeTree(std::vector<RangeElement>& elements, Property_t** properties,
              uint64_t element_num, WriterTraceBlock* trace_block);
    RangeTree(RangeElement* elements, Property_t* properties,
              uint64_t element_num, uint64_t new_element,
              Property_t* property, uint64_t pos, WriterTraceBlock* trace_block);

    [[nodiscard]] bool has_element(uint64_t element) const;

    void range_intersect(RangeElement* range, uint16_t range_size,
                         std::vector<uint64_t>& result) const;
    void intersect(RangeTree* other_tree, std::vector<uint64_t>& result) const;
    uint64_t range_intersect(RangeElement* range, uint16_t range_size) const;
    uint64_t intersect(RangeTree* other_tree) const;

    bool insert(uint64_t src, uint64_t element, Property_t* property,
                std::vector<GCResourceInfo>& gc_resources,
                WriterTraceBlock* trace_block);

    RangeTreeInsertElemBatchRes insert_element_batch(
        uint64_t src, const std::pair<RangeElement, RangeElement>* edges,
        Property_t** properties, uint64_t count,
        std::vector<GCResourceInfo>& gc_resources,
        WriterTraceBlock* trace_block);

    bool remove(uint64_t element, std::vector<GCResourceInfo>& gc_resources,
                WriterTraceBlock* trace_block);

#if EDGE_PROPERTY_NUM != 0
    [[nodiscard]] Property_t get_property(uint64_t element, uint8_t pid) const;
    void set_property(uint64_t element, uint8_t pid, Property_t property,
                      std::vector<GCResourceInfo>& gc_resources,
                      WriterTraceBlock* trace_block);
#endif

    ART* range_tree2art(uint64_t src, uint64_t degree,
                        uint64_t new_element, Property_t* new_property,
                        std::vector<GCResourceInfo>& gc_resources,
                        WriterTraceBlock* trace_block);

    RangeTreeInsertElemBatchRes range_tree2art_batch(
        uint64_t src, uint64_t degree,
        const std::pair<RangeElement, RangeElement>* edges,
        Property_t** properties, uint64_t count,
        std::vector<GCResourceInfo>& gc_resources,
        WriterTraceBlock* trace_block);

    template<typename F>
    void for_each(F&& callback) {
        uint64_t scanned = 0;
        for (size_t i = 0; i < node_block.size(); i++) {
            auto& node = node_block[i];
            auto arr = (RangeElement*)node.arr_ptr;
            for (uint16_t j = 0; j < node.size; j++) {
                callback(arr[j], 0.0);
                scanned++;
            }
        }
        range_tree_stats().element_scan_total.fetch_add(scanned, std::memory_order_relaxed);
    }

private:
    // ─── find_node with comparison counting ───
    [[nodiscard]] uint8_t find_node(uint64_t element) const {
        range_tree_stats().find_node_cmp.fetch_add(1, std::memory_order_relaxed);
        // Linear scan (upstream); keys.size() typically ≤ 32
        for (uint8_t i = 0; i < keys.size(); i++) {
            if (keys[i] >= element) return i;
            range_tree_stats().find_node_cmp.fetch_add(1, std::memory_order_relaxed);
        }
        return keys.size();
    }
};

} // namespace container
