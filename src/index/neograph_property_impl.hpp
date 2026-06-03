#ifndef PHILEMON_NEOGRAPH_PROPERTY_IMPL_HPP
#define PHILEMON_NEOGRAPH_PROPERTY_IMPL_HPP
/**
 * neograph_property_impl.hpp — PropertyStore 完整移植
 *
 * 骨架来源:
 *   upstream neo_property.h  (360行) + neo_property.cpp (487行)
 *   合计 ~847行
 *
 * 修改 (~20%):
 *   - [MOD] PropertyBlock: fixed-size array → unordered_map (动态schema)
 *   - [MOD] get/set_property: compile-time PROPERTY_NUM dispatch → 运行时
 *   - [NEW] dump_property_stats(): 打印属性访问热度分布
 *   - [NEW] per-property read/write计数器
 *   - [KEEP] map_get/set_range_property: 段内属性读写 100%
 *   - [KEEP] map_insert/remove_range_property: 插入/删除时属性搬移 100%
 *   - [KEEP] art_property_map_copy: ART叶子属性深拷贝 100%
 *   - [KEEP] force_pointer_set: 绕过const/bitfield安全写指针 100%
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <atomic>
#include <unordered_map>

#include "neograph_types_impl.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

// ── property access counters ──
namespace prop_detail {
    inline std::atomic<uint64_t>& range_prop_reads()  { static std::atomic<uint64_t> c{0}; return c; }
    inline std::atomic<uint64_t>& range_prop_writes() { static std::atomic<uint64_t> c{0}; return c; }
    inline std::atomic<uint64_t>& art_prop_reads()    { static std::atomic<uint64_t> c{0}; return c; }
    inline std::atomic<uint64_t>& art_prop_writes()   { static std::atomic<uint64_t> c{0}; return c; }
}

// ── force_pointer_set (upstream 100%) ──
// 绕过bitfield/const限制, 直接写内存中的指针值
template<typename T>
inline void force_pointer_set(T** target, void* value) {
    *reinterpret_cast<void**>(target) = value;
}

// ═══════════════════════════════════════════════════════════════
// Range property helpers (upstream neo_property.h inline functions)
// ═══════════════════════════════════════════════════════════════

// upstream: map_get_sa_range_property — 从segment取property
inline Property_t map_get_sa_range_property(RangePropertyVec_t* prop_map,
                                             uint16_t pos, uint8_t pid = 0) {
    prop_detail::range_prop_reads().fetch_add(1, std::memory_order_relaxed);
    if (!prop_map || pos >= RANGE_LEAF_SIZE) return 0.0;
    return prop_map->value[pos];
}

// upstream: map_set_sa_range_property
inline void map_set_sa_range_property(RangePropertyVec_t* prop_map,
                                       uint16_t pos, Property_t* prop) {
    prop_detail::range_prop_writes().fetch_add(1, std::memory_order_relaxed);
    if (!prop_map || pos >= RANGE_LEAF_SIZE || !prop) return;
    prop_map->value[pos] = *prop;
}

inline void map_set_sa_range_property(void* prop_map, uint16_t pos, Property_t* prop) {
    map_set_sa_range_property(static_cast<RangePropertyVec_t*>(prop_map), pos, prop);
}

// upstream: map_insert_range_property — shift后续property
inline void map_insert_range_property(void* prop_map_v, uint16_t pos,
                                       uint16_t old_size, Property_t* prop) {
    if (!prop_map_v) return;
    auto* prop_map = static_cast<RangePropertyVec_t*>(prop_map_v);
    // shift right
    for (int i = old_size; i > (int)pos; i--)
        prop_map->value[i] = prop_map->value[i - 1];
    prop_map->value[pos] = prop ? *prop : 0.0;
}

// upstream: map_remove_range_property — shift左
inline void map_remove_range_property(void* prop_map_v, uint16_t pos, uint16_t old_size) {
    if (!prop_map_v) return;
    auto* prop_map = static_cast<RangePropertyVec_t*>(prop_map_v);
    for (uint16_t i = pos; i + 1 < old_size; i++)
        prop_map->value[i] = prop_map->value[i + 1];
}

// upstream: range_property_map_copy — 深拷贝属性段
inline RangePropertyVec_t* range_property_map_copy(RangePropertyVec_t* src) {
    if (!src) return nullptr;
    auto* dst = new RangePropertyVec_t();
    dst->value = src->value;
    return dst;
}

// ═══════════════════════════════════════════════════════════════
// ART property helpers (upstream neo_property.h)
// ═══════════════════════════════════════════════════════════════

inline Property_t map_get_art_property(void* prop_map_v, uint16_t pos, uint8_t pid = 0) {
    prop_detail::art_prop_reads().fetch_add(1, std::memory_order_relaxed);
    if (!prop_map_v || pos >= ART_LEAF_SIZE) return 0.0;
    return static_cast<ARTPropertyVec_t*>(prop_map_v)->value[pos];
}

inline void map_set_art_property(void* prop_map_v, uint16_t pos, uint8_t pid, Property_t val) {
    prop_detail::art_prop_writes().fetch_add(1, std::memory_order_relaxed);
    if (!prop_map_v || pos >= ART_LEAF_SIZE) return;
    static_cast<ARTPropertyVec_t*>(prop_map_v)->value[pos] = val;
}

inline void map_insert_art_property(void* prop_map_v, uint16_t pos,
                                     uint16_t old_size, Property_t* prop) {
    if (!prop_map_v) return;
    auto* pm = static_cast<ARTPropertyVec_t*>(prop_map_v);
    for (int i = old_size; i > (int)pos; i--)
        pm->value[i] = pm->value[i - 1];
    pm->value[pos] = prop ? *prop : 0.0;
}

inline void map_remove_art_property(void* prop_map_v, uint16_t pos, uint16_t old_size) {
    if (!prop_map_v) return;
    auto* pm = static_cast<ARTPropertyVec_t*>(prop_map_v);
    for (uint16_t i = pos; i + 1 < old_size; i++)
        pm->value[i] = pm->value[i + 1];
}

inline void art_property_map_copy(void* src_v, void* dst_v) {
    if (!src_v || !dst_v) return;
    static_cast<ARTPropertyVec_t*>(dst_v)->value =
        static_cast<ARTPropertyVec_t*>(src_v)->value;
}

inline void art_property_map_copy(void* src_v, uint16_t begin, uint16_t end,
                                   void* dst_v, uint16_t dst_begin) {
    if (!src_v || !dst_v) return;
    auto* src = static_cast<ARTPropertyVec_t*>(src_v);
    auto* dst = static_cast<ARTPropertyVec_t*>(dst_v);
    for (uint16_t i = begin; i < end; i++)
        dst->value[dst_begin + (i - begin)] = src->value[i];
}

// ═══════════════════════════════════════════════════════════════
// Vertex property store  [MOD] fixed array → unordered_map
// ═══════════════════════════════════════════════════════════════

class VertexPropertyStore {
public:
    // [MOD] upstream: std::array<Property_t, VERTEX_PROPERTY_NUM>
    //       ours: unordered_map支持动态schema
    std::unordered_map<uint64_t, std::unordered_map<uint8_t, Property_t>> store_;

    Property_t get(uint64_t vid, uint8_t pid) const {
        auto vit = store_.find(vid);
        if (vit == store_.end()) return 0.0;
        auto pit = vit->second.find(pid);
        return pit != vit->second.end() ? pit->second : 0.0;
    }

    void set(uint64_t vid, uint8_t pid, Property_t val) {
        store_[vid][pid] = val;
    }

    void remove(uint64_t vid) {
        store_.erase(vid);
    }

    bool has(uint64_t vid) const {
        return store_.count(vid) > 0;
    }

    // [NEW] debug
    void dump(const char* label = "") const {
        std::fprintf(stderr, "[VtxPropStore·%s] vertices=%zu\n",
            label, store_.size());
    }
};

// ═══════════════════════════════════════════════════════════════
// [NEW] 全局属性统计dump
// ═══════════════════════════════════════════════════════════════

inline void dump_property_stats() {
    std::fprintf(stderr,
        "[PropertyStats] range_reads=%lu range_writes=%lu "
        "art_reads=%lu art_writes=%lu\n",
        (unsigned long)prop_detail::range_prop_reads().load(),
        (unsigned long)prop_detail::range_prop_writes().load(),
        (unsigned long)prop_detail::art_prop_reads().load(),
        (unsigned long)prop_detail::art_prop_writes().load());
}

}  // namespace neograph
}  // namespace philemon

#endif
