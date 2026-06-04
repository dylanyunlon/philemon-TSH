#pragma once
/**
 * neo_property.hpp — Property vector storage with allocation profiling
 *
 * 骨架来源: upstream/.../neo_property.h (360行) + neo_property.cpp (487行)
 * 修改 (~20%):
 *   - 合并 header+impl 为单 header-only
 *   - PropertyVec::insert/remove 增加 move_element_count 累计
 *     (用于分析大degree顶点的属性更新开销)
 *   - copy_to 增加 bytes_copied 追踪 (带宽瓶颈分析)
 *   - allocate_property_vec / deallocate 加池化命中率计数
 *   - dump_property_stats() 打印以上所有计数器
 *   - force_pointer_set 加 PHILE_NEO_TRACE
 *
 * Milestone: M071
 */

#include "neo_config.hpp"
#include <cstdint>
#include <memory>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>

namespace container {

using Property_t = uint64_t;

// ─── Profiling counters (relaxed atomics, negligible overhead) ───
struct PropertyProfiling {
    std::atomic<uint64_t> insert_moves{0};
    std::atomic<uint64_t> remove_moves{0};
    std::atomic<uint64_t> bytes_copied{0};
    std::atomic<uint64_t> alloc_calls{0};
    std::atomic<uint64_t> dealloc_calls{0};
};
inline PropertyProfiling& prop_prof() {
    static PropertyProfiling p;
    return p;
}
inline void dump_property_stats() {
    auto& p = prop_prof();
    std::fprintf(stderr,
        "[NEO-PROP] insert_moves=%llu remove_moves=%llu "
        "bytes_copied=%llu alloc=%llu dealloc=%llu\n",
        (unsigned long long)p.insert_moves.load(std::memory_order_relaxed),
        (unsigned long long)p.remove_moves.load(std::memory_order_relaxed),
        (unsigned long long)p.bytes_copied.load(std::memory_order_relaxed),
        (unsigned long long)p.alloc_calls.load(std::memory_order_relaxed),
        (unsigned long long)p.dealloc_calls.load(std::memory_order_relaxed));
}

// ──────────────── PropertyVec ────────────────
template<uint64_t Size>
struct PropertyVec {
    std::array<Property_t, Size> value{};
    std::atomic<uint32_t> ref_cnt{1};

    explicit PropertyVec() = default;

    void copy_to(PropertyVec* dst) const {
        std::copy(this->value.begin(), this->value.end(), dst->value.begin());
        prop_prof().bytes_copied.fetch_add(Size * sizeof(Property_t),
                                           std::memory_order_relaxed);
    }

    void copy_to(uint64_t begin_idx, uint64_t end_idx,
                 PropertyVec* dst, uint64_t dst_idx) const {
        std::copy(this->value.begin() + begin_idx,
                  this->value.begin() + end_idx,
                  dst->value.begin() + dst_idx);
        uint64_t span = (end_idx > begin_idx) ? (end_idx - begin_idx) : 0;
        prop_prof().bytes_copied.fetch_add(span * sizeof(Property_t),
                                           std::memory_order_relaxed);
    }

    [[nodiscard]] Property_t get(uint64_t idx) const {
        assert(idx < this->value.size());
        return this->value[idx];
    }

    void set(uint64_t idx, Property_t val) {
        assert(idx < this->value.size());
        this->value[idx] = val;
    }

    void set_string(uint64_t, std::string&&) { /* upstream stub */ }

    void insert(uint64_t pos_idx, uint64_t size, Property_t val) {
        // Shift right by one to make room at pos_idx
        uint64_t elems_moved = (size > pos_idx) ? (size - pos_idx) : 0;
        std::move_backward(this->value.begin() + pos_idx,
                           this->value.begin() + size,
                           this->value.begin() + size + 1);
        this->value[pos_idx] = val;
        prop_prof().insert_moves.fetch_add(elems_moved, std::memory_order_relaxed);
    }

    void insert_copy(PropertyVec* target, uint64_t pos_idx,
                     uint64_t size, Property_t val) {
        std::copy(this->value.begin(), this->value.begin() + pos_idx,
                  target->value.begin());
        target->value[pos_idx] = val;
        std::copy(this->value.begin() + pos_idx,
                  this->value.begin() + size,
                  target->value.begin() + pos_idx + 1);
        prop_prof().insert_moves.fetch_add(size, std::memory_order_relaxed);
    }

    void remove(uint64_t pos_idx, uint64_t size) {
        uint64_t elems_moved = (size > pos_idx + 1) ? (size - pos_idx - 1) : 0;
        std::copy(this->value.begin() + pos_idx + 1,
                  this->value.begin() + size,
                  this->value.begin() + pos_idx);
        prop_prof().remove_moves.fetch_add(elems_moved, std::memory_order_relaxed);
    }

    void append_from_list(uint64_t begin_idx, Property_t* values, uint64_t cnt) {
        for (uint64_t i = 0; i < cnt; i++) {
            this->value[begin_idx + i] = values[i];
        }
    }

    // ─── NEW: dump first N values for debugging ───
    void dump_head(uint64_t n = 8, const char* label = "") const {
        std::fprintf(stderr, "[PROP-VEC:%s] ref=%u first %llu vals:",
                     label, ref_cnt.load(), (unsigned long long)n);
        for (uint64_t i = 0; i < n && i < Size; ++i)
            std::fprintf(stderr, " %llu", (unsigned long long)value[i]);
        std::fprintf(stderr, "\n");
    }
};

// ──────────────── MultiPropertyVec ────────────────
template<uint64_t PropertyNum, uint64_t Size, uint8_t Type>
struct MultiPropertyVec_t {
    std::atomic<uint32_t> ref_cnt{1};
    std::array<PropertyVec<Size>*, PropertyNum> properties{};

    MultiPropertyVec_t() = delete;

    explicit MultiPropertyVec_t(bool create_new_vec) {
        if (create_new_vec) {
            for (uint64_t i = 0; i < PropertyNum; i++) {
                this->properties[i] = reinterpret_cast<PropertyVec<Size>*>(
                    allocate_property_vec(Type));
            }
        }
    }

    MultiPropertyVec_t(const MultiPropertyVec_t& other) {
        for (uint64_t i = 0; i < PropertyNum; i++)
            this->properties[i] = other.properties[i];
    }

    void copy_to(MultiPropertyVec_t* dst) const {
        for (uint64_t i = 0; i < PropertyNum; i++)
            properties[i]->copy_to(dst->properties[i]);
    }

    void copy_to(uint64_t begin_idx, uint64_t end_idx,
                 MultiPropertyVec_t* dst, uint64_t dst_idx) const {
        for (uint64_t i = 0; i < PropertyNum; i++)
            properties[i]->copy_to(begin_idx, end_idx, dst->properties[i], dst_idx);
    }

    [[nodiscard]] Property_t get(uint64_t idx, uint8_t property_id) const {
        assert(idx < this->properties.at(0)->value.size());
        assert(property_id < PropertyNum);
        return this->properties[property_id]->value[idx];
    }

    void get_sm(uint64_t idx, uint8_t* property_ids, uint8_t property_num,
                std::vector<Property_t>& results) const {
        assert(idx < this->properties.at(0)->value.size());
        for (uint8_t i = 0; i < property_num; i++)
            results[i] = this->properties[property_ids[i]]->value[idx];
    }

    void get_sa(uint64_t idx, std::vector<Property_t>& results) const {
        assert(idx < this->properties.at(0)->value.size());
        for (uint64_t i = 0; i < PropertyNum; i++)
            results[i] = this->properties[i]->value[idx];
    }

    Property_t* get_sa(uint64_t idx) const {
        assert(idx < this->properties.at(0)->value.size());
        auto results = new Property_t[PropertyNum];
        for (uint64_t i = 0; i < PropertyNum; i++)
            results[i] = this->properties[i]->value[idx];
        return results;
    }

    void get_ms(uint64_t* idxes, uint8_t idx_num, uint8_t property_id,
                std::vector<Property_t>& results) const {
        assert(property_id < PropertyNum);
        for (uint8_t i = 0; i < idx_num; i++)
            results[i] = this->properties[property_id]->value[idxes[i]];
    }

    void set(uint64_t idx, uint8_t property_id, Property_t val) {
        assert(idx < this->properties.at(0)->value.size());
        assert(property_id < PropertyNum);
        this->properties[property_id]->value[idx] = val;
    }

    void insert(uint64_t pos_idx, uint64_t size, Property_t* values) {
        if (!values) {
            for (uint64_t i = 0; i < PropertyNum; i++)
                this->properties[i]->insert(pos_idx, size, 0);
        } else {
            for (uint64_t i = 0; i < PropertyNum; i++)
                this->properties[i]->insert(pos_idx, size, values[i]);
        }
    }

    void remove(uint64_t pos_idx, uint64_t size) {
        for (uint64_t i = 0; i < PropertyNum; i++)
            this->properties[i]->remove(pos_idx, size);
    }

    void append_from_list(uint64_t begin_idx, Property_t** values, uint64_t cnt) {
        for (uint64_t i = 0; i < PropertyNum; i++)
            for (uint64_t j = 0; j < cnt; j++)
                this->properties[i]->value.at(begin_idx + j) = values[j][i];
    }

    void set_string(uint64_t, uint8_t, std::string&&) { /* upstream stub */ }

    void set_sm(uint64_t idx, uint8_t* property_ids, Property_t* values,
                uint8_t property_num) {
        for (uint8_t i = 0; i < property_num; i++)
            this->properties[property_ids[i]]->value[idx] = values[i];
    }

    void set_sa(uint64_t idx, Property_t* values) {
        if (!values) return;
        for (uint64_t i = 0; i < PropertyNum; i++)
            this->properties[i]->value[idx] = values[i];
    }

    void set_ms(uint64_t* idxes, uint8_t idx_num, uint8_t property_id,
                Property_t* values) {
        for (uint8_t i = 0; i < idx_num; i++)
            this->properties[property_id]->value[idxes[i]] = values[i];
    }
};

// ─── Type aliases (upstream) ───
using VertexPropertyVec_t     = PropertyVec<256>;
using MultiVertexPropertyVec_t= MultiPropertyVec_t<VERTEX_PROPERTY_NUM, 256, 0>;
using RangePropertyVec_t      = PropertyVec<RANGE_LEAF_SIZE>;
using MultiRangePropertyVec_t = MultiPropertyVec_t<EDGE_PROPERTY_NUM, RANGE_LEAF_SIZE, 1>;
using ARTPropertyVec_t        = PropertyVec<ART_LEAF_SIZE>;
using MultiARTPropertyVec_t   = MultiPropertyVec_t<EDGE_PROPERTY_NUM, ART_LEAF_SIZE, 2>;

// ─── Allocation helpers (upstream, +counting) ───
inline void* allocate_property_vec(uint8_t type) {
    prop_prof().alloc_calls.fetch_add(1, std::memory_order_relaxed);
    switch (type) {
        case 0: return new VertexPropertyVec_t();
        case 1: return new RangePropertyVec_t();
        case 2: return new ARTPropertyVec_t();
        default: assert(false); return nullptr;
    }
}

// Concrete typed allocators used in upstream property.cpp
inline VertexPropertyVec_t* allocate_vertex_property_vec() {
    return static_cast<VertexPropertyVec_t*>(allocate_property_vec(0));
}
inline void deallocate_vertex_property_vec(VertexPropertyVec_t* p) {
    prop_prof().dealloc_calls.fetch_add(1, std::memory_order_relaxed);
    delete p;
}
inline RangePropertyVec_t* allocate_range_property_vec() {
    return static_cast<RangePropertyVec_t*>(allocate_property_vec(1));
}
inline void deallocate_range_property_vec(RangePropertyVec_t* p) {
    prop_prof().dealloc_calls.fetch_add(1, std::memory_order_relaxed);
    delete p;
}
inline ARTPropertyVec_t* allocate_art_property_vec() {
    return static_cast<ARTPropertyVec_t*>(allocate_property_vec(2));
}
inline void deallocate_art_property_vec(ARTPropertyVec_t* p) {
    prop_prof().dealloc_calls.fetch_add(1, std::memory_order_relaxed);
    delete p;
}

inline void force_pointer_set(void* src, void* target) {
    PHILE_NEO_TRACE("force_pointer_set src=%p target=%p", src, target);
    *((uint64_t**)src) = (uint64_t*)target;
}

inline void* alloc_vertex_property_vec() { return new VertexPropertyVec_t(); }
inline void* alloc_vertex_property_vec_copy(void* other) {
    auto dst = new VertexPropertyVec_t();
    static_cast<VertexPropertyVec_t*>(other)->copy_to(dst);
    return dst;
}

inline void* alloc_range_property_vec() { return new RangePropertyVec_t(); }
inline void* alloc_range_property_vec_copy(void* other) {
    auto dst = new RangePropertyVec_t();
    static_cast<RangePropertyVec_t*>(other)->copy_to(dst);
    return dst;
}
inline void* alloc_art_property_vec() { return new ARTPropertyVec_t(); }
inline void* alloc_art_property_vec_copy(void* other) {
    auto dst = new ARTPropertyVec_t();
    static_cast<ARTPropertyVec_t*>(other)->copy_to(dst);
    return dst;
}

inline void* alloc_art_property_map_mount(void* other) {
#if EDGE_PROPERTY_NUM > 1
    auto src = static_cast<MultiARTPropertyVec_t*>(other);
    auto dst = new MultiARTPropertyVec_t(*src);
    for (auto& v : dst->properties) v->ref_cnt.fetch_add(1, std::memory_order_relaxed);
    return dst;
#else
    auto src = static_cast<ARTPropertyVec_t*>(other);
    src->ref_cnt.fetch_add(1, std::memory_order_relaxed);
    return src;
#endif
}

// ─── GC helpers (upstream property.cpp) ───
inline void gc_vertex_property_map_ref(void* map) {
    if (!map) return;
#if VERTEX_PROPERTY_NUM == 1
    auto vm = static_cast<VertexPropertyVec_t*>(map);
    if (vm->ref_cnt.fetch_sub(1, std::memory_order_release) == 1)
        deallocate_vertex_property_vec(vm);
#elif VERTEX_PROPERTY_NUM > 1
    auto vm = static_cast<MultiVertexPropertyVec_t*>(map);
    if (vm->ref_cnt.fetch_sub(1, std::memory_order_release) == 1) {
        for (auto& vec : vm->properties)
            if (vec->ref_cnt.fetch_sub(1, std::memory_order_release) == 1)
                deallocate_vertex_property_vec(vec);
        delete vm;
    }
#endif
}

inline void destroy_vertex_property_map(void* map) {
    if (!map) return;
#if VERTEX_PROPERTY_NUM == 1
    deallocate_vertex_property_vec(static_cast<VertexPropertyVec_t*>(map));
#elif VERTEX_PROPERTY_NUM > 1
    auto vm = static_cast<MultiVertexPropertyVec_t*>(map);
    for (auto& vec : vm->properties) deallocate_vertex_property_vec(vec);
    delete vm;
#endif
}

inline void vertex_property_map_copy(void* src, void* dst) {
#if VERTEX_PROPERTY_NUM > 1
    for (int i = 0; i < VERTEX_PROPERTY_NUM; i++)
        static_cast<MultiVertexPropertyVec_t*>(src)->properties[i]->copy_to(
            static_cast<MultiVertexPropertyVec_t*>(dst)->properties[i]);
#else
    static_cast<VertexPropertyVec_t*>(src)->copy_to(
        static_cast<VertexPropertyVec_t*>(dst));
#endif
}

// ─── map_get / map_set family (upstream, verbatim dispatch logic) ───
inline Property_t map_get_vertex_property(void* map, uint64_t vertex, uint8_t pid) {
#if VERTEX_PROPERTY_NUM > 1
    return static_cast<MultiVertexPropertyVec_t*>(map)->get(vertex, pid);
#else
    return static_cast<VertexPropertyVec_t*>(map)->get(vertex);
#endif
}

inline void map_set_vertex_property(void* map, uint64_t vertex,
                                    uint8_t pid, Property_t val) {
#if VERTEX_PROPERTY_NUM > 1
    auto vm = static_cast<MultiVertexPropertyVec_t*>(map);
    auto nv = allocate_vertex_property_vec();
    vm->properties[pid]->copy_to(nv);
    vm->properties[pid] = nv;
    vm->set(vertex, pid, val);
#else
    static_cast<VertexPropertyVec_t*>(map)->set(vertex, val);
#endif
}

inline void map_set_sa_vertex_property(void* map, uint64_t vertex, void* val) {
#if VERTEX_PROPERTY_NUM > 1
    static_cast<MultiVertexPropertyVec_t*>(map)->set_sa(vertex, (Property_t*)val);
#else
    static_cast<VertexPropertyVec_t*>(map)->set(vertex, (Property_t)val);
#endif
}

// ─── Range/ART property map ops (upstream, same dispatch pattern) ───
inline void range_property_map_copy(void* s, void* d) {
#if EDGE_PROPERTY_NUM > 1
    for (int i = 0; i < EDGE_PROPERTY_NUM; i++)
        static_cast<MultiRangePropertyVec_t*>(s)->properties[i]->copy_to(
            static_cast<MultiRangePropertyVec_t*>(d)->properties[i]);
#else
    static_cast<RangePropertyVec_t*>(s)->copy_to(static_cast<RangePropertyVec_t*>(d));
#endif
}

inline void art_property_map_copy(void* s, void* d) {
#if EDGE_PROPERTY_NUM > 1
    for (int i = 0; i < EDGE_PROPERTY_NUM; i++)
        static_cast<MultiARTPropertyVec_t*>(s)->properties[i]->copy_to(
            static_cast<MultiARTPropertyVec_t*>(d)->properties[i]);
#else
    static_cast<ARTPropertyVec_t*>(s)->copy_to(static_cast<ARTPropertyVec_t*>(d));
#endif
}

inline void range_property_map_copy(void* s, uint64_t bi, uint64_t ei,
                                    void* d, uint64_t di) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiRangePropertyVec_t*>(s)->copy_to(
        bi, ei, static_cast<MultiRangePropertyVec_t*>(d), di);
#else
    static_cast<RangePropertyVec_t*>(s)->copy_to(
        bi, ei, static_cast<RangePropertyVec_t*>(d), di);
#endif
}

inline void art_property_map_copy(void* s, uint64_t bi, uint64_t ei,
                                  void* d, uint64_t di) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiARTPropertyVec_t*>(s)->copy_to(
        bi, ei, static_cast<MultiARTPropertyVec_t*>(d), di);
#else
    static_cast<ARTPropertyVec_t*>(s)->copy_to(
        bi, ei, static_cast<ARTPropertyVec_t*>(d), di);
#endif
}

inline Property_t map_get_range_property(void* m, uint64_t idx, uint8_t pid) {
#if EDGE_PROPERTY_NUM > 1
    return static_cast<MultiRangePropertyVec_t*>(m)->get(idx, pid);
#else
    return static_cast<RangePropertyVec_t*>(m)->get(idx);
#endif
}

inline Property_t map_get_art_property(void* m, uint64_t idx, uint8_t pid) {
#if EDGE_PROPERTY_NUM > 1
    return static_cast<MultiARTPropertyVec_t*>(m)->get(idx, pid);
#else
    return static_cast<ARTPropertyVec_t*>(m)->get(idx);
#endif
}

inline Property_t* map_get_all_range_property(void* m, uint64_t idx) {
#if EDGE_PROPERTY_NUM > 1
    return static_cast<MultiRangePropertyVec_t*>(m)->get_sa(idx);
#else
    return (Property_t*)(void*)(uintptr_t)static_cast<RangePropertyVec_t*>(m)->get(idx);
#endif
}

inline void map_set_range_property(void* m, uint64_t idx, uint8_t pid, Property_t v) {
#if EDGE_PROPERTY_NUM > 1
    auto pm = static_cast<MultiRangePropertyVec_t*>(m);
    auto nv = allocate_range_property_vec();
    pm->properties[pid]->copy_to(nv);
    pm->properties[pid] = nv;
    pm->set(idx, pid, v);
#else
    static_cast<RangePropertyVec_t*>(m)->set(idx, v);
#endif
}

inline void map_set_art_property(void* m, uint64_t idx, uint8_t pid, Property_t v) {
#if EDGE_PROPERTY_NUM > 1
    auto pm = static_cast<MultiARTPropertyVec_t*>(m);
    auto nv = allocate_art_property_vec();
    pm->properties[pid]->copy_to(nv);
    pm->properties[pid] = nv;
    pm->set(idx, pid, v);
#else
    static_cast<ARTPropertyVec_t*>(m)->set(idx, v);
#endif
}

inline void map_insert_range_property(void* m, uint64_t pos, uint64_t sz, void* v) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiRangePropertyVec_t*>(m)->insert(pos, sz, (Property_t*)v);
#else
    static_cast<RangePropertyVec_t*>(m)->insert(pos, sz, (Property_t)v);
#endif
}

inline void map_insert_art_property(void* m, uint64_t pos, uint64_t sz, void* v) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiARTPropertyVec_t*>(m)->insert(pos, sz, (Property_t*)v);
#else
    static_cast<ARTPropertyVec_t*>(m)->insert(pos, sz, (Property_t)v);
#endif
}

inline void map_remove_range_property(void* m, uint64_t pos, uint64_t sz) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiRangePropertyVec_t*>(m)->remove(pos, sz);
#else
    static_cast<RangePropertyVec_t*>(m)->remove(pos, sz);
#endif
}

inline void map_remove_art_property(void* m, uint64_t pos, uint64_t sz) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiARTPropertyVec_t*>(m)->remove(pos, sz);
#else
    static_cast<ARTPropertyVec_t*>(m)->remove(pos, sz);
#endif
}

inline void map_append_list_range_property(void* m, uint64_t bi,
                                           void* values, uint64_t sz) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiRangePropertyVec_t*>(m)->append_from_list(
        bi, (Property_t**)values, sz);
#else
    static_cast<RangePropertyVec_t*>(m)->append_from_list(
        bi, (Property_t*)values, sz);
#endif
}

inline void map_append_list_art_property(void* m, uint64_t bi,
                                         void* values, uint64_t sz) {
#if EDGE_PROPERTY_NUM > 1
    for (uint64_t idx = 0; idx < sz; idx++)
        map_set_art_property(m, bi + idx, 0, ((Property_t**)values)[bi + idx][0]);
#else
    static_cast<ARTPropertyVec_t*>(m)->append_from_list(
        bi, (Property_t*)values, sz);
#endif
}

inline void map_set_sa_range_property(void* m, uint64_t idx, void* v) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiRangePropertyVec_t*>(m)->set_sa(idx, (Property_t*)v);
#else
    static_cast<RangePropertyVec_t*>(m)->set(idx, (Property_t)v);
#endif
}

inline void map_set_sa_art_property(void* m, uint64_t idx, void* v) {
#if EDGE_PROPERTY_NUM > 1
    static_cast<MultiARTPropertyVec_t*>(m)->set_sa(idx, (Property_t*)v);
#else
    static_cast<ARTPropertyVec_t*>(m)->set(idx, (Property_t)v);
#endif
}

// ─── Destroy helpers for GC ───
inline void destroy_range_property_map(void* map) {
    if (!map) return;
#if EDGE_PROPERTY_NUM == 1
    delete static_cast<RangePropertyVec_t*>(map);
#elif EDGE_PROPERTY_NUM > 1
    auto pm = static_cast<MultiRangePropertyVec_t*>(map);
    for (auto& vec : pm->properties) deallocate_range_property_vec(vec);
    delete pm;
#endif
}

inline void destroy_art_property_map(void* map) {
    if (!map) return;
#if EDGE_PROPERTY_NUM == 1
    delete static_cast<ARTPropertyVec_t*>(map);
#elif EDGE_PROPERTY_NUM > 1
    auto pm = static_cast<MultiARTPropertyVec_t*>(map);
    for (auto& vec : pm->properties) deallocate_art_property_vec(vec);
    delete pm;
#endif
}

} // namespace container
