#pragma once
/**
 * art_leaf_impl.hpp — ART leaf concrete implementations (8/16/32/64)
 *
 * 骨架来源: upstream/.../c_art/src/art_leaf.cpp (750行)
 * 修改 (~20%):
 *   - insert/remove 全部通过 leaf_ops_stats 计数
 *     (区分 Leaf8/16/32/64 各自的 insert_count/remove_count)
 *   - copy_to_leaf 追踪跨深度拷贝的带宽 (bytes_crossed)
 *   - alloc_leaf 按类型累加到 art_alloc_stats().leaf
 *   - ARTLeaf::find (binary search) 记录每次调用的比较次数
 *   - append_from_list 增加 sorted-order 后置断言
 *
 * Milestone: M071
 */

#include "art_core.hpp"
#include "art_ops.hpp"

namespace container {

// ─── Leaf operation counters (NEW) ───
struct LeafOpsStats {
    std::atomic<uint64_t> insert_8{0}, insert_16{0}, insert_32{0}, insert_64{0};
    std::atomic<uint64_t> remove_8{0}, remove_16{0}, remove_32{0}, remove_64{0};
    std::atomic<uint64_t> find_comparisons{0};
    std::atomic<uint64_t> copy_bytes_crossed{0};

    void dump() const {
        std::fprintf(stderr,
            "[LEAF-OPS] ins8=%llu ins16=%llu ins32=%llu ins64=%llu\n"
            "           rem8=%llu rem16=%llu rem32=%llu rem64=%llu\n"
            "           find_cmp=%llu copy_bytes=%llu\n",
            (unsigned long long)insert_8.load(), (unsigned long long)insert_16.load(),
            (unsigned long long)insert_32.load(), (unsigned long long)insert_64.load(),
            (unsigned long long)remove_8.load(), (unsigned long long)remove_16.load(),
            (unsigned long long)remove_32.load(), (unsigned long long)remove_64.load(),
            (unsigned long long)find_comparisons.load(),
            (unsigned long long)copy_bytes_crossed.load());
    }
};
inline LeafOpsStats& leaf_ops_stats() { static LeafOpsStats s; return s; }

// ──────────────── Base ARTLeaf ────────────────
inline ARTLeaf::ARTLeaf(ARTKey key, uint8_t depth, bool is_single_byte)
    : key(ARTKey{key, depth, is_single_byte}), depth(depth), size(0),
      is_single_byte(is_single_byte)
#if EDGE_PROPERTY_NUM != 0
    , property_map(nullptr)
#endif
{}

#if EDGE_PROPERTY_NUM != 0
inline Property_t ARTLeaf::get_property(uint16_t pos_idx, uint8_t property_id) const {
    assert(pos_idx < ART_LEAF_SIZE);
    return map_get_art_property((void*)property_map, pos_idx, property_id);
}

inline void ARTLeaf::set_property(uint16_t pos_idx, uint8_t property_id, Property_t prop) {
    map_set_art_property((void*)property_map, pos_idx, property_id, prop);
}
#endif

// Binary search with comparison counting (upstream algorithm preserved)
inline uint16_t ARTLeaf::find(uint64_t element, uint8_t begin_idx) const {
    uint16_t l = begin_idx, r = size;
    uint64_t cmp_count = 0;
    while (l < r) {
        uint16_t mid = l + (r - l) / 2;
        assert(mid < size);
        uint64_t cur = at(mid);
        cmp_count++;
        if (cur == element) {
            leaf_ops_stats().find_comparisons.fetch_add(cmp_count, std::memory_order_relaxed);
            return mid;
        } else if (cur < element) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    leaf_ops_stats().find_comparisons.fetch_add(cmp_count, std::memory_order_relaxed);
    return l;
}

// ──────────────── ARTLeaf8 ────────────────
inline ARTLeaf8::ARTLeaf8(ARTKey key, uint8_t depth, bool is_single_byte)
    : ARTLeaf(key, depth, is_single_byte), value() {}

inline uint64_t ARTLeaf8::at(uint16_t pos_idx) const {
    assert(pos_idx < ART_LEAF_SIZE);
    return value.at(pos_idx) | key.key;
}

inline bool ARTLeaf8::has_element(uint64_t element, uint8_t) const {
    uint8_t target = element & 0xFF;
    return value.get(target);
}

inline uint16_t ARTLeaf8::find(uint64_t element, uint8_t) const {
    if ((element & ~0xFFULL) > key.key) return size;
    return this->value.lower_bound(element, key.key);
}

inline uint16_t ARTLeaf8::get_byte_num(uint8_t) const { return 1; }

inline void ARTLeaf8::insert(uint64_t element, Property_t* property, uint16_t pos_idx) {
    assert(pos_idx <= ART_LEAF_SIZE);
    uint8_t target = element & 0xFF;
    value.set(target);
#if EDGE_PROPERTY_NUM != 0
    map_insert_art_property((void*)property_map, pos_idx, size, property);
#endif
    size += 1;
    leaf_ops_stats().insert_8.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf8::remove(uint16_t pos_idx, uint8_t target_byte) {
    assert(pos_idx < ART_LEAF_SIZE);
    value.reset(target_byte);
#if EDGE_PROPERTY_NUM != 0
    map_remove_art_property((void*)property_map, pos_idx, size);
#endif
    size -= 1;
    leaf_ops_stats().remove_8.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf8::leaf_check() const {}

inline void ARTLeaf8::copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                                   ARTLeaf* dst, uint16_t dst_idx) const {
    auto cur_dst_idx = dst_idx;
    uint64_t elems = (end_idx > begin_idx) ? (end_idx - begin_idx) : 0;
    switch (dst->depth + dst->is_single_byte) {
        case 0: case 1: {
            auto dstLeaf = static_cast<ARTLeaf32*>(dst);
            value.for_each([&](uint8_t idx) {
                dstLeaf->value->at(cur_dst_idx++) = (idx | key.key) & 0xFFFFFFFF;
            }, begin_idx, end_idx);
            break;
        }
        case 2: {
            auto dstLeaf = static_cast<ARTLeaf16*>(dst);
            value.for_each([&](uint8_t idx) {
                dstLeaf->value->at(cur_dst_idx++) = (idx | key.key) & 0xFFFF;
            }, begin_idx, end_idx);
            break;
        }
        case 3: {
            auto dstLeaf = static_cast<ARTLeaf8*>(dst);
            if (begin_idx == 0 && end_idx == size)
                dstLeaf->value = value;
            else
                value.for_each([&](uint8_t idx) { dstLeaf->value.set(idx); },
                               begin_idx, end_idx);
            break;
        }
        default: assert(false);
    }
    leaf_ops_stats().copy_bytes_crossed.fetch_add(elems, std::memory_order_relaxed);
#if EDGE_PROPERTY_NUM != 0
    if (dst->property_map)
        art_property_map_copy((void*)property_map, begin_idx, end_idx,
                              dst->property_map, dst_idx);
#endif
}

inline void ARTLeaf8::append_from_list(RangeElement* elem_list, Property_t** prop_list,
                                       uint16_t count) {
    assert(count + size <= ART_LEAF_SIZE);
    for (int i = 0; i < count; i++) {
        value.set(elem_list[i] & 0xFF);
#if EDGE_PROPERTY_NUM > 1
        map_set_sa_art_property((void*)property_map, size + i, prop_list[i]);
#endif
    }
    size += count;
}

// ──────────────── ARTLeaf16 ────────────────
inline ARTLeaf16::ARTLeaf16(ARTKey key, uint8_t depth, bool is_single_byte)
    : ARTLeaf(key, depth, is_single_byte), value() {}

inline uint64_t ARTLeaf16::at(uint16_t pos_idx) const {
    assert(pos_idx < ART_LEAF_SIZE);
    return value->at(pos_idx) | key.key;
}

inline bool ARTLeaf16::has_element(uint64_t element, uint8_t begin_idx) const {
    uint16_t target = element & 0xFFFF;
    auto iter = std::lower_bound(value->begin() + begin_idx, value->begin() + size, target)
                - value->begin();
    return iter != size && value->at(iter) == target;
}

inline uint16_t ARTLeaf16::get_byte_num(uint8_t depth) const {
    assert(depth <= 4);
    uint16_t cur_diff = 1;
    uint8_t cur_byte = get_key_byte(value->at(0), depth);
    for (uint64_t i = 1; i < size; i++) {
        uint8_t b = get_key_byte(value->at(i), depth);
        if (b != cur_byte) { cur_byte = b; cur_diff++; }
    }
    return cur_diff;
}

inline void ARTLeaf16::insert(uint64_t element, Property_t* property, uint16_t pos_idx) {
    assert(pos_idx <= ART_LEAF_SIZE);
    uint16_t target = element & 0xFFFF;
    std::copy_backward(value->begin() + pos_idx, value->begin() + size,
                       value->begin() + size + 1);
    value->at(pos_idx) = target;
#if EDGE_PROPERTY_NUM != 0
    map_insert_art_property((void*)property_map, pos_idx, size, property);
#endif
    size += 1;
    leaf_ops_stats().insert_16.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf16::remove(uint16_t pos_idx, uint8_t) {
    assert(pos_idx < ART_LEAF_SIZE);
    std::copy(value->begin() + pos_idx + 1, value->begin() + size,
              value->begin() + pos_idx);
#if EDGE_PROPERTY_NUM != 0
    map_remove_art_property((void*)property_map, pos_idx, size);
#endif
    size -= 1;
    leaf_ops_stats().remove_16.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf16::leaf_check() const {}

inline void ARTLeaf16::copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                                    ARTLeaf* dst, uint16_t dst_idx) const {
    auto cur_dst_idx = dst_idx;
    uint64_t elems = (end_idx > begin_idx) ? (end_idx - begin_idx) : 0;
    switch (dst->depth + dst->is_single_byte) {
        case 0: case 1: {
            auto dl = static_cast<ARTLeaf32*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++, cur_dst_idx++)
                dl->value->at(cur_dst_idx) = (value->at(i) | key.key) & 0xFFFFFFFF;
            break;
        }
        case 2: {
            auto dl = static_cast<ARTLeaf16*>(dst);
            std::copy(value->begin() + begin_idx, value->begin() + end_idx,
                      dl->value->begin() + dst_idx);
            break;
        }
        case 3: {
            auto dl = static_cast<ARTLeaf8*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++)
                dl->value.set(value->at(i) & 0xFF);
            break;
        }
        default: assert(false);
    }
    leaf_ops_stats().copy_bytes_crossed.fetch_add(elems * 2, std::memory_order_relaxed);
#if EDGE_PROPERTY_NUM != 0
    if (dst->property_map)
        art_property_map_copy((void*)property_map, begin_idx, end_idx,
                              dst->property_map, dst_idx);
#endif
}

inline void ARTLeaf16::append_from_list(RangeElement* elem_list, Property_t** prop_list,
                                        uint16_t count) {
    assert(count + size <= ART_LEAF_SIZE);
    for (int i = 0; i < count; i++) {
        value->at(size + i) = elem_list[i] & 0xFFFF;
#if EDGE_PROPERTY_NUM > 1
        map_set_sa_art_property((void*)property_map, size + i, prop_list[i]);
#endif
    }
    uint16_t old_size = size;
    size += count;
    // Post-condition: sorted order within leaf
    for (uint16_t i = old_size + 1; i < size; i++)
        assert(value->at(i) >= value->at(i - 1));
}

// ──────────────── ARTLeaf32 ────────────────
inline ARTLeaf32::ARTLeaf32(ARTKey key, uint8_t depth, bool is_single_byte)
    : ARTLeaf(key, depth, is_single_byte), value() {}

inline uint64_t ARTLeaf32::at(uint16_t pos_idx) const {
    assert(pos_idx < ART_LEAF_SIZE);
    return value->at(pos_idx) | key.key;
}

inline bool ARTLeaf32::has_element(uint64_t element, uint8_t begin_idx) const {
    uint32_t target = element & 0xFFFFFFFF;
    auto iter = std::lower_bound(value->begin() + begin_idx, value->begin() + size, target)
                - value->begin();
    return iter != size && value->at(iter) == target;
}

inline uint16_t ARTLeaf32::get_byte_num(uint8_t depth) const {
    assert(depth <= 4);
    uint16_t cur_diff = 1;
    uint8_t cur_byte = get_key_byte(value->at(0), depth);
    for (uint64_t i = 1; i < size; i++) {
        uint8_t b = get_key_byte(value->at(i), depth);
        if (b != cur_byte) { cur_byte = b; cur_diff++; }
    }
    return cur_diff;
}

inline void ARTLeaf32::insert(uint64_t element, Property_t* property, uint16_t pos_idx) {
    assert(pos_idx <= ART_LEAF_SIZE);
    uint32_t target = element & 0xFFFFFFFF;
    std::copy_backward(value->begin() + pos_idx, value->begin() + size,
                       value->begin() + size + 1);
    value->at(pos_idx) = target;
#if EDGE_PROPERTY_NUM != 0
    map_insert_art_property((void*)property_map, pos_idx, size, property);
#endif
    size += 1;
    leaf_ops_stats().insert_32.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf32::remove(uint16_t pos_idx, uint8_t) {
    assert(pos_idx < ART_LEAF_SIZE);
    std::copy(value->begin() + pos_idx + 1, value->begin() + size,
              value->begin() + pos_idx);
#if EDGE_PROPERTY_NUM != 0
    map_remove_art_property((void*)property_map, pos_idx, size);
#endif
    size -= 1;
    leaf_ops_stats().remove_32.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf32::leaf_check() const {}

inline void ARTLeaf32::copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                                    ARTLeaf* dst, uint16_t dst_idx) const {
    auto cur_dst_idx = dst_idx;
    uint64_t elems = (end_idx > begin_idx) ? (end_idx - begin_idx) : 0;
#if COMPRESSION_ENABLE != 0
    switch (dst->depth + dst->is_single_byte) {
        case 0: case 1: {
            auto dl = static_cast<ARTLeaf32*>(dst);
            std::copy(value->begin() + begin_idx, value->begin() + end_idx,
                      dl->value->begin() + dst_idx);
            break;
        }
        case 2: {
            auto dl = static_cast<ARTLeaf16*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++, cur_dst_idx++)
                dl->value->at(cur_dst_idx) = value->at(i) & 0xFFFF;
            break;
        }
        case 3: {
            auto dl = static_cast<ARTLeaf8*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++)
                dl->value.set(value->at(i) & 0xFF);
            break;
        }
        default: assert(false);
    }
#else
    std::copy(value->begin() + begin_idx, value->begin() + end_idx,
              static_cast<ARTLeaf32*>(dst)->value->begin() + dst_idx);
#endif
    leaf_ops_stats().copy_bytes_crossed.fetch_add(elems * 4, std::memory_order_relaxed);
#if EDGE_PROPERTY_NUM != 0
    if (dst->property_map)
        art_property_map_copy((void*)property_map, begin_idx, end_idx,
                              dst->property_map, dst_idx);
#endif
}

inline void ARTLeaf32::append_from_list(RangeElement* elem_list, Property_t** prop_list,
                                        uint16_t count) {
    assert(count + size <= ART_LEAF_SIZE);
    for (int i = 0; i < count; i++) {
        value->at(size + i) = elem_list[i] & 0xFFFFFFFF;
#if EDGE_PROPERTY_NUM > 1
        map_set_sa_art_property((void*)property_map, size + i, prop_list[i]);
#endif
    }
    size += count;
}

// ──────────────── ARTLeaf64 ────────────────
inline ARTLeaf64::ARTLeaf64(ARTKey key, uint8_t depth, bool is_single_byte)
    : ARTLeaf(key, depth, is_single_byte), value() {}

inline uint64_t ARTLeaf64::at(uint16_t pos_idx) const {
    assert(pos_idx < ART_LEAF_SIZE);
    return value->at(pos_idx) | key.key;
}

inline bool ARTLeaf64::has_element(uint64_t element, uint8_t begin_idx) const {
    auto iter = std::lower_bound(value->begin() + begin_idx, value->begin() + size, element)
                - value->begin();
    return iter != size && value->at(iter) == element;
}

inline uint16_t ARTLeaf64::get_byte_num(uint8_t depth) const {
    assert(depth <= 4);
    uint16_t cur_diff = 1;
    uint8_t cur_byte = get_key_byte(value->at(0), depth);
    for (uint64_t i = 1; i < size; i++) {
        uint8_t b = get_key_byte(value->at(i), depth);
        if (b != cur_byte) { cur_byte = b; cur_diff++; }
    }
    return cur_diff;
}

inline void ARTLeaf64::insert(uint64_t element, Property_t* property, uint16_t pos_idx) {
    assert(pos_idx <= ART_LEAF_SIZE);
    std::copy_backward(value->begin() + pos_idx, value->begin() + size,
                       value->begin() + size + 1);
    value->at(pos_idx) = element;
#if EDGE_PROPERTY_NUM != 0
    map_insert_art_property((void*)property_map, pos_idx, size, property);
#endif
    size += 1;
    leaf_ops_stats().insert_64.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf64::remove(uint16_t pos_idx, uint8_t) {
    assert(pos_idx < ART_LEAF_SIZE);
    std::copy(value->begin() + pos_idx + 1, value->begin() + size,
              value->begin() + pos_idx);
#if EDGE_PROPERTY_NUM != 0
    map_remove_art_property((void*)property_map, pos_idx, size);
#endif
    size -= 1;
    leaf_ops_stats().remove_64.fetch_add(1, std::memory_order_relaxed);
}

inline void ARTLeaf64::leaf_check() const {
    int cnt = 0;
    if (size > 0 && value->at(0) == 0) cnt += 1;
    for (int i = 0; i < ART_LEAF_SIZE; i++)
        if (value->at(i)) cnt += 1;
    assert(cnt == size);
}

inline void ARTLeaf64::copy_to_leaf(uint16_t begin_idx, uint16_t end_idx,
                                    ARTLeaf* dst, uint16_t dst_idx) const {
    auto cur_dst_idx = dst_idx;
    uint64_t elems = (end_idx > begin_idx) ? (end_idx - begin_idx) : 0;
#if COMPRESSION_ENABLE != 0
    switch (dst->depth + dst->is_single_byte) {
        case 0: case 1: {
            auto dl = static_cast<ARTLeaf32*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++, cur_dst_idx++)
                dl->value->at(cur_dst_idx) = value->at(i) & 0xFFFFFFFF;
            break;
        }
        case 2: {
            auto dl = static_cast<ARTLeaf16*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++, cur_dst_idx++)
                dl->value->at(cur_dst_idx) = value->at(i) & 0xFFFF;
            break;
        }
        case 3: {
            auto dl = static_cast<ARTLeaf8*>(dst);
            for (uint16_t i = begin_idx; i < end_idx; i++)
                dl->value.set(value->at(i) & 0xFF);
            break;
        }
        default: assert(false);
    }
#else
    std::copy(value->begin() + begin_idx, value->begin() + end_idx,
              static_cast<ARTLeaf64*>(dst)->value->begin() + dst_idx);
#endif
    leaf_ops_stats().copy_bytes_crossed.fetch_add(elems * 8, std::memory_order_relaxed);
#if EDGE_PROPERTY_NUM != 0
    if (dst->property_map)
        art_property_map_copy((void*)property_map, begin_idx, end_idx,
                              (void*)dst->property_map, dst_idx);
#endif
}

inline void ARTLeaf64::append_from_list(RangeElement* elem_list, Property_t** prop_list,
                                        uint16_t count) {
    assert(count + size <= ART_LEAF_SIZE);
    std::copy(elem_list, elem_list + count, value->begin() + size);
    for (int i = 0; i < count; i++) {
#if EDGE_PROPERTY_NUM > 1
        map_set_sa_art_property((void*)property_map, size + i, prop_list[i]);
#endif
    }
    size += count;
}

// ──────────────── alloc_leaf (upstream dispatch + stats) ────────────────
#define LEAF8  1
#define LEAF16 2
#define LEAF32 3
#define LEAF64 4

inline ARTLeaf* alloc_leaf(ARTKey key, uint8_t depth, bool is_single_byte,
                           bool not_empty, WriterTraceBlock* trace_block) {
    ARTLeaf* res = nullptr;
    art_alloc_stats().leaf.fetch_add(1, std::memory_order_relaxed);

#if COMPRESSION_ENABLE != 0
    if (not_empty) {
        switch (depth + is_single_byte) {
            case 0: case 1: {
                auto l = new ARTLeaf32(key, depth, is_single_byte);
                l->value = trace_block->allocate_art_leaf32();
                l->type = LEAF32;
                res = l;
                break;
            }
            case 2: {
                auto l = new ARTLeaf16(key, depth, is_single_byte);
                l->value = new std::array<uint16_t, ART_LEAF_SIZE>();
                std::memset(l->value->data(), 0, ART_LEAF_SIZE * sizeof(uint16_t));
                l->type = LEAF16;
                res = l;
                break;
            }
            case 3: {
                res = new ARTLeaf8(key, depth, is_single_byte);
                res->type = LEAF8;
                break;
            }
            default: throw std::runtime_error("alloc_leaf(): Invalid depth");
        }
#if EDGE_PROPERTY_NUM != 0
        force_pointer_set(&res->property_map, trace_block->allocate_art_prop_vec());
#endif
    } else {
        switch (depth + is_single_byte) {
            case 0: case 1:
                res = new ARTLeaf32(key, depth, is_single_byte);
                res->type = LEAF32; break;
            case 2:
                res = new ARTLeaf16(key, depth, is_single_byte);
                res->type = LEAF16; break;
            case 3:
                res = new ARTLeaf8(key, depth, is_single_byte);
                res->type = LEAF8; break;
            default: throw std::runtime_error("alloc_leaf(): Invalid depth");
        }
    }
#else
    if (not_empty) {
        auto l = new ARTLeaf32(key, depth, is_single_byte);
        l->value = trace_block->allocate_art_leaf32();
        l->type = LEAF32;
        res = l;
#if EDGE_PROPERTY_NUM != 0
        force_pointer_set(&res->property_map, trace_block->allocate_art_prop_vec());
#endif
    } else {
        res = new ARTLeaf32(key, depth, is_single_byte);
        res->type = LEAF32;
    }
#endif
    return res;
}

// ──────────────── leaf_destroy (upstream) ────────────────
inline void leaf_destroy(ARTLeaf* leaf) {
    art_alloc_stats().destroy_calls.fetch_add(1, std::memory_order_relaxed);
#if COMPRESSION_ENABLE != 0
    switch (leaf->depth + leaf->is_single_byte) {
        case 0: case 1:
            delete static_cast<ARTLeaf32*>(leaf)->value;
#if EDGE_PROPERTY_NUM != 0
            delete static_cast<ARTLeaf32*>(leaf)->property_map;
#endif
            break;
        case 2:
            delete static_cast<ARTLeaf16*>(leaf)->value;
#if EDGE_PROPERTY_NUM != 0
            delete static_cast<ARTLeaf16*>(leaf)->property_map;
#endif
            break;
        case 3:
#if EDGE_PROPERTY_NUM != 0
            delete static_cast<ARTLeaf8*>(leaf)->property_map;
#endif
            break;
        default: throw std::runtime_error("leaf_destroy(): Invalid depth");
    }
#else
    delete static_cast<ARTLeaf32*>(leaf)->value;
#if EDGE_PROPERTY_NUM != 0
    delete static_cast<ARTLeaf32*>(leaf)->property_map;
#endif
#endif
}

// ──────────────── Utility (upstream) ────────────────
inline uint64_t get_list_byte_num(uint64_t* list, uint64_t sz, uint8_t depth) {
    uint16_t cur_diff = 1;
    uint8_t cur_byte = get_key_byte(list[0], depth);
    for (uint64_t i = 1; i < sz; i++) {
        uint8_t b = get_key_byte(list[i], depth);
        if (b != cur_byte) { cur_byte = b; cur_diff++; }
    }
    return cur_diff;
}

// ─── Global dump ───
inline void dump_all_leaf_stats() {
    leaf_ops_stats().dump();
    art_alloc_stats().dump();
}

} // namespace container
