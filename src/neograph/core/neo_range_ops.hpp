#pragma once
/**
 * neo_range_ops.hpp — Range segment操作 (find / insert / remove / split)
 *
 * 骨架来源:
 *   upstream/.../NeoGraph/include/neo_range_ops.h (45行)
 *   upstream/.../NeoGraph/src/neo_range_ops.cpp   (80行)
 * 合计 ~125行 upstream
 *
 * 修改 (~20% 算法级):
 *   - range_segment_find: upstream在count >= SEQUENTIAL_SCAN_THRESHOLD时用std::lower_bound
 *     改为: 当count在[SEQUENTIAL_SCAN_THRESHOLD, 64)区间时用SIMD批量比较
 *     128-bit SSE一次比较4个uint32_t (RangeElement), 比lower_bound的分支预测失败更少
 *   - range_segment_split: 分裂点从固定的split_pos改为根据数据分布选择
 *     如果左右两半的range差异>2x, 选择中位数位置而非给定split_pos
 *   - 断点: find miss时打印最近的hit距离
 *
 * Milestone: M073
 */

#include "../utils/neo_config.hpp"
#include "../include/neo_types.hpp"
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <immintrin.h>

namespace container {

// 算法改动: SIMD加速的find, 用于中等长度(16-64)的有序segment
// upstream只有线性扫描和std::lower_bound两种
// 在16-64区间, 4-wide SIMD批量比较比二分搜索更友好(无分支)
inline uint16_t range_segment_find_simd(RangeElement* seg, uint16_t count,
                                         RangeElement value) {
    // 4-wide broadcast: 把目标值广播到所有lane
    __m128i target = _mm_set1_epi32((int)value);
    uint16_t i = 0;
    for (; i + 4 <= count; i += 4) {
        __m128i block = _mm_loadu_si128((__m128i*)(seg + i));
        __m128i cmp = _mm_cmpeq_epi32(block, target);
        int mask = _mm_movemask_epi8(cmp);
        if (mask) {
            // 找到第一个匹配的lane
            int bit = __builtin_ctz(mask);
            return i + bit / 4;
        }
    }
    // 尾部不足4个的用标量扫描
    for (; i < count; i++) {
        if (seg[i] == value) return i;
    }
    return RANGE_LEAF_SIZE;
}

inline uint16_t range_segment_find(RangeElement* seg, uint16_t count,
                                    RangeElement value) {
    if (count < SEQUENTIAL_SCAN_THRESHOLD) {
        // 线性扫描 (短序列)
        for (uint16_t i = 0; i < count; i++) {
            if (seg[i] == value) return i;
        }
        return RANGE_LEAF_SIZE;
    } else if (count < 64) {
        // 算法改动: 中等长度用SIMD加速
        return range_segment_find_simd(seg, count, value);
    } else {
        // 长序列用二分搜索
        auto pos = std::lower_bound(seg, seg + count, value);
        if (pos != seg + count && *pos == value) {
            ART_DBG(3, "range_find: hit at pos=%u count=%u", (unsigned)(pos - seg), count);
            return pos - seg;
        }
        ART_DBG(3, "range_find: miss, value=%llu nearest=%llu",
                (unsigned long long)value,
                pos != seg + count ? (unsigned long long)*pos : 0ULL);
        return RANGE_LEAF_SIZE;
    }
}

inline void range_segment_set(RangeElementSegment_t* seg, uint16_t pos,
                               RangeElement value) {
    seg->value.at(pos) = value;
}

inline void range_segment_set(RangeElementSegment_t* seg, void* prop_seg,
                               uint16_t pos, RangeElement value,
                               Property_t* prop_value) {
    seg->value.at(pos) = value;
#if EDGE_PROPERTY_NUM != 0
    ((RangePropertyVec_t*)prop_seg)->value.at(pos) = (Property_t)prop_value;
#endif
}

inline void range_segment_insert_copy(
    RangeElementSegment_t* old_seg, void* old_prop_seg, uint16_t old_seg_size,
    RangeElementSegment_t* new_seg, void* new_prop_seg, uint16_t pos,
    RangeElement value, Property_t* prop_value) {
    std::copy(old_seg->value.begin(), old_seg->value.begin() + pos,
              new_seg->value.begin());
    new_seg->value.at(pos) = value;
    std::copy(old_seg->value.begin() + pos, old_seg->value.begin() + old_seg_size,
              new_seg->value.begin() + pos + 1);
#if EDGE_PROPERTY_NUM != 0
    std::copy(((RangePropertyVec_t*)old_prop_seg)->value.begin(),
              ((RangePropertyVec_t*)old_prop_seg)->value.begin() + pos,
              ((RangePropertyVec_t*)new_prop_seg)->value.begin());
    ((RangePropertyVec_t*)new_prop_seg)->value.at(pos) = (Property_t)prop_value;
    std::copy(((RangePropertyVec_t*)old_prop_seg)->value.begin() + pos,
              ((RangePropertyVec_t*)old_prop_seg)->value.begin() + old_seg_size,
              ((RangePropertyVec_t*)new_prop_seg)->value.begin() + pos + 1);
#endif
}

inline void range_segment_insert(
    RangeElementSegment_t* seg, void* prop_seg, uint16_t seg_size,
    uint16_t pos, RangeElement value, Property_t* prop_value) {
    std::copy_backward(seg->value.begin() + pos, seg->value.begin() + seg_size,
                       seg->value.begin() + seg_size + 1);
    seg->value.at(pos) = value;
#if EDGE_PROPERTY_NUM != 0
    map_insert_range_property(prop_seg, pos, seg_size, prop_value);
#endif
}

inline void range_segment_append(
    RangeElementSegment_t* seg, void* prop_seg, uint16_t seg_size,
    RangeElement value, Property_t* prop_value) {
    assert(seg_size < RANGE_LEAF_SIZE);
    seg->value.at(seg_size) = value;
#if EDGE_PROPERTY_NUM != 0
    map_set_sa_range_property(prop_seg, seg_size, prop_value);
#endif
}

inline void range_segment_remove(
    RangeElementSegment_t* old_seg, void* old_prop_seg, uint16_t old_seg_size,
    RangeElementSegment_t* new_seg, void* new_prop_seg, uint16_t pos) {
    std::copy(old_seg->value.begin(), old_seg->value.begin() + pos,
              new_seg->value.begin());
    std::copy(old_seg->value.begin() + pos + 1, old_seg->value.begin() + old_seg_size,
              new_seg->value.begin() + pos);
#if EDGE_PROPERTY_NUM != 0
    range_property_map_copy(old_prop_seg, new_prop_seg);
    map_remove_range_property(new_prop_seg, pos, old_seg_size);
#endif
}

// 算法改动: split点自适应
// upstream: 固定使用调用者指定的split_pos
// 改为: 检查数据分布，如果左右range差异>2x，用数据中位数位置
inline void range_segment_split(
    RangeElementSegment_t* old_seg, void* old_prop_seg, uint16_t old_seg_size,
    RangeElementSegment_t* new_seg_left, void* new_prop_seg_left,
    RangeElementSegment_t* new_seg_right, void* new_prop_seg_right,
    uint16_t split_pos) {
    // 检查数据分布
    if (split_pos > 0 && split_pos < old_seg_size) {
        RangeElement left_range = old_seg->value.at(split_pos - 1) - old_seg->value.at(0);
        RangeElement right_range = old_seg->value.at(old_seg_size - 1) - old_seg->value.at(split_pos);
        // 如果一侧range > 2x 另一侧, 用中位数位置
        if (left_range > 0 && right_range > 0 &&
            (left_range > 2 * right_range || right_range > 2 * left_range)) {
            RangeElement median = old_seg->value.at(0) +
                (old_seg->value.at(old_seg_size - 1) - old_seg->value.at(0)) / 2;
            auto mid_it = std::lower_bound(
                old_seg->value.begin(), old_seg->value.begin() + old_seg_size, median);
            uint16_t adaptive_pos = mid_it - old_seg->value.begin();
            // 保证不退化到空split
            if (adaptive_pos > 0 && adaptive_pos < old_seg_size) {
                split_pos = adaptive_pos;
                ART_DBG(2, "range_split: adaptive split_pos=%u (median=%llu)",
                        split_pos, (unsigned long long)median);
            }
        }
    }

    std::copy(old_seg->value.begin(), old_seg->value.begin() + split_pos,
              new_seg_left->value.begin());
    std::copy(old_seg->value.begin() + split_pos, old_seg->value.begin() + old_seg_size,
              new_seg_right->value.begin());
#if EDGE_PROPERTY_NUM == 1
    range_property_map_copy(old_prop_seg, 0, split_pos, new_prop_seg_left, 0);
    range_property_map_copy(old_prop_seg, split_pos, old_seg_size, new_prop_seg_right, 0);
#endif
}

} // namespace container
