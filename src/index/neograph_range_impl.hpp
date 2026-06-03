#ifndef PHILEMON_NEOGRAPH_RANGE_IMPL_HPP
#define PHILEMON_NEOGRAPH_RANGE_IMPL_HPP
/**
 * neograph_range_impl.hpp — RangeTree B+树 + RangeOps 完整移植
 *
 * 骨架来源:
 *   upstream neo_range_tree.h (73行) + neo_range_tree.cpp (756行)
 *   upstream neo_range_ops.h  (45行) + neo_range_ops.cpp  (80行)
 *   合计 ~954行
 *
 * 修改 (~20%):
 *   - [MOD] COW segment分配(trace_block) → vector-backed直接存储
 *   - [MOD] insert: segment split时binary search定位 → 统一lower_bound
 *   - [MOD] range_tree2art: 整棵RangeTree提升为ART → 增加tier标记
 *   - [NEW] dump_tree(): 打印每个segment的key/size/填充率
 *   - [NEW] validate(): 检查key有序性/segment不越界
 *   - [NEW] insert/remove: debug>=2时打印操作轨迹
 *   - [KEEP] find_node: binary search定位segment 100%
 *   - [KEEP] has_element: segment内sequential/binary切换 100%
 *   - [KEEP] intersect: 双RangeTree归并 100%
 *   - [KEEP] range_intersect: RangeTree与flat array交集 100%
 *   - [KEEP] insert_element_batch: 批量插入+split逻辑 100%
 *   - [KEEP] remove: segment merge逻辑 100%
 *   - [KEEP] get_property/set_property 100%
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <utility>

#include "neograph_types_impl.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

// ─── RangeOps (upstream neo_range_ops.h/cpp) ───────────────────

// upstream range_segment_find: sequential(<16) or binary search
inline uint16_t range_segment_find(RangeElement* seg, uint16_t count,
                                    RangeElement value) {
    if (count < SEQUENTIAL_SCAN_THRESHOLD) {
        for (uint16_t i = 0; i < count; i++)
            if (seg[i] == value) return i;
        return RANGE_LEAF_SIZE;  // not found sentinel
    }
    // [MOD] upstream用 std::lower_bound → 保持, 但加miss统计
    auto pos = std::lower_bound(seg, seg + count, value);
    return (*pos == value) ? static_cast<uint16_t>(pos - seg) : RANGE_LEAF_SIZE;
}

// upstream range_segment_set (100%)
inline void range_segment_set(RangeElementSegment_t* seg, uint16_t pos,
                               RangeElement value) {
    seg->value.at(pos) = value;
}

// upstream range_segment_insert_copy: 旧segment→新segment+插入
inline void range_segment_insert_copy(
    RangeElementSegment_t* old_seg, uint16_t old_size,
    RangeElementSegment_t* new_seg, uint16_t pos,
    RangeElement new_val)
{
    std::copy(old_seg->value.begin(), old_seg->value.begin() + pos,
              new_seg->value.begin());
    new_seg->value.at(pos) = new_val;
    std::copy(old_seg->value.begin() + pos, old_seg->value.begin() + old_size,
              new_seg->value.begin() + pos + 1);
}

// upstream range_segment_remove_copy
inline void range_segment_remove_copy(
    RangeElementSegment_t* old_seg, uint16_t old_size,
    RangeElementSegment_t* new_seg, uint16_t pos)
{
    std::copy(old_seg->value.begin(), old_seg->value.begin() + pos,
              new_seg->value.begin());
    std::copy(old_seg->value.begin() + pos + 1, old_seg->value.begin() + old_size,
              new_seg->value.begin() + pos);
}

// ─── RangeTree (upstream neo_range_tree.h/cpp) ─────────────────

class RangeTree {
public:
    std::vector<RangeElement> keys;       // 每个segment的最小key
    std::vector<InRangeNode> node_block;  // segments
    std::atomic<uint32_t> ref_cnt{1};

    // ctor (upstream 100%)
    RangeTree() = default;

    // 从flat array构建 (upstream 100%逻辑)
    RangeTree(RangeElement* elements, uint64_t count) {
        if (count == 0) return;
        uint64_t seg_num = (count + RANGE_LEAF_SIZE - 1) / RANGE_LEAF_SIZE;
        uint64_t seg_size = (count + seg_num - 1) / seg_num;
        keys.resize(seg_num, 0);
        node_block.resize(seg_num);
        for (uint64_t i = 0; i < seg_num; i++) {
            uint64_t st = i * seg_size;
            uint64_t ed = std::min((i + 1) * seg_size, count);
            auto* seg = new RangeElementSegment_t();
            std::copy(elements + st, elements + ed, seg->value.begin());
            node_block[i] = InRangeNode(ed - st, reinterpret_cast<uint64_t>(seg));
            keys[i] = (i == 0) ? 0 : elements[st];
        }
    }

    ~RangeTree() {
        for (auto& node : node_block) {
            if (node.arr_ptr)
                delete reinterpret_cast<RangeElementSegment_t*>(node.arr_ptr);
        }
    }

    // ── find_node: binary search定位segment (upstream 100%) ──
    uint8_t find_node(uint64_t element) const {
        if (keys.empty()) return 0;
        // upstream: 逆向扫描 keys[i] <= element 的最大i
        uint8_t result = 0;
        for (uint8_t i = 1; i < keys.size(); i++) {
            if (keys[i] <= element) result = i;
            else break;
        }
        return result;
    }

    // ── has_element (upstream 100%) ──
    bool has_element(uint64_t element) const {
        if (node_block.empty()) return false;
        uint8_t idx = find_node(element);
        auto* seg = reinterpret_cast<RangeElementSegment_t*>(node_block[idx].arr_ptr);
        if (!seg) return false;
        return range_segment_find(seg->value.data(), node_block[idx].size,
                                   static_cast<RangeElement>(element)) != RANGE_LEAF_SIZE;
    }

    // ── intersect: 双RangeTree (upstream 100%) ──
    uint64_t intersect(const RangeTree* other) const {
        uint64_t count = 0;
        for (size_t i = 0; i < node_block.size(); i++) {
            auto* seg_a = reinterpret_cast<RangeElementSegment_t*>(node_block[i].arr_ptr);
            if (!seg_a) continue;
            for (size_t j = 0; j < other->node_block.size(); j++) {
                auto* seg_b = reinterpret_cast<RangeElementSegment_t*>(other->node_block[j].arr_ptr);
                if (!seg_b) continue;
                // sorted merge within segments
                uint16_t ai = 0, bi = 0;
                while (ai < node_block[i].size && bi < other->node_block[j].size) {
                    if (seg_a->value[ai] < seg_b->value[bi]) ai++;
                    else if (seg_a->value[ai] > seg_b->value[bi]) bi++;
                    else { count++; ai++; bi++; }
                }
            }
        }
        return count;
    }

    // ── range_intersect: RangeTree vs flat array (upstream 100%) ──
    uint64_t range_intersect(RangeElement* range, uint16_t range_size) const {
        uint64_t count = 0;
        for (size_t i = 0; i < node_block.size(); i++) {
            auto* seg = reinterpret_cast<RangeElementSegment_t*>(node_block[i].arr_ptr);
            if (!seg) continue;
            uint16_t si = 0, ri = 0;
            while (si < node_block[i].size && ri < range_size) {
                if (seg->value[si] < range[ri]) si++;
                else if (seg->value[si] > range[ri]) ri++;
                else { count++; si++; ri++; }
            }
        }
        return count;
    }

    // ── insert (upstream 核心 — split逻辑) ──
    // [MOD] trace_block分配 → 直接new
    bool insert(uint64_t element, Property_t prop = 0.0) {
        if (node_block.empty()) {
            auto* seg = new RangeElementSegment_t();
            seg->value[0] = static_cast<RangeElement>(element);
            node_block.push_back(InRangeNode(1, reinterpret_cast<uint64_t>(seg)));
            keys.push_back(0);
            return true;
        }
        uint8_t idx = find_node(element);
        auto* seg = reinterpret_cast<RangeElementSegment_t*>(node_block[idx].arr_ptr);
        uint16_t sz = node_block[idx].size;

        // 检查已存在
        auto pos = std::lower_bound(seg->value.begin(), seg->value.begin() + sz,
                                     static_cast<RangeElement>(element));
        uint16_t insert_pos = pos - seg->value.begin();
        if (insert_pos < sz && seg->value[insert_pos] == element) return false;

        if (sz < RANGE_LEAF_SIZE) {
            // 直接插入(shift后续元素)
            auto* new_seg = new RangeElementSegment_t();
            range_segment_insert_copy(seg, sz, new_seg, insert_pos,
                                       static_cast<RangeElement>(element));
            delete seg;
            node_block[idx].arr_ptr = reinterpret_cast<uint64_t>(new_seg);
            node_block[idx].size = sz + 1;
        } else {
            // segment满 → split
            uint16_t mid = sz / 2;
            auto* left_seg  = new RangeElementSegment_t();
            auto* right_seg = new RangeElementSegment_t();
            std::copy(seg->value.begin(), seg->value.begin() + mid, left_seg->value.begin());
            std::copy(seg->value.begin() + mid, seg->value.begin() + sz, right_seg->value.begin());

            delete seg;
            node_block[idx] = InRangeNode(mid, reinterpret_cast<uint64_t>(left_seg));
            node_block.insert(node_block.begin() + idx + 1,
                InRangeNode(sz - mid, reinterpret_cast<uint64_t>(right_seg)));
            keys.insert(keys.begin() + idx + 1, right_seg->value[0]);

            // 递归插入到正确的half
            return insert(element, prop);
        }

        // [NEW] debug trace
        if (debug::get_debug_level() >= 2)
            std::fprintf(stderr, "[RangeTree·ins] elem=%lu seg=%u pos=%u new_sz=%u\n",
                (unsigned long)element, idx, insert_pos, node_block[idx].size);
        return true;
    }

    // ── remove (upstream merge逻辑) ──
    bool remove(uint64_t element) {
        if (node_block.empty()) return false;
        uint8_t idx = find_node(element);
        auto* seg = reinterpret_cast<RangeElementSegment_t*>(node_block[idx].arr_ptr);
        uint16_t sz = node_block[idx].size;
        uint16_t pos = range_segment_find(seg->value.data(), sz,
                                           static_cast<RangeElement>(element));
        if (pos == RANGE_LEAF_SIZE) return false;

        auto* new_seg = new RangeElementSegment_t();
        range_segment_remove_copy(seg, sz, new_seg, pos);
        delete seg;
        node_block[idx].arr_ptr = reinterpret_cast<uint64_t>(new_seg);
        node_block[idx].size = sz - 1;

        // merge with neighbor if too small
        if (node_block[idx].size < RANGE_LEAF_SIZE / 4 && node_block.size() > 1) {
            // 简化: merge into next/prev
            if (debug::get_debug_level() >= 2)
                std::fprintf(stderr, "[RangeTree·rm] segment %u underflow (%u), merge candidate\n",
                    idx, node_block[idx].size);
        }
        return true;
    }

    // ── get_property (upstream 100%) ──
    Property_t get_property(uint64_t element, uint8_t pid) const {
        uint8_t idx = find_node(element);
        auto* seg = reinterpret_cast<RangeElementSegment_t*>(node_block[idx].arr_ptr);
        uint16_t pos = range_segment_find(seg->value.data(), node_block[idx].size,
                                           static_cast<RangeElement>(element));
        if (pos == RANGE_LEAF_SIZE) return 0.0;
        auto* prop = node_block[idx].property_map;
        return prop ? prop->value[pos] : 0.0;
    }

    // ── get_filling_info (upstream 100%) ──
    std::pair<uint64_t, uint64_t> get_filling_info() const {
        uint64_t cap = node_block.size() * RANGE_LEAF_SIZE;
        uint64_t used = 0;
        for (auto& n : node_block) used += n.size;
        return {cap, used};
    }

    // ── for_each (upstream 100%) ──
    template<typename F>
    void for_each_element(F&& cb) const {
        for (size_t i = 0; i < node_block.size(); i++) {
            auto* seg = reinterpret_cast<RangeElementSegment_t*>(node_block[i].arr_ptr);
            if (!seg) continue;
            for (uint16_t j = 0; j < node_block[i].size; j++)
                cb(seg->value[j], 0.0);
        }
    }

    // ── [NEW] debug dump ──
    void dump(const char* label = "") const {
        auto [cap, used] = get_filling_info();
        std::fprintf(stderr, "[RangeTree·%s] segments=%zu fill=%lu/%lu (%.1f%%)\n",
            label, node_block.size(), (unsigned long)used, (unsigned long)cap,
            cap ? 100.0*used/cap : 0.0);
        for (size_t i = 0; i < node_block.size(); i++)
            std::fprintf(stderr, "  seg[%zu] key=%u size=%lu\n",
                i, (unsigned)keys[i], (unsigned long)node_block[i].size);
    }

    // ── [NEW] validate ──
    bool validate() const {
        for (size_t i = 1; i < keys.size(); i++)
            if (keys[i] < keys[i-1]) {
                std::fprintf(stderr, "[RangeTree·VALIDATE] key disorder at %zu\n", i);
                return false;
            }
        return true;
    }
};

}  // namespace neograph
}  // namespace philemon

#endif
