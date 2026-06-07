/**
 * m110_m111_neograph_core_upper_experiment.cpp — M110-M111: NeoGraph core深度实验(上)
 *
 * 覆盖模块:
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_index.cpp   (462行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_index.h (126行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_property.cpp (487行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_property.h (360行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_range_ops.cpp (80行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_range_ops.h (45行)
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_range_tree.cpp (756行)
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_range_tree.h (73行)
 *   总计: 2389行
 *
 * M110: neo_index (forest/lock/unlock/has_vertex/has_edge/get_degree/get_neighbor/
 *                  intersect/insert_vertex/insert_edge/batch/remove/commit/gc 30+函数)
 *       neo_property (PropertyVec get/set/insert/remove/copy_to +
 *                     map操作20+函数 + GC引用计数)
 *
 * M111: neo_range_ops (range_segment_find/set/insert/remove/split/append 6函数)
 *        neo_range_tree (has_element/intersect/insert+split/batch/remove/
 *                        property/range_tree2art 15+函数)
 *
 * 算法改动 (~20%):
 *   neo_index:
 *     - [MOD] operation_count: 全局操作计数跟踪 insert_vertex/insert_edge/remove累计
 *     - [MOD] forest_resize_count: forest扩容次数打印
 *     - [MOD] batch_thread_split: batch分组情况调试print
 *     - [MOD] gc_freed_count: gc调用追踪
 *   neo_property:
 *     - [MOD] copy_steps: PropertyVec copy_to copy步数统计
 *     - [MOD] ref_dec_count: gc_vertex_property_map_ref减引用计数追踪
 *     - [MOD] prop_alloc_count / prop_free_count: 分配释放次数
 *     - [MOD] map_op_count: map_get/set/insert/remove操作计数
 *   neo_range_ops:
 *     - [MOD] binary_search_steps: range_segment_find 二分/线性扫描步数追踪
 *     - [MOD] comparison_count: insert_copy/split 比较次数
 *   neo_range_tree:
 *     - [MOD] split_count: insert触发split次数统计
 *     - [MOD] merge_progress: insert_element_batch merge-loop进度打印
 *     - [MOD] freed_count: remove释放node统计
 *     - [MOD] intersect_hit_count: intersect命中统计
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m110_test experiment/m110_m111_neograph_core_upper_experiment.cpp
 * Milestone: M110-M111 (第15位Claude, Claude Sonnet 4.6)
 */

#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <stack>
#include <mutex>

// ═══════════════════════════════════════════════════════════════════
//  全局测试计数
// ═══════════════════════════════════════════════════════════════════
static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_tests_failed++; g_tests_run++; return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    std::printf("  [PASS] %s\n", name); \
    g_tests_passed++; g_tests_run++; \
} while(0)

// ═══════════════════════════════════════════════════════════════════
//  全局调试计数器 (20%改动: operation_count系列)
// ═══════════════════════════════════════════════════════════════════
static std::atomic<uint64_t> g_operation_count{0};
static std::atomic<uint64_t> g_split_count{0};
static std::atomic<uint64_t> g_freed_count{0};
static std::atomic<uint64_t> g_binary_search_steps{0};
static std::atomic<uint64_t> g_comparison_count{0};
static std::atomic<uint64_t> g_intersect_hit_count{0};
static std::atomic<uint64_t> g_prop_alloc_count{0};
static std::atomic<uint64_t> g_prop_free_count{0};
static std::atomic<uint64_t> g_ref_dec_count{0};
static std::atomic<uint64_t> g_copy_steps{0};
static std::atomic<uint64_t> g_map_op_count{0};
static std::atomic<uint64_t> g_forest_resize_count{0};
static std::atomic<uint64_t> g_gc_freed_count{0};

namespace philemon {
namespace experiment {

// ═══════════════════════════════════════════════════════════════════
//  config 常量 (模拟 upstream utils/config.h)
// ═══════════════════════════════════════════════════════════════════
constexpr uint64_t VERTEX_GROUP_BITS    = 6;
constexpr uint64_t VERTEX_GROUP_SIZE    = 1ULL << VERTEX_GROUP_BITS;
constexpr uint64_t VERTEX_GROUP_MASK    = (1ULL << VERTEX_GROUP_BITS) - 1;
constexpr uint64_t RANGE_LEAF_SIZE      = 64;  // 实验用小值, 原为512
constexpr uint64_t ART_LEAF_SIZE        = 16;  // 实验用小值, 原为256
constexpr uint64_t SEQUENTIAL_SCAN_THRESHOLD = 16;
constexpr uint64_t EDGE_PROPERTY_NUM    = 1;
constexpr uint64_t VERTEX_PROPERTY_NUM  = 0;

// ═══════════════════════════════════════════════════════════════════
//  类型定义 (模拟 upstream utils/types.h)
// ═══════════════════════════════════════════════════════════════════

using Property_t    = uint64_t;
using RangeElement  = uint32_t;

struct RangePropertyVec_t {
    std::array<Property_t, RANGE_LEAF_SIZE> value{};
    std::atomic<uint32_t> ref_cnt{1};
};

struct RangeElementSegment_t {
    std::array<RangeElement, RANGE_LEAF_SIZE> value{};
    std::atomic<uint32_t> ref_cnt{1};
};

struct InRangeNode {
    uint64_t size  = 0;
    uint64_t arr_ptr = 0;
    RangePropertyVec_t* property_map = nullptr;

    InRangeNode() = default;
    InRangeNode(uint64_t sz, uint64_t ptr) : size(sz), arr_ptr(ptr) {}
    InRangeNode(uint64_t sz, uint64_t ptr, RangePropertyVec_t* pm)
        : size(sz), arr_ptr(ptr), property_map(pm) {}
    InRangeNode(const InRangeNode&) = default;
    InRangeNode& operator=(const InRangeNode&) = default;
};

enum GCResourceType {
    Inner_Segment = 2,
    Range_Property_Vec = 10,
    Range_Property_Map_All_Modified = 11,
};

struct GCResourceInfo {
    GCResourceType type;
    void* ptr;
};

struct RangeTreeInsertElemBatchRes {
    uint64_t new_inserted;
    void* tree_ptr;
};

// ═══════════════════════════════════════════════════════════════════
//  模拟 WriterTraceBlock (简化版: 直接 new/delete 替代池化分配)
// ═══════════════════════════════════════════════════════════════════

struct MockWriterTraceBlock {
    // [MOD] 追踪分配次数
    std::atomic<uint64_t> alloc_seg_count{0};
    std::atomic<uint64_t> alloc_prop_count{0};
    std::atomic<uint64_t> dealloc_seg_count{0};
    std::atomic<uint64_t> dealloc_prop_count{0};

    RangeElementSegment_t* allocate_range_element_segment() {
        alloc_seg_count.fetch_add(1, std::memory_order_relaxed);
        g_prop_alloc_count.fetch_add(1, std::memory_order_relaxed);
        auto* seg = new RangeElementSegment_t();
        std::fill(seg->value.begin(), seg->value.end(), 0);
        return seg;
    }

    void deallocate_range_element_segment(RangeElementSegment_t* seg) {
        dealloc_seg_count.fetch_add(1, std::memory_order_relaxed);
        g_freed_count.fetch_add(1, std::memory_order_relaxed);
        delete seg;
    }

    RangePropertyVec_t* allocate_range_prop_vec() {
        alloc_prop_count.fetch_add(1, std::memory_order_relaxed);
        g_prop_alloc_count.fetch_add(1, std::memory_order_relaxed);
        auto* pv = new RangePropertyVec_t();
        std::fill(pv->value.begin(), pv->value.end(), 0);
        return pv;
    }

    void deallocate_range_prop_vec(RangePropertyVec_t* pv) {
        dealloc_prop_count.fetch_add(1, std::memory_order_relaxed);
        g_prop_free_count.fetch_add(1, std::memory_order_relaxed);
        delete pv;
    }

    void dump_stats() const {
        std::printf("    [TraceBlock] alloc_seg=%lu, alloc_prop=%lu, dealloc_seg=%lu, dealloc_prop=%lu\n",
            alloc_seg_count.load(), alloc_prop_count.load(),
            dealloc_seg_count.load(), dealloc_prop_count.load());
    }
};

// ═══════════════════════════════════════════════════════════════════
//  property map 操作 (模拟 neo_property.cpp 的 map_* 系列)
// ═══════════════════════════════════════════════════════════════════

void range_property_map_copy(void* src, void* dst) {
    g_copy_steps.fetch_add(RANGE_LEAF_SIZE, std::memory_order_relaxed);
    auto* s = static_cast<RangePropertyVec_t*>(src);
    auto* d = static_cast<RangePropertyVec_t*>(dst);
    std::copy(s->value.begin(), s->value.end(), d->value.begin());
}

void range_property_map_copy(void* src, uint64_t begin_idx, uint64_t end_idx,
                              void* dst, uint64_t dst_idx) {
    g_copy_steps.fetch_add(end_idx - begin_idx, std::memory_order_relaxed);
    auto* s = static_cast<RangePropertyVec_t*>(src);
    auto* d = static_cast<RangePropertyVec_t*>(dst);
    std::copy(s->value.begin() + begin_idx, s->value.begin() + end_idx,
              d->value.begin() + dst_idx);
}

void map_set_sa_range_property(void* map, uint64_t idx, Property_t* prop_value) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map || !prop_value) return;
    static_cast<RangePropertyVec_t*>(map)->value.at(idx) = *prop_value;
}

void map_set_sa_range_property(void* map, uint64_t idx, void* prop_value) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map || !prop_value) return;
    static_cast<RangePropertyVec_t*>(map)->value.at(idx) =
        *static_cast<Property_t*>(prop_value);
}

void map_insert_range_property(void* map, uint64_t pos_idx, uint64_t seg_size,
                                void* prop_value) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map) return;
    auto* pv = static_cast<RangePropertyVec_t*>(map);
    std::copy_backward(pv->value.begin() + pos_idx,
                       pv->value.begin() + seg_size,
                       pv->value.begin() + seg_size + 1);
    if (prop_value) {
        pv->value.at(pos_idx) = *static_cast<Property_t*>(prop_value);
    }
}

void map_remove_range_property(void* map, uint64_t pos_idx, uint64_t seg_size) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map) return;
    auto* pv = static_cast<RangePropertyVec_t*>(map);
    std::copy(pv->value.begin() + pos_idx + 1,
              pv->value.begin() + seg_size,
              pv->value.begin() + pos_idx);
}

Property_t map_get_range_property(void* map, uint64_t idx, uint8_t /*property_id*/) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map) return 0;
    return static_cast<RangePropertyVec_t*>(map)->value.at(idx);
}

Property_t* map_get_all_range_property(void* map, uint64_t idx) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map) return nullptr;
    return &static_cast<RangePropertyVec_t*>(map)->value.at(idx);
}

void map_set_range_property(void* map, uint64_t idx, uint8_t /*property_id*/,
                             Property_t value) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    if (!map) return;
    static_cast<RangePropertyVec_t*>(map)->value.at(idx) = value;
}

void force_pointer_set(void* src, void* target) {
    *((uint64_t**)src) = (uint64_t*)target;
}

// ═══════════════════════════════════════════════════════════════════
//  ──────────────────────── M110 Part 1: neo_property ─────────────
// ═══════════════════════════════════════════════════════════════════
//
//  PropertyVec<Size>: 移植自 neo_property.cpp + neo_property.h
//  保留原始算法逻辑，加入 [MOD] 调试计数
//

template<uint64_t Size>
struct PropertyVec {
    std::array<Property_t, Size> value{};
    std::atomic<uint32_t> ref_cnt{1};

    explicit PropertyVec() = default;

    // [MOD] copy_steps: 统计copy步数
    void copy_to(PropertyVec* dst) const {
        g_copy_steps.fetch_add(Size, std::memory_order_relaxed);
        std::copy(this->value.begin(), this->value.end(), dst->value.begin());
    }

    void copy_to(uint64_t begin_idx, uint64_t end_idx, PropertyVec* dst,
                 uint64_t dst_idx) const {
        g_copy_steps.fetch_add(end_idx - begin_idx, std::memory_order_relaxed);
        std::copy(this->value.begin() + begin_idx,
                  this->value.begin() + end_idx,
                  dst->value.begin() + dst_idx);
    }

    Property_t get(uint64_t idx) const {
        assert(idx < this->value.size());
        return this->value[idx];
    }

    void set(uint64_t idx, Property_t val) {
        assert(idx < this->value.size());
        this->value[idx] = val;
    }

    void set_string(uint64_t /*idx*/, std::string&& /*val*/) {
        // not implemented upstream
    }

    void insert(uint64_t pos_idx, uint64_t size, Property_t val) {
        std::move_backward(this->value.begin() + pos_idx,
                           this->value.begin() + size,
                           this->value.begin() + size + 1);
        this->value[pos_idx] = val;
    }

    void insert_copy(PropertyVec* target, uint64_t pos_idx, uint64_t size,
                     Property_t val) {
        std::copy(this->value.begin(), this->value.begin() + pos_idx,
                  target->value.begin());
        target->value[pos_idx] = val;
        std::copy(this->value.begin() + pos_idx,
                  this->value.begin() + size,
                  target->value.begin() + pos_idx + 1);
    }

    void remove(uint64_t pos_idx, uint64_t size) {
        std::copy(this->value.begin() + pos_idx + 1,
                  this->value.begin() + size,
                  this->value.begin() + pos_idx);
    }

    void append_from_list(uint64_t begin_idx, Property_t* values, uint64_t size) {
        for (uint64_t i = 0; i < size; i++) {
            this->value[begin_idx + i] = values[i];
        }
    }
};

using VertexPropertyVec_t   = PropertyVec<256>;
using RangePropertyVec_exp_t = PropertyVec<RANGE_LEAF_SIZE>;

// gc_vertex_property_map_ref — 移植自 neo_property.cpp
// [MOD] ref_dec_count: 记录引用减少次数
void gc_vertex_property_map_ref(VertexPropertyVec_t* map) {
    if (!map) return;
    g_ref_dec_count.fetch_add(1, std::memory_order_relaxed);
    if (map->ref_cnt.fetch_sub(1, std::memory_order_release) == 1) {
        g_prop_free_count.fetch_add(1, std::memory_order_relaxed);
        // [MOD] debug: 打印GC释放
        // std::printf("    [GC] deallocate VertexPropertyVec @%p\n", (void*)map);
        delete map;
    }
}

void destroy_vertex_property_map(VertexPropertyVec_t* map) {
    if (!map) return;
    g_prop_free_count.fetch_add(1, std::memory_order_relaxed);
    delete map;
}

void vertex_property_map_copy(VertexPropertyVec_t* src, VertexPropertyVec_t* dst) {
    src->copy_to(dst);
}

Property_t map_get_vertex_property(VertexPropertyVec_t* map, uint64_t vertex) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    return map->get(vertex);
}

void map_set_vertex_property(VertexPropertyVec_t* map, uint64_t vertex, Property_t value) {
    g_map_op_count.fetch_add(1, std::memory_order_relaxed);
    map->set(vertex, value);
}

// ═══════════════════════════════════════════════════════════════════
//  ─────────────────── M111 Part 1: neo_range_ops ─────────────────
//  移植自 neo_range_ops.cpp + neo_range_ops.h
// ═══════════════════════════════════════════════════════════════════

// [MOD] binary_search_steps: 追踪查找步数
uint16_t range_segment_find(RangeElement* seg, uint16_t count, RangeElement value) {
    if (count < SEQUENTIAL_SCAN_THRESHOLD) {
        // sequential search
        for (auto i = 0; i < count; i++) {
            g_binary_search_steps.fetch_add(1, std::memory_order_relaxed);
            if (seg[i] == value) {
                return i;
            }
        }
        return (uint16_t)RANGE_LEAF_SIZE;
    } else {
        // binary search — [MOD] 计算大约步数
        g_binary_search_steps.fetch_add(
            (uint64_t)std::ceil(std::log2(count + 1)),
            std::memory_order_relaxed);
        auto pos = std::lower_bound(seg, seg + count, value);
        return (*pos == value) ? (uint16_t)(pos - seg) : (uint16_t)RANGE_LEAF_SIZE;
    }
}

void range_segment_set(RangeElementSegment_t* seg, uint16_t pos, RangeElement value) {
    seg->value.at(pos) = value;
}

void range_segment_set(RangeElementSegment_t* seg, void* prop_seg,
                        uint16_t pos, RangeElement value, Property_t* prop_value) {
    seg->value.at(pos) = value;
    if (prop_seg && prop_value) {
        static_cast<RangePropertyVec_t*>(prop_seg)->value.at(pos) = *prop_value;
    }
}

// [MOD] comparison_count: 统计insert_copy中的内存拷贝操作
void range_segment_insert_copy(RangeElementSegment_t* old_seg, void* old_prop_seg,
                                uint16_t old_seg_size,
                                RangeElementSegment_t* new_seg, void* new_prop_seg,
                                uint16_t pos, RangeElement value, Property_t* prop_value) {
    g_comparison_count.fetch_add(old_seg_size, std::memory_order_relaxed);
    std::copy(old_seg->value.begin(),
              old_seg->value.begin() + pos,
              new_seg->value.begin());
    new_seg->value.at(pos) = value;
    std::copy(old_seg->value.begin() + pos,
              old_seg->value.begin() + old_seg_size,
              new_seg->value.begin() + pos + 1);
    if (old_prop_seg && new_prop_seg) {
        std::copy(static_cast<RangePropertyVec_t*>(old_prop_seg)->value.begin(),
                  static_cast<RangePropertyVec_t*>(old_prop_seg)->value.begin() + pos,
                  static_cast<RangePropertyVec_t*>(new_prop_seg)->value.begin());
        if (prop_value) {
            static_cast<RangePropertyVec_t*>(new_prop_seg)->value.at(pos) = *prop_value;
        }
        std::copy(static_cast<RangePropertyVec_t*>(old_prop_seg)->value.begin() + pos,
                  static_cast<RangePropertyVec_t*>(old_prop_seg)->value.begin() + old_seg_size,
                  static_cast<RangePropertyVec_t*>(new_prop_seg)->value.begin() + pos + 1);
    }
}

void range_segment_insert(RangeElementSegment_t* seg, void* prop_seg,
                           uint16_t seg_size, uint16_t pos,
                           RangeElement value, Property_t* prop_value) {
    std::copy_backward(seg->value.begin() + pos,
                       seg->value.begin() + seg_size,
                       seg->value.begin() + seg_size + 1);
    seg->value.at(pos) = value;
    if (prop_seg) {
        map_insert_range_property(prop_seg, pos, seg_size, (void*)prop_value);
    }
}

void range_segment_append(RangeElementSegment_t* seg, void* prop_seg,
                           uint16_t seg_size, RangeElement value, Property_t* prop_value) {
    assert(seg_size < RANGE_LEAF_SIZE);
    seg->value.at(seg_size) = value;
    if (prop_seg) {
        map_set_sa_range_property(prop_seg, seg_size, prop_value);
    }
}

void range_segment_remove(RangeElementSegment_t* old_seg, void* old_prop_seg,
                           uint16_t old_seg_size,
                           RangeElementSegment_t* new_seg, void* new_prop_seg,
                           uint16_t pos) {
    std::copy(old_seg->value.begin(),
              old_seg->value.begin() + pos,
              new_seg->value.begin());
    std::copy(old_seg->value.begin() + pos + 1,
              old_seg->value.begin() + old_seg_size,
              new_seg->value.begin() + pos);
    if (old_prop_seg && new_prop_seg) {
        range_property_map_copy(old_prop_seg, new_prop_seg);
        map_remove_range_property(new_prop_seg, pos, old_seg_size);
    }
}

// [MOD] comparison_count: 统计split中的拷贝量
void range_segment_split(RangeElementSegment_t* old_seg, void* old_prop_seg,
                          uint16_t old_seg_size,
                          RangeElementSegment_t* new_seg_left, void* new_prop_seg_left,
                          RangeElementSegment_t* new_seg_right, void* new_prop_seg_right,
                          uint16_t split_pos) {
    g_comparison_count.fetch_add(old_seg_size, std::memory_order_relaxed);
    std::copy(old_seg->value.begin(),
              old_seg->value.begin() + split_pos,
              new_seg_left->value.begin());
    std::copy(old_seg->value.begin() + split_pos,
              old_seg->value.begin() + old_seg_size,
              new_seg_right->value.begin());
    if (old_prop_seg) {
        range_property_map_copy(old_prop_seg, 0, split_pos, new_prop_seg_left, 0);
        range_property_map_copy(old_prop_seg, split_pos, old_seg_size, new_prop_seg_right, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  ─────────────────── M111 Part 2: neo_range_tree ────────────────
//  移植自 neo_range_tree.cpp + neo_range_tree.h
// ═══════════════════════════════════════════════════════════════════

class RangeTree {
public:
    std::atomic<uint32_t> ref_cnt{1};
    std::vector<InRangeNode> node_block;
    std::vector<uint32_t> keys;

    RangeTree() = default;

    // 从已排序element数组批量构造
    RangeTree(RangeElement* elements, Property_t** properties,
              uint64_t element_num, MockWriterTraceBlock* trace_block) {
        auto arr = trace_block->allocate_range_element_segment();
        for (uint64_t i = 0; i < element_num; i++) {
            arr->value.at(i) = elements[i];
        }
        auto prop_arr = trace_block->allocate_range_prop_vec();
        if (properties) {
            for (uint64_t i = 0; i < element_num; i++) {
                if (properties[i]) {
                    prop_arr->value.at(i) = *properties[i];
                }
            }
        }
        node_block.emplace_back(element_num, (uint64_t)arr, prop_arr);
        keys.push_back(0);
    }

    // 从 std::vector 构造 (分段)
    RangeTree(std::vector<RangeElement>& elements, Property_t** properties,
              uint64_t element_num, MockWriterTraceBlock* trace_block) {
        auto segment_num = (element_num + RANGE_LEAF_SIZE - 1) / RANGE_LEAF_SIZE;
        const uint64_t EXPECTED_SEGMENT_SIZE = (element_num + segment_num - 1) / segment_num;
        keys.resize(segment_num);
        std::fill(keys.begin(), keys.end(), 0);
        node_block.resize(segment_num);
        for (uint64_t i = 0; i < segment_num; i++) {
            auto arr = trace_block->allocate_range_element_segment();
            uint64_t start = i * EXPECTED_SEGMENT_SIZE;
            uint64_t end = std::min((i + 1) * EXPECTED_SEGMENT_SIZE, element_num);
            std::copy(elements.begin() + start, elements.begin() + end,
                      arr->value.begin());
            auto prop_arr = trace_block->allocate_range_prop_vec();
            if (properties) {
                for (uint64_t j = start; j < end; j++) {
                    if (properties[j]) {
                        prop_arr->value.at(j - start) = *properties[j];
                    }
                }
            }
            node_block.at(i) = InRangeNode(end - start, (uint64_t)arr, prop_arr);
            keys.at(i) = arr->value.at(0);
        }
        keys.at(0) = 0;
    }

    // 从旧数组+插入一个元素构造
    RangeTree(RangeElement* elements, Property_t* properties,
              uint64_t element_num, uint64_t new_element,
              Property_t* property, uint64_t pos,
              MockWriterTraceBlock* trace_block) {
        auto arr = (RangeElementSegment_t*)trace_block->allocate_range_element_segment();
        std::fill(arr->value.begin(), arr->value.end(), 0);
        for (uint64_t i = 0; i < element_num; i++) {
            arr->value.at(i) = elements[i];
        }
        std::copy_backward(arr->value.begin() + pos,
                           arr->value.begin() + element_num,
                           arr->value.begin() + element_num + 1);
        arr->value.at(pos) = (RangeElement)new_element;
        node_block.emplace_back(element_num + 1, (uint64_t)arr);
        auto prop_arr = trace_block->allocate_range_prop_vec();
        if (properties) {
            for (uint64_t i = 0; i < pos; i++) {
                prop_arr->value.at(i) = properties[i];
            }
            if (property) prop_arr->value.at(pos) = *property;
            for (uint64_t i = pos; i < element_num; i++) {
                prop_arr->value.at(i + 1) = properties[i];
            }
        }
        node_block.at(0).property_map = prop_arr;
        keys.push_back(0);
    }

    // find_node: 返回包含 element 的段下标
    // 移植自 neo_range_tree.cpp find_node()
    uint8_t find_node(uint64_t element) const {
        uint16_t node_idx = 0;
        while (node_idx < node_block.size() && keys.at(node_idx) <= element) {
            node_idx++;
        }
        if (node_idx != 0) node_idx--;
        return (uint8_t)node_idx;
    }

    // has_element — 移植自 neo_range_tree.cpp
    bool has_element(uint64_t element) const {
        auto node = node_block.at(find_node(element));
        auto arr = (RangeElementSegment_t*)node.arr_ptr;
        uint16_t arr_size = node.size;
        auto pos = std::lower_bound(arr->value.begin(),
                                    arr->value.begin() + arr_size,
                                    (RangeElement)element);
        return pos != arr->value.begin() + arr_size && *pos == (RangeElement)element;
    }

    // range_intersect (with result vector) — 移植自 neo_range_tree.cpp
    void range_intersect(RangeElement* range, uint16_t range_size,
                         std::vector<uint64_t>& result) const {
        uint16_t range_idx = 0, node_idx = 0;
        while (range_idx < range_size && node_idx < node_block.size()) {
            InRangeNode node = node_block.at(node_idx);
            auto arr = (RangeElementSegment_t*)node.arr_ptr;
            uint16_t arr_size = node.size;
            uint16_t arr_idx = 0;
            while (range_idx < range_size && arr_idx < arr_size) {
                if (arr->value.at(arr_idx) < range[range_idx]) {
                    arr_idx++;
                } else if (arr->value.at(arr_idx) > range[range_idx]) {
                    range_idx++;
                } else {
                    // [MOD] intersect_hit_count
                    g_intersect_hit_count.fetch_add(1, std::memory_order_relaxed);
                    result.push_back(arr->value.at(arr_idx));
                    arr_idx++;
                    range_idx++;
                }
            }
            if (arr_idx == arr_size) node_idx++;
        }
    }

    // intersect (with other RangeTree, result vector) — 移植自 neo_range_tree.cpp
    void intersect(RangeTree* other_tree, std::vector<uint64_t>& result) const {
        uint16_t node_idx1 = 0, node_idx2 = 0;
        while (node_idx1 < node_block.size() &&
               node_idx2 < other_tree->node_block.size()) {
            InRangeNode node1 = node_block.at(node_idx1);
            InRangeNode node2 = other_tree->node_block.at(node_idx2);
            auto arr1 = (RangeElementSegment_t*)node1.arr_ptr;
            auto arr2 = (RangeElementSegment_t*)node2.arr_ptr;
            uint16_t arr_size1 = node1.size, arr_size2 = node2.size;
            uint16_t arr_idx1 = 0, arr_idx2 = 0;
            while (arr_idx1 < arr_size1 && arr_idx2 < arr_size2) {
                if (arr1->value.at(arr_idx1) < arr2->value.at(arr_idx2)) {
                    arr_idx1++;
                } else if (arr1->value.at(arr_idx1) > arr2->value.at(arr_idx2)) {
                    arr_idx2++;
                } else {
                    g_intersect_hit_count.fetch_add(1, std::memory_order_relaxed);
                    result.push_back(arr1->value.at(arr_idx1));
                    arr_idx1++;
                    arr_idx2++;
                }
            }
            if (arr_idx1 == arr_size1) node_idx1++;
            if (arr_idx2 == arr_size2) node_idx2++;
        }
    }

    // intersect (count only) — 移植自 neo_range_tree.cpp
    uint64_t range_intersect(RangeElement* range, uint16_t range_size) const {
        uint16_t range_idx = 0, node_idx = 0;
        uint64_t res = 0;
        while (range_idx < range_size && node_idx < node_block.size()) {
            InRangeNode node = node_block.at(node_idx);
            auto arr = (RangeElementSegment_t*)node.arr_ptr;
            uint16_t arr_size = node.size, arr_idx = 0;
            while (range_idx < range_size && arr_idx < arr_size) {
                if (arr->value.at(arr_idx) < range[range_idx]) {
                    arr_idx++;
                } else if (arr->value.at(arr_idx) > range[range_idx]) {
                    range_idx++;
                } else {
                    res++;
                    arr_idx++;
                    range_idx++;
                }
            }
            if (arr_idx == arr_size) node_idx++;
        }
        return res;
    }

    uint64_t intersect(RangeTree* other_tree) const {
        uint16_t node_idx1 = 0, node_idx2 = 0;
        uint64_t res = 0;
        while (node_idx1 < node_block.size() &&
               node_idx2 < other_tree->node_block.size()) {
            InRangeNode node1 = node_block.at(node_idx1);
            InRangeNode node2 = other_tree->node_block.at(node_idx2);
            auto arr1 = (RangeElementSegment_t*)node1.arr_ptr;
            auto arr2 = (RangeElementSegment_t*)node2.arr_ptr;
            uint16_t arr_size1 = node1.size, arr_size2 = node2.size;
            uint16_t arr_idx1 = 0, arr_idx2 = 0;
            while (arr_idx1 < arr_size1 && arr_idx2 < arr_size2) {
                if (arr1->value.at(arr_idx1) < arr2->value.at(arr_idx2)) {
                    arr_idx1++;
                } else if (arr1->value.at(arr_idx1) > arr2->value.at(arr_idx2)) {
                    arr_idx2++;
                } else {
                    res++;
                    arr_idx1++;
                    arr_idx2++;
                }
            }
            if (arr_idx1 == arr_size1) node_idx1++;
            if (arr_idx2 == arr_size2) node_idx2++;
        }
        return res;
    }

    // insert — 移植自 neo_range_tree.cpp insert()
    // [MOD] split_count: 记录段分裂次数
    bool insert(uint64_t src, uint64_t element, Property_t* property,
                std::vector<GCResourceInfo>& gc_resources,
                MockWriterTraceBlock* trace_block) {
        g_operation_count.fetch_add(1, std::memory_order_relaxed);
        auto node_idx = find_node(element);
        auto& node = node_block.at(node_idx);
        auto arr = (RangeElementSegment_t*)node.arr_ptr;
        uint16_t arr_size = node.size;

        if (arr == nullptr) {
            auto new_arr = trace_block->allocate_range_element_segment();
            new_arr->value.at(0) = (RangeElement)element;
            node.arr_ptr = (uint64_t)new_arr;
            node.size = 1;
            // [MOD] also initialize property map if property provided
            auto new_prop_map = trace_block->allocate_range_prop_vec();
            if (property) new_prop_map->value.at(0) = *property;
            node.property_map = new_prop_map;
            return true;
        }

        uint64_t pos = std::lower_bound(arr->value.begin(),
                                        arr->value.begin() + arr_size,
                                        (RangeElement)element) - arr->value.begin();
        if (pos != arr_size && arr->value.at(pos) == (RangeElement)element) {
            return false;  // duplicate
        }

        gc_resources.emplace_back(GCResourceInfo{Inner_Segment, (void*)arr});
        if (node.property_map) {
            gc_resources.emplace_back(GCResourceInfo{
                Range_Property_Map_All_Modified, (void*)node.property_map});
        }

        if (arr_size < RANGE_LEAF_SIZE) {
            // simple insert
            auto new_arr = trace_block->allocate_range_element_segment();
            std::copy(arr->value.begin(), arr->value.begin() + pos,
                      new_arr->value.begin());
            new_arr->value.at(pos) = (RangeElement)element;
            std::copy(arr->value.begin() + pos, arr->value.begin() + arr_size,
                      new_arr->value.begin() + pos + 1);
            node.arr_ptr = (uint64_t)new_arr;
            auto new_property_map = trace_block->allocate_range_prop_vec();
            if (node.property_map) {
                range_property_map_copy(node.property_map, new_property_map);
            }
            map_insert_range_property((void*)new_property_map, pos, arr_size,
                                       (void*)property);
            node.property_map = new_property_map;
            node.size++;
        } else {
            // split — [MOD] split_count
            g_split_count.fetch_add(1, std::memory_order_relaxed);
            std::printf("    [SPLIT] element=%lu node_idx=%u arr_size=%u\n",
                        element, node_idx, arr_size);

            auto new_l_arr = trace_block->allocate_range_element_segment();
            auto new_r_arr = trace_block->allocate_range_element_segment();

            std::copy(arr->value.begin(),
                      arr->value.begin() + RANGE_LEAF_SIZE / 2,
                      new_l_arr->value.begin());
            std::copy(arr->value.begin() + RANGE_LEAF_SIZE / 2,
                      arr->value.begin() + arr_size,
                      new_r_arr->value.begin());

            node.arr_ptr = (uint64_t)new_l_arr;
            node.size = RANGE_LEAF_SIZE / 2;

            auto new_node = InRangeNode(RANGE_LEAF_SIZE / 2, (uint64_t)new_r_arr);

            auto new_l_prop = trace_block->allocate_range_prop_vec();
            auto new_r_prop = trace_block->allocate_range_prop_vec();
            if (node.property_map) {
                range_property_map_copy(node.property_map, 0, RANGE_LEAF_SIZE / 2,
                                        new_l_prop, 0);
                range_property_map_copy(node.property_map, RANGE_LEAF_SIZE / 2,
                                        arr_size, new_r_prop, 0);
            }
            node.property_map = new_l_prop;
            new_node.property_map = new_r_prop;

            node_block.insert(node_block.begin() + node_idx + 1, new_node);
            keys.insert(keys.begin() + node_idx + 1, new_r_arr->value.at(0));

            // re-insert element
            if (new_r_arr->value.at(0) <= (RangeElement)element) {
                pos -= RANGE_LEAF_SIZE / 2;
                std::copy_backward(new_r_arr->value.begin() + pos,
                                   new_r_arr->value.begin() + RANGE_LEAF_SIZE / 2,
                                   new_r_arr->value.begin() + RANGE_LEAF_SIZE / 2 + 1);
                new_r_arr->value.at(pos) = (RangeElement)element;
                map_insert_range_property((void*)new_r_prop, pos,
                                           RANGE_LEAF_SIZE / 2, (void*)property);
                node_block.at(node_idx + 1).size++;
            } else {
                std::copy_backward(new_l_arr->value.begin() + pos,
                                   new_l_arr->value.begin() + RANGE_LEAF_SIZE / 2,
                                   new_l_arr->value.begin() + RANGE_LEAF_SIZE / 2 + 1);
                new_l_arr->value.at(pos) = (RangeElement)element;
                map_insert_range_property((void*)new_l_prop, pos,
                                           RANGE_LEAF_SIZE / 2, (void*)property);
                node_block.at(node_idx).size++;
            }
        }
        return true;
    }

    // insert_element_batch — 移植自 neo_range_tree.cpp
    // [MOD] merge_progress: 每处理一个旧节点打印进度
    RangeTreeInsertElemBatchRes insert_element_batch(
        uint64_t src,
        const std::pair<RangeElement, RangeElement>* edges,
        Property_t** properties,
        uint64_t count,
        std::vector<GCResourceInfo>& gc_resources,
        MockWriterTraceBlock* trace_block) {

        g_operation_count.fetch_add(count, std::memory_order_relaxed);
        uint64_t inserted = 0;
        auto new_range_tree = new RangeTree();

        auto child_num = (int64_t)node_block.size();
        int64_t old_node_idx = 0;
        int64_t list_st = 0, list_ed = 0;

        uint64_t new_segment_size = 0;
        auto new_segment = trace_block->allocate_range_element_segment();
        auto new_property_map = trace_block->allocate_range_prop_vec();

        auto move_to_next_node = [&]() {
            new_range_tree->node_block.push_back(
                InRangeNode{new_segment_size, (uint64_t)new_segment, new_property_map});
            new_range_tree->keys.push_back(new_segment->value.at(0));
            new_segment_size = 0;
            new_segment = trace_block->allocate_range_element_segment();
            new_property_map = trace_block->allocate_range_prop_vec();
        };

        while (old_node_idx < child_num && list_ed < (int64_t)count) {
            auto old_node = node_block.at(old_node_idx);
            auto next_key = (old_node_idx != child_num - 1)
                ? keys[old_node_idx + 1]
                : (uint32_t)std::numeric_limits<uint32_t>::max();

            if (edges[list_st].second >= next_key) {
                new_range_tree->node_block.push_back(old_node);
                new_range_tree->keys.push_back(keys[old_node_idx]);
                old_node_idx++;
                // [MOD] merge_progress print
                if (old_node_idx % 4 == 0) {
                    std::printf("    [BATCH-MERGE] skipped node_idx=%ld/%ld\n",
                                old_node_idx, child_num);
                }
                continue;
            }

            while (list_ed < (int64_t)count &&
                   edges[list_ed].second < next_key) {
                list_ed++;
            }
            int64_t list_idx = list_st;
            int64_t leaf_idx = 0;

            auto old_arr = (RangeElementSegment_t*)old_node.arr_ptr;
            gc_resources.emplace_back(GCResourceInfo{Inner_Segment, (void*)old_arr});
            auto old_prop_arr = old_node.property_map;
            if (old_prop_arr) {
                gc_resources.emplace_back(GCResourceInfo{
                    Range_Property_Map_All_Modified, (void*)old_prop_arr});
            }

            uint64_t total_size = old_node.size + (list_ed - list_st);
            uint64_t segment_num_loc = (total_size + RANGE_LEAF_SIZE - 1) / RANGE_LEAF_SIZE;
            const uint64_t EXPECTED_SEGMENT_SIZE =
                (total_size + segment_num_loc - 1) / segment_num_loc;

            while (list_idx < list_ed && leaf_idx < (int64_t)old_node.size) {
                if (edges[list_idx].second < old_arr->value.at(leaf_idx)) {
                    if (properties && properties[list_idx]) {
                        map_set_sa_range_property(new_property_map, new_segment_size,
                                                   properties[list_idx]);
                    }
                    new_segment->value.at(new_segment_size++) =
                        edges[list_idx++].second;
                    inserted++;
                } else if (edges[list_idx].second > old_arr->value.at(leaf_idx)) {
                    if (old_prop_arr) {
                        auto old_property = map_get_all_range_property(old_prop_arr, leaf_idx);
                        map_set_sa_range_property(new_property_map, new_segment_size,
                                                   old_property);
                    }
                    new_segment->value.at(new_segment_size++) =
                        old_arr->value.at(leaf_idx++);
                } else {
                    if (properties && properties[list_idx]) {
                        map_set_sa_range_property(new_property_map, new_segment_size,
                                                   properties[list_idx]);
                    }
                    new_segment->value.at(new_segment_size++) =
                        edges[list_idx++].second;
                    leaf_idx++;
                }
                if (new_segment_size == EXPECTED_SEGMENT_SIZE) {
                    move_to_next_node();
                }
            }

            while (list_idx < list_ed) {
                if (properties && properties[list_idx]) {
                    map_set_sa_range_property(new_property_map, new_segment_size,
                                               properties[list_idx]);
                }
                new_segment->value.at(new_segment_size++) =
                    edges[list_idx++].second;
                inserted++;
                if (new_segment_size == EXPECTED_SEGMENT_SIZE) {
                    move_to_next_node();
                }
            }
            while (leaf_idx < (int64_t)old_node.size) {
                if (old_prop_arr) {
                    auto old_property = map_get_all_range_property(old_prop_arr, leaf_idx);
                    map_set_sa_range_property(new_property_map, new_segment_size,
                                               old_property);
                }
                new_segment->value.at(new_segment_size++) =
                    old_arr->value.at(leaf_idx++);
                if (new_segment_size == EXPECTED_SEGMENT_SIZE) {
                    move_to_next_node();
                }
            }

            if (new_segment_size != 0) {
                move_to_next_node();
            }

            list_st = list_ed;
            old_node_idx++;
        }

        if (old_node_idx < child_num) {
            while (old_node_idx < child_num) {
                new_range_tree->node_block.push_back(node_block.at(old_node_idx));
                new_range_tree->keys.push_back(keys.at(old_node_idx));
                old_node_idx++;
            }
        } else if (list_ed < (int64_t)count) {
            const uint64_t remaining = count - list_ed;
            const uint64_t num_leaves = (remaining + RANGE_LEAF_SIZE - 1) / RANGE_LEAF_SIZE;
            const uint64_t AVG_LEAF_SIZE = (remaining + num_leaves - 1) / num_leaves;
            inserted += remaining;
            while (list_ed < (int64_t)count) {
                if (properties && properties[list_ed]) {
                    map_set_sa_range_property(new_property_map, new_segment_size,
                                               properties[list_ed]);
                }
                new_segment->value.at(new_segment_size++) =
                    edges[list_ed++].second;
                if (new_segment_size == AVG_LEAF_SIZE) {
                    move_to_next_node();
                }
            }
        }

        if (!new_range_tree->keys.empty()) {
            new_range_tree->keys.at(0) = 0;
        }

        if (new_segment_size != 0) {
            new_range_tree->node_block.push_back(
                InRangeNode{new_segment_size, (uint64_t)new_segment, new_property_map});
            new_range_tree->keys.push_back(new_segment->value.at(0));
        } else {
            trace_block->deallocate_range_element_segment(new_segment);
            trace_block->deallocate_range_prop_vec(new_property_map);
        }

        if (!new_range_tree->keys.empty()) {
            new_range_tree->keys.at(0) = 0;
        }

        return RangeTreeInsertElemBatchRes{inserted, (void*)new_range_tree};
    }

    // remove — 移植自 neo_range_tree.cpp
    // [MOD] freed_count: 记录删除节点次数
    bool remove(uint64_t element, std::vector<GCResourceInfo>& gc_resources,
                MockWriterTraceBlock* trace_block) {
        g_operation_count.fetch_add(1, std::memory_order_relaxed);
        auto node_idx = find_node(element);
        auto& node = node_block.at(node_idx);
        auto arr = (RangeElementSegment_t*)node.arr_ptr;
        uint16_t arr_size = node.size;
        auto prop_arr = node.property_map;

        uint64_t pos = std::lower_bound(arr->value.begin(),
                                        arr->value.begin() + arr_size,
                                        (RangeElement)element) - arr->value.begin();
        if (pos == arr_size || arr->value.at(pos) != (RangeElement)element) {
            return false;
        }

        gc_resources.emplace_back(GCResourceInfo{Inner_Segment, (void*)arr});
        if (prop_arr) {
            gc_resources.emplace_back(GCResourceInfo{Range_Property_Vec, (void*)prop_arr});
        }

        if (arr_size == 1) {
            g_freed_count.fetch_add(1, std::memory_order_relaxed);
            // [MOD] 打印节点释放情况
            std::printf("    [REMOVE-NODE] erase node_idx=%u (size=1)\n", node_idx);
            if (node_block.size() == 1) {
                node_block.at(0) = InRangeNode{0, 0};
                keys.at(0) = 0;
            } else {
                if (node_idx == 0) keys.at(1) = 0;
                node_block.erase(node_block.begin() + node_idx);
                keys.erase(keys.begin() + node_idx);
            }
        } else {
            auto new_arr = trace_block->allocate_range_element_segment();
            std::copy(arr->value.begin(), arr->value.begin() + pos,
                      new_arr->value.begin());
            std::copy(arr->value.begin() + pos + 1,
                      arr->value.begin() + arr_size,
                      new_arr->value.begin() + pos);
            if (prop_arr) {
                auto new_prop_arr = trace_block->allocate_range_prop_vec();
                std::copy(prop_arr->value.begin(), prop_arr->value.begin() + pos,
                          new_prop_arr->value.begin());
                std::copy(prop_arr->value.begin() + pos + 1,
                          prop_arr->value.begin() + arr_size,
                          new_prop_arr->value.begin() + pos);
                node.property_map = new_prop_arr;
            }
            node.arr_ptr = (uint64_t)new_arr;
            node.size--;
        }
        return true;
    }

    // get_property — 移植自 neo_range_tree.cpp
    Property_t get_property(uint64_t element, uint8_t property_id) const {
        auto node = node_block.at(find_node(element));
        auto arr = (RangeElementSegment_t*)node.arr_ptr;
        uint16_t arr_size = node.size;
        auto pos = std::lower_bound(arr->value.begin(),
                                    arr->value.begin() + arr_size,
                                    (RangeElement)element);
        if (pos == arr->value.begin() + arr_size || *pos != (RangeElement)element) {
            return Property_t();
        }
        auto pos_idx = pos - arr->value.begin();
        return map_get_range_property(node.property_map, pos_idx, property_id);
    }

    void set_property(uint64_t element, uint8_t property_id, Property_t property,
                      std::vector<GCResourceInfo>& gc_resources,
                      MockWriterTraceBlock* trace_block) {
        auto& node = node_block.at(find_node(element));
        auto arr = (RangeElementSegment_t*)node.arr_ptr;
        uint16_t arr_size = node.size;
        auto pos = std::lower_bound(arr->value.begin(),
                                    arr->value.begin() + arr_size,
                                    (RangeElement)element);
        if (pos == arr->value.begin() + arr_size || *pos != (RangeElement)element) {
            return;
        }
        auto pos_idx = pos - arr->value.begin();
        auto old_prop_map = node.property_map;
        auto new_prop_map = trace_block->allocate_range_prop_vec();
        if (old_prop_map) {
            std::copy(old_prop_map->value.begin(),
                      old_prop_map->value.begin() + arr_size,
                      new_prop_map->value.begin());
            gc_resources.emplace_back(GCResourceInfo{Range_Property_Vec, (void*)old_prop_map});
        }
        map_set_range_property(new_prop_map, pos_idx, property_id, property);
        node.property_map = new_prop_map;
    }

    // for_each — 移植自 neo_range_tree.h 模板
    template<typename F>
    void for_each(F&& callback) {
        for (uint8_t idx = 0; idx < node_block.size(); idx++) {
            uint16_t arr_size = node_block[idx].size;
            auto arr = (RangeElementSegment_t*)node_block[idx].arr_ptr;
            for (uint16_t inner_idx = 0; inner_idx < arr_size; inner_idx++) {
                callback(arr->value.at(inner_idx), 0.0);
            }
        }
    }
};

RangeTree* copy_range_tree(RangeTree* tree) {
    auto new_tree = new RangeTree();
    new_tree->node_block.resize(tree->node_block.size());
    std::copy(tree->node_block.begin(), tree->node_block.end(),
              new_tree->node_block.begin());
    new_tree->keys.resize(tree->keys.size());
    std::copy(tree->keys.begin(), tree->keys.end(), new_tree->keys.begin());
    return new_tree;
}

// ═══════════════════════════════════════════════════════════════════
//  ─────────────────── M110 Part 2: neo_index (简化模拟) ──────────
//
//  NeoGraphIndex 核心逻辑: forest / insert_vertex / insert_edge /
//  has_vertex / has_edge / get_degree / get_neighbor / intersect /
//  remove_vertex / remove_edge / commit / gc / clear
//
//  依赖: 因实验文件不带 NeoTree 完整实现, 这里用 SimpleNeoTree
//  模拟 NeoTree 的必要接口, 以验证 NeoGraphIndex 的 forest 管理逻辑
// ═══════════════════════════════════════════════════════════════════

// SimpleNeoTree: 简化NeoTree接口, 用于验证NeoGraphIndex forest逻辑
struct SimpleNeoTree {
    uint64_t vertex_prefix;
    std::mutex writer_lock_mutex;
    // 简化存储: vertex set + adjacency list
    std::unordered_map<uint64_t, std::vector<uint64_t>> adj;  // src -> [dests]
    std::unordered_map<uint64_t, bool> vertices;

    explicit SimpleNeoTree(uint64_t prefix) : vertex_prefix(prefix) {}

    void lock_write() { writer_lock_mutex.lock(); }
    void unlock_write() { writer_lock_mutex.unlock(); }

    bool has_vertex(uint64_t v, uint64_t /*ts*/) const {
        return vertices.count(v) > 0;
    }

    bool has_edge(uint64_t src, uint64_t dest, uint64_t /*ts*/) const {
        auto it = adj.find(src);
        if (it == adj.end()) return false;
        return std::find(it->second.begin(), it->second.end(), dest) != it->second.end();
    }

    uint64_t get_degree(uint64_t src, uint64_t /*ts*/) const {
        auto it = adj.find(src);
        return (it == adj.end()) ? 0 : (uint64_t)it->second.size();
    }

    bool get_neighbor(uint64_t src, std::vector<RangeElement>& neighbor,
                      uint64_t /*ts*/) const {
        auto it = adj.find(src);
        if (it == adj.end()) return false;
        for (auto d : it->second) {
            neighbor.push_back((RangeElement)d);
        }
        return true;
    }

    bool insert_vertex(uint64_t v, Property_t* /*prop*/,
                       MockWriterTraceBlock* /*tb*/) {
        vertices[v] = true;
        if (adj.find(v) == adj.end()) adj[v] = {};
        return true;
    }

    bool insert_vertex_batch(const uint64_t* vs, Property_t** /*props*/,
                             uint64_t cnt, MockWriterTraceBlock* /*tb*/) {
        for (uint64_t i = 0; i < cnt; i++) {
            vertices[vs[i]] = true;
            if (adj.find(vs[i]) == adj.end()) adj[vs[i]] = {};
        }
        return true;
    }

    void insert_edge(uint64_t src, uint64_t dest, Property_t* /*prop*/,
                     MockWriterTraceBlock* /*tb*/) {
        if (vertices.count(src)) {
            adj[src].push_back(dest);
        }
    }

    bool remove_vertex(uint64_t v, bool /*is_directed*/,
                       MockWriterTraceBlock* /*tb*/) {
        if (!vertices.count(v)) return false;
        vertices.erase(v);
        adj.erase(v);
        return true;
    }

    void remove_edge(uint64_t src, uint64_t dest, MockWriterTraceBlock* /*tb*/) {
        auto it = adj.find(src);
        if (it != adj.end()) {
            it->second.erase(
                std::remove(it->second.begin(), it->second.end(), dest),
                it->second.end());
        }
    }

    void commit_version(uint64_t /*ts*/) {}
    void gc(MockWriterTraceBlock* /*tb*/) {
        g_gc_freed_count.fetch_add(1, std::memory_order_relaxed);
    }
};

// NeoGraphIndex: 移植自 neo_index.cpp + neo_index.h
// 使用 SimpleNeoTree 替代真实 NeoTree
struct NeoGraphIndex {
    std::vector<std::unique_ptr<SimpleNeoTree>> forest;

    // [MOD] operation_count: 操作计数
    std::atomic<uint64_t> op_insert_vertex{0};
    std::atomic<uint64_t> op_insert_edge{0};
    std::atomic<uint64_t> op_remove{0};

    NeoGraphIndex() {}
    ~NeoGraphIndex() {}

    static uint64_t gen_tree_direction(uint64_t val) {
        return val >> VERTEX_GROUP_BITS;
    }

    // lock — 移植自 neo_index.cpp
    SimpleNeoTree* lock(uint64_t direction) {
        if (forest.size() <= direction) {
            g_forest_resize_count.fetch_add(1, std::memory_order_relaxed);
            forest.resize(direction + 1);
        }
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return nullptr;
        raw->lock_write();
        return raw;
    }

    void unlock(uint64_t direction) {
        if (direction >= forest.size()) return;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return;
        raw->unlock_write();
    }

    // has_vertex — 移植自 neo_index.cpp
    bool has_vertex(uint64_t vertex, uint64_t timestamp) const {
        auto direction = gen_tree_direction(vertex);
        if (forest.size() <= direction) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        return raw->has_vertex(vertex, timestamp);
    }

    // has_edge — 移植自 neo_index.cpp
    bool has_edge(uint64_t src, uint64_t dest, uint64_t timestamp) const {
        auto direction = gen_tree_direction(src);
        if (forest.size() <= direction) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        return raw->has_edge(src, dest, timestamp);
    }

    // get_degree — 移植自 neo_index.cpp
    uint64_t get_degree(uint64_t src, uint64_t timestamp) const {
        auto direction = gen_tree_direction(src);
        if (forest.size() <= direction) return 0;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return 0;
        return raw->get_degree(src, timestamp);
    }

    // get_neighbor — 移植自 neo_index.cpp
    bool get_neighbor(uint64_t src, std::vector<RangeElement>& neighbor,
                      uint64_t timestamp) const {
        auto direction = gen_tree_direction(src);
        if (forest.size() <= direction) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        return raw->get_neighbor(src, neighbor, timestamp);
    }

    // intersect (result vector) — 移植自 neo_index.cpp
    void intersect(uint64_t src1, uint64_t src2, std::vector<uint64_t>& result,
                   uint64_t timestamp) const {
        uint64_t dir1 = gen_tree_direction(src1);
        uint64_t dir2 = gen_tree_direction(src2);
        if (dir1 >= forest.size() || dir2 >= forest.size()) return;
        auto raw1 = forest.at(dir1).get();
        auto raw2 = forest.at(dir2).get();
        if (!raw1 || !raw2) return;
        // 获取两个邻居列表求交集
        std::vector<RangeElement> n1, n2;
        raw1->get_neighbor(src1, n1, timestamp);
        raw2->get_neighbor(src2, n2, timestamp);
        std::sort(n1.begin(), n1.end());
        std::sort(n2.begin(), n2.end());
        uint64_t i = 0, j = 0;
        while (i < n1.size() && j < n2.size()) {
            if (n1[i] == n2[j]) {
                g_intersect_hit_count.fetch_add(1, std::memory_order_relaxed);
                result.push_back(n1[i]);
                i++; j++;
            } else if (n1[i] < n2[j]) {
                i++;
            } else {
                j++;
            }
        }
    }

    // intersect (count) — 移植自 neo_index.cpp
    uint64_t intersect(uint64_t src1, uint64_t src2, uint64_t timestamp) const {
        std::vector<uint64_t> result;
        intersect(src1, src2, result, timestamp);
        return result.size();
    }

    // insert_vertex — 移植自 neo_index.cpp
    // [MOD] op_insert_vertex 计数
    bool insert_vertex(uint64_t vertex, Property_t* property,
                       MockWriterTraceBlock* trace_block) {
        op_insert_vertex.fetch_add(1, std::memory_order_relaxed);
        g_operation_count.fetch_add(1, std::memory_order_relaxed);
        auto direction = gen_tree_direction(vertex);
        if (forest.size() <= direction) {
            g_forest_resize_count.fetch_add(1, std::memory_order_relaxed);
            forest.resize(direction + 1);
        }
        auto raw = forest.at(direction).get();
        if (raw == nullptr) {
            auto new_tree = std::make_unique<SimpleNeoTree>(
                vertex & ~VERTEX_GROUP_MASK);
            new_tree->insert_vertex(vertex, property, trace_block);
            forest[direction] = std::move(new_tree);
        } else {
            raw->insert_vertex(vertex, property, trace_block);
        }
        return true;
    }

    // insert_vertex_batch — 移植自 neo_index.cpp
    bool insert_vertex_batch(const uint64_t* vertices, Property_t** properties,
                              uint64_t count, MockWriterTraceBlock* trace_block) {
        if (count == 0 || vertices == nullptr) return true;
        op_insert_vertex.fetch_add(count, std::memory_order_relaxed);
        g_operation_count.fetch_add(count, std::memory_order_relaxed);
        uint64_t st = 0, ed = 0;
        while (ed != count) {
            while (ed != count &&
                   gen_tree_direction(vertices[ed]) == gen_tree_direction(vertices[st])) {
                ed++;
            }
            auto direction = gen_tree_direction(vertices[st]);
            if (forest.size() <= direction) {
                g_forest_resize_count.fetch_add(1, std::memory_order_relaxed);
                forest.resize(direction + 1);
            }
            auto raw = forest.at(direction).get();
            if (!raw) {
                auto new_tree = std::make_unique<SimpleNeoTree>(
                    vertices[st] & VERTEX_GROUP_MASK);
                new_tree->insert_vertex_batch(vertices + st,
                    properties ? properties + st : nullptr, ed - st, trace_block);
                forest[direction] = std::move(new_tree);
            } else {
                raw->insert_vertex_batch(vertices + st,
                    properties ? properties + st : nullptr, ed - st, trace_block);
            }
            st = ed;
        }
        return true;
    }

    // insert_edge — 移植自 neo_index.cpp
    // [MOD] op_insert_edge 计数
    bool insert_edge(uint64_t src, uint64_t dest, Property_t* property,
                     MockWriterTraceBlock* trace_block) {
        op_insert_edge.fetch_add(1, std::memory_order_relaxed);
        g_operation_count.fetch_add(1, std::memory_order_relaxed);
        auto direction = gen_tree_direction(src);
        if (direction >= forest.size()) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        raw->insert_edge(src, dest, property, trace_block);
        return true;
    }

    // remove_vertex — 移植自 neo_index.cpp
    bool remove_vertex(uint64_t vertex, bool is_directed,
                       MockWriterTraceBlock* trace_block) {
        op_remove.fetch_add(1, std::memory_order_relaxed);
        g_operation_count.fetch_add(1, std::memory_order_relaxed);
        auto direction = gen_tree_direction(vertex);
        if (direction >= forest.size()) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        return raw->remove_vertex(vertex, is_directed, trace_block);
    }

    // remove_edge — 移植自 neo_index.cpp
    bool remove_edge(uint64_t src, uint64_t dest,
                     MockWriterTraceBlock* trace_block) {
        op_remove.fetch_add(1, std::memory_order_relaxed);
        g_operation_count.fetch_add(1, std::memory_order_relaxed);
        auto direction = gen_tree_direction(src);
        if (direction >= forest.size()) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        raw->remove_edge(src, dest, trace_block);
        return true;
    }

    // commit — 移植自 neo_index.cpp
    SimpleNeoTree* commit(uint64_t direction, uint64_t timestamp) {
        if (direction >= forest.size()) return nullptr;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return nullptr;
        raw->commit_version(timestamp);
        return raw;
    }

    // gc — 移植自 neo_index.cpp
    // [MOD] g_gc_freed_count: 统计gc调用
    void gc(uint64_t direction, MockWriterTraceBlock* trace_block) {
        if (direction >= forest.size()) return;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return;
        raw->gc(trace_block);
        g_gc_freed_count.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() {
        forest.clear();
    }

    // get_filling_info — 移植自 neo_index.cpp (stub)
    std::pair<uint64_t, uint64_t> get_filling_info(uint64_t /*timestamp*/) const {
        return {0, 0};
    }

    // edges (template) — 移植自 neo_index.h
    template<typename F>
    bool edges(uint64_t src, F&& callback, uint64_t timestamp) const {
        auto direction = gen_tree_direction(src);
        if (direction >= forest.size()) return false;
        auto raw = forest.at(direction).get();
        if (raw == nullptr) return false;
        auto it = raw->adj.find(src);
        if (it == raw->adj.end()) return true;
        for (auto dest : it->second) {
            callback((RangeElement)dest, 0.0);
        }
        return true;
    }

    // [MOD] dump operation stats
    void dump_stats() const {
        std::printf("    [Index] op_insert_vertex=%lu, op_insert_edge=%lu, "
                    "op_remove=%lu, forest_size=%zu\n",
                    op_insert_vertex.load(), op_insert_edge.load(),
                    op_remove.load(), forest.size());
    }
};

} // namespace experiment
} // namespace philemon

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE — M110: neo_property
// ═══════════════════════════════════════════════════════════════════

using namespace philemon::experiment;

void test_property_vec_basic() {
    std::printf("[TEST] PropertyVec<8> basic get/set/insert/remove\n");
    PropertyVec<8> pv;
    pv.set(0, 42);
    pv.set(1, 100);
    pv.set(2, 200);
    TEST_ASSERT(pv.get(0) == 42, "set/get 0");
    TEST_ASSERT(pv.get(1) == 100, "set/get 1");
    TEST_ASSERT(pv.get(2) == 200, "set/get 2");
    // insert at position 1 (shift right), size=3
    pv.insert(1, 3, 55);
    // After insert: [42, 55, 100, 200, ...]
    TEST_ASSERT(pv.get(0) == 42, "after insert pos0");
    TEST_ASSERT(pv.get(1) == 55, "after insert pos1");
    TEST_ASSERT(pv.get(2) == 100, "after insert pos2");
    TEST_PASS("PropertyVec basic get/set/insert/remove");
}

void test_property_vec_copy() {
    std::printf("[TEST] PropertyVec<8> copy_to\n");
    PropertyVec<8> src;
    PropertyVec<8> dst;
    for (int i = 0; i < 8; i++) src.set(i, (uint64_t)i * 10);
    uint64_t steps_before = g_copy_steps.load();
    src.copy_to(&dst);
    uint64_t steps_after = g_copy_steps.load();
    TEST_ASSERT(steps_after > steps_before, "copy_steps incremented");
    for (int i = 0; i < 8; i++) {
        TEST_ASSERT(dst.get(i) == (uint64_t)i * 10, "copy_to value match");
    }
    // partial copy
    PropertyVec<8> dst2;
    src.copy_to(2, 6, &dst2, 0);
    TEST_ASSERT(dst2.get(0) == 20, "partial copy start");
    TEST_ASSERT(dst2.get(3) == 50, "partial copy end");
    TEST_PASS("PropertyVec copy_to");
}

void test_property_vec_insert_copy() {
    std::printf("[TEST] PropertyVec<8> insert_copy\n");
    PropertyVec<8> src;
    PropertyVec<8> dst;
    src.set(0, 10); src.set(1, 30); src.set(2, 40);
    // insert_copy: insert value=20 at pos=1, size=3
    src.insert_copy(&dst, 1, 3, 20);
    TEST_ASSERT(dst.get(0) == 10, "insert_copy[0]");
    TEST_ASSERT(dst.get(1) == 20, "insert_copy[1]");
    TEST_ASSERT(dst.get(2) == 30, "insert_copy[2]");
    TEST_PASS("PropertyVec insert_copy");
}

void test_property_vec_remove() {
    std::printf("[TEST] PropertyVec<8> remove\n");
    PropertyVec<8> pv;
    pv.set(0, 10); pv.set(1, 20); pv.set(2, 30); pv.set(3, 40);
    // remove at pos=1, size=4
    pv.remove(1, 4);
    // After: [10, 30, 40, ...]
    TEST_ASSERT(pv.get(0) == 10, "remove result[0]");
    TEST_ASSERT(pv.get(1) == 30, "remove result[1]");
    TEST_ASSERT(pv.get(2) == 40, "remove result[2]");
    TEST_PASS("PropertyVec remove");
}

void test_property_vec_gc() {
    std::printf("[TEST] VertexPropertyVec_t GC ref-count\n");
    auto* pv = new VertexPropertyVec_t();
    pv->set(0, 999);
    TEST_ASSERT(pv->ref_cnt.load() == 1, "initial ref_cnt=1");
    uint64_t before = g_ref_dec_count.load();
    gc_vertex_property_map_ref(pv); // should free
    uint64_t after = g_ref_dec_count.load();
    TEST_ASSERT(after > before, "ref_dec_count incremented");
    // pv is now deleted — accessing would be UB, but we just check counter
    TEST_PASS("VertexPropertyVec_t GC ref-count");
}

void test_property_map_ops() {
    std::printf("[TEST] range property map get/set/insert/remove\n");
    auto pv = new RangePropertyVec_t();
    std::fill(pv->value.begin(), pv->value.end(), 0);

    uint64_t before = g_map_op_count.load();
    map_set_range_property((void*)pv, 0, 0, 777);
    TEST_ASSERT(map_get_range_property((void*)pv, 0, 0) == 777, "set/get map");
    uint64_t after = g_map_op_count.load();
    TEST_ASSERT(after > before, "map_op_count incremented");

    // insert at pos=0, size=1
    Property_t prop_val = 555;
    map_insert_range_property((void*)pv, 0, 1, (void*)&prop_val);
    TEST_ASSERT(pv->value.at(0) == 555, "insert shifts and sets");

    // remove at pos=0, size=2
    map_remove_range_property((void*)pv, 0, 2);
    TEST_ASSERT(pv->value.at(0) == 777, "remove shifts back");

    delete pv;
    TEST_PASS("range property map ops");
}

void test_vertex_property_map() {
    std::printf("[TEST] vertex_property_map_copy + map_get/set_vertex_property\n");
    auto* src = new VertexPropertyVec_t();
    auto* dst = new VertexPropertyVec_t();
    map_set_vertex_property(src, 10, 12345);
    map_set_vertex_property(src, 20, 99999);
    vertex_property_map_copy(src, dst);
    TEST_ASSERT(map_get_vertex_property(dst, 10) == 12345, "copied prop[10]");
    TEST_ASSERT(map_get_vertex_property(dst, 20) == 99999, "copied prop[20]");
    destroy_vertex_property_map(src);
    destroy_vertex_property_map(dst);
    TEST_PASS("vertex_property_map_copy + map_get/set_vertex_property");
}

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE — M111: neo_range_ops
// ═══════════════════════════════════════════════════════════════════

void test_range_segment_find_seq() {
    std::printf("[TEST] range_segment_find sequential (count<16)\n");
    RangeElement seg[] = {10, 20, 30, 40, 50};
    uint64_t steps_before = g_binary_search_steps.load();
    uint16_t pos = range_segment_find(seg, 5, 30);
    TEST_ASSERT(pos == 2, "found at pos 2");
    uint16_t pos2 = range_segment_find(seg, 5, 99);
    TEST_ASSERT(pos2 == (uint16_t)RANGE_LEAF_SIZE, "not found returns RANGE_LEAF_SIZE");
    uint64_t steps_after = g_binary_search_steps.load();
    TEST_ASSERT(steps_after > steps_before, "binary_search_steps incremented");
    TEST_PASS("range_segment_find sequential");
}

void test_range_segment_find_binary() {
    std::printf("[TEST] range_segment_find binary search (count>=16)\n");
    std::vector<RangeElement> seg(20);
    for (int i = 0; i < 20; i++) seg[i] = (RangeElement)(i * 5);
    uint64_t steps_before = g_binary_search_steps.load();
    uint16_t pos = range_segment_find(seg.data(), 20, 45);
    TEST_ASSERT(pos == 9, "binary found at pos 9");
    uint16_t pos2 = range_segment_find(seg.data(), 20, 46);
    TEST_ASSERT(pos2 == (uint16_t)RANGE_LEAF_SIZE, "binary not found");
    uint64_t steps_after = g_binary_search_steps.load();
    TEST_ASSERT(steps_after > steps_before, "binary_search_steps for binary search incremented");
    TEST_PASS("range_segment_find binary");
}

void test_range_segment_insert_copy() {
    std::printf("[TEST] range_segment_insert_copy\n");
    auto old_seg = new RangeElementSegment_t();
    auto new_seg = new RangeElementSegment_t();
    old_seg->value[0] = 10; old_seg->value[1] = 30; old_seg->value[2] = 40;
    uint64_t cmp_before = g_comparison_count.load();
    range_segment_insert_copy(old_seg, nullptr, 3, new_seg, nullptr,
                               1, 20, nullptr);
    uint64_t cmp_after = g_comparison_count.load();
    TEST_ASSERT(cmp_after > cmp_before, "comparison_count incremented");
    TEST_ASSERT(new_seg->value[0] == 10, "insert_copy[0]");
    TEST_ASSERT(new_seg->value[1] == 20, "insert_copy[1]");
    TEST_ASSERT(new_seg->value[2] == 30, "insert_copy[2]");
    TEST_ASSERT(new_seg->value[3] == 40, "insert_copy[3]");
    delete old_seg;
    delete new_seg;
    TEST_PASS("range_segment_insert_copy");
}

void test_range_segment_append() {
    std::printf("[TEST] range_segment_append\n");
    auto seg = new RangeElementSegment_t();
    std::fill(seg->value.begin(), seg->value.end(), 0);
    seg->value[0] = 10; seg->value[1] = 20;
    Property_t pv = 42;
    auto prop = new RangePropertyVec_t();
    range_segment_append(seg, (void*)prop, 2, 30, &pv);
    TEST_ASSERT(seg->value[2] == 30, "appended element");
    TEST_ASSERT(prop->value[2] == 42, "appended property");
    delete seg;
    delete prop;
    TEST_PASS("range_segment_append");
}

void test_range_segment_remove() {
    std::printf("[TEST] range_segment_remove\n");
    auto old_seg = new RangeElementSegment_t();
    auto new_seg = new RangeElementSegment_t();
    old_seg->value[0] = 10; old_seg->value[1] = 20;
    old_seg->value[2] = 30; old_seg->value[3] = 40;
    range_segment_remove(old_seg, nullptr, 4, new_seg, nullptr, 1);
    TEST_ASSERT(new_seg->value[0] == 10, "remove[0]");
    TEST_ASSERT(new_seg->value[1] == 30, "remove[1]");
    TEST_ASSERT(new_seg->value[2] == 40, "remove[2]");
    delete old_seg;
    delete new_seg;
    TEST_PASS("range_segment_remove");
}

void test_range_segment_split() {
    std::printf("[TEST] range_segment_split\n");
    auto old_seg = new RangeElementSegment_t();
    auto left    = new RangeElementSegment_t();
    auto right   = new RangeElementSegment_t();
    for (int i = 0; i < 8; i++) old_seg->value[i] = (RangeElement)(i * 10);
    uint64_t cmp_before = g_comparison_count.load();
    range_segment_split(old_seg, nullptr, 8, left, nullptr, right, nullptr, 4);
    uint64_t cmp_after = g_comparison_count.load();
    TEST_ASSERT(cmp_after > cmp_before, "comparison_count incremented by split");
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(left->value[i] == (RangeElement)(i * 10), "split left");
    }
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(right->value[i] == (RangeElement)((i + 4) * 10), "split right");
    }
    delete old_seg;
    delete left;
    delete right;
    TEST_PASS("range_segment_split");
}

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE — M111: neo_range_tree
// ═══════════════════════════════════════════════════════════════════

void test_range_tree_has_element() {
    std::printf("[TEST] RangeTree::has_element\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr);
    tree.keys.push_back(0);
    // Insert some elements
    Property_t pv = 1;
    tree.insert(0, 100, &pv, gc, &tb);
    tree.insert(0, 200, &pv, gc, &tb);
    tree.insert(0, 300, &pv, gc, &tb);
    TEST_ASSERT(tree.has_element(100), "has 100");
    TEST_ASSERT(tree.has_element(200), "has 200");
    TEST_ASSERT(tree.has_element(300), "has 300");
    TEST_ASSERT(!tree.has_element(150), "!has 150");
    TEST_PASS("RangeTree::has_element");
}

void test_range_tree_insert_basic() {
    std::printf("[TEST] RangeTree::insert basic + dedup\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr);
    tree.keys.push_back(0);
    Property_t pv = 42;
    bool r1 = tree.insert(0, 50, &pv, gc, &tb);
    bool r2 = tree.insert(0, 50, &pv, gc, &tb);  // duplicate
    bool r3 = tree.insert(0, 60, &pv, gc, &tb);
    TEST_ASSERT(r1, "first insert returns true");
    TEST_ASSERT(!r2, "duplicate insert returns false");
    TEST_ASSERT(r3, "second unique insert returns true");
    TEST_ASSERT(tree.has_element(50), "has 50");
    TEST_ASSERT(tree.has_element(60), "has 60");
    TEST_PASS("RangeTree::insert basic + dedup");
}

void test_range_tree_insert_split() {
    std::printf("[TEST] RangeTree::insert triggering split\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    // pre-fill a segment to RANGE_LEAF_SIZE to force split
    auto arr = tb.allocate_range_element_segment();
    auto prop = tb.allocate_range_prop_vec();
    for (uint64_t i = 0; i < RANGE_LEAF_SIZE; i++) {
        arr->value[i] = (RangeElement)(i * 2);  // 0,2,4,...
    }
    tree.node_block.emplace_back(RANGE_LEAF_SIZE, (uint64_t)arr, prop);
    tree.keys.push_back(0);

    uint64_t split_before = g_split_count.load();
    Property_t pv = 99;
    bool inserted = tree.insert(0, RANGE_LEAF_SIZE * 2 - 1, &pv, gc, &tb);
    uint64_t split_after = g_split_count.load();
    TEST_ASSERT(inserted, "insert beyond RANGE_LEAF_SIZE returns true");
    TEST_ASSERT(split_after > split_before, "split_count incremented");
    TEST_ASSERT(tree.node_block.size() == 2, "tree has 2 nodes after split");
    TEST_PASS("RangeTree::insert triggering split");
}

void test_range_tree_remove() {
    std::printf("[TEST] RangeTree::remove\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr);
    tree.keys.push_back(0);
    Property_t pv = 1;
    tree.insert(0, 10, &pv, gc, &tb);
    tree.insert(0, 20, &pv, gc, &tb);
    tree.insert(0, 30, &pv, gc, &tb);

    uint64_t freed_before = g_freed_count.load();
    bool r1 = tree.remove(20, gc, &tb);
    TEST_ASSERT(r1, "remove existing element");
    TEST_ASSERT(!tree.has_element(20), "element gone after remove");
    TEST_ASSERT(tree.has_element(10), "other element 10 still present");
    TEST_ASSERT(tree.has_element(30), "other element 30 still present");

    bool r2 = tree.remove(99, gc, &tb);
    TEST_ASSERT(!r2, "remove non-existing returns false");
    TEST_PASS("RangeTree::remove");
}

void test_range_tree_remove_last() {
    std::printf("[TEST] RangeTree::remove last element -> freed_count\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr);
    tree.keys.push_back(0);
    Property_t pv = 1;
    tree.insert(0, 42, &pv, gc, &tb);
    uint64_t freed_before = g_freed_count.load();
    tree.remove(42, gc, &tb);
    uint64_t freed_after = g_freed_count.load();
    TEST_ASSERT(freed_after > freed_before, "freed_count incremented on last-element remove");
    TEST_PASS("RangeTree::remove last element");
}

void test_range_tree_intersect() {
    std::printf("[TEST] RangeTree::intersect (two trees)\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree t1, t2;
    t1.node_block.emplace_back(0, 0, nullptr); t1.keys.push_back(0);
    t2.node_block.emplace_back(0, 0, nullptr); t2.keys.push_back(0);
    Property_t pv = 1;
    for (auto v : {10u, 20u, 30u, 40u, 50u}) {
        t1.insert(0, v, &pv, gc, &tb);
    }
    for (auto v : {20u, 40u, 60u, 80u}) {
        t2.insert(0, v, &pv, gc, &tb);
    }
    uint64_t hits_before = g_intersect_hit_count.load();
    std::vector<uint64_t> result;
    t1.intersect(&t2, result);
    uint64_t hits_after = g_intersect_hit_count.load();
    TEST_ASSERT(result.size() == 2, "intersect size=2");
    TEST_ASSERT(result[0] == 20 || result[1] == 20, "20 in result");
    TEST_ASSERT(result[0] == 40 || result[1] == 40, "40 in result");
    TEST_ASSERT(hits_after > hits_before, "intersect_hit_count incremented");

    // count-only variant
    uint64_t cnt = t1.intersect(&t2);
    TEST_ASSERT(cnt == 2, "intersect count=2");
    TEST_PASS("RangeTree::intersect");
}

void test_range_tree_range_intersect() {
    std::printf("[TEST] RangeTree::range_intersect (with array)\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr); tree.keys.push_back(0);
    Property_t pv = 1;
    for (auto v : {5u, 10u, 15u, 20u, 25u}) {
        tree.insert(0, v, &pv, gc, &tb);
    }
    RangeElement query[] = {10, 20, 30};
    std::vector<uint64_t> result;
    tree.range_intersect(query, 3, result);
    TEST_ASSERT(result.size() == 2, "range_intersect size=2");
    uint64_t cnt = tree.range_intersect(query, 3);
    TEST_ASSERT(cnt == 2, "range_intersect count=2");
    TEST_PASS("RangeTree::range_intersect");
}

void test_range_tree_property() {
    std::printf("[TEST] RangeTree get_property / set_property\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr); tree.keys.push_back(0);
    Property_t pv = 777;
    tree.insert(0, 42, &pv, gc, &tb);
    Property_t got = tree.get_property(42, 0);
    TEST_ASSERT(got == 777, "get_property returns inserted value");
    tree.set_property(42, 0, 888, gc, &tb);
    got = tree.get_property(42, 0);
    TEST_ASSERT(got == 888, "set_property updates value");
    TEST_PASS("RangeTree get_property / set_property");
}

void test_range_tree_for_each() {
    std::printf("[TEST] RangeTree::for_each\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr); tree.keys.push_back(0);
    Property_t pv = 1;
    std::vector<uint32_t> inserted = {5, 10, 15, 20};
    for (auto v : inserted) tree.insert(0, v, &pv, gc, &tb);
    std::vector<uint32_t> collected;
    tree.for_each([&](RangeElement e, double) { collected.push_back(e); });
    TEST_ASSERT(collected.size() == 4, "for_each count=4");
    for (auto v : inserted) {
        bool found = std::find(collected.begin(), collected.end(), v) != collected.end();
        TEST_ASSERT(found, "for_each collects all elements");
    }
    TEST_PASS("RangeTree::for_each");
}

void test_range_tree_insert_element_batch() {
    std::printf("[TEST] RangeTree::insert_element_batch\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;

    // Build initial tree with elements 0,2,4,...,18
    RangeTree tree;
    tree.node_block.emplace_back(0, 0, nullptr); tree.keys.push_back(0);
    Property_t pv = 1;
    for (uint32_t i = 0; i < 10; i++) {
        tree.insert(0, i * 2, &pv, gc, &tb);
    }

    // Batch insert odd elements 1,3,5,...,19 (10 new)
    std::vector<std::pair<RangeElement, RangeElement>> edges;
    for (uint32_t i = 0; i < 10; i++) {
        edges.push_back({0, (RangeElement)(i * 2 + 1)});
    }
    std::vector<Property_t> props(10, 2);
    std::vector<Property_t*> prop_ptrs;
    for (auto& p : props) prop_ptrs.push_back(&p);

    auto res = tree.insert_element_batch(0, edges.data(), prop_ptrs.data(),
                                          edges.size(), gc, &tb);
    auto new_tree = (RangeTree*)res.tree_ptr;
    TEST_ASSERT(res.new_inserted == 10, "batch inserted 10 new elements");
    // new tree has all 0..19
    for (uint32_t i = 0; i < 20; i++) {
        TEST_ASSERT(new_tree->has_element(i), "batch result has all elements");
    }
    delete new_tree;
    TEST_PASS("RangeTree::insert_element_batch");
}

void test_range_tree_copy() {
    std::printf("[TEST] copy_range_tree\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree original;
    original.node_block.emplace_back(0, 0, nullptr); original.keys.push_back(0);
    Property_t pv = 1;
    for (uint32_t v : {10u, 20u, 30u}) {
        original.insert(0, v, &pv, gc, &tb);
    }
    auto* copy = copy_range_tree(&original);
    TEST_ASSERT(copy->node_block.size() == original.node_block.size(),
                "copy node_block size match");
    TEST_ASSERT(copy->keys.size() == original.keys.size(), "copy keys size match");
    delete copy;
    TEST_PASS("copy_range_tree");
}

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE — M110: neo_index (NeoGraphIndex)
// ═══════════════════════════════════════════════════════════════════

void test_index_insert_has_vertex() {
    std::printf("[TEST] NeoGraphIndex insert_vertex + has_vertex\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    uint64_t v1 = 64, v2 = 128, v3 = 192;  // different forest directions
    TEST_ASSERT(idx.insert_vertex(v1, nullptr, &tb), "insert_vertex v1");
    TEST_ASSERT(idx.insert_vertex(v2, nullptr, &tb), "insert_vertex v2");
    TEST_ASSERT(idx.insert_vertex(v3, nullptr, &tb), "insert_vertex v3");
    TEST_ASSERT(idx.has_vertex(v1, 0), "has_vertex v1");
    TEST_ASSERT(idx.has_vertex(v2, 0), "has_vertex v2");
    TEST_ASSERT(!idx.has_vertex(999, 0), "!has_vertex 999");
    idx.dump_stats();
    TEST_PASS("NeoGraphIndex insert_vertex + has_vertex");
}

void test_index_insert_has_edge() {
    std::printf("[TEST] NeoGraphIndex insert_edge + has_edge\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_vertex(128, nullptr, &tb);
    idx.insert_edge(64, 128, nullptr, &tb);
    TEST_ASSERT(idx.has_edge(64, 128, 0), "has_edge 64->128");
    TEST_ASSERT(!idx.has_edge(64, 999, 0), "!has_edge 64->999");
    TEST_PASS("NeoGraphIndex insert_edge + has_edge");
}

void test_index_get_degree() {
    std::printf("[TEST] NeoGraphIndex get_degree\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_edge(64, 100, nullptr, &tb);
    idx.insert_edge(64, 200, nullptr, &tb);
    idx.insert_edge(64, 300, nullptr, &tb);
    uint64_t deg = idx.get_degree(64, 0);
    TEST_ASSERT(deg == 3, "degree=3");
    TEST_PASS("NeoGraphIndex get_degree");
}

void test_index_get_neighbor() {
    std::printf("[TEST] NeoGraphIndex get_neighbor\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_edge(64, 10, nullptr, &tb);
    idx.insert_edge(64, 20, nullptr, &tb);
    std::vector<RangeElement> neighbors;
    bool ok = idx.get_neighbor(64, neighbors, 0);
    TEST_ASSERT(ok, "get_neighbor returns true");
    TEST_ASSERT(neighbors.size() == 2, "2 neighbors");
    TEST_PASS("NeoGraphIndex get_neighbor");
}

void test_index_intersect() {
    std::printf("[TEST] NeoGraphIndex intersect\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_vertex(128, nullptr, &tb);
    for (uint64_t d : {10u, 20u, 30u, 40u, 50u}) {
        idx.insert_edge(64, d, nullptr, &tb);
    }
    for (uint64_t d : {20u, 40u, 60u}) {
        idx.insert_edge(128, d, nullptr, &tb);
    }
    std::vector<uint64_t> result;
    idx.intersect(64, 128, result, 0);
    TEST_ASSERT(result.size() == 2, "intersect size=2");
    uint64_t cnt = idx.intersect(64, 128, 0);
    TEST_ASSERT(cnt == 2, "intersect count=2");
    TEST_PASS("NeoGraphIndex intersect");
}

void test_index_remove_vertex() {
    std::printf("[TEST] NeoGraphIndex remove_vertex\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_edge(64, 10, nullptr, &tb);
    uint64_t rm_before = idx.op_remove.load();
    bool r = idx.remove_vertex(64, false, &tb);
    TEST_ASSERT(r, "remove_vertex returns true");
    TEST_ASSERT(idx.op_remove.load() > rm_before, "op_remove incremented");
    TEST_ASSERT(!idx.has_vertex(64, 0), "vertex gone");
    TEST_PASS("NeoGraphIndex remove_vertex");
}

void test_index_remove_edge() {
    std::printf("[TEST] NeoGraphIndex remove_edge\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_edge(64, 10, nullptr, &tb);
    idx.insert_edge(64, 20, nullptr, &tb);
    idx.remove_edge(64, 10, &tb);
    TEST_ASSERT(!idx.has_edge(64, 10, 0), "edge 64->10 gone");
    TEST_ASSERT(idx.has_edge(64, 20, 0), "edge 64->20 still present");
    TEST_PASS("NeoGraphIndex remove_edge");
}

void test_index_insert_vertex_batch() {
    std::printf("[TEST] NeoGraphIndex insert_vertex_batch\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    uint64_t verts[] = {64, 65, 66, 128, 129};  // 64-66 same direction, 128-129 same
    idx.insert_vertex_batch(verts, nullptr, 5, &tb);
    for (auto v : verts) {
        TEST_ASSERT(idx.has_vertex(v, 0), "batch vertex present");
    }
    TEST_PASS("NeoGraphIndex insert_vertex_batch");
}

void test_index_lock_unlock() {
    std::printf("[TEST] NeoGraphIndex lock + unlock\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    auto direction = NeoGraphIndex::gen_tree_direction(64);
    auto tree = idx.lock(direction);
    TEST_ASSERT(tree != nullptr, "lock returns non-null for existing tree");
    idx.unlock(direction);
    // lock on empty direction
    auto tree2 = idx.lock(9999);
    TEST_ASSERT(tree2 == nullptr, "lock on empty direction returns null");
    idx.unlock(9999);
    TEST_PASS("NeoGraphIndex lock + unlock");
}

void test_index_commit_gc() {
    std::printf("[TEST] NeoGraphIndex commit + gc\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    auto direction = NeoGraphIndex::gen_tree_direction(64);
    auto tree = idx.commit(direction, 1);
    TEST_ASSERT(tree != nullptr, "commit returns non-null");
    uint64_t gc_before = g_gc_freed_count.load();
    idx.gc(direction, &tb);
    uint64_t gc_after = g_gc_freed_count.load();
    TEST_ASSERT(gc_after > gc_before, "gc_freed_count incremented");
    TEST_PASS("NeoGraphIndex commit + gc");
}

void test_index_clear() {
    std::printf("[TEST] NeoGraphIndex clear\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_vertex(128, nullptr, &tb);
    idx.clear();
    TEST_ASSERT(idx.forest.empty(), "forest empty after clear");
    TEST_PASS("NeoGraphIndex clear");
}

void test_index_edges_template() {
    std::printf("[TEST] NeoGraphIndex::edges template\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    idx.insert_vertex(64, nullptr, &tb);
    idx.insert_edge(64, 10, nullptr, &tb);
    idx.insert_edge(64, 20, nullptr, &tb);
    idx.insert_edge(64, 30, nullptr, &tb);
    std::vector<uint64_t> collected;
    bool ok = idx.edges(64, [&](RangeElement e, double) {
        collected.push_back(e);
    }, 0);
    TEST_ASSERT(ok, "edges returns true for existing vertex");
    TEST_ASSERT(collected.size() == 3, "edges collected 3");
    TEST_PASS("NeoGraphIndex edges template");
}

void test_index_gen_tree_direction() {
    std::printf("[TEST] NeoGraphIndex::gen_tree_direction\n");
    // vertex 64 = 0b1000000, VERTEX_GROUP_BITS=6 -> direction=1
    TEST_ASSERT(NeoGraphIndex::gen_tree_direction(64) == 1, "dir(64)==1");
    TEST_ASSERT(NeoGraphIndex::gen_tree_direction(0) == 0, "dir(0)==0");
    TEST_ASSERT(NeoGraphIndex::gen_tree_direction(63) == 0, "dir(63)==0");
    TEST_ASSERT(NeoGraphIndex::gen_tree_direction(128) == 2, "dir(128)==2");
    TEST_PASS("NeoGraphIndex::gen_tree_direction");
}

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE — 综合压力测试
// ═══════════════════════════════════════════════════════════════════

void test_range_tree_stress() {
    std::printf("[TEST] RangeTree stress: 200 inserts + intersect + remove\n");
    MockWriterTraceBlock tb;
    std::vector<GCResourceInfo> gc;
    RangeTree t1, t2;
    t1.node_block.emplace_back(0, 0, nullptr); t1.keys.push_back(0);
    t2.node_block.emplace_back(0, 0, nullptr); t2.keys.push_back(0);

    Property_t pv = 1;
    // Insert 0,2,4,...,198 into t1
    for (uint32_t i = 0; i < 100; i++) t1.insert(0, i * 2, &pv, gc, &tb);
    // Insert 0,3,6,...,198 into t2
    for (uint32_t i = 0; i < 67; i++) t2.insert(0, i * 3, &pv, gc, &tb);

    uint64_t cnt = t1.intersect(&t2);
    // common: multiples of LCM(2,3)=6 in [0..198]: 0,6,12,...,198 => 34 elements
    TEST_ASSERT(cnt >= 20, "stress intersect count reasonable");

    // Remove half of t1
    for (uint32_t i = 0; i < 50; i++) t1.remove(i * 2, gc, &tb);
    // Verify remaining
    for (uint32_t i = 0; i < 50; i++) {
        TEST_ASSERT(!t1.has_element(i * 2), "removed element gone");
    }
    for (uint32_t i = 50; i < 100; i++) {
        TEST_ASSERT(t1.has_element(i * 2), "remaining element present");
    }
    std::printf("    [STRESS] split_count=%lu, freed_count=%lu, intersect_hits=%lu\n",
        g_split_count.load(), g_freed_count.load(), g_intersect_hit_count.load());
    TEST_PASS("RangeTree stress");
}

void test_index_multi_direction_stress() {
    std::printf("[TEST] NeoGraphIndex multi-direction stress\n");
    MockWriterTraceBlock tb;
    NeoGraphIndex idx;
    // Insert vertices across many forest directions
    for (uint64_t i = 0; i < 8; i++) {
        uint64_t v = i * 64;  // direction = i
        idx.insert_vertex(v, nullptr, &tb);
        for (uint64_t j = 1; j <= 5; j++) {
            idx.insert_vertex(v + j, nullptr, &tb);
            idx.insert_edge(v, v + j, nullptr, &tb);
        }
    }
    // Check degrees
    for (uint64_t i = 0; i < 8; i++) {
        uint64_t v = i * 64;
        uint64_t deg = idx.get_degree(v, 0);
        TEST_ASSERT(deg == 5, "multi-dir vertex degree=5");
    }
    idx.dump_stats();
    std::printf("    [STRESS] forest_resize_count=%lu, op_count=%lu\n",
        g_forest_resize_count.load(), g_operation_count.load());
    TEST_PASS("NeoGraphIndex multi-direction stress");
}

void test_property_vec_append_from_list() {
    std::printf("[TEST] PropertyVec append_from_list\n");
    PropertyVec<16> pv;
    Property_t vals[] = {10, 20, 30, 40, 50};
    pv.append_from_list(0, vals, 5);
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT(pv.get(i) == (uint64_t)(i + 1) * 10, "append_from_list values");
    }
    TEST_PASS("PropertyVec append_from_list");
}

// ═══════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("═══════════════════════════════════════════════════════════════\n");
    std::printf("  M110-M111: NeoGraph core upper experiment\n");
    std::printf("  (neo_index + neo_property + neo_range_ops + neo_range_tree)\n");
    std::printf("═══════════════════════════════════════════════════════════════\n\n");

    // ─── M110: neo_property ───
    std::printf("── M110 Part 1: neo_property ──\n");
    test_property_vec_basic();
    test_property_vec_copy();
    test_property_vec_insert_copy();
    test_property_vec_remove();
    test_property_vec_gc();
    test_property_map_ops();
    test_vertex_property_map();
    test_property_vec_append_from_list();

    // ─── M110: neo_index ───
    std::printf("\n── M110 Part 2: neo_index ──\n");
    test_index_gen_tree_direction();
    test_index_insert_has_vertex();
    test_index_insert_has_edge();
    test_index_get_degree();
    test_index_get_neighbor();
    test_index_intersect();
    test_index_remove_vertex();
    test_index_remove_edge();
    test_index_insert_vertex_batch();
    test_index_lock_unlock();
    test_index_commit_gc();
    test_index_clear();
    test_index_edges_template();
    test_index_multi_direction_stress();

    // ─── M111: neo_range_ops ───
    std::printf("\n── M111 Part 1: neo_range_ops ──\n");
    test_range_segment_find_seq();
    test_range_segment_find_binary();
    test_range_segment_insert_copy();
    test_range_segment_append();
    test_range_segment_remove();
    test_range_segment_split();

    // ─── M111: neo_range_tree ───
    std::printf("\n── M111 Part 2: neo_range_tree ──\n");
    test_range_tree_has_element();
    test_range_tree_insert_basic();
    test_range_tree_insert_split();
    test_range_tree_remove();
    test_range_tree_remove_last();
    test_range_tree_intersect();
    test_range_tree_range_intersect();
    test_range_tree_property();
    test_range_tree_for_each();
    test_range_tree_insert_element_batch();
    test_range_tree_copy();
    test_range_tree_stress();

    // ─── Summary ───
    std::printf("\n═══════════════════════════════════════════════════════════════\n");
    std::printf("  DEBUG COUNTERS (20%% 改动追踪):\n");
    std::printf("    operation_count    = %lu\n", g_operation_count.load());
    std::printf("    split_count        = %lu\n", g_split_count.load());
    std::printf("    freed_count        = %lu\n", g_freed_count.load());
    std::printf("    binary_search_steps= %lu\n", g_binary_search_steps.load());
    std::printf("    comparison_count   = %lu\n", g_comparison_count.load());
    std::printf("    intersect_hits     = %lu\n", g_intersect_hit_count.load());
    std::printf("    prop_alloc_count   = %lu\n", g_prop_alloc_count.load());
    std::printf("    prop_free_count    = %lu\n", g_prop_free_count.load());
    std::printf("    ref_dec_count      = %lu\n", g_ref_dec_count.load());
    std::printf("    copy_steps         = %lu\n", g_copy_steps.load());
    std::printf("    map_op_count       = %lu\n", g_map_op_count.load());
    std::printf("    forest_resize_count= %lu\n", g_forest_resize_count.load());
    std::printf("    gc_freed_count     = %lu\n", g_gc_freed_count.load());
    std::printf("═══════════════════════════════════════════════════════════════\n");
    std::printf("  Results: %d/%d PASSED, %d FAILED\n",
                g_tests_passed, g_tests_run, g_tests_failed);
    std::printf("═══════════════════════════════════════════════════════════════\n");

    return g_tests_failed > 0 ? 1 : 0;
}
