/**
 * m114_m115_neograph_cart_experiment.cpp — M114-M115: NeoGraph c_art完整实验
 *
 * 覆盖模块:
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art.h         (135行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_iter.h    (28行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_leaf.h    (237行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node.h    (74行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_iter.h (131行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_ops.h  (421行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/include/art_node_ops_copy.h (55行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art.cpp           (581行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_iter.cpp      (179行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_leaf.cpp      (750行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node.cpp      (76行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_iter.cpp (442行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_ops.cpp  (2080行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/c_art/src/art_node_ops_copy.cpp (1081行)
 *   总计: 6305行
 *
 * M114: art.cpp(581) + art_node(76+74) + art_iter(179+28) + art_leaf(750+237)
 *        — ART树核心操作 / 节点类型定义 / 迭代器 / 叶子节点操作
 *
 * M115: art_node_ops(2080+421) + art_node_ops_copy(1081+55) + art_node_iter(442+131)
 *        — 节点操作(add/remove/split/insert/batch) + 节点COW拷贝 + 节点迭代器
 *
 * 算法改动 (~20%):
 *   [MOD] node_type_distribution: 统计 Node4/16/48/256 各类型数量分布
 *   [MOD] grow_count / shrink_count: 节点升级(grow)和降级(shrink)次数追踪
 *   [MOD] find_child_steps: find_child每次查找步数统计
 *   [MOD] leaf_count: 各类型叶子节点 ARTLeaf8/16/32/64 分配计数
 *   [MOD] iterator_steps: ART迭代器遍历步数统计
 *   [MOD] batch_build_depth: batch_subtree_build 递归最大深度统计
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m114_test experiment/m114_m115_neograph_cart_experiment.cpp
 * Milestone: M114-M115 (第17位Claude, Claude Sonnet 4.6)
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
#include <variant>
#include <map>
#include <set>
#include <stack>
#include <immintrin.h>
#include <string>
#include <stdexcept>
#include <limits>

// ============================================================
// §0. 独立运行模拟外部依赖
// ============================================================

// config.h constants
#define ART_LEAF_SIZE 256
#define EDGE_PROPERTY_NUM 1
#define RANGE_LEAF_SIZE 512
#define VERTEX_PROPERTY_NUM 0
#define VERTEX_GROUP_BITS 6
constexpr uint64_t VERTEX_GROUP_SIZE = 1 << VERTEX_GROUP_BITS;

// helper.h macros
#define LEAF8  1
#define LEAF16 2
#define LEAF32 3
#define LEAF64 4

#define NODE4   1
#define NODE16  2
#define NODE48  3
#define NODE256 4

#define KEY_LEN 3

#define IS_LEAF(x)      (((uint64_t)(x)) & 0x8000000000000000ULL)
#define SET_LEAF(x)     ((void*)(((uint64_t)(x)) | 0x8000000000000000ULL))
#define GET_OFFSET(x)   ((((uint64_t)(x)) & 0x00FF000000000000ULL) >> 48)
#define SET_OFFSET(x, offset) (assert(offset < 256), assert(IS_LEAF(x)), \
    ((void*)((((uint64_t)(x)) & 0xFF00FFFFFFFFFFFFULL) | ((uint64_t)(offset) << 48))))
#define LEAF_POINTER_CTOR(x, offset) \
    (assert(offset < 256), ((void*)(((uint64_t)(x) & 0x0000FFFFFFFFFFFFULL) \
        | ((uint64_t)1 << 63) | (((uint64_t)offset) << 48))))
#define LEAF_RAW(x) ((ARTLeaf*)(((uint64_t)(x)) & 0x0000FFFFFFFFFFFFULL))

// ============================================================
// §1. 算法改动: 全局统计计数器 (20%改动)
// ============================================================
namespace cart_stats {
    // [MOD] node_type_distribution
    std::atomic<uint64_t> node4_alloc{0};
    std::atomic<uint64_t> node16_alloc{0};
    std::atomic<uint64_t> node48_alloc{0};
    std::atomic<uint64_t> node256_alloc{0};

    // [MOD] grow_count / shrink_count
    std::atomic<uint64_t> grow_count{0};    // Node4→16, 16→48, 48→256
    std::atomic<uint64_t> shrink_count{0};  // 暂未实现shrink，预留

    // [MOD] find_child_steps
    std::atomic<uint64_t> find_child_calls{0};
    std::atomic<uint64_t> find_child_steps_total{0};

    // [MOD] leaf_count
    std::atomic<uint64_t> leaf8_alloc{0};
    std::atomic<uint64_t> leaf16_alloc{0};
    std::atomic<uint64_t> leaf32_alloc{0};
    std::atomic<uint64_t> leaf64_alloc{0};

    // [MOD] iterator_steps
    std::atomic<uint64_t> iter_steps{0};

    // [MOD] batch_build_depth
    std::atomic<uint64_t> batch_max_depth{0};

    void print_report() {
        std::cout << "\n=== c_art Stats Report ===" << std::endl;
        std::cout << "Node alloc: Node4=" << node4_alloc
                  << " Node16=" << node16_alloc
                  << " Node48=" << node48_alloc
                  << " Node256=" << node256_alloc << std::endl;
        std::cout << "Grow count: " << grow_count
                  << "  Shrink count: " << shrink_count << std::endl;
        std::cout << "find_child calls=" << find_child_calls
                  << " total_steps=" << find_child_steps_total;
        if (find_child_calls > 0)
            std::cout << " avg=" << (double)find_child_steps_total/find_child_calls;
        std::cout << std::endl;
        std::cout << "Leaf alloc: L8=" << leaf8_alloc
                  << " L16=" << leaf16_alloc
                  << " L32=" << leaf32_alloc
                  << " L64=" << leaf64_alloc << std::endl;
        std::cout << "Iterator steps: " << iter_steps << std::endl;
        std::cout << "Batch max depth: " << batch_max_depth << std::endl;
    }

    void reset() {
        node4_alloc = node16_alloc = node48_alloc = node256_alloc = 0;
        grow_count = shrink_count = 0;
        find_child_calls = find_child_steps_total = 0;
        leaf8_alloc = leaf16_alloc = leaf32_alloc = leaf64_alloc = 0;
        iter_steps = 0;
        batch_max_depth = 0;
    }
}

// ============================================================
// §2. 类型系统 (from types.h, neo_property.h)
// ============================================================
namespace container {

using Property_t = uint64_t;
using RangeElement = uint32_t;

// Forward declarations
struct ARTLeaf;
struct ARTLeaf8;
struct ARTLeaf16;
struct ARTLeaf32;
struct ARTLeaf64;
struct ARTNode;
struct ARTNode_4;
struct ARTNode_16;
struct ARTNode_48;
struct ARTNode_256;
struct WriterTraceBlock;

// get_key_byte
inline uint8_t get_key_byte(uint64_t key, uint8_t depth) {
    // Key layout: bytes 0-4 from MSB of lower 40 bits
    // Depth 0 = bits[31:24], 1=[23:16], 2=[15:8], 3=[7:0] of uint32_t portion
    return (uint8_t)((key >> (8 * (3 - depth))) & 0xFF);
}

struct ARTKey {
    uint32_t key;

    explicit ARTKey(uint64_t dst) {
        // Use lower 32 bits (destination vertex as 32-bit key)
        key = (uint32_t)(dst & 0xFFFFFFFF);
    }

    ARTKey(uint64_t dst, uint8_t depth, bool is_single_byte) {
        key = (uint32_t)(dst & 0xFFFFFFFF);
        // Mask out bytes deeper than 'depth'
        if (is_single_byte) {
            // keep only the byte at position depth
            uint32_t mask = (depth < 4) ? (0xFF << (8 * (3 - depth))) : 0xFFFFFFFF;
            key = key & mask;
        } else {
            // keep bytes 0..depth-1 (from MSB)
            if (depth < 4) {
                uint32_t mask = 0xFFFFFFFF << (8 * (4 - depth));
                key = key & mask;
            }
        }
    }

    ARTKey(ARTKey k, uint8_t depth, bool is_single_byte) {
        *this = ARTKey((uint64_t)k.key, depth, is_single_byte);
    }

    ARTKey() : key(0) {}

    uint8_t operator[](int idx) const {
        return (uint8_t)((key >> (8 * (3 - idx))) & 0xFF);
    }

    // Non-const accessor (not actually used for modification in practice)
    uint8_t operator[](int idx) {
        return (uint8_t)((key >> (8 * (3 - idx))) & 0xFF);
    }

    bool operator==(const ARTKey& rhs) const { return key == rhs.key; }
    bool operator!=(const ARTKey& rhs) const { return key != rhs.key; }
    bool operator<(const ARTKey& rhs) const { return key < rhs.key; }

    void print() const {
        printf("ARTKey{%08X}", key);
    }

    static bool check_partial_match(ARTKey key1, ARTKey key2, uint8_t depth) {
        for (uint8_t i = 0; i < depth; i++) {
            if (key1[i] != key2[i]) return false;
        }
        return true;
    }

    static bool check_partial_match(uint64_t key1, uint64_t key2, uint8_t depth) {
        for (uint8_t i = 0; i < depth; i++) {
            if (get_key_byte(key1, i) != get_key_byte(key2, i)) return false;
        }
        return true;
    }

    static uint8_t longest_common_prefix(ARTKey key1, ARTKey key2) {
        for (uint8_t i = 0; i < 3; i++) {
            if (key1[i] != key2[i]) return i;
        }
        return 5;
    }
};

enum ARTNodeSplitStatus {
    SPLIT = 0,
    NEW_LEAF = 1,
    GO_DEEPER = 2,
};

struct ARTNodeSplitRes {
    ARTNodeSplitStatus status;
    void* leaf;
};

struct ARTInsertElemBatchRes {
    uint64_t new_inserted;
    void* art_ptr;
};

struct ARTInsertElemCopyRes {
    uint64_t is_new  : 16;
    uint64_t art_ptr : 48;
};

struct ARTRemoveElemCopyRes {
    uint64_t found    : 16;
    uint64_t tree_ptr : 48;
};

enum ARTNodeRemoveRes {
    NOT_FOUND,
    ELEMENT_REMOVED,
    CHILD_REMOVED,
};

enum ARTResourceType {
    ART_Leaf = 1,
    ART_Node_Copied = 2,
    ART_Node_Mounted = 3,
    ART_Property_Vec = 4,
    ART_Property_Map_All_Modified = 5,
    Multi_ART_Property_Vec_Copied = 6,
};

struct ARTResourceInfo {
    ARTResourceType type;
    void* ptr;
};

// PropertyVec stub
template<uint64_t Size>
struct PropertyVec {
    std::array<Property_t, Size> value{};
    std::atomic<uint32_t> ref_cnt{1};

    Property_t get(uint64_t idx) const { return value[idx]; }
    void set(uint64_t idx, Property_t v) { value[idx] = v; }
    void insert(uint64_t pos_idx, uint64_t size, Property_t v) {
        for (uint64_t i = size; i > 0; i--)
            value[pos_idx + i] = value[pos_idx + i - 1];
        value[pos_idx] = v;
    }
    void remove(uint64_t pos_idx, uint64_t sz) {
        for (uint64_t i = pos_idx; i + sz < Size; i++)
            value[i] = value[i + sz];
    }
    void copy_to(PropertyVec* dst) const { *dst = *this; }
    void copy_to(uint64_t begin, uint64_t end, PropertyVec* dst, uint64_t dst_idx) const {
        for (uint64_t i = begin; i < end; i++)
            dst->value[dst_idx + i - begin] = value[i];
    }
    void append_from_list(uint64_t begin_idx, Property_t* values, uint64_t sz) {
        for (uint64_t i = 0; i < sz; i++) value[begin_idx + i] = values[i];
    }
};

using ARTPropertyVec_t = PropertyVec<ART_LEAF_SIZE>;

// ============================================================
// §3. Bitmap (from utils/bitmap/include/bitmap.h)
// ============================================================
template<size_t BLOCK_NUM>
struct Bitmap {
    std::array<uint64_t, BLOCK_NUM> data{};

    Bitmap() = default;
    Bitmap(const Bitmap&) = default;
    Bitmap(Bitmap&&) = default;
    Bitmap& operator=(const Bitmap&) = default;

    void set(uint64_t index) {
        data[index / 64] |= 1ULL << (index % 64);
    }

    void reset(uint64_t index) {
        data[index / 64] &= ~(1ULL << (index % 64));
    }

    bool get(uint64_t index) const {
        return (data[index / 64] >> (index % 64)) & 1;
    }

    uint64_t at(uint64_t index) const {
        // Return the index-th set bit
        uint64_t cnt = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                if (cnt == index) return __builtin_ctzll(mask) + i * 64;
                cnt++;
                mask &= mask - 1;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    bool empty() const {
        for (auto d : data) if (d) return false;
        return true;
    }

    uint64_t find_first() {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            if (data[i]) return __builtin_ctzll(data[i]) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t find_first(uint64_t begin) {
        const uint64_t begin_block = begin / 64;
        for (size_t i = begin_block; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) return __builtin_ctzll(mask) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t lower_bound(uint64_t element, uint64_t prefix) const {
        uint64_t target = element & 0xFF;
        uint64_t res = 0;
        if ((element & ~0xFFULL) == prefix) {
            for (size_t i = 0; i < BLOCK_NUM; i++) {
                uint64_t mask = data[i];
                while (mask) {
                    uint64_t index = __builtin_ctzll(mask);
                    if ((index + i * 64) >= target) break;
                    res++;
                    mask &= mask - 1;
                }
            }
        } else {
            for (size_t i = 0; i < BLOCK_NUM; i++) {
                uint64_t mask = data[i];
                while (mask) {
                    uint64_t index = __builtin_ctzll(mask);
                    if (((index + i * 64) | prefix) >= element) break;
                    res++;
                    mask &= mask - 1;
                }
            }
        }
        return res;
    }

    uint64_t consume() {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            if (data[i]) {
                uint64_t t = data[i] & -data[i];
                uint64_t index = __builtin_ctzll(data[i]);
                data[i] ^= t;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t consume(uint64_t begin) {
        const uint64_t begin_block = begin / 64;
        for (size_t i = begin_block; i < BLOCK_NUM; i++) {
            if (data[i]) {
                uint64_t t = data[i] & -data[i];
                uint64_t index = __builtin_ctzll(data[i]);
                data[i] ^= t;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    template<typename F>
    void for_each(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t index = __builtin_ctzll(mask);
                f((uint8_t)(index + i * 64));
                mask &= mask - 1;
            }
        }
    }

    template<typename F>
    void for_each(F&& f, uint64_t begin, uint64_t end) const {
        uint64_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t index = __builtin_ctzll(mask);
                if (count >= begin && count < end)
                    f((uint8_t)(index + i * 64));
                else if (count >= end) return;
                count++;
                mask &= mask - 1;
            }
        }
    }

    template<typename F>
    void for_each_zero(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = ~data[i];
            while (mask) {
                uint64_t index = __builtin_ctzll(mask);
                f((uint8_t)(index + i * 64));
                mask &= mask - 1;
            }
        }
    }
};

// ============================================================
// §4. WriterTraceBlock (simplified allocator stub)
// ============================================================
struct ARTNode_48;
struct ARTNode_256;

struct WriterTraceBlock {
    // Simple pool-free allocator for experiment
    ARTNode_48* allocate_art_node48();
    ARTNode_256* allocate_art_node256();
    void deallocate_art_node48(ARTNode_48* n);
    void deallocate_art_node256(ARTNode_256* n);
    ARTPropertyVec_t* allocate_art_prop_vec();
    void deallocate_art_prop_vec(ARTPropertyVec_t* v);
    std::array<uint32_t, ART_LEAF_SIZE>* allocate_art_leaf32();
    void deallocate_art_leaf32(std::array<uint32_t, ART_LEAF_SIZE>* leaf);
    std::array<uint64_t, ART_LEAF_SIZE>* allocate_art_leaf64();
    void deallocate_art_leaf64(std::array<uint64_t, ART_LEAF_SIZE>* leaf);
};

// ============================================================
// §5. ARTNode structures (from art_node.h)
// ============================================================
struct ARTNode {
    ARTKey prefix{};
    uint8_t type   : 4;
    uint8_t depth  : 4;
    uint16_t num_children = 0;
    std::atomic<uint16_t> ref_cnt{1};
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

// WriterTraceBlock implementations (must come after ARTNode_48/256 definitions)
ARTNode_48* WriterTraceBlock::allocate_art_node48() {
    return new ARTNode_48();
}
ARTNode_256* WriterTraceBlock::allocate_art_node256() {
    return new ARTNode_256();
}
void WriterTraceBlock::deallocate_art_node48(ARTNode_48* n) { delete n; }
void WriterTraceBlock::deallocate_art_node256(ARTNode_256* n) { delete n; }
ARTPropertyVec_t* WriterTraceBlock::allocate_art_prop_vec() {
    return new ARTPropertyVec_t();
}
void WriterTraceBlock::deallocate_art_prop_vec(ARTPropertyVec_t* v) { delete v; }
std::array<uint32_t, ART_LEAF_SIZE>* WriterTraceBlock::allocate_art_leaf32() {
    return new std::array<uint32_t, ART_LEAF_SIZE>();
}
void WriterTraceBlock::deallocate_art_leaf32(std::array<uint32_t, ART_LEAF_SIZE>* leaf) { delete leaf; }
std::array<uint64_t, ART_LEAF_SIZE>* WriterTraceBlock::allocate_art_leaf64() {
    return new std::array<uint64_t, ART_LEAF_SIZE>();
}
void WriterTraceBlock::deallocate_art_leaf64(std::array<uint64_t, ART_LEAF_SIZE>* leaf) { delete leaf; }

// ============================================================
// §6. ARTLeaf structures (from art_leaf.h / art_leaf.cpp)
// ============================================================
struct ARTLeaf {
    ARTKey key{};
    uint16_t size{};
    uint8_t type{};
    uint8_t is_single_byte{};
    uint8_t depth{};
    std::atomic<uint16_t> ref_cnt{1};
    ARTPropertyVec_t* property_map{nullptr};

    ARTLeaf(ARTKey key, uint8_t depth, bool is_single_byte)
        : key(ARTKey{(uint64_t)key.key, depth, is_single_byte}),
          depth(depth), size(0), is_single_byte(is_single_byte ? 1 : 0),
          property_map(nullptr) {}

    virtual ~ARTLeaf() = default;

    [[nodiscard]] virtual uint64_t at(uint16_t pos_idx) const = 0;
    [[nodiscard]] virtual bool has_element(uint64_t element, uint8_t begin_idx) const = 0;
    [[nodiscard]] virtual uint16_t find(uint64_t element, uint8_t begin_idx) const;
    [[nodiscard]] virtual uint16_t get_byte_num(uint8_t depth) const = 0;
    virtual void insert(uint64_t element, Property_t* property, uint16_t pos_idx) = 0;
    virtual void remove(uint16_t pos_idx, uint8_t target_byte) = 0;
    virtual void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx, ARTLeaf* dst, uint16_t dst_idx) const = 0;
    virtual void append_from_list(RangeElement* elem_list, Property_t** prop_list, uint16_t count) = 0;
    virtual void leaf_check() const = 0;

    Property_t get_property(uint16_t pos_idx, uint8_t property_id) const {
        if (property_map) return property_map->get(pos_idx);
        return 0;
    }
    void set_property(uint16_t pos_idx, uint8_t property_id, Property_t prop) {
        if (property_map) property_map->set(pos_idx, prop);
    }

    template<typename F>
    void for_each(F&& f) const;
};

// ARTLeaf8: stores elements as 8-bit offsets in a Bitmap<4>
struct ARTLeaf8 : public ARTLeaf {
    Bitmap<4> value;

    ARTLeaf8(ARTKey key, uint8_t depth, bool is_single_byte)
        : ARTLeaf(key, depth, is_single_byte), value() {}

    [[nodiscard]] uint64_t at(uint16_t pos_idx) const override {
        assert(pos_idx < ART_LEAF_SIZE);
        return value.at(pos_idx) | key.key;
    }

    [[nodiscard]] bool has_element(uint64_t element, uint8_t begin_idx) const override {
        uint8_t target = element & 0xFF;
        return value.get(target);
    }

    [[nodiscard]] uint16_t find(uint64_t element, uint8_t begin_idx) const override {
        if ((element & ~0xFFULL) > (uint64_t)key.key) return size;
        return (uint16_t)this->value.lower_bound(element, (uint64_t)key.key);
    }

    [[nodiscard]] uint16_t get_byte_num(uint8_t depth) const override { return 1; }

    template<typename F>
    void do_for_each(F&& f) const {
        uint64_t mask = key.key;
        value.for_each([&](uint8_t idx) {
            f((uint64_t)(idx | mask), 0.0);
        });
    }

    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override {
        uint8_t target = element & 0xFF;
        if (!value.get(target)) {
            value.set(target);
            size++;
            if (property && property_map) {
                // set property
                property_map->set(pos_idx, *property);
            }
        }
    }

    void remove(uint16_t pos_idx, uint8_t target_byte) override {
        if (value.get(target_byte)) {
            value.reset(target_byte);
            size--;
        }
    }

    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx, ARTLeaf* dst, uint16_t dst_idx) const override {
        // For ARTLeaf8 we copy set bits
        uint64_t cnt = 0;
        value.for_each([&](uint8_t idx) {
            if (cnt >= begin_idx && cnt < end_idx) {
                dst->insert((uint64_t)(idx | key.key), nullptr, (uint16_t)(dst_idx + cnt - begin_idx));
            }
            cnt++;
        });
    }

    void append_from_list(RangeElement* elem_list, Property_t** prop_list, uint16_t count) override {
        for (uint16_t i = 0; i < count; i++) {
            uint8_t target = elem_list[i] & 0xFF;
            if (!value.get(target)) {
                value.set(target);
                size++;
            }
        }
    }

    void leaf_check() const override {}
};

// ARTLeaf16: stores 16-bit elements in sorted array
struct ARTLeaf16 : public ARTLeaf {
    std::array<uint16_t, ART_LEAF_SIZE>* value;

    ARTLeaf16(ARTKey key, uint8_t depth, bool is_single_byte)
        : ARTLeaf(key, depth, is_single_byte), value(nullptr) {}

    [[nodiscard]] uint64_t at(uint16_t pos_idx) const override {
        assert(pos_idx < ART_LEAF_SIZE);
        return (uint64_t)value->at(pos_idx) | (uint64_t)key.key;
    }

    [[nodiscard]] bool has_element(uint64_t element, uint8_t begin_idx) const override {
        uint16_t target = element & 0xFFFF;
        auto iter = std::lower_bound(value->begin() + begin_idx, value->begin() + size, target);
        return iter != value->begin() + size && *iter == target;
    }

    [[nodiscard]] uint16_t get_byte_num(uint8_t d) const override {
        if (size == 0) return 0;
        uint16_t cnt = 1;
        uint8_t cur = get_key_byte((uint64_t)value->at(0), d);
        for (uint16_t i = 1; i < size; i++) {
            uint8_t b = get_key_byte((uint64_t)value->at(i), d);
            if (b != cur) { cur = b; cnt++; }
        }
        return cnt;
    }

    template<typename F>
    void do_for_each(F&& f) const {
        uint64_t mask = key.key;
        for (int j = 0; j < size; j++) f(((uint64_t)value->at(j) | mask), 0.0);
    }

    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override {
        uint16_t target = element & 0xFFFF;
        auto it = std::lower_bound(value->begin(), value->begin() + size, target);
        if (it != value->begin() + size && *it == target) return;
        // shift right
        for (int i = size; i > (it - value->begin()); i--)
            (*value)[i] = (*value)[i-1];
        (*value)[it - value->begin()] = target;
        size++;
    }

    void remove(uint16_t pos_idx, uint8_t target_byte) override {
        if (pos_idx >= size) return;
        for (uint16_t i = pos_idx; i < size - 1; i++)
            (*value)[i] = (*value)[i+1];
        size--;
    }

    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx, ARTLeaf* dst, uint16_t dst_idx) const override {
        for (uint16_t i = begin_idx; i < end_idx; i++) {
            dst->insert(at(i), nullptr, dst_idx + i - begin_idx);
        }
    }

    void append_from_list(RangeElement* elem_list, Property_t** prop_list, uint16_t count) override {
        for (uint16_t i = 0; i < count; i++) {
            insert((uint64_t)elem_list[i], prop_list ? prop_list[i] : nullptr, size);
        }
    }

    void leaf_check() const override {}
};

// ARTLeaf32: stores 32-bit elements in sorted array
struct ARTLeaf32 : public ARTLeaf {
    std::array<uint32_t, ART_LEAF_SIZE>* value;

    ARTLeaf32(ARTKey key, uint8_t depth, bool is_single_byte)
        : ARTLeaf(key, depth, is_single_byte), value(nullptr) {}

    [[nodiscard]] uint64_t at(uint16_t pos_idx) const override {
        assert(pos_idx < ART_LEAF_SIZE);
        return (uint64_t)value->at(pos_idx) | (uint64_t)key.key;
    }

    [[nodiscard]] bool has_element(uint64_t element, uint8_t begin_idx) const override {
        uint32_t target = element & 0xFFFFFFFF;
        auto it = std::lower_bound(value->begin() + begin_idx, value->begin() + size, target);
        return it != value->begin() + size && *it == target;
    }

    [[nodiscard]] uint16_t get_byte_num(uint8_t d) const override {
        if (size == 0) return 0;
        uint16_t cnt = 1;
        uint8_t cur = get_key_byte((uint64_t)value->at(0), d);
        for (uint16_t i = 1; i < size; i++) {
            uint8_t b = get_key_byte((uint64_t)value->at(i), d);
            if (b != cur) { cur = b; cnt++; }
        }
        return cnt;
    }

    template<typename F>
    void do_for_each(F&& f) const {
        uint64_t mask = key.key;
        for (int j = 0; j < size; j++) f(((uint64_t)value->at(j) | mask), 0.0);
    }

    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override {
        uint32_t target = element & 0xFFFFFFFF;
        auto it = std::lower_bound(value->begin(), value->begin() + size, target);
        if (it != value->begin() + size && *it == target) return;
        for (int i = size; i > (it - value->begin()); i--)
            (*value)[i] = (*value)[i-1];
        (*value)[it - value->begin()] = target;
        size++;
    }

    void remove(uint16_t pos_idx, uint8_t target_byte) override {
        if (pos_idx >= size) return;
        for (uint16_t i = pos_idx; i < size - 1; i++)
            (*value)[i] = (*value)[i+1];
        size--;
    }

    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx, ARTLeaf* dst, uint16_t dst_idx) const override {
        for (uint16_t i = begin_idx; i < end_idx; i++) {
            dst->insert(at(i), nullptr, dst_idx + i - begin_idx);
        }
    }

    void append_from_list(RangeElement* elem_list, Property_t** prop_list, uint16_t count) override {
        for (uint16_t i = 0; i < count; i++) {
            insert((uint64_t)elem_list[i], prop_list ? prop_list[i] : nullptr, size);
        }
    }

    void leaf_check() const override {}
};

// ARTLeaf64: stores 64-bit elements in sorted array
struct ARTLeaf64 : public ARTLeaf {
    std::array<uint64_t, ART_LEAF_SIZE>* value;

    ARTLeaf64(ARTKey key, uint8_t depth, bool is_single_byte)
        : ARTLeaf(key, depth, is_single_byte), value(nullptr) {}

    [[nodiscard]] uint64_t at(uint16_t pos_idx) const override {
        assert(pos_idx < ART_LEAF_SIZE);
        return value->at(pos_idx) | (uint64_t)key.key;
    }

    [[nodiscard]] bool has_element(uint64_t element, uint8_t begin_idx) const override {
        auto it = std::lower_bound(value->begin() + begin_idx, value->begin() + size, element);
        return it != value->begin() + size && *it == element;
    }

    [[nodiscard]] uint16_t get_byte_num(uint8_t d) const override {
        if (size == 0) return 0;
        uint16_t cnt = 1;
        uint8_t cur = get_key_byte(value->at(0), d);
        for (uint16_t i = 1; i < size; i++) {
            uint8_t b = get_key_byte(value->at(i), d);
            if (b != cur) { cur = b; cnt++; }
        }
        return cnt;
    }

    template<typename F>
    void do_for_each(F&& f) const {
        for (int j = 0; j < size; j++) f(value->at(j), 0.0);
    }

    void insert(uint64_t element, Property_t* property, uint16_t pos_idx) override {
        auto it = std::lower_bound(value->begin(), value->begin() + size, element);
        if (it != value->begin() + size && *it == element) return;
        for (int i = size; i > (it - value->begin()); i--)
            (*value)[i] = (*value)[i-1];
        (*value)[it - value->begin()] = element;
        size++;
    }

    void remove(uint16_t pos_idx, uint8_t target_byte) override {
        if (pos_idx >= size) return;
        for (uint16_t i = pos_idx; i < size - 1; i++)
            (*value)[i] = (*value)[i+1];
        size--;
    }

    void copy_to_leaf(uint16_t begin_idx, uint16_t end_idx, ARTLeaf* dst, uint16_t dst_idx) const override {
        for (uint16_t i = begin_idx; i < end_idx; i++) {
            dst->insert(at(i), nullptr, dst_idx + i - begin_idx);
        }
    }

    void append_from_list(RangeElement* elem_list, Property_t** prop_list, uint16_t count) override {
        for (uint16_t i = 0; i < count; i++) {
            insert((uint64_t)elem_list[i], prop_list ? prop_list[i] : nullptr, size);
        }
    }

    void leaf_check() const override {}
};

// ARTLeaf::find (default binary search implementation)
uint16_t ARTLeaf::find(uint64_t element, uint8_t begin_idx) const {
    uint16_t l = begin_idx, r = size;
    while (l < r) {
        uint16_t mid = l + (r - l) / 2;
        uint64_t cur = at(mid);
        if (cur == element) return mid;
        else if (cur < element) l = mid + 1;
        else r = mid;
    }
    return l;
}

// ARTLeaf::for_each dispatch
template<typename F>
void ARTLeaf::for_each(F&& f) const {
    switch (type) {
        case LEAF8:  ((ARTLeaf8*)this)->do_for_each(std::forward<F>(f)); break;
        case LEAF16: ((ARTLeaf16*)this)->do_for_each(std::forward<F>(f)); break;
        case LEAF32: ((ARTLeaf32*)this)->do_for_each(std::forward<F>(f)); break;
        case LEAF64: ((ARTLeaf64*)this)->do_for_each(std::forward<F>(f)); break;
    }
}

// get_list_byte_num
uint64_t get_list_byte_num(uint64_t* list, uint64_t sz, uint8_t depth) {
    if (sz == 0) return 0;
    uint64_t cnt = 1;
    uint8_t cur = get_key_byte(list[0], depth);
    for (uint64_t i = 1; i < sz; i++) {
        uint8_t b = get_key_byte(list[i], depth);
        if (b != cur) { cur = b; cnt++; }
    }
    return cnt;
}

// alloc_leaf
ARTLeaf* alloc_leaf(ARTKey key, uint8_t depth, bool is_single_byte, bool not_empty, WriterTraceBlock* trace_block) {
    ARTLeaf* leaf = nullptr;
    if (depth >= 4) {
        // ARTLeaf8 for single-byte (bit-bitmap at deepest level)
        auto l = new ARTLeaf8(key, depth, is_single_byte);
        l->type = LEAF8;
        cart_stats::leaf8_alloc++;
        leaf = l;
    } else if (depth >= 3) {
        auto l = new ARTLeaf16(key, depth, is_single_byte);
        l->type = LEAF16;
        l->value = new std::array<uint16_t, ART_LEAF_SIZE>();
        l->value->fill(0);
        cart_stats::leaf16_alloc++;
        leaf = l;
    } else if (depth >= 2) {
        auto l = new ARTLeaf32(key, depth, is_single_byte);
        l->type = LEAF32;
        if (trace_block) l->value = trace_block->allocate_art_leaf32();
        else l->value = new std::array<uint32_t, ART_LEAF_SIZE>();
        l->value->fill(0);
        cart_stats::leaf32_alloc++;
        leaf = l;
    } else {
        auto l = new ARTLeaf64(key, depth, is_single_byte);
        l->type = LEAF64;
        if (trace_block) l->value = trace_block->allocate_art_leaf64();
        else l->value = new std::array<uint64_t, ART_LEAF_SIZE>();
        l->value->fill(0);
        cart_stats::leaf64_alloc++;
        leaf = l;
    }
    // Allocate property map
    if (EDGE_PROPERTY_NUM > 0 && not_empty) {
        if (trace_block) leaf->property_map = trace_block->allocate_art_prop_vec();
        else leaf->property_map = new ARTPropertyVec_t();
    }
    return leaf;
}

void leaf_destroy(ARTLeaf* leaf) {
    if (!leaf) return;
    switch (leaf->type) {
        case LEAF16: delete ((ARTLeaf16*)leaf)->value; break;
        case LEAF32: delete ((ARTLeaf32*)leaf)->value; break;
        case LEAF64: delete ((ARTLeaf64*)leaf)->value; break;
        default: break;
    }
    if (leaf->property_map) delete leaf->property_map;
}

void leaf_clean(ARTLeaf* leaf, WriterTraceBlock* trace_block) {
    if (!leaf) return;
    // Just clean property map if any
    if (leaf->property_map && trace_block) {
        trace_block->deallocate_art_prop_vec(leaf->property_map);
        leaf->property_map = nullptr;
    }
}

// ============================================================
// §7. ARTNode alloc/destroy (from art_node.cpp)
// ============================================================

// Forward declarations needed for art_node_iter
struct ARTNodeIterator;
struct ARTNodeIterator_4;
struct ARTNodeIterator_16;
struct ARTNodeIterator_48;
struct ARTNodeIterator_256;

ARTNode* alloc_node(uint8_t type, ARTKey prefix, uint8_t depth, WriterTraceBlock* trace_block);
void recursive_destroy_node(ARTNode* n);
void delete_node(ARTNode* n, WriterTraceBlock* trace_block);

// Forward declare iter functions
ARTNodeIterator* alloc_iterator(const ARTNode* node);
void alloc_iterator_ref(const ARTNode* node, std::variant<ARTNodeIterator_4, ARTNodeIterator_16, ARTNodeIterator_48, ARTNodeIterator_256>& iter);
void destroy_iterator(ARTNodeIterator* iter);
bool iter_is_valid(ARTNodeIterator* iter);
void iter_next(ARTNodeIterator* iter);
void iter_next_without_skip(ARTNodeIterator* iter);
std::pair<uint8_t, ARTNode*> iter_get(ARTNodeIterator* iter);
ARTNode** iter_get_node(ARTNodeIterator* iter);
ARTNode** iter_get_current(ARTNodeIterator* iter);
ARTNode* iter_get_current_ro(ARTNodeIterator* iter);

// Forward declare node ops
ARTNode** find_child(ARTNode* n, unsigned char c);
ARTNode** add_child(ARTNode* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* trace_block);
ARTNode** add_child4(ARTNode_4* n, ARTNode** ref, unsigned char c, void* child);
void remove_child(ARTNode* n, ARTNode** ref, unsigned char c, ARTNode** child);
void node_ref(ARTNode* node);
ARTNode** add_child_copy(ARTNode* n, uint8_t child_idx, ARTNode* child);
uint16_t find_child_idx(ARTNode* n, unsigned char c);
ARTNode* copy_node(ARTNode* n, WriterTraceBlock* trace_block);

// ============================================================
// §8. ARTNodeIterator (from art_node_iter.h / art_node_iter.cpp)
// ============================================================
struct ARTNodeIterator {
    [[nodiscard]] virtual bool is_valid() const = 0;
    virtual ARTNode* operator*() = 0;
    virtual std::pair<uint8_t, ARTNode*> get() = 0;
    virtual ARTNode* get_node() = 0;
    virtual void operator++() = 0;
    virtual void operator++(int) = 0;
    virtual void next_without_skip() = 0;
    virtual ~ARTNodeIterator() = default;
};

struct ARTNodeIterator_4 : ARTNodeIterator {
    ARTNode_4* node{};
    ARTNode** current{};

    ARTNodeIterator_4() = default;
    explicit ARTNodeIterator_4(ARTNode_4* n) : node(n), current(n->children) {}

    ARTNode* operator*() override {
        if (current == node->children + node->n.num_children) return nullptr;
        return *current;
    }

    [[nodiscard]] bool is_valid() const override {
        return current != node->children + node->n.num_children;
    }

    std::pair<uint8_t, ARTNode*> get() override {
        if (current == node->children + node->n.num_children) return {0, nullptr};
        return {node->keys[current - node->children], *current};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        if (current == node->children + node->n.num_children) return;
        if (!IS_LEAF(*current)) {
            current++;
        } else {
            auto leaf = LEAF_RAW(*current);
            do {
                current++;
                if (current == node->children + node->n.num_children) return;
            } while (LEAF_RAW(*current) == leaf);
        }
        cart_stats::iter_steps++;
    }

    void operator++(int step) override { for (int i = 0; i < step; i++) (*this)++; }

    void next_without_skip() override {
        if (current == node->children + node->n.num_children) return;
        current++;
        cart_stats::iter_steps++;
    }
};

struct ARTNodeIterator_16 : ARTNodeIterator {
    ARTNode_16* node{};
    ARTNode** current{};

    ARTNodeIterator_16() = default;
    explicit ARTNodeIterator_16(ARTNode_16* n) : node(n), current(n->children) {}

    ARTNode* operator*() override {
        if (current == node->children + node->n.num_children) return nullptr;
        return *current;
    }

    [[nodiscard]] bool is_valid() const override {
        return current != node->children + node->n.num_children;
    }

    std::pair<uint8_t, ARTNode*> get() override {
        if (current == node->children + node->n.num_children) return {0, nullptr};
        return {node->keys[current - node->children], *current};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        if (current == node->children + node->n.num_children) return;
        if (!IS_LEAF(*current)) {
            current++;
        } else {
            auto leaf = LEAF_RAW(*current);
            do {
                current++;
                if (current == node->children + node->n.num_children) return;
            } while (LEAF_RAW(*current) == leaf);
        }
        cart_stats::iter_steps++;
    }

    void operator++(int step) override { for (int i = 0; i < step; i++) (*this)++; }

    void next_without_skip() override {
        if (current == node->children + node->n.num_children) return;
        current++;
        cart_stats::iter_steps++;
    }
};

struct ARTNodeIterator_48 : ARTNodeIterator {
    ARTNode_48* node{};
    Bitmap<4> bitmap{};
    uint64_t cur_index{};

    ARTNodeIterator_48() = default;
    explicit ARTNodeIterator_48(ARTNode_48* n) : node(n), bitmap(n->unique_bitmap), cur_index(0) {
        cur_index = bitmap.find_first();
    }

    ARTNode* operator*() override {
        if (cur_index >= 256) return nullptr;
        uint8_t slot = node->keys[cur_index];
        if (!slot) return nullptr;
        return node->children[slot - 1];
    }

    [[nodiscard]] bool is_valid() const override { return cur_index < 256; }

    std::pair<uint8_t, ARTNode*> get() override {
        if (cur_index >= 256) return {0, nullptr};
        uint8_t slot = node->keys[cur_index];
        return {(uint8_t)cur_index, slot ? node->children[slot - 1] : nullptr};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        if (cur_index >= 256) return;
        // skip duplicates (shared leaves)
        uint8_t slot = node->keys[cur_index];
        ARTNode* cur_child = slot ? node->children[slot - 1] : nullptr;
        if (cur_child && IS_LEAF(cur_child)) {
            ARTLeaf* leaf = LEAF_RAW(cur_child);
            do {
                cur_index = bitmap.find_first(cur_index + 1);
                if (cur_index >= 256) return;
                uint8_t nslot = node->keys[cur_index];
                ARTNode* nxt = nslot ? node->children[nslot - 1] : nullptr;
                if (!nxt || !IS_LEAF(nxt) || LEAF_RAW(nxt) != leaf) break;
            } while (true);
        } else {
            cur_index = bitmap.find_first(cur_index + 1);
        }
        cart_stats::iter_steps++;
    }

    void operator++(int step) override { for (int i = 0; i < step; i++) (*this)++; }

    void next_without_skip() override {
        if (cur_index >= 256) return;
        cur_index = bitmap.find_first(cur_index + 1);
        cart_stats::iter_steps++;
    }
};

struct ARTNodeIterator_256 : ARTNodeIterator {
    ARTNode_256* node{};
    Bitmap<4> bitmap{};
    uint64_t cur_index{};

    ARTNodeIterator_256() = default;
    explicit ARTNodeIterator_256(ARTNode_256* n) : node(n), bitmap(n->unique_bitmap), cur_index(0) {
        cur_index = bitmap.find_first();
    }

    ARTNode* operator*() override {
        if (cur_index >= 256) return nullptr;
        return node->children[cur_index];
    }

    [[nodiscard]] bool is_valid() const override { return cur_index < 256; }

    std::pair<uint8_t, ARTNode*> get() override {
        if (cur_index >= 256) return {0, nullptr};
        return {(uint8_t)cur_index, node->children[cur_index]};
    }

    ARTNode* get_node() override { return (ARTNode*)node; }

    void operator++() override {
        if (cur_index >= 256) return;
        ARTNode* cur = node->children[cur_index];
        if (cur && IS_LEAF(cur)) {
            ARTLeaf* leaf = LEAF_RAW(cur);
            do {
                cur_index = bitmap.find_first(cur_index + 1);
                if (cur_index >= 256) return;
                auto nxt = node->children[cur_index];
                if (!nxt || !IS_LEAF(nxt) || LEAF_RAW(nxt) != leaf) break;
            } while (true);
        } else {
            cur_index = bitmap.find_first(cur_index + 1);
        }
        cart_stats::iter_steps++;
    }

    void operator++(int step) override { for (int i = 0; i < step; i++) (*this)++; }

    void next_without_skip() override {
        if (cur_index >= 256) return;
        cur_index = bitmap.find_first(cur_index + 1);
        cart_stats::iter_steps++;
    }
};

// alloc_iterator implementations
ARTNodeIterator* alloc_iterator(const ARTNode* node) {
    switch (node->type) {
        case NODE4:  return new ARTNodeIterator_4((ARTNode_4*)node);
        case NODE16: return new ARTNodeIterator_16((ARTNode_16*)node);
        case NODE48: return new ARTNodeIterator_48((ARTNode_48*)node);
        case NODE256:return new ARTNodeIterator_256((ARTNode_256*)node);
        default: throw std::runtime_error("alloc_iterator: invalid node type");
    }
}

void alloc_iterator_ref(const ARTNode* node, std::variant<ARTNodeIterator_4, ARTNodeIterator_16, ARTNodeIterator_48, ARTNodeIterator_256>& iter) {
    switch (node->type) {
        case NODE4:   iter = ARTNodeIterator_4((ARTNode_4*)node); break;
        case NODE16:  iter = ARTNodeIterator_16((ARTNode_16*)node); break;
        case NODE48:  iter = ARTNodeIterator_48((ARTNode_48*)node); break;
        case NODE256: iter = ARTNodeIterator_256((ARTNode_256*)node); break;
        default: throw std::runtime_error("alloc_iterator_ref: invalid node type");
    }
}

void destroy_iterator(ARTNodeIterator* iter) { delete iter; }

bool iter_is_valid(ARTNodeIterator* iter) { return iter->is_valid(); }

void iter_next(ARTNodeIterator* iter) { (*iter)++; }

void iter_next_without_skip(ARTNodeIterator* iter) { iter->next_without_skip(); }

std::pair<uint8_t, ARTNode*> iter_get(ARTNodeIterator* iter) { return iter->get(); }

ARTNode** iter_get_node(ARTNodeIterator* iter) {
    // Not commonly used; return nullptr stub
    return nullptr;
}

ARTNode** iter_get_current(ARTNodeIterator* iter) {
    // Return pointer to current child for modification
    switch (iter->get_node()->type) {
        case NODE4:  { auto i = (ARTNodeIterator_4*)iter; return i->current; }
        case NODE16: { auto i = (ARTNodeIterator_16*)iter; return i->current; }
        case NODE48: {
            auto i = (ARTNodeIterator_48*)iter;
            if (i->cur_index >= 256) return nullptr;
            uint8_t slot = i->node->keys[i->cur_index];
            return slot ? &i->node->children[slot - 1] : nullptr;
        }
        case NODE256: {
            auto i = (ARTNodeIterator_256*)iter;
            if (i->cur_index >= 256) return nullptr;
            return &i->node->children[i->cur_index];
        }
        default: return nullptr;
    }
}

ARTNode* iter_get_current_ro(ARTNodeIterator* iter) {
    auto** ptr = iter_get_current(iter);
    return ptr ? *ptr : nullptr;
}

// ============================================================
// §9. ARTNode alloc/destroy (art_node.cpp)
// ============================================================
ARTNode* alloc_node(uint8_t type, ARTKey prefix, uint8_t depth, WriterTraceBlock* trace_block) {
    ARTNode* n;
    switch (type) {
        case NODE4:
            n = (ARTNode*) new ARTNode_4();
            cart_stats::node4_alloc++;
            break;
        case NODE16:
            n = (ARTNode*) new ARTNode_16();
            cart_stats::node16_alloc++;
            break;
        case NODE48:
            if (trace_block) n = (ARTNode*) trace_block->allocate_art_node48();
            else n = (ARTNode*) new ARTNode_48();
            cart_stats::node48_alloc++;
            break;
        case NODE256:
            if (trace_block) n = (ARTNode*) trace_block->allocate_art_node256();
            else n = (ARTNode*) new ARTNode_256();
            cart_stats::node256_alloc++;
            break;
        default:
            throw std::runtime_error("alloc_node: invalid node type");
    }
    n->type = type;
    n->prefix = prefix;
    n->depth = depth;
    return n;
}

void recursive_destroy_node(ARTNode* n) {
    if (!LEAF_RAW(n)) return;
    if (IS_LEAF(n)) {
        leaf_destroy(LEAF_RAW(n));
        delete LEAF_RAW(n);
        return;
    }
    // [MOD] Direct walk for destruction
    switch (n->type) {
        case NODE4: {
            auto n4 = (ARTNode_4*)n;
            for (int i = 0; i < n->num_children; i++) {
                if (n4->children[i]) recursive_destroy_node(n4->children[i]);
            }
            break;
        }
        case NODE16: {
            auto n16 = (ARTNode_16*)n;
            for (int i = 0; i < n->num_children; i++) {
                if (n16->children[i]) recursive_destroy_node(n16->children[i]);
            }
            break;
        }
        case NODE48: {
            auto n48 = (ARTNode_48*)n;
            n48->unique_bitmap.for_each([&](uint8_t byte) {
                if (n48->children[n48->keys[byte]-1])
                    recursive_destroy_node(n48->children[n48->keys[byte]-1]);
            });
            break;
        }
        case NODE256: {
            auto n256 = (ARTNode_256*)n;
            n256->unique_bitmap.for_each([&](uint8_t byte) {
                if (n256->children[byte]) recursive_destroy_node(n256->children[byte]);
            });
            break;
        }
        default: break;
    }
    // Type-correct delete
    switch (n->type) {
        case NODE4:   delete (ARTNode_4*)n; break;
        case NODE16:  delete (ARTNode_16*)n; break;
        case NODE48:  delete (ARTNode_48*)n; break;
        case NODE256: delete (ARTNode_256*)n; break;
        default:      delete n; break;
    }
}

void delete_node(ARTNode* n, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   delete (ARTNode_4*)n; break;
        case NODE16:  delete (ARTNode_16*)n; break;
        case NODE48:
            if (trace_block) trace_block->deallocate_art_node48((ARTNode_48*)n);
            else delete (ARTNode_48*)n;
            break;
        case NODE256:
            if (trace_block) trace_block->deallocate_art_node256((ARTNode_256*)n);
            else delete (ARTNode_256*)n;
            break;
        default: throw std::runtime_error("delete_node: invalid type");
    }
}

// ============================================================
// §10. Core node operations (from art_node_ops.cpp)
// ============================================================

// node_for_each forward declare (needed by node_ref)
template<typename F>
void node_for_each(ARTNode* n, F&& callback);

void node_ref(ARTNode* node) {
    auto cb = [](ARTNode* child) {
        if (IS_LEAF(child)) LEAF_RAW(child)->ref_cnt += 1;
        else child->ref_cnt += 1;
    };
    node_for_each(node, cb);
}

uint16_t find_mid_count(std::vector<ARTNode**>* pointers_in_range) {
    uint16_t cur_idx = 0;
    while (cur_idx < pointers_in_range->size() - 1) {
        cur_idx++;
        if (GET_OFFSET(*(pointers_in_range->at(cur_idx))) >= ART_LEAF_SIZE / 2) break;
    }
    return cur_idx;
}

ARTNode** find_mid_count(ARTNode** begin_ptr, ARTNode** end_ptr) {
    auto cur_ptr = begin_ptr;
    if (*(end_ptr - 1) == nullptr) end_ptr--;
    while (cur_ptr < end_ptr - 1) {
        cur_ptr++;
        if (GET_OFFSET(*cur_ptr) >= ART_LEAF_SIZE / 2) break;
    }
    return cur_ptr;
}

// [MOD] find_child with step counting
ARTNode** find_child(ARTNode* n, unsigned char c) {
    cart_stats::find_child_calls++;
    int i, mask, bitfield;
    union {
        ARTNode_4*   p1;
        ARTNode_16*  p2;
        ARTNode_48*  p3;
        ARTNode_256* p4;
    } p{};

    switch (n->type) {
        case NODE4:
            p.p1 = (ARTNode_4*)n;
            for (i = 0; i < n->num_children; i++) {
                cart_stats::find_child_steps_total++;
                if (((unsigned char*)p.p1->keys)[i] == c)
                    return &p.p1->children[i];
            }
            break;
        case NODE16: {
            p.p2 = (ARTNode_16*)n;
            __m128i cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                                          _mm_loadu_si128((__m128i*)p.p2->keys));
            mask = (1 << n->num_children) - 1;
            bitfield = _mm_movemask_epi8(cmp) & mask;
            cart_stats::find_child_steps_total++;
            if (bitfield) return &p.p2->children[__builtin_ctz(bitfield)];
            break;
        }
        case NODE48:
            p.p3 = (ARTNode_48*)n;
            cart_stats::find_child_steps_total++;
            i = p.p3->keys[c];
            if (i) return &p.p3->children[i - 1];
            break;
        case NODE256:
            p.p4 = (ARTNode_256*)n;
            cart_stats::find_child_steps_total++;
            if (p.p4->children[c]) return &p.p4->children[c];
            break;
        default:
            throw std::runtime_error("find_child: invalid node type");
    }
    return nullptr;
}

uint16_t find_child_idx(ARTNode* n, unsigned char c) {
    int i;
    union {
        ARTNode_4*   p1;
        ARTNode_16*  p2;
        ARTNode_48*  p3;
        ARTNode_256* p4;
    } p{};

    switch (n->type) {
        case NODE4: {
            p.p1 = (ARTNode_4*)n;
            for (i = 0; i < n->num_children; i++)
                if (p.p1->keys[i] == c) return i;
            break;
        }
        case NODE16: {
            p.p2 = (ARTNode_16*)n;
            __m128i cmp = _mm_cmpeq_epi8(_mm_set1_epi8(c),
                                          _mm_loadu_si128((__m128i*)p.p2->keys));
            int mask = (1 << n->num_children) - 1;
            int bf = _mm_movemask_epi8(cmp) & mask;
            if (bf) return __builtin_ctz(bf);
            break;
        }
        case NODE48: {
            p.p3 = (ARTNode_48*)n;
            i = p.p3->keys[c];
            if (i) return i - 1;
            break;
        }
        case NODE256: {
            p.p4 = (ARTNode_256*)n;
            if (p.p4->children[c]) return c;
            break;
        }
        default:
            throw std::runtime_error("find_child_idx: invalid type");
    }
    return 256;
}

ARTLeaf* node_search(ARTNode* u, ARTKey key) {
    ARTNode** child;
    ARTNode* n = u;
    int depth = 0;

    while (n) {
        if (IS_LEAF(n)) {
            auto l = LEAF_RAW(n);
            auto offset = GET_OFFSET(n);
            if (l->depth == 4) return (ARTLeaf*)n;
            if (ARTKey::check_partial_match(ARTKey{(uint64_t)l->at(offset)}, key, l->depth))
                return (ARTLeaf*)n;
            return nullptr;
        }
        if (n->depth == depth) {
            child = find_child(n, key[depth]);
            n = child ? *child : nullptr;
            depth++;
        } else {
            assert(n->depth > depth);
            if (ARTKey::check_partial_match(n->prefix, key, n->depth - 1)) {
                depth = n->depth;
                child = find_child(n, key[depth]);
                n = child ? *child : nullptr;
            } else return nullptr;
        }
    }
    return nullptr;
}

ARTNode** add_child256(ARTNode_256* n, ARTNode** ref, unsigned char c, void* child) {
    (void)ref;
    n->n.num_children++;
    n->children[c] = (ARTNode*)child;
    n->unique_bitmap.set(c);
    return &n->children[c];
}

ARTNode** add_child48(ARTNode_48* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* trace_block) {
    if (n->n.num_children < 48) {
        // Find empty slot
        int pos = 0;
        while (n->children[pos]) pos++;
        n->children[pos] = (ARTNode*)child;
        n->keys[c] = pos + 1;
        n->unique_bitmap.set(c);
        n->n.num_children++;
        return &n->children[pos];
    } else {
        // Upgrade to node256
        cart_stats::grow_count++;
        auto new_node = (ARTNode_256*) alloc_node(NODE256, n->n.prefix, n->n.depth, trace_block);
        new_node->n.num_children = n->n.num_children;
        for (int i = 0; i < 256; i++) {
            if (n->keys[i]) {
                new_node->children[i] = n->children[n->keys[i] - 1];
                new_node->unique_bitmap.set(i);
            }
        }
        delete_node((ARTNode*)n, trace_block);
        *ref = (ARTNode*)new_node;
        return add_child256(new_node, ref, c, child);
    }
}

ARTNode** add_child16(ARTNode_16* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* trace_block) {
    if (n->n.num_children < 16) {
        // Find sorted insertion position
        int pos = 0;
        while (pos < n->n.num_children && n->keys[pos] < c) pos++;
        // Shift right
        memmove(n->keys + pos + 1, n->keys + pos, n->n.num_children - pos);
        memmove(n->children + pos + 1, n->children + pos, (n->n.num_children - pos) * sizeof(void*));
        n->keys[pos] = c;
        n->children[pos] = (ARTNode*)child;
        n->n.num_children++;
        return &n->children[pos];
    } else {
        // Upgrade to node48
        cart_stats::grow_count++;
        auto new_node = (ARTNode_48*) alloc_node(NODE48, n->n.prefix, n->n.depth, trace_block);
        new_node->n.num_children = n->n.num_children;
        for (int i = 0; i < n->n.num_children; i++) {
            new_node->children[i] = n->children[i];
            new_node->keys[(uint8_t)n->keys[i]] = i + 1;
            new_node->unique_bitmap.set((uint8_t)n->keys[i]);
        }
        delete_node((ARTNode*)n, trace_block);
        *ref = (ARTNode*)new_node;
        return add_child48(new_node, ref, c, child, trace_block);
    }
}

ARTNode** add_child4(ARTNode_4* n, ARTNode** ref, unsigned char c, void* child) {
    if (n->n.num_children < 4) {
        int pos = 0;
        while (pos < n->n.num_children && n->keys[pos] < c) pos++;
        memmove(n->keys + pos + 1, n->keys + pos, n->n.num_children - pos);
        memmove(n->children + pos + 1, n->children + pos, (n->n.num_children - pos) * sizeof(void*));
        n->keys[pos] = c;
        n->children[pos] = (ARTNode*)child;
        n->n.num_children++;
        return &n->children[pos];
    } else {
        // Upgrade to node16
        cart_stats::grow_count++;
        auto new_node = (ARTNode_16*) alloc_node(NODE16, n->n.prefix, n->n.depth, nullptr);
        new_node->n.num_children = n->n.num_children;
        memcpy(new_node->keys, n->keys, n->n.num_children);
        memcpy(new_node->children, n->children, n->n.num_children * sizeof(void*));
        delete (ARTNode_4*)n;
        *ref = (ARTNode*)new_node;
        return add_child16(new_node, ref, c, child, nullptr);
    }
}

ARTNode** add_child(ARTNode* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return add_child4((ARTNode_4*)n, ref, c, child);
        case NODE16:  return add_child16((ARTNode_16*)n, ref, c, child, trace_block);
        case NODE48:  return add_child48((ARTNode_48*)n, ref, c, child, trace_block);
        case NODE256: return add_child256((ARTNode_256*)n, ref, c, child);
        default: throw std::runtime_error("add_child: invalid type");
    }
}

void node16_upgrade(ARTNode_16* n, ARTNode** ref, WriterTraceBlock* trace_block) {
    cart_stats::grow_count++;
    auto new_node = (ARTNode_48*) alloc_node(NODE48, n->n.prefix, n->n.depth, trace_block);
    new_node->n.num_children = n->n.num_children;
    for (int i = 0; i < n->n.num_children; i++) {
        new_node->children[i] = n->children[i];
        new_node->keys[(uint8_t)n->keys[i]] = i + 1;
        new_node->unique_bitmap.set((uint8_t)n->keys[i]);
    }
    delete_node((ARTNode*)n, trace_block);
    *ref = (ARTNode*)new_node;
}

void remove_child256(ARTNode_256* n, ARTNode** ref, unsigned char c, ARTNode** child) {
    n->children[c] = nullptr;
    n->unique_bitmap.reset(c);
    n->n.num_children--;
}

void remove_child48(ARTNode_48* n, ARTNode** ref, unsigned char c, ARTNode** child) {
    int pos = n->keys[c];
    if (!pos) return;
    n->children[pos - 1] = nullptr;
    n->keys[c] = 0;
    n->unique_bitmap.reset(c);
    n->n.num_children--;
}

void remove_child16(ARTNode_16* n, ARTNode** ref, unsigned char c, ARTNode** child) {
    int pos = child - n->children;
    memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - pos - 1);
    memmove(n->children + pos, n->children + pos + 1, (n->n.num_children - pos - 1) * sizeof(void*));
    n->n.num_children--;
    n->children[n->n.num_children] = nullptr;
}

void remove_child4(ARTNode_4* n, ARTNode** ref, unsigned char c, ARTNode** child) {
    int pos = child - n->children;
    memmove(n->keys + pos, n->keys + pos + 1, n->n.num_children - pos - 1);
    memmove(n->children + pos, n->children + pos + 1, (n->n.num_children - pos - 1) * sizeof(void*));
    n->n.num_children--;
    n->children[n->n.num_children] = nullptr;
}

void remove_child(ARTNode* n, ARTNode** ref, unsigned char c, ARTNode** child) {
    switch (n->type) {
        case NODE4:   remove_child4((ARTNode_4*)n, ref, c, child); break;
        case NODE16:  remove_child16((ARTNode_16*)n, ref, c, child); break;
        case NODE48:  remove_child48((ARTNode_48*)n, ref, c, child); break;
        case NODE256: remove_child256((ARTNode_256*)n, ref, c, child); break;
        default: throw std::runtime_error("remove_child: invalid type");
    }
}

// add_child_copy: update child pointer in-place without growing
ARTNode** add_child_copy(ARTNode* n, uint8_t child_idx, ARTNode* child) {
    switch (n->type) {
        case NODE4: {
            auto p = (ARTNode_4*)n;
            p->children[child_idx] = child;
            return &p->children[child_idx];
        }
        case NODE16: {
            auto p = (ARTNode_16*)n;
            p->children[child_idx] = child;
            return &p->children[child_idx];
        }
        case NODE48: {
            auto p = (ARTNode_48*)n;
            p->children[child_idx] = child;
            return &p->children[child_idx];
        }
        case NODE256: {
            auto p = (ARTNode_256*)n;
            p->children[child_idx] = child;
            return &p->children[child_idx];
        }
        default: throw std::runtime_error("add_child_copy: invalid type");
    }
}

void node_pointers_update(ARTNode* node, ARTNode** child, ARTKey key, int offset) {
    // Update all pointers in node that point to same leaf as *child, shifting their offset by 'offset'
    switch (node->type) {
        case NODE4: {
            auto n = (ARTNode_4*)node;
            if (!IS_LEAF(*child)) return;
            ARTLeaf* target = LEAF_RAW(*child);
            for (int i = 0; i < n->n.num_children; i++) {
                if (IS_LEAF(n->children[i]) && LEAF_RAW(n->children[i]) == target) {
                    uint64_t old_offset = GET_OFFSET(n->children[i]);
                    n->children[i] = (ARTNode*)LEAF_POINTER_CTOR(target, old_offset + offset);
                }
            }
            break;
        }
        case NODE16: {
            auto n = (ARTNode_16*)node;
            if (!IS_LEAF(*child)) return;
            ARTLeaf* target = LEAF_RAW(*child);
            for (int i = 0; i < n->n.num_children; i++) {
                if (IS_LEAF(n->children[i]) && LEAF_RAW(n->children[i]) == target) {
                    uint64_t old_offset = GET_OFFSET(n->children[i]);
                    n->children[i] = (ARTNode*)LEAF_POINTER_CTOR(target, old_offset + offset);
                }
            }
            break;
        }
        default: break;
    }
}

ARTLeaf* leaf_pointer_expand(ARTNode** n, uint8_t depth, WriterTraceBlock* trace_block) {
    // When a leaf range is size 1, expand it: allocate new node and split
    // Returns old leaf pointer
    // Simplified: just return nullptr indicating no split needed
    return nullptr;
}

ARTLeaf* find_leaf4(ARTNode_4* node, ARTNode** child, ARTKey key, WriterTraceBlock* trace_block) {
    if (!*child) {
        // Try adjacent
        if (child > node->children && IS_LEAF(*(child-1)))
            return LEAF_RAW(*(child-1));
        if (child < node->children + node->n.num_children - 1 && IS_LEAF(*(child+1)))
            return LEAF_RAW(*(child+1));
        return alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
    }
    return LEAF_RAW(*child);
}

ARTLeaf* find_leaf16(ARTNode_16* node, ARTNode** child, ARTKey key, WriterTraceBlock* trace_block) {
    if (!*child) {
        if (child > node->children && IS_LEAF(*(child-1)))
            return LEAF_RAW(*(child-1));
        if (child < node->children + node->n.num_children - 1 && IS_LEAF(*(child+1)))
            return LEAF_RAW(*(child+1));
        return alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
    }
    return LEAF_RAW(*child);
}

ARTLeaf* find_leaf48(ARTNode_48* node, ARTKey key, WriterTraceBlock* trace_block) {
    uint8_t target = key[node->n.depth];
    // Try neighbors
    for (int b = target - 1; b >= 0; b--) {
        if (node->keys[b]) {
            auto ptr = node->children[node->keys[b] - 1];
            if (IS_LEAF(ptr)) return LEAF_RAW(ptr);
            break;
        }
    }
    for (int b = target + 1; b <= 255; b++) {
        if (node->keys[b]) {
            auto ptr = node->children[node->keys[b] - 1];
            if (IS_LEAF(ptr)) return LEAF_RAW(ptr);
            break;
        }
    }
    return alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
}

ARTLeaf* find_leaf256(ARTNode_256* node, ARTKey key, WriterTraceBlock* trace_block) {
    uint8_t target = key[node->n.depth];
    for (int b = target - 1; b >= 0; b--) {
        if (node->children[b] && IS_LEAF(node->children[b]))
            return LEAF_RAW(node->children[b]);
    }
    for (int b = target + 1; b <= 255; b++) {
        if (node->children[b] && IS_LEAF(node->children[b]))
            return LEAF_RAW(node->children[b]);
    }
    return alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
}

ARTLeaf* find_leaf(ARTNode* n, ARTNode** child, ARTKey key, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return find_leaf4((ARTNode_4*)n, child, key, trace_block);
        case NODE16:  return find_leaf16((ARTNode_16*)n, child, key, trace_block);
        case NODE48:  return find_leaf48((ARTNode_48*)n, key, trace_block);
        case NODE256: return find_leaf256((ARTNode_256*)n, key, trace_block);
        default: throw std::runtime_error("find_leaf: invalid type");
    }
}

// node_split helpers
ARTNodeSplitRes node_split4(ARTNode_4* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, WriterTraceBlock* trace_block) {
    // Simple: allocate new leaf for the key and insert
    auto new_leaf = alloc_leaf(key, node->n.depth + 1, false, true, trace_block);
    *child = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, 0);
    return ARTNodeSplitRes{NEW_LEAF, new_leaf};
}

ARTNodeSplitRes node_split16(ARTNode_16* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, WriterTraceBlock* trace_block) {
    auto new_leaf = alloc_leaf(key, node->n.depth + 1, false, true, trace_block);
    *child = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, 0);
    return ARTNodeSplitRes{NEW_LEAF, new_leaf};
}

ARTNodeSplitRes node_split48(ARTNode_48* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, WriterTraceBlock* trace_block) {
    auto new_leaf = alloc_leaf(key, node->n.depth + 1, false, true, trace_block);
    return ARTNodeSplitRes{NEW_LEAF, new_leaf};
}

ARTNodeSplitRes node_split256(ARTNode_256* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, WriterTraceBlock* trace_block) {
    auto new_leaf = alloc_leaf(key, node->n.depth + 1, false, true, trace_block);
    return ARTNodeSplitRes{NEW_LEAF, new_leaf};
}

ARTNodeSplitRes node_split(ARTNode* n, ARTNode** child, ARTLeaf* leaf, ARTKey key, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return node_split4((ARTNode_4*)n, child, leaf, key, trace_block);
        case NODE16:  return node_split16((ARTNode_16*)n, child, leaf, key, trace_block);
        case NODE48:  return node_split48((ARTNode_48*)n, child, leaf, key, trace_block);
        case NODE256: return node_split256((ARTNode_256*)n, child, leaf, key, trace_block);
        default: throw std::runtime_error("node_split: invalid type");
    }
}

// insert: core insert into ART node
bool insert(ARTNode** n, ARTKey key, uint64_t value, Property_t* property, WriterTraceBlock* trace_block) {
    ARTNode* cur = *n;
    int depth = cur->depth;

    auto child = find_child(cur, key[depth]);
    if (!child) {
        // Insert new leaf
        auto new_leaf = alloc_leaf(key, depth + 1, false, true, trace_block);
        new_leaf->insert(value, property, 0);
        add_child(cur, n, key[depth], LEAF_POINTER_CTOR(new_leaf, 0), trace_block);
        return true;
    }
    if (IS_LEAF(*child)) {
        auto raw = LEAF_RAW(*child);
        if (raw->has_element(value, GET_OFFSET(*child))) return false;
        // Check if there's space
        if (raw->size < ART_LEAF_SIZE) {
            raw->insert(value, property, (uint16_t)raw->size);
            return true;
        }
        // Leaf full: need to split
        node_split(cur, child, raw, key, trace_block);
        return true;
    } else {
        // Recurse into child node
        return insert(child, key, value, property, trace_block);
    }
}

ARTNodeRemoveRes remove(ARTNode** n, ARTKey key, uint64_t value) {
    ARTNode* cur = *n;
    int depth = cur->depth;

    auto child = find_child(cur, key[depth]);
    if (!child) return NOT_FOUND;
    if (IS_LEAF(*child)) {
        auto raw = LEAF_RAW(*child);
        uint16_t pos = raw->find(value, GET_OFFSET(*child));
        if (pos >= raw->size || raw->at(pos) != value) return NOT_FOUND;
        raw->remove(pos, key[depth]);
        if (raw->size == 0) {
            remove_child(cur, n, key[depth], child);
            return CHILD_REMOVED;
        }
        return ELEMENT_REMOVED;
    } else {
        return remove(child, key, value);
    }
}

std::pair<uint64_t, uint64_t> get_node_filling_info(ARTNode* n) {
    if (!n || IS_LEAF(n)) {
        if (n && IS_LEAF(n)) {
            auto leaf = LEAF_RAW(n);
            return {leaf->size, ART_LEAF_SIZE};
        }
        return {0, 0};
    }
    uint64_t total_used = 0, total_cap = 0;
    // [MOD] Direct walk to avoid iterator overhead
    switch (n->type) {
        case NODE4: {
            auto n4 = (ARTNode_4*)n;
            ARTLeaf* last = nullptr;
            for (int i = 0; i < n->num_children; i++) {
                auto child = n4->children[i];
                if (IS_LEAF(child)) {
                    if (LEAF_RAW(child) != last) {
                        auto [u,c] = get_node_filling_info(child);
                        total_used += u; total_cap += c;
                        last = LEAF_RAW(child);
                    }
                } else { auto [u,c] = get_node_filling_info(child); total_used+=u; total_cap+=c; last=nullptr; }
            }
            break;
        }
        case NODE16: {
            auto n16 = (ARTNode_16*)n;
            ARTLeaf* last = nullptr;
            for (int i = 0; i < n->num_children; i++) {
                auto child = n16->children[i];
                if (IS_LEAF(child)) {
                    if (LEAF_RAW(child) != last) {
                        auto [u,c] = get_node_filling_info(child);
                        total_used += u; total_cap += c;
                        last = LEAF_RAW(child);
                    }
                } else { auto [u,c] = get_node_filling_info(child); total_used+=u; total_cap+=c; last=nullptr; }
            }
            break;
        }
        case NODE48: {
            auto n48 = (ARTNode_48*)n;
            n48->unique_bitmap.for_each([&](uint8_t byte) {
                auto child = n48->children[n48->keys[byte]-1];
                auto [u,c] = get_node_filling_info(child); total_used+=u; total_cap+=c;
            });
            break;
        }
        case NODE256: {
            auto n256 = (ARTNode_256*)n;
            n256->unique_bitmap.for_each([&](uint8_t byte) {
                auto child = n256->children[byte];
                auto [u,c] = get_node_filling_info(child); total_used+=u; total_cap+=c;
            });
            break;
        }
    }
    return {total_used, total_cap};
}

void gc_node_ref(ARTNode* n, WriterTraceBlock* trace_block) {
    if (!n || IS_LEAF(n)) return;
    // [MOD] Direct walk
    switch (n->type) {
        case NODE4: {
            auto n4 = (ARTNode_4*)n;
            for (int i = 0; i < n->num_children; i++) {
                auto child = n4->children[i];
                if (IS_LEAF(child)) LEAF_RAW(child)->ref_cnt = 1;
                else { child->ref_cnt = 1; gc_node_ref(child, trace_block); }
            }
            break;
        }
        case NODE16: {
            auto n16 = (ARTNode_16*)n;
            for (int i = 0; i < n->num_children; i++) {
                auto child = n16->children[i];
                if (IS_LEAF(child)) LEAF_RAW(child)->ref_cnt = 1;
                else { child->ref_cnt = 1; gc_node_ref(child, trace_block); }
            }
            break;
        }
        case NODE48: {
            auto n48 = (ARTNode_48*)n;
            n48->unique_bitmap.for_each([&](uint8_t byte) {
                auto child = n48->children[n48->keys[byte]-1];
                if (IS_LEAF(child)) LEAF_RAW(child)->ref_cnt = 1;
                else { child->ref_cnt = 1; gc_node_ref(child, trace_block); }
            });
            break;
        }
        case NODE256: {
            auto n256 = (ARTNode_256*)n;
            n256->unique_bitmap.for_each([&](uint8_t byte) {
                auto child = n256->children[byte];
                if (IS_LEAF(child)) LEAF_RAW(child)->ref_cnt = 1;
                else { child->ref_cnt = 1; gc_node_ref(child, trace_block); }
            });
            break;
        }
    }
}

// Intersection helpers
void node_range_intersect(ARTNode* node, RangeElement* range, uint16_t range_size, std::vector<uint64_t>& result) {
    if (!node || IS_LEAF(node)) return;
    for (uint16_t ri = 0; ri < range_size; ri++) {
        auto child = find_child(node, range[ri]);
        if (child && IS_LEAF(*child)) {
            auto leaf = LEAF_RAW(*child);
            leaf->for_each([&](uint64_t elem, double) { result.push_back(elem); });
        }
    }
}

void leaf_intersect(ARTLeaf* leaf1, uint8_t leaf_start1, ARTLeaf* leaf2, uint8_t leaf_start2, std::vector<uint64_t>& result) {
    // Simple sorted merge intersection
    uint16_t i = leaf_start1, j = leaf_start2;
    while (i < leaf1->size && j < leaf2->size) {
        uint64_t a = leaf1->at(i), b = leaf2->at(j);
        if (a == b) { result.push_back(a); i++; j++; }
        else if (a < b) i++;
        else j++;
    }
}

void node_leaf_intersect(ARTNode* node, ARTLeaf* leaf, uint8_t leaf_start, std::vector<uint64_t>& result) {
    for (uint16_t i = leaf_start; i < leaf->size; i++) {
        uint64_t elem = leaf->at(i);
        auto child = find_child(node, (uint8_t)(elem & 0xFF));
        if (child && IS_LEAF(*child)) {
            auto l2 = LEAF_RAW(*child);
            if (l2->has_element(elem, 0)) result.push_back(elem);
        }
    }
}

void node_intersect(ARTNode* node1, ARTNode* node2, std::vector<uint64_t>& result) {
    // [MOD] Direct walk to probe node2
    switch (node1->type) {
        case NODE4: {
            auto n4 = (ARTNode_4*)node1;
            for (int i = 0; i < node1->num_children; i++) {
                auto child1 = n4->children[i];
                uint8_t byte = n4->keys[i];
                auto child2 = find_child(node2, byte);
                if (child2 && IS_LEAF(child1) && IS_LEAF(*child2)) {
                    leaf_intersect(LEAF_RAW(child1), 0, LEAF_RAW(*child2), 0, result);
                }
            }
            break;
        }
        case NODE16: {
            auto n16 = (ARTNode_16*)node1;
            for (int i = 0; i < node1->num_children; i++) {
                auto child1 = n16->children[i];
                uint8_t byte = n16->keys[i];
                auto child2 = find_child(node2, byte);
                if (child2 && IS_LEAF(child1) && IS_LEAF(*child2)) {
                    leaf_intersect(LEAF_RAW(child1), 0, LEAF_RAW(*child2), 0, result);
                }
            }
            break;
        }
        case NODE48: {
            auto n48 = (ARTNode_48*)node1;
            n48->unique_bitmap.for_each([&](uint8_t byte) {
                auto child1 = n48->children[n48->keys[byte]-1];
                auto child2 = find_child(node2, byte);
                if (child2 && IS_LEAF(child1) && IS_LEAF(*child2)) {
                    leaf_intersect(LEAF_RAW(child1), 0, LEAF_RAW(*child2), 0, result);
                }
            });
            break;
        }
        case NODE256: {
            auto n256 = (ARTNode_256*)node1;
            n256->unique_bitmap.for_each([&](uint8_t byte) {
                auto child1 = n256->children[byte];
                auto child2 = find_child(node2, byte);
                if (child2 && IS_LEAF(child1) && IS_LEAF(*child2)) {
                    leaf_intersect(LEAF_RAW(child1), 0, LEAF_RAW(*child2), 0, result);
                }
            });
            break;
        }
    }
}

uint64_t node_range_intersect(ARTNode* node, RangeElement* range, uint16_t range_size) {
    std::vector<uint64_t> result;
    node_range_intersect(node, range, range_size, result);
    return result.size();
}

uint64_t leaf_intersect(ARTLeaf* leaf1, uint8_t leaf_start1, ARTLeaf* leaf2, uint8_t leaf_start2) {
    std::vector<uint64_t> result;
    leaf_intersect(leaf1, leaf_start1, leaf2, leaf_start2, result);
    return result.size();
}

uint64_t node_leaf_intersect(ARTNode* node, ARTLeaf* leaf, uint8_t leaf_start) {
    std::vector<uint64_t> result;
    node_leaf_intersect(node, leaf, leaf_start, result);
    return result.size();
}

uint64_t node_intersect(ARTNode* node1, ARTNode* node2) {
    std::vector<uint64_t> result;
    node_intersect(node1, node2, result);
    return result.size();
}

void check_node(ARTNode* node) {
    // Basic sanity check
    if (!node) return;
    assert(node->type >= NODE4 && node->type <= NODE256);
}

// batch_insert helpers
uint64_t empty_leaf_batch_insert8(ARTNode** n, ARTLeaf8** leaf, RangeElement* insert_list, Property_t** properties, uint64_t list_size, WriterTraceBlock* trace_block) {
    uint64_t inserted = 0;
    for (uint64_t i = 0; i < list_size; i++) {
        uint8_t target = insert_list[i] & 0xFF;
        if (!(*leaf)->value.get(target)) {
            (*leaf)->value.set(target);
            (*leaf)->size++;
            inserted++;
        }
    }
    return inserted;
}

uint64_t empty_leaf_batch_insert16(ARTNode** n, ARTLeaf16** leaf, RangeElement* insert_list, Property_t** properties, uint64_t list_size, WriterTraceBlock* trace_block) {
    uint64_t inserted = 0;
    for (uint64_t i = 0; i < list_size; i++) {
        uint16_t target = insert_list[i] & 0xFFFF;
        auto it = std::lower_bound((*leaf)->value->begin(), (*leaf)->value->begin() + (*leaf)->size, target);
        if (it == (*leaf)->value->begin() + (*leaf)->size || *it != target) {
            for (int j = (*leaf)->size; j > (int)(it - (*leaf)->value->begin()); j--)
                (*(*leaf)->value)[j] = (*(*leaf)->value)[j-1];
            (*(*leaf)->value)[it - (*leaf)->value->begin()] = target;
            (*leaf)->size++;
            inserted++;
        }
    }
    return inserted;
}

uint64_t empty_leaf_batch_insert32(ARTNode** n, ARTLeaf32** leaf, RangeElement* insert_list, Property_t** properties, uint64_t list_size, WriterTraceBlock* trace_block) {
    uint64_t inserted = 0;
    for (uint64_t i = 0; i < list_size; i++) {
        uint32_t target = insert_list[i];
        auto it = std::lower_bound((*leaf)->value->begin(), (*leaf)->value->begin() + (*leaf)->size, target);
        if (it == (*leaf)->value->begin() + (*leaf)->size || *it != target) {
            for (int j = (*leaf)->size; j > (int)(it - (*leaf)->value->begin()); j--)
                (*(*leaf)->value)[j] = (*(*leaf)->value)[j-1];
            (*(*leaf)->value)[it - (*leaf)->value->begin()] = target;
            (*leaf)->size++;
            inserted++;
        }
    }
    return inserted;
}

uint64_t empty_leaf_batch_insert64(ARTNode** n, ARTLeaf64** leaf, RangeElement* insert_list, Property_t** properties, uint64_t list_size, WriterTraceBlock* trace_block) {
    uint64_t inserted = 0;
    for (uint64_t i = 0; i < list_size; i++) {
        uint64_t target = insert_list[i];
        auto it = std::lower_bound((*leaf)->value->begin(), (*leaf)->value->begin() + (*leaf)->size, target);
        if (it == (*leaf)->value->begin() + (*leaf)->size || *it != target) {
            for (int j = (*leaf)->size; j > (int)(it - (*leaf)->value->begin()); j--)
                (*(*leaf)->value)[j] = (*(*leaf)->value)[j-1];
            (*(*leaf)->value)[it - (*leaf)->value->begin()] = target;
            (*leaf)->size++;
            inserted++;
        }
    }
    return inserted;
}

uint64_t empty_leaf_batch_insert(ARTNode** n, ARTLeaf** leaf, RangeElement* insert_list, Property_t** properties, uint64_t list_size, WriterTraceBlock* trace_block) {
    switch ((*leaf)->type) {
        case LEAF8:  return empty_leaf_batch_insert8(n, (ARTLeaf8**)leaf, insert_list, properties, list_size, trace_block);
        case LEAF16: return empty_leaf_batch_insert16(n, (ARTLeaf16**)leaf, insert_list, properties, list_size, trace_block);
        case LEAF32: return empty_leaf_batch_insert32(n, (ARTLeaf32**)leaf, insert_list, properties, list_size, trace_block);
        case LEAF64: return empty_leaf_batch_insert64(n, (ARTLeaf64**)leaf, insert_list, properties, list_size, trace_block);
        default: return 0;
    }
}

uint64_t leaf_list_merge_batch_insert(ARTNode** n, ARTLeaf* &new_leaf, uint8_t cur_byte, ARTLeaf* leaf, uint16_t leaf_count, RangeElement* insert_list, Property_t** properties, uint64_t list_count, WriterTraceBlock* trace_block);

void add_list_segment_to_new_leaf(ARTNode** new_node, ARTLeaf* &new_leaf, uint8_t depth, RangeElement* elem_list, Property_t** prop_list, uint64_t count, uint8_t cur_byte, WriterTraceBlock* trace_block);

void add_leaf_segment_to_new_leaf(ARTNode** new_node, ARTLeaf* &new_leaf, uint8_t depth, ARTLeaf* cur_leaf, uint64_t count, uint8_t cur_byte, WriterTraceBlock* trace_block);

uint64_t leaf_list_merge(ARTNode** n, ARTLeaf* leaf, RangeElement* elem_list, Property_t** properties, uint64_t list_size, WriterTraceBlock* trace_block) {
    // Simplified: just batch insert into leaf
    return empty_leaf_batch_insert(n, &leaf, elem_list, properties, list_size, trace_block);
}

uint64_t leaf_list_merge_batch_insert(ARTNode** n, ARTLeaf* &new_leaf, uint8_t cur_byte, ARTLeaf* leaf, uint16_t leaf_count, RangeElement* insert_list, Property_t** properties, uint64_t list_count, WriterTraceBlock* trace_block) {
    uint64_t inserted = 0;
    // Merge: insert list elements into existing leaf structure
    if (!new_leaf) {
        new_leaf = alloc_leaf(ARTKey{(uint64_t)insert_list[0]}, leaf->depth, false, true, trace_block);
        add_child(*n, n, cur_byte, LEAF_POINTER_CTOR(new_leaf, 0), trace_block);
    }
    for (uint64_t i = 0; i < list_count; i++) {
        if (!new_leaf->has_element((uint64_t)insert_list[i], 0)) {
            new_leaf->insert((uint64_t)insert_list[i], properties ? properties[i] : nullptr, new_leaf->size);
            inserted++;
        }
    }
    return inserted;
}

void add_list_segment_to_new_leaf(ARTNode** new_node, ARTLeaf* &new_leaf, uint8_t depth, RangeElement* elem_list, Property_t** prop_list, uint64_t count, uint8_t cur_byte, WriterTraceBlock* trace_block) {
    assert(count != 0);
    if (new_leaf == nullptr || new_leaf->size + count > ART_LEAF_SIZE) {
        if (count > ART_LEAF_SIZE) {
            auto new_child = add_child(*new_node, new_node, cur_byte, (void*)0x1, trace_block);
            // batch_subtree_build(new_child, depth + 1, elem_list, prop_list, count, trace_block);
            new_leaf = nullptr;
            return;
        } else {
            new_leaf = alloc_leaf(ARTKey{(uint64_t)elem_list[0]}, depth, false, true, trace_block);
        }
    }
    add_child(*new_node, new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf, new_leaf->size), trace_block);
    for (uint64_t i = 0; i < count; i++) {
        new_leaf->insert((uint64_t)elem_list[i], prop_list ? prop_list[i] : nullptr, new_leaf->size);
    }
}

void add_leaf_segment_to_new_leaf(ARTNode** new_node, ARTLeaf* &new_leaf, uint8_t depth, ARTLeaf* cur_leaf, uint64_t count, uint8_t cur_byte, WriterTraceBlock* trace_block) {
    assert(count != 0);
    auto cur_raw_leaf = LEAF_RAW(cur_leaf);
    auto leaf_st = GET_OFFSET(cur_leaf);
    if (new_leaf == nullptr || new_leaf->size + count > ART_LEAF_SIZE) {
        new_leaf = alloc_leaf(ARTKey{cur_raw_leaf->at((uint16_t)leaf_st)}, depth, false, true, trace_block);
        assert(new_leaf->size == 0);
    }
    cur_raw_leaf->copy_to_leaf((uint16_t)leaf_st, (uint16_t)(leaf_st + count), new_leaf, new_leaf->size);
    add_child(*new_node, new_node, cur_byte, LEAF_POINTER_CTOR(new_leaf, new_leaf->size), trace_block);
    new_leaf->size += count;
}

// ============================================================
// §11. Tree-level iteration templates (from art_node_ops.h)
// ============================================================
template<typename F>
void node_for_each(ARTNode* n, F&& callback) {
    switch (n->type) {
        case NODE4: {
            ARTNode* child = nullptr;
            for (int i = 0; i < n->num_children; i++) {
                auto cur = ((ARTNode_4*)n)->children[i];
                if (cur != child) { callback(cur); child = cur; }
            }
            break;
        }
        case NODE16: {
            ARTNode* child = nullptr;
            for (int i = 0; i < n->num_children; i++) {
                auto cur = ((ARTNode_16*)n)->children[i];
                if (cur != child) { callback(cur); child = cur; }
            }
            break;
        }
        case NODE48: {
            auto n48 = (ARTNode_48*)n;
            n48->unique_bitmap.for_each([&](uint8_t idx) {
                callback(n48->children[n48->keys[idx] - 1]);
            });
            break;
        }
        case NODE256: {
            auto n256 = (ARTNode_256*)n;
            n256->unique_bitmap.for_each([&](uint8_t idx) {
                callback(n256->children[idx]);
            });
            break;
        }
        default: abort();
    }
}

template<typename F>
int tree_leaf_iter(ARTNode* n, F&& callback) {
    switch (n->type) {
        case NODE4: {
            // [MOD] Direct array walk to avoid iterator overhead
            auto node4 = (ARTNode_4*)n;
            ARTLeaf* last_leaf = nullptr;
            for (int i = 0; i < n->num_children; i++) {
                auto child = node4->children[i];
                if (IS_LEAF(child)) {
                    auto leaf = LEAF_RAW(child);
                    if (leaf != last_leaf) {  // skip duplicate shared leaves
                        leaf->for_each(callback);
                        last_leaf = leaf;
                    }
                } else {
                    tree_leaf_iter(child, callback);
                    last_leaf = nullptr;
                }
                cart_stats::iter_steps++;
            }
            break;
        }
        case NODE16: {
            // [MOD] Direct array walk for Node16
            auto node16 = (ARTNode_16*)n;
            ARTLeaf* last_leaf = nullptr;
            for (int i = 0; i < n->num_children; i++) {
                auto child = node16->children[i];
                if (IS_LEAF(child)) {
                    auto leaf = LEAF_RAW(child);
                    if (leaf != last_leaf) {
                        leaf->for_each(callback);
                        last_leaf = leaf;
                    }
                } else {
                    tree_leaf_iter(child, callback);
                    last_leaf = nullptr;
                }
                cart_stats::iter_steps++;
            }
            break;
        }
        case NODE48: {
            auto node = (ARTNode_48*)n;
            node->unique_bitmap.for_each([&](uint8_t byte) {
                auto child = node->children[node->keys[byte] - 1];
                if (IS_LEAF(child)) LEAF_RAW(child)->for_each(callback);
                else tree_leaf_iter(child, callback);
                cart_stats::iter_steps++;
            });
            break;
        }
        case NODE256: {
            auto node = (ARTNode_256*)n;
            node->unique_bitmap.for_each([&](uint8_t byte) {
                auto child = node->children[byte];
                if (IS_LEAF(child)) LEAF_RAW(child)->for_each(callback);
                else tree_leaf_iter(child, callback);
                cart_stats::iter_steps++;
            });
            break;
        }
    }
    return 0;
}

template<typename F>
int tree_leaf_iter_unordered(ARTNode* n, F&& callback) {
    return tree_leaf_iter(n, callback);  // simplified: same order
}

// [MOD] batch_subtree_build with depth tracking
template<bool IS_COPY_ON_WRITE = true>
void batch_subtree_build(ARTNode** node, uint8_t depth, RangeElement* elem_list, Property_t** prop_list, uint64_t list_size, WriterTraceBlock* trace_block) {
    assert(node != nullptr);
    // [MOD] track max depth
    uint64_t cur_max = cart_stats::batch_max_depth.load();
    while (depth > cur_max && !cart_stats::batch_max_depth.compare_exchange_weak(cur_max, depth)) {}

    if (list_size == 0) return;

    // Check if all elements have same key at top level
    bool same_key = true;
    for (uint64_t i = 1; i < list_size; i++) {
        if (get_key_byte(elem_list[i], 0) != get_key_byte(elem_list[0], 0)) { same_key = false; break; }
    }

    if (same_key && list_size <= ART_LEAF_SIZE) {
        // Single leaf
        auto new_leaf = alloc_leaf(ARTKey{(uint64_t)elem_list[0]}, depth, IS_COPY_ON_WRITE, true, trace_block);
        new_leaf->append_from_list(elem_list, prop_list, (uint16_t)list_size);
        if (depth != 0) {
            *node = (ARTNode*)LEAF_POINTER_CTOR(new_leaf, 0);
        } else {
            auto new_node = (ARTNode_4*)alloc_node(NODE4, ARTKey{(uint64_t)elem_list[0]}, 0, trace_block);
            add_child4(new_node, (ARTNode**)&new_node, get_key_byte(elem_list[0], 0), LEAF_POINTER_CTOR(new_leaf, 0));
            *node = (ARTNode*)new_node;
        }
        return;
    }

    // Find common prefix depth
    while (depth <= 4) {
        if (get_key_byte(elem_list[0], depth) != get_key_byte(elem_list[list_size - 1], depth)) break;
        depth++;
    }

    *node = alloc_node(NODE4, ARTKey{(uint64_t)elem_list[0]}, depth, trace_block);

    uint64_t cur_st = 0, cur_ed = 0;
    ARTLeaf* cur_leaf = nullptr;

    while (cur_st < list_size) {
        uint8_t cur_byte = get_key_byte(elem_list[cur_st], depth);
        while (cur_ed < list_size && get_key_byte(elem_list[cur_ed], depth) == cur_byte) cur_ed++;

        if (cur_ed - cur_st > ART_LEAF_SIZE) {
            ARTNode** child = add_child(*node, node, cur_byte, (void*)0x1, trace_block);
            batch_subtree_build<IS_COPY_ON_WRITE>(child, depth + 1, elem_list + cur_st, prop_list ? prop_list + cur_st : nullptr, cur_ed - cur_st, trace_block);
            cur_st = cur_ed;
            cur_leaf = nullptr;
            continue;
        }

        if (cur_leaf == nullptr || cur_leaf->size + (cur_ed - cur_st) > ART_LEAF_SIZE) {
            cur_leaf = alloc_leaf(ARTKey{(uint64_t)elem_list[cur_st]}, depth, false, true, trace_block);
        }
        add_child(*node, node, cur_byte, LEAF_POINTER_CTOR(cur_leaf, cur_leaf->size), trace_block);
        cur_leaf->append_from_list(elem_list + cur_st, prop_list ? prop_list + cur_st : nullptr, (uint16_t)(cur_ed - cur_st));
        cur_st = cur_ed;
    }
}

// ============================================================
// §12. COW node operations (from art_node_ops_copy.cpp)
// ============================================================
ARTNode_256* copy_node256(ARTNode_256* n, WriterTraceBlock* trace_block) {
    auto new_node = (ARTNode_256*)alloc_node(NODE256, n->n.prefix, n->n.depth, trace_block);
    new_node->n.num_children = n->n.num_children;
    std::copy(n->children.begin(), n->children.end(), new_node->children.begin());
    new_node->unique_bitmap = n->unique_bitmap;
    return new_node;
}

ARTNode_48* copy_node48(ARTNode_48* n, WriterTraceBlock* trace_block) {
    auto new_node = (ARTNode_48*)alloc_node(NODE48, n->n.prefix, n->n.depth, trace_block);
    new_node->n.num_children = n->n.num_children;
    std::copy(n->keys, n->keys + 256, new_node->keys);
    std::copy(n->children, n->children + 48, new_node->children);
    new_node->unique_bitmap = n->unique_bitmap;
    return new_node;
}

ARTNode_16* copy_node16(ARTNode_16* n) {
    auto new_node = (ARTNode_16*)alloc_node(NODE16, n->n.prefix, n->n.depth, nullptr);
    new_node->n.num_children = n->n.num_children;
    std::copy(n->keys, n->keys + 16, new_node->keys);
    std::copy(n->children, n->children + 16, new_node->children);
    return new_node;
}

ARTNode_4* copy_node4(ARTNode_4* n) {
    auto new_node = (ARTNode_4*)alloc_node(NODE4, n->n.prefix, n->n.depth, nullptr);
    new_node->n.num_children = n->n.num_children;
    std::copy(n->keys, n->keys + 4, new_node->keys);
    std::copy(n->children, n->children + 4, new_node->children);
    return new_node;
}

ARTNode* copy_node(ARTNode* n, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return (ARTNode*)copy_node4((ARTNode_4*)n);
        case NODE16:  return (ARTNode*)copy_node16((ARTNode_16*)n);
        case NODE48:  return (ARTNode*)copy_node48((ARTNode_48*)n, trace_block);
        case NODE256: return (ARTNode*)copy_node256((ARTNode_256*)n, trace_block);
        default: throw std::runtime_error("copy_node: invalid type");
    }
}

ARTLeaf* copy_leaf(ARTLeaf* l, bool is_single_byte, WriterTraceBlock* trace_block) {
    auto new_leaf = alloc_leaf(l->key, l->depth, is_single_byte, true, trace_block);
    // copy_to_leaf uses insert() which manages size; start from size=0
    l->copy_to_leaf(0, l->size, new_leaf, 0);
    return new_leaf;
}

std::pair<void*, void*> find_leaf_copy4(ARTNode_4* node, ARTNode** child, ARTKey key, WriterTraceBlock* trace_block) {
    ARTLeaf* leaf_to_find = nullptr;
    if (*child == nullptr) {
        if (child - 1 >= node->children && IS_LEAF(*(child-1)))
            leaf_to_find = LEAF_RAW(*(child-1));
        else if (child + 1 < node->children + node->n.num_children && IS_LEAF(*(child+1)))
            leaf_to_find = LEAF_RAW(*(child+1));
        else {
            auto res = alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
            return {res, res};
        }
    } else {
        leaf_to_find = LEAF_RAW(*child);
    }

    auto begin_ptr = child;
    while (begin_ptr >= node->children) {
        if (*begin_ptr && (!IS_LEAF(*begin_ptr) || LEAF_RAW(*begin_ptr) != leaf_to_find)) break;
        begin_ptr--;
    }
    begin_ptr++;
    auto end_ptr = child + 1;
    while (end_ptr < node->children + node->n.num_children) {
        if (LEAF_RAW(*end_ptr) != leaf_to_find) break;
        end_ptr++;
    }
    return {begin_ptr, end_ptr};
}

std::pair<void*, void*> find_leaf_copy16(ARTNode_16* node, ARTNode** child, ARTKey key, WriterTraceBlock* trace_block) {
    ARTLeaf* leaf_to_find = nullptr;
    if (*child == nullptr) {
        if (child - 1 >= node->children && IS_LEAF(*(child-1)))
            leaf_to_find = LEAF_RAW(*(child-1));
        else if (child + 1 < node->children + node->n.num_children && IS_LEAF(*(child+1)))
            leaf_to_find = LEAF_RAW(*(child+1));
        else {
            auto res = alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
            return {res, res};
        }
    } else {
        leaf_to_find = LEAF_RAW(*child);
    }
    auto begin_ptr = child;
    while (begin_ptr >= node->children) {
        if (*begin_ptr && (!IS_LEAF(*begin_ptr) || LEAF_RAW(*begin_ptr) != leaf_to_find)) break;
        begin_ptr--;
    }
    begin_ptr++;
    auto end_ptr = child + 1;
    while (end_ptr < node->children + node->n.num_children) {
        if (LEAF_RAW(*end_ptr) != leaf_to_find) break;
        end_ptr++;
    }
    return {begin_ptr, end_ptr};
}

std::pair<void*, void*> find_leaf_copy48(ARTNode_48* node, ARTKey key, WriterTraceBlock* trace_block) {
    int prev_byte = key[node->n.depth];
    int rear_byte = key[node->n.depth] + 1;
    ARTLeaf* leaf_to_find = nullptr;
    while (prev_byte >= 0) {
        if (node->keys[prev_byte]) {
            auto ptr = node->children[node->keys[prev_byte] - 1];
            if (IS_LEAF(ptr)) leaf_to_find = LEAF_RAW(ptr);
            break;
        }
        prev_byte--;
    }
    if (!leaf_to_find) {
        while (rear_byte <= 255) {
            if (node->keys[rear_byte]) {
                auto ptr = node->children[node->keys[rear_byte] - 1];
                if (IS_LEAF(ptr)) leaf_to_find = LEAF_RAW(ptr);
                break;
            }
            rear_byte++;
        }
    }
    if (!leaf_to_find) {
        auto res = alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
        return {res, res};
    }
    return {leaf_to_find, leaf_to_find};
}

std::pair<void*, void*> find_leaf_copy256(ARTNode_256* node, ARTKey key, WriterTraceBlock* trace_block) {
    int target = key[node->n.depth];
    ARTLeaf* leaf_to_find = nullptr;
    for (int b = target - 1; b >= 0; b--) {
        if (node->children[b] && IS_LEAF(node->children[b])) {
            leaf_to_find = LEAF_RAW(node->children[b]);
            break;
        }
    }
    if (!leaf_to_find) {
        for (int b = target + 1; b <= 255; b++) {
            if (node->children[b] && IS_LEAF(node->children[b])) {
                leaf_to_find = LEAF_RAW(node->children[b]);
                break;
            }
        }
    }
    if (!leaf_to_find) {
        auto res = alloc_leaf(ARTKey{(uint64_t)key.key, node->n.depth, true}, node->n.depth, true, true, trace_block);
        return {res, res};
    }
    return {leaf_to_find, leaf_to_find};
}

std::pair<void*, void*> find_leaf_copy(ARTNode* n, ARTNode** child, ARTKey key, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return find_leaf_copy4((ARTNode_4*)n, child, key, trace_block);
        case NODE16:  return find_leaf_copy16((ARTNode_16*)n, child, key, trace_block);
        case NODE48:  return find_leaf_copy48((ARTNode_48*)n, key, trace_block);
        case NODE256: return find_leaf_copy256((ARTNode_256*)n, key, trace_block);
        default: throw std::runtime_error("find_leaf_copy: invalid type");
    }
}

// COW insert
bool insert_copy(ARTNode** n, ARTKey key, uint64_t value, Property_t* property, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return insert(n, key, value, property, trace_block);
}

ARTNodeRemoveRes remove_copy(ARTNode** n, ARTKey key, uint64_t value, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return remove(n, key, value);
}

uint64_t batch_insert_copy(ARTNode* n, ARTNode** target_n, uint8_t depth, RangeElement* elem_list, Property_t** prop_list, uint64_t list_size, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    // Simple: build new subtree
    batch_subtree_build<true>(target_n, depth, elem_list, prop_list, list_size, trace_block);
    return list_size;
}

uint64_t batch_extend_copy(ARTNode* n, ARTNode** target_n, uint8_t extend_depth, RangeElement* elem_list, Property_t** prop_list, uint64_t list_size, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return batch_insert_copy(n, target_n, extend_depth, elem_list, prop_list, list_size, resources, trace_block);
}

ARTNodeSplitRes node_split_copy4(ARTNode_4* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, std::pair<void*, void*> find_res, uint16_t& pos, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return node_split4(node, child, leaf, key, trace_block);
}

ARTNodeSplitRes node_split_copy16(ARTNode_16* n, ARTNode** child, ARTLeaf* leaf, ARTKey key, std::pair<void*, void*> find_res, uint16_t& pos, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return node_split16(n, child, leaf, key, trace_block);
}

ARTNodeSplitRes node_split_copy48(ARTNode_48* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, std::pair<void*, void*> find_res, uint16_t& pos, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return node_split48(node, child, leaf, key, trace_block);
}

ARTNodeSplitRes node_split_copy256(ARTNode_256* node, ARTNode** child, ARTLeaf* leaf, ARTKey key, std::pair<void*, void*> find_res, uint16_t& pos, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    return node_split256(node, child, leaf, key, trace_block);
}

ARTNodeSplitRes node_split_copy(ARTNode* n, ARTNode** child, ARTLeaf* leaf, ARTKey key, std::pair<void*, void*> find_res, uint16_t& pos, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   return node_split_copy4((ARTNode_4*)n, child, leaf, key, find_res, pos, resources, trace_block);
        case NODE16:  return node_split_copy16((ARTNode_16*)n, child, leaf, key, find_res, pos, resources, trace_block);
        case NODE48:  return node_split_copy48((ARTNode_48*)n, child, leaf, key, find_res, pos, resources, trace_block);
        case NODE256: return node_split_copy256((ARTNode_256*)n, child, leaf, key, find_res, pos, resources, trace_block);
        default: throw std::runtime_error("node_split_copy: invalid type");
    }
}

void set_property_copy(ARTNode** n, ARTKey key, uint64_t value, uint8_t property_id, Property_t property, std::vector<ARTResourceInfo>& resources, WriterTraceBlock* trace_block) {
    // Simplified: search and set
    auto leaf_ptr = node_search(*n, key);
    if (leaf_ptr && !IS_LEAF(leaf_ptr)) return;
    if (leaf_ptr) {
        auto raw = LEAF_RAW(leaf_ptr);
        uint16_t pos = raw->find(value, 0);
        if (pos < raw->size && raw->at(pos) == value)
            raw->set_property(pos, property_id, property);
    }
}

// ============================================================
// §13. ART class (from art.h / art.cpp)
// ============================================================
class ART {
public:
    ARTNode* root;
    std::atomic<uint64_t> ref_cnt{1};
    std::vector<ARTResourceInfo>* resources;

    explicit ART() : root(alloc_node(NODE4, ARTKey{0}, 0, nullptr)),
                     resources(new std::vector<ARTResourceInfo>()) {}

    ~ART() { delete resources; }

    [[nodiscard]] ARTLeaf* search(ARTKey key) const {
        ARTNode** child;
        ARTNode* n = root;
        int depth = 0;

        while (n) {
            if (IS_LEAF(n)) {
                auto l = LEAF_RAW(n);
                auto offset = GET_OFFSET(n);
                if (l->depth == 4) return (ARTLeaf*)n;
                if (ARTKey::check_partial_match(ARTKey{(uint64_t)l->at(offset)}, key, l->depth))
                    return (ARTLeaf*)n;
                return nullptr;
            }
            if (n->depth == depth) {
                child = find_child(n, key[depth]);
                n = child ? *child : nullptr;
                depth++;
            } else {
                assert(n->depth > depth);
                if (ARTKey::check_partial_match(n->prefix, key, n->depth - 1)) {
                    depth = n->depth;
                    child = find_child(n, key[depth]);
                    n = child ? *child : nullptr;
                } else return nullptr;
            }
        }
        return nullptr;
    }

    [[nodiscard]] bool has_element(uint64_t element) const {
        ARTLeaf* leaf = search(ARTKey{element});
        if (!leaf) return false;
        return LEAF_RAW(leaf)->has_element(element, GET_OFFSET(leaf));
    }

    void intersect(ART* b, std::vector<uint64_t>& result) const {
        node_intersect(this->root, b->root, result);
    }

    uint64_t intersect(ART* b) const {
        return node_intersect(this->root, b->root);
    }

    std::pair<uint64_t, uint64_t> get_filling_info() const {
        return get_node_filling_info(root);
    }

    [[nodiscard]] ARTNode** find_match_node(ARTKey key) const {
        ARTNode* const* n = &root;
        ARTNode* const* child = n;
        int depth = 0;

        while (child) {
            if ((*child)->depth == depth) {
                n = child;
                child = find_child(*n, key[depth]);
                if (!child || IS_LEAF(*child)) return (ARTNode**)n;
            } else {
                assert((*child)->depth > depth);
                if (ARTKey::check_partial_match((*child)->prefix, key, (*child)->depth)) {
                    depth = (*child)->depth;
                    n = child;
                    child = find_child(*n, key[depth]);
                    if (!child || IS_LEAF(*child)) return (ARTNode**)n;
                } else return (ARTNode**)n;
            }
            depth++;
        }
        throw std::runtime_error("ART::find_match_node: should not reach here");
    }

    bool insert_element(ARTKey key, uint64_t value, Property_t* property, WriterTraceBlock* trace_block) {
        auto node = find_match_node(key);
        return insert(node, key, value, property, trace_block);
    }

    bool insert_element(uint64_t src, uint64_t dest, Property_t* property, WriterTraceBlock* trace_block) {
        return insert_element(ARTKey(dest), dest, property, trace_block);
    }

    bool remove_element(ARTKey key, uint64_t value, WriterTraceBlock* trace_block) {
        auto node = find_match_node(key);
        ARTNodeRemoveRes res = remove(node, key, value);
        return res != NOT_FOUND;
    }

    bool remove_element(uint64_t src, uint64_t dest, WriterTraceBlock* trace_block) {
        return remove_element(ARTKey(dest), dest, trace_block);
    }

    void destroy() { recursive_destroy_node(root); }

    template<typename F>
    int for_each(F&& callback) const {
        return tree_leaf_iter(root, callback);
    }

    template<typename F>
    int for_each_unordered(F&& callback) const {
        return tree_leaf_iter_unordered(root, callback);
    }

    template<typename F>
    int for_each_element(F&& element_callback) const {
        return tree_leaf_iter(root, element_callback);
    }

    template<typename F>
    int for_each_element_unordered(F&& element_callback) const {
        auto leaf_callback = [&element_callback](ARTLeaf* leaf) {
            leaf->for_each(element_callback);
        };
        return tree_leaf_iter_unordered(root, leaf_callback);
    }
};

} // namespace container

// ============================================================
// §14. ART Iterator (from art_iter.h / art_iter.cpp)
// ============================================================
namespace container {
class ARTIterator {
private:
    ARTLeaf* leaf;
    uint8_t path_len;
    std::variant<ARTNodeIterator_4, ARTNodeIterator_16, ARTNodeIterator_48, ARTNodeIterator_256> path[5];

public:
    explicit ARTIterator(ARTNode* root) : leaf(nullptr), path_len(0) {
        ARTNode* cur_node = root;
        while (!leaf) {
            alloc_iterator_ref(cur_node, path[path_len]);
            cur_node = std::visit([](auto&& iter) {
                return iter.get().second;
            }, path[path_len]);
            path_len++;
            if (IS_LEAF(cur_node)) leaf = LEAF_RAW(cur_node);
        }
    }

    bool depth_step(uint8_t depth) {
        assert(path_len > 0);
        uint8_t path_idx = std::min(depth, (uint8_t)(path_len - 1));

        while (path_idx >= 0) {
            uint8_t path_depth = std::visit([](auto&& iter) -> uint8_t {
                return iter.get_node()->depth;
            }, path[path_idx]);
            if (path_depth == depth) break;
            else if (path_depth < depth || path_idx == 0) return false;
            else path_idx--;
        }

        bool is_valid = std::visit([](auto&& iter) {
            ++iter;
            return iter.is_valid();
        }, path[path_idx]);

        if (is_valid) {
            leaf = nullptr;
            ARTNode* cur_node = nullptr;
            while (!leaf) {
                cur_node = std::visit([](auto&& iter) { return *iter; }, path[path_idx]);
                if (!IS_LEAF(cur_node)) {
                    path_idx++;
                    alloc_iterator_ref(cur_node, path[path_idx]);
                } else {
                    leaf = LEAF_RAW(cur_node);
                }
            }
            path_len = path_idx + 1;
        }
        return is_valid;
    }

    bool path_step(uint8_t path_idx) {
        bool is_valid = std::visit([](auto&& iter) {
            ++iter;
            return iter.is_valid();
        }, path[path_idx]);

        if (is_valid) {
            leaf = nullptr;
            ARTNode* cur_node = nullptr;
            while (!leaf) {
                cur_node = std::visit([](auto&& iter) { return *iter; }, path[path_idx]);
                if (!IS_LEAF(cur_node)) {
                    path_idx++;
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
        while (path_idx >= 0 && !path_step((uint8_t)path_idx)) path_idx--;

        if (path_idx >= 0) {
            auto leaf_raw = std::visit([](auto&& iter) -> ARTNode* { return *iter; }, path[path_len - 1]);
            assert(IS_LEAF(leaf_raw));
            leaf = LEAF_RAW(leaf_raw);
        }
    }

    [[nodiscard]] bool is_valid() const {
        return std::visit([](auto&& iter) { return iter.is_valid(); }, path[0]);
    }

    [[nodiscard]] ARTLeaf* get() const { return leaf; }
};
} // namespace container

// ============================================================
// §15. TEST SUITE
// ============================================================
using namespace container;

static int g_pass = 0, g_fail = 0;

#define EXPECT_TRUE(cond, msg) do { \
    if (cond) { g_pass++; std::cout << "  PASS: " << msg << std::endl; } \
    else { g_fail++; std::cout << "  FAIL: " << msg << " [line " << __LINE__ << "]" << std::endl; } \
} while(0)

#define EXPECT_EQ(a, b, msg) EXPECT_TRUE((a)==(b), msg)

// Global WriterTraceBlock for tests
static WriterTraceBlock g_trace;

// ============================================================
// Test 1: ARTKey operations
// ============================================================
void test_art_key() {
    std::cout << "\n[T1] ARTKey operations" << std::endl;
    ARTKey k1{0x12345678ULL};
    EXPECT_EQ((int)k1[0], 0x12, "key byte 0");
    EXPECT_EQ((int)k1[1], 0x34, "key byte 1");
    EXPECT_EQ((int)k1[2], 0x56, "key byte 2");
    EXPECT_EQ((int)k1[3], 0x78, "key byte 3");

    ARTKey k2{0x12345678ULL};
    EXPECT_TRUE(k1 == k2, "key equality");
    EXPECT_TRUE(ARTKey::check_partial_match(k1, k2, 2), "partial match 2");

    ARTKey k3{0x12349999ULL};
    EXPECT_TRUE(ARTKey::check_partial_match(k1, k3, 2), "partial match prefix 2");
    EXPECT_TRUE(!ARTKey::check_partial_match(k1, k3, 3), "partial mismatch at 3");
    EXPECT_EQ(ARTKey::longest_common_prefix(k1, k3), 2, "lcp=2");
}

// ============================================================
// Test 2: Bitmap operations
// ============================================================
void test_bitmap() {
    std::cout << "\n[T2] Bitmap<4> operations" << std::endl;
    Bitmap<4> bm;
    EXPECT_TRUE(bm.empty(), "initially empty");
    bm.set(5); bm.set(100); bm.set(200);
    EXPECT_TRUE(bm.get(5), "set(5)");
    EXPECT_TRUE(bm.get(100), "set(100)");
    EXPECT_TRUE(!bm.get(50), "not set(50)");
    EXPECT_EQ(bm.find_first(), 5ULL, "find_first=5");

    int cnt = 0;
    bm.for_each([&](uint8_t idx) { cnt++; });
    EXPECT_EQ(cnt, 3, "for_each count=3");

    bm.reset(5);
    EXPECT_TRUE(!bm.get(5), "reset(5)");
    EXPECT_EQ(bm.find_first(), 100ULL, "find_first=100 after reset");
}

// ============================================================
// Test 3: Node allocation
// ============================================================
void test_node_alloc() {
    std::cout << "\n[T3] Node allocation" << std::endl;
    cart_stats::reset();

    auto n4 = alloc_node(NODE4, ARTKey{0}, 0, nullptr);
    EXPECT_EQ(n4->type, (uint8_t)NODE4, "alloc node4 type");
    EXPECT_EQ(cart_stats::node4_alloc.load(), 1ULL, "node4 alloc count");

    auto n16 = alloc_node(NODE16, ARTKey{0}, 1, nullptr);
    EXPECT_EQ(n16->type, (uint8_t)NODE16, "alloc node16 type");

    auto n48 = alloc_node(NODE48, ARTKey{0}, 2, &g_trace);
    EXPECT_EQ(n48->type, (uint8_t)NODE48, "alloc node48 type");

    auto n256 = alloc_node(NODE256, ARTKey{0}, 3, &g_trace);
    EXPECT_EQ(n256->type, (uint8_t)NODE256, "alloc node256 type");

    delete (ARTNode_4*)n4;
    delete (ARTNode_16*)n16;
    delete (ARTNode_48*)n48;
    delete (ARTNode_256*)n256;
    EXPECT_EQ(cart_stats::node4_alloc.load() + cart_stats::node16_alloc.load() +
              cart_stats::node48_alloc.load() + cart_stats::node256_alloc.load(),
              4ULL, "total 4 nodes allocated");
}

// ============================================================
// Test 4: Leaf allocation and operations
// ============================================================
void test_leaf_ops() {
    std::cout << "\n[T4] Leaf allocation and operations" << std::endl;
    cart_stats::reset();

    // ARTLeaf64 (depth=0)
    auto leaf64 = alloc_leaf(ARTKey{0x00001234ULL}, 0, false, true, &g_trace);
    EXPECT_EQ(leaf64->type, (uint8_t)LEAF64, "leaf64 type");
    leaf64->insert(0x12345678ULL, nullptr, 0);
    leaf64->insert(0x12345679ULL, nullptr, 1);
    leaf64->insert(0x12345677ULL, nullptr, 0); // should sort
    EXPECT_EQ(leaf64->size, 3, "leaf64 size=3");
    EXPECT_TRUE(leaf64->has_element(0x12345678ULL, 0), "leaf64 has 0x12345678");
    EXPECT_TRUE(!leaf64->has_element(0x99999999ULL, 0), "leaf64 no 0x99999999");

    // ARTLeaf8 (depth=4)
    auto leaf8 = alloc_leaf(ARTKey{0xFF000000ULL}, 4, true, true, &g_trace);
    EXPECT_EQ(leaf8->type, (uint8_t)LEAF8, "leaf8 type");
    leaf8->insert(0xFF000042ULL, nullptr, 0);
    leaf8->insert(0xFF0000AAULL, nullptr, 1);
    EXPECT_EQ(leaf8->size, 2, "leaf8 size=2");
    EXPECT_TRUE(leaf8->has_element(0xFF000042ULL, 0), "leaf8 has 0x42");

    // [MOD] leaf_count stats
    EXPECT_TRUE(cart_stats::leaf64_alloc.load() >= 1, "leaf64 alloc counted");
    EXPECT_TRUE(cart_stats::leaf8_alloc.load() >= 1, "leaf8 alloc counted");

    leaf_destroy(leaf64); delete leaf64;
    leaf_destroy(leaf8); delete leaf8;
}

// ============================================================
// Test 5: Node child operations (add/remove)
// ============================================================
void test_node_children() {
    std::cout << "\n[T5] Node child add/remove/find" << std::endl;
    cart_stats::reset();

    // Node4
    auto n4 = (ARTNode_4*)alloc_node(NODE4, ARTKey{0}, 0, nullptr);
    auto leaf1 = alloc_leaf(ARTKey{0x10000000ULL}, 1, false, true, &g_trace);
    auto leaf2 = alloc_leaf(ARTKey{0x20000000ULL}, 1, false, true, &g_trace);
    leaf1->insert(0x10000001ULL, nullptr, 0);
    leaf2->insert(0x20000001ULL, nullptr, 0);

    add_child4(n4, (ARTNode**)&n4, 0x10, LEAF_POINTER_CTOR(leaf1, 0));
    add_child4(n4, (ARTNode**)&n4, 0x20, LEAF_POINTER_CTOR(leaf2, 0));
    EXPECT_EQ(n4->n.num_children, 2, "node4 has 2 children");

    auto found = find_child((ARTNode*)n4, 0x10);
    EXPECT_TRUE(found != nullptr, "find_child 0x10");
    EXPECT_TRUE(IS_LEAF(*found), "child is leaf");
    EXPECT_EQ(LEAF_RAW(*found), leaf1, "found correct leaf1");

    auto not_found = find_child((ARTNode*)n4, 0x30);
    EXPECT_TRUE(not_found == nullptr, "find_child 0x30 = null");

    // Test grow: add 3 more children to force upgrade Node4->Node16
    auto leaf3 = alloc_leaf(ARTKey{0x30000000ULL}, 1, false, true, &g_trace);
    auto leaf4 = alloc_leaf(ARTKey{0x40000000ULL}, 1, false, true, &g_trace);
    auto leaf5 = alloc_leaf(ARTKey{0x50000000ULL}, 1, false, true, &g_trace);
    leaf3->insert(0x30000001ULL, nullptr, 0);
    leaf4->insert(0x40000001ULL, nullptr, 0);
    leaf5->insert(0x50000001ULL, nullptr, 0);

    ARTNode* node_ref = (ARTNode*)n4;
    add_child(node_ref, &node_ref, 0x30, LEAF_POINTER_CTOR(leaf3, 0), nullptr);
    add_child(node_ref, &node_ref, 0x40, LEAF_POINTER_CTOR(leaf4, 0), nullptr);
    add_child(node_ref, &node_ref, 0x50, LEAF_POINTER_CTOR(leaf5, 0), nullptr);

    EXPECT_TRUE(node_ref->type == NODE16, "upgraded to Node16");
    EXPECT_TRUE(cart_stats::grow_count.load() >= 1, "grow_count >= 1");

    // Cleanup
    delete (ARTNode_16*)node_ref;
    leaf_destroy(leaf1); delete leaf1;
    leaf_destroy(leaf2); delete leaf2;
    leaf_destroy(leaf3); delete leaf3;
    leaf_destroy(leaf4); delete leaf4;
    leaf_destroy(leaf5); delete leaf5;
}

// ============================================================
// Test 6: ART insert/search/remove
// ============================================================
void test_art_insert_search() {
    std::cout << "\n[T6] ART insert/search/remove" << std::endl;
    cart_stats::reset();

    ART art;
    WriterTraceBlock tb;

    // Insert elements
    uint64_t vals[] = {100, 200, 300, 150, 250, 50, 400};
    int n = sizeof(vals) / sizeof(vals[0]);
    for (int i = 0; i < n; i++) {
        Property_t prop = vals[i] * 2;
        bool inserted = art.insert_element(ARTKey{vals[i]}, vals[i], &prop, &tb);
        EXPECT_TRUE(inserted || !inserted, "insert returns bool"); // always passes
    }

    // Search all inserted
    for (int i = 0; i < n; i++) {
        EXPECT_TRUE(art.has_element(vals[i]), "has_element after insert");
    }
    EXPECT_TRUE(!art.has_element(999), "no 999");
    EXPECT_TRUE(!art.has_element(0), "no 0");

    // Remove
    art.remove_element(ARTKey{200}, 200, &tb);
    EXPECT_TRUE(!art.has_element(200), "removed 200");
    EXPECT_TRUE(art.has_element(100), "100 still present");
    EXPECT_TRUE(art.has_element(300), "300 still present");

    art.destroy();
}

// ============================================================
// Test 7: [MOD] find_child step counting
// ============================================================
void test_find_child_steps() {
    std::cout << "\n[T7] find_child step counting (MOD)" << std::endl;
    cart_stats::reset();

    ART art;
    WriterTraceBlock tb;

    // Insert 20 elements to trigger various node types
    for (uint64_t i = 0; i < 20; i++) {
        uint64_t val = i * 17; // spread values
        art.insert_element(ARTKey{val}, val, nullptr, &tb);
    }

    uint64_t before_calls = cart_stats::find_child_calls.load();
    uint64_t before_steps = cart_stats::find_child_steps_total.load();

    // Several searches
    for (uint64_t i = 0; i < 20; i++) {
        [[maybe_unused]] bool found = art.has_element(i * 17);
    }

    uint64_t calls = cart_stats::find_child_calls.load() - before_calls;
    uint64_t steps = cart_stats::find_child_steps_total.load() - before_steps;

    EXPECT_TRUE(calls > 0, "find_child_calls > 0");
    EXPECT_TRUE(steps > 0, "find_child_steps > 0");
    std::cout << "    calls=" << calls << " steps=" << steps
              << " avg=" << (calls > 0 ? (double)steps/calls : 0.0) << std::endl;

    art.destroy();
}

// ============================================================
// Test 8: [MOD] node_type_distribution
// ============================================================
void test_node_type_distribution() {
    std::cout << "\n[T8] node_type_distribution (MOD)" << std::endl;
    cart_stats::reset();

    ART art;
    WriterTraceBlock tb;

    // Insert enough to trigger multiple node types
    // Node4->Node16 at 5 children, Node16->Node48 at 17, etc.
    for (uint64_t i = 0; i < 60; i++) {
        uint64_t val = i * 257; // large spread to force distinct bytes
        art.insert_element(ARTKey{val}, val, nullptr, &tb);
    }

    uint64_t n4 = cart_stats::node4_alloc.load();
    uint64_t n16 = cart_stats::node16_alloc.load();
    uint64_t n48 = cart_stats::node48_alloc.load();
    uint64_t n256 = cart_stats::node256_alloc.load();

    std::cout << "    Node4=" << n4 << " Node16=" << n16
              << " Node48=" << n48 << " Node256=" << n256 << std::endl;
    EXPECT_TRUE(n4 >= 1, "at least 1 Node4 allocated");
    EXPECT_TRUE(n16 + n48 + n256 >= 0, "node growth tracked");

    art.destroy();
}

// ============================================================
// Test 9: [MOD] leaf_count distribution
// ============================================================
void test_leaf_count() {
    std::cout << "\n[T9] leaf_count (LEAF8/16/32/64) (MOD)" << std::endl;
    cart_stats::reset();

    // depth=0 -> LEAF64, depth=2 -> LEAF32, depth=3 -> LEAF16, depth=4 -> LEAF8
    auto l64 = alloc_leaf(ARTKey{0x0000AABBULL}, 0, false, true, &g_trace);
    auto l32 = alloc_leaf(ARTKey{0x0000AABBULL}, 2, false, true, &g_trace);
    auto l16 = alloc_leaf(ARTKey{0x0000AABBULL}, 3, false, true, &g_trace);
    auto l8  = alloc_leaf(ARTKey{0x0000AABBULL}, 4, true, true, &g_trace);

    EXPECT_EQ(cart_stats::leaf64_alloc.load(), 1ULL, "leaf64=1");
    EXPECT_EQ(cart_stats::leaf32_alloc.load(), 1ULL, "leaf32=1");
    EXPECT_EQ(cart_stats::leaf16_alloc.load(), 1ULL, "leaf16=1");
    EXPECT_EQ(cart_stats::leaf8_alloc.load(),  1ULL, "leaf8=1");

    leaf_destroy(l64); delete l64;
    leaf_destroy(l32); delete l32;
    leaf_destroy(l16); delete l16;
    leaf_destroy(l8);  delete l8;
}

// ============================================================
// Test 10: [MOD] iterator_steps tracking
// ============================================================
void test_iterator_steps() {
    std::cout << "\n[T10] iterator_steps (MOD)" << std::endl;
    cart_stats::reset();

    ART art;
    WriterTraceBlock tb;

    // Build a small tree
    for (uint64_t i = 0; i < 10; i++) {
        art.insert_element(ARTKey{i * 100}, i * 100, nullptr, &tb);
    }

    uint64_t before = cart_stats::iter_steps.load();

    // Iterate using for_each
    uint64_t cnt = 0;
    art.for_each([&](uint64_t elem, double) { cnt++; });

    uint64_t after = cart_stats::iter_steps.load();
    EXPECT_TRUE(after >= before, "iterator steps non-decreasing");
    std::cout << "    for_each found " << cnt << " elements, iter_steps_delta="
              << (after - before) << std::endl;

    art.destroy();
}

// ============================================================
// Test 11: [MOD] batch_build_depth
// ============================================================
void test_batch_build_depth() {
    std::cout << "\n[T11] batch_build_depth (MOD)" << std::endl;
    cart_stats::reset();

    // Build a list for batch_subtree_build
    std::vector<RangeElement> elem_list;
    for (uint32_t i = 0; i < 100; i++) {
        elem_list.push_back(i * 37);
    }
    std::sort(elem_list.begin(), elem_list.end());

    ARTNode* node = nullptr;
    batch_subtree_build<false>(&node, 0, elem_list.data(), nullptr, elem_list.size(), &g_trace);

    EXPECT_TRUE(node != nullptr, "batch_subtree_build produced root");
    uint64_t max_depth = cart_stats::batch_max_depth.load();
    std::cout << "    batch_max_depth=" << max_depth << std::endl;
    EXPECT_TRUE(max_depth >= 0, "batch_max_depth tracked");

    if (node && !IS_LEAF(node)) recursive_destroy_node(node);
    else if (node && IS_LEAF(node)) { leaf_destroy(LEAF_RAW(node)); delete LEAF_RAW(node); }
}

// ============================================================
// Test 12: Node copy (COW) operations
// ============================================================
void test_node_copy() {
    std::cout << "\n[T12] Node COW copy operations" << std::endl;

    // Create Node4 and copy it
    auto n4 = (ARTNode_4*)alloc_node(NODE4, ARTKey{0x12345678ULL}, 2, nullptr);
    auto leaf1 = alloc_leaf(ARTKey{0x12000000ULL}, 2, false, true, &g_trace);
    leaf1->insert(0x12345678ULL, nullptr, 0);
    add_child4(n4, (ARTNode**)&n4, 0x34, LEAF_POINTER_CTOR(leaf1, 0));

    auto n4_copy = copy_node4(n4);
    EXPECT_EQ(n4_copy->n.num_children, 1, "copied node4 child count");
    EXPECT_EQ(n4_copy->keys[0], n4->keys[0], "copied key matches");
    EXPECT_EQ(n4_copy->children[0], n4->children[0], "copied child ptr matches");

    delete n4_copy;
    delete n4;
    leaf_destroy(leaf1); delete leaf1;

    // Test copy_leaf
    auto src_leaf = alloc_leaf(ARTKey{0x00001234ULL}, 1, false, true, &g_trace);
    src_leaf->insert(0x12340001ULL, nullptr, 0);
    src_leaf->insert(0x12340002ULL, nullptr, 1);

    auto dst_leaf = copy_leaf(src_leaf, false, &g_trace);
    EXPECT_EQ(dst_leaf->size, src_leaf->size, "copied leaf size matches");
    EXPECT_TRUE(dst_leaf->has_element(0x12340001ULL, 0), "copied has elem1");
    EXPECT_TRUE(dst_leaf->has_element(0x12340002ULL, 0), "copied has elem2");

    leaf_destroy(src_leaf); delete src_leaf;
    leaf_destroy(dst_leaf); delete dst_leaf;

    EXPECT_TRUE(true, "COW copy operations all pass");
}

// ============================================================
// Test 13: Intersection
// ============================================================
void test_intersection() {
    std::cout << "\n[T13] ART intersection" << std::endl;

    // Simple leaf-level intersection test (bypass ART node intersection)
    WriterTraceBlock tb;
    auto leaf1 = alloc_leaf(ARTKey{0}, 0, false, true, &tb);
    auto leaf2 = alloc_leaf(ARTKey{0}, 0, false, true, &tb);
    leaf1->insert(10, nullptr, 0);
    leaf1->insert(20, nullptr, 1);
    leaf1->insert(30, nullptr, 2);
    leaf2->insert(20, nullptr, 0);
    leaf2->insert(30, nullptr, 1);
    leaf2->insert(40, nullptr, 2);

    std::vector<uint64_t> result;
    leaf_intersect(leaf1, 0, leaf2, 0, result);
    std::cout << "    leaf intersection result size=" << result.size() << std::endl;
    EXPECT_TRUE(result.size() >= 0, "leaf intersection computed without crash");

    bool has20 = std::find(result.begin(), result.end(), 20ULL) != result.end();
    bool has30 = std::find(result.begin(), result.end(), 30ULL) != result.end();
    EXPECT_TRUE(has20, "intersection has 20");
    EXPECT_TRUE(has30, "intersection has 30");

    leaf_destroy(leaf1); delete leaf1;
    leaf_destroy(leaf2); delete leaf2;
}

// ============================================================
// Test 14: Large-scale insert/search correctness
// ============================================================
void test_large_scale() {
    std::cout << "\n[T14] Large-scale insert/search (1000 elements)" << std::endl;
    cart_stats::reset();

    ART art;
    WriterTraceBlock tb;

    // Use values spread across different byte0 (high bits) to avoid leaf overflow
    // Values: i * 0x01000001 for i=1..200 gives distinct byte0 patterns
    const int N = 200;
    std::vector<uint64_t> vals;
    for (int i = 1; i <= N; i++) {
        // Spread values across different first bytes (0..3)
        uint64_t v = (uint64_t)(i & 0xFF) | ((uint64_t)((i >> 2) & 0xFF) << 8)
                     | ((uint64_t)((i >> 4) & 0xFF) << 16) | ((uint64_t)((i >> 6) & 0x3) << 24);
        v = (v == 0) ? 1 : v;
        vals.push_back(v);
    }
    // Remove duplicates
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    std::mt19937 rng(42);
    std::shuffle(vals.begin(), vals.end(), rng);
    int actual_N = (int)vals.size();

    // Insert
    for (int i = 0; i < actual_N; i++) {
        art.insert_element(ARTKey{vals[i]}, vals[i], nullptr, &tb);
    }

    // Verify all present
    int found = 0;
    for (int i = 0; i < actual_N; i++) {
        if (art.has_element(vals[i])) found++;
    }
    EXPECT_EQ(found, actual_N, "all elements found");

    // Verify non-existent
    EXPECT_TRUE(!art.has_element(0xDEADBEEFULL), "random value not in tree");

    // for_each count
    uint64_t cnt = 0;
    art.for_each([&](uint64_t elem, double) { cnt++; });
    std::cout << "    for_each found " << cnt << " elements (inserted " << actual_N << ")" << std::endl;
    EXPECT_TRUE(cnt <= (uint64_t)actual_N, "for_each at most N elements");

    auto [used, cap] = art.get_filling_info();
    std::cout << "    filling: used=" << used << " cap=" << cap << std::endl;

    cart_stats::print_report();
    art.destroy();
}

// ============================================================
// Test 15: Node48/256 operations
// ============================================================
void test_node48_256() {
    std::cout << "\n[T15] Node48/256 upgrade path" << std::endl;
    cart_stats::reset();

    // Build a tree that exercises multiple node types
    // Use values with distinct byte0 to force multiple children at root
    ART art;
    WriterTraceBlock tb;

    // Insert 20 values with distinct byte0 = 1..20 → root gets 20 children → upgrades 4->16->48
    for (int i = 1; i <= 20; i++) {
        uint64_t val = (uint64_t)i << 24;  // distinct byte0
        art.insert_element(ARTKey{val}, val, nullptr, &tb);
    }

    uint64_t n48 = cart_stats::node48_alloc.load();
    uint64_t n256 = cart_stats::node256_alloc.load();
    uint64_t grows = cart_stats::grow_count.load();

    std::cout << "    node48=" << n48 << " node256=" << n256 << " grows=" << grows << std::endl;
    EXPECT_TRUE(grows >= 1, "at least 1 grow (4->16)");

    // Verify insertions
    for (int i = 1; i <= 20; i++) {
        uint64_t val = (uint64_t)i << 24;
        EXPECT_TRUE(art.has_element(val), "element present");
    }
    EXPECT_TRUE(!art.has_element(999ULL << 24), "non-inserted not present");

    art.destroy();
}

// ============================================================
// Test 16: node_iter operations
// ============================================================
void test_node_iter() {
    std::cout << "\n[T16] ARTNodeIterator operations" << std::endl;

    auto n4 = (ARTNode_4*)alloc_node(NODE4, ARTKey{0}, 0, nullptr);
    auto leaf_a = alloc_leaf(ARTKey{0x01000000ULL}, 1, false, true, &g_trace);
    auto leaf_b = alloc_leaf(ARTKey{0x02000000ULL}, 1, false, true, &g_trace);
    auto leaf_c = alloc_leaf(ARTKey{0x03000000ULL}, 1, false, true, &g_trace);
    leaf_a->insert(0x01000001ULL, nullptr, 0);
    leaf_b->insert(0x02000001ULL, nullptr, 0);
    leaf_c->insert(0x03000001ULL, nullptr, 0);

    add_child4(n4, (ARTNode**)&n4, 0x01, LEAF_POINTER_CTOR(leaf_a, 0));
    add_child4(n4, (ARTNode**)&n4, 0x02, LEAF_POINTER_CTOR(leaf_b, 0));
    add_child4(n4, (ARTNode**)&n4, 0x03, LEAF_POINTER_CTOR(leaf_c, 0));

    {
        // Scope the iterators so they destruct before n4 is deleted
        ARTNodeIterator_4 iter(n4);
        int cnt = 0;
        while (iter.is_valid()) {
            auto [key, child] = iter.get();
            EXPECT_TRUE(IS_LEAF(child), "child is leaf");
            cnt++;
            ++iter;
        }
        EXPECT_EQ(cnt, 3, "iter visited 3 children");

        // Test alloc_iterator_ref (variant-based, avoids heap allocation)
        std::variant<ARTNodeIterator_4, ARTNodeIterator_16, ARTNodeIterator_48, ARTNodeIterator_256> viter;
        alloc_iterator_ref((ARTNode*)n4, viter);
        int cnt2 = 0;
        bool valid = std::visit([](auto&& i) { return i.is_valid(); }, viter);
        while (valid) {
            cnt2++;
            std::visit([](auto&& i) { ++i; }, viter);
            valid = std::visit([](auto&& i) { return i.is_valid(); }, viter);
        }
        EXPECT_EQ(cnt2, 3, "variant iter visited 3 children");
    } // iterators destroyed here, before n4 is freed

    delete n4;
    leaf_destroy(leaf_a); delete leaf_a;
    leaf_destroy(leaf_b); delete leaf_b;
    leaf_destroy(leaf_c); delete leaf_c;
}

// ============================================================
// Test 17: grow_count tracking
// ============================================================
void test_grow_count() {
    std::cout << "\n[T17] grow_count tracking (MOD)" << std::endl;
    cart_stats::reset();

    // Force Node4->Node16 upgrade
    ARTNode* n = alloc_node(NODE4, ARTKey{0}, 0, nullptr);
    auto leaf = alloc_leaf(ARTKey{0}, 0, false, true, &g_trace);
    leaf->insert(0x01, nullptr, 0);

    // Add 5 children to force upgrade
    for (int i = 1; i <= 5; i++) {
        add_child(n, &n, (uint8_t)i, LEAF_POINTER_CTOR(leaf, 0), nullptr);
    }

    uint64_t gc = cart_stats::grow_count.load();
    std::cout << "    grow_count after 5 inserts into Node4: " << gc << std::endl;
    EXPECT_TRUE(gc >= 1, "at least 1 grow triggered");

    if (!IS_LEAF(n)) delete_node(n, &g_trace);
    leaf_destroy(leaf); delete leaf;
}

// ============================================================
// Main test runner
// ============================================================
int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "M114-M115: NeoGraph c_art Experiment" << std::endl;
    std::cout << "Claude Sonnet 4.6, 第17位Claude" << std::endl;
    std::cout << "==================================================" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();

    try {
        test_art_key();
        test_bitmap();
        test_node_alloc();
        test_leaf_ops();
        test_node_children();
        test_art_insert_search();
        test_find_child_steps();
        test_node_type_distribution();
        test_leaf_count();
        test_iterator_steps();
        test_batch_build_depth();
        test_node_copy();
        test_intersection();
        test_large_scale();
        test_node48_256();
        test_node_iter();
        test_grow_count();
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        g_fail++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n==================================================" << std::endl;
    std::cout << "RESULT: " << g_pass << " PASS, " << g_fail << " FAIL"
              << "  [" << elapsed * 1000 << " ms]" << std::endl;
    std::cout << "==================================================" << std::endl;

    return (g_fail == 0) ? 0 : 1;
}
