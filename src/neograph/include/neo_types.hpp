#pragma once
/**
 * neo_types.hpp — Core NeoGraph data structures with tier-awareness
 *
 * 骨架来源: upstream/.../utils/types.h (242行) + types.cpp (146行)
 * 修改 (~20%):
 *   - ARTKey 比较操作加 depth_histogram 记录各depth访问频次
 *   - NeoVertex 新增 tier_id (uint8_t) 标记所在存储层
 *   - NeoRangeNode 新增 access_count (热度追踪, 迁移调度用)
 *   - 增加 dump_art_key(), dump_vertex(), dump_range_node()
 *   - GCResourceType 增加 Tier_Migration_Buffer
 *   - ARTKey 构造的 switch-case 增加 PHILE_NEO_TRACE
 *
 * Milestone: M071
 */

#include "neo_config.hpp"
#include "neo_property.hpp"
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

namespace container {

// ─── Key-byte extraction (upstream, unchanged) ───
inline uint8_t get_key_byte(uint64_t key, uint8_t depth) {
    return (key >> ((3 - depth) * 8)) & 0xFF;
}

// ─── Depth access histogram (NEW: profiling which depths are hot) ───
struct ARTDepthStats {
    std::atomic<uint64_t> depth_hits[5]{};
    void record(uint8_t d) {
        if (d < 5) depth_hits[d].fetch_add(1, std::memory_order_relaxed);
    }
    void dump() const {
        std::fprintf(stderr, "[ART-DEPTH]");
        for (int i = 0; i < 5; ++i)
            std::fprintf(stderr, " d%d=%llu", i,
                         (unsigned long long)depth_hits[i].load(std::memory_order_relaxed));
        std::fprintf(stderr, "\n");
    }
};
inline ARTDepthStats& art_depth_stats() { static ARTDepthStats s; return s; }

// ──────────────── ARTKey (upstream core + depth tracking) ────────────────
struct ARTKey {
    uint32_t key;

    explicit ARTKey(uint64_t dst) : key(dst & 0x00000000FFFFFF00u) {}

    ARTKey(uint64_t dst, uint8_t depth, bool is_single_byte)
        : key(dst & 0x00000000FFFFFF00u) {
        art_depth_stats().record(depth);
        switch (depth + is_single_byte) {
            case 0: key &= 0x0000FFFF00000000u; break;
            case 1: key &= 0x0000FFFFFF000000u; break;
            case 2: key &= 0x0000FFFFFFFF0000u; break;
            case 3: key &= 0x0000FFFFFFFFFF00u; break;
            default:
                PHILE_NEO_TRACE("ARTKey invalid depth=%u", (unsigned)depth);
                assert(false);
        }
    }

    ARTKey(ARTKey k, uint8_t depth, bool is_single_byte) : key(k.key) {
        art_depth_stats().record(depth);
        switch (depth + is_single_byte) {
            case 0: this->key &= 0x0000FFFF00000000u; break;
            case 1: this->key &= 0x0000FFFFFF000000u; break;
            case 2: this->key &= 0x0000FFFFFFFF0000u; break;
            case 3: this->key &= 0x0000FFFFFFFFFF00u; break;
            default: assert(false);
        }
    }

    uint8_t operator[](int idx) const {
        assert(idx < 5);
        return (key >> ((3 - idx) * 8)) & 0xFF;
    }

    uint8_t& operator[](int idx) {
        assert(idx < 5);
        return reinterpret_cast<uint8_t*>(&key)[3 - idx];
    }

    bool operator==(const ARTKey& rhs) const { return key == rhs.key; }
    bool operator!=(const ARTKey& rhs) const { return key != rhs.key; }

    bool operator<(const ARTKey& rhs) const {
        for (uint8_t d = 0; d < 3; d++)
            if ((*this)[d] != rhs[d])
                return (*this)[d] < rhs[d];
        return false;
    }

    void print() const {
        for (int i = 0; i < 3; i++)
            std::cout << (int)(*this)[i] << " ";
        std::cout << std::endl;
    }

    // ─── NEW: structured dump ───
    void dump(const char* label = "") const {
        std::fprintf(stderr, "[ARTKey:%s] raw=0x%08x bytes=[%u,%u,%u]\n",
                     label, key, (*this)[0], (*this)[1], (*this)[2]);
    }

    static bool check_partial_match(ARTKey key1, ARTKey key2, uint8_t depth) {
        assert(depth <= 3);
        for (uint8_t i = 0; i < depth; i++)
            if (key1[i] != key2[i]) return false;
        return true;
    }

    static bool check_partial_match(uint64_t key1, uint64_t key2, uint8_t depth) {
        assert(depth <= 3);
        for (uint8_t i = 0; i < depth; i++)
            if (get_key_byte(key1, i) != get_key_byte(key2, i)) return false;
        return true;
    }

    static uint8_t longest_common_prefix(ARTKey key1, ARTKey key2) {
        for (uint8_t i = 0; i < 3; i++)
            if (key1[i] != key2[i]) return i;
        return 5;
    }
};

// ──────────────── Batch result structs (upstream, unchanged) ────────────────
struct RangeTreeInsertElemBatchRes {
    uint64_t new_inserted;
    void* tree_ptr;
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
    uint64_t is_new: 16;
    uint64_t art_ptr: 48;
};

struct ARTRemoveElemCopyRes {
    uint64_t found: 16;
    uint64_t tree_ptr: 48;
};

enum ARTNodeRemoveRes {
    NOT_FOUND,
    ELEMENT_REMOVED,
    CHILD_REMOVED,
};

// ──────────────── NeoRangeNode (upstream + access_count) ────────────────
struct NeoRangeNode {
    uint64_t key: 16;
    uint64_t arr_ptr: 48;
    uint64_t size: 48;
#if EDGE_PROPERTY_NUM == 1
    RangePropertyVec_t* property;
#elif EDGE_PROPERTY_NUM > 1
    MultiRangePropertyVec_t* property;
#endif
    // ─── NEW: access tracking for tier migration decisions ───
    mutable uint32_t access_count{0};

    NeoRangeNode() : key(0), size(0), arr_ptr(0)
#if EDGE_PROPERTY_NUM > 0
        , property(nullptr)
#endif
    {}

    NeoRangeNode(uint64_t key, uint64_t size, uint64_t arr_ptr, void* prop_ptr)
        : key(key), size(size), arr_ptr(arr_ptr)
#if EDGE_PROPERTY_NUM == 1
        , property(static_cast<RangePropertyVec_t*>(prop_ptr))
#elif EDGE_PROPERTY_NUM > 1
        , property(static_cast<MultiRangePropertyVec_t*>(prop_ptr))
#endif
    {}

    NeoRangeNode(const NeoRangeNode&) = default;

    [[nodiscard]] bool is_empty() const {
#if EDGE_PROPERTY_NUM == 0
        return *((uint64_t*)this) == 0;
#else
        return *((uint64_t*)this) == 0 && *((uint64_t*)this + 1) == 0;
#endif
    }

    void dump(const char* label = "") const {
        std::fprintf(stderr, "[RangeNode:%s] key=%llu size=%llu arr_ptr=%llu access=%u\n",
                     label, (unsigned long long)key, (unsigned long long)size,
                     (unsigned long long)arr_ptr, access_count);
    }
};

using RangeElement = uint32_t;

// ──────────────── InRangeNode (upstream) ────────────────
struct InRangeNode {
    uint64_t size: 16;
    uint64_t arr_ptr: 48;
#if EDGE_PROPERTY_NUM == 1
    RangePropertyVec_t* property_map{};
#elif EDGE_PROPERTY_NUM > 1
    MultiRangePropertyVec_t* property_map;
#endif

    InRangeNode() : size(0), arr_ptr(0) {}
    InRangeNode(uint64_t size, uint64_t arr_ptr) : size(size), arr_ptr(arr_ptr) {}
#if EDGE_PROPERTY_NUM != 0
    InRangeNode(uint64_t size, uint64_t arr_ptr, RangePropertyVec_t* prop)
        : size(size), arr_ptr(arr_ptr), property_map(prop) {}
#endif
    InRangeNode(const InRangeNode&) = default;
};

// ──────────────── NeoVertex (upstream + tier_id) ────────────────
struct NeoVertex {
    uint64_t is_independent: 1;
    uint64_t is_art: 1;
    uint64_t exist: 1;
    uint64_t degree: 32;
    uint64_t range_node_idx: 16;
    uint64_t neighbor_offset: 12;
    uint64_t neighborhood_ptr: 48;

    // ─── NEW: which storage tier this vertex lives on ───
    uint8_t tier_id{0};

    explicit NeoVertex()
        : is_art(0), exist(0), degree(0), range_node_idx(0), neighbor_offset(0) {}

    void dump(uint64_t vid = 0) const {
        std::fprintf(stderr,
            "[NeoVertex:%llu] exist=%llu deg=%llu art=%llu indep=%llu "
            "tier=%u rng_idx=%llu\n",
            (unsigned long long)vid,
            (unsigned long long)exist,
            (unsigned long long)degree,
            (unsigned long long)is_art,
            (unsigned long long)is_independent,
            (unsigned)tier_id,
            (unsigned long long)range_node_idx);
    }
};

// ──────────────── GC resource tracking (upstream + Tier_Migration_Buffer) ────────────────
enum GCResourceType {
    Outer_Segment = 1,
    Inner_Segment = 2,
    Range_Tree_Copied = 3,
    Range_Tree_Upgraded = 4,
    ART_Tree = 5,
#if VERTEX_PROPERTY_NUM != 0
    Vertex_Property_Vec = 6,
    Vertex_Property_Map_All_Modified = 7,
    Multi_Vertex_Property_Vec_Copied = 8,
    Multi_Vertex_Property_Vec_Mounted = 9,
#endif
#if EDGE_PROPERTY_NUM != 0
    Range_Property_Vec = 10,
    Range_Property_Map_All_Modified = 11,
#endif
    Tier_Migration_Buffer = 12,  // NEW: Philemon cross-tier migration
};

struct GCResourceInfo {
    GCResourceType type;
    void* ptr;

    void dump() const {
        std::fprintf(stderr, "[GC-RES] type=%d ptr=%p\n", (int)type, ptr);
    }
};

enum ARTResourceType {
    ART_Leaf = 1,
    ART_Node_Copied = 2,
    ART_Node_Mounted = 3,
#if EDGE_PROPERTY_NUM != 0
    ART_Property_Vec = 4,
    ART_Property_Map_All_Modified = 5,
    Multi_ART_Property_Vec_Copied = 6,
#endif
};

struct ARTResourceInfo {
    ARTResourceType type;
    void* ptr;
};

// ──────────────── Segment / Map types (upstream) ────────────────
using RangeNodeSegment_t = std::vector<NeoRangeNode>;
using VertexMap_t = std::array<NeoVertex, VERTEX_GROUP_SIZE>;
struct RangeElementSegment_t {
    std::array<RangeElement, RANGE_LEAF_SIZE> value;
    std::atomic<uint32_t> ref_cnt{1};
};

struct RequiredLock {
    uint64_t idx;
    bool is_exclusive;
};

} // namespace container
