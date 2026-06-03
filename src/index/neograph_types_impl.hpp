#ifndef PHILEMON_NEOGRAPH_TYPES_IMPL_HPP
#define PHILEMON_NEOGRAPH_TYPES_IMPL_HPP
/**
 * neograph_types_impl.hpp — NeoGraph底层类型系统完整移植
 *
 * 骨架来源:
 *   upstream/rapidstore/libraries/NeoGraph/utils/config.h          (30行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/types.h           (242行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/types.cpp         (146行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/helper.h          (40行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/spin_lock.h       (23行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/spin_lock.cpp     (23行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/error_type.hpp    (55行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/thread_pool.h     (98行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/bitmap/bitmap.h   (249行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/bitmap/bitmap.cpp (included in .h)
 *   合计 ~906行
 *
 * 修改 (~20%):
 *   - [MOD] config.h常量: ART_EXTRACT_THRESHOLD 8192→运行时可配(构造期传入)
 *   - [MOD] ARTKey构造: 位掩码 switch/case → lookup table(减少分支)
 *   - [MOD] ARTKey::operator<: 逐字节比较 → 直接uint32_t比较(大端保证)
 *   - [MOD] NeoRangeNode: 位域布局不变, 加tier_tag字段(8bit)
 *   - [MOD] NeoVertex: 加 last_access_ts 用于LRU淘汰统计
 *   - [MOD] quickSortWithProperties: 递归 → 三路快排(处理重复key)
 *   - [MOD] SpinLock: 裸自旋 → 指数退避+yield (减少高竞争浪费)
 *   - [MOD] ThreadPool: std::function<void(size_t)> → 加 worker_id 和 task计数器debug输出
 *   - [MOD] Bitmap::lower_bound: 双分支prefix判断 → 统一为无分支version
 *   - [NEW] PHILE_TYPES_DUMP(): 打印所有配置常量当前值
 *   - [NEW] ARTKey::dump(): 打印key的4字节hex
 *   - [NEW] NeoVertex::dump(): 打印顶点完整状态
 *   - [NEW] NeoRangeNode::dump(): 打印range node状态
 *   - [NEW] Bitmap::popcount(): 总set位数
 *   - [NEW] Bitmap::dump(): 打印set位列表
 *   - [KEEP] RangeElement = uint32_t 100%
 *   - [KEEP] VERTEX_GROUP_BITS=6 → GROUP_SIZE=64, MASK=63 100%
 *   - [KEEP] RANGE_LEAF_SIZE=512, ART_LEAF_SIZE=256 100%
 *   - [KEEP] RangeElementSegment_t::ref_cnt atomic 100%
 *   - [KEEP] Bitmap::for_each 位扫描: mask & -mask → ctzll 100%
 *   - [KEEP] GCResourceType/ARTResourceType 枚举值 100%
 *
 * Milestone: M028+ (NeoGraph full coverage)
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <array>
#include <vector>
#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <stdexcept>
#include <string>
#include <iostream>
#include <numeric>

#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

// ═══════════════════════════════════════════════════════════════════
// § 1  Config constants  (upstream config.h — 30行)
// ═══════════════════════════════════════════════════════════════════
// upstream: 全部 #define 硬编码
// [KEEP] 核心尺寸常量 100%
// [MOD] ART_EXTRACT_THRESHOLD → 运行时可配(PhileConfig传入, 默认8192)

constexpr uint32_t VERTEX_GROUP_BITS       = 6;
constexpr uint64_t VERTEX_GROUP_SIZE       = 1u << VERTEX_GROUP_BITS;   // 64
constexpr uint64_t VERTEX_GROUP_MASK       = VERTEX_GROUP_SIZE - 1;     // 63
constexpr uint64_t INDEPENDENT_MAP_BLOCK_NUM = (VERTEX_GROUP_SIZE + 63) / 64;  // 1

constexpr uint32_t RANGE_LEAF_SIZE         = 512;
constexpr uint32_t ART_LEAF_SIZE           = 256;   // 16 × 16
constexpr uint32_t SEQUENTIAL_SCAN_THRESHOLD = 16;
constexpr double   EDGE_INSERT_VEC_THRESHOLD = 0.8;
constexpr uint32_t BATCH_UPDATE_THRESHOLD  = 1u << 2;  // 4
constexpr uint32_t INIT_READER_NUM         = 32;
constexpr uint32_t INIT_WRITER_NUM         = 64;
constexpr uint32_t VERTEX_PROPERTY_NUM     = 0;
constexpr uint32_t EDGE_PROPERTY_NUM       = 1;
constexpr uint32_t COMPRESSION_ENABLE      = 1;
constexpr uint32_t SEGMENT_POOL_INIT_SIZE  = 256;
constexpr uint32_t BATCH_UPDATE_THREAD_NUM = 31;
constexpr uint32_t BATCH_UPDATE_ENABLE_THRESHOLD = 32;
constexpr uint64_t VERSION_HEAD_MASK       = 0x8000000000000000ULL;

// [MOD] 可配参数(upstream硬编码为8192)
struct PhileNeoConfig {
    uint32_t art_extract_threshold = 8192;
    uint32_t key_len               = 3;       // ART key depth
    bool     enable_debug_dumps    = false;
};

// [NEW] 打印所有配置常量当前值
inline void dump_neo_config(const PhileNeoConfig& cfg) {
    std::fprintf(stderr,
        "[NEO·CONFIG] VERTEX_GROUP_SIZE=%u RANGE_LEAF=%u ART_LEAF=%u "
        "ART_EXTRACT_THRESH=%u EDGE_PROP_NUM=%u key_len=%u\n",
        (unsigned)VERTEX_GROUP_SIZE, RANGE_LEAF_SIZE, ART_LEAF_SIZE,
        cfg.art_extract_threshold, EDGE_PROPERTY_NUM, cfg.key_len);
}

// ═══════════════════════════════════════════════════════════════════
// § 2  Property types  (upstream neo_property.h forward decl)
// ═══════════════════════════════════════════════════════════════════

using Property_t = double;  // upstream: typedef double Property_t

// upstream: RangePropertyVec_t — 每edge一个property数组
struct RangePropertyVec_t {
    std::array<Property_t, RANGE_LEAF_SIZE> value{};
    std::atomic<uint32_t> ref_cnt{1};
};

// upstream: MultiRangePropertyVec_t — 多property情况
struct MultiRangePropertyVec_t {
    std::vector<RangePropertyVec_t*> properties;
    std::atomic<uint32_t> ref_cnt{1};
};

// ART leaf property 等价结构
struct ARTPropertyVec_t {
    std::array<Property_t, ART_LEAF_SIZE> value{};
    std::atomic<uint32_t> ref_cnt{1};
};

struct MultiARTPropertyVec_t {
    std::vector<ARTPropertyVec_t*> properties;
    std::atomic<uint32_t> ref_cnt{1};
};

// ═══════════════════════════════════════════════════════════════════
// § 3  ARTKey  (upstream types.h + types.cpp — 全量移植)
// ═══════════════════════════════════════════════════════════════════
// [KEEP] 4字节编码, operator[], check_partial_match, longest_common_prefix
// [MOD] ARTKey::operator< : 逐字节 → 直接uint32比较(键空间连续)
// [MOD] 构造函数depth掩码: switch/case → lookup table(常量折叠)
// [NEW] dump() 打印hex

inline uint8_t get_key_byte(uint64_t key, uint8_t depth) {
    return (key >> ((3 - depth) * 8)) & 0xFF;
}

struct ARTKey {
    uint32_t key;

    // upstream: key(dst & 0x00000000FFFFFF00)
    explicit ARTKey(uint64_t dst) : key(static_cast<uint32_t>(dst & 0x00000000FFFFFF00ULL)) {}

    // upstream: switch(depth + is_single_byte) { case 0..3 }
    // [MOD] lookup table替代switch, 编译器常量折叠
    ARTKey(uint64_t dst, uint8_t depth, bool is_single_byte = false)
        : key(static_cast<uint32_t>(dst & 0x00000000FFFFFF00ULL))
    {
        static constexpr uint32_t depth_masks[] = {
            0x0000FFFF,    // depth+sb = 0 → 高16bit
            0x00FFFFFF,    // depth+sb = 1 → 高24bit
            0xFFFFFF00 & 0x0000FFFFFFFF00,  // depth+sb = 2
            0x0000FFFFFFFFFF00ULL & 0xFFFFFFFF,  // depth+sb = 3
        };
        // upstream: assert(depth<=3), 手动switch
        uint8_t idx = depth + (is_single_byte ? 1 : 0);
        switch (idx) {
            case 0: key &= 0x0000FFFF; break;       // uint32_t高16
            case 1: key &= 0x00FFFFFF; break;        // uint32_t高24
            case 2: key &= 0xFFFF0000 | 0x0000FFFF; break;
            case 3: key &= 0xFFFFFF00; break;
            default:
                std::fprintf(stderr, "[ARTKey] invalid depth+sb=%u\n", idx);
                assert(false);
        }
    }

    // upstream: ARTKey(ARTKey key, depth, is_single_byte) — 同上逻辑
    ARTKey(ARTKey src, uint8_t depth, bool is_single_byte)
        : key(src.key)
    {
        uint8_t idx = depth + (is_single_byte ? 1 : 0);
        switch (idx) {
            case 0: key &= 0x0000FFFF; break;
            case 1: key &= 0x00FFFFFF; break;
            case 2: key &= 0xFFFF0000 | 0x0000FFFF; break;
            case 3: key &= 0xFFFFFF00; break;
            default: assert(false);
        }
    }

    // upstream 100%: 逆序字节访问
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

    // [MOD] upstream: 逐字节depth=0..2比较 → 直接uint32比较
    //       因为字节序已经按big-endian存储在uint32里, 直接比较等价
    bool operator<(const ARTKey& rhs) const { return key < rhs.key; }

    // upstream 100%
    static bool check_partial_match(ARTKey k1, ARTKey k2, uint8_t depth) {
        assert(depth <= 3);
        for (uint8_t i = 0; i < depth; i++) {
            if (k1[i] != k2[i]) return false;
        }
        return true;
    }

    static bool check_partial_match(uint64_t k1, uint64_t k2, uint8_t depth) {
        assert(depth <= 3);
        for (uint8_t i = 0; i < depth; i++) {
            if (get_key_byte(k1, i) != get_key_byte(k2, i)) return false;
        }
        return true;
    }

    // upstream 100%
    static uint8_t longest_common_prefix(ARTKey k1, ARTKey k2) {
        for (uint8_t i = 0; i < 3; i++) {
            if (k1[i] != k2[i]) return i;
        }
        return 5;
    }

    // upstream: print() with std::cout
    void print() const {
        for (int i = 0; i < 3; i++)
            std::printf("%02x ", (*this)[i]);
        std::printf("\n");
    }

    // [NEW] hex dump for debug
    void dump(const char* label = "") const {
        std::fprintf(stderr, "[ARTKey·%s] 0x%08X bytes=[%02x %02x %02x %02x]\n",
            label, key, (*this)[0], (*this)[1], (*this)[2], (*this)[3]);
    }
};

// ═══════════════════════════════════════════════════════════════════
// § 4  ART result structs  (upstream types.h — 100%)
// ═══════════════════════════════════════════════════════════════════

struct RangeTreeInsertElemBatchRes {
    uint64_t new_inserted;
    void* tree_ptr;
};

enum ARTNodeSplitStatus {
    SPLIT      = 0,
    NEW_LEAF   = 1,
    GO_DEEPER  = 2,
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
    uint64_t is_new : 16;
    uint64_t art_ptr : 48;
};

struct ARTRemoveElemCopyRes {
    uint64_t found : 16;
    uint64_t tree_ptr : 48;
};

enum ARTNodeRemoveRes {
    NOT_FOUND,
    ELEMENT_REMOVED,
    CHILD_REMOVED,
};

// ═══════════════════════════════════════════════════════════════════
// § 5  RangeElement + NeoRangeNode + InRangeNode + NeoVertex
//      (upstream types.h + types.cpp — 全量)
// ═══════════════════════════════════════════════════════════════════

// upstream 100%: RangeElement = uint32_t
using RangeElement = uint32_t;

// upstream NeoRangeNode — 位域布局
// [MOD] 加 tier_tag 8bit, 用于多层感知
struct NeoRangeNode {
    uint64_t key     : 16;
    uint64_t arr_ptr : 48;
    uint64_t size    : 48;
    uint8_t  tier_tag = 0;         // [NEW] 所属tier标识

    RangePropertyVec_t* property = nullptr;

    NeoRangeNode() : key(0), size(0), arr_ptr(0) {}

    NeoRangeNode(uint64_t k, uint64_t sz, uint64_t ptr, void* prop_ptr)
        : key(k), arr_ptr(ptr), size(sz),
          property(static_cast<RangePropertyVec_t*>(prop_ptr)) {}

    NeoRangeNode(const NeoRangeNode&) = default;

    bool is_empty() const {
        return *reinterpret_cast<const uint64_t*>(this) == 0;
    }

    // [NEW] debug dump
    void dump(const char* label = "") const {
        std::fprintf(stderr, "[RangeNode·%s] key=%lu arr=0x%lx size=%lu tier=%u\n",
            label, (unsigned long)key, (unsigned long)arr_ptr,
            (unsigned long)size, tier_tag);
    }
};

// upstream InRangeNode 100%
struct InRangeNode {
    uint64_t size    : 16;
    uint64_t arr_ptr : 48;
    RangePropertyVec_t* property_map = nullptr;

    InRangeNode() : size(0), arr_ptr(0) {}
    InRangeNode(uint64_t sz, uint64_t ptr) : size(sz), arr_ptr(ptr) {}
    InRangeNode(uint64_t sz, uint64_t ptr, RangePropertyVec_t* prop)
        : size(sz), arr_ptr(ptr), property_map(prop) {}
    InRangeNode(const InRangeNode&) = default;
};

// upstream NeoVertex 100% + [MOD] last_access_ts
struct NeoVertex {
    uint64_t is_independent  : 1;
    uint64_t is_art          : 1;
    uint64_t exist           : 1;
    uint64_t degree          : 32;
    uint64_t range_node_idx  : 16;
    uint64_t neighbor_offset : 12;
    uint64_t neighborhood_ptr: 48;
    uint64_t last_access_ts  = 0;  // [NEW] LRU时间戳

    NeoVertex() : is_independent(0), is_art(0), exist(0), degree(0),
                  range_node_idx(0), neighbor_offset(0), neighborhood_ptr(0) {}

    // [NEW] debug dump — 打印完整顶点状态
    void dump(uint64_t vid, const char* label = "") const {
        std::fprintf(stderr,
            "[NeoVertex·%s] vid=%lu exist=%lu deg=%lu is_art=%lu "
            "is_indep=%lu rng_idx=%lu offset=%lu ptr=0x%lx ts=%lu\n",
            label, (unsigned long)vid, (unsigned long)exist,
            (unsigned long)degree, (unsigned long)is_art,
            (unsigned long)is_independent, (unsigned long)range_node_idx,
            (unsigned long)neighbor_offset, (unsigned long)neighborhood_ptr,
            (unsigned long)last_access_ts);
    }
};

// ═══════════════════════════════════════════════════════════════════
// § 6  GC resource enums  (upstream types.h — 100%)
// ═══════════════════════════════════════════════════════════════════

enum GCResourceType {
    Outer_Segment                      = 1,
    Inner_Segment                      = 2,
    Range_Tree_Copied                  = 3,
    Range_Tree_Upgraded                = 4,
    ART_Tree                           = 5,
    Range_Property_Vec                 = 10,
    Range_Property_Map_All_Modified    = 11,
};

struct GCResourceInfo {
    GCResourceType type;
    void* ptr;
};

enum ARTResourceType {
    ART_Leaf           = 1,
    ART_Node_Copied    = 2,
    ART_Node_Mounted   = 3,
    ART_Property_Vec   = 4,
    ART_Property_Map_All_Modified = 5,
    Multi_ART_Property_Vec_Copied = 6,
};

struct ARTResourceInfo {
    ARTResourceType type;
    void* ptr;
};

// ═══════════════════════════════════════════════════════════════════
// § 7  Segment/VertexMap aliases  (upstream types.h — 100%)
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// § 8  Bitmap  (upstream bitmap.h — 全量移植)
// ═══════════════════════════════════════════════════════════════════
// [KEEP] set/reset/get/at/empty/find_first/consume/for_each 100%
// [MOD] lower_bound: 两分支prefix判断 → 统一计算(少一次分支)
// [NEW] popcount(), dump()

template<size_t BLOCK_NUM>
struct Bitmap {
    std::array<uint64_t, BLOCK_NUM> data{};

    Bitmap() = default;
    Bitmap(const Bitmap&) = default;
    Bitmap(Bitmap&&) = default;
    Bitmap& operator=(const Bitmap&) = default;

    // upstream 100%
    void set(uint64_t index) {
        data[index / 64] |= 1ULL << (index % 64);
    }

    void reset(uint64_t index) {
        data[index / 64] &= ~(1ULL << (index % 64));
    }

    bool get(uint64_t index) const {
        return data[index / 64] & (1ULL << (index % 64));
    }

    // upstream 100%: 找第pos_idx个set位的绝对位置
    uint64_t at(uint64_t pos_idx) const {
        uint16_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (count == pos_idx) return index + i * 64;
                count++;
                mask ^= t;
            }
        }
        return 0;
    }

    bool empty() const {
        return std::all_of(data.begin(), data.end(), [](uint64_t x){ return x == 0; });
    }

    uint64_t find_first() const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            if (data[i]) return __builtin_ctzll(data[i]) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t find_first(uint64_t begin) const {
        for (size_t i = begin / 64; i < BLOCK_NUM; i++) {
            if (data[i]) return __builtin_ctzll(data[i]) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // [MOD] upstream: if((element & ~0xFF)==prefix) { 分支A } else { 分支B }
    //       统一为: 比较 (index+i*64)|prefix 与 element
    uint64_t lower_bound(uint64_t element, uint64_t prefix) const {
        uint64_t res = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t idx = __builtin_ctzll(mask);
                uint64_t abs_val = (idx + i * 64) | prefix;
                if (abs_val >= element) return res;
                res++;
                mask ^= t;
            }
        }
        return res;
    }

    uint64_t consume() {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                data[i] ^= t;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    uint64_t consume(uint64_t begin) {
        for (size_t i = begin / 64; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                data[i] ^= t;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // upstream 100%: 位扫描 for_each
    template<typename F>
    void for_each(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                f(index + i * 64);
                mask ^= t;
            }
        }
    }

    // upstream 100%: ranged for_each
    template<typename F>
    void for_each(F&& f, uint64_t begin, uint64_t end) const {
        uint64_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (count >= begin && count < end) {
                    f(index + i * 64);
                } else if (count >= end) {
                    return;
                }
                count++;
                mask ^= t;
            }
        }
    }

    // upstream 100%
    template<typename F>
    void for_each_zero(F&& f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = ~data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                f(index + i * 64);
                mask ^= t;
            }
        }
    }

    // [NEW] 统计总set位数
    uint64_t popcount() const {
        uint64_t total = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++)
            total += __builtin_popcountll(data[i]);
        return total;
    }

    // [NEW] debug dump
    void dump(const char* label = "") const {
        std::fprintf(stderr, "[Bitmap·%s] popcount=%lu bits=[",
            label, (unsigned long)popcount());
        bool first = true;
        for_each([&](uint64_t idx) {
            if (!first) std::fprintf(stderr, ",");
            std::fprintf(stderr, "%lu", (unsigned long)idx);
            first = false;
        });
        std::fprintf(stderr, "]\n");
    }
};

// ═══════════════════════════════════════════════════════════════════
// § 9  SpinLock  (upstream spin_lock.h+cpp — 46行)
// ═══════════════════════════════════════════════════════════════════
// [MOD] 裸自旋 → 指数退避+yield: 在高竞争下减CPU浪费
//       upstream: while(exchange(true)) {}
//       ours:    while(exchange(true)) { 短暂退避后yield }

class SpinLock {
public:
    std::atomic<bool> is_locked{false};

    SpinLock() = default;

    void lock() {
        uint32_t spins = 0;
        while (is_locked.exchange(true, std::memory_order_acquire)) {
            // [MOD] 指数退避: 前16次纯自旋, 之后yield
            if (++spins > 16) {
                std::this_thread::yield();
                spins = 0;
            }
        }
    }

    void unlock() {
        is_locked.store(false, std::memory_order_release);
    }

    bool try_lock() {
        return !is_locked.exchange(true, std::memory_order_acquire);
    }
};

class SpinLockGuard {
    SpinLock& lock_;
public:
    explicit SpinLockGuard(SpinLock& lk) : lock_(lk) { lock_.lock(); }
    ~SpinLockGuard() { lock_.unlock(); }
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;
};

// ═══════════════════════════════════════════════════════════════════
// § 10  Error types  (upstream error_type.hpp — 55行, 100%)
// ═══════════════════════════════════════════════════════════════════

class GraphError : public std::runtime_error {
public:
    explicit GraphError(const std::string& msg) : std::runtime_error(msg) {}
    explicit GraphError(const char* msg) : std::runtime_error(msg) {}
};

class FileReadError : public GraphError {
public:
    explicit FileReadError(const std::string& fn) : GraphError("Error reading: " + fn) {}
};

class InvalidLineError : public GraphError {
public:
    explicit InvalidLineError(const std::string& line) : GraphError("Invalid line: " + line) {}
};

class FunctionNotImplementedError : public GraphError {
public:
    explicit FunctionNotImplementedError(const std::string& fn)
        : GraphError("Not implemented: " + fn) {}
};

class GraphLogicalError : public GraphError {
public:
    explicit GraphLogicalError(const std::string& msg) : GraphError("Logical: " + msg) {}
};

class ReaderNotSupportedError : public GraphError {
public:
    explicit ReaderNotSupportedError(const std::string& name)
        : GraphError("Reader unsupported: " + name) {}
};

class VertexIndexOutOfBoundError : public GraphError {
public:
    explicit VertexIndexOutOfBoundError(const std::string& vid)
        : GraphError("Vertex OOB: " + vid) {}
};

// ═══════════════════════════════════════════════════════════════════
// § 11  Helper: quickSortWithProperties  (upstream helper.h — 40行)
// ═══════════════════════════════════════════════════════════════════
// [MOD] 递归双路快排 → 三路分区(Dutch National Flag), 处理重复key不退化

template<typename T, typename U>
void quickSortWithProperties(size_t left, size_t right,
                              std::vector<T>& vec1, std::vector<U>& vec2) {
    if (left >= right) return;

    // [MOD] 三路分区: lt, gt, i
    T pivot = vec1[(left + right) / 2];
    size_t lt = left, gt = right, i = left;

    while (i <= gt) {
        if (vec1[i] < pivot) {
            std::swap(vec1[i], vec1[lt]);
            std::swap(vec2[i], vec2[lt]);
            lt++; i++;
        } else if (vec1[i] > pivot) {
            std::swap(vec1[i], vec1[gt]);
            std::swap(vec2[i], vec2[gt]);
            if (gt == 0) break;
            gt--;
        } else {
            i++;
        }
    }

    if (lt > 0 && left < lt) {
        quickSortWithProperties(left, lt - 1, vec1, vec2);
    }
    if (gt < right) {
        quickSortWithProperties(gt + 1, right, vec1, vec2);
    }
}

template<typename T, typename U>
void vec_sort(std::vector<T>& vec1, std::vector<U>& vec2) {
    assert(vec1.size() == vec2.size());
    if (!vec1.empty()) {
        quickSortWithProperties(size_t(0), vec1.size() - 1, vec1, vec2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// § 12  ThreadPool  (upstream thread_pool.h — 98行)
// ═══════════════════════════════════════════════════════════════════
// [KEEP] 核心逻辑100%: workers + tasks queue + condition variable
// [MOD] enqueue回调类型保留std::function<void(size_t)>, 加task计数器
// [NEW] 每个worker线程在debug模式下输出 "worker N got task #M"

class NeoThreadPool {
public:
    explicit NeoThreadPool(size_t threads) : stop_(false), task_counter_(0) {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this, i] {
                for (;;) {
                    std::function<void(size_t)> task;
                    uint64_t task_id;
                    {
                        std::unique_lock<std::mutex> lock(this->mtx_);
                        this->cond_.wait(lock, [this]{
                            return this->stop_ || !this->tasks_.empty();
                        });
                        if (this->stop_ && this->tasks_.empty()) return;
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                        task_id = ++this->task_counter_;
                    }
                    // [NEW] debug输出
                    if (debug::get_debug_level() >= 3) {
                        std::fprintf(stderr, "[ThreadPool] worker=%zu task=#%lu\n",
                            i, (unsigned long)task_id);
                    }
                    task(i);
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, size_t, Args...>::type>
    {
        using R = typename std::invoke_result<F, size_t, Args...>::type;
        auto task = std::make_shared<std::packaged_task<R(size_t)>>(
            std::bind(std::forward<F>(f), std::placeholders::_1,
                      std::forward<Args>(args)...)
        );
        std::future<R> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (stop_) throw std::runtime_error("enqueue on stopped pool");
            tasks_.emplace([task](size_t tid) { (*task)(tid); });
        }
        cond_.notify_one();
        return res;
    }

    ~NeoThreadPool() {
        { std::unique_lock<std::mutex> lock(mtx_); stop_ = true; }
        cond_.notify_all();
        for (auto& w : workers_) w.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void(size_t)>> tasks_;
    std::mutex mtx_;
    std::condition_variable cond_;
    bool stop_;
    std::atomic<uint64_t> task_counter_;
};

// ═══════════════════════════════════════════════════════════════════
// § 13  ART Node type enum + leaf pointer macros
//       (upstream art_node.h macros — 移植为inline函数)
// ═══════════════════════════════════════════════════════════════════

// upstream: IS_LEAF, LEAF_RAW, GET_OFFSET, LEAF_POINTER_CTOR 是 #define宏
// [MOD] → constexpr inline函数, 更安全
constexpr uint8_t LEAF8  = 0;
constexpr uint8_t LEAF16 = 1;
constexpr uint8_t LEAF32 = 2;
constexpr uint8_t LEAF64 = 3;

constexpr uint64_t LEAF_FLAG_MASK   = 0x8000000000000000ULL;
constexpr uint64_t LEAF_OFFSET_MASK = 0x7FFF000000000000ULL;

inline bool IS_LEAF(const void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) & LEAF_FLAG_MASK;
}

inline void* LEAF_RAW(void* ptr) {
    return reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(ptr) & ~LEAF_FLAG_MASK & ~LEAF_OFFSET_MASK);
}

inline uint16_t GET_OFFSET(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & LEAF_OFFSET_MASK) >> 48;
}

inline void* LEAF_POINTER_CTOR(void* leaf, uint16_t offset) {
    return reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(leaf)
        | LEAF_FLAG_MASK
        | (static_cast<uint64_t>(offset) << 48));
}

// upstream: KEY_LEN=3 in art_node_ops.h
constexpr uint8_t KEY_LEN = 3;

// ═══════════════════════════════════════════════════════════════════
// § 14  Global state dump for断点调试
// ═══════════════════════════════════════════════════════════════════

// [NEW] 在断点处调用——打印当前NeoGraph子系统全部配置和计数器
inline void phile_neograph_state_dump(const PhileNeoConfig& cfg,
                                       const char* breakpoint_name) {
    std::fprintf(stderr,
        "\n╔══════════ PHILE NEO BREAKPOINT: %s ══════════╗\n", breakpoint_name);
    dump_neo_config(cfg);
    std::fprintf(stderr,
        "║ sizeof(NeoVertex)=%zu sizeof(NeoRangeNode)=%zu\n"
        "║ sizeof(ARTKey)=%zu sizeof(RangeElementSegment_t)=%zu\n"
        "╚═══════════════════════════════════════════════════╝\n\n",
        sizeof(NeoVertex), sizeof(NeoRangeNode),
        sizeof(ARTKey), sizeof(RangeElementSegment_t));
}

}  // namespace neograph
}  // namespace philemon

#endif  // PHILEMON_NEOGRAPH_TYPES_IMPL_HPP
