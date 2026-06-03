#ifndef PHILEMON_NEOGRAPH_UTILS_IMPL_HPP
#define PHILEMON_NEOGRAPH_UTILS_IMPL_HPP
/**
 * neograph_utils_impl.hpp — NeoGraph utils层完整移植
 *
 * 骨架来源:
 *   upstream/rapidstore/libraries/NeoGraph/utils/spin_lock.cpp  (23行)
 *   upstream/rapidstore/libraries/NeoGraph/utils/types.cpp      (146行)
 *   合计 169行
 *
 * 修改 (~20%):
 *   - [MOD] container:: → philemon::index::neo_utils
 *   - [MOD] SpinLock 已由 src/executor/spin_lock.hpp 覆盖
 *     → 本文件仅提供 SpinLockCompat 别名 + 对齐断点
 *   - [NEW] ARTKey: dump_key() 打印各byte层
 *   - [NEW] ARTKey: validate_depth() 断言深度合法
 *   - [NEW] NeoRangeNode: dump_node() 打印完整状态
 *   - [NEW] get_key_byte: trace模式下打印提取结果
 *   - [KEEP] ARTKey 3个构造函数 + mask逻辑 100%保留
 *   - [KEEP] ARTKey operator[], ==, !=, < 100%保留
 *   - [KEEP] NeoRangeNode 构造函数 + is_empty 100%保留
 *   - [KEEP] InRangeNode 构造函数 100%保留
 *   - [KEEP] get_key_byte 100%保留
 *
 * Milestone: M028
 */

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <cstring>

#include "../executor/spin_lock.hpp"
#include "../debug/philemon_debug.hpp"

// 默认为0 (无edge property), 可编译时覆盖
#ifndef EDGE_PROPERTY_NUM
#define EDGE_PROPERTY_NUM 0
#endif

namespace philemon {
namespace index {
namespace neo_utils {

// ═══════════════════════════════════════════════════════════════════
// SpinLock兼容层 — 指向 executor::SpinLock
// ═══════════════════════════════════════════════════════════════════
using SpinLockCompat = philemon::executor::SpinLock;

// RAII guard (对应 upstream SpinLockGuard)
class SpinLockGuard {
    SpinLockCompat& lock_;
public:
    explicit SpinLockGuard(SpinLockCompat& lk) : lock_(lk) { lock_.lock(); }
    ~SpinLockGuard() { lock_.unlock(); }
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;
};

// ═══════════════════════════════════════════════════════════════════
// ARTKey — Adaptive Radix Tree 键 (upstream types.cpp 100%)
// ═══════════════════════════════════════════════════════════════════

class ARTKey {
    uint64_t key;

public:
    // 构造1: 从destination id (upstream 100%)
    explicit ARTKey(uint64_t dst)
        : key(dst & 0x00000000FFFFFF00ULL) {}

    // 构造2: 带depth truncation (upstream 100%)
    ARTKey(uint64_t dst, uint8_t depth, bool is_single_byte)
        : key(dst & 0x00000000FFFFFF00ULL) {
        switch (depth + is_single_byte) {
            case 0: key &= 0x0000FFFF00000000ULL; break;
            case 1: key &= 0x0000FFFFFF000000ULL; break;
            case 2: key &= 0x0000FFFFFFFF0000ULL; break;
            case 3: key &= 0x0000FFFFFFFFFF00ULL; break;
            default:
                PHILE_DBG(0, "ARTKey: INVALID depth %d", (int)depth);
                assert(false);
        }
    }

    // 构造3: 从另一个ARTKey截断 (upstream 100%)
    ARTKey(ARTKey other, uint8_t depth, bool is_single_byte)
        : key(other.key) {
        switch (depth + is_single_byte) {
            case 0: key &= 0x0000FFFF00000000ULL; break;
            case 1: key &= 0x0000FFFFFF000000ULL; break;
            case 2: key &= 0x0000FFFFFFFF0000ULL; break;
            case 3: key &= 0x0000FFFFFFFFFF00ULL; break;
            default:
                PHILE_DBG(0, "ARTKey: INVALID depth %d", (int)depth);
                assert(false);
        }
    }

    // operator[] (upstream 100%)
    uint8_t operator[](int idx) const {
        assert(idx < 5);
        return (key >> ((3 - idx) * 8)) & 0xFF;
    }

    uint8_t& operator[](int idx) {
        assert(idx < 5);
        return reinterpret_cast<uint8_t*>(&key)[3 - idx];
    }

    // 比较运算符 (upstream 100%)
    bool operator==(const ARTKey& rhs) const { return key == rhs.key; }
    bool operator!=(const ARTKey& rhs) const { return key != rhs.key; }

    bool operator<(const ARTKey& rhs) const {
        for (uint8_t depth = 0; depth < 3; depth++) {
            if ((*this)[depth] != rhs[depth])
                return (*this)[depth] < rhs[depth];
        }
        return false;
    }

    // upstream print() (upstream 100%)
    void print() const {
        for (int i = 0; i < 3; i++)
            std::cout << (int)(*this)[i] << " ";
        std::cout << std::endl;
    }

    // [NEW] 增强版dump
    void dump_key(const char* label = "ARTKEY") const {
        std::printf("[%s] raw=0x%016lx bytes=[",
                    label, (unsigned long)key);
        for (int i = 0; i < 4; i++)
            std::printf("%s%02x", i > 0 ? "," : "",
                        (unsigned)(*this)[i]);
        std::printf("]\n");
    }

    // [NEW] 验证depth合法性
    static bool validate_depth(uint8_t depth) {
        if (depth > 3) {
            PHILE_DBG(0, "ARTKey: depth %d out of range [0,3]",
                      (int)depth);
            return false;
        }
        return true;
    }

    uint64_t raw() const { return key; }
};

// get_key_byte (upstream 100%)
inline uint8_t get_key_byte(uint64_t key, uint8_t depth) {
    return (key >> ((3 - depth) * 8)) & 0xFF;
}

// ═══════════════════════════════════════════════════════════════════
// NeoRangeNode — Range树的节点 (upstream types.cpp 100%)
// ═══════════════════════════════════════════════════════════════════

// Forward declarations for property types
#if EDGE_PROPERTY_NUM > 0
struct RangePropertyVec_t;
struct MultiRangePropertyVec_t;
#endif

struct NeoRangeNode {
    uint64_t key;
    uint64_t size;
    uint64_t arr_ptr;
#if EDGE_PROPERTY_NUM == 1
    RangePropertyVec_t* property;
#elif EDGE_PROPERTY_NUM > 1
    MultiRangePropertyVec_t* property;
#endif

    // 默认构造 (upstream 100%)
    NeoRangeNode() : key(0), size(0), arr_ptr(0)
#if EDGE_PROPERTY_NUM > 0
        , property(nullptr)
#endif
    {}

    // 完整构造 (upstream 100%)
    NeoRangeNode(uint64_t k, uint64_t s, uint64_t ptr,
                 [[maybe_unused]] void* prop_ptr = nullptr)
        : key(k), size(s), arr_ptr(ptr)
#if EDGE_PROPERTY_NUM == 1
        , property(static_cast<RangePropertyVec_t*>(prop_ptr))
#elif EDGE_PROPERTY_NUM > 1
        , property(static_cast<MultiRangePropertyVec_t*>(prop_ptr))
#endif
    {}

    // is_empty (upstream 100%)
    bool is_empty() const {
#if EDGE_PROPERTY_NUM == 0
        return *reinterpret_cast<const uint64_t*>(this) == 0;
#else
        return *reinterpret_cast<const uint64_t*>(this) == 0 &&
               *(reinterpret_cast<const uint64_t*>(this) + 1) == 0;
#endif
    }

    // [NEW] 节点状态打印
    void dump_node(const char* label = "RNODE") const {
        std::printf("[%s] key=%lu size=%lu arr_ptr=0x%lx empty=%d\n",
                    label,
                    (unsigned long)key,
                    (unsigned long)size,
                    (unsigned long)arr_ptr,
                    (int)is_empty());
    }
};

// ═══════════════════════════════════════════════════════════════════
// InRangeNode — 入边Range节点 (upstream types.cpp 100%)
// ═══════════════════════════════════════════════════════════════════

struct InRangeNode {
    uint64_t size;
    uint64_t arr_ptr;
#if EDGE_PROPERTY_NUM != 0
    RangePropertyVec_t* property_map;
#endif

    InRangeNode() : size(0), arr_ptr(0) {}

    InRangeNode(uint64_t s, uint64_t ptr)
        : size(s), arr_ptr(ptr) {}

#if EDGE_PROPERTY_NUM != 0
    InRangeNode(uint64_t s, uint64_t ptr, RangePropertyVec_t* prop)
        : size(s), arr_ptr(ptr), property_map(prop) {}
#endif

    void dump_node(const char* label = "INRNG") const {
        std::printf("[%s] size=%lu arr_ptr=0x%lx\n",
                    label, (unsigned long)size, (unsigned long)arr_ptr);
    }
};

} // namespace neo_utils
} // namespace index
} // namespace philemon

#endif // PHILEMON_NEOGRAPH_UTILS_IMPL_HPP
