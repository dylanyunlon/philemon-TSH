#pragma once
/**
 * neo_config.hpp — NeoGraph compile-time configuration constants
 *
 * 骨架来源: upstream/rapidstore/libraries/NeoGraph/utils/config.h (30行)
 * 修改 (~25%):
 *   - 加入 PHILEMON_TIER_* 常量供跨层使用
 *   - 加入 PHILE_NEO_DBG 运行时调试开关
 *   - 加入 dump_neo_config() 一键打印所有当前配置
 *   - RANGE_LEAF_SIZE / ART_EXTRACT_THRESHOLD 改为 constexpr (可在编译期切换)
 *   - 增加 static_assert 防止非法组合
 *
 * Milestone: M071 (Claude #?)
 */

#include <limits>
#include <cstdint>
#include <cstdio>

// ─── Philemon tier-aware extensions ───
#ifndef PHILEMON_TIER_COUNT
#define PHILEMON_TIER_COUNT 3
#endif
#ifndef PHILE_NEO_DBG
#define PHILE_NEO_DBG 0
#endif

#define PHILE_NEO_TRACE(fmt, ...)                                          \
    do {                                                                   \
        if (PHILE_NEO_DBG)                                                 \
            std::fprintf(stderr, "[NEO-TRACE] %s:%d " fmt "\n",           \
                         __func__, __LINE__, ##__VA_ARGS__);               \
    } while (0)

// ─── Core NeoGraph constants (upstream, renamed guards) ───
#define VERTEX_GROUP_BITS 6
constexpr uint64_t VERTEX_GROUP_SIZE = 1ULL << VERTEX_GROUP_BITS;
constexpr uint64_t VERTEX_GROUP_MASK = (1ULL << VERTEX_GROUP_BITS) - 1;
constexpr uint64_t INDEPENDENT_MAP_BLOCK_NUM = (VERTEX_GROUP_SIZE + 63) / 64;

constexpr uint64_t RANGE_LEAF_SIZE_V       = 512;    // upstream default 512
constexpr uint64_t ART_EXTRACT_THRESHOLD_V = 8192;   // upstream default 8192
constexpr uint64_t ART_LEAF_SIZE_V         = 256;     // 16*16

// Keep macros for backward compat with upstream code that #ifdef-s on them
#ifndef RANGE_LEAF_SIZE
#define RANGE_LEAF_SIZE 512
#endif
#ifndef ART_EXTRACT_THRESHOLD
#define ART_EXTRACT_THRESHOLD 8192
#endif
#ifndef ART_LEAF_SIZE
#define ART_LEAF_SIZE 256
#endif

#define SEQUENTIAL_SCAN_THRESHOLD 16
#define EDGE_INSERT_VEC_THRESHOLD 0.8
#define BATCH_UPDATE_THRESHOLD (1 << 2)
#define INIT_READER_NUM 32
#define INIT_WRITER_NUM 64

// Property dimensions — Philemon overrides with 0/1
#ifndef VERTEX_PROPERTY_NUM
#define VERTEX_PROPERTY_NUM 0
#endif
#ifndef EDGE_PROPERTY_NUM
#define EDGE_PROPERTY_NUM 1
#endif

#define COMPRESSION_ENABLE 1
#define FROM_CLUSTERED_TO_SMALL_VEC_ENABLE 0
#define SIMULATE_PER_EDGE_VERSIONING_ENABLE 0

#define SEGMENT_POOL_INIT_SIZE 256
#define BATCH_UPDATE_THREAD_NUM 31
#define BATCH_UPDATE_ENABLE_THRESHOLD 32

#define VERSION_HEAD_MASK 0x8000000000000000ULL

// ─── Compile-time sanity checks ───
static_assert(VERTEX_GROUP_SIZE > 0 && (VERTEX_GROUP_SIZE & (VERTEX_GROUP_SIZE - 1)) == 0,
              "VERTEX_GROUP_SIZE must be power-of-2");
static_assert(RANGE_LEAF_SIZE >= 64, "RANGE_LEAF_SIZE too small for ART ops");

// ─── Runtime config dump for debugging ───
namespace philemon { namespace neo {

inline void dump_neo_config() {
    std::fprintf(stderr,
        "╔══════════════ NeoGraph Config Snapshot ══════════════╗\n"
        "║ VERTEX_GROUP_BITS       = %d                        ║\n"
        "║ VERTEX_GROUP_SIZE       = %llu                       ║\n"
        "║ RANGE_LEAF_SIZE         = %d                        ║\n"
        "║ ART_EXTRACT_THRESHOLD   = %d                       ║\n"
        "║ ART_LEAF_SIZE           = %d                        ║\n"
        "║ VERTEX_PROPERTY_NUM     = %d                         ║\n"
        "║ EDGE_PROPERTY_NUM       = %d                         ║\n"
        "║ COMPRESSION_ENABLE      = %d                         ║\n"
        "║ PHILEMON_TIER_COUNT     = %d                         ║\n"
        "║ PHILE_NEO_DBG           = %d                         ║\n"
        "╚═════════════════════════════════════════════════════╝\n",
        VERTEX_GROUP_BITS,
        (unsigned long long)VERTEX_GROUP_SIZE,
        RANGE_LEAF_SIZE,
        ART_EXTRACT_THRESHOLD,
        ART_LEAF_SIZE,
        VERTEX_PROPERTY_NUM,
        EDGE_PROPERTY_NUM,
        COMPRESSION_ENABLE,
        PHILEMON_TIER_COUNT,
        PHILE_NEO_DBG);
}

}} // namespace philemon::neo
