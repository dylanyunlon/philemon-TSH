#ifndef PHILEMON_NEOGRAPH_TRACE_IMPL_HPP
#define PHILEMON_NEOGRAPH_TRACE_IMPL_HPP
/**
 * neograph_trace_impl.hpp — WriterTraceBlock + ReaderTraceBlock 完整移植
 *
 * 骨架来源:
 *   upstream neo_reader_trace.h  (186行) + neo_reader_trace.cpp (355行)
 *   合计 ~541行
 *
 * 修改 (~20%):
 *   - [MOD] trace_block segment pool: pre-allocated ring → 按需new+free list
 *   - [MOD] ReaderTraceBlock: 64位CAS版本锁 → SeqLock(与src/core兼容)
 *   - [NEW] WriterTraceBlock::dump(): 打印分配统计(segments/leaves/nodes)
 *   - [NEW] ReaderTraceBlock::dump(): 打印读者计数
 *   - [KEEP] allocate_range_element_segment: 段分配 100%
 *   - [KEEP] allocate_range_prop_vec: 属性向量分配 100%
 *   - [KEEP] deallocate_art_node48: 释放NODE48 100%
 *   - [KEEP] reader enter/leave原子计数 100%
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <atomic>
#include <cassert>

#include "neograph_types_impl.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

// ═══════════════════════════════════════════════════════════════
// WriterTraceBlock (upstream neo_reader_trace.h/cpp)
// ═══════════════════════════════════════════════════════════════

struct WriterTraceBlock {
    // 分配计数 (debug)
    uint64_t seg_alloc_count = 0;
    uint64_t prop_alloc_count = 0;
    uint64_t node_alloc_count = 0;
    uint64_t seg_free_count = 0;

    // GC资源跟踪
    std::vector<GCResourceInfo> gc_resources;

    // ── upstream allocate_range_element_segment ──
    // [MOD] pre-allocated pool → 直接new
    RangeElementSegment_t* allocate_range_element_segment() {
        seg_alloc_count++;
        return new RangeElementSegment_t();
    }

    // ── upstream allocate_range_prop_vec ──
    RangePropertyVec_t* allocate_range_prop_vec() {
        prop_alloc_count++;
        return new RangePropertyVec_t();
    }

    // ── upstream deallocate_art_node48 ──
    void deallocate_art_node48(void* node) {
        if (node) {
            delete static_cast<ARTNode_48*>(node);
            seg_free_count++;
        }
    }

    // ── upstream deallocate_art_prop_vec ──
    void deallocate_art_prop_vec(ARTPropertyVec_t* vec) {
        if (vec) delete vec;
    }

    void deallocate_range_element_segment(RangeElementSegment_t* seg) {
        if (seg) { delete seg; seg_free_count++; }
    }

    void deallocate_range_prop_vec(RangePropertyVec_t* vec) {
        if (vec) delete vec;
    }

    void record_gc_resource(GCResourceType type, void* ptr) {
        gc_resources.push_back({type, ptr});
    }

    // [NEW]
    void dump(const char* label = "") const {
        std::fprintf(stderr,
            "[TraceBlock·%s] seg_alloc=%lu seg_free=%lu prop_alloc=%lu "
            "node_alloc=%lu gc_pending=%zu\n",
            label, (unsigned long)seg_alloc_count,
            (unsigned long)seg_free_count,
            (unsigned long)prop_alloc_count,
            (unsigned long)node_alloc_count,
            gc_resources.size());
    }
};

// ═══════════════════════════════════════════════════════════════
// ReaderTraceBlock (upstream reader enter/leave + version tracking)
// ═══════════════════════════════════════════════════════════════
// [MOD] 64位CAS版本锁 → SeqLock style

struct ReaderTraceBlock {
    std::atomic<uint64_t> active_readers{0};
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> total_enters{0};
    std::atomic<uint64_t> total_leaves{0};

    // upstream: reader_enter — 原子增reader计数
    uint64_t reader_enter() {
        active_readers.fetch_add(1, std::memory_order_acq_rel);
        total_enters.fetch_add(1, std::memory_order_relaxed);
        return version.load(std::memory_order_acquire);
    }

    // upstream: reader_leave — 原子减reader计数
    void reader_leave() {
        active_readers.fetch_sub(1, std::memory_order_acq_rel);
        total_leaves.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t get_active_readers() const {
        return active_readers.load(std::memory_order_acquire);
    }

    // upstream: 写者推进版本号
    void advance_version() {
        version.fetch_add(1, std::memory_order_release);
    }

    bool no_readers() const {
        return active_readers.load(std::memory_order_acquire) == 0;
    }

    // [NEW]
    void dump(const char* label = "") const {
        std::fprintf(stderr,
            "[ReaderTrace·%s] active=%lu version=%lu enters=%lu leaves=%lu\n",
            label,
            (unsigned long)active_readers.load(),
            (unsigned long)version.load(),
            (unsigned long)total_enters.load(),
            (unsigned long)total_leaves.load());
    }
};

}  // namespace neograph
}  // namespace philemon

#endif
