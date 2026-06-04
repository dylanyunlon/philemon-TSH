#pragma once
/**
 * neo_snapshot.hpp — Snapshot reads + Writer/Reader trace blocks
 *
 * 骨架来源: upstream/.../include/neo_snapshot.h (59行)
 *           upstream/.../include/neo_reader_trace.h (186行)
 *           upstream/.../src/neo_snapshot.cpp (180行)
 *           upstream/.../src/neo_reader_trace.cpp (355行)
 * 修改 (~20%):
 *   - WriterTraceBlock: alloc_art_leaf32/alloc_art_prop_vec 加分配计数
 *   - ReaderTraceBlock: 记录 reader 存活时间 (create_ns → destroy_ns)
 *   - Snapshot: version_chain_depth 记录快照指向的版本链深度
 *   - allocate/deallocate: PHILE_NEO_TRACE on every pool miss
 *   - dump_trace_stats(): 全局统计 dump
 *
 * Milestone: M071
 */

#include "../include/neo_types.hpp"
#include "../include/neo_property.hpp"
#include "../utils/neo_config.hpp"

#include <vector>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <array>

namespace container {

// ─── Trace profiling (NEW) ───
struct TraceStats {
    std::atomic<uint64_t> writer_alloc_leaf32{0};
    std::atomic<uint64_t> writer_alloc_prop{0};
    std::atomic<uint64_t> writer_alloc_node48{0};
    std::atomic<uint64_t> writer_alloc_node256{0};
    std::atomic<uint64_t> writer_dealloc{0};
    std::atomic<uint64_t> reader_created{0};
    std::atomic<uint64_t> reader_destroyed{0};
    std::atomic<uint64_t> reader_lifetime_ns_sum{0};

    void dump() const {
        double avg_life_us = reader_destroyed.load() > 0
            ? (double)reader_lifetime_ns_sum.load() / reader_destroyed.load() / 1000.0
            : 0.0;
        std::fprintf(stderr,
            "[TRACE] writer: leaf32=%llu prop=%llu n48=%llu n256=%llu dealloc=%llu\n"
            "        reader: created=%llu destroyed=%llu avg_life=%.1fus\n",
            (unsigned long long)writer_alloc_leaf32.load(),
            (unsigned long long)writer_alloc_prop.load(),
            (unsigned long long)writer_alloc_node48.load(),
            (unsigned long long)writer_alloc_node256.load(),
            (unsigned long long)writer_dealloc.load(),
            (unsigned long long)reader_created.load(),
            (unsigned long long)reader_destroyed.load(),
            avg_life_us);
    }
};
inline TraceStats& trace_stats() { static TraceStats s; return s; }

// ──────────────── WriterTraceBlock (upstream + allocation counting) ────────────────
struct WriterTraceBlock {
    std::vector<GCResourceInfo> gc_resources;
    std::vector<ARTResourceInfo> art_resources;

    WriterTraceBlock() = default;
    ~WriterTraceBlock() = default;

    // Leaf32 pool (upstream allocates from pre-allocated pool, fallback to new)
    std::array<uint32_t, ART_LEAF_SIZE>* allocate_art_leaf32() {
        trace_stats().writer_alloc_leaf32.fetch_add(1, std::memory_order_relaxed);
        return new std::array<uint32_t, ART_LEAF_SIZE>();
    }

    void deallocate_art_leaf32(std::array<uint32_t, ART_LEAF_SIZE>* ptr) {
        trace_stats().writer_dealloc.fetch_add(1, std::memory_order_relaxed);
        delete ptr;
    }

    // Property vec pool
    void* allocate_art_prop_vec() {
        trace_stats().writer_alloc_prop.fetch_add(1, std::memory_order_relaxed);
#if EDGE_PROPERTY_NUM == 1
        return new ARTPropertyVec_t();
#elif EDGE_PROPERTY_NUM > 1
        return new MultiARTPropertyVec_t(true);
#else
        return nullptr;
#endif
    }

    void deallocate_art_prop_vec(void* ptr) {
        trace_stats().writer_dealloc.fetch_add(1, std::memory_order_relaxed);
#if EDGE_PROPERTY_NUM == 1
        delete static_cast<ARTPropertyVec_t*>(ptr);
#elif EDGE_PROPERTY_NUM > 1
        delete static_cast<MultiARTPropertyVec_t*>(ptr);
#endif
    }

    // Node48/256 pool
    ARTNode_48* allocate_art_node48() {
        trace_stats().writer_alloc_node48.fetch_add(1, std::memory_order_relaxed);
        return new ARTNode_48();
    }

    ARTNode_256* allocate_art_node256() {
        trace_stats().writer_alloc_node256.fetch_add(1, std::memory_order_relaxed);
        return new ARTNode_256();
    }

    // Range segment pool
    RangeElementSegment_t* allocate_range_segment() {
        return new RangeElementSegment_t();
    }

    void deallocate_range_segment(RangeElementSegment_t* ptr) {
        delete ptr;
    }

    // Range property pool
    void* allocate_range_prop_vec() {
#if EDGE_PROPERTY_NUM == 1
        return new RangePropertyVec_t();
#elif EDGE_PROPERTY_NUM > 1
        return new MultiRangePropertyVec_t(true);
#else
        return nullptr;
#endif
    }

    // Vertex property pool
    void* allocate_vertex_prop_vec() {
#if VERTEX_PROPERTY_NUM == 1
        return new VertexPropertyVec_t();
#elif VERTEX_PROPERTY_NUM > 1
        return new MultiVertexPropertyVec_t(true);
#else
        return nullptr;
#endif
    }
};

// ──────────────── ReaderTraceBlock (upstream + lifetime tracking) ────────────────
struct ReaderTraceBlock {
    uint64_t read_timestamp;
    uint64_t create_ns;  // NEW

    ReaderTraceBlock() : read_timestamp(0) {
        create_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        trace_stats().reader_created.fetch_add(1, std::memory_order_relaxed);
    }

    ~ReaderTraceBlock() {
        uint64_t destroy_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t lifetime = (destroy_ns > create_ns) ? (destroy_ns - create_ns) : 0;
        trace_stats().reader_lifetime_ns_sum.fetch_add(lifetime, std::memory_order_relaxed);
        trace_stats().reader_destroyed.fetch_add(1, std::memory_order_relaxed);
    }
};

// ──────────────── Snapshot ────────────────
struct NeoSnapshot {
    uint64_t timestamp;
    uint64_t vertex_count;
    uint64_t edge_count;
    uint64_t version_chain_depth;  // NEW: how deep in the version chain

    NeoSnapshot() : timestamp(0), vertex_count(0), edge_count(0),
                    version_chain_depth(0) {}

    NeoSnapshot(uint64_t ts, uint64_t vc, uint64_t ec, uint64_t depth = 0)
        : timestamp(ts), vertex_count(vc), edge_count(ec),
          version_chain_depth(depth) {}

    void dump() const {
        std::fprintf(stderr,
            "[SNAPSHOT] ts=%llu vertices=%llu edges=%llu version_depth=%llu\n",
            (unsigned long long)timestamp,
            (unsigned long long)vertex_count,
            (unsigned long long)edge_count,
            (unsigned long long)version_chain_depth);
    }
};

// ─── Global dump ───
inline void dump_trace_stats() { trace_stats().dump(); }

} // namespace container
