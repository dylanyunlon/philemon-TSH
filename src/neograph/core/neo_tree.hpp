#pragma once
/**
 * neo_tree.hpp — Version-chain tree with MVCC and GC profiling
 *
 * 骨架来源: upstream/.../include/neo_tree.h (127行) + src/neo_tree.cpp (446行)
 * 修改 (~20%):
 *   - find_version: 记录版本链遍历长度 (version_walk_length_sum)
 *     → 诊断版本链膨胀 (如果平均 walk > 5, GC太慢)
 *   - version_gc: 记录回收前后的版本数差 + 延迟ns
 *   - commit_version: PHILE_NEO_TRACE 打印 timestamp 和当前版本数
 *   - edges 模板: 累加 neighbor_scan_count (degree分布诊断)
 *   - insert_edge / insert_vertex: 操作计数器
 *   - release_version: 增加 ref_cnt underflow 断言
 *
 * Milestone: M071
 */

#include "../art/art_core.hpp"
#include "../art/art_ops.hpp"
#include "../utils/neo_spin_lock.hpp"
#include "../utils/neo_config.hpp"
#include "../include/neo_types.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <cstring>
#include <atomic>

namespace container {

// Forward declare
class NeoTreeVersion;
struct WriterTraceBlock;
struct RangeTree;

// ─── Tree-level profiling (NEW) ───
struct NeoTreeStats {
    std::atomic<uint64_t> version_walk_sum{0};
    std::atomic<uint64_t> version_walk_calls{0};
    std::atomic<uint64_t> gc_reclaimed{0};
    std::atomic<uint64_t> commit_count{0};
    std::atomic<uint64_t> insert_vertex_count{0};
    std::atomic<uint64_t> insert_edge_count{0};
    std::atomic<uint64_t> remove_vertex_count{0};
    std::atomic<uint64_t> remove_edge_count{0};
    std::atomic<uint64_t> neighbor_scan_total{0};

    void dump() const {
        double avg_walk = version_walk_calls.load() > 0
            ? (double)version_walk_sum.load() / version_walk_calls.load()
            : 0.0;
        std::fprintf(stderr,
            "[NEO-TREE] commits=%llu avg_version_walk=%.2f gc_reclaimed=%llu\n"
            "           ins_v=%llu ins_e=%llu rem_v=%llu rem_e=%llu\n"
            "           neighbor_scans=%llu\n",
            (unsigned long long)commit_count.load(), avg_walk,
            (unsigned long long)gc_reclaimed.load(),
            (unsigned long long)insert_vertex_count.load(),
            (unsigned long long)insert_edge_count.load(),
            (unsigned long long)remove_vertex_count.load(),
            (unsigned long long)remove_edge_count.load(),
            (unsigned long long)neighbor_scan_total.load());
    }
};
inline NeoTreeStats& neo_tree_stats() { static NeoTreeStats s; return s; }

// ──────────────── NeoTree ────────────────
class NeoTree {
public:
    NeoTreeVersion* version_head{};
    NeoTreeVersion* uncommited_version{};
    uint16_t version_num: 15;
    uint16_t direct_gc_flag: 1;
    SpinLock writer_lock{};

    explicit NeoTree(uint64_t prefix);
    ~NeoTree();

    // ─── Read operations ───
    [[nodiscard]] bool has_vertex(uint64_t vertex, uint64_t timestamp) const;
    [[nodiscard]] bool has_edge(uint64_t src, uint64_t dest, uint64_t timestamp) const;
    [[nodiscard]] uint64_t get_degree(uint64_t vertex, uint64_t timestamp) const;
    [[nodiscard]] RangeElement* get_neighbor_addr(uint64_t vertex, uint64_t timestamp) const;
    [[nodiscard]] std::pair<uint64_t, uint64_t> get_filling_info(uint64_t timestamp) const;

#if VERTEX_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_vertex_property(uint64_t vertex, uint8_t property_id,
                                                 uint64_t timestamp) const;
#endif
#if EDGE_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_edge_property(uint64_t src, uint64_t dest,
                                               uint8_t property_id,
                                               uint64_t timestamp) const;
#endif

    bool get_neighbor(uint64_t src, std::vector<RangeElement>& neighbor,
                      uint64_t timestamp) const;

    // ─── Edges iteration (upstream template + scan counting) ───
    template<typename F>
    void edges(uint64_t src, F&& callback, uint64_t timestamp) const;

    static void intersect(NeoTree* tree1, uint64_t src1,
                          NeoTree* tree2, uint64_t src2,
                          std::vector<uint64_t>& result, uint64_t timestamp);
    static uint64_t intersect(NeoTree* tree1, uint64_t src1,
                              NeoTree* tree2, uint64_t src2,
                              uint64_t timestamp);

    // ─── Write operations ───
    void insert_vertex(uint64_t vertex, Property_t* property,
                       WriterTraceBlock* trace_block);
    void insert_vertex_batch(const uint64_t* vertices, Property_t** properties,
                             uint64_t count, WriterTraceBlock* trace_block);

#if VERTEX_PROPERTY_NUM >= 1
    void set_vertex_property(uint64_t vertex, uint8_t property_id, Property_t property);
#endif

    void insert_edge(uint64_t src, uint64_t dest, Property_t* property,
                     WriterTraceBlock* trace_block);
    void insert_edge_batch(const std::pair<RangeElement, RangeElement>* edges,
                           Property_t** properties, uint64_t count,
                           WriterTraceBlock* trace_block);

#if EDGE_PROPERTY_NUM >= 1
    void set_edge_property(uint64_t src, uint64_t dest, uint8_t property_id,
                           Property_t property, WriterTraceBlock* trace_block);
#endif

    bool remove_vertex(uint64_t vertex, bool is_directed,
                       WriterTraceBlock* trace_block);
    void remove_edge(uint64_t src, uint64_t dest, WriterTraceBlock* trace_block);
    void clean(WriterTraceBlock* trace_block);

    // ─── Version management (upstream + walk counting) ───
    [[nodiscard]] NeoTreeVersion* find_version(uint64_t timestamp) const {
        neo_tree_stats().version_walk_calls.fetch_add(1, std::memory_order_relaxed);
        uint64_t walk = 0;
        auto v = version_head;
        while (v != nullptr) {
            walk++;
            if (v->timestamp <= timestamp) {
                v->ref_cnt.fetch_add(1, std::memory_order_acquire);
                neo_tree_stats().version_walk_sum.fetch_add(walk, std::memory_order_relaxed);
                return v;
            }
            v = v->next;
        }
        neo_tree_stats().version_walk_sum.fetch_add(walk, std::memory_order_relaxed);
        return nullptr;
    }

    static void release_version(NeoTreeVersion* version) {
        if (version == nullptr) return;
        auto prev = version->ref_cnt.fetch_sub(1, std::memory_order_release);
        assert(prev > 0);  // NEW: underflow guard
        (void)prev;
    }

    bool finish_version(NeoTreeVersion* version);
    bool commit_version(uint64_t timestamp);
    void version_gc(NeoTreeVersion*& version, std::vector<uint64_t>& readers,
                    WriterTraceBlock* trace_block);
    void gc(WriterTraceBlock* trace_block);
};

// ─── edges template implementation (upstream + scan counting) ───
template<typename F>
void NeoTree::edges(uint64_t src, F&& callback, uint64_t timestamp) const {
    auto version = find_version(timestamp);
    if (version == nullptr) return;

    auto& vertex = version->vertex_map->at(src & VERTEX_GROUP_MASK);
    uint64_t degree = vertex.degree;
    neo_tree_stats().neighbor_scan_total.fetch_add(degree, std::memory_order_relaxed);

    if (!vertex.is_independent) {
        auto iter = (RangeElement*)version->node_block->at(vertex.range_node_idx).arr_ptr
                    + vertex.neighbor_offset;
        for (uint64_t i = 0; i < degree; i++) {
            uint64_t dst = iter[i];
            callback(dst, 0.0);
        }
    } else if (!vertex.is_art) {
        ((RangeTree*)vertex.neighborhood_ptr)->for_each(callback);
    } else {
        ((ART*)vertex.neighborhood_ptr)->for_each_element(callback);
    }
    release_version(version);
}

} // namespace container
