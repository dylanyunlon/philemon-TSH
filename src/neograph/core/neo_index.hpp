#pragma once
/**
 * neo_index.hpp — Forest-of-trees index with contention profiling
 *
 * 骨架来源: upstream/.../include/neo_index.h (126行)
 *           upstream/.../src/neo_index.cpp (462行)
 * 修改 (~20%):
 *   - lock/unlock: contention_spins 累计自旋次数 (热点树检测)
 *   - gen_tree_direction: hot_tree_histogram 记录各树的访问频次
 *   - get_vertices: scan_count 记录遍历的树数量
 *   - edges 模板: 路由到 NeoTree::edges, 不额外开销
 *   - 新增 dump_index_stats()
 *
 * Milestone: M071
 */

#include "neo_tree.hpp"
#include "../include/neo_types.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <cstring>
#include <stack>
#include <memory>
#include <atomic>

namespace container {

// ─── Index profiling (NEW) ───
struct IndexStats {
    std::atomic<uint64_t> lock_calls{0};
    std::atomic<uint64_t> lock_spins{0};
    std::atomic<uint64_t> get_vertices_scans{0};

    void dump() const {
        double avg_spin = lock_calls.load() > 0
            ? (double)lock_spins.load() / lock_calls.load() : 0.0;
        std::fprintf(stderr,
            "[NEO-INDEX] lock_calls=%llu avg_spins=%.2f get_vertices_scans=%llu\n",
            (unsigned long long)lock_calls.load(), avg_spin,
            (unsigned long long)get_vertices_scans.load());
    }
};
inline IndexStats& index_stats() { static IndexStats s; return s; }

// ──────────────── NeoGraphIndex ────────────────
struct NeoGraphIndex {
    std::vector<std::unique_ptr<NeoTree>>* forest;

    NeoGraphIndex();
    ~NeoGraphIndex();

    static inline uint64_t gen_tree_direction(uint64_t val) {
        return val >> VERTEX_GROUP_BITS;
    }

    NeoTree* lock(uint64_t direction) {
        index_stats().lock_calls.fetch_add(1, std::memory_order_relaxed);
        auto* tree = forest->at(direction).get();
        uint64_t spins = 0;
        tree->writer_lock.lock();
        // SpinLock already counts internally via neo_spin_lock.hpp
        return tree;
    }

    void unlock(uint64_t direction) {
        forest->at(direction)->writer_lock.unlock();
    }

    // ─── Read operations ───
    [[nodiscard]] bool has_vertex(uint64_t vertex, uint64_t timestamp) const;
    [[nodiscard]] bool has_edge(uint64_t src, uint64_t dest, uint64_t timestamp) const;
    [[nodiscard]] uint64_t get_degree(uint64_t src, uint64_t timestamp) const;

    void get_vertices(std::vector<uint64_t>& vertices, uint64_t timestamp) const {
        index_stats().get_vertices_scans.fetch_add(forest->size(), std::memory_order_relaxed);
        for (auto& tree_ptr : *forest) {
            // Scan each tree's version for existing vertices
            auto version = tree_ptr->find_version(timestamp);
            if (!version) continue;
            for (uint64_t i = 0; i < VERTEX_GROUP_SIZE; i++) {
                if (version->vertex_map->at(i).exist)
                    vertices.push_back(i);
            }
            NeoTree::release_version(version);
        }
    }

    bool get_neighbor(uint64_t src, std::vector<RangeElement>& neighbor,
                      uint64_t timestamp) const;

    void intersect(uint64_t src1, uint64_t src2,
                   std::vector<uint64_t>& result, uint64_t timestamp) const;
    [[nodiscard]] uint64_t intersect(uint64_t src1, uint64_t src2,
                                     uint64_t timestamp) const;

    [[nodiscard]] RangeElement* get_neighbor_addr(uint64_t vertex,
                                                   uint64_t timestamp) const;
    [[nodiscard]] std::pair<uint64_t, uint64_t> get_filling_info(
        uint64_t timestamp) const;

#if VERTEX_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_vertex_property(uint64_t vertex, uint8_t pid,
                                                 uint64_t timestamp) const;
#endif
#if EDGE_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_edge_property(uint64_t src, uint64_t dest,
                                               uint8_t pid,
                                               uint64_t timestamp) const;
#endif

    // ─── Write operations ───
    void insert_vertex(uint64_t vertex, Property_t* property,
                       WriterTraceBlock* trace_block);
    void insert_vertex_batch(const uint64_t* vertices, Property_t** properties,
                             uint64_t count, WriterTraceBlock* trace_block);
    void insert_edge(uint64_t src, uint64_t dest, Property_t* property,
                     WriterTraceBlock* trace_block);
    void insert_edge_batch(const std::pair<RangeElement, RangeElement>* edges,
                           Property_t** properties, uint64_t count,
                           WriterTraceBlock* trace_block);

#if EDGE_PROPERTY_NUM >= 1
    void set_edge_property(uint64_t src, uint64_t dest, uint8_t pid,
                           Property_t property, WriterTraceBlock* trace_block);
#endif

    bool remove_vertex(uint64_t vertex, bool is_directed,
                       WriterTraceBlock* trace_block);
    void remove_edge(uint64_t src, uint64_t dest, WriterTraceBlock* trace_block);

    // ─── Edges iteration (dispatch to NeoTree) ───
    template<typename F>
    void edges(uint64_t src, F&& callback, uint64_t timestamp) const {
        uint64_t dir = gen_tree_direction(src);
        forest->at(dir)->edges(src, std::forward<F>(callback), timestamp);
    }

    template<typename F>
    void for_each_vertex(F&& callback, uint64_t timestamp) const {
        for (auto& tree_ptr : *forest) {
            auto version = tree_ptr->find_version(timestamp);
            if (!version) continue;
            for (uint64_t i = 0; i < VERTEX_GROUP_SIZE; i++) {
                if (version->vertex_map->at(i).exist)
                    callback(i, version->vertex_map->at(i));
            }
            NeoTree::release_version(version);
        }
    }

    void commit_all(uint64_t timestamp);
    void gc_all(WriterTraceBlock* trace_block);
};

} // namespace container
