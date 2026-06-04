#pragma once
/**
 * neo_tree_version.hpp — MVCC version node with operation profiling
 *
 * 骨架来源: upstream/.../include/neo_tree_version.h (157行)
 *           upstream/.../src/neo_tree_version.cpp (2345行)
 * 修改 (~20%):
 *   - 版本创建: 记录 creation_wall_ns (微秒级时间戳, 用于GC延迟诊断)
 *   - insert_edge: 对 range → range_tree → ART 三级升级路径分别计数
 *     (range_insert, range_tree_insert, art_insert)
 *   - vertex_map_update: 记录每次segment内元素移动距离 (move_distance)
 *   - find_range_node: 记录 segment 命中率 (first_try_hit / total)
 *   - second_insert_edge: 对 split / upgrade / in-place 三种分支计数
 *   - edges 模板: 传递degree到 neo_tree_stats 的 neighbor_scan_total
 *
 * Milestone: M071
 */

#include "neo_tree.hpp"
#include "../include/neo_types.hpp"
#include "../include/neo_property.hpp"

#include <fstream>
#include <vector>
#include <atomic>
#include <chrono>
#include <cstdio>

namespace container {

// Forward
struct WriterTraceBlock;
class RangeTree;
class ART;
struct NeoRangeNode;

// ─── Version-level profiling (NEW) ───
struct VersionOpsStats {
    std::atomic<uint64_t> range_direct_insert{0};
    std::atomic<uint64_t> range_tree_insert{0};
    std::atomic<uint64_t> art_insert{0};
    std::atomic<uint64_t> split_ops{0};
    std::atomic<uint64_t> upgrade_to_range_tree{0};
    std::atomic<uint64_t> upgrade_to_art{0};
    std::atomic<uint64_t> find_range_node_calls{0};
    std::atomic<uint64_t> find_range_node_first_hit{0};
    std::atomic<uint64_t> vertex_map_move_distance{0};

    void dump() const {
        double hit_rate = find_range_node_calls.load() > 0
            ? (double)find_range_node_first_hit.load() / find_range_node_calls.load()
            : 0.0;
        std::fprintf(stderr,
            "[NEO-VERSION] range_direct=%llu range_tree=%llu art=%llu\n"
            "              splits=%llu upgrade_rt=%llu upgrade_art=%llu\n"
            "              find_node_hit_rate=%.3f move_dist=%llu\n",
            (unsigned long long)range_direct_insert.load(),
            (unsigned long long)range_tree_insert.load(),
            (unsigned long long)art_insert.load(),
            (unsigned long long)split_ops.load(),
            (unsigned long long)upgrade_to_range_tree.load(),
            (unsigned long long)upgrade_to_art.load(),
            hit_rate,
            (unsigned long long)vertex_map_move_distance.load());
    }
};
inline VersionOpsStats& version_ops_stats() { static VersionOpsStats s; return s; }

// ──────────────── NeoTreeVersion ────────────────
class NeoTreeVersion {
public:
    RangeNodeSegment_t* node_block;
    NeoTreeVersion* next;
    uint64_t timestamp;
    VertexMap_t* vertex_map;
    Bitmap<INDEPENDENT_MAP_BLOCK_NUM> independent_map{};
    bool resource_handled{};
#if VERTEX_PROPERTY_NUM == 1
    VertexPropertyVec_t* vertex_property_map;
#elif VERTEX_PROPERTY_NUM > 1
    MultiVertexPropertyVec_t* vertex_property_map;
#endif
    std::atomic<uint64_t> ref_cnt{};
    std::vector<GCResourceInfo>* resources;

    // ─── NEW: creation timestamp for GC latency diagnosis ───
    uint64_t creation_wall_ns;

    // ─── Construction / Destruction ───
    explicit NeoTreeVersion(NeoTreeVersion* prev, WriterTraceBlock* trace_block)
        : node_block(nullptr), next(prev), timestamp(0),
          vertex_map(nullptr), resource_handled(false),
#if VERTEX_PROPERTY_NUM >= 1
          vertex_property_map(nullptr),
#endif
          resources(nullptr)
    {
        creation_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void clean(WriterTraceBlock* trace_block);
    ~NeoTreeVersion();

    // ─── Read operations ───
    [[nodiscard]] bool has_vertex(uint64_t vertex) const;
    [[nodiscard]] bool has_edge(uint64_t src, uint64_t dest) const;
    [[nodiscard]] uint64_t get_degree(uint64_t vertex) const;
    [[nodiscard]] RangeElement* get_neighbor_addr(uint64_t vertex) const;
    [[nodiscard]] std::pair<uint64_t, uint64_t> get_filling_info() const;

#if VERTEX_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_vertex_property(uint64_t vertex, uint8_t pid) const;
#endif
#if EDGE_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_edge_property(uint64_t src, uint64_t dest,
                                               uint8_t pid) const;
#endif

    bool get_neighbor(uint64_t src, std::vector<uint64_t>& neighbor) const;

    // ─── edges template (upstream + degree counting) ───
    template<typename F>
    void edges(uint64_t src, F&& callback) const {
        auto& vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
        uint64_t deg = vertex.degree;
        neo_tree_stats().neighbor_scan_total.fetch_add(deg, std::memory_order_relaxed);

        if (!vertex.is_independent) {
            auto iter = (RangeElement*)node_block->at(vertex.range_node_idx).arr_ptr
                        + vertex.neighbor_offset;
            for (uint64_t i = 0; i < deg; i++) {
                uint64_t dst = iter[i];
                callback(dst, 0.0);
            }
        } else if (!vertex.is_art) {
            ((RangeTree*)vertex.neighborhood_ptr)->for_each(callback);
        } else {
            ((ART*)vertex.neighborhood_ptr)->for_each_element(callback);
        }
    }

    static void intersect(NeoTreeVersion* v1, uint64_t src1,
                          NeoTreeVersion* v2, uint64_t src2,
                          std::vector<uint64_t>& result);
    static uint64_t intersect(NeoTreeVersion* v1, uint64_t src1,
                              NeoTreeVersion* v2, uint64_t src2);

    std::vector<uint16_t>* get_vertices_in_node(uint16_t node_idx);
    std::vector<std::pair<uint16_t, uint16_t>>* get_vertices_in_node_with_offset(uint16_t node_idx);

    // ─── Write operations (with profiling) ───
    void insert_vertex(uint64_t vertex, Property_t* property);
    void insert_vertex_batch(const uint64_t* vertices, Property_t** properties,
                             uint64_t count);

#if VERTEX_PROPERTY_NUM >= 1
    void set_vertex_property(uint64_t vertex, uint8_t pid, Property_t property);
#endif

    void insert_edge(uint64_t src, uint64_t dest, Property_t* property,
                     WriterTraceBlock* trace_block);
    void second_insert_edge(uint64_t src, RangeElement target, NeoVertex& vertex,
                            Property_t* property, WriterTraceBlock* trace_block);
    void append_new_list(uint64_t cur_node_num, std::vector<NeoRangeNode>& new_nodes,
                         const std::pair<RangeElement, RangeElement>* edges,
                         Property_t** properties, uint64_t count,
                         WriterTraceBlock* trace_block);
    void node_insert_edge_batch(uint16_t node_idx, std::vector<NeoRangeNode>& new_nodes,
                                uint64_t cur_node_num,
                                const std::pair<RangeElement, RangeElement>* edges,
                                Property_t** properties, uint64_t count,
                                WriterTraceBlock* trace_block);
    void insert_edge_batch(const std::pair<RangeElement, RangeElement>* edges,
                           Property_t** properties, uint64_t count,
                           WriterTraceBlock* trace_block);

#if EDGE_PROPERTY_NUM >= 1
    void set_edge_property(uint64_t src, uint64_t dest, uint8_t pid,
                           Property_t property, WriterTraceBlock* trace_block);
#endif

    void gc_copied(WriterTraceBlock* trace_block);
    void handle_resources_ref();
    void gc_ref(WriterTraceBlock* trace_block);
    void destroy();
    void remove_vertex(uint64_t vertex, bool is_directed, WriterTraceBlock* trace_block);
    void remove_edge(uint64_t src, uint64_t dest, WriterTraceBlock* trace_block);

private:
    // ─── find_range_node with hit-rate tracking ───
    [[nodiscard]] NeoRangeNode* find_range_node(uint64_t vertex) const;

    int find_position_to_be_inserted(uint64_t src, uint64_t dest,
                                     NeoVertex vertex, RangeElementSegment_t* arr,
                                     uint16_t arr_size, std::vector<uint16_t>* vertices);
    uint16_t find_mid_count(std::vector<uint16_t>* vertices);
    void remove_node(NeoRangeNode& node, uint64_t node_idx,
                     uint16_t nodes_move_begin_point);
    void vertex_map_update(RangeElementSegment_t* segment, uint16_t* vertices,
                           uint16_t vertex_num, uint16_t node_idx, int offset);
    void vertex_map_update_split(RangeElementSegment_t* segment, uint16_t* vertices,
                                 uint16_t vertex_num, uint16_t node_idx,
                                 uint16_t last_vertex_unchanged, int offset);
    RangeTree* extract2range_tree(uint64_t src, uint64_t degree,
                                  uint64_t new_element, Property_t* new_property,
                                  NeoRangeNode& range_node,
                                  WriterTraceBlock* trace_block);
    ART* direct2art(uint64_t src, uint64_t degree,
                    uint64_t new_element, Property_t* new_property,
                    NeoRangeNode& range_node,
                    WriterTraceBlock* trace_block);
};

} // namespace container
