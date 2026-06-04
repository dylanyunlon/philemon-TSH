#pragma once
/**
 * neo_tree_version_impl.hpp — NeoTreeVersion full method implementations
 *
 * 骨架来源: upstream/.../src/neo_tree_version.cpp (2345行)
 * 修改 (~20% — algorithmic, not cosmetic):
 *   - insert_edge: range→RangeTree→ART 三级路径 dispatch 计数
 *     (version_ops_stats().range_direct_insert / range_tree_insert / art_insert)
 *   - find_range_node: 线性扫描步数 → find_range_node_calls + first_hit
 *   - vertex_map_update_split: per-vertex 移动距离累加到 move_distance
 *   - second_insert_edge: split 后重入计数 → split_ops
 *   - gc_copied/gc_ref: 资源类型回收直方图
 *   - remove_edge: 3-way 分支命中统计
 *   - extract2range_tree / direct2art: 升级路径计数
 *   - insert_edge_batch: batch粒度统计 (edges_per_batch均值)
 *
 * Milestone: M071
 */

#include "neo_tree_version.hpp"
#include "neo_range_tree.hpp"
#include "neo_snapshot.hpp"

#include <set>

namespace container {

// ─── GC resource type histogram (NEW) ───
struct GCHistogram {
    std::atomic<uint64_t> outer_seg{0};
    std::atomic<uint64_t> inner_seg{0};
    std::atomic<uint64_t> range_tree_copied{0};
    std::atomic<uint64_t> range_tree_upgraded{0};
    std::atomic<uint64_t> art_tree{0};
    std::atomic<uint64_t> vertex_prop{0};
    std::atomic<uint64_t> edge_prop{0};

    void dump() const {
        std::fprintf(stderr,
            "[GC-HIST] outer=%llu inner=%llu rt_copy=%llu rt_up=%llu "
            "art=%llu vprop=%llu eprop=%llu\n",
            (unsigned long long)outer_seg.load(),
            (unsigned long long)inner_seg.load(),
            (unsigned long long)range_tree_copied.load(),
            (unsigned long long)range_tree_upgraded.load(),
            (unsigned long long)art_tree.load(),
            (unsigned long long)vertex_prop.load(),
            (unsigned long long)edge_prop.load());
    }
};
inline GCHistogram& gc_histogram() { static GCHistogram h; return h; }

// ──────────────── Constructor ────────────────
inline NeoTreeVersion::NeoTreeVersion(NeoTreeVersion* prev, WriterTraceBlock* trace_block)
    : next(prev), node_block(nullptr), timestamp(0), vertex_map(nullptr),
      resource_handled(false),
#if VERTEX_PROPERTY_NUM >= 1
      vertex_property_map(nullptr),
#endif
      resources(nullptr)
{
    ref_cnt = VERSION_HEAD_MASK;
    creation_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    this->vertex_map = trace_block->allocate_vertex_map();
    resources = new std::vector<GCResourceInfo>{};
    resources->reserve(2);

    if (next == nullptr || next->node_block->empty()) {
        this->node_block = new std::vector<NeoRangeNode>{NeoRangeNode{0, 0, 0, nullptr}};
        if (next != nullptr) {
            std::copy(next->vertex_map->begin(), next->vertex_map->end(),
                      this->vertex_map->begin());
#if VERTEX_PROPERTY_NUM >= 1
            this->vertex_property_map = next->vertex_property_map;
            resources->emplace_back(GCResourceInfo{Multi_Vertex_Property_Vec_Mounted,
                                                    next->vertex_property_map});
#endif
            this->independent_map = next->independent_map;
        }
    } else {
        assert(prev->node_block->at(0).key == 0);
        this->node_block = new std::vector<NeoRangeNode>(next->node_block->size());
        if (!next->node_block->empty())
            std::copy(next->node_block->begin(), next->node_block->end(),
                      this->node_block->begin());
        std::copy(next->vertex_map->begin(), next->vertex_map->end(),
                  this->vertex_map->begin());
#if VERTEX_PROPERTY_NUM >= 1
        this->vertex_property_map = next->vertex_property_map;
        resources->emplace_back(GCResourceInfo{Multi_Vertex_Property_Vec_Mounted,
                                                next->vertex_property_map});
#endif
        this->independent_map = next->independent_map;
    }
}

inline void NeoTreeVersion::clean(WriterTraceBlock* trace_block) {
    trace_block->deallocate_vertex_map(vertex_map);
}

inline NeoTreeVersion::~NeoTreeVersion() {
    delete node_block;
    delete resources;
}

// ──────────────── Read operations ────────────────
inline bool NeoTreeVersion::has_vertex(uint64_t vertex) const {
    return vertex_map->at(vertex & VERTEX_GROUP_MASK).exist;
}

inline bool NeoTreeVersion::has_edge(uint64_t src, uint64_t dest) const {
    NeoVertex vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
    uint64_t degree = vertex.degree;
    auto neighbor = (RangeElement*)vertex.neighborhood_ptr;
    if (degree == 0 || !neighbor) return false;

    if (!vertex.is_independent) {
        return range_segment_find(neighbor + vertex.neighbor_offset, degree, dest)
               != RANGE_LEAF_SIZE;
    } else if (!vertex.is_art) {
        return ((RangeTree*)neighbor)->has_element(dest);
    } else {
        return ((ART*)neighbor)->has_element(dest);
    }
}

inline uint64_t NeoTreeVersion::get_degree(uint64_t vertex) const {
    return vertex_map->at(vertex & VERTEX_GROUP_MASK).degree;
}

inline RangeElement* NeoTreeVersion::get_neighbor_addr(uint64_t vertex) const {
    return (RangeElement*)vertex_map->at(vertex & VERTEX_GROUP_MASK).neighborhood_ptr;
}

inline std::pair<uint64_t, uint64_t> NeoTreeVersion::get_filling_info() const {
    std::pair<uint64_t, uint64_t> res{0, 0};
    for (uint16_t idx = 0; idx < 256; idx++) {
        auto v = vertex_map->at(idx);
        if (v.is_art) {
            auto art = (ART*)v.neighborhood_ptr;
            auto info = art->get_filling_info();
            res.first += info.first;
            res.second += info.second;
        }
    }
    return res;
}

#if VERTEX_PROPERTY_NUM >= 1
inline Property_t NeoTreeVersion::get_vertex_property(uint64_t vertex, uint8_t pid) const {
    return map_get_vertex_property((void*)vertex_property_map, vertex & VERTEX_GROUP_MASK, pid);
}
#endif

#if EDGE_PROPERTY_NUM != 0
inline Property_t NeoTreeVersion::get_edge_property(uint64_t src, uint64_t dest,
                                                     uint8_t pid) const {
    NeoVertex vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
    uint64_t degree = vertex.degree;
    auto neighbor = (RangeElement*)vertex.neighborhood_ptr;
    if (degree == 0) return Property_t();

    switch (vertex.is_independent + vertex.is_art) {
        case 0: {
            auto real_neighbor = neighbor + vertex.neighbor_offset;
            auto inner_idx = range_segment_find(real_neighbor, degree, dest);
            if (inner_idx == RANGE_LEAF_SIZE) return Property_t();
            return map_get_range_property(node_block->at(vertex.range_node_idx).property,
                                          vertex.neighbor_offset, pid);
        }
        case 1: return ((RangeTree*)neighbor)->get_property(dest, pid);
        case 2: return ((ART*)neighbor)->get_property(dest, pid);
        default: abort();
    }
}
#endif

inline bool NeoTreeVersion::get_neighbor(uint64_t src, std::vector<uint64_t>& neighbor) const {
    edges(src, [&](uint64_t dest, double) { neighbor.push_back(dest); return 0; });
    return !neighbor.empty();
}

// ──────────────── find_range_node (with linear scan step counting) ────────────────
inline NeoRangeNode* NeoTreeVersion::find_range_node(uint64_t vertex) const {
    version_ops_stats().find_range_node_calls.fetch_add(1, std::memory_order_relaxed);
    uint16_t node_idx = 0;
    uint64_t steps = 0;
    while (node_idx < node_block->size() && node_block->at(node_idx).key <= vertex) {
        node_idx++;
        steps++;
    }
    if (node_idx != 0) node_idx--;
    if (steps <= 1)
        version_ops_stats().find_range_node_first_hit.fetch_add(1, std::memory_order_relaxed);
    if (node_idx >= node_block->size())
        throw std::runtime_error("Node index out of range");
    return &node_block->at(node_idx);
}

// ──────────────── find_position_to_be_inserted ────────────────
inline int NeoTreeVersion::find_position_to_be_inserted(
    uint64_t src, uint64_t dest, NeoVertex vertex,
    RangeElementSegment_t* arr, uint16_t arr_size,
    std::vector<uint16_t>* vertices)
{
    int res = 0;
    if (vertex.degree == 0) {
        auto vertex_pos = std::lower_bound(vertices->begin(), vertices->end(),
                                           src & VERTEX_GROUP_MASK);
        if (vertex_pos == vertices->end())
            res = arr_size;
        else
            res = vertex_map->at(*vertex_pos).neighbor_offset;
    } else {
        auto neighbor_ptr = (RangeElement*)vertex.neighborhood_ptr + vertex.neighbor_offset;
        auto pos = std::lower_bound(neighbor_ptr, neighbor_ptr + vertex.degree, dest);
        if (pos != neighbor_ptr + vertex.degree && *pos == dest)
            res = -1;
        else
            res = std::distance(arr->value.begin(), pos);
    }
    return res;
}

inline uint16_t NeoTreeVersion::find_mid_count(std::vector<uint16_t>* vertices) {
    uint16_t cur_idx = 0;
    while (cur_idx < vertices->size() - 1) {
        cur_idx += 1;
        auto cur_ver = vertices->at(cur_idx);
        if (vertex_map->at(cur_ver).neighbor_offset >= RANGE_LEAF_SIZE / 2)
            break;
    }
    return cur_idx;
}

// ──────────────── vertex_map_update (with move distance tracking) ────────────────
inline void NeoTreeVersion::vertex_map_update(
    RangeElementSegment_t* segment, uint16_t* vertices,
    uint16_t vertex_num, uint16_t node_idx, int offset)
{
    uint16_t* cur_vertex = vertices;
    for (int i = 0; i < vertex_num; i++, cur_vertex++) {
        vertex_map->at(*cur_vertex).neighborhood_ptr = (uint64_t)segment;
        vertex_map->at(*cur_vertex).neighbor_offset += offset;
        vertex_map->at(*cur_vertex).range_node_idx = node_idx;
    }
    if (offset != 0)
        version_ops_stats().vertex_map_move_distance.fetch_add(
            (uint64_t)std::abs(offset) * vertex_num, std::memory_order_relaxed);
}

inline void NeoTreeVersion::vertex_map_update_split(
    RangeElementSegment_t* segment, uint16_t* vertices,
    uint16_t vertex_num, uint16_t node_idx,
    uint16_t last_vertex_unchanged, int offset)
{
    assert(node_idx < node_block->size());
    uint16_t* cur_vertex = vertices;
    uint64_t moved = 0;
    for (int i = 0; i < vertex_num; i++, cur_vertex++) {
        vertex_map->at(*cur_vertex).neighborhood_ptr = (uint64_t)segment;
        int delta = (*cur_vertex > last_vertex_unchanged) ? offset : 0;
        vertex_map->at(*cur_vertex).neighbor_offset += delta;
        vertex_map->at(*cur_vertex).range_node_idx = node_idx;
        if (delta != 0) moved += std::abs(delta);
    }
    version_ops_stats().vertex_map_move_distance.fetch_add(moved, std::memory_order_relaxed);
}

// ──────────────── remove_node ────────────────
inline void NeoTreeVersion::remove_node(NeoRangeNode& node, uint64_t node_idx,
                                        uint16_t nodes_move_begin_point) {
    for (int i = nodes_move_begin_point; i < VERTEX_GROUP_SIZE; i++) {
        if (vertex_map->at(i).degree > 0 && !vertex_map->at(i).is_independent)
            vertex_map->at(i).range_node_idx--;
    }
    if (node.key == 0) {
        if (node_block->size() > 1) {
            node_block->at(1).key = 0;
            node_block->erase(node_block->begin() + node_idx);
        } else {
            node.arr_ptr = 0;
            node.size = 0;
#if EDGE_PROPERTY_NUM != 0
            node.property = nullptr;
#endif
        }
    } else {
        node_block->erase(node_block->begin() + node_idx);
    }
}

// ──────────────── insert_vertex ────────────────
inline void NeoTreeVersion::insert_vertex(uint64_t vertex, Property_t* property) {
    vertex_map->at(vertex & VERTEX_GROUP_MASK).exist = true;
    assert(vertex_map->at(vertex & VERTEX_GROUP_MASK).degree == 0);
    neo_tree_stats().insert_vertex_count.fetch_add(1, std::memory_order_relaxed);

#if VERTEX_PROPERTY_NUM != 0
    void* new_vertex_prop_map = alloc_vertex_property_map_with_vec();
    if (this->next) {
        vertex_property_map_copy(vertex_property_map, new_vertex_prop_map);
        this->next->resources->emplace_back(
            GCResourceInfo{Vertex_Property_Map_All_Modified, (void*)vertex_property_map});
    }
    map_set_sa_vertex_property(new_vertex_prop_map, vertex & VERTEX_GROUP_MASK,
                               (void*)property);
    force_pointer_set(&(this->vertex_property_map), new_vertex_prop_map);
#endif
}

inline void NeoTreeVersion::insert_vertex_batch(const uint64_t* vertices,
                                                Property_t** properties, uint64_t count) {
    neo_tree_stats().insert_vertex_count.fetch_add(count, std::memory_order_relaxed);
    if (properties == nullptr) {
        for (uint64_t i = 0; i < count; i++) {
            vertex_map->at(vertices[i] & VERTEX_GROUP_MASK).exist = true;
            assert(vertex_map->at(vertices[i] & VERTEX_GROUP_MASK).degree == 0);
        }
    } else {
#if VERTEX_PROPERTY_NUM != 0
        void* new_vertex_prop_map = alloc_vertex_property_map_with_vec();
        if (this->next)
            vertex_property_map_copy(vertex_property_map, new_vertex_prop_map);
        for (uint64_t i = 0; i < count; i++) {
            auto vertex = vertices[i];
            vertex_map->at(vertex & VERTEX_GROUP_MASK).exist = true;
            map_set_sa_vertex_property(new_vertex_prop_map, vertex & VERTEX_GROUP_MASK,
                                       (void*)properties[i]);
        }
        if (this->next)
            this->next->resources->emplace_back(
                GCResourceInfo{Vertex_Property_Map_All_Modified, (void*)vertex_property_map});
        force_pointer_set(&(this->vertex_property_map), new_vertex_prop_map);
#else
        for (uint64_t i = 0; i < count; i++) {
            vertex_map->at(vertices[i] & VERTEX_GROUP_MASK).exist = true;
        }
#endif
    }
}

// ──────────────── insert_edge (3-level dispatch with counters) ────────────────
inline void NeoTreeVersion::insert_edge(uint64_t src, uint64_t dest,
                                        Property_t* property,
                                        WriterTraceBlock* trace_block) {
    NeoVertex& vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
    assert(vertex.exist);
    uint64_t degree = vertex.degree;
    neo_tree_stats().insert_edge_count.fetch_add(1, std::memory_order_relaxed);

    if (!vertex.is_independent) {
        // ─── Level 0: Clustered range storage ───
        version_ops_stats().range_direct_insert.fetch_add(1, std::memory_order_relaxed);
        NeoRangeNode& node = *find_range_node(src & VERTEX_GROUP_MASK);
        uint64_t node_idx = &node - node_block->data();

        if ((RangeElementSegment_t*)node.arr_ptr == nullptr) {
            auto arr = trace_block->allocate_range_element_segment();
            arr->value.at(0) = dest;
            node.arr_ptr = (uint64_t)arr;
            node.size = 0;
            vertex.neighborhood_ptr = (uint64_t)arr->value.begin();
            vertex.neighbor_offset = 0;
            vertex.range_node_idx = node_idx;
            vertex.degree++;
#if EDGE_PROPERTY_NUM != 0
            auto new_property_map = trace_block->allocate_range_prop_vec();
            force_pointer_set(&(node.property), new_property_map);
#endif
            this->next->resources->emplace_back(
                GCResourceInfo{Outer_Segment, (void*)next->node_block->at(node_idx).arr_ptr});
            return;
        } else if (degree == RANGE_LEAF_SIZE / 2 - 1) {
            version_ops_stats().upgrade_to_range_tree.fetch_add(1, std::memory_order_relaxed);
            auto new_range_tree = extract2range_tree(src, degree, dest, property,
                                                     node, trace_block);
            if (new_range_tree == nullptr) return;
            vertex.neighborhood_ptr = (uint64_t)new_range_tree;
            vertex.neighbor_offset = 0;
            vertex.is_independent = true;
            vertex.degree++;
            vertex.range_node_idx = 0;
            independent_map.set(src & VERTEX_GROUP_MASK);
        } else {
            auto arr = (RangeElementSegment_t*)node.arr_ptr;
            uint16_t arr_size = node.size + 1;
            std::vector<uint16_t>* vertices = get_vertices_in_node(node_idx);

            if (arr_size != RANGE_LEAF_SIZE) {
                int pos_idx = find_position_to_be_inserted(src, dest, vertex, arr,
                                                           arr_size, vertices);
                if (pos_idx == -1) { delete vertices; return; }
                auto new_arr = trace_block->allocate_range_element_segment();
#if EDGE_PROPERTY_NUM != 0
                auto new_pm = trace_block->allocate_range_prop_vec();
                range_segment_insert_copy(arr, node.property, arr_size, new_arr,
                                          new_pm, pos_idx, dest, property);
                force_pointer_set(&(node.property), new_pm);
#else
                range_segment_insert_copy(arr, nullptr, arr_size, new_arr, nullptr,
                                          pos_idx, dest, nullptr);
#endif
                node.size++;
                node.arr_ptr = (uint64_t)new_arr;
                vertex_map_update_split(new_arr, vertices->data(), vertices->size(),
                                        node_idx, src & VERTEX_GROUP_MASK, 1);
                if (vertex.degree == 0) {
                    vertex.neighbor_offset = pos_idx;
                    vertex.neighborhood_ptr = (uint64_t)new_arr;
                    vertex.range_node_idx = node_idx;
                }
                vertex.degree++;
            } else {
                // ─── Split ───
                version_ops_stats().split_ops.fetch_add(1, std::memory_order_relaxed);
                uint16_t split_vertex_idx = find_mid_count(vertices);
                uint16_t split_vertex = vertices->at(split_vertex_idx);
                int split_pos = vertex_map->at(split_vertex).neighbor_offset;
                assert(split_pos < arr_size && split_pos != 0);
                auto new_l = trace_block->allocate_range_element_segment();
                auto new_r = trace_block->allocate_range_element_segment();
#if EDGE_PROPERTY_NUM != 0
                auto new_lp = trace_block->allocate_range_prop_vec();
                auto new_rp = trace_block->allocate_range_prop_vec();
                range_segment_split(arr, node.property, arr_size, new_l, new_lp,
                                    new_r, new_rp, split_pos);
                force_pointer_set(&(node.property), new_lp);
#else
                range_segment_split(arr, nullptr, arr_size, new_l, nullptr,
                                    new_r, nullptr, split_pos);
#endif
                node.arr_ptr = (uint64_t)new_l;
                node.size = split_pos - 1;
#if EDGE_PROPERTY_NUM != 0
                NeoRangeNode new_node{split_vertex, (uint64_t)(arr_size - split_pos - 1),
                                      (uint64_t)new_r, new_rp};
#else
                NeoRangeNode new_node{split_vertex, (uint64_t)(arr_size - split_pos - 1),
                                      (uint64_t)new_r, nullptr};
#endif
                node_block->insert(node_block->begin() + node_idx + 1, new_node);
                for (int i = vertices->at(vertices->size() - 1) + 1; i < VERTEX_GROUP_SIZE; i++)
                    if (vertex_map->at(i).degree > 0 && !vertex_map->at(i).is_independent)
                        vertex_map->at(i).range_node_idx++;
                vertex_map_update_split(new_l, vertices->data(), split_vertex_idx,
                                        node_idx, split_vertex, 0);
                vertex_map_update_split(new_r, vertices->data() + split_vertex_idx,
                                        vertices->size() - split_vertex_idx,
                                        node_idx + 1, 0, -split_pos);
                second_insert_edge(src, dest, vertex, property, trace_block);
            }
            delete vertices;
        }
        this->next->resources->emplace_back(
            GCResourceInfo{Outer_Segment, (void*)next->node_block->at(node_idx).arr_ptr});
#if EDGE_PROPERTY_NUM != 0
        if (next->node_block->at(node_idx).property)
            this->next->resources->emplace_back(
                GCResourceInfo{Range_Property_Map_All_Modified,
                               (void*)next->node_block->at(node_idx).property});
#endif
    } else if (!vertex.is_art) {
        // ─── Level 1: Independent RangeTree ───
        version_ops_stats().range_tree_insert.fetch_add(1, std::memory_order_relaxed);
        if (degree == ART_EXTRACT_THRESHOLD - 1) {
            version_ops_stats().upgrade_to_art.fetch_add(1, std::memory_order_relaxed);
            this->next->resources->emplace_back(
                GCResourceInfo{Range_Tree_Upgraded, (void*)vertex.neighborhood_ptr});
            auto new_art = ((RangeTree*)vertex.neighborhood_ptr)->range_tree2art(
                src, degree, dest, property, *this->next->resources, trace_block);
            if (new_art == nullptr) { this->next->resources->pop_back(); return; }
            vertex.neighborhood_ptr = (uint64_t)new_art;
            vertex.is_art = true;
            vertex.degree++;
        } else {
            auto vrt = (RangeTree*)vertex.neighborhood_ptr;
            auto new_rt = copy_range_tree(vrt);
            this->next->resources->emplace_back(
                GCResourceInfo{Range_Tree_Copied, (void*)vrt});
            if (!new_rt->insert(src, dest, property, *this->next->resources, trace_block)) {
                delete new_rt;
                this->next->resources->pop_back();
                return;
            }
            vertex.neighborhood_ptr = (uint64_t)new_rt;
            vertex.degree++;
        }
    } else {
        // ─── Level 2: ART storage ───
        version_ops_stats().art_insert.fetch_add(1, std::memory_order_relaxed);
        assert(vertex.is_art);
        auto vertex_art = (ART*)vertex.neighborhood_ptr;
        ARTInsertElemCopyRes res = vertex_art->insert_element_copy(src, dest, property,
                                                                    trace_block);
        if (res.is_new) {
            vertex.degree++;
            vertex.neighborhood_ptr = (uint64_t)res.art_ptr;
            this->next->resources->emplace_back(
                GCResourceInfo{ART_Tree, (void*)vertex_art});
        }
    }
}

// ──────────────── second_insert_edge (post-split re-entry) ────────────────
inline void NeoTreeVersion::second_insert_edge(uint64_t src, RangeElement target,
                                               NeoVertex& vertex, Property_t* property,
                                               WriterTraceBlock* trace_block) {
    NeoRangeNode& node = *find_range_node(src & VERTEX_GROUP_MASK);
    uint64_t node_idx = &node - node_block->data();
    std::vector<uint16_t>* vertices = get_vertices_in_node(node_idx);

    auto arr = (RangeElementSegment_t*)node.arr_ptr;
    uint16_t arr_size = node.size + 1;

    int pos_idx = find_position_to_be_inserted(src, target, vertex, arr, arr_size, vertices);
    if (pos_idx == -1) { delete vertices; return; }

#if EDGE_PROPERTY_NUM != 0
    range_segment_insert(arr, node.property, arr_size, pos_idx, target, property);
#else
    range_segment_insert(arr, nullptr, arr_size, pos_idx, target, nullptr);
#endif
    node.size++;
    if (vertex.degree == 0) {
        vertex.neighbor_offset = pos_idx;
        vertex.neighborhood_ptr = (uint64_t)arr;
        vertex.range_node_idx = node_idx;
    }
    vertex.degree++;
    vertex_map_update_split(arr, vertices->data(), vertices->size(),
                            node_idx, src & VERTEX_GROUP_MASK, 1);
    delete vertices;
}

// ──────────────── extract2range_tree ────────────────
inline RangeTree* NeoTreeVersion::extract2range_tree(uint64_t src, uint64_t degree,
                                                     uint64_t new_element,
                                                     Property_t* new_property,
                                                     NeoRangeNode& range_node,
                                                     WriterTraceBlock* trace_block) {
    auto arr = (RangeElementSegment_t*)range_node.arr_ptr;
    auto vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
    uint64_t arr_size = range_node.size + 1;
    uint64_t node_idx = &range_node - node_block->data();

    std::vector<uint16_t>* vertices = get_vertices_in_node(node_idx);
    int pos_idx = find_position_to_be_inserted(src, new_element, vertex, arr, arr_size, vertices);
    if (pos_idx == -1) { delete vertices; return nullptr; }

    RangeElement* neighbor_begin = (RangeElement*)vertex.neighborhood_ptr + vertex.neighbor_offset;
    pos_idx -= vertex.neighbor_offset;

#if EDGE_PROPERTY_NUM != 0
    Property_t* prop_arr = range_node.property->value.begin() + vertex.neighbor_offset;
    auto new_rt = new RangeTree{neighbor_begin, prop_arr, degree, new_element,
                                new_property, (uint64_t)pos_idx, trace_block};
#else
    auto new_rt = new RangeTree{neighbor_begin, nullptr, degree, new_element,
                                nullptr, (uint64_t)pos_idx, trace_block};
#endif

    // clean old array
    if (arr_size > degree) {
        auto new_arr = trace_block->allocate_range_element_segment();
        range_node.arr_ptr = (uint64_t)new_arr;
        range_node.size = arr_size - degree - 1;
        std::copy(arr->value.begin(), neighbor_begin, new_arr->value.begin());
        std::copy(neighbor_begin + degree, arr->value.begin() + arr_size,
                  new_arr->value.begin() + vertex.neighbor_offset);
#if EDGE_PROPERTY_NUM != 0
        if (prop_arr) {
            auto new_pa = trace_block->allocate_range_prop_vec();
            range_property_map_copy(prop_arr, 0, vertex.neighbor_offset, new_pa, 0);
            range_property_map_copy(prop_arr, vertex.neighbor_offset + degree, arr_size,
                                    new_pa, vertex.neighbor_offset);
            range_node.property = new_pa;
        }
#endif
        vertex_map_update_split(new_arr, vertices->data(), vertices->size(),
                                node_idx, src & VERTEX_GROUP_MASK, -(int)vertex.degree);
        if (range_node.key != 0 && vertex.neighbor_offset == 0) {
            assert(vertices->at(0) == (src & VERTEX_GROUP_MASK) && vertices->size() > 1);
            range_node.key = vertices->at(1);
        }
    } else {
        remove_node(range_node, node_idx, vertices->at(vertices->size() - 1) + 1);
    }
    delete vertices;
    return new_rt;
}

// ──────────────── direct2art ────────────────
inline ART* NeoTreeVersion::direct2art(uint64_t src, uint64_t degree,
                                       uint64_t new_element, Property_t* new_property,
                                       NeoRangeNode& range_node,
                                       WriterTraceBlock* trace_block) {
    version_ops_stats().upgrade_to_art.fetch_add(1, std::memory_order_relaxed);
    auto arr = (RangeElementSegment_t*)range_node.arr_ptr;
    auto vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
    RangeElement* neighbor_begin = (RangeElement*)vertex.neighborhood_ptr + vertex.neighbor_offset;

    std::vector<RangeElement> extract_list;
    extract_list.reserve(degree + 1);
    for (uint64_t i = 0; i < degree; i++)
        extract_list.push_back(neighbor_begin[i]);
    auto new_element_pos = std::lower_bound(extract_list.begin(), extract_list.end(), new_element);
    if (*new_element_pos == new_element) return nullptr;
    extract_list.insert(new_element_pos, new_element);

#if EDGE_PROPERTY_NUM != 0
    Property_t* prop_arr = range_node.property->value.begin() + vertex.neighbor_offset;
    std::vector<Property_t*> extract_prop_list;
    extract_prop_list.reserve(degree + 1);
    for (uint64_t i = 0; i < degree; i++)
        extract_prop_list.push_back(prop_arr ? map_get_all_range_property(prop_arr, i) : nullptr);
    extract_prop_list.insert(
        extract_prop_list.begin() + std::distance(extract_list.begin(), new_element_pos),
        new_property);
#endif

    ART* new_art = new ART();
    delete new_art->root;
    assert(extract_list.size() == degree + 1);
#if EDGE_PROPERTY_NUM != 0
    batch_subtree_build<false>(&new_art->root, 0, extract_list.data(),
                               extract_prop_list.data(), degree + 1, trace_block);
#else
    batch_subtree_build<false>(&new_art->root, 0, extract_list.data(), nullptr,
                               degree + 1, trace_block);
#endif
    return new_art;
}

// ──────────────── remove_edge (3-way dispatch with stats) ────────────────
inline void NeoTreeVersion::remove_edge(uint64_t src, uint64_t dest,
                                        WriterTraceBlock* trace_block) {
    NeoVertex& vertex = vertex_map->at(src & VERTEX_GROUP_MASK);
    assert(vertex.exist);
    neo_tree_stats().remove_edge_count.fetch_add(1, std::memory_order_relaxed);

    switch (vertex.is_independent + vertex.is_art) {
        case 0: { // Outer Range
            NeoRangeNode& node = *find_range_node(src & VERTEX_GROUP_MASK);
            uint64_t node_idx = &node - node_block->data();
            if ((RangeElementSegment_t*)node.arr_ptr == nullptr) return;
            auto arr = (RangeElementSegment_t*)vertex.neighborhood_ptr;
            auto neighbor = (RangeElement*)vertex.neighborhood_ptr + vertex.neighbor_offset;
            uint64_t arr_size = node.size + 1;
            auto pos = std::lower_bound(neighbor, neighbor + vertex.degree, dest);
            if (pos == neighbor + vertex.degree || *pos != dest) return;
            if (arr_size != 1) {
                auto vertices = get_vertices_in_node(node_idx);
                auto new_arr = trace_block->allocate_range_element_segment();
                auto pos_idx = std::distance(arr->value.begin(), pos);
#if EDGE_PROPERTY_NUM != 0
                auto new_pm = trace_block->allocate_range_prop_vec();
                range_segment_remove(arr, node.property, arr_size, new_arr, new_pm, pos_idx);
                force_pointer_set(&(node.property), new_pm);
#else
                range_segment_remove(arr, nullptr, arr_size, new_arr, nullptr, pos_idx);
#endif
                node.size -= 1;
                node.arr_ptr = (uint64_t)new_arr;
                vertex_map_update_split(new_arr, vertices->data(), vertices->size(),
                                        node_idx, src & VERTEX_GROUP_MASK, -1);
                delete vertices;
            } else {
                remove_node(node, node_idx, (src & VERTEX_GROUP_MASK) + 1);
            }
            vertex.degree -= 1;
            if (vertex.degree == 0) {
                vertex.neighborhood_ptr = 0;
                vertex.neighbor_offset = 0;
                vertex.range_node_idx = 0;
            }
            this->next->resources->emplace_back(
                GCResourceInfo{Outer_Segment, (void*)next->node_block->at(node_idx).arr_ptr});
            break;
        }
        case 1: { // Inner Range
            auto vrt = (RangeTree*)vertex.neighborhood_ptr;
            auto new_rt = copy_range_tree(vrt);
            this->next->resources->emplace_back(GCResourceInfo{Range_Tree_Copied, (void*)vrt});
            if (!new_rt->remove(dest, *this->next->resources, trace_block)) {
                delete new_rt;
                this->next->resources->pop_back();
                return;
            }
            vertex.neighborhood_ptr = (uint64_t)new_rt;
            vertex.degree--;
            break;
        }
        case 2: { // ART
            auto vertex_art = (ART*)vertex.neighborhood_ptr;
            this->next->resources->emplace_back(GCResourceInfo{ART_Tree, (void*)vertex_art});
            ARTRemoveElemCopyRes res = vertex_art->remove_element_copy(src, dest, trace_block);
            if (res.found) {
                vertex.degree--;
                vertex.neighborhood_ptr = (uint64_t)res.tree_ptr;
            } else {
                this->next->resources->pop_back();
            }
            break;
        }
    }
}

// ──────────────── remove_vertex ────────────────
inline void NeoTreeVersion::remove_vertex(uint64_t vtx, bool is_directed,
                                          WriterTraceBlock* trace_block) {
    NeoVertex& entry = vertex_map->at(vtx);
    neo_tree_stats().remove_vertex_count.fetch_add(1, std::memory_order_relaxed);
    if (!is_directed) {
        edges(vtx, [&](uint64_t dest, double) {
            remove_edge(dest, vtx, trace_block);
            return 0;
        });
    }
    switch (entry.is_independent + entry.is_art) {
        case 0:
            edges(vtx, [&](uint64_t dest, double) {
                remove_edge(vtx, dest, trace_block);
                return 0;
            });
            break;
        case 1: {
            auto vrt = (RangeTree*)entry.neighborhood_ptr;
            this->next->resources->emplace_back(
                GCResourceInfo{Range_Tree_Upgraded, (void*)vrt});
            for (size_t i = 0; i < vrt->node_block.size(); i++)
                resources->emplace_back(
                    GCResourceInfo{Inner_Segment, (void*)vrt->node_block.at(i).arr_ptr});
            break;
        }
        case 2:
            this->next->resources->emplace_back(
                GCResourceInfo{ART_Tree, (void*)entry.neighborhood_ptr});
            break;
    }
    entry.degree = 0;
    entry.exist = false;
    entry.neighborhood_ptr = 0;
    entry.neighbor_offset = 0;
}

// ──────────────── gc_copied (with type histogram) ────────────────
inline void NeoTreeVersion::gc_copied(WriterTraceBlock* trace_block) {
    for (size_t idx = 0; idx < resources->size(); idx++) {
        auto res = resources->at(idx);
        switch (res.type) {
            case Outer_Segment:
                gc_histogram().outer_seg.fetch_add(1, std::memory_order_relaxed);
                trace_block->deallocate_range_segment((RangeElementSegment_t*)res.ptr);
                break;
            case Inner_Segment:
                gc_histogram().inner_seg.fetch_add(1, std::memory_order_relaxed);
                trace_block->deallocate_range_segment((RangeElementSegment_t*)res.ptr);
                break;
            case Range_Tree_Copied:
                gc_histogram().range_tree_copied.fetch_add(1, std::memory_order_relaxed);
                delete (RangeTree*)res.ptr;
                break;
            case Range_Tree_Upgraded:
                gc_histogram().range_tree_upgraded.fetch_add(1, std::memory_order_relaxed);
                for (size_t i = 0; i < ((RangeTree*)res.ptr)->node_block.size(); i++) {
                    auto node = ((RangeTree*)res.ptr)->node_block.at(i);
                    trace_block->deallocate_range_segment((RangeElementSegment_t*)node.arr_ptr);
                }
                delete (RangeTree*)res.ptr;
                break;
            case ART_Tree:
                gc_histogram().art_tree.fetch_add(1, std::memory_order_relaxed);
                ((ART*)res.ptr)->handle_resources_copied(trace_block);
                delete (ART*)res.ptr;
                break;
#if VERTEX_PROPERTY_NUM != 0
            case Vertex_Property_Vec:
                gc_histogram().vertex_prop.fetch_add(1, std::memory_order_relaxed);
                deallocate_vertex_property_vec((VertexPropertyVec_t*)res.ptr);
                break;
            case Vertex_Property_Map_All_Modified:
                gc_histogram().vertex_prop.fetch_add(1, std::memory_order_relaxed);
                gc_vertex_property_map_copied(res.ptr);
                break;
            case Multi_Vertex_Property_Vec_Copied:
                delete (MultiVertexPropertyVec_t*)res.ptr;
                break;
            case Multi_Vertex_Property_Vec_Mounted:
                break;
#endif
#if EDGE_PROPERTY_NUM != 0
            case Range_Property_Vec:
                gc_histogram().edge_prop.fetch_add(1, std::memory_order_relaxed);
                trace_block->deallocate_range_prop_vec((RangePropertyVec_t*)res.ptr);
                break;
            case Range_Property_Map_All_Modified:
                gc_histogram().edge_prop.fetch_add(1, std::memory_order_relaxed);
                trace_block->deallocate_range_prop_vec((RangePropertyVec_t*)res.ptr);
                break;
#endif
            default:
                std::cerr << "Invalid GC resource type" << std::endl;
                abort();
        }
    }
    delete resources;
    resources = nullptr;
}

// ──────────────── handle_resources_ref ────────────────
inline void NeoTreeVersion::handle_resources_ref() {
    if (!resource_handled) {
        resource_handled = true;
#if VERTEX_PROPERTY_NUM >= 1
        vertex_property_map->ref_cnt += 1;
#endif
        if (!node_block->empty() && node_block->at(0).arr_ptr != 0) {
            for (auto& node : *node_block) {
                auto arr = (RangeElementSegment_t*)node.arr_ptr;
                arr->ref_cnt += 1;
#if EDGE_PROPERTY_NUM >= 1
                if (node.property) node.property->ref_cnt += 1;
#endif
            }
        }
        auto add_ref = [&](uint16_t idx) {
            if (vertex_map->at(idx).is_art)
                ((ART*)vertex_map->at(idx).neighborhood_ptr)->ref_cnt += 1;
            else
                ((RangeTree*)vertex_map->at(idx).neighborhood_ptr)->ref_cnt += 1;
        };
        independent_map.for_each(add_ref);
    }

    for (GCResourceInfo& res : *resources) {
        switch (res.type) {
            case Outer_Segment:
            case Inner_Segment:
                ((RangeElementSegment_t*)res.ptr)->ref_cnt -= 1;
                break;
            case Range_Tree_Copied: {
                auto rt = (RangeTree*)res.ptr;
                rt->ref_cnt -= 1;
                for (size_t i = 0; i < rt->node_block.size(); i++) {
                    ((RangeElementSegment_t*)rt->node_block.at(i).arr_ptr)->ref_cnt += 1;
                }
                break;
            }
            case Range_Tree_Upgraded:
                ((RangeTree*)res.ptr)->ref_cnt -= 1;
                break;
            case ART_Tree:
                ((ART*)res.ptr)->handle_resources_ref();
                ((ART*)res.ptr)->ref_cnt -= 1;
                break;
#if VERTEX_PROPERTY_NUM != 0
            case Vertex_Property_Vec:
                ((VertexPropertyVec_t*)res.ptr)->ref_cnt -= 1; break;
            case Vertex_Property_Map_All_Modified:
                ((VertexPropertyVec_t*)res.ptr)->ref_cnt -= 1; break;
            case Multi_Vertex_Property_Vec_Copied:
            case Multi_Vertex_Property_Vec_Mounted:
                break;
#endif
#if EDGE_PROPERTY_NUM != 0
            case Range_Property_Vec:
            case Range_Property_Map_All_Modified:
                ((RangePropertyVec_t*)res.ptr)->ref_cnt -= 1; break;
#endif
            default: abort();
        }
    }
    resources->clear();
}

// ──────────────── gc_ref ────────────────
inline void NeoTreeVersion::gc_ref(WriterTraceBlock* trace_block) {
#if VERTEX_PROPERTY_NUM != 0
    gc_vertex_property_map_ref(vertex_property_map);
#endif
    if (node_block->at(0).arr_ptr != 0) {
        for (auto& node : *node_block) {
            auto arr = (RangeElementSegment_t*)node.arr_ptr;
            if (arr->ref_cnt.fetch_sub(1, std::memory_order_release) == 1) {
                trace_block->deallocate_range_segment(arr);
#if EDGE_PROPERTY_NUM != 0
                trace_block->deallocate_range_prop_vec(node.property);
#endif
            }
        }
    }
    auto dec_ref = [&](uint16_t idx) {
        if (vertex_map->at(idx).is_art) {
            auto art = (ART*)vertex_map->at(idx).neighborhood_ptr;
            if (art->ref_cnt.fetch_sub(1, std::memory_order_release) == 1) {
                art->gc_ref(trace_block);
                delete art;
            }
        } else {
            auto rt = (RangeTree*)vertex_map->at(idx).neighborhood_ptr;
            if (rt->ref_cnt.fetch_sub(1, std::memory_order_release) == 1) {
                for (size_t i = 0; i < rt->node_block.size(); i++) {
                    auto a = (RangeElementSegment_t*)rt->node_block.at(i).arr_ptr;
                    if (a->ref_cnt.fetch_sub(1, std::memory_order_release) == 1)
                        trace_block->deallocate_range_segment(a);
                }
                delete rt;
            }
        }
    };
    independent_map.for_each(dec_ref);
}

// ──────────────── destroy ────────────────
inline void NeoTreeVersion::destroy() {
#if VERTEX_PROPERTY_NUM != 0
    destroy_vertex_property_map(vertex_property_map);
#endif
    for (auto& it : *vertex_map) {
        if (it.is_independent) {
            if (!it.is_art) {
                auto tree = (RangeTree*)it.neighborhood_ptr;
                for (size_t i = 0; i < tree->node_block.size(); i++)
                    delete (RangeElementSegment_t*)tree->node_block.at(i).arr_ptr;
                delete tree;
            } else {
                ((ART*)it.neighborhood_ptr)->destroy();
                delete (ART*)it.neighborhood_ptr;
            }
            it.neighborhood_ptr = 0;
        }
    }
    for (auto& node : *node_block) {
        delete (RangeElementSegment_t*)node.arr_ptr;
        node.arr_ptr = 0;
    }
    delete resources; resources = nullptr;
    delete node_block; node_block = nullptr;
    delete vertex_map; vertex_map = nullptr;
}

// ──────────────── Intersect (upstream pair-wise) ────────────────
inline void NeoTreeVersion::intersect(NeoTreeVersion* v1, uint64_t src1,
                                      NeoTreeVersion* v2, uint64_t src2,
                                      std::vector<uint64_t>& result) {
    NeoVertex vtx1 = v1->vertex_map->at(src1 & VERTEX_GROUP_MASK);
    NeoVertex vtx2 = v2->vertex_map->at(src2 & VERTEX_GROUP_MASK);
    uint8_t st1 = vtx1.is_independent + vtx1.is_art;
    uint8_t st2 = vtx2.is_independent + vtx2.is_art;
    if (st1 < st2) std::swap(vtx1, vtx2);
    if (vtx1.degree == 0 || vtx2.degree == 0) return;

    switch (st1) {
        case 0: {
            auto n1 = (RangeElement*)vtx1.neighborhood_ptr + vtx1.neighbor_offset;
            auto n2 = (RangeElement*)vtx2.neighborhood_ptr + vtx2.neighbor_offset;
            uint64_t i1 = 0, i2 = 0;
            while (i1 < vtx1.degree && i2 < vtx2.degree) {
                if (n1[i1] == n2[i2]) { result.push_back(n1[i1]); i1++; i2++; }
                else if (n1[i1] < n2[i2]) i1++;
                else i2++;
            }
            break;
        }
        case 1: {
            auto rt1 = (RangeTree*)vtx1.neighborhood_ptr;
            if (st2 == 0) {
                auto n2 = (RangeElement*)vtx2.neighborhood_ptr + vtx2.neighbor_offset;
                rt1->range_intersect(n2, vtx2.degree, result);
            } else {
                rt1->intersect((RangeTree*)vtx2.neighborhood_ptr, result);
            }
            break;
        }
        case 2: {
            auto a1 = (ART*)vtx1.neighborhood_ptr;
            if (st2 == 0) {
                auto n2 = (RangeElement*)vtx2.neighborhood_ptr + vtx2.neighbor_offset;
                a1->range_intersect(n2, vtx2.degree, result);
            } else if (st2 == 1) {
                auto rt2 = (RangeTree*)vtx2.neighborhood_ptr;
                for (size_t i = 0; i < rt2->node_block.size(); i++)
                    a1->range_intersect(
                        ((RangeElementSegment_t*)rt2->node_block.at(i).arr_ptr)->value.begin(),
                        rt2->node_block.at(i).size, result);
            } else {
                a1->intersect((ART*)vtx2.neighborhood_ptr, result);
            }
            break;
        }
    }
}

inline uint64_t NeoTreeVersion::intersect(NeoTreeVersion* v1, uint64_t src1,
                                          NeoTreeVersion* v2, uint64_t src2) {
    NeoVertex vtx1 = v1->vertex_map->at(src1 & VERTEX_GROUP_MASK);
    NeoVertex vtx2 = v2->vertex_map->at(src2 & VERTEX_GROUP_MASK);
    uint8_t st1 = vtx1.is_independent + vtx1.is_art;
    uint8_t st2 = vtx2.is_independent + vtx2.is_art;
    if (st1 < st2) { std::swap(vtx1, vtx2); std::swap(st1, st2); }
    if (vtx1.degree == 0 || vtx2.degree == 0) return 0;
    uint64_t res = 0;

    switch (st1) {
        case 0: {
            auto n1 = (RangeElement*)vtx1.neighborhood_ptr + vtx1.neighbor_offset;
            auto n2 = (RangeElement*)vtx2.neighborhood_ptr + vtx2.neighbor_offset;
            uint64_t i1 = 0, i2 = 0;
            while (i1 < vtx1.degree && i2 < vtx2.degree) {
                if (n1[i1] == n2[i2]) { res++; i1++; i2++; }
                else if (n1[i1] < n2[i2]) i1++;
                else i2++;
            }
            break;
        }
        case 1: {
            auto rt1 = (RangeTree*)vtx1.neighborhood_ptr;
            if (st2 == 0) {
                res = rt1->range_intersect((RangeElement*)vtx2.neighborhood_ptr + vtx2.neighbor_offset, vtx2.degree);
            } else {
                res = rt1->intersect((RangeTree*)vtx2.neighborhood_ptr);
            }
            break;
        }
        case 2: {
            auto a1 = (ART*)vtx1.neighborhood_ptr;
            if (st2 == 0) {
                res = a1->range_intersect((RangeElement*)vtx2.neighborhood_ptr + vtx2.neighbor_offset, vtx2.degree);
            } else if (st2 == 1) {
                auto rt2 = (RangeTree*)vtx2.neighborhood_ptr;
                for (size_t i = 0; i < rt2->node_block.size(); i++)
                    res += a1->range_intersect(
                        ((RangeElementSegment_t*)rt2->node_block.at(i).arr_ptr)->value.begin(),
                        rt2->node_block.at(i).size);
            } else {
                res = a1->intersect((ART*)vtx2.neighborhood_ptr);
            }
            break;
        }
    }
    return res;
}

// ──────────────── get_vertices_in_node ────────────────
inline std::vector<uint16_t>* NeoTreeVersion::get_vertices_in_node(uint16_t node_idx) {
    auto target = node_block->at(node_idx).arr_ptr;
    uint16_t start_v = node_block->at(node_idx).key;
    uint16_t end_v = (node_idx == node_block->size() - 1)
                     ? VERTEX_GROUP_SIZE : node_block->at(node_idx + 1).key;
    auto vertices = new std::vector<uint16_t>();
    vertices->reserve(end_v - start_v);
    for (int cur = start_v; cur < end_v; cur++)
        if (vertex_map->at(cur).neighborhood_ptr == target)
            vertices->push_back(cur);
    return vertices;
}

inline std::vector<std::pair<uint16_t, uint16_t>>*
NeoTreeVersion::get_vertices_in_node_with_offset(uint16_t node_idx) {
    auto target = node_block->at(node_idx).arr_ptr;
    auto vertices = new std::vector<std::pair<uint16_t, uint16_t>>;
    if (target == 0) return vertices;
    uint16_t start_v = node_block->at(node_idx).key;
    uint16_t end_v = (node_idx == node_block->size() - 1)
                     ? VERTEX_GROUP_SIZE : node_block->at(node_idx + 1).key;
    vertices->reserve(end_v - start_v);
    for (int cur = start_v; cur < end_v; cur++)
        if (vertex_map->at(cur).neighborhood_ptr == target)
            vertices->emplace_back(cur, (uint16_t)vertex_map->at(cur).neighbor_offset);
    return vertices;
}

// ──────────────── Global dump ────────────────
inline void dump_all_version_stats() {
    version_ops_stats().dump();
    gc_histogram().dump();
    neo_tree_stats().dump();
}

} // namespace container
