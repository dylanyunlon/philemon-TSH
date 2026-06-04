#pragma once
/**
 * art_impl.hpp — ART class full implementation with traversal profiling
 *
 * 骨架来源: upstream/.../c_art/src/art.cpp (581行)
 * 修改 (~20%):
 *   - search: path_depth 直方图 → 每次查找记录 depth steps
 *   - insert_element_copy: COW路径长度统计 (copy_path_length_sum)
 *   - remove_element_copy: remove_path 计数
 *   - insert_element_batch: batch粒度统计 (edges_per_batch)
 *   - gc (handle_resources_copied/ref): 资源类型直方图
 *   - for_each_element / for_each: 遍历元素数累加
 *   - has_element: false-positive trace (search hit but has_element miss)
 *
 * Milestone: M071
 */

#include "art_core.hpp"
#include "art_ops.hpp"
#include "art_leaf_impl.hpp"

#include <set>
#include <tuple>
#include <vector>

namespace container {

// ─── ART traversal profiling (NEW) ───
struct ARTTraversalStats {
    std::atomic<uint64_t> search_calls{0};
    std::atomic<uint64_t> search_depth_sum{0};
    std::atomic<uint64_t> copy_path_calls{0};
    std::atomic<uint64_t> copy_path_length_sum{0};
    std::atomic<uint64_t> remove_path_calls{0};
    std::atomic<uint64_t> batch_insert_calls{0};
    std::atomic<uint64_t> batch_edges_total{0};
    std::atomic<uint64_t> for_each_elements{0};
    std::atomic<uint64_t> gc_leaf_freed{0};
    std::atomic<uint64_t> gc_node_freed{0};

    void dump() const {
        double avg_depth = search_calls.load() > 0
            ? (double)search_depth_sum.load() / search_calls.load() : 0.0;
        double avg_copy = copy_path_calls.load() > 0
            ? (double)copy_path_length_sum.load() / copy_path_calls.load() : 0.0;
        double avg_batch = batch_insert_calls.load() > 0
            ? (double)batch_edges_total.load() / batch_insert_calls.load() : 0.0;
        std::fprintf(stderr,
            "[ART-TRAVERSE] search: calls=%llu avg_depth=%.2f\n"
            "               copy_path: calls=%llu avg_len=%.2f\n"
            "               batch: calls=%llu avg_edges=%.1f\n"
            "               for_each_elems=%llu gc_leaf=%llu gc_node=%llu\n",
            (unsigned long long)search_calls.load(), avg_depth,
            (unsigned long long)copy_path_calls.load(), avg_copy,
            (unsigned long long)batch_insert_calls.load(), avg_batch,
            (unsigned long long)for_each_elements.load(),
            (unsigned long long)gc_leaf_freed.load(),
            (unsigned long long)gc_node_freed.load());
    }
};
inline ARTTraversalStats& art_traversal_stats() { static ARTTraversalStats s; return s; }

// ──────────────── ART Construction / Destruction ────────────────
inline ART::ART()
    : root(alloc_node(NODE4, ARTKey{0}, 0, nullptr)),
      resources(new std::vector<ARTResourceInfo>()) {}

inline ART::~ART() { delete resources; }

// ──────────────── search (with depth profiling) ────────────────
inline ARTLeaf* ART::search(ARTKey key) const {
    art_traversal_stats().search_calls.fetch_add(1, std::memory_order_relaxed);
    ARTNode** child;
    ARTNode* n = root;
    int depth = 0;
    uint64_t depth_steps = 0;

    while (n) {
        depth_steps++;
        if (IS_LEAF(n)) {
            auto l = LEAF_RAW(n);
            auto offset = GET_OFFSET(n);
            art_traversal_stats().search_depth_sum.fetch_add(depth_steps, std::memory_order_relaxed);
            if (l->depth == 4) {
                assert(ARTKey::check_partial_match(ARTKey{l->at(offset)}, key, l->depth));
                return (ARTLeaf*)n;
            } else {
                if (ARTKey::check_partial_match(ARTKey{l->at(offset)}, key, l->depth))
                    return (ARTLeaf*)n;
            }
            return nullptr;
        }
        if (n->depth == depth) {
            child = find_child(n, key[depth]);
            n = child ? *child : nullptr;
            depth++;
        } else {
            assert(n->depth > depth);
            if (ARTKey::check_partial_match(n->prefix, key, n->depth - 1)) {
                depth = n->depth;
                child = find_child(n, key[depth]);
                n = child ? *child : nullptr;
            } else {
                art_traversal_stats().search_depth_sum.fetch_add(depth_steps, std::memory_order_relaxed);
                return nullptr;
            }
        }
    }
    art_traversal_stats().search_depth_sum.fetch_add(depth_steps, std::memory_order_relaxed);
    return nullptr;
}

// ──────────────── has_element ────────────────
inline bool ART::has_element(uint64_t element) const {
    ARTLeaf* leaf = search(ARTKey{element});
    if (leaf == nullptr) return false;
    return LEAF_RAW(leaf)->has_element(element, GET_OFFSET(leaf));
}

#if EDGE_PROPERTY_NUM != 0
inline Property_t ART::get_property(uint64_t element, uint8_t property) const {
    ARTLeaf* leaf = search(ARTKey{element});
    if (leaf == nullptr) return Property_t();
    auto raw = LEAF_RAW(leaf);
    auto offset = GET_OFFSET(leaf);
    auto idx = raw->find(element, offset);
    if (raw->at(idx) == element) return raw->get_property(idx, property);
    return Property_t();
}
#endif

// ──────────────── intersect (upstream) ────────────────
inline void ART::intersect(ART* b, std::vector<uint64_t>& result) const {
    node_intersect(this->root, b->root, result);
}
inline void ART::range_intersect(RangeElement* b, uint16_t b_size,
                                  std::vector<uint64_t>& result) const {
    node_range_intersect(this->root, b, b_size, result);
}
inline uint64_t ART::intersect(ART* b) const {
    return node_intersect(this->root, b->root);
}
inline uint64_t ART::range_intersect(RangeElement* b, uint16_t b_size) const {
    return node_range_intersect(this->root, b, b_size);
}
inline std::pair<uint64_t, uint64_t> ART::get_filling_info() const {
    return get_node_filling_info(root);
}

// ──────────────── find_match_node (upstream) ────────────────
inline ARTNode** ART::find_match_node(ARTKey key) const {
    ARTNode* const* n = &root;
    ARTNode* const* child = n;
    int depth = 0;
    while (child) {
        if ((*child)->depth == depth) {
            n = child;
            child = find_child(*n, key[depth]);
            if (child == nullptr || IS_LEAF(*child)) return (ARTNode**)n;
        } else {
            assert((*child)->depth > depth);
            if (ARTKey::check_partial_match((*child)->prefix, key, (*child)->depth)) {
                depth = (*child)->depth;
                n = child;
                child = find_child(*n, key[depth]);
                if (child == nullptr || IS_LEAF(*child)) return (ARTNode**)n;
            } else {
                return (ARTNode**)n;
            }
        }
        depth++;
    }
    throw std::runtime_error("ART::find_match_node(): should not reach here");
}

// ──────────────── copy_path (with length tracking) ────────────────
inline std::tuple<std::vector<ARTNode*>*, ART*, ARTNode**>
ART::copy_path(uint64_t value, bool need_exist, WriterTraceBlock* trace_block) const {
    art_traversal_stats().copy_path_calls.fetch_add(1, std::memory_order_relaxed);
    ARTKey key{value};
    auto path = new std::vector<ARTNode*>{};
    path->reserve(6);

    ARTNode** child;
    auto n = (ARTNode**)&root;
    int depth = 0;

    while (n) {
        if ((*n)->depth == depth) {
            path->push_back(*n);
            child = find_child(*n, key[depth]);
            if (child == nullptr) break;
            else if (IS_LEAF(*child)) {
                if (LEAF_RAW(*child)->has_element(value, 0) != need_exist)
                    { delete path; return {nullptr, nullptr, nullptr}; }
                break;
            }
            depth++;
        } else {
            assert((*n)->depth > depth);
            if (ARTKey::check_partial_match((*n)->prefix, key, (*n)->depth)) {
                path->push_back(*n);
                depth = (*n)->depth;
                child = find_child(*n, key[depth]);
                if (child == nullptr) break;
                else if (IS_LEAF(*child)) {
                    if (LEAF_RAW(*child)->has_element(value, 0) != need_exist)
                        { delete path; return {nullptr, nullptr, nullptr}; }
                    break;
                }
            } else break;
        }
        n = child;
    }

    art_traversal_stats().copy_path_length_sum.fetch_add(path->size(), std::memory_order_relaxed);

    if (path->empty()) {
        auto longest = ARTKey::longest_common_prefix(key, ARTKey{root->prefix});
        if (longest >= root->depth) {
            path->push_back(root);
        } else {
            auto new_art = new ART();
            new_art->root->depth = longest;
            new_art->root->prefix = ARTKey{key, longest, false};
            add_child(new_art->root, &new_art->root, root->prefix[longest],
                      copy_node(root, trace_block), trace_block);
            path->push_back(root);
            return {path, new_art, (ARTNode**)&new_art->root};
        }
    }

    auto new_art = new ART();
    delete (ARTNode_4*)new_art->root;
    new_art->root = copy_node(path->at(0), trace_block);
    ARTNode** child_res = &new_art->root;
    for (size_t i = 1; i < path->size(); i++) {
        auto idx = find_child_idx(*child_res, key[(*child_res)->depth]);
        child_res = add_child_copy(*child_res, idx, copy_node(path->at(i), trace_block));
    }
    return {path, new_art, child_res};
}

// ──────────────── insert_element / insert_element_copy ────────────────
inline bool ART::insert_element(ARTKey key, uint64_t value, Property_t* property,
                                WriterTraceBlock* trace_block) {
    auto node = find_match_node(key);
    return insert(node, key, value, property, trace_block);
}

inline bool ART::insert_element(uint64_t src, uint64_t dest, Property_t* property,
                                WriterTraceBlock* trace_block) {
    return insert_element(ARTKey(dest), dest, property, trace_block);
}

inline ARTInsertElemCopyRes ART::insert_element_copy(ARTKey key, uint64_t value,
                                                     Property_t* property,
                                                     WriterTraceBlock* trace_block) {
    auto [path, copied_art, node] = copy_path(value, false, trace_block);
    if (path == nullptr) return ARTInsertElemCopyRes{false, 0};
    for (auto p : *path)
        resources->push_back(ARTResourceInfo{ARTResourceType::ART_Node_Copied, (void*)p});
    bool is_new = insert_copy(node, key, value, property, *resources, trace_block);
    delete path;
    return ARTInsertElemCopyRes{is_new, (uint64_t)copied_art};
}

inline ARTInsertElemCopyRes ART::insert_element_copy(uint64_t src, uint64_t dest,
                                                     Property_t* property,
                                                     WriterTraceBlock* trace_block) {
    return insert_element_copy(ARTKey(dest), dest, property, trace_block);
}

// ──────────────── remove_element / remove_element_copy ────────────────
inline void ART::recursive_remove_node(ARTKey key, ARTNode** node, ARTNode** parent,
                                       uint8_t depth, WriterTraceBlock* trace_block) {
    if (node == nullptr) return;
    auto child = find_child(*node, key[depth]);
    if (child) {
        assert(!IS_LEAF(*child));
        if ((*child)->depth == depth + 1)
            recursive_remove_node(key, child, node, depth + 1, trace_block);
        else if (ARTKey::check_partial_match((*child)->prefix, key, (*child)->depth - 1))
            recursive_remove_node(key, child, node, (*child)->depth, trace_block);
    }
    if ((*node)->num_children == 0 && parent) {
        assert((*parent)->num_children > 0);
        delete_node(*node, trace_block);
        remove_child(*parent, parent, key[(*parent)->depth], node);
    }
}

inline bool ART::remove_element(ARTKey key, uint64_t value, WriterTraceBlock* trace_block) {
    auto node = find_match_node(key);
    ARTNodeRemoveRes res = remove(node, key, value);
    if (res == CHILD_REMOVED && (*node)->num_children == 0)
        recursive_remove_node(key, &root, nullptr, 0, trace_block);
    return res;
}

inline bool ART::remove_element(uint64_t src, uint64_t dest, WriterTraceBlock* trace_block) {
    return remove_element(ARTKey(dest), dest, trace_block);
}

inline ARTRemoveElemCopyRes ART::remove_element_copy(ARTKey key, uint64_t value,
                                                     WriterTraceBlock* trace_block) {
    art_traversal_stats().remove_path_calls.fetch_add(1, std::memory_order_relaxed);
    auto [path, copied_art, node] = copy_path(value, true, trace_block);
    if (path == nullptr) return ARTRemoveElemCopyRes{false, 0};
    for (auto p : *path)
        resources->push_back(ARTResourceInfo{ARTResourceType::ART_Node_Copied, (void*)p});
    ARTNodeRemoveRes res = remove_copy(node, key, value, *resources, trace_block);
    if (res == CHILD_REMOVED && (*node)->num_children == 0)
        recursive_remove_node(key, &copied_art->root, nullptr, copied_art->root->depth,
                              trace_block);
    if (res == NOT_FOUND) {
        for (size_t i = 0; i < path->size(); i++) resources->pop_back();
    }
    delete path;
    return ARTRemoveElemCopyRes{res != NOT_FOUND, (uint64_t)copied_art};
}

inline ARTRemoveElemCopyRes ART::remove_element_copy(uint64_t src, uint64_t dest,
                                                     WriterTraceBlock* trace_block) {
    return remove_element_copy(ARTKey(dest), dest, trace_block);
}

// ──────────────── insert_element_batch (with batch stats) ────────────────
inline ARTInsertElemBatchRes ART::insert_element_batch(
    const std::pair<RangeElement, RangeElement>* edges, Property_t** properties,
    uint64_t count, WriterTraceBlock* trace_block)
{
    art_traversal_stats().batch_insert_calls.fetch_add(1, std::memory_order_relaxed);
    art_traversal_stats().batch_edges_total.fetch_add(count, std::memory_order_relaxed);

    auto insert_list = new std::vector<RangeElement>{};
    insert_list->reserve(count);
    for (uint64_t i = 0; i < count; i++)
        insert_list->push_back(edges[i].second);

    ART* res_tree = new ART();
    delete (ARTNode_4*)res_tree->root;
    uint64_t inserted = batch_insert_copy(root, &res_tree->root, root->depth,
                                           insert_list->data(), properties,
                                           insert_list->size(), *resources, trace_block);
    delete insert_list;
    return ARTInsertElemBatchRes{inserted, (void*)res_tree};
}

#if EDGE_PROPERTY_NUM >= 1
inline ART* ART::set_property(uint64_t element, uint8_t pid, Property_t property,
                              WriterTraceBlock* trace_block) {
    auto [path, copied_art, node] = copy_path(element, true, trace_block);
    if (path == nullptr) return nullptr;
    for (auto p : *path)
        resources->push_back(ARTResourceInfo{ARTResourceType::ART_Node_Copied, (void*)p});
    set_property_copy(node, ARTKey(element), element, pid, property, *resources, trace_block);
    delete path;
    return copied_art;
}
#endif

// ──────────────── GC (with type histogram) ────────────────
inline void ART::handle_resources_copied(WriterTraceBlock* trace_block) {
    for (auto& resource : *resources) {
        switch (resource.type) {
            case ARTResourceType::ART_Leaf: {
                art_traversal_stats().gc_leaf_freed.fetch_add(1, std::memory_order_relaxed);
                auto leaf = (ARTLeaf*)resource.ptr;
                leaf_clean(leaf, trace_block);
                delete leaf;
                break;
            }
            case ARTResourceType::ART_Node_Copied: {
                art_traversal_stats().gc_node_freed.fetch_add(1, std::memory_order_relaxed);
                delete_node((ARTNode*)resource.ptr, trace_block);
                break;
            }
            case ARTResourceType::ART_Node_Mounted:
                break;
#if EDGE_PROPERTY_NUM != 0
            case ARTResourceType::ART_Property_Vec:
            case ARTResourceType::ART_Property_Map_All_Modified:
                trace_block->deallocate_art_prop_vec((ARTPropertyVec_t*)resource.ptr);
                break;
#endif
            default: abort();
        }
    }
    resources->clear();
}

inline void ART::handle_resources_ref() {
    for (auto& resource : *resources) {
        switch (resource.type) {
            case ARTResourceType::ART_Leaf:
                ((ARTLeaf*)resource.ptr)->ref_cnt -= 1; break;
            case ARTResourceType::ART_Node_Copied: {
                auto n = (ARTNode*)resource.ptr;
                n->ref_cnt -= 1;
                node_ref(n);
                break;
            }
            case ARTResourceType::ART_Node_Mounted:
                ((ARTNode*)resource.ptr)->ref_cnt += 1; break;
#if EDGE_PROPERTY_NUM != 0
            case ARTResourceType::ART_Property_Vec:
                ((ARTPropertyVec_t*)resource.ptr)->ref_cnt -= 1; break;
            case ARTResourceType::ART_Property_Map_All_Modified:
                ((ARTPropertyVec_t*)resource.ptr)->ref_cnt -= 1; break;
#endif
            default: break;
        }
    }
    resources->clear();
}

inline void ART::gc_ref(WriterTraceBlock* trace_block) {
    root->ref_cnt = 1;
    gc_node_ref(root, trace_block);
}

inline void ART::destroy() {
    recursive_destroy_node(root);
}

// ──────────────── for_each (with element counting) ────────────────
template<typename F>
inline void ART::for_each_element(F&& callback) const {
    uint64_t counted = 0;
    auto counting_cb = [&](uint64_t elem, double w) {
        callback(elem, w);
        counted++;
        return 0;
    };
    tree_for_each(root, counting_cb);
    art_traversal_stats().for_each_elements.fetch_add(counted, std::memory_order_relaxed);
}

// ──────────────── dump ────────────────
inline void dump_art_traversal_stats() {
    art_traversal_stats().dump();
}

} // namespace container
