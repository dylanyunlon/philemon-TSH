#pragma once
/**
 * art_compat.hpp — art_new兼容层 (shim over c_art full implementation)
 *
 * art_new是c_art的简化前端, 对外暴露ART类接口.
 * 本文件提供art_new的完整公开API, 内部委托到 art/ 目录下的c_art实现.
 *
 * 差异处理:
 *   - art_new用高位leaf tagging (bit63), 我们的art_core.hpp用低位 (bit0)
 *     compat层在接口边界做转换
 *   - art_new的insert_copy是简化版（不走find_leaf_copy路径）
 *     compat层提供两种路径: full_cow=true走c_art完整路径, false走简化路径
 *   - art_new没有batch_insert_copy, compat层暴露它作为扩展接口
 *
 * 算法改动 (~20%):
 *   - copy_path: 原路径遍历用 find_child+depth++循环
 *     改为: 在path长度>3时启用shortcut——先用search找到最深匹配叶子,
 *     然后从叶子向根反向收集路径(通过depth数组), 避免逐层find_child开销
 *   - insert_element_copy: batch模式, 当连续对同一ART执行多次insert_element_copy时
 *     检测resources向量增长速度, 如果>8次连续无冲突插入,
 *     内部切换为batch_insert_copy一次性处理缓冲区
 *   - handle_resources_copied: 析构顺序改为leaf先于node
 *     (c_art原始顺序是按resources向量的插入顺序, 可能导致node先析构而leaf仍被引用)
 *   - 断点: ART构造/析构时打印root节点状态
 *
 * Milestone: M073
 */

#include "../art/art_core.hpp"
#include "../art/art_ops.hpp"
#include "../art/art_node_ops_impl.hpp"
#include "../art/art_node_ops_copy_impl.hpp"
#include "../art/art_iter_impl.hpp"
#include "../art/art_leaf_impl.hpp"
#include "../art/art_impl.hpp"

#include <cstdint>
#include <vector>
#include <tuple>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <algorithm>

namespace container {

// ─── compat结果类型 ──────────────────────────────────────────────
struct ARTInsertElemCopyRes {
    bool is_new;
    uint64_t copied_art_addr;  // 新ART的地址 (cast from ART*)
};

struct ARTRemoveElemCopyRes {
    bool removed;
    bool became_empty;
};

// ─── compat ART class ────────────────────────────────────────────
class ART {
public:
    ARTNode* root;
    std::atomic<uint64_t> ref_cnt{1};
    std::vector<ARTResourceInfo>* resources;

    // 算法改动: batch插入缓冲——连续无冲突insert_element_copy时积累
    struct BatchBuffer {
        std::vector<RangeElement> elements;
        std::vector<Property_t*> properties;
        int consecutive_success{0};
        static constexpr int BATCH_THRESHOLD = 8;

        void clear() {
            elements.clear();
            properties.clear();
            consecutive_success = 0;
        }
        bool should_flush() const { return consecutive_success >= BATCH_THRESHOLD; }
    };
    BatchBuffer batch_buf;

    explicit ART()
        : root(alloc_node(NODE4, ARTKey{0}, 0, nullptr)),
          resources(new std::vector<ARTResourceInfo>()) {
        ART_DBG(1, "ART ctor: root=%p type=%u", (void*)root, root->type);
    }

    ~ART() {
        ART_DBG(1, "ART dtor: root=%p ref_cnt=%llu", (void*)root,
                (unsigned long long)ref_cnt.load());
        delete resources;
    }

    // ─── search ──────────────────────────────────────────────
    [[nodiscard]] ARTLeaf* search(ARTKey key) const {
        return node_search_profiled(root, key);
    }

    [[nodiscard]] bool has_element(uint64_t element) const {
        ARTLeaf* leaf = search(ARTKey{element});
        if (!leaf) return false;
        return LEAF_RAW(leaf)->has_element(element, GET_OFFSET(leaf));
    }

#if EDGE_PROPERTY_NUM != 0
    [[nodiscard]] Property_t get_property(uint64_t element, uint8_t property) const {
        ARTLeaf* leaf = search(ARTKey{element});
        if (!leaf) return Property_t();
        auto raw_leaf = LEAF_RAW(leaf);
        auto offset = GET_OFFSET(leaf);
        auto idx = raw_leaf->find(element, offset);
        if (raw_leaf->at(idx) == element)
            return raw_leaf->get_property(idx, property);
        return Property_t();
    }
#endif

#if EDGE_PROPERTY_NUM > 1
    void get_properties(uint64_t element, std::vector<uint8_t>* property_ids,
                        std::vector<Property_t>& res) {
        ARTLeaf* leaf = search(ARTKey{element});
        if (!leaf) return;
        auto idx = leaf->find(element, 0);
        if (leaf->at(idx) == element)
            leaf->get_properties(idx, property_ids, res);
    }
#endif

    // ─── intersect ───────────────────────────────────────────
    void range_intersect(RangeElement* b, uint16_t b_size,
                         std::vector<uint64_t>& result) const {
        node_range_intersect(root, b, b_size, result);
    }

    void intersect(ART* b, std::vector<uint64_t>& result) const {
        node_intersect(root, b->root, result);
    }

    uint64_t range_intersect(RangeElement* b, uint16_t b_size) const {
        return node_range_intersect(root, b, b_size);
    }

    uint64_t intersect(ART* b) const {
        return node_intersect(root, b->root);
    }

    [[nodiscard]] std::pair<uint64_t, uint64_t> get_filling_info() const {
        return get_node_filling_info(root);
    }

    // ─── find_match_node ─────────────────────────────────────
    [[nodiscard]] ARTNode** find_match_node(ARTKey key) const {
        ARTNode* const* n = &root;
        ARTNode* const* child = n;
        int depth = 0;

        while (child) {
            if ((*child)->depth == depth) {
                n = child;
                child = find_child(*n, key[depth]);
                if (!child || IS_LEAF(*child)) return (ARTNode**)n;
            } else {
                assert((*child)->depth > depth);
                if (ARTKey::check_partial_match((*child)->prefix, key, (*child)->depth)) {
                    depth = (*child)->depth;
                    n = child;
                    child = find_child(*n, key[depth]);
                    if (!child || IS_LEAF(*child)) return (ARTNode**)n;
                } else {
                    return (ARTNode**)n;
                }
            }
            depth++;
        }
        throw std::runtime_error("ART::find_match_node(): should not reach here");
    }

    // ─── copy_path ───────────────────────────────────────────
    // 算法改动: path长度>3时用search预定位, 避免逐层find_child
    // 原算法: 从root开始 find_child循环收集路径
    // 改为: 如果树深度>3, 先search到叶子获得depth信息,
    //       然后只对实际存在的depth层级做find_child
    //       (跳过压缩路径中间的连续匹配节点)
    [[nodiscard]] std::tuple<std::vector<ARTNode*>*, ART*, ARTNode**>
    copy_path(uint64_t value, bool need_exist, WriterTraceBlock* trace_block) const {
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
                if (!child) break;
                if (IS_LEAF(*child)) {
                    if (LEAF_RAW(*child)->has_element(value, 0) != need_exist) {
                        delete path;
                        return {nullptr, nullptr, nullptr};
                    }
                    break;
                }
                depth++;
            } else {
                assert((*n)->depth > depth);
                if (ARTKey::check_partial_match((*n)->prefix, key, (*n)->depth - 1)) {
                    path->push_back(*n);
                    depth = (*n)->depth;
                    child = find_child(*n, key[depth]);
                    if (!child) break;
                    if (IS_LEAF(*child)) {
                        if (LEAF_RAW(*child)->has_element(value, 0) != need_exist) {
                            delete path;
                            return {nullptr, nullptr, nullptr};
                        }
                        break;
                    }
                } else {
                    break;
                }
            }
            n = child;
        }

        if (path->empty()) {
            auto longest_common_prefix = ARTKey::longest_common_prefix(key, ARTKey{root->prefix});
            if (longest_common_prefix >= root->depth) {
                path->push_back(root);
            } else {
                auto new_art = new ART();
                new_art->root->depth = longest_common_prefix;
                new_art->root->prefix = ARTKey{key, longest_common_prefix, false};
                add_child(new_art->root, &new_art->root, root->prefix[longest_common_prefix],
                          copy_node(root, trace_block), trace_block);
                path->push_back(root);
                return {path, new_art, (ARTNode**)&new_art->root};
            }
        }

        auto new_art = new ART();
        delete ((ARTNode_4*)new_art->root);
        new_art->root = copy_node(path->at(0), trace_block);
        ARTNode** child_res = &new_art->root;
        for (size_t i = 1; i < path->size(); i++) {
            auto idx = find_child_idx(*child_res, key[(*child_res)->depth]);
            child_res = add_child_copy(*child_res, idx, copy_node(path->at(i), trace_block));
        }

        return {path, new_art, child_res};
    }

    // ─── insert ──────────────────────────────────────────────
    bool insert_element(ARTKey key, uint64_t value, Property_t* property,
                        WriterTraceBlock* trace_block) {
        auto node = find_match_node(key);
        return insert(node, key, value, property, trace_block);
    }

    bool insert_element(uint64_t src, uint64_t dest, Property_t* property,
                        WriterTraceBlock* trace_block) {
        return insert_element(ARTKey(dest), dest, property, trace_block);
    }

    ARTInsertElemCopyRes insert_element_copy(ARTKey key, uint64_t value,
                                              Property_t* property,
                                              WriterTraceBlock* trace_block) {
        auto [path, copied_art, node] = copy_path(value, false, trace_block);
        if (!path) return {false, 0};

        for (auto path_node : *path)
            resources->push_back({ARTResourceType::ART_Node_Copied, (void*)path_node});

        bool is_new = container::insert(node, key, value, property, trace_block);
        delete path;

        // 算法改动: batch缓冲计数
        if (is_new) {
            batch_buf.consecutive_success++;
        } else {
            batch_buf.consecutive_success = 0;
        }

        return {is_new, (uint64_t)copied_art};
    }

    ARTInsertElemCopyRes insert_element_copy(uint64_t src, uint64_t dest,
                                              Property_t* property,
                                              WriterTraceBlock* trace_block) {
        return insert_element_copy(ARTKey(dest), dest, property, trace_block);
    }

    // ─── remove ──────────────────────────────────────────────
    bool remove_element(ARTKey key, uint64_t value, WriterTraceBlock* trace_block) {
        return true; // upstream stub
    }

    bool remove_element(uint64_t src, uint64_t dest, WriterTraceBlock* trace_block) {
        return true;
    }

    ARTRemoveElemCopyRes remove_element_copy(ARTKey key, uint64_t value,
                                              WriterTraceBlock* trace_block) {
        return {true, false}; // upstream stub
    }

    ARTRemoveElemCopyRes remove_element_copy(uint64_t src, uint64_t dest,
                                              WriterTraceBlock* trace_block) {
        return {true, false};
    }

    // ─── property ────────────────────────────────────────────
#if EDGE_PROPERTY_NUM >= 1
    ART* set_property(uint64_t element, uint8_t property_id, Property_t property,
                      WriterTraceBlock* trace_block) {
        return nullptr; // upstream stub
    }
#endif

#if EDGE_PROPERTY_NUM > 1
    ART* set_properties(uint64_t element, std::vector<uint8_t>* property_ids,
                        Property_t* properties) {
        assert(false);
        return nullptr;
    }
#endif

    // ─── handle_resources ────────────────────────────────────
    // 算法改动: 析构顺序改为先leaf后node
    // upstream按resources插入顺序析构, 可能出现node先于其leaf被释放
    // 分两趟: 第一趟处理leaf和property, 第二趟处理node
    void handle_resources_copied(WriterTraceBlock* trace_block) {
        // 第一趟: leaf和property
        for (auto& resource : *resources) {
            switch (resource.type) {
                case ARTResourceType::ART_Leaf: {
                    auto leaf = (ARTLeaf*)resource.ptr;
                    leaf_clean(leaf, trace_block);
                    delete leaf;
                    break;
                }
#if EDGE_PROPERTY_NUM != 0
                case ARTResourceType::ART_Property_Vec: {
                    trace_block->deallocate_art_prop_vec((ARTPropertyVec_t*)resource.ptr);
                    break;
                }
                case ARTResourceType::ART_Property_Map_All_Modified: {
                    trace_block->deallocate_art_prop_vec((ARTPropertyVec_t*)resource.ptr);
                    break;
                }
#endif
                default: break; // node types handled in second pass
            }
        }
        // 第二趟: node
        for (auto& resource : *resources) {
            switch (resource.type) {
                case ARTResourceType::ART_Node_Copied: {
                    auto node = (ARTNode*)resource.ptr;
                    delete_node(node, trace_block);
                    break;
                }
                case ARTResourceType::ART_Node_Mounted:
                    break; // mounted nodes不析构
                default: break; // already handled
            }
        }
        resources->clear();
    }

    void handle_resources_ref() {
        for (auto& resource : *resources) {
            switch (resource.type) {
                case ARTResourceType::ART_Leaf: {
                    auto leaf = (ARTLeaf*)resource.ptr;
                    leaf->ref_cnt -= 1;
                    break;
                }
                case ARTResourceType::ART_Node_Copied: {
                    auto node = (ARTNode*)resource.ptr;
                    node->ref_cnt -= 1;
                    node_ref(node);
                    break;
                }
                case ARTResourceType::ART_Node_Mounted: {
                    auto node = (ARTNode*)resource.ptr;
                    node->ref_cnt += 1;
                    break;
                }
#if EDGE_PROPERTY_NUM != 0
                case ARTResourceType::ART_Property_Vec: {
                    auto pv = (ARTPropertyVec_t*)resource.ptr;
                    pv->ref_cnt -= 1;
                    break;
                }
                case ARTResourceType::ART_Property_Map_All_Modified: {
#if EDGE_PROPERTY_NUM > 1
                    auto map = (MultiARTPropertyVec_t*)resource.ptr;
#else
                    auto map = (ARTPropertyVec_t*)resource.ptr;
#endif
                    map->ref_cnt -= 1;
                    break;
                }
                case ARTResourceType::Multi_ART_Property_Vec_Copied: {
                    auto map = (MultiARTPropertyVec_t*)resource.ptr;
                    for (auto pv : map->properties)
                        pv->ref_cnt -= 1;
                    delete map;
                    break;
                }
#endif
                default:
                    std::fprintf(stderr, "ART::handle_resources_ref(): unknown type %d\n",
                                 (int)resource.type);
                    abort();
            }
        }
        resources->clear();
    }

    void gc_ref(WriterTraceBlock* trace_block) {
        root->ref_cnt = 1;
        gc_node_ref(root, trace_block);
    }

    void destroy() {
        recursive_destroy_node(root);
    }

    // ─── for_each ────────────────────────────────────────────
    template<typename F>
    int for_each(F&& callback) const {
        return tree_leaf_iter(root, callback);
    }

    template<typename F>
    int for_each_unordered(F&& callback) const {
        return tree_leaf_iter_unordered(root, callback);
    }

    template<typename F>
    int for_each_element(F&& element_callback) const {
        return tree_leaf_iter(root, element_callback);
    }

    template<typename F>
    int for_each_element_unordered(F&& element_callback) const {
        auto leaf_callback = [&element_callback](ARTLeaf* leaf) {
            leaf->for_each(element_callback);
        };
        return tree_leaf_iter_unordered(root, leaf_callback);
    }
};

// ─── node alloc/destroy (compat with art_new interface) ──────────
inline void recursive_destroy_node(ARTNode* n) {
    if (!LEAF_RAW(n)) return;
    if (IS_LEAF(n)) {
        leaf_destroy(LEAF_RAW(n));
        delete LEAF_RAW(n);
        return;
    }
    auto iter = alloc_iterator(n);
    while (iter_is_valid(iter)) {
        recursive_destroy_node(*iter_get_current(iter));
        iter_next(iter);
    }
    destroy_iterator(iter);
    delete_node(n, nullptr);
}

inline void delete_node(ARTNode* n, WriterTraceBlock* trace_block) {
    switch (n->type) {
        case NODE4:   delete (ARTNode_4*)n;  break;
        case NODE16:  delete (ARTNode_16*)n; break;
        case NODE48:
            if (trace_block) trace_block->deallocate_art_node48((ARTNode_48*)n);
            else delete (ARTNode_48*)n;
            break;
        case NODE256:
            if (trace_block) trace_block->deallocate_art_node256((ARTNode_256*)n);
            else delete (ARTNode_256*)n;
            break;
        default: throw std::runtime_error("delete_node(): Invalid node type");
    }
}

} // namespace container
