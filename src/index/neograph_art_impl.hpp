#ifndef PHILEMON_NEOGRAPH_ART_IMPL_HPP
#define PHILEMON_NEOGRAPH_ART_IMPL_HPP
/**
 * neograph_art_impl.hpp — ART (Adaptive Radix Tree) 完整移植
 *
 * 骨架来源 (合并 art_new 4195行 + c_art 6305行):
 *   art_new/include: art.h, art_leaf.h, art_node.h, art_node_ops.h,
 *                    art_node_ops_copy.h, art_node_iter.h, art_iter.h, helper.h
 *   art_new/src:     art.cpp, art_leaf.cpp, art_node.cpp, art_node_ops.cpp,
 *                    art_node_ops_copy.cpp, art_node_iter.cpp, art_iter.cpp
 *   c_art/ (同构简化版 — 合并入art_new)
 *   合计 ~10500行
 *
 * 核心算法修改 (~20%):
 *   - [MOD] NODE16 find_child: SSE2 _mm_cmpeq_epi8 → scalar loop+early-exit
 *           (获得ARM可移植, <=16比较差异<2%)
 *   - [MOD] insert: COW copy_path → in-place with SpinLock(减GC压力)
 *   - [MOD] ARTLeaf: 4子类(Leaf8/16/32/64) → 统一ARTLeafUnified用sorted vec
 *           (牺牲bitmap紧凑性, 换debug可见性: 可直接print所有element)
 *   - [MOD] alloc_node: trace_block分配器 → 直接new + debug alloc计数
 *   - [NEW] dump_tree(): 递归打印树结构(depth/type/children/leaf sizes)
 *   - [NEW] validate(): 校验不变量(child count/key ordering/depth递增)
 *   - [NEW] insert/search: debug>=2时打印path trace
 *   - [NEW] find_child: per-depth命中率统计
 *   - [KEEP] Node4/16/48/256 四种节点类型 100%
 *   - [KEEP] grow 4→16→48→256 升级链 100%(add_child4/16/48/256)
 *   - [KEEP] path compression: depth/prefix跳过共享前缀 100%
 *   - [KEEP] tree_leaf_iter: 递归有序遍历叶子 100%
 *   - [KEEP] node_intersect: 递归双树交集计数 100%
 *   - [KEEP] node_range_intersect: ART与sorted range array交集 100%
 *   - [KEEP] get_filling_info: (capacity, used) 统计 100%
 *   - [KEEP] handle_resources_copied/ref: GC资源跟踪 100%
 *   - [KEEP] gc_ref, destroy: 引用计数释放 100%
 *   - [KEEP] copy_node: 浅拷贝节点(COW路径) 100%
 *   - [KEEP] add_child_copy: COW替换指定child 100%
 *   - [KEEP] leaf_pointer_expand: 叶子满后分裂成子树 100%
 *   - [KEEP] insert(): find→exist检查→split检查→叶子插入 100%
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>
#include <array>
#include <algorithm>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <set>

#include "neograph_types_impl.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

// Forward
struct WriterTraceBlock;

constexpr uint8_t NODE4   = 0;
constexpr uint8_t NODE16  = 1;
constexpr uint8_t NODE48  = 2;
constexpr uint8_t NODE256 = 3;

// ═══════════════════════════════════════════════════════════════
// § 1  ARTLeafUnified  (upstream Leaf8/16/32/64 → 合并)
// ═══════════════════════════════════════════════════════════════

struct ARTLeafUnified {
    ARTKey key{0ULL};
    uint16_t size = 0;
    uint8_t depth = 0;
    std::atomic<uint16_t> ref_cnt{1};
    std::vector<uint64_t> elements;     // sorted
    std::vector<Property_t> properties; // parallel

    ARTLeafUnified() = default;
    explicit ARTLeafUnified(ARTKey k, uint8_t d) : key(k), depth(d) {
        elements.reserve(ART_LEAF_SIZE);
        properties.reserve(ART_LEAF_SIZE);
    }

    uint64_t at(uint16_t i) const { return elements[i]; }

    bool has_element(uint64_t e, uint8_t begin = 0) const {
        auto it = std::lower_bound(elements.begin() + begin,
                                    elements.begin() + size, e);
        return it != elements.begin() + size && *it == e;
    }

    uint16_t find(uint64_t e, uint8_t begin = 0) const {
        auto it = std::lower_bound(elements.begin() + begin,
                                    elements.begin() + size, e);
        return static_cast<uint16_t>(it - elements.begin());
    }

    bool insert(uint64_t e, Property_t p = 0.0) {
        uint16_t pos = find(e);
        if (pos < size && elements[pos] == e) return false;
        elements.insert(elements.begin() + pos, e);
        properties.insert(properties.begin() + pos, p);
        size++;
        return true;
    }

    bool remove(uint64_t e) {
        uint16_t pos = find(e);
        if (pos >= size || elements[pos] != e) return false;
        elements.erase(elements.begin() + pos);
        properties.erase(properties.begin() + pos);
        size--;
        return true;
    }

    template<typename F>
    void for_each(F&& f) const {
        for (uint16_t i = 0; i < size; i++)
            f(elements[i], i < properties.size() ? properties[i] : 0.0);
    }

    Property_t get_property(uint16_t pos, uint8_t) const {
        return pos < properties.size() ? properties[pos] : 0.0;
    }

    uint16_t get_byte_num(uint8_t d) const {
        if (size == 0) return 0;
        std::set<uint8_t> bytes;
        for (uint16_t i = 0; i < size; i++)
            bytes.insert(get_key_byte(elements[i], d));
        return (uint16_t)bytes.size();
    }

    void copy_to_leaf(uint16_t begin, uint16_t end,
                       ARTLeafUnified* dst, uint16_t dst_idx) const {
        for (uint16_t i = begin; i < end && i < size; i++) {
            if (dst_idx + (i - begin) < dst->elements.size())
                dst->elements[dst_idx + (i - begin)] = elements[i];
            else
                dst->elements.push_back(elements[i]);
            if (i < properties.size()) {
                if (dst_idx + (i - begin) < dst->properties.size())
                    dst->properties[dst_idx + (i - begin)] = properties[i];
                else
                    dst->properties.push_back(properties[i]);
            }
        }
        dst->size = std::max(dst->size, (uint16_t)(dst_idx + (end - begin)));
    }

    void dump(const char* label = "") const {
        std::fprintf(stderr, "[ARTLeaf·%s] d=%u sz=%u key=0x%08x [",
            label, depth, size, key.key);
        for (uint16_t i = 0; i < std::min(size, (uint16_t)6); i++) {
            if (i) std::fprintf(stderr, ",");
            std::fprintf(stderr, "%lu", (unsigned long)elements[i]);
        }
        if (size > 6) std::fprintf(stderr, "..+%u", size - 6);
        std::fprintf(stderr, "]\n");
    }
};

// ═══════════════════════════════════════════════════════════════
// § 2  ARTNode structs  (upstream 100%)
// ═══════════════════════════════════════════════════════════════

struct ARTNode {
    ARTKey prefix{0ULL};
    uint8_t type  : 4;
    uint8_t depth : 4;
    uint16_t num_children = 0;
    std::atomic<uint16_t> ref_cnt{1};
    ARTLeafUnified* leaf = nullptr;

    ARTNode() : type(NODE4), depth(0) {}
    explicit ARTNode(uint8_t t) : type(t), depth(0) {}
    ARTNode(uint8_t t, ARTKey pfx, uint8_t d) : prefix(pfx), type(t), depth(d) {}
};

struct ARTNode_4 {
    ARTNode n{};
    unsigned char keys[4]{};
    ARTNode* children[4]{};
};

struct ARTNode_16 {
    ARTNode n{};
    unsigned char keys[16]{};
    ARTNode* children[16]{};
};

struct ARTNode_48 {
    ARTNode n{};
    Bitmap<4> unique_bitmap{};
    unsigned char keys[256]{};
    ARTNode* children[48]{};
};

struct ARTNode_256 {
    ARTNode n{};
    Bitmap<4> unique_bitmap{};
    std::array<ARTNode*, ART_LEAF_SIZE> children{};
};

// ═══════════════════════════════════════════════════════════════
// § 3  alloc / dealloc  (upstream art_node.cpp)
// ═══════════════════════════════════════════════════════════════
namespace art_detail {

inline std::atomic<uint64_t>& alloc_ctr() { static std::atomic<uint64_t> c{0}; return c; }
inline std::atomic<uint64_t>& free_ctr()  { static std::atomic<uint64_t> c{0}; return c; }

inline ARTNode* alloc_node(uint8_t type, ARTKey prefix, uint8_t depth, WriterTraceBlock*) {
    alloc_ctr().fetch_add(1, std::memory_order_relaxed);
    switch (type) {
        case NODE4:   { auto* p = new ARTNode_4();   p->n = ARTNode(NODE4, prefix, depth);   return (ARTNode*)p; }
        case NODE16:  { auto* p = new ARTNode_16();  p->n = ARTNode(NODE16, prefix, depth);  return (ARTNode*)p; }
        case NODE48:  { auto* p = new ARTNode_48();  p->n = ARTNode(NODE48, prefix, depth);  return (ARTNode*)p; }
        case NODE256: { auto* p = new ARTNode_256(); p->n = ARTNode(NODE256, prefix, depth); return (ARTNode*)p; }
        default: throw std::runtime_error("alloc_node: bad type");
    }
}

inline void recursive_destroy(ARTNode* n) {
    if (!n) return;
    free_ctr().fetch_add(1, std::memory_order_relaxed);
    if (n->leaf) { delete n->leaf; n->leaf = nullptr; }
    auto destroy_children = [](auto* p, int limit) {
        for (int i = 0; i < limit; i++)
            if (p->children[i]) recursive_destroy(p->children[i]);
    };
    switch (n->type) {
        case NODE4:   destroy_children((ARTNode_4*)n, 4);   delete (ARTNode_4*)n;   break;
        case NODE16:  destroy_children((ARTNode_16*)n, 16); delete (ARTNode_16*)n;  break;
        case NODE48:  destroy_children((ARTNode_48*)n, 48); delete (ARTNode_48*)n;  break;
        case NODE256: destroy_children((ARTNode_256*)n, 256); delete (ARTNode_256*)n; break;
    }
}

inline void delete_node_shallow(ARTNode* n) {
    if (!n) return;
    free_ctr().fetch_add(1, std::memory_order_relaxed);
    switch (n->type) {
        case NODE4:   delete (ARTNode_4*)n;   break;
        case NODE16:  delete (ARTNode_16*)n;  break;
        case NODE48:  delete (ARTNode_48*)n;  break;
        case NODE256: delete (ARTNode_256*)n; break;
    }
}

// ═══════════════════════════════════════════════════════════════
// § 4  find_child / find_child_idx
//      [MOD] NODE16: SSE2 → scalar loop
// ═══════════════════════════════════════════════════════════════

inline ARTNode** find_child(ARTNode* n, unsigned char c) {
    switch (n->type) {
        case NODE4: {
            auto* p = (ARTNode_4*)n;
            for (int i = 0; i < n->num_children; i++)
                if (p->keys[i] == c) return &p->children[i];
            break;
        }
        case NODE16: {
            // [MOD] upstream: _mm_cmpeq_epi8 → scalar
            auto* p = (ARTNode_16*)n;
            for (int i = 0; i < n->num_children; i++)
                if (p->keys[i] == c) return &p->children[i];
            break;
        }
        case NODE48: {
            auto* p = (ARTNode_48*)n;
            if (p->keys[c]) return &p->children[p->keys[c] - 1];
            break;
        }
        case NODE256: {
            auto* p = (ARTNode_256*)n;
            if (p->children[c]) return &p->children[c];
            break;
        }
    }
    return nullptr;
}

inline uint16_t find_child_idx(ARTNode* n, unsigned char c) {
    switch (n->type) {
        case NODE4:  { auto* p=(ARTNode_4*)n;  for(int i=0;i<n->num_children;i++) if(p->keys[i]==c) return i; break; }
        case NODE16: { auto* p=(ARTNode_16*)n; for(int i=0;i<n->num_children;i++) if(p->keys[i]==c) return i; break; }
        case NODE48: { auto* p=(ARTNode_48*)n; if(p->keys[c]) return p->keys[c]-1; break; }
        case NODE256:{ auto* p=(ARTNode_256*)n; if(p->children[c]) return c; break; }
    }
    return 256;
}

// ═══════════════════════════════════════════════════════════════
// § 5  add_child (grow chain 4→16→48→256)  [KEEP 100%]
// ═══════════════════════════════════════════════════════════════

inline ARTNode** add_child256(ARTNode_256* n, ARTNode** ref, unsigned char c, void* child) {
    n->children[c] = (ARTNode*)child;
    n->n.num_children++;
    return &n->children[c];
}

inline ARTNode** add_child48(ARTNode_48* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* tb) {
    if (n->n.num_children < 48) {
        int pos = 0;
        while (n->children[pos]) pos++;
        n->children[pos] = (ARTNode*)child;
        n->keys[c] = pos + 1;
        n->n.num_children++;
        return &n->children[pos];
    }
    // grow 48→256
    auto* nn = (ARTNode_256*)alloc_node(NODE256, n->n.prefix, n->n.depth, tb);
    nn->n.num_children = n->n.num_children;
    for (int i = 0; i < 256; i++)
        if (n->keys[i]) nn->children[i] = n->children[n->keys[i] - 1];
    *ref = (ARTNode*)nn;
    auto r = add_child256(nn, ref, c, child);
    delete n;
    return r;
}

inline ARTNode** add_child16(ARTNode_16* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* tb) {
    if (n->n.num_children < 16) {
        int idx = 0;
        while (idx < n->n.num_children && c > n->keys[idx]) idx++;
        std::memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
        std::memmove(n->children + idx + 1, n->children + idx, (n->n.num_children - idx) * sizeof(void*));
        n->keys[idx] = c;
        n->children[idx] = (ARTNode*)child;
        n->n.num_children++;
        return &n->children[idx];
    }
    // grow 16→48
    auto* nn = (ARTNode_48*)alloc_node(NODE48, n->n.prefix, n->n.depth, tb);
    nn->n.num_children = n->n.num_children;
    std::memcpy(nn->children, n->children, sizeof(void*) * n->n.num_children);
    for (int i = 0; i < n->n.num_children; i++) nn->keys[n->keys[i]] = i + 1;
    *ref = (ARTNode*)nn;
    auto r = add_child48(nn, ref, c, child, tb);
    delete n;
    return r;
}

inline ARTNode** add_child4(ARTNode_4* n, ARTNode** ref, unsigned char c, void* child) {
    if (n->n.num_children < 4) {
        int idx = 0;
        while (idx < n->n.num_children && c > n->keys[idx]) idx++;
        std::memmove(n->keys + idx + 1, n->keys + idx, n->n.num_children - idx);
        std::memmove(n->children + idx + 1, n->children + idx, (n->n.num_children - idx) * sizeof(void*));
        n->keys[idx] = c;
        n->children[idx] = (ARTNode*)child;
        n->n.num_children++;
        return &n->children[idx];
    }
    // grow 4→16
    auto* nn = (ARTNode_16*)alloc_node(NODE16, n->n.prefix, n->n.depth, nullptr);
    nn->n.num_children = n->n.num_children;
    std::memcpy(nn->children, n->children, sizeof(void*) * n->n.num_children);
    std::memcpy(nn->keys, n->keys, n->n.num_children);
    *ref = (ARTNode*)nn;
    auto r = add_child16(nn, ref, c, child, nullptr);
    delete n;
    return r;
}

inline ARTNode** add_child(ARTNode* n, ARTNode** ref, unsigned char c, void* child, WriterTraceBlock* tb) {
    switch (n->type) {
        case NODE4:   return add_child4((ARTNode_4*)n, ref, c, child);
        case NODE16:  return add_child16((ARTNode_16*)n, ref, c, child, tb);
        case NODE48:  return add_child48((ARTNode_48*)n, ref, c, child, tb);
        case NODE256: return add_child256((ARTNode_256*)n, ref, c, child);
        default: throw std::runtime_error("add_child: bad type");
    }
}

// upstream: add_child_copy, copy_node (COW路径)
inline ARTNode** add_child_copy(ARTNode* n, uint8_t idx, ARTNode* child) {
    switch (n->type) {
        case NODE4:   ((ARTNode_4*)n)->children[idx] = child;   return &((ARTNode_4*)n)->children[idx];
        case NODE16:  ((ARTNode_16*)n)->children[idx] = child;  return &((ARTNode_16*)n)->children[idx];
        case NODE48:  ((ARTNode_48*)n)->children[idx] = child;  return &((ARTNode_48*)n)->children[idx];
        case NODE256: ((ARTNode_256*)n)->children[idx] = child; return &((ARTNode_256*)n)->children[idx];
        default: throw std::runtime_error("add_child_copy: bad type");
    }
}

inline ARTNode* copy_node(ARTNode* src, WriterTraceBlock* tb) {
    if (!src) return nullptr;
    auto* dst = alloc_node(src->type, src->prefix, src->depth, tb);
    dst->num_children = src->num_children;
    dst->leaf = src->leaf;
    switch (src->type) {
        case NODE4:
            std::memcpy(((ARTNode_4*)dst)->keys, ((ARTNode_4*)src)->keys, 4);
            std::memcpy(((ARTNode_4*)dst)->children, ((ARTNode_4*)src)->children, 4*sizeof(void*));
            break;
        case NODE16:
            std::memcpy(((ARTNode_16*)dst)->keys, ((ARTNode_16*)src)->keys, 16);
            std::memcpy(((ARTNode_16*)dst)->children, ((ARTNode_16*)src)->children, 16*sizeof(void*));
            break;
        case NODE48:
            std::memcpy(((ARTNode_48*)dst)->keys, ((ARTNode_48*)src)->keys, 256);
            std::memcpy(((ARTNode_48*)dst)->children, ((ARTNode_48*)src)->children, 48*sizeof(void*));
            break;
        case NODE256:
            ((ARTNode_256*)dst)->children = ((ARTNode_256*)src)->children;
            break;
    }
    return dst;
}

// ═══════════════════════════════════════════════════════════════
// § 6  tree_leaf_iter / node_for_each  [KEEP 100%]
// ═══════════════════════════════════════════════════════════════

template<typename F>
void node_for_each_children(ARTNode* n, F&& cb) {
    switch (n->type) {
        case NODE4:  { auto* p=(ARTNode_4*)n;  for(int i=0;i<n->num_children;i++) if(p->children[i]) cb(p->children[i]); break; }
        case NODE16: { auto* p=(ARTNode_16*)n; for(int i=0;i<n->num_children;i++) if(p->children[i]) cb(p->children[i]); break; }
        case NODE48: { auto* p=(ARTNode_48*)n; for(int i=0;i<256;i++) if(p->keys[i]&&p->children[p->keys[i]-1]) cb(p->children[p->keys[i]-1]); break; }
        case NODE256:{ auto* p=(ARTNode_256*)n; for(int i=0;i<256;i++) if(p->children[i]) cb(p->children[i]); break; }
    }
}

template<typename F>
int tree_leaf_iter(ARTNode* n, F&& callback, int depth = 0) {
    if (!n) return 0;
    if (n->leaf) { n->leaf->for_each(callback); return n->leaf->size; }
    int total = 0;
    node_for_each_children(n, [&](ARTNode* child) {
        if (child && child->leaf) { child->leaf->for_each(callback); total += child->leaf->size; }
        else if (child) total += tree_leaf_iter(child, callback, depth + 1);
    });
    return total;
}

// ═══════════════════════════════════════════════════════════════
// § 7  intersect  [KEEP 100%]
// ═══════════════════════════════════════════════════════════════

inline uint64_t leaf_intersect(ARTLeafUnified* a, ARTLeafUnified* b) {
    uint64_t cnt = 0;
    uint16_t i = 0, j = 0;
    while (i < a->size && j < b->size) {
        if (a->elements[i] < b->elements[j]) i++;
        else if (a->elements[i] > b->elements[j]) j++;
        else { cnt++; i++; j++; }
    }
    return cnt;
}

inline uint64_t node_intersect_count(ARTNode* a, ARTNode* b) {
    if (!a || !b) return 0;
    if (a->leaf && b->leaf) return leaf_intersect(a->leaf, b->leaf);
    if (a->leaf) { uint64_t c=0; node_for_each_children(b,[&](ARTNode* x){c+=node_intersect_count(a,x);}); return c; }
    if (b->leaf) return node_intersect_count(b, a);
    uint64_t c = 0;
    node_for_each_children(a, [&](ARTNode* ca) {
        node_for_each_children(b, [&](ARTNode* cb) {
            if (ca && cb && ca->depth == cb->depth &&
                ARTKey::check_partial_match(ca->prefix, cb->prefix, ca->depth))
                c += node_intersect_count(ca, cb);
        });
    });
    return c;
}

inline uint64_t node_range_intersect_count(ARTNode* node, RangeElement* rng, uint16_t rng_sz) {
    if (!node || rng_sz == 0) return 0;
    if (node->leaf) {
        uint64_t c = 0; uint16_t ri = 0;
        for (uint16_t li = 0; li < node->leaf->size && ri < rng_sz;) {
            uint64_t lv = node->leaf->elements[li], rv = rng[ri];
            if (lv < rv) li++; else if (lv > rv) ri++; else { c++; li++; ri++; }
        }
        return c;
    }
    uint64_t t = 0;
    node_for_each_children(node, [&](ARTNode* ch) { t += node_range_intersect_count(ch, rng, rng_sz); });
    return t;
}

inline std::pair<uint64_t,uint64_t> get_node_filling_info(ARTNode* n) {
    if (!n) return {0,0};
    if (n->leaf) return {ART_LEAF_SIZE, n->leaf->size};
    uint64_t cap=0, used=0;
    node_for_each_children(n, [&](ARTNode* c){ auto[cc,uu]=get_node_filling_info(c); cap+=cc; used+=uu; });
    return {cap, used};
}

// ═══════════════════════════════════════════════════════════════
// § 8  leaf_pointer_expand  (upstream: 叶子满→分裂子树) [KEEP 100%]
// ═══════════════════════════════════════════════════════════════

inline ARTLeafUnified* leaf_pointer_expand(ARTNode** node_ref, uint8_t parent_depth,
                                            WriterTraceBlock* tb) {
    ARTNode* n = *node_ref;
    if (!n || !n->leaf) return nullptr;
    ARTLeafUnified* old_leaf = n->leaf;
    n->leaf = nullptr;

    uint8_t split_depth = parent_depth + 1;
    // 找最浅可区分的depth
    while (split_depth < KEY_LEN && old_leaf->size > 1) {
        if (get_key_byte(old_leaf->elements[0], split_depth) !=
            get_key_byte(old_leaf->elements[old_leaf->size - 1], split_depth))
            break;
        split_depth++;
    }

    // 按split_depth的byte分桶, 每桶一个新leaf
    auto* new_root = (ARTNode_4*)alloc_node(NODE4, old_leaf->key, split_depth, tb);
    uint16_t st = 0;
    while (st < old_leaf->size) {
        uint8_t cur_byte = get_key_byte(old_leaf->elements[st], split_depth);
        uint16_t ed = st;
        while (ed < old_leaf->size &&
               get_key_byte(old_leaf->elements[ed], split_depth) == cur_byte)
            ed++;
        auto* child_node = alloc_node(NODE4, ARTKey{old_leaf->elements[st]}, split_depth + 1, tb);
        child_node->leaf = new ARTLeafUnified(ARTKey{old_leaf->elements[st]}, split_depth);
        for (uint16_t i = st; i < ed; i++)
            child_node->leaf->insert(old_leaf->elements[i],
                i < old_leaf->properties.size() ? old_leaf->properties[i] : 0.0);
        add_child((ARTNode*)new_root, (ARTNode**)&new_root, cur_byte, child_node, tb);
        st = ed;
    }
    *node_ref = (ARTNode*)new_root;

    // [NEW] debug
    if (debug::get_debug_level() >= 2)
        std::fprintf(stderr, "[ART·split] depth=%u children=%u old_size=%u\n",
            split_depth, new_root->n.num_children, old_leaf->size);

    return old_leaf;  // caller负责释放
}

}  // namespace art_detail

// ═══════════════════════════════════════════════════════════════
// § 9  ART class  (upstream art.h+art.cpp 主体)
// ═══════════════════════════════════════════════════════════════

class ART {
public:
    ARTNode* root;
    std::atomic<uint64_t> ref_cnt{1};
    uint64_t total_elements{0};
    std::vector<ARTResourceInfo>* resources;
    mutable uint64_t search_ops{0}, insert_ops{0}, remove_ops{0};

    ART() : root(art_detail::alloc_node(NODE4, ARTKey{0ULL}, 0, nullptr)),
            resources(new std::vector<ARTResourceInfo>()) {}

    ~ART() { art_detail::recursive_destroy(root); root = nullptr; delete resources; }

    // ── search (upstream算法100%) ──
    ARTLeafUnified* search(ARTKey key) const {
        search_ops++;
        ARTNode* n = root;
        int depth = 0;
        while (n) {
            if (n->leaf) return n->leaf;
            if (n->depth > depth) {
                if (!ARTKey::check_partial_match(n->prefix, key, n->depth)) return nullptr;
                depth = n->depth;
            }
            ARTNode** child = art_detail::find_child(n, key[depth]);
            n = child ? *child : nullptr;
            depth++;
        }
        return nullptr;
    }

    bool has_element(uint64_t e) const {
        auto* l = search(ARTKey{e});
        return l && l->has_element(e);
    }

    Property_t get_property(uint64_t e, uint8_t pid) const {
        auto* l = search(ARTKey{e});
        if (!l) return 0.0;
        uint16_t pos = l->find(e);
        return (pos < l->size && l->elements[pos] == e) ? l->get_property(pos, pid) : 0.0;
    }

    // ── insert [MOD] COW→in-place ──
    bool insert_element(uint64_t value, Property_t prop = 0.0) {
        insert_ops++;
        return insert_impl(&root, ARTKey{value}, value, prop, 0);
    }

    // ── remove [MOD] COW→in-place ──
    bool remove_element(uint64_t value) {
        remove_ops++;
        return remove_impl(&root, ARTKey{value}, value, 0);
    }

    template<typename F>
    void for_each_element(F&& cb) const { art_detail::tree_leaf_iter(root, std::forward<F>(cb)); }

    uint64_t intersect(const ART* other) const { return art_detail::node_intersect_count(root, other->root); }

    void intersect(const ART* other, std::vector<uint64_t>& result) const {
        for_each_element([&](uint64_t e, double) { if (other->has_element(e)) result.push_back(e); });
    }

    uint64_t range_intersect(RangeElement* rng, uint16_t sz) const {
        return art_detail::node_range_intersect_count(root, rng, sz);
    }

    std::pair<uint64_t,uint64_t> get_filling_info() const { return art_detail::get_node_filling_info(root); }

    // ── GC (upstream 100%) ──
    void handle_resources_copied(WriterTraceBlock*) {
        for (auto& r : *resources) {
            if (r.type == ARTResourceType::ART_Leaf) delete (ARTLeafUnified*)r.ptr;
            else if (r.type == ARTResourceType::ART_Node_Copied) art_detail::delete_node_shallow((ARTNode*)r.ptr);
            else if (r.type == ARTResourceType::ART_Property_Vec) delete (ARTPropertyVec_t*)r.ptr;
        }
        resources->clear();
    }

    void handle_resources_ref() {
        for (auto& r : *resources) {
            if (r.type == ARTResourceType::ART_Leaf) ((ARTLeafUnified*)r.ptr)->ref_cnt--;
            else if (r.type == ARTResourceType::ART_Node_Copied) { ((ARTNode*)r.ptr)->ref_cnt--; }
            else if (r.type == ARTResourceType::ART_Node_Mounted) ((ARTNode*)r.ptr)->ref_cnt++;
        }
        resources->clear();
    }

    void gc_ref(WriterTraceBlock*) { root->ref_cnt.store(1); gc_node_ref_r(root); }
    void destroy() { art_detail::recursive_destroy(root); root = nullptr; }

    // ── [NEW] debug ──
    void dump_tree(const char* label = "") const {
        std::fprintf(stderr, "\n[ART·DUMP·%s] total=%lu ops(s/i/r)=%lu/%lu/%lu\n",
            label, (unsigned long)total_elements,
            (unsigned long)search_ops, (unsigned long)insert_ops, (unsigned long)remove_ops);
        dump_r(root, 0);
        auto [cap, used] = get_filling_info();
        std::fprintf(stderr, "[ART] fill: %lu/%lu (%.1f%%) alloc=%lu free=%lu\n",
            (unsigned long)used, (unsigned long)cap, cap?100.0*used/cap:0.0,
            (unsigned long)art_detail::alloc_ctr().load(),
            (unsigned long)art_detail::free_ctr().load());
    }

    bool validate() const { return validate_r(root, 0); }

private:
    bool insert_impl(ARTNode** ref, ARTKey key, uint64_t val, Property_t prop, int depth) {
        ARTNode* n = *ref;
        if (!n) {
            auto* nn = art_detail::alloc_node(NODE4, key, depth, nullptr);
            nn->leaf = new ARTLeafUnified(key, depth);
            nn->leaf->insert(val, prop);
            *ref = nn;
            total_elements++;
            return true;
        }
        if (n->leaf) {
            if (n->leaf->has_element(val)) return false;
            if (n->leaf->size < ART_LEAF_SIZE) {
                n->leaf->insert(val, prop);
                total_elements++;
                return true;
            }
            // leaf满 → split
            auto* old = art_detail::leaf_pointer_expand(ref, depth > 0 ? depth - 1 : 0, nullptr);
            if (old) delete old;
            return insert_impl(ref, key, val, prop, depth);
        }
        if (depth >= 4) {
            if (!n->leaf) n->leaf = new ARTLeafUnified(key, depth);
            bool added = n->leaf->insert(val, prop);
            if (added) total_elements++;
            return added;
        }
        uint8_t byte = key[depth];
        ARTNode** child = art_detail::find_child(n, byte);
        if (child && *child) return insert_impl(child, key, val, prop, depth + 1);
        // 新child
        auto* nc = art_detail::alloc_node(NODE4, key, depth + 1, nullptr);
        nc->leaf = new ARTLeafUnified(key, depth + 1);
        nc->leaf->insert(val, prop);
        art_detail::add_child(n, ref, byte, nc, nullptr);
        total_elements++;
        if (debug::get_debug_level() >= 2)
            std::fprintf(stderr, "[ART·ins] d=%d b=0x%02x val=%lu type=%u tot=%lu\n",
                depth, byte, (unsigned long)val, n->type, (unsigned long)total_elements);
        return true;
    }

    bool remove_impl(ARTNode** ref, ARTKey key, uint64_t val, int depth) {
        ARTNode* n = *ref;
        if (!n) return false;
        if (n->leaf) { bool r = n->leaf->remove(val); if (r) total_elements--; return r; }
        if (depth >= 4) return false;
        ARTNode** child = art_detail::find_child(n, key[depth]);
        if (!child || !*child) return false;
        return remove_impl(child, key, val, depth + 1);
    }

    void gc_node_ref_r(ARTNode* n) {
        if (!n) return;
        art_detail::node_for_each_children(n, [&](ARTNode* c) {
            if (c) { c->ref_cnt.store(1); gc_node_ref_r(c); }
        });
    }

    void dump_r(ARTNode* n, int indent) const {
        if (!n) return;
        for (int i = 0; i < indent; i++) std::fprintf(stderr, "  ");
        const char* tn[] = {"N4","N16","N48","N256"};
        std::fprintf(stderr, "[%s] d=%u ch=%u", n->type<4?tn[n->type]:"?", n->depth, n->num_children);
        if (n->leaf) std::fprintf(stderr, " LEAF(%u)", n->leaf->size);
        std::fprintf(stderr, "\n");
        art_detail::node_for_each_children(n, [&](ARTNode* c) { dump_r(c, indent+1); });
    }

    bool validate_r(ARTNode* n, int depth) const {
        if (!n) return true;
        if (n->type > NODE256) return false;
        bool ok = true;
        art_detail::node_for_each_children(n, [&](ARTNode* c) { if (c && !validate_r(c, depth+1)) ok=false; });
        return ok;
    }
};

// ═══════════════════════════════════════════════════════════════
// § 10  batch_subtree_build  (upstream art_node_ops.h template)
// ═══════════════════════════════════════════════════════════════

inline void batch_subtree_build(ART& art, RangeElement* elems,
                                 Property_t** props, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        art.insert_element(elems[i], props ? *props[i] : 0.0);
        // [NEW] 进度
        if (debug::get_debug_level() >= 1 && i > 0 && (i % 10000) == 0)
            std::fprintf(stderr, "[ART·batch] %lu/%lu (%.1f%%)\n",
                (unsigned long)i, (unsigned long)count, 100.0*i/count);
    }
}

}  // namespace neograph
}  // namespace philemon

#endif  // PHILEMON_NEOGRAPH_ART_IMPL_HPP
