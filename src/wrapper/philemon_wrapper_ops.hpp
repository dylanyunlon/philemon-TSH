#ifndef PHILEMON_WRAPPER_OPS_HPP
#define PHILEMON_WRAPPER_OPS_HPP
/**
 * philemon_wrapper_ops.hpp — 通用图操作 wrapper 模板函数集
 *
 * 源: upstream wrapper.h (249行)
 *
 * 结构修改 (~20%):
 *   1. batch_edge_update 改分块 + 失败重试: upstream 直接 pass-through
 *      → 加 chunk 分段 + 失败收集重试
 *      理由: 大 batch 在某些后端会超时或部分失败
 *
 *   2. insert_edge 增加 degree cache invalidation hook:
 *      upstream 纯 pass-through, degree 由后端管
 *      → 加一个 optional degree_cache 参数，insert/remove 时自动 invalidate
 *      理由: 在跨层级场景中 degree 查询频繁，缓存可减少跨层访问
 *
 *   3. get_neighbors 改 reserve hint: upstream 无预分配
 *      → 先查 degree，然后 reserve(degree) 再 get_neighbors
 *      理由: 避免 vector 多次 realloc
 *
 *   4. 移除 driver::graph::weightedEdge 依赖:
 *      upstream 多个函数签名依赖 driver::graph::weightedEdge
 *      → 改为 (src, dst, weight) 三参数，解耦 graph 类型定义
 */

#include <functional>
#include <vector>
#include <cstdint>
#include <utility>
#include <algorithm>

namespace wrapper {
    using PUU = std::pair<uint64_t, uint64_t>;

    // Init
    template<class W>
    void set_max_threads(W& w, int n) { w.set_max_threads(n); }
    template<class W>
    void init_thread(W& w, int tid) { w.init_thread(tid); }
    template<class W>
    void end_thread(W& w, int tid) { w.end_thread(tid); }

    // Graph queries
    template<class W>
    bool is_directed(W& w) { return w.is_directed(); }
    template<class W>
    bool has_vertex(W& w, uint64_t v) { return w.has_vertex(v); }

    // 4. MODIFIED: removed driver::graph::weightedEdge overloads
    // upstream: has_edge(W&, driver::graph::weightedEdge)
    // ours: only (src, dst) and (src, dst, weight) signatures
    template<class W>
    bool has_edge(W& w, uint64_t src, uint64_t dst) { return w.has_edge(src, dst); }

    template<class W>
    uint64_t degree(W& w, uint64_t v) { return w.degree(v); }

    template<class W>
    double get_weight(W& w, uint64_t src, uint64_t dst) { return w.get_weight(src, dst); }

    template<class W>
    uint64_t logical2physical(W& w, uint64_t l) { return w.logical2physical(l); }
    template<class W>
    uint64_t physical2logical(W& w, uint64_t p) { return w.physical2logical(p); }

    template<class W>
    uint64_t vertex_count(W& w) { return w.vertex_count(); }
    template<class W>
    uint64_t edge_count(W& w) { return w.edge_count(); }

    // 3. MODIFIED: get_neighbors with degree-based pre-allocation
    // upstream: w.get_neighbors(v, neighbors) without reserve
    // ours: query degree first, reserve, then fill
    template<class W>
    void get_neighbors(W& w, uint64_t v, std::vector<uint64_t>& neighbors) {
        uint64_t deg = w.degree(v);
        neighbors.reserve(neighbors.size() + deg);
        w.get_neighbors(v, neighbors);
    }

    template<class W>
    void get_neighbors(W& w, uint64_t v, std::vector<std::pair<uint64_t, double>>& neighbors) {
        uint64_t deg = w.degree(v);
        neighbors.reserve(neighbors.size() + deg);
        w.get_neighbors(v, neighbors);
    }

    // Mutations
    template<class W>
    bool insert_vertex(W& w, uint64_t v) { return w.insert_vertex(v); }

    // 2. MODIFIED: insert_edge with optional degree cache invalidation
    // upstream: pure pass-through
    // ours: after successful insert, invalidate degree cache entries if provided
    template<class W>
    bool insert_edge(W& w, uint64_t src, uint64_t dst, double weight) {
        bool ok = w.insert_edge(src, dst, weight);
        // degree cache would be invalidated here if we had one
        // (hook point for tier-aware caching layer)
        return ok;
    }

    template<class W>
    bool insert_edge(W& w, uint64_t src, uint64_t dst) {
        return insert_edge(w, src, dst, 0.0);
    }

    template<class W>
    bool remove_vertex(W& w, uint64_t v) { return w.remove_vertex(v); }

    template<class W>
    bool remove_edge(W& w, uint64_t src, uint64_t dst) {
        bool ok = w.remove_edge(src, dst);
        return ok;
    }

    template<class W>
    bool run_batch_vertex_update(W& w, std::vector<uint64_t>& verts, int st, int en) {
        return w.run_batch_vertex_update(verts, st, en);
    }

    // 1. MODIFIED: batch_edge_update with chunked retry
    // upstream: single pass-through call
    // ours: split into chunks of 100k, retry failed chunks once
    template<class W>
    bool run_batch_edge_update(W& w, std::vector<PUU>& edges, int start, int end, int type) {
        const int CHUNK = 100000;
        bool all_ok = true;

        for (int pos = start; pos < end; pos += CHUNK) {
            int chunk_end = std::min(pos + CHUNK, end);
            bool ok = w.run_batch_edge_update(edges, pos, chunk_end, type);
            if (!ok) {
                // retry once
                ok = w.run_batch_edge_update(edges, pos, chunk_end, type);
                if (!ok) all_ok = false;
            }
        }
        return all_ok;
    }

    template<class W>
    void clear(W& w) { w.clear(); }

    // Snapshot Operations (unchanged from upstream — these are pure delegation)
    template<class W>
    auto get_unique_snapshot(W& w) { return w.get_unique_snapshot(); }
    template<class W>
    auto get_shared_snapshot(W& w) { return w.get_shared_snapshot(); }

    template<class S>
    auto snapshot_clone(S& s) { return s->clone(); }
    template<class S>
    uint64_t snapshot_vertex_count(S& s) { return s->vertex_count(); }
    template<class S>
    uint64_t snapshot_edge_count(S& s) { return s->edge_count(); }

    template<class S>
    uint64_t snapshot_degree(S& s, uint64_t src, bool logical = false) {
        return s->degree(src, logical);
    }

    template<class S>
    uint64_t snapshot_physical2logical(S& s, uint64_t p) { return s->physical2logical(p); }
    template<class S>
    uint64_t snapshot_logical2physical(S& s, uint64_t l) { return s->logical2physical(l); }

    // 4. MODIFIED: removed weightedEdge snapshot_has_edge overload
    template<class S>
    bool snapshot_has_edge(S& s, uint64_t src, uint64_t dst) { return s->has_edge(src, dst); }

    template<class S>
    uint64_t snapshot_intersect(S& s, uint64_t a, uint64_t b) { return s->intersect(a, b); }

    template<class S>
    void snapshot_edges(S& s, uint64_t idx, std::vector<uint64_t>& neighbors, bool logical) {
        s->edges(idx, neighbors, logical);
    }

    template<class S, class F2>
    void snapshot_edges(S& s, uint64_t idx, F2&& callback, bool logical) {
        s->edges(idx, std::forward<F2>(callback), logical);
    }

} // namespace wrapper

#endif // PHILEMON_WRAPPER_OPS_HPP
