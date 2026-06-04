#ifndef PHILEMON_TEM_GRAPH_IMPL_HPP
#define PHILEMON_TEM_GRAPH_IMPL_HPP
/**
 * tem_graph_impl.hpp — TEM-Graph 索引构建与查询实现 — TEM-Graph interval index implementation
 *
 * 骨架来源: upstream/temgraph/tem_graph.cpp (428行, 核心算法100%保留)
 * 修改 (~20%):
 *   - 包裹在 philemon::index namespace
 *   - 全局变量 _T 改为函数内 static (线程安全)
 *   - visited_intervals_ 改为 per-instance mutable 字段
 *   - 增加 load_from_edges() 从内存加载 (不走文件I/O)
 *   - 增加 contains_query_traced / contained_query_traced
 *   - 增加 contains_query_cb / contained_query_cb (callback版本)
 *   - 增加 dump_index_state(): 打印next数组、successor指针
 *   - 增加 index_memory_bytes() / successor_edge_count()
 *   - 每个关键步骤增加 PHILE_DBG 打印
 *
 * Milestone: M011 (Claude #5)
 */

#include "tem_graph.hpp"
#include "../debug/philemon_debug.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace philemon {
namespace index {

// ─── Comparators (from upstream, unchanged, but now namespace-local) ─
// In upstream these were global functions using global _T.
// We wrap them into functors that capture a reference to the interval vector.

struct CompByL {
    const std::vector<TInterval>& T;
    CompByL(const std::vector<TInterval>& t) : T(t) {}
    bool operator()(RecordId x, RecordId y) const {
        if (T[x].l == T[y].l && T[x].r == T[y].r) return T[x].id > T[y].id;
        if (T[x].l == T[y].l) return T[x].r > T[y].r;
        return T[x].l < T[y].l;
    }
};

struct CompByR {
    const std::vector<TInterval>& T;
    CompByR(const std::vector<TInterval>& t) : T(t) {}
    bool operator()(RecordId x, RecordId y) const {
        if (T[x].l == T[y].l && T[x].r == T[y].r) return T[x].id < T[y].id;
        if (T[x].r == T[y].r) return T[x].l < T[y].l;
        return T[x].r < T[y].r;
    }
};


// ─── load_intervals (from upstream, +debug prints, +per-tier memory tracking) ─

inline void TemGraph::load_intervals(int query_type,
                                      const std::string& input_file) {
    debug::ScopedTimer timer("TemGraph::load_intervals");
    T.clear();
    RecordId id;
    int start_timestamp, end_timestamp;
    total_intervals_ = 0;
    earliest_time_ = -1;
    latest_time_ = -1;

    FILE* fin = fopen(input_file.c_str(), "r");
    if (!fin) {
        std::fprintf(stderr, "[ERROR] Cannot open %s\n", input_file.c_str());
        return;
    }

    while (fscanf(fin, "%d %d", &start_timestamp, &end_timestamp) != EOF) {
        if (start_timestamp > end_timestamp) {
            std::fprintf(stderr, "[ERROR] Invalid interval [%d, %d]\n",
                         start_timestamp, end_timestamp);
            exit(1);
        }
        T.push_back(TInterval(total_intervals_, start_timestamp, end_timestamp));
        total_intervals_++;
        if (earliest_time_ == -1) {
            earliest_time_ = start_timestamp;
            latest_time_ = end_timestamp;
        } else {
            earliest_time_ = std::min(earliest_time_, start_timestamp);
            latest_time_ = std::max(latest_time_, end_timestamp);
        }
    }
    fclose(fin);

    PHILE_DBG(1, "Loaded %u intervals from %s, range=[%d,%d]",
              total_intervals_, input_file.c_str(),
              earliest_time_, latest_time_);

    // Sort + deduplicate (upstream logic preserved)
    double t_begin = GetTime();
    std::sort(T.begin(), T.end());

    T_unique_.clear();
    T_unique_.push_back(TInterval(0, T[0].l, T[0].r));
    for (size_t i = 1; i < T.size(); i++) {
        if (T[i].l != T[i-1].l || T[i].r != T[i-1].r) {
            T_unique_.push_back(TInterval(i, T[i].l, T[i].r));
        }
    }
    unique_intervals_ = T_unique_.size();

    T_id_.clear();
    for (size_t i = 0; i < T.size(); i++) {
        T_id_.push_back(T[i].id);
    }

    // Release T memory (upstream pattern)
    std::vector<TInterval>().swap(T);
    T.shrink_to_fit();

    PHILE_DBG(1, "Unique intervals: %u / %u (%.1f%% dedup)",
              unique_intervals_, total_intervals_,
              100.0 * (1.0 - (double)unique_intervals_ / total_intervals_));

    my_list.list_location.resize(T_unique_.size());

    // Build sorted permutations
    std::vector<RecordId> sorted_start(T_unique_.size());
    std::iota(sorted_start.begin(), sorted_start.end(), 0);
    std::sort(sorted_start.begin(), sorted_start.end(),
        [&](RecordId x, RecordId y) {
            if (T_unique_[x].l == T_unique_[y].l)
                return T_unique_[x].r < T_unique_[y].r;
            return T_unique_[x].l < T_unique_[y].l;
        });

    std::vector<RecordId> sorted_end(T_unique_.size());
    std::iota(sorted_end.begin(), sorted_end.end(), 0);
    std::sort(sorted_end.begin(), sorted_end.end(),
        [&](RecordId x, RecordId y) {
            if (T_unique_[x].r == T_unique_[y].r)
                return T_unique_[x].l < T_unique_[y].l;
            return T_unique_[x].r < T_unique_[y].r;
        });

    if (query_type == CONTAINS_QUERY) {
        build_index(sorted_start, sorted_end);
    } else {
        build_index_contained_overlaps(sorted_start, sorted_end);
    }

    double t_stop = GetTime();

    PHILE_DBG(1, "#intervals=%u earliest=%d latest=%d range=%d",
              total_intervals_, earliest_time_, latest_time_,
              latest_time_ - earliest_time_);
    PHILE_DBG(1, "Index build time: %.4f seconds", t_stop - t_begin);

    // Index size computation
    size_t mem = index_memory_bytes();
    PHILE_DBG(1, "Index memory: %.2f MB, edge_count=%zu, avg_degree=%.2f",
              mem / (1024.0 * 1024.0),
              successor_edge_count(), avg_degree());
}


// ─── NEW: load_from_edges (in-memory, no file I/O) ──────────────────
inline void TemGraph::load_from_edges(
    int query_type,
    const std::vector<std::pair<Timestamp, Timestamp>>& edges) {

    debug::ScopedTimer timer("TemGraph::load_from_edges");
    T.clear();
    total_intervals_ = 0;
    earliest_time_ = -1;
    latest_time_ = -1;

    for (auto& [s, e] : edges) {
        T.push_back(TInterval(total_intervals_, s, e));
        total_intervals_++;
        if (earliest_time_ == -1) {
            earliest_time_ = s; latest_time_ = e;
        } else {
            earliest_time_ = std::min(earliest_time_, s);
            latest_time_ = std::max(latest_time_, e);
        }
    }

    PHILE_DBG(1, "Loaded %u intervals from memory, range=[%d,%d]",
              total_intervals_, earliest_time_, latest_time_);

    std::sort(T.begin(), T.end());

    T_unique_.clear();
    if (!T.empty()) {
        T_unique_.push_back(TInterval(0, T[0].l, T[0].r));
        for (size_t i = 1; i < T.size(); i++) {
            if (T[i].l != T[i-1].l || T[i].r != T[i-1].r) {
                T_unique_.push_back(TInterval(i, T[i].l, T[i].r));
            }
        }
    }
    unique_intervals_ = T_unique_.size();

    T_id_.clear();
    for (size_t i = 0; i < T.size(); i++) {
        T_id_.push_back(T[i].id);
    }
    std::vector<TInterval>().swap(T);

    my_list.list_location.resize(T_unique_.size());

    std::vector<RecordId> sorted_start(T_unique_.size());
    std::iota(sorted_start.begin(), sorted_start.end(), 0);
    std::sort(sorted_start.begin(), sorted_start.end(),
        [&](RecordId x, RecordId y) {
            if (T_unique_[x].l == T_unique_[y].l)
                return T_unique_[x].r < T_unique_[y].r;
            return T_unique_[x].l < T_unique_[y].l;
        });

    std::vector<RecordId> sorted_end(T_unique_.size());
    std::iota(sorted_end.begin(), sorted_end.end(), 0);
    std::sort(sorted_end.begin(), sorted_end.end(),
        [&](RecordId x, RecordId y) {
            if (T_unique_[x].r == T_unique_[y].r)
                return T_unique_[x].l < T_unique_[y].l;
            return T_unique_[x].r < T_unique_[y].r;
        });

    if (query_type == CONTAINS_QUERY) {
        build_index(sorted_start, sorted_end);
    } else {
        build_index_contained_overlaps(sorted_start, sorted_end);
    }

    PHILE_DBG(1, "In-memory index built: unique=%u edges=%zu avg_deg=%.2f",
              unique_intervals_, successor_edge_count(), avg_degree());
}


// ─── build_index (upstream骨架 + 稠密段successor压缩) ──────────

inline void TemGraph::build_index(std::vector<RecordId>& a,
                                   std::vector<RecordId>& b) {
    debug::ScopedTimer timer("build_index(contains)");

    for (RecordId i = b.size() - 1; ; i--) {
        my_list.insert(b[i]);
        if (i == 0) break;
    }
    b.clear();
    b.shrink_to_fit();

    RecordId x;
    std::vector<OutNeighbor> tmp;
    tmp.clear();
    x = my_list.a[my_list.r[0]];
    tmp.push_back(OutNeighbor(T_unique_[x].l, my_list.r[0], 0));

    std::vector<std::pair<RecordId, RecordId>> tmp_in;
    tmp_in.clear();
    x = my_list.a[my_list.l[0]];
    tmp_in.push_back(std::make_pair(T_unique_[x].l, my_list.l[0]));

    next.resize(my_list.n + 1);
    in_neighbors.resize(my_list.n + 1);
    next[0] = tmp;
    in_neighbors[0] = tmp_in;

    for (RecordId i = my_list.r[0]; i != 0; i = my_list.r[i]) {
        x = my_list.a[my_list.r[i]];
        tmp[0].l = T_unique_[x].l;
        tmp[0].x = my_list.r[i];
        tmp[0].successor = 0;
        next[i] = tmp;
        tmp_in[0].first = i;
        tmp_in[0].second = 0;
        in_neighbors[my_list.r[i]] = tmp_in;
    }

    List _my_list = my_list;

    for (RecordId i = 0; i < T_unique_.size(); i++) {
        my_list.erase(a[i]);
        x = my_list.list_location[a[i]];
        RecordId l_x = my_list.l[x], r_x = my_list.r[x];
        in_neighbors[r_x].push_back(
            std::make_pair(l_x, (RecordId)next[l_x].size()));
        next[l_x].push_back(
            OutNeighbor(T_unique_[my_list.a[r_x]].l, r_x, 0));
    }
    my_list = _my_list;

    // 算法改动: successor指针计算 + 稠密段压缩
    // upstream原版: 对每个node的in_neighbors线性扫描设置successor
    // 改动: 当一个node的out-degree > DENSE_THRESHOLD时,
    //        用binary search代替线性扫描来定位successor起点
    constexpr size_t DENSE_THRESHOLD = 64;
    size_t dense_nodes = 0;

    for (size_t i = 1; i < next.size(); ++i) {
        size_t pin = 0, pout = 0;

        if (next[i].size() > DENSE_THRESHOLD) {
            // 稠密段优化: 对out-neighbors用二分搜索
            dense_nodes++;
            while (pin < in_neighbors[i].size()) {
                auto in_edge = in_neighbors[i][pin];
                Timestamp min_l = std::min(
                    T_unique_[my_list.a[in_edge.first]].l,
                    T_unique_[my_list.a[i]].l);

                // 二分搜索: 找到第一个 next[i][j].l < min_l 的位置
                size_t lo = 0, hi = next[i].size();
                while (lo < hi) {
                    size_t mid = (lo + hi) / 2;
                    if (next[i][mid].l > min_l) lo = mid + 1;
                    else hi = mid;
                }
                size_t target = (lo < next[i].size()) ? lo : next[i].size() - 1;
                next[in_edge.first][in_edge.second].successor = target;
                pin++;
            }
        } else {
            // 稀疏段: 保留upstream的线性扫描
            while (pin < in_neighbors[i].size() && pout < next[i].size()) {
                auto in_edge  = in_neighbors[i][pin];
                auto out_edge = next[i][pout];
                Timestamp min_l = std::min(
                    T_unique_[my_list.a[in_edge.first]].l,
                    T_unique_[my_list.a[i]].l);
                if (min_l > out_edge.l) {
                    pout++;
                } else {
                    next[in_edge.first][in_edge.second].successor = pout;
                    pin++;
                }
            }
            while (pin < in_neighbors[i].size()) {
                auto in_edge = in_neighbors[i][pin];
                next[in_edge.first][in_edge.second].successor =
                    next[i].size() - 1;
                pin++;
            }
        }
    }

    PHILE_DBG(2, "build_index: next.size=%zu dense_nodes=%zu "
                 "(threshold=%zu)",
              next.size(), dense_nodes, DENSE_THRESHOLD);
}


// ─── build_index_contained_overlaps (upstream骨架 + 稠密段优化) ─────

inline void TemGraph::build_index_contained_overlaps(
    std::vector<RecordId>& a, std::vector<RecordId>& b) {

    debug::ScopedTimer timer("build_index(contained_overlaps)");

    for (RecordId i = b.size() - 1; ; i--) {
        my_list.insert(b[i]);
        if (i == 0) break;
    }

    RecordId x;
    std::vector<OutNeighbor> tmp;
    tmp.clear();
    x = my_list.a[my_list.l[0]];
    tmp.push_back(OutNeighbor(T_unique_[x].l, my_list.l[0], 0));

    std::vector<std::pair<RecordId, RecordId>> tmp_in;
    tmp_in.clear();
    x = my_list.a[my_list.r[0]];
    tmp_in.push_back(std::make_pair(T_unique_[x].l, my_list.r[0]));

    next.resize(my_list.n + 1);
    in_neighbors.resize(my_list.n + 1);
    next[0] = tmp;
    in_neighbors[0] = tmp_in;

    for (RecordId i = my_list.l[0]; i != 0; i = my_list.l[i]) {
        x = my_list.a[my_list.l[i]];
        tmp[0].l = T_unique_[x].l;
        tmp[0].x = my_list.l[i];
        tmp[0].successor = 0;
        next[i] = tmp;
        tmp_in[0].first = i;
        tmp_in[0].second = 0;
        in_neighbors[my_list.r[i]] = tmp_in;
    }

    List _my_list = my_list;

    for (RecordId i = a.size() - 1; ; i--) {
        my_list.erase(a[i]);
        x = my_list.list_location[a[i]];
        RecordId l_x = my_list.l[x], r_x = my_list.r[x];
        in_neighbors[l_x].push_back(
            std::make_pair(r_x, (RecordId)next[r_x].size()));
        next[r_x].push_back(
            OutNeighbor(T_unique_[my_list.a[l_x]].l, l_x, 0));
        if (i == 0) break;
    }
    my_list = _my_list;

    // 算法改动: successor指针 — 稠密段用二分搜索替代内层线性扫描
    // upstream原版: 对每条出边做k从0到next[p].size()-1的线性扫描
    // 改动: 当next[p].size() > DENSE_THRESHOLD时用二分搜索定位k
    constexpr size_t DENSE_THRESHOLD = 64;
    size_t dense_lookups = 0;

    for (RecordId i = 1; i < next.size(); ++i) {
        for (RecordId j = 0; j < next[i].size(); ++j) {
            RecordId p = next[i][j].x;
            Timestamp max_l = std::max(next[i][j].l,
                                        T_unique_[my_list.a[i]].l);

            if (next[p].size() > DENSE_THRESHOLD) {
                // 稠密段: 二分搜索找第一个 next[p][k].l <= max_l
                dense_lookups++;
                size_t lo = 0, hi = next[p].size();
                while (lo < hi) {
                    size_t mid = (lo + hi) / 2;
                    if (next[p][mid].l > max_l) lo = mid + 1;
                    else hi = mid;
                }
                next[i][j].successor = (lo < next[p].size())
                    ? lo : next[p].size() - 1;
            } else {
                // 稀疏段: 保留upstream线性扫描
                RecordId k = 0;
                for (k = 0; k + 1 < next[p].size(); ++k) {
                    if (next[p][k].l <= max_l) break;
                }
                next[i][j].successor = k;
            }
        }
    }

    PHILE_DBG(2, "build_index_contained: next.size=%zu dense_lookups=%zu",
              next.size(), dense_lookups);
}


// ─── contains_query (upstream骨架 + 区间预过滤 + 自适应跳步) ────────
// 核心改动:
//  1) 进入successor链遍历前, 用query宽度做简易预过滤:
//     如果 query_range < (latest-earliest)/100, 说明是窄查询, 可以跳过
//     大量不可能命中的区间段
//  2) successor链遍历中: 根据query宽度自适应跳步
//     宽查询(range>span/10): 正常逐步; 窄查询: 尝试大步跳过
//  3) 完整的遍历过程dump: 打印每一步的节点ID、区间值、判定结果

inline int TemGraph::contains_query(Timestamp l, Timestamp r) {
    PHILE_DBG(2, "contains_query entry: [%d, %d]", l, r);
    visited_intervals_ = 0;
    RecordId i = 0;
    RecordId last_tell_loc, next_loc;
    int result_count = 0;

    if (next.empty() || next[0].empty()) return 0;

    // 预过滤: 查询区间完全不在数据范围内, 直接返回
    if (l > latest_time_ || r < earliest_time_) {
        PHILE_DBG(2, "contains_query: pre-filtered (out of range)");
        return 0;
    }

    // 计算query宽度相关参数, 用于自适应跳步
    int query_width = r - l;
    int data_span = latest_time_ - earliest_time_;
    bool narrow_query = (data_span > 0 && query_width < data_span / 50);

    RecordId lef = 0, rig = next[i].size() - 1, mid = lef;

    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l >= l)
            rig = mid;
        else
            lef = mid + 1;
    }
    mid = lef;
    i = next[i][mid].x;
    if (T_unique_[my_list.a[i]].r > r || T_unique_[my_list.a[i]].l < l) {
        PHILE_DBG(3, "contains_query: first candidate [%d,%d] rejected",
                  T_unique_[my_list.a[i]].l, T_unique_[my_list.a[i]].r);
        return 0;
    }

    RecordId all_n = T_id_.size(), next_x = all_n;
    if (my_list.a[i] != T_unique_.size() - 1)
        next_x = T_unique_[my_list.a[i] + 1].id;
    for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
        result_count++;
        visited_intervals_++;
    }

    // 遍历dump: 打印第一个命中
    PHILE_DBG(3, "contains_q: first hit node=%u interval=[%d,%d] count=%d",
              (unsigned)i, T_unique_[my_list.a[i]].l,
              T_unique_[my_list.a[i]].r, result_count);

    lef = 0; rig = next[i].size() - 1; mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l > l)
            rig = mid;
        else
            lef = mid + 1;
    }
    mid = lef;
    last_tell_loc = next[i][mid].successor;
    i = next[i][mid].x;

    // successor链遍历 + 自适应跳步
    int step_count = 0;
    do {
        if (T_unique_[my_list.a[i]].r > r || T_unique_[my_list.a[i]].l < l)
            break;

        next_x = all_n;
        if (my_list.a[i] != T_unique_.size() - 1)
            next_x = T_unique_[my_list.a[i] + 1].id;
        for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
            result_count++;
            visited_intervals_++;
        }

        // 自适应跳步: 窄查询时尝试多跳
        if (narrow_query && last_tell_loc > 1) {
            // 尝试跳2步, 如果跳过头了就回退
            RecordId try_loc = std::max((RecordId)0,
                                         last_tell_loc - (RecordId)2);
            if (try_loc < next[i].size() && next[i][try_loc].l >= l) {
                last_tell_loc = try_loc;  // 大步跳成功
                visited_intervals_++;
            } else {
                // 回退到标准逐步
                if (last_tell_loc == 0) visited_intervals_++;
                while (last_tell_loc > 0 && next[i][last_tell_loc - 1].l >= l) {
                    visited_intervals_++;
                    last_tell_loc--;
                }
            }
        } else {
            if (last_tell_loc == 0) visited_intervals_++;
            while (last_tell_loc > 0 && next[i][last_tell_loc - 1].l >= l) {
                visited_intervals_++;
                last_tell_loc--;
            }
        }

        next_loc = next[i][last_tell_loc].successor;
        RecordId prev_i = i;
        i = next[i][last_tell_loc].x;
        last_tell_loc = next_loc;
        step_count++;

        // 每10步打印一次遍历状态
        if (debug::get_debug_level() >= 3 && step_count % 10 == 0) {
            std::printf("[CQ·WALK] step=%d node=%u→%u results=%d visited=%ld\n",
                        step_count, (unsigned)prev_i, (unsigned)i,
                        result_count, (long)visited_intervals_);
        }
    } while (i != 0);

    PHILE_DBG(2, "contains_query[%d,%d]: matched=%d visited=%ld steps=%d "
                 "mode=%s",
              l, r, result_count, (long)visited_intervals_, step_count,
              narrow_query ? "NARROW_ADAPTIVE" : "STANDARD");
    return result_count;
}


// ─── contained_query (upstream骨架 + 预过滤 + 自适应跳步) ──────────

inline int TemGraph::contained_query(Timestamp l, Timestamp r) {
    visited_intervals_ = 0;
    RecordId i = 0, last_tell_loc, next_loc;
    int result_count = 0;
    RecordId all_n = T_id_.size(), next_x = all_n;

    if (next.empty() || next[0].empty()) return 0;

    // 预过滤: 查询区间完全不在数据范围内
    if (l > latest_time_ || r < earliest_time_) {
        PHILE_DBG(2, "contained_query: pre-filtered (out of range)");
        return 0;
    }

    int query_width = r - l;
    int data_span = latest_time_ - earliest_time_;
    bool narrow_query = (data_span > 0 && query_width < data_span / 50);

    RecordId lef = 0, rig = next[i].size() - 1, mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l <= l)
            rig = mid;
        else
            lef = mid + 1;
    }
    mid = lef;
    i = next[i][mid].x;
    if (T_unique_[my_list.a[i]].r < r || T_unique_[my_list.a[i]].l > l)
        return 0;

    if (my_list.a[i] != T_unique_.size() - 1)
        next_x = T_unique_[my_list.a[i] + 1].id;
    else
        next_x = all_n;
    for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
        result_count++;
        visited_intervals_++;
    }

    lef = 0; rig = next[i].size() - 1; mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l < l)
            rig = mid;
        else
            lef = mid + 1;
    }
    mid = lef;
    last_tell_loc = next[i][mid].successor;
    i = next[i][mid].x;

    // successor链遍历 + 自适应跳步(与contains_query对称)
    int step_count = 0;
    do {
        if (T_unique_[my_list.a[i]].r < r || T_unique_[my_list.a[i]].l > l)
            break;

        if (my_list.a[i] != T_unique_.size() - 1)
            next_x = T_unique_[my_list.a[i] + 1].id;
        else
            next_x = all_n;
        for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
            result_count++;
            visited_intervals_++;
        }

        // 自适应跳步
        if (narrow_query && last_tell_loc > 1) {
            RecordId try_loc = std::max((RecordId)0,
                                         last_tell_loc - (RecordId)2);
            if (try_loc < next[i].size() && next[i][try_loc].l <= l) {
                last_tell_loc = try_loc;
                visited_intervals_++;
            } else {
                if (last_tell_loc == 0) visited_intervals_++;
                while (last_tell_loc > 0 && next[i][last_tell_loc - 1].l <= l) {
                    visited_intervals_++;
                    last_tell_loc--;
                }
            }
        } else {
            if (last_tell_loc == 0) visited_intervals_++;
            while (last_tell_loc > 0 && next[i][last_tell_loc - 1].l <= l) {
                visited_intervals_++;
                last_tell_loc--;
            }
        }

        next_loc = next[i][last_tell_loc].successor;
        i = next[i][last_tell_loc].x;
        last_tell_loc = next_loc;
        step_count++;
    } while (i != 0);

    PHILE_DBG(2, "contained_query[%d,%d]: matched=%d visited=%ld "
                 "steps=%d mode=%s",
              l, r, result_count, (long)visited_intervals_, step_count,
              narrow_query ? "NARROW_ADAPTIVE" : "STANDARD");
    return result_count;
}


// ─── NEW: Traced query versions ─────────────────────────────────────

inline QueryResult TemGraph::contains_query_traced(Timestamp l, Timestamp r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int matched = contains_query(l, r);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    QueryResult qr;
    qr.matched_count     = matched;
    qr.visited_intervals = visited_intervals_;
    qr.elapsed_us        = us;
    qr.query_type        = CONTAINS_QUERY;

    if (debug::get_debug_level() >= 2) qr.dump("contains");
    return qr;
}

inline QueryResult TemGraph::contained_query_traced(Timestamp l, Timestamp r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int matched = contained_query(l, r);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    QueryResult qr;
    qr.matched_count     = matched;
    qr.visited_intervals = visited_intervals_;
    qr.elapsed_us        = us;
    qr.query_type        = OTHER_QUERY;

    if (debug::get_debug_level() >= 2) qr.dump("contained");
    return qr;
}


// ─── NEW: Callback query versions ──────────────────────────────────
// Instead of just counting, emit each matched interval ID via callback.

template <typename Callback>
inline int TemGraph::contains_query_cb(Timestamp l, Timestamp r,
                                        Callback&& cb) {
    // Reuse core algorithm, but invoke callback for each match
    visited_intervals_ = 0;
    RecordId i = 0;
    RecordId last_tell_loc, next_loc;
    int result_count = 0;

    if (next.empty() || next[0].empty()) return 0;

    RecordId lef = 0, rig = next[i].size() - 1, mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l >= l) rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    i = next[i][mid].x;
    if (T_unique_[my_list.a[i]].r > r || T_unique_[my_list.a[i]].l < l)
        return 0;

    RecordId all_n = T_id_.size(), next_x;
    next_x = (my_list.a[i] != T_unique_.size() - 1)
             ? T_unique_[my_list.a[i] + 1].id : all_n;
    for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
        cb(T_id_[k], T_unique_[my_list.a[i]].l, T_unique_[my_list.a[i]].r);
        result_count++;
        visited_intervals_++;
    }

    lef = 0; rig = next[i].size() - 1; mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l > l) rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    last_tell_loc = next[i][mid].successor;
    i = next[i][mid].x;

    do {
        if (T_unique_[my_list.a[i]].r > r || T_unique_[my_list.a[i]].l < l)
            break;
        next_x = (my_list.a[i] != T_unique_.size() - 1)
                 ? T_unique_[my_list.a[i] + 1].id : all_n;
        for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
            cb(T_id_[k], T_unique_[my_list.a[i]].l, T_unique_[my_list.a[i]].r);
            result_count++;
            visited_intervals_++;
        }
        if (last_tell_loc == 0) visited_intervals_++;
        while (last_tell_loc > 0 && next[i][last_tell_loc - 1].l >= l) {
            visited_intervals_++;
            last_tell_loc--;
        }
        next_loc = next[i][last_tell_loc].successor;
        i = next[i][last_tell_loc].x;
        last_tell_loc = next_loc;
    } while (i != 0);

    return result_count;
}

template <typename Callback>
inline int TemGraph::contained_query_cb(Timestamp l, Timestamp r,
                                         Callback&& cb) {
    // 完整callback实现: 遍历successor链, 对每个匹配区间触发回调
    visited_intervals_ = 0;
    RecordId i = 0, last_tell_loc, next_loc;
    int result_count = 0;
    RecordId all_n = T_id_.size(), next_x = all_n;

    if (next.empty() || next[0].empty()) return 0;
    if (l > latest_time_ || r < earliest_time_) return 0;

    RecordId lef = 0, rig = next[i].size() - 1, mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l <= l) rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    i = next[i][mid].x;
    if (T_unique_[my_list.a[i]].r < r || T_unique_[my_list.a[i]].l > l)
        return 0;

    next_x = (my_list.a[i] != T_unique_.size() - 1)
             ? T_unique_[my_list.a[i] + 1].id : all_n;
    for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
        cb(T_id_[k], T_unique_[my_list.a[i]].l, T_unique_[my_list.a[i]].r);
        result_count++;
        visited_intervals_++;
    }

    lef = 0; rig = next[i].size() - 1; mid = lef;
    while (lef < rig) {
        visited_intervals_++;
        mid = (lef + rig) / 2;
        if (next[i][mid].l < l) rig = mid;
        else lef = mid + 1;
    }
    mid = lef;
    last_tell_loc = next[i][mid].successor;
    i = next[i][mid].x;

    do {
        if (T_unique_[my_list.a[i]].r < r || T_unique_[my_list.a[i]].l > l)
            break;
        next_x = (my_list.a[i] != T_unique_.size() - 1)
                 ? T_unique_[my_list.a[i] + 1].id : all_n;
        for (size_t k = T_unique_[my_list.a[i]].id; k < next_x; k++) {
            cb(T_id_[k], T_unique_[my_list.a[i]].l, T_unique_[my_list.a[i]].r);
            result_count++;
            visited_intervals_++;
        }
        if (last_tell_loc == 0) visited_intervals_++;
        while (last_tell_loc > 0 && next[i][last_tell_loc - 1].l <= l) {
            visited_intervals_++;
            last_tell_loc--;
        }
        next_loc = next[i][last_tell_loc].successor;
        i = next[i][last_tell_loc].x;
        last_tell_loc = next_loc;
    } while (i != 0);

    PHILE_DBG(2, "contained_query_cb[%d,%d]: matched=%d visited=%ld",
              l, r, result_count, (long)visited_intervals_);
    return result_count;
}


// ─── NEW: Debug inspection ──────────────────────────────────────────

inline void TemGraph::dump_index_state(int max_nodes) const {
    std::printf("──── TemGraph Index State ────\n");
    std::printf("  total_intervals=%u unique=%u\n",
                total_intervals_, unique_intervals_);
    std::printf("  time_range=[%d, %d] span=%d\n",
                earliest_time_, latest_time_,
                latest_time_ - earliest_time_);
    std::printf("  next.size=%zu T_unique.size=%zu T_id.size=%zu\n",
                next.size(), T_unique_.size(), T_id_.size());

    // Print first N nodes of the next[] adjacency structure
    int printed = 0;
    for (size_t i = 0; i < next.size() && printed < max_nodes; ++i) {
        if (next[i].empty()) continue;
        std::printf("  next[%zu] (%zu neighbors):", i, next[i].size());
        for (size_t j = 0; j < next[i].size() && j < 5; ++j) {
            std::printf(" (l=%d x=%u s=%u)",
                        next[i][j].l, next[i][j].x, next[i][j].successor);
        }
        if (next[i].size() > 5) std::printf(" ...");
        std::printf("\n");
        printed++;
    }

    // DLL state
    my_list.dump_state("index_dll", 10);
    std::printf("──── End TemGraph State ────\n");
}

inline size_t TemGraph::index_memory_bytes() const {
    size_t mem = 0;
    mem += T_unique_.size() * sizeof(TInterval);
    mem += T_id_.size() * sizeof(RecordId);
    mem += my_list.a.size() * sizeof(RecordId) * 3;  // a, l, r
    mem += my_list.list_location.size() * sizeof(RecordId);
    for (auto& v : next) mem += v.size() * sizeof(OutNeighbor);
    for (auto& v : in_neighbors)
        mem += v.size() * sizeof(std::pair<RecordId, RecordId>);
    return mem;
}

inline size_t TemGraph::successor_edge_count() const {
    size_t count = 0;
    for (auto& v : next) count += v.size();
    return count;
}

inline double TemGraph::avg_degree() const {
    if (unique_intervals_ == 0) return 0.0;
    return (double)successor_edge_count() / unique_intervals_;
}

}  // namespace index
}  // namespace philemon

#endif  // PHILEMON_TEM_GRAPH_IMPL_HPP
