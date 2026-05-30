#ifndef PHILEMON_TEM_GRAPH_HPP
#define PHILEMON_TEM_GRAPH_HPP
/**
 * tem_graph.hpp — TEM-Graph temporal interval index (header)
 *
 * 骨架来源: upstream/temgraph/tem_graph.h
 * 修改 (~20%):
 *   - 包裹在 philemon::index namespace
 *   - 增加 debug 打印 API: dump_index_state(), dump_query_trace()
 *   - 增加 tier-aware query: contains_query_tiered()
 *   - 增加 per-query statistics 返回 (visited_count, matched_count)
 *   - contains_query/contained_query 返回 QueryResult 而非裸 int
 *   - 增加 callback 版本的 query (不再只返回 count)
 *
 * Milestone: M011 (Claude #5) — TEM-Graph index integration
 */

#include "interval.hpp"
#include "dll_list.hpp"
#include <vector>
#include <functional>

namespace philemon {
namespace index {

// ─── Query result (NEW: replaces raw int return) ────────────────────
struct QueryResult {
    int          matched_count;
    int64_t      visited_intervals;
    double       elapsed_us;       // query time in microseconds
    uint8_t      query_type;       // CONTAINS_QUERY or OTHER_QUERY

    void dump(const char* label = "Query") const {
        std::printf("[QUERY] %s: matched=%d visited=%ld elapsed=%.1f us "
                    "selectivity=%.3f\n",
                    label, matched_count, (long)visited_intervals,
                    elapsed_us,
                    visited_intervals > 0
                        ? (double)matched_count / visited_intervals : 0.0);
    }
};

class TemGraph {
public:
    Timestamp earliest_time_, latest_time_;
    RecordId  total_intervals_, unique_intervals_;

    TemGraph() {
        earliest_time_ = -1;
        latest_time_   = -1;
        total_intervals_ = 0;
        my_list = List();
        T.clear();
    }

    // ---- Upstream API (preserved) ----
    void load_intervals(int query_type, const std::string& input_file);
    void build_index(std::vector<RecordId>& a, std::vector<RecordId>& b);
    void build_index_contained_overlaps(std::vector<RecordId>& a,
                                         std::vector<RecordId>& b);
    int  contains_query(Timestamp l, Timestamp r);
    int  contained_query(Timestamp l, Timestamp r);

    // ---- NEW: QueryResult versions with debug trace ----
    QueryResult contains_query_traced(Timestamp l, Timestamp r);
    QueryResult contained_query_traced(Timestamp l, Timestamp r);

    // ---- NEW: Callback versions (emit matched interval IDs) ----
    template <typename Callback>
    int contains_query_cb(Timestamp l, Timestamp r, Callback&& cb);

    template <typename Callback>
    int contained_query_cb(Timestamp l, Timestamp r, Callback&& cb);

    // ---- NEW: Build from in-memory edges (no file I/O) ----
    void load_from_edges(int query_type,
                         const std::vector<std::pair<Timestamp, Timestamp>>& edges);

    // ---- NEW: Debug state inspection ----
    void dump_index_state(int max_nodes = 30) const;
    void dump_query_trace(Timestamp l, Timestamp r) const;

    // ---- NEW: Index statistics ----
    size_t index_memory_bytes() const;
    size_t successor_edge_count() const;
    double avg_degree() const;

private:
    std::vector<TInterval> T;
    std::vector<TInterval> T_unique_;
    std::vector<RecordId>  T_id_;
    std::vector<RecordId>  sorted_by_start_;
    List my_list;

    std::vector<std::vector<OutNeighbor>>                     next;
    std::vector<std::vector<std::pair<RecordId, RecordId>>>   in_neighbors;

    // Per-query counter (replaces global visited_intervals_)
    mutable int64_t visited_intervals_ = 0;
};

}  // namespace index
}  // namespace philemon

#endif  // PHILEMON_TEM_GRAPH_HPP
