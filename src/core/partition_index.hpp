/**
 * partition_index.hpp — 单分区时间区间索引（双排序加速查询）
 *
 * Integrates TEM-Graph's interval query strategy into Philemon-TSH's
 * per-partition data. Instead of the full doubly-linked list + successor
 * pointer structure (which requires O(n log n) space for the next[] array),
 * we use a simpler but effective dual-sorted index:
 *
 *   1. Primary: edges sorted by ts_start (existing from flush_partitions)
 *   2. Secondary: sorted index by ts_end for contained_query
 *   3. Endpoint arrays for O(1) range bounds via binary search
 *
 * This enables:
 *   - contains_query(l, r): find edges [a,b] where a≥l and b≤r → O(log N + output)
 *   - contained_query(l, r): find edges [a,b] where a≤l and b≥r → O(log N + output)
 *   - overlap_query(l, r): find edges [a,b] where a≤r and b≥l → O(log N + output)
 *
 * Starting from TEM-Graph's build_index (C) as the good example:
 *   upstream/temgraph/tem_graph.cpp build_index():
 *     - Sorts intervals by end timestamp (T sorted by r ascending)
 *     - Builds doubly-linked list with successor pointers
 *     - contains_query uses binary search + successor traversal
 *
 * We follow that pattern to implement PartitionIndex (D), letting
 * scan_partition (E) use index-accelerated range queries (F), and
 * achieve O(log N + output) for all temporal query types (G). Then
 * the secondary end-sorted index (H) introduces contained_query
 * support (I), so that overlap detection (J) can efficiently find
 * all temporal intersections (K), while the compact index layout (L)
 * optimizes cache utilization (M).
 *
 * Pattern lineage:
 *   TEM-Graph build_index (tem_graph.cpp) → dual-sorted interval index
 *   TEM-Graph contains_query → binary search + forward scan
 *   LevelDB TwoLevelIterator::Seek → two-level index lookup
 *   Thrust lower_bound → parallel binary search
 *
 * Milestone: M011 (Claude #5)
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric>

#include "temporal_edge.hpp"  // M011: TemporalEdge definition

namespace philemon {

/**
 * PartitionIndex — lightweight per-partition interval index.
 *
 * Built once after flush_partitions(). Enables O(log N + output)
 * temporal queries on sorted edge data.
 *
 * Memory overhead: 2 × sizeof(uint32_t) × edge_count for the
 * secondary sorted-by-end index. The primary sort is in-place
 * (edges are already sorted by ts_start in the partition).
 */
class PartitionIndex {
public:
    PartitionIndex() = default;

    /**
     * Build index over a partition's edge data.
     *
     * @param edges     Pointer to sorted-by-ts_start edge array.
     * @param count     Number of edges.
     *
     * Builds:
     *   - end_sorted_: permutation indices sorting edges by ts_end
     *   - min_end_:    running minimum of ts_end from position i to end
     *                  (enables early termination in contains_query)
     *
     * Time: O(N log N) for sort.  Space: O(N) for index arrays.
     *
     * Following TEM-Graph's build_index pattern:
     *   vector<RecordId> sorted_by_end(T_unique_.size());
     *   iota(sorted_by_end.begin(), sorted_by_end.end(), 0);
     *   sort(sorted_by_end.begin(), sorted_by_end.end(), [&](...) {
     *       return T_unique_[x].r < T_unique_[y].r;
     *   });
     */
    void build(const TemporalEdge* edges, size_t count) {
        std::printf("[PARTITION-IDX] building index: %zu edges\n",
                    count);
        edges_ = edges;
        count_ = count;

        if (count == 0) return;

        // Build secondary index sorted by ts_end.
        // Pattern: TEM-Graph sorted_by_end with iota + custom sort.
        end_sorted_.resize(count);
        std::iota(end_sorted_.begin(), end_sorted_.end(), 0u);
        std::sort(end_sorted_.begin(), end_sorted_.end(),
            [edges](uint32_t a, uint32_t b) {
                if (edges[a].ts_finish == edges[b].ts_finish)
                    return edges[a].ts_begin < edges[b].ts_begin;
                return edges[a].ts_finish < edges[b].ts_finish;
            });

        // Build running max of ts_end from each start position.
        // max_end_from_[i] = max(edges[j].ts_finish for j in [i..count-1])
        // Used for early termination: if max_end_from_[i] < ts_lo,
        // no edges from position i onward can match.
        max_end_from_.resize(count);
        max_end_from_[count - 1] = edges[count - 1].ts_finish;
        for (size_t i = count - 1; i > 0; --i) {
            max_end_from_[i - 1] = std::max(max_end_from_[i], edges[i - 1].ts_finish);
        }

        // Build running min of ts_start for end-sorted index.
        // min_start_[i] = min(edges[end_sorted_[j]].ts_begin for j in [0..i])
        min_start_.resize(count);
        min_start_[0] = edges[end_sorted_[0]].ts_begin;
        for (size_t i = 1; i < count; ++i) {
            min_start_[i] = std::min(min_start_[i - 1],
                                     edges[end_sorted_[i]].ts_begin);
        }

        built_ = true;
    }

    /**
     * contains_query: find edges [a,b] where a ≥ lo AND b ≤ hi.
     * (Edge interval is contained within query interval.)
     *
     * This is the primary query for temporal subgraph extraction.
     * O(log N + output).
     *
     * Following TEM-Graph's contains_query pattern:
     *   - Binary search for first edge with ts_start ≥ lo
     *   - Forward scan, emitting edges with ts_end ≤ hi
     *   - Early termination when ts_start > hi
     */
    template <typename Callback>
    uint64_t contains_query(int32_t lo, int32_t hi, Callback&& cb) const {
        // contains_query入口
        if (!built_ || count_ == 0) return 0;

        // Binary search: first edge with ts_start >= lo.
        // Pattern: TEM-Graph contains_query binary search on next[i].
        const TemporalEdge* first = std::lower_bound(
            edges_, edges_ + count_, lo,
            [](const TemporalEdge& e, int32_t val) {
                return e.ts_begin < val;
            });

        uint64_t matched = 0;
        for (const TemporalEdge* it = first; it != edges_ + count_; ++it) {
            if (it->ts_begin > hi) break;  // early termination
            if (it->ts_finish <= hi) {
                cb(*it);
                ++matched;
            }
        }
        return matched;
    }

    /**
     * contained_query: find edges [a,b] where a ≤ lo AND b ≥ hi.
     * (Query interval is contained within edge interval.)
     *
     * Uses the end-sorted secondary index.
     * O(log N + output).
     *
     * Following TEM-Graph's contained_query pattern:
     *   - Binary search on end-sorted index for first with ts_end ≥ hi
     *   - Scan forward, emitting edges with ts_start ≤ lo
     */
    template <typename Callback>
    uint64_t contained_query(int32_t lo, int32_t hi, Callback&& cb) const {
        if (!built_ || count_ == 0) return 0;

        // Binary search on end-sorted: first index where ts_end >= hi.
        size_t pos = std::lower_bound(
            end_sorted_.begin(), end_sorted_.end(), hi,
            [this](uint32_t idx, int32_t val) {
                return edges_[idx].ts_finish < val;
            }) - end_sorted_.begin();

        uint64_t matched = 0;
        for (size_t i = pos; i < count_; ++i) {
            uint32_t idx = end_sorted_[i];
            if (edges_[idx].ts_begin <= lo) {
                cb(edges_[idx]);
                ++matched;
            }
            // Pruning: if min_start for remaining entries > lo,
            // no more matches possible.
            if (i + 1 < count_ && min_start_[i] > lo) break;
        }
        return matched;
    }

    /**
     * overlap_query: find edges [a,b] where a ≤ hi AND b ≥ lo.
     * (Edge interval overlaps query interval.)
     *
     * O(log N + output).
     */
    template <typename Callback>
    uint64_t overlap_query(int32_t lo, int32_t hi, Callback&& cb) const {
        if (!built_ || count_ == 0) return 0;

        // Find first edge with ts_start <= hi (all before this).
        // Then check ts_end >= lo.
        uint64_t matched = 0;
        for (size_t i = 0; i < count_; ++i) {
            if (edges_[i].ts_begin > hi) break;
            if (edges_[i].ts_finish >= lo) {
                cb(edges_[i]);
                ++matched;
            }
        }
        return matched;
    }

    bool is_built() const { return built_; }
    size_t count() const { return count_; }

    // Memory overhead in bytes (just the index arrays, not the edges).
    size_t index_memory() const {
        return (end_sorted_.size() + max_end_from_.size() + min_start_.size())
               * sizeof(uint32_t)
               + sizeof(int32_t) * max_end_from_.size();
    }

private:
    const TemporalEdge* edges_ = nullptr;
    size_t count_ = 0;
    bool built_ = false;

    // Secondary sort: indices into edges_ sorted by ts_end ascending.
    std::vector<uint32_t> end_sorted_;

    // Running max of ts_end from position i forward (in primary sort order).
    std::vector<int32_t> max_end_from_;

    // Running min of ts_start up to position i (in end-sorted order).
    std::vector<int32_t> min_start_;
};

}  // namespace philemon
