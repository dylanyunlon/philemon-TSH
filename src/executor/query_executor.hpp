#ifndef PHILEMON_QUERY_EXECUTOR_HPP
#define PHILEMON_QUERY_EXECUTOR_HPP
/**
 * query_executor.hpp — 并发时间查询执行器 — Concurrent temporal query executor
 *
 * 新文件，基于 thread_pool_base.hpp + TEM-Graph index + TieredSnapshot 组合。
 * 骨架借鉴:
 *   - ThreadPool 的 enqueue/future 模式
 *   - RapidStore wrapper 的 snapshot_edges 回调模式
 *   - TEM-Graph 的 contains_query_traced
 *
 * 功能:
 *   - submit_contains(l, r) → future<QueryResult>
 *   - submit_contained(l, r) → future<QueryResult>
 *   - submit_algorithm(AlgoType, params) → future<AlgoResult>
 *   - batch_query() → submit N queries, collect all results
 *   - dump_executor_state(): 打印 pending/completed/pool stats
 *
 * Milestone: M015–M016 (Claude #5–6)
 */

#include "thread_pool_base.hpp"
#include "../index/tem_graph.hpp"
#include "../index/tem_graph_impl.hpp"
#include "../debug/philemon_debug.hpp"

#include <vector>
#include <future>
#include <memory>
#include <chrono>
#include <cstdio>
#include <functional>

namespace philemon {
namespace executor {

// ─── Query types ────────────────────────────────────────────────────
enum class QueryType : uint8_t {
    CONTAINS,
    CONTAINED,
};

struct QueryRequest {
    QueryType         type;
    index::Timestamp  l, r;
    uint64_t          query_id;  // for tracking
};

// ─── Algorithm types ────────────────────────────────────────────────
enum class AlgoType : uint8_t {
    BFS,
    PAGERANK,
    SSSP,
    WCC,
    TC,
};

struct AlgoResult {
    AlgoType   type;
    double     elapsed_ms;
    uint64_t   result_count;  // vertices/components/triangles
    std::string summary;
};

// ─── Concurrent Query Executor ──────────────────────────────────────
class QueryExecutor {
public:
    explicit QueryExecutor(size_t num_threads,
                           index::TemGraph* index = nullptr)
        : pool_(num_threads), index_(index),
          total_queries_(0), total_completed_(0) {
        PHILE_DBG(1, "QueryExecutor: %zu threads, index=%s",
                  num_threads, index ? "yes" : "no");
    }

    ~QueryExecutor() {
        drain();
    }

    // Set/replace the TEM-Graph index
    void set_index(index::TemGraph* idx) { index_ = idx; }

    // ─── Submit single temporal queries ─────────────────────────────

    std::future<index::QueryResult> submit_contains(
        index::Timestamp l, index::Timestamp r) {
        uint64_t qid = total_queries_.fetch_add(1);
        PHILE_DBG(2, "submit_contains[%lu]: [%d, %d]",
                  (unsigned long)qid, l, r);

        return pool_.enqueue([this, l, r, qid](size_t /*worker_id*/) {
            debug::ScopedTimer t("query_contains");
            PHILE_TRACE(debug::TraceEvent::QUERY_BEGIN, qid);

            index::QueryResult result;
            if (index_) {
                result = index_->contains_query_traced(l, r);
            } else {
                result.matched_count = 0;
                result.visited_intervals = 0;
                result.elapsed_us = 0;
                result.query_type = index::CONTAINS_QUERY;
            }

            PHILE_TRACE(debug::TraceEvent::QUERY_END, qid,
                        0, 0, result.matched_count, result.visited_intervals);
            total_completed_.fetch_add(1);
            return result;
        });
    }

    std::future<index::QueryResult> submit_contained(
        index::Timestamp l, index::Timestamp r) {
        uint64_t qid = total_queries_.fetch_add(1);
        PHILE_DBG(2, "submit_contained[%lu]: [%d, %d]",
                  (unsigned long)qid, l, r);

        return pool_.enqueue([this, l, r, qid](size_t /*worker_id*/) {
            debug::ScopedTimer t("query_contained");
            PHILE_TRACE(debug::TraceEvent::QUERY_BEGIN, qid);

            index::QueryResult result;
            if (index_) {
                result = index_->contained_query_traced(l, r);
            } else {
                result.matched_count = 0;
                result.visited_intervals = 0;
                result.elapsed_us = 0;
                result.query_type = index::OTHER_QUERY;
            }

            PHILE_TRACE(debug::TraceEvent::QUERY_END, qid,
                        0, 0, result.matched_count, result.visited_intervals);
            total_completed_.fetch_add(1);
            return result;
        });
    }

    // ─── Batch query submission ─────────────────────────────────────

    struct BatchResult {
        std::vector<index::QueryResult> results;
        double total_elapsed_ms;
        uint64_t total_matched;
        uint64_t total_visited;

        void dump() const {
            std::printf("──── Batch Query Results ────\n");
            std::printf("  queries=%zu total_matched=%lu total_visited=%lu "
                        "elapsed=%.1f ms\n",
                        results.size(),
                        (unsigned long)total_matched,
                        (unsigned long)total_visited,
                        total_elapsed_ms);
            double avg_sel = total_visited > 0
                ? (double)total_matched / total_visited : 0;
            std::printf("  avg_selectivity=%.4f avg_latency=%.1f us\n",
                        avg_sel,
                        results.empty() ? 0 :
                            total_elapsed_ms * 1000.0 / results.size());
            std::printf("──── End Batch ────\n");
        }
    };

    BatchResult batch_query(const std::vector<QueryRequest>& requests) {
        debug::ScopedTimer timer("batch_query");
        auto t0 = std::chrono::high_resolution_clock::now();

        // Submit all queries
        std::vector<std::future<index::QueryResult>> futures;
        futures.reserve(requests.size());
        for (auto& req : requests) {
            if (req.type == QueryType::CONTAINS) {
                futures.push_back(submit_contains(req.l, req.r));
            } else {
                futures.push_back(submit_contained(req.l, req.r));
            }
        }

        // Collect results
        BatchResult batch;
        batch.total_matched = 0;
        batch.total_visited = 0;
        batch.results.reserve(requests.size());

        for (auto& f : futures) {
            auto qr = f.get();
            batch.total_matched += qr.matched_count;
            batch.total_visited += qr.visited_intervals;
            batch.results.push_back(qr);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        batch.total_elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        PHILE_DBG(1, "Batch: %zu queries in %.1f ms (%.0f qps)",
                  requests.size(), batch.total_elapsed_ms,
                  requests.size() / (batch.total_elapsed_ms / 1000.0));
        return batch;
    }

    // ─── Executor state ─────────────────────────────────────────────

    void drain() { pool_.drain(); }

    void dump_state() const {
        std::printf("──── QueryExecutor State ────\n");
        std::printf("  total_queries=%lu completed=%lu pending=%lu\n",
                    (unsigned long)total_queries_.load(),
                    (unsigned long)total_completed_.load(),
                    (unsigned long)(total_queries_.load() -
                                   total_completed_.load()));
        pool_.dump_stats();
        std::printf("──── End Executor ────\n");
    }

    uint64_t total_queries() const { return total_queries_.load(); }
    uint64_t total_completed() const { return total_completed_.load(); }

private:
    ThreadPool pool_;
    index::TemGraph* index_;
    std::atomic<uint64_t> total_queries_;
    std::atomic<uint64_t> total_completed_;
};

}  // namespace executor
}  // namespace philemon

#endif  // PHILEMON_QUERY_EXECUTOR_HPP
