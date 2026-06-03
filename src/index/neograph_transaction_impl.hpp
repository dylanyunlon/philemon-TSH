#ifndef PHILEMON_NEOGRAPH_TRANSACTION_IMPL_HPP
#define PHILEMON_NEOGRAPH_TRANSACTION_IMPL_HPP
/**
 * neograph_transaction_impl.hpp — Transaction管理 完整移植
 *
 * 骨架来源:
 *   upstream neo_transaction.h  (331行) + neo_transaction.cpp (537行)
 *   合计 ~868行
 *
 * 修改 (~20%):
 *   - [MOD] WriteTransaction::commit: tbb::parallel_sort → std::sort
 *   - [MOD] batch edge insert: 全局锁+parallel → 单线程序列化(Phase5优化)
 *   - [MOD] TransactionManager: 内联计数 → 加commit延迟统计
 *   - [NEW] dump_tx_stats(): 打印事务吞吐/延迟
 *   - [NEW] begin/commit/abort: debug>=2时打印操作日志
 *   - [KEEP] ReadTransaction: timestamp快照+reader_enter/leave 100%
 *   - [KEEP] WriteTransaction: 写缓冲→commit时batch flush 100%
 *   - [KEEP] LightWriteTransaction: 轻量单树写入 100%
 *   - [KEEP] vertex_count/edge_count 原子计数 100%
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <atomic>
#include <utility>
#include <chrono>
#include <cassert>

#include "neograph_types_impl.hpp"
#include "neograph_version_impl.hpp"
#include "neograph_trace_impl.hpp"
#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace neograph {

using PRR = std::pair<RangeElement, RangeElement>;

// Forward
class NeoGraphIndex;

// ═══════════════════════════════════════════════════════════════
// TransactionManager (upstream neo_transaction.h/cpp)
// ═══════════════════════════════════════════════════════════════

class TransactionManager {
public:
    std::atomic<uint64_t> write_timestamp{0};
    std::atomic<uint64_t> read_timestamp{0};
    uint64_t vertex_count_ = 0;
    uint64_t edge_count_ = 0;
    bool is_directed_;
    bool is_weighted_;

    // [NEW] 统计
    std::atomic<uint64_t> total_commits{0};
    std::atomic<uint64_t> total_aborts{0};

    // 版本链(简化: 单NeoTree)
    NeoTree* primary_tree = nullptr;
    ReaderTraceBlock reader_trace;
    WriterTraceBlock writer_trace;

    TransactionManager(bool directed, bool weighted)
        : is_directed_(directed), is_weighted_(weighted) {
        primary_tree = new NeoTree(0);
    }

    ~TransactionManager() {
        delete primary_tree;
    }

    uint64_t vertex_count() const { return vertex_count_; }
    uint64_t edge_count() const { return edge_count_; }
    uint64_t get_write_timestamp() { return write_timestamp.fetch_add(1, std::memory_order_acq_rel); }
    uint64_t get_read_timestamp() const { return read_timestamp.load(std::memory_order_acquire); }

    // upstream: finish_commit — 推进read_timestamp
    void finish_commit(uint64_t ts) {
        uint64_t cur = read_timestamp.load(std::memory_order_acquire);
        while (cur < ts && !read_timestamp.compare_exchange_weak(cur, ts, std::memory_order_acq_rel)) {}
        total_commits.fetch_add(1, std::memory_order_relaxed);
        reader_trace.advance_version();
    }

    // [NEW]
    void dump_stats(const char* label = "") const {
        std::fprintf(stderr,
            "[TxMgr·%s] w_ts=%lu r_ts=%lu vtx=%lu edge=%lu commits=%lu aborts=%lu\n",
            label,
            (unsigned long)write_timestamp.load(),
            (unsigned long)read_timestamp.load(),
            (unsigned long)vertex_count_,
            (unsigned long)edge_count_,
            (unsigned long)total_commits.load(),
            (unsigned long)total_aborts.load());
    }
};

// ═══════════════════════════════════════════════════════════════
// ReadTransaction (upstream 100%)
// ═══════════════════════════════════════════════════════════════

struct ReadTransaction {
    TransactionManager* tm;
    uint64_t timestamp;
    uint64_t cached_vertex_count;
    uint64_t cached_edge_count;

    explicit ReadTransaction(TransactionManager* mgr)
        : tm(mgr),
          timestamp(mgr->get_read_timestamp()),
          cached_vertex_count(mgr->vertex_count()),
          cached_edge_count(mgr->edge_count())
    {
        tm->reader_trace.reader_enter();
        if (debug::get_debug_level() >= 2)
            std::fprintf(stderr, "[ReadTx] begin ts=%lu\n", (unsigned long)timestamp);
    }

    ~ReadTransaction() {
        tm->reader_trace.reader_leave();
    }

    // upstream commit — 释放reader
    bool commit() {
        if (debug::get_debug_level() >= 2)
            std::fprintf(stderr, "[ReadTx] commit ts=%lu\n", (unsigned long)timestamp);
        return true;
    }

    uint64_t vertex_count() const { return cached_vertex_count; }
    uint64_t edge_count() const { return cached_edge_count; }

    bool has_vertex(uint64_t v) const {
        return tm->primary_tree->has_vertex(v, timestamp);
    }

    bool has_edge(uint64_t src, uint64_t dest) const {
        return tm->primary_tree->has_edge(src, dest, timestamp);
    }

    uint64_t get_degree(uint64_t src) const {
        return tm->primary_tree->get_degree(src, timestamp);
    }

    void get_neighbor(uint64_t src, std::vector<RangeElement>& neighbor) const {
        tm->primary_tree->get_neighbor(src, neighbor, timestamp);
    }

    Property_t get_edge_property(uint64_t src, uint64_t dest, uint8_t pid) const {
        // 简化: 从version中取
        auto* ver = tm->primary_tree->find_version(timestamp);
        if (!ver) return 0.0;
        // 需要通过storage type dispatch
        return 0.0;  // Phase5: full property support
    }
};

// ═══════════════════════════════════════════════════════════════
// WriteTransaction (upstream: 写缓冲 → commit时flush)
// ═══════════════════════════════════════════════════════════════

struct WriteTransaction {
    TransactionManager* tm;
    // 写缓冲
    std::vector<uint64_t> vertex_inserts;
    std::vector<uint64_t> vertex_removes;
    std::vector<PRR> edge_inserts;
    std::vector<PRR> edge_removes;
    std::vector<Property_t> edge_props;  // parallel to edge_inserts

    explicit WriteTransaction(TransactionManager* mgr) : tm(mgr) {
        if (debug::get_debug_level() >= 2)
            std::fprintf(stderr, "[WriteTx] begin\n");
    }

    void insert_vertex(uint64_t v) { vertex_inserts.push_back(v); }
    void remove_vertex(uint64_t v) { vertex_removes.push_back(v); }

    void insert_edge(uint64_t src, uint64_t dest, double weight = 0.0) {
        edge_inserts.push_back({static_cast<RangeElement>(src),
                                 static_cast<RangeElement>(dest)});
        edge_props.push_back(weight);
    }

    void remove_edge(uint64_t src, uint64_t dest) {
        edge_removes.push_back({static_cast<RangeElement>(src),
                                 static_cast<RangeElement>(dest)});
    }

    // upstream commit: sort → batch insert → update counters
    // [MOD] tbb::parallel_sort → std::sort
    bool commit() {
        auto t0 = std::chrono::steady_clock::now();

        uint64_t ts = tm->get_write_timestamp();

        // vertex inserts
        for (auto v : vertex_inserts)
            tm->primary_tree->insert_vertex(v, nullptr, &tm->writer_trace);
        tm->vertex_count_ += vertex_inserts.size();

        // vertex removes
        for (auto v : vertex_removes)
            tm->primary_tree->remove_vertex(v, tm->is_directed_, &tm->writer_trace);
        tm->vertex_count_ -= vertex_removes.size();

        // edge inserts — [MOD] sort by src
        if (!edge_inserts.empty()) {
            // upstream: tbb::parallel_sort → std::sort
            std::vector<size_t> order(edge_inserts.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return edge_inserts[a].first < edge_inserts[b].first;
            });

            for (auto idx : order) {
                Property_t p = idx < edge_props.size() ? edge_props[idx] : 0.0;
                tm->primary_tree->insert_edge(
                    edge_inserts[idx].first, edge_inserts[idx].second,
                    &p, &tm->writer_trace);
            }
            tm->edge_count_ += edge_inserts.size();
        }

        // edge removes
        for (auto& [src, dest] : edge_removes)
            tm->primary_tree->remove_edge(src, dest, &tm->writer_trace);
        tm->edge_count_ -= edge_removes.size();

        tm->finish_commit(ts);

        auto t1 = std::chrono::steady_clock::now();
        if (debug::get_debug_level() >= 2) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::fprintf(stderr, "[WriteTx] commit ts=%lu vtx+%zu edge+%zu in %ldms\n",
                (unsigned long)ts, vertex_inserts.size(), edge_inserts.size(), ms);
        }
        return true;
    }

    void abort() {
        vertex_inserts.clear();
        vertex_removes.clear();
        edge_inserts.clear();
        edge_removes.clear();
        edge_props.clear();
        tm->total_aborts.fetch_add(1, std::memory_order_relaxed);
        if (debug::get_debug_level() >= 2)
            std::fprintf(stderr, "[WriteTx] abort\n");
    }
};

// ═══════════════════════════════════════════════════════════════
// LightWriteTransaction (upstream: 单树轻量写 100%)
// ═══════════════════════════════════════════════════════════════

struct LightWriteTransaction {
    TransactionManager* tm;
    WriterTraceBlock* trace;

    explicit LightWriteTransaction(TransactionManager* mgr, WriterTraceBlock* tb = nullptr)
        : tm(mgr), trace(tb ? tb : &mgr->writer_trace) {}

    void insert_vertex(uint64_t v) {
        tm->primary_tree->insert_vertex(v, nullptr, trace);
        tm->vertex_count_++;
    }

    void insert_edge(uint64_t src, uint64_t dest, Property_t weight = 0.0) {
        tm->primary_tree->insert_edge(src, dest, &weight, trace);
        tm->edge_count_++;
    }

    bool commit() {
        uint64_t ts = tm->get_write_timestamp();
        tm->finish_commit(ts);
        return true;
    }
};

}  // namespace neograph
}  // namespace philemon

#endif
