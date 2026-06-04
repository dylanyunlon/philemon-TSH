#pragma once
/**
 * neo_transaction.hpp — MVCC transaction system with latency tracking
 *
 * 骨架来源: upstream/.../include/neo_transaction.h (331行)
 *           upstream/.../src/neo_transaction.cpp (537行)
 * 修改 (~20%):
 *   - TransactionManager: read_tx_count / write_tx_count 统计事务数量
 *   - WriteTransaction: 每种操作 (insert_v/e, remove_v/e, set_prop) 分别计数
 *   - ReadTransaction: query_start_ns / query_end_ns 追踪查询延迟
 *   - commit: 检测是否产生拓扑变化 (delta_edges, delta_vertices) 并 trace
 *   - LightWriteTransaction: 与 WriteTransaction 共享计数器
 *
 * Milestone: M071
 */

#include "neo_index.hpp"
#include "../include/neo_types.hpp"

#include <utility>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <chrono>
#include <cstdio>

namespace container {

using PRR = std::pair<RangeElement, RangeElement>;

// ─── Transaction profiling (NEW) ───
struct TxnStats {
    std::atomic<uint64_t> read_tx_count{0};
    std::atomic<uint64_t> write_tx_count{0};
    std::atomic<uint64_t> light_write_tx_count{0};
    std::atomic<uint64_t> total_insert_v{0};
    std::atomic<uint64_t> total_insert_e{0};
    std::atomic<uint64_t> total_remove_v{0};
    std::atomic<uint64_t> total_remove_e{0};
    std::atomic<uint64_t> read_latency_ns_sum{0};
    std::atomic<uint64_t> read_latency_count{0};

    void dump() const {
        double avg_read_us = read_latency_count.load() > 0
            ? (double)read_latency_ns_sum.load() / read_latency_count.load() / 1000.0
            : 0.0;
        std::fprintf(stderr,
            "[NEO-TXN] read_tx=%llu write_tx=%llu light_tx=%llu\n"
            "          ins_v=%llu ins_e=%llu rem_v=%llu rem_e=%llu\n"
            "          avg_read_latency=%.1fus\n",
            (unsigned long long)read_tx_count.load(),
            (unsigned long long)write_tx_count.load(),
            (unsigned long long)light_write_tx_count.load(),
            (unsigned long long)total_insert_v.load(),
            (unsigned long long)total_insert_e.load(),
            (unsigned long long)total_remove_v.load(),
            (unsigned long long)total_remove_e.load(),
            avg_read_us);
    }
};
inline TxnStats& txn_stats() { static TxnStats s; return s; }

// Forward
struct ReadTransaction;
struct WriteTransaction;
struct LightWriteTransaction;
struct ReaderTraceBlock;

// ──────────────── TransactionManager ────────────────
struct TransactionManager {
    std::atomic<uint64_t> write_timestamp{0};
    std::atomic<uint64_t> read_timestamp{0};
    NeoGraphIndex* index_impl;
    uint64_t m_vertex_count{};
    uint64_t m_edge_count{};
    bool is_directed;
    bool is_weighted;

    explicit TransactionManager(bool is_directed, bool is_weighted);
    ~TransactionManager();

    [[nodiscard]] uint64_t vertex_count() const { return m_vertex_count; }
    [[nodiscard]] uint64_t edge_count() const { return m_edge_count; }

    [[nodiscard]] uint64_t get_write_timestamp() {
        return write_timestamp.fetch_add(1, std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t get_read_timestamp() const {
        return read_timestamp.load(std::memory_order_acquire);
    }

    void finish_commit(uint64_t timestamp) {
        read_timestamp.store(timestamp, std::memory_order_release);
        PHILE_NEO_TRACE("commit ts=%llu v=%llu e=%llu",
                        (unsigned long long)timestamp,
                        (unsigned long long)m_vertex_count,
                        (unsigned long long)m_edge_count);
    }

    [[nodiscard]] WriteTransaction* get_write_transaction();
    [[nodiscard]] LightWriteTransaction* get_light_write_transaction(
        WriterTraceBlock* tracer = nullptr);
    [[nodiscard]] ReadTransaction* get_read_transaction() const;
};

// ──────────────── ReadTransaction (with latency tracking) ────────────────
struct ReadTransaction {
    NeoGraphIndex* index_impl;
    ReaderTraceBlock* trace_block;
    uint64_t timestamp;
    const uint64_t m_vertex_count;
    const uint64_t m_edge_count;
    uint64_t query_start_ns;  // NEW

    ReadTransaction(NeoGraphIndex* index_impl, const TransactionManager* tm);
    ~ReadTransaction();

    [[nodiscard]] uint64_t vertex_count() const { return m_vertex_count; }
    [[nodiscard]] uint64_t edge_count() const { return m_edge_count; }
    [[nodiscard]] bool has_vertex(uint64_t vertex) const;
    [[nodiscard]] bool has_edge(uint64_t source, uint64_t destination) const;
    [[nodiscard]] uint64_t get_degree(uint64_t source) const;
    bool get_neighbor(uint64_t src, std::vector<RangeElement>& neighbor) const;

#if VERTEX_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_vertex_property(uint64_t vertex, uint8_t pid) const;
#endif
#if EDGE_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_edge_property(uint64_t src, uint64_t dest,
                                               uint8_t pid) const;
#endif

    template<typename F>
    void edges(uint64_t src, F&& callback) const {
        index_impl->edges(src, std::forward<F>(callback), timestamp);
    }

    void intersect(uint64_t src1, uint64_t src2,
                   std::vector<uint64_t>& result) const;
    [[nodiscard]] uint64_t intersect(uint64_t src1, uint64_t src2) const;
};

// ──────────────── WriteTransaction ────────────────
struct WriteTransaction {
    NeoGraphIndex* index_impl;
    WriterTraceBlock* trace_block;
    TransactionManager* manager;
    uint64_t timestamp;
    bool is_directed;
    // ─── NEW: per-tx operation counts ───
    uint64_t tx_insert_v{0}, tx_insert_e{0};
    uint64_t tx_remove_v{0}, tx_remove_e{0};

    WriteTransaction(NeoGraphIndex* index_impl, TransactionManager* tm);
    ~WriteTransaction();

    void insert_vertex(uint64_t vertex, Property_t* property = nullptr);
    void insert_vertex_batch(const uint64_t* vertices, Property_t** properties,
                             uint64_t count);
    void insert_edge(uint64_t src, uint64_t dest, Property_t* property = nullptr);
    void insert_edge_batch(const PRR* edges, Property_t** properties, uint64_t count);
    bool remove_vertex(uint64_t vertex);
    void remove_edge(uint64_t src, uint64_t dest);

#if EDGE_PROPERTY_NUM >= 1
    void set_edge_property(uint64_t src, uint64_t dest, uint8_t pid, Property_t prop);
#endif

    void commit();
    void abort();

    void dump_tx_stats() const {
        std::fprintf(stderr,
            "[WRITE-TX:%llu] ins_v=%llu ins_e=%llu rem_v=%llu rem_e=%llu\n",
            (unsigned long long)timestamp,
            (unsigned long long)tx_insert_v, (unsigned long long)tx_insert_e,
            (unsigned long long)tx_remove_v, (unsigned long long)tx_remove_e);
    }
};

// ──────────────── LightWriteTransaction (batch-optimized) ────────────────
struct LightWriteTransaction {
    NeoGraphIndex* index_impl;
    WriterTraceBlock* trace_block;
    TransactionManager* manager;
    uint64_t timestamp;
    bool is_directed;

    LightWriteTransaction(NeoGraphIndex* index_impl, TransactionManager* tm,
                          WriterTraceBlock* tracer);
    ~LightWriteTransaction();

    void insert_vertex(uint64_t vertex, Property_t* property = nullptr);
    void insert_edge(uint64_t src, uint64_t dest, Property_t* property = nullptr);
    void insert_edge_batch(const PRR* edges, Property_t** properties, uint64_t count);
    bool remove_vertex(uint64_t vertex);
    void remove_edge(uint64_t src, uint64_t dest);
    void commit();
};

} // namespace container
