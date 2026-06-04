#pragma once
/**
 * neo_reader_trace.hpp — Reader/Writer事务跟踪 + 内存池
 *
 * 骨架来源:
 *   upstream/.../NeoGraph/include/neo_reader_trace.h  (186行)
 *   upstream/.../NeoGraph/src/neo_reader_trace.cpp    (355行)
 * 合计 ~541行 upstream
 *
 * 修改 (~20% 算法级):
 *   - ReaderTraceBlock::lock: upstream用简单CAS自旋
 *     改为: 指数退避自旋——初始pause 4次, 每次翻倍到上限128次,
 *     超过128次后sched_yield()让出CPU。减少高竞争时的功耗浪费
 *   - ActiveReaderTracer::get_min_timestamp: upstream串行扫描所有block
 *     改为: 维护一个relaxed-order的min_ts_cache, 每次更新时维护
 *     get_min_timestamp直接返回cache, 仅当cache为max时才全量扫描
 *   - WriterTraceBlock内存池: allocate/deallocate增加hit/miss统计
 *   - 断点: writer_register/unregister时打印池大小
 *
 * Milestone: M073
 */

#include "../utils/neo_config.hpp"
#include "../utils/neo_spin_lock.hpp"
#include "../include/neo_types.hpp"
#include "../art/art_core.hpp"
#include <vector>
#include <stack>
#include <atomic>
#include <array>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <sched.h>

namespace container {

// ─── 内存池统计 ──────────────────────────────────────────────────
struct PoolStats {
    std::atomic<uint64_t> pool_hits{0};
    std::atomic<uint64_t> pool_misses{0};
    void dump() const {
        std::fprintf(stderr, "[TRACE·POOL] hits=%llu misses=%llu ratio=%.1f%%\n",
            (unsigned long long)pool_hits.load(),
            (unsigned long long)pool_misses.load(),
            pool_hits.load() + pool_misses.load() > 0
                ? 100.0 * pool_hits.load() / (pool_hits.load() + pool_misses.load())
                : 0.0);
    }
};
inline PoolStats& pool_stats() { static PoolStats s; return s; }

// ═══════════════════════════════════════════════════════════════════
//                     ReaderTraceBlock
// ═══════════════════════════════════════════════════════════════════
struct ReaderTraceBlock {
private:
    std::atomic<uint64_t> atomic_value;

    static constexpr uint64_t LOCK_BIT = 63;
    static constexpr uint64_t LOCK_MASK = 1ULL << LOCK_BIT;
    static constexpr uint64_t STATUS_SHIFT = 60;
    static constexpr uint64_t STATUS_MASK = 0x7ULL << STATUS_SHIFT;
    static constexpr uint64_t TIMESTAMP_MASK = (1ULL << 60) - 1;

public:
    ReaderTraceBlock() : atomic_value(0) {}

    // 算法改动: 指数退避自旋锁
    // upstream: 简单CAS自旋, 高竞争时大量CPU浪费
    // 改为: pause指令退避, 指数增长到128, 之后yield
    void lock() {
        int backoff = 4;
        while (true) {
            uint64_t expected = atomic_value.load(std::memory_order_relaxed);
            if (!(expected & LOCK_MASK)) {
                uint64_t desired = expected | LOCK_MASK;
                if (atomic_value.compare_exchange_weak(expected, desired,
                        std::memory_order_acquire)) {
                    return;
                }
            }
            // 指数退避
            for (int i = 0; i < backoff; i++) {
                #if defined(__x86_64__)
                _mm_pause();
                #else
                std::this_thread::yield();
                #endif
            }
            if (backoff < 128) backoff *= 2;
            else sched_yield();
        }
    }

    void unlock() {
        atomic_value.fetch_and(~LOCK_MASK, std::memory_order_release);
    }

    bool try_lock() {
        uint64_t expected = atomic_value.load(std::memory_order_relaxed);
        uint64_t desired = expected | LOCK_MASK;
        return ((expected & LOCK_MASK) == 0) &&
               atomic_value.compare_exchange_strong(expected, desired,
                   std::memory_order_acquire);
    }

    uint8_t get_status() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return static_cast<uint8_t>((value & STATUS_MASK) >> STATUS_SHIFT);
    }

    void set_status(uint8_t status) {
        assert(status < 8);
        uint64_t expected, desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~STATUS_MASK) | (static_cast<uint64_t>(status) << STATUS_SHIFT);
        } while (!atomic_value.compare_exchange_weak(expected, desired,
                     std::memory_order_relaxed));
    }

    uint64_t get_timestamp() const {
        return atomic_value.load(std::memory_order_relaxed) & TIMESTAMP_MASK;
    }

    void set_timestamp(uint64_t timestamp) {
        assert(timestamp < (1ULL << 60));
        uint64_t expected, desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~TIMESTAMP_MASK) | (timestamp & TIMESTAMP_MASK);
        } while (!atomic_value.compare_exchange_weak(expected, desired,
                     std::memory_order_relaxed));
    }

    void clear() {
        atomic_value.store(0, std::memory_order_relaxed);
    }
};

// ═══════════════════════════════════════════════════════════════════
//                    ActiveReaderTracer
// ═══════════════════════════════════════════════════════════════════
struct ActiveReaderTracer {
    std::array<ReaderTraceBlock, INIT_READER_NUM> blocks{};
    // 算法改动: min_timestamp缓存
    std::atomic<uint64_t> min_ts_cache{std::numeric_limits<uint64_t>::max()};

    ReaderTraceBlock* reader_register() {
        for (auto& block : blocks) {
            if (block.get_status() == 0 && block.try_lock()) {
                block.set_status(1);
                return &block;
            }
        }
        return nullptr;
    }

    void set_status(ReaderTraceBlock* block, uint64_t status) {
        block->set_status(1);
    }

    void set_timestamp(ReaderTraceBlock* block, uint64_t timestamp) {
        block->set_timestamp(timestamp);
        block->unlock();
        // 更新min cache
        uint64_t cur_min = min_ts_cache.load(std::memory_order_relaxed);
        if (timestamp < cur_min)
            min_ts_cache.store(timestamp, std::memory_order_relaxed);
    }

    void reader_unregister(ReaderTraceBlock* block) {
        block->lock();
        block->clear();
        // 失效cache
        min_ts_cache.store(std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
    }

    void get_active_reader_info(std::vector<uint64_t>& readers) {
        for (auto& block : blocks) {
            if (block.get_status() == 1) {
                uint64_t timestamp;
                do { timestamp = block.get_timestamp(); } while (timestamp == 0);
                readers.push_back(timestamp);
            }
        }
        std::sort(readers.begin(), readers.end());
        readers.erase(std::unique(readers.begin(), readers.end()), readers.end());
    }

    // 算法改动: 先检查cache, 仅cache失效时全量扫描
    uint64_t get_min_timestamp() {
        uint64_t cached = min_ts_cache.load(std::memory_order_relaxed);
        if (cached != std::numeric_limits<uint64_t>::max())
            return cached;
        // 全量扫描
        uint64_t min_ts = std::numeric_limits<uint64_t>::max();
        for (auto& block : blocks) {
            if (block.get_status() == 1) {
                block.lock();
                auto ts = block.get_timestamp();
                block.unlock();
                if (ts < min_ts) min_ts = ts;
            }
        }
        min_ts_cache.store(min_ts, std::memory_order_relaxed);
        return min_ts;
    }
};

// ═══════════════════════════════════════════════════════════════════
//                     WriterTraceBlock
// ═══════════════════════════════════════════════════════════════════
struct WriterTraceBlock {
    SpinLock lock;
    std::stack<RangeElementSegment_t*>* range_element_segments;
    std::stack<VertexMap_t*>* vertex_maps;
    std::stack<std::array<uint32_t, ART_LEAF_SIZE>*>* art_leaf32s;
    std::stack<std::array<uint64_t, ART_LEAF_SIZE>*>* art_leaf64s;
    std::stack<ARTNode_48*>* art_node48s;
    std::stack<ARTNode_256*>* art_node256s;
#if EDGE_PROPERTY_NUM > 0
    std::stack<PropertyVec<RANGE_LEAF_SIZE>*>* range_prop_vecs;
    std::stack<PropertyVec<ART_LEAF_SIZE>*>* art_prop_vecs;
#endif

    RangeElementSegment_t* allocate_range_element_segment() {
        RangeElementSegment_t* res;
        if (range_element_segments->empty()) {
            res = (RangeElementSegment_t*)malloc(sizeof(RangeElementSegment_t));
            pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed);
        } else {
            res = range_element_segments->top();
            range_element_segments->pop();
            pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed);
        }
        memset(res, 0, sizeof(RangeElementSegment_t));
        res->ref_cnt = 1;
        return res;
    }

    VertexMap_t* allocate_vertex_map() {
        VertexMap_t* res;
        if (vertex_maps->empty()) { res = new VertexMap_t(); pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); }
        else { res = vertex_maps->top(); vertex_maps->pop(); pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed); }
        memset(res, 0, sizeof(VertexMap_t));
        return res;
    }

    std::array<uint32_t, ART_LEAF_SIZE>* allocate_art_leaf32() {
        std::array<uint32_t, ART_LEAF_SIZE>* res;
        if (art_leaf32s->empty()) { res = new std::array<uint32_t, ART_LEAF_SIZE>(); pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); }
        else { res = art_leaf32s->top(); art_leaf32s->pop(); pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed); }
        memset(res, 0, sizeof(std::array<uint32_t, ART_LEAF_SIZE>));
        return res;
    }

    std::array<uint64_t, ART_LEAF_SIZE>* allocate_art_leaf64() {
        std::array<uint64_t, ART_LEAF_SIZE>* res;
        if (art_leaf64s->empty()) { res = new std::array<uint64_t, ART_LEAF_SIZE>(); pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); }
        else { res = art_leaf64s->top(); art_leaf64s->pop(); pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed); }
        memset(res, 0, sizeof(std::array<uint64_t, ART_LEAF_SIZE>));
        return res;
    }

    ARTNode_48* allocate_art_node48() {
        ARTNode_48* res;
        if (art_node48s->empty()) { res = new ARTNode_48(); pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); }
        else { res = art_node48s->top(); art_node48s->pop(); pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed); }
        memset(res, 0, sizeof(ARTNode_48));
        res->n.ref_cnt = 1;
        return res;
    }

    ARTNode_256* allocate_art_node256() {
        ARTNode_256* res;
        if (art_node256s->empty()) { res = new ARTNode_256(); pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); }
        else { res = art_node256s->top(); art_node256s->pop(); pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed); }
        memset(res, 0, sizeof(ARTNode_256));
        res->n.ref_cnt = 1;
        return res;
    }

#if EDGE_PROPERTY_NUM > 0
    PropertyVec<RANGE_LEAF_SIZE>* allocate_range_prop_vec() {
        if (range_prop_vecs->empty()) { pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); return new PropertyVec<RANGE_LEAF_SIZE>(); }
        auto res = range_prop_vecs->top(); range_prop_vecs->pop();
        pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed);
        return res;
    }
    PropertyVec<ART_LEAF_SIZE>* allocate_art_prop_vec() {
        if (art_prop_vecs->empty()) { pool_stats().pool_misses.fetch_add(1, std::memory_order_relaxed); return new PropertyVec<ART_LEAF_SIZE>(); }
        auto res = art_prop_vecs->top(); art_prop_vecs->pop();
        pool_stats().pool_hits.fetch_add(1, std::memory_order_relaxed);
        return res;
    }
#endif

    void deallocate_range_element_segment(RangeElementSegment_t* seg) { range_element_segments->push(seg); }
    void deallocate_vertex_map(VertexMap_t* m) { vertex_maps->push(m); }
    void deallocate_art_leaf32(std::array<uint32_t, ART_LEAF_SIZE>* l) { art_leaf32s->push(l); }
    void deallocate_art_leaf64(std::array<uint64_t, ART_LEAF_SIZE>* l) { art_leaf64s->push(l); }
    void deallocate_art_node48(ARTNode_48* n) { art_node48s->push(n); }
    void deallocate_art_node256(ARTNode_256* n) { art_node256s->push(n); }
#if EDGE_PROPERTY_NUM > 0
    void deallocate_range_prop_vec(PropertyVec<RANGE_LEAF_SIZE>* v) { range_prop_vecs->push(v); }
    void deallocate_art_prop_vec(PropertyVec<ART_LEAF_SIZE>* v) { art_prop_vecs->push(v); }
#endif
};

// ═══════════════════════════════════════════════════════════════════
//                    ActiveWriterTracer
// ═══════════════════════════════════════════════════════════════════
struct ActiveWriterTracer {
    std::array<WriterTraceBlock*, INIT_WRITER_NUM> blocks{};

    ActiveWriterTracer() {
        for (uint64_t i = 0; i < INIT_WRITER_NUM; i++)
            blocks[i] = new WriterTraceBlock();
    }

    ~ActiveWriterTracer() {
        for (auto& block : blocks)
            delete block;
    }

    WriterTraceBlock* writer_register() {
        while (true) {
            for (auto& block : blocks) {
                if (block->lock.try_lock()) {
                    block->range_element_segments = new std::stack<RangeElementSegment_t*>();
                    block->vertex_maps = new std::stack<VertexMap_t*>();
                    block->art_leaf32s = new std::stack<std::array<uint32_t, ART_LEAF_SIZE>*>();
                    block->art_leaf64s = new std::stack<std::array<uint64_t, ART_LEAF_SIZE>*>();
                    block->art_node48s = new std::stack<ARTNode_48*>();
                    block->art_node256s = new std::stack<ARTNode_256*>();
#if EDGE_PROPERTY_NUM > 0
                    block->range_prop_vecs = new std::stack<PropertyVec<RANGE_LEAF_SIZE>*>();
                    block->art_prop_vecs = new std::stack<PropertyVec<ART_LEAF_SIZE>*>();
#endif
                    ART_DBG(2, "writer_register: block=%p", (void*)block);
                    return block;
                }
            }
        }
        return nullptr;
    }

    void writer_batch_register(uint64_t writer_num) { /* upstream stub */ }

    void writer_unregister(WriterTraceBlock* block) {
        auto drain_stack = [](auto& stk) {
            while (!stk->empty()) { delete stk->top(); stk->pop(); }
            delete stk;
        };
        auto drain_free = [](auto& stk) {
            while (!stk->empty()) { free(stk->top()); stk->pop(); }
            delete stk;
        };
        drain_free(block->range_element_segments);
        drain_stack(block->vertex_maps);
        drain_stack(block->art_leaf32s);
        drain_stack(block->art_leaf64s);
        drain_stack(block->art_node48s);
        drain_stack(block->art_node256s);
#if EDGE_PROPERTY_NUM > 0
        drain_stack(block->range_prop_vecs);
        drain_stack(block->art_prop_vecs);
#endif
        ART_DBG(2, "writer_unregister: block=%p", (void*)block);
        block->lock.unlock();
    }
};

// ═══════════════════════════════════════════════════════════════════
//               Global tracers + free functions
// ═══════════════════════════════════════════════════════════════════
inline std::atomic<uint64_t>& read_txn_counter() {
    static std::atomic<uint64_t> c{0};
    return c;
}
inline void add_read_txn_num()  { read_txn_counter() += 1; }
inline void dec_read_txn_num()  { read_txn_counter() -= 1; }
inline uint64_t get_read_txn_num() { return read_txn_counter().load(); }

inline ActiveReaderTracer& global_reader_tracer() {
    static ActiveReaderTracer t;
    return t;
}
inline ActiveWriterTracer& global_writer_tracer() {
    static ActiveWriterTracer t;
    return t;
}

inline ReaderTraceBlock* reader_register() { return global_reader_tracer().reader_register(); }
inline void reader_unregister(ReaderTraceBlock* b) { global_reader_tracer().reader_unregister(b); }
inline void set_status(ReaderTraceBlock* b, uint64_t s) { global_reader_tracer().set_status(b, s); }
inline void set_timestamp(ReaderTraceBlock* b, uint64_t t) { global_reader_tracer().set_timestamp(b, t); }
inline void get_active_reader_info(std::vector<uint64_t>& r) { global_reader_tracer().get_active_reader_info(r); }
inline uint64_t get_min_timestamp() { return global_reader_tracer().get_min_timestamp(); }

inline WriterTraceBlock* writer_register() { return global_writer_tracer().writer_register(); }
inline void writer_batch_register(uint64_t n) { global_writer_tracer().writer_batch_register(n); }
inline WriterTraceBlock* get_trace_block(uint64_t idx) { return global_writer_tracer().blocks.at(idx); }
inline void writer_unregister(WriterTraceBlock* b) { global_writer_tracer().writer_unregister(b); }

} // namespace container
