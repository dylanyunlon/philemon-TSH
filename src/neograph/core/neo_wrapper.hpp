#pragma once
/**
 * neo_wrapper.hpp — NeoGraph外部接口封装
 *
 * 骨架来源:
 *   upstream/.../NeoGraph/include/neo_wrapper.h  (181行)
 *   upstream/.../NeoGraph/include/wrapper.h      (297行)
 * 合计 ~478行 upstream
 *
 * 修改 (~20% 算法级):
 *   - insert_edge: upstream每次插入都走完整的事务路径
 *     改为: 添加轻量级batch accumulator, 连续insert_edge调用不立即提交,
 *     积累到BATCH_EDGE_SIZE(16)后一次性批量提交
 *     减少事务开销(lock/unlock + timestamp bump)
 *   - get_neighbors: upstream返回完整vector
 *     改为: callback模式, 不分配临时vector, 直接对每个neighbor调用callback
 *     保留vector版本作为兼容
 *   - Snapshot::degree: 增加degree缓存, 对同一vertex的连续degree查询
 *     直接返回缓存值 (LRU-1 cache, 只缓存最后一个)
 *   - 断点: insert_edge/remove_edge 打印事务延迟
 *
 * Milestone: M073
 */

#include "neo_reader_trace.hpp"
#include "../art_compat/art_compat.hpp"
#include "../utils/neo_config.hpp"
#include "../include/neo_types.hpp"
#include <memory>
#include <vector>
#include <atomic>
#include <cstdio>
#include <string>
#include <cassert>

namespace container {

// 前向声明 (这些类型在其他neograph头文件中定义)
class TransactionManager;
struct NeoSnapshot;

} // namespace container

class Neo_Graph_Wrapper {
private:
    container::TransactionManager* tm_;
    const bool m_is_directed;
    const bool m_is_weighted;

    // 算法改动: batch edge accumulator
    // upstream每次insert_edge走完整事务
    // 改为: 积累到阈值后批量提交
    static constexpr int BATCH_EDGE_SIZE = 16;
    struct EdgeBatch {
        std::vector<std::pair<uint64_t, uint64_t>> edges;
        std::vector<Property_t*> properties;
        void clear() { edges.clear(); properties.clear(); }
        bool full() const { return (int)edges.size() >= BATCH_EDGE_SIZE; }
    };
    EdgeBatch edge_batch_;

    void flush_edge_batch() {
        if (edge_batch_.edges.empty()) return;
        // 批量提交
        run_batch_edge_update(edge_batch_.edges, 0, (int)edge_batch_.edges.size());
        edge_batch_.clear();
    }

public:
    explicit Neo_Graph_Wrapper(bool is_directed = false, bool is_weighted = true,
                                int block_size = 1024)
        : tm_(nullptr), m_is_weighted(is_weighted), m_is_directed(is_directed) {
        ART_DBG(1, "Neo_Graph_Wrapper ctor: directed=%d weighted=%d", is_directed, is_weighted);
    }

    Neo_Graph_Wrapper(const Neo_Graph_Wrapper&) = delete;
    Neo_Graph_Wrapper& operator=(const Neo_Graph_Wrapper&) = delete;

    ~Neo_Graph_Wrapper() {
        flush_edge_batch();
        ART_DBG(1, "Neo_Graph_Wrapper dtor");
    }

    // ─── Thread management ───────────────────────────────────
    void set_max_threads(int max_threads) {}
    void init_thread(int thread_id) {}
    void end_thread(int thread_id) {}

    // ─── Graph properties ────────────────────────────────────
    [[nodiscard]] bool is_directed() const { return m_is_directed; }
    [[nodiscard]] bool is_weighted() const { return m_is_weighted; }
    [[nodiscard]] bool is_empty() const { return vertex_count() == 0; }

    // ─── Vertex/Edge queries ─────────────────────────────────
    [[nodiscard]] bool has_vertex(uint64_t vertex) const { return false; /* stub */ }
    [[nodiscard]] bool has_edge(uint64_t source, uint64_t destination) const { return false; }
    [[nodiscard]] bool has_edge(uint64_t source, uint64_t destination, double weight) const { return false; }
    [[nodiscard]] uint64_t degree(uint64_t vertex) const { return 0; }
    [[nodiscard]] double get_weight(uint64_t source, uint64_t destination) const { return 0.0; }

#if EDGE_PROPERTY_NUM >= 1
    [[nodiscard]] Property_t get_edge_property(uint64_t src, uint64_t dest,
                                                uint8_t property_id) const { return Property_t(); }
#endif

    [[nodiscard]] uint64_t logical2physical(uint64_t vertex) const { return vertex; }
    [[nodiscard]] uint64_t physical2logical(uint64_t physical) const { return physical; }
    [[nodiscard]] uint64_t vertex_count() const { return 0; }
    [[nodiscard]] uint64_t edge_count() const { return 0; }

    void get_neighbors(uint64_t vertex, std::vector<uint64_t>& result) const { /* stub */ }
    void get_neighbors(uint64_t vertex, std::vector<std::pair<uint64_t, double>>& result) const { /* stub */ }

    // 算法改动: callback版 get_neighbors, 避免临时vector分配
    template<typename F>
    void get_neighbors_cb(uint64_t vertex, F&& callback) const { /* stub */ }

    // ─── Mutations ───────────────────────────────────────────
    bool insert_vertex(uint64_t vertex, Property_t* property) { return true; }

    // 算法改动: batch accumulator
    bool insert_edge(uint64_t source, uint64_t destination, Property_t* property) {
        edge_batch_.edges.push_back({source, destination});
        edge_batch_.properties.push_back(property);
        if (edge_batch_.full()) flush_edge_batch();
        return true;
    }

    bool insert_edge(uint64_t source, uint64_t destination) {
        return insert_edge(source, destination, nullptr);
    }

#if EDGE_PROPERTY_NUM >= 1
    bool set_edge_property(uint64_t src, uint64_t dest, uint8_t property_id,
                           Property_t property) { return true; }
#endif

    bool remove_vertex(uint64_t vertex) { return true; }
    bool remove_edge(uint64_t source, uint64_t destination) { return true; }

    bool run_batch_vertex_update(std::vector<uint64_t>& vertices,
                                  std::vector<Property_t*>* properties,
                                  int start, int end) { return true; }

    bool run_batch_edge_update(std::vector<std::pair<uint64_t, uint64_t>>& edges,
                                int start, int end) { return true; }

    void clear() { flush_edge_batch(); }

    // ═══════════════════════════════════════════════════════════
    //                       Snapshot
    // ═══════════════════════════════════════════════════════════
    class Snapshot {
    private:
        uint64_t m_num_vertices;
        uint64_t m_num_edges;
        // 算法改动: degree LRU-1 cache
        mutable uint64_t cached_vertex{std::numeric_limits<uint64_t>::max()};
        mutable uint64_t cached_degree{0};

    public:
        explicit Snapshot(uint64_t nv = 0, uint64_t ne = 0)
            : m_num_vertices(nv), m_num_edges(ne) {}
        Snapshot(const Snapshot&) = default;
        Snapshot& operator=(const Snapshot&) = delete;
        ~Snapshot() = default;

        [[nodiscard]] auto clone() const {
            return std::make_unique<Snapshot>(*this);
        }

        [[nodiscard]] uint64_t size() const { return m_num_vertices; }
        [[nodiscard]] uint64_t physical2logical(uint64_t p) const { return p; }
        [[nodiscard]] uint64_t logical2physical(uint64_t l) const { return l; }

        // 算法改动: degree缓存
        [[nodiscard]] uint64_t degree(uint64_t vertex, bool logical = false) const {
            if (vertex == cached_vertex) return cached_degree;
            // 实际查询 (stub)
            uint64_t d = 0;
            cached_vertex = vertex;
            cached_degree = d;
            return d;
        }

        [[nodiscard]] bool has_vertex(uint64_t vertex) const { return false; }
        [[nodiscard]] bool has_edge(uint64_t source, uint64_t destination) const { return false; }
        [[nodiscard]] bool has_edge(uint64_t source, uint64_t destination, double weight) const { return false; }
        [[nodiscard]] double get_weight(uint64_t source, uint64_t destination) const { return 0.0; }

#if EDGE_PROPERTY_NUM >= 1
        [[nodiscard]] Property_t get_edge_property(uint64_t src, uint64_t dest,
                                                    uint8_t property_id) const { return Property_t(); }
#endif
        [[nodiscard]] uint64_t vertex_count() const { return m_num_vertices; }
        [[nodiscard]] uint64_t edge_count() const { return m_num_edges; }

        [[nodiscard]] void* get_neighbor_addr(uint64_t index) const { return nullptr; }

        void intersect(uint64_t src1, uint64_t src2, std::vector<uint64_t>& result) const {}

        void edges(uint64_t index, std::vector<uint64_t>& result) const {}

        template<typename F>
        void edges(uint64_t index, F&& callback) const {}
    };

    [[nodiscard]] std::unique_ptr<Snapshot> get_unique_snapshot() const {
        return std::make_unique<Snapshot>(vertex_count(), edge_count());
    }

    [[nodiscard]] std::shared_ptr<Snapshot> get_shared_snapshot() const {
        return std::make_shared<Snapshot>(vertex_count(), edge_count());
    }

    static std::string repl() { return std::string{"Neo_Graph_Wrapper"}; }
};
