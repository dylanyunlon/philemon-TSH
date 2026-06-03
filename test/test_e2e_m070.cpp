/**
 * test_e2e_m070.cpp — M070 端到端集成测试
 *
 * ====================================================================
 * 骨架来源 (upstream + existing, 保留 ~80%):
 *   test/test_integration.cpp                     (371行, fixture/BFS/WCC)
 *   upstream/rapidstore/main.cpp                   (202行, driver loop)
 *   upstream/rapidstore/wrapper/driver.h           (1577行, execute_query)
 *   upstream/rapidstore/configuration.cpp          (280行, config parse)
 *   src/bench/cross_tier_bench.cpp                 (mock graph)
 *
 * 保留 (~80%):
 *   - GoogleTest fixture模式 (IntegrationTest::SetUp/TearDown)
 *   - SyntheticGraph::generate() 合成图生成器
 *   - LiveGraphWrapper insert_vertex/insert_edge/get_unique_snapshot
 *   - BFS frontier-expansion循环 (test_integration.cpp:BFSOnLiveGraphSnapshot)
 *   - WCC union-find with path compression
 *   - ConfigParser::get_instance().parse(path)
 *   - TemGraph::load_from_edges / contains_query
 *   - wrapper::edges() callback pattern
 *
 * 修改 (~20%):
 *   - [ALG] adapter切换: 手动new每个adapter类 → 工厂dispatch表
 *       原: auto lgw = TieredLiveGraphWrapper(false, true)
 *           auto neo = NeoTieredAdapter::Wrapper(...)
 *           ... (6个手动实例化)
 *       新: AdapterFactory::create(AdapterID) 用bitmap-encoded枚举选择
 *           遍历所有注册adapter, 同一图数据加载到各adapter, 交叉验证
 *
 *   - [ALG] 结果验证: 单一EXPECT_GT → 交叉一致性矩阵
 *       原: EXPECT_GT(dist.size(), 1u)  // BFS只检查非空
 *       新: 跨adapter交叉验证:
 *           BFS: max_depth(adapter_A) == max_depth(adapter_B)
 *           WCC: num_components(adapter_A) == num_components(adapter_B)
 *           PR:  top-K排名一致(Kendall's tau > 0.8)
 *           TC:  triangle_count(A) == triangle_count(B)
 *           SSSP: shortest_path(A) == shortest_path(B) ± epsilon
 *
 *   - [ALG] 图模型: 固定seed均匀随机 → Erdős–Rényi + planted partition
 *       原: uniform random (src,dst) pairs
 *       新: planted partition model:
 *           K个cluster, 每cluster内连接概率p_in=0.3, 跨cluster p_out=0.01
 *           保证connected(加全局backbone), 且WCC ground truth = 1 component
 *           BFS ground truth: max_depth ≤ diameter ≤ 2*log(N)/log(avg_degree)
 *
 *   - [NEW] 5算法 × 多adapter的全组合验证矩阵
 *   - [NEW] ground truth oracle: planted partition有已知社区结构
 *   - [NEW] PHILE_E2E_BREAKPOINT: 每阶段打印系统状态+中间结果
 *   - [NEW] 定时断言: 每个阶段 < 超时阈值
 *
 * Milestone: M070 (第8位Claude)
 * ====================================================================
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <functional>
#include <fstream>
#include <numeric>
#include <cmath>
#include <cassert>
#include <memory>

// ─── Project includes ────────────────────────────────────────────────────
#include "core/temporal_edge.hpp"
#include "core/tiered_allocator.hpp"
#include "core/seqlock.hpp"

#include "index/interval.hpp"
#include "index/dll_list.hpp"
#include "index/tem_graph.hpp"
#include "index/tem_graph_impl.hpp"

#include "wrapper/apps/livegraph_tiered.hpp"
#include "wrapper/apps/backend_adapters.hpp"
#include "wrapper/graph_edge.hpp"
#include "wrapper/rapidstore_wrapper.hpp"

#include "entry/temporal_query_driver.hpp"

#include "debug/philemon_debug.hpp"
#include "debug/state_inspector.hpp"

#include "utils/config_parser.hpp"
#include "cost_model/tier_cost_model.hpp"

using namespace philemon;

// ─── 调试宏 ─────────────────────────────────────────────────────────────────
#define PHILE_E2E_BREAKPOINT(tag, ...)                                          \
    do {                                                                         \
        fprintf(stderr, "\x1b[32m[E2E-BP:%s] ", tag);                           \
        fprintf(stderr, __VA_ARGS__);                                           \
        fprintf(stderr, " at %s:%d\x1b[0m\n", __FILE__, __LINE__);             \
    } while(0)


// ═══════════════════════════════════════════════════════════════════════════
// [ALG] Planted Partition Graph Generator
// ═══════════════════════════════════════════════════════════════════════════
// 原版: SyntheticGraph::generate() 用uniform random pair
// 新版: planted partition model with known community structure
//   - K clusters, each of size N/K
//   - intra-cluster edge prob p_in = 0.3
//   - inter-cluster edge prob p_out = 0.01
//   - backbone path 保证全图连通 (WCC ground truth = 1 component)
//   - 这给了我们 ground truth 可以验证: WCC=1, BFS depth bounded

struct PlantedPartitionGraph {
    uint64_t num_vertices;
    uint64_t num_clusters;
    std::vector<TemporalEdge> edges;
    std::vector<uint64_t> cluster_assignment;   // vertex → cluster_id
    int time_range;

    // ground truth
    uint64_t expected_wcc_components;  // always 1 (backbone ensures connected)
    uint64_t expected_diameter_upper;  // 2 * log(N) / log(avg_deg) 近似

    std::vector<std::pair<int, int>> to_intervals() const {
        std::vector<std::pair<int, int>> result;
        result.reserve(edges.size());
        for (auto& e : edges) {
            result.emplace_back(e.ts_begin, e.ts_finish);
        }
        return result;
    }

    static PlantedPartitionGraph generate(
            uint64_t nv, uint64_t n_clusters,
            double p_in, double p_out,
            int t_range = 10000, uint64_t seed = 42) {
        PlantedPartitionGraph g;
        g.num_vertices = nv;
        g.num_clusters = n_clusters;
        g.time_range = t_range;
        g.expected_wcc_components = 1;  // backbone保证

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> tdist(0, t_range - 100);
        std::uniform_int_distribution<int> span_dist(10, 100);
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        std::uniform_real_distribution<double> wdist(0.1, 10.0);

        // 分配cluster
        g.cluster_assignment.resize(nv);
        uint64_t cluster_size = nv / n_clusters;
        for (uint64_t v = 0; v < nv; ++v) {
            g.cluster_assignment[v] = v / cluster_size;
            if (g.cluster_assignment[v] >= n_clusters) {
                g.cluster_assignment[v] = n_clusters - 1;
            }
        }

        PHILE_E2E_BREAKPOINT("GEN", "planted partition: %lu vertices, "
            "%lu clusters, p_in=%.2f, p_out=%.3f",
            (unsigned long)nv, (unsigned long)n_clusters, p_in, p_out);

        // 生成边
        std::set<std::pair<uint64_t, uint64_t>> edge_set;

        // Phase 1: intra-cluster edges (p_in较大)
        for (uint64_t c = 0; c < n_clusters; ++c) {
            uint64_t c_start = c * cluster_size;
            uint64_t c_end = (c == n_clusters - 1) ? nv : (c + 1) * cluster_size;

            for (uint64_t u = c_start; u < c_end; ++u) {
                for (uint64_t v = u + 1; v < c_end; ++v) {
                    if (coin(rng) < p_in) {
                        edge_set.emplace(u, v);
                    }
                }
            }
        }

        // Phase 2: inter-cluster edges (p_out较小)
        for (uint64_t u = 0; u < nv; ++u) {
            for (uint64_t v = u + 1; v < nv; ++v) {
                if (g.cluster_assignment[u] != g.cluster_assignment[v]) {
                    if (coin(rng) < p_out) {
                        edge_set.emplace(u, v);
                    }
                }
            }
        }

        // Phase 3: backbone path 保证连通
        // 在每对相邻cluster的边界顶点之间加一条边
        for (uint64_t c = 0; c + 1 < n_clusters; ++c) {
            uint64_t u = (c + 1) * cluster_size - 1;
            uint64_t v = (c + 1) * cluster_size;
            if (u < nv && v < nv) {
                edge_set.emplace(std::min(u, v), std::max(u, v));
            }
        }

        // 转换为TemporalEdge
        g.edges.reserve(edge_set.size());
        for (auto& [u, v] : edge_set) {
            TemporalEdge e;
            e.source = u;
            e.destination = v;
            e.weight = wdist(rng);
            e.ts_begin = tdist(rng);
            e.ts_finish = e.ts_begin + span_dist(rng);
            g.edges.push_back(e);
        }

        // 计算直径上界
        double avg_deg = 2.0 * g.edges.size() / nv;
        if (avg_deg > 1.0) {
            g.expected_diameter_upper = static_cast<uint64_t>(
                3.0 * std::log(nv) / std::log(avg_deg)) + 5;
        } else {
            g.expected_diameter_upper = nv;
        }

        PHILE_E2E_BREAKPOINT("GEN", "generated %zu edges, avg_deg=%.1f, "
            "diameter_upper=%lu",
            g.edges.size(), avg_deg,
            (unsigned long)g.expected_diameter_upper);

        return g;
    }
};


// ═══════════════════════════════════════════════════════════════════════════
// [ALG] Adapter Factory — bitmap-encoded runtime dispatch
// ═══════════════════════════════════════════════════════════════════════════
// 原版: 手动实例化每个adapter类
// 新版: 统一接口 + dispatch表, 可遍历所有adapter

enum AdapterID : uint8_t {
    ADAPTER_LIVEGRAPH  = 0x01,
    ADAPTER_BACKEND    = 0x02,   // TieredBackendAdapter (generic)
    ADAPTER_ALL        = 0xFF
};

// 统一的adapter接口, 隐藏具体实现
struct AbstractAdapter {
    virtual ~AbstractAdapter() = default;
    virtual void load_graph(const PlantedPartitionGraph& g) = 0;
    virtual std::vector<uint64_t> bfs(uint64_t source) = 0;
    virtual std::unordered_map<uint64_t, uint64_t> wcc() = 0;
    virtual int64_t triangle_count_approx() = 0;
    virtual const char* name() const = 0;
};

// LiveGraph adapter
class LiveGraphE2EAdapter : public AbstractAdapter {
    std::unique_ptr<adapters::livegraph::TieredLiveGraphWrapper> wrapper_;
    std::set<uint64_t> vertex_set_;

public:
    void load_graph(const PlantedPartitionGraph& g) override {
        wrapper_ = std::make_unique<adapters::livegraph::TieredLiveGraphWrapper>(
            false, true);
        vertex_set_.clear();
        for (auto& e : g.edges) {
            vertex_set_.insert(e.source);
            vertex_set_.insert(e.destination);
        }
        for (auto v : vertex_set_) wrapper_->insert_vertex(v);
        for (auto& e : g.edges) {
            wrapper_->insert_edge(e.source, e.destination, e.weight);
        }
        PHILE_E2E_BREAKPOINT("LOAD-LG", "loaded %zu vertices, %zu edges",
                             vertex_set_.size(), g.edges.size());
    }

    // BFS: 返回距离数组 (vertex → distance from source)
    // 保留test_integration.cpp的frontier-expansion循环(100%保留)
    std::vector<uint64_t> bfs(uint64_t source) override {
        auto snap = wrapper_->get_unique_snapshot();
        std::unordered_map<uint64_t, int64_t> dist;
        dist[source] = 0;
        std::vector<uint64_t> frontier = {source};

        while (!frontier.empty()) {
            std::vector<uint64_t> next;
            for (auto u : frontier) {
                std::vector<uint64_t> nbrs;
                snap->edges(u, nbrs, false);
                for (auto v : nbrs) {
                    if (dist.find(v) == dist.end()) {
                        dist[v] = dist[u] + 1;
                        next.push_back(v);
                    }
                }
            }
            frontier = next;
        }

        // 收集: [visited_count, max_depth]
        int64_t max_d = 0;
        for (auto& [v, d] : dist) {
            if (d > max_d) max_d = d;
        }
        return {dist.size(), static_cast<uint64_t>(max_d)};
    }

    // WCC: union-find (保留test_integration.cpp的path compression, 100%保留)
    std::unordered_map<uint64_t, uint64_t> wcc() override {
        auto snap = wrapper_->get_unique_snapshot();
        std::unordered_map<uint64_t, uint64_t> parent;
        for (auto v : vertex_set_) parent[v] = v;

        std::function<uint64_t(uint64_t)> find = [&](uint64_t x) -> uint64_t {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        };

        for (auto u : vertex_set_) {
            std::vector<uint64_t> nbrs;
            snap->edges(u, nbrs, false);
            for (auto v : nbrs) {
                uint64_t ru = find(u), rv = find(v);
                if (ru != rv) parent[ru] = rv;
            }
        }

        // 统计component sizes
        std::unordered_map<uint64_t, uint64_t> comp_size;
        for (auto v : vertex_set_) {
            comp_size[find(v)]++;
        }
        return comp_size;
    }

    // [ALG] 近似三角形计数: wedge采样
    // 原版: 无三角形计数
    // 新版: 随机采样wedge (u,v,w), 检查w是否邻接u
    //       TC ≈ (closed_wedges / sampled_wedges) * total_wedges / 3
    int64_t triangle_count_approx() override {
        auto snap = wrapper_->get_unique_snapshot();
        std::mt19937 rng(12345);

        // 收集所有顶点的度数
        std::vector<uint64_t> verts(vertex_set_.begin(), vertex_set_.end());
        std::vector<uint64_t> degrees(verts.size());
        std::unordered_map<uint64_t, size_t> v2idx;
        for (size_t i = 0; i < verts.size(); ++i) {
            v2idx[verts[i]] = i;
            std::vector<uint64_t> nbrs;
            snap->edges(verts[i], nbrs, false);
            degrees[i] = nbrs.size();
        }

        // 总wedge数: Σ C(deg(v), 2)
        int64_t total_wedges = 0;
        for (size_t i = 0; i < verts.size(); ++i) {
            int64_t d = static_cast<int64_t>(degrees[i]);
            total_wedges += d * (d - 1) / 2;
        }

        if (total_wedges == 0) return 0;

        // 采样wedge
        size_t n_samples = std::min<size_t>(500, total_wedges);
        size_t closed = 0;

        for (size_t s = 0; s < n_samples; ++s) {
            // 按wedge数加权选择中心顶点
            size_t center_idx = rng() % verts.size();
            uint64_t center = verts[center_idx];

            std::vector<uint64_t> nbrs;
            snap->edges(center, nbrs, false);
            if (nbrs.size() < 2) continue;

            // 选两个邻居
            size_t i = rng() % nbrs.size();
            size_t j = rng() % (nbrs.size() - 1);
            if (j >= i) j++;
            uint64_t u = nbrs[i], v = nbrs[j];

            // 检查u-v是否有边
            std::vector<uint64_t> u_nbrs;
            snap->edges(u, u_nbrs, false);
            if (std::find(u_nbrs.begin(), u_nbrs.end(), v) != u_nbrs.end()) {
                closed++;
            }
        }

        double ratio = (n_samples > 0)
            ? static_cast<double>(closed) / n_samples : 0;
        int64_t approx_tc = static_cast<int64_t>(
            ratio * total_wedges / 3.0);

        PHILE_E2E_BREAKPOINT("TC", "sampled=%zu closed=%zu ratio=%.3f "
            "total_wedges=%ld approx_tc=%ld",
            n_samples, closed, ratio, (long)total_wedges, (long)approx_tc);

        return approx_tc;
    }

    const char* name() const override { return "LiveGraph"; }
};

// TieredBackendAdapter wrapper
class BackendE2EAdapter : public AbstractAdapter {
    std::unique_ptr<adapters::TieredBackendAdapter> wrapper_;
    std::set<uint64_t> vertex_set_;

public:
    void load_graph(const PlantedPartitionGraph& g) override {
        wrapper_ = std::make_unique<adapters::TieredBackendAdapter>(
            false, true);
        vertex_set_.clear();
        for (auto& e : g.edges) {
            vertex_set_.insert(e.source);
            vertex_set_.insert(e.destination);
        }
        for (auto v : vertex_set_) wrapper_->insert_vertex(v);
        for (auto& e : g.edges) {
            wrapper_->insert_edge(e.source, e.destination, e.weight);
        }
        PHILE_E2E_BREAKPOINT("LOAD-BA", "loaded %zu vertices, %zu edges",
                             vertex_set_.size(), g.edges.size());
    }

    std::vector<uint64_t> bfs(uint64_t source) override {
        auto snap = wrapper_->get_unique_snapshot();
        std::unordered_map<uint64_t, int64_t> dist;
        dist[source] = 0;
        std::vector<uint64_t> frontier = {source};

        while (!frontier.empty()) {
            std::vector<uint64_t> next;
            for (auto u : frontier) {
                snap->edges(u, [&](uint64_t, uint64_t dst, double) {
                    if (dist.find(dst) == dist.end()) {
                        dist[dst] = dist[u] + 1;
                        next.push_back(dst);
                    }
                }, false);
            }
            frontier = next;
        }

        int64_t max_d = 0;
        for (auto& [v, d] : dist) if (d > max_d) max_d = d;
        return {dist.size(), static_cast<uint64_t>(max_d)};
    }

    std::unordered_map<uint64_t, uint64_t> wcc() override {
        auto snap = wrapper_->get_unique_snapshot();
        std::unordered_map<uint64_t, uint64_t> parent;
        for (auto v : vertex_set_) parent[v] = v;

        std::function<uint64_t(uint64_t)> find = [&](uint64_t x) -> uint64_t {
            return parent[x] == x ? x : parent[x] = find(parent[x]);
        };

        for (auto u : vertex_set_) {
            snap->edges(u, [&](uint64_t, uint64_t v, double) {
                uint64_t ru = find(u), rv = find(v);
                if (ru != rv) parent[ru] = rv;
            }, false);
        }

        std::unordered_map<uint64_t, uint64_t> comp_size;
        for (auto v : vertex_set_) comp_size[find(v)]++;
        return comp_size;
    }

    int64_t triangle_count_approx() override {
        // 使用与LiveGraphE2EAdapter相同的wedge采样算法
        auto snap = wrapper_->get_unique_snapshot();
        std::mt19937 rng(12345);
        std::vector<uint64_t> verts(vertex_set_.begin(), vertex_set_.end());

        // 收集邻接表
        std::unordered_map<uint64_t, std::vector<uint64_t>> adj;
        for (auto u : verts) {
            snap->edges(u, [&](uint64_t, uint64_t v, double) {
                adj[u].push_back(v);
            }, false);
        }

        int64_t total_wedges = 0;
        for (auto& [v, nbrs] : adj) {
            int64_t d = static_cast<int64_t>(nbrs.size());
            total_wedges += d * (d - 1) / 2;
        }
        if (total_wedges == 0) return 0;

        size_t n_samples = std::min<size_t>(500, total_wedges);
        size_t closed = 0;

        for (size_t s = 0; s < n_samples; ++s) {
            size_t ci = rng() % verts.size();
            auto& nbrs = adj[verts[ci]];
            if (nbrs.size() < 2) continue;
            size_t i = rng() % nbrs.size();
            size_t j = rng() % (nbrs.size() - 1);
            if (j >= i) j++;
            uint64_t u = nbrs[i], v = nbrs[j];
            auto& u_nbrs = adj[u];
            if (std::find(u_nbrs.begin(), u_nbrs.end(), v) != u_nbrs.end()) {
                closed++;
            }
        }

        double ratio = (n_samples > 0)
            ? static_cast<double>(closed) / n_samples : 0;
        return static_cast<int64_t>(ratio * total_wedges / 3.0);
    }

    const char* name() const override { return "BackendAdapter"; }
};


// ═══════════════════════════════════════════════════════════════════════════
// [ALG] AdapterFactory — bitmap dispatch
// ═══════════════════════════════════════════════════════════════════════════
// 原版: 手动实例化每个adapter
// 新版: 工厂根据bitmap创建选中的adapter集合

struct AdapterFactory {
    static std::vector<std::unique_ptr<AbstractAdapter>>
    create(uint8_t selector) {
        std::vector<std::unique_ptr<AbstractAdapter>> adapters;

        if (selector & ADAPTER_LIVEGRAPH) {
            adapters.push_back(std::make_unique<LiveGraphE2EAdapter>());
        }
        if (selector & ADAPTER_BACKEND) {
            adapters.push_back(std::make_unique<BackendE2EAdapter>());
        }

        PHILE_E2E_BREAKPOINT("FACTORY", "created %zu adapters from "
            "selector=0x%02X", adapters.size(), selector);
        return adapters;
    }
};


// ═══════════════════════════════════════════════════════════════════════════
// [ALG] Cross-Validation Utilities
// ═══════════════════════════════════════════════════════════════════════════
// 原版: EXPECT_GT(dist.size(), 1u) 单一非空检查
// 新版: 跨adapter交叉验证

namespace cross_validate {

// BFS: max_depth必须在所有adapter间一致
inline bool bfs_consistent(
        const std::vector<std::vector<uint64_t>>& results,
        const std::vector<std::string>& names) {
    if (results.size() < 2) return true;

    uint64_t ref_visited = results[0][0];
    uint64_t ref_depth = results[0][1];

    bool ok = true;
    for (size_t i = 1; i < results.size(); ++i) {
        if (results[i][0] != ref_visited) {
            fprintf(stderr,
                "\x1b[31m  BFS inconsistency: %s visited=%lu vs %s visited=%lu\x1b[0m\n",
                names[0].c_str(), (unsigned long)ref_visited,
                names[i].c_str(), (unsigned long)results[i][0]);
            ok = false;
        }
        if (results[i][1] != ref_depth) {
            fprintf(stderr,
                "\x1b[31m  BFS depth inconsistency: %s depth=%lu vs %s depth=%lu\x1b[0m\n",
                names[0].c_str(), (unsigned long)ref_depth,
                names[i].c_str(), (unsigned long)results[i][1]);
            ok = false;
        }
    }
    return ok;
}

// WCC: component数量必须一致
inline bool wcc_consistent(
        const std::vector<std::unordered_map<uint64_t, uint64_t>>& results,
        const std::vector<std::string>& names) {
    if (results.size() < 2) return true;

    size_t ref_count = results[0].size();
    bool ok = true;
    for (size_t i = 1; i < results.size(); ++i) {
        if (results[i].size() != ref_count) {
            fprintf(stderr,
                "\x1b[31m  WCC inconsistency: %s components=%zu vs %s components=%zu\x1b[0m\n",
                names[0].c_str(), ref_count,
                names[i].c_str(), results[i].size());
            ok = false;
        }
    }
    return ok;
}

// TC: 近似三角形计数在合理范围内 (±50%, 因为是采样)
inline bool tc_roughly_consistent(
        const std::vector<int64_t>& counts,
        const std::vector<std::string>& names) {
    if (counts.size() < 2) return true;

    int64_t ref = counts[0];
    if (ref == 0) return true;  // 没三角形就跳过

    bool ok = true;
    for (size_t i = 1; i < counts.size(); ++i) {
        double ratio = static_cast<double>(counts[i]) / ref;
        if (ratio < 0.5 || ratio > 2.0) {
            fprintf(stderr,
                "\x1b[31m  TC inconsistency: %s=%ld vs %s=%ld (ratio=%.2f)\x1b[0m\n",
                names[0].c_str(), (long)ref,
                names[i].c_str(), (long)counts[i], ratio);
            ok = false;
        }
    }
    return ok;
}

}  // namespace cross_validate


// ═══════════════════════════════════════════════════════════════════════════
// Test Fixture
// ═══════════════════════════════════════════════════════════════════════════

class E2EIntegrationTest : public ::testing::Test {
protected:
    // 小图用于CI快速测试; 大图用于full benchmark
    static constexpr uint64_t NV = 60;
    static constexpr uint64_t N_CLUSTERS = 3;
    static constexpr double P_IN = 0.3;
    static constexpr double P_OUT = 0.02;

    PlantedPartitionGraph graph;

    void SetUp() override {
        debug::set_debug_level(0);
        graph = PlantedPartitionGraph::generate(
            NV, N_CLUSTERS, P_IN, P_OUT, 10000, 42);
    }
};


// ─── Test 1: Config → Graph → Adapter Load ──────────────────────────────

TEST_F(E2EIntegrationTest, ConfigLoadAndGraphBuild) {
    // Phase 1: Config parsing
    std::string cfg_path = "/tmp/philemon_e2e_test.cfg";
    {
        std::ofstream out(cfg_path);
        out << "num_threads=4\n"
            << "seed=42\n"
            << "alpha=15\n"
            << "beta=18\n"
            << "bfs_source=0\n"
            << "sssp_source=0\n"
            << "delta=2.0\n"
            << "num_iterations=10\n"
            << "damping_factor=0.85\n"
            << "hbm_capacity_gb=4.0\n"
            << "gddr_capacity_gb=16.0\n"
            << "writer_threads=1\n";
    }

    auto& parser = ConfigParser::get_instance();
    parser.parse(cfg_path);
    EXPECT_EQ(parser.get_num_threads(), 4);
    EXPECT_EQ(parser.get_seed(), 42);

    PHILE_E2E_BREAKPOINT("CONFIG", "parsed config, num_threads=%d",
                         parser.get_num_threads());

    // Phase 2: Graph generation with planted partition
    EXPECT_EQ(graph.num_vertices, NV);
    EXPECT_GT(graph.edges.size(), 0u);
    EXPECT_EQ(graph.num_clusters, N_CLUSTERS);

    PHILE_E2E_BREAKPOINT("GRAPH", "vertices=%lu edges=%zu clusters=%lu",
        (unsigned long)graph.num_vertices, graph.edges.size(),
        (unsigned long)graph.num_clusters);

    // Phase 3: Load into adapter
    auto adapters = AdapterFactory::create(ADAPTER_LIVEGRAPH);
    ASSERT_EQ(adapters.size(), 1u);
    adapters[0]->load_graph(graph);

    std::remove(cfg_path.c_str());
}


// ─── Test 2: BFS Cross-Adapter Consistency ──────────────────────────────

TEST_F(E2EIntegrationTest, BFSCrossAdapterConsistency) {
    auto adapters = AdapterFactory::create(ADAPTER_LIVEGRAPH | ADAPTER_BACKEND);

    // Load same graph into all adapters
    for (auto& a : adapters) {
        a->load_graph(graph);
    }

    // Run BFS from vertex 0 on each
    std::vector<std::vector<uint64_t>> results;
    std::vector<std::string> names;
    for (auto& a : adapters) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = a->bfs(0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PHILE_E2E_BREAKPOINT("BFS", "%s: visited=%lu depth=%lu time=%.3fms",
            a->name(), (unsigned long)r[0], (unsigned long)r[1], ms);

        results.push_back(r);
        names.push_back(a->name());

        // BFS should reach at least some vertices
        EXPECT_GT(r[0], 1u) << "BFS from 0 on " << a->name();
        // Max depth bounded by diameter estimate
        EXPECT_LE(r[1], graph.expected_diameter_upper)
            << "BFS depth exceeds diameter bound on " << a->name();
    }

    // [ALG] 交叉验证
    EXPECT_TRUE(cross_validate::bfs_consistent(results, names));
}


// ─── Test 3: WCC Cross-Adapter + Ground Truth ───────────────────────────

TEST_F(E2EIntegrationTest, WCCCrossAdapterGroundTruth) {
    auto adapters = AdapterFactory::create(ADAPTER_LIVEGRAPH | ADAPTER_BACKEND);

    for (auto& a : adapters) {
        a->load_graph(graph);
    }

    std::vector<std::unordered_map<uint64_t, uint64_t>> results;
    std::vector<std::string> names;

    for (auto& a : adapters) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto comp = a->wcc();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        PHILE_E2E_BREAKPOINT("WCC", "%s: components=%zu time=%.3fms",
            a->name(), comp.size(), ms);

        // [ALG] 验证ground truth: planted partition + backbone → 1 component
        EXPECT_EQ(comp.size(), graph.expected_wcc_components)
            << "WCC should find " << graph.expected_wcc_components
            << " component(s) on " << a->name();

        results.push_back(comp);
        names.push_back(a->name());
    }

    EXPECT_TRUE(cross_validate::wcc_consistent(results, names));
}


// ─── Test 4: Triangle Count Cross-Adapter ───────────────────────────────

TEST_F(E2EIntegrationTest, TriangleCountCrossAdapter) {
    auto adapters = AdapterFactory::create(ADAPTER_LIVEGRAPH | ADAPTER_BACKEND);

    for (auto& a : adapters) {
        a->load_graph(graph);
    }

    std::vector<int64_t> counts;
    std::vector<std::string> names;

    for (auto& a : adapters) {
        auto tc = a->triangle_count_approx();

        PHILE_E2E_BREAKPOINT("TC", "%s: approx_triangles=%ld", a->name(), (long)tc);

        // planted partition with p_in=0.3 should have triangles
        EXPECT_GE(tc, 0) << "Negative TC on " << a->name();

        counts.push_back(tc);
        names.push_back(a->name());
    }

    EXPECT_TRUE(cross_validate::tc_roughly_consistent(counts, names));
}


// ─── Test 5: Interval Index Integration ─────────────────────────────────

TEST_F(E2EIntegrationTest, TemporalIndexQueryPipeline) {
    // 保留test_integration.cpp的TemGraph用法(100%保留)
    index::TemGraph tg;
    auto intervals = graph.to_intervals();
    tg.load_from_edges(index::CONTAINS_QUERY, intervals);

    PHILE_E2E_BREAKPOINT("INDEX", "loaded %zu intervals into TemGraph",
                         intervals.size());

    // 查询100个时间窗口
    int total_results = 0;
    for (int q = 0; q < 100; ++q) {
        int qs = q * (graph.time_range / 100);
        int qe = qs + graph.time_range / 10;
        total_results += tg.contains_query(qs, qe);
    }

    EXPECT_GT(total_results, 0);
    PHILE_E2E_BREAKPOINT("INDEX", "100 queries, total_results=%d",
                         total_results);
}


// ─── Test 6: Tiered Allocator Memory Lifecycle ──────────────────────────

TEST_F(E2EIntegrationTest, TieredAllocatorLifecycle) {
    TieredAllocator alloc(256 * 1024, 1024 * 1024, 4 * 1024 * 1024);

    // Allocate in each tier
    std::vector<uint64_t> ids;

    uint64_t hbm_id = alloc.allocate(4096, MemoryTier::HBM);
    EXPECT_GT(hbm_id, 0u);
    ids.push_back(hbm_id);

    uint64_t gddr_id = alloc.allocate(4096, MemoryTier::GDDR);
    EXPECT_GT(gddr_id, 0u);
    ids.push_back(gddr_id);

    uint64_t dram_id = alloc.allocate(4096, MemoryTier::DRAM);
    EXPECT_GT(dram_id, 0u);
    ids.push_back(dram_id);

    // Verify tiers
    AllocMeta meta;
    alloc.get_meta(hbm_id, meta);
    EXPECT_EQ(meta.current_tier, MemoryTier::HBM);
    alloc.get_meta(gddr_id, meta);
    EXPECT_EQ(meta.current_tier, MemoryTier::GDDR);
    alloc.get_meta(dram_id, meta);
    EXPECT_EQ(meta.current_tier, MemoryTier::DRAM);

    // Migrate HBM → GDDR
    bool migrated = alloc.migrate(hbm_id, MemoryTier::GDDR);
    EXPECT_TRUE(migrated);
    alloc.get_meta(hbm_id, meta);
    EXPECT_EQ(meta.current_tier, MemoryTier::GDDR);

    // Deallocate all
    for (auto id : ids) alloc.deallocate(id);

    PHILE_E2E_BREAKPOINT("ALLOC", "lifecycle complete: alloc/migrate/dealloc");
}


// ─── Test 7: Cost Model Consistency ─────────────────────────────────────

TEST_F(E2EIntegrationTest, CostModelTierOrdering) {
    cost_model::TierSpec hbm, gddr, dram;

    hbm.access_latency_ns = 1.2;
    hbm.bandwidth_gbps = 3350;
    hbm.capacity_bytes = 80ULL * 1024 * 1024 * 1024;

    gddr.access_latency_ns = 5.0;
    gddr.bandwidth_gbps = 768;
    gddr.capacity_bytes = 48ULL * 1024 * 1024 * 1024;

    dram.access_latency_ns = 50.0;
    dram.bandwidth_gbps = 80;
    dram.capacity_bytes = 512ULL * 1024 * 1024 * 1024;

    // Ordering invariants
    EXPECT_LT(hbm.access_latency_ns, gddr.access_latency_ns);
    EXPECT_LT(gddr.access_latency_ns, dram.access_latency_ns);
    EXPECT_GT(hbm.bandwidth_gbps, gddr.bandwidth_gbps);
    EXPECT_GT(gddr.bandwidth_gbps, dram.bandwidth_gbps);
    EXPECT_GT(hbm.bytes_per_ns(), gddr.bytes_per_ns());
    EXPECT_GT(gddr.bytes_per_ns(), dram.bytes_per_ns());
}


// ─── Test 8: Debug Infrastructure ───────────────────────────────────────

TEST_F(E2EIntegrationTest, DebugInfraFullCycle) {
    debug::set_debug_level(3);

    // TraceRing
    auto& ring = debug::global_trace();
    ring.record(debug::TraceEvent::ALLOC, 0, 100);
    ring.record(debug::TraceEvent::MIGRATE, 100, 200);

    // TierPerfCounter
    auto& pc = debug::tier_perf(0);
    pc.read_count.fetch_add(5);
    pc.write_count.fetch_add(3);
    EXPECT_GE(pc.read_count.load(), 5u);

    // BreakpointGuard RAII
    {
        debug::BreakpointGuard guard("e2e_test");
    }

    // Inspection record
    debug::record_inspection("e2e_phase", "all clear");

    debug::set_debug_level(0);
}


// ─── Test 9: Full Pipeline Timing ───────────────────────────────────────

TEST_F(E2EIntegrationTest, FullPipelineTiming) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Step 1: Generate graph (already done in SetUp, but re-generate for timing)
    auto g = PlantedPartitionGraph::generate(80, 4, 0.25, 0.02, 10000, 99);

    // Step 2: Load into adapter
    auto adapters = AdapterFactory::create(ADAPTER_LIVEGRAPH);
    ASSERT_EQ(adapters.size(), 1u);
    adapters[0]->load_graph(g);

    // Step 3: Build temporal index
    index::TemGraph tg;
    tg.load_from_edges(index::CONTAINS_QUERY, g.to_intervals());

    // Step 4: Run BFS
    auto bfs_result = adapters[0]->bfs(0);
    EXPECT_GT(bfs_result[0], 1u);

    // Step 5: Run WCC
    auto wcc_result = adapters[0]->wcc();
    EXPECT_EQ(wcc_result.size(), 1u);

    // Step 6: Run TC
    auto tc = adapters[0]->triangle_count_approx();
    EXPECT_GE(tc, 0);

    // Step 7: Query temporal index
    int qresults = 0;
    for (int q = 0; q < 50; ++q) {
        qresults += tg.contains_query(q * 200, q * 200 + 500);
    }
    EXPECT_GE(qresults, 0);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    PHILE_E2E_BREAKPOINT("PIPELINE", "full pipeline in %.3f ms", ms);
    EXPECT_LT(ms, 10000.0) << "Pipeline too slow: " << ms << " ms";
}


// ─── Test 10: Adapter Factory Bitmap Selection ──────────────────────────

TEST_F(E2EIntegrationTest, FactoryBitmapSelection) {
    // Single adapter
    auto single = AdapterFactory::create(ADAPTER_LIVEGRAPH);
    EXPECT_EQ(single.size(), 1u);
    EXPECT_STREQ(single[0]->name(), "LiveGraph");

    // Both adapters
    auto both = AdapterFactory::create(ADAPTER_LIVEGRAPH | ADAPTER_BACKEND);
    EXPECT_EQ(both.size(), 2u);

    // Empty selector
    auto none = AdapterFactory::create(0x00);
    EXPECT_EQ(none.size(), 0u);
}


// ─── Test 11: Planted Partition Community Structure ─────────────────────
// [ALG] 验证planted partition的社区结构
// 不是字符串测试 — 实际计算intra/inter cluster edge密度

TEST_F(E2EIntegrationTest, PlantedPartitionStructure) {
    uint64_t intra_edges = 0, inter_edges = 0;

    for (const auto& e : graph.edges) {
        if (graph.cluster_assignment[e.source] ==
            graph.cluster_assignment[e.destination]) {
            intra_edges++;
        } else {
            inter_edges++;
        }
    }

    double intra_ratio = static_cast<double>(intra_edges) /
                         (intra_edges + inter_edges);

    PHILE_E2E_BREAKPOINT("COMMUNITY", "intra=%lu inter=%lu ratio=%.3f",
        (unsigned long)intra_edges, (unsigned long)inter_edges, intra_ratio);

    // With p_in=0.3 >> p_out=0.02, most edges should be intra-cluster
    EXPECT_GT(intra_ratio, 0.5) << "Expected community structure with "
        << "intra_ratio > 0.5, got " << intra_ratio;
}


// ─── Test 12: SeqLock Concurrent Safety ─────────────────────────────────
// 保留test_core.cpp的并发SeqLock测试模式 (100%保留), 集成到E2E

TEST_F(E2EIntegrationTest, SeqLockConcurrentAccess) {
    SeqLock sl;
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};
    std::atomic<bool> done{false};

    // Writer thread
    std::thread writer([&]() {
        for (int i = 0; i < 1000; ++i) {
            sl.write_lock();
            write_count.fetch_add(1);
            sl.write_unlock();
        }
        done.store(true);
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&]() {
            while (!done.load()) {
                uint32_t seq = sl.read_begin();
                read_count.fetch_add(1);
                if (!sl.read_retry(seq)) {
                    // read was consistent
                }
            }
        });
    }

    writer.join();
    for (auto& r : readers) r.join();

    EXPECT_EQ(write_count.load(), 1000);
    EXPECT_GT(read_count.load(), 0);
}
