#pragma once
/**
 * gpu_topology.hpp — GPU拓扑探测+带宽建模+最优路由
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   src/cuda/hetero_bench.cu  HeteroAllocator构造函数 (134–177行)
 *     → cudaGetDeviceCount + cudaGetDeviceProperties 枚举
 *     → cudaDeviceCanAccessPeer + cudaDeviceEnablePeerAccess 拓扑探测
 *     → 嵌套for循环 peer access enable
 *
 *   src/cuda/hetero_bench.cu  experiment_bandwidth (293–357行)
 *     → 4种size × 4种tier pair 的带宽矩阵测量
 *     → CudaTimer + 5次迭代平均
 *
 *   src/cost_model/tier_cost_model.hpp  TierSpec结构 (40–87行)
 *     → bandwidth_gbps / latency_ns 硬件规格常量
 *
 * 算法改动 (~20%):
 *   [ALG1] 带宽建模: 原版用硬编码常量(HBM=3350GB/s等)
 *          → 运行时micro-benchmark: 分配probe buffer, 测实际带宽, 取中位数
 *   [ALG2] 拓扑表示: 原版bool can_peer[i][j]
 *          → 带权有向图 adj[i][j]=实测带宽(GB/s), 0=不可达
 *   [ALG3] 路由选择: 原版if/else手写路径
 *          → Dijkstra单源最短路(以1/bandwidth为权), 预算路由表
 *   [ALG4] NUMA感知: 原版忽略CPU-GPU亲和性
 *          → 解析/sys/bus/pci/.../numa_node, 对齐线程调度
 *
 * Milestone: M046 — GPU拓扑探测
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits>
#include <queue>
#include <functional>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "cuda_mem_manager.hpp"

namespace philemon {
namespace topology {

// ─── Node in topology graph ────────────────────────────────────────
struct DeviceNode {
    int          device_id;         // GPU id, or -1 for host
    std::string  name;              // "H100 NVL", "RTX A6000", "Host-NUMA0"
    size_t       total_mem;         // bytes
    size_t       free_mem;
    int          numa_node;         // -1 if unknown
    int          sm_major, sm_minor;
    int          pci_bus, pci_device, pci_domain;

    DeviceNode()
        : device_id(-1), total_mem(0), free_mem(0), numa_node(-1),
          sm_major(0), sm_minor(0), pci_bus(0), pci_device(0), pci_domain(0) {}
};

// ─── 带权边 (实测带宽) ──────────────────────────────────────────────
struct TopoEdge {
    int     src, dst;
    double  bandwidth_gbps;     // [ALG1] 实测, 非硬编码
    double  latency_us;         // 估算延迟
    bool    is_peer_direct;     // NVLink/PCIe peer
    bool    is_staged;          // 需要经host中转
};

// ─── [ALG3] 路由表项 ───────────────────────────────────────────────
struct RouteEntry {
    int    next_hop;             // 下一跳设备
    double total_cost;           // 1/bandwidth的累积cost
    double effective_bw_gbps;    // 路径瓶颈带宽
    std::vector<int> path;       // 完整路径
};

// ════════════════════════════════════════════════════════════════════════════
//  GpuTopology — 拓扑探测+带宽建模+路由
// ════════════════════════════════════════════════════════════════════════════

class GpuTopology {
public:
    static constexpr int MAX_NODES  = 5;     // 3 GPU + 1 Host + 1 spare
    static constexpr int HOST_NODE  = 3;     // 约定host的node id
    static constexpr size_t PROBE_SIZE = 16ULL << 20;  // 16MB probe buffer
    static constexpr int PROBE_ITERS = 7;    // 测量次数, 取中位数

    GpuTopology() {
        PHILE_CHECKPOINT("GpuTopology::ctor");
        discover_devices();
        probe_peer_access();
        measure_bandwidths();
        build_route_table();
        detect_numa_affinity();

        PHILE_DBG(1, "[Topo] discovery complete: %zu devices, %zu edges\n",
                  nodes_.size(), edges_.size());
    }

    // ── 查询最优拷贝路由 ─────────────────────────────────────────
    const RouteEntry& best_route(int src_node, int dst_node) const {
        return route_table_[src_node][dst_node];
    }

    // ── 查询两节点间的实测带宽 ───────────────────────────────────
    double measured_bandwidth(int src, int dst) const {
        return bw_matrix_[src][dst];
    }

    // ── 获取设备信息 ─────────────────────────────────────────────
    const DeviceNode& node(int id) const { return nodes_[id]; }
    size_t device_count() const { return gpu_count_; }

    // ── NUMA亲和性建议 ──────────────────────────────────────────
    int recommended_numa_for(int device_id) const {
        if (device_id < 0 || device_id >= static_cast<int>(nodes_.size()))
            return 0;
        return nodes_[device_id].numa_node >= 0 ? nodes_[device_id].numa_node : 0;
    }

    // ── 断点调试: 打印完整拓扑 ───────────────────────────────────
    void dump_topology() const {
        PHILE_SEPARATOR("GPU Topology");

        printf("[Topo] ── Devices ──\n");
        for (size_t i = 0; i < nodes_.size(); ++i) {
            auto& n = nodes_[i];
            printf("  Node %zu: %-20s  GPU=%d  NUMA=%d  Mem=%.1f/%.1f GB  SM=%d.%d\n",
                   i, n.name.c_str(), n.device_id, n.numa_node,
                   n.free_mem / (1024.0*1024*1024),
                   n.total_mem / (1024.0*1024*1024),
                   n.sm_major, n.sm_minor);
        }

        // [ALG2] 带权带宽矩阵
        printf("\n[Topo] ── Bandwidth Matrix (GB/s, [ALG1] runtime measured) ──\n");
        printf("  %8s", "src\\dst");
        for (size_t j = 0; j < nodes_.size(); ++j)
            printf("  %12s", nodes_[j].name.c_str());
        printf("\n");

        for (size_t i = 0; i < nodes_.size(); ++i) {
            printf("  %8s", nodes_[i].name.c_str());
            for (size_t j = 0; j < nodes_.size(); ++j) {
                if (i == j) printf("  %12s", "—");
                else printf("  %10.1f", bw_matrix_[i][j]);
            }
            printf("\n");
        }

        // [ALG3] 路由表
        printf("\n[Topo] ── Route Table ([ALG3] Dijkstra optimal) ──\n");
        for (size_t i = 0; i < nodes_.size(); ++i) {
            for (size_t j = 0; j < nodes_.size(); ++j) {
                if (i == j) continue;
                auto& r = route_table_[i][j];
                if (r.path.empty()) continue;
                printf("  %s → %s: bw=%.1f GB/s path=",
                       nodes_[i].name.c_str(), nodes_[j].name.c_str(),
                       r.effective_bw_gbps);
                for (size_t k = 0; k < r.path.size(); ++k) {
                    printf("%s%s", nodes_[r.path[k]].name.c_str(),
                           k + 1 < r.path.size() ? "→" : "");
                }
                printf("\n");
            }
        }

        // [ALG4] NUMA
        printf("\n[Topo] ── NUMA Affinity ──\n");
        for (size_t i = 0; i < nodes_.size(); ++i) {
            printf("  %s → NUMA node %d\n",
                   nodes_[i].name.c_str(), nodes_[i].numa_node);
        }

        PHILE_SEPARATOR("End GPU Topology");
    }

private:
    // ── 设备发现 ─────────────────────────────────────────────────
    void discover_devices() {
        PHILE_CHECKPOINT("discover_devices");

        #ifdef __CUDACC__
        int count = 0;
        cudaGetDeviceCount(&count);
        gpu_count_ = std::min(count, 3);

        for (int d = 0; d < gpu_count_; ++d) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, d);
            size_t free_m = 0, total_m = 0;
            cudaSetDevice(d);
            cudaMemGetInfo(&free_m, &total_m);

            DeviceNode node;
            node.device_id  = d;
            node.name       = prop.name;
            node.total_mem  = total_m;
            node.free_mem   = free_m;
            node.sm_major   = prop.major;
            node.sm_minor   = prop.minor;
            node.pci_bus    = prop.pciBusID;
            node.pci_device = prop.pciDeviceID;
            node.pci_domain = prop.pciDomainID;
            nodes_.push_back(node);
        }
        cudaSetDevice(0);
        #else
        gpu_count_ = 3;
        // CPU-only模拟: 3个虚拟GPU
        const char* names[] = {"RTX-A6000-0", "RTX-A6000-1", "H100-NVL"};
        size_t mems[] = {48ULL<<30, 48ULL<<30, 96ULL<<30};
        for (int d = 0; d < 3; ++d) {
            DeviceNode node;
            node.device_id = d;
            node.name      = names[d];
            node.total_mem = mems[d];
            node.free_mem  = mems[d];
            node.sm_major  = (d == 2) ? 9 : 8;
            node.sm_minor  = 0;
            nodes_.push_back(node);
        }
        #endif

        // Host node
        DeviceNode host;
        host.device_id = -1;
        host.name      = "Host-DRAM";
        host.total_mem = 1536ULL << 30;  // 1.5TB
        host.free_mem  = 1024ULL << 30;
        host.numa_node = 0;
        nodes_.push_back(host);

        PHILE_DBG(1, "[discover] %zu GPU + 1 Host\n", gpu_count_);
    }

    // ── Peer access探测 ──────────────────────────────────────────
    void probe_peer_access() {
        PHILE_CHECKPOINT("probe_peer_access");

        // 清零
        for (int i = 0; i < MAX_NODES; ++i)
            for (int j = 0; j < MAX_NODES; ++j)
                bw_matrix_[i][j] = 0.0;

        #ifdef __CUDACC__
        for (int i = 0; i < gpu_count_; ++i) {
            for (int j = 0; j < gpu_count_; ++j) {
                if (i == j) continue;
                int can = 0;
                cudaDeviceCanAccessPeer(&can, i, j);
                if (can) {
                    cudaSetDevice(i);
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    bool ok = (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
                    cudaGetLastError();
                    if (ok) {
                        TopoEdge e{i, j, 0, 0, true, false};
                        edges_.push_back(e);
                    }
                }
            }
            // GPU↔Host: 始终可达
            edges_.push_back({i, HOST_NODE, 0, 0, false, false});
            edges_.push_back({HOST_NODE, i, 0, 0, false, false});
        }
        cudaSetDevice(0);
        #else
        // CPU模拟
        edges_.push_back({0, 1, 0, 0, true, false});
        edges_.push_back({1, 0, 0, 0, true, false});
        for (int i = 0; i < 3; ++i) {
            edges_.push_back({i, HOST_NODE, 0, 0, false, false});
            edges_.push_back({HOST_NODE, i, 0, 0, false, false});
        }
        #endif
    }

    // ── [ALG1] 运行时带宽实测 ───────────────────────────────────
    // 原版: 硬编码常量 HBM=3350, GDDR=768, DRAM=80
    // 改动: 分配probe buffer → 多次拷贝 → 取中位数带宽
    void measure_bandwidths() {
        PHILE_CHECKPOINT("measure_bandwidths");

        for (auto& e : edges_) {
            double bw = run_bandwidth_probe(e.src, e.dst);
            e.bandwidth_gbps = bw;
            e.latency_us = (bw > 0) ? (PROBE_SIZE / (bw * 1e9)) * 1e6 : 9999.0;
            bw_matrix_[e.src][e.dst] = bw;

            PHILE_DBG(2, "[bw_probe] %d→%d: %.1f GB/s  latency=%.1f µs\n",
                      e.src, e.dst, bw, e.latency_us);
        }
    }

    double run_bandwidth_probe(int src_node, int dst_node) {
        #ifdef __CUDACC__
        void *src_buf = nullptr, *dst_buf = nullptr;

        // 分配probe buffer
        auto alloc_on = [](int node, size_t sz) -> void* {
            void* p = nullptr;
            if (node >= 0 && node < 3) {
                cudaSetDevice(node);
                if (cudaMalloc(&p, sz) != cudaSuccess) return nullptr;
            } else {
                if (cudaMallocHost(&p, sz) != cudaSuccess) return nullptr;
            }
            return p;
        };

        src_buf = alloc_on(src_node < HOST_NODE ? src_node : -1, PROBE_SIZE);
        dst_buf = alloc_on(dst_node < HOST_NODE ? dst_node : -1, PROBE_SIZE);
        if (!src_buf || !dst_buf) {
            if (src_buf) { /* free */ }
            if (dst_buf) { /* free */ }
            return 0.0;
        }

        // warmup
        cudaMemcpy(dst_buf, src_buf, PROBE_SIZE, cudaMemcpyDefault);

        // 测量多次, 取中位数
        std::vector<double> samples(PROBE_ITERS);
        for (int i = 0; i < PROBE_ITERS; ++i) {
            cudaEvent_t t0, t1;
            cudaEventCreate(&t0);
            cudaEventCreate(&t1);
            cudaEventRecord(t0);
            cudaMemcpy(dst_buf, src_buf, PROBE_SIZE, cudaMemcpyDefault);
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0;
            cudaEventElapsedTime(&ms, t0, t1);
            samples[i] = (PROBE_SIZE / (1024.0*1024*1024)) / (ms / 1000.0);
            cudaEventDestroy(t0);
            cudaEventDestroy(t1);
        }

        // 清理
        if (src_node < HOST_NODE) { cudaSetDevice(src_node); cudaFree(src_buf); }
        else { cudaFreeHost(src_buf); }
        if (dst_node < HOST_NODE) { cudaSetDevice(dst_node); cudaFree(dst_buf); }
        else { cudaFreeHost(dst_buf); }

        // 中位数
        std::sort(samples.begin(), samples.end());
        return samples[PROBE_ITERS / 2];

        #else
        // CPU模拟带宽
        if (src_node == dst_node) return 0.0;
        // 两个A6000之间(GPU0↔GPU1): PCIe peer ~24 GB/s
        if ((src_node == 0 && dst_node == 1) || (src_node == 1 && dst_node == 0))
            return 24.0;
        // GPU↔Host: PCIe Gen4 ~12 GB/s
        if (src_node == HOST_NODE || dst_node == HOST_NODE)
            return 12.5;
        // H100↔A6000: 无直连, 需staged
        if ((src_node == 2 && (dst_node == 0 || dst_node == 1)) ||
            (dst_node == 2 && (src_node == 0 || src_node == 1)))
            return 6.0;  // 经host的瓶颈带宽
        return 10.0;
        #endif
    }

    // ── [ALG3] Dijkstra路由表构建 ───────────────────────────────
    // 原版: 手写if/else判断peer直连还是staged
    // 改动: 以1/bandwidth为权重, Dijkstra算最优路径, 预算全对路由表
    void build_route_table() {
        PHILE_CHECKPOINT("build_route_table");
        size_t N = nodes_.size();

        for (size_t src = 0; src < N; ++src) {
            // Dijkstra从src出发
            std::vector<double> dist(N, std::numeric_limits<double>::infinity());
            std::vector<int> prev(N, -1);
            std::vector<bool> visited(N, false);
            dist[src] = 0.0;

            // 优先队列: (cost, node)
            using PQ = std::pair<double, int>;
            std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
            pq.push({0.0, static_cast<int>(src)});

            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (visited[u]) continue;
                visited[u] = true;

                for (auto& e : edges_) {
                    if (e.src != u) continue;
                    if (e.bandwidth_gbps <= 0) continue;

                    // cost = 1/bandwidth (越快越小)
                    double edge_cost = 1.0 / e.bandwidth_gbps;
                    if (dist[u] + edge_cost < dist[e.dst]) {
                        dist[e.dst] = dist[u] + edge_cost;
                        prev[e.dst] = u;
                        pq.push({dist[e.dst], e.dst});
                    }
                }
            }

            // 回溯路径, 填路由表
            for (size_t dst = 0; dst < N; ++dst) {
                RouteEntry& re = route_table_[src][dst];
                if (src == dst || dist[dst] >= std::numeric_limits<double>::infinity()) {
                    re.next_hop = -1;
                    re.total_cost = dist[dst];
                    re.effective_bw_gbps = 0;
                    continue;
                }

                // 回溯路径
                std::vector<int> path;
                for (int v = static_cast<int>(dst); v != -1; v = prev[v])
                    path.push_back(v);
                std::reverse(path.begin(), path.end());

                re.path = path;
                re.next_hop = (path.size() > 1) ? path[1] : path[0];
                re.total_cost = dist[dst];

                // 瓶颈带宽 = 路径上最小的边带宽
                double min_bw = std::numeric_limits<double>::infinity();
                for (size_t k = 0; k + 1 < path.size(); ++k) {
                    min_bw = std::min(min_bw, bw_matrix_[path[k]][path[k+1]]);
                }
                re.effective_bw_gbps = (min_bw < std::numeric_limits<double>::infinity())
                                     ? min_bw : 0.0;
            }
        }

        PHILE_DBG(1, "[route] built %zu×%zu route table\n", N, N);
    }

    // ── [ALG4] NUMA亲和性探测 ───────────────────────────────────
    // 原版: 完全忽略
    // 改动: 从/sys/bus/pci读取每个GPU的numa_node
    void detect_numa_affinity() {
        PHILE_CHECKPOINT("detect_numa_affinity");

        for (size_t i = 0; i < nodes_.size(); ++i) {
            auto& n = nodes_[i];
            if (n.device_id < 0) {
                n.numa_node = 0;  // host默认NUMA0
                continue;
            }

            #ifdef __CUDACC__
            // 尝试从sysfs读取
            char path[256];
            snprintf(path, sizeof(path),
                     "/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
                     n.pci_domain, n.pci_bus, n.pci_device);

            std::ifstream f(path);
            if (f.is_open()) {
                int numa = -1;
                f >> numa;
                n.numa_node = (numa >= 0) ? numa : 0;
            } else {
                n.numa_node = 1;  // 默认NUMA1 (ags1服务器配置)
            }
            #else
            n.numa_node = 1;  // 模拟: 所有GPU在NUMA1
            #endif

            PHILE_DBG(2, "[numa] GPU%d (%s) → NUMA%d\n",
                      n.device_id, n.name.c_str(), n.numa_node);
        }
    }

    // ── 数据成员 ───────────────────────────────────────────────────
    std::vector<DeviceNode> nodes_;
    std::vector<TopoEdge>   edges_;
    size_t                  gpu_count_ = 0;
    double                  bw_matrix_[MAX_NODES][MAX_NODES] = {};
    RouteEntry              route_table_[MAX_NODES][MAX_NODES] = {};
};

} // namespace topology
} // namespace philemon
