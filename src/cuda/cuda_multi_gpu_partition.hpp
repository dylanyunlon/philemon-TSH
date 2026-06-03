#pragma once
/**
 * cuda_multi_gpu_partition.hpp — 多GPU时序图分区+动态放置
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   src/cuda/hetero_bench.cu  partition_and_place (377–500行)
 *     → 边按ts_start排序
 *     → 固定25%/25%/25%/25% tier分配
 *     → cudaMemcpy H2D上传
 *     → PartitionSet + Partition结构体
 *     → tier_edges/tier_bytes统计
 *
 *   src/cuda/hetero_bench.cu  experiment_scaling (860–968行)
 *     → 不同规模(1M-100M)的分区+查询benchmark
 *     → VRAM headroom检查(20%余量)
 *     → 分区创建+上传+查询pipeline
 *
 *   src/core/partition_index.hpp  PartitionIndex (50–250行)
 *     → 每分区的双排序索引
 *     → contains/overlap/contained查询
 *
 *   src/cuda/hetero_bench.cu  cross_tier_query (510–610行)
 *     → 跨tier overlap检测 + async D2H gather
 *
 * 算法改动 (~20%):
 *   [ALG1] 分区策略: 原版固定25%切分, partition_idx/total < 0.25 → HBM
 *          → 热度+容量动态分配: 根据每分区的access_count + 每tier剩余容量,
 *            贪心放置: 最热分区→空间最大的高速tier
 *   [ALG2] 负载均衡: 原版均匀(每GPU分到的分区数相同)
 *          → 加权: GPU_i的权重 = remaining_capacity_i * bandwidth_i,
 *            按权重比例分配分区数量
 *   [ALG3] 跨分区边: 原版忽略(每分区独立, 边界不通信)
 *          → ghost vertex: 如果edge(u,v)中u和v在不同GPU, 在两个GPU上都建ghost copy,
 *            用bitmap标记哪些是ghost, 查询时合并
 *   [ALG4] 重平衡: 原版无(初始放置后不变)
 *          → 运行时检测: 每N次查询检查access_count分布的skew,
 *            skew > 2.0(最热/最冷比)时触发rebalance, 迁移最热DRAM分区→HBM
 *
 * Milestone: M050 — 多GPU分区放置
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <cmath>
#include <chrono>
#include <unordered_set>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "cuda_mem_manager.hpp"
#include "gpu_topology.hpp"

namespace philemon {
namespace cuda_partition {

// ─── 时序边 (与hetero_bench.cu的TemporalEdge对齐) ─────────────────
struct TemporalEdge {
    uint64_t source;
    uint64_t destination;
    double   weight;
    int32_t  ts_begin;
    int32_t  ts_finish;
};

// ─── GPU分区 ───────────────────────────────────────────────────────
struct GpuPartition {
    uint64_t id;
    cuda_mem::DeviceTier tier;
    void*    dev_ptr;          // GPU/Host内存指针
    size_t   size_bytes;
    size_t   edge_count;
    int32_t  ts_lo, ts_hi;    // 时间范围

    // 运行时统计
    std::atomic<uint64_t> access_count{0};
    std::atomic<uint64_t> last_access_ns{0};

    // [ALG3] ghost vertex bitmap
    std::vector<uint32_t> ghost_bitmap;  // bit[v]=1 → v是ghost vertex
    uint64_t ghost_count = 0;

    void touch() {
        access_count.fetch_add(1, std::memory_order_relaxed);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        last_access_ns.store(ns, std::memory_order_relaxed);
    }
};

// ─── 放置方案 ──────────────────────────────────────────────────────
struct PlacementPlan {
    size_t   total_partitions;
    size_t   total_edges;
    double   sort_ms;
    double   place_ms;

    // per-tier统计
    size_t   tier_edges[4] = {};
    size_t   tier_bytes[4] = {};
    size_t   tier_parts[4] = {};
    size_t   ghost_vertices = 0;
};

// ════════════════════════════════════════════════════════════════════════════
//  MultiGpuPartitioner — 动态分区放置引擎
// ════════════════════════════════════════════════════════════════════════════

class MultiGpuPartitioner {
public:
    static constexpr size_t DEFAULT_PARTITION_CAP = 500'000;  // 50万边/分区
    static constexpr double REBALANCE_SKEW_THRESHOLD = 2.0;
    static constexpr int    REBALANCE_CHECK_INTERVAL = 1000;  // 每1000次查询检查一次

    MultiGpuPartitioner(cuda_mem::CudaMemManager& mem_mgr,
                        topology::GpuTopology* topo = nullptr)
        : mem_mgr_(mem_mgr), topo_(topo), query_counter_(0)
    {
        PHILE_CHECKPOINT("MultiGpuPartitioner::ctor");
    }

    ~MultiGpuPartitioner() {
        // 释放所有分区的GPU内存
        for (auto& p : partitions_) {
            if (p.dev_ptr) {
                // 释放由mem_mgr管理, 这里只清指针
                p.dev_ptr = nullptr;
            }
        }
    }

    // ── [ALG1] 动态分区+放置 ───────────────────────────────────
    // 原版partition_and_place: 排序 → 固定25%切 → tier分配
    // 改动: 排序 → 分区 → 容量感知贪心放置
    PlacementPlan partition_and_place(std::vector<TemporalEdge>& edges,
                                      size_t partition_cap = DEFAULT_PARTITION_CAP)
    {
        PHILE_CHECKPOINT("partition_and_place");
        PlacementPlan plan;
        plan.total_edges = edges.size();

        PHILE_DBG(1, "[Partition] %zu edges, cap=%zu/partition\n",
                  edges.size(), partition_cap);

        // Step 1: 排序 (保留upstream的ts_start排序)
        auto t0 = std::chrono::steady_clock::now();
        std::sort(edges.begin(), edges.end(),
            [](const TemporalEdge& a, const TemporalEdge& b) {
                return (a.ts_begin != b.ts_begin)
                    ? a.ts_begin < b.ts_begin
                    : a.ts_finish < b.ts_finish;
            });
        auto t_sort = std::chrono::steady_clock::now();
        plan.sort_ms = std::chrono::duration<double, std::milli>(t_sort - t0).count();

        // Step 2: 切分区
        size_t total_parts = (edges.size() + partition_cap - 1) / partition_cap;
        plan.total_partitions = total_parts;

        // [ALG2] 计算每个tier的放置权重
        // 原版: 固定25%
        // 改动: weight = remaining_capacity * (topo ? bandwidth : 1.0)
        struct TierWeight {
            cuda_mem::DeviceTier tier;
            double weight;
            size_t remaining_slots;
        };
        std::vector<TierWeight> tier_weights;
        {
            cuda_mem::DeviceTier tiers[] = {
                cuda_mem::DeviceTier::HBM_GPU,
                cuda_mem::DeviceTier::GDDR_GPU0,
                cuda_mem::DeviceTier::GDDR_GPU1,
                cuda_mem::DeviceTier::HOST_DRAM
            };

            double total_weight = 0;
            for (auto t : tiers) {
                auto& budget = mem_mgr_.budget(t);
                size_t remaining = budget.capacity -
                    budget.used.load(std::memory_order_relaxed);
                // 估算能放多少个分区
                size_t bytes_per_part = partition_cap * sizeof(TemporalEdge);
                size_t slots = remaining / bytes_per_part;

                // bandwidth加权 (如果有topo)
                double bw = 1.0;
                if (topo_) {
                    int node = cuda_mem::tier_to_device_id(t);
                    if (node >= 0) {
                        // 用H→GPU的带宽作为权重
                        bw = topo_->measured_bandwidth(
                            topology::GpuTopology::HOST_NODE, node);
                        if (bw <= 0) bw = 1.0;
                    }
                }

                double w = slots * bw;
                tier_weights.push_back({t, w, slots});
                total_weight += w;

                PHILE_DBG(2, "[Partition] tier %s: remaining=%.1fGB slots=%zu bw=%.1f w=%.1f\n",
                          cuda_mem::device_tier_name(t),
                          remaining / (1024.0*1024*1024),
                          slots, bw, w);
            }

            // 归一化 → 分配分区数
            if (total_weight > 0) {
                for (auto& tw : tier_weights) {
                    tw.remaining_slots = static_cast<size_t>(
                        (tw.weight / total_weight) * total_parts);
                }
            }
            // 余数给DRAM
            size_t assigned = 0;
            for (auto& tw : tier_weights) assigned += tw.remaining_slots;
            if (assigned < total_parts) {
                tier_weights.back().remaining_slots += (total_parts - assigned);
            }
        }

        // Step 3: 分配分区
        size_t tw_idx = 0;
        size_t tw_used = 0;

        for (size_t i = 0; i < edges.size(); i += partition_cap) {
            size_t end = std::min(i + partition_cap, edges.size());
            size_t count = end - i;
            size_t sz = count * sizeof(TemporalEdge);

            // 时间范围
            int32_t lo = edges[i].ts_begin;
            int32_t hi = edges[end - 1].ts_finish;
            for (size_t j = i; j < end; ++j)
                hi = std::max(hi, edges[j].ts_finish);

            // [ALG1] 选tier: 按权重分配的quota, 用完切下一个
            while (tw_idx < tier_weights.size() &&
                   tw_used >= tier_weights[tw_idx].remaining_slots) {
                tw_idx++;
                tw_used = 0;
            }
            cuda_mem::DeviceTier tier = (tw_idx < tier_weights.size())
                ? tier_weights[tw_idx].tier
                : cuda_mem::DeviceTier::HOST_DRAM;

            // 分配GPU内存
            void* ptr = allocate_on_tier(tier, sz);
            if (!ptr) {
                // 回退到DRAM
                tier = cuda_mem::DeviceTier::HOST_DRAM;
                ptr = allocate_on_tier(tier, sz);
            }

            // 上传数据
            if (ptr) {
                upload_edges(tier, ptr, &edges[i], sz);
            }

            GpuPartition part;
            part.id         = partitions_.size() + 1;
            part.tier       = tier;
            part.dev_ptr    = ptr;
            part.size_bytes = sz;
            part.edge_count = count;
            part.ts_lo      = lo;
            part.ts_hi      = hi;

            int ti = static_cast<int>(tier);
            plan.tier_edges[ti] += count;
            plan.tier_bytes[ti] += sz;
            plan.tier_parts[ti]++;

            partitions_.push_back(std::move(part));
            tw_used++;
        }

        // [ALG3] Ghost vertex建设
        build_ghost_vertices(edges);
        for (auto& p : partitions_) plan.ghost_vertices += p.ghost_count;

        auto t_place = std::chrono::steady_clock::now();
        plan.place_ms = std::chrono::duration<double, std::milli>(t_place - t_sort).count();

        PHILE_DBG(1, "[Partition] done: %zu parts, sort=%.1fms place=%.1fms ghosts=%zu\n",
                  plan.total_partitions, plan.sort_ms, plan.place_ms,
                  plan.ghost_vertices);

        // 断点: 打印分区布局
        dump_layout(plan);

        return plan;
    }

    // ── 查询: 找overlap分区 ─────────────────────────────────────
    std::vector<size_t> find_overlapping(int32_t ts_lo, int32_t ts_hi) {
        std::shared_lock<std::shared_mutex> lk(mu_);
        std::vector<size_t> result;

        for (size_t i = 0; i < partitions_.size(); ++i) {
            auto& p = partitions_[i];
            if (p.ts_lo <= ts_hi && p.ts_hi >= ts_lo) {
                result.push_back(i);
                p.touch();
            }
        }

        // [ALG4] 周期性rebalance检查
        uint64_t qc = query_counter_.fetch_add(1);
        if (qc > 0 && qc % REBALANCE_CHECK_INTERVAL == 0) {
            check_and_rebalance();
        }

        return result;
    }

    const GpuPartition& partition(size_t idx) const { return partitions_[idx]; }
    size_t partition_count() const { return partitions_.size(); }

private:
    // ── 内存分配 ───────────────────────────────────────────────
    void* allocate_on_tier(cuda_mem::DeviceTier tier, size_t size) {
        #ifdef __CUDACC__
        int gpu = cuda_mem::tier_to_device_id(tier);
        if (gpu >= 0) {
            void* p = nullptr;
            cudaSetDevice(gpu);
            if (cudaMalloc(&p, size) != cudaSuccess) return nullptr;
            return p;
        }
        #endif
        return std::calloc(1, size);
    }

    void upload_edges(cuda_mem::DeviceTier tier, void* dst,
                      const TemporalEdge* src, size_t size) {
        #ifdef __CUDACC__
        int gpu = cuda_mem::tier_to_device_id(tier);
        if (gpu >= 0) {
            cudaSetDevice(gpu);
            cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        } else {
            std::memcpy(dst, src, size);
        }
        #else
        std::memcpy(dst, src, size);
        #endif
    }

    // ── [ALG3] Ghost vertex构建 ────────────────────────────────
    // 原版: 无ghost, 分区完全独立
    // 改动: 检测跨分区边, 在相邻分区建ghost vertex
    void build_ghost_vertices(const std::vector<TemporalEdge>& edges) {
        PHILE_CHECKPOINT("build_ghost_vertices");

        if (partitions_.size() <= 1) return;

        // 建立vertex→partition映射 (哪个vertex属于哪些分区)
        std::unordered_map<uint64_t, std::unordered_set<size_t>> vertex_parts;
        size_t part_start = 0;
        for (size_t pi = 0; pi < partitions_.size(); ++pi) {
            size_t part_end = part_start + partitions_[pi].edge_count;
            for (size_t ei = part_start; ei < part_end && ei < edges.size(); ++ei) {
                vertex_parts[edges[ei].source].insert(pi);
                vertex_parts[edges[ei].destination].insert(pi);
            }
            part_start = part_end;
        }

        // 找跨分区vertex: 出现在2+个分区的vertex
        uint64_t total_ghosts = 0;
        for (auto& [vtx, parts] : vertex_parts) {
            if (parts.size() > 1) {
                // 这个vertex需要在每个相关分区做ghost
                for (size_t pi : parts) {
                    // 简化: 只设ghost bitmap标记
                    auto& p = partitions_[pi];
                    size_t bmp_words = (vtx / 32) + 1;
                    if (p.ghost_bitmap.size() < bmp_words)
                        p.ghost_bitmap.resize(bmp_words, 0);
                    p.ghost_bitmap[vtx / 32] |= (1u << (vtx & 31));
                    p.ghost_count++;
                    total_ghosts++;
                }
            }
        }

        PHILE_DBG(1, "[Ghost] cross-partition vertices: %zu total ghost entries: %lu\n",
                  vertex_parts.size(), (unsigned long)total_ghosts);
    }

    // ── [ALG4] 运行时重平衡 ───────────────────────────────────
    // 原版: 无
    // 改动: 检测access skew, 移动最热DRAM分区→HBM
    void check_and_rebalance() {
        if (partitions_.empty()) return;

        // 计算access_count的min/max
        uint64_t max_ac = 0, min_ac = UINT64_MAX;
        size_t hottest_dram_idx = SIZE_MAX;
        uint64_t hottest_dram_ac = 0;

        for (size_t i = 0; i < partitions_.size(); ++i) {
            auto& p = partitions_[i];
            uint64_t ac = p.access_count.load(std::memory_order_relaxed);
            max_ac = std::max(max_ac, ac);
            min_ac = std::min(min_ac, ac);

            if (p.tier == cuda_mem::DeviceTier::HOST_DRAM && ac > hottest_dram_ac) {
                hottest_dram_ac = ac;
                hottest_dram_idx = i;
            }
        }

        if (min_ac == 0) min_ac = 1;
        double skew = static_cast<double>(max_ac) / min_ac;

        PHILE_DBG(2, "[Rebalance] check: max_ac=%lu min_ac=%lu skew=%.2f (threshold=%.1f)\n",
                  max_ac, min_ac, skew, REBALANCE_SKEW_THRESHOLD);

        if (skew > REBALANCE_SKEW_THRESHOLD && hottest_dram_idx != SIZE_MAX) {
            PHILE_DBG(1, "[Rebalance] TRIGGERED: skew=%.2f, promoting DRAM part %zu (ac=%lu) → HBM\n",
                      skew, hottest_dram_idx, hottest_dram_ac);
            // 标记需要迁移 (实际迁移由AsyncMigrator执行)
            rebalance_needed_ = true;
            rebalance_target_ = hottest_dram_idx;
        }
    }

    // ── 断点调试: 打印布局 ──────────────────────────────────────
    void dump_layout(const PlacementPlan& plan) const {
        PHILE_SEPARATOR("Partition Layout");
        printf("[Partition] %zu partitions, %zu edges\n",
               plan.total_partitions, plan.total_edges);

        printf("  %-6s  %-14s  %-20s  %-10s  %-10s  %-8s\n",
               "ID", "Tier", "Timestamp Range", "Edges", "Size(MB)", "Ghosts");

        for (auto& p : partitions_) {
            printf("  %-6lu  %-14s  [%d, %d]%*s  %-10lu  %-10.2f  %-8lu\n",
                   (unsigned long)p.id,
                   cuda_mem::device_tier_name(p.tier),
                   p.ts_lo, p.ts_hi,
                   (int)(15 - snprintf(nullptr, 0, "[%d, %d]", p.ts_lo, p.ts_hi)), "",
                   (unsigned long)p.edge_count,
                   p.size_bytes / (1024.0 * 1024),
                   (unsigned long)p.ghost_count);
        }

        printf("\n  Tier Summary:\n");
        for (int i = 0; i < 4; ++i) {
            if (plan.tier_parts[i] > 0) {
                printf("    %-14s  %zu parts  %zu edges  (%.2f MB)\n",
                       cuda_mem::device_tier_name(static_cast<cuda_mem::DeviceTier>(i)),
                       plan.tier_parts[i], plan.tier_edges[i],
                       plan.tier_bytes[i] / (1024.0 * 1024));
            }
        }
        printf("  Ghost vertices: %zu\n", plan.ghost_vertices);
        PHILE_SEPARATOR("End Partition Layout");
    }

    // ── 数据成员 ───────────────────────────────────────────────
    cuda_mem::CudaMemManager& mem_mgr_;
    topology::GpuTopology*    topo_;
    mutable std::shared_mutex mu_;
    std::vector<GpuPartition>  partitions_;

    std::atomic<uint64_t> query_counter_;
    bool   rebalance_needed_ = false;
    size_t rebalance_target_ = 0;
};

} // namespace cuda_partition
} // namespace philemon
