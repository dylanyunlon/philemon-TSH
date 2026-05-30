#ifndef PHILEMON_TIER_COST_MODEL_HPP
#define PHILEMON_TIER_COST_MODEL_HPP
/**
 * tier_cost_model.hpp — Heterogeneous Memory Cost Model
 *
 * 骨架来源:
 *   - upstream/rapidstore/wrapper/wrapper.h snapshot_edges 回调模式 (249行)
 *   - NCCL ncclMemAlloc tier dispatch pattern (infra-refs/nccl)
 *   - DeepSpeed PartitionedOptimizerSwapper swap cost estimation
 *
 * 修改 (~20%):
 *   - 增加 TierCostModel: 基于 HBM/GDDR/DRAM 硬件规格的代价计算
 *   - 增加 estimate_query_cost(): 预估 temporal query 跨层总代价
 *   - 增加 optimal_tier_assignment(): 贪心/ILP 最优 tier 分配
 *   - 增加 migration_cost(): 计算数据迁移开销 (cudaMemcpy model)
 *   - 增加 per-operation 代价追踪和统计打印
 *   - 增加 breakpoint/print 友好的状态快照
 *
 * Milestone: M018 — Tier cost model for cross-tier BFS/SSSP
 */

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <atomic>
#include <chrono>
#include <cassert>

#include "../debug/philemon_debug.hpp"
#include "../loader/ldbc_types.hpp"

namespace philemon {
namespace cost_model {

// ─── Hardware tier specifications ───────────────────────────────────
// Default values from H100 + A6000 + DDR5 system
struct TierSpec {
    const char*  name;
    double       access_latency_ns;    // per-access latency
    double       bandwidth_gbps;       // sustained bandwidth
    uint64_t     capacity_bytes;       // total capacity
    double       migration_overhead_us; // per-migration setup cost (μs)

    // Derived: bytes per nanosecond
    double bytes_per_ns() const {
        return bandwidth_gbps * 1e9 / 8.0 / 1e9;  // GB/s → B/ns
    }

    void dump() const {
        std::printf("  [%s] lat=%.1fns bw=%.1fGB/s cap=%luMB "
                    "migrate_overhead=%.1fμs bytes/ns=%.2f\n",
                    name, access_latency_ns, bandwidth_gbps,
                    (unsigned long)(capacity_bytes / (1024*1024)),
                    migration_overhead_us, bytes_per_ns());
    }
};

// ─── Cost model result ──────────────────────────────────────────────
struct CostEstimate {
    double   total_ns;           // total estimated time in nanoseconds
    double   access_ns;          // time spent in random access
    double   transfer_ns;        // time spent in bulk transfer
    double   migration_ns;       // time for any tier-to-tier migration
    uint64_t hbm_accesses;
    uint64_t gddr_accesses;
    uint64_t dram_accesses;
    uint64_t cross_tier_edges;   // edges crossing tier boundaries

    CostEstimate() : total_ns(0), access_ns(0), transfer_ns(0),
                     migration_ns(0), hbm_accesses(0), gddr_accesses(0),
                     dram_accesses(0), cross_tier_edges(0) {}

    void dump(const char* tag = "CostEstimate") const {
        std::printf("──── %s ────\n", tag);
        std::printf("  total=%.2f μs (access=%.2f + transfer=%.2f + migrate=%.2f)\n",
                    total_ns / 1000.0, access_ns / 1000.0,
                    transfer_ns / 1000.0, migration_ns / 1000.0);
        std::printf("  accesses: HBM=%lu GDDR=%lu DRAM=%lu cross_tier=%lu\n",
                    (unsigned long)hbm_accesses,
                    (unsigned long)gddr_accesses,
                    (unsigned long)dram_accesses,
                    (unsigned long)cross_tier_edges);
        std::printf("──── End %s ────\n", tag);
    }
};

// ─── Migration cost detail ──────────────────────────────────────────
struct MigrationPlan {
    uint8_t  from_tier;
    uint8_t  to_tier;
    uint64_t bytes;
    double   estimated_ns;
    uint64_t partition_id;

    void dump() const {
        const char* names[] = {"HBM", "GDDR", "DRAM"};
        std::printf("  [migrate] partition=%lu %s→%s %luKB est=%.2fμs\n",
                    (unsigned long)partition_id,
                    names[from_tier % 3], names[to_tier % 3],
                    (unsigned long)(bytes / 1024),
                    estimated_ns / 1000.0);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// TierCostModel — Main cost estimation engine
// ═══════════════════════════════════════════════════════════════════════
class TierCostModel {
private:
    TierSpec tiers_[3];   // HBM=0, GDDR=1, DRAM=2

    // ─── Accumulated statistics ─────────────────────────────────
    std::atomic<uint64_t> total_queries_{0};
    std::atomic<uint64_t> total_migrations_{0};
    double accumulated_cost_ns_{0.0};

public:
    TierCostModel() {
        // Default H100 + A6000 + DDR5 configuration
        tiers_[0] = {"HBM",  1.0,  3352.0,
                     80ULL * 1024 * 1024 * 1024, 2.0};
        tiers_[1] = {"GDDR", 5.0,  768.0,
                     48ULL * 1024 * 1024 * 1024, 5.0};
        tiers_[2] = {"DRAM", 50.0, 204.8,
                     512ULL * 1024 * 1024 * 1024, 50.0};
    }

    TierCostModel(TierSpec hbm, TierSpec gddr, TierSpec dram) {
        tiers_[0] = hbm;
        tiers_[1] = gddr;
        tiers_[2] = dram;
    }

    // ─── Single access cost ─────────────────────────────────────
    double access_cost_ns(uint8_t tier, uint64_t bytes) const {
        assert(tier < 3);
        const auto& t = tiers_[tier];
        // Latency + bandwidth-limited transfer time
        double transfer_time = static_cast<double>(bytes) / t.bytes_per_ns();
        return t.access_latency_ns + transfer_time;
    }

    // ─── Migration cost (from tier A to tier B) ─────────────────
    // Models cudaMemcpy overhead:
    //   cost = setup + max(read_from_src, write_to_dst)
    double migration_cost_ns(uint8_t from_tier, uint8_t to_tier,
                             uint64_t bytes) const {
        assert(from_tier < 3 && to_tier < 3);
        if (from_tier == to_tier) return 0.0;

        double setup = std::max(tiers_[from_tier].migration_overhead_us,
                                tiers_[to_tier].migration_overhead_us) * 1000.0;
        double read_ns = static_cast<double>(bytes) /
                         tiers_[from_tier].bytes_per_ns();
        double write_ns = static_cast<double>(bytes) /
                          tiers_[to_tier].bytes_per_ns();
        return setup + std::max(read_ns, write_ns);
    }

    // ─── Estimate query cost across multiple partitions ─────────
    // partitions: list of (tier_id, edge_count, edge_bytes) tuples
    CostEstimate estimate_query_cost(
        const std::vector<std::tuple<uint8_t, uint64_t, uint64_t>>& partitions,
        bool scan_mode = true) const
    {
        CostEstimate est;

        for (const auto& [tier, edges, bytes] : partitions) {
            double cost;
            if (scan_mode) {
                // Sequential scan: bandwidth-dominated
                cost = static_cast<double>(bytes) / tiers_[tier].bytes_per_ns();
                cost += tiers_[tier].access_latency_ns;  // initial access
            } else {
                // Random access: latency-dominated
                cost = edges * tiers_[tier].access_latency_ns;
            }

            est.access_ns += cost;

            switch (tier) {
                case 0: est.hbm_accesses += edges; break;
                case 1: est.gddr_accesses += edges; break;
                case 2: est.dram_accesses += edges; break;
            }
        }

        est.total_ns = est.access_ns + est.transfer_ns + est.migration_ns;
        return est;
    }

    // ─── Estimate BFS frontier expansion cost ───────────────────
    // For each level: frontier vertices may be in different tiers
    CostEstimate estimate_bfs_level_cost(
        const std::vector<std::pair<uint8_t, uint64_t>>& frontier_by_tier,
        uint64_t avg_degree) const
    {
        CostEstimate est;

        for (const auto& [tier, count] : frontier_by_tier) {
            // Each frontier vertex: 1 random access + avg_degree edge scans
            uint64_t edges = count * avg_degree;
            uint64_t bytes = edges * 24;  // ~24 bytes per edge (src+dst+ts)

            double vertex_cost = count * tiers_[tier].access_latency_ns;
            double edge_cost = static_cast<double>(bytes) /
                               tiers_[tier].bytes_per_ns();

            est.access_ns += vertex_cost + edge_cost;

            switch (tier) {
                case 0: est.hbm_accesses += edges; break;
                case 1: est.gddr_accesses += edges; break;
                case 2: est.dram_accesses += edges; break;
            }
        }

        // Cross-tier edges: neighbors of HBM vertices might be in DRAM
        // Penalize cross-tier traversals
        uint64_t total_frontier = 0;
        for (const auto& [tier, count] : frontier_by_tier) {
            total_frontier += count;
        }
        est.cross_tier_edges = total_frontier * avg_degree / 3;
        est.access_ns += est.cross_tier_edges *
                         (tiers_[2].access_latency_ns -
                          tiers_[0].access_latency_ns);

        est.total_ns = est.access_ns;
        return est;
    }

    // ─── Optimal tier assignment (greedy) ───────────────────────
    // Given partition access frequencies, assign to tiers to minimize
    // total cost subject to capacity constraints.
    //
    // partitions: (access_frequency, edge_count, bytes_per_partition)
    // Returns: optimal tier assignment for each partition
    std::vector<uint8_t> optimal_tier_assignment(
        const std::vector<std::tuple<double, uint64_t, uint64_t>>& partitions)
        const
    {
        debug::ScopedTimer timer("optimal_tier_assignment");

        size_t n = partitions.size();
        std::vector<uint8_t> assignment(n, 2);  // default: all DRAM

        // Sort by access frequency (descending) — hottest first
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return std::get<0>(partitions[a]) > std::get<0>(partitions[b]);
        });

        uint64_t remaining_hbm = tiers_[0].capacity_bytes;
        uint64_t remaining_gddr = tiers_[1].capacity_bytes;

        for (size_t i = 0; i < n; i++) {
            size_t idx = order[i];
            uint64_t bytes = std::get<2>(partitions[idx]);

            if (bytes <= remaining_hbm) {
                assignment[idx] = 0;  // HBM
                remaining_hbm -= bytes;
            } else if (bytes <= remaining_gddr) {
                assignment[idx] = 1;  // GDDR
                remaining_gddr -= bytes;
            } else {
                assignment[idx] = 2;  // DRAM
            }
        }

        // Debug: print assignment summary
        uint64_t counts[3] = {0, 0, 0};
        uint64_t bytes_per_tier[3] = {0, 0, 0};
        for (size_t i = 0; i < n; i++) {
            counts[assignment[i]]++;
            bytes_per_tier[assignment[i]] += std::get<2>(partitions[i]);
        }
        PHILE_DBG(1, "[optimal_assignment] HBM: %lu parts/%luMB  "
                  "GDDR: %lu parts/%luMB  DRAM: %lu parts/%luMB",
                  (unsigned long)counts[0],
                  (unsigned long)(bytes_per_tier[0] / (1024*1024)),
                  (unsigned long)counts[1],
                  (unsigned long)(bytes_per_tier[1] / (1024*1024)),
                  (unsigned long)counts[2],
                  (unsigned long)(bytes_per_tier[2] / (1024*1024)));

        return assignment;
    }

    // ─── Plan migration batch ───────────────────────────────────
    // Given current and target assignments, generate migration plan
    std::vector<MigrationPlan> plan_migrations(
        const std::vector<uint8_t>& current,
        const std::vector<uint8_t>& target,
        const std::vector<uint64_t>& partition_bytes) const
    {
        std::vector<MigrationPlan> plans;
        double total_cost = 0;

        for (size_t i = 0; i < current.size(); i++) {
            if (current[i] != target[i]) {
                MigrationPlan plan;
                plan.from_tier = current[i];
                plan.to_tier = target[i];
                plan.bytes = (i < partition_bytes.size()) ?
                             partition_bytes[i] : 0;
                plan.estimated_ns = migration_cost_ns(plan.from_tier,
                                                      plan.to_tier,
                                                      plan.bytes);
                plan.partition_id = i;

                total_cost += plan.estimated_ns;
                plans.push_back(plan);
            }
        }

        PHILE_DBG(1, "[plan_migrations] %zu migrations, total_cost=%.2fms",
                  plans.size(), total_cost / 1e6);

        // Print individual plans at debug level 2
        if (debug::get_debug_level() >= 2) {
            for (const auto& p : plans) {
                p.dump();
            }
        }

        return plans;
    }

    // ─── Full state dump ────────────────────────────────────────
    void dump() const {
        std::printf("════ TierCostModel ════\n");
        for (int i = 0; i < 3; i++) {
            tiers_[i].dump();
        }
        std::printf("  total_queries=%lu  total_migrations=%lu  "
                    "accumulated_cost=%.2fms\n",
                    (unsigned long)total_queries_.load(),
                    (unsigned long)total_migrations_.load(),
                    accumulated_cost_ns_ / 1e6);
        std::printf("════ End CostModel ════\n");
    }

    // ─── Tier spec accessors ────────────────────────────────────
    const TierSpec& tier(uint8_t idx) const { return tiers_[idx % 3]; }
    double latency_ratio(uint8_t from, uint8_t to) const {
        return tiers_[to].access_latency_ns / tiers_[from].access_latency_ns;
    }

    // ─── Prefetch benefit estimator ─────────────────────────────
    // Estimates how much time is saved by prefetching data from
    // a cold tier to a hot tier before the query arrives
    double prefetch_benefit_ns(uint8_t cold_tier, uint8_t hot_tier,
                                uint64_t bytes, uint64_t expected_accesses)
                                const {
        double cold_cost = expected_accesses * tiers_[cold_tier].access_latency_ns
                           + static_cast<double>(bytes) /
                             tiers_[cold_tier].bytes_per_ns();
        double hot_cost  = expected_accesses * tiers_[hot_tier].access_latency_ns
                           + static_cast<double>(bytes) /
                             tiers_[hot_tier].bytes_per_ns();
        double migrate   = migration_cost_ns(cold_tier, hot_tier, bytes);

        double benefit = cold_cost - hot_cost - migrate;

        PHILE_DBG(3, "[prefetch_benefit] %s→%s %luKB ×%lu accesses: "
                  "cold=%.1fμs hot=%.1fμs migrate=%.1fμs benefit=%.1fμs",
                  tiers_[cold_tier].name, tiers_[hot_tier].name,
                  (unsigned long)(bytes / 1024),
                  (unsigned long)expected_accesses,
                  cold_cost / 1000, hot_cost / 1000,
                  migrate / 1000, benefit / 1000);

        return benefit;
    }
};

}  // namespace cost_model
}  // namespace philemon

#endif  // PHILEMON_TIER_COST_MODEL_HPP
