#ifndef PHILEMON_COST_ESTIMATOR_HPP
#define PHILEMON_COST_ESTIMATOR_HPP
/**
 * cost_estimator.hpp — Roofline + AMAT 分层代价估算器
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/wrapper.h                              (249行)
 *     → snapshot_edges() 模板回调 s->edges(index, callback, logical)
 *     → snapshot_vertex_count / snapshot_degree
 *     → 100% 保留: 回调 pattern 用于采样代价
 *
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → execute_query() 的查询分发 for-loop (BFS/SSSP/PR/WCC/TC)
 *     → page_rank() 中 outgoing_contrib / incoming_total 的迭代模式
 *     → throughput = edges / duration 计算
 *     → 100% 保留: 查询分发结构 + throughput计算
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp     (537行)
 *     → finish_commit() 的 CAS loop (timestamp追赶)
 *     → get_write_timestamp() 的 fetch_add
 *     → 100% 保留: CAS + fetch_add 原子操作 pattern
 *
 * 算法修改 (~20%):
 *   - [MOD] 固定bandwidth_gbps → Roofline model: 区分 compute-bound
 *           和 memory-bound, 用 operational intensity (ops/byte)
 *           判断query瓶颈在哪; upstream的TierCostModel只用带宽
 *   - [MOD] 均匀access代价 → AMAT (Average Memory Access Time):
 *           AMAT = hit_time + miss_rate × miss_penalty, 递归应用于
 *           HBM→GDDR→DRAM三层; upstream用简单乘法
 *   - [MOD] 无带宽约束 → PCIe budget: 迁移代价受PCIe总带宽限制,
 *           并发迁移互相挤占; upstream假设独占带宽
 *   - [NEW] QueryCostProfile: per-query类型的代价模型参数
 *           (BFS=random-access heavy, PR=streaming heavy)
 *   - [NEW] CostBreakdown: 分层报告每个tier贡献的代价占比
 *
 * 断点调试:
 *   PHILE_COST_DUMP(estimator)     — 全量打印 Roofline参数+AMAT+预算
 *   PHILE_COST_PROFILE(estimator,q) — 打印某查询类型的代价明细
 *   PHILE_COST_BREAKPOINT(e,tag)   — RAII guard
 *
 * Milestone: M053 — Cost estimator
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <chrono>
#include <cassert>
#include <string>

#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace cost_model {

// ─── 查询类型 ───────────────────────────────────────────────────────
enum class QueryType : uint8_t { BFS, SSSP, PR, WCC, TC, MIXED };

static const char* query_type_name(QueryType q) {
    switch (q) {
        case QueryType::BFS:   return "BFS";
        case QueryType::SSSP:  return "SSSP";
        case QueryType::PR:    return "PR";
        case QueryType::WCC:   return "WCC";
        case QueryType::TC:    return "TC";
        case QueryType::MIXED: return "MIXED";
    }
    return "???";
}

// ─── Tier硬件规格 ──────────────────────────────────────────────────
// upstream的TierSpec保留, 扩展operational intensity参数
struct TierHWSpec {
    const char*  name;
    double       access_latency_ns;
    double       bandwidth_gbps;
    uint64_t     capacity_bytes;
    double       migration_overhead_us;
    // [NEW] Roofline参数
    double       peak_flops_gflops;    // 计算峰值 (用于roofline交叉点)

    double bytes_per_ns() const {
        return bandwidth_gbps * 1e9 / 8.0 / 1e9;
    }

    // [MOD] Roofline ridge point: operational intensity at which
    // compute becomes the bottleneck
    double ridge_point() const {
        if (bandwidth_gbps <= 0) return 0;
        return peak_flops_gflops / bandwidth_gbps;  // flops/byte
    }

    void dump() const {
        std::printf("  [%s] lat=%.1fns bw=%.1fGB/s cap=%luMB "
                    "peak=%.1fGF/s ridge=%.2f ops/byte\n",
                    name, access_latency_ns, bandwidth_gbps,
                    (unsigned long)(capacity_bytes / (1024*1024)),
                    peak_flops_gflops, ridge_point());
    }
};

// ─── [NEW] Per-query代价profile ─────────────────────────────────────
// 不同算法的访存模式差异很大, 这里为每种query类型设定参数
struct QueryCostProfile {
    QueryType type;
    double    ops_per_edge;          // 每条边的计算量 (FLOP)
    double    bytes_per_edge;        // 每条边的内存访问量
    double    random_access_ratio;   // [0,1] 随机访问占比 (影响cache)
    double    iteration_count;       // 迭代次数 (PR多轮, BFS一次)

    double operational_intensity() const {
        if (bytes_per_edge <= 0) return 0;
        return ops_per_edge / bytes_per_edge;
    }

    void dump() const {
        std::printf("  [%s] ops/edge=%.1f bytes/edge=%.1f "
                    "random=%.2f iters=%.0f OI=%.3f\n",
                    query_type_name(type), ops_per_edge, bytes_per_edge,
                    random_access_ratio, iteration_count,
                    operational_intensity());
    }
};

// 默认profile (基于典型图算法的特征)
inline QueryCostProfile default_profile(QueryType q) {
    switch (q) {
        // BFS: 每边1次比较, 读8B vertex + 8B edge, 高随机, 1次遍历
        case QueryType::BFS:
            return {q, 1.0, 16.0, 0.85, 1.0};
        // SSSP: 每边1次加+比较, 读16B (vertex+weight), 高随机
        case QueryType::SSSP:
            return {q, 2.0, 24.0, 0.80, 1.0};
        // PR: 每边1次乘+加, 读8B dest, 多轮迭代, 流式访问
        case QueryType::PR:
            return {q, 2.0, 8.0, 0.30, 20.0};
        // WCC: 每边1次union-find, 读8B, 中等随机
        case QueryType::WCC:
            return {q, 3.0, 16.0, 0.60, 1.0};
        // TC: 每边需要intersect, 读16B×2, 高计算
        case QueryType::TC:
            return {q, 10.0, 32.0, 0.70, 1.0};
        default:
            return {q, 2.0, 16.0, 0.50, 1.0};
    }
}

// ─── [MOD] AMAT model ──────────────────────────────────────────────
// upstream用 access_latency_ns * count 简单乘法.
// 这里用 AMAT = hit_time + miss_rate × miss_penalty 递归:
//   AMAT_HBM  = HBM_lat
//   AMAT_GDDR = GDDR_lat + P(miss_GDDR→DRAM) × DRAM_penalty
//   总 AMAT   = P(HBM) × AMAT_HBM + P(GDDR) × AMAT_GDDR + P(DRAM) × DRAM_lat

struct AMATResult {
    double amat_ns;          // 综合 AMAT
    double hbm_contrib_ns;
    double gddr_contrib_ns;
    double dram_contrib_ns;

    void dump(const char* tag = "AMAT") const {
        std::printf("  [%s] total=%.2fns "
                    "(HBM=%.2f GDDR=%.2f DRAM=%.2f)\n",
                    tag, amat_ns, hbm_contrib_ns,
                    gddr_contrib_ns, dram_contrib_ns);
    }
};

// ─── CostBreakdown — 一次估算的详细结果 ─────────────────────────────
struct CostBreakdown {
    double total_ns;
    double compute_ns;       // 纯计算耗时
    double memory_ns;        // 内存访问耗时
    double migration_ns;     // 可能的迁移开销
    AMATResult amat;

    // Roofline判定
    bool   is_compute_bound;
    double operational_intensity;
    double achievable_gflops;

    // 按tier的边数分布
    uint64_t edges_in_hbm;
    uint64_t edges_in_gddr;
    uint64_t edges_in_dram;
    uint64_t cross_tier_edges;

    CostBreakdown() { std::memset(this, 0, sizeof(*this)); }

    void dump(const char* tag = "CostBreakdown") const {
        std::printf("──── %s ────\n", tag);
        std::printf("  total=%.2fμs (compute=%.2f memory=%.2f "
                    "migrate=%.2f)\n",
                    total_ns / 1000, compute_ns / 1000,
                    memory_ns / 1000, migration_ns / 1000);
        std::printf("  roofline: OI=%.3f %s "
                    "achievable=%.1fGF/s\n",
                    operational_intensity,
                    is_compute_bound ? "COMPUTE-BOUND" : "MEMORY-BOUND",
                    achievable_gflops);
        std::printf("  edges: HBM=%lu GDDR=%lu DRAM=%lu cross=%lu\n",
                    (unsigned long)edges_in_hbm,
                    (unsigned long)edges_in_gddr,
                    (unsigned long)edges_in_dram,
                    (unsigned long)cross_tier_edges);
        amat.dump();
    }
};

// ═══════════════════════════════════════════════════════════════════════
// CostEstimator — 主类
// ═══════════════════════════════════════════════════════════════════════
class CostEstimator {
    std::array<TierHWSpec, 3> tiers_;

    // PCIe带宽预算 (所有迁移共享)
    double pcie_bandwidth_gbps_ = 32.0;  // PCIe 4.0 x16
    double pcie_utilization_ = 0.0;       // 当前利用率 [0,1]

    // upstream保留: 原子计数器
    std::atomic<uint64_t> total_estimates_{0};
    std::atomic<uint64_t> total_migrations_{0};
    double accumulated_cost_ns_ = 0;

public:
    CostEstimator() {
        // 默认: H100 HBM3 + A6000 GDDR6 + DDR5
        tiers_[0] = {"HBM3",  25.0, 3350.0,
                      80ULL * 1024*1024*1024, 5.0, 60000.0};
        tiers_[1] = {"GDDR6", 100.0, 768.0,
                      48ULL * 1024*1024*1024, 15.0, 20000.0};
        tiers_[2] = {"DDR5",  75.0, 76.8,
                      256ULL * 1024*1024*1024, 50.0, 1000.0};
    }

    void set_tier(uint8_t idx, TierHWSpec spec) {
        assert(idx < 3);
        tiers_[idx] = spec;
    }

    void set_pcie_budget(double gbps) { pcie_bandwidth_gbps_ = gbps; }

    // ── [MOD] AMAT 计算 ──
    // upstream: total_ns = access_count × tier.latency
    // 这里: AMAT递归, 考虑各tier的命中率
    AMATResult compute_amat(double hbm_hit_rate, double gddr_hit_rate) const {
        // HBM层: 命中直接返回
        double hbm_lat = tiers_[0].access_latency_ns;
        // GDDR层: 在GDDR上命中 or 穿透到DRAM
        double gddr_lat = tiers_[1].access_latency_ns;
        double dram_lat = tiers_[2].access_latency_ns;

        // GDDR的miss penalty = 访问DRAM的延迟 + 传输开销
        double gddr_miss_penalty = dram_lat + 50.0;  // +50ns DMA setup
        double amat_gddr = gddr_lat + (1.0 - gddr_hit_rate) * gddr_miss_penalty;

        // 总AMAT: HBM命中 → HBM_lat, 否则去GDDR层
        double hbm_miss_penalty = amat_gddr + 30.0;  // +30ns tier切换
        double amat_total = hbm_lat + (1.0 - hbm_hit_rate) * hbm_miss_penalty;

        AMATResult r;
        r.amat_ns = amat_total;
        r.hbm_contrib_ns = hbm_hit_rate * hbm_lat;
        r.gddr_contrib_ns = (1.0 - hbm_hit_rate) * gddr_hit_rate * amat_gddr;
        r.dram_contrib_ns = (1.0 - hbm_hit_rate) * (1.0 - gddr_hit_rate) *
                            (dram_lat + gddr_miss_penalty);

        PHILE_DBG(3, "[AMAT] hbm_hr=%.2f gddr_hr=%.2f → "
                   "amat=%.1fns (hbm=%.1f gddr=%.1f dram=%.1f)",
                   hbm_hit_rate, gddr_hit_rate,
                   r.amat_ns, r.hbm_contrib_ns,
                   r.gddr_contrib_ns, r.dram_contrib_ns);

        return r;
    }

    // ── [MOD] Roofline分析 ──
    // 给定query的operational intensity, 判断是compute还是memory bound
    struct RooflineResult {
        bool   is_compute_bound;
        double achievable_gflops;
        double ridge_point;
    };

    RooflineResult roofline_analysis(uint8_t primary_tier,
                                      double oi) const {
        auto& tier = tiers_[primary_tier % 3];
        double peak = tier.peak_flops_gflops;
        double bw = tier.bandwidth_gbps;
        double ridge = tier.ridge_point();

        RooflineResult r;
        r.ridge_point = ridge;
        r.is_compute_bound = (oi >= ridge);
        // achievable = min(peak, bandwidth × OI)
        r.achievable_gflops = std::min(peak, bw * oi);

        PHILE_DBG(3, "[Roofline] tier=%s OI=%.3f ridge=%.3f → %s "
                   "achievable=%.1fGF/s",
                   tier.name, oi, ridge,
                   r.is_compute_bound ? "COMPUTE" : "MEMORY",
                   r.achievable_gflops);

        return r;
    }

    // ── 核心: 估算查询代价 ──
    // upstream: estimate_query_cost 用简单的 edges × latency
    // 这里: Roofline + AMAT + PCIe budget
    CostBreakdown estimate_query(
        QueryType qtype,
        uint64_t total_edges,
        uint64_t edges_hbm,
        uint64_t edges_gddr,
        uint64_t edges_dram,
        uint64_t cross_tier = 0
    ) {
        total_estimates_.fetch_add(1, std::memory_order_relaxed);
        auto profile = default_profile(qtype);

        CostBreakdown bd;
        bd.edges_in_hbm = edges_hbm;
        bd.edges_in_gddr = edges_gddr;
        bd.edges_in_dram = edges_dram;
        bd.cross_tier_edges = cross_tier;

        // AMAT
        double hbm_hr = (total_edges > 0)
            ? static_cast<double>(edges_hbm) / total_edges : 0.0;
        double gddr_hr = (total_edges - edges_hbm > 0)
            ? static_cast<double>(edges_gddr) / (total_edges - edges_hbm)
            : 0.0;
        bd.amat = compute_amat(hbm_hr, gddr_hr);

        // memory时间 = total_edges × AMAT × iterations × random_penalty
        double random_penalty = 1.0 + profile.random_access_ratio * 2.0;
        bd.memory_ns = total_edges * bd.amat.amat_ns
                       * profile.iteration_count * random_penalty;

        // Roofline: 判断瓶颈
        uint8_t primary_tier = (edges_hbm >= edges_gddr && edges_hbm >= edges_dram)
            ? 0 : (edges_gddr >= edges_dram ? 1 : 2);
        auto rf = roofline_analysis(primary_tier,
                                     profile.operational_intensity());
        bd.is_compute_bound = rf.is_compute_bound;
        bd.operational_intensity = profile.operational_intensity();
        bd.achievable_gflops = rf.achievable_gflops;

        // compute时间 = total_ops / achievable_throughput
        double total_ops = total_edges * profile.ops_per_edge
                           * profile.iteration_count;
        bd.compute_ns = (rf.achievable_gflops > 0)
            ? total_ops / (rf.achievable_gflops * 1e9) * 1e9
            : total_ops;

        // 跨tier边的额外迁移代价
        bd.migration_ns = cross_tier * 200.0;  // ~200ns/edge 跨tier惩罚

        // 总代价取compute和memory的max (Roofline核心思想)
        bd.total_ns = std::max(bd.compute_ns, bd.memory_ns) + bd.migration_ns;

        accumulated_cost_ns_ += bd.total_ns;

        PHILE_DBG(2, "[CostEstimator] %s E=%lu → total=%.1fμs "
                   "(compute=%.1f memory=%.1f migrate=%.1f)",
                   query_type_name(qtype), (unsigned long)total_edges,
                   bd.total_ns / 1000, bd.compute_ns / 1000,
                   bd.memory_ns / 1000, bd.migration_ns / 1000);

        return bd;
    }

    // ── [MOD] PCIe预算约束下的迁移代价 ──
    // upstream: migration_cost = bytes / bandwidth + overhead
    // 这里: 考虑PCIe总预算, 并发迁移会互相减速
    double migration_cost_ns(uint8_t from, uint8_t to,
                              uint64_t bytes,
                              uint32_t concurrent_migrations = 1) const {
        double overhead = tiers_[from % 3].migration_overhead_us * 1000;
        // 有效带宽 = PCIe总带宽 / 并发数 (简化模型)
        double effective_bw = pcie_bandwidth_gbps_ / std::max(1u, concurrent_migrations);
        double transfer_ns = static_cast<double>(bytes) /
                             (effective_bw * 1e9 / 8.0) * 1e9;

        PHILE_DBG(3, "[MigCost] %s→%s %luKB concur=%u → "
                   "eff_bw=%.1fGB/s transfer=%.1fμs overhead=%.1fμs",
                   tiers_[from % 3].name, tiers_[to % 3].name,
                   (unsigned long)(bytes / 1024), concurrent_migrations,
                   effective_bw, transfer_ns / 1000, overhead / 1000);

        return overhead + transfer_ns;
    }

    // ── 迁移收益判定 ──
    // 返回: 迁移后预期节省的ns (正=值得, 负=不值得)
    double migration_benefit(uint8_t from, uint8_t to,
                              uint64_t bytes,
                              uint64_t expected_future_accesses,
                              QueryType qtype = QueryType::MIXED) const {
        auto profile = default_profile(qtype);

        // 不迁移的代价: 每次访问走from tier
        double stay_cost = expected_future_accesses *
                           tiers_[from % 3].access_latency_ns *
                           (1.0 + profile.random_access_ratio);

        // 迁移后: 走to tier
        double moved_cost = expected_future_accesses *
                            tiers_[to % 3].access_latency_ns *
                            (1.0 + profile.random_access_ratio);

        double mig_cost = migration_cost_ns(from, to, bytes);

        return stay_cost - moved_cost - mig_cost;
    }

    // ── 全量打印 ──
    void dump_all() const {
        std::printf("════ CostEstimator (Roofline+AMAT) ════\n");
        for (int i = 0; i < 3; i++) tiers_[i].dump();
        std::printf("  PCIe budget=%.1fGB/s util=%.1f%%\n",
                    pcie_bandwidth_gbps_, pcie_utilization_ * 100);
        std::printf("  total_estimates=%lu accumulated=%.2fms\n",
                    (unsigned long)total_estimates_.load(),
                    accumulated_cost_ns_ / 1e6);

        // 打印每种query类型的default profile
        std::printf("  Query profiles:\n");
        for (auto q : {QueryType::BFS, QueryType::SSSP, QueryType::PR,
                       QueryType::WCC, QueryType::TC}) {
            auto p = default_profile(q);
            p.dump();
        }
        std::printf("════ End CostEstimator ════\n");
    }

    const TierHWSpec& tier(uint8_t idx) const { return tiers_[idx % 3]; }
};

// ═══════════════════════════════════════════════════════════════════════
// 调试宏
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_COST_DUMP(estimator) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[COST_DUMP] at %s:%d\n", __FILE__, __LINE__); \
            (estimator).dump_all(); \
        } \
    } while(0)

class CostBreakpointGuard {
    const CostEstimator& est_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    CostBreakpointGuard(const CostEstimator& e, const char* n)
        : est_(e), name_(n), start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ COST_BP ENTER: %s ━━━━\n", name_);
            est_.dump_all();
        }
    }
    ~CostBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ COST_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)us);
            est_.dump_all();
        }
    }
};

#define PHILE_COST_BREAKPOINT(estimator, tag) \
    ::philemon::cost_model::CostBreakpointGuard \
        _phile_cost_bp_##__LINE__((estimator), (tag))

}  // namespace cost_model
}  // namespace philemon

#endif  // PHILEMON_COST_ESTIMATOR_HPP
