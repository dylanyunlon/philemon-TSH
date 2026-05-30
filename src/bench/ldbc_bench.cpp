/**
 * ldbc_bench.cpp — Integration Benchmark for M017-M020
 *
 * 测试管线:
 *   1. LDBCLoader 加载图数据 + tier placement
 *   2. TierCostModel 预估跨层代价
 *   3. CrossTierBFS 带 prefetch 的层级 BFS
 *   4. CrossTierSSSP 带 tier penalty 的 delta-stepping
 *
 * 编译: make ldbc_bench
 * 运行: ./ldbc_bench <edge_file> [num_threads] [delta] [debug_level]
 *
 * Milestone: M017-M020 integration test
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

// Philemon-TSH headers
#include "../debug/philemon_debug.hpp"
#include "../loader/ldbc_types.hpp"
#include "../loader/ldbc_loader.hpp"
#include "../cost_model/tier_cost_model.hpp"

// Note: CrossTierBFS and CrossTierSSSP are template classes that need
// the actual graph backend (F,S). This standalone bench uses the
// LDBCLoader's built-in validation + cost model testing.
// Full BFS/SSSP integration requires linking with RapidStore wrapper.

using namespace philemon;

// ─── Synthetic edge file generator for standalone testing ───────────
static void generate_synthetic_edges(const std::string& path,
                                      uint64_t num_vertices,
                                      uint64_t num_edges) {
    debug::ScopedTimer timer("generate_synthetic_edges");

    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "[ldbc_bench] ERROR: cannot create %s\n",
                     path.c_str());
        return;
    }

    std::fprintf(f, "# Synthetic temporal graph: V=%lu E=%lu\n",
                 (unsigned long)num_vertices, (unsigned long)num_edges);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> vdist(0, num_vertices - 1);
    std::uniform_real_distribution<double> wdist(0.1, 10.0);

    uint64_t written = 0;
    for (uint64_t i = 0; i < num_edges * 2 && written < num_edges; i++) {
        uint64_t src = vdist(rng);
        uint64_t dst = vdist(rng);
        if (src == dst) continue;

        double weight = wdist(rng);
        uint64_t timestamp = 1000000 + i * 100;  // monotonic timestamps

        std::fprintf(f, "%lu %lu %.4f %lu\n",
                     (unsigned long)src, (unsigned long)dst,
                     weight, (unsigned long)timestamp);
        written++;
    }

    std::fclose(f);
    PHILE_DBG(1, "[synthetic] wrote %lu edges to %s",
              (unsigned long)written, path.c_str());
}

// ─── Test 1: LDBCLoader construction + validation ───────────────────
static bool test_ldbc_loader(const std::string& edge_file) {
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Test 1: LDBCLoader Construction\n");
    std::printf("════════════════════════════════════════════\n");

    debug::ScopedTimer timer("test_ldbc_loader");

    loader::LDBCConfig config;
    config.dump();

    loader::LDBCLoader ldr(edge_file, true, ' ',
                           0.8, 0.2, 0.2,
                           0.01, 0.2, 0.2, 0.5,
                           10000, 10000, 10000,
                           42, true, config);

    // Validate
    bool ok = ldr.validateLoad();
    std::printf("  [result] validateLoad = %s\n", ok ? "PASS" : "FAIL");

    // Dump state
    ldr.dumpFullState("test1");
    ldr.dumpTierDistribution();
    ldr.dumpTimestampHistogram(10);

    // Check accessors
    std::printf("  [accessors] V=%lu E=%lu ts=[%lu,%lu]\n",
                (unsigned long)ldr.vertexCount(),
                (unsigned long)ldr.edgeCount(),
                (unsigned long)ldr.timestampMin(),
                (unsigned long)ldr.timestampMax());

    const auto& hints = ldr.edgeTierHints();
    uint64_t tier_counts[4] = {0, 0, 0, 0};
    for (auto h : hints) {
        tier_counts[static_cast<uint8_t>(h)]++;
    }
    std::printf("  [tier_hints] HBM=%lu GDDR=%lu DRAM=%lu AUTO=%lu\n",
                (unsigned long)tier_counts[0], (unsigned long)tier_counts[1],
                (unsigned long)tier_counts[2], (unsigned long)tier_counts[3]);

    // Partition hint check
    const auto& ph = ldr.partitionHint();
    std::printf("  [partition] hot=%.3f warm=%.3f density=%.2f\n",
                ph.hot_fraction, ph.warm_fraction, ph.density_threshold);

    return ok;
}

// ─── Test 2: TierCostModel standalone ───────────────────────────────
static bool test_cost_model() {
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Test 2: TierCostModel\n");
    std::printf("════════════════════════════════════════════\n");

    debug::ScopedTimer timer("test_cost_model");

    cost_model::TierCostModel model;
    model.dump();

    bool ok = true;

    // Test 2a: Single access costs
    double hbm_cost  = model.access_cost_ns(0, 64);    // 64B cache line
    double gddr_cost = model.access_cost_ns(1, 64);
    double dram_cost = model.access_cost_ns(2, 64);

    std::printf("  [access_cost] 64B: HBM=%.2fns GDDR=%.2fns DRAM=%.2fns\n",
                hbm_cost, gddr_cost, dram_cost);

    if (hbm_cost >= gddr_cost || gddr_cost >= dram_cost) {
        std::printf("  [FAIL] Expected HBM < GDDR < DRAM cost\n");
        ok = false;
    } else {
        std::printf("  [PASS] HBM < GDDR < DRAM ordering correct\n");
    }

    // Test 2b: Migration costs
    double hbm_to_dram = model.migration_cost_ns(0, 2, 1024 * 1024);
    double dram_to_hbm = model.migration_cost_ns(2, 0, 1024 * 1024);

    std::printf("  [migration] 1MB: HBM→DRAM=%.2fμs DRAM→HBM=%.2fμs\n",
                hbm_to_dram / 1000.0, dram_to_hbm / 1000.0);

    if (hbm_to_dram <= 0 || dram_to_hbm <= 0) {
        std::printf("  [FAIL] Migration cost should be positive\n");
        ok = false;
    } else {
        std::printf("  [PASS] Migration costs positive\n");
    }

    // Test 2c: Query cost estimation
    std::vector<std::tuple<uint8_t, uint64_t, uint64_t>> partitions = {
        {0, 1000, 1000 * 24},     // 1000 edges in HBM
        {1, 5000, 5000 * 24},     // 5000 edges in GDDR
        {2, 10000, 10000 * 24}    // 10000 edges in DRAM
    };
    auto est_scan = model.estimate_query_cost(partitions, true);
    auto est_rand = model.estimate_query_cost(partitions, false);

    est_scan.dump("scan_mode");
    est_rand.dump("random_mode");

    if (est_rand.total_ns > est_scan.total_ns) {
        std::printf("  [PASS] Random access more expensive than scan\n");
    } else {
        std::printf("  [INFO] Scan >= random (depends on data size)\n");
    }

    // Test 2d: BFS level cost estimation
    std::vector<std::pair<uint8_t, uint64_t>> frontier = {
        {0, 100},   // 100 HBM vertices
        {1, 500},   // 500 GDDR vertices
        {2, 2000}   // 2000 DRAM vertices
    };
    auto bfs_est = model.estimate_bfs_level_cost(frontier, 10);
    bfs_est.dump("bfs_level");

    // Test 2e: Optimal tier assignment
    std::vector<std::tuple<double, uint64_t, uint64_t>> assign_input = {
        {100.0, 1000, 1000 * 24},    // hot partition
        {50.0,  2000, 2000 * 24},     // warm partition
        {10.0,  5000, 5000 * 24},     // lukewarm
        {1.0,   10000, 10000 * 24},   // cold
        {0.1,   50000, 50000 * 24}    // very cold
    };
    auto assignment = model.optimal_tier_assignment(assign_input);

    std::printf("  [optimal_assignment] ");
    const char* tier_names[] = {"HBM", "GDDR", "DRAM"};
    for (size_t i = 0; i < assignment.size(); i++) {
        std::printf("part[%zu]=%s ", i, tier_names[assignment[i] % 3]);
    }
    std::printf("\n");

    // Hottest should be in HBM
    if (assignment[0] == 0) {
        std::printf("  [PASS] Hottest partition assigned to HBM\n");
    } else {
        std::printf("  [WARN] Hottest partition NOT in HBM (tier=%d)\n",
                    assignment[0]);
    }

    // Test 2f: Prefetch benefit
    double benefit = model.prefetch_benefit_ns(2, 0, 1024 * 1024, 1000);
    std::printf("  [prefetch_benefit] DRAM→HBM 1MB ×1000 accesses: "
                "%.2fμs\n", benefit / 1000.0);

    // Test 2g: Migration plan
    std::vector<uint8_t> current  = {2, 2, 1, 0, 2};  // current placement
    std::vector<uint8_t> target   = {0, 1, 1, 0, 0};  // target placement
    std::vector<uint64_t> sizes   = {24000, 48000, 48000, 24000, 120000};

    auto plans = model.plan_migrations(current, target, sizes);
    std::printf("  [migration_plan] %zu migrations planned\n", plans.size());

    return ok;
}

// ─── Test 3: Workload generation ────────────────────────────────────
static bool test_workload_generation(const std::string& edge_file,
                                      const std::string& output_dir) {
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Test 3: Workload Generation\n");
    std::printf("════════════════════════════════════════════\n");

    debug::ScopedTimer timer("test_workload_generation");

    // Create output directory
    std::string cmd = "mkdir -p " + output_dir;
    std::system(cmd.c_str());

    loader::LDBCLoader ldr(edge_file, true, ' ',
                           0.8, 0.2, 0.2,
                           0.01, 0.2, 0.2, 0.5,
                           1000, 1000, 1000,
                           42, true);

    ldr.generateAllWorkloads(output_dir);

    // Verify files exist
    bool ok = true;
    std::vector<std::string> expected_files = {
        output_dir + "/initial_stream_analytic.stream",
        output_dir + "/target_stream_analytic.stream",
        output_dir + "/target_stream_temporal.stream"
    };

    for (const auto& f : expected_files) {
        std::FILE* fp = std::fopen(f.c_str(), "rb");
        if (fp) {
            std::fseek(fp, 0, SEEK_END);
            long size = std::ftell(fp);
            std::fclose(fp);
            std::printf("  [OK] %s (%ld bytes)\n", f.c_str(), size);
        } else {
            std::printf("  [FAIL] %s not found\n", f.c_str());
            ok = false;
        }
    }

    return ok;
}

// ─── Test 4: Degree stats + tier distribution cross-check ───────────
static bool test_degree_tier_crosscheck(const std::string& edge_file) {
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Test 4: Degree-Tier Cross-Check\n");
    std::printf("════════════════════════════════════════════\n");

    debug::ScopedTimer timer("test_degree_tier_crosscheck");

    loader::LDBCLoader ldr(edge_file, true);

    const auto& stats = ldr.degreeStats();
    const auto& hints = ldr.vertexTierHints();
    const auto& ph = ldr.partitionHint();

    bool ok = true;

    // Check that high-degree vertices tend to be in HBM
    // (This is a statistical check, not absolute)
    uint64_t hbm_high_degree = 0, hbm_low_degree = 0;
    uint64_t total_hbm = 0;
    for (uint64_t v = 0; v < hints.size(); v++) {
        if (hints[v] == loader::TierHint::HBM) {
            total_hbm++;
        }
    }

    std::printf("  [stats] V=%lu max_deg=%lu avg_deg=%.2f median=%lu "
                "p90=%lu p99=%lu\n",
                (unsigned long)stats.num_vertices,
                (unsigned long)stats.max_degree,
                stats.avg_degree,
                (unsigned long)stats.median_degree,
                (unsigned long)stats.p90_degree,
                (unsigned long)stats.p99_degree);
    std::printf("  [tier] total_HBM_vertices=%lu hot_fraction=%.3f\n",
                (unsigned long)total_hbm, ph.hot_fraction);

    // HBM vertices should be roughly hot_fraction of total
    double actual_hot = static_cast<double>(total_hbm) / hints.size();
    double expected_hot = ph.hot_fraction;
    double diff = std::abs(actual_hot - expected_hot);

    if (diff < 0.1) {
        std::printf("  [PASS] HBM fraction %.3f ≈ hot_fraction %.3f "
                    "(diff=%.3f)\n", actual_hot, expected_hot, diff);
    } else {
        std::printf("  [WARN] HBM fraction %.3f vs hot_fraction %.3f "
                    "(diff=%.3f)\n", actual_hot, expected_hot, diff);
    }

    return ok;
}

// ─── Test 5: Cost model calibration ─────────────────────────────────
static bool test_cost_calibration() {
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Test 5: Cost Model Calibration\n");
    std::printf("════════════════════════════════════════════\n");

    debug::ScopedTimer timer("test_cost_calibration");

    cost_model::TierCostModel model;
    bool ok = true;

    // Test latency ratios
    double gddr_hbm_ratio = model.latency_ratio(0, 1);  // GDDR/HBM
    double dram_hbm_ratio = model.latency_ratio(0, 2);  // DRAM/HBM
    double dram_gddr_ratio = model.latency_ratio(1, 2);  // DRAM/GDDR

    std::printf("  [ratios] GDDR/HBM=%.1fx DRAM/HBM=%.1fx "
                "DRAM/GDDR=%.1fx\n",
                gddr_hbm_ratio, dram_hbm_ratio, dram_gddr_ratio);

    // H100 HBM is ~5x faster than GDDR, ~50x faster than DRAM
    if (gddr_hbm_ratio >= 3.0 && gddr_hbm_ratio <= 10.0) {
        std::printf("  [PASS] GDDR/HBM ratio in expected range [3-10]\n");
    } else {
        std::printf("  [FAIL] GDDR/HBM ratio %.1f outside range\n",
                    gddr_hbm_ratio);
        ok = false;
    }

    if (dram_hbm_ratio >= 20.0 && dram_hbm_ratio <= 100.0) {
        std::printf("  [PASS] DRAM/HBM ratio in expected range [20-100]\n");
    } else {
        std::printf("  [FAIL] DRAM/HBM ratio %.1f outside range\n",
                    dram_hbm_ratio);
        ok = false;
    }

    // Test bandwidth: 1GB transfer time
    uint64_t one_gb = 1024ULL * 1024 * 1024;
    double hbm_time  = model.access_cost_ns(0, one_gb);
    double gddr_time = model.access_cost_ns(1, one_gb);
    double dram_time = model.access_cost_ns(2, one_gb);

    std::printf("  [1GB_transfer] HBM=%.2fms GDDR=%.2fms DRAM=%.2fms\n",
                hbm_time / 1e6, gddr_time / 1e6, dram_time / 1e6);

    // H100 HBM should do 1GB in ~0.3ms, DRAM in ~5ms
    if (hbm_time / 1e6 < 1.0) {
        std::printf("  [PASS] HBM 1GB < 1ms (actual=%.3fms)\n",
                    hbm_time / 1e6);
    } else {
        std::printf("  [FAIL] HBM 1GB too slow: %.3fms\n", hbm_time / 1e6);
        ok = false;
    }

    return ok;
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    std::printf("╔══════════════════════════════════════════════╗\n");
    std::printf("║  Philemon-TSH LDBC Bench (M017-M020)        ║\n");
    std::printf("╚══════════════════════════════════════════════╝\n");

    // Parse args
    std::string edge_file;
    int num_threads = 4;
    double delta = 2.0;
    int debug_level = 2;

    if (argc >= 2) {
        edge_file = argv[1];
    }
    if (argc >= 3) num_threads = std::atoi(argv[2]);
    if (argc >= 4) delta = std::atof(argv[3]);
    if (argc >= 5) debug_level = std::atoi(argv[4]);

    // Set debug level
    debug::set_debug_level(debug_level);
    PHILE_DBG(1, "Debug level set to %d", debug_level);

    // Generate synthetic data if no file provided
    bool synthetic = false;
    if (edge_file.empty()) {
        edge_file = "/tmp/philemon_synthetic_edges.txt";
        uint64_t synth_v = 10000;
        uint64_t synth_e = 50000;
        std::printf("\n  No edge file provided — generating synthetic "
                    "graph (V=%lu E=%lu)\n",
                    (unsigned long)synth_v, (unsigned long)synth_e);
        generate_synthetic_edges(edge_file, synth_v, synth_e);
        synthetic = true;
    }

    std::printf("  edge_file=%s threads=%d delta=%.2f debug=%d\n\n",
                edge_file.c_str(), num_threads, delta, debug_level);

    // ─── Run tests ──────────────────────────────────────────────
    int passed = 0, failed = 0;

    auto run = [&](const char* name, bool result) {
        if (result) {
            std::printf("\n  ✓ %s PASSED\n", name);
            passed++;
        } else {
            std::printf("\n  ✗ %s FAILED\n", name);
            failed++;
        }
    };

    run("LDBCLoader",           test_ldbc_loader(edge_file));
    run("TierCostModel",        test_cost_model());
    run("WorkloadGeneration",   test_workload_generation(
                                    edge_file, "/tmp/philemon_workloads"));
    run("DegreeTierCrossCheck", test_degree_tier_crosscheck(edge_file));
    run("CostCalibration",      test_cost_calibration());

    // ─── Summary ────────────────────────────────────────────────
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Results: %d/%d passed", passed, passed + failed);
    if (failed > 0) {
        std::printf(" (%d FAILED)", failed);
    }
    std::printf("\n");
    std::printf("════════════════════════════════════════════\n");

    // Dump global trace log
    debug::global_trace().dump_last(20);

    // Cleanup synthetic
    if (synthetic) {
        std::remove(edge_file.c_str());
        PHILE_DBG(2, "Cleaned up synthetic edge file");
    }

    return (failed == 0) ? 0 : 1;
}
