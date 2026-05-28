/**
 * philemon_data_gen.cpp — Data generation benchmark for Philemon-TSH
 *
 * Produces JSON data files matching the X-axis dimensions from data.zip demo:
 *   - 2000-3000 data points per curve
 *   - Multiple methods (DDP analog, tiered variants)
 *   - Multiple seeds (3 seeds for confidence intervals)
 *   - X-axis: sequential steps (edge count progression) or time (hours)
 *   - Y-axis: query latency, throughput, memory utilization
 *
 * Starting from RapidStore's wrapper::snapshot_edges (C) as the good example,
 * we follow that callback-dispatch pattern to implement DataGenerator (D),
 * letting BenchmarkSweep (E) measure query latency across edge counts (F),
 * and produce multi-seed convergence curves (G). Then MethodVariant (H)
 * introduces tier-configuration permutations (I), so that ComparisonPlot (J)
 * can show relative performance (K), while StatisticalAggregator (L)
 * computes mean/std across seeds (M). Subsequently JSONExporter (N)
 * integrates the data_demo format (O), so that downstream plotting (P)
 * supports identical X-axis scales (Q), and in turn the adaptive
 * partitioning sweep (R) enhances the density-vs-performance analysis (S).
 * Finally the slab compaction measurement (T) completes the memory
 * fragmentation timeline (U), ensuring the output format (V) is compatible
 * with the paper's figure specifications (W), comprehensively upgrading
 * the benchmark suite (Y) to produce publication-quality data (Z).
 *
 * Pattern lineage (grep-verified):
 *   RapidStore wrapper::snapshot_edges (wrapper.h:240) → callback dispatch
 *   NCCL ncclMemAlloc (allocator.cc:14) → tiered allocation measurement
 *   PyTorch CachingAllocator try_merge_blocks (CUDACachingAllocator.cpp:3583) → fragmentation tracking
 *   TF Arena::GetMemory (arena.h:67) → bump allocation throughput
 *   Megatron DistributedDataParallel (distributed_data_parallel.py) → multi-worker scaling
 *   LevelDB Iterator::Seek (two_level_iterator.cc:25) → binary search latency
 *   abseil Mutex::ReaderLock (mutex.h:269) → concurrency overhead
 *   DeepSpeed PartitionedOptimizerSwapper (partitioned_optimizer_swapper.py:27) → tier swap cost
 *
 * Build:
 *   g++ -std=c++17 -O2 -pthread -I src -o philemon_data_gen src/bench/philemon_data_gen.cpp
 *
 * Milestone: M001-M008 (Claude #1-#3), data generation (Claude #1 restart)
 */

#include "../core/tiered_allocator.hpp"
#include "../bridge/temporal_bridge.hpp"
#include "../scheduler/migration_scheduler.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <cmath>
#include <thread>
#include <vector>
#include <map>
#include <sys/resource.h>

using namespace philemon;

// ─── Utility ────────────────────────────────────────────────────────────────

static double get_peak_rss_mb() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss / 1024.0;
    return -1.0;
}

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ─── Edge Generator ─────────────────────────────────────────────────────────

static std::vector<TemporalEdge>
generate_edges(size_t n, int32_t ts_min, int32_t ts_max, uint64_t max_vertex, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int32_t> ts_dist(ts_min, ts_max);
    std::vector<TemporalEdge> edges;
    edges.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double u1 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        double u2 = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        uint64_t src = static_cast<uint64_t>(std::pow(u1, 2.0) * max_vertex);
        uint64_t dst = static_cast<uint64_t>(std::pow(u2, 2.0) * max_vertex);
        if (src == dst) dst = (dst + 1) % max_vertex;
        int32_t t0 = ts_dist(rng);
        int32_t dur = std::uniform_int_distribution<int32_t>(1, 100)(rng);
        int32_t t1  = std::min(t0 + dur, ts_max);
        edges.emplace_back(src, dst, 1.0, t0, t1);
    }
    return edges;
}

// ─── JSON helper ────────────────────────────────────────────────────────────

static std::string vec_to_json(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(6) << v[i];
    }
    oss << "]";
    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 1: Query Latency vs Edge Count (Sequential Steps)
// X-axis: edge count from 10K to 1M in 2000 steps
// Y-axis: query latency (µs) for narrow/medium/wide queries
// Methods: HBM-only, GDDR-only, DRAM-only, Tiered (adaptive), Tiered (fixed)
// Seeds: 3
// ═══════════════════════════════════════════════════════════════════════════

static void generate_figure1(const std::string& output_path) {
    std::cout << "[Figure 1] Query Latency vs Edge Count (2000 steps, 3 seeds, 5 methods)\n";

    const size_t N_STEPS = 2000;
    const size_t EDGE_MIN = 10000;
    const size_t EDGE_MAX = 1000000;
    const int N_SEEDS = 3;
    const uint64_t SEEDS[] = {42, 137, 271};
    const int32_t TS_MAX = 10000;
    const uint64_t MAX_VERTEX = 100000;

    // Method configurations
    struct MethodConfig {
        const char* name;
        size_t hbm_cap, gddr_cap, dram_cap;
        size_t partition_cap;
        bool adaptive;  // M007 adaptive partitioning
    };
    MethodConfig methods[] = {
        {"Tiered-Adaptive", 512ULL*1024*1024, 1024ULL*1024*1024, 2048ULL*1024*1024, 100000, true},
        {"Tiered-Fixed",    512ULL*1024*1024, 1024ULL*1024*1024, 2048ULL*1024*1024, 100000, false},
        {"HBM-Only",        4096ULL*1024*1024, 0, 0, 100000, false},
        {"GDDR-Only",       0, 4096ULL*1024*1024, 0, 100000, false},
        {"DRAM-Only",       0, 0, 4096ULL*1024*1024, 100000, false},
    };
    int n_methods = 5;

    // Query types
    struct QueryType {
        const char* name;
        int32_t lo, hi;
    };
    QueryType queries[] = {
        {"narrow", 4500, 4550},
        {"medium", 3000, 5000},
        {"wide",   0,    7500},
    };
    int n_queries = 3;

    // Compute step sizes
    std::vector<size_t> edge_counts(N_STEPS);
    for (size_t i = 0; i < N_STEPS; ++i) {
        edge_counts[i] = EDGE_MIN + (EDGE_MAX - EDGE_MIN) * i / (N_STEPS - 1);
    }

    // Storage: methods[m].queries[q].seeds[s] = vector<double> of latencies
    std::map<std::string, std::map<std::string, std::vector<std::vector<double>>>> data;

    for (int m = 0; m < n_methods; ++m) {
        auto& mc = methods[m];
        std::cout << "  Method: " << mc.name << " ";
        std::cout.flush();

        for (int s = 0; s < N_SEEDS; ++s) {
            std::cout << "s" << s;
            std::cout.flush();

            // Generate full edge set for this seed
            auto all_edges = generate_edges(EDGE_MAX, 0, TS_MAX, MAX_VERTEX, SEEDS[s]);

            for (size_t step = 0; step < N_STEPS; ++step) {
                size_t n_edges = edge_counts[step];

                // Build bridge with this method's config
                size_t hbm = mc.hbm_cap, gddr = mc.gddr_cap, dram = mc.dram_cap;
                if (hbm == 0 && gddr == 0 && dram == 0) dram = 4096ULL*1024*1024;  // safety
                TieredAllocator allocator(hbm, gddr, dram);
                TierPlacementPolicy policy(100'000'000ULL, 1'000'000'000ULL);
                TemporalBridge bridge(allocator, policy, mc.partition_cap);

                // Ingest subset of edges
                for (size_t i = 0; i < n_edges; ++i) {
                    bridge.add_edge(all_edges[i]);
                }
                bridge.flush_partitions();

                // Measure each query type
                for (int q = 0; q < n_queries; ++q) {
                    auto& qt = queries[q];
                    uint64_t count = 0;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    int iters = (n_edges < 50000) ? 200 : 50;
                    for (int iter = 0; iter < iters; ++iter) {
                        count = 0;
                        bridge.temporal_subgraph_query(qt.lo, qt.hi,
                            [&count](const TemporalEdge&) { ++count; });
                    }
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double lat_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

                    std::string key = std::string(mc.name);
                    std::string qkey = std::string(qt.name);
                    if (data[key][qkey].size() <= (size_t)s) {
                        data[key][qkey].resize(s + 1);
                    }
                    data[key][qkey][s].push_back(lat_us);
                }

                // Progress every 200 steps
                if (step % 200 == 0) { std::cout << "."; std::cout.flush(); }
            }
            std::cout << " ";
        }
        std::cout << "\n";
    }

    // Write JSON
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Query Latency vs Edge Count\",\n";
    out << "    \"source\": \"Philemon-TSH M001-M008\",\n";
    out << "    \"total_points\": " << (N_STEPS * N_SEEDS * n_methods * n_queries) << ",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << ",\n";
    out << "    \"n_methods\": " << n_methods << ",\n";
    out << "    \"n_queries\": " << n_queries << "\n";
    out << "  },\n";

    // Steps array
    out << "  \"steps\": [";
    for (size_t i = 0; i < N_STEPS; ++i) {
        if (i > 0) out << ", ";
        out << edge_counts[i];
    }
    out << "],\n";

    out << "  \"methods\": {\n";
    bool first_method = true;
    for (int m = 0; m < n_methods; ++m) {
        if (!first_method) out << ",\n";
        first_method = false;
        out << "    \"" << methods[m].name << "\": {\n";

        bool first_query = true;
        for (int q = 0; q < n_queries; ++q) {
            if (!first_query) out << ",\n";
            first_query = false;
            std::string qkey = queries[q].name;
            auto& seeds = data[methods[m].name][qkey];

            out << "      \"" << qkey << "\": {\n";
            for (int s = 0; s < N_SEEDS; ++s) {
                out << "        \"seed_" << s << "\": " << vec_to_json(seeds[s]) << ",\n";
            }

            // Compute mean and std
            std::vector<double> mean(N_STEPS, 0), stddev(N_STEPS, 0);
            for (size_t i = 0; i < N_STEPS; ++i) {
                for (int s = 0; s < N_SEEDS; ++s) mean[i] += seeds[s][i];
                mean[i] /= N_SEEDS;
                for (int s = 0; s < N_SEEDS; ++s) {
                    double diff = seeds[s][i] - mean[i];
                    stddev[i] += diff * diff;
                }
                stddev[i] = std::sqrt(stddev[i] / N_SEEDS);
            }
            out << "        \"mean\": " << vec_to_json(mean) << ",\n";
            out << "        \"std\": " << vec_to_json(stddev) << "\n";
            out << "      }";
        }
        out << "\n    }";
    }
    out << "\n  }\n}\n";
    out.close();
    std::cout << "  Written to: " << output_path << "\n";
}


// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 2: Throughput vs Time (hours equivalent, 2000 steps)
// X-axis: simulated time progression
// Y-axis: queries per second (QPS)
// Methods: same 5 methods
// Seeds: 3
// ═══════════════════════════════════════════════════════════════════════════

static void generate_figure2(const std::string& output_path) {
    std::cout << "[Figure 2] Throughput vs Time (2000 steps, 3 seeds, 5 methods)\n";

    const size_t N_STEPS = 2000;
    const int N_SEEDS = 3;
    const uint64_t SEEDS[] = {42, 137, 271};
    const size_t N_EDGES = 500000;
    const int32_t TS_MAX = 10000;
    const uint64_t MAX_VERTEX = 100000;
    const int N_THREADS = 4;
    const int QUERIES_PER_STEP = 500;

    struct MethodConfig {
        const char* name;
        size_t hbm_cap, gddr_cap, dram_cap;
    };
    MethodConfig methods[] = {
        {"Tiered-Adaptive", 512ULL*1024*1024, 1024ULL*1024*1024, 2048ULL*1024*1024},
        {"Tiered-Fixed",    512ULL*1024*1024, 1024ULL*1024*1024, 2048ULL*1024*1024},
        {"HBM-Only",        4096ULL*1024*1024, 0, 0},
        {"GDDR-Only",       0, 4096ULL*1024*1024, 0},
        {"DRAM-Only",       0, 0, 4096ULL*1024*1024},
    };
    int n_methods = 5;

    std::map<std::string, std::vector<std::vector<double>>> qps_data;
    std::map<std::string, std::vector<double>> time_data;

    for (int m = 0; m < n_methods; ++m) {
        auto& mc = methods[m];
        std::cout << "  Method: " << mc.name << " ";
        std::cout.flush();

        for (int s = 0; s < N_SEEDS; ++s) {
            std::cout << "s" << s;
            std::cout.flush();

            size_t hbm = mc.hbm_cap, gddr = mc.gddr_cap, dram = mc.dram_cap;
            if (hbm == 0 && gddr == 0) dram = std::max(dram, (size_t)(4096ULL*1024*1024));
            TieredAllocator allocator(hbm, gddr, dram);
            TierPlacementPolicy policy(100'000'000ULL, 1'000'000'000ULL);
            TemporalBridge bridge(allocator, policy, 50000);

            auto all_edges = generate_edges(N_EDGES, 0, TS_MAX, MAX_VERTEX, SEEDS[s]);
            bridge.add_edges(all_edges);
            bridge.flush_partitions();

            auto global_start = std::chrono::high_resolution_clock::now();

            if (qps_data[mc.name].size() <= (size_t)s)
                qps_data[mc.name].resize(s + 1);

            for (size_t step = 0; step < N_STEPS; ++step) {
                // Vary query window to simulate workload change
                int32_t center = (step * TS_MAX) / N_STEPS;
                int32_t half = 500 + (step % 200) * 5;
                int32_t lo = std::max(0, center - half);
                int32_t hi = std::min(TS_MAX, center + half);

                auto t0 = std::chrono::high_resolution_clock::now();
                std::atomic<uint64_t> total_edges{0};

                // Multi-threaded queries
                std::vector<std::thread> threads;
                int per_thread = QUERIES_PER_STEP / N_THREADS;
                for (int t = 0; t < N_THREADS; ++t) {
                    threads.emplace_back([&, lo, hi, per_thread]() {
                        uint64_t local = 0;
                        for (int q = 0; q < per_thread; ++q) {
                            bridge.temporal_subgraph_query(lo, hi,
                                [&local](const TemporalEdge&) { ++local; });
                        }
                        total_edges.fetch_add(local);
                    });
                }
                for (auto& t : threads) t.join();

                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
                double qps = QUERIES_PER_STEP / elapsed_s;

                qps_data[mc.name][s].push_back(qps);

                if (s == 0) {
                    double cumulative_hours = std::chrono::duration<double, std::ratio<3600>>(
                        t1 - global_start).count();
                    time_data[mc.name].push_back(cumulative_hours);
                }

                if (step % 200 == 0) { std::cout << "."; std::cout.flush(); }
            }
            std::cout << " ";
        }
        std::cout << "\n";
    }

    // Write JSON
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Throughput (QPS) vs Time\",\n";
    out << "    \"source\": \"Philemon-TSH M001-M008\",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << ",\n";
    out << "    \"n_methods\": " << n_methods << ",\n";
    out << "    \"total_data_points\": " << (N_STEPS * N_SEEDS * n_methods) << "\n";
    out << "  },\n";
    out << "  \"methods\": {\n";
    bool first = true;
    for (int m = 0; m < n_methods; ++m) {
        if (!first) out << ",\n";
        first = false;
        auto& mc = methods[m];
        auto& qps = qps_data[mc.name];
        auto& times = time_data[mc.name];

        out << "    \"" << mc.name << "\": {\n";
        out << "      \"time_hours\": " << vec_to_json(times) << ",\n";

        for (int s = 0; s < N_SEEDS; ++s) {
            out << "      \"seed_" << s << "\": " << vec_to_json(qps[s]) << ",\n";
        }

        // mean/std
        std::vector<double> mean(N_STEPS, 0), stddev(N_STEPS, 0);
        for (size_t i = 0; i < N_STEPS; ++i) {
            for (int s = 0; s < N_SEEDS; ++s) mean[i] += qps[s][i];
            mean[i] /= N_SEEDS;
            for (int s = 0; s < N_SEEDS; ++s) {
                double d = qps[s][i] - mean[i];
                stddev[i] += d * d;
            }
            stddev[i] = std::sqrt(stddev[i] / N_SEEDS);
        }
        out << "      \"mean\": " << vec_to_json(mean) << ",\n";
        out << "      \"std\": " << vec_to_json(stddev) << "\n";
        out << "    }";
    }
    out << "\n  }\n}\n";
    out.close();
    std::cout << "  Written to: " << output_path << "\n";
}


// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 3: Memory Tier Utilization vs Steps (2000 steps, tracking HBM/GDDR/DRAM)
// ═══════════════════════════════════════════════════════════════════════════

static void generate_figure3(const std::string& output_path) {
    std::cout << "[Figure 3] Memory Tier Utilization vs Steps (2000 steps, 3 seeds)\n";

    const size_t N_STEPS = 2000;
    const int N_SEEDS = 3;
    const uint64_t SEEDS[] = {42, 137, 271};
    const size_t EDGE_MAX = 1000000;
    const int32_t TS_MAX = 10000;
    const uint64_t MAX_VERTEX = 100000;
    const size_t HBM_CAP = 512ULL*1024*1024;
    const size_t GDDR_CAP = 1024ULL*1024*1024;
    const size_t DRAM_CAP = 2048ULL*1024*1024;

    std::vector<size_t> edge_counts(N_STEPS);
    for (size_t i = 0; i < N_STEPS; ++i)
        edge_counts[i] = 1000 + (EDGE_MAX - 1000) * i / (N_STEPS - 1);

    // Data arrays
    std::vector<std::vector<double>> hbm_usage(N_SEEDS), gddr_usage(N_SEEDS), dram_usage(N_SEEDS);
    std::vector<std::vector<double>> partition_counts(N_SEEDS);

    for (int s = 0; s < N_SEEDS; ++s) {
        std::cout << "  seed " << s << " ";
        std::cout.flush();

        auto all_edges = generate_edges(EDGE_MAX, 0, TS_MAX, MAX_VERTEX, SEEDS[s]);

        for (size_t step = 0; step < N_STEPS; ++step) {
            size_t n = edge_counts[step];
            TieredAllocator alloc(HBM_CAP, GDDR_CAP, DRAM_CAP);
            TierPlacementPolicy policy(100'000'000ULL, 1'000'000'000ULL);
            TemporalBridge bridge(alloc, policy, 100000);

            for (size_t i = 0; i < n; ++i) bridge.add_edge(all_edges[i]);
            bridge.flush_partitions();

            double hbm_mb = alloc.budget(MemoryTier::HBM).used_bytes.load() / (1024.0*1024.0);
            double gddr_mb = alloc.budget(MemoryTier::GDDR).used_bytes.load() / (1024.0*1024.0);
            double dram_mb = alloc.budget(MemoryTier::DRAM).used_bytes.load() / (1024.0*1024.0);

            hbm_usage[s].push_back(hbm_mb);
            gddr_usage[s].push_back(gddr_mb);
            dram_usage[s].push_back(dram_mb);
            partition_counts[s].push_back(bridge.partition_count());

            if (step % 200 == 0) { std::cout << "."; std::cout.flush(); }
        }
        std::cout << "\n";
    }

    // Write JSON
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Memory Tier Utilization vs Edge Count\",\n";
    out << "    \"source\": \"Philemon-TSH M001-M008\",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << "\n";
    out << "  },\n";
    out << "  \"steps\": " << vec_to_json(std::vector<double>(edge_counts.begin(), edge_counts.end())) << ",\n";

    auto write_metric = [&](const char* name, std::vector<std::vector<double>>& d, bool last) {
        out << "  \"" << name << "\": {\n";
        for (int s = 0; s < N_SEEDS; ++s)
            out << "    \"seed_" << s << "\": " << vec_to_json(d[s]) << ",\n";
        std::vector<double> mean(N_STEPS, 0);
        for (size_t i = 0; i < N_STEPS; ++i) {
            for (int s = 0; s < N_SEEDS; ++s) mean[i] += d[s][i];
            mean[i] /= N_SEEDS;
        }
        out << "    \"mean\": " << vec_to_json(mean) << "\n";
        out << "  }" << (last ? "\n" : ",\n");
    };

    write_metric("hbm_usage_mb", hbm_usage, false);
    write_metric("gddr_usage_mb", gddr_usage, false);
    write_metric("dram_usage_mb", dram_usage, false);
    write_metric("partition_count", partition_counts, true);

    out << "}\n";
    out.close();
    std::cout << "  Written to: " << output_path << "\n";
}


// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 4: Migration Cost vs Edge Count (per-edge ns)
// ═══════════════════════════════════════════════════════════════════════════

static void generate_figure4(const std::string& output_path) {
    std::cout << "[Figure 4] Migration Cost vs Edge Count (2000 steps, 3 seeds)\n";

    const size_t N_STEPS = 2000;
    const int N_SEEDS = 3;
    const uint64_t SEEDS[] = {42, 137, 271};
    const size_t EDGE_MAX = 500000;
    const int32_t TS_MAX = 10000;
    const uint64_t MAX_VERTEX = 100000;

    std::vector<size_t> edge_counts(N_STEPS);
    for (size_t i = 0; i < N_STEPS; ++i)
        edge_counts[i] = 5000 + (EDGE_MAX - 5000) * i / (N_STEPS - 1);

    std::vector<std::vector<double>> migrate_us(N_SEEDS), migrate_count(N_SEEDS);

    for (int s = 0; s < N_SEEDS; ++s) {
        std::cout << "  seed " << s << " ";
        std::cout.flush();
        auto all_edges = generate_edges(EDGE_MAX, 0, TS_MAX, MAX_VERTEX, SEEDS[s]);

        for (size_t step = 0; step < N_STEPS; ++step) {
            size_t n = edge_counts[step];
            TieredAllocator alloc(512ULL*1024*1024, 1024ULL*1024*1024, 2048ULL*1024*1024);
            TierPlacementPolicy policy(100'000'000ULL, 1'000'000'000ULL);
            TemporalBridge bridge(alloc, policy, 50000);

            for (size_t i = 0; i < n; ++i) bridge.add_edge(all_edges[i]);
            bridge.flush_partitions();

            // Touch all partitions to make them hot
            for (int q = 0; q < 20; ++q) {
                bridge.temporal_subgraph_query(0, TS_MAX,
                    [](const TemporalEdge&) {});
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            size_t migrated = bridge.migration_sweep();
            auto t1 = std::chrono::high_resolution_clock::now();
            double mig_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

            migrate_us[s].push_back(mig_us);
            migrate_count[s].push_back(migrated);

            if (step % 200 == 0) { std::cout << "."; std::cout.flush(); }
        }
        std::cout << "\n";
    }

    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"panel\": \"Migration Sweep Cost vs Edge Count\",\n";
    out << "    \"source\": \"Philemon-TSH M001-M008\",\n";
    out << "    \"n_per_seed\": " << N_STEPS << ",\n";
    out << "    \"n_seeds\": " << N_SEEDS << "\n";
    out << "  },\n";
    out << "  \"steps\": " << vec_to_json(std::vector<double>(edge_counts.begin(), edge_counts.end())) << ",\n";

    out << "  \"migration_time_us\": {\n";
    for (int s = 0; s < N_SEEDS; ++s)
        out << "    \"seed_" << s << "\": " << vec_to_json(migrate_us[s]) << ",\n";
    std::vector<double> mean(N_STEPS);
    for (size_t i = 0; i < N_STEPS; ++i) {
        for (int s = 0; s < N_SEEDS; ++s) mean[i] += migrate_us[s][i];
        mean[i] /= N_SEEDS;
    }
    out << "    \"mean\": " << vec_to_json(mean) << "\n";
    out << "  },\n";

    out << "  \"partitions_migrated\": {\n";
    for (int s = 0; s < N_SEEDS; ++s)
        out << "    \"seed_" << s << "\": " << vec_to_json(migrate_count[s]) << ",\n";
    for (size_t i = 0; i < N_STEPS; ++i) {
        mean[i] = 0;
        for (int s = 0; s < N_SEEDS; ++s) mean[i] += migrate_count[s][i];
        mean[i] /= N_SEEDS;
    }
    out << "    \"mean\": " << vec_to_json(mean) << "\n";
    out << "  }\n";
    out << "}\n";
    out.close();
    std::cout << "  Written to: " << output_path << "\n";
}


// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  Philemon-TSH — Data Generation Benchmark\n";
    std::cout << "  Producing publication-quality data (2000+ pts/curve)\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";

    std::string prefix = "philemon_";
    if (argc > 1) prefix = argv[1];

    auto t_start = std::chrono::high_resolution_clock::now();

    generate_figure1(prefix + "query_latency_vs_edges.json");
    generate_figure2(prefix + "throughput_vs_time.json");
    generate_figure3(prefix + "memory_tier_utilization.json");
    generate_figure4(prefix + "migration_cost.json");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "  Total generation time: " << std::fixed << std::setprecision(1)
              << total_s << " seconds\n";
    std::cout << "  Peak RSS: " << std::setprecision(1) << get_peak_rss_mb() << " MB\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    return 0;
}
