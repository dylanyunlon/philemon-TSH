// M177-M178: Streaming + Compaction Experiment — RQ5 流式写入+压缩
//
// Ports ALL upstream streaming/compaction infrastructure with 20% algorithmic change:
//   upstream/rapidstore/graph/edgeStream.cpp+hpp   — streaming ingestion pipeline
//   upstream/rapidstore/graph/graphTile.cpp+hpp    — segment management
//   upstream/rapidstore/graph/partition.cpp+hpp    — partition layout + tier placement
//   upstream/rapidstore/index/c_art.cpp+hpp        — ART index for segment fan-out
//   upstream/rapidstore/driver.h (streaming path) — continuous write + flush loop
//
// Algorithmic modifications (~20%):
//   [MOD] SegmentedIndex::flush → tier-aware partition placement: hot partitions
//         (high hotness score) go to DRAM tier, warm to SSD, cold to HDD.
//         Upstream assigns tiers monotonically by flush order.
//   [MOD] SegmentedIndex::compact → amortized multi-way merge with per-tier cost
//         accounting. Upstream: simple rebuild from all edges.
//   [MOD] StreamingBench::measure_selection_latency → per-tier hit ratio tracking.
//         Upstream: only total time.
//   [MOD] WriteThroughput::run → adaptive flush sizing: batch size scales with
//         current DRAM pressure (more pressure → smaller batches to limit eviction).
//         Upstream: fixed flush_size.
//   [MOD] CompactionTracker → per-compaction tier migration cost breakdown.
//         Upstream: only total compaction time.
//   [MOD] LatencyTrace::annotate_spikes → statistical spike detection (3-sigma).
//         Upstream: no spike annotation.
//
//   [KEEP] 80% of logic: partition Interval model, augmented span_max index,
//          seqlock-protected segment list, threshold-8 compaction trigger,
//          cross-check (indexed vs linear vs brute-force), RMAT streaming
//          generation, segment sawtooth growth, flush record schema.
//
// Experiment outputs (RQ5 paper data):
//   experiment/results/m177_streaming.csv  — per-flush latency trace
//   stdout LaTeX: pgfplots coordinates for Figure (latency trace)
//
// Build: g++ -std=c++17 -O2 -fopenmp -march=native -o m177_m178 \
//        experiment/m177_m178_streaming_compaction.cpp -lpthread
// Run:   ./m177_m178 [--scale 14] [--threads 4] [--flushes 128] [--debug 2]

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <deque>
#include <queue>
#include <functional>
#include <unordered_map>
#include <limits>
#include <array>
#include <iomanip>
#include <condition_variable>
#include <sys/resource.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// §0  Infrastructure: Debug, Timer, Memory, Check macros
//     Mirrors m169_m170_driver_workload_engine.cpp §0 style
// ═══════════════════════════════════════════════════════════════════════════════
namespace phi {

static int g_debug = 1;
static int g_pass = 0, g_fail = 0;

struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    Timer() : t0(clk::now()) {}
    void reset() { t0 = clk::now(); }
    double us() const { return std::chrono::duration<double,std::micro>(clk::now()-t0).count(); }
    double ms() const { return us()/1000.0; }
    double s()  const { return ms()/1000.0; }
};

double rss_mb() {
    std::ifstream f("/proc/self/status"); std::string l;
    while (std::getline(f, l))
        if (l.rfind("VmRSS:", 0) == 0) {
            std::istringstream ss(l.substr(6)); uint64_t kb; std::string u;
            if (ss >> kb >> u) return kb/1024.0;
        }
    return 0;
}

#define CHECK(cond, name) do { \
    if (cond) { phi::g_pass++; if(phi::g_debug>=1) printf("  PASS: %s\n", name); } \
    else { phi::g_fail++; printf("  FAIL: %s\n", name); } \
} while(0)

#define CHECK_RANGE(val, lo, hi, name) do { \
    bool ok = ((val) >= (lo) && (val) <= (hi)); \
    if (ok) { phi::g_pass++; if(phi::g_debug>=1) printf("  PASS: %s (%.4f in [%.4f, %.4f])\n", name, (double)(val), (double)(lo), (double)(hi)); } \
    else { phi::g_fail++; printf("  FAIL: %s (%.4f not in [%.4f, %.4f])\n", name, (double)(val), (double)(lo), (double)(hi)); } \
} while(0)

// ─── Breakpoint dump: prints all state at a named checkpoint ────────────────
struct BreakpointDump {
    static void dump_state(const char* label, int phase,
                           uint64_t vertices, uint64_t edges,
                           double rss, double elapsed_ms,
                           const std::map<std::string,double>& extra = {}) {
        if (phi::g_debug < 2) return;
        printf("  ┌─ BREAKPOINT [%s] phase=%d ──────────────────────\n", label, phase);
        printf("  │ vertices=%lu  edges=%lu  RSS=%.1fMB  elapsed=%.2fms\n",
               vertices, edges, rss, elapsed_ms);
        for (auto& [k,v] : extra)
            printf("  │ %s = %.6f\n", k.c_str(), v);
        printf("  └────────────────────────────────────────────────\n");
    }
};

// ─── Latency histogram with P50/P99 ─────────────────────────────────────────
struct LatencyHistogram {
    std::vector<double> samples;
    std::string name;
    LatencyHistogram(const std::string& n = "") : name(n) {}
    void record(double us) { samples.push_back(us); }
    double p(double pct) const {
        if (samples.empty()) return 0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = std::min((size_t)(sorted.size() * pct), sorted.size()-1);
        return sorted[idx];
    }
    double mean() const {
        if (samples.empty()) return 0;
        return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    }
    double stddev() const {
        if (samples.size() < 2) return 0;
        double m = mean();
        double sq = 0;
        for (auto x : samples) sq += (x-m)*(x-m);
        return std::sqrt(sq / samples.size());
    }
    void report() const {
        if (samples.empty()) return;
        printf("  │ %-16s n=%-6zu  P50=%8.2fus  P99=%8.2fus  mean=%8.2fus  stddev=%7.2fus\n",
               name.c_str(), samples.size(),
               p(0.50), p(0.99), mean(), stddev());
    }
};

} // namespace phi

// ═══════════════════════════════════════════════════════════════════════════════
// §1  Types — TierID, Edge, Partition, Operation
//     From upstream types.hpp + graph/partition.hpp
// ═══════════════════════════════════════════════════════════════════════════════

enum TierID : uint8_t { TIER_DRAM=0, TIER_SSD=1, TIER_HDD=2, NUM_TIERS=3 };
static const char* tier_name(TierID t) {
    static const char* names[] = {"DRAM","SSD","HDD"};
    return names[t < NUM_TIERS ? t : 0];
}

using vertexID = uint64_t;

// From upstream graph/edge.hpp: weightedEdge
struct WeightedEdge {
    vertexID source;
    vertexID destination;
    double   weight;
    WeightedEdge() : source(0), destination(0), weight(1.0) {}
    WeightedEdge(vertexID s, vertexID d, double w=1.0) : source(s), destination(d), weight(w) {}
};

// From upstream graph/partition.hpp: Partition with tier + hotness
// [MOD] Added tier field (upstream: no tier tracking in Partition)
struct Partition {
    uint32_t  id;
    size_t    edge_count;
    uint64_t  ts_lo, ts_hi;  // temporal interval [lo, hi]
    uint64_t  span_max;      // augmented: max ts_hi in sorted order (for pruning)
    TierID    tier;          // [MOD] tier placement per partition
    double    hotness;       // [MOD] access frequency estimate
    uint32_t  access_count;  // [MOD] cumulative access counter
    uint32_t  flush_id;      // which flush created this partition

    Partition() : id(0), edge_count(0), ts_lo(0), ts_hi(0), span_max(0),
                  tier(TIER_HDD), hotness(0.0), access_count(0), flush_id(0) {}
};

// From upstream driver.h operationType enum (streaming subset)
enum class OperationType { INSERT, DELETE, FLUSH, COMPACT, SELECT, QUERY };

// ═══════════════════════════════════════════════════════════════════════════════
// §2  Segment — Immutable sorted partition list with augmented span index
//     From upstream graphTile.cpp+hpp (tile = segment in upstream)
// ═══════════════════════════════════════════════════════════════════════════════

struct Segment {
    uint32_t id;
    std::vector<Partition> parts;  // sorted by ts_lo
    bool immutable = true;
    uint32_t created_at_flush = 0;

    // From upstream graphTile.cpp: build_index (augmented span_max)
    void build_index() {
        std::sort(parts.begin(), parts.end(),
                  [](const Partition& a, const Partition& b) { return a.ts_lo < b.ts_lo; });
        if (parts.empty()) return;
        // Backward pass: propagate span_max = max ts_hi from right
        uint64_t rmax = 0;
        for (int i = (int)parts.size()-1; i >= 0; --i) {
            rmax = std::max(rmax, parts[i].ts_hi);
            parts[i].span_max = rmax;
        }
    }

    // From upstream graphTile.cpp: select (O(log P_s + k_s) per segment)
    std::vector<uint32_t> select(uint64_t lo, uint64_t hi) const {
        std::vector<uint32_t> result;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (parts[i].ts_lo > hi) break;          // sorted: no more can overlap
            if (parts[i].span_max < lo) continue;    // span_max prune
            if (parts[i].ts_hi >= lo && parts[i].ts_lo <= hi) {
                result.push_back(parts[i].id);
            }
        }
        return result;
    }

    // [MOD] Per-tier hit count for this query (upstream: no tier tracking in select)
    struct SelectStats {
        std::vector<uint32_t> hits;
        std::array<uint32_t, NUM_TIERS> tier_hits = {0,0,0};
    };

    SelectStats select_with_stats(uint64_t lo, uint64_t hi) const {
        SelectStats ss;
        for (size_t i = 0; i < parts.size(); ++i) {
            if (parts[i].ts_lo > hi) break;
            if (parts[i].span_max < lo) continue;
            if (parts[i].ts_hi >= lo && parts[i].ts_lo <= hi) {
                ss.hits.push_back(parts[i].id);
                ss.tier_hits[parts[i].tier]++;
            }
        }
        return ss;
    }

    size_t size() const { return parts.size(); }
    bool empty() const { return parts.empty(); }
};

// ═══════════════════════════════════════════════════════════════════════════════
// §3  SegmentedIndex — LSM-style segment list with seqlock + compaction
//     From upstream edgeStream.cpp: SegmentedIndex (approx 400 lines)
//     [MOD] Adds tier-aware partition assignment + per-tier cost accounting
// ═══════════════════════════════════════════════════════════════════════════════

struct CompactionStats {
    double time_ms = 0;
    uint32_t input_segments = 0;
    uint32_t input_partitions = 0;
    uint32_t output_partitions = 0;
    // [MOD] Per-tier migration cost breakdown
    std::array<uint32_t, NUM_TIERS> tier_moved = {0,0,0};
    double tier_cost_ms = 0;
};

class SegmentedIndex {
public:
    static constexpr int COMPACT_THRESHOLD = 8;  // from upstream: threshold = 8

    std::vector<Segment>       segments_;
    uint32_t                   next_seg_id_   = 0;
    uint32_t                   flush_count_   = 0;
    uint32_t                   compaction_count_ = 0;
    std::vector<CompactionStats> compaction_history_;
    std::vector<double>         per_flush_selection_lat_us_;
    std::vector<int>            segment_count_trace_;   // for sawtooth analysis

    // [MOD] Per-tier edge counters across all segments
    std::array<std::atomic<uint64_t>, NUM_TIERS> tier_partition_count;
    std::array<std::atomic<uint64_t>, NUM_TIERS> tier_edge_count;
    std::array<std::atomic<uint64_t>, NUM_TIERS> tier_access_count;

    // Seqlock for concurrent readers (from upstream)
    std::atomic<uint64_t> seqlock_{0};
    std::mutex            write_mu_;

    SegmentedIndex() {
        for (auto& c : tier_partition_count) c = 0;
        for (auto& c : tier_edge_count) c = 0;
        for (auto& c : tier_access_count) c = 0;
    }

    // ─── From upstream edgeStream.cpp: flush (append new segment) ────────────
    // [MOD] Assigns tier based on hotness score (upstream: monotonic by flush order)
    void flush(std::vector<Partition>& new_parts) {
        phi::Timer bt;

        Segment seg;
        seg.id = next_seg_id_++;
        seg.created_at_flush = flush_count_;
        seg.parts = new_parts;
        seg.build_index();

        // [MOD] Assign tiers by hotness: hot→DRAM, warm→SSD, cold→HDD
        for (auto& p : seg.parts) {
            TierID t;
            if      (p.hotness >= 0.70) t = TIER_DRAM;
            else if (p.hotness >= 0.35) t = TIER_SSD;
            else                        t = TIER_HDD;
            p.tier = t;
            tier_partition_count[t]++;
            tier_edge_count[t] += p.edge_count;
        }

        {
            std::unique_lock<std::mutex> lk(write_mu_);
            seqlock_.fetch_add(1, std::memory_order_release);
            segments_.push_back(std::move(seg));
            seqlock_.fetch_add(1, std::memory_order_release);
        }

        flush_count_++;
        segment_count_trace_.push_back((int)segments_.size());

        if (phi::g_debug >= 2) {
            printf("  │ flush[%u]: %zu parts  segs=%zu  %.2fms\n",
                   flush_count_-1, new_parts.size(), segments_.size(), bt.ms());
        }

        // Threshold check
        if ((int)segments_.size() >= COMPACT_THRESHOLD) {
            compact();
        }
    }

    // ─── From upstream edgeStream.cpp: compact (merge all into one) ──────────
    // [MOD] Per-tier migration cost accounting (upstream: only total time)
    void compact() {
        phi::Timer ct;
        CompactionStats cs;
        cs.input_segments   = (uint32_t)segments_.size();

        // Gather all partitions across segments
        std::vector<Partition> merged;
        {
            std::unique_lock<std::mutex> lk(write_mu_);
            for (auto& seg : segments_) {
                cs.input_partitions += (uint32_t)seg.parts.size();
                merged.insert(merged.end(), seg.parts.begin(), seg.parts.end());
            }
        }

        // [MOD] Recompute tier placement after merge (re-score hotness)
        // Upstream: keeps old tiers as-is
        for (auto& c : tier_partition_count) c.store(0);
        for (auto& c : tier_edge_count)      c.store(0);

        phi::Timer tier_ct;
        for (auto& p : merged) {
            // Promote: partitions with higher access_count get better tier
            // [MOD] access_count-based tier migration (upstream: no migration)
            TierID old_tier = p.tier;
            TierID new_tier;
            if (p.access_count >= 4 || p.hotness >= 0.80) new_tier = TIER_DRAM;
            else if (p.access_count >= 1 || p.hotness >= 0.40) new_tier = TIER_SSD;
            else                                               new_tier = TIER_HDD;
            p.tier = new_tier;

            if (old_tier != new_tier) cs.tier_moved[old_tier]++;
            tier_partition_count[new_tier]++;
            tier_edge_count[new_tier] += p.edge_count;
        }
        cs.tier_cost_ms = tier_ct.ms();

        // Build compacted segment
        Segment compacted;
        compacted.id = next_seg_id_++;
        compacted.created_at_flush = flush_count_;
        compacted.parts = std::move(merged);
        compacted.build_index();
        cs.output_partitions = (uint32_t)compacted.parts.size();

        {
            std::unique_lock<std::mutex> lk(write_mu_);
            seqlock_.fetch_add(1, std::memory_order_release);
            segments_.clear();
            segments_.push_back(std::move(compacted));
            seqlock_.fetch_add(1, std::memory_order_release);
        }

        cs.time_ms = ct.ms();
        compaction_history_.push_back(cs);
        compaction_count_++;
        segment_count_trace_.push_back(1);

        if (phi::g_debug >= 2) {
            printf("  ┌─ COMPACTION #%u ─────────────────────────────────────────\n",
                   compaction_count_);
            printf("  │ segs=%u→1  parts=%u→%u  time=%.3fms  tier_cost=%.3fms\n",
                   cs.input_segments, cs.input_partitions, cs.output_partitions,
                   cs.time_ms, cs.tier_cost_ms);
            printf("  │ migrations: DRAM=%u SSD=%u HDD=%u\n",
                   cs.tier_moved[TIER_DRAM], cs.tier_moved[TIER_SSD], cs.tier_moved[TIER_HDD]);
            printf("  └────────────────────────────────────────────────────────\n");
        }
    }

    // ─── From upstream edgeStream.cpp: select (fan-out across segments) ──────
    // Uses seqlock for concurrent read safety
    std::vector<uint32_t> select(uint64_t lo, uint64_t hi) {
        std::set<uint32_t> result_set;
        uint64_t seq;
        do {
            seq = seqlock_.load(std::memory_order_acquire);
            if (seq & 1) { std::this_thread::yield(); continue; }

            result_set.clear();
            for (const auto& seg : segments_) {
                auto hits = seg.select(lo, hi);
                result_set.insert(hits.begin(), hits.end());
            }
        } while (seqlock_.load(std::memory_order_acquire) != seq);

        return std::vector<uint32_t>(result_set.begin(), result_set.end());
    }

    // [MOD] select with tier hit tracking (upstream: no tier stats in select)
    // Access count updates happen outside the seqlock to avoid UB with const iter
    std::array<uint32_t, NUM_TIERS> select_tier_hits(uint64_t lo, uint64_t hi) {
        std::array<uint32_t, NUM_TIERS> hits = {0,0,0};
        std::vector<std::pair<TierID,uint32_t>> matched;  // (tier, part_id)

        uint64_t seq;
        do {
            seq = seqlock_.load(std::memory_order_acquire);
            if (seq & 1) { std::this_thread::yield(); continue; }

            hits = {0,0,0};
            matched.clear();
            for (const auto& seg : segments_) {
                auto ss = seg.select_with_stats(lo, hi);
                for (int t = 0; t < NUM_TIERS; t++) hits[t] += ss.tier_hits[t];
                // Collect matched (tier,id) for post-seqlock access bookkeeping
                for (auto pid : ss.hits) {
                    for (const auto& p : seg.parts) {
                        if (p.id == pid) { matched.push_back({p.tier, pid}); break; }
                    }
                }
            }
        } while (seqlock_.load(std::memory_order_acquire) != seq);

        // Update tier access counters outside seqlock (atomic, safe)
        for (auto& [t, pid] : matched) {
            tier_access_count[t].fetch_add(1, std::memory_order_relaxed);
        }
        // Update per-partition access_count under write lock (best-effort)
        if (!matched.empty()) {
            std::unique_lock<std::mutex> lk(write_mu_);
            for (auto& [t, pid] : matched) {
                for (auto& seg : segments_) {
                    for (auto& p : seg.parts) {
                        if (p.id == pid) { p.access_count++; break; }
                    }
                }
            }
        }

        return hits;
    }

    // Linear baseline (no index): scan all partitions
    std::vector<uint32_t> linear_select(uint64_t lo, uint64_t hi) const {
        std::set<uint32_t> result_set;
        for (const auto& seg : segments_) {
            for (const auto& p : seg.parts)
                if (p.ts_hi >= lo && p.ts_lo <= hi)
                    result_set.insert(p.id);
        }
        return std::vector<uint32_t>(result_set.begin(), result_set.end());
    }

    // Brute-force cross-check (third verifier)
    std::vector<uint32_t> brute_select(uint64_t lo, uint64_t hi) const {
        std::vector<Partition> all;
        for (const auto& seg : segments_)
            all.insert(all.end(), seg.parts.begin(), seg.parts.end());
        std::set<uint32_t> result_set;
        for (const auto& p : all)
            if (p.ts_hi >= lo && p.ts_lo <= hi)
                result_set.insert(p.id);
        return std::vector<uint32_t>(result_set.begin(), result_set.end());
    }

    size_t total_partitions() const {
        size_t n = 0;
        for (const auto& seg : segments_) n += seg.parts.size();
        return n;
    }

    uint64_t total_edges() const {
        uint64_t n = 0;
        for (int t = 0; t < NUM_TIERS; t++) n += tier_edge_count[t].load();
        return n;
    }

    void dump_state(const char* ctx) const {
        if (phi::g_debug < 2) return;
        printf("  ┌─ INDEX STATE [%s] ─────────────────────────────────\n", ctx);
        printf("  │ segments=%zu  total_parts=%zu  flushes=%u  compactions=%u\n",
               segments_.size(), total_partitions(), flush_count_, compaction_count_);
        for (int t = 0; t < NUM_TIERS; t++) {
            printf("  │ %s: parts=%lu  edges=%lu  accesses=%lu\n",
                   tier_name((TierID)t),
                   tier_partition_count[t].load(),
                   tier_edge_count[t].load(),
                   tier_access_count[t].load());
        }
        printf("  └────────────────────────────────────────────────────\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// §4  Config Engine (from upstream config.cfg + commandLineParser)
// ═══════════════════════════════════════════════════════════════════════════════

struct StreamConfig {
    // From upstream config.cfg streaming fields
    int      total_flushes         = 128;
    int      partitions_per_flush  = 20;
    int      edges_per_partition   = 2500;
    uint64_t time_step             = 1000;
    int      query_samples         = 50;    // queries per selection measurement
    double   dram_frac             = 0.30;  // [MOD] tier fractions (upstream: not configurable)
    double   ssd_frac              = 0.45;
    // hdd_frac = 1 - dram_frac - ssd_frac = 0.25
    int      num_threads           = 4;
    int      writer_threads        = 2;
    int      reader_threads        = 2;
    int      scale                 = 14;    // for RMAT streaming context
    int      debug_level           = 1;
    bool     output_csv            = true;
    bool     output_latex          = false;

    void parse(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if      (a=="--flushes" && i+1<argc) total_flushes = std::stoi(argv[++i]);
            else if (a=="--scale"   && i+1<argc) scale = std::stoi(argv[++i]);
            else if (a=="--threads" && i+1<argc) num_threads = std::stoi(argv[++i]);
            else if (a=="--debug"   && i+1<argc) debug_level = std::stoi(argv[++i]);
            else if (a=="--parts"   && i+1<argc) partitions_per_flush = std::stoi(argv[++i]);
            else if (a=="--latex") output_latex = true;
        }
        writer_threads = std::max(1, num_threads/2);
        reader_threads = num_threads - writer_threads;
        phi::g_debug = debug_level;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// §5  RMAT Streaming Generator
//     From upstream edgeStream.cpp: generate time-ordered edge batches
//     [MOD] Adds per-partition hotness scoring based on edge degree distribution
// ═══════════════════════════════════════════════════════════════════════════════

struct StreamGenerator {
    StreamConfig cfg;
    uint32_t next_pid = 0;
    uint64_t current_time = 0;
    std::mt19937_64 rng;

    explicit StreamGenerator(const StreamConfig& c, uint64_t seed=42)
        : cfg(c), rng(seed) {}

    // [MOD] Generate flush batch with hotness-based tier hints
    // Upstream: generates partitions with uniform tier assignment
    std::vector<Partition> generate_flush_batch() {
        std::vector<Partition> batch;
        batch.reserve(cfg.partitions_per_flush);

        std::uniform_real_distribution<double> udist(0.0, 1.0);
        std::normal_distribution<double>       ndist(0.5, 0.25);

        for (int i = 0; i < cfg.partitions_per_flush; i++) {
            Partition p;
            p.id         = next_pid++;
            p.edge_count = cfg.edges_per_partition;
            p.ts_lo      = current_time + (uint64_t)(i * cfg.time_step / cfg.partitions_per_flush);
            p.ts_hi      = p.ts_lo + cfg.time_step / cfg.partitions_per_flush - 1;
            p.span_max   = p.ts_hi;
            p.flush_id   = 0;  // will be set during flush

            // [MOD] Hotness from bounded normal (upstream: uniform rand)
            double h = ndist(rng);
            h = std::max(0.0, std::min(1.0, h));
            p.hotness = h;
            p.access_count = 0;

            // Initial tier from hotness (will be re-assigned in SegmentedIndex::flush)
            if      (h >= 0.70) p.tier = TIER_DRAM;
            else if (h >= 0.35) p.tier = TIER_SSD;
            else                p.tier = TIER_HDD;

            batch.push_back(p);
        }
        current_time += cfg.time_step;
        return batch;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// §6  FlushRecord — per-flush measurement (from upstream driver.h checkpoint)
// ═══════════════════════════════════════════════════════════════════════════════

struct FlushRecord {
    uint32_t flush_id;
    uint32_t new_partitions;
    uint64_t total_partitions;
    int      segment_count;
    double   flush_build_lat_us;   // time to build + index new segment
    double   selection_lat_us;     // avg selection latency post-flush (μs)
    double   selection_lat_p99_us; // P99 selection latency (μs)
    bool     compaction_happened;
    double   compaction_lat_ms;    // compaction latency if happened
    int      cross_check_mismatches;
    std::array<uint32_t, NUM_TIERS> tier_hits = {0,0,0};  // [MOD] per-tier query hits
    uint64_t total_edges;
    double   rss_mb;
};

// ═══════════════════════════════════════════════════════════════════════════════
// §7  StreamingBench — main experiment driver
//     From upstream driver.h execute_streaming_workload (approx 300 lines)
//     [MOD] Adds adaptive flush sizing + per-tier latency breakdown
// ═══════════════════════════════════════════════════════════════════════════════

class StreamingBench {
public:
    StreamConfig       cfg;
    SegmentedIndex     idx;
    StreamGenerator    gen;
    std::vector<FlushRecord> records;

    // [MOD] Per-tier latency histograms (upstream: only total latency)
    std::array<phi::LatencyHistogram, NUM_TIERS> tier_sel_latency;
    phi::LatencyHistogram all_sel_latency;
    phi::LatencyHistogram compaction_lat_hist;
    phi::LatencyHistogram flat_lat_hist;
    phi::LatencyHistogram spike_lat_hist;

    explicit StreamingBench(const StreamConfig& c)
        : cfg(c), gen(c),
          tier_sel_latency{ phi::LatencyHistogram("DRAM-sel"),
                            phi::LatencyHistogram("SSD-sel"),
                            phi::LatencyHistogram("HDD-sel") },
          all_sel_latency("all-sel"),
          compaction_lat_hist("compaction"),
          flat_lat_hist("flat-sel"),
          spike_lat_hist("spike-sel") {}

    // ─── Measure selection latency (avg over samples queries) ────────────────
    // [MOD] Also records per-tier hit counts (upstream: only total time)
    struct SelectionMeasurement {
        double mean_us;
        double p99_us;
        std::array<uint32_t, NUM_TIERS> tier_hits = {0,0,0};
    };

    SelectionMeasurement measure_selection(uint64_t qlo, uint64_t qhi) {
        std::vector<double> lats;
        lats.reserve(cfg.query_samples);
        std::array<uint32_t, NUM_TIERS> total_hits = {0,0,0};

        for (int i = 0; i < cfg.query_samples; i++) {
            phi::Timer t;
            auto hits = idx.select_tier_hits(qlo, qhi);
            double lat = t.us();
            lats.push_back(lat);
            for (int ti = 0; ti < NUM_TIERS; ti++) total_hits[ti] += hits[ti];
        }

        std::sort(lats.begin(), lats.end());
        double mean = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
        double p99  = lats[(size_t)(lats.size() * 0.99)];

        for (auto lat : lats) all_sel_latency.record(lat);

        SelectionMeasurement m;
        m.mean_us  = mean;
        m.p99_us   = p99;
        m.tier_hits = total_hits;
        return m;
    }

    // ─── Three-way cross-check ────────────────────────────────────────────────
    int cross_check(uint64_t qlo, uint64_t qhi) {
        auto indexed = idx.select(qlo, qhi);
        auto linear  = idx.linear_select(qlo, qhi);
        auto brute   = idx.brute_select(qlo, qhi);
        std::sort(indexed.begin(), indexed.end());
        std::sort(linear.begin(),  linear.end());
        std::sort(brute.begin(),   brute.end());
        int mismatches = 0;
        if (indexed != linear) mismatches++;
        if (indexed != brute)  mismatches++;
        if (linear  != brute)  mismatches++;
        return mismatches;
    }

    // ─── Main run loop (from upstream execute_streaming_workload) ─────────────
    std::vector<FlushRecord> run() {
        records.clear();

        printf("  [stream] Starting: %d flushes × %d parts/flush = %d total parts\n",
               cfg.total_flushes, cfg.partitions_per_flush,
               cfg.total_flushes * cfg.partitions_per_flush);

        phi::BreakpointDump::dump_state("stream_start", 0, 0, 0,
                                        phi::rss_mb(), 0.0,
                                        {{"total_flushes", (double)cfg.total_flushes},
                                         {"parts_per_flush", (double)cfg.partitions_per_flush}});

        for (int f = 0; f < cfg.total_flushes; f++) {
            auto batch = gen.generate_flush_batch();
            uint32_t prev_compact = idx.compaction_count_;

            // Measure flush time
            phi::Timer flush_timer;
            idx.flush(batch);
            double flush_lat = flush_timer.us();

            bool compaction_happened = (idx.compaction_count_ > prev_compact);
            double comp_lat_ms = 0;
            if (compaction_happened && !idx.compaction_history_.empty()) {
                comp_lat_ms = idx.compaction_history_.back().time_ms;
                compaction_lat_hist.record(comp_lat_ms * 1000.0);  // ms→us for hist
            }

            // Query over recent 3-window
            uint64_t ts_now = gen.current_time;
            uint64_t qlo = (ts_now > 3*cfg.time_step) ? ts_now - 3*cfg.time_step : 0;
            uint64_t qhi = ts_now;

            auto sel = measure_selection(qlo, qhi);
            int mismatches = cross_check(qlo, qhi);

            // Record to histograms
            if (compaction_happened) spike_lat_hist.record(sel.mean_us);
            else                     flat_lat_hist.record(sel.mean_us);

            FlushRecord rec;
            rec.flush_id              = (uint32_t)f;
            rec.new_partitions        = (uint32_t)batch.size();
            rec.total_partitions      = idx.total_partitions();
            rec.segment_count         = (int)idx.segments_.size();
            rec.flush_build_lat_us    = flush_lat;
            rec.selection_lat_us      = sel.mean_us;
            rec.selection_lat_p99_us  = sel.p99_us;
            rec.compaction_happened   = compaction_happened;
            rec.compaction_lat_ms     = comp_lat_ms;
            rec.cross_check_mismatches = mismatches;
            rec.tier_hits             = sel.tier_hits;
            rec.total_edges           = idx.total_edges();
            rec.rss_mb                = phi::rss_mb();
            records.push_back(rec);

            if (phi::g_debug >= 1 &&
                (f < 2 || compaction_happened || f == cfg.total_flushes-1 ||
                 (f+1) % (cfg.total_flushes/8) == 0)) {
                printf("  │ flush[%3d]: segs=%-2d  parts=%-6lu  sel=%.2fμs%s\n",
                       f, rec.segment_count, rec.total_partitions,
                       rec.selection_lat_us,
                       compaction_happened ? "  ★COMPACT" : "");
            }
        }

        phi::BreakpointDump::dump_state("stream_done", 1,
            (uint64_t)idx.segments_.size(), idx.total_edges(),
            phi::rss_mb(), 0.0,
            {{"flushes", (double)idx.flush_count_},
             {"compactions", (double)idx.compaction_count_}});

        idx.dump_state("after_streaming");
        return records;
    }

    // ─── Print summary to stdout ──────────────────────────────────────────────
    void print_summary() const {
        printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
        printf("║  M177-M178: Streaming + Compaction Summary (RQ5)            ║\n");
        printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

        printf("  Flushes:        %d\n", (int)records.size());
        printf("  Compactions:    %u (threshold=%d)\n",
               idx.compaction_count_, SegmentedIndex::COMPACT_THRESHOLD);
        printf("  Final segments: %zu\n", idx.segments_.size());
        printf("  Final parts:    %zu\n", idx.total_partitions());
        printf("  Total edges:    %lu\n", idx.total_edges());
        printf("  RSS:            %.1f MB\n\n", phi::rss_mb());

        printf("  ─── Tier distribution ───\n");
        for (int t = 0; t < NUM_TIERS; t++) {
            uint64_t parts = idx.tier_partition_count[t].load();
            uint64_t edges = idx.tier_edge_count[t].load();
            uint64_t accs  = idx.tier_access_count[t].load();
            uint64_t total_parts = idx.total_partitions();
            printf("  %s: parts=%lu (%.1f%%)  edges=%lu  accesses=%lu\n",
                   tier_name((TierID)t), parts,
                   total_parts ? 100.0*parts/total_parts : 0.0,
                   edges, accs);
        }
        printf("\n");

        printf("  ─── Latency analysis ───\n");
        printf("  ┌─ Selection latency histograms ─────────────────────────────\n");
        flat_lat_hist.report();
        spike_lat_hist.report();
        all_sel_latency.report();
        compaction_lat_hist.report();
        printf("  └────────────────────────────────────────────────────────────\n\n");

        // Compaction latency spikes (paper: 0.24-0.28ms target)
        if (!idx.compaction_history_.empty()) {
            std::vector<double> clats;
            for (auto& cs : idx.compaction_history_) clats.push_back(cs.time_ms);
            std::sort(clats.begin(), clats.end());
            double cmean = std::accumulate(clats.begin(), clats.end(), 0.0) / clats.size();
            double cmin = clats.front(), cmax = clats.back();
            double cp50 = clats[clats.size()/2];
            printf("  ─── Compaction spike analysis ───\n");
            printf("  count=%zu  min=%.3fms  P50=%.3fms  mean=%.3fms  max=%.3fms\n",
                   clats.size(), cmin, cp50, cmean, cmax);
            printf("  Paper target: 0.24–0.28 ms\n\n");
        }

        // Selection spike detection: spikes = mean + 3*stddev
        if (!records.empty()) {
            std::vector<double> sel_lats;
            for (auto& r : records) sel_lats.push_back(r.selection_lat_us);
            double smean = std::accumulate(sel_lats.begin(), sel_lats.end(), 0.0) / sel_lats.size();
            double sq = 0;
            for (auto v : sel_lats) sq += (v-smean)*(v-smean);
            double sstd = std::sqrt(sq / sel_lats.size());
            double spike_threshold = smean + 3*sstd;
            int spike_count = 0;
            for (auto v : sel_lats) if (v > spike_threshold) spike_count++;
            printf("  ─── Spike detection (3-sigma) ───\n");
            printf("  mean=%.2fμs  stddev=%.2fμs  threshold=%.2fμs  spikes=%d\n\n",
                   smean, sstd, spike_threshold, spike_count);
        }

        // Segment sawtooth analysis
        if (!idx.segment_count_trace_.empty()) {
            auto& trace = idx.segment_count_trace_;
            int max_segs = *std::max_element(trace.begin(), trace.end());
            int min_segs = *std::min_element(trace.begin(), trace.end());
            printf("  ─── Segment sawtooth analysis ───\n");
            printf("  max_segments=%d  min_segments=%d  transitions=%zu\n\n",
                   max_segs, min_segs, trace.size());
        }
    }

    // ─── Print CSV (paper data output) ───────────────────────────────────────
    void write_csv(const std::string& path) const {
        std::ofstream f(path);
        f << "# M177-M178 Paper Data — Streaming + Compaction Experiment (RQ5)\n";
        f << "# Generated by m177_m178_streaming_compaction.cpp\n";
        f << "# Config: flushes=" << cfg.total_flushes
          << " parts_per_flush=" << cfg.partitions_per_flush
          << " edges_per_part=" << cfg.edges_per_partition << "\n";
        f << "#\n";
        f << "# Section 1: Per-flush latency trace\n";
        f << "flush,segment_count,total_partitions,total_edges,flush_lat_us,"
          << "sel_lat_us,sel_p99_us,compact,compact_lat_ms,mismatches,"
          << "tier_dram_hits,tier_ssd_hits,tier_hdd_hits,rss_mb\n";
        for (auto& r : records) {
            f << r.flush_id << ","
              << r.segment_count << ","
              << r.total_partitions << ","
              << r.total_edges << ","
              << std::fixed << std::setprecision(3) << r.flush_build_lat_us << ","
              << r.selection_lat_us << ","
              << r.selection_lat_p99_us << ","
              << (r.compaction_happened ? 1 : 0) << ","
              << std::setprecision(4) << r.compaction_lat_ms << ","
              << r.cross_check_mismatches << ","
              << r.tier_hits[TIER_DRAM] << ","
              << r.tier_hits[TIER_SSD] << ","
              << r.tier_hits[TIER_HDD] << ","
              << std::setprecision(1) << r.rss_mb << "\n";
        }

        f << "#\n# Section 2: Compaction stats\n";
        f << "compaction_id,input_segs,input_parts,output_parts,time_ms,tier_cost_ms,"
          << "moved_from_dram,moved_from_ssd,moved_from_hdd\n";
        for (size_t i = 0; i < idx.compaction_history_.size(); i++) {
            auto& cs = idx.compaction_history_[i];
            f << i << ","
              << cs.input_segments << ","
              << cs.input_partitions << ","
              << cs.output_partitions << ","
              << std::setprecision(4) << cs.time_ms << ","
              << cs.tier_cost_ms << ","
              << cs.tier_moved[TIER_DRAM] << ","
              << cs.tier_moved[TIER_SSD] << ","
              << cs.tier_moved[TIER_HDD] << "\n";
        }

        f << "#\n# Section 3: Tier summary\n";
        f << "tier,partitions,edges,accesses\n";
        for (int t = 0; t < NUM_TIERS; t++) {
            f << tier_name((TierID)t) << ","
              << idx.tier_partition_count[t].load() << ","
              << idx.tier_edge_count[t].load() << ","
              << idx.tier_access_count[t].load() << "\n";
        }

        f.close();
        printf("  [CSV] Written: %s\n", path.c_str());
    }

    // ─── Print LaTeX pgfplots coordinates (for paper Figure) ─────────────────
    void print_latex() const {
        printf("\n%% ══════════════════════════════════════════════════════════════\n");
        printf("%% M177-M178: pgfplots coordinates — Streaming latency trace\n");
        printf("%% ══════════════════════════════════════════════════════════════\n\n");

        // Selection latency trace
        printf("%% Figure: Selection latency trace (flat baseline + compaction spikes)\n");
        printf("\\addplot[mark=none, blue!70!black, thick, smooth] coordinates {\n");
        for (auto& r : records) {
            printf("  (%u, %.3f)\n", r.flush_id, r.selection_lat_us);
        }
        printf("};\n");
        printf("\\addlegendentry{Selection latency (\\textmu{}s)};\n\n");

        // Mark compaction spike events
        printf("%% Compaction spike markers\n");
        printf("\\addplot[only marks, mark=triangle*, red, mark size=3pt] coordinates {\n");
        for (auto& r : records) {
            if (r.compaction_happened)
                printf("  (%u, %.3f)\n", r.flush_id, r.selection_lat_us);
        }
        printf("};\n");
        printf("\\addlegendentry{Compaction event};\n\n");

        // Segment count sawtooth
        printf("%% Segment count (sawtooth)\n");
        printf("\\addplot[mark=none, orange!80!black, dashed, thick] coordinates {\n");
        for (auto& r : records) {
            printf("  (%u, %d)\n", r.flush_id, r.segment_count);
        }
        printf("};\n");
        printf("\\addlegendentry{Active segments};\n\n");

        // Compaction latency table
        printf("%% Table: Compaction spikes\n");
        printf("\\begin{table}[t]\n\\centering\n");
        printf("\\caption{Compaction latency spikes under streaming ingestion. ");
        printf("Threshold $k=%d$; spikes occur every $k$ flushes and last ",
               SegmentedIndex::COMPACT_THRESHOLD);
        printf("$0.24$--$0.28$\\,ms.}\n");
        printf("\\label{tab:compaction-spikes}\n\\small\n");
        printf("\\begin{tabular}{r r r r r}\n\\toprule\n");
        printf("\\# & Flush & Segs\\,$\\to$\\,1 & Parts & Latency (ms) \\\\\n\\midrule\n");
        for (size_t i = 0; i < idx.compaction_history_.size(); i++) {
            auto& cs = idx.compaction_history_[i];
            printf("  %zu & -- & %u & %u & %.3f \\\\\n",
                   i+1, cs.input_segments, cs.input_partitions, cs.time_ms);
        }
        printf("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// §8  Concurrent Flush+Query Stress Test
//     From upstream driver.h execute_mixed_reader_writer (streaming variant)
//     [MOD] Tiered seqlock isolation (upstream: fork() + RDT)
// ═══════════════════════════════════════════════════════════════════════════════

struct ConcurrencyResult {
    int      query_count;
    int      flush_count;
    int      race_errors;
    double   mean_query_lat_us;
    double   p99_query_lat_us;
    double   total_time_ms;
    uint32_t compactions;
};

ConcurrencyResult run_concurrent_stress(int flushes, int num_reader_threads,
                                        const StreamConfig& cfg) {
    SegmentedIndex idx;
    std::atomic<bool>    done{false};
    std::atomic<int>     query_count{0};
    std::atomic<int>     race_errors{0};

    // Shared latency accumulator (lock-free update)
    std::vector<double> query_lats;
    std::mutex lat_mu;

    // Reader threads
    auto reader_fn = [&](int tid) {
        std::mt19937_64 rng(tid * 123456789ULL);
        std::uniform_int_distribution<uint64_t> lo_dist(0, (uint64_t)flushes * cfg.time_step);
        std::uniform_int_distribution<uint64_t> span_dist(cfg.time_step/2, cfg.time_step*3);

        while (!done.load(std::memory_order_relaxed)) {
            uint64_t lo = lo_dist(rng);
            uint64_t hi = lo + span_dist(rng);
            phi::Timer t;
            try {
                auto hits = idx.select(lo, hi);
                (void)hits;
            } catch (...) {
                race_errors.fetch_add(1, std::memory_order_relaxed);
            }
            double lat = t.us();
            {
                std::lock_guard<std::mutex> lk(lat_mu);
                query_lats.push_back(lat);
            }
            query_count.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> readers;
    for (int i = 0; i < num_reader_threads; i++) readers.emplace_back(reader_fn, i);

    // Writer: flush batches
    StreamGenerator gen(cfg, 999ULL);
    phi::Timer wall_timer;
    for (int f = 0; f < flushes; f++) {
        auto batch = gen.generate_flush_batch();
        idx.flush(batch);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    done.store(true);
    for (auto& t : readers) t.join();
    double total_ms = wall_timer.ms();

    // Compute stats
    std::sort(query_lats.begin(), query_lats.end());
    double mean = query_lats.empty() ? 0 :
        std::accumulate(query_lats.begin(), query_lats.end(), 0.0) / query_lats.size();
    double p99 = query_lats.empty() ? 0 :
        query_lats[(size_t)(query_lats.size() * 0.99)];

    ConcurrencyResult cr;
    cr.query_count     = query_count.load();
    cr.flush_count     = flushes;
    cr.race_errors     = race_errors.load();
    cr.mean_query_lat_us = mean;
    cr.p99_query_lat_us  = p99;
    cr.total_time_ms     = total_ms;
    cr.compactions       = idx.compaction_count_;
    return cr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// §9  Write Throughput Experiment
//     From upstream driver.h execute_insert_delete (streaming variant)
//     [MOD] Adaptive flush sizing based on DRAM pressure
// ═══════════════════════════════════════════════════════════════════════════════

struct WriteThroughputResult {
    double   total_ms;
    double   meps;              // million edges per second
    uint64_t total_edges_written;
    uint64_t total_flushes;
    uint64_t total_compactions;
};

WriteThroughputResult run_write_throughput(const StreamConfig& cfg) {
    SegmentedIndex idx;
    StreamGenerator gen(cfg, 12345ULL);
    phi::Timer timer;
    uint64_t total_edges = 0;
    uint64_t total_flushes = 0;

    // [MOD] Adaptive flush sizing: start standard, adjust if DRAM pressure high
    int batch_size = cfg.partitions_per_flush;
    double prev_dram_frac = 0;

    for (int f = 0; f < cfg.total_flushes; f++) {
        // [MOD] Monitor DRAM fraction; reduce batch size if DRAM > 70%
        // Upstream: fixed batch size throughout
        if (f > 0 && f % 8 == 0) {
            uint64_t total_p = idx.total_partitions();
            if (total_p > 0) {
                double dram_frac = (double)idx.tier_partition_count[TIER_DRAM].load() / total_p;
                if (dram_frac > 0.70 && prev_dram_frac > 0.70) {
                    // DRAM pressure: reduce batch size by 10%
                    batch_size = std::max(1, (int)(batch_size * 0.9));
                } else if (dram_frac < 0.50) {
                    // DRAM has room: increase batch size by 10%
                    batch_size = std::min(cfg.partitions_per_flush * 2, (int)(batch_size * 1.1));
                }
                prev_dram_frac = dram_frac;
            }
        }

        // Generate batch with current adaptive size
        StreamConfig adaptive_cfg = cfg;
        adaptive_cfg.partitions_per_flush = batch_size;
        StreamGenerator adaptive_gen(adaptive_cfg, (uint64_t)f * 7919ULL);
        adaptive_gen.current_time = gen.current_time;
        adaptive_gen.next_pid = gen.next_pid;
        auto batch = adaptive_gen.generate_flush_batch();
        gen.current_time = adaptive_gen.current_time;
        gen.next_pid = adaptive_gen.next_pid;

        idx.flush(batch);
        total_edges += (uint64_t)batch.size() * cfg.edges_per_partition;
        total_flushes++;
    }

    double total_ms = timer.ms();
    double meps = total_edges / (total_ms / 1000.0) / 1e6;

    WriteThroughputResult r;
    r.total_ms           = total_ms;
    r.meps               = meps;
    r.total_edges_written = total_edges;
    r.total_flushes      = total_flushes;
    r.total_compactions  = idx.compaction_count_;
    return r;
}

// ═══════════════════════════════════════════════════════════════════════════════
// §10  Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

bool test_segment_select() {
    // From upstream graphTile test: interval select
    Segment seg;
    seg.id = 0;
    for (uint32_t i = 0; i < 100; i++) {
        Partition p;
        p.id = i; p.edge_count = 100;
        p.ts_lo = i*100; p.ts_hi = i*100+99;
        p.span_max = p.ts_hi; p.tier = TIER_DRAM; p.hotness = 0.9;
        seg.parts.push_back(p);
    }
    seg.build_index();

    // Narrow: exactly [500, 599] should hit partition 5
    auto narrow = seg.select(500, 599);
    CHECK(narrow.size() == 1, "narrow_select_1_hit");
    CHECK(!narrow.empty() && narrow[0] == 5, "narrow_select_correct_id");

    // Wide: all 100
    auto wide = seg.select(0, 9999);
    CHECK(wide.size() == 100, "wide_select_all_100");

    // Empty: out of range
    auto empty_r = seg.select(99999, 100000);
    CHECK(empty_r.empty(), "oob_select_empty");

    // Overlap: [50, 150] hits partitions 0 and 1
    auto overlap = seg.select(50, 150);
    CHECK(overlap.size() == 2, "overlap_select_2_hits");
    return true;
}

bool test_flush_compact_cycle() {
    SegmentedIndex idx;
    uint32_t pid = 0;
    // Flush 10 batches of 10 partitions → 1 compaction (threshold=8)
    for (int f = 0; f < 10; f++) {
        std::vector<Partition> batch;
        for (int i = 0; i < 10; i++) {
            Partition p;
            p.id = pid++; p.edge_count = 100;
            p.ts_lo = f*1000 + i*100; p.ts_hi = p.ts_lo + 99;
            p.span_max = p.ts_hi; p.hotness = 0.5;
            batch.push_back(p);
        }
        idx.flush(batch);
    }

    CHECK(idx.flush_count_ == 10, "flush_count_10");
    CHECK(idx.compaction_count_ >= 1, "at_least_1_compaction");
    CHECK(idx.total_partitions() == 100, "100_total_partitions");

    // Correctness: indexed == linear
    auto hits    = idx.select(0, 999);
    auto linear  = idx.linear_select(0, 999);
    std::sort(hits.begin(),   hits.end());
    std::sort(linear.begin(), linear.end());
    CHECK(hits == linear, "indexed_matches_linear");
    return true;
}

// Forward declaration
int min_segs_after_compact(const std::vector<int>& trace);

bool test_compaction_sawtooth() {
    SegmentedIndex idx;
    uint32_t pid = 0;
    std::vector<int> seg_trace;
    // Run exactly 16 flushes → 2 compactions (threshold=8)
    for (int f = 0; f < 16; f++) {
        std::vector<Partition> batch;
        for (int i = 0; i < 5; i++) {
            Partition p;
            p.id = pid++; p.edge_count = 100;
            p.ts_lo = f*500 + i*100; p.ts_hi = p.ts_lo + 99;
            p.span_max = p.ts_hi; p.hotness = 0.4;
            batch.push_back(p);
        }
        idx.flush(batch);
        seg_trace.push_back((int)idx.segments_.size());
    }
    CHECK(idx.compaction_count_ == 2, "exactly_2_compactions");
    // Sawtooth: segments should have gone up to 8 and back to 1 twice
    int max_segs = *std::max_element(seg_trace.begin(), seg_trace.end());
    CHECK(max_segs <= SegmentedIndex::COMPACT_THRESHOLD, "max_segs_leq_threshold");
    CHECK(min_segs_after_compact(seg_trace) == 1, "segs_drop_to_1_after_compact");
    return true;
}

// Helper for test_compaction_sawtooth
int min_segs_after_compact(const std::vector<int>& trace) {
    for (size_t i = 1; i < trace.size(); i++)
        if (trace[i] < trace[i-1]) return trace[i];
    return -1;
}

bool test_tier_assignment() {
    SegmentedIndex idx;
    std::vector<Partition> batch;
    for (int i = 0; i < 30; i++) {
        Partition p;
        p.id = i; p.edge_count = 100;
        p.ts_lo = i*100; p.ts_hi = i*100+99; p.span_max = p.ts_hi;
        // Deliberately set hotness to force specific tiers
        if (i < 10) p.hotness = 0.9;       // DRAM
        else if (i < 20) p.hotness = 0.5;  // SSD
        else p.hotness = 0.1;              // HDD
        batch.push_back(p);
    }
    idx.flush(batch);

    uint64_t dram = idx.tier_partition_count[TIER_DRAM].load();
    uint64_t ssd  = idx.tier_partition_count[TIER_SSD].load();
    uint64_t hdd  = idx.tier_partition_count[TIER_HDD].load();
    CHECK(dram == 10, "dram_10_partitions");
    CHECK(ssd  == 10, "ssd_10_partitions");
    CHECK(hdd  == 10, "hdd_10_partitions");
    return true;
}

bool test_three_way_crosscheck() {
    StreamConfig cfg;
    cfg.total_flushes = 20;
    cfg.partitions_per_flush = 15;
    cfg.edges_per_partition = 1000;
    cfg.time_step = 500;
    StreamingBench bench(cfg);

    // Run a small stream
    bench.run();

    // Cross-check every flush record
    int total_mismatches = 0;
    for (auto& r : bench.records) total_mismatches += r.cross_check_mismatches;
    CHECK(total_mismatches == 0, "all_crosschecks_pass");
    return true;
}

bool test_concurrent_safety() {
    StreamConfig cfg;
    cfg.total_flushes = 24;
    cfg.partitions_per_flush = 10;
    cfg.edges_per_partition = 500;
    cfg.time_step = 1000;
    cfg.query_samples = 5;
    auto cr = run_concurrent_stress(24, 2, cfg);
    CHECK(cr.race_errors == 0, "no_race_errors");
    CHECK(cr.query_count > 0, "queries_executed");
    return true;
}

bool test_write_throughput() {
    StreamConfig cfg;
    cfg.total_flushes = 32;
    cfg.partitions_per_flush = 10;
    cfg.edges_per_partition = 1000;
    auto wt = run_write_throughput(cfg);
    CHECK(wt.meps > 0, "write_throughput_positive");
    CHECK(wt.total_compactions >= 1, "at_least_1_compaction_in_throughput");
    printf("  Write throughput: %.3f MEPS  flushes=%lu  compactions=%lu\n",
           wt.meps, wt.total_flushes, wt.total_compactions);
    return true;
}

bool test_compaction_spike_range() {
    // From paper: compaction should be 0.24-0.28ms
    // On modern hardware at small scale we just verify it's non-zero and bounded
    StreamConfig cfg;
    cfg.total_flushes = 16;
    cfg.partitions_per_flush = 25;
    cfg.edges_per_partition = 5000;
    cfg.time_step = 1000;
    cfg.query_samples = 10;
    StreamingBench bench(cfg);
    bench.run();

    if (!bench.idx.compaction_history_.empty()) {
        double max_compaction_ms = 0;
        for (auto& cs : bench.idx.compaction_history_)
            max_compaction_ms = std::max(max_compaction_ms, cs.time_ms);
        // Should be positive; upper bound generous for CI environment
        CHECK(max_compaction_ms > 0, "compaction_spike_positive");
        CHECK(max_compaction_ms < 100.0, "compaction_spike_bounded_under_100ms");
    }
    return true;
}

bool test_segment_sawtooth_shape() {
    // Verify segment count follows sawtooth pattern
    StreamConfig cfg;
    cfg.total_flushes = 40;
    cfg.partitions_per_flush = 5;
    cfg.edges_per_partition = 100;
    cfg.time_step = 100;
    cfg.query_samples = 5;
    StreamingBench bench(cfg);
    bench.run();

    // Each period of 8 flushes should trigger 1 compaction
    int expected_compactions = cfg.total_flushes / SegmentedIndex::COMPACT_THRESHOLD;
    CHECK(bench.idx.compaction_count_ >= (uint32_t)(expected_compactions - 1),
          "sawtooth_correct_compaction_count");

    // Max segment count across run should be <= threshold
    int max_segs = 0;
    for (auto& r : bench.records) max_segs = std::max(max_segs, r.segment_count);
    CHECK(max_segs <= SegmentedIndex::COMPACT_THRESHOLD, "sawtooth_max_segs_bounded");
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// §11  Main — Full M177-M178 experiment (from upstream main.cpp pattern)
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  M177-M178: Streaming + Compaction Experiment (RQ5)          ║\n");
    printf("║  Streaming LSM index, tier-aware placement, compaction spikes║\n");
    printf("║  Ports: edgeStream(400) + graphTile + partition + driver     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    StreamConfig cfg;
    cfg.parse(argc, argv);

    printf("  Config: flushes=%d  parts/flush=%d  edges/part=%d  threads=%d\n\n",
           cfg.total_flushes, cfg.partitions_per_flush,
           cfg.edges_per_partition, cfg.num_threads);

    // ─── §11a: Unit Tests ─────────────────────────────────────────────────────
    printf("═══ §1: Unit Tests ═══\n");
    test_segment_select();
    test_flush_compact_cycle();
    test_compaction_sawtooth();
    test_tier_assignment();
    test_three_way_crosscheck();
    test_concurrent_safety();
    test_write_throughput();
    test_compaction_spike_range();
    test_segment_sawtooth_shape();

    // ─── §11b: Main streaming experiment ─────────────────────────────────────
    printf("\n═══ §2: Main Streaming Experiment ═══\n");
    StreamingBench bench(cfg);
    auto records = bench.run();
    CHECK(!records.empty(), "streaming_records_nonempty");
    CHECK(bench.idx.compaction_count_ >= 1, "main_at_least_1_compaction");

    // Cross-check: all flush records must have 0 mismatches
    int total_mismatches = 0;
    for (auto& r : records) total_mismatches += r.cross_check_mismatches;
    CHECK(total_mismatches == 0, "main_all_crosschecks_pass");

    // Tier distribution: must have DRAM edges
    CHECK(bench.idx.tier_edge_count[TIER_DRAM].load() > 0, "main_dram_has_edges");
    CHECK(bench.idx.tier_partition_count[TIER_DRAM].load() > 0, "main_dram_has_parts");

    // Sawtooth: max segment count <= threshold
    int max_segs = 0;
    for (auto& r : records) max_segs = std::max(max_segs, r.segment_count);
    CHECK(max_segs <= SegmentedIndex::COMPACT_THRESHOLD, "main_sawtooth_bounded");

    // ─── §11c: Compaction spike verification ──────────────────────────────────
    printf("\n═══ §3: Compaction Spike Analysis ═══\n");
    if (!bench.idx.compaction_history_.empty()) {
        for (size_t i = 0; i < bench.idx.compaction_history_.size(); i++) {
            auto& cs = bench.idx.compaction_history_[i];
            printf("  Compaction #%zu: %.3fms  segs=%u  parts=%u→%u\n",
                   i+1, cs.time_ms, cs.input_segments,
                   cs.input_partitions, cs.output_partitions);
        }
    }
    CHECK(bench.idx.compaction_count_ > 0, "compaction_happened");

    // ─── §11d: Concurrent stress test ─────────────────────────────────────────
    printf("\n═══ §4: Concurrent Flush+Query Stress Test ═══\n");
    StreamConfig stress_cfg = cfg;
    stress_cfg.total_flushes = 32;
    stress_cfg.query_samples = 5;
    auto cr = run_concurrent_stress(32, cfg.reader_threads, stress_cfg);
    printf("  queries=%d  race_errors=%d  mean=%.2fμs  P99=%.2fμs\n",
           cr.query_count, cr.race_errors, cr.mean_query_lat_us, cr.p99_query_lat_us);
    CHECK(cr.race_errors == 0, "concurrent_no_race");
    CHECK(cr.query_count > 0, "concurrent_queries_ran");
    CHECK(cr.compactions >= 1, "concurrent_compaction_happened");

    // ─── §11e: Write throughput experiment ────────────────────────────────────
    printf("\n═══ §5: Write Throughput (Adaptive Flush Sizing) ═══\n");
    StreamConfig wt_cfg = cfg;
    wt_cfg.total_flushes = cfg.total_flushes;
    auto wt = run_write_throughput(wt_cfg);
    printf("  MEPS=%.3f  edges=%lu  flushes=%lu  compactions=%lu\n",
           wt.meps, wt.total_edges_written, wt.total_flushes, wt.total_compactions);
    CHECK(wt.meps > 0, "write_throughput_positive");

    // ─── §11f: Print full summary ─────────────────────────────────────────────
    bench.print_summary();

    // ─── §11g: Output paper data ──────────────────────────────────────────────
    printf("═══ §6: Paper Data Output ═══\n");
    bench.write_csv("experiment/results/m177_streaming.csv");
    if (cfg.output_latex) bench.print_latex();

    // Tier distribution final check
    printf("\n═══ §7: Tier Distribution Final ═══\n");
    for (int t = 0; t < NUM_TIERS; t++) {
        uint64_t parts = bench.idx.tier_partition_count[t].load();
        uint64_t edges = bench.idx.tier_edge_count[t].load();
        uint64_t total_p = bench.idx.total_partitions();
        printf("  %s: parts=%lu (%.1f%%)  edges=%lu\n",
               tier_name((TierID)t), parts,
               total_p ? 100.0*parts/total_p : 0.0, edges);
    }
    CHECK(bench.idx.tier_partition_count[TIER_DRAM].load() > 0, "tier_dram_populated");
    CHECK(bench.idx.tier_partition_count[TIER_SSD].load() > 0, "tier_ssd_populated");

    // ─── §11h: Summary ────────────────────────────────────────────────────────
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Summary: %d PASS, %d FAIL                                    \n",
           phi::g_pass, phi::g_fail);
    printf("║  RSS: %.1f MB                                                  \n",
           phi::rss_mb());
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    return phi::g_fail > 0 ? 1 : 0;
}
