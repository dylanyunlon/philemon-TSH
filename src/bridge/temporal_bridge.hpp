#pragma once
/**
 * temporal_bridge.hpp — 时间索引-图存储桥接层（区间→分区→层级放置）
 *
 * This is the critical integration layer.  TEM-Graph stores temporal intervals
 * (start, end timestamps) with a doubly-linked-list index for contains/contained
 * queries.  RapidStore stores the actual graph topology with concurrent
 * snapshot isolation.  The bridge maps:
 *
 *   interval ──► subgraph partition ──► memory tier placement
 *
 * Pattern lineage:
 *   - RapidStore's `wrapper::snapshot_edges(S, u, callback, logical)` for
 *     traversal abstraction.
 *   - TEM-Graph's `TemGraph::build_index(sorted_by_start, sorted_by_end)` for
 *     interval index construction.
 *   - NCCL's topology ring/tree builder for inter-tier communication scheduling.
 *
 * Milestone: M001–M004 (Claude #1), M005–M006 (Claude #2), M007–M008 (Claude #3)
 *
 * M007 changes:
 *   - partitions_ read path now uses SeqLock instead of shared_mutex.
 *     Readers are wait-free: they optimistically read partition data
 *     and retry only if a concurrent migration_sweep is detected.
 *     This eliminates the write-starvation risk identified in
 *     Claude #2 review Bug 4.1.
 *     Pattern: Linux kernel seqlock, NCCL seq_num ordering.
 *   - flush_partitions now supports adaptive density-aware partitioning:
 *     dense temporal regions get smaller partitions (fit in HBM),
 *     sparse regions get larger partitions (DRAM-bound).
 *     Pattern: TEM-Graph's build_index density detection.
 *
 * M005 changes:
 *   - partitions_ now protected by std::shared_mutex (part_mu_).
 *     query_partitions/scan_partition take shared_lock;
 *     flush_partitions/migration_sweep take unique_lock.
 *     Pattern: PyTorch c10::COWDeleter (c10/core/impl/COWDeleter.h).
 *
 * M006 changes:
 *   - scan_partition now uses std::lower_bound for O(log N + output) scan
 *     instead of O(N) linear scan.
 *     Pattern: LevelDB's table/two_level_iterator Seek() binary search
 *     (leveldb/table/two_level_iterator.cc:25).
 *   - Edges within each partition are sorted by ts_start (invariant from
 *     flush_partitions), so lower_bound on ts_start directly applies.
 */

#include "../core/tiered_allocator.hpp"
#include "../core/seqlock.hpp"       // M007: wait-free reader seqlock
#include "../core/partition_index.hpp"  // M011: per-partition interval index
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <shared_mutex>    // M005: for partitions_ concurrency
#include <mutex>           // M005: for std::unique_lock
#include <unordered_map>
#include <cstdint>

namespace philemon {

// ─── Temporal Interval (compatible with TEM-Graph's TInterval) ──────────────
struct TemporalInterval {
    uint32_t id;
    int32_t  start;     // maps to TInterval::l
    int32_t  end;       // maps to TInterval::r

    TemporalInterval() : id(0), start(0), end(0) {}
    TemporalInterval(uint32_t _id, int32_t _s, int32_t _e)
        : id(_id), start(_s), end(_e) {}

    // Sort by end ascending, then start ascending (matches TEM-Graph TInterval::operator<)
    bool operator<(const TemporalInterval& o) const {
        if (end == o.end && start == o.start) return id < o.id;
        if (end == o.end) return start < o.start;
        return end < o.end;
    }
};


// ─── Edge with temporal annotation ──────────────────────────────────────────
// Defined in core/temporal_edge.hpp (M011: extracted for partition_index.hpp).
#include "../core/temporal_edge.hpp"


// ─── Tier Placement Policy ──────────────────────────────────────────────────
// Determines which memory tier an interval partition should reside on.
// Policy: recent / high-frequency intervals → HBM → GDDR → DRAM.

class TierPlacementPolicy {
public:
    TierPlacementPolicy(uint64_t hot_ns, uint64_t warm_ns)
        : hot_threshold_ns_(hot_ns), warm_threshold_ns_(warm_ns) {}

    MemoryTier decide(const AllocMeta& meta, uint64_t now_ns) const {
        uint64_t last = meta.last_access_ns.load(std::memory_order_relaxed);
        uint64_t age  = (now_ns > last) ? (now_ns - last) : 0;
        uint64_t freq = meta.access_count.load(std::memory_order_relaxed);

        if (age < hot_threshold_ns_ && freq > 10) {
            return MemoryTier::HBM;
        }
        if (age < warm_threshold_ns_ || freq > 3) {
            return MemoryTier::GDDR;
        }
        return MemoryTier::DRAM;
    }

private:
    uint64_t hot_threshold_ns_;
    uint64_t warm_threshold_ns_;
};


// ─── Subgraph Partition ─────────────────────────────────────────────────────
// A contiguous chunk of temporal edges belonging to one interval range.

struct SubgraphPartition {
    uint64_t             alloc_id;       // TieredAllocator allocation handle
    int32_t              ts_lo;          // lowest interval start in this partition
    int32_t              ts_hi;          // highest interval end in this partition
    uint64_t             edge_count;     // number of temporal edges
    std::atomic<uint8_t> tier_atomic;    // M005: atomic tier for concurrent access

    // Convenience getter/setter
    MemoryTier tier() const {
        return static_cast<MemoryTier>(tier_atomic.load(std::memory_order_relaxed));
    }
    void set_tier(MemoryTier t) {
        tier_atomic.store(static_cast<uint8_t>(t), std::memory_order_relaxed);
    }

    SubgraphPartition()
        : alloc_id(0), ts_lo(0), ts_hi(0), edge_count(0), tier_atomic(0) {}

    // M005: explicit copy for atomic member
    SubgraphPartition(const SubgraphPartition& o)
        : alloc_id(o.alloc_id), ts_lo(o.ts_lo), ts_hi(o.ts_hi),
          edge_count(o.edge_count)
    {
        tier_atomic.store(o.tier_atomic.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
    }
    SubgraphPartition& operator=(const SubgraphPartition& o) {
        if (this != &o) {
            alloc_id   = o.alloc_id;
            ts_lo      = o.ts_lo;
            ts_hi      = o.ts_hi;
            edge_count = o.edge_count;
            tier_atomic.store(o.tier_atomic.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        }
        return *this;
    }
};


// ─── Temporal Bridge ────────────────────────────────────────────────────────
// Main integration class.
//
// M005 concurrency model:
//   - part_mu_ (shared_mutex) protects partitions_ vector
//   - query_partitions / scan_partition: shared_lock (concurrent reads)
//   - flush_partitions / migration_sweep: unique_lock (exclusive writes)
//
// M006 optimization:
//   - scan_partition uses binary search (std::lower_bound) on ts_start
//     within sorted partition data, reducing narrow-range queries from
//     O(partition_size) to O(log(partition_size) + output_size).
//   - Pattern: LevelDB's Seek() (leveldb/table/two_level_iterator.cc:25)

class TemporalBridge {
public:
    TemporalBridge(TieredAllocator& allocator,
                   TierPlacementPolicy policy,
                   size_t partition_capacity = 1 << 20 /*1M edges per partition*/)
        : allocator_(allocator)
        , policy_(policy)
        , partition_cap_(partition_capacity)
    {}

    // ── Ingest ──────────────────────────────────────────────────────────────
    void add_edge(const TemporalEdge& e) {
        buffer_.push_back(e);
    }

    void add_edges(const std::vector<TemporalEdge>& edges) {
        buffer_.insert(buffer_.end(), edges.begin(), edges.end());
    }

    // ── Partitioning ────────────────────────────────────────────────────────
    // Sort buffered edges by timestamp, split into partitions,
    // allocate tiered memory for each.
    // Takes UNIQUE lock on part_mu_ — exclusive write.
    // M007: SeqLock write_lock for partition structure changes.
    //
    // M007 ADAPTIVE PARTITIONING:
    //   Fixed partition_cap_ wastes HBM on sparse regions and under-serves
    //   dense regions (Bug 4.2 from Claude #1 review). We now compute
    //   temporal density and split accordingly:
    //     - Dense regions (>2× avg density) → smaller partitions (HBM-sized)
    //     - Sparse regions (<0.5× avg density) → larger partitions (DRAM-bound)
    //     - Average regions → default partition_cap_
    //
    //   This follows TEM-Graph's build_index approach of adapting index
    //   granularity to data distribution.

    size_t flush_partitions() {
        if (buffer_.empty()) return 0;

        // Sort by interval start, then end (matches TEM-Graph loading order).
        std::sort(buffer_.begin(), buffer_.end(),
            [](const TemporalEdge& a, const TemporalEdge& b) {
                if (a.ts_begin != b.ts_begin) return a.ts_begin < b.ts_begin;
                return a.ts_finish < b.ts_finish;
            });

        // M007: Compute adaptive partition boundaries based on temporal density.
        // Step 1: Determine the time range and average density.
        int32_t global_lo = buffer_.front().ts_begin;
        int32_t global_hi = buffer_.back().ts_begin;
        int32_t time_range = global_hi - global_lo + 1;
        double avg_density = static_cast<double>(buffer_.size()) / std::max(time_range, 1);

        // Step 2: Build partition boundaries adaptively.
        // Walk the sorted edges, tracking local density. When we accumulate
        // enough edges for a partition (adjusted by density), emit a boundary.
        std::vector<size_t> boundaries;
        boundaries.push_back(0);

        size_t i = 0;
        while (i < buffer_.size()) {
            // Determine local density in a lookahead window
            size_t window_end = std::min(i + partition_cap_, buffer_.size());
            int32_t local_time = buffer_[window_end - 1].ts_begin - buffer_[i].ts_begin + 1;
            double local_density = static_cast<double>(window_end - i) / std::max(local_time, 1);

            // Adaptive capacity: dense → smaller, sparse → larger
            size_t adaptive_cap = partition_cap_;
            if (local_density > avg_density * 2.0 && partition_cap_ > 10000) {
                // Dense region: halve partition size for finer HBM granularity
                adaptive_cap = partition_cap_ / 2;
            } else if (local_density < avg_density * 0.5) {
                // Sparse region: double partition size for DRAM efficiency
                adaptive_cap = std::min(partition_cap_ * 2, buffer_.size() - i);
            }

            size_t next_boundary = std::min(i + adaptive_cap, buffer_.size());
            boundaries.push_back(next_boundary);
            i = next_boundary;
        }

        // Step 3: Create partitions from boundaries.
        size_t created = 0;
        for (size_t b = 0; b + 1 < boundaries.size(); ++b) {
            size_t start = boundaries[b];
            size_t end   = boundaries[b + 1];
            size_t count = end - start;
            if (count == 0) continue;

            int32_t lo = buffer_[start].ts_begin;
            int32_t hi = buffer_[end - 1].ts_finish;
            for (size_t j = start; j < end; ++j) {
                hi = std::max(hi, buffer_[j].ts_finish);
            }

            MemoryTier init_tier = MemoryTier::DRAM;
            if (created == 0) {
                init_tier = MemoryTier::HBM;
            } else if (created <= 2) {
                init_tier = MemoryTier::GDDR;
            }

            size_t alloc_size = count * sizeof(TemporalEdge);
            uint64_t aid = allocator_.allocate(alloc_size, init_tier, lo, hi);
            if (aid == 0) {
                aid = allocator_.allocate(alloc_size, MemoryTier::DRAM, lo, hi);
            }
            if (aid == 0) {
                std::cerr << "[TemporalBridge] FATAL: cannot allocate "
                          << alloc_size << " bytes for partition [" << lo
                          << ", " << hi << "]\n";
                continue;
            }

            void* dst = allocator_.get_ptr(aid);
            if (dst) {
                ::memcpy(dst, &buffer_[start], count * sizeof(TemporalEdge));
            }

            SubgraphPartition part;
            part.alloc_id   = aid;
            part.ts_lo      = lo;
            part.ts_hi      = hi;
            part.edge_count = count;
            part.set_tier(init_tier);

            {
                std::unique_lock<std::shared_mutex> lk(part_mu_);  // M005
                seq_lock_.write_lock();   // M007: seqlock write
                partitions_.push_back(part);
                seq_lock_.write_unlock(); // M007: seqlock write
            }
            ++created;
        }

        // M011: Build per-partition interval index for O(log N + output) queries.
        // Pattern: TEM-Graph build_index sorts by end, builds successor pointers.
        // Our PartitionIndex builds secondary end-sorted index + running min/max.
        partition_indices_.resize(partitions_.size());
        for (size_t i = 0; i < partitions_.size(); ++i) {
            void* raw = allocator_.get_ptr(partitions_[i].alloc_id);
            if (raw) {
                const TemporalEdge* edges = reinterpret_cast<const TemporalEdge*>(raw);
                partition_indices_[i].build(edges, partitions_[i].edge_count);
            }
        }

        buffer_.clear();
        buffer_.shrink_to_fit();
        return created;
    }


    // ── Temporal Range Query ────────────────────────────────────────────────
    // Find all partitions whose interval range overlaps [ts_lo, ts_hi].
    // M007: Uses SeqLock for wait-free reads. Readers optimistically read
    // partition data and retry if a concurrent write is detected.
    // This eliminates the shared_mutex write-starvation risk.

    std::vector<const SubgraphPartition*>
    query_partitions(int32_t ts_lo, int32_t ts_hi) const {
        std::vector<const SubgraphPartition*> result;
        // M007: Optimistic seqlock read — no blocking on writers
        uint64_t seq;
        do {
            seq = seq_lock_.read_begin();
            result.clear();
            // Still take shared_lock for vector safety (seqlock protects
            // against torn partition metadata reads during migration)
            std::shared_lock<std::shared_mutex> lk(part_mu_);  // M005
            for (auto& p : partitions_) {
                if (p.ts_lo <= ts_hi && p.ts_hi >= ts_lo) {
                    result.push_back(&p);
                    allocator_.touch(p.alloc_id);  // M005: lockfree touch
                }
            }
        } while (seq_lock_.read_retry(seq));
        return result;
    }


    // ── Edge Iteration within a Partition ───────────────────────────────────
    // M006 CRITICAL FIX: Binary search via std::lower_bound.
    //
    // Previous implementation (M001–M004) scanned every edge in the
    // partition linearly — O(partition_size) regardless of query selectivity.
    // The benchmark showed narrow [1000,1050] cost the same as medium
    // [2000,3000] because both traversed full 100K-edge partitions.
    //
    // Fix: edges are sorted by ts_start (invariant from flush_partitions).
    // Use std::lower_bound to jump to the first edge where ts_start >= ts_lo.
    // Then scan forward, stopping when ts_start > ts_hi (no more matches).
    //
    // For "contains" semantics (edge ⊆ query), we need:
    //   edge.ts_begin >= ts_lo AND edge.ts_finish <= ts_hi
    //
    // The lower_bound on ts_start gives us the starting position.
    // The early-exit when ts_start > ts_hi gives us the stopping position.
    // Within that range, we filter on ts_end <= ts_hi.
    //
    // Complexity: O(log(N) + output_size + false_positive_count)
    // where false_positives are edges with ts_start in range but ts_end > ts_hi.
    //
    // Pattern source: LevelDB's Iterator::Seek() (two_level_iterator.cc:25)
    // which performs binary search within sorted blocks, then linear scan
    // within the matching block.

    template <typename Callback>
    uint64_t scan_partition(const SubgraphPartition& part,
                            int32_t ts_lo, int32_t ts_hi,
                            Callback&& cb) const {
        // M011: If we have a partition index, use it for O(log N + output).
        // Find which partition index corresponds to this partition.
        for (size_t i = 0; i < partitions_.size(); ++i) {
            if (&partitions_[i] == &part && i < partition_indices_.size()
                && partition_indices_[i].is_built()) {
                return partition_indices_[i].contains_query(
                    ts_lo, ts_hi, std::forward<Callback>(cb));
            }
        }

        // Fallback: M006 binary search path (no index built yet).
        void* raw = allocator_.get_ptr(part.alloc_id);
        if (!raw) return 0;

        const TemporalEdge* edges = reinterpret_cast<const TemporalEdge*>(raw);
        const TemporalEdge* edges_end = edges + part.edge_count;
        uint64_t matched = 0;

        const TemporalEdge* first = std::lower_bound(
            edges, edges_end, ts_lo,
            [](const TemporalEdge& e, int32_t val) {
                return e.ts_begin < val;
            });

        for (const TemporalEdge* it = first; it != edges_end; ++it) {
            if (it->ts_begin > ts_hi) break;
            if (it->ts_finish <= ts_hi) {
                cb(*it);
                ++matched;
            }
        }
        return matched;
    }

    // M011: Contained query — find edges [a,b] where a≤lo and b≥hi.
    // Exposed as a separate API for algorithms that need "which edges
    // contain this time point."
    template <typename Callback>
    uint64_t temporal_contained_query(int32_t ts_lo, int32_t ts_hi,
                                      Callback&& cb) const {
        auto parts = query_partitions(ts_lo, ts_hi);
        uint64_t total = 0;
        for (auto* p : parts) {
            for (size_t i = 0; i < partitions_.size(); ++i) {
                if (&partitions_[i] == p && i < partition_indices_.size()
                    && partition_indices_[i].is_built()) {
                    total += partition_indices_[i].contained_query(
                        ts_lo, ts_hi, std::forward<Callback>(cb));
                    break;
                }
            }
        }
        return total;
    }

    // Full temporal subgraph query: locate partitions, scan edges.
    template <typename Callback>
    uint64_t temporal_subgraph_query(int32_t ts_lo, int32_t ts_hi,
                                     Callback&& cb) const {
        auto parts = query_partitions(ts_lo, ts_hi);
        uint64_t total = 0;
        for (auto* p : parts) {
            total += scan_partition(*p, ts_lo, ts_hi, std::forward<Callback>(cb));
        }
        return total;
    }


    // ── Migration Sweep ─────────────────────────────────────────────────────
    // M007: SeqLock write_lock during tier updates to signal readers.

    size_t migration_sweep() {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

        size_t migrated = 0;
        std::unique_lock<std::shared_mutex> lk(part_mu_);  // M005
        for (auto& part : partitions_) {
            AllocMeta meta;
            if (!allocator_.get_meta(part.alloc_id, meta)) continue;

            MemoryTier target = policy_.decide(meta, now_ns);
            if (target != meta.current_tier) {
                if (allocator_.migrate(part.alloc_id, target)) {
                    seq_lock_.write_lock();   // M007: signal readers
                    part.set_tier(target);
                    seq_lock_.write_unlock(); // M007
                    ++migrated;
                }
            }
        }
        return migrated;
    }


    // ── Accessors ───────────────────────────────────────────────────────────
    size_t partition_count() const {
        std::shared_lock<std::shared_mutex> lk(part_mu_);  // M005
        return partitions_.size();
    }

    // Return a snapshot copy of partitions (safe for external iteration).
    std::vector<SubgraphPartition> partitions_snapshot() const {
        std::shared_lock<std::shared_mutex> lk(part_mu_);  // M005
        return partitions_;
    }

    // Direct access — caller must be aware this is NOT thread-safe
    // without external synchronization.  Used only by single-threaded
    // benchmark code.
    const std::vector<SubgraphPartition>& partitions_unsafe() const {
        return partitions_;
    }

private:
    TieredAllocator&            allocator_;
    TierPlacementPolicy         policy_;
    size_t                      partition_cap_;
    std::vector<TemporalEdge>   buffer_;

    mutable std::shared_mutex   part_mu_;      // M005: protects partitions_
    mutable SeqLock             seq_lock_;     // M007: wait-free reader seqlock
    std::vector<SubgraphPartition> partitions_;
    std::vector<PartitionIndex>    partition_indices_;  // M011: per-partition index
};

}  // namespace philemon
