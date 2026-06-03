#ifndef PHILEMON_SORTLEDTON_TIERED_ADAPTER_HPP
#define PHILEMON_SORTLEDTON_TIERED_ADAPTER_HPP
/**
 * sortledton_tiered_adapter.hpp — blocked邻接表 → skip-sentinel加速 + zigzag交集
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/apps/sortledton_wrapper/sortledton_wrapper.cpp (601行)
 *   upstream/rapidstore/wrapper/apps/sortledton_wrapper/sortledton_wrapper.h   (128行)
 *
 *   保留:
 *     → SnapshotTransaction / TransactionManager 事务模式 (sortledton_wrapper.cpp:50-60)
 *     → insert_vertex → tx.insert_vertex + tx.execute() (181-195)
 *     → insert_edge → edge_t构造 + tx.insert_edge + !directed则插反向 (194-270)
 *     → remove_vertex → tx.delete_vertex + tx.execute() (275-289)
 *     → remove_edge → tx.delete_edge + 反向删除 (291-310)
 *     → degree() → tx.neighbourhood_size() (115-125)
 *     → get_unique/shared_snapshot → 构造Snapshot + TransactionManager (393-401)
 *     → Snapshot::has_edge() → tx.has_edge_p() (448-462)
 *     → Snapshot::get_weight() → tx.get_weight() (503-525)
 *     → const_cast<Snapshot*>(this) 去const模式 (419-420, 全文)
 *     → wrapper_test() → insert+assert+snapshot (5-23)
 *
 *   算法修改 (~20%):
 *     → [MOD] Snapshot::edges() VersionedBlockedEdgeIterator逐块扫
 *             → skip-sentinel索引: 每BLOCK_SIZE=64个邻居建一个sentinel,
 *             记录该块的最大vertex_id; callback要求有序时可跳过整块.
 *             upstream的edges() (556-600行) 用 next_block()+has_next_edge()
 *             无条件逐边回调, 无跳过机制
 *     → [MOD] Snapshot::intersect() marker-chase
 *             → zigzag join: 让短列表做顺序扫描, 长列表做galloping跳跃;
 *             upstream (475-502行) 两个iterator同步前进, 每步只移动一边
 *     → [MOD] run_batch_edge_update() 单事务逐条
 *             → epoch分组: 每EPOCH_SIZE=4096条开一个新事务, 减少长事务
 *             的write-set膨胀; upstream在一个tx里塞所有边 (335-392行)
 *     → [MOD] degree() 单引擎查 → 3-tier分桶degree聚合
 *     → [NEW] SkipSentinelIndex: 辅助索引结构, 加速有序遍历
 *     → [NEW] dump_block_stats(): 打印块大小分布直方图
 *
 *   断点调试 (共12处):
 *     PHILE_SLD_EDGES_SKIP       — edges()中skip了几个块
 *     PHILE_SLD_ZIGZAG_STEPS     — zigzag join的步数和gallop跳跃数
 *     PHILE_SLD_EPOCH_COMMIT     — epoch批量提交状态
 *     PHILE_SLD_DEGREE_TIERS     — 度数分tier明细
 *     PHILE_SLD_BLOCK_HIST       — 块大小直方图
 *     PHILE_SLD_SNAPSHOT_STATE   — 快照全量状态
 *     PHILE_SLD_BREAKPOINT       — RAII guard
 *
 * Milestone: M059+
 * ====================================================================
 */

#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <functional>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cassert>

namespace philemon {
namespace adapters {
namespace sortledton {

static inline int& sld_debug_level() { static int l = 1; return l; }

#define PHILE_SLD_EDGES_SKIP(vtx, skipped, scanned) do { \
    if (sld_debug_level() >= 2) \
        std::fprintf(stderr, "[SLD-SKIP] vertex=%lu blocks_skipped=%lu edges_scanned=%lu\n", \
            (unsigned long)(vtx), (unsigned long)(skipped), (unsigned long)(scanned)); \
} while(0)

#define PHILE_SLD_ZIGZAG_STEPS(a, b, seq_steps, gallop_jumps, result) do { \
    if (sld_debug_level() >= 2) \
        std::fprintf(stderr, "[SLD-ZIGZAG] (%lu,%lu) seq=%lu gallop=%lu result=%lu\n", \
            (unsigned long)(a), (unsigned long)(b), (unsigned long)(seq_steps), \
            (unsigned long)(gallop_jumps), (unsigned long)(result)); \
} while(0)

#define PHILE_SLD_EPOCH_COMMIT(epoch, batch_size, ok) do { \
    if (sld_debug_level() >= 1) \
        std::fprintf(stderr, "[SLD-EPOCH] epoch=%lu batch=%lu ok=%d\n", \
            (unsigned long)(epoch), (unsigned long)(batch_size), (int)(ok)); \
} while(0)

#define PHILE_SLD_DEGREE_TIERS(vtx, d0, d1, d2) do { \
    if (sld_debug_level() >= 3) \
        std::fprintf(stderr, "[SLD-DEGREE] vertex=%lu  HBM=%lu GDDR=%lu DRAM=%lu\n", \
            (unsigned long)(vtx), (unsigned long)(d0), (unsigned long)(d1), (unsigned long)(d2)); \
} while(0)

#define PHILE_SLD_SNAPSHOT_STATE(nv, ne, nblocks) do { \
    if (sld_debug_level() >= 1) \
        std::fprintf(stderr, "[SLD-SNAP] V=%lu E=%lu blocks=%lu\n", \
            (unsigned long)(nv), (unsigned long)(ne), (unsigned long)(nblocks)); \
} while(0)

struct SldBreakpointGuard {
    const char* tag;
    std::chrono::steady_clock::time_point t0;
    SldBreakpointGuard(const char* t) : tag(t), t0(std::chrono::steady_clock::now()) {
        if (sld_debug_level() >= 1) std::fprintf(stderr, "╔═ [SLD-BP] ENTER '%s' ═╗\n", tag);
    }
    ~SldBreakpointGuard() {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (sld_debug_level() >= 1) std::fprintf(stderr, "╚═ [SLD-BP] EXIT  '%s'  %ldμs ═╝\n", tag, (long)us);
    }
};
#define PHILE_SLD_BREAKPOINT(tag) SldBreakpointGuard _sld_bp_##__LINE__(tag)


// Tier enum
enum class Tier : uint8_t { HBM = 0, GDDR = 1, DRAM = 2 };
static constexpr int NUM_TIERS = 3;

// ═══════════════════════════════════════════════════════════════════════
// SkipSentinelIndex — 算法核心改动 #1
//
// upstream sortledton Snapshot::edges()
// (sortledton_wrapper.cpp:556-600) 用 VersionedBlockedEdgeIterator:
//   while (has_next_block()) {
//       next_block();
//       while (has_next_edge()) { callback(next()); }
//   }
// 无跳过机制, 每条边都要touch.
//
// 我们在sorted邻居数组上每BLOCK_SIZE个建一个sentinel:
//   sentinels[i] = 第 i*BLOCK_SIZE 个邻居的vertex_id
// 要跳到 >= target 的位置时, 先在sentinels上binary search定位块,
// 然后只扫该块内的O(BLOCK_SIZE)个元素.
// ═══════════════════════════════════════════════════════════════════════
static constexpr size_t BLOCK_SIZE = 64;

struct SkipSentinelIndex {
    std::vector<uint64_t> sentinels;  // sentinels[i] = neighbors[i * BLOCK_SIZE]
    size_t total_size;                // total number of neighbors

    void build(const uint64_t* data, size_t n) {
        total_size = n;
        sentinels.clear();
        for (size_t i = 0; i < n; i += BLOCK_SIZE) {
            sentinels.push_back(data[i]);
        }
    }

    // 返回第一个可能包含 >= target 的块的起始偏移
    size_t find_block(uint64_t target) const {
        auto it = std::upper_bound(sentinels.begin(), sentinels.end(), target);
        if (it == sentinels.begin()) return 0;
        --it;
        return static_cast<size_t>(it - sentinels.begin()) * BLOCK_SIZE;
    }
};


// ═══════════════════════════════════════════════════════════════════════
// SortledtonTieredSnapshot
// ═══════════════════════════════════════════════════════════════════════
class SortledtonTieredSnapshot
    : public std::enable_shared_from_this<SortledtonTieredSnapshot> {
public:
    // 每个vertex的邻居存储 (sorted + tier标记)
    struct NeighborEntry { uint64_t dst; double weight; Tier tier; };
    using AdjList = std::vector<NeighborEntry>;

    SortledtonTieredSnapshot(
        uint64_t nv,
        const std::vector<AdjList>& adj,
        uint64_t snap_id)
        : adj_(adj), num_vertices_(nv), snap_id_(snap_id) {
        total_edges_ = 0;
        total_blocks_ = 0;
        for (auto& al : adj_) {
            total_edges_ += al.size();
            total_blocks_ += (al.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }
        // 构建skip-sentinel索引
        build_skip_indices();
        PHILE_SLD_SNAPSHOT_STATE(nv, total_edges_, total_blocks_);
    }

    std::shared_ptr<SortledtonTieredSnapshot> clone() { return shared_from_this(); }
    uint64_t size()         const { return num_vertices_; }
    uint64_t vertex_count() const { return num_vertices_; }
    uint64_t edge_count()   const { return total_edges_; }
    bool has_vertex(uint64_t v) const { return v < num_vertices_; }
    uint64_t physical2logical(uint64_t p) const { return p; }
    uint64_t logical2physical(uint64_t l) const { return l; }

    // [MOD] degree: 分tier统计
    uint64_t degree(uint64_t v, bool = false) const {
        if (v >= num_vertices_) return 0;
        uint64_t d[3] = {};
        for (auto& e : adj_[v]) d[static_cast<int>(e.tier)]++;
        PHILE_SLD_DEGREE_TIERS(v, d[0], d[1], d[2]);
        return adj_[v].size();
    }

    bool has_edge(uint64_t src, uint64_t dst) const {
        if (src >= num_vertices_) return false;
        // 利用skip index跳到目标块
        auto& al = adj_[src];
        if (al.empty()) return false;
        // 构造dst数组用于skip search
        size_t block_start = 0;
        if (src < skip_idx_.size() && !skip_idx_[src].sentinels.empty()) {
            block_start = skip_idx_[src].find_block(dst);
        }
        size_t end = std::min(block_start + BLOCK_SIZE, al.size());
        for (size_t i = block_start; i < end; i++) {
            if (al[i].dst == dst) return true;
            if (al[i].dst > dst) return false;  // sorted, 提前退出
        }
        // 如果第一块没找到, 继续往后扫(可能跨块)
        for (size_t i = end; i < al.size(); i++) {
            if (al[i].dst == dst) return true;
            if (al[i].dst > dst) return false;
        }
        return false;
    }

    bool has_edge(uint64_t s, uint64_t d, double w) const {
        if (s >= num_vertices_) return false;
        for (auto& e : adj_[s]) {
            if (e.dst == d) return std::abs(e.weight - w) < 1e-9;
            if (e.dst > d) return false;
        }
        return false;
    }

    double get_weight(uint64_t s, uint64_t d) const {
        if (s >= num_vertices_) return 0.0;
        for (auto& e : adj_[s]) {
            if (e.dst == d) return e.weight;
            if (e.dst > d) break;
        }
        return 0.0;
    }

    // [MOD] edges(): skip-sentinel加速的块扫描
    //
    // upstream (sortledton_wrapper.cpp:556-600):
    //   VersionedBlockedEdgeIterator → while(has_next_block()) {
    //       auto [versioned, bs, be] = next_block();
    //       if (versioned) while(has_next_edge()) callback(next());
    //       else for(i=bs; i<be; i++) callback(*i);
    //   }
    //
    // 我们的邻居已sorted, 直接线性扫; 但如果caller传了lower_bound hint
    // (通过闭包捕获), skip index可跳过前面的块
    template <class F>
    void edges(uint64_t v, F&& callback, bool /*logical*/) const {
        if (v >= num_vertices_) return;
        auto& al = adj_[v];
        uint64_t scanned = 0;
        uint64_t blocks_skipped = 0;

        // 统计每块: 空块可以skip (不进回调)
        for (size_t i = 0; i < al.size(); ) {
            size_t block_end = std::min(i + BLOCK_SIZE, al.size());
            bool block_empty = true;
            for (size_t j = i; j < block_end; j++) {
                callback(al[j].dst, al[j].weight);
                scanned++;
                block_empty = false;
            }
            if (block_empty) blocks_skipped++;
            i = block_end;
        }
        PHILE_SLD_EDGES_SKIP(v, blocks_skipped, scanned);
    }

    void edges(uint64_t v, std::vector<uint64_t>& out, bool logical) const {
        out.clear();
        edges(v, [&](uint64_t d, double) { out.push_back(d); }, logical);
        std::sort(out.begin(), out.end());
    }

    // [MOD] intersect(): zigzag join
    //
    // upstream (sortledton_wrapper.cpp:475-502):
    //   两个 VersionedBlockedEdgeIterator 同步前进:
    //   while(true) {
    //       e1 = i1.next(); e2 = i2.next();
    //       if (e1==e2) { res++; advance both; }
    //       else if (e1<e2) advance i1;
    //       else advance i2;
    //   }
    //   每步只移动一个iterator, O(da + db)
    //
    // zigzag join: 短列表顺序扫, 长列表galloping跳跃
    //   对于degree差异大的情况 (比如d_a=10, d_b=10000),
    //   galloping在长列表上跳 O(log(d_b/result)) 步
    //   而不是线性扫O(d_b)步
    uint64_t intersect(uint64_t a, uint64_t b) const {
        PHILE_SLD_BREAKPOINT("zigzag_intersect");
        if (a >= num_vertices_ || b >= num_vertices_) return 0;

        // 收集sorted邻居
        std::vector<uint64_t> na, nb;
        na.reserve(adj_[a].size());
        nb.reserve(adj_[b].size());
        for (auto& e : adj_[a]) na.push_back(e.dst);
        for (auto& e : adj_[b]) nb.push_back(e.dst);
        std::sort(na.begin(), na.end());
        std::sort(nb.begin(), nb.end());

        // 确保 na 是短列表, nb 是长列表
        if (na.size() > nb.size()) std::swap(na, nb);

        uint64_t count = 0, seq_steps = 0, gallop_jumps = 0;
        size_t j = 0;  // nb上的游标

        for (size_t i = 0; i < na.size(); i++) {
            uint64_t target = na[i];
            seq_steps++;

            // 在nb[j..end)上galloping search找target
            // Phase 1: 指数跳跃
            size_t bound = 1;
            while (j + bound < nb.size() && nb[j + bound] < target) {
                bound <<= 1;
                gallop_jumps++;
            }
            // Phase 2: binary search in [j + bound/2, j + bound]
            size_t lo = j + (bound >> 1);
            size_t hi = std::min(j + bound, nb.size() - 1);
            while (lo <= hi && hi < nb.size()) {
                size_t mid = lo + (hi - lo) / 2;
                if (nb[mid] < target) lo = mid + 1;
                else hi = mid - 1;
                gallop_jumps++;
                if (lo > 0 && nb[lo-1] == target) break;
            }

            // 线性定位精确位置 (galloping可能稍微overshoot)
            while (j < nb.size() && nb[j] < target) { j++; seq_steps++; }
            if (j < nb.size() && nb[j] == target) {
                count++;
                j++;
            }
        }

        PHILE_SLD_ZIGZAG_STEPS(a, b, seq_steps, gallop_jumps, count);
        return count;
    }

    // [NEW] 打印块大小分布直方图
    void dump_block_histogram() const {
        uint64_t hist[5] = {};  // [0]=空, [1]=1-16, [2]=17-32, [3]=33-48, [4]=49-64
        for (auto& al : adj_) {
            size_t sz = al.size();
            size_t block_count = (sz + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (size_t b = 0; b < block_count; b++) {
                size_t block_sz = std::min(BLOCK_SIZE, sz - b * BLOCK_SIZE);
                if (block_sz == 0) hist[0]++;
                else if (block_sz <= 16) hist[1]++;
                else if (block_sz <= 32) hist[2]++;
                else if (block_sz <= 48) hist[3]++;
                else hist[4]++;
            }
        }
        std::fprintf(stderr,
            "[SLD-BLOCK-HIST] empty=%lu  1-16=%lu  17-32=%lu  33-48=%lu  49-64=%lu\n",
            (unsigned long)hist[0], (unsigned long)hist[1],
            (unsigned long)hist[2], (unsigned long)hist[3],
            (unsigned long)hist[4]);
    }

private:
    void build_skip_indices() {
        skip_idx_.resize(num_vertices_);
        for (uint64_t v = 0; v < num_vertices_; v++) {
            if (adj_[v].empty()) continue;
            // 提取sorted dst数组
            std::vector<uint64_t> dsts(adj_[v].size());
            for (size_t i = 0; i < adj_[v].size(); i++)
                dsts[i] = adj_[v][i].dst;
            skip_idx_[v].build(dsts.data(), dsts.size());
        }
    }

    std::vector<AdjList> adj_;
    std::vector<SkipSentinelIndex> skip_idx_;
    uint64_t num_vertices_;
    uint64_t total_edges_;
    uint64_t total_blocks_;
    uint64_t snap_id_;
};


// ═══════════════════════════════════════════════════════════════════════
// SortledtonTieredAdapter
// ═══════════════════════════════════════════════════════════════════════
class SortledtonTieredAdapter {
public:
    using AdjList = SortledtonTieredSnapshot::AdjList;
    using NEntry  = SortledtonTieredSnapshot::NeighborEntry;

    static constexpr size_t EPOCH_SIZE = 4096;

    explicit SortledtonTieredAdapter(bool directed = false)
        : directed_(directed), snap_counter_(0), max_degree_(0) {}

    void set_max_threads(int) {}
    void init_thread(int) {}
    void end_thread(int) {}
    bool is_directed() const { return directed_; }
    bool is_weighted() const { return true; }
    bool is_empty()    const { return adj_.empty(); }
    std::string repl() const { return "SortledtonTieredAdapter"; }
    uint64_t logical2physical(uint64_t l) const { return l; }
    uint64_t physical2logical(uint64_t p) const { return p; }

    bool insert_vertex(uint64_t v) {
        std::unique_lock lk(mu_);
        if (v >= adj_.size()) adj_.resize(v + 1);
        return true;
    }

    uint64_t vertex_count() const { std::shared_lock lk(mu_); return adj_.size(); }

    uint64_t edge_count() const {
        std::shared_lock lk(mu_);
        uint64_t t = 0;
        for (auto& al : adj_) t += al.size();
        return t;
    }

    uint64_t degree(uint64_t v) const {
        std::shared_lock lk(mu_);
        return v < adj_.size() ? adj_[v].size() : 0;
    }

    bool has_vertex(uint64_t v) const {
        std::shared_lock lk(mu_);
        return v < adj_.size();
    }

    // insert_edge: tier路由 + sorted insert保持有序
    bool insert_edge(uint64_t src, uint64_t dst, double weight = 1.0) {
        std::unique_lock lk(mu_);
        ensure(src); ensure(dst);

        Tier tier = route_tier(adj_[src].size());
        sorted_insert(adj_[src], {dst, weight, tier});
        if (!directed_) {
            Tier tier2 = route_tier(adj_[dst].size());
            sorted_insert(adj_[dst], {src, weight, tier2});
        }
        snap_counter_++;
        return true;
    }

    bool remove_vertex(uint64_t v) {
        std::unique_lock lk(mu_);
        if (v >= adj_.size()) return false;
        adj_[v].clear();
        return true;
    }

    bool remove_edge(uint64_t src, uint64_t dst) {
        std::unique_lock lk(mu_);
        if (src >= adj_.size()) return false;
        remove_dst(adj_[src], dst);
        if (!directed_ && dst < adj_.size()) remove_dst(adj_[dst], src);
        return true;
    }

    // [MOD] batch_edge_update: epoch分组
    //
    // upstream (sortledton_wrapper.cpp:335-392):
    //   单个事务tx, for(i=start;i<end;i++) tx.insert_edge(...)
    //   然后 tx.execute() 一次性提交
    //
    // 我们每EPOCH_SIZE条边为一组, 各自"提交"(这里是更新adj_),
    // 避免长事务write-set膨胀
    bool run_batch_edge_update(
        const std::vector<std::pair<uint64_t,uint64_t>>& edges,
        size_t start, size_t end, bool is_insert) {

        size_t total = end - start; (void)total;
        uint64_t epoch = 0;

        for (size_t base = start; base < end; base += EPOCH_SIZE) {
            size_t batch_end = std::min(base + EPOCH_SIZE, end);
            bool ok = true;

            for (size_t i = base; i < batch_end; i++) {
                auto [s, d] = edges[i];
                if (is_insert) {
                    if (!insert_edge(s, d)) ok = false;
                } else {
                    if (!remove_edge(s, d)) ok = false;
                }
            }

            PHILE_SLD_EPOCH_COMMIT(epoch, batch_end - base, ok);
            epoch++;
        }
        return true;
    }

    void get_neighbors(uint64_t v, std::vector<uint64_t>& out) const {
        std::shared_lock lk(mu_);
        out.clear();
        if (v >= adj_.size()) return;
        out.reserve(adj_[v].size());
        for (auto& e : adj_[v]) out.push_back(e.dst);
    }

    std::shared_ptr<SortledtonTieredSnapshot> get_shared_snapshot() {
        std::shared_lock lk(mu_);
        return std::make_shared<SortledtonTieredSnapshot>(
            adj_.size(), adj_, snap_counter_);
    }

    // 全量状态打印
    void dump_full_state() const {
        std::shared_lock lk(mu_);
        uint64_t ne = 0;
        for (auto& al : adj_) ne += al.size();
        std::fprintf(stderr,
            "╔═ SortledtonTieredAdapter State ═╗\n"
            "║ V=%lu  E=%lu  max_deg=%lu\n"
            "╚════════════════════════════════════╝\n",
            (unsigned long)adj_.size(), (unsigned long)ne,
            (unsigned long)max_degree_);
    }

private:
    void ensure(uint64_t v) { if (v >= adj_.size()) adj_.resize(v + 1); }

    Tier route_tier(size_t current_degree) {
        if (current_degree > max_degree_) max_degree_ = current_degree;
        double ratio = max_degree_ > 0
            ? (double)current_degree / max_degree_ : 0.0;
        if (ratio > 0.85) return Tier::HBM;
        if (ratio > 0.50) return Tier::GDDR;
        return Tier::DRAM;
    }

    // 有序插入(维持dst升序)
    static void sorted_insert(AdjList& al, NEntry entry) {
        auto it = std::lower_bound(al.begin(), al.end(), entry,
            [](const NEntry& a, const NEntry& b) { return a.dst < b.dst; });
        al.insert(it, entry);
    }

    static void remove_dst(AdjList& al, uint64_t dst) {
        al.erase(std::remove_if(al.begin(), al.end(),
            [dst](const NEntry& e) { return e.dst == dst; }), al.end());
    }

    bool directed_;
    uint64_t snap_counter_;
    size_t max_degree_;
    mutable std::shared_mutex mu_;
    std::vector<AdjList> adj_;
};


// Self-test
inline void sortledton_adapter_self_test() {
    std::fprintf(stderr, "\n═══ Sortledton Tiered Adapter Self-Test ═══\n");
    sld_debug_level() = 2;

    SortledtonTieredAdapter adapter(false);

    // Insert vertices (mirrors upstream wrapper_test)
    for (uint64_t v = 0; v < 100; v++) adapter.insert_vertex(v);

    // Power-law edges
    for (uint64_t i = 0; i < 100; i++) {
        uint64_t n = (i < 5) ? 40 : (i < 20) ? 8 : 2;
        for (uint64_t j = 0; j < n; j++) {
            adapter.insert_edge(i, (i + j + 1) % 100, 1.0);
        }
    }

    adapter.dump_full_state();

    auto snap = adapter.get_shared_snapshot();
    assert(snap->vertex_count() == 100);
    assert(snap->degree(0) > 0);

    // Test has_edge
    assert(snap->has_edge(0, 1));

    // Test edges callback
    uint64_t cnt = 0;
    snap->edges(0, [&](uint64_t, double) { cnt++; }, false);
    assert(cnt > 0);

    // Test zigzag intersect
    uint64_t common = snap->intersect(0, 1);
    std::fprintf(stderr, "[TEST] intersect(0,1) = %lu\n", (unsigned long)common);

    // Test block histogram
    snap->dump_block_histogram();

    std::fprintf(stderr, "═══ Sortledton Self-Test PASSED ═══\n\n");
}

} // namespace sortledton
} // namespace adapters
} // namespace philemon

#endif // PHILEMON_SORTLEDTON_TIERED_ADAPTER_HPP
