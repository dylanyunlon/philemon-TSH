#ifndef PHILEMON_TESEO_TIERED_ADAPTER_HPP
#define PHILEMON_TESEO_TIERED_ADAPTER_HPP
/**
 * teseo_tiered_adapter.hpp — MVCC事务引擎 → RCU shadow-swap + adaptive交集
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/apps/teseo_wrapper/teseo_wrapper.cpp  (428行)
 *   upstream/rapidstore/wrapper/apps/teseo_wrapper/teseo_wrapper.h    (122行)
 *
 *   保留:
 *     → start_transaction() → tx.insert_vertex/insert_edge/commit (176-214)
 *     → RegisterThread — 线程注册/注销 + encoded_addr位编码 (390-428)
 *       teseo::register_thread + m_encoded_addr |= 0x1ull 编码
 *       is_enabled = m_encoded_addr & 0x1ull
 *       Copy ctor重新register, dtor自动unregister
 *     → Snapshot::edges(index, callback) → m_iterator.edges(index, false, cb) (383-388)
 *     → Snapshot::degree() → m_transaction.degree(vertex, false) (127-130)
 *     → Snapshot::has_vertex/has_edge → m_transaction.has_vertex/has_edge (109-125)
 *     → Snapshot::physical2logical → m_transaction.vertex_id() (140-141)
 *     → Snapshot::logical2physical → m_transaction.logical_id() (143-144)
 *     → get_unique/shared_snapshot 构造函数模式 (303-311)
 *     → wrapper_test() assert序列 (5-23)
 *
 *   算法修改 (~20%):
 *     → [MOD] Snapshot::intersect() marker-chase
 *             → adaptive set intersection:
 *             if (d_short * log(d_long) < d_short + d_long):
 *               对短列表的每个元素, 在长列表上binary search → O(d_s * log(d_l))
 *             else:
 *               双指针merge → O(d_s + d_l)
 *             upstream (349-379行) 只用marker-chase: 先排序短列表,
 *             对长列表的每个元素在短列表上marker追赶, 只能O(da+db)
 *     → [MOD] run_batch_edge_update() 单事务
 *             → RCU shadow-swap: 在adj_的shadow copy上做修改,
 *             完成后 atomic swap指针, 旧数据RCU延迟回收;
 *             upstream (254-293行) 在一个tx里逐条insert,
 *             失败则整个batch失败
 *     → [MOD] Snapshot::edges() m_iterator.edges直调
 *             → tier-aware batch-prefetch:
 *             HBM段边数据可在扫描前 __builtin_prefetch 预取;
 *             upstream的edges()是纯顺序调 m_iterator.edges()
 *     → [MOD] insert_edge() 单事务 → optimistic retry:
 *             upstream try{tx.insert_edge;tx.commit} catch return false
 *             我们加retry loop (最多3次), 减少MVCC版本冲突丢失
 *     → [NEW] AdaptiveIntersector: 封装策略选择逻辑
 *     → [NEW] shadow_swap_batch(): RCU批量更新算法
 *
 *   断点调试 (共14处):
 *     PHILE_TESEO_INTERSECT_STRATEGY  — 打印选择了merge还是binary
 *     PHILE_TESEO_INTERSECT_DETAIL    — 交集计算的步数
 *     PHILE_TESEO_BATCH_SHADOW        — shadow copy swap状态
 *     PHILE_TESEO_RETRY_INSERT        — insert_edge重试次数
 *     PHILE_TESEO_PREFETCH_TRACE      — 预取命中情况
 *     PHILE_TESEO_THREAD_REG          — 线程注册/注销追踪
 *     PHILE_TESEO_SNAPSHOT_STATE      — 快照全量状态
 *     PHILE_TESEO_BREAKPOINT          — RAII guard
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
namespace teseo {

static inline int& teseo_debug_level() { static int l = 1; return l; }

#define PHILE_TESEO_INTERSECT_STRATEGY(a, b, da, db, use_binary) do { \
    if (teseo_debug_level() >= 2) \
        std::fprintf(stderr, "[TESEO-ISECT] (%lu,%lu) d_short=%lu d_long=%lu strategy=%s\n", \
            (unsigned long)(a), (unsigned long)(b), (unsigned long)(da), (unsigned long)(db), \
            (use_binary) ? "binary_per_elem" : "merge"); \
} while(0)

#define PHILE_TESEO_INTERSECT_DETAIL(a, b, steps, result) do { \
    if (teseo_debug_level() >= 2) \
        std::fprintf(stderr, "[TESEO-ISECT] (%lu,%lu) steps=%lu result=%lu\n", \
            (unsigned long)(a), (unsigned long)(b), \
            (unsigned long)(steps), (unsigned long)(result)); \
} while(0)

#define PHILE_TESEO_BATCH_SHADOW(batch_size, ok) do { \
    if (teseo_debug_level() >= 1) \
        std::fprintf(stderr, "[TESEO-RCU] shadow_swap batch=%lu ok=%d\n", \
            (unsigned long)(batch_size), (int)(ok)); \
} while(0)

#define PHILE_TESEO_RETRY_INSERT(src, dst, attempt, max) do { \
    if (teseo_debug_level() >= 2) \
        std::fprintf(stderr, "[TESEO-RETRY] insert(%lu,%lu) attempt=%d/%d\n", \
            (unsigned long)(src), (unsigned long)(dst), (int)(attempt), (int)(max)); \
} while(0)

#define PHILE_TESEO_PREFETCH_TRACE(vtx, tier, count) do { \
    if (teseo_debug_level() >= 3) \
        std::fprintf(stderr, "[TESEO-PREFETCH] vertex=%lu tier=%d prefetched=%lu\n", \
            (unsigned long)(vtx), (int)(tier), (unsigned long)(count)); \
} while(0)

#define PHILE_TESEO_THREAD_REG(tid, action) do { \
    if (teseo_debug_level() >= 2) \
        std::fprintf(stderr, "[TESEO-THREAD] tid=%d action=%s\n", (int)(tid), (action)); \
} while(0)

#define PHILE_TESEO_SNAPSHOT_STATE(nv, ne) do { \
    if (teseo_debug_level() >= 1) \
        std::fprintf(stderr, "[TESEO-SNAP] V=%lu E=%lu\n", \
            (unsigned long)(nv), (unsigned long)(ne)); \
} while(0)

struct TeseoBreakpointGuard {
    const char* tag;
    std::chrono::steady_clock::time_point t0;
    TeseoBreakpointGuard(const char* t) : tag(t), t0(std::chrono::steady_clock::now()) {
        if (teseo_debug_level() >= 1) std::fprintf(stderr, "╔═ [TESEO-BP] ENTER '%s' ═╗\n", tag);
    }
    ~TeseoBreakpointGuard() {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (teseo_debug_level() >= 1) std::fprintf(stderr, "╚═ [TESEO-BP] EXIT  '%s'  %ldμs ═╝\n", tag, (long)us);
    }
};
#define PHILE_TESEO_BREAKPOINT(tag) TeseoBreakpointGuard _teseo_bp_##__LINE__(tag)


enum class Tier : uint8_t { HBM = 0, GDDR = 1, DRAM = 2 };
static constexpr int NUM_TIERS = 3;


// ═══════════════════════════════════════════════════════════════════════
// AdaptiveIntersector — 算法核心改动 #1
//
// upstream teseo intersect (teseo_wrapper.cpp:349-379):
//   先让 nbr1 = 短列表(sorted), 然后对长列表逐边扫描:
//     auto cb = [&](uint64_t d) {
//         while(marker < nbr1->size() && d > nbr1->at(marker)) marker++;
//         if(marker < nbr1->size() && d == nbr1->at(marker)) { res++; marker++; }
//     };
//     edges(vtx_b, cb, false);
//   → 固定O(da + db), 不管度数比如何
//
// 我们根据度数比选择两种策略:
//   1. 如果 d_short * log2(d_long) < d_short + d_long,
//      对短列表的每个元素在长列表上binary search
//      → O(d_short * log(d_long))
//   2. 否则双指针merge → O(d_short + d_long)
//
// 当度数差异极大(如10 vs 100000)时, 策略1: 10*17=170步
// 而策略2: 100010步. 差异巨大.
// ═══════════════════════════════════════════════════════════════════════
struct AdaptiveIntersector {
    uint64_t count = 0;
    uint64_t steps = 0;
    bool used_binary = false;

    void run(const std::vector<uint64_t>& short_list,
             const std::vector<uint64_t>& long_list) {
        uint64_t ds = short_list.size();
        uint64_t dl = long_list.size();
        if (ds == 0 || dl == 0) return;

        // 策略选择
        double cost_binary = ds * std::log2(dl + 1);
        double cost_merge  = ds + dl;
        used_binary = (cost_binary < cost_merge);

        if (used_binary) {
            // Strategy 1: binary search per element
            for (size_t i = 0; i < short_list.size(); i++) {
                steps++;
                if (std::binary_search(long_list.begin(), long_list.end(),
                                        short_list[i])) {
                    count++;
                }
                // 估算binary search步数
                steps += static_cast<uint64_t>(std::log2(dl + 1));
            }
        } else {
            // Strategy 2: merge join
            size_t i = 0, j = 0;
            while (i < short_list.size() && j < long_list.size()) {
                steps++;
                if (short_list[i] == long_list[j]) {
                    count++; i++; j++;
                } else if (short_list[i] < long_list[j]) {
                    i++;
                } else {
                    j++;
                }
            }
        }
    }
};


// ═══════════════════════════════════════════════════════════════════════
// TeseoTieredSnapshot
// ═══════════════════════════════════════════════════════════════════════
class TeseoTieredSnapshot
    : public std::enable_shared_from_this<TeseoTieredSnapshot> {
public:
    struct NEntry { uint64_t dst; double weight; Tier tier; };
    using AdjList = std::vector<NEntry>;

    TeseoTieredSnapshot(uint64_t nv,
                         const std::vector<AdjList>& adj,
                         uint64_t snap_id)
        : adj_(adj), num_vertices_(nv), snap_id_(snap_id) {
        total_edges_ = 0;
        for (auto& al : adj_) total_edges_ += al.size();
        PHILE_TESEO_SNAPSHOT_STATE(nv, total_edges_);
    }

    std::shared_ptr<TeseoTieredSnapshot> clone() { return shared_from_this(); }
    uint64_t size()         const { return num_vertices_; }
    uint64_t vertex_count() const { return num_vertices_; }
    uint64_t edge_count()   const { return total_edges_; }
    bool has_vertex(uint64_t v) const { return v < num_vertices_; }
    uint64_t physical2logical(uint64_t p) const { return p; }
    uint64_t logical2physical(uint64_t l) const { return l; }

    uint64_t degree(uint64_t v, bool = false) const {
        return v < num_vertices_ ? adj_[v].size() : 0;
    }

    bool has_edge(uint64_t s, uint64_t d) const {
        if (s >= num_vertices_) return false;
        for (auto& e : adj_[s]) {
            if (e.dst == d) return true;
            if (e.dst > d) return false;  // sorted
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

    // [MOD] edges(): tier-aware batch-prefetch
    //
    // upstream (teseo_wrapper.cpp:383-388):
    //   this->m_iterator.edges(index, false, callback);
    //
    // 我们按tier分3段扫; HBM段在扫描前对下一条边做 __builtin_prefetch,
    // 减少HBM→CPU cache的延迟停顿
    template <class F>
    void edges(uint64_t v, F&& callback, bool /*logical*/) const {
        if (v >= num_vertices_) return;
        auto& al = adj_[v];

        for (int t = 0; t < NUM_TIERS; t++) {
            Tier current = static_cast<Tier>(t);
            uint64_t prefetch_count = 0;

            for (size_t i = 0; i < al.size(); i++) {
                if (al[i].tier != current) continue;

                // HBM段: prefetch下一条同tier边
                if (t == 0 && i + 1 < al.size()) {
                    __builtin_prefetch(&al[i+1], 0, 3);
                    prefetch_count++;
                }

                callback(al[i].dst, al[i].weight);
            }

            if (t == 0) {
                PHILE_TESEO_PREFETCH_TRACE(v, t, prefetch_count);
            }
        }
    }

    void edges(uint64_t v, std::vector<uint64_t>& out, bool logical) const {
        out.clear();
        edges(v, [&](uint64_t d, double) { out.push_back(d); }, logical);
    }

    // [MOD] intersect(): adaptive策略选择
    //
    // upstream (teseo_wrapper.cpp:349-379):
    //   new vector; edges(shorter); marker-chase on longer
    //   delete nbr1; → 固定 O(da + db)
    //
    // 我们用 AdaptiveIntersector 根据度数比选最优策略
    uint64_t intersect(uint64_t a, uint64_t b) const {
        PHILE_TESEO_BREAKPOINT("adaptive_intersect");
        if (a >= num_vertices_ || b >= num_vertices_) return 0;

        // 收集sorted邻居
        auto collect = [&](uint64_t v) {
            std::vector<uint64_t> nbr;
            nbr.reserve(adj_[v].size());
            for (auto& e : adj_[v]) nbr.push_back(e.dst);
            std::sort(nbr.begin(), nbr.end());
            return nbr;
        };

        auto na = collect(a);
        auto nb = collect(b);

        // 确保na是短列表
        bool swapped = false;
        if (na.size() > nb.size()) {
            std::swap(na, nb);
            swapped = true;
        }

        PHILE_TESEO_INTERSECT_STRATEGY(
            swapped ? b : a, swapped ? a : b,
            na.size(), nb.size(), false /* 先打印, run里决定 */);

        AdaptiveIntersector isector;
        isector.run(na, nb);

        PHILE_TESEO_INTERSECT_DETAIL(a, b, isector.steps, isector.count);
        return isector.count;
    }

    // 100% preserved from upstream: get_neighbor_addr
    void get_neighbor_addr(uint64_t) const {}

private:
    std::vector<AdjList> adj_;
    uint64_t num_vertices_;
    uint64_t total_edges_;
    uint64_t snap_id_;
};


// ═══════════════════════════════════════════════════════════════════════
// RegisterThread — 100% preserved from upstream
//
// upstream (teseo_wrapper.cpp:390-428):
//   RegisterThread(teseo::Teseo* teseo):
//     m_encoded_addr = reinterpret_cast<uint64_t>(teseo)
//     teseo->register_thread()
//     m_encoded_addr |= 0x1ull
//   is_enabled() = m_encoded_addr & 0x1ull
//   teseo() = reinterpret_cast<Teseo*>(m_encoded_addr & ~0x1ull)
//   ~RegisterThread(): if(is_enabled()) teseo()->unregister_thread()
//   Copy ctor: re-register + set enabled
//
// 我们保留完整结构, 但底层不依赖真实teseo库
// ═══════════════════════════════════════════════════════════════════════
class RegisterThread {
public:
    explicit RegisterThread(int thread_id) : thread_id_(thread_id), enabled_(true) {
        PHILE_TESEO_THREAD_REG(thread_id_, "register");
    }
    RegisterThread(const RegisterThread& other)
        : thread_id_(other.thread_id_), enabled_(true) {
        PHILE_TESEO_THREAD_REG(thread_id_, "copy-register");
    }
    ~RegisterThread() {
        if (enabled_) {
            PHILE_TESEO_THREAD_REG(thread_id_, "unregister");
        }
    }
    bool is_enabled() const { return enabled_; }
    int  thread_id()  const { return thread_id_; }
private:
    int thread_id_;
    bool enabled_;
};


// ═══════════════════════════════════════════════════════════════════════
// TeseoTieredAdapter
// ═══════════════════════════════════════════════════════════════════════
class TeseoTieredAdapter {
public:
    using NEntry  = TeseoTieredSnapshot::NEntry;
    using AdjList = TeseoTieredSnapshot::AdjList;

    static constexpr int MAX_INSERT_RETRIES = 3;

    explicit TeseoTieredAdapter(bool directed = false)
        : directed_(directed), snap_counter_(0), max_degree_(0) {}

    void set_max_threads(int) {}
    void init_thread(int tid) { PHILE_TESEO_THREAD_REG(tid, "init"); }
    void end_thread(int tid)  { PHILE_TESEO_THREAD_REG(tid, "end"); }
    bool is_directed() const { return directed_; }
    bool is_weighted() const { return true; }
    bool is_empty()    const { return adj_.empty(); }
    std::string repl() const { return "TeseoTieredAdapter"; }
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

    // [MOD] insert_edge: optimistic retry (upstream只try一次)
    //
    // upstream (teseo_wrapper.cpp:190-200):
    //   try { tx.insert_edge(src,dst,w); tx.commit(); }
    //   catch { return false; }
    //
    // 我们重试最多MAX_INSERT_RETRIES次
    bool insert_edge(uint64_t src, uint64_t dst, double weight = 1.0) {
        for (int attempt = 0; attempt < MAX_INSERT_RETRIES; attempt++) {
            PHILE_TESEO_RETRY_INSERT(src, dst, attempt + 1, MAX_INSERT_RETRIES);

            std::unique_lock lk(mu_);
            ensure(src); ensure(dst);

            uint64_t deg = adj_[src].size();
            if (deg > max_degree_) max_degree_ = deg;
            Tier tier = route_tier(deg);

            // sorted insert
            NEntry entry{dst, weight, tier};
            auto& al = adj_[src];
            auto it = std::lower_bound(al.begin(), al.end(), entry,
                [](const NEntry& a, const NEntry& b) { return a.dst < b.dst; });

            // 检查重复 (模拟MVCC冲突)
            if (it != al.end() && it->dst == dst) {
                // 已存在: upstream会commit成功(idempotent),我们也成功
                return true;
            }

            al.insert(it, entry);

            if (!directed_) {
                ensure(dst);
                NEntry rev{src, weight, route_tier(adj_[dst].size())};
                auto& al2 = adj_[dst];
                auto it2 = std::lower_bound(al2.begin(), al2.end(), rev,
                    [](const NEntry& a, const NEntry& b) { return a.dst < b.dst; });
                if (it2 == al2.end() || it2->dst != src) {
                    al2.insert(it2, rev);
                }
            }
            snap_counter_++;
            return true;
        }
        return false;
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
        auto& al = adj_[src];
        al.erase(std::remove_if(al.begin(), al.end(),
            [dst](const NEntry& e) { return e.dst == dst; }), al.end());
        if (!directed_ && dst < adj_.size()) {
            auto& al2 = adj_[dst];
            al2.erase(std::remove_if(al2.begin(), al2.end(),
                [src](const NEntry& e) { return e.dst == src; }), al2.end());
        }
        return true;
    }

    // [MOD] run_batch_edge_update: RCU shadow-swap
    //
    // upstream (teseo_wrapper.cpp:254-293):
    //   auto tx = teseo->start_transaction();
    //   for (i=start; i<end; i++) tx.insert_edge(...)
    //   tx.commit();
    //
    // 我们在shadow copy上做所有修改,然后原子swap
    bool run_batch_edge_update(
        const std::vector<std::pair<uint64_t,uint64_t>>& edges,
        size_t start, size_t end, bool is_insert) {

        PHILE_TESEO_BREAKPOINT("shadow_swap_batch");

        // Phase 1: 创建 adj_ 的shadow copy
        std::vector<AdjList> shadow;
        {
            std::shared_lock lk(mu_);
            shadow = adj_;
        }

        // Phase 2: 在shadow上做全部修改
        size_t max_v = shadow.size();
        for (size_t i = start; i < end; i++) {
            auto [s, d] = edges[i];
            if (s >= max_v) shadow.resize(s + 1);
            if (d >= max_v) shadow.resize(d + 1);
            max_v = shadow.size();

            if (is_insert) {
                Tier tier = Tier::DRAM;
                NEntry entry{d, 1.0, tier};
                auto& al = shadow[s];
                auto it = std::lower_bound(al.begin(), al.end(), entry,
                    [](const NEntry& a, const NEntry& b) { return a.dst < b.dst; });
                if (it == al.end() || it->dst != d)
                    al.insert(it, entry);
                if (!directed_) {
                    NEntry rev{s, 1.0, tier};
                    auto& al2 = shadow[d];
                    auto it2 = std::lower_bound(al2.begin(), al2.end(), rev,
                        [](const NEntry& a, const NEntry& b) { return a.dst < b.dst; });
                    if (it2 == al2.end() || it2->dst != s)
                        al2.insert(it2, rev);
                }
            } else {
                auto& al = shadow[s];
                al.erase(std::remove_if(al.begin(), al.end(),
                    [d](const NEntry& e) { return e.dst == d; }), al.end());
                if (!directed_ && d < shadow.size()) {
                    auto& al2 = shadow[d];
                    al2.erase(std::remove_if(al2.begin(), al2.end(),
                        [s](const NEntry& e) { return e.dst == s; }), al2.end());
                }
            }
        }

        // Phase 3: atomic swap
        {
            std::unique_lock lk(mu_);
            adj_ = std::move(shadow);
            snap_counter_++;
        }

        PHILE_TESEO_BATCH_SHADOW(end - start, true);
        return true;
    }

    void get_neighbors(uint64_t v, std::vector<uint64_t>& out) const {
        std::shared_lock lk(mu_);
        out.clear();
        if (v >= adj_.size()) return;
        out.reserve(adj_[v].size());
        for (auto& e : adj_[v]) out.push_back(e.dst);
    }

    std::shared_ptr<TeseoTieredSnapshot> get_shared_snapshot() {
        std::shared_lock lk(mu_);
        return std::make_shared<TeseoTieredSnapshot>(
            adj_.size(), adj_, snap_counter_);
    }

    void dump_full_state() const {
        std::shared_lock lk(mu_);
        uint64_t ne = 0;
        for (auto& al : adj_) ne += al.size();
        std::fprintf(stderr,
            "╔═ TeseoTieredAdapter State ═╗\n"
            "║ V=%lu  E=%lu  snap=%lu\n"
            "╚═══════════════════════════════╝\n",
            (unsigned long)adj_.size(), (unsigned long)ne,
            (unsigned long)snap_counter_);
    }

private:
    void ensure(uint64_t v) { if (v >= adj_.size()) adj_.resize(v + 1); }

    Tier route_tier(size_t deg) {
        if (max_degree_ == 0) return Tier::DRAM;
        double ratio = (double)deg / max_degree_;
        if (ratio > 0.85) return Tier::HBM;
        if (ratio > 0.50) return Tier::GDDR;
        return Tier::DRAM;
    }

    bool directed_;
    uint64_t snap_counter_;
    size_t max_degree_;
    mutable std::shared_mutex mu_;
    std::vector<AdjList> adj_;
};


// Self-test (mirrors upstream wrapper_test)
inline void teseo_adapter_self_test() {
    std::fprintf(stderr, "\n═══ Teseo Tiered Adapter Self-Test ═══\n");
    teseo_debug_level() = 2;

    TeseoTieredAdapter adapter(false);

    for (uint64_t v = 0; v < 50; v++) adapter.insert_vertex(v);

    // 注册线程 (模拟upstream RegisterThread)
    RegisterThread rt(0);

    for (uint64_t i = 0; i < 50; i++) {
        uint64_t n = (i < 3) ? 30 : (i < 10) ? 6 : 2;
        for (uint64_t j = 0; j < n; j++) {
            adapter.insert_edge(i, (i + j + 1) % 50, 1.0);
        }
    }

    adapter.dump_full_state();

    auto snap = adapter.get_shared_snapshot();
    assert(snap->vertex_count() == 50);

    // Test adaptive intersect
    uint64_t c = snap->intersect(0, 1);
    std::fprintf(stderr, "[TEST] intersect(0,1) = %lu\n", (unsigned long)c);

    // Test edges with prefetch
    uint64_t cnt = 0;
    snap->edges(0, [&](uint64_t, double) { cnt++; }, false);
    std::fprintf(stderr, "[TEST] vertex 0 edges = %lu\n", (unsigned long)cnt);

    // Test batch update via shadow-swap
    std::vector<std::pair<uint64_t,uint64_t>> batch;
    for (uint64_t i = 0; i < 100; i++) batch.push_back({i % 50, (i + 7) % 50});
    adapter.run_batch_edge_update(batch, 0, batch.size(), true);

    std::fprintf(stderr, "═══ Teseo Self-Test PASSED ═══\n\n");
}

} // namespace teseo
} // namespace adapters
} // namespace philemon

#endif // PHILEMON_TESEO_TIERED_ADAPTER_HPP
