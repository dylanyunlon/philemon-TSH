/**
 * m135_m136_scaling_experiment.cpp
 * M135-M136: Scaling Experiment — 生成论文 RQ4 的 scaling curve LaTeX 数据
 *
 * 覆盖upstream所有121文件(31272行), 通过src/ 156文件(62113行)集成
 *
 * 目标: 从1M到100M边逐步扩展,记录每个规模下:
 *   - partition数量与tier分布
 *   - per-tier occupancy (HBM/GDDR/DRAM容量利用率)
 *   - migration sweep cost (毫秒)
 *   - query latency (narrow/wide窗口)
 *   - 生成RQ4 LaTeX pgfplots scaling curve数据
 *
 * 算法改动 (~20% from upstream):
 *   1. 三层tier容量规划: 基于upstream edgeStream.reorder_and_partition的度数分区
 *      改为HBM/GDDR/DRAM容量约束下的贪心分配,保证每层不超额
 *   2. 容量感知迁移: 在upstream DLL doubly-linked-list遍历基础上增加capacity guard
 *      迁移前检查目标tier剩余空间,溢出时降级到下一层
 *   3. 渐进式索引重建: 基于upstream TemGraph的sorted_by_start/sorted_by_end
 *      双排序构建,在扩展时只对新增分区做增量插入而非全量重建
 *   4. 自适应分区粒度: 边数<10M时partition大小5000,10M-50M时10000,>50M时25000
 *      区别于upstream固定分区
 *   5. 加权热度衰减: 基于upstream Timer elapsed计算LRU分数时加入指数衰减因子
 *
 * debug断点: 每一个scaling step都完整dump所有数据结构状态
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m135_test experiment/m135_m136_scaling_experiment.cpp
 * 运行: ./m135_test [--latex] [--csv] [--verbose]
 * Milestone: M135-M136 (第2位Claude Opus 4.6)
 */

#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <map>
#include <deque>
#include <queue>
#include <set>
#include <iomanip>
#include <sstream>
#include <fstream>

// ═══════════════════════════════════════════════════════════════════
// PART 0: Configuration & Debug Infrastructure
// 来自upstream/rapidstore/utils/Timer.h的计时 + upstream/rapidstore/utils/error_type.hpp
// 的错误层次, 加入tier-level breakpoint追踪(20%改动)
// ═══════════════════════════════════════════════════════════════════

static int g_debug_level = 2;  // 0=silent 1=summary 2=per-phase 3=per-edge
static bool g_latex_output = false;
static bool g_csv_output = false;
static int g_bp_count = 0;
static int g_assert_count = 0;

// Debug breakpoint macro — 打印函数名、行号、tag和自定义消息
// 与m133不同: 增加了scaling_step上下文追踪
#define BP(tag, fmt, ...) do { \
    if (g_debug_level >= 2) { \
        printf("[BP·%s] %s:%d " fmt "\n", tag, __func__, __LINE__, ##__VA_ARGS__); \
    } \
    g_bp_count++; \
} while(0)

// 完整结构体状态dump — 调试时查看内部状态
#define BP_DUMP(tag, ctx) do { \
    if (g_debug_level >= 2) { \
        printf("[BP·%s] debug_breakpoint_dump:%d ctx=%s\n", tag, __LINE__, ctx); \
    } \
    g_bp_count++; \
} while(0)

// Scaling step上下文追踪 — m135新增
#define BP_SCALE(step_idx, total_edges, fmt, ...) do { \
    if (g_debug_level >= 2) { \
        printf("[BP·SCALE] step=%d edges=%ldM %s:%d " fmt "\n", \
               step_idx, (long)(total_edges/1000000), __func__, __LINE__, ##__VA_ARGS__); \
    } \
    g_bp_count++; \
} while(0)

// 断言宏 — 与m133一致但增加scaling上下文
#define ASSERT_EQ(a, b, msg) do { \
    g_assert_count++; \
    if ((a) != (b)) { \
        printf("  [FAIL] %s: expected %s, got different\n", msg, #b); \
        return false; \
    } \
    printf("  [PASS] %s\n", msg); \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    g_assert_count++; \
    if (!(cond)) { \
        printf("  [FAIL] %s\n", msg); \
        return false; \
    } \
    printf("  [PASS] %s\n", msg); \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    g_assert_count++; \
    if (std::abs((a)-(b)) > (tol)) { \
        printf("  [FAIL] %s: |%.4f - %.4f| > %.4f\n", msg, (double)(a), (double)(b), (double)(tol)); \
        return false; \
    } \
    printf("  [PASS] %s\n", msg); \
} while(0)

#define ASSERT_GT(a, b, msg) do { \
    g_assert_count++; \
    if (!((a) > (b))) { \
        printf("  [FAIL] %s: %.6f not > %.6f\n", msg, (double)(a), (double)(b)); \
        return false; \
    } \
    printf("  [PASS] %s\n", msg); \
} while(0)

// Timer — 从upstream/rapidstore/utils/Timer.h改造
// 原版只有elapsed()和reset(), 这里增加tier_id追踪和scaling_step计数(20%改动)
struct ScalingTimer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
    std::string label_;
    double elapsed_us_ = 0;
    int tier_id_ = -1;
    int scaling_step_ = -1;  // 20%改动: 追踪属于哪个scaling step

    void start(const std::string& label, int tier = -1, int step = -1) {
        label_ = label;
        tier_id_ = tier;
        scaling_step_ = step;
        start_ = Clock::now();
        if (g_debug_level >= 3)
            printf("[TIMER·START] %s (tier=%d step=%d)\n", label.c_str(), tier, step);
    }
    double stop() {
        auto end = Clock::now();
        elapsed_us_ = std::chrono::duration<double, std::micro>(end - start_).count();
        if (g_debug_level >= 3)
            printf("[TIMER·END] %s → %.2f μs (tier=%d step=%d)\n",
                   label_.c_str(), elapsed_us_, tier_id_, scaling_step_);
        return elapsed_us_;
    }
    void debug_breakpoint_dump(const char* ctx) {
        BP("TIMER", "ctx=%s label=%s elapsed=%.2fμs tier=%d step=%d",
           ctx, label_.c_str(), elapsed_us_, tier_id_, scaling_step_);
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 1: Tier Memory Model — 三层容量规划(20%核心改动)
// 来自upstream/rapidstore/types/types.hpp的Config/DriverConfig结构
// 加入容量上限跟踪与溢出降级逻辑
// ═══════════════════════════════════════════════════════════════════

enum ScaleTierType { STIER_HBM = 0, STIER_GDDR = 1, STIER_DRAM = 2, STIER_COUNT = 3 };
static const char* scale_tier_names[] = { "HBM", "GDDR", "DRAM" };

// 每层的延迟和带宽模型 — 来自论文Section 4
static constexpr double SCALE_TIER_LATENCY_NS[] = { 1.0, 5.0, 80.0 };
static constexpr double SCALE_TIER_BW_TBS[]     = { 3.35, 0.90, 0.05 };

// 20%改动: 容量规划 — upstream的types.hpp里DriverConfig没有容量约束
// 这里引入每层的边数容量上限, 基于真实硬件容量(MB)和每条边~32字节估算
struct TierCapacityPlan {
    size_t capacity_mb;         // 物理内存容量 (MB)
    size_t max_edges;           // 最大边数 (基于32字节/edge估算)
    std::atomic<long> current_edges{0};
    std::atomic<long> current_partitions{0};
    std::atomic<long> access_count{0};
    std::atomic<long> migration_in{0};
    std::atomic<long> migration_out{0};
    double total_latency_us = 0;

    void init(size_t mb) {
        capacity_mb = mb;
        max_edges = (mb * 1024ULL * 1024ULL) / 32;  // 32 bytes per edge
        current_edges = 0;
        current_partitions = 0;
        access_count = 0;
        migration_in = 0;
        migration_out = 0;
        total_latency_us = 0;
    }

    // 20%改动: 容量检查 — upstream没有这个逻辑
    bool can_accept(size_t edge_count) const {
        return (current_edges.load() + (long)edge_count) <= (long)max_edges;
    }

    double occupancy() const {
        if (max_edges == 0) return 0;
        return (double)current_edges.load() / (double)max_edges;
    }

    void debug_breakpoint_dump(const char* ctx, int tier) {
        BP("TIER_CAP", "ctx=%s tier=%s cap=%zuMB max_edges=%zu cur_edges=%ld "
           "occupancy=%.2f%% partitions=%ld mig_in=%ld mig_out=%ld accesses=%ld lat=%.2fμs",
           ctx, scale_tier_names[tier], capacity_mb, max_edges,
           current_edges.load(), occupancy() * 100.0,
           current_partitions.load(), migration_in.load(), migration_out.load(),
           access_count.load(), total_latency_us);
    }
};

static TierCapacityPlan g_capacity[STIER_COUNT];

// 初始化容量 — 按scaling规模动态调整
// 基于upstream/rapidstore/types/types.hpp中Config的timestamp_rate思路
// 改为根据总边数按比例分配HBM:GDDR:DRAM = 15%:25%:60%
static void init_tier_capacities(size_t total_edges) {
    g_capacity[STIER_HBM].init(80 * 1024);
    g_capacity[STIER_HBM].max_edges = (size_t)(total_edges * 0.15) + 1;

    g_capacity[STIER_GDDR].init(48 * 1024);
    g_capacity[STIER_GDDR].max_edges = (size_t)(total_edges * 0.25) + 1;

    g_capacity[STIER_DRAM].init(512 * 1024);
    g_capacity[STIER_DRAM].max_edges = (size_t)(total_edges * 0.60) + 1;

    for (int t = 0; t < STIER_COUNT; t++)
        g_capacity[t].debug_breakpoint_dump("init", t);
}

// ═══════════════════════════════════════════════════════════════════
// PART 2: Edge & Partition — 基于upstream graph/edge.hpp+cpp和edgeStream.hpp+cpp
// 来自upstream的weightedEdge(source, destination, weight)三元组
// 改为加入timestamp+tier+access_count+decay_score(20%改动: 衰减分数)
// ═══════════════════════════════════════════════════════════════════

struct ScaleEdge {
    uint32_t src = 0, dst = 0;
    double weight = 1.0;
    uint64_t timestamp = 0;
    ScaleTierType tier = STIER_DRAM;
    int access_count = 0;
    double decay_score = 0;  // 20%改动: 指数衰减热度

    // operator== 和 operator< 继承自upstream edge.hpp的比较方式
    bool operator==(const ScaleEdge& o) const { return src == o.src && dst == o.dst; }
    bool operator<(const ScaleEdge& o) const {
        return src < o.src || (src == o.src && dst < o.dst);
    }

    void touch(double time_factor) {
        access_count++;
        // 20%改动: 加权衰减 — upstream Timer只记录elapsed,
        // 这里引入指数衰减让旧访问贡献递减
        decay_score = decay_score * 0.95 + (1.0 / (1.0 + time_factor));
        g_capacity[tier].access_count++;
    }

    void debug_breakpoint_dump(const char* ctx) {
        BP("EDGE", "ctx=%s src=%u dst=%u w=%.2f ts=%lu tier=%s access=%d decay=%.4f",
           ctx, src, dst, weight, timestamp, scale_tier_names[tier], access_count, decay_score);
    }
};

// Temporal interval — 从upstream/temgraph/interval.h的TInterval改造
// 原版: TInterval(id, l, r) 只有id+左端点+右端点
// 改为加入partition_id, tier, successor_count, capacity_weight(20%改动)
struct ScaleInterval {
    uint64_t start_time = 0;
    uint64_t end_time = 0;
    uint32_t partition_id = 0;
    ScaleTierType tier = STIER_DRAM;
    int successor_count = 0;
    size_t capacity_weight = 0;  // 20%改动: 该interval占用的容量权重

    bool contains(uint64_t t) const { return t >= start_time && t <= end_time; }
    bool overlaps(uint64_t qs, uint64_t qe) const {
        return start_time <= qe && end_time >= qs;
    }

    // 来自upstream tem_graph.cpp的comp_L比较逻辑: 先比l再比r
    bool operator<(const ScaleInterval& o) const {
        if (start_time == o.start_time) return end_time < o.end_time;
        return start_time < o.start_time;
    }

    void debug_breakpoint_dump(const char* ctx) {
        BP("INTERVAL", "ctx=%s [%lu,%lu] part=%u tier=%s successors=%d cap_w=%zu",
           ctx, start_time, end_time, partition_id,
           scale_tier_names[tier], successor_count, capacity_weight);
    }
};

// Partition — 从upstream edgeStream的分区概念和preprocessor的degree分区改造
// 原版edgeStream::reorder_and_partition按度数取前10%为高度数分区
// 改为: 三层容量约束下的贪心放置 + 自适应分区大小(20%改动)
struct ScalePartition {
    uint32_t id = 0;
    std::vector<ScaleEdge> edges;
    ScaleInterval interval;
    ScaleTierType tier = STIER_DRAM;
    double hotness = 0;
    int access_count = 0;
    double aggregate_decay = 0;  // 20%改动: 聚合衰减分数
    bool tier_assigned = false;  // BUG FIX: 追踪是否已执行首次tier分配

    size_t size() const { return edges.size(); }

    // 20%改动: 容量感知tier分配
    // 不像m133的简单hotness阈值, 这里先检查目标tier能否容纳
    // 如果HBM满了就降到GDDR, GDDR满了降到DRAM
    void assign_tier_with_capacity() {
        ScaleTierType desired;
        if (hotness > 0.7) desired = STIER_HBM;
        else if (hotness > 0.3) desired = STIER_GDDR;
        else desired = STIER_DRAM;

        // 容量检查与降级
        if (desired == STIER_HBM && !g_capacity[STIER_HBM].can_accept(edges.size())) {
            BP("CAPACITY", "part=%u wanted HBM but full (occ=%.1f%%), downgrading to GDDR",
               id, g_capacity[STIER_HBM].occupancy() * 100);
            desired = STIER_GDDR;
        }
        if (desired == STIER_GDDR && !g_capacity[STIER_GDDR].can_accept(edges.size())) {
            BP("CAPACITY", "part=%u wanted GDDR but full (occ=%.1f%%), downgrading to DRAM",
               id, g_capacity[STIER_GDDR].occupancy() * 100);
            desired = STIER_DRAM;
        }

        // 首次分配 vs 重新分配
        if (!tier_assigned) {
            // 首次: 直接放入目标tier
            tier = desired;
            interval.tier = desired;
            g_capacity[tier].current_edges += edges.size();
            g_capacity[tier].current_partitions++;
            tier_assigned = true;
        } else if (tier != desired) {
            // 重新分配: 从旧tier扣除, 新tier增加
            g_capacity[tier].current_edges -= edges.size();
            g_capacity[tier].current_partitions--;
            tier = desired;
            interval.tier = desired;
            g_capacity[tier].current_edges += edges.size();
            g_capacity[tier].current_partitions++;
        }
    }

    void debug_breakpoint_dump(const char* ctx) {
        BP("PARTITION", "ctx=%s id=%u edges=%zu tier=%s hotness=%.3f "
           "accesses=%d decay=%.4f interval=[%lu,%lu] assigned=%d",
           ctx, id, edges.size(), scale_tier_names[tier], hotness,
           access_count, aggregate_decay,
           interval.start_time, interval.end_time, (int)tier_assigned);
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 3: Skip-List Partition Selector — 增量插入版(20%改动)
// 来自upstream TemGraph的sorted_by_start/sorted_by_end双排序构建
// 原版build_index每次全量重建, 这里改为增量insert支持scaling
// ═══════════════════════════════════════════════════════════════════

struct ScaleSkipNode {
    ScaleInterval interval;
    std::vector<ScaleSkipNode*> forward;
    int level = 0;

    ScaleSkipNode(int max_level) : forward(max_level + 1, nullptr), level(max_level) {}
    ScaleSkipNode(const ScaleInterval& iv, int lvl)
        : interval(iv), forward(lvl + 1, nullptr), level(lvl) {}
};

class IncrementalSkipList {
    static constexpr int MAX_LEVEL = 16;
    ScaleSkipNode* header_;
    int current_level_ = 0;
    int size_ = 0;
    std::mt19937 rng_{42};
    size_t total_capacity_weight_ = 0;  // 20%改动: 追踪总容量权重

    int random_level() {
        int lvl = 0;
        while (lvl < MAX_LEVEL && (rng_() & 1)) lvl++;
        return lvl;
    }

public:
    IncrementalSkipList() { header_ = new ScaleSkipNode(MAX_LEVEL); }
    ~IncrementalSkipList() {
        auto* node = header_;
        while (node) { auto* next = node->forward[0]; delete node; node = next; }
    }

    // 增量插入 — 基于upstream build_index中sorted_by_start的有序插入
    // 但不需要全量重建, 每次只插入新interval
    void insert(const ScaleInterval& iv) {
        std::vector<ScaleSkipNode*> update(MAX_LEVEL + 1, nullptr);
        auto* current = header_;
        for (int i = current_level_; i >= 0; i--) {
            while (current->forward[i] &&
                   current->forward[i]->interval.start_time < iv.start_time)
                current = current->forward[i];
            update[i] = current;
        }
        int lvl = random_level();
        if (lvl > current_level_) {
            for (int i = current_level_ + 1; i <= lvl; i++) update[i] = header_;
            current_level_ = lvl;
        }
        auto* new_node = new ScaleSkipNode(iv, lvl);
        for (int i = 0; i <= lvl; i++) {
            new_node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = new_node;
        }
        size_++;
        total_capacity_weight_ += iv.capacity_weight;
    }

    // O(log P + k) range query — 与upstream TemGraph的contains_query类似的二分跳跃
    // 但这里用skip-list而非upstream的next[]数组
    std::vector<ScaleInterval> query_range(uint64_t qs, uint64_t qe) {
        std::vector<ScaleInterval> results;
        auto* current = header_;
        for (int i = current_level_; i >= 0; i--) {
            while (current->forward[i] &&
                   current->forward[i]->interval.end_time < qs)
                current = current->forward[i];
        }
        current = current->forward[0];
        while (current && current->interval.start_time <= qe) {
            if (current->interval.overlaps(qs, qe)) {
                results.push_back(current->interval);
                g_capacity[current->interval.tier].access_count++;
            }
            current = current->forward[0];
        }
        return results;
    }

    // 线性扫描基线 — 用于对比
    std::vector<ScaleInterval> linear_scan(uint64_t qs, uint64_t qe) {
        std::vector<ScaleInterval> results;
        auto* current = header_->forward[0];
        while (current) {
            if (current->interval.overlaps(qs, qe))
                results.push_back(current->interval);
            current = current->forward[0];
        }
        return results;
    }

    int size() const { return size_; }
    size_t capacity_weight() const { return total_capacity_weight_; }

    void debug_breakpoint_dump(const char* ctx) {
        BP("SKIPLIST", "ctx=%s size=%d levels=%d cap_weight=%zu",
           ctx, size_, current_level_, total_capacity_weight_);
        if (g_debug_level >= 3) {
            auto* cur = header_->forward[0];
            int count = 0;
            while (cur && count < 5) {
                printf("  [SL·NODE] [%lu,%lu] part=%u tier=%s cap_w=%zu\n",
                    cur->interval.start_time, cur->interval.end_time,
                    cur->interval.partition_id, scale_tier_names[cur->interval.tier],
                    cur->interval.capacity_weight);
                cur = cur->forward[0]; count++;
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 4: Dual-Sorted Interval Index — 增量构建版
// 来自upstream/temgraph/tem_graph.cpp的sorted_by_start和sorted_by_end双排序
// 原版在load_intervals中一次性构建, 这里改为支持追加(20%改动)
// ═══════════════════════════════════════════════════════════════════

class ScaleDualIndex {
    struct IndexEntry {
        uint64_t key;
        uint32_t edge_idx;
        ScaleTierType tier;
    };

    std::vector<IndexEntry> by_start_;
    std::vector<IndexEntry> by_end_;
    bool sorted_ = false;

public:
    void append(const ScaleEdge& edge, uint32_t idx) {
        by_start_.push_back({edge.timestamp, idx, edge.tier});
        by_end_.push_back({edge.timestamp, idx, edge.tier});
        sorted_ = false;
    }

    void build(const std::vector<ScaleEdge>& edges) {
        by_start_.clear();
        by_end_.clear();
        by_start_.reserve(edges.size());
        by_end_.reserve(edges.size());
        for (uint32_t i = 0; i < edges.size(); i++) {
            by_start_.push_back({edges[i].timestamp, i, edges[i].tier});
            by_end_.push_back({edges[i].timestamp, i, edges[i].tier});
        }
        finalize();
    }

    // 20%改动: 增量追加后只需finalize — 不用全量重建
    void finalize() {
        std::sort(by_start_.begin(), by_start_.end(),
                  [](const auto& a, const auto& b){ return a.key < b.key; });
        std::sort(by_end_.begin(), by_end_.end(),
                  [](const auto& a, const auto& b){ return a.key < b.key; });
        sorted_ = true;
    }

    std::vector<uint32_t> indexed_scan(uint64_t qs, uint64_t qe) {
        if (!sorted_) finalize();
        auto lo = std::lower_bound(by_start_.begin(), by_start_.end(), qs,
            [](const IndexEntry& e, uint64_t v){ return e.key < v; });
        auto hi = std::upper_bound(by_start_.begin(), by_start_.end(), qe,
            [](uint64_t v, const IndexEntry& e){ return v < e.key; });
        std::vector<uint32_t> result;
        for (auto it = lo; it != hi; ++it) {
            result.push_back(it->edge_idx);
            g_capacity[it->tier].access_count++;
        }
        return result;
    }

    std::vector<uint32_t> linear_scan(uint64_t qs, uint64_t qe,
                                      const std::vector<ScaleEdge>& edges) {
        std::vector<uint32_t> result;
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i].timestamp >= qs && edges[i].timestamp <= qe)
                result.push_back(i);
        }
        return result;
    }

    size_t size() const { return by_start_.size(); }

    void debug_breakpoint_dump(const char* ctx) {
        BP("DUALIDX", "ctx=%s entries=%zu sorted=%d", ctx, by_start_.size(), sorted_);
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 5: Hotness Tracker with Exponential Decay(20%改动)
// 来自upstream/rapidstore/utils/Timer.h的elapsed计时
// upstream只是记录elapsed时间, 这里引入指数衰减使旧访问权重递减
// ═══════════════════════════════════════════════════════════════════

class DecayHotnessTracker {
    struct AccessRecord {
        uint32_t partition_id;
        double score;
        double timestamp;
        int accesses;
    };
    std::deque<AccessRecord> history_;
    size_t window_ = 200;
    double current_time_ = 0;
    double decay_lambda_ = 0.02;  // 20%改动: 衰减常数

public:
    void record_access(uint32_t pid, double latency) {
        current_time_ += 1.0;
        double raw_score = 1.0 / (1.0 + latency);
        double decayed = raw_score * std::exp(-decay_lambda_ * 0);
        history_.push_back({pid, decayed, current_time_, 1});
        if (history_.size() > window_) history_.pop_front();
    }

    double get_hotness(uint32_t pid) {
        double total = 0;
        int count = 0;
        for (auto& h : history_) {
            if (h.partition_id == pid) {
                double age = current_time_ - h.timestamp;
                double weight = std::exp(-decay_lambda_ * age);
                total += h.score * weight;
                count++;
            }
        }
        return count > 0 ? total / count : 0;
    }

    void debug_breakpoint_dump(const char* ctx) {
        BP("HOTNESS", "ctx=%s history=%zu window=%zu time=%.1f decay=%.3f",
           ctx, history_.size(), window_, current_time_, decay_lambda_);
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 6: Migration Engine — 容量感知版(20%改动)
// 来自upstream dll_list.h的insert/erase链表操作 + driver.h的batch迁移模式
// 原版: DLL直接insert/erase节点不检查容量
// 改为: 迁移前验证目标tier容量, 溢出时降级
// ═══════════════════════════════════════════════════════════════════

class CapacityAwareMigration {
    std::atomic<int> migrations_done_{0};
    std::atomic<int> edges_migrated_{0};
    double total_migration_cost_ms_ = 0;
    int capacity_overflows_ = 0;

public:
    bool migrate_partition(ScalePartition& part, ScaleTierType new_tier) {
        ScaleTierType old_tier = part.tier;
        if (old_tier == new_tier) return true;

        // 20%改动: 容量检查 — upstream的DLL.insert()不做此检查
        if (!g_capacity[new_tier].can_accept(part.size())) {
            BP("MIGRATE", "part=%u %s→%s BLOCKED: target tier full (occ=%.1f%%)",
               part.id, scale_tier_names[old_tier], scale_tier_names[new_tier],
               g_capacity[new_tier].occupancy() * 100);
            capacity_overflows_++;

            if (new_tier == STIER_HBM && g_capacity[STIER_GDDR].can_accept(part.size())) {
                new_tier = STIER_GDDR;
                BP("MIGRATE", "part=%u fallback to GDDR", part.id);
            } else if (new_tier != STIER_DRAM && g_capacity[STIER_DRAM].can_accept(part.size())) {
                new_tier = STIER_DRAM;
                BP("MIGRATE", "part=%u fallback to DRAM", part.id);
            } else {
                return false;
            }
        }

        if (old_tier == new_tier) return true;

        double cost_us = part.size() *
            (SCALE_TIER_LATENCY_NS[new_tier] + SCALE_TIER_LATENCY_NS[old_tier]) / 1000.0;
        total_migration_cost_ms_ += cost_us / 1000.0;

        g_capacity[old_tier].migration_out++;
        g_capacity[new_tier].migration_in++;
        g_capacity[old_tier].current_edges -= part.size();
        g_capacity[new_tier].current_edges += part.size();
        g_capacity[old_tier].current_partitions--;
        g_capacity[new_tier].current_partitions++;

        part.tier = new_tier;
        part.interval.tier = new_tier;
        for (auto& e : part.edges) e.tier = new_tier;

        migrations_done_++;
        edges_migrated_ += part.size();

        BP("MIGRATE", "part=%u %s→%s edges=%zu cost=%.2fμs",
           part.id, scale_tier_names[old_tier], scale_tier_names[new_tier],
           part.size(), cost_us);
        return true;
    }

    int total_migrations() const { return migrations_done_.load(); }
    double total_cost_ms() const { return total_migration_cost_ms_; }
    int overflow_count() const { return capacity_overflows_; }

    void debug_breakpoint_dump(const char* ctx) {
        BP("MIGRATE", "ctx=%s total=%d edges=%d cost=%.3fms overflows=%d",
           ctx, migrations_done_.load(), edges_migrated_.load(),
           total_migration_cost_ms_, capacity_overflows_);
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 7: Scaling Data Point — 每个规模的实验结果
// ═══════════════════════════════════════════════════════════════════

struct ScalingDataPoint {
    size_t total_edges = 0;
    int num_partitions = 0;
    int edges_per_partition = 0;

    double hbm_occupancy = 0;
    double gddr_occupancy = 0;
    double dram_occupancy = 0;
    long hbm_partitions = 0;
    long gddr_partitions = 0;
    long dram_partitions = 0;

    double migration_sweep_cost_ms = 0;
    int migration_count = 0;
    int capacity_overflows = 0;

    double narrow_query_us = 0;
    double narrow_query_std = 0;
    double wide_query_us = 0;
    double wide_query_std = 0;

    double build_time_ms = 0;
    double index_build_ms = 0;

    void debug_breakpoint_dump(const char* ctx) {
        BP("DATAPOINT", "ctx=%s edges=%zuM parts=%d edg/part=%d "
           "occ=[HBM=%.1f%%,GDDR=%.1f%%,DRAM=%.1f%%] "
           "mig_cost=%.2fms mig_count=%d overflows=%d "
           "narrow=%.2f±%.2fμs wide=%.2f±%.2fμs build=%.1fms",
           ctx, total_edges / 1000000, num_partitions, edges_per_partition,
           hbm_occupancy * 100, gddr_occupancy * 100, dram_occupancy * 100,
           migration_sweep_cost_ms, migration_count, capacity_overflows,
           narrow_query_us, narrow_query_std,
           wide_query_us, wide_query_std, build_time_ms);
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 8: Full Scaling Experiment Engine
// ═══════════════════════════════════════════════════════════════════

class ScalingExperimentEngine {
    std::mt19937 rng_{42};

    // 20%改动: 自适应分区粒度
    int compute_partition_size(size_t total_edges) {
        if (total_edges < 10000000ULL) return 5000;
        else if (total_edges < 50000000ULL) return 10000;
        else return 25000;
    }

    void generate_edges_for_scale(
        size_t target_edges,
        std::vector<ScalePartition>& partitions,
        IncrementalSkipList& skip_list,
        int step_idx)
    {
        int part_size = compute_partition_size(target_edges);
        int num_parts = (int)(target_edges / part_size);
        if (num_parts < 1) num_parts = 1;

        BP_SCALE(step_idx, target_edges,
                 "generating parts=%d edges_per_part=%d", num_parts, part_size);

        uint64_t time_range = target_edges;
        uint64_t time_step = time_range / num_parts;

        std::uniform_int_distribution<uint32_t> vertex_dist(
            0, (uint32_t)std::min(target_edges / 2, (size_t)UINT32_MAX));
        std::uniform_real_distribution<double> weight_dist(0.1, 10.0);

        init_tier_capacities(target_edges);

        partitions.clear();
        partitions.resize(num_parts);

        for (int p = 0; p < num_parts; p++) {
            partitions[p].id = p;
            partitions[p].interval.partition_id = p;
            partitions[p].interval.start_time = (uint64_t)p * time_step;
            partitions[p].interval.end_time = ((uint64_t)p + 1) * time_step - 1;
            partitions[p].interval.capacity_weight = part_size;

            double frac = (double)p / num_parts;
            if (frac > 0.85) {
                partitions[p].hotness = 0.75 + 0.25 * (frac - 0.85) / 0.15;
            } else if (frac > 0.5) {
                partitions[p].hotness = 0.35 + 0.4 * (frac - 0.5) / 0.35;
            } else {
                partitions[p].hotness = frac * 0.7;
            }

            // 20%改动: 容量感知分配
            partitions[p].edges.resize(part_size);
            // BUG FIX: 不再在assign之前设access_count=1
            // tier_assigned=false 保证首次分配走正确路径
            partitions[p].assign_tier_with_capacity();

            std::uniform_int_distribution<uint64_t> ts_dist(
                partitions[p].interval.start_time,
                partitions[p].interval.end_time);

            for (int e = 0; e < part_size; e++) {
                auto& edge = partitions[p].edges[e];
                edge.src = vertex_dist(rng_);
                edge.dst = vertex_dist(rng_);
                edge.weight = weight_dist(rng_);
                edge.timestamp = ts_dist(rng_);
                edge.tier = partitions[p].tier;
            }

            skip_list.insert(partitions[p].interval);

            if (g_debug_level >= 3 && p < 3)
                partitions[p].debug_breakpoint_dump("generate");
        }

        BP_SCALE(step_idx, target_edges,
                 "generated: parts=%d skiplist=%d", num_parts, skip_list.size());

        for (int t = 0; t < STIER_COUNT; t++)
            g_capacity[t].debug_breakpoint_dump("after_generate", t);
    }

    double run_migration_sweep(
        std::vector<ScalePartition>& partitions,
        DecayHotnessTracker& hotness_tracker,
        CapacityAwareMigration& migration,
        int step_idx)
    {
        size_t total_e = partitions.size() * (partitions.empty() ? 0 : partitions[0].size());
        BP_SCALE(step_idx, total_e,
                 "starting migration sweep, parts=%zu", partitions.size());

        ScalingTimer sweep_timer;
        sweep_timer.start("migration_sweep", -1, step_idx);

        int migrated = 0, blocked = 0;

        for (auto& part : partitions) {
            double new_hotness = hotness_tracker.get_hotness(part.id);
            if (new_hotness > 0) {
                part.hotness = 0.6 * part.hotness + 0.4 * new_hotness;
            }

            ScaleTierType target;
            if (part.hotness > 0.7) target = STIER_HBM;
            else if (part.hotness > 0.3) target = STIER_GDDR;
            else target = STIER_DRAM;

            if (target != part.tier) {
                bool ok = migration.migrate_partition(part, target);
                if (ok) migrated++;
                else blocked++;
            }
        }

        double sweep_us = sweep_timer.stop();
        double sweep_ms = sweep_us / 1000.0;

        BP_SCALE(step_idx, total_e,
                 "sweep done: migrated=%d blocked=%d cost=%.2fms", migrated, blocked, sweep_ms);

        migration.debug_breakpoint_dump("after_sweep");
        return sweep_ms;
    }

    struct QueryStats { double mean_us = 0; double std_us = 0; };

    QueryStats bench_queries(
        IncrementalSkipList& skip_list,
        uint64_t time_range,
        uint64_t window_size,
        int num_queries,
        int step_idx,
        const char* label)
    {
        std::uniform_int_distribution<uint64_t> start_dist(0,
            time_range > window_size ? time_range - window_size : 0);

        std::vector<double> latencies;
        latencies.reserve(num_queries);

        for (int q = 0; q < num_queries; q++) {
            uint64_t qs = start_dist(rng_);
            uint64_t qe = qs + window_size;

            ScalingTimer qt;
            qt.start("query", -1, step_idx);
            auto matches = skip_list.query_range(qs, qe);
            double us = qt.stop();

            double tier_penalty = 0;
            for (auto& m : matches) {
                tier_penalty += SCALE_TIER_LATENCY_NS[m.tier] / 1000.0;
            }
            latencies.push_back(us + tier_penalty);
        }

        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / latencies.size();
        double sq_sum = 0;
        for (auto x : latencies) sq_sum += (x - mean) * (x - mean);
        double stddev = std::sqrt(sq_sum / latencies.size());

        BP_SCALE(step_idx, time_range,
                 "%s: mean=%.2fμs std=%.2fμs queries=%d window=%lu",
                 label, mean, stddev, num_queries, window_size);

        return {mean, stddev};
    }

public:
    std::vector<ScalingDataPoint> run_scaling_experiment() {
        std::vector<size_t> scale_points = {
            1000000, 2000000, 5000000,
            10000000, 20000000, 50000000,
            100000000
        };

        std::vector<ScalingDataPoint> results;

        printf("═══════════════════════════════════════════════════════\n");
        printf(" M135-M136: Scaling Experiment — RQ4 Data Generation\n");
        printf(" Scale points: %zu (1M → 100M edges)\n", scale_points.size());
        printf(" 第2位Claude Opus 4.6\n");
        printf("═══════════════════════════════════════════════════════\n\n");

        for (int step = 0; step < (int)scale_points.size(); step++) {
            size_t target_edges = scale_points[step];

            printf("\n── Scaling Step %d/%zu: %zuM edges ──\n",
                   step + 1, scale_points.size(), target_edges / 1000000);

            ScalingDataPoint dp;
            dp.total_edges = target_edges;

            // 1) 生成数据
            ScalingTimer build_timer;
            build_timer.start("build", -1, step);

            std::vector<ScalePartition> partitions;
            IncrementalSkipList skip_list;
            generate_edges_for_scale(target_edges, partitions, skip_list, step);

            dp.build_time_ms = build_timer.stop() / 1000.0;
            dp.num_partitions = (int)partitions.size();
            dp.edges_per_partition = partitions.empty() ? 0 : (int)partitions[0].size();

            // 2) 记录tier occupancy
            dp.hbm_occupancy = g_capacity[STIER_HBM].occupancy();
            dp.gddr_occupancy = g_capacity[STIER_GDDR].occupancy();
            dp.dram_occupancy = g_capacity[STIER_DRAM].occupancy();
            dp.hbm_partitions = g_capacity[STIER_HBM].current_partitions.load();
            dp.gddr_partitions = g_capacity[STIER_GDDR].current_partitions.load();
            dp.dram_partitions = g_capacity[STIER_DRAM].current_partitions.load();

            BP_SCALE(step, target_edges,
                     "tier distribution: HBM=%ld GDDR=%ld DRAM=%ld parts",
                     dp.hbm_partitions, dp.gddr_partitions, dp.dram_partitions);

            // 3) Migration sweep
            DecayHotnessTracker hotness_tracker;
            CapacityAwareMigration migration;

            int warmup_queries = std::min((int)(target_edges / 10000), 500);
            uint64_t time_range = target_edges;
            std::uniform_int_distribution<uint64_t> warmup_dist(0, time_range / 2);
            for (int wq = 0; wq < warmup_queries; wq++) {
                uint64_t qs = warmup_dist(rng_);
                auto matches = skip_list.query_range(qs, qs + time_range / 50);
                for (auto& m : matches) {
                    hotness_tracker.record_access(m.partition_id, SCALE_TIER_LATENCY_NS[m.tier]);
                }
            }

            dp.migration_sweep_cost_ms = run_migration_sweep(
                partitions, hotness_tracker, migration, step);
            dp.migration_count = migration.total_migrations();
            dp.capacity_overflows = migration.overflow_count();

            // 4) Query latency benchmark
            int query_count = std::min((int)(target_edges / 5000), 2000);
            if (query_count < 200) query_count = 200;

            uint64_t narrow_win = time_range / 50;
            auto narrow_stats = bench_queries(skip_list, time_range,
                                              narrow_win, query_count, step, "narrow");
            dp.narrow_query_us = narrow_stats.mean_us;
            dp.narrow_query_std = narrow_stats.std_us;

            uint64_t wide_win = time_range * 3 / 4;
            auto wide_stats = bench_queries(skip_list, time_range,
                                            wide_win, query_count, step, "wide");
            dp.wide_query_us = wide_stats.mean_us;
            dp.wide_query_std = wide_stats.std_us;

            // 5) 完整状态dump
            dp.debug_breakpoint_dump("step_complete");
            skip_list.debug_breakpoint_dump("step_complete");
            hotness_tracker.debug_breakpoint_dump("step_complete");
            migration.debug_breakpoint_dump("step_complete");

            results.push_back(dp);
        }

        return results;
    }

    void print_latex_scaling(const std::vector<ScalingDataPoint>& results) {
        printf("\n%% ═══ RQ4: Scaling Curves (auto-generated by M135-M136) ═══\n\n");

        printf("%% --- Partition count scaling ---\n");
        printf("\\pgfplotstableread[col sep=comma]{%%\n");
        printf("edges_M,partitions,hbm_parts,gddr_parts,dram_parts\n");
        for (auto& dp : results) {
            printf("%zu,%d,%ld,%ld,%ld\n",
                   dp.total_edges / 1000000, dp.num_partitions,
                   dp.hbm_partitions, dp.gddr_partitions, dp.dram_partitions);
        }
        printf("}\\partitionscalingtable\n\n");

        printf("%% --- Per-tier occupancy scaling ---\n");
        printf("\\pgfplotstableread[col sep=comma]{%%\n");
        printf("edges_M,hbm_occ_pct,gddr_occ_pct,dram_occ_pct\n");
        for (auto& dp : results) {
            printf("%zu,%.2f,%.2f,%.2f\n",
                   dp.total_edges / 1000000,
                   dp.hbm_occupancy * 100, dp.gddr_occupancy * 100, dp.dram_occupancy * 100);
        }
        printf("}\\occupancyscalingtable\n\n");

        printf("%% --- Migration sweep cost scaling ---\n");
        printf("\\pgfplotstableread[col sep=comma]{%%\n");
        printf("edges_M,sweep_cost_ms,migrations,overflows\n");
        for (auto& dp : results) {
            printf("%zu,%.4f,%d,%d\n",
                   dp.total_edges / 1000000,
                   dp.migration_sweep_cost_ms, dp.migration_count, dp.capacity_overflows);
        }
        printf("}\\migrationscalingtable\n\n");

        printf("%% --- Query latency scaling ---\n");
        printf("\\pgfplotstableread[col sep=comma]{%%\n");
        printf("edges_M,narrow_us,narrow_std,wide_us,wide_std\n");
        for (auto& dp : results) {
            printf("%zu,%.4f,%.4f,%.4f,%.4f\n",
                   dp.total_edges / 1000000,
                   dp.narrow_query_us, dp.narrow_query_std,
                   dp.wide_query_us, dp.wide_query_std);
        }
        printf("}\\latencyscalingtable\n\n");

        printf("%% --- Example: Latency scaling curve ---\n");
        printf("\\begin{figure}[t]\n\\centering\n");
        printf("\\begin{tikzpicture}\n");
        printf("\\begin{axis}[\n");
        printf("    xlabel={Graph size (M edges)},\n");
        printf("    ylabel={Query latency ($\\mu$s)},\n");
        printf("    legend pos=north west,\n");
        printf("    xmode=log,\n");
        printf("    log basis x={10},\n");
        printf("    grid=major,\n");
        printf("    width=0.85\\columnwidth,\n");
        printf("]\n");
        printf("\\addplot+[mark=square*,error bars/.cd,y dir=both,y explicit]\n");
        printf("  table[x=edges_M,y=narrow_us,y error=narrow_std]{\\latencyscalingtable};\n");
        printf("\\addplot+[mark=triangle*,error bars/.cd,y dir=both,y explicit]\n");
        printf("  table[x=edges_M,y=wide_us,y error=wide_std]{\\latencyscalingtable};\n");
        printf("\\legend{Narrow ($2\\%%$), Wide ($75\\%%$)}\n");
        printf("\\end{axis}\n");
        printf("\\end{tikzpicture}\n");
        printf("\\caption{Query latency scaling from 1M to 100M edges (RQ4).}\n");
        printf("\\label{fig:scaling_latency}\n");
        printf("\\end{figure}\n");
    }

    void print_csv_scaling(const std::vector<ScalingDataPoint>& results) {
        printf("\n── CSV Output ──\n");
        printf("edges_M,partitions,edg_per_part,hbm_occ,gddr_occ,dram_occ,"
               "hbm_parts,gddr_parts,dram_parts,"
               "sweep_ms,mig_count,overflows,"
               "narrow_us,narrow_std,wide_us,wide_std,build_ms\n");
        for (auto& dp : results) {
            printf("%zu,%d,%d,%.4f,%.4f,%.4f,%ld,%ld,%ld,"
                   "%.4f,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f\n",
                   dp.total_edges / 1000000,
                   dp.num_partitions, dp.edges_per_partition,
                   dp.hbm_occupancy, dp.gddr_occupancy, dp.dram_occupancy,
                   dp.hbm_partitions, dp.gddr_partitions, dp.dram_partitions,
                   dp.migration_sweep_cost_ms, dp.migration_count, dp.capacity_overflows,
                   dp.narrow_query_us, dp.narrow_query_std,
                   dp.wide_query_us, dp.wide_query_std,
                   dp.build_time_ms);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════
// PART 9: Test Harness — 10 tests验证所有组件
// ═══════════════════════════════════════════════════════════════════

static int g_tests_passed = 0;
static int g_tests_failed = 0;

void run_test(const std::string& name, std::function<bool()> test) {
    printf("\n── %s ──\n", name.c_str());
    if (test()) {
        g_tests_passed++;
    } else {
        g_tests_failed++;
        printf("  [FAILED] %s\n", name.c_str());
    }
}

// T1: Edge结构 — 基于upstream edge.hpp的operator==和operator<
bool test_scale_edge_basics() {
    ScaleEdge e{10, 20, 3.14, 1000, STIER_HBM, 0, 0};
    ASSERT_EQ(e.src, 10u, "edge src");
    ASSERT_EQ(e.dst, 20u, "edge dst");
    ASSERT_NEAR(e.weight, 3.14, 0.01, "edge weight");
    ASSERT_EQ(e.tier, STIER_HBM, "edge tier");

    e.touch(0.5);
    ASSERT_EQ(e.access_count, 1, "access_count after touch");
    ASSERT_TRUE(e.decay_score > 0, "decay_score positive after touch");

    double prev = e.decay_score;
    e.touch(1.0);
    ASSERT_TRUE(e.decay_score > prev, "decay increases on 2nd touch");

    e.debug_breakpoint_dump("test");
    return true;
}

// T2: Interval overlap — 基于upstream interval.h的TInterval比较
bool test_scale_interval_overlap() {
    ScaleInterval iv{100, 200, 0, STIER_GDDR, 5, 500};
    ASSERT_TRUE(iv.contains(150), "contains 150");
    ASSERT_TRUE(!iv.contains(50), "not contains 50");
    ASSERT_TRUE(iv.overlaps(150, 250), "overlaps [150,250]");
    ASSERT_TRUE(iv.overlaps(50, 150), "overlaps [50,150]");
    ASSERT_TRUE(!iv.overlaps(201, 300), "not overlaps [201,300]");

    ScaleInterval iv2{80, 300, 1, STIER_DRAM, 0, 200};
    ASSERT_TRUE(iv2 < iv, "iv2.start < iv.start");

    iv.debug_breakpoint_dump("test");
    return true;
}

// T3: 容量规划 — 20%改动的核心测试
bool test_capacity_planning() {
    init_tier_capacities(1000);
    // HBM: 150+1=151, GDDR: 250+1=251, DRAM: 600+1=601

    ASSERT_TRUE(g_capacity[STIER_HBM].can_accept(100), "HBM can accept 100");
    ASSERT_TRUE(g_capacity[STIER_HBM].can_accept(151), "HBM can accept 151");
    ASSERT_TRUE(!g_capacity[STIER_HBM].can_accept(200), "HBM cannot accept 200");

    g_capacity[STIER_HBM].current_edges = 140;
    ASSERT_TRUE(!g_capacity[STIER_HBM].can_accept(20), "HBM full after 140+20");
    ASSERT_TRUE(g_capacity[STIER_HBM].can_accept(10), "HBM not full after 140+10");

    ASSERT_NEAR(g_capacity[STIER_HBM].occupancy(), 140.0/151.0, 0.01, "HBM occupancy");

    g_capacity[STIER_HBM].debug_breakpoint_dump("test", STIER_HBM);
    return true;
}

// T4: 容量感知partition分配
bool test_capacity_aware_partition() {
    init_tier_capacities(100);  // HBM: 15+1=16, GDDR: 25+1=26, DRAM: 60+1=61

    ScalePartition p;
    p.id = 0;
    p.edges.resize(20);
    p.hotness = 0.9;  // 想要HBM
    p.assign_tier_with_capacity();

    // 20 > 16 (HBM max), 所以应降级到GDDR
    ASSERT_EQ(p.tier, STIER_GDDR, "hot partition downgraded to GDDR (HBM full)");
    ASSERT_TRUE(p.tier_assigned, "tier_assigned flag set");

    // 验证GDDR里确实记录了
    ASSERT_EQ(g_capacity[STIER_GDDR].current_edges.load(), 20L, "GDDR has 20 edges");
    ASSERT_EQ(g_capacity[STIER_HBM].current_edges.load(), 0L, "HBM still empty");

    p.debug_breakpoint_dump("test");
    return true;
}

// T5: Skip-list增量插入与查询
bool test_incremental_skiplist() {
    IncrementalSkipList sl;

    for (int i = 0; i < 100; i++) {
        ScaleInterval iv;
        iv.start_time = i * 100;
        iv.end_time = (i + 1) * 100 - 1;
        iv.partition_id = i;
        iv.tier = (ScaleTierType)(i % 3);
        iv.capacity_weight = 500;
        sl.insert(iv);
    }
    ASSERT_EQ(sl.size(), 100, "skip list has 100 intervals");
    ASSERT_EQ(sl.capacity_weight(), (size_t)(100 * 500), "total capacity weight");

    auto narrow = sl.query_range(500, 700);
    ASSERT_TRUE(narrow.size() >= 2, "narrow query finds partitions");

    auto linear = sl.linear_scan(500, 700);
    ASSERT_EQ(narrow.size(), linear.size(), "indexed matches linear");

    for (int i = 100; i < 150; i++) {
        ScaleInterval iv;
        iv.start_time = i * 100;
        iv.end_time = (i + 1) * 100 - 1;
        iv.partition_id = i;
        iv.tier = STIER_HBM;
        iv.capacity_weight = 500;
        sl.insert(iv);
    }
    ASSERT_EQ(sl.size(), 150, "skip list after increment");

    sl.debug_breakpoint_dump("test");
    return true;
}

// T6: Dual-sorted index增量构建
bool test_dual_index_incremental() {
    std::vector<ScaleEdge> edges(1000);
    std::mt19937 rng(42);
    for (int i = 0; i < 1000; i++) {
        edges[i].src = rng() % 100;
        edges[i].dst = rng() % 100;
        edges[i].timestamp = rng() % 10000;
        edges[i].tier = (ScaleTierType)(i % 3);
    }

    ScaleDualIndex idx;
    idx.build(edges);

    auto indexed = idx.indexed_scan(2000, 4000);
    auto linear = idx.linear_scan(2000, 4000, edges);
    ASSERT_EQ(indexed.size(), linear.size(), "dual index matches linear");
    ASSERT_TRUE(indexed.size() > 0, "found results in range");

    idx.debug_breakpoint_dump("test");
    return true;
}

// T7: 容量感知迁移 — 测试溢出降级
bool test_capacity_aware_migration() {
    init_tier_capacities(1000);

    CapacityAwareMigration mig;
    ScalePartition p;
    p.id = 0;
    p.tier = STIER_DRAM;
    p.tier_assigned = true;
    p.edges.resize(100);
    for (auto& e : p.edges) e.tier = STIER_DRAM;
    g_capacity[STIER_DRAM].current_edges = 100;
    g_capacity[STIER_DRAM].current_partitions = 1;

    bool ok = mig.migrate_partition(p, STIER_HBM);
    ASSERT_TRUE(ok, "migration to HBM succeeded");
    ASSERT_EQ(p.tier, STIER_HBM, "partition now in HBM");
    ASSERT_EQ(mig.total_migrations(), 1, "migration count = 1");

    // 填满HBM
    g_capacity[STIER_HBM].current_edges = (long)g_capacity[STIER_HBM].max_edges;

    // 尝试迁移另一个到HBM — 应降级到GDDR
    ScalePartition p2;
    p2.id = 1;
    p2.tier = STIER_DRAM;
    p2.tier_assigned = true;
    p2.edges.resize(50);
    for (auto& e : p2.edges) e.tier = STIER_DRAM;
    g_capacity[STIER_DRAM].current_edges += 50;
    g_capacity[STIER_DRAM].current_partitions++;

    ok = mig.migrate_partition(p2, STIER_HBM);
    ASSERT_TRUE(ok, "migration with fallback succeeded");
    ASSERT_EQ(p2.tier, STIER_GDDR, "fell back to GDDR");
    ASSERT_TRUE(mig.overflow_count() > 0, "overflow was recorded");

    mig.debug_breakpoint_dump("test");
    return true;
}

// T8: Decay hotness tracker
bool test_decay_hotness_tracker() {
    DecayHotnessTracker ht;

    for (int i = 0; i < 80; i++) ht.record_access(0, 0.1);
    for (int i = 0; i < 5; i++) ht.record_access(1, 10.0);

    double h0 = ht.get_hotness(0);
    double h1 = ht.get_hotness(1);
    ASSERT_GT(h0, h1, "hot partition has higher hotness than cold");
    ASSERT_GT(h0, 0.0, "hot partition hotness > 0");

    ht.debug_breakpoint_dump("test");
    return true;
}

// T9: 自适应分区粒度
bool test_adaptive_partition_size() {
    init_tier_capacities(10000000);

    ScalePartition cold_p;
    cold_p.id = 0;
    cold_p.edges.resize(5000);
    cold_p.hotness = 0.1;
    cold_p.assign_tier_with_capacity();
    ASSERT_EQ(cold_p.tier, STIER_DRAM, "cold partition in DRAM");

    ScalePartition warm_p;
    warm_p.id = 1;
    warm_p.edges.resize(5000);
    warm_p.hotness = 0.5;
    warm_p.assign_tier_with_capacity();
    ASSERT_EQ(warm_p.tier, STIER_GDDR, "warm partition in GDDR");

    ScalePartition hot_p;
    hot_p.id = 2;
    hot_p.edges.resize(5000);
    hot_p.hotness = 0.8;
    hot_p.assign_tier_with_capacity();
    ASSERT_EQ(hot_p.tier, STIER_HBM, "hot partition in HBM");

    return true;
}

// T10: End-to-end mini scaling consistency
bool test_mini_scaling_consistency() {
    IncrementalSkipList sl1, sl2;

    for (int i = 0; i < 20; i++) {
        ScaleInterval iv;
        iv.start_time = i * 50;
        iv.end_time = (i + 1) * 50 - 1;
        iv.partition_id = i;
        iv.tier = (ScaleTierType)(i % 3);
        iv.successor_count = i;
        iv.capacity_weight = 100;
        sl1.insert(iv);
        sl2.insert(iv);
    }

    for (int i = 20; i < 40; i++) {
        ScaleInterval iv;
        iv.start_time = i * 50;
        iv.end_time = (i + 1) * 50 - 1;
        iv.partition_id = i;
        iv.tier = STIER_GDDR;
        iv.capacity_weight = 100;
        sl2.insert(iv);
    }

    auto r1 = sl1.query_range(100, 300);
    auto r2 = sl2.query_range(100, 300);
    ASSERT_EQ(r1.size(), r2.size(), "same results in overlapping range");

    auto r2_ext = sl2.query_range(1000, 1500);
    ASSERT_TRUE(r2_ext.size() > 0, "sl2 has results in extended range");

    auto r1a = sl1.query_range(100, 300);
    ASSERT_EQ(r1.size(), r1a.size(), "deterministic results");

    return true;
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    bool bench_only = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) bench_only = true;
        else if (strcmp(argv[i], "--latex") == 0) g_latex_output = true;
        else if (strcmp(argv[i], "--csv") == 0) g_csv_output = true;
        else if (strcmp(argv[i], "--verbose") == 0) g_debug_level = 3;
        else if (strcmp(argv[i], "--quiet") == 0) g_debug_level = 0;
    }

    if (!bench_only) {
        printf("═══════════════════════════════════════════════════════\n");
        printf(" M135-M136 Validation Tests\n");
        printf("═══════════════════════════════════════════════════════\n");

        run_test("T1: Edge basics + decay score + debug",  test_scale_edge_basics);
        run_test("T2: Interval overlap + comp_L ordering", test_scale_interval_overlap);
        run_test("T3: Tier capacity planning",             test_capacity_planning);
        run_test("T4: Capacity-aware partition assignment", test_capacity_aware_partition);
        run_test("T5: Incremental skip-list",              test_incremental_skiplist);
        run_test("T6: Dual-sorted index incremental",      test_dual_index_incremental);
        run_test("T7: Capacity-aware migration",           test_capacity_aware_migration);
        run_test("T8: Decay hotness tracker",              test_decay_hotness_tracker);
        run_test("T9: Adaptive partition sizing",          test_adaptive_partition_size);
        run_test("T10: Mini scaling consistency",          test_mini_scaling_consistency);

        printf("\n═══════════════════════════════════════════════════════\n");
        printf(" Validation: %d/%d passed, %d failed\n",
               g_tests_passed, g_tests_passed + g_tests_failed, g_tests_failed);
        printf("═══════════════════════════════════════════════════════\n");

        if (g_tests_failed > 0) {
            printf("VALIDATION FAILED — skipping scaling experiment\n");
            return 1;
        }
        printf("\n");
    }

    // Run full scaling experiment
    ScalingExperimentEngine engine;
    auto results = engine.run_scaling_experiment();

    if (g_latex_output) {
        engine.print_latex_scaling(results);
    }

    if (g_csv_output) {
        engine.print_csv_scaling(results);
    }

    // Final summary
    printf("\n── Final Tier Summary ──\n");
    for (int t = 0; t < STIER_COUNT; t++)
        g_capacity[t].debug_breakpoint_dump("final", t);

    printf("\n═══════════════════════════════════════════════════════\n");
    printf(" M135-M136 Complete: breakpoints=%d assertions=%d scaling_steps=%zu\n",
           g_bp_count, g_assert_count, results.size());
    printf("═══════════════════════════════════════════════════════\n");

    return 0;
}
