#ifndef PHILEMON_DYNAMIC_REBALANCER_HPP
#define PHILEMON_DYNAMIC_REBALANCER_HPP
/**
 * dynamic_rebalancer.hpp — 双水位 + 迁移代价感知的动态rebalancer
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/driver.h  (1577行)
 *     → execute_mixed_reader_writer():
 *       · chunk_size = (total + n_threads - 1) / n_threads  (行1060)
 *       · writer lambda: init_thread → for(j=start;j<end;j++) remove+insert  (行1096-1113)
 *       · reader lambda: init_thread → snapshot_clone → page_rank  (行1121-1137)
 *       · bind_thread_to_core(threads[i], i%hardware_concurrency)  (行1115)
 *       · Barrier arrive_and_wait + sleep(2) 同步  (行49-60, 1119)
 *       · thread_time/thread_speed/check_point 统计  (行1063-1067)
 *       · 100% 保留: chunk分发, worker lambda, core binding, barrier
 *
 *     → page_rank():
 *       · outgoing_contrib[src] = result[src] / degree  (行860)
 *       · dangling_sum accumulate → /= size  (行865)
 *       · result[v] = base + damping*(incoming + dangling)  (行878)
 *       · 100% 保留: 迭代框架, contrib计算
 *
 *     → UnionFind: root[], find(path compress), unite  (行806-830)
 *       · 100% 保留: union-find结构
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp  (537行)
 *     → finish_commit() CAS: while(!CAS(target,timestamp)){}  (行33-36)
 *     → get_write_timestamp() fetch_add(1)  (行29-31)
 *     → 100% 保留: CAS追赶 + 原子递增
 *
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → WriterTraceBlock stack pool: push/pop  (行75-90)
 *     → 100% 保留
 *
 * 算法修改 (~20%):
 *   [MOD-1] PageRank式迭代收敛 → 梯度下降迁移:
 *     upstream的page_rank是 outgoing_contrib → incoming → 新score, 固定迭代次数.
 *     这里把同样的"迭代+贡献累加"框架改为: 每tier有一个"压力值",
 *     迭代不是算PageRank而是做梯度下降 — 每轮计算梯度
 *     grad[t] = utilization[t] - target, 然后 pressure[t] -= lr*grad[t],
 *     直到所有tier的|grad|<threshold 收敛. 收敛后按pressure排序决定迁移.
 *     upstream: 固定num_iterations, 不看收敛;  这里: 自适应收敛+梯度驱动.
 *
 *   [MOD-2] UnionFind组件合并 → 亲和度感知分组:
 *     upstream的wcc用UnionFind把连通vertex合为同一component.
 *     这里用同样的UnionFind + path compression, 但不是找连通性,
 *     而是把访问模式相似的vertex合为"迁移组" — 如果vertex A和B在
 *     同一时间窗口内都被访问, unite(A,B), 迁移时整组一起搬,
 *     避免breaking access locality.
 *     upstream: unite条件是edge存在;  这里: unite条件是时间窗口共现.
 *
 *   [MOD-3] 固定batch大小 → 代价感知动态sizing:
 *     upstream的chunk_size = total / n_threads, 每个worker处理固定量.
 *     这里: batch_size = f(gradient_magnitude, tier_bandwidth),
 *     梯度越陡 → batch越大(紧急), 但受限于PCIe带宽估计.
 *     upstream: 不考虑带宽;  这里: batch_bytes <= bw * time_budget.
 *
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <stack>
#include <queue>
#include <algorithm>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <cmath>
#include <cassert>
#include <numeric>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"

namespace philemon {
namespace rebalance {

// ─── upstream Barrier 100% 保留 (driver.h 行49-60) ───
class RebalanceBarrier {
    std::mutex mtx_;
    std::condition_variable cv_;
    std::size_t count_;
    std::atomic<std::size_t> waiting_{0};
public:
    explicit RebalanceBarrier(std::size_t count) : count_(count) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        std::size_t w = ++waiting_;
        if (w == count_) {
            waiting_.store(0);
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return waiting_.load() == 0; });
        }
    }
};

// ─── upstream bind_thread_to_core 100% 保留 (driver.h 行36-43) ───
static void bind_thread_to_core(std::thread& t, int core_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(),
                                     sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::fprintf(stderr, "Error pthread_setaffinity_np: %d\n", rc);
    }
#endif
    (void)t; (void)core_id;
}

// ─── Tier物理描述 ───
struct TierDesc {
    uint64_t capacity_bytes;
    double   bandwidth_gbps;       // PCIe/HBM bandwidth estimate
    std::atomic<uint64_t> used_bytes{0};
    std::atomic<uint64_t> vertex_count{0};

    double utilization() const {
        return capacity_bytes > 0
            ? static_cast<double>(used_bytes.load()) / capacity_bytes : 0;
    }
    void dump(int idx) const {
        std::printf("    tier[%d] %luMB/%luMB (%.1f%%) bw=%.1fGB/s vtx=%lu\n",
                    idx,
                    (unsigned long)(used_bytes.load()>>20),
                    (unsigned long)(capacity_bytes>>20),
                    utilization()*100, bandwidth_gbps,
                    (unsigned long)vertex_count.load());
    }
};

// ─── [MOD-2] 亲和度 UnionFind — upstream的UnionFind结构100%保留 ───
// upstream (driver.h 行806-830): root[], find(path compress), unite
// 改动: unite的条件不是"边存在", 而是"同一时间窗口内共同被访问"
class AffinityUnionFind {
public:
    std::vector<uint64_t> root_;
    std::vector<uint64_t> rank_;   // union by rank (upstream只有简单unite)
    uint64_t n_;

    AffinityUnionFind() : n_(0) {}
    explicit AffinityUnionFind(uint64_t size) : n_(size) {
        root_.resize(size);
        rank_.resize(size, 0);
        for (uint64_t i = 0; i < size; i++) root_[i] = i;
    }

    // upstream find + path compression 100%保留
    uint64_t find(uint64_t x) {
        if (x >= n_) return x;
        if (x == root_[x]) return x;
        return root_[x] = find(root_[x]);
    }

    // upstream unite保留, 但加了union-by-rank (upstream没有rank)
    void unite(uint64_t x, uint64_t y) {
        uint64_t rx = find(x), ry = find(y);
        if (rx == ry) return;
        // [MOD] union by rank: upstream直接 root[ry] = rx
        if (rank_[rx] < rank_[ry]) std::swap(rx, ry);
        root_[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
    }

    // 获取同组所有vertex
    std::vector<uint64_t> group_of(uint64_t x) {
        uint64_t rx = find(x);
        std::vector<uint64_t> grp;
        for (uint64_t i = 0; i < n_; i++) {
            if (find(i) == rx) grp.push_back(i);
        }
        return grp;
    }

    void dump() const {
        if (n_ == 0) return;
        // 统计组数
        std::vector<bool> seen(n_, false);
        uint64_t groups = 0;
        // 注意: find会修改root_, 但dump是const
        // 用一个临时copy
        auto tmp = root_;
        std::function<uint64_t(uint64_t)> find_tmp =
            [&](uint64_t x) -> uint64_t {
                return tmp[x] == x ? x : (tmp[x] = find_tmp(tmp[x]));
            };
        for (uint64_t i = 0; i < n_; i++) {
            uint64_t r = find_tmp(i);
            if (!seen[r]) { seen[r] = true; groups++; }
        }
        std::printf("    [AffinityUF] n=%lu groups=%lu\n",
                    (unsigned long)n_, (unsigned long)groups);
    }
};

// ─── MigrationBatch ───
struct MigrationBatch {
    uint64_t batch_id;
    uint64_t epoch;
    uint8_t  from_tier;
    uint8_t  to_tier;
    std::vector<uint64_t> vertices;
    uint64_t total_bytes = 0;
    double   gradient_magnitude = 0;  // 驱动此batch的梯度大小
    enum class Status : uint8_t { PLANNED, EXECUTING, DONE, FAILED };
    std::atomic<Status> status{Status::PLANNED};
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point finished_at;

    MigrationBatch() : batch_id(0), epoch(0), from_tier(0), to_tier(0),
                       created_at(std::chrono::steady_clock::now()) {}
    double duration_ms() const {
        auto end = (status.load() == Status::DONE)
            ? finished_at : std::chrono::steady_clock::now();
        return std::chrono::duration<double,std::milli>(end-created_at).count();
    }
    void dump() const {
        const char* st[] = {"PLANNED","EXEC","DONE","FAIL"};
        std::printf("    [Batch %lu] epoch=%lu %u→%u vtx=%zu bytes=%luKB "
                    "grad=%.3f %s %.1fms\n",
                    (unsigned long)batch_id, (unsigned long)epoch,
                    from_tier, to_tier, vertices.size(),
                    (unsigned long)(total_bytes>>10),
                    gradient_magnitude,
                    st[(int)status.load()], duration_ms());
    }
};

// ─── MigrationBatchPool — upstream WriterTraceBlock stack pool保留 ───
class MigrationBatchPool {
    std::stack<MigrationBatch*> free_pool_;
    std::mutex mu_;
    uint64_t next_id_ = 0;
public:
    MigrationBatch* allocate() {
        std::lock_guard<std::mutex> lk(mu_);
        MigrationBatch* b;
        if (free_pool_.empty()) {
            b = new MigrationBatch();
        } else {
            b = free_pool_.top(); free_pool_.pop();
        }
        b->batch_id = next_id_++;
        b->vertices.clear();
        b->total_bytes = 0;
        b->status.store(MigrationBatch::Status::PLANNED);
        b->created_at = std::chrono::steady_clock::now();
        return b;
    }
    void deallocate(MigrationBatch* b) {
        std::lock_guard<std::mutex> lk(mu_);
        free_pool_.push(b);
    }
    ~MigrationBatchPool() {
        while (!free_pool_.empty()) { delete free_pool_.top(); free_pool_.pop(); }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// DynamicRebalancer
// ═══════════════════════════════════════════════════════════════════════
class DynamicRebalancer {
    static constexpr int NUM_TIERS = 3;

    std::array<TierDesc, NUM_TIERS> tiers_;

    // [MOD-1] 梯度下降的状态
    std::array<double, NUM_TIERS> pressure_{};   // 类比page_rank的result[]
    std::array<double, NUM_TIERS> gradient_{};   // 类比outgoing_contrib[]
    double target_utilization_ = 0.65;
    double learning_rate_ = 0.3;
    double convergence_threshold_ = 0.01;
    int max_iterations_ = 50;

    // [MOD-2] 亲和度分组
    AffinityUnionFind affinity_uf_;
    std::vector<std::pair<uint64_t,uint64_t>> covisit_buffer_;
    std::mutex covisit_mu_;
    static constexpr size_t COVISIT_WINDOW = 64;
    std::vector<uint64_t> recent_access_;
    std::mutex recent_mu_;

    // upstream CAS: epoch追赶 (neo_transaction.cpp 行33-36)
    std::atomic<uint64_t> migration_epoch_{0};
    std::atomic<uint64_t> committed_epoch_{0};

    MigrationBatchPool batch_pool_;

    // 回调
    using MigrateFn = std::function<bool(uint64_t vertex, uint8_t from,
                                          uint8_t to, uint64_t bytes)>;
    MigrateFn migrate_fn_;

    // 后台
    bool running_ = false;
    std::thread monitor_thread_;
    std::mutex monitor_mu_;
    std::condition_variable monitor_cv_;

    // 统计
    std::atomic<uint64_t> total_promotes_{0};
    std::atomic<uint64_t> total_demotes_{0};
    std::atomic<uint64_t> total_bytes_moved_{0};
    std::atomic<uint64_t> total_batches_{0};
    std::atomic<uint64_t> total_gd_iters_{0};

public:
    DynamicRebalancer() {
        tiers_[0].capacity_bytes = 80ULL<<30;  tiers_[0].bandwidth_gbps = 900;
        tiers_[1].capacity_bytes = 48ULL<<30;  tiers_[1].bandwidth_gbps = 200;
        tiers_[2].capacity_bytes = 256ULL<<30; tiers_[2].bandwidth_gbps = 50;
        for (int i = 0; i < NUM_TIERS; i++) pressure_[i] = 0.5;
    }

    void set_tier(int idx, uint64_t cap, double bw) {
        tiers_[idx].capacity_bytes = cap;
        tiers_[idx].bandwidth_gbps = bw;
    }
    void set_target(double t) { target_utilization_ = t; }
    void set_lr(double lr) { learning_rate_ = lr; }
    void set_migrate_fn(MigrateFn fn) { migrate_fn_ = std::move(fn); }

    void update_usage(uint8_t tier, uint64_t used, uint64_t vtx) {
        tiers_[tier].used_bytes.store(used);
        tiers_[tier].vertex_count.store(vtx);
    }

    void init_affinity(uint64_t n_vertices) {
        affinity_uf_ = AffinityUnionFind(n_vertices);
        recent_access_.reserve(COVISIT_WINDOW);
    }

    // ── [MOD-2] 记录访问, 共现的vertex unite到同组 ──
    // upstream的wcc: for(all vertices) edges→unite(src,dst)
    // 这里: 时间窗口内共同出现的vertex → unite
    void record_access(uint64_t vertex) {
        std::lock_guard<std::mutex> lk(recent_mu_);
        // 把当前vertex和窗口内所有vertex做unite
        for (uint64_t prev : recent_access_) {
            if (prev != vertex) {
                affinity_uf_.unite(prev, vertex);
            }
        }
        recent_access_.push_back(vertex);
        if (recent_access_.size() > COVISIT_WINDOW) {
            recent_access_.erase(recent_access_.begin());
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // [MOD-1] 梯度下降迁移决策
    //
    // upstream page_rank的迭代框架:
    //   for(iter) {
    //     for(v) outgoing_contrib[v] = result[v]/degree;   ← 计算贡献
    //     dangling_sum = accumulate(dangling) / size;
    //     for(v) result[v] = base + damping*(incoming+dangling);  ← 更新
    //   }
    //
    // 改为梯度下降:
    //   for(iter) {
    //     for(t) gradient[t] = utilization[t] - target;      ← 计算梯度
    //     coupling = mean(gradient) (类比dangling_sum/size)   ← 全局耦合项
    //     for(t) pressure[t] -= lr*(gradient[t] + coupling);  ← 梯度更新
    //     if(max|gradient| < threshold) break;                ← 收敛判断
    //   }
    // ═══════════════════════════════════════════════════════════════════
    struct GDResult {
        int iterations_used;
        bool converged;
        std::array<double, 3> final_pressure;
        std::array<double, 3> final_gradient;
        int demote_tier;   // -1=不迁, 否则=源tier
        int promote_tier;  // -1=不迁, 否则=源tier
    };

    GDResult run_gradient_descent() {
        GDResult res;
        res.demote_tier = -1;
        res.promote_tier = -1;

        PHILE_DBG(2, "[GD] starting: target=%.2f lr=%.3f thresh=%.4f",
                   target_utilization_, learning_rate_,
                   convergence_threshold_);

        // upstream page_rank: for(iter < num_iterations)
        int iter = 0;
        for (; iter < max_iterations_; iter++) {
            // ── 梯度计算 (类比 outgoing_contrib[v] = result[v]/degree) ──
            double max_abs_grad = 0;
            double grad_sum = 0;
            for (int t = 0; t < NUM_TIERS; t++) {
                gradient_[t] = tiers_[t].utilization() - target_utilization_;
                grad_sum += gradient_[t];
                max_abs_grad = std::max(max_abs_grad, std::abs(gradient_[t]));
            }

            // ── 耦合项 (类比 dangling_sum /= size) ──
            double coupling = grad_sum / NUM_TIERS;

            // ── 压力更新 (类比 result[v] = base + damping*(incoming+dangling)) ──
            for (int t = 0; t < NUM_TIERS; t++) {
                // upstream: result = base_score + damping * (incoming + dangling)
                // 改为:    pressure -= lr * (gradient + coupling)
                double update = learning_rate_ * (gradient_[t] + coupling);
                pressure_[t] -= update;
                // clamp到[0,1]
                pressure_[t] = std::max(0.0, std::min(1.0, pressure_[t]));
            }

            PHILE_DBG(3, "[GD] iter=%d grad=[%.4f,%.4f,%.4f] "
                       "pressure=[%.3f,%.3f,%.3f] max|g|=%.4f",
                       iter,
                       gradient_[0], gradient_[1], gradient_[2],
                       pressure_[0], pressure_[1], pressure_[2],
                       max_abs_grad);

            // ── 收敛检查 (upstream page_rank固定迭代, 这里自适应) ──
            if (max_abs_grad < convergence_threshold_) {
                PHILE_DBG(2, "[GD] converged at iter=%d", iter);
                break;
            }
        }

        res.iterations_used = iter;
        res.converged = (iter < max_iterations_);
        res.final_pressure = pressure_;
        res.final_gradient = gradient_;
        total_gd_iters_.fetch_add(iter);

        // 决策: 梯度最正的tier需要demote, 最负的可以promote
        int most_overloaded = 0, most_underloaded = 0;
        for (int t = 1; t < NUM_TIERS; t++) {
            if (gradient_[t] > gradient_[most_overloaded]) most_overloaded = t;
            if (gradient_[t] < gradient_[most_underloaded]) most_underloaded = t;
        }

        if (gradient_[most_overloaded] > convergence_threshold_ &&
            most_overloaded < NUM_TIERS - 1) {
            res.demote_tier = most_overloaded;
        }
        if (gradient_[most_underloaded] < -convergence_threshold_ &&
            most_underloaded > 0) {
            res.promote_tier = most_underloaded;
        }

        return res;
    }

    // ── [MOD-3] 代价感知batch sizing ──
    // upstream: chunk_size = total / n_threads (固定)
    // 改为: batch_bytes = bandwidth * time_budget, batch_count = batch_bytes / avg_vertex_size
    size_t compute_batch_size(int tier, double gradient_mag,
                               double time_budget_ms = 5.0) const {
        double bw_bytes_per_ms = tiers_[tier].bandwidth_gbps * 1e6; // GB/s → bytes/ms
        double max_bytes = bw_bytes_per_ms * time_budget_ms;

        // 梯度越大 → 用更多带宽预算 (线性缩放)
        double urgency = std::min(1.0, gradient_mag / 0.3);
        double budget_bytes = max_bytes * urgency;

        uint64_t avg_vertex_bytes = 65536;
        if (tiers_[tier].vertex_count.load() > 0) {
            avg_vertex_bytes = tiers_[tier].used_bytes.load()
                               / tiers_[tier].vertex_count.load();
        }

        size_t count = static_cast<size_t>(budget_bytes / avg_vertex_bytes);
        count = std::max(size_t(1), std::min(count, size_t(4096)));

        PHILE_DBG(2, "[BatchSize] tier=%d grad=%.3f urgency=%.2f "
                   "budget=%.0fKB avg_vtx=%luB → count=%zu",
                   tier, gradient_mag, urgency,
                   budget_bytes/1024, (unsigned long)avg_vertex_bytes, count);
        return count;
    }

    // ── 执行迁移 ──
    // 保留upstream的 for(j=start;j<end;j++) 循环 + CAS epoch + checkpoint
    bool execute_batch(MigrationBatch* batch) {
        if (!migrate_fn_) return false;

        // upstream: get_write_timestamp的fetch_add (neo_transaction.cpp 行29)
        uint64_t epoch = migration_epoch_.fetch_add(1,
            std::memory_order_relaxed) + 1;
        batch->epoch = epoch;
        batch->status.store(MigrationBatch::Status::EXECUTING);

        PHILE_DBG(1, "[DynRebal] batch %lu: %zu vtx %u→%u epoch=%lu "
                   "grad=%.3f",
                   (unsigned long)batch->batch_id, batch->vertices.size(),
                   batch->from_tier, batch->to_tier,
                   (unsigned long)epoch, batch->gradient_magnitude);

        // upstream: for(j=start;j<end;j++) { remove+insert } (driver.h 行1096-1113)
        uint64_t success_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t j = 0; j < batch->vertices.size(); j++) {
            uint64_t vtx = batch->vertices[j];
            uint64_t est_bytes = 65536;

            // [MOD-2] 亲和度: 如果vertex属于某个组, 把整组一起迁移
            auto group = affinity_uf_.group_of(vtx);
            for (uint64_t gv : group) {
                bool ok = migrate_fn_(gv, batch->from_tier,
                                       batch->to_tier, est_bytes);
                if (ok) {
                    success_count++;
                    batch->total_bytes += est_bytes;
                    tiers_[batch->from_tier].used_bytes.fetch_sub(est_bytes);
                    tiers_[batch->from_tier].vertex_count.fetch_sub(1);
                    tiers_[batch->to_tier].used_bytes.fetch_add(est_bytes);
                    tiers_[batch->to_tier].vertex_count.fetch_add(1);
                }
            }

            // upstream checkpoint: if((j-start)%100000==0) cout
            if ((j+1) % 128 == 0) {
                auto mid = std::chrono::high_resolution_clock::now();
                double us = std::chrono::duration_cast<
                    std::chrono::microseconds>(mid - start_time).count();
                PHILE_DBG(3, "[DynRebal] ckpt j=%zu/%zu succ=%lu %.1fμs",
                           j+1, batch->vertices.size(),
                           (unsigned long)success_count, us);
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        batch->finished_at = end_time;
        batch->status.store(MigrationBatch::Status::DONE);

        // upstream finish_commit CAS追赶 (neo_transaction.cpp 行33-36)
        uint64_t target = epoch - 1;
        while (!committed_epoch_.compare_exchange_weak(
                   target, epoch, std::memory_order_relaxed)) {
            target = epoch - 1;
        }

        // upstream throughput: global_speed = ops / duration * 1e6
        double dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - std::chrono::steady_clock::time_point(
                std::chrono::steady_clock::duration(
                    std::chrono::steady_clock::now().time_since_epoch()
                    - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        end_time - batch->created_at)))).count();
        // 简化
        double batch_dur_ns = batch->duration_ms() * 1e6;
        double tp = batch_dur_ns > 0
            ? static_cast<double>(success_count) / batch_dur_ns * 1e6 : 0;

        if (batch->from_tier > batch->to_tier) {
            total_promotes_.fetch_add(success_count);
        } else {
            total_demotes_.fetch_add(success_count);
        }
        total_bytes_moved_.fetch_add(batch->total_bytes);
        total_batches_.fetch_add(1);

        PHILE_DBG(1, "[DynRebal] batch %lu done: %lu vtx %.1fKB %.1fms "
                   "%.1f kops",
                   (unsigned long)batch->batch_id,
                   (unsigned long)success_count,
                   batch->total_bytes/1024.0, batch->duration_ms(), tp);

        return success_count > 0;
    }

    // ── 一轮完整的 GD→select→execute ──
    void rebalance_once() {
        auto gd = run_gradient_descent();

        // demote
        if (gd.demote_tier >= 0 && gd.demote_tier < NUM_TIERS - 1) {
            double gm = std::abs(gd.final_gradient[gd.demote_tier]);
            size_t batch_sz = compute_batch_size(gd.demote_tier, gm);
            auto* batch = batch_pool_.allocate();
            batch->from_tier = gd.demote_tier;
            batch->to_tier = gd.demote_tier + 1;
            batch->gradient_magnitude = gm;
            // 选最"冷"的vertex — 用pressure作为proxy
            // 简化: 随机选batch_sz个
            uint64_t vtx_total = tiers_[batch->from_tier].vertex_count.load();
            for (size_t i = 0; i < batch_sz && i < vtx_total; i++) {
                batch->vertices.push_back(i);  // 占位, 实际应有热度排序
            }
            execute_batch(batch);
            batch_pool_.deallocate(batch);
        }

        // promote
        if (gd.promote_tier > 0) {
            double gm = std::abs(gd.final_gradient[gd.promote_tier]);
            size_t batch_sz = compute_batch_size(gd.promote_tier, gm);
            auto* batch = batch_pool_.allocate();
            batch->from_tier = gd.promote_tier;
            batch->to_tier = gd.promote_tier - 1;
            batch->gradient_magnitude = gm;
            uint64_t vtx_total = tiers_[batch->from_tier].vertex_count.load();
            for (size_t i = 0; i < batch_sz && i < vtx_total; i++) {
                batch->vertices.push_back(i);
            }
            execute_batch(batch);
            batch_pool_.deallocate(batch);
        }
    }

    // ── 后台监控 ──
    void start(int bind_core = -1, uint32_t interval_ms = 200) {
        if (running_) return;
        running_ = true;
        monitor_thread_ = std::thread([this, interval_ms] {
            while (running_) {
                {
                    std::unique_lock<std::mutex> lk(monitor_mu_);
                    monitor_cv_.wait_for(lk, std::chrono::milliseconds(interval_ms));
                }
                if (!running_) break;
                rebalance_once();
                if (total_batches_.load() % 10 == 0 &&
                    debug::get_debug_level() >= 2) { dump_all(); }
            }
        });
        if (bind_core >= 0) bind_thread_to_core(monitor_thread_, bind_core);
    }

    void stop() {
        running_ = false;
        monitor_cv_.notify_all();
        if (monitor_thread_.joinable()) monitor_thread_.join();
    }
    ~DynamicRebalancer() { stop(); }

    void dump_all() const {
        std::printf("════ DynamicRebalancer (GD+Affinity) ════\n");
        for (int i = 0; i < NUM_TIERS; i++) tiers_[i].dump(i);
        std::printf("  pressure=[%.3f,%.3f,%.3f] gradient=[%.4f,%.4f,%.4f]\n",
                    pressure_[0], pressure_[1], pressure_[2],
                    gradient_[0], gradient_[1], gradient_[2]);
        std::printf("  epoch: mig=%lu committed=%lu gd_iters=%lu\n",
                    (unsigned long)migration_epoch_.load(),
                    (unsigned long)committed_epoch_.load(),
                    (unsigned long)total_gd_iters_.load());
        std::printf("  stats: promotes=%lu demotes=%lu bytes=%luMB batches=%lu\n",
                    (unsigned long)total_promotes_.load(),
                    (unsigned long)total_demotes_.load(),
                    (unsigned long)(total_bytes_moved_.load()>>20),
                    (unsigned long)total_batches_.load());
        affinity_uf_.dump();
        std::printf("════ End DynRebalancer ════\n");
    }
};

// ─── 调试宏 ───
#define PHILE_DYN_REBAL_DUMP(r) do { \
    if (::philemon::debug::get_debug_level()>=1) { \
        std::printf("[DYN_REBAL_DUMP] %s:%d\n",__FILE__,__LINE__); \
        (r).dump_all(); } } while(0)

class DynRebalBP {
    const DynamicRebalancer& r_; const char* n_;
    std::chrono::steady_clock::time_point t0_;
public:
    DynRebalBP(const DynamicRebalancer& r, const char* n)
        :r_(r),n_(n),t0_(std::chrono::steady_clock::now()){
        if(debug::get_debug_level()>=2){
            std::printf("━━ REBAL_BP ENTER: %s ━━\n",n_); r_.dump_all();}
    }
    ~DynRebalBP(){
        if(debug::get_debug_level()>=2){
            auto us=std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()-t0_).count();
            std::printf("━━ REBAL_BP EXIT: %s (%ldμs) ━━\n",n_,us);
            r_.dump_all();}
    }
};
#define PHILE_DYN_REBAL_BP(r,tag) \
    ::philemon::rebalance::DynRebalBP _drb_##__LINE__((r),(tag))

}  // namespace rebalance
}  // namespace philemon
#endif
