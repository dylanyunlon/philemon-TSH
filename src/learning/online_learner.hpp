#ifndef PHILEMON_ONLINE_LEARNER_HPP
#define PHILEMON_ONLINE_LEARNER_HPP
/**
 * online_learner.hpp — Thompson Sampling + UCB混合多臂老虎机
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp  (537行)
 *     → finish_commit() CAS loop: while(!CAS(target,timestamp)){}  (行33-36)
 *     → get_write_timestamp() fetch_add(1, relaxed)  (行29-31)
 *     → WriteTransaction::insert_edge():
 *       tree = lock(group) → insert → get_timestamp → commit_version
 *       → finish_commit → gc → unlock  (行356-404)
 *     → 100% 保留: CAS追赶, fetch_add, lock→op→commit→gc lifecycle
 *
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → ReaderTraceBlock atomic_value 位打包: lock|status|timestamp  (行21-50)
 *     → CAS自旋获取lock  (行53-74)
 *     → WriterTraceBlock stack pool: push/pop  (行75-90)
 *     → 100% 保留: 位打包, CAS自旋, stack pool
 *
 *   upstream/rapidstore/wrapper/driver.h  (1577行)
 *     → sssp() Dijkstra: priority_queue<pdv, greater> + relaxation  (行785-805)
 *     → bfs() queue + visited + level callback  (行762-784)
 *     → execute_microbenchmarks() checkpoint + throughput  (行500-650)
 *     → 100% 保留: 优先队列pattern, 队列遍历, 统计
 *
 * 算法修改 (~20%):
 *   [MOD-1] Dijkstra relaxation → Thompson Sampling:
 *     upstream sssp: priority_queue取最小dist, 对每个邻居做relaxation
 *       if(next_dist < result[dest]) result[dest]=next_dist, push.
 *     改为: priority_queue取最大Thompson采样值, 对每个arm做"exploration":
 *       sample = Beta(α,β), 如果sample > best_so_far, 更新best.
 *     数据结构相同(priority_queue), 但比较方向相反(max而非min),
 *     且queue里存的不是(distance,vertex)而是(sample_value,arm_index).
 *
 *   [MOD-2] BFS visited → UCB exploration bonus:
 *     upstream bfs: visited[]标记已访问, level[]记层数.
 *     改为: explored[]标记已尝试的arm, confidence_bound[]记探索奖励.
 *     UCB1: score = mean_reward + sqrt(2*ln(total_pulls) / arm_pulls).
 *     upstream: visited=bool, 只访问一次;
 *     这里: explored记录次数, UCB允许重复访问但减少奖励.
 *
 *   [MOD-3] CAS自旋lock的位打包 → 原子arm状态:
 *     upstream reader_trace: 把lock|status|timestamp打包成uint64_t,
 *     用CAS自旋获取. 改为: arm_state打包 alpha_bits|beta_bits|lock,
 *     更新时CAS自旋, 保证并发安全.
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
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <random>
#include <functional>
#include <cassert>
#include <string>

#include "../debug/philemon_debug.hpp"

namespace philemon {
namespace learning {

// ═══════════════════════════════════════════════════════════════════════
// [MOD-3] ArmState — 原子位打包
//
// upstream ReaderTraceBlock (neo_reader_trace.h 行21-50):
//   atomic_value = lock(1bit) | status(1bit) | timestamp(62bit)
//   CAS自旋获取lock: while(locked) { expected = unlocked; CAS(expected, locked) }
//
// 改为: alpha(20bit) | beta(20bit) | pulls(23bit) | lock(1bit)
// alpha/beta以定点数存储: 实际值 = 存储值 / 100.0
// ═══════════════════════════════════════════════════════════════════════
struct PackedArmState {
    std::atomic<uint64_t> packed{0};

    static constexpr uint64_t LOCK_BIT   = 1ULL;
    static constexpr uint64_t PULLS_SHIFT = 1;
    static constexpr uint64_t PULLS_MASK  = ((1ULL<<23)-1) << PULLS_SHIFT;
    static constexpr uint64_t BETA_SHIFT  = 24;
    static constexpr uint64_t BETA_MASK   = ((1ULL<<20)-1) << BETA_SHIFT;
    static constexpr uint64_t ALPHA_SHIFT = 44;
    static constexpr uint64_t ALPHA_MASK  = ((1ULL<<20)-1) << ALPHA_SHIFT;

    void init(double alpha, double beta) {
        uint64_t a = static_cast<uint64_t>(alpha * 100) & ((1ULL<<20)-1);
        uint64_t b = static_cast<uint64_t>(beta * 100)  & ((1ULL<<20)-1);
        packed.store((a << ALPHA_SHIFT) | (b << BETA_SHIFT), std::memory_order_relaxed);
    }

    double get_alpha() const {
        return ((packed.load() & ALPHA_MASK) >> ALPHA_SHIFT) / 100.0;
    }
    double get_beta() const {
        return ((packed.load() & BETA_MASK) >> BETA_SHIFT) / 100.0;
    }
    uint64_t get_pulls() const {
        return (packed.load() & PULLS_MASK) >> PULLS_SHIFT;
    }

    // upstream CAS自旋lock (neo_reader_trace.h 行53-74)
    void lock() {
        uint64_t expected = packed.load() & ~LOCK_BIT;
        while (!packed.compare_exchange_weak(
                   expected, expected | LOCK_BIT,
                   std::memory_order_acquire)) {
            expected = packed.load() & ~LOCK_BIT;
        }
    }
    void unlock() {
        packed.fetch_and(~LOCK_BIT, std::memory_order_release);
    }

    // 带CAS的原子更新alpha/beta/pulls
    void update(double reward) {
        lock();
        uint64_t val = packed.load();
        double a = ((val & ALPHA_MASK) >> ALPHA_SHIFT) / 100.0;
        double b = ((val & BETA_MASK) >> BETA_SHIFT) / 100.0;
        uint64_t pulls = (val & PULLS_MASK) >> PULLS_SHIFT;

        a += reward;
        b += (1.0 - reward);
        pulls++;

        uint64_t new_a = static_cast<uint64_t>(a * 100) & ((1ULL<<20)-1);
        uint64_t new_b = static_cast<uint64_t>(b * 100) & ((1ULL<<20)-1);
        uint64_t new_p = pulls & ((1ULL<<23)-1);
        uint64_t new_val = (new_a << ALPHA_SHIFT) | (new_b << BETA_SHIFT)
                           | (new_p << PULLS_SHIFT);  // lock bit cleared
        packed.store(new_val, std::memory_order_release);
    }

    double mean() const {
        double a = get_alpha(), b = get_beta();
        return (a+b > 0) ? a/(a+b) : 0.5;
    }

    void dump(int idx, double arm_val) const {
        std::printf("      arm[%d]=%.3f α=%.2f β=%.2f pulls=%lu mean=%.3f\n",
                    idx, arm_val, get_alpha(), get_beta(),
                    (unsigned long)get_pulls(), mean());
    }
};

// ─── TrialBlock — upstream WriteTransaction lifecycle ───
// lock → insert → get_timestamp → commit → gc → unlock
struct TrialBlock {
    uint64_t trial_id = 0;
    std::vector<size_t> chosen_arms;
    std::vector<double> chosen_values;
    std::vector<double> thompson_samples;  // 记录采样值
    std::vector<double> ucb_scores;        // 记录UCB分数
    double reward = 0;
    bool committed = false;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

    double duration_ms() const {
        return std::chrono::duration<double,std::milli>(
            end_time - start_time).count();
    }
    void dump() const {
        std::printf("    [Trial %lu] reward=%.3f %s %.1fms\n",
                    (unsigned long)trial_id, reward,
                    committed?"committed":"aborted", duration_ms());
        for (size_t i = 0; i < chosen_arms.size(); i++) {
            std::printf("      param[%zu]: arm=%zu val=%.3f ts=%.4f ucb=%.4f\n",
                        i, chosen_arms[i], chosen_values[i],
                        i<thompson_samples.size() ? thompson_samples[i] : 0,
                        i<ucb_scores.size() ? ucb_scores[i] : 0);
        }
    }
};

// ─── TrialBlockPool — upstream WriterTraceBlock stack pool ───
class TrialBlockPool {
    std::stack<TrialBlock*> free_;
    std::mutex mu_;
public:
    TrialBlock* allocate() {
        std::lock_guard<std::mutex> lk(mu_);
        TrialBlock* b;
        if (free_.empty()) { b = new TrialBlock(); }
        else { b = free_.top(); free_.pop(); }
        b->chosen_arms.clear(); b->chosen_values.clear();
        b->thompson_samples.clear(); b->ucb_scores.clear();
        b->reward = 0; b->committed = false;
        b->start_time = std::chrono::steady_clock::now();
        return b;
    }
    void deallocate(TrialBlock* b) {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push(b);
    }
    ~TrialBlockPool() {
        while(!free_.empty()) { delete free_.top(); free_.pop(); }
    }
};

// ─── 参数定义 ───
struct ParameterDef {
    std::string name;
    std::vector<double> arms;
};

// ═══════════════════════════════════════════════════════════════════════
// OnlineLearner
// ═══════════════════════════════════════════════════════════════════════
class OnlineLearner {
    std::vector<ParameterDef> params_;
    std::vector<std::vector<PackedArmState>> arm_states_;

    // upstream CAS epoch追赶
    std::atomic<uint64_t> trial_epoch_{0};
    std::atomic<uint64_t> committed_epoch_{0};

    TrialBlockPool pool_;

    // [MOD-2] explored计数 (替代BFS的visited[])
    // upstream bfs: bool visited[n]; 这里: uint64_t explored[n_arms]
    // UCB1需要总pull数和per-arm pull数
    std::atomic<uint64_t> total_pulls_{0};

    // 历史
    static constexpr size_t HISTORY_SIZE = 128;
    std::array<TrialBlock*, HISTORY_SIZE> history_{};
    std::atomic<size_t> history_head_{0};
    mutable std::shared_mutex history_mu_;

    // 随机数
    std::mt19937 rng_{std::random_device{}()};
    std::mutex rng_mu_;

    // 统计
    std::atomic<uint64_t> total_trials_{0};
    double cumulative_reward_ = 0;

    // Thompson/UCB混合权重
    double thompson_weight_ = 0.7;
    double ucb_weight_ = 0.3;

public:
    OnlineLearner() = default;

    void add_parameter(const std::string& name,
                       const std::vector<double>& arms) {
        params_.push_back({name, arms});
        arm_states_.emplace_back(arms.size());
        for (auto& s : arm_states_.back()) s.init(1.0, 1.0);
    }

    void set_weights(double ts_w, double ucb_w) {
        thompson_weight_ = ts_w; ucb_weight_ = ucb_w;
    }

    // ═══════════════════════════════════════════════════════════════════
    // [MOD-1] Thompson Sampling via priority_queue
    //
    // upstream sssp (driver.h 行785-805):
    //   priority_queue<pdv, vector, greater> → 取最小dist → relaxation
    //   while(!queue.empty()) {
    //     cur = queue.top(); queue.pop();
    //     if(cur_dist > result[cur]) continue;   ← 剪枝
    //     for(neighbor) { if(next < result[n]) update, push }
    //   }
    //
    // 改为:
    //   priority_queue<(sample, arm_idx), vector, less> → 取最大sample
    //   while(!queue.empty()) {
    //     (sample, arm) = queue.top(); queue.pop();
    //     if(sample < best_threshold) continue;    ← 剪枝(方向反转)
    //     best_arm = arm;  break;                   ← Dijkstra只取最优
    //   }
    //
    // 数据结构相同, 比较方向相反, relaxation改为sampling.
    // ═══════════════════════════════════════════════════════════════════
    TrialBlock* begin_trial() {
        auto* block = pool_.allocate();
        block->trial_id = trial_epoch_.fetch_add(1, std::memory_order_relaxed) + 1;

        std::lock_guard<std::mutex> rng_lk(rng_mu_);
        uint64_t total = total_pulls_.load();

        for (size_t p = 0; p < params_.size(); p++) {
            size_t K = params_[p].arms.size();

            // ── Thompson采样 via priority_queue (upstream sssp pattern) ──
            // upstream: priority_queue<pdv, vector, greater>  (min-heap)
            // 改为: max-heap, 存(sample_value, arm_index)
            using ScoredArm = std::pair<double, size_t>;
            std::priority_queue<ScoredArm> pq;   // max-heap (方向反转)

            for (size_t a = 0; a < K; a++) {
                double alpha = arm_states_[p][a].get_alpha();
                double beta  = arm_states_[p][a].get_beta();

                // Thompson: 从Beta(α,β)采样 via gamma
                std::gamma_distribution<double> ga(std::max(alpha, 0.01), 1.0);
                std::gamma_distribution<double> gb(std::max(beta, 0.01), 1.0);
                double x = ga(rng_), y = gb(rng_);
                double ts_sample = x / (x + y + 1e-15);

                // [MOD-2] UCB1 exploration bonus
                // upstream bfs: visited[dest] = true, level[dest] = level
                // 改为: explored[arm] = pulls, ucb = mean + sqrt(2ln(N)/n)
                uint64_t arm_pulls = arm_states_[p][a].get_pulls();
                double ucb_bonus = 0;
                if (arm_pulls > 0 && total > 0) {
                    ucb_bonus = arm_states_[p][a].mean()
                        + std::sqrt(2.0 * std::log(total + 1) / arm_pulls);
                } else {
                    ucb_bonus = 1e9;  // 未探索arm有无限奖励(upstream: 未visited优先)
                }

                // 混合分数
                double combined = thompson_weight_ * ts_sample
                                + ucb_weight_ * ucb_bonus;

                // upstream sssp: push({next_dist, destination})
                pq.push({combined, a});

                PHILE_DBG(3, "[TS+UCB] param[%zu] arm[%zu]=%.3f "
                           "α=%.1f β=%.1f ts=%.4f ucb=%.4f combined=%.4f",
                           p, a, params_[p].arms[a],
                           alpha, beta, ts_sample, ucb_bonus, combined);
            }

            // upstream sssp: cur = queue.top(); pop(); if(dist>result) continue;
            // 取最大(方向反转): 第一个就是最优
            size_t best_arm = 0;
            double best_score = -1;
            if (!pq.empty()) {
                auto top = pq.top();
                best_arm = top.second;
                best_score = top.first;
            }

            block->chosen_arms.push_back(best_arm);
            block->chosen_values.push_back(params_[p].arms[best_arm]);
            block->thompson_samples.push_back(best_score * thompson_weight_);
            block->ucb_scores.push_back(best_score * ucb_weight_);

            total_pulls_.fetch_add(1);
        }

        PHILE_DBG(2, "[Learner] trial %lu started",
                   (unsigned long)block->trial_id);
        return block;
    }

    double get_param(const TrialBlock* t, const std::string& name) const {
        for (size_t i = 0; i < params_.size(); i++)
            if (params_[i].name == name) return t->chosen_values[i];
        return 0;
    }

    // ── 提交: upstream WriteTransaction commit lifecycle ──
    // lock → insert → get_timestamp → commit_version → finish_commit → gc → unlock
    void commit_trial(TrialBlock* block, double raw_reward) {
        block->end_time = std::chrono::steady_clock::now();
        double reward = std::max(0.0, std::min(1.0, raw_reward));
        block->reward = reward;
        block->committed = true;

        // [MOD-3] CAS自旋更新arm state (upstream reader_trace CAS lock)
        for (size_t p = 0; p < params_.size(); p++) {
            size_t arm = block->chosen_arms[p];
            arm_states_[p][arm].update(reward);

            PHILE_DBG(2, "[TS update] '%s' arm[%zu]: α→%.2f β→%.2f "
                       "(reward=%.3f)",
                       params_[p].name.c_str(), arm,
                       arm_states_[p][arm].get_alpha(),
                       arm_states_[p][arm].get_beta(), reward);
        }

        // upstream finish_commit CAS追赶
        uint64_t target = block->trial_id - 1;
        while (!committed_epoch_.compare_exchange_weak(
                   target, block->trial_id, std::memory_order_relaxed)) {
            target = block->trial_id - 1;
        }

        total_trials_.fetch_add(1);
        cumulative_reward_ += reward;

        // 写入历史
        {
            std::unique_lock<std::shared_mutex> lk(history_mu_);
            size_t idx = history_head_.fetch_add(1) % HISTORY_SIZE;
            history_[idx] = block;  // pool不回收, 留在history
        }
    }

    void abort_trial(TrialBlock* b) { pool_.deallocate(b); }

    std::vector<double> best_params() const {
        std::vector<double> result;
        for (size_t p = 0; p < params_.size(); p++) {
            size_t best = 0; double best_mean = -1;
            for (size_t a = 0; a < params_[p].arms.size(); a++) {
                double m = arm_states_[p][a].mean();
                if (m > best_mean) { best_mean = m; best = a; }
            }
            result.push_back(params_[p].arms[best]);
        }
        return result;
    }

    void dump_all() const {
        std::printf("════ OnlineLearner (TS+UCB hybrid) ════\n");
        std::printf("  trials=%lu pulls=%lu reward=%.2f epoch=%lu/%lu "
                    "ts_w=%.2f ucb_w=%.2f\n",
                    (unsigned long)total_trials_.load(),
                    (unsigned long)total_pulls_.load(),
                    cumulative_reward_,
                    (unsigned long)trial_epoch_.load(),
                    (unsigned long)committed_epoch_.load(),
                    thompson_weight_, ucb_weight_);
        for (size_t p = 0; p < params_.size(); p++) {
            std::printf("  Param '%s' (%zu arms):\n",
                        params_[p].name.c_str(), params_[p].arms.size());
            for (size_t a = 0; a < params_[p].arms.size(); a++) {
                arm_states_[p][a].dump(a, params_[p].arms[a]);
            }
        }
        auto bp = best_params();
        std::printf("  Best:");
        for (size_t i = 0; i < bp.size(); i++)
            std::printf(" %s=%.3f", params_[i].name.c_str(), bp[i]);
        std::printf("\n");
        // 最近3条
        {
            std::shared_lock<std::shared_mutex> lk(history_mu_);
            size_t head = history_head_.load();
            for (int i = 0; i < 3 && i < (int)head; i++) {
                size_t idx = (head-1-i) % HISTORY_SIZE;
                if (history_[idx]) history_[idx]->dump();
            }
        }
        std::printf("════ End OnlineLearner ════\n");
    }
};

// 调试宏
#define PHILE_LEARNER_DUMP(l) do { \
    if(::philemon::debug::get_debug_level()>=1){ \
        std::printf("[LEARNER_DUMP] %s:%d\n",__FILE__,__LINE__); \
        (l).dump_all();}} while(0)

class LearnerBP {
    const OnlineLearner& l_; const char* n_;
    std::chrono::steady_clock::time_point t0_;
public:
    LearnerBP(const OnlineLearner& l, const char* n)
        :l_(l),n_(n),t0_(std::chrono::steady_clock::now()){
        if(debug::get_debug_level()>=2){
            std::printf("━━ LEARNER_BP ENTER: %s ━━\n",n_); l_.dump_all();}
    }
    ~LearnerBP(){
        if(debug::get_debug_level()>=2){
            auto us=std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()-t0_).count();
            std::printf("━━ LEARNER_BP EXIT: %s (%ldμs) ━━\n",n_,us);
            l_.dump_all();}
    }
};
#define PHILE_LEARNER_BP(l,tag) \
    ::philemon::learning::LearnerBP _lb_##__LINE__((l),(tag))

}  // namespace learning
}  // namespace philemon
#endif
