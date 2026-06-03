#ifndef PHILEMON_INTEGRATION_ORCHESTRATOR_HPP
#define PHILEMON_INTEGRATION_ORCHESTRATOR_HPP
/**
 * integration_orchestrator.hpp — pipeline编排 + 依赖DAG调度
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/wrapper/driver.h  (1577行)
 *     → execute_query() 查询分发:
 *       for(num_threads) for(op_types) { switch: BFS/SSSP/PR/WCC/TC }
 *       fopen/fclose + log_set_fp + try-catch  (行700-760)
 *       100% 保留: 分发循环, 日志绑定, 异常处理
 *
 *     → execute_microbenchmarks() 线程模型:
 *       chunk_size = (total+n-1)/n  (行510)
 *       worker lambda: init_thread → switch(op_type){GET_V/GET_E/SCAN} → ckpt
 *       thread_time[]/thread_speed[] → global_speed = ops/dur*1e6  (行600-645)
 *       bind_thread_to_core  (行648)
 *       100% 保留: chunk, worker, switch分发, 统计, core bind
 *
 *     → bfs() queue遍历:
 *       queue.push(source); while(!empty) { cur=front; pop;
 *         for(neighbor) if(!visited) push }  (行762-784)
 *       100% 保留: BFS的queue + visited + level框架
 *
 *     → execute_mixed_reader_writer():
 *       snapshot = get_shared_snapshot → reader/writer并行  (行1050-1150)
 *       100% 保留: snapshot, 并行
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_transaction.cpp  (537行)
 *     → ReadTransaction构造: register → get_timestamp → set_timestamp
 *     → commit: unregister  (行69-100)
 *     → 100% 保留
 *
 * 算法修改 (~20%):
 *   [MOD-1] for循环分发 → DAG拓扑排序调度:
 *     upstream execute_query: for(types) switch{BFS/SSSP/...}  顺序执行.
 *     改为: 4个stage有依赖关系, 用BFS拓扑排序决定执行顺序.
 *     stage的依赖是一个DAG, 入度为0的先执行.
 *     upstream: 每个query type独立, 不关心依赖;
 *     这里: stage间有数据依赖, Prefetch必须在Cost之前, Cost在Exec之前.
 *     拓扑排序用的BFS和upstream的bfs()结构完全一致(queue+visited+level),
 *     只是图是stage DAG不是vertex图.
 *
 *   [MOD-2] 固定线程数 → 自适应worker pool:
 *     upstream: num_threads在config里写死, chunk_size固定.
 *     改为: 根据当前pending query数自适应调整worker数:
 *       active_workers = clamp(pending/4, min_workers, max_workers)
 *     chunk_size也随之动态变化.
 *     upstream: 不看负载;  这里: feedback-driven sizing.
 *
 *   [MOD-3] switch分发 → 虚函数stage + pipeline数据流:
 *     upstream: switch(op_type){case BFS: bfs(); case SSSP: sssp();}
 *     改为: 每个stage是一个StageExecutor对象, execute()虚方法,
 *     前一个stage的output是后一个的input, 形成pipeline.
 *     upstream: 各query独立, 无数据传递;
 *     这里: StageResult在stage之间流动.
 *
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <cmath>
#include <cassert>
#include <string>
#include <queue>
#include <algorithm>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"

namespace philemon {
namespace orchestrator {

// ─── StageResult: stage间传递的数据包 ───
struct StageResult {
    bool success = false;
    double latency_ns = 0;
    int stage_id = -1;
    // 通用payload
    uint64_t prefetch_hits = 0;
    uint64_t vertices_migrated = 0;
    double query_throughput_kops = 0;
    double learned_reward = 0;

    void dump() const {
        std::printf("    [Stage%d] %s %.1fμs tp=%.1f\n",
                    stage_id, success?"OK":"FAIL",
                    latency_ns/1000, query_throughput_kops);
    }
};

// ─── [MOD-3] StageExecutor 虚基类 ───
// upstream: switch(op_type){case BFS:...case SSSP:...}
// 改为: 每个stage是StageExecutor子类, execute()虚方法
class StageExecutor {
public:
    int stage_id;
    std::string name;
    std::vector<int> dependencies;  // 入边 (DAG)

    StageExecutor(int id, const std::string& n) : stage_id(id), name(n) {}
    virtual ~StageExecutor() = default;

    virtual StageResult execute(const std::vector<StageResult>& prev_results) = 0;
};

// ─── PipelineMetrics ───
struct PipelineMetrics {
    uint64_t query_id = 0;
    std::vector<double> stage_latency_ns;
    double total_latency_ns = 0;
    int topo_order_length = 0;

    void dump() const {
        std::printf("  [Pipeline Q%lu] total=%.1fμs topo=%d stages\n",
                    (unsigned long)query_id, total_latency_ns/1000,
                    topo_order_length);
        for (size_t i = 0; i < stage_latency_ns.size(); i++) {
            std::printf("    stage[%zu]: %.1fμs (%.1f%%)\n",
                        i, stage_latency_ns[i]/1000,
                        total_latency_ns>0
                            ? stage_latency_ns[i]/total_latency_ns*100 : 0);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// IntegrationOrchestrator
// ═══════════════════════════════════════════════════════════════════════
class IntegrationOrchestrator {
    std::vector<StageExecutor*> stages_;   // 不持有, 外部管理生命周期
    std::vector<std::vector<int>> adj_;    // DAG邻接表
    std::vector<int> in_degree_;

    // upstream query_id fetch_add
    std::atomic<uint64_t> next_query_id_{1};

    // [MOD-2] 自适应worker pool
    uint32_t min_workers_ = 1;
    uint32_t max_workers_ = 16;
    std::atomic<uint64_t> pending_queries_{0};

    // 统计
    std::atomic<uint64_t> total_queries_{0};
    std::atomic<uint64_t> total_pipeline_runs_{0};
    double accumulated_latency_ns_ = 0;

    // metrics history
    static constexpr size_t MH_SIZE = 64;
    std::array<PipelineMetrics, MH_SIZE> metrics_hist_{};
    std::atomic<size_t> mh_head_{0};
    mutable std::shared_mutex mh_mu_;

    // upstream bind_thread_to_core
    static void bind_to_core(std::thread& t, int core_id) {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
        (void)t; (void)core_id;
    }

public:
    IntegrationOrchestrator() = default;

    void set_worker_range(uint32_t mn, uint32_t mx) {
        min_workers_ = mn; max_workers_ = mx;
    }

    // ── 注册stage + 依赖 ──
    void add_stage(StageExecutor* s) {
        while (stages_.size() <= (size_t)s->stage_id) {
            stages_.push_back(nullptr);
            adj_.push_back({});
            in_degree_.push_back(0);
        }
        stages_[s->stage_id] = s;
    }

    void add_dependency(int from_stage, int to_stage) {
        while (adj_.size() <= (size_t)from_stage) {
            adj_.push_back({}); in_degree_.push_back(0);
        }
        while (adj_.size() <= (size_t)to_stage) {
            adj_.push_back({}); in_degree_.push_back(0);
        }
        adj_[from_stage].push_back(to_stage);
        in_degree_[to_stage]++;
    }

    // ═══════════════════════════════════════════════════════════════════
    // [MOD-1] BFS拓扑排序
    //
    // upstream bfs() (driver.h 行762-784):
    //   queue.push(source);
    //   while(!queue.empty()) {
    //     cur = queue.front(); queue.pop();
    //     visited[cur] = true;
    //     level = result[cur] + 1;
    //     for(neighbor) if(!visited[n]) {
    //       visited[n]=true; queue.push(n); result[n]=level;
    //     }
    //   }
    //
    // 改为: Kahn's algorithm (也是BFS, 结构100%一致):
    //   for(node) if(in_degree[node]==0) queue.push(node);   ← BFS seed
    //   while(!queue.empty()) {
    //     cur = queue.front(); queue.pop();                   ← 同
    //     topo_order.push_back(cur);                          ← result[cur]=level
    //     for(neighbor in adj[cur]) {
    //       in_degree[n]--;
    //       if(in_degree[n]==0) queue.push(n);                ← 同!visited check
    //     }
    //   }
    //
    // 数据结构完全相同: std::queue, visited用in_degree==0替代,
    // level用topo_order的位置替代. 但图含义不同(stage DAG vs vertex图).
    // ═══════════════════════════════════════════════════════════════════
    std::vector<int> topo_sort() const {
        size_t n = stages_.size();
        std::vector<int> deg(in_degree_.begin(), in_degree_.end());
        deg.resize(n, 0);

        // upstream bfs: queue.push(source)
        // 这里: 所有入度0的stage是"source"
        std::queue<int> bfs_queue;
        for (size_t i = 0; i < n; i++) {
            if (deg[i] == 0 && stages_[i] != nullptr) {
                bfs_queue.push(i);
            }
        }

        std::vector<int> topo_order;

        // upstream: while(!queue.empty())
        while (!bfs_queue.empty()) {
            int cur = bfs_queue.front();
            bfs_queue.pop();
            // upstream: visited[cur] = true; result[cur] = level;
            topo_order.push_back(cur);

            // upstream: for(neighbor) if(!visited) push
            if ((size_t)cur < adj_.size()) {
                for (int next : adj_[cur]) {
                    deg[next]--;
                    // upstream: if(!visited[dest]) → 这里: if(in_degree==0)
                    if (deg[next] == 0) {
                        bfs_queue.push(next);
                    }
                }
            }
        }

        PHILE_DBG(2, "[Orch] topo sort: %zu stages", topo_order.size());
        for (int s : topo_order) {
            PHILE_DBG(2, "  stage[%d] = %s", s,
                       stages_[s] ? stages_[s]->name.c_str() : "?");
        }

        return topo_order;
    }

    // ── Pipeline执行 ──
    PipelineMetrics run_pipeline() {
        uint64_t qid = next_query_id_.fetch_add(1, std::memory_order_relaxed);
        pending_queries_.fetch_add(1);

        PipelineMetrics metrics;
        metrics.query_id = qid;

        auto topo = topo_sort();
        metrics.topo_order_length = topo.size();

        // 收集所有stage的结果
        std::vector<StageResult> all_results(stages_.size());

        auto pipeline_start = std::chrono::steady_clock::now();

        // upstream execute_query: for(op_types) { try { switch... } catch }
        for (int sid : topo) {
            if (!stages_[sid]) continue;

            auto t0 = std::chrono::steady_clock::now();

            // 收集依赖stage的结果
            std::vector<StageResult> dep_results;
            for (int dep : stages_[sid]->dependencies) {
                if (dep >= 0 && dep < (int)all_results.size()) {
                    dep_results.push_back(all_results[dep]);
                }
            }

            // upstream: try { bfs()/sssp()/... } catch(exception& e)
            try {
                all_results[sid] = stages_[sid]->execute(dep_results);
                all_results[sid].success = true;
            } catch (std::exception& e) {
                std::fprintf(stderr, "[Orch] stage %s error: %s\n",
                              stages_[sid]->name.c_str(), e.what());
                all_results[sid].success = false;
            }

            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                t1 - t0).count();
            all_results[sid].latency_ns = ns;
            all_results[sid].stage_id = sid;
            metrics.stage_latency_ns.push_back(ns);
        }

        auto pipeline_end = std::chrono::steady_clock::now();
        metrics.total_latency_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(pipeline_end - pipeline_start).count();

        total_queries_.fetch_add(1);
        total_pipeline_runs_.fetch_add(1);
        pending_queries_.fetch_sub(1);
        accumulated_latency_ns_ += metrics.total_latency_ns;

        // 写history
        {
            std::unique_lock<std::shared_mutex> lk(mh_mu_);
            size_t idx = mh_head_.fetch_add(1) % MH_SIZE;
            metrics_hist_[idx] = metrics;
        }

        PHILE_DBG(1, "[Orch] pipeline Q%lu done: %.1fμs %zu stages",
                   (unsigned long)qid, metrics.total_latency_ns/1000,
                   topo.size());

        return metrics;
    }

    // ── [MOD-2] 自适应batch执行 ──
    // upstream: chunk_size = (total + n_threads - 1) / n_threads  (固定)
    // 改为: active_workers = clamp(pending / 4, min, max), 动态调整
    void run_batch(size_t n_queries) {
        pending_queries_.store(n_queries);

        // [MOD-2] 动态worker数
        uint64_t pending = pending_queries_.load();
        uint32_t workers = static_cast<uint32_t>(
            std::max((uint64_t)min_workers_,
                     std::min((uint64_t)max_workers_, pending / 4 + 1)));

        // upstream chunk_size分发
        size_t chunk = (n_queries + workers - 1) / workers;
        std::vector<std::thread> threads;
        std::vector<double> thread_times(workers, 0);

        PHILE_DBG(1, "[Orch] batch: %zu queries, %u workers (adaptive), "
                   "chunk=%zu",
                   n_queries, workers, chunk);

        auto start = std::chrono::steady_clock::now();

        // upstream: for(i < num_threads) { threads.emplace_back(worker, i) }
        for (uint32_t t = 0; t < workers; t++) {
            threads.emplace_back([this, &thread_times, chunk, t, n_queries] {
                size_t begin = t * chunk;
                size_t end = std::min(begin + chunk, n_queries);
                auto t0 = std::chrono::steady_clock::now();

                for (size_t i = begin; i < end; i++) {
                    run_pipeline();

                    // upstream checkpoint: if((j-start)%check_point_size==0)
                    if ((i - begin + 1) % 50 == 0) {
                        auto mid = std::chrono::steady_clock::now();
                        double ms = std::chrono::duration<double,std::milli>(
                            mid - t0).count();
                        PHILE_DBG(2, "[Orch] worker %u: %zu/%zu %.1fms",
                                   t, i-begin+1, end-begin, ms);
                    }
                }

                auto t1 = std::chrono::steady_clock::now();
                thread_times[t] = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(t1 - t0).count();
            });
            // upstream: bind_thread_to_core(threads[i], i % hw_concurrency)
            bind_to_core(threads.back(),
                         t % std::thread::hardware_concurrency());
        }

        for (auto& th : threads) th.join();

        auto end = std::chrono::steady_clock::now();
        double total_ns = std::chrono::duration_cast<
            std::chrono::nanoseconds>(end - start).count();

        // upstream: global_speed = ops / duration * 1e6
        double throughput = static_cast<double>(n_queries) / total_ns * 1e6;

        PHILE_DBG(1, "[Orch] batch done: %zu queries %u workers "
                   "%.1fms throughput=%.1f kqps",
                   n_queries, workers, total_ns/1e6, throughput);
    }

    void dump_all() const {
        std::printf("════ IntegrationOrchestrator (DAG) ════\n");
        std::printf("  stages=%zu queries=%lu pipeline=%lu "
                    "avg=%.1fμs pending=%lu\n",
                    stages_.size(),
                    (unsigned long)total_queries_.load(),
                    (unsigned long)total_pipeline_runs_.load(),
                    total_queries_.load() > 0
                        ? accumulated_latency_ns_/total_queries_.load()/1000 : 0,
                    (unsigned long)pending_queries_.load());

        // DAG结构
        std::printf("  DAG edges:\n");
        for (size_t i = 0; i < adj_.size(); i++) {
            for (int j : adj_[i]) {
                std::printf("    %zu(%s) → %d(%s)\n",
                            i, stages_[i] ? stages_[i]->name.c_str() : "?",
                            j, (j<(int)stages_.size() && stages_[j])
                                ? stages_[j]->name.c_str() : "?");
            }
        }

        // 最近3条
        {
            std::shared_lock<std::shared_mutex> lk(mh_mu_);
            size_t head = mh_head_.load();
            for (int i = 0; i < 3 && i < (int)head; i++) {
                metrics_hist_[(head-1-i) % MH_SIZE].dump();
            }
        }
        std::printf("════ End Orchestrator ════\n");
    }
};

// 调试宏
#define PHILE_ORCH_DUMP(o) do { \
    if(::philemon::debug::get_debug_level()>=1){ \
        std::printf("[ORCH_DUMP] %s:%d\n",__FILE__,__LINE__); \
        (o).dump_all();}} while(0)

class OrchBP {
    const IntegrationOrchestrator& o_; const char* n_;
    std::chrono::steady_clock::time_point t0_;
public:
    OrchBP(const IntegrationOrchestrator& o, const char* n)
        :o_(o),n_(n),t0_(std::chrono::steady_clock::now()){
        if(debug::get_debug_level()>=2){
            std::printf("━━ ORCH_BP ENTER: %s ━━\n",n_); o_.dump_all();}
    }
    ~OrchBP(){
        if(debug::get_debug_level()>=2){
            auto us=std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()-t0_).count();
            std::printf("━━ ORCH_BP EXIT: %s (%ldμs) ━━\n",n_,us);
            o_.dump_all();}
    }
};
#define PHILE_ORCH_BP(o,tag) \
    ::philemon::orchestrator::OrchBP _ob_##__LINE__((o),(tag))

}  // namespace orchestrator
}  // namespace philemon
#endif
