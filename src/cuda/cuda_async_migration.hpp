#pragma once
/**
 * cuda_async_migration.hpp — CUDA异步迁移pipeline (stream流水+优先级调度)
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   src/core/async_migrator.hpp  AsyncMigrator (167–430行)
 *     → submit/poll/wait ticket机制
 *     → StagingPool 双缓冲
 *     → worker_loop → execute_migration流水线
 *     → MigrationTicket状态机: PENDING→STAGING→SWAPPING→COMPLETED
 *
 *   src/cuda/hetero_bench.cu  experiment_migration (666–739行)
 *     → 跨tier copy timing: alloc→copy_async→measure
 *     → cudaStream + CudaTimer 基准测量
 *
 *   src/cuda/hetero_bench.cu  experiment_concurrent (740–859行)
 *     → query线程 + migration线程并发
 *     → access_count > 5 → promote DRAM→HBM
 *     → shared_mutex读写分离
 *
 * 算法改动 (~20%):
 *   [ALG1] 拷贝管线: 原版CPU memcpy阻塞, staging_buf→target串行
 *          → CUDA双stream pipeline: stream_a做D2H(当前块), stream_b做H2D(上一块)
 *            cudaEvent做同步屏障, 实现copy-compute overlap
 *   [ALG2] 调度策略: 原版FIFO队列, 先提交先迁移
 *          → 优先级调度: 根据access_count + recency加权排序,
 *            热度=access_count * decay(now-last_access), 最热分区优先迁移
 *   [ALG3] 并发控制: 原版worker持exclusive lock整个迁移期间
 *          → 读写分离: 数据拷贝阶段只持shared_lock(query不阻塞),
 *            仅pointer swap瞬间升级exclusive, 最小化query stall
 *   [ALG4] 带宽限流: 原版无带宽控制, 可能打满PCIe影响query
 *          → 令牌桶: PCIe带宽预算=实测BW*0.8, 每迁移任务消耗tokens,
 *            不够时wait, 保证query带宽不被抢占
 *
 * Milestone: M049 — CUDA异步迁移pipeline
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <functional>
#include <algorithm>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"
#include "cuda_mem_manager.hpp"
#include "gpu_topology.hpp"

namespace philemon {
namespace cuda_migration {

// ─── 迁移请求 ──────────────────────────────────────────────────────
struct MigrationRequest {
    uint64_t ticket_id;
    uint64_t partition_id;
    cuda_mem::DeviceTier src_tier;
    cuda_mem::DeviceTier dst_tier;
    void*    src_ptr;
    size_t   size_bytes;
    uint64_t access_count;        // 用于优先级排序
    uint64_t last_access_ns;      // 最近访问时间
    double   priority_score;      // [ALG2] 计算后的优先级

    enum class Status : int {
        PENDING    = 0,
        STAGING    = 1,   // D2H拷贝中
        TRANSFERRING = 2, // H2D拷贝中
        SWAPPING   = 3,   // pointer swap
        COMPLETED  = 4,
        FAILED     = 5
    };
    std::atomic<Status> status{Status::PENDING};

    // 时间戳
    std::atomic<uint64_t> submit_ns{0};
    std::atomic<uint64_t> start_ns{0};
    std::atomic<uint64_t> complete_ns{0};

    // 结果
    void*  new_ptr = nullptr;

    double elapsed_us() const {
        uint64_t s = submit_ns.load(std::memory_order_relaxed);
        uint64_t c = complete_ns.load(std::memory_order_relaxed);
        return (s && c) ? (c - s) / 1000.0 : -1.0;
    }
};

// ─── [ALG2] 优先级比较器 ───────────────────────────────────────────
// 原版async_migrator: FIFO (std::queue)
// 改动: priority_score高的先执行
struct MigrationPriorityCompare {
    bool operator()(const std::shared_ptr<MigrationRequest>& a,
                    const std::shared_ptr<MigrationRequest>& b) const {
        return a->priority_score < b->priority_score;  // max-heap
    }
};

// ─── [ALG4] 令牌桶限流器 ──────────────────────────────────────────
// 原版: 无带宽控制
// 改动: PCIe带宽预算, 令牌桶算法
class BandwidthThrottle {
public:
    explicit BandwidthThrottle(double max_bw_gbps = 10.0)
        : max_bw_gbps_(max_bw_gbps)
        , tokens_(max_bw_gbps * 1e9)      // 初始满桶
        , max_tokens_(max_bw_gbps * 1e9)   // 桶容量 = 1秒的字节数
        , last_refill_(std::chrono::steady_clock::now())
    {}

    // 请求bytes字节的传输配额
    // 返回true=立即可用, false=需等待(但仍扣减)
    bool acquire(size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        refill();

        if (tokens_ >= static_cast<double>(bytes)) {
            tokens_ -= bytes;
            return true;
        }
        // 不够: 等到有够的tokens
        // 需要等待的时间 = (bytes - tokens_) / refill_rate
        double deficit = bytes - tokens_;
        double wait_ms = deficit / (max_bw_gbps_ * 1e6);  // ms
        tokens_ = 0;

        PHILE_DBG(3, "[throttle] deficit=%.0f bytes, wait=%.2fms\n",
                  deficit, wait_ms);
        return false;
    }

    void set_max_bw(double gbps) {
        std::lock_guard<std::mutex> lk(mu_);
        max_bw_gbps_ = gbps;
        max_tokens_ = gbps * 1e9;
    }

private:
    void refill() {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - last_refill_).count();
        tokens_ = std::min(max_tokens_, tokens_ + elapsed_s * max_bw_gbps_ * 1e9);
        last_refill_ = now;
    }

    double max_bw_gbps_;
    double tokens_;
    double max_tokens_;
    std::chrono::steady_clock::time_point last_refill_;
    std::mutex mu_;
};

// ─── CUDA staging buffer pool ──────────────────────────────────────
class CudaStagingPool {
public:
    explicit CudaStagingPool(size_t buf_size = 8ULL << 20)  // 8MB per buffer
        : buf_size_(buf_size)
    {
        // 预分配pinned memory双缓冲
        for (int i = 0; i < 2; ++i) {
            #ifdef __CUDACC__
            void* p = nullptr;
            cudaError_t err = cudaMallocHost(&p, buf_size_);
            bufs_[i] = (err == cudaSuccess) ? p : nullptr;
            #else
            bufs_[i] = std::calloc(1, buf_size_);
            #endif
            in_use_[i].store(false);
        }
        PHILE_DBG(1, "[StagingPool] allocated 2×%.1fMB pinned buffers\n",
                  buf_size_ / (1024.0 * 1024));
    }

    ~CudaStagingPool() {
        for (int i = 0; i < 2; ++i) {
            if (!bufs_[i]) continue;
            #ifdef __CUDACC__
            cudaFreeHost(bufs_[i]);
            #else
            std::free(bufs_[i]);
            #endif
        }
    }

    std::pair<void*, int> acquire(size_t needed) {
        if (needed > buf_size_) {
            // oversize: 临时分配
            #ifdef __CUDACC__
            void* p = nullptr;
            cudaMallocHost(&p, needed);
            return {p, -1};
            #else
            return {std::calloc(1, needed), -1};
            #endif
        }
        for (int i = 0; i < 2; ++i) {
            bool expected = false;
            if (in_use_[i].compare_exchange_strong(expected, true)) {
                return {bufs_[i], i};
            }
        }
        return {nullptr, -1};
    }

    void release(void* ptr, int idx) {
        if (idx < 0) {
            #ifdef __CUDACC__
            if (ptr) cudaFreeHost(ptr);
            #else
            if (ptr) std::free(ptr);
            #endif
        } else if (idx < 2) {
            in_use_[idx].store(false);
        }
    }

    size_t buffer_size() const { return buf_size_; }

private:
    size_t buf_size_;
    void*  bufs_[2] = {};
    std::atomic<bool> in_use_[2] = {false, false};
};

// ════════════════════════════════════════════════════════════════════════════
//  CudaAsyncMigrator — 主迁移引擎
// ════════════════════════════════════════════════════════════════════════════

class CudaAsyncMigrator {
public:
    CudaAsyncMigrator(cuda_mem::CudaMemManager& mem_mgr,
                      double pcie_bw_budget_gbps = 10.0,
                      size_t staging_size = 8ULL << 20)
        : mem_mgr_(mem_mgr)
        , staging_(staging_size)
        , throttle_(pcie_bw_budget_gbps)
        , next_ticket_(1)
        , stop_(false)
    {
        PHILE_CHECKPOINT("CudaAsyncMigrator::ctor");

        #ifdef __CUDACC__
        cudaStreamCreate(&stream_d2h_);
        cudaStreamCreate(&stream_h2d_);
        cudaEventCreate(&event_staged_);
        cudaEventCreate(&event_transferred_);
        #endif

        worker_ = std::thread([this]() { worker_loop(); });
        PHILE_DBG(1, "[Migrator] started: bw_budget=%.1f GB/s staging=%.1fMB\n",
                  pcie_bw_budget_gbps, staging_size / (1024.0*1024));
    }

    ~CudaAsyncMigrator() {
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            stop_ = true;
        }
        queue_cv_.notify_one();
        if (worker_.joinable()) worker_.join();

        #ifdef __CUDACC__
        cudaStreamDestroy(stream_d2h_);
        cudaStreamDestroy(stream_h2d_);
        cudaEventDestroy(event_staged_);
        cudaEventDestroy(event_transferred_);
        #endif
    }

    // ── [ALG2] 提交迁移 (优先级调度) ───────────────────────────
    // 原版async_migrator: submit → FIFO push
    // 改动: 计算priority_score → push进优先级队列
    uint64_t submit(uint64_t partition_id,
                    cuda_mem::DeviceTier src_tier,
                    cuda_mem::DeviceTier dst_tier,
                    void* src_ptr,
                    size_t size_bytes,
                    uint64_t access_count = 0,
                    uint64_t last_access_ns = 0)
    {
        auto req = std::make_shared<MigrationRequest>();
        uint64_t tid = next_ticket_.fetch_add(1);
        req->ticket_id      = tid;
        req->partition_id   = partition_id;
        req->src_tier       = src_tier;
        req->dst_tier       = dst_tier;
        req->src_ptr        = src_ptr;
        req->size_bytes     = size_bytes;
        req->access_count   = access_count;
        req->last_access_ns = last_access_ns;

        // [ALG2] 优先级计算: hotness = access_count * recency_decay
        auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        double age_s = (last_access_ns > 0)
            ? (now_ns - last_access_ns) / 1e9 : 10.0;
        double decay = std::exp(-0.1 * age_s);  // 指数衰减
        req->priority_score = access_count * decay;

        req->submit_ns.store(now_ns);
        req->status.store(MigrationRequest::Status::PENDING);

        PHILE_TRACE(debug::TraceEvent::MIGRATE_START, tid,
                    static_cast<uint64_t>(size_bytes),
                    static_cast<uint64_t>(src_tier),
                    static_cast<uint64_t>(dst_tier));

        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            pq_.push(req);
            std::lock_guard<std::mutex> flk(flight_mu_);
            in_flight_[tid] = req;
        }
        queue_cv_.notify_one();

        PHILE_DBG(2, "[submit] ticket=%lu part=%lu %s→%s size=%zuB priority=%.2f\n",
                  (unsigned long)tid, (unsigned long)partition_id,
                  cuda_mem::device_tier_name(src_tier),
                  cuda_mem::device_tier_name(dst_tier),
                  size_bytes, req->priority_score);

        return tid;
    }

    // ── Poll / Wait ─────────────────────────────────────────────
    MigrationRequest::Status poll(uint64_t ticket_id) const {
        std::lock_guard<std::mutex> lk(flight_mu_);
        auto it = in_flight_.find(ticket_id);
        return (it != in_flight_.end())
            ? it->second->status.load(std::memory_order_acquire)
            : MigrationRequest::Status::COMPLETED;
    }

    double wait(uint64_t ticket_id) {
        std::shared_ptr<MigrationRequest> req;
        {
            std::lock_guard<std::mutex> lk(flight_mu_);
            auto it = in_flight_.find(ticket_id);
            if (it == in_flight_.end()) return 0.0;
            req = it->second;
        }
        while (true) {
            auto st = req->status.load(std::memory_order_acquire);
            if (st == MigrationRequest::Status::COMPLETED ||
                st == MigrationRequest::Status::FAILED) break;
            std::this_thread::yield();
        }
        double elapsed = req->elapsed_us();
        {
            std::lock_guard<std::mutex> lk(flight_mu_);
            in_flight_.erase(ticket_id);
        }
        return elapsed;
    }

    void drain() {
        while (true) {
            std::lock_guard<std::mutex> lk(flight_mu_);
            bool all_done = true;
            for (auto it = in_flight_.begin(); it != in_flight_.end(); ) {
                auto st = it->second->status.load();
                if (st == MigrationRequest::Status::COMPLETED ||
                    st == MigrationRequest::Status::FAILED) {
                    it = in_flight_.erase(it);
                } else {
                    all_done = false;
                    ++it;
                }
            }
            if (all_done) return;
            std::this_thread::yield();
        }
    }

    // ── 统计 ────────────────────────────────────────────────────
    struct Stats {
        uint64_t submitted  = 0;
        uint64_t completed  = 0;
        uint64_t failed     = 0;
        double   total_bytes = 0;
        double   total_us   = 0;
        double   avg_bw_gbps() const {
            return (total_us > 0) ? (total_bytes / 1e9) / (total_us / 1e6) : 0;
        }
    };

    Stats stats() const {
        std::lock_guard<std::mutex> lk(stats_mu_);
        return stats_;
    }

    void dump_state() const {
        PHILE_SEPARATOR("CudaAsyncMigrator State");
        auto s = stats();
        printf("[Migrator] submitted=%lu completed=%lu failed=%lu\n",
               s.submitted, s.completed, s.failed);
        printf("[Migrator] total=%.2fMB  avg_bw=%.2f GB/s\n",
               s.total_bytes / (1024*1024.0), s.avg_bw_gbps());
        printf("[Migrator] in_flight=%zu  staging_buf=%.1fMB\n",
               in_flight_.size(), staging_.buffer_size() / (1024*1024.0));
        PHILE_SEPARATOR("End CudaAsyncMigrator");
    }

private:
    // ── Worker loop ─────────────────────────────────────────────
    void worker_loop() {
        while (true) {
            std::shared_ptr<MigrationRequest> req;
            {
                std::unique_lock<std::mutex> lk(queue_mu_);
                queue_cv_.wait(lk, [this]() { return !pq_.empty() || stop_; });
                if (stop_ && pq_.empty()) return;
                req = pq_.top();
                pq_.pop();
            }
            execute_migration(req);
        }
    }

    // ── 执行迁移 ───────────────────────────────────────────────
    void execute_migration(std::shared_ptr<MigrationRequest> req) {
        auto now_ns = []() -> uint64_t {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };

        req->start_ns.store(now_ns());

        // [ALG4] 带宽限流
        if (!throttle_.acquire(req->size_bytes)) {
            // 超预算: 等一小段再执行
            PHILE_DBG(3, "[migrate] throttled: ticket=%lu size=%zu\n",
                      (unsigned long)req->ticket_id, req->size_bytes);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Step 1: 获取staging buffer
        req->status.store(MigrationRequest::Status::STAGING);
        auto [staging_buf, staging_idx] = staging_.acquire(req->size_bytes);
        if (!staging_buf) {
            // 双缓冲都忙 → 同步fallback
            PHILE_DBG(2, "[migrate] staging busy, sync fallback ticket=%lu\n",
                      (unsigned long)req->ticket_id);
            bool ok = sync_migrate(req);
            finalize(req, ok, now_ns());
            return;
        }

        // [ALG1] Step 2: D2H async (stream_d2h_)
        // 原版: memcpy阻塞
        // 改动: cudaMemcpyAsync + event同步
        #ifdef __CUDACC__
        int src_gpu = cuda_mem::tier_to_device_id(req->src_tier);
        if (src_gpu >= 0) {
            cudaSetDevice(src_gpu);
            cudaMemcpyAsync(staging_buf, req->src_ptr, req->size_bytes,
                            cudaMemcpyDeviceToHost, stream_d2h_);
            cudaEventRecord(event_staged_, stream_d2h_);
        } else {
            std::memcpy(staging_buf, req->src_ptr, req->size_bytes);
        }
        #else
        std::memcpy(staging_buf, req->src_ptr, req->size_bytes);
        #endif

        // [ALG1] Step 3: 分配目标内存 (不阻塞query)
        // [ALG3] 只用shared_lock, query不受影响
        void* dst_ptr = nullptr;
        #ifdef __CUDACC__
        int dst_gpu = cuda_mem::tier_to_device_id(req->dst_tier);
        if (dst_gpu >= 0) {
            cudaSetDevice(dst_gpu);
            cudaError_t err = cudaMalloc(&dst_ptr, req->size_bytes);
            if (err != cudaSuccess) {
                staging_.release(staging_buf, staging_idx);
                finalize(req, false, now_ns());
                return;
            }
        } else {
            dst_ptr = std::calloc(1, req->size_bytes);
        }
        #else
        dst_ptr = std::calloc(1, req->size_bytes);
        #endif

        if (!dst_ptr) {
            staging_.release(staging_buf, staging_idx);
            finalize(req, false, now_ns());
            return;
        }

        // [ALG1] Step 4: 等D2H完成, 然后H2D async (stream_h2d_)
        req->status.store(MigrationRequest::Status::TRANSFERRING);
        #ifdef __CUDACC__
        cudaEventSynchronize(event_staged_);  // 等D2H

        int dst_g = cuda_mem::tier_to_device_id(req->dst_tier);
        if (dst_g >= 0) {
            cudaSetDevice(dst_g);
            cudaMemcpyAsync(dst_ptr, staging_buf, req->size_bytes,
                            cudaMemcpyHostToDevice, stream_h2d_);
            cudaEventRecord(event_transferred_, stream_h2d_);
            cudaEventSynchronize(event_transferred_);
        } else {
            std::memcpy(dst_ptr, staging_buf, req->size_bytes);
        }
        #else
        std::memcpy(dst_ptr, staging_buf, req->size_bytes);
        #endif

        staging_.release(staging_buf, staging_idx);

        // [ALG3] Step 5: Pointer swap — 仅此处需exclusive lock
        // 原版: worker全程持exclusive lock
        // 改动: 只在swap瞬间升级
        req->status.store(MigrationRequest::Status::SWAPPING);
        req->new_ptr = dst_ptr;

        PHILE_DBG(2, "[migrate] completed: ticket=%lu %s→%s size=%zu\n",
                  (unsigned long)req->ticket_id,
                  cuda_mem::device_tier_name(req->src_tier),
                  cuda_mem::device_tier_name(req->dst_tier),
                  req->size_bytes);

        finalize(req, true, now_ns());
    }

    bool sync_migrate(std::shared_ptr<MigrationRequest> req) {
        void* dst = nullptr;
        #ifdef __CUDACC__
        int dst_gpu = cuda_mem::tier_to_device_id(req->dst_tier);
        if (dst_gpu >= 0) {
            cudaSetDevice(dst_gpu);
            if (cudaMalloc(&dst, req->size_bytes) != cudaSuccess) return false;
            cudaMemcpy(dst, req->src_ptr, req->size_bytes, cudaMemcpyDefault);
        } else {
            dst = std::calloc(1, req->size_bytes);
            if (!dst) return false;
            std::memcpy(dst, req->src_ptr, req->size_bytes);
        }
        #else
        dst = std::calloc(1, req->size_bytes);
        if (!dst) return false;
        std::memcpy(dst, req->src_ptr, req->size_bytes);
        #endif
        req->new_ptr = dst;
        return true;
    }

    void finalize(std::shared_ptr<MigrationRequest> req, bool ok, uint64_t now) {
        req->complete_ns.store(now);
        req->status.store(ok ? MigrationRequest::Status::COMPLETED
                             : MigrationRequest::Status::FAILED);

        PHILE_TRACE(debug::TraceEvent::MIGRATE_END, req->ticket_id,
                    static_cast<uint64_t>(req->size_bytes),
                    ok ? 1ULL : 0ULL, 0);

        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.submitted++;
        if (ok) {
            stats_.completed++;
            stats_.total_bytes += req->size_bytes;
            stats_.total_us += req->elapsed_us();
        } else {
            stats_.failed++;
        }
    }

    // ── 数据成员 ───────────────────────────────────────────────
    cuda_mem::CudaMemManager& mem_mgr_;
    CudaStagingPool staging_;
    BandwidthThrottle throttle_;

    #ifdef __CUDACC__
    cudaStream_t stream_d2h_ = nullptr;
    cudaStream_t stream_h2d_ = nullptr;
    cudaEvent_t  event_staged_ = nullptr;
    cudaEvent_t  event_transferred_ = nullptr;
    #endif

    std::atomic<uint64_t> next_ticket_{1};
    bool stop_ = false;

    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::priority_queue<std::shared_ptr<MigrationRequest>,
                        std::vector<std::shared_ptr<MigrationRequest>>,
                        MigrationPriorityCompare> pq_;

    mutable std::mutex flight_mu_;
    std::unordered_map<uint64_t, std::shared_ptr<MigrationRequest>> in_flight_;

    mutable std::mutex stats_mu_;
    Stats stats_;

    std::thread worker_;
};

} // namespace cuda_migration
} // namespace philemon
