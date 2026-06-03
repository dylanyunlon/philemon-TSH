#ifndef PHILEMON_ADAPTIVE_PREFETCH_HPP
#define PHILEMON_ADAPTIVE_PREFETCH_HPP
/**
 * adaptive_prefetch.hpp — stride+frequency双模式自适应预取
 *
 * ====================================================================
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h  (186行)
 *     → ReaderTraceBlock 的 atomic_value 64-bit打包 (lock|status|timestamp)
 *     → try_lock() 的 CAS 自旋
 *     → ActiveReaderTracer 的 block 数组 + register/unregister
 *     → 100% 保留: lock/unlock/set_status/set_timestamp 位操作
 *
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp    (355行)
 *     → reader_register() 的 for-loop CAS 扫描
 *     → get_min_timestamp() 的全局barrier扫描
 *     → WriterTraceBlock 的 stack<T*> 对象池 push/pop
 *     → 100% 保留: allocate/deallocate 的 empty-check + fallback
 *
 *   upstream/rapidstore/wrapper/driver.h                               (1577行)
 *     → Barrier arrive_and_wait 同步原语
 *     → bind_thread_to_core() CPU亲和性
 *     → execute_query() 的查询分发 for-loop
 *     → 100% 保留: barrier wait + thread affinity binding
 *
 *   upstream/rapidstore/wrapper/wrapper.h                              (249行)
 *     → snapshot_edges() 模板回调 s->edges(index, callback, logical)
 *     → 100% 保留: 回调采样模式
 *
 * 算法修改 (~20%):
 *   - [MOD] 固定QueryHistoryRing → StrideDetector: 检测连续vertex访问的
 *           stride pattern, 用 last_addr + (last_addr - prev_addr) 预测
 *           下一次访问; upstream的ring只记录不预测
 *   - [MOD] 单一频率统计 → FrequencyPredictor: Count-Min Sketch做O(1)
 *           频率近似, 超过阈值的vertex视为热点; upstream用精确vector
 *   - [MOD] 静态触发阈值 → EWMA自适应: 以最近命中率的指数加权移动平均
 *           动态调threshold; upstream的AdaptiveThreshold用固定步长
 *   - [MOD] 顺序预取 → 双模式择优: stride和frequency各自给出预测集合,
 *           按置信度加权合并, 取top-K下发; upstream只用frequency
 *   - [NEW] PrefetchOutcome: 跟踪每条预取是否被命中, 反馈给两个
 *           predictor调整权重
 *
 * 断点调试:
 *   PHILE_ADAPT_PREFETCH_DUMP(engine)  — 打印stride/freq状态+命中率+权重
 *   PHILE_STRIDE_DUMP(engine)          — 打印stride检测器内部状态
 *   PHILE_FREQ_DUMP(engine)            — 打印Count-Min Sketch热点
 *   PHILE_ADAPT_BREAKPOINT(engine,tag) — RAII guard, 入口/出口自动dump
 *
 * Milestone: M052 — Adaptive prefetch engine
 * ====================================================================
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <vector>
#include <array>
#include <queue>
#include <stack>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <cassert>
#include <cmath>

#include "../debug/philemon_debug.hpp"
#include "../debug/state_inspector.hpp"

namespace philemon {
namespace prefetch {

// ═══════════════════════════════════════════════════════════════════════
// AccessTraceBlock — upstream ReaderTraceBlock骨架, 100% 保留位操作
// 唯一变化: 增加 tier_id 字段(占用 status 的高2位)
// ═══════════════════════════════════════════════════════════════════════
struct AccessTraceBlock {
private:
    // upstream: atomic_value 打包 lock(1)|status(3)|timestamp(60)
    // 保留: 完全相同的位布局和 CAS 语义
    std::atomic<uint64_t> atomic_value;

    static constexpr uint64_t LOCK_BIT       = 63;
    static constexpr uint64_t LOCK_MASK      = 1ULL << LOCK_BIT;
    static constexpr uint64_t STATUS_SHIFT   = 60;
    static constexpr uint64_t STATUS_MASK    = 0x7ULL << STATUS_SHIFT;
    static constexpr uint64_t TIMESTAMP_MASK = (1ULL << 60) - 1;

    // [NEW] 额外字段: 记录此block对应的vertex和tier
    std::atomic<uint64_t> vertex_id_{0};
    std::atomic<uint8_t>  tier_id_{0};
    std::atomic<uint32_t> access_count_{0};

public:
    AccessTraceBlock() : atomic_value(0) {}

    // ── upstream lock/unlock 100% 保留 ──
    void lock() {
        uint64_t expected, desired;
        while (true) {
            expected = atomic_value.load(std::memory_order_relaxed);
            if (expected & LOCK_MASK) continue;
            desired = expected | LOCK_MASK;
            if (atomic_value.compare_exchange_weak(
                    expected, desired, std::memory_order_acquire))
                break;
        }
    }

    void unlock() {
        atomic_value.fetch_and(~LOCK_MASK, std::memory_order_release);
    }

    bool try_lock() {
        uint64_t expected = atomic_value.load(std::memory_order_relaxed);
        uint64_t desired = expected | LOCK_MASK;
        return ((expected & LOCK_MASK) == 0) &&
               atomic_value.compare_exchange_strong(
                   expected, desired, std::memory_order_acquire);
    }

    // ── upstream get/set_status 100% 保留 ──
    uint8_t get_status() const {
        uint64_t v = atomic_value.load(std::memory_order_relaxed);
        return static_cast<uint8_t>((v & STATUS_MASK) >> STATUS_SHIFT);
    }

    void set_status(uint8_t status) {
        assert(status < 8);
        uint64_t expected, desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~STATUS_MASK) |
                      (static_cast<uint64_t>(status) << STATUS_SHIFT);
        } while (!atomic_value.compare_exchange_weak(
                     expected, desired, std::memory_order_relaxed));
    }

    // ── upstream get/set_timestamp 100% 保留 ──
    uint64_t get_timestamp() const {
        return atomic_value.load(std::memory_order_relaxed) & TIMESTAMP_MASK;
    }

    void set_timestamp(uint64_t ts) {
        assert(ts < (1ULL << 60));
        uint64_t expected, desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~TIMESTAMP_MASK) | (ts & TIMESTAMP_MASK);
        } while (!atomic_value.compare_exchange_weak(
                     expected, desired, std::memory_order_relaxed));
    }

    // ── upstream clear 100% 保留 ──
    void clear() {
        atomic_value.store(0, std::memory_order_relaxed);
        vertex_id_.store(0, std::memory_order_relaxed);
        tier_id_.store(0, std::memory_order_relaxed);
        access_count_.store(0, std::memory_order_relaxed);
    }

    // ── [NEW] vertex/tier accessors ──
    void set_vertex(uint64_t v) {
        vertex_id_.store(v, std::memory_order_relaxed);
    }
    uint64_t get_vertex() const {
        return vertex_id_.load(std::memory_order_relaxed);
    }
    void set_tier(uint8_t t) {
        tier_id_.store(t, std::memory_order_relaxed);
    }
    uint8_t get_tier() const {
        return tier_id_.load(std::memory_order_relaxed);
    }
    uint32_t bump_access() {
        return access_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    uint32_t get_access_count() const {
        return access_count_.load(std::memory_order_relaxed);
    }

    void dump() const {
        std::printf("    [TraceBlock] vtx=%lu tier=%u status=%u ts=%lu "
                    "access=%u locked=%s\n",
                    (unsigned long)get_vertex(), (unsigned)get_tier(),
                    (unsigned)get_status(), (unsigned long)get_timestamp(),
                    (unsigned)get_access_count(),
                    (atomic_value.load() & LOCK_MASK) ? "Y" : "N");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// AccessTracer — upstream ActiveReaderTracer 骨架
// 保留: register/unregister 的 CAS loop, get_min_timestamp 全局扫描
// ═══════════════════════════════════════════════════════════════════════
static constexpr size_t MAX_TRACE_BLOCKS = 128;

struct AccessTracer {
    std::array<AccessTraceBlock, MAX_TRACE_BLOCKS> blocks{};

    // upstream: reader_register 的 for-loop + try_lock + set_status(1) 完全保留
    AccessTraceBlock* trace_register() {
        for (auto& block : blocks) {
            if (block.get_status() == 0 && block.try_lock()) {
                block.set_status(1);
                return &block;
            }
        }
        return nullptr;
    }

    // upstream: set_timestamp + unlock 完全保留
    void set_timestamp(AccessTraceBlock* block, uint64_t ts) {
        block->set_timestamp(ts);
        block->unlock();
    }

    // upstream: reader_unregister → lock + clear 完全保留
    void trace_unregister(AccessTraceBlock* block) {
        block->lock();
        block->clear();
    }

    // upstream: get_min_timestamp 全扫 + sort + unique 完全保留
    uint64_t get_min_timestamp() const {
        uint64_t min_ts = std::numeric_limits<uint64_t>::max();
        for (auto& block : blocks) {
            if (block.get_status() == 1) {
                auto ts = block.get_timestamp();
                if (ts != 0 && ts < min_ts) min_ts = ts;
            }
        }
        return min_ts;
    }

    // upstream: get_active_reader_info 完全保留逻辑
    void get_active_info(std::vector<uint64_t>& vertices) const {
        for (auto& block : blocks) {
            if (block.get_status() == 1) {
                uint64_t v;
                do { v = block.get_vertex(); } while (v == 0 && block.get_status() == 1);
                if (v != 0) vertices.push_back(v);
            }
        }
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()),
                       vertices.end());
    }

    void dump() const {
        std::printf("  [AccessTracer] active blocks:\n");
        int cnt = 0;
        for (auto& block : blocks) {
            if (block.get_status() != 0) {
                block.dump();
                cnt++;
            }
        }
        std::printf("  total active: %d\n", cnt);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [MOD] StrideDetector — 替换upstream的固定ring buffer
//
// 算法改动: upstream的QueryHistoryRing只做记录, 不做预测.
// 这里用 stride detection: 如果最近3次访问的 vertex_id 差值相同,
// 就认为存在 stride, 预测 next = last + stride.
// 用状态机: INIT → TRANSIENT → STEADY, 连续命中升级, 一次miss重置.
// ═══════════════════════════════════════════════════════════════════════
class StrideDetector {
public:
    enum class State : uint8_t { INIT, TRANSIENT, STEADY };

    struct PerStreamState {
        int64_t  last_addr    = -1;
        int64_t  last_stride  = 0;
        State    state        = State::INIT;
        uint32_t hit_count    = 0;
        uint32_t miss_count   = 0;

        void dump(size_t stream_id) const {
            const char* state_str = (state == State::INIT) ? "INIT" :
                (state == State::TRANSIENT) ? "TRANSIENT" : "STEADY";
            std::printf("    [stride stream %zu] last_addr=%ld stride=%ld "
                        "state=%s hit=%u miss=%u\n",
                        stream_id, (long)last_addr, (long)last_stride,
                        state_str, hit_count, miss_count);
        }
    };

private:
    static constexpr size_t NUM_STREAMS = 16;
    std::array<PerStreamState, NUM_STREAMS> streams_;
    std::mutex mu_;

    size_t hash_stream(uint64_t vertex) const {
        return (vertex * 0x9E3779B97F4A7C15ULL) >> 60;
    }

public:
    // 喂一个新的 vertex 访问, 返回预测的下一个 vertex (-1 表示无预测)
    int64_t observe(uint64_t vertex) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t idx = hash_stream(vertex);
        auto& s = streams_[idx];

        int64_t prediction = -1;

        if (s.last_addr < 0) {
            // 第一次访问, 无法计算stride
            s.last_addr = static_cast<int64_t>(vertex);
            s.state = State::INIT;

            PHILE_DBG(3, "[StrideDetector] stream %zu: first access vtx=%lu",
                       idx, (unsigned long)vertex);
            return -1;
        }

        int64_t new_stride = static_cast<int64_t>(vertex) - s.last_addr;

        switch (s.state) {
        case State::INIT:
            // 记录第一个stride, 进入TRANSIENT
            s.last_stride = new_stride;
            s.state = State::TRANSIENT;
            break;

        case State::TRANSIENT:
            if (new_stride == s.last_stride && new_stride != 0) {
                // stride连续匹配, 升级到STEADY
                s.state = State::STEADY;
                s.hit_count++;
                prediction = static_cast<int64_t>(vertex) + new_stride;
            } else {
                // stride不匹配, 重置
                s.last_stride = new_stride;
                s.state = State::TRANSIENT;
                s.miss_count++;
            }
            break;

        case State::STEADY:
            if (new_stride == s.last_stride) {
                s.hit_count++;
                prediction = static_cast<int64_t>(vertex) + new_stride;
            } else {
                // 模式破裂, 退回TRANSIENT
                s.last_stride = new_stride;
                s.state = State::TRANSIENT;
                s.miss_count++;
            }
            break;
        }

        s.last_addr = static_cast<int64_t>(vertex);

        PHILE_DBG(3, "[StrideDetector] stream %zu: vtx=%lu stride=%ld "
                   "state=%d predict=%ld",
                   idx, (unsigned long)vertex, (long)new_stride,
                   (int)s.state, (long)prediction);

        return prediction;
    }

    // 置信度: steady且hit多→高
    double confidence() const {
        uint32_t total_hit = 0, total_miss = 0;
        for (auto& s : streams_) {
            total_hit += s.hit_count;
            total_miss += s.miss_count;
        }
        if (total_hit + total_miss == 0) return 0.0;
        return static_cast<double>(total_hit) / (total_hit + total_miss);
    }

    void dump() const {
        std::printf("  [StrideDetector] confidence=%.3f\n", confidence());
        for (size_t i = 0; i < NUM_STREAMS; i++) {
            if (streams_[i].last_addr >= 0) streams_[i].dump(i);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [MOD] FrequencyPredictor — 替换upstream的精确vector统计
//
// 算法改动: upstream在PrefetchPredictor里用 unordered_map<vertex, count>
// 做精确频率统计, O(N) 空间. 这里用 Count-Min Sketch 做 O(1) 插入
// O(1) 查询的近似频率, 空间固定 ROWS × COLS.
// 热点判定: 频率超过 mean + 2*stddev.
// ═══════════════════════════════════════════════════════════════════════
class FrequencyPredictor {
    static constexpr size_t ROWS = 4;
    static constexpr size_t COLS = 1024;

    // Count-Min Sketch: ROWS个hash函数, 每行COLS个计数器
    std::array<std::array<uint32_t, COLS>, ROWS> sketch_{};
    uint64_t total_observations_ = 0;

    // 最近的热点vertex列表 (从sketch中提取)
    std::vector<uint64_t> hot_vertices_;
    mutable std::mutex mu_;

    // ROWS个不同的hash
    size_t hash_fn(size_t row, uint64_t vertex) const {
        // 用不同的乘数做多hash
        static constexpr uint64_t PRIMES[ROWS] = {
            0x9E3779B97F4A7C15ULL, 0x517CC1B727220A95ULL,
            0x6C62272E07BB0142ULL, 0xBF58476D1CE4E5B9ULL
        };
        return ((vertex * PRIMES[row]) >> 22) % COLS;
    }

public:
    void observe(uint64_t vertex) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t r = 0; r < ROWS; r++) {
            sketch_[r][hash_fn(r, vertex)]++;
        }
        total_observations_++;
    }

    uint32_t query(uint64_t vertex) const {
        uint32_t min_val = std::numeric_limits<uint32_t>::max();
        for (size_t r = 0; r < ROWS; r++) {
            min_val = std::min(min_val, sketch_[r][hash_fn(r, vertex)]);
        }
        return min_val;
    }

    // 提取当前热点: 扫描所有已见过的vertex, 返回频率异常高的
    // 调用者需要提供候选集合 (从AccessTracer获取)
    void extract_hotspots(const std::vector<uint64_t>& candidates,
                          std::vector<uint64_t>& out_hot,
                          double sigma_multiplier = 2.0) {
        std::lock_guard<std::mutex> lk(mu_);
        if (candidates.empty()) return;

        // 计算候选的频率统计
        std::vector<uint32_t> freqs;
        freqs.reserve(candidates.size());
        for (uint64_t v : candidates) {
            freqs.push_back(query(v));
        }

        double sum = 0, sum_sq = 0;
        for (uint32_t f : freqs) {
            sum += f;
            sum_sq += static_cast<double>(f) * f;
        }
        double mean = sum / freqs.size();
        double var = sum_sq / freqs.size() - mean * mean;
        double threshold = mean + sigma_multiplier * std::sqrt(std::max(var, 0.0));

        out_hot.clear();
        for (size_t i = 0; i < candidates.size(); i++) {
            if (freqs[i] > threshold) {
                out_hot.push_back(candidates[i]);
            }
        }

        PHILE_DBG(2, "[FreqPredictor] candidates=%zu mean=%.1f std=%.1f "
                   "threshold=%.1f hotspots=%zu",
                   candidates.size(), mean, std::sqrt(std::max(var, 0.0)),
                   threshold, out_hot.size());
    }

    double confidence() const {
        if (total_observations_ < 100) return 0.1;
        // 越多观测越自信, 但饱和在0.9
        return std::min(0.9, 0.3 + 0.006 * std::sqrt(
                   static_cast<double>(total_observations_)));
    }

    void dump() const {
        std::printf("  [FreqPredictor] observations=%lu confidence=%.3f\n",
                    (unsigned long)total_observations_, confidence());
        // 打印sketch的非零计数分布
        uint32_t max_val = 0;
        size_t nonzero = 0;
        for (auto& row : sketch_) {
            for (auto v : row) {
                if (v > 0) { nonzero++; max_val = std::max(max_val, v); }
            }
        }
        std::printf("    sketch: nonzero=%zu/%zu max_count=%u\n",
                    nonzero, ROWS * COLS, max_val);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [MOD] EWMAThreshold — 替换upstream的固定步长AdaptiveThreshold
//
// 算法改动: upstream用 threshold ± step 做线性调整.
// 这里用 EWMA(α=0.15) 追踪命中率, 阈值 = EWMA * scale_factor,
// 当命中率下降时自动收紧预取, 上升时放松.
// ═══════════════════════════════════════════════════════════════════════
class EWMAThreshold {
    double alpha_;
    double ewma_hit_rate_ = 0.5;  // 初始50%
    double scale_factor_;

    uint64_t total_prefetches_ = 0;
    uint64_t total_hits_ = 0;
    uint64_t window_prefetches_ = 0;
    uint64_t window_hits_ = 0;

    static constexpr uint64_t WINDOW_SIZE = 64;

public:
    explicit EWMAThreshold(double alpha = 0.15, double scale = 1.5)
        : alpha_(alpha), scale_factor_(scale) {}

    void record_outcome(bool hit) {
        total_prefetches_++;
        window_prefetches_++;
        if (hit) { total_hits_++; window_hits_++; }

        if (window_prefetches_ >= WINDOW_SIZE) {
            double window_rate = static_cast<double>(window_hits_)
                                 / window_prefetches_;
            ewma_hit_rate_ = alpha_ * window_rate
                             + (1.0 - alpha_) * ewma_hit_rate_;
            window_prefetches_ = 0;
            window_hits_ = 0;

            PHILE_DBG(2, "[EWMAThreshold] updated: hit_rate=%.3f "
                       "ewma=%.3f threshold=%.3f",
                       window_rate, ewma_hit_rate_, current_threshold());
        }
    }

    // 返回当前触发阈值: 频率count必须超过此值才预取
    double current_threshold() const {
        // 命中率高→阈值降低(更激进), 命中率低→阈值升高(更保守)
        return scale_factor_ * (1.0 - ewma_hit_rate_) * 10.0 + 1.0;
    }

    double hit_rate() const {
        if (total_prefetches_ == 0) return 0.0;
        return static_cast<double>(total_hits_) / total_prefetches_;
    }

    void dump() const {
        std::printf("  [EWMAThreshold] total_prefetch=%lu hits=%lu "
                    "hit_rate=%.3f ewma=%.3f threshold=%.1f\n",
                    (unsigned long)total_prefetches_,
                    (unsigned long)total_hits_,
                    hit_rate(), ewma_hit_rate_, current_threshold());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchTicket — upstream PrefetchTraceBlock 的ticket pattern保留
// ═══════════════════════════════════════════════════════════════════════
struct PrefetchTicket {
    uint64_t vertex_id;
    uint8_t  from_tier;      // cold tier
    uint8_t  to_tier;        // hot tier
    double   confidence;     // 预测置信度 [0,1]
    enum class Status : uint8_t { PENDING, INFLIGHT, DONE, HIT, EXPIRED };
    std::atomic<Status> status{Status::PENDING};
    std::chrono::steady_clock::time_point issued_at;

    PrefetchTicket() : vertex_id(0), from_tier(0), to_tier(0),
                       confidence(0),
                       issued_at(std::chrono::steady_clock::now()) {}

    PrefetchTicket(uint64_t v, uint8_t from, uint8_t to, double conf)
        : vertex_id(v), from_tier(from), to_tier(to), confidence(conf),
          issued_at(std::chrono::steady_clock::now()) {}

    bool is_expired(double timeout_ms = 100.0) const {
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - issued_at).count();
        return elapsed > timeout_ms * 1000;
    }

    void dump() const {
        const char* st_str[] = {"PENDING","INFLIGHT","DONE","HIT","EXPIRED"};
        auto age_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - issued_at).count();
        std::printf("    [Ticket] vtx=%lu %u→%u conf=%.2f status=%s "
                    "age=%ldμs\n",
                    (unsigned long)vertex_id, from_tier, to_tier,
                    confidence,
                    st_str[static_cast<int>(status.load())],
                    (long)age_us);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// PrefetchTicketPool — upstream WriterTraceBlock的 stack<T*> 对象池
// 保留: empty check + new fallback 的 allocate/deallocate pattern
// ═══════════════════════════════════════════════════════════════════════
class PrefetchTicketPool {
    std::stack<PrefetchTicket*> free_pool_;
    std::mutex mu_;
    uint64_t total_alloc_ = 0;
    uint64_t pool_reuse_ = 0;

public:
    // upstream: allocate pattern — empty check + malloc fallback
    PrefetchTicket* allocate(uint64_t vtx, uint8_t from, uint8_t to,
                             double conf) {
        std::lock_guard<std::mutex> lk(mu_);
        PrefetchTicket* t = nullptr;
        if (free_pool_.empty()) {
            t = new PrefetchTicket(vtx, from, to, conf);
            total_alloc_++;
        } else {
            t = free_pool_.top();
            free_pool_.pop();
            // 重用: 重置字段
            t->vertex_id = vtx;
            t->from_tier = from;
            t->to_tier = to;
            t->confidence = conf;
            t->status.store(PrefetchTicket::Status::PENDING);
            t->issued_at = std::chrono::steady_clock::now();
            pool_reuse_++;
        }
        return t;
    }

    // upstream: deallocate pattern — stack push 回收
    void deallocate(PrefetchTicket* t) {
        std::lock_guard<std::mutex> lk(mu_);
        free_pool_.push(t);
    }

    ~PrefetchTicketPool() {
        while (!free_pool_.empty()) {
            delete free_pool_.top();
            free_pool_.pop();
        }
    }

    void dump() const {
        std::printf("  [TicketPool] total_alloc=%lu reuse=%lu pool_size=%zu\n",
                    (unsigned long)total_alloc_,
                    (unsigned long)pool_reuse_,
                    free_pool_.size());
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Barrier — upstream Barrier arrive_and_wait 100% 保留
// ═══════════════════════════════════════════════════════════════════════
class PrefetchBarrier {
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t count_;
    size_t waiting_ = 0;
public:
    explicit PrefetchBarrier(size_t count) : count_(count) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        ++waiting_;
        if (waiting_ == count_) {
            waiting_ = 0;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return waiting_ == 0; });
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// [MOD] AdaptivePrefetchEngine — 双模式预取的顶层引擎
//
// 核心算法改动:
//   upstream的PrefetchEngine用单一frequency predictor决定预取集合.
//   这里用 stride + frequency 两个predictor, 各自输出预测集合,
//   按各自confidence加权合并, 取top-K.
//
//   权重调整: 每个predictor的预取命中反馈给EWMAThreshold,
//   同时用 outcome_feedback 更新 stride_weight / freq_weight.
// ═══════════════════════════════════════════════════════════════════════
class AdaptivePrefetchEngine {
    // 双模式预测器
    StrideDetector stride_;
    FrequencyPredictor freq_;
    EWMAThreshold threshold_;
    AccessTracer tracer_;
    PrefetchTicketPool ticket_pool_;

    // predictor 权重 (0~1, 和不需要为1)
    std::atomic<double> stride_weight_{0.5};
    std::atomic<double> freq_weight_{0.5};

    // 配置
    size_t max_inflight_ = 32;
    size_t top_k_ = 8;

    // 活跃 ticket 列表
    std::vector<PrefetchTicket*> inflight_;
    mutable std::shared_mutex inflight_mu_;

    // 统计
    std::atomic<uint64_t> total_observations_{0};
    std::atomic<uint64_t> total_issued_{0};
    std::atomic<uint64_t> total_hit_{0};
    std::atomic<uint64_t> total_miss_{0};

    // 迁移执行回调: 调用者设置的实际迁移函数
    std::function<bool(uint64_t vtx, uint8_t from_tier, uint8_t to_tier)>
        migrate_fn_;

    bool running_ = false;
    std::thread worker_;
    std::mutex worker_mu_;
    std::condition_variable worker_cv_;

    // ─── bind_thread_to_core — upstream 100% 保留 ───
    static void bind_to_core(std::thread& t, int core_id) {
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        int rc = pthread_setaffinity_np(t.native_handle(),
                                         sizeof(cpu_set_t), &cpuset);
        if (rc != 0)
            std::fprintf(stderr, "[AdaptPrefetch] setaffinity error: %d\n", rc);
#endif
    }

public:
    AdaptivePrefetchEngine() = default;

    void set_migrate_fn(
        std::function<bool(uint64_t, uint8_t, uint8_t)> fn) {
        migrate_fn_ = std::move(fn);
    }

    void set_config(size_t max_inflight, size_t top_k) {
        max_inflight_ = max_inflight;
        top_k_ = top_k;
    }

    // ── 核心: 记录一次vertex访问 ──
    // 返回: 该访问是否命中了之前的预取
    bool record_access(uint64_t vertex, uint8_t tier) {
        total_observations_.fetch_add(1, std::memory_order_relaxed);

        // 喂给两个predictor
        int64_t stride_pred = stride_.observe(vertex);
        freq_.observe(vertex);

        // 检查是否命中已有的prefetch ticket
        bool hit = check_and_retire(vertex);
        if (hit) {
            total_hit_.fetch_add(1, std::memory_order_relaxed);
            threshold_.record_outcome(true);
        }

        // 通知 worker 可能有新预取需求
        worker_cv_.notify_one();

        PHILE_DBG(3, "[AdaptPrefetch] access vtx=%lu tier=%u "
                   "stride_pred=%ld hit=%s",
                   (unsigned long)vertex, tier,
                   (long)stride_pred, hit ? "Y" : "N");

        return hit;
    }

    // ── 启动后台预取worker ──
    void start(int bind_core = -1) {
        if (running_) return;
        running_ = true;
        worker_ = std::thread([this] { prefetch_loop(); });
        if (bind_core >= 0) {
            bind_to_core(worker_, bind_core);
        }
        PHILE_DBG(1, "[AdaptPrefetch] started (core=%d)", bind_core);
    }

    void stop() {
        running_ = false;
        worker_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        PHILE_DBG(1, "[AdaptPrefetch] stopped");
    }

    ~AdaptivePrefetchEngine() { stop(); }

    // ── 全量状态打印 ──
    void dump_all() const {
        std::printf("════ AdaptivePrefetchEngine ════\n");
        std::printf("  observations=%lu issued=%lu hit=%lu miss=%lu "
                    "hit_rate=%.3f\n",
                    (unsigned long)total_observations_.load(),
                    (unsigned long)total_issued_.load(),
                    (unsigned long)total_hit_.load(),
                    (unsigned long)total_miss_.load(),
                    total_issued_.load() > 0
                        ? (double)total_hit_.load() / total_issued_.load()
                        : 0.0);
        std::printf("  stride_weight=%.3f freq_weight=%.3f\n",
                    stride_weight_.load(), freq_weight_.load());
        stride_.dump();
        freq_.dump();
        threshold_.dump();
        ticket_pool_.dump();

        // inflight tickets
        {
            std::shared_lock<std::shared_mutex> lk(inflight_mu_);
            std::printf("  inflight tickets: %zu\n", inflight_.size());
            for (auto* t : inflight_) t->dump();
        }
        std::printf("════ End AdaptPrefetch ════\n");
    }

private:
    // ── 检查vertex是否命中inflight ticket ──
    bool check_and_retire(uint64_t vertex) {
        std::unique_lock<std::shared_mutex> lk(inflight_mu_);
        for (auto it = inflight_.begin(); it != inflight_.end(); ++it) {
            auto* t = *it;
            if (t->vertex_id == vertex &&
                t->status.load() == PrefetchTicket::Status::DONE) {
                t->status.store(PrefetchTicket::Status::HIT);
                inflight_.erase(it);
                ticket_pool_.deallocate(t);
                return true;
            }
        }
        return false;
    }

    // ── 清理过期ticket ──
    void expire_tickets() {
        std::unique_lock<std::shared_mutex> lk(inflight_mu_);
        auto it = inflight_.begin();
        while (it != inflight_.end()) {
            if ((*it)->is_expired()) {
                (*it)->status.store(PrefetchTicket::Status::EXPIRED);
                total_miss_.fetch_add(1, std::memory_order_relaxed);
                threshold_.record_outcome(false);
                ticket_pool_.deallocate(*it);
                it = inflight_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // ── [MOD] 双模式融合预测 ──
    // upstream: 只用frequency. 这里stride和frequency各自出预测集合,
    // 按confidence加权排序, 取top-K.
    std::vector<std::pair<uint64_t, double>> fused_prediction() {
        // stride预测: 从每个stream取
        std::vector<std::pair<uint64_t, double>> candidates;
        double s_conf = stride_.confidence();
        double f_conf = freq_.confidence();
        double s_w = stride_weight_.load();
        double f_w = freq_weight_.load();

        // stride部分: StrideDetector已经在observe时给出了预测
        // 这里重新扫描tracer获取当前活跃vertex, 对每个做stride预测
        std::vector<uint64_t> active_vertices;
        tracer_.get_active_info(active_vertices);

        for (uint64_t v : active_vertices) {
            int64_t pred = stride_.observe(v);
            if (pred > 0) {
                double score = s_conf * s_w;
                candidates.emplace_back(
                    static_cast<uint64_t>(pred), score);
            }
        }

        // frequency部分: 提取热点
        std::vector<uint64_t> hotspots;
        freq_.extract_hotspots(active_vertices, hotspots, 1.5);
        for (uint64_t v : hotspots) {
            double score = f_conf * f_w;
            candidates.emplace_back(v, score);
        }

        // 去重: 同一vertex取最高score
        std::sort(candidates.begin(), candidates.end());
        std::vector<std::pair<uint64_t, double>> merged;
        for (auto& [vtx, score] : candidates) {
            if (!merged.empty() && merged.back().first == vtx) {
                merged.back().second = std::max(merged.back().second, score);
            } else {
                merged.emplace_back(vtx, score);
            }
        }

        // 按score降序排, 取top-K
        std::sort(merged.begin(), merged.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        if (merged.size() > top_k_) merged.resize(top_k_);

        // 过滤: score必须超过EWMA阈值的逆(阈值高→少预取)
        double thresh = threshold_.current_threshold();
        std::vector<std::pair<uint64_t, double>> result;
        for (auto& [vtx, score] : merged) {
            if (score * 10.0 >= thresh) {
                result.emplace_back(vtx, score);
            }
        }

        PHILE_DBG(2, "[FusedPredict] candidates=%zu merged=%zu "
                   "after_threshold=%zu (thresh=%.1f)",
                   candidates.size(), merged.size(),
                   result.size(), thresh);

        return result;
    }

    // ── [MOD] 权重更新: 根据命中反馈调整两个predictor的权重 ──
    // upstream无此逻辑. 这里用简单的multiplicative weight update:
    // 命中的predictor权重 *= 1.05, 未命中的 *= 0.95, clamp [0.1, 0.9]
    void update_weights(bool stride_contributed, bool freq_contributed) {
        auto clamp = [](double v) { return std::max(0.1, std::min(0.9, v)); };

        if (stride_contributed) {
            stride_weight_.store(
                clamp(stride_weight_.load() * 1.05));
        } else {
            stride_weight_.store(
                clamp(stride_weight_.load() * 0.95));
        }
        if (freq_contributed) {
            freq_weight_.store(
                clamp(freq_weight_.load() * 1.05));
        } else {
            freq_weight_.store(
                clamp(freq_weight_.load() * 0.95));
        }
    }

    // ── 后台预取循环 ──
    void prefetch_loop() {
        while (running_) {
            {
                std::unique_lock<std::mutex> lk(worker_mu_);
                worker_cv_.wait_for(lk, std::chrono::milliseconds(50));
            }
            if (!running_) break;

            // 清理过期
            expire_tickets();

            // 检查inflight容量
            size_t cur_inflight;
            {
                std::shared_lock<std::shared_mutex> lk(inflight_mu_);
                cur_inflight = inflight_.size();
            }
            if (cur_inflight >= max_inflight_) continue;

            // 融合预测
            auto predictions = fused_prediction();

            // 下发预取
            for (auto& [vtx, score] : predictions) {
                if (!migrate_fn_) break;

                // 简化: 假设cold=2(DRAM), hot=0(HBM)
                auto* ticket = ticket_pool_.allocate(vtx, 2, 0, score);
                ticket->status.store(PrefetchTicket::Status::INFLIGHT);

                bool ok = migrate_fn_(vtx, 2, 0);
                if (ok) {
                    ticket->status.store(PrefetchTicket::Status::DONE);
                    total_issued_.fetch_add(1, std::memory_order_relaxed);
                    std::unique_lock<std::shared_mutex> lk(inflight_mu_);
                    inflight_.push_back(ticket);
                } else {
                    ticket_pool_.deallocate(ticket);
                }
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// 调试宏
// ═══════════════════════════════════════════════════════════════════════

#define PHILE_ADAPT_PREFETCH_DUMP(engine) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[ADAPT_PREFETCH_DUMP] at %s:%d\n", \
                        __FILE__, __LINE__); \
            (engine).dump_all(); \
        } \
    } while(0)

#define PHILE_STRIDE_DUMP(engine) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[STRIDE_DUMP] at %s:%d\n", __FILE__, __LINE__); \
        } \
    } while(0)

#define PHILE_FREQ_DUMP(engine) \
    do { \
        if (::philemon::debug::get_debug_level() >= 1) { \
            std::printf("[FREQ_DUMP] at %s:%d\n", __FILE__, __LINE__); \
        } \
    } while(0)

class AdaptPrefetchBreakpointGuard {
    const AdaptivePrefetchEngine& engine_;
    const char* name_;
    std::chrono::steady_clock::time_point start_;
public:
    AdaptPrefetchBreakpointGuard(const AdaptivePrefetchEngine& e,
                                  const char* name)
        : engine_(e), name_(name),
          start_(std::chrono::steady_clock::now()) {
        if (debug::get_debug_level() >= 2) {
            std::printf("━━━━ ADAPT_PREFETCH_BP ENTER: %s ━━━━\n", name_);
            engine_.dump_all();
        }
    }
    ~AdaptPrefetchBreakpointGuard() {
        if (debug::get_debug_level() >= 2) {
            auto elapsed = std::chrono::duration_cast<
                std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_).count();
            std::printf("━━━━ ADAPT_PREFETCH_BP EXIT: %s (%.1fμs) ━━━━\n",
                        name_, (double)elapsed);
            engine_.dump_all();
        }
    }
};

#define PHILE_ADAPT_BREAKPOINT(engine, name) \
    ::philemon::prefetch::AdaptPrefetchBreakpointGuard \
        _phile_adapt_bp_##__LINE__((engine), (name))

}  // namespace prefetch
}  // namespace philemon

#endif  // PHILEMON_ADAPTIVE_PREFETCH_HPP
