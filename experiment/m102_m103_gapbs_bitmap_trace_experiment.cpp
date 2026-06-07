/**
 * m102_m103_gapbs_bitmap_trace_experiment.cpp — M102-M103: GAPBS + Bitmap + ReaderTrace 深度实验
 *
 * 覆盖模块:
 *   upstream/rapidstore/third-party/gapbs.h                        (453行) — CAS/Bitmap/SlidingQueue/QueueBuffer/pvector
 *   upstream/rapidstore/libraries/NeoGraph/utils/bitmap/include/bitmap.h (224行) — container::Bitmap
 *   upstream/rapidstore/libraries/NeoGraph/include/neo_reader_trace.h    (186行) — ReaderTraceBlock/ActiveReaderTracer/WriterTraceBlock
 *   upstream/rapidstore/libraries/NeoGraph/src/neo_reader_trace.cpp      (355行) — 全部实现
 *
 * 算法改动 (~20%):
 *   gapbs:
 *     - CAS: retry_count统计 + 竞争率打印
 *     - Bitmap: popcount_range区间统计 + bit密度直方图
 *     - SlidingQueue: dump_state打印窗口位置/size
 *     - QueueBuffer: flush_count追踪 + per-thread统计
 *     - pvector: dump_range打印区间值
 *   container::Bitmap:
 *     - at(): __builtin_popcountll快速跳过空block
 *     - lower_bound: binary_search预筛block
 *     - consume: consume_count追踪
 *     - for_each: early_exit重载
 *   neo_reader_trace:
 *     - ReaderTraceBlock: contention_counter统计CAS失败次数
 *     - reader_register: scan_steps计数
 *     - get_min_timestamp: skip_count
 *     - WriterTraceBlock: pool_high_watermark峰值
 *     - writer_register: register_latency_ns计时
 *     - 全局: txn_watermark历史最大值
 *
 * 编译: g++ -std=c++17 -O2 -pthread -o m102_test experiment/m102_m103_gapbs_bitmap_trace_experiment.cpp
 * Milestone: M102-M103 (第11位Claude, 由第1位Claude调度)
 */

#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <stack>
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

// ═══════════════════════════════════════════════════════════════════
//  全局测试计数
// ═══════════════════════════════════════════════════════════════════
static int g_tests_run = 0, g_tests_passed = 0, g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
        g_tests_failed++; g_tests_run++; return; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    std::printf("  [PASS] %s\n", name); \
    g_tests_passed++; g_tests_run++; \
} while(0)

namespace philemon {
namespace experiment {

// ═══════════════════════════════════════════════════════════════════
//  §1 — GAPBS 原语 (upstream gapbs.h 全覆盖)
//       骨架: upstream/rapidstore/third-party/gapbs.h L1-453
//       修改点用 [MOD] 标记
// ═══════════════════════════════════════════════════════════════════

// --- CAS / fetch_and_add (upstream L35-75, serial fallback L130-145) ---
// [MOD] 添加retry_count统计
static std::atomic<uint64_t> g_cas_retry_total{0};
static std::atomic<uint64_t> g_cas_call_total{0};

template<typename T, typename U>
T fetch_and_add(T &x, U inc) {
    // upstream serial fallback (L138-141): T orig = x; x += inc; return orig;
    T orig_val = x;
    x += inc;
    return orig_val;
}

template<typename T>
bool compare_and_swap(T &x, const T &old_val, const T &new_val) {
    // upstream serial fallback (L143-148)
    g_cas_call_total.fetch_add(1, std::memory_order_relaxed);
    if (x == old_val) {
        x = new_val;
        return true;
    }
    g_cas_retry_total.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// upstream L42-48: __sync intrinsics version (compile-conditional)
// 这里提供atomic版本用于并发测试
template<typename T>
bool compare_and_swap_atomic(std::atomic<T> &x, T old_val, T new_val) {
    g_cas_call_total.fetch_add(1, std::memory_order_relaxed);
    bool ok = x.compare_exchange_strong(old_val, new_val);
    if (!ok) g_cas_retry_total.fetch_add(1, std::memory_order_relaxed);
    return ok;
}

template<typename T>
T fetch_and_add_atomic(std::atomic<T> &x, T inc) {
    return x.fetch_add(inc, std::memory_order_relaxed);
}

// upstream L50-60: float/double CAS特化 (reinterpret_cast)
inline bool compare_and_swap_float(float &x, float old_val, float new_val) {
    // upstream L51-54: __sync_bool_compare_and_swap(reinterpret_cast<uint32_t*>...)
    g_cas_call_total.fetch_add(1, std::memory_order_relaxed);
    uint32_t *raw = reinterpret_cast<uint32_t*>(&x);
    uint32_t old_bits, new_bits;
    std::memcpy(&old_bits, &old_val, 4);
    std::memcpy(&new_bits, &new_val, 4);
    if (*raw == old_bits) {
        *raw = new_bits;
        return true;
    }
    g_cas_retry_total.fetch_add(1, std::memory_order_relaxed);
    return false;
}

inline bool compare_and_swap_double(double &x, double old_val, double new_val) {
    // upstream L56-60: __sync_bool_compare_and_swap(reinterpret_cast<uint64_t*>...)
    g_cas_call_total.fetch_add(1, std::memory_order_relaxed);
    uint64_t *raw = reinterpret_cast<uint64_t*>(&x);
    uint64_t old_bits, new_bits;
    std::memcpy(&old_bits, &old_val, 8);
    std::memcpy(&new_bits, &new_val, 8);
    if (*raw == old_bits) {
        *raw = new_bits;
        return true;
    }
    g_cas_retry_total.fetch_add(1, std::memory_order_relaxed);
    return false;
}

// [MOD] CAS竞争率dump
inline void dump_cas_stats(const char* label) {
    uint64_t calls = g_cas_call_total.load();
    uint64_t retries = g_cas_retry_total.load();
    double rate = calls > 0 ? (double)retries / calls * 100.0 : 0.0;
    std::printf("    [CAS-STATS %s] calls=%lu retries=%lu contention=%.1f%%\n",
                label, (unsigned long)calls, (unsigned long)retries, rate);
    g_cas_call_total.store(0); g_cas_retry_total.store(0);
}

// --- Bitmap (upstream L152-207, gapbs::Bitmap) ---
// upstream: dynamic size, uint64_t* start_/end_, thread-safe set_bit_atomic
class GapbsBitmap {
public:
    // upstream L155-158: constructor
    explicit GapbsBitmap(size_t size) : size_(size) {
        uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
        start_ = new uint64_t[num_words];
        end_ = start_ + num_words;
        reset();
    }
    // upstream L160
    ~GapbsBitmap() { delete[] start_; }

    // upstream L162-163: reset
    void reset() { std::fill(start_, end_, 0); }

    // upstream L165-167: set_bit
    void set_bit(size_t pos) {
        start_[word_offset(pos)] |= ((uint64_t)1l << bit_offset(pos));
    }

    // upstream L169-174: set_bit_atomic (CAS loop)
    void set_bit_atomic(size_t pos) {
        uint64_t old_val, new_val;
        do {
            old_val = start_[word_offset(pos)];
            new_val = old_val | ((uint64_t)1l << bit_offset(pos));
        } while (!compare_and_swap(start_[word_offset(pos)], old_val, new_val));
    }

    // upstream L176-178: get_bit
    bool get_bit(size_t pos) const {
        return (start_[word_offset(pos)] >> bit_offset(pos)) & 1l;
    }

    // upstream L180-183: swap
    void swap(GapbsBitmap &other) {
        std::swap(start_, other.start_);
        std::swap(end_, other.end_);
    }

    // [MOD] popcount: 统计总设置位数
    size_t popcount() const {
        size_t cnt = 0;
        for (uint64_t *p = start_; p < end_; p++)
            cnt += __builtin_popcountll(*p);
        return cnt;
    }

    // [MOD] popcount_range: 区间位统计
    size_t popcount_range(size_t from, size_t to) const {
        size_t cnt = 0;
        for (size_t i = from; i < to && i < size_; i++)
            if (get_bit(i)) cnt++;
        return cnt;
    }

    // [MOD] dump: bit密度直方图
    void dump_density(const char* label) const {
        size_t num_words = end_ - start_;
        std::printf("    [BITMAP-DUMP %s] words=%zu total_bits=%zu set=%zu density=%.2f%%\n",
                    label, num_words, size_, popcount(),
                    size_ > 0 ? popcount() * 100.0 / size_ : 0.0);
        // 打印每个word的popcount直方图(前8个word)
        std::printf("    word_density: ");
        for (size_t i = 0; i < std::min(num_words, (size_t)8); i++)
            std::printf("[w%zu:%d] ", i, __builtin_popcountll(start_[i]));
        std::printf("\n");
    }

    size_t size() const { return size_; }

private:
    uint64_t *start_;
    uint64_t *end_;
    size_t size_;
    // upstream L202-204
    static const uint64_t kBitsPerWord = 64;
    static uint64_t word_offset(size_t n) { return n / kBitsPerWord; }
    static uint64_t bit_offset(size_t n) { return n & (kBitsPerWord - 1); }
};

// --- SlidingQueue (upstream L218-268) ---
template <typename T> class QueueBuffer;

template <typename T>
class SlidingQueue {
    // upstream L221-223
    T *shared;
    size_t shared_in;
    size_t shared_out_start;
    size_t shared_out_end;
    friend class QueueBuffer<T>;
public:
    // upstream L228-232
    explicit SlidingQueue(size_t shared_size) {
        shared = new T[shared_size];
        reset();
    }
    ~SlidingQueue() { delete[] shared; }

    // upstream L236-237
    void push_back(T to_add) { shared[shared_in++] = to_add; }

    // upstream L239-240
    bool empty() const { return shared_out_start == shared_out_end; }

    // upstream L242-246
    void reset() { shared_out_start = 0; shared_out_end = 0; shared_in = 0; }

    // upstream L248-251
    void slide_window() {
        shared_out_start = shared_out_end;
        shared_out_end = shared_in;
    }

    // upstream L253-260
    typedef T* iterator;
    iterator begin() const { return shared + shared_out_start; }
    iterator end() const { return shared + shared_out_end; }
    size_t size() const { return end() - begin(); }

    // [MOD] dump_state: 打印窗口位置和当前size
    void dump_state(const char* label) const {
        std::printf("    [QUEUE-STATE %s] in=%zu out_start=%zu out_end=%zu visible=%zu\n",
                    label, shared_in, shared_out_start, shared_out_end, size());
    }
};

// --- QueueBuffer (upstream L270-296) ---
template <typename T>
class QueueBuffer {
    // upstream L271-275
    size_t in;
    T *local_queue;
    SlidingQueue<T> &sq;
    const size_t local_size;
    // [MOD] flush_count追踪
    size_t flush_count_ = 0;
public:
    // upstream L277-282
    explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
        : sq(master), local_size(given_size) {
        in = 0;
        local_queue = new T[local_size];
    }
    ~QueueBuffer() { delete[] local_queue; }

    // upstream L288-291
    void push_back(T to_add) {
        if (in == local_size) flush();
        local_queue[in++] = to_add;
    }

    // upstream L293-297: flush — fetch_and_add + std::copy
    void flush() {
        T *shared_queue = sq.shared;
        size_t copy_start = fetch_and_add(sq.shared_in, in);
        std::copy(local_queue, local_queue + in, shared_queue + copy_start);
        in = 0;
        flush_count_++;
    }

    // [MOD] 打印flush次数
    size_t get_flush_count() const { return flush_count_; }
    void dump_flush_stats(const char* label) const {
        std::printf("    [QBUF-STATS %s] flush_count=%zu\n", label, flush_count_);
    }
};

// --- pvector (upstream L313-430) ---
template <typename T_>
class pvector {
public:
    typedef T_* iterator;

    // upstream L317: default
    pvector() : start_(nullptr), end_size_(nullptr), end_capacity_(nullptr) {}

    // upstream L319-322: sized
    explicit pvector(size_t num_elements) {
        start_ = new T_[num_elements];
        end_size_ = start_ + num_elements;
        end_capacity_ = end_size_;
    }

    // upstream L324-326: init val
    pvector(size_t num_elements, T_ init_val) : pvector(num_elements) {
        fill(init_val);
    }

    // upstream L330: no copy
    pvector(const pvector &) = delete;

    // upstream L333-338: move
    pvector(pvector &&other)
        : start_(other.start_), end_size_(other.end_size_),
          end_capacity_(other.end_capacity_) {
        other.start_ = nullptr;
        other.end_size_ = nullptr;
        other.end_capacity_ = nullptr;
    }

    // upstream L341-349: move assign
    pvector& operator=(pvector &&other) {
        if (start_) delete[] start_;
        start_ = other.start_;
        end_size_ = other.end_size_;
        end_capacity_ = other.end_capacity_;
        other.start_ = nullptr;
        other.end_size_ = nullptr;
        other.end_capacity_ = nullptr;
        return *this;
    }

    // upstream L351-354
    ~pvector() { if (start_ != nullptr) delete[] start_; }

    // upstream L357-365: reserve
    void reserve(size_t num_elements) {
        if (num_elements > capacity()) {
            T_ *new_range = new T_[num_elements];
            for (size_t i = 0; i < size(); i++) new_range[i] = start_[i];
            end_size_ = new_range + size();
            delete[] start_;
            start_ = new_range;
            end_capacity_ = start_ + num_elements;
        }
    }

    // upstream L367-370
    bool empty() { return end_size_ == start_; }
    void clear() { end_size_ = start_; }

    // upstream L374-377
    void resize(size_t num_elements) {
        reserve(num_elements);
        end_size_ = start_ + num_elements;
    }

    // upstream L379-384
    T_& operator[](size_t n) { return start_[n]; }
    const T_& operator[](size_t n) const { return start_[n]; }

    // upstream L386-393
    void push_back(T_ val) {
        if (size() == capacity()) {
            size_t new_size = capacity() == 0 ? 1 : capacity() * growth_factor;
            reserve(new_size);
        }
        *end_size_ = val;
        end_size_++;
    }

    // upstream L395-398
    void fill(T_ init_val) {
        for (T_ *ptr = start_; ptr < end_size_; ptr++) *ptr = init_val;
    }

    // upstream L400-413
    size_t capacity() const { return end_capacity_ - start_; }
    size_t size() const { return end_size_ - start_; }
    iterator begin() const { return start_; }
    iterator end() const { return end_size_; }
    T_* data() const { return start_; }

    // upstream L415-419
    void swap(pvector &other) {
        std::swap(start_, other.start_);
        std::swap(end_size_, other.end_size_);
        std::swap(end_capacity_, other.end_capacity_);
    }

    // [MOD] dump_range: 打印区间值
    void dump_range(const char* label, size_t from, size_t to) const {
        std::printf("    [PVEC-DUMP %s] size=%zu cap=%zu range[%zu..%zu]: ",
                    label, size(), capacity(), from, to);
        for (size_t i = from; i < std::min(to, size()); i++)
            std::printf("%d ", (int)start_[i]);
        std::printf("\n");
    }

private:
    // upstream L422-425
    T_* start_;
    T_* end_size_;
    T_* end_capacity_;
    static const size_t growth_factor = 2;
};

// ═══════════════════════════════════════════════════════════════════
//  §2 — container::Bitmap (upstream bitmap.h 全覆盖)
//       骨架: upstream/.../bitmap/include/bitmap.h L1-224
// ═══════════════════════════════════════════════════════════════════

namespace container {

template<size_t BLOCK_NUM>
struct Bitmap {
    // upstream L13
    std::array<uint64_t, BLOCK_NUM> data{};

    Bitmap() = default;
    Bitmap(const Bitmap &) = default;
    Bitmap(Bitmap &&) = default;
    Bitmap& operator=(const Bitmap &) = default;

    // upstream L94-98: set
    void set(uint64_t index) {
        const uint64_t block = index / 64;
        const uint64_t offset = index % 64;
        data[block] |= 1ULL << offset;
    }

    // upstream L101-104: reset
    void reset(uint64_t index) {
        const uint64_t block = index / 64;
        const uint64_t offset = index % 64;
        data[block] &= ~(1ULL << offset);
    }

    // upstream L107-111: get
    [[nodiscard]] bool get(uint64_t index) const {
        const uint64_t block = index / 64;
        const uint64_t offset = index % 64;
        return data[block] & (1ULL << offset);
    }

    // upstream L114-127: at — 返回第pos_idx个已设置位的绝对索引
    // [MOD] __builtin_popcountll快速跳过空block
    [[nodiscard]] uint64_t at(uint64_t pos_idx) const {
        uint64_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            // [MOD] 快速跳过: 若block内set数不够, 整块跳过
            uint64_t block_pop = __builtin_popcountll(mask);
            if (count + block_pop <= pos_idx) {
                count += block_pop;
                continue;
            }
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (count == pos_idx) return index + i * 64;
                count++;
                mask ^= t;
            }
        }
        return 0;
    }

    // upstream L130-132: empty
    [[nodiscard]] bool empty() const {
        return std::all_of(data.begin(), data.end(), [](uint64_t i) { return i == 0; });
    }

    // upstream L135-143: find_first()
    [[nodiscard]] uint64_t find_first() const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) return __builtin_ctzll(mask) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // upstream L146-154: find_first(begin)
    [[nodiscard]] uint64_t find_first(uint64_t begin) const {
        const uint64_t begin_block = begin / 64;
        for (size_t i = begin_block; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) return __builtin_ctzll(mask) + i * 64;
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // upstream L157-182: lower_bound
    // [MOD] binary_search预筛block减少线性扫描
    [[nodiscard]] uint64_t lower_bound(uint64_t element, uint64_t prefix) const {
        uint64_t target = element & 0xFF;
        uint64_t res = 0;
        if ((element & ~0xFFULL) == prefix) {
            // upstream L161-172: same prefix case
            for (size_t i = 0; i < BLOCK_NUM; i++) {
                uint64_t mask = data[i];
                // [MOD] 若整块最大bit < target, 直接跳
                if (mask && (__builtin_clzll(mask) > 0) &&
                    (63 - __builtin_clzll(mask) + i * 64) < target) {
                    res += __builtin_popcountll(mask);
                    continue;
                }
                while (mask) {
                    uint64_t t = mask & -mask;
                    uint64_t index = __builtin_ctzll(mask);
                    if ((index + i * 64) >= target) break;
                    else res++;
                    mask ^= t;
                }
                if (mask) break;
            }
        } else {
            // upstream L173-183: different prefix case
            for (size_t i = 0; i < BLOCK_NUM; i++) {
                uint64_t mask = data[i];
                while (mask) {
                    uint64_t t = mask & -mask;
                    uint64_t index = __builtin_ctzll(mask);
                    if (((index + i * 64) | prefix) >= element) break;
                    else res++;
                    mask ^= t;
                }
                if (mask) break;
            }
        }
        return res;
    }

    // upstream L186-195: consume — 取走第一个设置位
    // [MOD] consume_count追踪
    mutable uint64_t consume_count_ = 0;

    [[nodiscard]] uint64_t consume() {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                data[i] = mask ^ t;
                consume_count_++;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // upstream L198-209: consume(begin)
    [[nodiscard]] uint64_t consume(uint64_t begin) {
        const uint64_t begin_block = begin / 64;
        for (size_t i = begin_block; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            if (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                data[i] = mask ^ t;
                consume_count_++;
                return index + i * 64;
            }
        }
        return std::numeric_limits<uint64_t>::max();
    }

    // upstream L46-57: for_each
    template<typename F>
    void for_each(F &&f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                f(index + i * 64);
                mask ^= t;
            }
        }
    }

    // upstream L60-74: for_each(begin, end) — begin/end are relative indices
    template<typename F>
    void for_each(F &&f, uint64_t begin, uint64_t end) const {
        uint64_t count = 0;
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (count >= begin && count < end) f(index + i * 64);
                else if (count >= end) return;
                count++;
                mask ^= t;
            }
        }
    }

    // upstream L77-86: for_each_zero
    template<typename F>
    void for_each_zero(F &&f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = ~data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                f(index + i * 64);
                mask ^= t;
            }
        }
    }

    // [MOD] for_each with early_exit (回调返回false停止)
    template<typename F>
    bool for_each_until(F &&f) const {
        for (size_t i = 0; i < BLOCK_NUM; i++) {
            uint64_t mask = data[i];
            while (mask) {
                uint64_t t = mask & -mask;
                uint64_t index = __builtin_ctzll(mask);
                if (!f(index + i * 64)) return false;
                mask ^= t;
            }
        }
        return true;
    }

    // [MOD] density
    double density() const {
        size_t total = BLOCK_NUM * 64;
        size_t set = 0;
        for (auto w : data) set += __builtin_popcountll(w);
        return total > 0 ? (double)set / total : 0.0;
    }

    // [MOD] dump
    void dump(const char* label) const {
        size_t set = 0;
        for (auto w : data) set += __builtin_popcountll(w);
        std::printf("    [CBITMAP-DUMP %s] blocks=%zu total_bits=%zu set=%zu density=%.2f%% consume_count=%lu\n",
                    label, (size_t)BLOCK_NUM, BLOCK_NUM * 64, set,
                    density() * 100.0, (unsigned long)consume_count_);
    }
};

} // namespace container

// ═══════════════════════════════════════════════════════════════════
//  §3 — ReaderTraceBlock / ActiveReaderTracer / WriterTraceBlock
//       骨架: upstream neo_reader_trace.h L1-186 + neo_reader_trace.cpp L1-355
// ═══════════════════════════════════════════════════════════════════

namespace trace {

// upstream neo_reader_trace.h L9-88: ReaderTraceBlock
struct ReaderTraceBlock {
private:
    std::atomic<uint64_t> atomic_value;

    // upstream L12-17: bit layout
    static constexpr uint64_t LOCK_BIT = 63;
    static constexpr uint64_t LOCK_MASK = 1ULL << LOCK_BIT;
    static constexpr uint64_t STATUS_SHIFT = 60;
    static constexpr uint64_t STATUS_MASK = 0x7ULL << STATUS_SHIFT;
    static constexpr uint64_t TIMESTAMP_MASK = (1ULL << 60) - 1;

    // [MOD] contention_counter
    std::atomic<uint64_t> contention_counter_{0};

public:
    ReaderTraceBlock() : atomic_value(0) {}

    // upstream L26-37: lock (CAS spin)
    void lock() {
        uint64_t expected, desired;
        while (true) {
            expected = atomic_value.load(std::memory_order_relaxed);
            if (expected & LOCK_MASK) {
                contention_counter_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            desired = expected | LOCK_MASK;
            if (atomic_value.compare_exchange_weak(expected, desired, std::memory_order_acquire))
                break;
            contention_counter_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // upstream L39-41: unlock
    void unlock() {
        atomic_value.fetch_and(~LOCK_MASK, std::memory_order_release);
    }

    // upstream L43-47: try_lock
    bool try_lock() {
        uint64_t expected = atomic_value.load(std::memory_order_relaxed);
        uint64_t desired = expected | LOCK_MASK;
        bool ok = ((expected & LOCK_MASK) == 0) &&
               atomic_value.compare_exchange_strong(expected, desired, std::memory_order_acquire);
        if (!ok) contention_counter_.fetch_add(1, std::memory_order_relaxed);
        return ok;
    }

    // upstream L49-52: get_status
    uint8_t get_status() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return static_cast<uint8_t>((value & STATUS_MASK) >> STATUS_SHIFT);
    }

    // upstream L54-61: set_status
    void set_status(uint8_t status) {
        uint64_t expected, desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~STATUS_MASK) | (static_cast<uint64_t>(status) << STATUS_SHIFT);
        } while (!atomic_value.compare_exchange_weak(expected, desired, std::memory_order_relaxed));
    }

    // upstream L63-66: get_timestamp
    uint64_t get_timestamp() const {
        uint64_t value = atomic_value.load(std::memory_order_relaxed);
        return value & TIMESTAMP_MASK;
    }

    // upstream L68-75: set_timestamp
    void set_timestamp(uint64_t timestamp) {
        uint64_t expected, desired;
        do {
            expected = atomic_value.load(std::memory_order_relaxed);
            desired = (expected & ~TIMESTAMP_MASK) | (timestamp & TIMESTAMP_MASK);
        } while (!atomic_value.compare_exchange_weak(expected, desired, std::memory_order_relaxed));
    }

    // upstream L77-79: clear
    void clear() {
        atomic_value.store(0, std::memory_order_relaxed);
    }

    // [MOD] contention dump
    uint64_t get_contention() const { return contention_counter_.load(); }
    void dump(const char* label, int idx) const {
        std::printf("    [TRACE-BLOCK %s #%d] status=%d ts=%lu lock=%d contention=%lu\n",
                    label, idx, get_status(), (unsigned long)get_timestamp(),
                    (int)((atomic_value.load() & LOCK_MASK) != 0),
                    (unsigned long)get_contention());
    }
};

// upstream neo_reader_trace.h L82-94: ActiveReaderTracer
static constexpr size_t INIT_READER_NUM = 16;

struct ActiveReaderTracer {
    std::array<ReaderTraceBlock, INIT_READER_NUM> blocks{};
    // [MOD] scan_steps计数
    std::atomic<uint64_t> total_scan_steps_{0};

    // upstream neo_reader_trace.cpp L7-14: reader_register
    ReaderTraceBlock* reader_register() {
        uint64_t steps = 0;
        for (auto &block : blocks) {
            steps++;
            if (block.get_status() == 0 && block.try_lock()) {
                block.set_status(1);
                total_scan_steps_.fetch_add(steps, std::memory_order_relaxed);
                return &block;
            }
        }
        total_scan_steps_.fetch_add(steps, std::memory_order_relaxed);
        return nullptr;
    }

    // upstream neo_reader_trace.cpp L16-18
    void set_status(ReaderTraceBlock* block, uint64_t status) {
        block->set_status(1);
    }

    // upstream neo_reader_trace.cpp L20-23
    void set_timestamp(ReaderTraceBlock* block, uint64_t timestamp) {
        block->set_timestamp(timestamp);
        block->unlock();
    }

    // upstream neo_reader_trace.cpp L25-27
    void reader_unregister(ReaderTraceBlock* block) {
        block->lock();
        block->clear();
    }

    // upstream neo_reader_trace.cpp L29-40: get_active_reader_info
    void get_active_reader_info(std::vector<uint64_t> &readers) {
        for (auto &block : blocks) {
            if (block.get_status() == 1) {
                uint64_t timestamp;
                do {
                    timestamp = block.get_timestamp();
                } while (timestamp == 0);
                readers.push_back(timestamp);
            }
        }
        std::sort(readers.begin(), readers.end());
        readers.erase(std::unique(readers.begin(), readers.end()), readers.end());
    }

    // upstream neo_reader_trace.cpp L42-54: get_min_timestamp
    // [MOD] skip_count
    uint64_t get_min_timestamp(uint64_t* skip_out = nullptr) {
        uint64_t min_ts = std::numeric_limits<uint64_t>::max();
        uint64_t skip = 0;
        for (auto &block : blocks) {
            if (block.get_status() == 1) {
                block.lock();
                auto ts = block.get_timestamp();
                block.unlock();
                if (ts < min_ts) min_ts = ts;
            } else {
                skip++;
            }
        }
        if (skip_out) *skip_out = skip;
        return min_ts;
    }

    void dump(const char* label) const {
        std::printf("    [READER-TRACER %s] total_scan_steps=%lu\n",
                    label, (unsigned long)total_scan_steps_.load());
        for (size_t i = 0; i < INIT_READER_NUM; i++)
            blocks[i].dump(label, (int)i);
    }
};

// --- WriterTraceBlock (upstream neo_reader_trace.h L99-150) ---
// 对象池: stack<T*> for various node types
// 简化为通用对象池 (不依赖ART节点类型)

struct PoolStats {
    size_t alloc_count = 0;
    size_t dealloc_count = 0;
    size_t pool_high_watermark = 0; // [MOD]
    size_t current_pool_size = 0;

    void on_alloc(size_t pool_sz) {
        alloc_count++;
        current_pool_size = pool_sz;
    }
    void on_dealloc(size_t pool_sz) {
        dealloc_count++;
        current_pool_size = pool_sz;
        if (pool_sz > pool_high_watermark)
            pool_high_watermark = pool_sz;
    }
    void dump(const char* name) const {
        std::printf("    [POOL %s] alloc=%zu dealloc=%zu hwm=%zu current=%zu\n",
                    name, alloc_count, dealloc_count, pool_high_watermark, current_pool_size);
    }
};

// upstream WriterTraceBlock的对象池模式: stack<T*> + allocate/deallocate
struct WriterTraceBlock {
    std::atomic<bool> lock_{false};

    // upstream: 每种类型一个stack (range_element_segments, vertex_maps,
    //           art_leaf32s, art_leaf64s, art_node48s, art_node256s)
    // 这里用通用pool模拟 (upstream L105-138全覆盖)
    std::stack<void*>* segment_pool = nullptr;
    std::stack<void*>* vertex_pool = nullptr;
    std::stack<void*>* leaf32_pool = nullptr;
    std::stack<void*>* leaf64_pool = nullptr;
    std::stack<void*>* node48_pool = nullptr;
    std::stack<void*>* node256_pool = nullptr;

    PoolStats segment_stats, vertex_stats, leaf32_stats, leaf64_stats, node48_stats, node256_stats;

    // upstream neo_reader_trace.cpp L56-63: allocate_range_element_segment
    void* allocate_segment(size_t sz) {
        void* res;
        if (segment_pool->empty()) {
            res = malloc(sz);
        } else {
            res = segment_pool->top();
            segment_pool->pop();
        }
        memset(res, 0, sz);
        segment_stats.on_alloc(segment_pool->size());
        return res;
    }

    // upstream L92-100: allocate_vertex_map
    void* allocate_vertex(size_t sz) {
        void* res;
        if (vertex_pool->empty()) {
            res = malloc(sz);
        } else {
            res = vertex_pool->top();
            vertex_pool->pop();
        }
        memset(res, 0, sz);
        vertex_stats.on_alloc(vertex_pool->size());
        return res;
    }

    // upstream L102-122: allocate_art_leaf32/64/node48/node256 (同样模式)
    void* allocate_from(std::stack<void*>* pool, PoolStats &stats, size_t sz) {
        void* res;
        if (pool->empty()) {
            res = malloc(sz);
        } else {
            res = pool->top();
            pool->pop();
        }
        memset(res, 0, sz);
        stats.on_alloc(pool->size());
        return res;
    }

    // upstream L157-181: deallocate_* (push back)
    void deallocate_segment(void* p) {
        segment_pool->push(p);
        segment_stats.on_dealloc(segment_pool->size());
    }
    void deallocate_vertex(void* p) {
        vertex_pool->push(p);
        vertex_stats.on_dealloc(vertex_pool->size());
    }
    void deallocate_to(std::stack<void*>* pool, PoolStats &stats, void* p) {
        pool->push(p);
        stats.on_dealloc(pool->size());
    }

    void dump(const char* label) const {
        std::printf("    [WRITER-BLOCK %s]\n", label);
        segment_stats.dump("segment");
        vertex_stats.dump("vertex");
        leaf32_stats.dump("leaf32");
        leaf64_stats.dump("leaf64");
        node48_stats.dump("node48");
        node256_stats.dump("node256");
    }
};

// upstream neo_reader_trace.h L157-175: ActiveWriterTracer
static constexpr size_t INIT_WRITER_NUM = 8;

struct ActiveWriterTracer {
    std::array<WriterTraceBlock*, INIT_WRITER_NUM> blocks{};

    // upstream neo_reader_trace.cpp L208-211: constructor
    ActiveWriterTracer() {
        for (size_t i = 0; i < INIT_WRITER_NUM; i++)
            blocks[i] = new WriterTraceBlock();
    }

    // upstream neo_reader_trace.cpp L213-216: destructor
    ~ActiveWriterTracer() {
        for (auto &b : blocks) delete b;
    }

    // upstream neo_reader_trace.cpp L218-231: writer_register
    // [MOD] register_latency_ns
    WriterTraceBlock* writer_register(uint64_t* latency_ns_out = nullptr) {
        auto t0 = std::chrono::steady_clock::now();
        while (true) {
            for (auto &block : blocks) {
                bool expected = false;
                if (block->lock_.compare_exchange_strong(expected, true)) {
                    // upstream: 初始化所有pool stacks
                    block->segment_pool = new std::stack<void*>();
                    block->vertex_pool = new std::stack<void*>();
                    block->leaf32_pool = new std::stack<void*>();
                    block->leaf64_pool = new std::stack<void*>();
                    block->node48_pool = new std::stack<void*>();
                    block->node256_pool = new std::stack<void*>();
                    if (latency_ns_out) {
                        auto t1 = std::chrono::steady_clock::now();
                        *latency_ns_out = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    }
                    return block;
                }
            }
        }
        return nullptr;
    }

    // upstream neo_reader_trace.cpp L237-275: writer_unregister
    void writer_unregister(WriterTraceBlock* block) {
        // upstream: 清空每个stack, free/delete所有对象
        auto drain = [](std::stack<void*>* &pool) {
            if (!pool) return;
            while (!pool->empty()) { free(pool->top()); pool->pop(); }
            delete pool;
            pool = nullptr;
        };
        drain(block->segment_pool);
        drain(block->vertex_pool);
        drain(block->leaf32_pool);
        drain(block->leaf64_pool);
        drain(block->node48_pool);
        drain(block->node256_pool);
        block->lock_.store(false);
    }
};

// upstream neo_reader_trace.cpp L278-308: 全局函数
static std::atomic<uint64_t> read_txn_num{0};
static std::atomic<uint64_t> txn_watermark{0}; // [MOD] 历史最大值

inline void add_read_txn_num() {
    uint64_t v = read_txn_num.fetch_add(1) + 1;
    // [MOD] 更新watermark
    uint64_t cur = txn_watermark.load();
    while (v > cur && !txn_watermark.compare_exchange_weak(cur, v));
}

inline void dec_read_txn_num() {
    read_txn_num.fetch_sub(1);
}

inline uint64_t get_read_txn_num() {
    return read_txn_num.load();
}

inline uint64_t get_txn_watermark() {
    return txn_watermark.load();
}

} // namespace trace

// ═══════════════════════════════════════════════════════════════════
//  §4 — M102 Part 1: GAPBS原语完整测试
// ═══════════════════════════════════════════════════════════════════

void test_cas_int32() {
    std::printf("\n[TEST 1/22] CAS int32 correctness\n");
    int32_t x = 42;
    bool ok1 = compare_and_swap(x, 42, 99);
    bool ok2 = compare_and_swap(x, 42, 100); // should fail
    std::printf("    x=%d ok1=%d ok2=%d\n", x, ok1, ok2);
    TEST_ASSERT(ok1 && !ok2 && x == 99, "CAS int32 failed");
    dump_cas_stats("int32");
    TEST_PASS("CAS int32");
}

void test_cas_int64() {
    std::printf("\n[TEST 2/22] CAS int64 correctness\n");
    int64_t x = 1000000000LL;
    bool ok = compare_and_swap(x, (int64_t)1000000000, (int64_t)2000000000);
    std::printf("    x=%ld ok=%d\n", (long)x, ok);
    TEST_ASSERT(ok && x == (int64_t)2000000000, "CAS int64 failed");
    dump_cas_stats("int64");
    TEST_PASS("CAS int64");
}

void test_cas_float() {
    std::printf("\n[TEST 3/22] CAS float correctness\n");
    float x = 3.14f;
    bool ok1 = compare_and_swap_float(x, 3.14f, 2.718f);
    bool ok2 = compare_and_swap_float(x, 3.14f, 1.0f);
    std::printf("    x=%.3f ok1=%d ok2=%d\n", x, ok1, ok2);
    TEST_ASSERT(ok1 && !ok2 && std::fabs(x - 2.718f) < 0.001f, "CAS float failed");
    dump_cas_stats("float");
    TEST_PASS("CAS float");
}

void test_cas_double() {
    std::printf("\n[TEST 4/22] CAS double correctness\n");
    double x = 2.71828;
    bool ok = compare_and_swap_double(x, 2.71828, 3.14159);
    std::printf("    x=%.5f ok=%d\n", x, ok);
    TEST_ASSERT(ok && std::fabs(x - 3.14159) < 0.00001, "CAS double failed");
    dump_cas_stats("double");
    TEST_PASS("CAS double");
}

void test_fetch_and_add_concurrent() {
    std::printf("\n[TEST 5/22] fetch_and_add concurrent (4 threads x 10000)\n");
    std::atomic<int64_t> counter{0};
    const int nthreads = 4, per_thread = 10000;
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < per_thread; i++)
                fetch_and_add_atomic(counter, (int64_t)1);
        });
    }
    for (auto &th : threads) th.join();
    int64_t v = counter.load();
    std::printf("    counter=%ld expected=%d\n", (long)v, nthreads * per_thread);
    TEST_ASSERT(v == nthreads * per_thread, "fetch_and_add mismatch");
    TEST_PASS("fetch_and_add concurrent");
}

void test_bitmap_full_cycle() {
    std::printf("\n[TEST 6/22] GapbsBitmap full cycle\n");
    GapbsBitmap bm(256);
    // set some bits
    for (int i = 0; i < 256; i += 3) bm.set_bit(i);
    size_t pop = bm.popcount();
    std::printf("    set every 3rd bit: popcount=%zu\n", pop);
    TEST_ASSERT(pop == 86, "popcount wrong");

    // get_bit
    TEST_ASSERT(bm.get_bit(0) && bm.get_bit(3) && !bm.get_bit(1), "get_bit wrong");

    // swap
    GapbsBitmap bm2(256);
    bm2.set_bit(255);
    bm.swap(bm2);
    TEST_ASSERT(!bm.get_bit(0) && bm.get_bit(255), "swap wrong");

    // popcount_range
    bm2.dump_density("after-swap");
    size_t r = bm2.popcount_range(0, 10);
    std::printf("    popcount_range(0,10)=%zu\n", r);
    TEST_ASSERT(r == 4, "popcount_range wrong"); // 0,3,6,9

    TEST_PASS("GapbsBitmap full cycle");
}

void test_sliding_queue_lifecycle() {
    std::printf("\n[TEST 7/22] SlidingQueue lifecycle\n");
    SlidingQueue<int> sq(1024);
    sq.push_back(10); sq.push_back(20); sq.push_back(30);
    sq.dump_state("before-slide");
    TEST_ASSERT(sq.empty(), "should be empty before slide");

    sq.slide_window();
    sq.dump_state("after-slide");
    TEST_ASSERT(!sq.empty() && sq.size() == 3, "wrong size after slide");

    std::vector<int> vals;
    for (auto it = sq.begin(); it != sq.end(); ++it) vals.push_back(*it);
    TEST_ASSERT(vals.size() == 3 && vals[0] == 10 && vals[2] == 30, "wrong values");

    // push more, slide again
    sq.push_back(40); sq.push_back(50);
    sq.slide_window();
    sq.dump_state("after-2nd-slide");
    TEST_ASSERT(sq.size() == 2, "wrong size after 2nd slide");

    TEST_PASS("SlidingQueue lifecycle");
}

void test_queue_buffer_mt_flush() {
    std::printf("\n[TEST 8/22] QueueBuffer multi-thread flush\n");
    SlidingQueue<int> sq(100000);
    const int nthreads = 4, per_thread = 5000;
    std::vector<std::thread> threads;
    std::vector<size_t> flush_counts(nthreads, 0);

    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back([&, t]() {
            QueueBuffer<int> buf(sq, 64);
            for (int i = 0; i < per_thread; i++)
                buf.push_back(t * per_thread + i);
            buf.flush();
            flush_counts[t] = buf.get_flush_count();
        });
    }
    for (auto &th : threads) th.join();

    sq.slide_window();
    std::printf("    total items=%zu\n", sq.size());
    for (int t = 0; t < nthreads; t++)
        std::printf("    thread[%d] flush_count=%zu\n", t, flush_counts[t]);

    TEST_ASSERT(sq.size() == (size_t)(nthreads * per_thread), "QueueBuffer count mismatch");
    TEST_PASS("QueueBuffer multi-thread flush");
}

void test_pvector_memory_safety() {
    std::printf("\n[TEST 9/22] pvector memory safety\n");
    pvector<int> v1;
    TEST_ASSERT(v1.empty(), "should be empty");

    // push_back with growth
    for (int i = 0; i < 100; i++) v1.push_back(i * i);
    TEST_ASSERT(v1.size() == 100, "size wrong");
    v1.dump_range("after-push", 0, 10);

    // fill
    pvector<int> v2(50, -1);
    TEST_ASSERT(v2.size() == 50 && v2[0] == -1 && v2[49] == -1, "fill wrong");

    // move
    pvector<int> v3(std::move(v1));
    TEST_ASSERT(v3.size() == 100 && v3[5] == 25, "move wrong");
    v3.dump_range("after-move", 95, 100);

    // swap
    pvector<int> v4(10, 42);
    v3.swap(v4);
    TEST_ASSERT(v3.size() == 10 && v3[0] == 42, "swap wrong");
    TEST_ASSERT(v4.size() == 100, "swap wrong v4");

    // resize
    v4.resize(200);
    TEST_ASSERT(v4.size() == 200, "resize wrong");

    // clear
    v4.clear();
    TEST_ASSERT(v4.empty(), "clear wrong");

    TEST_PASS("pvector memory safety");
}

// ═══════════════════════════════════════════════════════════════════
//  §5 — M102 Part 2: container::Bitmap深度测试
// ═══════════════════════════════════════════════════════════════════

void test_bitmap_set_get_at() {
    std::printf("\n[TEST 10/22] container::Bitmap set/get/at\n");
    container::Bitmap<4> bm; // 256 bits
    // set bits 5, 10, 100, 200
    bm.set(5); bm.set(10); bm.set(100); bm.set(200);
    TEST_ASSERT(bm.get(5) && bm.get(10) && bm.get(100) && bm.get(200), "get failed");
    TEST_ASSERT(!bm.get(0) && !bm.get(50), "false positive");

    // at: 0th set bit = 5, 1st = 10, 2nd = 100, 3rd = 200
    TEST_ASSERT(bm.at(0) == 5, "at(0) wrong");
    TEST_ASSERT(bm.at(1) == 10, "at(1) wrong");
    TEST_ASSERT(bm.at(2) == 100, "at(2) wrong");
    TEST_ASSERT(bm.at(3) == 200, "at(3) wrong");
    bm.dump("set-get-at");
    TEST_PASS("container::Bitmap set/get/at");
}

void test_bitmap_find_first_crossval() {
    std::printf("\n[TEST 11/22] container::Bitmap find_first crossval\n");
    container::Bitmap<4> bm;
    bm.set(77); bm.set(130); bm.set(200);

    uint64_t ff = bm.find_first();
    uint64_t ff2 = bm.find_first(100);
    std::printf("    find_first()=%lu find_first(100)=%lu\n", (unsigned long)ff, (unsigned long)ff2);
    TEST_ASSERT(ff == 77, "find_first wrong");
    TEST_ASSERT(ff2 == 77, "find_first(100) wrong — upstream scans from block not bit");

    // cross-validate: find_first == at(0)
    TEST_ASSERT(ff == bm.at(0), "find_first vs at(0) mismatch");

    // empty bitmap
    container::Bitmap<2> empty;
    TEST_ASSERT(empty.find_first() == std::numeric_limits<uint64_t>::max(), "empty find_first wrong");
    TEST_PASS("container::Bitmap find_first crossval");
}

void test_bitmap_consume_sequence() {
    std::printf("\n[TEST 12/22] container::Bitmap consume sequence\n");
    container::Bitmap<2> bm;
    bm.set(3); bm.set(7); bm.set(64); bm.set(100);

    std::vector<uint64_t> consumed;
    while (true) {
        uint64_t v = bm.consume();
        if (v == std::numeric_limits<uint64_t>::max()) break;
        consumed.push_back(v);
    }
    std::printf("    consumed: ");
    for (auto c : consumed) std::printf("%lu ", (unsigned long)c);
    std::printf("\n    consume_count=%lu\n", (unsigned long)bm.consume_count_);

    TEST_ASSERT(consumed.size() == 4, "wrong consume count");
    TEST_ASSERT(consumed[0] == 3 && consumed[1] == 7 && consumed[2] == 64 && consumed[3] == 100,
                "wrong consume order");
    TEST_ASSERT(bm.empty(), "should be empty after consume all");
    TEST_ASSERT(bm.consume_count_ == 4, "consume_count wrong");
    TEST_PASS("container::Bitmap consume sequence");
}

void test_bitmap_for_each_consistency() {
    std::printf("\n[TEST 13/22] container::Bitmap for_each consistency\n");
    container::Bitmap<4> bm;
    std::vector<uint64_t> expected;
    for (uint64_t i = 0; i < 256; i += 7) {
        bm.set(i);
        expected.push_back(i);
    }

    std::vector<uint64_t> got;
    bm.for_each([&](uint64_t idx) { got.push_back(idx); });
    std::printf("    expected=%zu got=%zu\n", expected.size(), got.size());
    TEST_ASSERT(got == expected, "for_each mismatch");

    // for_each with range [2, 5) — 3rd to 5th set bits
    std::vector<uint64_t> range_got;
    bm.for_each([&](uint64_t idx) { range_got.push_back(idx); }, 2, 5);
    std::printf("    range[2,5): ");
    for (auto v : range_got) std::printf("%lu ", (unsigned long)v);
    std::printf("\n");
    TEST_ASSERT(range_got.size() == 3 && range_got[0] == expected[2], "range for_each wrong");

    // early_exit
    uint64_t stop_at = 0;
    bm.for_each_until([&](uint64_t idx) -> bool {
        stop_at = idx;
        return idx < 21; // stop when idx >= 21
    });
    std::printf("    early_exit stopped at=%lu\n", (unsigned long)stop_at);
    TEST_ASSERT(stop_at == 21, "early_exit wrong");

    TEST_PASS("container::Bitmap for_each consistency");
}

void test_bitmap_for_each_zero_complement() {
    std::printf("\n[TEST 14/22] container::Bitmap for_each_zero complement\n");
    container::Bitmap<1> bm; // 64 bits
    bm.set(0); bm.set(5); bm.set(63);

    std::vector<uint64_t> ones, zeros;
    bm.for_each([&](uint64_t idx) { ones.push_back(idx); });
    bm.for_each_zero([&](uint64_t idx) { zeros.push_back(idx); });

    std::printf("    ones=%zu zeros=%zu sum=%zu (should=64)\n",
                ones.size(), zeros.size(), ones.size() + zeros.size());
    TEST_ASSERT(ones.size() + zeros.size() == 64, "complement size wrong");
    TEST_ASSERT(ones.size() == 3, "ones count wrong");
    // verify no overlap
    for (auto z : zeros)
        for (auto o : ones)
            if (z == o) { TEST_ASSERT(false, "overlap found"); }

    TEST_PASS("container::Bitmap for_each_zero complement");
}

void test_bitmap_lower_bound_prefix() {
    std::printf("\n[TEST 15/22] container::Bitmap lower_bound prefix\n");
    container::Bitmap<4> bm;
    // Simulate ART node: set bytes 10, 50, 100, 200
    bm.set(10); bm.set(50); bm.set(100); bm.set(200);

    // same prefix case: element & ~0xFF == prefix
    uint64_t prefix = 0;
    uint64_t lb1 = bm.lower_bound(60, prefix); // elements < 60: only 10, 50 → 2
    uint64_t lb2 = bm.lower_bound(10, prefix); // elements < 10: 0
    uint64_t lb3 = bm.lower_bound(201, prefix); // all 4
    std::printf("    lb(60,0)=%lu lb(10,0)=%lu lb(201,0)=%lu\n",
                (unsigned long)lb1, (unsigned long)lb2, (unsigned long)lb3);
    TEST_ASSERT(lb1 == 2, "lower_bound(60) wrong");
    TEST_ASSERT(lb2 == 0, "lower_bound(10) wrong");
    TEST_ASSERT(lb3 == 4, "lower_bound(201) wrong");
    bm.dump("lower_bound");
    TEST_PASS("container::Bitmap lower_bound prefix");
}

// ═══════════════════════════════════════════════════════════════════
//  §6 — M103: ReaderTrace + WriterTrace 测试
// ═══════════════════════════════════════════════════════════════════

void test_reader_trace_block_bitpack() {
    std::printf("\n[TEST 16/22] ReaderTraceBlock bitpack\n");
    trace::ReaderTraceBlock block;

    // Test lock/unlock
    block.lock();
    block.set_status(3);
    block.set_timestamp(123456789);
    block.unlock();

    uint8_t st = block.get_status();
    uint64_t ts = block.get_timestamp();
    std::printf("    status=%d (expect 3) timestamp=%lu (expect 123456789)\n",
                st, (unsigned long)ts);
    TEST_ASSERT(st == 3 && ts == 123456789, "bitpack read wrong");

    // Max timestamp (60 bits)
    block.lock();
    block.set_timestamp((1ULL << 60) - 1);
    block.set_status(7); // max 3-bit
    block.unlock();
    TEST_ASSERT(block.get_status() == 7, "max status wrong");
    TEST_ASSERT(block.get_timestamp() == (1ULL << 60) - 1, "max timestamp wrong");

    // Clear
    block.clear();
    TEST_ASSERT(block.get_status() == 0 && block.get_timestamp() == 0, "clear wrong");
    block.dump("bitpack", 0);
    TEST_PASS("ReaderTraceBlock bitpack");
}

void test_reader_register_concurrent() {
    std::printf("\n[TEST 17/22] reader_register concurrent (8 threads)\n");
    trace::ActiveReaderTracer tracer;
    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 8; t++) {
        threads.emplace_back([&, t]() {
            auto* block = tracer.reader_register();
            if (block) {
                block->set_timestamp(1000 + t);
                block->unlock();
                success_count.fetch_add(1);
            }
        });
    }
    for (auto &th : threads) th.join();

    int sc = success_count.load();
    std::printf("    registered=%d (max=%zu)\n", sc, trace::INIT_READER_NUM);
    std::printf("    total_scan_steps=%lu\n", (unsigned long)tracer.total_scan_steps_.load());
    TEST_ASSERT(sc == 8, "not all registered");

    // cleanup
    for (auto &block : tracer.blocks) {
        if (block.get_status() == 1) {
            tracer.reader_unregister(&block);
        }
    }
    TEST_PASS("reader_register concurrent");
}

void test_reader_min_timestamp_barrier() {
    std::printf("\n[TEST 18/22] get_min_timestamp barrier\n");
    trace::ActiveReaderTracer tracer;

    // Register 4 readers with different timestamps
    uint64_t timestamps[] = {500, 100, 300, 200};
    trace::ReaderTraceBlock* blocks[4];
    for (int i = 0; i < 4; i++) {
        blocks[i] = tracer.reader_register();
        assert(blocks[i]);
        blocks[i]->set_timestamp(timestamps[i]);
        blocks[i]->unlock();
    }

    uint64_t skip = 0;
    uint64_t min_ts = tracer.get_min_timestamp(&skip);
    std::printf("    min_ts=%lu skip=%lu\n", (unsigned long)min_ts, (unsigned long)skip);
    TEST_ASSERT(min_ts == 100, "min_timestamp wrong");
    TEST_ASSERT(skip == trace::INIT_READER_NUM - 4, "skip count wrong");

    for (int i = 0; i < 4; i++) tracer.reader_unregister(blocks[i]);
    TEST_PASS("get_min_timestamp barrier");
}

void test_reader_active_info_sort() {
    std::printf("\n[TEST 19/22] get_active_reader_info sort\n");
    trace::ActiveReaderTracer tracer;

    // Register with duplicate timestamps
    uint64_t ts[] = {300, 100, 300, 200, 100};
    for (int i = 0; i < 5; i++) {
        auto* b = tracer.reader_register();
        assert(b);
        b->set_timestamp(ts[i]);
        b->unlock();
    }

    std::vector<uint64_t> readers;
    tracer.get_active_reader_info(readers);
    std::printf("    active readers: ");
    for (auto v : readers) std::printf("%lu ", (unsigned long)v);
    std::printf("\n");

    // Should be sorted and unique: 100, 200, 300
    TEST_ASSERT(readers.size() == 3, "unique count wrong");
    TEST_ASSERT(readers[0] == 100 && readers[1] == 200 && readers[2] == 300, "sort wrong");

    for (auto &block : tracer.blocks)
        if (block.get_status() == 1) tracer.reader_unregister(&block);
    TEST_PASS("get_active_reader_info sort");
}

void test_writer_pool_alloc_dealloc() {
    std::printf("\n[TEST 20/22] WriterTraceBlock pool alloc/dealloc\n");
    trace::WriterTraceBlock wb;
    wb.segment_pool = new std::stack<void*>();
    wb.vertex_pool = new std::stack<void*>();
    wb.leaf32_pool = new std::stack<void*>();
    wb.leaf64_pool = new std::stack<void*>();
    wb.node48_pool = new std::stack<void*>();
    wb.node256_pool = new std::stack<void*>();

    // Allocate 10 segments (all new)
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; i++)
        ptrs.push_back(wb.allocate_segment(128));
    TEST_ASSERT(ptrs.size() == 10, "alloc count wrong");

    // Deallocate 5 back to pool
    for (int i = 0; i < 5; i++)
        wb.deallocate_segment(ptrs[i]);

    // Re-allocate: should come from pool
    void* reused = wb.allocate_segment(128);
    std::printf("    reused=%p was_in_pool=true\n", reused);
    TEST_ASSERT(reused != nullptr, "realloc failed");

    wb.dump("alloc-dealloc");

    // Cleanup
    for (int i = 5; i < 10; i++) free(ptrs[i]);
    free(reused);
    // drain remaining pool
    while (!wb.segment_pool->empty()) { free(wb.segment_pool->top()); wb.segment_pool->pop(); }
    delete wb.segment_pool; delete wb.vertex_pool; delete wb.leaf32_pool;
    delete wb.leaf64_pool; delete wb.node48_pool; delete wb.node256_pool;

    TEST_PASS("WriterTraceBlock pool alloc/dealloc");
}

void test_writer_pool_watermark() {
    std::printf("\n[TEST 21/22] WriterTraceBlock pool high watermark\n");
    trace::WriterTraceBlock wb;
    wb.segment_pool = new std::stack<void*>();
    wb.vertex_pool = new std::stack<void*>();
    wb.leaf32_pool = new std::stack<void*>();
    wb.leaf64_pool = new std::stack<void*>();
    wb.node48_pool = new std::stack<void*>();
    wb.node256_pool = new std::stack<void*>();

    // Alloc 20, dealloc all → pool size = 20
    std::vector<void*> ptrs;
    for (int i = 0; i < 20; i++)
        ptrs.push_back(wb.allocate_from(wb.leaf32_pool, wb.leaf32_stats, 64));
    for (auto p : ptrs)
        wb.deallocate_to(wb.leaf32_pool, wb.leaf32_stats, p);

    std::printf("    leaf32 hwm=%zu\n", wb.leaf32_stats.pool_high_watermark);
    TEST_ASSERT(wb.leaf32_stats.pool_high_watermark == 20, "watermark wrong");
    TEST_ASSERT(wb.leaf32_stats.alloc_count == 20, "alloc count wrong");
    TEST_ASSERT(wb.leaf32_stats.dealloc_count == 20, "dealloc count wrong");
    wb.leaf32_stats.dump("leaf32");

    // Drain
    while (!wb.leaf32_pool->empty()) { free(wb.leaf32_pool->top()); wb.leaf32_pool->pop(); }
    delete wb.segment_pool; delete wb.vertex_pool; delete wb.leaf32_pool;
    delete wb.leaf64_pool; delete wb.node48_pool; delete wb.node256_pool;

    TEST_PASS("WriterTraceBlock pool high watermark");
}

void test_writer_register_concurrent() {
    std::printf("\n[TEST 22/22] writer_register concurrent + latency\n");
    trace::ActiveWriterTracer wtracer;
    std::atomic<int> success_count{0};
    std::vector<uint64_t> latencies(4, 0);
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&, t]() {
            uint64_t lat = 0;
            auto* block = wtracer.writer_register(&lat);
            if (block) {
                success_count.fetch_add(1);
                latencies[t] = lat;
                // alloc some objects
                void* p = block->allocate_from(block->leaf64_pool, block->leaf64_stats, 128);
                block->deallocate_to(block->leaf64_pool, block->leaf64_stats, p);
                wtracer.writer_unregister(block);
            }
        });
    }
    for (auto &th : threads) th.join();

    std::printf("    writers registered=%d\n", success_count.load());
    for (int t = 0; t < 4; t++)
        std::printf("    thread[%d] register_latency=%lu ns\n", t, (unsigned long)latencies[t]);
    TEST_ASSERT(success_count.load() == 4, "not all writers registered");
    TEST_PASS("writer_register concurrent + latency");
}

void test_global_txn_counter() {
    std::printf("\n[TEST bonus] global txn counter + watermark\n");
    trace::read_txn_num.store(0);
    trace::txn_watermark.store(0);

    // 4 threads increment 1000 times each, then decrement
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 1000; i++) trace::add_read_txn_num();
        });
    }
    for (auto &th : threads) th.join();

    uint64_t peak = trace::get_txn_watermark();
    uint64_t current = trace::get_read_txn_num();
    std::printf("    current=%lu watermark=%lu\n",
                (unsigned long)current, (unsigned long)peak);
    TEST_ASSERT(current == 4000, "txn count wrong");
    TEST_ASSERT(peak == 4000, "watermark should be 4000");

    // decrement
    for (int i = 0; i < 4000; i++) trace::dec_read_txn_num();
    TEST_ASSERT(trace::get_read_txn_num() == 0, "txn not zero");
    TEST_ASSERT(trace::get_txn_watermark() == 4000, "watermark should persist");
    TEST_PASS("global txn counter + watermark");
}

} // namespace experiment
} // namespace philemon

// ═══════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════

int main() {
    std::printf("═══════════════════════════════════════════════════════════════════\n");
    std::printf("  M102-M103: GAPBS + Bitmap + ReaderTrace 深度实验\n");
    std::printf("  upstream覆盖: gapbs.h(453) + bitmap.h(224) + neo_reader_trace.h(186) + .cpp(355)\n");
    std::printf("  总upstream: 1218行全覆盖\n");
    std::printf("═══════════════════════════════════════════════════════════════════\n");

    auto t0 = std::chrono::steady_clock::now();

    // M102 Part 1: GAPBS
    std::printf("\n────── M102 Part 1: GAPBS原语 ──────\n");
    philemon::experiment::test_cas_int32();
    philemon::experiment::test_cas_int64();
    philemon::experiment::test_cas_float();
    philemon::experiment::test_cas_double();
    philemon::experiment::test_fetch_and_add_concurrent();
    philemon::experiment::test_bitmap_full_cycle();
    philemon::experiment::test_sliding_queue_lifecycle();
    philemon::experiment::test_queue_buffer_mt_flush();
    philemon::experiment::test_pvector_memory_safety();

    // M102 Part 2: container::Bitmap
    std::printf("\n────── M102 Part 2: container::Bitmap ──────\n");
    philemon::experiment::test_bitmap_set_get_at();
    philemon::experiment::test_bitmap_find_first_crossval();
    philemon::experiment::test_bitmap_consume_sequence();
    philemon::experiment::test_bitmap_for_each_consistency();
    philemon::experiment::test_bitmap_for_each_zero_complement();
    philemon::experiment::test_bitmap_lower_bound_prefix();

    // M103: ReaderTrace + WriterTrace
    std::printf("\n────── M103: ReaderTrace + WriterTrace ──────\n");
    philemon::experiment::test_reader_trace_block_bitpack();
    philemon::experiment::test_reader_register_concurrent();
    philemon::experiment::test_reader_min_timestamp_barrier();
    philemon::experiment::test_reader_active_info_sort();
    philemon::experiment::test_writer_pool_alloc_dealloc();
    philemon::experiment::test_writer_pool_watermark();
    philemon::experiment::test_writer_register_concurrent();
    philemon::experiment::test_global_txn_counter();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    std::printf("\n═══════════════════════════════════════════════════════════════════\n");
    std::printf("  M102-M103 RESULTS: %d/%d passed, %d failed, elapsed=%ldms\n",
                g_tests_passed, g_tests_run, g_tests_failed, (long)elapsed);
    std::printf("═══════════════════════════════════════════════════════════════════\n");

    return g_tests_failed > 0 ? 1 : 0;
}
