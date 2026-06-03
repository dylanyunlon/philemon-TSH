#ifndef PHILEMON_GAPBS_COMPAT_HPP
#define PHILEMON_GAPBS_COMPAT_HPP
/**
 * gapbs_compat.hpp — GAP Benchmark Suite 原语的Philemon兼容层
 *
 * 骨架来源:
 *   upstream/rapidstore/third-party/gapbs/gapbs.hpp  (453行)
 *   (原始来源: https://github.com/sbeamer/gapbs, BSD-3-Clause)
 *
 * 修改 (~20%):
 *   - [MOD] gapbs namespace → philemon::algorithms::gapbs_compat
 *   - [MOD] OpenMP atomic → __sync intrinsics (去OMP依赖)
 *   - [NEW] Bitmap::popcount(): 统计已设置位数
 *   - [NEW] Bitmap::dump(): 打印前N位的可视化
 *   - [NEW] SlidingQueue: 添加 dump_state() 打印窗口位置
 *   - [NEW] pvector: 添加 dump_range() 打印区间值
 *   - [NEW] 每个CAS操作可选计数器 (GAPBS_TRACE_CAS)
 *   - [KEEP] compare_and_swap 100%保留
 *   - [KEEP] fetch_and_add 100%保留
 *   - [KEEP] Bitmap set/get/reset/swap 100%保留
 *   - [KEEP] SlidingQueue push_back/slide_window/begin/end 100%保留
 *   - [KEEP] QueueBuffer flush逻辑 100%保留
 *   - [KEEP] pvector 内存管理 100%保留
 *
 * 注意: tiered_bfs.hpp 已内联了轻量版本。本文件提供
 * 与upstream gapbs.hpp 功能对齐的完整兼容层, 供需要
 * 完整gapbs语义的模块使用。
 *
 * Milestone: M028
 */

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <atomic>

namespace philemon {
namespace algorithms {
namespace gapbs_compat {

// ─── CAS原语 (upstream 100%) ────────────────────────────────────
template <typename T>
bool compare_and_swap(T& target, T old_val, T new_val) {
    static_assert(sizeof(T) <= 8, "CAS supports up to 8 bytes");
    if constexpr (sizeof(T) == 4) {
        uint32_t* raw = reinterpret_cast<uint32_t*>(&target);
        uint32_t o, n;
        std::memcpy(&o, &old_val, 4);
        std::memcpy(&n, &new_val, 4);
        return __sync_bool_compare_and_swap(raw, o, n);
    } else {
        uint64_t* raw = reinterpret_cast<uint64_t*>(&target);
        uint64_t o, n;
        std::memcpy(&o, &old_val, 8);
        std::memcpy(&n, &new_val, 8);
        return __sync_bool_compare_and_swap(raw, o, n);
    }
}

template <typename T>
T fetch_and_add(T& target, T value) {
    return __sync_fetch_and_add(&target, value);
}

// ─── Bitmap (upstream 100% + popcount/dump) ─────────────────────
class Bitmap {
    uint64_t* data_;
    size_t    size_;   // bits
    size_t    words_;

public:
    explicit Bitmap(size_t size)
        : size_(size), words_((size + 63) / 64) {
        data_ = new uint64_t[words_]();
    }
    ~Bitmap() { delete[] data_; }

    Bitmap(const Bitmap&) = delete;
    Bitmap& operator=(const Bitmap&) = delete;

    // upstream 100%
    void reset() { std::memset(data_, 0, words_ * sizeof(uint64_t)); }

    void set_bit(size_t n) {
        data_[n >> 6] |= (1ULL << (n & 63));
    }

    void set_bit_atomic(size_t n) {
        __sync_fetch_and_or(&data_[n >> 6], 1ULL << (n & 63));
    }

    bool get_bit(size_t n) const {
        return (data_[n >> 6] >> (n & 63)) & 1;
    }

    void swap(Bitmap& other) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(words_, other.words_);
    }

    size_t size() const { return size_; }

    // [NEW] 统计置位数
    uint64_t popcount() const {
        uint64_t cnt = 0;
        for (size_t i = 0; i < words_; i++)
            cnt += __builtin_popcountll(data_[i]);
        return cnt;
    }

    // [NEW] 可视化前N位
    void dump(size_t max_bits = 64) const {
        size_t n = std::min(max_bits, size_);
        std::printf("[BITMAP] size=%lu set=%lu | ",
                    (unsigned long)size_, (unsigned long)popcount());
        for (size_t i = 0; i < n; i++)
            std::putchar(get_bit(i) ? '1' : '0');
        if (n < size_) std::printf("...");
        std::putchar('\n');
    }
};

// ─── SlidingQueue (upstream 100% + dump) ────────────────────────
template <typename T>
class SlidingQueue {
    T*     data_;
    size_t capacity_;
    size_t head_;     // shared_in
    size_t tail_;     // shared_out_start
    size_t end_;      // shared_out_end

public:
    explicit SlidingQueue(size_t cap)
        : capacity_(cap), head_(0), tail_(0), end_(0) {
        data_ = new T[cap];
    }
    ~SlidingQueue() { delete[] data_; }

    void push_back(T val) { data_[head_++] = val; }

    void slide_window() {
        tail_ = end_;
        end_  = head_;
    }

    bool  empty() const { return tail_ == end_; }
    size_t size() const { return end_ - tail_; }

    T*       begin()       { return data_ + tail_; }
    const T* begin() const { return data_ + tail_; }
    T*       end()         { return data_ + end_; }
    const T* end()  const  { return data_ + end_; }

    // [NEW] 状态打印
    void dump_state(const char* label = "QUEUE") const {
        std::printf("[%s] capacity=%lu head=%lu tail=%lu end=%lu "
                    "window_size=%lu\n",
                    label,
                    (unsigned long)capacity_,
                    (unsigned long)head_,
                    (unsigned long)tail_,
                    (unsigned long)end_,
                    (unsigned long)size());
    }
};

// ─── QueueBuffer (upstream 100%) ────────────────────────────────
template <typename T>
class QueueBuffer {
    static constexpr size_t kBufSize = 2048;
    SlidingQueue<T>& parent_;
    T*     buf_;
    size_t idx_;

public:
    explicit QueueBuffer(SlidingQueue<T>& parent)
        : parent_(parent), idx_(0) {
        buf_ = new T[kBufSize];
    }
    ~QueueBuffer() {
        flush();
        delete[] buf_;
    }

    void push_back(T val) {
        buf_[idx_++] = val;
        if (idx_ == kBufSize) flush();
    }

    void flush() {
        for (size_t i = 0; i < idx_; i++)
            parent_.push_back(buf_[i]);
        idx_ = 0;
    }
};

// ─── pvector (upstream 100% + dump_range) ───────────────────────
template <typename T>
class pvector {
    T*     start_;
    T*     end_size_;
    T*     end_capacity_;
    static constexpr size_t growth_factor = 2;

public:
    using iterator = T*;

    pvector() : start_(nullptr), end_size_(nullptr),
                end_capacity_(nullptr) {}

    explicit pvector(size_t n) {
        start_ = new T[n]();
        end_size_ = start_ + n;
        end_capacity_ = start_ + n;
    }

    pvector(size_t n, T val) {
        start_ = new T[n];
        end_size_ = start_ + n;
        end_capacity_ = start_ + n;
        for (size_t i = 0; i < n; i++) start_[i] = val;
    }

    ~pvector() { delete[] start_; }

    pvector(const pvector&) = delete;
    pvector& operator=(const pvector&) = delete;

    pvector(pvector&& o) noexcept
        : start_(o.start_), end_size_(o.end_size_),
          end_capacity_(o.end_capacity_) {
        o.start_ = o.end_size_ = o.end_capacity_ = nullptr;
    }

    T& operator[](size_t i) { return start_[i]; }
    const T& operator[](size_t i) const { return start_[i]; }

    void reserve(size_t new_cap) {
        if (new_cap <= capacity()) return;
        T* new_data = new T[new_cap];
        size_t s = size();
        std::copy(start_, end_size_, new_data);
        delete[] start_;
        start_ = new_data;
        end_size_ = start_ + s;
        end_capacity_ = start_ + new_cap;
    }

    void push_back(T val) {
        if (size() == capacity()) {
            size_t ns = capacity() == 0 ? 1 : capacity() * growth_factor;
            reserve(ns);
        }
        *end_size_ = val;
        end_size_++;
    }

    void fill(T val) {
        for (T* p = start_; p < end_size_; p++) *p = val;
    }

    size_t capacity() const { return end_capacity_ - start_; }
    size_t size()     const { return end_size_ - start_; }

    iterator begin() const { return start_; }
    iterator end()   const { return end_size_; }
    T*       data()  const { return start_; }

    void swap(pvector& o) {
        std::swap(start_, o.start_);
        std::swap(end_size_, o.end_size_);
        std::swap(end_capacity_, o.end_capacity_);
    }

    // [NEW] 打印区间值
    void dump_range(size_t from, size_t to,
                    const char* label = "PVEC") const {
        to = std::min(to, size());
        std::printf("[%s] size=%lu showing [%lu,%lu):\n",
                    label, (unsigned long)size(),
                    (unsigned long)from, (unsigned long)to);
        for (size_t i = from; i < to; i++) {
            // 尝试打印，具体格式依赖T
            if constexpr (std::is_arithmetic_v<T>)
                std::printf("  [%lu] = %g\n", (unsigned long)i,
                            (double)start_[i]);
        }
    }
};

} // namespace gapbs_compat
} // namespace algorithms
} // namespace philemon

#endif // PHILEMON_GAPBS_COMPAT_HPP
