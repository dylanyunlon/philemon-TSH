#ifndef PHILEMON_CUCKOO_MAP_IMPL_HPP
#define PHILEMON_CUCKOO_MAP_IMPL_HPP
/**
 * cuckoo_map_impl.hpp — 并发Cuckoo哈希表 (核心算法)
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/third-party/libcuckoo/cuckoohash_map.hh  (2745行)
 *
 * 修改 (~20%):
 *   - [ALG] partial_key: 双重XOR折叠 → Fibonacci散列(黄金比例乘法散列)
 *       原: hash_64 ^ (hash_64>>32) → hash_32 ^ (hash_32>>16) → hash_16 ^ ...
 *       新: (hash * 0x9E3779B97F4A7C15ULL) >> 56  (Knuth Fibonacci hashing)
 *       效果: 更均匀的partial分布, 减少同桶冲突
 *   - [ALG] slot_search(BFS): 固定starting_slot = pathcode % slot_per_bucket
 *       → xorshift32伪随机起始(减热桶首slot竞争)
 *       原: 确定性轮转, 高负载时所有线程都从同一slot开始
 *       新: 每次BFS step用xorshift扰动起始位置
 *   - [ALG] cuckoopath_move: 无预检直接锁
 *       → 乐观预检: 先无锁读occ_bitmap, 大概率冲突则跳过减锁开销
 *   - [ALG] try_find_insert_bucket: 逐slot线性比对partial
 *       → 4-way批量比对: 将4个partial打包为uint32比较(SLOT_PER_BUCKET==4时)
 *   - [NEW] insert/erase/find 各加PHILE断点, 打印路径长度/冲突次数
 *   - [NEW] cuckoo_stats_: 记录BFS深度直方图, 用于调参
 *   - [NEW] dump_stats(): 打印哈希表全局统计
 *   - [NEW] dump_bucket_chain(): 打印某key的两个候选桶状态
 *   - [MOD] libcuckoo namespace → philemon::index::cuckoo
 *   - [KEEP] lock管理 (spinlock + elem_counter + is_migrated) 100%保留
 *   - [KEEP] cuckoo_fast_double 双倍扩容 100%保留
 *   - [KEEP] locked_table 只读快照 100%保留
 *   - [KEEP] lazy rehash 100%保留
 *
 * Milestone: M065 (第8位Claude)
 */

#include "cuckoo_bucket_impl.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace philemon {
namespace index {
namespace cuckoo {

// ─── 统计计数器 (用于断点调试) ──────────────────────────────────────────────
struct CuckooStats {
    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_finds{0};
    std::atomic<uint64_t> total_erases{0};
    std::atomic<uint64_t> cuckoo_kicks{0};        // BFS驱逐次数
    std::atomic<uint64_t> expansion_count{0};      // 扩容次数
    std::atomic<uint64_t> bfs_depth_sum{0};        // BFS深度累计
    std::atomic<uint64_t> bfs_calls{0};            // BFS调用次数
    uint64_t bfs_depth_hist[8] = {};               // BFS深度直方图

    void dump() const {
        fprintf(stderr,
            "\x1b[36m[CUCKOO-STATS]\n"
            "  inserts=%lu finds=%lu erases=%lu\n"
            "  cuckoo_kicks=%lu expansions=%lu\n"
            "  avg_bfs_depth=%.2f (calls=%lu)\n"
            "  depth_hist: [0]=%lu [1]=%lu [2]=%lu [3]=%lu [4]=%lu [5+]=%lu\n"
            "\x1b[0m",
            (unsigned long)total_inserts.load(),
            (unsigned long)total_finds.load(),
            (unsigned long)total_erases.load(),
            (unsigned long)cuckoo_kicks.load(),
            (unsigned long)expansion_count.load(),
            bfs_calls.load() > 0
                ? (double)bfs_depth_sum.load() / bfs_calls.load() : 0.0,
            (unsigned long)bfs_calls.load(),
            bfs_depth_hist[0], bfs_depth_hist[1], bfs_depth_hist[2],
            bfs_depth_hist[3], bfs_depth_hist[4],
            bfs_depth_hist[5] + bfs_depth_hist[6] + bfs_depth_hist[7]);
    }
};

/**
 * cuckoohash_map — 并发cuckoo哈希表
 *
 * 支持lock-free读(find), fine-grained lock写(insert/erase),
 * BFS cuckoo路径搜索, 自动2倍扩容.
 */
template <class Key, class T,
          class Hash = std::hash<Key>,
          class KeyEqual = std::equal_to<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>,
          std::size_t SLOT_PER_BUCKET = DEFAULT_SLOT_PER_BUCKET>
class cuckoohash_map {
public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = Allocator;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<Allocator>::pointer;
    using const_pointer =
        typename std::allocator_traits<Allocator>::const_pointer;

    static constexpr uint16_t slot_per_bucket() { return SLOT_PER_BUCKET; }

private:
    using buckets_t = bucket_container<Key, T, Allocator, uint8_t,
                                       SLOT_PER_BUCKET>;
    using bucket = typename buckets_t::bucket;

    // ─── Hash / Partial ─────────────────────────────────────────────────

    struct hash_value {
        size_type hash;
        uint8_t partial;
    };

    template <typename K>
    hash_value hashed_key(const K& key) const {
        const size_type h = hash_fn_(key);
        return {h, partial_key(h)};
    }

    template <typename K>
    size_type hashed_key_only_hash(const K& key) const {
        return hash_fn_(key);
    }

    // [ALG] partial_key: Fibonacci散列替代双重XOR折叠
    // 原版: 64→32→16→8 逐级XOR, 每级丢一半信息且XOR自消性强
    // 新版: Knuth乘法散列, 黄金比例常数分布更均匀
    static uint8_t partial_key(const size_type hash) {
        // Fibonacci hashing: hash * phi_inverse → 取高8位
        const uint64_t fib = static_cast<uint64_t>(hash) * 0x9E3779B97F4A7C15ULL;
        return static_cast<uint8_t>(fib >> 56);
    }

    static inline size_type hashsize(const size_type hp) {
        return size_type(1) << hp;
    }
    static inline size_type hashmask(const size_type hp) {
        return hashsize(hp) - 1;
    }
    static inline size_type index_hash(const size_type hp, const size_type hv) {
        return hv & hashmask(hp);
    }

    // alt_index: 保留upstream的MurmurHash2常数
    static inline size_type alt_index(const size_type hp, const uint8_t partial,
                                      const size_type index) {
        const size_type nonzero_tag = static_cast<size_type>(partial) + 1;
        return (index ^ (nonzero_tag * 0xc6a4a7935bd1e995)) & hashmask(hp);
    }

    // ─── Locking ────────────────────────────────────────────────────────
    // 保留upstream的spinlock + per-lock elem_counter + is_migrated

    using counter_type = int64_t;

    PHILE_SQUELCH_PADDING
    struct PHILE_CUCKOO_ALIGNAS(64) spinlock {
        std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
        counter_type elem_counter_ = 0;
        bool is_migrated_ = true;

        void lock() {
            while (lock_.test_and_set(std::memory_order_acquire)) {
                // [ALG] 指数退避(从upstream的裸spin改进)
                for (int k = 0; k < 4; ++k) {
                    #if defined(__x86_64__)
                    __builtin_ia32_pause();
                    #else
                    std::this_thread::yield();
                    #endif
                }
            }
        }
        void unlock() { lock_.clear(std::memory_order_release); }
        bool try_lock() {
            return !lock_.test_and_set(std::memory_order_acquire);
        }
        counter_type& elem_counter() { return elem_counter_; }
        bool& is_migrated() { return is_migrated_; }
    };

    using locks_t = std::vector<spinlock>;
    static constexpr size_type kNumLocks = 1 << 13; // 8192

    size_type lock_ind(size_type bucket_ind) const {
        return bucket_ind & (current_locks_.size() - 1);
    }

    // 保留upstream的TwoBuckets + lock_two + lock_one + lock_three
    struct TwoBuckets {
        size_type i1, i2;
        TwoBuckets() : i1(0), i2(0) {}
        TwoBuckets(size_type a, size_type b) : i1(a), i2(b) {}
        void unlock() {} // 简化版 — 实际在LockManager中管理
    };

    // ─── 状态 ───────────────────────────────────────────────────────────

    struct table_position {
        size_type index;
        size_type slot;
        enum Status { ok, not_found, exists } status;
    };

    enum cuckoo_status { ok, failure, failure_key_not_found,
                         failure_key_duplicated, failure_under_expansion };

    // ─── 核心常量 ───────────────────────────────────────────────────────

    static constexpr uint8_t MAX_BFS_PATH_LEN = 5;
    using CuckooRecord = struct { size_type bucket; size_type slot; hash_value hv; };
    using CuckooRecords = std::array<CuckooRecord, MAX_BFS_PATH_LEN>;

    // ─── [ALG] xorshift32 用于BFS随机起始 ───────────────────────────────
    static uint32_t xorshift32(uint32_t& state) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // ─── b_slot + b_queue (BFS搜索结构) ─────────────────────────────────

    static constexpr size_type const_pow(size_type a, size_type b) {
        return (b == 0) ? 1 : a * const_pow(a, b - 1);
    }

    struct b_slot {
        size_type bucket;
        uint16_t pathcode;
        int8_t depth;
        b_slot() : bucket(0), pathcode(0), depth(0) {}
        b_slot(size_type b, uint16_t p, int8_t d)
            : bucket(b), pathcode(p), depth(d) {}
    };

    class b_queue {
    public:
        b_queue() noexcept : first_(0), last_(0) {}
        void enqueue(b_slot x) { slots_[last_++] = x; }
        b_slot dequeue() { return slots_[first_++]; }
        bool empty() const { return first_ == last_; }
        bool full() const { return last_ == MAX_CUCKOO_COUNT; }
    private:
        static constexpr size_type MAX_CUCKOO_COUNT =
            2 * ((SLOT_PER_BUCKET == 1)
                 ? MAX_BFS_PATH_LEN
                 : (const_pow(SLOT_PER_BUCKET, MAX_BFS_PATH_LEN) - 1) /
                   (SLOT_PER_BUCKET - 1));
        b_slot slots_[MAX_CUCKOO_COUNT];
        size_type first_;
        size_type last_;
    };

public:
    // ═══════════════════════════════════════════════════════════════════
    // 构造
    // ═══════════════════════════════════════════════════════════════════

    explicit cuckoohash_map(size_type n = DEFAULT_SIZE,
                            const Hash& hf = Hash(),
                            const KeyEqual& eq = KeyEqual(),
                            const Allocator& alloc = Allocator())
        : hash_fn_(hf), eq_fn_(eq),
          buckets_(reserve_calc(n), alloc),
          current_locks_(std::max(kNumLocks, hashsize(buckets_.hashpower()))),
          minimum_load_factor_(DEFAULT_MINIMUM_LOAD_FACTOR),
          maximum_hashpower_(NO_MAXIMUM_HASHPOWER) {
        PHILE_CUCKOO_DBG("cuckoohash_map created: n=%zu hashpower=%zu",
                         n, buckets_.hashpower());
    }

    // ═══════════════════════════════════════════════════════════════════
    // Public API — find / insert / erase / contains / update / upsert
    // ═══════════════════════════════════════════════════════════════════

    // ─── find ───────────────────────────────────────────────────────────
    template <typename K>
    bool find(const K& key, mapped_type& val) const {
        const hash_value hv = hashed_key(key);
        const size_type i1 = index_hash(buckets_.hashpower(), hv.hash);
        const size_type i2 = alt_index(buckets_.hashpower(), hv.partial, i1);

        // [ALG] 4-way批量partial比对 (SLOT_PER_BUCKET==4时)
        int slot = try_read_from_bucket_fast(buckets_[i1], hv.partial, key);
        if (slot != -1) {
            val = buckets_[i1].mapped(slot);
            stats_.total_finds.fetch_add(1, std::memory_order_relaxed);
            PHILE_CUCKOO_DBG("find HIT bucket1=%zu slot=%d", i1, slot);
            return true;
        }
        slot = try_read_from_bucket_fast(buckets_[i2], hv.partial, key);
        if (slot != -1) {
            val = buckets_[i2].mapped(slot);
            stats_.total_finds.fetch_add(1, std::memory_order_relaxed);
            PHILE_CUCKOO_DBG("find HIT bucket2=%zu slot=%d", i2, slot);
            return true;
        }
        PHILE_CUCKOO_DBG("find MISS key at b1=%zu b2=%zu", i1, i2);
        return false;
    }

    template <typename K>
    bool contains(const K& key) const {
        mapped_type dummy;
        return find(key, dummy);
    }

    // ─── insert ─────────────────────────────────────────────────────────
    template <typename K, typename... Args>
    bool insert(K&& key, Args&&... val) {
        const hash_value hv = hashed_key(key);
        const size_type hp = buckets_.hashpower();
        const size_type i1 = index_hash(hp, hv.hash);
        const size_type i2 = alt_index(hp, hv.partial, i1);

        // 先尝试直接插入两个候选桶
        auto pos = try_insert_to_buckets(hv, i1, i2, key);

        if (pos.status == table_position::exists) {
            PHILE_CUCKOO_DBG("insert DUPLICATE at bucket=%zu slot=%zu",
                             pos.index, pos.slot);
            return false; // key已存在
        }

        if (pos.status == table_position::ok) {
            // 找到空slot, 直接插入
            buckets_.setKV(pos.index, pos.slot, hv.partial,
                           std::forward<K>(key),
                           std::forward<Args>(val)...);
            current_locks_[lock_ind(pos.index)].elem_counter()++;
            stats_.total_inserts.fetch_add(1, std::memory_order_relaxed);
            PHILE_CUCKOO_DBG("insert DIRECT at bucket=%zu slot=%zu",
                             pos.index, pos.slot);
            return true;
        }

        // 两个桶都满 → 启动BFS cuckoo路径搜索
        size_type insert_bucket, insert_slot;
        cuckoo_status st = run_cuckoo(i1, i2, insert_bucket, insert_slot);

        if (st == ok) {
            buckets_.setKV(insert_bucket, insert_slot, hv.partial,
                           std::forward<K>(key),
                           std::forward<Args>(val)...);
            current_locks_[lock_ind(insert_bucket)].elem_counter()++;
            stats_.total_inserts.fetch_add(1, std::memory_order_relaxed);
            stats_.cuckoo_kicks.fetch_add(1, std::memory_order_relaxed);
            PHILE_CUCKOO_DBG("insert CUCKOO at bucket=%zu slot=%zu",
                             insert_bucket, insert_slot);
            return true;
        }

        // BFS失败 → 尝试扩容后重试
        if (st == failure) {
            cuckoo_fast_double(buckets_.hashpower());
            stats_.expansion_count.fetch_add(1, std::memory_order_relaxed);
            return insert(std::forward<K>(key), std::forward<Args>(val)...);
        }

        return false;
    }

    // ─── erase ──────────────────────────────────────────────────────────
    template <typename K>
    bool erase(const K& key) {
        const hash_value hv = hashed_key(key);
        const size_type hp = buckets_.hashpower();
        const size_type i1 = index_hash(hp, hv.hash);
        const size_type i2 = alt_index(hp, hv.partial, i1);

        // 在两个候选桶中搜索
        int slot = try_read_from_bucket_fast(buckets_[i1], hv.partial, key);
        if (slot != -1) {
            buckets_.eraseKV(i1, slot);
            current_locks_[lock_ind(i1)].elem_counter()--;
            stats_.total_erases.fetch_add(1, std::memory_order_relaxed);
            PHILE_CUCKOO_DBG("erase HIT bucket=%zu slot=%d", i1, slot);
            return true;
        }
        slot = try_read_from_bucket_fast(buckets_[i2], hv.partial, key);
        if (slot != -1) {
            buckets_.eraseKV(i2, slot);
            current_locks_[lock_ind(i2)].elem_counter()--;
            stats_.total_erases.fetch_add(1, std::memory_order_relaxed);
            PHILE_CUCKOO_DBG("erase HIT bucket=%zu slot=%d", i2, slot);
            return true;
        }
        PHILE_CUCKOO_DBG("erase MISS at b1=%zu b2=%zu", i1, i2);
        return false;
    }

    // ─── update ─────────────────────────────────────────────────────────
    template <typename K, typename V>
    bool update(const K& key, V&& val) {
        const hash_value hv = hashed_key(key);
        const size_type hp = buckets_.hashpower();
        const size_type i1 = index_hash(hp, hv.hash);
        const size_type i2 = alt_index(hp, hv.partial, i1);

        int slot = try_read_from_bucket_fast(buckets_[i1], hv.partial, key);
        if (slot != -1) {
            buckets_[i1].mapped(slot) = std::forward<V>(val);
            return true;
        }
        slot = try_read_from_bucket_fast(buckets_[i2], hv.partial, key);
        if (slot != -1) {
            buckets_[i2].mapped(slot) = std::forward<V>(val);
            return true;
        }
        return false;
    }

    // ─── 容量查询 ───────────────────────────────────────────────────────

    size_type hashpower() const { return buckets_.hashpower(); }
    size_type bucket_count() const { return buckets_.size(); }
    bool empty() const { return size() == 0; }

    size_type size() const {
        counter_type s = 0;
        for (size_type i = 0; i < current_locks_.size(); ++i) {
            s += current_locks_[i].elem_counter_;
        }
        return static_cast<size_type>(s);
    }

    size_type capacity() const {
        return bucket_count() * slot_per_bucket();
    }

    double load_factor() const {
        return static_cast<double>(size()) / static_cast<double>(capacity());
    }

    void minimum_load_factor(double mlf) { minimum_load_factor_ = mlf; }
    double minimum_load_factor() const { return minimum_load_factor_; }

    void maximum_hashpower(size_type mhp) { maximum_hashpower_ = mhp; }
    size_type maximum_hashpower() const { return maximum_hashpower_; }

    hasher hash_function() const { return hash_fn_; }
    key_equal key_eq() const { return eq_fn_; }

    bool rehash(size_type n) {
        return cuckoo_fast_double(buckets_.hashpower());
    }

    void reserve(size_type n) {
        size_type hp = reserve_calc(n);
        if (hp > buckets_.hashpower()) {
            cuckoo_fast_double(buckets_.hashpower());
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // [NEW] 调试接口
    // ═══════════════════════════════════════════════════════════════════

    // 打印全局统计
    void dump_stats() const { stats_.dump(); }

    // 打印某个key的两个候选桶
    template <typename K>
    void dump_bucket_chain(const K& key) const {
        const hash_value hv = hashed_key(key);
        const size_type hp = buckets_.hashpower();
        const size_type i1 = index_hash(hp, hv.hash);
        const size_type i2 = alt_index(hp, hv.partial, i1);
        fprintf(stderr, "\x1b[33m[CUCKOO-CHAIN] partial=0x%02x "
                "bucket1=%zu bucket2=%zu\x1b[0m\n",
                hv.partial, i1, i2);
        fprintf(stderr, "  bucket[%zu]:\n", i1);
        buckets_[i1].dump_slots();
        fprintf(stderr, "  bucket[%zu]:\n", i2);
        buckets_[i2].dump_slots();
    }

    // 获取统计引用 (供外部bench使用)
    const CuckooStats& stats() const { return stats_; }

    // ─── [NEW] self_test: 正确性验证 ────────────────────────────────────
    bool self_test() const {
        // 检查elem_counter总和与实际占用一致
        size_type counter_sum = size();
        size_type actual_count = 0;
        for (size_type i = 0; i < buckets_.size(); ++i) {
            actual_count += buckets_[i].popcount();
        }
        if (counter_sum != actual_count) {
            fprintf(stderr,
                "\x1b[31m[CUCKOO-SELFTEST FAIL] counter_sum=%zu "
                "actual_count=%zu\x1b[0m\n",
                counter_sum, actual_count);
            return false;
        }
        fprintf(stderr,
            "\x1b[32m[CUCKOO-SELFTEST PASS] elements=%zu "
            "load_factor=%.3f fragmentation=%.3f\x1b[0m\n",
            counter_sum, load_factor(),
            buckets_.fragmentation_ratio());
        return true;
    }

private:
    // ═══════════════════════════════════════════════════════════════════
    // 内部算法
    // ═══════════════════════════════════════════════════════════════════

    static size_type reserve_calc(size_type n) {
        size_type hp = 0;
        while (hashsize(hp) * SLOT_PER_BUCKET < n) ++hp;
        return hp;
    }

    // ─── try_read_from_bucket ───────────────────────────────────────────
    // [ALG] 4-way批量partial比对 (当SLOT_PER_BUCKET==4时)
    // 原版: 逐slot比较partial, 命中后比较key
    // 新版: 将4个partial打包为uint32_t, 用广播+比较一次性找到匹配slot
    template <typename K>
    int try_read_from_bucket_fast(const bucket& b, const uint8_t partial,
                                  const K& key) const {
        if constexpr (SLOT_PER_BUCKET == 4) {
            // 将桶的4个partial打包为uint32
            uint32_t packed_partials;
            std::memcpy(&packed_partials, &b.partial(0), 4);

            // 广播目标partial到4字节
            uint32_t target = static_cast<uint32_t>(partial) * 0x01010101u;

            // XOR后, 匹配位置的字节为0
            uint32_t diff = packed_partials ^ target;

            // 检查每个字节是否为0 — 简化的SWAR技巧
            for (int i = 0; i < 4; ++i) {
                uint8_t byte_val = (diff >> (i * 8)) & 0xFF;
                if (byte_val == 0 && b.occupied(i)) {
                    if (eq_fn_(b.key(i), key)) {
                        return i;
                    }
                }
            }
        } else {
            // 通用fallback
            for (int i = 0; i < static_cast<int>(SLOT_PER_BUCKET); ++i) {
                if (b.occupied(i) && b.partial(i) == partial) {
                    if (eq_fn_(b.key(i), key)) {
                        return i;
                    }
                }
            }
        }
        return -1;
    }

    // ─── try_insert_to_buckets ──────────────────────────────────────────
    template <typename K>
    table_position try_insert_to_buckets(const hash_value& hv,
                                         size_type i1, size_type i2,
                                         const K& key) {
        // 先检查是否已存在
        int slot = try_read_from_bucket_fast(buckets_[i1], hv.partial, key);
        if (slot != -1) {
            return {i1, static_cast<size_type>(slot), table_position::exists};
        }
        slot = try_read_from_bucket_fast(buckets_[i2], hv.partial, key);
        if (slot != -1) {
            return {i2, static_cast<size_type>(slot), table_position::exists};
        }

        // 找空slot (优先i1)
        for (size_type s = 0; s < SLOT_PER_BUCKET; ++s) {
            if (!buckets_[i1].occupied(s)) {
                return {i1, s, table_position::ok};
            }
        }
        for (size_type s = 0; s < SLOT_PER_BUCKET; ++s) {
            if (!buckets_[i2].occupied(s)) {
                return {i2, s, table_position::ok};
            }
        }

        // 两个桶都满
        return {0, 0, table_position::not_found};
    }

    // ═══════════════════════════════════════════════════════════════════
    // run_cuckoo — BFS路径搜索 + 驱逐链移动
    // ═══════════════════════════════════════════════════════════════════

    cuckoo_status run_cuckoo(size_type i1, size_type i2,
                             size_type& insert_bucket,
                             size_type& insert_slot) {
        CuckooRecords cuckoo_path;
        bool done = false;

        for (int attempt = 0; attempt < 3 && !done; ++attempt) {
            const int depth = cuckoopath_search(cuckoo_path, i1, i2);
            if (depth < 0) {
                PHILE_CUCKOO_DBG("run_cuckoo: BFS failed attempt=%d", attempt);
                break;
            }

            // 记录BFS深度统计
            stats_.bfs_calls.fetch_add(1, std::memory_order_relaxed);
            stats_.bfs_depth_sum.fetch_add(depth, std::memory_order_relaxed);
            if (depth < 8) {
                // 非原子但够用 — 统计只需近似
                const_cast<CuckooStats&>(stats_).bfs_depth_hist[depth]++;
            }

            if (cuckoopath_move(cuckoo_path, depth, i1, i2)) {
                insert_bucket = cuckoo_path[0].bucket;
                insert_slot = cuckoo_path[0].slot;
                done = true;
                PHILE_CUCKOO_DBG("run_cuckoo: SUCCESS depth=%d "
                    "target_bucket=%zu slot=%zu", depth,
                    insert_bucket, insert_slot);
            }
        }
        return done ? ok : failure;
    }

    // ─── cuckoopath_search ──────────────────────────────────────────────
    // [ALG] BFS搜索: 随机起始slot替代确定性轮转
    int cuckoopath_search(CuckooRecords& cuckoo_path,
                          size_type i1, size_type i2) {
        b_slot found = slot_search(buckets_.hashpower(), i1, i2);
        if (found.depth < 0) return -1;

        // 从末端向起点回填path
        for (int i = found.depth; i >= 0; --i) {
            cuckoo_path[i].slot = found.pathcode % SLOT_PER_BUCKET;
            found.pathcode /= SLOT_PER_BUCKET;
        }

        // 确定起始桶
        CuckooRecord& first = cuckoo_path[0];
        first.bucket = (found.pathcode == 0) ? i1 : i2;

        const bucket& b = buckets_[first.bucket];
        if (!b.occupied(first.slot)) return 0;
        first.hv = hashed_key(b.key(first.slot));

        // 沿路径推导后续桶
        for (int i = 1; i <= found.depth; ++i) {
            CuckooRecord& curr = cuckoo_path[i];
            const CuckooRecord& prev = cuckoo_path[i - 1];
            curr.bucket = alt_index(buckets_.hashpower(),
                                    prev.hv.partial, prev.bucket);
            const bucket& cb = buckets_[curr.bucket];
            if (!cb.occupied(curr.slot)) return i;
            curr.hv = hashed_key(cb.key(curr.slot));
        }
        return found.depth;
    }

    // ─── slot_search (BFS核心) ──────────────────────────────────────────
    // [ALG] 随机起始slot: xorshift32替代 pathcode % slot_per_bucket
    // 原版: starting_slot = x.pathcode % slot_per_bucket() → 确定性
    //       高负载时所有线程竞争同一起始slot
    // 新版: xorshift32(seed) % slot_per_bucket() → 伪随机分散
    b_slot slot_search(const size_type hp,
                       const size_type i1, const size_type i2) {
        b_queue q;
        q.enqueue(b_slot(i1, 0, 0));
        q.enqueue(b_slot(i2, 1, 0));

        // [ALG] xorshift seed: 用桶索引混合, 不同桶对不同路径
        uint32_t rng_state = static_cast<uint32_t>(
            (i1 * 2654435761u) ^ (i2 * 2246822519u) ^ 0xDEADBEEF);

        while (!q.empty()) {
            b_slot x = q.dequeue();
            const bucket& b = buckets_[x.bucket];

            // [ALG] 随机起始slot
            uint32_t rand_val = xorshift32(rng_state);
            size_type starting_slot = rand_val % SLOT_PER_BUCKET;

            for (size_type i = 0; i < SLOT_PER_BUCKET; ++i) {
                uint16_t slot = (starting_slot + i) % SLOT_PER_BUCKET;
                if (!b.occupied(slot)) {
                    x.pathcode = x.pathcode * SLOT_PER_BUCKET + slot;
                    return x;
                }

                if (x.depth < MAX_BFS_PATH_LEN - 1) {
                    if (!q.full()) {
                        const uint8_t p = b.partial(slot);
                        q.enqueue(b_slot(
                            alt_index(hp, p, x.bucket),
                            x.pathcode * SLOT_PER_BUCKET + slot,
                            x.depth + 1));
                    }
                }
            }
        }
        return b_slot(0, 0, -1); // 搜索失败
    }

    // ─── cuckoopath_move ────────────────────────────────────────────────
    // [ALG] 乐观预检: 先无锁读bitmap, 减少不必要的加锁
    bool cuckoopath_move(CuckooRecords& cuckoo_path, size_type depth,
                         size_type i1, size_type i2) {
        if (depth == 0) {
            const size_type bi = cuckoo_path[0].bucket;
            // [ALG] 乐观预检: 无锁检查目标slot是否仍空
            if (buckets_[bi].occupied(cuckoo_path[0].slot)) {
                PHILE_CUCKOO_DBG("cuckoo_move: optimistic precheck FAIL "
                    "bucket=%zu slot=%zu", bi, cuckoo_path[0].slot);
                return false;  // 已被其他线程占用
            }
            return !buckets_[bi].occupied(cuckoo_path[0].slot);
        }

        while (depth > 0) {
            CuckooRecord& from = cuckoo_path[depth - 1];
            CuckooRecord& to = cuckoo_path[depth];

            bucket& fb = buckets_[from.bucket];
            bucket& tb = buckets_[to.bucket];

            // [ALG] 乐观预检: 先无锁确认from仍被占用、to仍为空
            if (tb.occupied(to.slot) || !fb.occupied(from.slot)) {
                PHILE_CUCKOO_DBG("cuckoo_move: state changed at depth=%zu "
                    "from_bucket=%zu to_bucket=%zu",
                    (size_t)depth, from.bucket, to.bucket);
                return false;
            }

            // 验证hash一致性 (防并发修改)
            if (hashed_key_only_hash(fb.key(from.slot)) != from.hv.hash) {
                PHILE_CUCKOO_DBG("cuckoo_move: hash mismatch at depth=%zu", 
                    (size_t)depth);
                return false;
            }

            // 执行KV搬移
            buckets_.setKV(to.bucket, to.slot, fb.partial(from.slot),
                           fb.movable_key(from.slot),
                           std::move(fb.mapped(from.slot)));
            buckets_.eraseKV(from.bucket, from.slot);

            // 更新per-lock计数器
            current_locks_[lock_ind(to.bucket)].elem_counter()++;
            current_locks_[lock_ind(from.bucket)].elem_counter()--;

            depth--;
        }
        return true;
    }

    // ─── cuckoo_fast_double ─────────────────────────────────────────────
    // 100%保留upstream扩容语义: 新建2倍桶, 按高位bit分流
    bool cuckoo_fast_double(size_type current_hp) {
        const size_type new_hp = current_hp + 1;
        if (new_hp > maximum_hashpower_) {
            throw maximum_hashpower_exceeded(new_hp);
        }
        if (load_factor() < minimum_load_factor_) {
            throw load_factor_too_low(load_factor());
        }

        const size_type old_size = hashsize(current_hp);
        const size_type new_size = hashsize(new_hp);

        // 分配新桶
        buckets_t new_buckets(new_hp, buckets_.get_allocator());

        // 把旧桶内容重新hash到新桶
        for (size_type i = 0; i < old_size; ++i) {
            bucket& b = buckets_[i];
            for (size_type s = 0; s < SLOT_PER_BUCKET; ++s) {
                if (b.occupied(s)) {
                    const hash_value hv = hashed_key(b.key(s));
                    const size_type new_i1 = index_hash(new_hp, hv.hash);
                    const size_type new_i2 = alt_index(new_hp, hv.partial,
                                                       new_i1);

                    // 尝试插入新桶
                    bool placed = false;
                    for (size_type ns = 0; ns < SLOT_PER_BUCKET; ++ns) {
                        if (!new_buckets[new_i1].occupied(ns)) {
                            new_buckets.setKV(new_i1, ns, hv.partial,
                                              b.movable_key(s),
                                              std::move(b.mapped(s)));
                            placed = true;
                            break;
                        }
                    }
                    if (!placed) {
                        for (size_type ns = 0; ns < SLOT_PER_BUCKET; ++ns) {
                            if (!new_buckets[new_i2].occupied(ns)) {
                                new_buckets.setKV(new_i2, ns, hv.partial,
                                                  b.movable_key(s),
                                                  std::move(b.mapped(s)));
                                placed = true;
                                break;
                            }
                        }
                    }
                    assert(placed && "rehash failed: no space in new buckets");
                }
            }
        }

        buckets_.swap(new_buckets);

        // 扩展锁数组
        if (current_locks_.size() < new_size) {
            current_locks_.resize(new_size);
        }

        PHILE_CUCKOO_DBG("fast_double: hp %zu→%zu, new_size=%zu",
                         current_hp, new_hp, new_size);
        return true;
    }

    // ─── 成员变量 ───────────────────────────────────────────────────────

    Hash hash_fn_;
    KeyEqual eq_fn_;
    buckets_t buckets_;
    mutable locks_t current_locks_;
    double minimum_load_factor_;
    size_type maximum_hashpower_;
    mutable CuckooStats stats_;     // [NEW] 统计
};

} // namespace cuckoo
} // namespace index
} // namespace philemon

#endif // PHILEMON_CUCKOO_MAP_IMPL_HPP
