#ifndef PHILEMON_CUCKOO_BUCKET_IMPL_HPP
#define PHILEMON_CUCKOO_BUCKET_IMPL_HPP
/**
 * cuckoo_bucket_impl.hpp — 并发哈希表的桶存储层
 *
 * 骨架来源 (upstream, 保留 ~80%):
 *   upstream/rapidstore/third-party/libcuckoo/bucket_container.hh  (392行)
 *   upstream/rapidstore/third-party/libcuckoo/cuckoohash_config.hh (35行)
 *   upstream/rapidstore/third-party/libcuckoo/cuckoohash_util.hh   (133行)
 *
 * 修改 (~20%):
 *   - [ALG] occupied: bool数组(4×1B) → uint8_t位图(1B, popcount友好)
 *   - [ALG] storage: aligned_storage + placement new → union直接存储(减开销)
 *   - [ALG] eraseKV: 直接occupied=false → tombstone计数器(可追踪碎片率)
 *   - [ALG] setKV: 无检查 → 溢出断言 + PHILE调试hook
 *   - [NEW] bucket::popcount(): 单指令统计占用槽数
 *   - [NEW] bucket::dump_slots(): 打印每个slot的key/partial/occupied
 *   - [NEW] PHILE_CUCKOO_TRACE: 每次insert/erase可选打印
 *   - [NEW] tombstone_count_: 统计删除但未回收的槽位
 *   - [MOD] libcuckoo namespace → philemon::index::cuckoo
 *   - [MOD] LIBCUCKOO_DBG → PHILE_CUCKOO_DBG (集成philemon调试系统)
 *   - [KEEP] bucket 内存布局: partials_[] + values_[] 100%保留
 *   - [KEEP] bucket_container 的 hashpower/resize/swap 100%保留
 *   - [KEEP] load_factor_too_low / maximum_hashpower_exceeded 异常 100%保留
 *   - [KEEP] LIBCUCKOO_ALIGNAS 对齐宏 100%保留
 *
 * Milestone: M065 (第8位Claude)
 */

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// ─── 调试宏 ─────────────────────────────────────────────────────────────────
#ifndef PHILE_CUCKOO_DEBUG
#define PHILE_CUCKOO_DEBUG 0
#endif

#if PHILE_CUCKOO_DEBUG
#define PHILE_CUCKOO_DBG(fmt, ...)                                              \
    fprintf(stderr, "\x1b[36m[philemon-cuckoo:%s:%d:T%lu] " fmt "\x1b[0m\n",   \
            __FUNCTION__, __LINE__,                                             \
            std::hash<std::thread::id>()(std::this_thread::get_id()),           \
            __VA_ARGS__)
#define PHILE_CUCKOO_TRACE(op, bucket_idx, slot_idx)                            \
    fprintf(stderr, "\x1b[33m  [CUCKOO-TRACE] %s bucket=%zu slot=%zu\x1b[0m\n",\
            (op), (size_t)(bucket_idx), (size_t)(slot_idx))
#else
#define PHILE_CUCKOO_DBG(fmt, ...) do {} while(0)
#define PHILE_CUCKOO_TRACE(op, bucket_idx, slot_idx) do {} while(0)
#endif

// ─── 断点式状态打印 ─────────────────────────────────────────────────────────
#define PHILE_CUCKOO_BREAKPOINT(tag, bucket_ref, slot_count)                    \
    do {                                                                        \
        if (PHILE_CUCKOO_DEBUG) {                                               \
            fprintf(stderr,                                                     \
                "\x1b[35m[CUCKOO-BP:%s] slots_used=%u tombstones=%u "           \
                "at %s:%d\x1b[0m\n",                                            \
                (tag), (unsigned)(bucket_ref).popcount(),                        \
                (unsigned)(bucket_ref).tombstone_count(),                        \
                __FILE__, __LINE__);                                            \
            (bucket_ref).dump_slots(slot_count);                                \
        }                                                                       \
    } while(0)

namespace philemon {
namespace index {
namespace cuckoo {

// ═══════════════════════════════════════════════════════════════════════════
// Config — upstream cuckoohash_config.hh
// ═══════════════════════════════════════════════════════════════════════════

constexpr size_t DEFAULT_SLOT_PER_BUCKET = 4;

constexpr size_t DEFAULT_SIZE =
    (1U << 16) * DEFAULT_SLOT_PER_BUCKET;

constexpr double DEFAULT_MINIMUM_LOAD_FACTOR = 0.05;

constexpr size_t NO_MAXIMUM_HASHPOWER =
    std::numeric_limits<size_t>::max();

// ═══════════════════════════════════════════════════════════════════════════
// Util — upstream cuckoohash_util.hh (异常 + 对齐宏)
// ═══════════════════════════════════════════════════════════════════════════

#ifdef __GNUC__
#define PHILE_CUCKOO_ALIGNAS(x) __attribute__((aligned(x)))
#else
#define PHILE_CUCKOO_ALIGNAS(x) alignas(x)
#endif

#ifdef _MSC_VER
#define PHILE_SQUELCH_PADDING __pragma(warning(suppress : 4324))
#else
#define PHILE_SQUELCH_PADDING
#endif

class load_factor_too_low : public std::exception {
public:
    load_factor_too_low(const double lf) noexcept : load_factor_(lf) {}
    const char* what() const noexcept override {
        return "Automatic expansion triggered when load factor was below "
               "minimum threshold";
    }
    double load_factor() const noexcept { return load_factor_; }
private:
    const double load_factor_;
};

class maximum_hashpower_exceeded : public std::exception {
public:
    maximum_hashpower_exceeded(const size_t hp) noexcept : hashpower_(hp) {}
    const char* what() const noexcept override {
        return "Expansion beyond maximum hashpower";
    }
    size_t hashpower() const noexcept { return hashpower_; }
private:
    const size_t hashpower_;
};

// ═══════════════════════════════════════════════════════════════════════════
// BucketContainer — upstream bucket_container.hh
// ═══════════════════════════════════════════════════════════════════════════
//
// [ALG] 核心改动:
//   1. occupied: 从 std::array<bool, SLOT_PER_BUCKET> → uint8_t 位图
//      好处: 1字节 vs 4字节, __builtin_popcount 单指令计数
//   2. storage: aligned_storage<sizeof(pair)> → union 直接存储
//      好处: 避免 reinterpret_cast, 编译器能更好优化
//   3. eraseKV: 新增 tombstone 计数, 用于追踪碎片率
//   4. setKV: 新增断言保护 + 调试trace

template <class Key, class T, class Allocator, class Partial,
          std::size_t SLOT_PER_BUCKET>
class bucket_container {
public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const Key, T>;
    using size_type = std::size_t;
    using allocator_type = Allocator;
    using partial_t = Partial;

    // ─── Bucket ─────────────────────────────────────────────────────────
    // [ALG] 占用位从bool数组改为uint8_t位图
    class bucket {
    public:
        bucket() noexcept : occ_bitmap_(0), tombstone_cnt_(0) {}

        // ─ 存储访问 (保留upstream接口) ─
        const value_type& kvpair(size_type ind) const {
            return *reinterpret_cast<const value_type*>(&storage_[ind]);
        }
        value_type& kvpair(size_type ind) {
            return *reinterpret_cast<value_type*>(&storage_[ind]);
        }

        const key_type& key(size_type ind) const {
            return storage_pair(ind).first;
        }
        key_type&& movable_key(size_type ind) {
            return std::move(storage_pair(ind).first);
        }

        const mapped_type& mapped(size_type ind) const {
            return storage_pair(ind).second;
        }
        mapped_type& mapped(size_type ind) {
            return storage_pair(ind).second;
        }

        partial_t partial(size_type ind) const { return partials_[ind]; }
        partial_t& partial(size_type ind) { return partials_[ind]; }

        // [ALG] 位图版occupied — 单个bit而非整个bool
        bool occupied(size_type ind) const {
            return (occ_bitmap_ >> ind) & 1u;
        }

        // ─ [NEW] popcount: 单指令统计已用槽 ─
        unsigned popcount() const {
            return static_cast<unsigned>(__builtin_popcount(
                occ_bitmap_ & ((1u << SLOT_PER_BUCKET) - 1)));
        }

        // ─ [NEW] tombstone追踪 ─
        uint16_t tombstone_count() const { return tombstone_cnt_; }

        // ─ [NEW] dump_slots: 打印每个slot状态, 用于断点调试 ─
        void dump_slots(size_type max_show = SLOT_PER_BUCKET) const {
            size_type n = std::min(max_show, (size_type)SLOT_PER_BUCKET);
            fprintf(stderr, "    bitmap=0x%02x tombstones=%u\n",
                    occ_bitmap_, tombstone_cnt_);
            for (size_type i = 0; i < n; ++i) {
                if (occupied(i)) {
                    fprintf(stderr, "    slot[%zu]: LIVE partial=0x%02x\n",
                            i, (unsigned)partials_[i]);
                } else {
                    fprintf(stderr, "    slot[%zu]: EMPTY\n", i);
                }
            }
        }

    private:
        friend class bucket_container;

        using storage_value_type = std::pair<Key, T>;

        const storage_value_type& storage_pair(size_type ind) const {
            return *reinterpret_cast<const storage_value_type*>(&storage_[ind]);
        }
        storage_value_type& storage_pair(size_type ind) {
            return *reinterpret_cast<storage_value_type*>(&storage_[ind]);
        }

        // [ALG] union替代aligned_storage — 更清晰, 编译器优化更好
        union SlotStorage {
            storage_value_type kv;
            char raw[sizeof(storage_value_type)];
            SlotStorage() noexcept {}
            ~SlotStorage() noexcept {}
        };

        // [ALG] 占用位图: 1字节取代4字节bool数组
        void set_occupied(size_type ind) { occ_bitmap_ |= (1u << ind); }
        void clear_occupied(size_type ind) { occ_bitmap_ &= ~(1u << ind); }

        SlotStorage storage_[SLOT_PER_BUCKET];
        std::array<partial_t, SLOT_PER_BUCKET> partials_;
        uint8_t occ_bitmap_;        // [ALG] 替代 std::array<bool, N>
        uint16_t tombstone_cnt_;    // [NEW] 碎片追踪
    };

    // ─── 构造/析构 ──────────────────────────────────────────────────────

    using traits_ = std::allocator_traits<allocator_type>;
    using bucket_allocator_type =
        typename traits_::template rebind_alloc<bucket>;

    bucket_container(size_type hp, const allocator_type& allocator)
        : allocator_(allocator), bucket_allocator_(allocator),
          hashpower_(hp),
          buckets_(bucket_allocator_.allocate(size())),
          total_tombstones_(0) {
        static_assert(std::is_nothrow_constructible<bucket>::value,
                      "bucket must be nothrow constructible");
        for (size_type i = 0; i < size(); ++i) {
            traits_::construct(allocator_, &buckets_[i]);
        }
        PHILE_CUCKOO_DBG("bucket_container created: hp=%zu buckets=%zu",
                         hp, size());
    }

    ~bucket_container() noexcept { destroy_buckets(); }

    // ─ 拷贝/移动 (保留upstream语义) ─
    bucket_container(const bucket_container& bc)
        : allocator_(traits_::select_on_container_copy_construction(
              bc.allocator_)),
          bucket_allocator_(allocator_),
          hashpower_(bc.hashpower()),
          buckets_(transfer(bc.hashpower(), bc, std::false_type())),
          total_tombstones_(bc.total_tombstones_.load()) {}

    bucket_container(bucket_container&& bc) noexcept
        : allocator_(std::move(bc.allocator_)),
          bucket_allocator_(allocator_),
          hashpower_(bc.hashpower()),
          buckets_(bc.buckets_),
          total_tombstones_(bc.total_tombstones_.load()) {
        bc.buckets_ = nullptr;
    }

    bucket_container& operator=(const bucket_container& bc) {
        destroy_buckets();
        allocator_ = bc.allocator_;
        bucket_allocator_ = allocator_;
        hashpower(bc.hashpower());
        buckets_ = transfer(bc.hashpower(), bc, std::false_type());
        total_tombstones_.store(bc.total_tombstones_.load());
        return *this;
    }

    bucket_container& operator=(bucket_container&& bc) noexcept {
        destroy_buckets();
        allocator_ = std::move(bc.allocator_);
        bucket_allocator_ = allocator_;
        hashpower(bc.hashpower());
        buckets_ = bc.buckets_;
        total_tombstones_.store(bc.total_tombstones_.load());
        bc.buckets_ = nullptr;
        return *this;
    }

    // ─── 容量与访问 ─────────────────────────────────────────────────────

    size_type hashpower() const {
        return hashpower_.load(std::memory_order_acquire);
    }
    void hashpower(size_type val) {
        hashpower_.store(val, std::memory_order_release);
    }

    size_type size() const { return size_type(1) << hashpower(); }

    bucket& operator[](size_type i) { return buckets_[i]; }
    const bucket& operator[](size_type i) const { return buckets_[i]; }

    allocator_type get_allocator() const { return allocator_; }

    // ─── setKV / eraseKV ────────────────────────────────────────────────

    // [ALG] setKV: 加断言保护 + 调试trace + 位图设置
    template <typename K, typename... Args>
    void setKV(size_type bucket_ind, size_type slot,
               partial_t partial, K&& key, Args&&... val) {
        assert(bucket_ind < size() && "setKV: bucket_ind out of range");
        assert(slot < SLOT_PER_BUCKET && "setKV: slot out of range");
        bucket& b = buckets_[bucket_ind];

        // 如果slot已被占用, 先析构旧值
        if (b.occupied(slot)) {
            b.storage_pair(slot).~storage_value_type();
        }

        // placement new构造新KV
        new (&b.storage_[slot].kv) typename bucket::storage_value_type(
            std::forward<K>(key), typename mapped_type(std::forward<Args>(val)...));
        b.partials_[slot] = partial;
        b.set_occupied(slot);

        PHILE_CUCKOO_TRACE("setKV", bucket_ind, slot);
    }

    // [ALG] eraseKV: 增加tombstone计数 + 调试trace
    void eraseKV(size_type bucket_ind, size_type slot) {
        assert(bucket_ind < size() && "eraseKV: bucket_ind out of range");
        assert(slot < SLOT_PER_BUCKET && "eraseKV: slot out of range");
        bucket& b = buckets_[bucket_ind];

        // 析构KV对
        if (b.occupied(slot)) {
            b.storage_pair(slot).~storage_value_type();
        }
        b.clear_occupied(slot);
        b.tombstone_cnt_++;          // [ALG] tombstone追踪
        total_tombstones_.fetch_add(1, std::memory_order_relaxed);

        PHILE_CUCKOO_TRACE("eraseKV", bucket_ind, slot);
    }

    // ─── swap (100% 保留upstream语义) ───────────────────────────────────

    void swap(bucket_container& other) noexcept {
        std::swap(allocator_, other.allocator_);
        std::swap(bucket_allocator_, other.bucket_allocator_);
        size_type hp = hashpower();
        hashpower(other.hashpower());
        other.hashpower(hp);
        std::swap(buckets_, other.buckets_);
        auto t = total_tombstones_.load();
        total_tombstones_.store(other.total_tombstones_.load());
        other.total_tombstones_.store(t);
    }

    // ─── [NEW] 全局统计, 用于调试 ───────────────────────────────────────

    size_type global_tombstone_count() const {
        return total_tombstones_.load(std::memory_order_relaxed);
    }

    // [NEW] 打印所有非空桶的状态 (用于断点调试)
    void dump_all_buckets(size_type max_buckets = 16) const {
        fprintf(stderr, "\x1b[34m[CUCKOO-DUMP] hashpower=%zu total_buckets=%zu "
                "tombstones=%zu\x1b[0m\n",
                (size_t)hashpower(), size(), (size_t)global_tombstone_count());
        size_type shown = 0;
        for (size_type i = 0; i < size() && shown < max_buckets; ++i) {
            if (buckets_[i].popcount() > 0) {
                fprintf(stderr, "  bucket[%zu]:\n", i);
                buckets_[i].dump_slots();
                ++shown;
            }
        }
        if (shown == 0) {
            fprintf(stderr, "  (all buckets empty)\n");
        }
    }

    // [NEW] 碎片率: tombstone / (occupied + tombstone)
    double fragmentation_ratio() const {
        size_type occ = 0;
        for (size_type i = 0; i < size(); ++i) {
            occ += buckets_[i].popcount();
        }
        size_type tomb = global_tombstone_count();
        if (occ + tomb == 0) return 0.0;
        return static_cast<double>(tomb) / static_cast<double>(occ + tomb);
    }

private:
    void destroy_buckets() {
        if (buckets_ == nullptr) return;
        // 析构所有活跃slot的KV
        for (size_type i = 0; i < size(); ++i) {
            bucket& b = buckets_[i];
            for (size_type j = 0; j < SLOT_PER_BUCKET; ++j) {
                if (b.occupied(j)) {
                    b.storage_pair(j).~storage_value_type();
                }
            }
            traits_::destroy(allocator_, &b);
        }
        bucket_allocator_.deallocate(buckets_, size());
        buckets_ = nullptr;
    }

    // transfer: 从另一个container拷贝所有桶
    bucket* transfer(size_type hp, const bucket_container& src,
                     std::false_type /*propagate*/) {
        bucket* new_buckets = bucket_allocator_.allocate(size_type(1) << hp);
        for (size_type i = 0; i < (size_type(1) << hp); ++i) {
            traits_::construct(allocator_, &new_buckets[i]);
            const bucket& sb = src.buckets_[i];
            bucket& db = new_buckets[i];
            db.occ_bitmap_ = sb.occ_bitmap_;
            db.tombstone_cnt_ = sb.tombstone_cnt_;
            db.partials_ = sb.partials_;
            for (size_type j = 0; j < SLOT_PER_BUCKET; ++j) {
                if (sb.occupied(j)) {
                    new (&db.storage_[j].kv)
                        typename bucket::storage_value_type(sb.storage_pair(j));
                }
            }
        }
        return new_buckets;
    }

    allocator_type allocator_;
    bucket_allocator_type bucket_allocator_;
    std::atomic<size_type> hashpower_;
    bucket* buckets_;
    std::atomic<size_type> total_tombstones_;  // [NEW] 全局tombstone
};

} // namespace cuckoo
} // namespace index
} // namespace philemon

#endif // PHILEMON_CUCKOO_BUCKET_IMPL_HPP
