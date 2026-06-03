/**
 * test_core.cpp — 核心模块单元测试
 *
 * M036: core模块UT覆盖
 *   - SlabAllocator: 初始化/分配/回收/size class
 *   - TieredAllocator: 多tier分配/迁移/deallocate/budget/遍历
 *   - SeqLock: 读写一致性/并发读写
 *   - TierPtr: 构造/move/void特化
 *   - TemporalEdge: 构造/比较/hash/interval判断
 *   - Debug: level/TraceRing/TierPerfCounter
 */

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <set>
#include <cstring>

#include "core/slab_allocator.hpp"
#include "core/tiered_allocator.hpp"
#include "core/seqlock.hpp"
#include "core/tier_ptr.hpp"
#include "core/temporal_edge.hpp"
#include "debug/philemon_debug.hpp"

using namespace philemon;

// ═══════════════════════════════════════════════════════════════════
// SlabAllocator Tests
// ═══════════════════════════════════════════════════════════════════

class SlabAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override { debug::set_debug_level(0); }
};

TEST_F(SlabAllocatorTest, SlabPageInit) {
    constexpr size_t PAGE_SIZE = 4096;
    constexpr size_t SLOT_SIZE = 64;
    alignas(64) uint8_t raw[PAGE_SIZE];

    SlabPage page;
    page.init(raw, PAGE_SIZE, SLOT_SIZE);

    EXPECT_EQ(page.slot_size, SLOT_SIZE);
    EXPECT_GT(page.slot_count, 0u);
    EXPECT_FALSE(page.is_full());
    EXPECT_TRUE(page.is_empty());
}

TEST_F(SlabAllocatorTest, SizeClassRounding) {
    // slab_size_class: bytes → class index
    // class 0 = 4KB, class 1 = 8KB, ...
    EXPECT_EQ(slab_size_class(1), 0u);      // ≤4KB → class 0
    EXPECT_EQ(slab_size_class(4096), 0u);   // exactly 4KB → class 0
    EXPECT_EQ(slab_size_class(4097), 1u);   // >4KB → class 1 (8KB)
    EXPECT_EQ(slab_size_class(8192), 1u);   // exactly 8KB → class 1
}

TEST_F(SlabAllocatorTest, ClassSizeInverse) {
    // slab_class_size(cls) returns the byte size of a class
    EXPECT_EQ(slab_class_size(0), 4096u);     // class 0 = 4KB
    EXPECT_EQ(slab_class_size(1), 8192u);     // class 1 = 8KB
    EXPECT_EQ(slab_class_size(2), 16384u);    // class 2 = 16KB
}

TEST_F(SlabAllocatorTest, SlabPoolDefaultConstruct) {
    SlabPool pool;
    EXPECT_EQ(pool.size_class, 0u);
    EXPECT_EQ(pool.slot_size, 0u);
}

TEST_F(SlabAllocatorTest, SlabPoolParameterized) {
    SlabPool pool(1, 8192);
    EXPECT_EQ(pool.size_class, 1u);
    EXPECT_EQ(pool.slot_size, 8192u);
}

// ═══════════════════════════════════════════════════════════════════
// TieredAllocator Tests
// ═══════════════════════════════════════════════════════════════════

class TieredAllocatorTest : public ::testing::Test {
protected:
    // Constructor requires (hbm_cap, gddr_cap, dram_cap)
    TieredAllocator allocator{1024*1024, 4*1024*1024, 16*1024*1024};

    void SetUp() override { debug::set_debug_level(0); }
};

TEST_F(TieredAllocatorTest, AllocateToPreferredTier) {
    uint64_t id = allocator.allocate(4096, MemoryTier::HBM);
    EXPECT_GT(id, 0u);

    AllocMeta meta;
    EXPECT_TRUE(allocator.get_meta(id, meta));
    EXPECT_EQ(meta.size_bytes, 4096u);
    EXPECT_EQ(meta.current_tier, MemoryTier::HBM);
}

TEST_F(TieredAllocatorTest, AllocateFallbackWhenFull) {
    // Fill HBM (1MB budget), then allocate more
    std::vector<uint64_t> ids;
    for (int i = 0; i < 300; i++) {
        uint64_t id = allocator.allocate(4096, MemoryTier::HBM);
        if (id == 0) break;
        ids.push_back(id);
    }
    // Budget exhausted — next alloc should fallback
    uint64_t id = allocator.allocate(4096, MemoryTier::HBM);
    if (id > 0) {
        AllocMeta meta;
        allocator.get_meta(id, meta);
        EXPECT_NE(meta.current_tier, MemoryTier::HBM);
    }
    for (auto aid : ids) allocator.deallocate(aid);
}

TEST_F(TieredAllocatorTest, MigrateBetweenTiers) {
    uint64_t id = allocator.allocate(4096, MemoryTier::DRAM);
    EXPECT_GT(id, 0u);

    AllocMeta before;
    allocator.get_meta(id, before);
    EXPECT_EQ(before.current_tier, MemoryTier::DRAM);

    bool ok = allocator.migrate(id, MemoryTier::HBM);
    EXPECT_TRUE(ok);

    AllocMeta after;
    allocator.get_meta(id, after);
    EXPECT_EQ(after.current_tier, MemoryTier::HBM);
}

TEST_F(TieredAllocatorTest, DeallocateFreesMemory) {
    size_t before_total = allocator.total_allocated();
    uint64_t id = allocator.allocate(4096, MemoryTier::DRAM);
    EXPECT_GT(allocator.total_allocated(), before_total);

    allocator.deallocate(id);
    AllocMeta meta;
    EXPECT_FALSE(allocator.get_meta(id, meta));
}

TEST_F(TieredAllocatorTest, TouchUpdatesAccessTime) {
    uint64_t id = allocator.allocate(4096, MemoryTier::DRAM);
    AllocMeta m1;
    allocator.get_meta(id, m1);

    allocator.touch(id);

    AllocMeta m2;
    allocator.get_meta(id, m2);
    EXPECT_GE(m2.last_access_ns.load(), m1.last_access_ns.load());
}

TEST_F(TieredAllocatorTest, ForEachAllocIteratesAll) {
    uint64_t id1 = allocator.allocate(4096, MemoryTier::HBM);
    uint64_t id2 = allocator.allocate(4096, MemoryTier::GDDR);
    uint64_t id3 = allocator.allocate(4096, MemoryTier::DRAM);

    std::set<uint64_t> seen;
    allocator.for_each_alloc([&](uint64_t aid, const AllocMeta&) {
        seen.insert(aid);
    });
    EXPECT_TRUE(seen.count(id1));
    EXPECT_TRUE(seen.count(id2));
    EXPECT_TRUE(seen.count(id3));
}

// ═══════════════════════════════════════════════════════════════════
// SeqLock Tests
// ═══════════════════════════════════════════════════════════════════

class SeqLockTest : public ::testing::Test {
protected:
    SeqLock lock;
};

TEST_F(SeqLockTest, BasicReadWrite) {
    int shared_data = 0;
    lock.write_lock();
    shared_data = 42;
    lock.write_unlock();

    uint64_t seq;
    int read_val;
    do {
        seq = lock.read_begin();
        read_val = shared_data;
    } while (lock.read_retry(seq));

    EXPECT_EQ(read_val, 42);
}

TEST_F(SeqLockTest, SequenceIncrementsOnWrite) {
    uint64_t s1 = lock.sequence();
    lock.write_lock();
    lock.write_unlock();
    uint64_t s2 = lock.sequence();
    EXPECT_GT(s2, s1);
}

TEST_F(SeqLockTest, WriteGuardRAII) {
    uint64_t s1 = lock.sequence();
    {
        SeqLockWriteGuard guard(lock);
    }
    uint64_t s2 = lock.sequence();
    EXPECT_GT(s2, s1);
}

TEST_F(SeqLockTest, ConcurrentReaders) {
    std::atomic<int> shared{0};
    std::atomic<int> success_count{0};

    lock.write_lock();
    shared.store(100);
    lock.write_unlock();

    std::vector<std::thread> readers;
    for (int i = 0; i < 8; i++) {
        readers.emplace_back([&]() {
            for (int j = 0; j < 1000; j++) {
                uint64_t seq;
                int val;
                do {
                    seq = lock.read_begin();
                    val = shared.load();
                } while (lock.read_retry(seq));
                if (val == 100) success_count.fetch_add(1);
            }
        });
    }
    for (auto& t : readers) t.join();
    EXPECT_EQ(success_count.load(), 8 * 1000);
}

// ═══════════════════════════════════════════════════════════════════
// TierPtr Tests
// ═══════════════════════════════════════════════════════════════════

TEST(TierPtrTest, DefaultConstruct) {
    TierPtr<uint64_t> ptr;
    EXPECT_EQ(ptr.size(), 0u);
}

TEST(TierPtrTest, MoveSemantics) {
    std::shared_mutex mu;
    uint64_t data[5] = {1, 2, 3, 4, 5};
    std::shared_lock<std::shared_mutex> lk(mu);

    TierPtr<uint64_t> a(data, 5 * sizeof(uint64_t), std::move(lk));
    EXPECT_EQ(a.size(), 5 * sizeof(uint64_t));

    TierPtr<uint64_t> b(std::move(a));
    EXPECT_EQ(b.size(), 5 * sizeof(uint64_t));
    EXPECT_EQ(a.size(), 0u);
}

TEST(TierPtrTest, VoidSpecialization) {
    std::shared_mutex mu;
    uint8_t data[1024];
    std::shared_lock<std::shared_mutex> lk(mu);

    TierPtr<void> vp(data, 1024, std::move(lk));
    EXPECT_EQ(vp.size(), 1024u);
}

// ═══════════════════════════════════════════════════════════════════
// TemporalEdge Tests
// ═══════════════════════════════════════════════════════════════════

TEST(TemporalEdgeTest, ConstructionAndEquality) {
    TemporalEdge e1{10, 20, 1.5, 100, 200};
    TemporalEdge e2{10, 20, 1.5, 100, 200};
    TemporalEdge e3{10, 21, 1.5, 100, 200};
    EXPECT_EQ(e1, e2);
    EXPECT_FALSE(e1 == e3);
}

TEST(TemporalEdgeTest, ContainedIn) {
    TemporalEdge e{1, 2, 1.0, 50, 150};
    EXPECT_TRUE(e.contained_in(0, 200));
    EXPECT_FALSE(e.contained_in(60, 140));
    EXPECT_TRUE(e.contained_in(50, 150));
}

TEST(TemporalEdgeTest, Overlaps) {
    TemporalEdge e{1, 2, 1.0, 50, 150};
    EXPECT_TRUE(e.overlaps(0, 100));
    EXPECT_TRUE(e.overlaps(100, 200));
    EXPECT_TRUE(e.overlaps(60, 140));
    EXPECT_FALSE(e.overlaps(200, 300));
    EXPECT_FALSE(e.overlaps(0, 49));
}

TEST(TemporalEdgeTest, HashDeterministic) {
    TemporalEdge e{5, 10, 2.0, 100, 200};
    EXPECT_EQ(e.hash(), e.hash());

    TemporalEdge e2{5, 11, 2.0, 100, 200};
    EXPECT_NE(e.hash(), e2.hash());
}

// ═══════════════════════════════════════════════════════════════════
// Debug Tests
// ═══════════════════════════════════════════════════════════════════

TEST(DebugTest, SetAndGetLevel) {
    debug::set_debug_level(0);
    EXPECT_EQ(debug::get_debug_level(), 0);
    debug::set_debug_level(3);
    EXPECT_EQ(debug::get_debug_level(), 3);
    debug::set_debug_level(0);
}

TEST(DebugTest, TraceRingDoesNotCrash) {
    debug::set_debug_level(3);
    auto& ring = debug::global_trace();
    ring.record(debug::TraceEvent::ALLOC, 0, 42);
    ring.record(debug::TraceEvent::MIGRATE_START, 1, 99);
    debug::set_debug_level(0);
}

TEST(DebugTest, TierPerfCounterAccumulates) {
    auto& pc = debug::tier_perf(0);
    uint64_t before = pc.read_count.load();
    pc.read_count.fetch_add(10);
    EXPECT_EQ(pc.read_count.load(), before + 10);
}
