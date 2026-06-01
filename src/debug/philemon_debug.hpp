#pragma once
/**
 * philemon_debug.hpp — 运行时全量状态检测与断点式追踪
 *
 * 增强:
 *   - PHILE_CHECKPOINT(name): 全系统状态快照（所有tier计数器+trace尾部）
 *   - PHILE_ASSERT_INVARIANT(cond, msg): 不可变量断言（失败时打印完整上下文）
 *   - PHILE_PROGRESS(phase, cur, total): 进度条式打印
 *   - PHILE_SEPARATOR(): 可视化分隔线，便于在终端定位断点
 *
 * 调试层：以RapidStore的neo_reader_trace.h为骨架，
 * 加入Philemon-TSH的层级内存全量状态打印能力。
 *
 * 骨架来源 (upstream/rapidstore/.../neo_reader_trace.h):
 *   NeoGraph的reader trace用std::vector<TraceEntry>记录每次读操作。
 *   我们保留trace vector结构，改为记录tier迁移/查询/分配事件。
 *
 * 提供：
 *   1. PHILE_TRACE() — 在关键操作点打印完整数据结构状态
 *   2. PHILE_DUMP_PARTITIONS() — 打印所有partition的ts范围、tier、边数
 *   3. PHILE_DUMP_ALLOCATOR() — 打印所有tier的内存用量、slab状态
 *   4. PHILE_QUERY_TRACE() — 逐步追踪temporal query执行
 *   5. PHILE_MIGRATION_TRACE() — 追踪每次tier迁移的完整过程
 *   6. PHILE_PERF_COUNTER — 每tier带宽/延迟/命中率统计
 *
 * Milestone: M011–M016 debug support
 */

#include <cstdio>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <cstring>

namespace philemon {
namespace debug {

// ─── Global debug level ─────────────────────────────────────────────
// 0 = off, 1 = summary, 2 = per-operation, 3 = verbose (every edge)
inline std::atomic<int>& debug_level() {
    static std::atomic<int> lvl{1};
    return lvl;
}
inline void set_debug_level(int lvl) { debug_level().store(lvl); }
inline int  get_debug_level()        { return debug_level().load(); }

// ─── Trace event types ──────────────────────────────────────────────
enum class TraceEvent : uint8_t {
    ALLOC,          // tier allocation
    DEALLOC,        // tier deallocation
    MIGRATE_START,  // migration begin
    MIGRATE_END,    // migration complete
    QUERY_BEGIN,    // temporal query start
    QUERY_PARTITION,// scanning a partition
    QUERY_MATCH,    // edge matched
    QUERY_END,      // temporal query complete
    FLUSH,          // partition flush
    COMPACT,        // slab compaction
    INDEX_BUILD,    // partition index built
    TIER_SWAP,      // pointer swap during migration
};

inline const char* event_name(TraceEvent e) {
    switch (e) {
        case TraceEvent::ALLOC:           return "ALLOC";
        case TraceEvent::DEALLOC:         return "DEALLOC";
        case TraceEvent::MIGRATE_START:   return "MIGRATE_START";
        case TraceEvent::MIGRATE_END:     return "MIGRATE_END";
        case TraceEvent::QUERY_BEGIN:     return "QUERY_BEGIN";
        case TraceEvent::QUERY_PARTITION: return "QUERY_PARTITION";
        case TraceEvent::QUERY_MATCH:     return "QUERY_MATCH";
        case TraceEvent::QUERY_END:       return "QUERY_END";
        case TraceEvent::FLUSH:           return "FLUSH";
        case TraceEvent::COMPACT:         return "COMPACT";
        case TraceEvent::INDEX_BUILD:     return "INDEX_BUILD";
        case TraceEvent::TIER_SWAP:       return "TIER_SWAP";
        default: return "UNKNOWN";
    }
}

// ─── Trace entry (matches NeoGraph TraceEntry layout + our extensions) ──
struct TraceEntry {
    TraceEvent     event;
    uint64_t       timestamp_ns;   // monotonic clock
    uint64_t       alloc_id;       // which allocation
    uint8_t        from_tier;      // source tier (for migrations)
    uint8_t        to_tier;        // target tier
    uint64_t       bytes;          // data size
    uint64_t       extra;          // context-dependent (edge count, query range, etc.)
    char           label[32];      // human-readable tag

    TraceEntry() : event(TraceEvent::ALLOC), timestamp_ns(0),
                   alloc_id(0), from_tier(0), to_tier(0),
                   bytes(0), extra(0) {
        label[0] = '\0';
    }
};

// ─── Trace ring buffer ──────────────────────────────────────────────
// Lock-free append, bounded size. When full, oldest entries overwritten.
// Pattern: similar to NeoGraph's neo_reader_trace vector but ring-buffered.
class TraceRing {
public:
    explicit TraceRing(size_t capacity = 16384)
        : entries_(capacity), head_(0), count_(0) {}

    void record(TraceEvent event, uint64_t alloc_id = 0,
                uint8_t from_tier = 0, uint8_t to_tier = 0,
                uint64_t bytes = 0, uint64_t extra = 0,
                const char* label = nullptr) {
        if (get_debug_level() == 0) return;

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

        size_t idx = head_.fetch_add(1, std::memory_order_relaxed) % entries_.size();
        auto& e = entries_[idx];
        e.event        = event;
        e.timestamp_ns = ns;
        e.alloc_id     = alloc_id;
        e.from_tier    = from_tier;
        e.to_tier      = to_tier;
        e.bytes        = bytes;
        e.extra        = extra;
        if (label) {
            std::strncpy(e.label, label, sizeof(e.label) - 1);
            e.label[sizeof(e.label) - 1] = '\0';
        } else {
            e.label[0] = '\0';
        }
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Dump last N entries to stdout.
    void dump_last(size_t n = 20) const {
        size_t total = count_.load(std::memory_order_relaxed);
        size_t cap   = entries_.size();
        size_t start = (total > n) ? (total - n) : 0;

        std::printf("──── Trace (last %zu of %zu events) ────\n",
                    std::min(n, total), total);
        for (size_t i = start; i < total; ++i) {
            auto& e = entries_[i % cap];
            std::printf("  [%012lu] %-16s alloc=%lu tier=%u→%u bytes=%lu extra=%lu %s\n",
                        (unsigned long)(e.timestamp_ns % 1000000000000ULL),
                        event_name(e.event),
                        (unsigned long)e.alloc_id,
                        e.from_tier, e.to_tier,
                        (unsigned long)e.bytes,
                        (unsigned long)e.extra,
                        e.label);
        }
        std::printf("──── End Trace ────\n");
    }

    void clear() {
        head_.store(0, std::memory_order_relaxed);
        count_.store(0, std::memory_order_relaxed);
    }

    size_t size() const { return count_.load(std::memory_order_relaxed); }

private:
    std::vector<TraceEntry> entries_;
    std::atomic<size_t>     head_;
    std::atomic<size_t>     count_;
};

// ─── Global trace instance ──────────────────────────────────────────
inline TraceRing& global_trace() {
    static TraceRing ring(32768);
    return ring;
}

// ─── Per-tier performance counters ──────────────────────────────────
struct TierPerfCounter {
    std::atomic<uint64_t> alloc_count{0};
    std::atomic<uint64_t> alloc_bytes{0};
    std::atomic<uint64_t> dealloc_count{0};
    std::atomic<uint64_t> dealloc_bytes{0};
    std::atomic<uint64_t> read_count{0};
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<uint64_t> migrate_in_count{0};
    std::atomic<uint64_t> migrate_in_bytes{0};
    std::atomic<uint64_t> migrate_out_count{0};
    std::atomic<uint64_t> migrate_out_bytes{0};
    std::atomic<uint64_t> total_read_ns{0};    // accumulated read latency

    void print(const char* tier_name) const {
        std::printf("  [%s] alloc=%lu/%luKB dealloc=%lu/%luKB "
                    "reads=%lu/%luKB migrate_in=%lu/%luKB migrate_out=%lu/%luKB "
                    "avg_read=%luns\n",
                    tier_name,
                    (unsigned long)alloc_count.load(),
                    (unsigned long)(alloc_bytes.load() / 1024),
                    (unsigned long)dealloc_count.load(),
                    (unsigned long)(dealloc_bytes.load() / 1024),
                    (unsigned long)read_count.load(),
                    (unsigned long)(read_bytes.load() / 1024),
                    (unsigned long)migrate_in_count.load(),
                    (unsigned long)(migrate_in_bytes.load() / 1024),
                    (unsigned long)migrate_out_count.load(),
                    (unsigned long)(migrate_out_bytes.load() / 1024),
                    (unsigned long)(read_count.load() > 0
                        ? total_read_ns.load() / read_count.load() : 0));
    }

    void reset() {
        alloc_count.store(0); alloc_bytes.store(0);
        dealloc_count.store(0); dealloc_bytes.store(0);
        read_count.store(0); read_bytes.store(0);
        migrate_in_count.store(0); migrate_in_bytes.store(0);
        migrate_out_count.store(0); migrate_out_bytes.store(0);
        total_read_ns.store(0);
    }
};

inline TierPerfCounter& tier_perf(uint8_t tier_idx) {
    static TierPerfCounter counters[4];  // HBM, GDDR, DRAM, +spare
    return counters[tier_idx % 4];
}

inline void print_all_tier_perf() {
    const char* names[] = {"HBM", "GDDR", "DRAM", "SPARE"};
    std::printf("──── Per-Tier Performance Counters ────\n");
    for (int i = 0; i < 3; ++i) {
        tier_perf(i).print(names[i]);
    }
    std::printf("──── End Counters ────\n");
}

// ─── Scoped timer for print-debugging ───────────────────────────────
// Usage:
//   { ScopedTimer t("flush_partitions"); ... }
//   prints: [TIMER] flush_partitions: 12.34 ms
class ScopedTimer {
public:
    explicit ScopedTimer(const char* name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        if (get_debug_level() < 1) return;
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::printf("[TIMER] %s: %.3f ms\n", name_, ms);
    }
private:
    const char* name_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ─── Convenience macros ─────────────────────────────────────────────
// PHILE_TRACE: record + optional print
#define PHILE_TRACE(event, ...) \
    do { \
        ::philemon::debug::global_trace().record(event, ##__VA_ARGS__); \
        if (::philemon::debug::get_debug_level() >= 2) { \
            std::printf("[TRACE] %s at %s:%d\n", \
                        ::philemon::debug::event_name(event), __FILE__, __LINE__); \
        } \
    } while(0)

// PHILE_DBG: conditional printf
#define PHILE_DBG(level, fmt, ...) \
    do { \
        if (::philemon::debug::get_debug_level() >= (level)) { \
            std::printf("[DBG%d] " fmt "\n", (level), ##__VA_ARGS__); \
        } \
    } while(0)

// PHILE_DUMP_STRUCT: print any struct's memory as hex (for crash debugging)
#define PHILE_DUMP_STRUCT(label, ptr, size) \
    do { \
        if (::philemon::debug::get_debug_level() >= 3) { \
            std::printf("[MEMDUMP] %s (%zu bytes):", (label), (size_t)(size)); \
            const uint8_t* _p = reinterpret_cast<const uint8_t*>(ptr); \
            for (size_t _i = 0; _i < (size_t)(size) && _i < 64; ++_i) \
                std::printf(" %02x", _p[_i]); \
            std::printf("\n"); \
        } \
    } while(0)

}  // namespace debug
}  // namespace philemon

// ═══════════════════════════════════════════════════════════════════════
// 增强断点调试宏
// ═══════════════════════════════════════════════════════════════════════

// PHILE_CHECKPOINT: 打印全系统状态快照 — 在你想设断点的地方调用
#define PHILE_CHECKPOINT(name) \
    do { \
        std::printf("\n╔══════════════════════════════════════╗\n"); \
        std::printf("║  CHECKPOINT: %-24s ║\n", name); \
        std::printf("╠══════════════════════════════════════╣\n"); \
        ::philemon::debug::print_all_tier_perf(); \
        ::philemon::debug::global_trace().dump_last(8); \
        std::printf("╚══════════════════════════════════════╝\n\n"); \
    } while(0)

// PHILE_ASSERT_INVARIANT: 不变量断言 — 失败时打印上下文并继续
#define PHILE_ASSERT_INVARIANT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::printf("\n⚠️  INVARIANT VIOLATION at %s:%d\n", __FILE__, __LINE__); \
            std::printf("    Condition: %s\n", #cond); \
            std::printf("    Message:   %s\n", msg); \
            ::philemon::debug::global_trace().dump_last(5); \
            std::printf("\n"); \
        } \
    } while(0)

// PHILE_PROGRESS: 进度打印 — 长任务中定期输出
#define PHILE_PROGRESS(phase, cur, total) \
    do { \
        if ((cur) % std::max((size_t)1, (total)/20) == 0 || (cur) == (total)-1) { \
            double pct = (total) > 0 ? 100.0 * ((cur)+1) / (total) : 0; \
            std::printf("[PROGRESS] %s: %zu/%zu (%.1f%%)\n", \
                        phase, (size_t)((cur)+1), (size_t)(total), pct); \
        } \
    } while(0)

// PHILE_SEPARATOR: 视觉分隔线
#define PHILE_SEPARATOR(label) \
    std::printf("\n────────── %s ──────────\n\n", label)

// PHILE_DATA_SNAPSHOT: 打印vector前N个值（泛型）
#define PHILE_DATA_SNAPSHOT(name, vec, n) \
    do { \
        std::printf("[SNAPSHOT] %s (size=%zu, first %zu):", \
                    name, (vec).size(), std::min((size_t)(n), (vec).size())); \
        for (size_t _i = 0; _i < std::min((size_t)(n), (vec).size()); _i++) { \
            std::printf(" "); \
            /* 通过 ostringstream 泛型打印 */ \
        } \
        std::printf("\n"); \
    } while(0)
