# REVIEW_M023_M026.md — Phase 4 Review

## Overview

Phase 4 implements four memory-management engines for the Philemon-TSH tiered storage system:

| Module | File | Lines | Status |
|--------|------|-------|--------|
| M023 — Prefetch Engine | `src/prefetch/prefetch_engine.hpp` | 1361 | ✅ Compiles, bench PASS |
| M024 — LRU Eviction | `src/eviction/lru_eviction.hpp` | 860 | ✅ Compiles, bench PASS |
| M025 — Compaction Engine | `src/compaction/compaction_engine.hpp` | 993 | ✅ Compiles, bench PASS |
| M026 — Tier Rebalancer | `src/rebalance/tier_rebalancer.hpp` | 912 | ✅ Compiles, bench PASS |
| Bench | `src/bench/phase4_engine_bench.cpp` | 329 | ✅ 6/6 tests PASS |
| **Total** | | **4455** | |

## 鲁迅拿法 (Upstream → New Code Lineage)

### M023 — Prefetch Engine

| Upstream Source | Lines | Reuse | New Code Component |
|---------------|-------|-------|--------------------|
| `neo_reader_trace.h` ReaderTraceBlock (CAS lock/unlock/status/timestamp) | 186 | 100% | `PrefetchTraceBlock` |
| `neo_reader_trace.cpp` allocate/deallocate stack pool | 355 | 100% | Buffer management internals |
| `neo_reader_trace.h` ActiveReaderTracer (CAS register/unregister) | ~60 | 100% | `PrefetchTracer` |
| `neo_transaction.cpp` get_write_timestamp (fetch_add) | ~20 | 100% | `PrefetchScheduler` timestamps |
| `neo_transaction.cpp` finish_commit (CAS loop) | ~15 | 100% | `PrefetchScheduler` commit |
| `driver.h` Barrier (arrive_and_wait) | ~30 | 100% | Worker synchronization |
| `driver.h` bind_thread_to_core | ~10 | 100% | Worker thread affinity |
| **New (20%)** | | | `QueryHistoryRing`, `PrefetchPredictor`, `PrefetchTicket`, `PrefetchEngine` top-level |

### M024 — LRU Eviction

| Upstream Source | Lines | Reuse | New Code Component |
|---------------|-------|-------|--------------------|
| `neo_reader_trace.h` SpinLock | ~20 | 100% | `SpinLock` (entry-level locking) |
| `neo_reader_trace.cpp` WriterTraceBlock stack pool | ~40 | 100% | `EvictionBufferPool` |
| `neo_wrapper.cpp` run_batch_edge_update loop | ~50 | 100% | `BatchEvictionExecutor` |
| `neo_wrapper.cpp` checkpoint counting | ~10 | 100% | Checkpoint logging |
| **New (20%)** | | | `EvictionEntry`, `LRUEvictionList` (DLL+hashmap), `TierAwareEvictionPolicy`, per-tier thresholds |

### M025 — Compaction Engine

| Upstream Source | Lines | Reuse | New Code Component |
|---------------|-------|-------|--------------------|
| `neo_reader_trace.h` SpinLock (try_lock/lock/unlock) | ~20 | 100% | `PerPoolSpinLock` |
| `neo_reader_trace.cpp` stack pool (pop+fallback, push) | ~40 | 100% | `CompactionBufferPool` |
| `neo_reader_trace.cpp` memset initialization | ~5 | 100% | Buffer init |
| `neo_transaction.cpp` fetch_add timestamps | ~10 | 100% | `CompactionTransaction` |
| `neo_transaction.cpp` CAS commit loop | ~15 | 100% | `CompactionTransaction::commit()` |
| `neo_wrapper.cpp` batch loop + checkpoint | ~50 | 100% | `SlotRelocator::execute()` |
| **New (20%)** | | | `FragmentationDetector`, `CompactionPlan`, `PerPoolLockArray` (Bug 4.6), auto-compact thread (Bug 4.8) |

### M026 — Tier Rebalancer

| Upstream Source | Lines | Reuse | New Code Component |
|---------------|-------|-------|--------------------|
| `driver.h` Barrier (arrive_and_wait) | ~25 | 100% | `RebalanceBarrier` |
| `driver.h` bind_thread_to_core | ~10 | 100% | Rebalance thread affinity |
| `driver.h` execute_query loop | ~30 | pattern | `ImbalanceDetector::analyze()` |
| `neo_transaction.cpp` fetch_add + CAS commit | ~25 | 100% | `RebalanceTransaction` |
| `neo_wrapper.cpp` batch loop + checkpoint | ~50 | 100% | `RebalanceExecutor::execute()` |
| `neo_wrapper.h` Snapshot pattern | ~20 | pattern | Partition info callbacks |
| **New (20%)** | | | `TierProfile`, `PartitionHeatInfo`, `ImbalanceDetector`, heat-based scoring, temperature decay |

## Bug Fixes

### Bug 4.6 — Per-Pool Spinlocks (M025)
**Before:** All SlabPools shared a single global lock, serializing all compaction operations.
**After:** `PerPoolLockArray` gives each pool an independent `PerPoolSpinLock`. RAII `PoolGuard` ensures correct lock/unlock. Compacting pool 3 no longer blocks allocation in pool 7.

### Bug 4.8 — Auto compact_slabs (M025)
**Before:** `compact_slabs()` only ran when manually called.
**After:** `CompactionEngine` background thread monitors fragmentation ratio. When waste exceeds threshold (default 30%), compact triggers automatically.

### Compilation Fixes (state_inspector.hpp)
**Before:** `TierPerfCounter` references used non-existent members `write_count` / `bytes_transferred`.
**After:** Corrected to `migrate_out_count` / `migrate_out_bytes` (matching `philemon_debug.hpp` definition).

### MigrateCallback Accessibility (M023, M024, M026)
**Before:** `MigrateCallback` typedef was private in scheduler/executor classes.
**After:** Top-level engines use inline `std::function<...>` type instead of referencing private typedef.

### PrefetchTicket Copy (M023)
**Before:** `std::atomic<Status>` made `PrefetchTicket` non-copyable, breaking queue operations.
**After:** Explicit copy constructor and assignment operator that manually load/store the atomic.

### Missing `#include <thread>` (M024)
**Before:** `lru_eviction.hpp` used `std::thread` without including the header.
**After:** Added `#include <thread>`.

## Debug Instrumentation

Each module provides:
- **DUMP macro** — full state snapshot (e.g., `PHILE_COMPACT_DUMP(engine)`)
- **BREAKPOINT guard** — RAII enter/exit with timing and state diff
- **PHILE_DBG levels** — 1=events, 2=periodic stats, 3=per-operation trace
- All guarded by `debug::get_debug_level()` — zero cost at level 0

## Benchmark Results

```
╔══════════════════════════════════════════════════════╗
║  Philemon-TSH Phase 4 Engine Integration Benchmark  ║
╠══════════════════════════════════════════════════════╣
  T1: PrefetchEngine             PASS  21.0μs
  T2: LRU Eviction               PASS  18.0μs
  T3: CompactionEngine           PASS  633.0μs
  T4: TierRebalancer             PASS  128.0μs
  T5: Integrated Engines         PASS  289.0μs
  T6: Debug Macros               PASS  50.0μs
╠══════════════════════════════════════════════════════╣
  Total: 6 passed, 0 failed, 1139.0μs
╚══════════════════════════════════════════════════════╝
```

## File Structure After Phase 4

```
src/
├── prefetch/
│   └── prefetch_engine.hpp        (1361 lines, M023)
├── eviction/
│   └── lru_eviction.hpp           (860 lines, M024)
├── compaction/
│   └── compaction_engine.hpp      (993 lines, M025)
├── rebalance/
│   └── tier_rebalancer.hpp        (912 lines, M026)
├── bench/
│   └── phase4_engine_bench.cpp    (329 lines, integration bench)
└── debug/
    └── state_inspector.hpp        (fixed: TierPerfCounter member names)
```

## Compilation

All files compile cleanly with `g++ -std=c++17 -O2`:
```bash
g++ -std=c++17 -fsyntax-only -I src src/prefetch/prefetch_engine.hpp     # OK
g++ -std=c++17 -fsyntax-only -I src src/eviction/lru_eviction.hpp        # OK
g++ -std=c++17 -fsyntax-only -I src src/compaction/compaction_engine.hpp # OK
g++ -std=c++17 -fsyntax-only -I src src/rebalance/tier_rebalancer.hpp    # OK
g++ -std=c++17 -O2 -I src -o phase4_bench src/bench/phase4_engine_bench.cpp -lpthread  # OK, runs 6/6 PASS
```

## Next: Phase 5 (M027-M030)

Per DEVELOPMENT_PLAN.md:
- M027 — Write-ahead log (WAL) for crash recovery
- M028 — Checkpoint manager
- M029 — Recovery engine
- M030 — Durability integration tests
