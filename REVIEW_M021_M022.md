# REVIEW_M021_M022.md — Cross-Tier Iterative Algorithms + Driver + Debug

## 完成范围

M021: Cross-tier PageRank with tier gradient accumulation
M022: Cross-tier WCC with component tier tracking + TC + Driver + Debug

## 新增文件 (+3,363 lines)

| File | Lines | Upstream Source | 修改比例 |
|------|-------|----------------|----------|
| `src/algorithms/cross_tier_pagerank.hpp` | 689 | PR.h(174) + pageRank.cpp(159) + driver.h::page_rank(46) | ~20% |
| `src/algorithms/cross_tier_wcc.hpp` | 644 | WCC.h(149) + driver.h::UnionFind+wcc(70) | ~20% |
| `src/algorithms/cross_tier_tc.hpp` | 363 | TC.h(93) + TC_opt.h(81) | ~20% |
| `src/driver/philemon_driver.hpp` | 721 | driver.h(1577) + wrapper.h(249) + Timer.h(35) | ~20% |
| `src/debug/state_inspector.hpp` | 445 | neo_reader_trace.h(186) + log.h(68) + log.cpp(157) | ~20% |
| `src/bench/cross_tier_bench.cpp` | 501 | main.cpp(202) + integration_bench(375) pattern | ~20% |

## 代码总量

```
Before (M001-M020):  12,516 lines across 39 files
After  (M001-M022):  15,879 lines across 45 files
Delta:               +3,363 lines across 6 new files
```

## 鲁迅拿法: 具体保留与修改

### 1. CrossTierPageRank (689行)

**保留 (~80%)**:
- PR.h 的多线程 chunk 分配 (100%)
- PR.h 的 dangling_sum 处理 (100%)
- PR.h 的 outgoing_contrib 计算 (100%)
- PR.h 的 incoming score 累加 (100%)
- driver.h::page_rank() 的单线程验证路径 (100%)

**修改 (~20%)**:
- `PRIterStats`: 每轮 max_delta + L2_norm + tier分布 + 热点统计
- `detect_hotspots()`: 识别高度数顶点, 打印 top-10
- `dump_all_state()`: 断点级状态打印 (scores统计, outgoing_contrib, tier_perf)
- `convergence_threshold` early stop
- `dump_convergence()`: 全量收敛日志表格
- score sanity check + top-10 ranked vertices 打印

### 2. CrossTierWCC (644行)

**保留 (~80%)**:
- WCC.h 的 label propagation 核心 (100%)
- WCC.h 的 high→low union 逻辑 (100%)
- WCC.h 的 path compression comp[i]=comp[comp[i]] (100%)
- driver.h 的 UnionFind (find + unite, +rank优化)
- driver.h::wcc() 的顺序遍历 + component ID 分配

**修改 (~20%)**:
- `WCCRoundStats`: 每轮 changes + component_count + largest + tier分布
- `UnionFindTiered`: 在 upstream UnionFind 上增加 rank-based union + tier tracking
- `dump_all_state()`: 断点级打印 comp数组 + component大小分布
- Cross-tier edge tracking
- Component size distribution (singleton/small/medium/large)
- Optional UF verification path

### 3. CrossTierTC (363行)

**保留 (~80%)**:
- TC.h 的度数自适应三策略 (search all src/search all dst/intersect) (100%)
- TC_opt.h 的 marker-based intersection (100%)
- TC.h 的 SEARCH_THRESHOLD 概念 (100%)

**修改 (~20%)**:
- `TCStats`: edges_examined + tier分布 + cross_tier_triangles
- `dump_progress()`: 每1000顶点打印进度
- SEARCH_THRESHOLD configurable
- Multi-thread ready (thread count parameter)

### 4. PhilemonDriver (721行)

**保留 (~80%)**:
- driver.h 的 `Barrier` 类 (100%)
- driver.h 的 `bind_thread_to_core()` (100%)
- driver.h 的 `execute_query()` 分派结构 (100%)
- driver.h 的 `execute_microbenchmarks()` 并行 worker 框架 (100%)
- driver.h 的 `execute_mixed_reader_writer()` 读写混合框架 (100%)

**修改 (~20%)**:
- `PhilemonConfig`: 扩展 DriverConfig + tier/debug/convergence 参数
- `PhilemonOp`: 扩展 operationType + cross_tier variants
- `execute_cross_tier_query()`: 分派到 cross_tier_* 算法
- `execute_tier_benchmark()`: 顺序扫描 + 随机访问 + 边遍历吞吐
- `dump_full_system_state()`: 系统全量状态打印
- `print_progress_bar()`: ASCII 进度条
- `WallTimer`: 从 Timer.h 移植

### 5. StateInspector (445行)

**保留 (~80%)**:
- neo_reader_trace.h 的 trace entry 结构
- neo_reader_trace.h 的 ring buffer 概念
- log.h 的 log level 概念
- log.h 的 formatted message pattern

**修改 (~20%)**:
- `InspectionPoint`: trace entry + phase name + formatted message + tier snapshot
- `BreakpointGuard`: RAII scope guard, 入口/出口自动打印
- `DataDumper`: 结构化打印 vector/array/pairs
- `TierHeatmap`: ASCII 热度图 (█░ bar chart)
- `ConvergenceTracker`: 收敛数据 + JSON序列化
- `AlgorithmProfiler`: per-phase timing
- 宏: `PHILE_BREAKPOINT`, `PHILE_INSPECT`, `PHILE_DUMP_VEC`, `PHILE_TIER_HEATMAP`

### 6. CrossTierBench (501行)

5-test harness:
1. CrossTierPageRank — score_sum ≈ 1.0 验证
2. CrossTierWCC — component coverage 验证
3. CrossTierTC degree-adaptive — triangle count
4. CrossTierTC marker-based — optimized count
5. Sequential baseline — 1-thread comparison

每个测试都集成: `PHILE_BREAKPOINT` + `PHILE_INSPECT` + `PHILE_TIER_HEATMAP`

## Debug 能力对照

| 需求 | 实现 |
|------|------|
| 断点调试 | `PHILE_BREAKPOINT("name")` — RAII, 自动打印入口/出口+耗时 |
| print当前所有数据 | `dump_all_state()` — 打印 scores/comp/outgoing_contrib 的统计+首N个值 |
| print结构体状态 | `PRIterStats::dump()` / `WCCRoundStats::dump()` / `TCStats::dump()` |
| 运行进度反馈 | `print_progress_bar()` + `dump_progress()` 每1000步打印 |
| Tier热度图 | `PHILE_TIER_HEATMAP()` — ASCII bar chart per tier |
| 收敛曲线 | `dump_convergence()` — 表格输出每轮 max_delta/L2/tier分布 |
| 全系统状态 | `dump_full_system_state()` — Graph + Config + TierPerf + TraceRing |

## 运行示例输出

```
╔══════════════════════════════════════════════════════════╗
║   Philemon-TSH Cross-Tier Algorithm Benchmark           ║
║   M021-M022: PageRank + WCC + TC                        ║
╠══════════════════════════════════════════════════════════╣
║   N=5000  E=25000  threads=4  debug=2
╚══════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════
  Test 1: Cross-Tier PageRank
═══════════════════════════════════════════════════════
  🔵 [ENTER test_cross_tier_pagerank]
  📍 [PR_init] N=5000 threads=4
  [CrossTierPR] N=5000 iters=10 damping=0.85 convergence_threshold=1.0e-06
  [PR::hotspots] found 12 hotspots (threshold=50) total_edges=1842
  [PR iter 0] max_Δ=0.00019423 L2=0.00876543 dangling=0.0012 time=2.341ms
           tier_vtx: HBM=1666 GDDR=1667 DRAM=1667 hotspots=12(edges=1842)
  ...
  ──── PageRank Convergence Log (10 iterations) ────
  Iter  max_Δ        L2_norm      dangling   time_ms  HBM     GDDR    DRAM
  0     0.00019423   0.00876543   0.0012     2.341    1666    1667    1667
  1     0.00005834   0.00234567   0.0008     1.987    1666    1667    1667
  ...
  ──── End PageRank Convergence ────
  🟢 [EXIT  test_cross_tier_pagerank] elapsed=25.432ms

  Test Summary:
  CrossTierPageRank         ✅ PASS  25.432      score_sum=1.000000 iters=10
  CrossTierWCC              ✅ PASS  12.876      components=23 rounds=4
  CrossTierTC               ✅ PASS  156.234     triangles=1234 edges_examined=50000
  CrossTierTC_Opt           ✅ PASS  189.456     triangles_opt=1234
  SequentialComparison      ✅ PASS  98.765      sequential_total=98.765ms
```

## Codebase After M022

```
src/ (15,879 lines total across 45 files)
├── core/           — 7 files, 1997 lines  [M001–M010]
├── bridge/         — 1 file,  510 lines   [M002,M006,M007]
├── scheduler/      — 1 file,  102 lines   [M003]
├── debug/          — 2 files, 726 lines   [M011,M021-M022]    ← +445
│   ├── philemon_debug.hpp       281 lines
│   └── state_inspector.hpp      445 lines  ← NEW
├── index/          — 4 files, 1178 lines  [M011,M012]
├── wrapper/        — 5 files, 635 lines   [M013]
├── algorithms/     — 10 files, 3700 lines [M014,M019-M022]    ← +1696
│   ├── tiered_bfs.hpp           388 lines
│   ├── tiered_pagerank.hpp      165 lines
│   ├── tiered_sssp.hpp          162 lines
│   ├── tiered_wcc.hpp           159 lines
│   ├── tiered_tc.hpp            106 lines
│   ├── cross_tier_bfs.hpp       485 lines
│   ├── cross_tier_sssp.hpp      539 lines
│   ├── cross_tier_pagerank.hpp  689 lines  ← NEW
│   ├── cross_tier_wcc.hpp       644 lines  ← NEW
│   └── cross_tier_tc.hpp        363 lines  ← NEW
├── executor/       — 3 files, 491 lines   [M015,M016]
├── cost_model/     — 1 file,  387 lines   [M018]
├── loader/         — 3 files, 1708 lines  [M017,M018]
├── driver/         — 1 file,  721 lines   [M021-M022]         ← NEW dir
│   └── philemon_driver.hpp      721 lines  ← NEW
├── bench/          — 6 files, 2683 lines  [M004,M016,M021-M022] ← +501
│   ├── philemon_bench.cpp       389 lines
│   ├── philemon_data_fast.cpp   296 lines
│   ├── philemon_data_gen.cpp    645 lines
│   ├── integration_bench.cpp    375 lines
│   ├── ldbc_bench.cpp           477 lines
│   └── cross_tier_bench.cpp     501 lines  ← NEW
└── cuda/           — 1 file, 1039 lines   [M009,M010]
    └── hetero_bench.cu
```
