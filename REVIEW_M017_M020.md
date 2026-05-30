# REVIEW_M017_M020.md — LDBC Loader + Tier Cost Model + Cross-Tier Algorithms

## 概览 (Overview)

本次 Claude session (第2位Claude) 完成 Phase 3 的 M017-M020，涵盖:
- **M017**: LDBC SNB 时序图加载器 (3 files)
- **M018**: 异构内存层级代价模型 (1 file)
- **M019**: 跨层 BFS + frontier prefetch (1 file)
- **M020**: 跨层 SSSP + tier penalty delta-stepping (1 file)
- **集成基准**: ldbc_bench.cpp (1 file)
- **Makefile 更新**: 新增 ldbc_bench target

## 新增文件清单

| 文件 | 行数 | 来源 | 修改比例 |
|------|------|------|----------|
| `src/loader/ldbc_types.hpp` | ~180 | upstream dataset_preprocessor/types.hpp (284行) | ~20% |
| `src/loader/ldbc_loader.hpp` | ~520 | upstream dataset_preprocessor/*.{hpp,cpp} (657行) | ~20% |
| `src/loader/ldbc_driver.hpp` | ~430 | upstream wrapper/driver.h (1577行) | ~20% |
| `src/cost_model/tier_cost_model.hpp` | ~350 | NCCL allocator + DeepSpeed swapper patterns | ~20% |
| `src/algorithms/cross_tier_bfs.hpp` | ~420 | upstream BFS.h (330行) + tiered_bfs.hpp (388行) | ~20% |
| `src/algorithms/cross_tier_sssp.hpp` | ~420 | upstream SSSP.h (182行) | ~20% |
| `src/bench/ldbc_bench.cpp` | ~340 | 新写 | 100% |
| **合计** | **~2660** | | |

## 各模块设计决策

### M017: LDBC SNB Loader

**ldbc_types.hpp**:
- 从 upstream 保留: `operationType` enum (INSERT/DELETE/BFS/SSSP/PR/TC/WCC/GET_EDGE/...)
- 新增: `TEMPORAL_QUERY`, `CROSS_TIER_BFS`, `CROSS_TIER_SSSP` 操作类型
- 从 upstream 保留: `weightedEdge` → 改名 `TemporalEdge` (新增 `timestamp` 字段)
- 新增: `TierHint` enum (HBM/GDDR/DRAM/AUTO)
- 新增: `LDBCConfig` (hbm_capacity=80GB, gddr_capacity=48GB, dram_capacity=512GB, latency params)
- 新增: `DegreeStats` (max/avg/median/p90/p99/high/low counts)
- 每个 struct 都带 `dump()` 打印方法

**ldbc_loader.hpp**:
- 6-stage construction pipeline:
  1. loadEdges (含 timestamp 解析)
  2. removeDuplicateEdges (upstream 原样)
  3. randomShuffle (upstream 原样)
  4. computeDegreeDistribution (+ DegreeStats 采集)
  5. selectNodesByDegree (upstream 原样)
  6. **NEW**: computeTierPlacement + calibratePartitionThresholds
- Tier placement 策略: degree 排序 → top hot_fraction% 入 HBM, warm% 入 GDDR, 余下 DRAM
  - 边的 tier 继承两端点中较"热"的 tier
  - 时间戳加权: recent edges 升级一层
- 全量调试: ScopedTimer, per-stage 日志, validateLoad() 一致性断言

**ldbc_driver.hpp**:
- `TieredDriver<F,S>` 模板类
- `tiered_initialize()`: 多线程插入 + per-tier 计数 + RSS 追踪 + 进度打印
- `tiered_insert()`: stream-based batch insert + per-thread timing
- `tiered_query_bench()`: GET_EDGE/SCAN_NEIGHBOR/GET_VERTEX 微基准
- `execute_cross_tier_bfs/sssp()`: 独立 BFS/SSSP 入口 (简化版, 用于验证)
- `run_full_pipeline()`: 4-stage 编排: init → BFS → SSSP → state dump

### M018: Tier Cost Model

- `TierSpec`: 硬件规格 (latency_ns, bandwidth_gbps, capacity_bytes, migration_overhead_us)
  - 默认: H100 HBM (1ns/3352GB/s/80GB), A6000 GDDR (5ns/768GB/s/48GB), DDR5 (50ns/204.8GB/s/512GB)
- `access_cost_ns()`: latency + bytes/bandwidth
- `migration_cost_ns()`: setup + max(read, write) 模型 (cudaMemcpy 模式)
- `estimate_query_cost()`: 多分区代价, scan vs random access 模式
- `estimate_bfs_level_cost()`: 按 frontier tier 分布 + 跨层惩罚
- `optimal_tier_assignment()`: 贪心分配, 按访问频率排序, 依次填充 HBM → GDDR → DRAM
- `plan_migrations()`: 生成迁移计划 + 代价估算
- `prefetch_benefit_ns()`: 预取收益 = cold_cost - hot_cost - migrate_cost

### M019: Cross-Tier BFS

- 完整保留 upstream TDStep/BUStep direction-optimizing 核心
- 复用 src/algorithms/tiered_bfs.hpp 的 pvector/Bitmap/SlidingQueue
- 新增 `BFSLevelStats`: 每层追踪 frontier size, edges, tier 分布, estimated vs actual cost
- 新增 `prefetch_cold_neighbors()`: 大 frontier 时对冷层发 prefetch hint
- 新增 `collect_level_stats()`: 用 TierCostModel 估算每层代价
- 新增 `dump_convergence()`: 表格格式输出所有层数据 (可直接粘贴到论文)

### M020: Cross-Tier SSSP

- 完整保留 upstream delta-stepping 核心: compare_and_swap, local_bins, frontier 循环
- 新增 `SSSPIterStats`: 每次迭代的 bin_index, frontier_size, relaxations, edges, tier 分布
- 新增 `tier_adjusted_weight()`: DRAM 边权 +0.01, GDDR +0.001, HBM +0.0 (模拟真实延迟差)
- 新增 `prefetch_cold_frontier()`: 冷层 vertex 超过 25% 时触发预取
- 新增 `collect_iter_stats()`: 用 TierCostModel 估算每迭代代价
- 新增 `dump_convergence()`: 含 est/act 比值 (模型校准指标)

### Integration Bench (ldbc_bench.cpp)

5 个独立测试:
1. **LDBCLoader Construction**: 加载 + validate + state dump
2. **TierCostModel**: access cost ordering, migration, query estimation, optimal assignment, prefetch benefit
3. **Workload Generation**: .stream 文件生成验证
4. **Degree-Tier Cross-Check**: 验证 HBM fraction ≈ hot_fraction
5. **Cost Model Calibration**: 延迟比值范围检查, 1GB 传输时间合理性

可独立编译运行:
```bash
make ldbc_bench
./ldbc_bench                      # 用合成数据
./ldbc_bench /path/to/edges.txt   # 用真实数据
```

## 调试支持 (Debug Instrumentation)

每个新文件都包含:
- `ScopedTimer` on 所有主要函数
- `PHILE_DBG` 分级日志 (level 0-3)
- `PHILE_TRACE` 事件追踪 (TraceEvent 类型)
- 所有 struct 带 `dump()` 方法
- `validateLoad()` 做加载后一致性断言
- 收敛日志可直接用于论文图表数据

## 下一步 (Next Steps → Phase 4, M021-M022)

- M021: 从 upstream PR.h + WCC.h 移植跨层 PageRank + WCC
- M022: TC_opt.h 移植跨层 Triangle Counting
- M023-M024: 端到端管线集成 + 真实 LDBC SNB 数据评测
- 后续需要实际链接 NeoGraph library (upstream/rapidstore/libraries/NeoGraph/, 18920行)

## 代码风格

- namespace: `philemon::loader`, `philemon::cost_model`, `philemon::algorithms`
- Header-only: 所有 .hpp，内联实现
- 调试: 全量 printf (不依赖 iostream 格式化)
- upstream 代码原样保留段落用 `// from upstream, preserved` 标注
- 新增段落用 `// NEW:` 标注
