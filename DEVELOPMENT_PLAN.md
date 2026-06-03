# Philemon-TSH: Multi-Claude Development Plan

## System Architecture Overview

Philemon-TSH是一个分层异构内存(HBM/GDDR/DRAM)上的时序子图索引系统。
上游代码基础: upstream/rapidstore (31,272行) + upstream/temgraph。

## Claude Session Progress

| Claude # | Milestones | Status | Summary |
|----------|-----------|--------|---------|
| 第1位Claude | M001–M033 | ✅ 完成 | upstream全覆盖算法移植(31272行→62文件24881行) + 20%差异化 + 断点调试增强 + LiveGraph tiered适配 + temporal query驱动 + entry适配 |
| 第2位Claude | M034–M040 | ✅ 完成 | CMake编译系统 + GoogleTest框架 + 94个单元测试全绿 (ctest 94/94 passed) |
| 第3位Claude | M041–M047 | 🔲 待开始 | 多GPU拓扑感知调度 + CUDA流水线迁移 + HBM bandwidth profiler |
| 第4位Claude | M048–M054 | 🔲 待开始 | 自适应预取策略 + 查询代价估算器 + 动态tier rebalance |
| 第5位Claude | M055–M061 | 🔲 待开始 | 端到端benchmark矩阵(LDBC/LiveJournal/Twitter) + CI集成 + 性能回归检测 |
| 第6位Claude | M062–M068 | 🔲 待开始 | 论文数据生成流水线 + 图表自动化 + 论文写作 + 投稿准备 |

---

## Detailed Milestone Plan

### 第1位Claude: M001–M033 ✅ (已完成)
**upstream全量算法移植 + 系统骨架 + gap closure**

已完成的模块覆盖:

| Milestone | 对应upstream | 产出 src/ 文件 |
|-----------|------------|--------------|
| M001–M010 | core系统 (temporal_edge, slab_allocator, tiered_allocator, seqlock, tier_ptr, async_migrator, partition_index) | src/core/*.hpp (7文件) |
| M011–M016 | temgraph index + RapidStore bridge + wrapper | src/index/tem_graph*.hpp, src/bridge/*.hpp, src/wrapper/*.hpp, src/executor/*.hpp |
| M017–M020 | algorithms (BFS, SSSP, PR, WCC, TC) + LDBC loader + cost model | src/algorithms/*.hpp (10文件), src/loader/*.hpp, src/cost_model/*.hpp |
| M021–M026 | driver + debug + eviction + compaction + rebalance + prefetch | src/driver/*.hpp, src/debug/*.hpp, src/eviction/*.hpp, src/compaction/*.hpp, etc. |
| M027–M029 | types + utils + readers + config_parser + preprocessor | src/types/*.hpp, src/utils/*.hpp, src/readers/*.hpp, src/preprocessor/*.hpp |
| M030–M031 | 6个wrapper apps (3808行) → unified TieredBackendAdapter + **LiveGraph tiered adapter** | src/wrapper/apps/backend_adapters.hpp, **src/wrapper/apps/livegraph_tiered.hpp** (924行) |
| M032 | main.cpp + **driver_main.h** + **main_tem_graph.cpp** → entry points | src/entry/philemon_main.hpp, **src/entry/temporal_query_driver.hpp** (591行), **src/entry/driver_entry.hpp** (224行) |
| M033 | NeoGraph library (18920行) → tiered_index + internals | src/index/neograph_tiered_index.hpp, src/index/neograph_internals.hpp |

关键算法改动 (20%差异化):
- edges()遍历: 单级引擎iterate → 多级TierRouter路由 + per-tier统计
- insert_edge: 单引擎直插 → TierRouter.route_by_hash/degree分级插入
- batch_update: 逐条顺序 → 按tier分桶(bucket)批量提交
- intersect: 各引擎各异(marker-chase/binary-search) → 全局排序归并
- NeoTree版本链: +VersionTier标记, ART升级阈值运行时可配
- RangeTree: COW segment分配 → sorted vector
- ART: COW path copy → in-place + 节点类型自动升级
- Config: boost::program_options → 手动key=value INI解析
- 6个Backend Adapter(NeoGraph/CSR/LiveGraph/Aspen/Sortledton/Teseo) → 统一TieredAdjacencyStore
- **LiveGraph: tbb::concurrent_hash_map → unordered_map+shared_mutex; lg::Transaction → TierRouter; batch_loader → bucket-by-tier flush**
- **Temporal Query: GetTime() → QueryPhaseTimer(p50/p95/p99); warmup phase; per-query tier-hit counters**
- **Driver Entry: Intel VTune __itt_pause删除; hardcoded config → CLI args; --dump-config/--dry-run**

### 第2位Claude: M034–M040 ✅ (已完成)
**编译系统 + 单元测试框架**

| Milestone | 内容 |
|-----------|------|
| M034 | CMakeLists.txt构建系统: 编译全部62个hpp为可执行测试 |
| M035 | Google Test集成 + test/目录骨架 |
| M036 | core模块UT: slab_allocator, tiered_allocator, seqlock, tier_ptr |
| M037 | index模块UT: tem_graph, neograph_tiered_index, neograph_internals |
| M038 | algorithms模块UT: BFS/SSSP/PR/WCC/TC正确性验证 |
| M039 | wrapper模块UT: backend_adapters, **livegraph_tiered**, readers, config_parser |
| M040 | 集成测试: config加载→建图→运行算法→验证结果的全流程 (含**temporal_query_driver**和**driver_entry**) |

### 第3位Claude: M041–M047 (待开始)
**GPU拓扑感知 + CUDA迁移**

| Milestone | 内容 |
|-----------|------|
| M041 | CUDA内存管理: cudaMalloc/cudaMemcpy封装为TierPtr<HBM> |
| M042 | GPU拓扑探测: nvidia-smi拓扑解析 + NVLink/PCIe带宽建模 |
| M043 | CUDA kernel: 并行BFS (warp-level edge scan) |
| M044 | CUDA kernel: 并行PageRank (shared-memory reduction) |
| M045 | 异步数据迁移流水线: CUDA streams + host-to-device overlap |
| M046 | Multi-GPU分区: 按vertex range分配到不同GPU的HBM |
| M047 | GPU性能profiler: kernel时间 + HBM带宽 + PCIe传输量 |

### 第4位Claude: M048–M054 (待开始)
**自适应策略 + 代价估算**

| Milestone | 内容 |
|-----------|------|
| M048 | 查询代价估算器: 基于degree分布预测BFS/SSSP层数和边扫描量 |
| M049 | 自适应预取: 根据历史访问模式动态调整prefetch depth |
| M050 | 热度追踪器: LFU/LRU混合策略,按时间窗口衰减 |
| M051 | 动态tier rebalance: 定期(每N次insert后)重新分配HBM/GDDR/DRAM |
| M052 | 在线代价学习: 用运行时统计修正代价模型参数 |
| M053 | 内存压力检测: /proc/meminfo + HBM利用率 → 触发eviction |
| M054 | 自适应batch size: 根据insert throughput动态调整batch大小 |

### 第5位Claude: M055–M061 (待开始)
**端到端Benchmark + CI**

| Milestone | 内容 |
|-----------|------|
| M055 | LDBC SNB数据集集成: SF1/SF10/SF100 数据加载 |
| M056 | LiveJournal/Twitter/UK-2002图数据集支持 |
| M057 | Benchmark矩阵: 6数据集 × 5算法 × 3tier配置 |
| M058 | 对比基准: 与upstream RapidStore的性能对比 |
| M059 | GitHub Actions CI: 自动编译 + UT + 基准测试回归 |
| M060 | 性能回归检测: 每次commit后与baseline对比,超过5%告警 |
| M061 | 可视化dashboard: 性能趋势图 + tier利用率热力图 |

### 第6位Claude: M062–M068 (待开始)
**论文 + 投稿准备**

| Milestone | 内容 |
|-----------|------|
| M062 | 实验数据收集: 运行完整benchmark矩阵,收集CSV |
| M063 | 图表生成: matplotlib/pgfplots自动化 (throughput, latency, tier分布) |
| M064 | 论文Section 1-2: Introduction + System Design |
| M065 | 论文Section 3-4: Algorithm + Evaluation |
| M066 | 论文Section 5-6: Related Work + Conclusion |
| M067 | 数据一致性校验: 论文中每个数字都能追溯到benchmark输出 |
| M068 | 投稿准备: camera-ready格式 + supplementary material |

---

## File Statistics

| 指标 | 数值 |
|------|------|
| upstream总行数 | 31,272 |
| upstream文件数 | 121 |
| src/文件数 | 62 |
| src/总行数 | 24,881 |
| 覆盖率 | 100% (121/121 upstream files) |
| 算法差异化 | ~20% per file |

## Changelog

| 日期 | Claude # | 变更 |
|------|----------|------|
| 初始 | 第1位 (Session 1-3) | M001–M029: core/index/bridge/wrapper/algorithms/debug/driver/entry 共52文件20031行 |
| 初始 | 第1位 (Session 4) | M030–M031: NeoGraph internals 2文件 |
| 2026-06-02 | 第1位 (Session 5) | M030–M033 gap closure: livegraph_tiered.hpp(924行) + temporal_query_driver.hpp(591行) + driver_entry.hpp(224行) + PHILE_BREAKPOINT_NAMED宏 → 62文件24881行, 121/121 upstream全覆盖 |
| 2026-06-03 | 第2位 | M034–M040: CMakeLists.txt + GoogleTest v1.14.0 + test/{test_core,test_index,test_algorithms,test_wrapper,test_integration}.cpp → 94/94 tests passed. 源码修复: temporal_query_driver.hpp PHILE_LG_TRACE_FMT宏前移 |
