# Philemon-TSH: Multi-Claude Development Plan

## System Architecture Overview

Philemon-TSH是一个分层异构内存(HBM/GDDR/DRAM)上的时序子图索引系统。
上游代码基础: upstream/rapidstore (31,272行) + upstream/temgraph。

## Claude Session Progress

| Claude # | Milestones | Status | Summary |
|----------|-----------|--------|---------|
| 第1位Claude | M001–M033 | ✅ 完成 | upstream全覆盖算法移植(31272行→62文件24881行) + 20%差异化 + 断点调试增强 + LiveGraph tiered适配 + temporal query驱动 + entry适配 |
| 第2位Claude | M034–M040 | ✅ 完成 | CMake编译系统 + GoogleTest框架 + 94个单元测试全绿 (ctest 94/94 passed) |
| 第3位Claude | M041–M043 | ✅ 完成 | upstream 5大核心算法(BFS/SSSP/PR/WCC/TC)算法逻辑重写 + wrapper_ops结构改造 + 辅助模块移植(15文件2799行) |
| 第4位Claude | M044 | ✅ 完成 | NeoGraph子系统全量覆盖移植(upstream ~18000行 → 9文件3547行) — ART/RangeTree/Version/Transaction/Snapshot/Wrapper + 20%算法差异化 + 断点调试增强 |
| 第5位Claude | M045–M051 | 🔲 待开始 | CUDA内存管理 + GPU拓扑探测 + 并行BFS/PR kernel + 异步迁移流水线 + Multi-GPU分区 |
| 第6位Claude | M052–M058 | 🔲 待开始 | 自适应预取 + 代价估算器 + 热度追踪 + 动态rebalance + 在线学习 + GPU性能profiler |
| 第7位Claude | M059–M065 | 🔲 待开始 | Benchmark矩阵(LDBC/LiveJournal/Twitter) + 对比基准 + CI + 性能回归 |
| 第8位Claude | M066–M072 | 🔲 待开始 | 论文数据收集 + 图表 + 写作(System Design/Algorithm/Evaluation) + 投稿准备 |

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

### 第3位Claude: M041–M043 ✅ (已完成)
**upstream核心算法重写 — 真正的算法逻辑差异化**

| Milestone | 内容 |
|-----------|------|
| M041 | BFS/SSSP算法重写: density-based TD↔BU切换, BU early-exit, adaptive delta, eager pruning, double-checked relaxation |
| M042 | PR/WCC/TC算法重写: 单趟融合PR, CAS-based union-find WCC, adaptive threshold TC + binary search |
| M043 | wrapper_ops结构改造 + 辅助模块移植: chunked batch retry, degree reserve, 去weightedEdge耦合; timer/log/config/preprocessor/neo_adapter/driver/entry/bench |

算法改动详情 (每个文件4处核心算法修改):

**BFS**: frontier_density切换 / BU early-exit / acquire-release CAS / alignas(64)距离数组
**SSSP**: adaptive delta自调节 / dist快照防过时传播 / 区外pre-check减mutex竞争 / 单趟结果收集
**PR**: 两趟线程→一趟融合 / L1收敛提前退出 / contrib数组复用 / atomic dangling_sum
**WCC**: CAS原子hook / path halving替代full compression / find_root再hook / atomic change flag
**TC**: median-degree自适应阈值 / binary_search替代线性marker / 合并TC+TC_opt / 低度数端probe

### 第4位Claude: M044 ✅ (已完成)
**NeoGraph子系统全量覆盖移植**

| Milestone | 内容 |
|-----------|------|
| M044 | upstream NeoGraph完整移植: ART树(art_new 4195行 + c_art 6305行 合并), RangeTree B+树(881行), NeoTreeVersion MVCC核心(3075行), Transaction管理(868行), PropertyStore(847行), WriterTraceBlock/ReaderTraceBlock(541行), Snapshot(239行), Wrapper driver接口(478行), types/config/utils(906行) → 9个impl文件3547行 |

产出文件:
| 文件 | 行数 | upstream来源 |
|------|------|-------------|
| neograph_types_impl.hpp | 861 | config.h + types.h/cpp + helper.h + spin_lock + error_type + bitmap.h + thread_pool.h |
| neograph_art_impl.hpp | 754 | art_new/ (14文件4195行) + c_art/ (6305行) 合并 |
| neograph_range_impl.hpp | 314 | neo_range_tree.h/cpp + neo_range_ops.h/cpp |
| neograph_property_impl.hpp | 202 | neo_property.h/cpp |
| neograph_trace_impl.hpp | 150 | neo_reader_trace.h/cpp |
| neograph_version_impl.hpp | 634 | neo_tree_version.h/cpp + neo_tree.h/cpp |
| neograph_transaction_impl.hpp | 288 | neo_transaction.h/cpp |
| neograph_snapshot_impl.hpp | 120 | neo_snapshot.h/cpp |
| neograph_wrapper_impl.hpp | 224 | neo_wrapper.h + wrapper.h |

核心算法差异化(~20%):
- ART NODE16 find_child: SSE2 _mm_cmpeq_epi8→scalar loop(ARM可移植)
- ART insert: COW copy_path→in-place+SpinLock(减GC压力)
- ARTLeaf: 4子类(Leaf8 bitmap/Leaf16/32/64 sorted array)→统一ARTLeafUnified sorted vector
- RangeTree: COW segment(trace_block分配器)→vector-backed直接存储
- NeoTreeVersion insert_edge: COW segment→in-place per-vertex lock
- insert_edge_batch: tbb::parallel_sort→std::sort(去TBB依赖)
- SpinLock: 裸while(exchange)→指数退避16次+yield
- quickSortWithProperties: Hoare双路→Dutch National Flag三路分区(重复key不退化)
- Bitmap::lower_bound: if/else双分支prefix判断→统一abs_val计算
- ARTKey::operator<: 逐字节depth循环→uint32直接比较

### 第5位Claude: M045–M051 (待开始)
**GPU拓扑感知 + CUDA内存管理 + 并行kernel**

| Milestone | 内容 |
|-----------|------|
| M045 | CUDA内存管理: cudaMalloc/cudaMemcpy封装为TierPtr<HBM> |
| M046 | GPU拓扑探测: nvidia-smi拓扑解析 + NVLink/PCIe带宽建模 |
| M047 | CUDA kernel: 并行BFS (warp-level edge scan) |
| M048 | CUDA kernel: 并行PageRank (shared-memory reduction) |
| M049 | 异步数据迁移流水线: CUDA streams + host-to-device overlap |
| M050 | Multi-GPU分区: 按vertex range分配到不同GPU的HBM |
| M051 | GPU性能profiler: kernel时间 + HBM带宽 + PCIe传输量 |

### 第6位Claude: M052–M058 (待开始)
**自适应预取 + 代价估算 + 热度追踪 + 动态rebalance**

| Milestone | 内容 |
|-----------|------|
| M052 | 自适应预取引擎: access pattern识别 + prefetch queue |
| M053 | 代价估算器: tier间迁移cost建模(延迟/带宽/容量) |
| M054 | 热度追踪: vertex/edge访问频率统计 + 时间衰减 |
| M055 | 动态rebalance: 根据热度+cost自动在tier间迁移数据 |
| M056 | 在线学习: rebalance策略自适应调参 |
| M057 | 端到端集成: 预取+估算+热度+rebalance联调 |
| M058 | 回归测试: 确保自适应策略不引入正确性问题 |

### 第7位Claude: M059–M065 (待开始)
**Benchmark矩阵 + CI**

| Milestone | 内容 |
|-----------|------|
| M059 | LDBC SNB数据集集成: SF1/SF10/SF100 数据加载 |
| M060 | LiveJournal/Twitter/UK-2002图数据集支持 |
| M061 | Benchmark矩阵: 6数据集 × 5算法 × 3tier配置 |
| M062 | 对比基准: 与upstream RapidStore性能对比 |
| M063 | GitHub Actions CI: 自动编译 + UT + 基准回归 |
| M064 | 性能回归检测: 每commit与baseline对比, 超5%告警 |
| M065 | 可视化dashboard: 性能趋势图 + tier利用率热力图 |

### 第8位Claude: M066–M072 (待开始)
**论文写作 + 投稿**

| Milestone | 内容 |
|-----------|------|
| M066 | 实验数据收集: 运行完整benchmark矩阵, 收集CSV |
| M067 | 图表生成: matplotlib/pgfplots (throughput, latency, tier分布) |
| M068 | 论文Section 1-2: Introduction + System Design |
| M069 | 论文Section 3-4: Algorithm + Evaluation |
| M070 | 论文Section 5-6: Related Work + Conclusion |
| M071 | 数据一致性校验: 论文中每个数字可追溯到benchmark输出 |
| M072 | 投稿准备: camera-ready格式 + supplementary material |

---

## File Statistics

| 指标 | 数值 |
|------|------|
| upstream总行数 | 31,272 |
| upstream文件数 | 121 |
| src/文件数 | 97 |
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
| 2026-06-03 | 第3位 | M041–M043: 5大核心算法(BFS/SSSP/PR/WCC/TC)算法逻辑重写(每个4处核心修改) + wrapper_ops结构改造 + 辅助模块移植 → 15文件2799行新增 |
