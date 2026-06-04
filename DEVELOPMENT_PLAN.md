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
| 第5位Claude | M045–M051 | ✅ 完成 | CUDA内存管理(瀑布分配+slab池化) + GPU拓扑(Dijkstra路由+NUMA) + 并行BFS(warp-level+ballot) + PR(shared-mem reduce+L1收敛) + 异步迁移(双stream+优先级+限流) + Multi-GPU分区(加权放置+ghost vertex+rebalance) + GPU profiler(分层计时+滑动窗口+瓶颈检测) → 7文件3554行 |
| 第6位Claude | M052–M058 | ✅ 完成 | 自适应预取(stride+frequency双模式+EWMA+Count-Min) + 代价估算(Roofline+AMAT三层) + 热度追踪(分桶CLOCK+指数衰减+16-shard) + 动态rebalance(梯度下降+共现亲和+bandwidth×urgency) + 在线学习(Thompson Sampling+UCB1) + pipeline编排(Kahn拓扑排序+自适应worker) + 回归测试(shadow-run+Welch t-test) → 7文件4224行 |
| 第7位Claude | M059–M064 | ✅ 完成 | 6个独立wrapper adapter全量移植(upstream 3808行→6文件3860行): CSR(分段CSR+galloping+3路归并) + Sortledton(skip-sentinel+zigzag join+epoch事务) + Teseo(adaptive intersect+RCU shadow-swap+optimistic retry) + Aspen(B+树叶链直扫+fractional cascading+CompactBloom+16桶batch) + LiveGraph(OpenAddrHashSet交集+batch-gather+WAL merge+degree cache) + Neo(delta-compressed+4-way unrolled merge+partition-sort-batch) + 编译验证6/6 PASS |
| 第8位Claude | M065–M070 | ✅ 已完成 | Benchmark矩阵(LDBC/LiveJournal/Twitter) + 对比基准(RapidStore/Teseo/Sortledton原版) + CI pipeline + 性能回归检测 + 端到端集成测试 + cuckoo index gap close + dataset loader |
| 第8位Claude(续) | M071–M072 | ✅ 已完成 | NeoGraph子系统移植(19文件/5529行) + upstream算法核心20%重写(9文件22个算法级改动点: tier编码/跨tier惩罚/采样剪枝/adaptive delta/分层L1收敛/三步jump/稠密段二分/自适应跳步/加权去重/IQR阈值/power-law/work-stealing) |
| 第9位Claude | M073–M078 | 🔲 待开始 | 论文实验数据收集 + 图表生成(matplotlib/pgfplots) + 统计显著性分析 + 数据一致性校验 + 实验可复现脚本 + supplementary material |
| 第10位Claude | M079–M084 | 🔲 待开始 | 论文写作: Introduction + System Design + Algorithm + Evaluation + Related Work + Conclusion |
| 第11位Claude | M085–M088 | 🔲 待开始 | 代码优化: SIMD向量化 + memory pool tuning + lock-free数据结构 + profile-guided优化 + 跨平台适配(ARM/RISC-V) + 文档生成(Doxygen) |
| 第12位Claude | M089–M094 | 🔲 待开始 | 投稿准备: camera-ready + rebuttal模板 + artifact evaluation + 开源release + README国际化 + CI/CD badge |

### 当前开发状态 (按新一轮Claude编号)

```
第一位Claude 已完成: M065–M072
  → libcuckoo gap + dataset loader + benchmark matrix + comparison baseline
  → CI pipeline + regression detector + E2E integration test
  → NeoGraph子系统移植(19文件/5529行)
  → upstream算法核心20%重写: 9文件22个算法级改动点
    (tier编码/跨tier惩罚/采样剪枝/adaptive delta/tier延迟加权/
     tier温度boost/分层L1收敛/边权过滤/三步jump/稠密段二分/
     自适应跳步/加权去重/IQR阈值/power-law拟合/确定性hash权重/
     degree加权采样/work-stealing)

第二位Claude 待开始: M073–M078
  → 论文实验数据收集 + 图表生成(matplotlib/pgfplots)
  → 统计显著性分析(Welch t-test/bootstrap CI)
  → 数据一致性校验 + 实验可复现脚本 + supplementary material

第三位Claude 待开始: M079–M084
  → 论文写作: Introduction + System Design + Algorithm
  → Evaluation + Related Work + Conclusion
  → 6 sections全文, Springer Nature sn-jnl格式

第四位Claude 待开始: M085–M088
  → 代码优化: SIMD向量化(BFS/WCC frontier ops)
  → memory pool tuning + lock-free数据结构
  → profile-guided优化 + 跨平台适配(ARM/RISC-V)

第五位Claude 待开始: M089–M091
  → 投稿准备: camera-ready格式化 + rebuttal模板预写
  → artifact evaluation badge + 实验复现Docker

第六位Claude 待开始: M092–M094
  → 开源release: LICENSE(Apache 2.0) + CI badge
  → README中英双语 + v1.0 tag + Zenodo DOI
```

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

### 第5位Claude: M045–M051 ✅ (已完成)
**GPU拓扑感知 + CUDA内存管理 + 并行kernel + 异步迁移 + 多GPU分区 + profiler**

| Milestone | 内容 |
|-----------|------|
| M045 | CUDA内存管理: 瀑布回退分配(HBM→GDDR→DRAM) + slab池化(≤4MB) + 智能copy路径选择(peer/staged/chunked) + 碎片回收 |
| M046 | GPU拓扑探测: 运行时bandwidth micro-benchmark + 带权有向拓扑图 + Dijkstra最优路由 + NUMA sysfs解析 |
| M047 | CUDA BFS kernel: warp-level邻接表并行扫描 + ballot投票BU + warp-aggregated frontier push + 动态密度切换 |
| M048 | CUDA PageRank kernel: 单趟融合dangling+contrib + shared-mem block-reduce + L1-norm收敛退出 + warp-shuffle规约 |
| M049 | 异步迁移pipeline: CUDA双stream D2H/H2D overlap + access_count×recency优先级队列 + 读写分离 + 令牌桶PCIe限流 |
| M050 | Multi-GPU分区: 容量×带宽加权贪心放置 + GPU权重比例分配 + ghost vertex bitmap + 运行时skew检测rebalance |
| M051 | GPU profiler: 三级分层(kernel/phase/tier) + 滑动窗口p50/p95/p99 + 自动热点+PCIe饱和预警 + JSON输出 |

产出文件:
| 文件 | 行数 | upstream来源 | 算法改动 | 调试断点 |
|------|------|-------------|---------|---------|
| cuda_mem_manager.hpp | 687 | hetero_bench.cu HeteroAllocator (134-268行) + tiered_allocator.hpp | 18 | 21 |
| gpu_topology.hpp | 492 | hetero_bench.cu constructor+experiment_bandwidth (134-357行) + tier_cost_model.hpp | 14 | 13 |
| cuda_bfs_kernel.hpp | 469 | BFS.h TDStep/BUStep/bfs (330行) + BFS.cpp (302行) | 14 | 11 |
| cuda_pagerank_kernel.hpp | 371 | PR.h page_rank (174行) + pageRank.cpp (159行) | 13 | 9 |
| cuda_async_migration.hpp | 610 | async_migrator.hpp (430行) + hetero_bench.cu E4/E5 (666-859行) | 15 | 12 |
| cuda_multi_gpu_partition.hpp | 482 | hetero_bench.cu partition_and_place+scaling (377-968行) + partition_index.hpp | 12 | 11 |
| cuda_gpu_profiler.hpp | 443 | hetero_bench.cu CudaTimer+experiments (269-968行) + state_inspector.hpp | 11 | 7 |
| **合计** | **3554** | | **97** | **84** |

核心算法差异化(~20%):
- **M045 allocate**: 固定tier→瀑布回退(HBM→GDDR0→GDDR1→DRAM+watermark); cudaMemcpyDefault→peer-direct/staged-via-host/chunked pipeline; per-call cudaMalloc→64MB slab池bump分配; 无compaction→watermark<25%回收
- **M046 topology**: 硬编码BW常量→runtime micro-benchmark 7次取中位数; bool[][] →带权有向图adj[i][j]=GB/s; if/else路由→Dijkstra全对最短路; 忽略NUMA→/sys/bus/pci解析
- **M047 BFS**: OMP thread-level→warp-level 32线程stride-scan邻接表+atomicCAS; per-vertex BU→__ballot_sync投票减分歧; SlidingQueue→atomicAdd warp-leader预留slot; 固定alpha/beta→0.03*sqrt(V/E)自适应
- **M048 PR**: 两趟dangling+contrib→单趟融合warp-reduce; 独立per-thread→shared-mem block-reduce; 固定迭代→L1<epsilon(1e-6)提前退出; per-thread sum数组→warp-shuffle+block-shared+atomicAdd
- **M049 migration**: memcpy阻塞→CUDA双stream event-based overlap; FIFO→access_count*exp(-0.1*age)优先级队列; exclusive全程→shared_lock+swap时exclusive; 无限制→令牌桶PCIe 80%预算
- **M050 partition**: 固定25%切分→capacity*bandwidth加权贪心; 均匀分配→GPU权重比例; 无ghost→cross-partition vertex bitmap; 无rebalance→skew>2.0触发迁移
- **M051 profiler**: CudaEvent pair→kernel/phase/tier三级分层; 单次平均→64-sample滑动窗口+p50/p95/p99; 人工读→自动top-3热点+PCIe饱和预警; printf→JSON+text双输出

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
| M059 | CSR独立adapter: 分段CSR(HBM/GDDR/DRAM) + galloping search + 3路归并交集 |
| M060 | Sortledton独立adapter: skip-sentinel索引 + zigzag join + epoch分组事务 |
| M061 | Teseo独立adapter: adaptive set intersection + RCU shadow-swap + optimistic retry |
| M062 | Aspen独立adapter: B+树叶链直扫 + fractional cascading + CompactBloom + 16桶batch |
| M063 | LiveGraph独立adapter: OpenAddrHashSet交集 + batch-gather + WAL merge + degree cache |
| M064 | Neo独立adapter: delta-compressed遍历 + 4-way unrolled merge + partition-sort-batch |

### 第8位Claude: M065–M070 (✅ 已完成)
**Benchmark矩阵 + CI + 端到端测试**

| Milestone | 内容 | 状态 |
|-----------|------|------|
| M065 | libcuckoo gap close (cuckoo_bucket/map_impl.hpp) + dataset_loader.hpp (auto-detect+mmap+FNV-1a renumber) | ✅ |
| M066 | Benchmark矩阵: 6数据集 × 5算法 × 3tier配置 × 6 adapter, Welch t-test + JSON/Markdown export | ✅ |
| M067 | 对比基准: Philemon vs 5竞品, 4维profile(lat/thr/mem/scal), 归一化+A/B Welch t-test | ✅ |
| M068 | GitHub Actions CI: Debug/Release × GCC-13/Clang-17 4-way, ccache, ASan, lcov coverage | ✅ |
| M069 | 性能回归检测: Welch t-test + Page's CUSUM变点检测, 三级告警, 移动平均平滑, 历史趋势 | ✅ |
| M070 | 端到端集成测试: planted partition + AdapterFactory bitmap dispatch + BFS/WCC/TC交叉验证 (12 tests) | ✅ |

### 第9位Claude: M071–M076 (待开始)
**论文实验数据 + 图表**

| Milestone | 内容 |
|-----------|------|
| M071 | 实验数据收集: 运行完整benchmark矩阵, 收集CSV |
| M072 | 图表生成: matplotlib/pgfplots (throughput, latency, tier分布) |
| M073 | 统计显著性分析: 置信区间 + p-value + 多组比较校正 |
| M074 | 数据一致性校验: 论文中每个数字可追溯到benchmark输出 |
| M075 | 实验可复现脚本: one-click reproduce from scratch |
| M076 | Supplementary material: 附加图表 + 完整参数表 |

### 第10位Claude: M077–M082 (待开始)
**论文写作**

| Milestone | 内容 |
|-----------|------|
| M077 | 论文Section 1: Introduction + Motivation |
| M078 | 论文Section 2: System Design (Philemon-TSH架构) |
| M079 | 论文Section 3: Algorithm (6个adapter的算法改进) |
| M080 | 论文Section 4: Evaluation (实验结果分析) |
| M081 | 论文Section 5: Related Work |
| M082 | 论文Section 6: Conclusion + Future Work |

### 第11位Claude: M083–M088 (待开始)
**代码优化 + 文档**

| Milestone | 内容 |
|-----------|------|
| M083 | SIMD向量化: SSE/AVX2加速intersect/edges |
| M084 | Memory pool tuning: slab/arena allocator for adapter内部结构 |
| M085 | Lock-free数据结构: 无锁degree cache + 无锁bloom filter |
| M086 | Profile-guided优化: PGO编译 + 热点分析 |
| M087 | 跨平台适配: ARM NEON + RISC-V Vector |
| M088 | 文档生成: Doxygen + API reference + architecture diagram |

### 第12位Claude: M089–M094 (待开始)
**投稿准备 + 开源**

| Milestone | 内容 |
|-----------|------|
| M089 | Camera-ready格式化: ACM/IEEE template |
| M090 | Rebuttal模板: 常见reviewer质疑预写 |
| M091 | Artifact evaluation: 可复现badge申请 |
| M092 | 开源release: LICENSE + CONTRIBUTING + CI badge |
| M093 | README国际化: 中英双语文档 |
| M094 | 项目归档: tag v1.0 + DOI注册 |

---

## File Statistics

| 指标 | 数值 |
|------|------|
| upstream总行数 | 31,272 |
| upstream文件数 | 121 |
| src/文件数 | 122 |
| src/总行数 | ~48,785 |
| test/文件数 | 7 |
| test/总行数 | ~2,901 |
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
| 2026-06-03 | 第4位 | M044: NeoGraph子系统全量覆盖移植(upstream ~18000行 → 9文件3547行) — ART/RangeTree/Version/Transaction/Snapshot/Wrapper + 20%算法差异化 + 断点调试增强 |
| 2026-06-03 | 第5位 | M045–M051: CUDA GPU子系统 — 内存管理(瀑布+slab) + 拓扑(Dijkstra+NUMA) + 并行BFS(warp+ballot) + PR(shared-mem+L1) + 异步迁移(stream pipeline+优先级+限流) + Multi-GPU分区(加权+ghost+rebalance) + profiler(分层+滑动窗口+瓶颈) → 7文件3554行, 97个[ALG]算法改动标记 + 84个PHILE_调试断点 |
| 2026-06-03 | 第6位 | M052–M058: 自适应预取+代价估算+热度追踪+动态rebalance+在线学习+pipeline编排+回归测试 → 7文件4224行 |
| 2026-06-03 | 第7位 | M059–M064: 6个独立wrapper adapter(CSR/Sortledton/Teseo/Aspen/LiveGraph/Neo) — upstream 3808行→6文件3860行, 每个adapter 4处核心算法改动 + 12-14处断点调试宏 + self_test, 编译验证6/6 PASS |
| 2026-06-04 | 第8位(新一轮第1位) | M065–M070: libcuckoo gap close(cuckoo_bucket/map_impl 1322行) + dataset_loader(618行) + benchmark_matrix(674行) + comparison_baseline(454行) + CI pipeline(262行) + regression_detector(500行) + E2E test(1047行) → 8文件4877行新增 |
