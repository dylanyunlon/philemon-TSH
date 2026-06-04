# Philemon-TSH 开发进度计划

## 里程碑总览

| Milestone | 状态 | 内容 |
|-----------|------|------|
| M001-M070 | ✅ 已完成 | 基础框架 (123文件/49824行, 之前开发者完成) |
| M071 | ✅ 完成 | NeoGraph核心子系统移植 (19文件/5529行) |
| M072 | ✅ 完成 | NeoGraph M072 patch (合入M071) |
| **M073** | **✅ 第1位Claude完成** | **ART完整实现 + art_new compat + NeoGraph gaps (7新文件/4715行)** |
| M074-M076 | 🔜 第2位Claude | Driver补全 + NeoGraph property完善 (~1550行) |
| M077-M079 | 🔜 第3位Claude | 跨模块集成头文件 + CMake + 测试桩 (~900行) |
| M080-M082 | 🔜 第4位Claude | 性能基准 Benchmark (~1600行) |
| M083-M085 | 🔜 第5位Claude | 文档 + 代码质量 (~1300行) |
| M086-M088 | 🔜 第6位Claude | 最终集成 + 发布准备 (~300行) |

---

## 第一位Claude (已完成): M071

**交付物**: `src/neograph/` — 19文件, 5529行

```
src/neograph/
├── include/    neo_property.hpp (合并360+487行), neo_types.hpp (合并242+146行)
├── utils/      neo_config.hpp, neo_error_type.hpp, neo_helper.hpp,
│               neo_spin_lock.hpp, neo_thread_pool.hpp
├── art/        art_core.hpp, art_ops.hpp, art_leaf_impl.hpp, art_impl.hpp
├── bitmap/     neo_bitmap.hpp
└── core/       neo_tree.hpp, neo_tree_version.hpp, neo_tree_version_impl.hpp,
                neo_range_tree.hpp, neo_index.hpp, neo_transaction.hpp,
                neo_snapshot.hpp
```

**修改手法**: 鲁迅式~20%算法修改
- atomic计数器 on every hot path (insert/remove/search/GC)
- 深度直方图 (ART search depth, version chain walk)
- 分支命中率追踪 (3-level storage dispatch)
- 带宽统计 (copy_bytes_crossed)
- 延迟采样 (creation_wall_ns, reader lifetime)
- 后置断言 (sorted-order, ref_cnt underflow)

---

## 第二位Claude (实际由第9位Claude完成): M072-M075 → M073

**任务**: NeoGraph ART完整实现 + art_new兼容层 + NeoGraph gap文件
**状态**: ✅ M073已完成 (合并了原M072-M075范围 + 额外gap文件)

| Milestone | 来源文件 | 目标 | 行数 | 状态 |
|-----------|---------|------|------|------|
| M073 | c_art node_ops.cpp(2080) + node_ops_copy.cpp(1081) + art_iter.cpp(179) + art_node_iter.cpp(442) + art_new全部15文件 + neo_range_ops(125) + neo_reader_trace(541) + neo_wrapper(181) + wrapper.h(297) | 见下方 | 4715行 | ✅ |

**实际产出** (8 files, 4715 new lines):
- `src/neograph/art/art_node_ops_impl.hpp` (1792行) — ART节点操作完整实现
- `src/neograph/art/art_iter_impl.hpp` (666行) — ART迭代器 (node+tree level)
- `src/neograph/art/art_node_ops_copy_impl.hpp` (951行) — ART COW copy操作
- `src/neograph/art_compat/art_compat.hpp` (490行) — art_new兼容shim层
- `src/neograph/core/neo_range_ops.hpp` (188行) — Range segment操作
- `src/neograph/core/neo_reader_trace.hpp` (411行) — Reader/Writer跟踪+内存池
- `src/neograph/core/neo_wrapper.hpp` (217行) — NeoGraph外部接口封装
- `src/neograph/art/art_ops.hpp` (modified) — 补充LEAF_POINTER_CTOR/SET_OFFSET宏

**算法改动 (28处)**:
- art_node_ops_impl: SSE prefetch, branchless comparison, bitmap batch, galloping search, probe mode, right-skewed split
- art_iter_impl: CTZ O(1) next_without_skip, path cache, IteratorStats
- art_node_ops_copy_impl: generation tag COW, batch_insert_copy fast path
- art_compat: two-pass cleanup, batch insert accumulator, copy_path shortcut
- neo_range_ops: SIMD 4-wide find, adaptive split point
- neo_reader_trace: exponential backoff spinlock, min_timestamp cache, pool stats
- neo_wrapper: batch edge accumulator, callback get_neighbors, degree LRU-1 cache

---

---

# 后续开发进度规划 (第2-6位Claude)

根据第一位Claude完成M073后的全面盘点，upstream文件迁移覆盖率已很高。
主要缺口是**Driver补全**(~800行) 和一些胶水集成工作。

## 迁移覆盖状态

| 模块 | Upstream行数 | src/已有行数 | 覆盖 |
|------|-------------|-------------|------|
| NeoGraph ART (c_art) | 6,305 | 5,506 (art+art_compat) | ✅ |
| NeoGraph ART (art_new) | 4,195 | (merged into art_compat) | ✅ |
| NeoGraph core (9 .cpp+11 .h) | 7,590 | 4,753 (core+include+utils+bitmap) + 5,506 (art) | ✅ |
| NeoGraph utils | 881 | 561 (utils/) + 228 (bitmap/) | ✅ |
| Algorithms | 961 | 5,475 | ✅ |
| Wrapper Algorithms | 1,009 | 1,053 | ✅ |
| Graph IO (edge/edgeStream) | 179 | 524 | ✅ |
| Readers | 248 | 2,189 (readers+loader) | ✅ |
| Utils (Timer/Log/CLI) | 960 | 817 | ✅ |
| Types | 150 | 321 | ✅ |
| Dataset Preprocessor | 1,168 | 902 | ✅ |
| TemGraph | 810 | 1,175 | ✅ |
| Main | 337 | 1,134 (entry/) | ✅ |
| Wrapper (NeoGraph+rapidstore) | 727 | 817 (wrappers) | ✅ |
| **Driver** | **1,577** | **947** | **⚠️ 缺~800行** |

## 第2位Claude: M074-M076 — Driver补全 + NeoGraph property完善

| Milestone | 任务 | 来源 | 预计行数 |
|-----------|------|------|---------|
| M074 | driver.h缺失函数: execute_qos, execute_concurrent, execute_insert_real_ldbc, execute_batch_insert, execute_update, initialize_graph | upstream driver.h 残余~800行 | ~900行 |
| M075 | driver.h图算法委托: bfs/sssp/wcc/page_rank snapshot委托, 性能计数集成 | upstream driver.h (724-890行) | ~400行 |
| M076 | neo_property.cpp 完善 (upstream 487行, src 329行偏简), edge property map ops | upstream neo_property.cpp | ~250行 |

修改重点: benchmark timing μs精度, warmup检测, QoS latency直方图, property map的COW追踪

**预计产出**: ~3-5文件, ~1550行

## 第3位Claude: M077-M079 — 跨模块集成测试桩 + CMake

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M077 | neograph.hpp 汇总头文件 (one-include), 条件编译宏, 全局dump_all_stats() | ~200行 |
| M078 | CMakeLists.txt模板 (含新文件), 编译flag, 依赖检测 | ~300行 |
| M079 | 集成测试桩: ART单元测试骨架, NeoGraph插入/查询烟雾测试 | ~400行 |

修改重点: 编译开关控制ART_DEBUG/EDGE_PROPERTY_NUM, test fixture, assert宏

**预计产出**: ~3-4文件, ~900行

## 第4位Claude: M080-M082 — 性能基准 + Benchmark补全

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M080 | ART micro-benchmark: insert/search/iterate吞吐量测量, node利用率报告 | ~500行 |
| M081 | NeoGraph端到端benchmark: 模拟LDBC-style workload, 混合读写延迟分布 | ~600行 |
| M082 | 跨层benchmark: HBM/GDDR/DRAM tier间数据迁移延迟, tier选择策略效果对比 | ~500行 |

修改重点: 统计收集器聚合, histogram输出, CSV dump

**预计产出**: ~3-4文件, ~1600行

## 第5位Claude: M083-M085 — 文档 + 代码质量

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M083 | 全项目API文档: 每个public class/struct的Doxygen注释, 模块依赖图 | ~800行注释 |
| M084 | 代码一致性扫描: 命名约定统一, include path整理, 未使用代码清理 | ~200行修改 |
| M085 | README更新: 构建指南, 架构图, 模块说明, 性能数据模板 | ~300行 |

**预计产出**: 修改多文件, 净新增~1300行

## 第6位Claude: M086-M088 — 最终集成 + 发布准备

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M086 | 编译验证: 全平台编译通过(GCC12+/Clang15+), warning清理 | 修改 |
| M087 | 依赖清单: third-party license, 版本固定, reproducible build脚本 | ~200行 |
| M088 | Release checklist: git tag, CHANGELOG, 迁移完成验证报告 | ~100行 |

**预计产出**: ~2-3文件, ~300行

---

## 总量预估 (最终)

| Phase | 文件数 | 行数 | 内容 |
|-------|--------|------|------|
| M001-M072 (之前8位Claude) | 133 | 51,697 | 全模块基础迁移 |
| **M073 (第1位Claude·本次)** | **7新+1改** | **4,715** | **ART完整实现 + art_new compat + NeoGraph gaps** |
| M074-M076 (第2位) | ~3-5 | ~1,550 | Driver补全 + property |
| M077-M079 (第3位) | ~3-4 | ~900 | 集成头文件 + CMake + 测试桩 |
| M080-M082 (第4位) | ~3-4 | ~1,600 | 性能基准 |
| M083-M085 (第5位) | 多文件修改 | ~1,300 | 文档 + 代码质量 |
| M086-M088 (第6位) | ~2-3 | ~300 | 发布准备 |
| **合计** | **~155+** | **~62,062** | **完整系统** |
