# Philemon-TSH 开发进度计划

## 里程碑总览

| Milestone | 状态 | 内容 |
|-----------|------|------|
| M001-M070 | ✅ 已完成 | 基础框架 (123文件/49824行, 之前开发者完成) |
| M071 | ✅ 完成 | NeoGraph核心子系统移植 (19文件/5529行) |
| M072 | ✅ 完成 | NeoGraph M072 patch (合入M071) |
| M073 | ✅ 完成 | ART完整实现 + art_new compat + NeoGraph gaps (7新文件/4715行) |
| **M074-M076** | **✅ 第1位Claude完成** | **Driver补全 + 真实数据集实验 (5新文件/2503行, LiveJournal 69M边全量验证)** |
| M077-M079 | 🔜 第2位Claude | upstream wrapper.h全量移植 + NeoGraph property补全 (~1200行) |
| M080-M082 | 🔜 第3位Claude | 跨模块集成头文件 + CMake统一编译 + 测试桩 (~900行) |
| M083-M085 | 🔜 第4位Claude | ART micro-benchmark + NeoGraph端到端bench (~1600行) |
| M086-M088 | 🔜 第5位Claude | 文档 + 代码一致性 + README (~1300行) |
| M089-M091 | 🔜 第6位Claude | 最终集成 + 编译验证 + 发布准备 (~300行) |

---

## 第1位Claude (已完成): M074-M076

**任务**: Driver缺失workload函数补全 + 算法委托 + 真实数据集实验

**交付物** (5新文件, 2503行):

| 文件 | 行数 | 来源 | 内容 |
|------|------|------|------|
| `src/driver/philemon_driver_workloads.hpp` | 416 | upstream driver.h:148-651 (~971行) | initialize_graph, execute_insert_delete, execute_batch_insert, execute_microbenchmarks |
| `src/driver/philemon_driver_algo_delegates.hpp` | 317 | upstream driver.h:724-889 (~144行) | bfs/sssp/wcc/page_rank snapshot委托, UnionFind |
| `experiment/philemon_realscale_experiment.cpp` | 522 | upstream main.cpp + driver.h execute() | 全流程实验入口, SNAP格式真实数据直读 |
| `experiment/philemon_experiment.cpp` | 1094 | upstream全模块集成 | 合成数据实验 + tier感知 |
| `experiment/run_experiment.sh` | 119 | upstream run.sh | numactl/系统信息/日志采集 |
| `experiment/philemon_experiment.cfg` | 35 | upstream config.cfg | 配置文件 |

**算法改动 (20%修改点)**:
- driver_workloads: 延迟直方图P50/P99, 吞吐率checkpoint, per-thread速率分解
- driver_algo_delegates: AlgoTierStats边遍历追踪, UnionFind按秩合并+路径压缩统计, per-level/per-iter断点
- realscale_experiment: 自动vertex range检测(无需vertex文件), 每100万边insert进度打印, 全算法结果统计

**实验验证**:
- email-Enron (36,692 V / 367,662 E): BFS 2.6ms, SSSP 5.9ms, PR 8.4ms, WCC 2.1ms
- **soc-LiveJournal1 (4,847,571 V / 68,993,773 E)**: 加载40.8s(2.56M/s), BFS 1.5s, SSSP 2.5s, PR 2.8s(5iter), WCC 0.8s, Scan 50.7M edges/s

---

## 后续开发进度规划 (第2-6位Claude)

### 第2位Claude: M077-M079 — upstream wrapper.h全量移植 + NeoGraph property

| Milestone | 任务 | 来源 | 预计行数 |
|-----------|------|------|---------|
| M077 | wrapper.h全部249行的模板函数移植 (snapshot_edges, snapshot_has_edge, snapshot_intersect等20+函数) | upstream wrapper.h | ~350行 |
| M078 | driver.h剩余workload: execute_mixed_reader_writer (含fork/QoS版), execute_qos, execute() dispatcher switch | upstream driver.h:890-1576 | ~500行 |
| M079 | neo_property.cpp补全: edge property map COW操作, property batch update | upstream neo_property.cpp | ~350行 |

修改重点: wrapper函数加tier感知计数器, mixed_reader_writer加读写延迟分离统计, property加COW generation tag

**预计产出**: ~3文件, ~1200行

### 第3位Claude: M080-M082 — 跨模块集成 + CMake + 测试桩

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M080 | neograph.hpp 一站式汇总头文件, 条件编译宏(ART_DEBUG/EDGE_PROPERTY_NUM), dump_all_stats() | ~200行 |
| M081 | CMakeLists.txt更新: 新文件纳入编译, neograph子目录CMake, 依赖检测(pthread/TBB) | ~300行 |
| M082 | 集成测试桩: ART insert/search单元测试, NeoGraph snapshot烟雾测试, driver workload冒烟测试 | ~400行 |

修改重点: 编译开关, test fixture, gtest assert, CI smoke

**预计产出**: ~4文件, ~900行

### 第4位Claude: M083-M085 — 性能基准Benchmark

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M083 | ART micro-benchmark: insert/search/iterate吞吐量, node利用率, 内存碎片报告 | ~500行 |
| M084 | NeoGraph端到端bench: LDBC-style workload, 混合读写延迟分布, snapshot开销 | ~600行 |
| M085 | 跨层benchmark: DRAM/CXL/SSD tier迁移延迟, tier选择策略效果, 热度追踪精度 | ~500行 |

修改重点: 统计聚合器, histogram/CSV输出, 多seed置信区间

**预计产出**: ~3文件, ~1600行

### 第5位Claude: M086-M088 — 文档 + 代码质量

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M086 | 全项目API文档: 每个public class的Doxygen注释, 模块依赖图(mermaid) | ~800行注释 |
| M087 | 代码一致性: 命名约定统一(snake_case), include路径整理, dead code清理 | ~200行修改 |
| M088 | README更新: 构建指南, 架构图, 模块说明, 实验复现步骤 | ~300行 |

**预计产出**: 多文件修改, 净新增~1300行

### 第6位Claude: M089-M091 — 最终集成 + 发布

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M089 | 全平台编译验证: GCC12+/Clang15+/CUDA11.5+ warning-free | 修改 |
| M090 | 依赖清单: third-party license, 版本固定, reproducible build脚本 | ~200行 |
| M091 | Release: git tag, CHANGELOG, 迁移完成验证报告 | ~100行 |

**预计产出**: ~2-3文件, ~300行

---

## 迁移覆盖进度

| 模块 | Upstream行数 | src/已有行数 | 覆盖 |
|------|-------------|-------------|------|
| NeoGraph ART (c_art) | 6,305 | 5,506 | ✅ |
| NeoGraph ART (art_new) | 4,195 | (merged) | ✅ |
| NeoGraph core | 7,590 | 10,259 | ✅ |
| NeoGraph utils | 881 | 789 | ✅ |
| Algorithms | 961 | 5,475 | ✅ |
| Wrapper Algorithms | 1,009 | 1,053 | ✅ |
| Graph IO | 179 | 524 | ✅ |
| Readers | 248 | 2,189 | ✅ |
| Utils | 960 | 817 | ✅ |
| Types | 150 | 321 | ✅ |
| Preprocessor | 1,168 | 902 | ✅ |
| TemGraph | 810 | 1,175 | ✅ |
| Main/Entry | 337 | 1,134 | ✅ |
| Wrapper (NeoGraph) | 727 | 817 | ✅ |
| **Driver** | **1,577** | **1,680** | **✅ M074补全** |
| **wrapper.h** | **249** | **188 (partial)** | **⚠️ M077补全** |
| **Experiment** | **(new)** | **1,770** | **✅ M076新增** |

## 总量

| Phase | 文件数 | 行数 |
|-------|--------|------|
| M001-M073 (之前) | 141 | 56,412 |
| **M074-M076 (第1位Claude)** | **+5** | **+2,503** |
| M077-M091 (第2-6位Claude) | ~15 | ~5,300 |
| **合计** | **~161** | **~64,215** |
