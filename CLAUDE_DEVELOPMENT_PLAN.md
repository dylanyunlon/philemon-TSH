# Philemon-TSH 开发进度计划

## 里程碑总览

| Milestone | 状态 | 内容 |
|-----------|------|------|
| M001-M070 | ✅ 已完成 | 基础框架 (123文件/49824行, 之前开发者完成) |
| M071 | ✅ 完成 | NeoGraph核心子系统移植 (19文件/5529行) |
| M072 | ✅ 完成 | NeoGraph M072 patch (合入M071) |
| M073 | ✅ 完成 | ART完整实现 + art_new compat + NeoGraph gaps (7新文件/4715行) |
| **M074-M076** | **✅ 第1位Claude完成** | **Driver补全 + 真实数据集实验 (5新文件/2503行, LiveJournal 69M边全量验证)** |
| **M077-M079** | **✅ 第1位Claude完成** | **LLM4Walking实验 + GPU树遍历 (6新文件/3830行, ART/Interval/Galloping)** |
| **M080-M082** | **✅ 第1位Claude调度Opus4.6完成** | **GPU warp-cooperative find_child + merge-path intersect + multi-GPU partition (1603行)** |
| **M083-M085** | **✅ 第1位Claude调度Opus4.6完成** | **TemGraph GPU时序查询: CSR化 + range query + successor walk (1745行)** |
| M086-M088 | 🔜 第4位Claude | NeoTree GPU MVCC snapshot + version chain scan (~1200行) |
| M089-M091 | 🔜 第5位Claude | 跨tier benchmark + 热度驱动placement (~1300行) |
| M092-M094 | 🔜 第6位Claude | 端到端集成 + LDBC workload + 论文复现 (~1200行) |

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

### 第1位Claude (续): M077-M079 — LLM4Walking实验 + GPU树遍历

**任务**: Walking实验框架 + 图遍历算法移植 + GPU树遍历kernel

**交付物** (6新文件, 3830行):

| 文件 | 行数 | 位置 | 核心算法 |
|------|------|------|---------|
| `walking_experiment.cpp` | 1095 | experiment/ | BFS direction-switch, PR收敛追踪, SSSP delta-stepping, WCC label propagation, TC |
| `walking_realscale.cpp` | 522 | experiment/ | 真实数据集加载 + 全算法执行 + per-100万checkpoint |
| `walking_inspector.cpp` | 633 | experiment/ | 三角不等式验证, PR归一化断言, WCC BFS交叉验证 |
| `walking_gpu_tree.cu` | 866 | src/cuda/ | GPU ART find_child/node_search/BFS + galloping intersect + interval stab |
| `walking.cfg` | 35 | experiment/ | 参数配置 |
| `llm4walking_run.sh` | 679 | experiment/ | pipeline编排 |

**算法改动 (~20%)**:
- BFS: tier_hit_count[3], direction-switch阈值打印
- PageRank: 二阶导数收敛率, 早停, top-5/每轮
- SSSP: 距离分桶直方图, 松弛率追踪
- WCC: 组件大小分布, 每轮merge计数
- ART find_child: 四路dispatch GPU化, 预判节点类型跳过升级链
- Intersect: galloping指数探测+二分, skew>4x自动切换 (84K skips实测)
- Interval stab: 二分预处理缩小线性扫描范围

---

## 后续开发进度规划 (第2-6位Claude)

### 第2位Claude(Opus 4.6, 由第1位调度): M080-M082 — GPU warp-cooperative树操作 ✅

**任务**: warp协作查找 + merge-path并行交集 + 多GPU ART分区

**交付物** (1新文件, 1603行):

| 文件 | 行数 | 位置 | 核心算法 |
|------|------|------|---------|
| `walking_warp_cooperative.cu` | 1603 | src/cuda/ | §1-§8: FlatART, CPU baseline, GPU serial, warp find_child, merge-path, multi-GPU partition |

**算法改动 (~20%)**:
- M080: Node16 warp `__ballot_sync` 16-lane并行, Node48 warp-shuffle probe
- M081: merge_path_partition对角线二分, P线程并行交集(两遍kernel)
- M082: hash(prefix_byte)%num_gpus子树分区, scatter+per-GPU launch

**实验验证**: M080 25K hits全对, M082 50000/50000 match (1/2/4 GPU), balance 1.00-1.02

| Milestone | 任务 | 来源 | 实际行数 |
|-----------|------|------|---------|
| M080 | warp-cooperative find_child | art_node_ops_impl find_child | ~400行 |
| M081 | merge-path intersect | galloping | ~500行 |
| M082 | multi-GPU ART partition | cuda_multi_gpu_partition.hpp | ~400行 |

### 第3位Claude(Opus 4.6, 由第1位调度): M083-M085 — TemGraph GPU时序查询 ✅

**任务**: successor链CSR化 + GPU temporal range query + successor walk batch

**交付物** (1新文件, 1745行):

| 文件 | 行数 | 位置 | 核心算法 |
|------|------|------|---------|
| `walking_temgraph_gpu.cu` | 1745 | src/cuda/ | FlatSuccessorCSR + FlatTemGraph + range query kernel + successor walk kernel |

**算法改动 (~20%)**:
- M083: successor链 linked list → CSR (row_ptr/col_idx/timestamps), crossval验证
- M084: contains/contained → GPU kern_temporal_range_query: 每thread二分查找时间窗口
- M085: 单起点串行遍历 → kern_successor_walk: N起点GPU并行, kern_successor_walk_timed: 时间约束遍历

**实验验证**: M083 CSR全对, M084 count 10000/10000 match, M085 walk 100/100 paths全对(4/8/16 hops)

| Milestone | 任务 | 来源 | 实际行数 |
|-----------|------|------|---------|
| M083 | successor链CSR化 | tem_graph_impl.hpp | ~500行 |
| M084 | GPU temporal range query | tem_graph_impl query | ~500行 |
| M085 | successor walk batch | tem_graph_impl successor_link | ~400行 |

### 第4位Claude: M086-M088 — NeoTree GPU MVCC

| Milestone | 任务 | 来源 | 预计行数 |
|-----------|------|------|---------|
| M086 | version chain GPU scan: 版本链flat化, GPU并行查timestamp | neo_tree_version_impl.hpp find_version | ~500行 |
| M087 | GPU snapshot read: GPU上snapshot_edges遍历 | neo_snapshot.hpp + neo_tree.hpp edges() | ~400行 |
| M088 | GC offload: 过期版本判定GPU化, 回收CPU执行 | neo_tree_version_impl GC | ~300行 |

### 第5位Claude: M089-M091 — 跨tier Benchmark

| Milestone | 任务 | 来源 | 预计行数 |
|-----------|------|------|---------|
| M089 | tier迁移延迟矩阵: DRAM↔CXL↔SSD↔GPU各路径P50/P99 | hetero_bench.cu E4 | ~500行 |
| M090 | 热度驱动placement: access_heat→promote/demote决策 | hotness_tracker + online_learner | ~400行 |
| M091 | 并发查询+后台迁移: 吞吐量衰减测量 | hetero_bench.cu E5 | ~400行 |

### 第6位Claude: M092-M094 — 端到端集成

| Milestone | 任务 | 预计行数 |
|-----------|------|---------|
| M092 | LDBC SNB workload端到端 | ~600行 |
| M093 | 论文实验复现: Table/Figure自动化 | ~400行 |
| M094 | Release: 编译验证 + CHANGELOG + 回归检测 | ~200行 |

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
