# Philemon-TSH 开发进度计划

## 里程碑总览

| Milestone | 状态 | 内容 |
|-----------|------|------|
| M001-M070 | ✅ 已完成 | 基础框架 (123文件/49824行, 之前开发者完成) |
| M071 | ✅ 第一位Claude完成 | NeoGraph核心子系统移植 (19文件/5529行) |
| M072-M075 | 🔜 第二位Claude | NeoGraph ART完整实现 + art_new兼容层 |
| M076-M079 | 🔜 第三位Claude | 图算法层 + Wrapper算法层 |
| M080-M084 | 🔜 第四位Claude | 6个Wrapper适配器 + Driver框架 |
| M085-M088 | 🔜 第五位Claude | IO层 + 工具层 + TemGraph时序图 |
| M089-M092 | 🔜 第六位Claude | 入口集成 + 数据集预处理 + 全局胶水 |

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

## 第二位Claude: M072-M075

**任务**: NeoGraph ART完整实现 + art_new兼容层

| Milestone | 来源文件 | 目标 | 行数 |
|-----------|---------|------|------|
| M072 | `art_node_ops.cpp` (2080行) | `src/neograph/art/art_node_ops_impl.hpp` | ~2400行 |
| M073 | `art_node_ops_copy.cpp` (1081行) | `src/neograph/art/art_node_ops_copy_impl.hpp` | ~1250行 |
| M074 | `art_iter.cpp` (179行) + `art_node_iter.cpp` (442行) | `src/neograph/art/art_iter_impl.hpp` | ~720行 |
| M075 | `art_new/` 全部14文件 | `src/neograph/art_compat/` | ~3500行 |

**修改重点**:
- M072: `find_child` SSE路径加SIMD miss计数, `add_child` upgrade chain (4→16→48→256) 每级加计数器
- M073: COW copy path 加 generation counter + 共享度追踪
- M074: iterator step counting, ordered/unordered 分支选择统计
- M075: art_new 作为 c_art 的 shim layer, 差异函数加 dispatch 桥接

**预计产出**: ~4文件, ~7870行

---

## 第三位Claude: M076-M079

**任务**: 图算法层 + Wrapper算法层

| Milestone | 来源文件 | 目标 | 行数 |
|-----------|---------|------|------|
| M076 | `algorithms/BFS.{hpp,cpp}` (354行) | `src/algorithms/neo_bfs.hpp` | ~420行 |
| M077 | `algorithms/SSSP+PageRank` (424行) | `src/algorithms/neo_sssp.hpp`, `neo_pagerank.hpp` | ~500行 |
| M078 | `algorithms/WCC` (183行) + wrapper algos (BFS.h, PR.h, SSSP.h) | `src/algorithms/neo_wcc.hpp`, `src/wrapper/algo_*.hpp` | ~900行 |
| M079 | wrapper algos (TC.h, TC_opt.h, WCC.h) | `src/wrapper/algo_tc.hpp`, `algo_wcc.hpp` | ~400行 |

**修改重点**:
- BFS: frontier expansion 计数器, TD/BU step 切换阈值追踪
- SSSP: relaxation 次数统计, 负环检测断言
- PageRank: 每轮 L1-norm 收敛速度记录
- WCC: union-find path compression 深度直方图
- TC: triangle 计数器 per-vertex 分布

**预计产出**: ~6-8文件, ~2220行

---

## 第四位Claude: M080-M084

**任务**: 6个Wrapper适配器 + Driver框架

| Milestone | 来源文件 | 目标 | 行数 |
|-----------|---------|------|------|
| M080 | `neo_wrapper.{h,cpp}` (913行) | `src/wrapper/neo_wrapper.hpp` | ~1060行 |
| M081 | `aspen_wrapper + csr_wrapper` (901行) | `src/wrapper/aspen_adapter.hpp`, `csr_adapter.hpp` | ~1050行 |
| M082 | `livegraph + sortledton_wrapper` (1444行) | `src/wrapper/livegraph_adapter.hpp`, `sortledton_adapter.hpp` | ~1680行 |
| M083 | `teseo_wrapper` (550行) + `wrapper.h` (249行) | `src/wrapper/teseo_adapter.hpp`, `wrapper_base.hpp` | ~930行 |
| M084 | `driver.h` (1577行) + `driver_main.h` (15行) | `src/wrapper/driver.hpp` | ~1840行 |

**修改重点**:
- 每个adapter: 操作计数器 (insert/query/delete 分别统计)
- neo_wrapper: NeoGraph特有的 MVCC snapshot 生命周期追踪
- driver: benchmark timing 精度从 ms 提升到 μs, 增加 warmup 检测

**预计产出**: ~7文件, ~6560行

---

## 第五位Claude: M085-M088

**任务**: IO层 + 工具层 + TemGraph时序图

| Milestone | 来源文件 | 目标 | 行数 |
|-----------|---------|------|------|
| M085 | `graph/edge.{hpp,cpp}` + `edgeStream.{hpp,cpp}` (179行) | `src/io/edge.hpp`, `edge_stream.hpp` | ~210行 |
| M086 | `readers/*.{hpp,cpp}` (248行) | `src/io/readers.hpp` | ~290行 |
| M087 | `utils/Timer.h` + `commandLineParser` + `log/` + `error_type` (980行) | `src/utils/timer.hpp`, `cli_parser.hpp`, `logger.hpp` | ~1140行 |
| M088 | `temgraph/*.{h,cpp}` (810行) | `src/temgraph/` | ~940行 |

**修改重点**:
- edgeStream: 边流读取 throughput 计数器 (edges/sec)
- readers: 文件解析 bytes_read 追踪, 格式检测断言
- Timer: 高精度计时 + lap() 嵌套追踪
- CLI parser: 参数验证后置断言
- TemGraph: 时间窗口查询 range scan 计数, interval tree 深度统计

**预计产出**: ~6-8文件, ~2580行

---

## 第六位Claude: M089-M092

**任务**: 入口集成 + 数据集预处理 + 全局胶水

| Milestone | 来源文件 | 目标 | 行数 |
|-----------|---------|------|------|
| M089 | `main.cpp` (202行) + `main_tem_graph.cpp` (135行) | `src/main_rapidstore.hpp`, `src/main_temgraph.hpp` | ~390行 |
| M090 | `dataset_preprocessor/*.{hpp,cpp}` (1168行) | `src/preprocessor/` | ~1350行 |
| M091 | `types/types.hpp` (150行) + `third-party/gapbs` (906行) | `src/types/`, `src/third_party/` | ~1220行 |
| M092 | 全局胶水: neograph.hpp 汇总头, CMakeLists 模板, 全局 dump_all_stats() | `src/neograph.hpp`, etc | ~300行 |

**修改重点**:
- main: startup banner + 全局 stats dump at exit
- preprocessor: 数据清洗 pipeline 的 stage 计数器
- types: 类型转换 narrowing 断言
- gapbs: pvector 分配追踪
- neograph.hpp: one-include 汇总, 条件编译宏

**预计产出**: ~8-10文件, ~3260行

---

## 总量预估

| Phase | 文件数 | 行数 | Upstream来源 |
|-------|--------|------|-------------|
| M001-M070 (已有) | 123 | 49,824 | — |
| M071 (第一位Claude) | 19 | 5,529 | NeoGraph core |
| M072-M075 (第二位) | ~4 | ~7,870 | ART impl + art_new |
| M076-M079 (第三位) | ~7 | ~2,220 | Algorithms |
| M080-M084 (第四位) | ~7 | ~6,560 | Wrappers + Driver |
| M085-M088 (第五位) | ~7 | ~2,580 | IO + Utils + TemGraph |
| M089-M092 (第六位) | ~9 | ~3,260 | Entry + Preprocessor + Glue |
| **合计** | **~176** | **~77,843** | **全部88个upstream文件** |

---

## 使用方式

```bash
# 应用第一位Claude的patch
git am M071-neograph-subsystem.patch

# 后续每位Claude生成自己的patch
# 第二位: M072-M075-art-impl.patch
# 第三位: M076-M079-algorithms.patch
# ...
```

## 每位Claude的交接协议

1. 读取本文件确认自己负责的里程碑范围
2. 读取 upstream/ 对应源文件
3. 按鲁迅式~20%算法修改规则创建文件
4. `git format-patch` 生成 patch, 作者 `dylanyunlon <dogechat@163.com>`
5. 更新本文件的完成状态
