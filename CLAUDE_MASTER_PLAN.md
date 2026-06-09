# Philemon-TSH — Master Development Plan (ATC'26)

> **目标**: 让 Philemon-TSH 在真实 GPU 硬件上跑出超越 SOTA 的数据，
> 填入 tex 论文，提交 ATC'26。

## 当前状态诊断

| 指标 | 当前值 | 目标 | 差距 |
|------|--------|------|------|
| BFS retention vs CSR | 110.6% | ≥100% | ✅ 已达标 |
| PageRank retention | 67.9% | ≥90% | ❌ 差 22% |
| SSSP retention | 89.2% | ≥90% | ⚠️ 差 0.8% |
| Memory ratio vs CSR | 8.99× | ≤4× | ❌ 差 5× |
| GPU 实验 | CPU 模拟 | 真实 H100+A6000 | ❌ 未开始 |
| Baseline 对比 | 硬编码参考值 | 同机实测 | ❌ 未开始 |
| 论文图表 | 0 张 | ≥8 张 | ❌ 未开始 |
| 数据集 | RMAT 2^18 + tiny SNAP | LDBC SF-10/100 | ❌ 不足 |

## 服务器硬件 (ags1)

- **CPU**: 2× AMD EPYC 9354 (128 cores, 2 NUMA nodes)
- **RAM**: ~1.5 TB DDR5 (774GB node0 + 774GB node1)
- **GPU0**: RTX A6000 49GB GDDR6 (sm_86) — NUMA1
- **GPU1**: RTX A6000 49GB GDDR6 (sm_86) — NUMA1
- **GPU2**: H100 NVL 96GB HBM2e (sm_90) — NUMA1
- **互连**: PCIe only (无 NVLink)
- **CUDA**: 11.5 (需升级到 12.x for H100)
- **Driver**: 550.144 (支持 CUDA 12.x)
- **OS**: Ubuntu 22.04

## 开发阶段与 Claude 分配

### 第一位 Claude (本轮): M181-M185 — 基础设施与计划

- **M181**: 创建本开发计划 + 环境搭建脚本 (`experiment/setup_ags1.sh`)
- **M182**: 创建 SOTA baseline 构建脚本 (从源码编译 RapidStore / Sortledton / Teseo / LiveGraph)
- **M183**: 创建自动化实验 → git push 流水线 (`experiment/auto_push_results.sh`)
- **M184**: 创建数据集下载脚本 (LDBC SNB SF-1/10, RMAT 2^20/22/24)
- **M185**: Push 所有基础设施到仓库

### 第二位 Claude: M186-M192 — 算法修复 (核心)

**目标**: 解决 PageRank 67.9% 和 Memory 8.99× 两个致命问题。

- **M186**: 诊断 PageRank 性能衰退根因 — profile `tiered_pagerank.hpp` 的
  tier 访问模式，定位是 cross-tier random access 还是 touch() 开销
- **M187**: 优化 PageRank 的 tier placement — 将 PR 迭代中高频访问的
  vertex score array 钉在 HBM，只把 edge list 做 tiered
- **M188**: 重写 `tiered_allocator.hpp` 的内存分配策略 — 当前 slab 粒度
  太粗 (每个 partition 独立 slab)，改为 shared pool + per-tier arena
- **M189**: 实现 degree-aware compact placement — 高度顶点的邻接边
  紧凑存储在同一 tier，减少跨 tier 访问
- **M190**: 优化 `cost_estimator.hpp` — 根据算法类型动态调整 tier 策略
  (BFS/SSSP 喜欢 locality → more HBM; PR 全图迭代 → bandwidth 优先)
- **M191**: 减少 per-partition metadata 开销 — 当前每个 partition 有
  interval index + skip list node + seqlock，在小分区时 overhead 太大
- **M192**: 验证修复后 PR ≥90%, Memory ≤4× CSR

### 第三位 Claude: M193-M198 — SOTA Baseline 实测

**目标**: 在 ags1 上编译运行所有 baseline，生成同机同条件对比数据。

- **M193**: 在 ags1 上 setup conda 环境 + CUDA 12.x toolkit
- **M194**: 编译 RapidStore (VLDB'25) 并跑 RMAT + LDBC benchmark
- **M195**: 编译 Sortledton (VLDB'22) + Teseo (VLDB'21) 并跑 benchmark
- **M196**: 编译 LiveGraph + Aspen 并跑 benchmark (或从 DynamicGraphStorage 框架统一跑)
- **M197**: 跑 Philemon-TSH 在真实 3-tier GPU 上的 benchmark (H100 HBM / A6000 GDDR / DRAM)
- **M198**: 汇总所有系统的 latency/throughput/memory 数据到 `experiment/results/sota_comparison.csv`

### 第四位 Claude: M199-M204 — 真实 GPU Tiered Memory 实验

**目标**: 在真实 GPU 上验证 tiered memory 的 6 个 RQ。

- **M199**: 实现真实的 HBM/GDDR/DRAM 三级分配器 — 用 `cudaMalloc`
  (H100 HBM) / `cudaMallocManaged` + prefetch to A6000 / host malloc
- **M200**: 跑 RQ1-RQ3 (index prediction, tier placement, scan speedup)
  用 RMAT 2^20 和 LDBC SF-10
- **M201**: 跑 RQ4 (scaling to 100M edges) 用 RMAT 2^24 + LDBC SF-100
- **M202**: 跑 RQ5 (streaming + compaction) 用 连续 edge stream
- **M203**: 跑 RQ6 (concurrent queries during migration)
- **M204**: 生成 2000-pts × 3-seeds 的完整 JSON 数据 (替换当前 CPU 模拟数据)

### 第五位 Claude: M205-M210 — 论文图表与数据填充

**目标**: 生成论文所需的所有图表，更新 tex。

- **M205**: 用 pgfplots/matplotlib 生成 Figure 1: Query Latency vs Window Width
  (4 curves: Tiered/HBM-Only/DRAM-Only/Linear)
- **M206**: 生成 Figure 2: QPS Scaling + Figure 3: Memory Utilization per Tier
- **M207**: 生成 Figure 4: SOTA Comparison Bar Chart
  (Philemon vs RapidStore vs Sortledton vs Teseo vs LiveGraph)
- **M208**: 生成 Figure 5-6: Scan/Selection Speedup + Migration Latency
- **M209**: 更新 `philemon_tsh.tex` — 替换所有实验数据为真实 GPU 数据，
  插入图表，修正 memory claim
- **M210**: 更新 `philemon_tsh_reconstructed.tex` — 同步更新 appendix 数据

### 第六位 Claude: M211-M215 — 最终验证与提交准备

- **M211**: 端到端回归测试 — 确保所有 44 个 test case 仍然 pass
- **M212**: 论文 self-review — 检查所有声称与数据一致
- **M213**: 编译 tex 生成 PDF，检查无 undefined refs
- **M214**: 代码清理 — 删除无用文件，统一命名
- **M215**: 最终 push + tag `atc26-submission`

## SOTA Baseline 系统及其仓库

| System | Venue | Repo | 角色 |
|--------|-------|------|------|
| RapidStore | VLDB'25 | `github.com/SJTU-Liquid/RapidStore` | 主要 baseline (上游) |
| Sortledton | VLDB'22 | `github.com/Hao-Zhang-SJTU/sortledton` | DGS baseline |
| Teseo | VLDB'21 | `github.com/cwida/teseo` | DGS baseline |
| LiveGraph | USENIX ATC'20 | `github.com/thu-pacman/LiveGraph-Binary` | DGS baseline |
| Aspen | PLDI'19 | `github.com/ldhulipala/aspen` | DGS baseline |
| DynamicGraphStorage | SIGMOD'26 | `github.com/SJTU-Liquid/DynamicGraphStorage` | 统一测试框架 |

## 自动化流水线设计

```
ags1 服务器                          GitHub 仓库
┌─────────────────┐                 ┌─────────────────┐
│ conda env: atc26│                 │ philemon-TSH    │
│                 │   git push      │                 │
│ 跑实验脚本      ├────────────────→│ experiment/     │
│ 生成 CSV/JSON   │                 │   results/      │
│                 │   git pull      │   sota_*.csv    │
│ Claude 拉取代码 │←────────────────┤                 │
│ 修改算法        │                 │ src/            │
│ push 修改       ├────────────────→│   algorithms/   │
└─────────────────┘                 └─────────────────┘
```

## 关键约束

1. **不开新分支** — 所有 commit 直接到 main
2. **不加后缀** — 不允许 `_v2`, `_port`, `_new`, `_old` 等后缀
3. **改算法不改字符串** — 修改 `.hpp` 中的实际算法逻辑
4. **每次 push 前确保编译通过**
5. **子 Claude 使用 opus 4.6 (medium)**
6. **截断时发送 Continue 继续执行**
