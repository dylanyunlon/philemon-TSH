# Philemon-TSH 多Claude开发进度计划

## 目标
ATC 2026 论文实验数据：超越现有SOTA（LiveGraph, Sortledton, Teseo, Aspen）
所有实验产出数据直接填入 LaTeX 论文 Table/Figure。

## 仓库规则（每位Claude必读）
1. **不开新分支**，全部在 main 上开发
2. **不用 v2/v3/port/old 等后缀**，文件命名统一
3. 代码改的是**算法核心**，不是字符串/docstring/格式化
4. push 前编译通过，测试全 PASS
5. commit author: `dylanyunlon <dogechat@163.com>`
6. 每位Claude完成后直接 `git push origin main`
7. 子Claude用 `claude_hk_chat.sh` 调度，选择 **Opus 4.6**
8. 如果输出被截断，发送 **Continue** 继续执行

## 服务器 ags1 信息
- **CPU**: 2× AMD EPYC 9354 (128核/256线程)
- **RAM**: 1.5TB (NUMA node0: 774GB, node1: 774GB)
- **GPU**: 2× RTX A6000 (49GB) + 1× H100 NVL (96GB)
- **CUDA**: 11.5, Driver 550.144.03
- **OS**: Ubuntu 22.04, kernel 5.15.0-164
- **所有GPU在NUMA node1**: `numactl --cpunodebind=1 --membind=1`

## 当前状态
- **95 commits**, 163 src文件, 42 experiment文件
- M001-M144 全部完成
- **M145-M146**: 第1位Claude完成 (algorithms/readers/utils 深度实验, 47/47 pass)

---

## 多Claude开发进度

### 第1位Claude（已完成）: M145-M146
**任务**: upstream algorithms/ (BFS/PR/SSSP/WCC) + readers/ + utils/ 全覆盖
**交付**: `experiment/m145_m146_algo_reader_utils_experiment.cpp` (1581行, 47/47 pass)
**SOTA数据**: BFS 3.4ms, PR 65ms, SSSP 19ms, WCC 12ms @ 100K vertices

### 第2位Claude: M147-M148
**任务**: upstream `main.cpp`(202行) + `wrapper.h`(249行) + `driver.h`(1577行) 深度实验
**重点算法改动**:
- driver 执行引擎：三层tier异步调度算法（DRAM→SSD→HDD pipeline overlap）
- wrapper graph接口：CSR/COO动态转换算法 + tier-aware edge iterator
- main入口：config-driven workload orchestration + 实验自动化
**预计行数**: ~1500行
**SOTA目标**: insert throughput > 2M edges/s（超越LiveGraph 1.8M）

### 第3位Claude: M149-M150
**任务**: NeoGraph include headers (1942行) 全覆盖 — `neo_types`, `neo_property`, `neo_config`, `neo_error_type`, `neo_helper`, `neo_spin_lock`, `neo_thread_pool`
**重点算法改动**:
- ART索引 tier-aware range查询算法
- B+Tree snapshot隔离算法（MVCC优化）
- 类型系统：temporal edge timestamp ordering + 压缩算法
**预计行数**: ~1200行
**SOTA目标**: point query < 1μs (超越Sortledton 1.3μs)

### 第4位Claude: M151-M152
**任务**: 全upstream交叉验证 + SOTA benchmark数据生成
**重点算法改动**:
- 跨tier一致性验证算法（三层snapshot对齐）
- LiveGraph/Sortledton/Teseo/Aspen 对比实验
- Twitter/uk-2005/Friendster 大规模真实图集
**预计行数**: ~1000行
**SOTA目标**: 产出论文 Table 1 (insert/delete throughput) + Table 2 (algorithm latency)

### 第5位Claude: M153-M154
**任务**: ags1 GPU+CPU联合benchmark + 论文Table/Figure数据
**重点算法改动**:
- GPU BFS kernel: warp-cooperative direction-optimizing（H100 NVL优化）
- GPU PageRank: multi-GPU partition（2×A6000 + H100 pipeline）
- CPU-GPU异步数据迁移算法
**预计行数**: ~800行
**SOTA目标**: 产出 Figure 3 (scalability) + Figure 4 (GPU speedup) + Figure 5 (tier分布)

### 第6位Claude: M155-M156
**任务**: 全回归 + CHANGELOG + Release + LaTeX数据整合
**重点算法改动**:
- 全链路regression检测算法（性能不退化保证）
- 自动化LaTeX数据表生成脚本
- ablation study: 各组件贡献量化
**预计行数**: ~500行
**SOTA目标**: 论文所有Table/Figure数据完整、可复现

---

## 运行流程

### ags1 上执行
```bash
cd /data/jiacheng/system/cache/temp/atc2026
git pull origin main
./experiment/run_ags1_ci.sh   # 自动编译+多规模+push日志
```

### 调度子Claude
```bash
./claude_hk_chat.sh  # 选择 Opus 4.6
# 把本文件 + 第一轮prompt附件 + 最新日志发给子Claude
# 子Claude读取 experiment/logs/ 中的 SUMMARY.md 和 CSV
# 子Claude完成后直接 git push origin main
```

### 论文数据目标 (SOTA baselines to beat)

| Metric | LiveGraph | Sortledton | Teseo | Aspen | **Philemon (目标)** |
|--------|-----------|------------|-------|-------|-------------------|
| Insert (M edges/s) | 1.8 | 1.2 | 2.1 | 0.9 | **≥3.0** |
| Delete (M edges/s) | 1.5 | 1.0 | 1.8 | 0.7 | **≥2.5** |
| BFS (ms, LiveJournal) | 850 | 1200 | 780 | 650 | **≤400** |
| PageRank (ms, 10iter) | 2800 | 3500 | 2600 | 2200 | **≤1500** |
| SSSP (ms) | 1500 | 2000 | 1400 | 1100 | **≤700** |
| WCC (ms) | 900 | 1300 | 850 | 700 | **≤450** |
| Point Query (μs) | 1.5 | 1.3 | 1.8 | 2.1 | **≤0.8** |
| Scan (M edges/s) | 45 | 35 | 50 | 55 | **≥80** |

---

## 子Claude指令模板

发给每位子Claude的开头：
```
你是philemon-TSH项目的第N位Claude (Opus 4.6)。
仓库: https://github.com/dylanyunlon/philemon-TSH
你的任务是 M{X}-M{Y}。

规则:
1. git clone, 在 main 分支上工作
2. 不开新分支, 不用v2/port/old后缀
3. 改算法核心, 不改格式
4. 编译通过, 测试全PASS后 git push origin main
5. commit author: dylanyunlon <dogechat@163.com>
6. 如果被截断, 用户会发 Continue

读取 CLAUDE_DEVELOPMENT_PLAN.md 了解全局进度。
读取 experiment/logs/ 了解前一位Claude的运行结果。
你的目标是产出超越SOTA的实验数据填入论文。
```
