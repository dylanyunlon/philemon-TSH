# Philemon-TSH 多Claude开发计划 — 面向ATC 2026

## 论文核心claim与实验对应

Philemon-TSH = RapidStore + 三层异构存储 (DRAM/SSD/HDD)
upstream = SJTU-Liquid/RapidStore (VLDB'25)

### 论文要回答的5个研究问题 (RQ)
1. **RQ1**: 三层存储对insert/delete吞吐的影响（vs 纯DRAM系统）
2. **RQ2**: graph analytics (BFS/PR/SSSP/WCC) 在三层上的性能与纯DRAM对比
3. **RQ3**: 并发读写下的性能隔离（读延迟增幅 <15%）
4. **RQ4**: 图规模从百万到十亿边的扩展性
5. **RQ5**: 各组件ablation（tier placement, 压缩, prefetch, eviction）

### SOTA竞品及真实数据 (来源: VLDB'25 RapidStore, LHGstore 2026)

| System | 来源 | Insert (MEPS@32T) | BFS (s, LJ) | PR 10iter (s, LJ) | Memory (GB, LJ) |
|--------|------|-------------------|-------------|-------------------|-----------------|
| RapidStore | VLDB'25 upstream | ~2.5 | ~25 | ~295 | ~6.2 |
| Sortledton | VLDB'22 | ~3.0 (最快) | ~25 | ~499 | ~3.7 |
| Teseo | VLDB'21 | ~1.5 | ~49 | ~295 | ~5.3 |
| LiveGraph | SOSP'20 | ~0.8 | ~69 | ~997 | ~7.0 |
| Aspen | PLDI'19 | ~1.2 | ~25 | ~517 | ~28.3 |
| LHGstore | 2026 | ~4.0 (learned) | ~22 | ~294 | ~4.1 |
| **Philemon目标** | **ATC'26** | **≥2.0** | **≤30** | **≤300** | **≤2.0 (含SSD)** |

注: 上表秒数来自LHGstore Table 3 (LiveJournal数据集), MEPS为估算
Philemon的优势不在于绝对速度, 而在于**以1/3内存+SSD实现90%性能**

### 真正的竞争定位

**不应该声称beat所有SOTA的绝对性能** — 纯DRAM系统在内存足够时必然更快。

**应该声称**: 
- 当图规模超出DRAM容量时(如Friendster 65B edges), 纯DRAM系统OOM, Philemon仍可运行
- 在等成本硬件(128GB DRAM + 2TB SSD vs 1.5TB DRAM)上, Philemon性能更优
- 三层tier placement使热数据自动进DRAM, 冷数据在SSD, 算法性能保持90%+
- 并发读写隔离优于Sortledton/Teseo(它们lock contention严重)

## 仓库规则
1. **不开新分支**, 全部main
2. **不用v2/port/old后缀**
3. **改算法核心**, 不改格式/docstring
4. commit author: `dylanyunlon <dogechat@163.com>`
5. push前编译通过, 测试全PASS
6. 子Claude用 `claude_hk_chat.sh` 调度 Opus 4.6
7. 截断发 Continue

## ags1 服务器
- 2×AMD EPYC 9354 (128核), 1.5TB RAM, 2×A6000+1×H100
- NUMA node1 (GPU侧): `numactl --cpunodebind=1 --membind=1`

## 开发进度

### 第1位Claude（已完成）: M145-M146
upstream algorithms/readers/utils 全覆盖, OpenMP并行算法, 26/26 pass
**自我批判**: 实验与论文claim脱节, 没有测三层存储, 没有竞品对比

### 第2位Claude: M147-M148
**任务**: 构建可复现的竞品对比实验框架
- 下载并编译 RapidStore upstream 作为 baseline
- 用 DynamicGraphStorage 测试框架的 config 格式
- 在 LiveJournal + Graph500-24 上跑 insert/scan/search microbenchmark
- 产出 Table: Philemon vs RapidStore vs CSR baseline
**关键**: 不是自己编benchmark数字, 而是用upstream的driver跑真实对比

### 第3位Claude: M149-M150
**任务**: 三层存储的真实I/O路径
- 实现 mmap + direct I/O 的 SSD tier (不是模拟)
- edge数据按hotness分层: hot→DRAM, warm→mmap, cold→SSD direct I/O
- 测量真实tier placement对算法性能的影响
**SOTA对比**: 当内存限制为64GB时, 纯DRAM系统在Graph500-26上OOM, Philemon仍可运行

### 第4位Claude: M151-M152
**任务**: 并发读写实验 (RQ3)
- 32线程混合workload: N readers + M writers
- 测量PR读延迟在不同writer压力下的增幅
- 对比Sortledton (lock contention 34%退化) vs Philemon (subgraph-centric)

### 第5位Claude: M153-M154
**任务**: 大规模扩展性 + GPU加速 (RQ4)
- Friendster (1.8B edges) 或 Twitter (1.5B edges) 规模测试
- GPU BFS/PR kernel on H100
- 产出 Figure: 性能随图规模的变化曲线

### 第6位Claude: M155-M156
**任务**: Ablation + 论文数据整合 (RQ5)
- 关闭tier placement → 性能退化多少
- 关闭prefetch → 退化多少
- 关闭压缩 → 内存增加多少
- 产出所有论文Table/Figure的最终数据

## 运行流程
```bash
# ags1
cd /data/jiacheng/system/cache/temp/atc2026/philemon-TSH
git pull origin main
./experiment/run_ags1_ci.sh
```

## 子Claude指令模板
```
你是philemon-TSH第N位Claude (Opus 4.6)。
仓库: github.com/dylanyunlon/philemon-TSH
任务: M{X}-M{Y}

规则: main分支, 不开新分支, 不用后缀, 改算法核心
编译通过+测试PASS后 git push origin main
author: dylanyunlon <dogechat@163.com>
截断时用户发 Continue

读 CLAUDE_DEVELOPMENT_PLAN.md 了解全局
读 experiment/logs/ 了解前一位的运行结果
目标: 产出论文实验数据, 证明三层存储在大图+有限内存下的优势
```
