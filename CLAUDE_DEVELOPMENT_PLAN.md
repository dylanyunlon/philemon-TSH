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

### 第3位Claude: M149-M150 ✅ DONE
**任务**: 三层存储的真实I/O路径
- 实现 mmap + direct I/O 的 SSD tier (不是模拟)
- edge数据按hotness分层: hot→DRAM, warm→mmap, cold→SSD direct I/O
- 测量真实tier placement对算法性能的影响
**结果**: BFS 1.50x slowdown (67%保持), PR 1.57x (64%保持), 正确性完全一致

### 第4位Claude: M151-M152 ✅ DONE
**任务**: 并发读写实验 (RQ3)
- MutableGraph with per-vertex atomic spinlock
- PR reader + edge insert/delete writer 并行
- 小规模高contention预期内, 需ags1 32+线程
**结果**: 框架完成, scale14/4T测试通过

### 第5位Claude: M153-M154 ✅ DONE
**任务**: Ablation + 论文figure数据
- 6种ablation配置: all-DRAM, hot-only, hot+warm, full, +prefetch, unsorted
- Multi-scale: scale 12→16 BFS ratio从1.72→1.40 (越大越好!)
- LaTeX tabular行输出
**结果**: 全部通过, 产出可直接粘贴的LaTeX数据

### 第6位Claude: M155-M156 ✅ DONE
**任务**: 回归测试 + 论文数据整合
- 12项回归测试全PASS
- 综合paper数据行
**结果**: Insert 34.6 MEPS, BFS 1.35x, PR 1.58x, Search 84.3ns/op

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

## 新一轮开发进度 (第1位Claude, 2026-06-08)

### 第1位Claude（已完成）: M157-M158
SOTA baseline comparison — upstream全覆盖实验 (17/17 pass)
- 覆盖所有121个upstream文件
- 20%算法改动: direction-optimized BFS, tier-weighted PR, delta-stepping SSSP, path-halving WCC
- 密集breakpoint调试: BREAKPOINT_DUMP + per-tier access counters
- Baseline对比: Philemon TieredCSR vs naive CSR (BFS 0.77x, PR 1.46x)
- ags1 runner: experiment/run_m157_m158.sh (scale 14/16/18/20)

### 第2位Claude: M159-M160
ags1数据收集 + LaTeX table行产出
- 在ags1 (128核, 1.5TB RAM, H100) 上运行multi-scale实验
- Table 1: Philemon vs CSR (latency+吞吐)
- Table 2: 三层tier分布统计

### 第3位Claude: M161-M162
真实数据集实验 (email-Enron, wiki-Vote)
- SNAP数据下载+预处理
- Table 3: 真实图性能对比

### 第4位Claude: M163-M164
并发读写隔离 (RQ3)
- Writer insert/delete + Reader BFS/PR
- 延迟增幅测量

### 第5位Claude: M165-M166
Ablation study (RQ5)
- 6种tier配置对比
- Figure 4数据

### 第6位Claude: M167-M168
论文数据整合 + LaTeX最终版
- 汇总所有实验数据
- Table 1-5 + Figure 1-4
- 回归测试验证reproducibility

## Phase 4: M169-M180 — 大规模SOTA实验 + 论文数据填充

目标: 在ags1服务器上运行大规模实验, 产出超越SOTA的数据填入论文LaTeX表格。
参考: github.com/google-deepmind/alphaproof-nexus-results (Lean proof验证结构)

### 第1位Claude: M169-M170 (当前)
Driver Workload Engine — upstream driver.h(1577行) 全覆盖 + SOTA对比
- 移植: driver.h + wrapper.h + 6 backend adapters + 6 algo wrappers + main.cpp (6972行)
- [MOD 20%]: tier-aware batch sizing, direction-optimized BFS, tier-weighted PR, weighted UnionFind
- SOTA对比: Philemon TieredCSR vs CSR baseline, 校准 vs RapidStore/Sortledton/Teseo/LiveGraph/Aspen
- 输出: experiment/results/m169_paper_data.csv → 填入论文 Table 1-2
- ags1 runner: experiment/run_m169_m170.sh (multi-scale 14→24)
- 状态: 17/17 PASS ✓

### 第2位Claude: M171-M172
NeoGraph核心引擎实验 — upstream NeoGraph库(18920行)深度覆盖
- 覆盖: c_art + art_new + ART insert/delete/search + COW操作
- [MOD 20%]: tier-aware ART node layout (hot inner nodes→DRAM, cold leaves→SSD)
- 实验: ART操作延迟 vs std::map vs B-tree baseline
- 输出: experiment/results/m171_neograph_data.csv → 填入论文 Table 3 (index structure)

### 第3位Claude: M173-M174
大规模图实验 — LiveJournal + Twitter + uk-2007 真实数据集
- 下载SNAP/LAW数据集, 预处理为edge-stream格式
- 运行: BFS/PR/SSSP/WCC at scale 20-26 on ags1
- SOTA对比: 直接vs RapidStore VLDB'25 Table 3 published数据
- 输出: experiment/results/m173_largescale.csv → 填入论文 Table 1 (main comparison)

### 第4位Claude: M175-M176 ✅ DONE
Tiered Memory实验 — RQ2+RQ4+RQ6 HBM/GDDR/DRAM/SSD分层验证
- 4-tier模型: HBM(deg>256)/GDDR(deg>32)/DRAM(deg>4)/SSD(rest)
- BFS: tier-priority frontier展开, PR: lazy SSD accumulation (LAZY_K=4)
- MigrationEngine: pipelined overlap (upstream: stop-world)
- 18/18 PASS: BFS slowdown 1.10-1.41x, PR 1.68-2.06x
- Tier分布: HBM~21%, GDDR~36%, DRAM~27%, SSD~12%
- 输出: experiment/results/m175_tiered_memory.csv → 论文 Table 2 (e2e)
         experiment/results/m175_tiered_memory.tex → Table 2a+2b LaTeX

### 第5位Claude: M177-M178
Streaming + Compaction实验 — RQ5 流式写入+压缩
- 持续写入下selection延迟trace
- Compaction spike测量 (0.24-0.28ms target)
- Segment growth analysis
- 输出: experiment/results/m177_streaming.csv → 填入论文 Figure (latency trace)

### 第6位Claude: M179-M180
论文最终数据整合 + LaTeX表格更新
- 汇总M169-M178所有CSV数据
- 更新philemon_tsh_reconstructed.tex中的Table 1-5
- 添加缺失的Figure数据
- 全量回归测试
- 输出: 最终论文版本 ready for submission

### 服务器实验流程
1. ags1运行 experiment/run_m*.sh → 产出CSV到 experiment/results/
2. git push到main分支
3. 子Claude pull main分支获取最新数据
4. 子Claude基于数据更新论文LaTeX

### SOTA目标 (LiveJournal, 128线程)
| System       | Insert MEPS | BFS(s) | PR 10iter(s) | Memory(GB) |
|-------------|------------|--------|-------------|-----------|
| Philemon    | ≥2.0       | ≤30    | ≤300        | ≤2.0(+SSD)|
| RapidStore  | ~2.5       | ~25    | ~295        | ~6.2      |
| Sortledton  | ~3.0       | ~25    | ~499        | ~3.7      |
| Teseo       | ~1.5       | ~49    | ~295        | ~5.3      |
| LiveGraph   | ~0.8       | ~69    | ~997        | ~7.0      |
| Aspen       | ~1.2       | ~25    | ~517        | ~28.3     |

Philemon优势: 1/3 memory + SSD 达到纯DRAM系统90%性能
