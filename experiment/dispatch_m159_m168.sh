#!/bin/bash
# 子Claude调度器 — M159-M168任务分配
# 通过 claude_hk_chat.sh 调用 Opus 4.6 执行后续里程碑
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CHAT="${REPO_DIR}/claude_hk_chat.sh"

# 拉取最新cookie
if [ -d /tmp/claude-hk-config ]; then
    git -C /tmp/claude-hk-config pull -q 2>/dev/null || true
else
    git clone --depth=1 -q https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config 2>/dev/null || true
fi
[ -f /tmp/claude-hk-config/cookie.txt ] && cp /tmp/claude-hk-config/cookie.txt /tmp/claude_hk_cookie.txt

export MODEL="claude-opus-4-6"
export EFFORT="medium"

PROMPT_TEMPLATE='你是philemon-TSH第%d位Claude (Opus 4.6)。
仓库: github.com/dylanyunlon/philemon-TSH
任务: M%03d-M%03d

规则: main分支, 不开新分支, 不用v2/port/old等后缀, 改算法核心20%%
编译通过+测试PASS后 git push origin main
author: dylanyunlon <dogechat@163.com>
token: 使用环境变量 $PHILEMON_TOKEN (从 claude-hk-config/token.txt 读取)

读 CLAUDE_DEVELOPMENT_PLAN.md 了解全局
读 experiment/logs/ 了解前一位的运行结果
读 experiment/m157_m158_sota_baseline_experiment.cpp 了解SOTA baseline对比格式

你的具体任务:
%s

截断时用户发 Continue 即可继续。'

# 任务定义
declare -A TASKS
TASKS[2]="M159-M160: 在ags1上运行M157-M158实验, 收集scale 14/16/18/20的数据, 产出LaTeX tabular行:
  - Table 1: Philemon vs CSR baseline (BFS/PR/SSSP/WCC latency+吞吐)
  - Table 2: 三层tier分布统计 (DRAM/SSD/HDD edge count + access ratio)
  - 数据push到 experiment/results/m159_m160_paper_data.csv"

TASKS[3]="M161-M162: 真实数据集实验 (email-Enron 36K vertices, wiki-Vote 7K vertices):
  - 从SNAP下载并预处理为edge list
  - 跑Philemon tiered vs CSR baseline BFS+PR
  - 产出Table 3行: 真实图上的性能对比"

TASKS[4]="M163-M164: 并发读写隔离实验 (RQ3):
  - Writer线程持续insert/delete, Reader线程跑BFS/PR
  - 测量reader latency增幅 (<15% target)
  - scale 14→18, writer_ratio 0.1/0.3/0.5"

TASKS[5]="M165-M166: Ablation study (RQ5):
  - 6种配置: all-DRAM, hot-only, hot+warm, full-tiered, +prefetch, +compaction
  - 对比每种配置的BFS/PR latency和内存占用
  - 产出Figure 4数据: grouped bar chart"

TASKS[6]="M167-M168: 论文数据整合 + LaTeX表格最终版:
  - 汇总M157-M166所有数据
  - 生成philemon_tsh.tex的Table 1-5 + Figure 1-4的数据行
  - 回归测试: 所有实验重跑验证reproducibility"

echo "=== Philemon-TSH 子Claude调度 ==="
echo "Model: $MODEL | Effort: $EFFORT"
echo ""

for CLAUDE_NUM in 2 3 4 5 6; do
    TASK="${TASKS[$CLAUDE_NUM]}"
    M_START=$((157 + (CLAUDE_NUM-2)*2))
    M_END=$((M_START + 1))
    PROMPT=$(printf "$PROMPT_TEMPLATE" "$CLAUDE_NUM" "$M_START" "$M_END" "$TASK")
    
    echo "── 第${CLAUDE_NUM}位Claude: M${M_START}-M${M_END} ──"
    echo "$PROMPT" | head -3
    echo "..."
    echo ""
    
    # 只打印计划，实际执行需要去掉下面的 echo
    echo "[DRY RUN] Would dispatch to claude_hk_chat.sh"
    # 实际执行: echo "$PROMPT" | MODEL=$MODEL EFFORT=$EFFORT bash "$CHAT"
done

echo ""
echo "调度计划完成。去掉DRY RUN后实际执行。"
