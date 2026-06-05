#!/usr/bin/env bash
# ================================================================
#  philemon_bench.sh — dylanyunlon/philemon-TSH benchmark via remote code execution
#  
#  让远端Claude:
#    1) git clone philemon-TSH
#    2) tree/diff/grep分析真实代码
#    3) 基于真实代码结构输出benchmark数字
#    4) 每个prompt独立新对话
# ================================================================
set -euo pipefail

ORG="0de6831b-fb77-41c7-bfb9-0899fb74f90f"
MODEL="${MODEL:-claude-sonnet-4-6}"
EFFORT="${EFFORT:-high}"
TIMEOUT="${TIMEOUT:-300}"  # code execution需要更长

# 重要: 用完整cookie
