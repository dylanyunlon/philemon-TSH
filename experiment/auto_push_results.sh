#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# auto_push_results.sh — 实验完成后自动 commit + push 到 GitHub
# M183: 自动化实验 → git push 流水线
#
# 用法:
#   bash experiment/auto_push_results.sh                    # push 所有新结果
#   bash experiment/auto_push_results.sh "RQ1 index data"   # 自定义 commit message
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

MSG="${1:-experiment: auto-push results $(date +%Y%m%d_%H%M%S)}"

# 确保 git 配置正确
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# 设置远程 URL (带 token)
REMOTE_URL=$(git remote get-url origin)
if [[ "${REMOTE_URL}" != *"ghp_"* ]]; then
    # Token should be set via: export GH_TOKEN=ghp_xxx
    # or: git remote set-url origin https://$GH_TOKEN@github.com/dylanyunlon/philemon-TSH.git
    if [ -n "${GH_TOKEN:-}" ]; then
        git remote set-url origin "https://${GH_TOKEN}@github.com/dylanyunlon/philemon-TSH.git"
    else
        echo "WARNING: GH_TOKEN not set. Push may fail without authentication."
    fi
fi

# 添加所有结果文件
git add -A experiment/results/
git add -A experiment/logs/
git add -A philemon/*.tex

# 检查是否有变更
if git diff --cached --quiet; then
    echo "No changes to push."
    exit 0
fi

# Commit and push
git commit -m "${MSG}"
git push origin main

echo "✓ Results pushed to github.com/dylanyunlon/philemon-TSH"
echo "  Commit: $(git rev-parse --short HEAD)"
echo "  Message: ${MSG}"
