#!/usr/bin/env bash
# ============================================================
#  初始化 claude-hk-config 仓库
#  用法: bash claude_hk_config_init.sh
#  
#  创建后push到 github.com/dylanyunlon/claude-hk-config
# ============================================================
set -euo pipefail

mkdir -p claude-hk-config
cd claude-hk-config
git init
git config user.name "dylanyunlon"
git config user.email "dogechat@163.com"

# 说明文件
cat > README.md << 'READMEEOF'
# claude-hk-config

claude.hk.cn API 配置中心。每天登录后更新 `raw_curl.txt`，
所有 bench 脚本自动从这里读取最新 cookie。

## 更新流程

1. 在浏览器打开 claude.hk.cn，登录
2. F12 → Network → 找任意一个 completion 请求 → 右键 Copy as cURL
3. 粘贴到 `raw_curl.txt`（覆盖全部内容）
4. `git add . && git commit -m "cookie $(date +%Y%m%d)" && git push`

## 文件

- `raw_curl.txt` — 浏览器抓取的完整 curl 命令
- `cookie.txt` — 提取出的纯 cookie 字符串（由 extract.py 自动生成）
- `extract.py` — 从 raw_curl.txt 提取 cookie 的脚本
READMEEOF

# cookie提取脚本
cat > extract.py << 'EXTRACTEOF'
#!/usr/bin/env python3
"""从 raw_curl.txt 提取 cookie 并写入 cookie.txt"""
import re, sys, os

raw_path = os.path.join(os.path.dirname(__file__), "raw_curl.txt")
cookie_path = os.path.join(os.path.dirname(__file__), "cookie.txt")

if not os.path.exists(raw_path):
    print("raw_curl.txt not found", file=sys.stderr)
    sys.exit(1)

text = open(raw_path).read()

# 提取 -b '...' 或 -b "..." 里的cookie
m = re.search(r"-b\s+['\"]([^'\"]+)['\"]", text)
if not m:
    # 也试 --cookie
    m = re.search(r"--cookie\s+['\"]([^'\"]+)['\"]", text)

if m:
    cookie = m.group(1)
    open(cookie_path, "w").write(cookie.strip() + "\n")
    print(f"Extracted cookie ({len(cookie)} chars) -> cookie.txt")
else:
    print("No cookie found in raw_curl.txt", file=sys.stderr)
    sys.exit(1)
EXTRACTEOF

# 放一个初始的 raw_curl.txt（空模板）
cat > raw_curl.txt << 'RAWEOF'
# 把浏览器抓取的完整 curl 命令粘贴到这里（替换这段注释）
# F12 → Network → 找 completion 请求 → 右键 Copy as cURL (bash)
RAWEOF

touch cookie.txt
echo "# 由 extract.py 自动生成" > cookie.txt

git add .
git commit -m "init: claude-hk-config cookie管理中心"

echo ""
echo "完成。接下来:"
echo "  1. gh repo create dylanyunlon/claude-hk-config --public --source=. --push"
echo "  2. 或者 git remote add origin git@github.com:dylanyunlon/claude-hk-config.git && git push -u origin main"
