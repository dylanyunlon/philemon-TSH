#!/usr/bin/env bash
# ================================================================
#  multi_model_bench.sh — opus/sonnet/haiku 三模型benchmark对比
#
#  cookie来源(优先级):
#    1. /tmp/claude_hk_cookie.txt
#    2. ./claude-hk-config/cookie.txt (会自动git pull)
#    3. 环境变量 CLAUDE_HK_COOKIE
#
#  用法:
#    REPO_URL=dylanyunlon/walpurgis-WTFGG bash multi_model_bench.sh
#    MODELS=claude-sonnet-4-6 REPO_URL=dylanyunlon/xxx bash multi_model_bench.sh
# ================================================================
set -euo pipefail

REPO_URL="${REPO_URL:?设置REPO_URL, 如 dylanyunlon/walpurgis-WTFGG}"
REPO_NAME="${REPO_NAME:-$(basename $REPO_URL)}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS="${MODELS:-claude-sonnet-4-6 claude-haiku-4-5 claude-opus-4-6}"
OUT="bench_results"; mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S)

R='\033[0m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'

# ── 自动同步cookie ──────────────────────────────────────────
sync_cookie() {
  local CONFIG_DIR="/tmp/claude-hk-config"
  
  # 已有本地cookie且不超过24小时就跳过
  if [ -f /tmp/claude_hk_cookie.txt ]; then
    local AGE=$(($(date +%s) - $(stat -c %Y /tmp/claude_hk_cookie.txt 2>/dev/null || echo 0)))
    if [ "$AGE" -lt 86400 ]; then
      echo -e "${G}cookie有效 (${AGE}s old)${R}"
      return 0
    fi
  fi

  echo -ne "${Y}同步cookie... ${R}"
  if [ -d "$CONFIG_DIR" ]; then
    git -C "$CONFIG_DIR" pull -q 2>/dev/null || true
  else
    git clone --depth=1 -q https://github.com/dylanyunlon/claude-hk-config.git "$CONFIG_DIR" 2>/dev/null || true
  fi

  # 从cookie.txt或raw_curl.txt提取
  if [ -f "$CONFIG_DIR/cookie.txt" ]; then
    local CK=$(cat "$CONFIG_DIR/cookie.txt" | head -1)
    if [ -n "$CK" ] && [ "$CK" != "#"* ]; then
      echo "$CK" > /tmp/claude_hk_cookie.txt
      echo -e "${G}✓${R}"
      return 0
    fi
  fi

  if [ -f "$CONFIG_DIR/raw_curl.txt" ]; then
    python3 "$CONFIG_DIR/extract.py" 2>/dev/null && \
      cp "$CONFIG_DIR/cookie.txt" /tmp/claude_hk_cookie.txt && \
      echo -e "${G}✓ (from raw_curl)${R}" && return 0
  fi

  echo -e "${Y}使用现有cookie${R}"
}

# ── 主流程 ──────────────────────────────────────────────────
echo -e "${Y}Multi-Model Benchmark: ${C}${REPO_URL}${R}"
echo -e "Models: ${C}${MODELS}${R}"
echo ""

sync_cookie

for MODEL in $MODELS; do
  TAG="${REPO_NAME}_${MODEL}_${TS}"
  OUT_FILE="$OUT/${TAG}.json"
  echo -ne "${G}[${MODEL}]${R} "

  timeout 360 python3 "${SCRIPT_DIR}/bench_one_model.py" \
    "$MODEL" "$REPO_URL" "$OUT_FILE" 300 2>&1 || echo "timeout"

  sleep 2
done

echo ""
echo -e "${Y}═══ 对比分析 ═══${R}"

python3 << PYEOF
import json, os

ts = "${TS}"
repo = "${REPO_NAME}"
models = "${MODELS}".split()
results = {}

for m in models:
    path = f"${OUT}/{repo}_{m}_{ts}.json"
    if os.path.exists(path):
        results[m] = json.load(open(path))

print(f"\n  {'Model':20s} {'Tools':>6s} {'Text':>7s} {'JSON':>5s} {'Time':>6s}")
print("  " + "-" * 50)
for m in models:
    if m in results:
        d = results[m]
        tc = d.get("num_tool_calls", 0)
        tl = len(d.get("text", ""))
        js = "✓" if "benchmark_data" in d else "✗"
        el = d.get("elapsed_s", "?")
        print(f"  {m:20s} {tc:>6d} {tl:>6d}c {js:>5s} {str(el):>5s}s")
    else:
        print(f"  {m:20s}   MISSING")

print(f"""
  推荐:
    sonnet-4-6: 生产级benchmark (tool call最多, JSON完整)
    haiku-4-5:  快速冒烟/quota紧张
    opus-4-6:   深度审计 (分析最彻底但慢)
""")

comparison = {"timestamp": ts, "repo": repo, "models": {}}
for m in models:
    if m in results:
        d = results[m]
        comparison["models"][m] = {
            "tool_calls": d.get("num_tool_calls", 0),
            "text_len": len(d.get("text", "")),
            "has_json": "benchmark_data" in d,
            "elapsed_s": d.get("elapsed_s"),
            "benchmark_data": d.get("benchmark_data"),
        }
cpath = f"${OUT}/comparison_{repo}_{ts}.json"
json.dump(comparison, open(cpath, "w"), indent=2, ensure_ascii=False)
print(f"  Saved: {cpath}")
PYEOF
