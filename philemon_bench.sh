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
CK='lastActiveOrg=0de6831b-fb77-41c7-bfb9-0899fb74f90f; CH-prefers-color-scheme=dark; g_state={"i_l":0,"i_ll":1778115570300,"i_b":"lqbhq/LibeNDVywiN36ev6CfJb/M9byi5v9sTOB0beg","i_e":{"enable_itp_optimization":0},"i_et":1778115570300,"i_t":1778201970306}; ajs_anonymous_id=claudeai.v1.d6ccc63a-21ab-47e6-9214-d6edfc518c7e; user-sidebar-pinned=false; share-session=6rrqz00r1ffzg0dizxnz0rb0zn63t3e7; lastActiveOrg=0de6831b-fb77-41c7-bfb9-0899fb74f90f; user-sidebar-visible-on-load=false; _dd_s=aid=5212c48b-ec8f-488b-929d-77b816ba6d67&rum=2&id=cb196ab3-7017-41c2-a1a3-96fd887bc5f1&created=1780623403660&expire=1780624622174'
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'
BASE="https://claude.hk.cn/api/organizations/${ORG}"

OUT="bench_results"; mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S)

R='\033[0m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'

# ── 发送函数(支持code execution) ────────────────────────────
bench_one() {
  local TAG="$1"
  local PROMPT_FILE="$2"
  local OUT_FILE="$OUT/${TAG}_${TS}.json"

  echo -ne "${G}[${TAG}]${R} "

  # 创建conversation
  local CONV_ID
  CONV_ID=$(curl -sf --max-time 15 "${BASE}/chat_conversations" \
    -X POST -H 'content-type: application/json' -b "$CK" \
    -H 'origin: https://claude.hk.cn' -H "user-agent: $UA" \
    --data-raw '{"name":"","project_uuid":null,"model":null}' \
    | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["uuid"])')

  [[ -z "$CONV_ID" ]] && { echo "conv创建失败"; return 1; }

  # 构建payload（关键: rendering_mode=messages + create_conversation_params）
  local H_UUID A_UUID
  H_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')
  A_UUID=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')

  python3 << PYEOF
import json
prompt = open("$PROMPT_FILE").read()
payload = {
    "prompt": prompt,
    "timezone": "Asia/Shanghai",
    "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
        "nameKey":"normal_style_name","prompt":"Normal\n",
        "summary":"Default responses from Claude",
        "summaryKey":"normal_style_summary","isDefault":True}],
    "locale": "en-US",
    "model": "${MODEL}",
    "effort": "${EFFORT}",
    "thinking_mode": "off",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {
        "human_message_uuid": "$H_UUID",
        "assistant_message_uuid": "$A_UUID"
    },
    "attachments": [],
    "files": [],
    "sync_sources": [],
    "rendering_mode": "messages",
    "create_conversation_params": {
        "name": "",
        "model": "${MODEL}",
        "include_conversation_preferences": True,
        "paprika_mode": None,
        "compass_mode": None,
        "tool_search_mode": "auto",
        "is_temporary": False,
        "enabled_imagine": True
    }
}
open("/tmp/_bench_payload.json","w").write(json.dumps(payload))
PYEOF

  # 发送
  curl -sf --max-time "$TIMEOUT" \
    "${BASE}/chat_conversations/${CONV_ID}/completion" \
    -H 'accept: text/event-stream' \
    -H 'content-type: application/json' \
    -H 'anthropic-client-platform: web_claude_ai' \
    -b "$CK" \
    -H 'origin: https://claude.hk.cn' \
    -H 'referer: https://claude.hk.cn/new' \
    -H "user-agent: $UA" \
    -d @/tmp/_bench_payload.json > /tmp/_bench_sse.txt 2>&1

  # 解析SSE: 提取text + tool_result
  python3 << PYEOF
import json, re

text_parts = []
tool_results = []
code_blocks = []

for line in open("/tmp/_bench_sse.txt"):
    line = line.strip()
    if not line.startswith("data: "): continue
    try:
        d = json.loads(line[6:])
    except: continue

    t = d.get("type", "")

    # text内容
    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta":
            text_parts.append(delta.get("text", ""))

    # tool结果（code execution输出）
    if t == "content_block_start":
        cb = d.get("content_block", {})
        if cb.get("type") == "tool_result":
            dc = cb.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    result = json.loads(dc["json_block"])
                    tool_results.append(result)
                except: pass

    # tool调用的代码
    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "tool_use_block_update_delta":
            dc = delta.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    code_info = json.loads(dc["json_block"])
                    code_blocks.append(code_info)
                except: pass

full_text = "".join(text_parts)

# 保存完整结果
result = {
    "text": full_text,
    "tool_results": tool_results,
    "code_executed": code_blocks,
    "num_tool_calls": len(tool_results)
}

# 尝试从text中提取JSON
parsed_json = None
for attempt_fn in [
    lambda: json.loads(full_text.strip()),
    lambda: json.loads(re.search(r'\x60\x60\x60(?:json)?\s*\n(.*?)\n\x60\x60\x60', full_text, re.DOTALL).group(1)),
    lambda: json.loads(re.search(r'\{[^{}]*("targets"|"results"|"models"|"baseline_MAE")[^}]*\}', full_text, re.DOTALL).group(0)),
]:
    try:
        parsed_json = attempt_fn()
        break
    except: pass

if parsed_json:
    result["benchmark_data"] = parsed_json

# 从tool_results里提取stdout（可能包含benchmark数据）
for tr in tool_results:
    stdout = tr.get("stdout", "")
    if stdout:
        result["exec_stdout"] = result.get("exec_stdout", "") + stdout

json.dump(result, open("$OUT_FILE", "w"), indent=2, ensure_ascii=False)

n_tools = len(tool_results)
n_text = len(full_text)
has_json = "benchmark_data" in result
print(f"tools={n_tools}, text={n_text}c, json={'✓' if has_json else '✗'}")
PYEOF

  # 清理conversation
  curl -sf -X DELETE "${BASE}/chat_conversations/${CONV_ID}" \
    -b "$CK" -H 'origin: https://claude.hk.cn' -H "user-agent: $UA" > /dev/null 2>&1 || true
}

# ── Prompts ───────────────────────────────────────────────────

# Prompt 1: Clone + 分析代码结构 + 输出baseline
cat > /tmp/_p1.txt << 'PROMPTEOF'
不要立刻查看所有内容,在你的linux上用tree、git branch 先查看架构. 使用clone工具进行git clone。没有tree你就apt install tree。 看看这个项目(upstream文件夹)关于代码移植的问题，我们需要每一个文件的每一行都用上。github.com/dylanyunlon/philemon-TSH 鲁迅那样的拿法，别全权复制啊。我的意思是在mv的基础上，动态修改算法的20%的内容就行了。注意多写一点关于断点调试（或者print当前所有数据、结构体状态）的内容, 让我们在运行实验的时候能像现实世界开发一样得到反馈

分析完代码后输出:
1. src/ vs upstream/ 的代码行数对比(C++/CUDA/h文件)
2. 逐文件改动行数统计(top 10)
3. 异构GPU存储迁移/GNN训练的published baselines(QPS, migration latency, memory utilization)
4. 基于真实代码改动估计每个改动对性能的影响
5. 相关领域SOTA系统排行(RapidStore, TempGraph, Hetero-Mark等)

最后汇总成JSON:
```json
{
  "code_analysis": {"src_lines": ..., "upstream_lines": ..., "top_changes": [{"file":"...","added":...,"removed":...}]},
  "baselines": {"metrics": {"qps": ..., "migration_latency_ms": ..., "memory_util_pct": ...}},
  "modifications_impact": [{"name":"...","file":"...","lines_changed":...,"estimated_improvement_pct":...}],
  "sota": [{"name":"...","year":...,"qps":...}],
  "expected_improvement": "..."
}
```
PROMPTEOF

# ── 执行 ──────────────────────────────────────────────────────
echo -e "${Y}Walpurgis Bench v3 — 远端Code Execution + Git Clone${R}"
echo -e "Model: ${C}${MODEL}${R}  Effort: ${C}${EFFORT}${R}  Timeout: ${TIMEOUT}s"
echo ""

bench_one "full_analysis" "/tmp/_p1.txt"

echo ""
echo -e "${Y}结果:${R} $OUT/full_analysis_${TS}.json"
echo ""

# 摘要
python3 << PYEOF
import json, glob

files = sorted(glob.glob("$OUT/full_analysis_${TS}.json"))
if not files:
    print("No results found")
    exit(0)

d = json.load(open(files[-1]))
print(f"Tool calls: {d.get('num_tool_calls', 0)}")
print(f"Text length: {len(d.get('text', ''))}")
print(f"Code blocks: {len(d.get('code_executed', []))}")

if d.get('exec_stdout'):
    print(f"\n=== Execution Output ===")
    print(d['exec_stdout'][:1000])

if d.get('benchmark_data'):
    print(f"\n=== Benchmark Data ===")
    print(json.dumps(d['benchmark_data'], indent=2, ensure_ascii=False)[:1000])
else:
    print(f"\n=== Text Response ===")
    print(d.get('text', '')[:1000])
PYEOF
