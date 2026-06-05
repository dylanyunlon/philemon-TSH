#!/usr/bin/env bash
# ============================================================
#  claude_hk_chat.sh — claude.hk.cn 交互式 CLI
#  支持 code execution / web search / artifacts
#  用法: bash claude_hk_chat.sh [选项]
# ============================================================
set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────
ORG="0de6831b-fb77-41c7-bfb9-0899fb74f90f"
BASE="https://claude.hk.cn/api/organizations/${ORG}"
MODEL="${MODEL:-claude-sonnet-4-6}"
EFFORT="${EFFORT:-high}"
THINKING="${THINKING:-off}"
TIMEOUT="${TIMEOUT:-300}"

COOKIES='lastActiveOrg=0de6831b-fb77-41c7-bfb9-0899fb74f90f; CH-prefers-color-scheme=dark; g_state={"i_l":0,"i_ll":1778115570300,"i_b":"lqbhq/LibeNDVywiN36ev6CfJb/M9byi5v9sTOB0beg","i_e":{"enable_itp_optimization":0},"i_et":1778115570300,"i_t":1778201970306}; ajs_anonymous_id=claudeai.v1.d6ccc63a-21ab-47e6-9214-d6edfc518c7e; user-sidebar-pinned=false; share-session=6rrqz00r1ffzg0dizxnz0rb0zn63t3e7; lastActiveOrg=0de6831b-fb77-41c7-bfb9-0899fb74f90f; user-sidebar-visible-on-load=false; _dd_s=aid=5212c48b-ec8f-488b-929d-77b816ba6d67&rum=2&id=cb196ab3-7017-41c2-a1a3-96fd887bc5f1&created=1780623403660&expire=1780624622174'
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'

# ── 当前conversation (新建或复用) ──────────────────────────────
CONV_ID=""

# ── 颜色 ─────────────────────────────────────────────────────
R='\033[0m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'; D='\033[90m'

# ── 创建新conversation ───────────────────────────────────────
new_conversation() {
  CONV_ID=$(curl -sf --max-time 15 "${BASE}/chat_conversations" \
    -X POST -H 'content-type: application/json' -b "$COOKIES" \
    -H 'origin: https://claude.hk.cn' -H "user-agent: $UA" \
    --data-raw '{"name":"","project_uuid":null,"model":null}' \
    | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["uuid"])')
  echo -e "${Y}新对话: ${D}${CONV_ID}${R}"
}

# ── 发送请求 ──────────────────────────────────────────────────
LAST_RAW=""

send_prompt() {
  local prompt="$1"

  # 构建payload（带完整tools声明）
  local h_uuid a_uuid
  h_uuid=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')
  a_uuid=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')

  python3 << PYEOF
import json
prompt = """${prompt}"""
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
    "thinking_mode": "${THINKING}",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {
        "human_message_uuid": "${h_uuid}",
        "assistant_message_uuid": "${a_uuid}"
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
open("/tmp/_chat_payload.json","w").write(json.dumps(payload))
PYEOF

  LAST_RAW=$(curl -sf --max-time "$TIMEOUT" \
    "${BASE}/chat_conversations/${CONV_ID}/completion" \
    -H 'accept: text/event-stream' \
    -H 'content-type: application/json' \
    -H 'anthropic-client-platform: web_claude_ai' \
    -b "$COOKIES" \
    -H 'origin: https://claude.hk.cn' \
    -H 'referer: https://claude.hk.cn/new' \
    -H "user-agent: $UA" \
    -d @/tmp/_chat_payload.json 2>&1)

  # 解析SSE: text + tool_use + tool_result
  echo "$LAST_RAW" | python3 << 'PYEOF'
import sys, json

text_parts = []
tool_calls = []
tool_results = []
limit_info = ""

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    try: d = json.loads(line[6:])
    except: continue

    t = d.get("type", "")

    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta":
            text_parts.append(delta["text"])
        elif delta.get("type") == "input_json_delta":
            pass  # tool input streaming
        elif delta.get("type") == "tool_use_block_update_delta":
            msg = delta.get("message", "")
            dc = delta.get("display_content")
            if dc and dc.get("type") == "json_block":
                try:
                    info = json.loads(dc["json_block"])
                    code = info.get("code", "")
                    lang = info.get("language", "")
                    if code:
                        tool_calls.append(f"[{lang}] {code[:200]}")
                except: pass

    elif t == "content_block_start":
        cb = d.get("content_block", {})
        if cb.get("type") == "tool_result":
            dc = cb.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    result = json.loads(dc["json_block"])
                    stdout = result.get("stdout", "")
                    stderr = result.get("stderr", "")
                    rc = result.get("returncode", "?")
                    if stdout: tool_results.append(stdout.rstrip())
                    if stderr and rc != 0: tool_results.append(f"[stderr] {stderr.rstrip()}")
                except: pass

    elif t == "message_limit":
        ml = d.get("message_limit", {})
        windows = ml.get("windows", {})
        parts = []
        for k, v in windows.items():
            u = v.get("utilization")
            if u is not None:
                parts.append(f"{k}: {u:.0%}")
        if parts:
            limit_info = " | ".join(parts)

# 输出tool执行过程
if tool_calls:
    print(f"\033[90m  [{len(tool_calls)} tool calls]\033[0m")
if tool_results:
    for r in tool_results:
        for line in r.split("\n")[:15]:
            print(f"\033[90m  > {line}\033[0m")
        if r.count("\n") > 15:
            print(f"\033[90m  > ... ({r.count(chr(10))} lines)\033[0m")

# 输出文本
text = "".join(text_parts)
if text.strip():
    print(f"\033[36mClaude:\033[0m")
    print(text)

if limit_info:
    print(f"\033[90m  [{limit_info}]\033[0m")

print()
PYEOF
}

# ── 帮助 ─────────────────────────────────────────────────────
usage() {
  cat << HELPEOF
${Y}claude_hk_chat.sh${R} — claude.hk.cn CLI (code exec enabled)

${G}环境变量:${R}
  MODEL     模型名     (默认: claude-sonnet-4-6)
  EFFORT    effort级别  (low/medium/high/max, 默认: high)
  THINKING  思考模式    (off/on, 默认: off)
  TIMEOUT   超时秒数    (默认: 300)

${G}交互命令:${R}
  /new             新建对话(清空上下文)
  /model <name>    切换模型
  /effort <level>  切换effort
  /think on|off    切换思考模式
  /status          显示当前配置
  /raw             显示上一次原始SSE
  /help            显示帮助
  /quit            退出

${G}能力:${R}
  code execution   远端sandbox里执行bash/python
  web search       搜索互联网
  artifacts        创建文件/代码

${G}示例:${R}
  bash claude_hk_chat.sh
  MODEL=claude-opus-4-6 EFFORT=max bash claude_hk_chat.sh
HELPEOF
}

# ── 主循环 ─────────────────────────────────────────────────────
main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage; exit 0
  fi

  echo -e "${Y}╔══════════════════════════════════════════════╗${R}"
  echo -e "${Y}║  claude.hk.cn CLI (code exec enabled)        ║${R}"
  echo -e "${Y}║  模型: ${C}${MODEL}${Y}                     ║${R}"
  echo -e "${Y}║  effort: ${C}${EFFORT}${Y}   thinking: ${C}${THINKING}${Y}           ║${R}"
  echo -e "${Y}║  /help 查看命令   /new 新建对话              ║${R}"
  echo -e "${Y}╚══════════════════════════════════════════════╝${R}"

  new_conversation

  while true; do
    echo -ne "${G}你: ${R}"
    IFS= read -r input || break
    [[ -z "$input" ]] && continue

    case "$input" in
      /quit|/exit|/q)
        echo -e "${Y}再见${R}"; break ;;
      /help|/h)
        usage ;;
      /new)
        new_conversation ;;
      /model\ *)
        MODEL="${input#/model }"
        echo -e "${Y}模型: ${C}${MODEL}${R}" ;;
      /effort\ *)
        EFFORT="${input#/effort }"
        echo -e "${Y}effort: ${C}${EFFORT}${R}" ;;
      /think\ *)
        THINKING="${input#/think }"
        echo -e "${Y}thinking: ${C}${THINKING}${R}" ;;
      /status)
        echo -e "  模型:     ${C}${MODEL}${R}"
        echo -e "  effort:   ${C}${EFFORT}${R}"
        echo -e "  thinking: ${C}${THINKING}${R}"
        echo -e "  conv:     ${D}${CONV_ID}${R}" ;;
      /raw)
        echo -e "${D}${LAST_RAW}${R}" ;;
      *)
        send_prompt "$input" ;;
    esac
  done
}

main "$@"
