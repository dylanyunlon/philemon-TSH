#!/usr/bin/env bash
# ============================================================
#  claude_hk_chat.sh — claude.hk.cn 交互式CLI
#  cookie自动从 /tmp/claude_hk_cookie.txt 或 claude-hk-config repo 读取
# ============================================================
set -euo pipefail

ORG="0de6831b-fb77-41c7-bfb9-0899fb74f90f"
BASE="https://claude.hk.cn/api/organizations/${ORG}"
MODEL="${MODEL:-claude-sonnet-4-6}"
EFFORT="${EFFORT:-high}"
THINKING="${THINKING:-off}"
TIMEOUT="${TIMEOUT:-300}"
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36'

# ── cookie读取 ───────────────────────────────────────────────
load_cookie() {
  if [ -f /tmp/claude_hk_cookie.txt ]; then
    COOKIES=$(head -1 /tmp/claude_hk_cookie.txt)
    [ -n "$COOKIES" ] && return 0
  fi
  # 从config repo
  local CFG="/tmp/claude-hk-config"
  if [ -d "$CFG" ]; then
    git -C "$CFG" pull -q 2>/dev/null || true
  else
    git clone --depth=1 -q https://github.com/dylanyunlon/claude-hk-config.git "$CFG" 2>/dev/null || true
  fi
  if [ -f "$CFG/cookie.txt" ]; then
    COOKIES=$(head -1 "$CFG/cookie.txt")
    echo "$COOKIES" > /tmp/claude_hk_cookie.txt
  fi
  if [ -z "${COOKIES:-}" ]; then
    echo "ERROR: No cookie. 把curl里的cookie写入 /tmp/claude_hk_cookie.txt"
    exit 1
  fi
}

COOKIES=""
load_cookie

CONV_ID=""
R='\033[0m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'; D='\033[90m'
LAST_RAW=""

new_conversation() {
  CONV_ID=$(curl -sf --max-time 15 "${BASE}/chat_conversations" \
    -X POST -H 'content-type: application/json' -b "$COOKIES" \
    -H 'origin: https://claude.hk.cn' -H "user-agent: $UA" \
    --data-raw '{"name":"","project_uuid":null,"model":null}' \
    | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["uuid"])')
  echo -e "${Y}新对话: ${D}${CONV_ID}${R}"
}

send_prompt() {
  local prompt="$1"
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
    "locale": "en-US", "model": "${MODEL}", "effort": "${EFFORT}",
    "thinking_mode": "${THINKING}",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {"human_message_uuid": "${h_uuid}", "assistant_message_uuid": "${a_uuid}"},
    "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
    "create_conversation_params":{"name":"","model":"${MODEL}",
        "include_conversation_preferences":True,"paprika_mode":None,"compass_mode":None,
        "tool_search_mode":"auto","is_temporary":False,"enabled_imagine":True}
}
open("/tmp/_chat_payload.json","w").write(json.dumps(payload))
PYEOF

  LAST_RAW=$(curl -sf --max-time "$TIMEOUT" \
    "${BASE}/chat_conversations/${CONV_ID}/completion" \
    -H 'accept: text/event-stream' -H 'content-type: application/json' \
    -H 'anthropic-client-platform: web_claude_ai' \
    -b "$COOKIES" -H 'origin: https://claude.hk.cn' \
    -H 'referer: https://claude.hk.cn/new' -H "user-agent: $UA" \
    -d @/tmp/_chat_payload.json 2>&1)

  echo "$LAST_RAW" | python3 << 'PYEOF'
import sys, json
text_parts, tool_calls, tool_results = [], [], []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    try: d = json.loads(line[6:])
    except: continue
    t = d.get("type", "")
    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta": text_parts.append(delta["text"])
        elif delta.get("type") == "tool_use_block_update_delta":
            dc = delta.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    info = json.loads(dc["json_block"])
                    if info.get("code"): tool_calls.append(f'[{info.get("language","")}] {info["code"][:150]}')
                except: pass
    elif t == "content_block_start":
        cb = d.get("content_block", {})
        if cb.get("type") == "tool_result":
            dc = cb.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try:
                    r = json.loads(dc["json_block"])
                    stdout = r.get("stdout", "")
                    if stdout: tool_results.append(stdout.rstrip())
                except: pass
if tool_calls: print(f"\033[90m  [{len(tool_calls)} tool calls]\033[0m")
if tool_results:
    for r in tool_results:
        for l in r.split("\n")[:10]: print(f"\033[90m  > {l}\033[0m")
text = "".join(text_parts)
if text.strip():
    print(f"\033[36mClaude:\033[0m")
    print(text)
print()
PYEOF
}

usage() {
  echo -e "${Y}claude_hk_chat.sh${R} — claude.hk.cn CLI (code exec)"
  echo -e "  /new   新建对话    /model <name>  切换模型"
  echo -e "  /effort <level>    /think on|off  /status  /raw  /quit"
  echo -e "  cookie从 /tmp/claude_hk_cookie.txt 或 claude-hk-config repo 自动读取"
}

main() {
  [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }
  echo -e "${Y}claude.hk.cn CLI | ${C}${MODEL}${Y} | effort=${C}${EFFORT}${R}"
  new_conversation
  while true; do
    echo -ne "${G}你: ${R}"
    IFS= read -r input || break
    [ -z "$input" ] && continue
    case "$input" in
      /quit|/q) echo -e "${Y}再见${R}"; break ;;
      /help|/h) usage ;;
      /new) new_conversation ;;
      /model\ *) MODEL="${input#/model }"; echo -e "${Y}模型: ${C}${MODEL}${R}" ;;
      /effort\ *) EFFORT="${input#/effort }"; echo -e "${Y}effort: ${C}${EFFORT}${R}" ;;
      /think\ *) THINKING="${input#/think }"; echo -e "${Y}thinking: ${C}${THINKING}${R}" ;;
      /status) echo -e "  模型:${C}${MODEL}${R} effort:${C}${EFFORT}${R} conv:${D}${CONV_ID}${R}" ;;
      /raw) echo -e "${D}${LAST_RAW}${R}" ;;
      *) send_prompt "$input" ;;
    esac
  done
}
main "$@"
