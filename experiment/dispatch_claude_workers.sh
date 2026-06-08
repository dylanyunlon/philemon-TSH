#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# dispatch_claude_workers.sh — 批量调度子Claude Opus 4.6
# 每位子Claude收到简洁的clone指令+任务描述
#
# 用法:
#   ./dispatch_claude_workers.sh           # 调度所有待完成milestone
#   ./dispatch_claude_workers.sh M147      # 只调度M147-M148
#   CONTINUE=3 ./dispatch_claude_workers.sh  # 最多发3次Continue
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── 配置 ──────────────────────────────────────────────────────
export MODEL="${MODEL:-claude-opus-4-6-20250514}"
export EFFORT="${EFFORT:-high}"
export THINKING="${THINKING:-off}"
export TIMEOUT="${TIMEOUT:-600}"
MAX_CONTINUE="${CONTINUE:-5}"
REPO_URL="https://github.com/dylanyunlon/philemon-TSH"
GIT_TOKEN="${GITHUB_TOKEN}"

# ── 加载cookie (复用claude_hk_chat.sh的逻辑) ────────────────
source_cookie() {
    if [ -f /tmp/claude_hk_cookie.txt ]; then
        COOKIES=$(head -1 /tmp/claude_hk_cookie.txt)
        [ -n "$COOKIES" ] && return
    fi
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
    [ -z "${COOKIES:-}" ] && { echo "ERROR: 无cookie"; exit 1; }
}

ORG="0de6831b-fb77-41c7-bfb9-0899fb74f90f"
BASE="https://claude.hk.cn/api/organizations/${ORG}"
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
COOKIES=""
source_cookie

# ── API调用 ───────────────────────────────────────────────────
CONV_ID=""

new_conversation() {
    CONV_ID=$(curl -sf --max-time 15 "${BASE}/chat_conversations" \
        -X POST -H 'content-type: application/json' -b "$COOKIES" \
        -H 'origin: https://claude.hk.cn' -H "user-agent: $UA" \
        --data-raw '{"name":"","project_uuid":null,"model":null}' \
        | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["uuid"])')
    echo "[调度] 新对话: ${CONV_ID}"
}

send_message() {
    local prompt="$1"
    local h_uuid a_uuid
    h_uuid=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')
    a_uuid=$(python3 -c 'import uuid; print(str(uuid.uuid4()))')

    python3 -c "
import json
payload = {
    'prompt': '''$prompt''',
    'timezone': 'Asia/Shanghai',
    'personalized_styles': [{'type':'default','key':'Default','name':'Normal',
        'nameKey':'normal_style_name','prompt':'Normal\n',
        'summary':'Default responses from Claude',
        'summaryKey':'normal_style_summary','isDefault':True}],
    'locale': 'en-US', 'model': '${MODEL}', 'effort': '${EFFORT}',
    'thinking_mode': '${THINKING}',
    'tools': [
        {'type': 'web_search_v0', 'name': 'web_search'},
        {'type': 'artifacts_v0', 'name': 'artifacts'},
        {'type': 'repl_v0', 'name': 'repl'}
    ],
    'turn_message_uuids': {'human_message_uuid': '${h_uuid}', 'assistant_message_uuid': '${a_uuid}'},
    'attachments':[],'files':[],'sync_sources':[],'rendering_mode':'messages',
    'create_conversation_params':{'name':'','model':'${MODEL}',
        'include_conversation_preferences':True,'paprika_mode':None,'compass_mode':None,
        'tool_search_mode':'auto','is_temporary':False,'enabled_imagine':True}
}
open('/tmp/_dispatch_payload.json','w').write(json.dumps(payload))
"

    local raw_response
    raw_response=$(curl -sf --max-time "$TIMEOUT" \
        "${BASE}/chat_conversations/${CONV_ID}/completion" \
        -H 'accept: text/event-stream' -H 'content-type: application/json' \
        -H 'anthropic-client-platform: web_claude_ai' \
        -b "$COOKIES" -H 'origin: https://claude.hk.cn' \
        -H 'referer: https://claude.hk.cn/new' -H "user-agent: $UA" \
        -d @/tmp/_dispatch_payload.json 2>&1)

    # 解析并显示响应
    local text
    text=$(echo "$raw_response" | python3 -c '
import sys, json
text_parts = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    try: d = json.loads(line[6:])
    except: continue
    t = d.get("type", "")
    if t == "content_block_delta":
        delta = d.get("delta", {})
        if delta.get("type") == "text_delta": text_parts.append(delta["text"])
text = "".join(text_parts)
print(text)
' 2>/dev/null || echo "")

    echo "$text"

    # 检查是否被截断 (最后没有正常结束标记)
    if echo "$raw_response" | grep -q '"stop_reason":"end_turn"'; then
        return 0  # 正常完成
    else
        return 1  # 可能被截断
    fi
}

# ── 任务定义 ──────────────────────────────────────────────────
# 每个任务就是一个简洁的clone-style prompt
get_task_prompt() {
    local task_id="$1"
    case "$task_id" in
        M147)
cat << 'PROMPT'
git clone https://github.com/dylanyunlon/philemon-TSH && cd philemon-TSH

你是第2位Claude (Opus 4.6)。读 CLAUDE_DEVELOPMENT_PLAN.md。

任务 M147-M148: 用upstream RapidStore的driver framework构建可复现竞品对比实验。

具体:
1. 读 upstream/rapidstore/utils/commandLineParser.cpp 和 upstream/rapidstore/wrapper/ 下所有文件
2. 在 experiment/ 下创建 m147_m148_baseline_benchmark.cpp
3. 实现: 用我们的TieredGraphWrapper (src/wrapper/rapidstore_wrapper.hpp) 跑 insert/scan/search microbenchmark
4. 对比: CSR baseline vs TieredSnapshot, 在 RMAT scale24 (16M vertices, 256M edges) 上
5. 产出数据: insert MEPS, scan MEPS, search latency (μs), 内存占用 (MB)
6. 修复M145-M146中BFS的vector<bool>数据竞争: 改用vector<char>或atomic
7. g++ -std=c++17 -O2 -fopenmp -march=native 编译通过, 全PASS

规则: main分支, 不开新分支, 不用v2/port后缀, 只改算法核心
git push用: git remote set-url origin https://${GITHUB_TOKEN}@github.com/dylanyunlon/philemon-TSH.git
commit author: dylanyunlon <dogechat@163.com>
PROMPT
        ;;
        M149)
cat << 'PROMPT'
git clone https://github.com/dylanyunlon/philemon-TSH && cd philemon-TSH

你是第3位Claude (Opus 4.6)。读 CLAUDE_DEVELOPMENT_PLAN.md。

任务 M149-M150: 三层存储真实I/O路径实现。

具体:
1. 读 src/core/tiered_allocator.hpp, src/eviction/lru_eviction.hpp, src/hotness/hotness_tracker.hpp
2. 在 experiment/ 下创建 m149_m150_tiered_io_experiment.cpp
3. 实现真实的三层数据放置:
   - Hot edges: 直接内存 (malloc)
   - Warm edges: mmap文件 (MAP_POPULATE for readahead)
   - Cold edges: pread/pwrite direct I/O
4. 实现hotness-based自动迁移: access count > threshold → promote, decay → demote
5. 测量: 限制DRAM到64MB时, 在1M vertex RMAT图上BFS/PR/WCC的性能
6. 对比: 全DRAM baseline vs tiered (DRAM+SSD)
7. 编译通过, 全PASS

规则: main分支, 不用后缀, 只改算法核心
git push用: git remote set-url origin https://${GITHUB_TOKEN}@github.com/dylanyunlon/philemon-TSH.git
commit author: dylanyunlon <dogechat@163.com>
PROMPT
        ;;
        M151)
cat << 'PROMPT'
git clone https://github.com/dylanyunlon/philemon-TSH && cd philemon-TSH

你是第4位Claude (Opus 4.6)。读 CLAUDE_DEVELOPMENT_PLAN.md。

任务 M151-M152: 并发读写实验 + 大规模扩展性。

具体:
1. 读 src/wrapper/rapidstore_wrapper.hpp 的 TieredGraphWrapper
2. 在 experiment/ 下创建 m151_m152_concurrent_scaling.cpp
3. 实现并发workload: N reader threads跑PR + M writer threads跑insert/delete
4. 测量: PR延迟在0/4/8/16/32 writers下的增幅
5. 扩展性: 从100K到10M vertices, 测量insert throughput和BFS时间的scaling曲线
6. 产出论文Table数据: 并发隔离比 (latency增幅%), throughput scaling factor
7. 编译通过, 全PASS

规则: main分支, 不用后缀, 只改算法核心
git push用: git remote set-url origin https://${GITHUB_TOKEN}@github.com/dylanyunlon/philemon-TSH.git
commit author: dylanyunlon <dogechat@163.com>
PROMPT
        ;;
        *)
            echo "未知任务: $task_id"
            return 1
        ;;
    esac
}

# ── 调度单个子Claude ─────────────────────────────────────────
dispatch_worker() {
    local task_id="$1"
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " 调度子Claude: ${task_id} | 模型: ${MODEL}"
    echo "═══════════════════════════════════════════════════════"

    local prompt
    prompt=$(get_task_prompt "$task_id")
    if [ -z "$prompt" ]; then
        echo "[ERROR] 无法获取任务prompt: $task_id"
        return 1
    fi

    new_conversation
    echo "[调度] 发送初始prompt ($(echo "$prompt" | wc -c) bytes)..."

    local continue_count=0
    if send_message "$prompt"; then
        echo "[调度] ${task_id} 初始响应完成"
    else
        echo "[调度] ${task_id} 可能被截断, 尝试Continue..."
        while [ $continue_count -lt $MAX_CONTINUE ]; do
            continue_count=$((continue_count + 1))
            echo "[调度] 发送 Continue ($continue_count/$MAX_CONTINUE)..."
            sleep 3
            if send_message "Continue"; then
                echo "[调度] ${task_id} Continue完成"
                break
            fi
        done
    fi

    echo "[调度] ${task_id} 对话结束 (${continue_count} continues)"
    echo "  对话链接: https://claude.hk.cn/chat/${CONV_ID}"
}

# ── 主流程 ────────────────────────────────────────────────────
main() {
    local filter="${1:-all}"

    echo "═══════════════════════════════════════════════════════"
    echo " Philemon-TSH 子Claude调度器"
    echo " 模型: ${MODEL} | 最大Continue: ${MAX_CONTINUE}"
    echo "═══════════════════════════════════════════════════════"

    local tasks=()
    case "$filter" in
        all)   tasks=(M147 M149 M151) ;;
        M147)  tasks=(M147) ;;
        M149)  tasks=(M149) ;;
        M151)  tasks=(M151) ;;
        *)     echo "用法: $0 [all|M147|M149|M151]"; exit 1 ;;
    esac

    for task in "${tasks[@]}"; do
        dispatch_worker "$task"
        echo ""
        echo "[调度] 等待30秒后调度下一位..."
        sleep 30
    done

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " 全部调度完成. 子Claude们正在工作."
    echo " 检查进度: git log --oneline -10"
    echo "═══════════════════════════════════════════════════════"
}

main "$@"
