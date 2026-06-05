#!/usr/bin/env python3
"""
bench_one_model.py — 单模型benchmark核心逻辑
用法: python3 bench_one_model.py <model> <repo_url> <output_file> [timeout]

配置来源(每次运行自动pull最新):
  cookie: /tmp/claude_hk_cookie.txt
  org_id: /tmp/claude_hk_config.json -> org_id
  均从 github.com/dylanyunlon/claude-hk-config 同步
"""
import sys, json, uuid, subprocess, re, os, time

MODEL = sys.argv[1]
REPO_URL = sys.argv[2]
OUT_FILE = sys.argv[3]
TIMEOUT = int(sys.argv[4]) if len(sys.argv) > 4 else 300
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
CONFIG_REPO = "https://github.com/dylanyunlon/claude-hk-config.git"
CONFIG_DIR = "/tmp/claude-hk-config"

# ── 同步config repo ────────────────────────────────────────
def sync_config():
    """git pull config repo, 提取cookie+org到本地缓存"""
    try:
        if os.path.exists(CONFIG_DIR):
            subprocess.run(["git", "-C", CONFIG_DIR, "pull", "-q"],
                          timeout=15, capture_output=True)
        else:
            subprocess.run(["git", "clone", "--depth=1", "-q",
                          CONFIG_REPO, CONFIG_DIR],
                          timeout=20, capture_output=True)
    except: pass

    # 运行extract.py从raw_curl.txt生成cookie.txt和config.json
    extract = os.path.join(CONFIG_DIR, "extract.py")
    if os.path.exists(extract):
        try:
            subprocess.run(["python3", extract], timeout=5, capture_output=True)
        except: pass

    # 同步到/tmp本地缓存
    ck_src = os.path.join(CONFIG_DIR, "cookie.txt")
    cfg_src = os.path.join(CONFIG_DIR, "config.json")
    if os.path.exists(ck_src):
        ck = open(ck_src).read().strip()
        if ck and not ck.startswith("#"):
            open("/tmp/claude_hk_cookie.txt", "w").write(ck)
    if os.path.exists(cfg_src):
        subprocess.run(["cp", cfg_src, "/tmp/claude_hk_config.json"],
                      capture_output=True)

def load_cookie():
    if os.path.exists("/tmp/claude_hk_cookie.txt"):
        ck = open("/tmp/claude_hk_cookie.txt").read().strip()
        if ck and not ck.startswith("#"): return ck
    ck = os.environ.get("CLAUDE_HK_COOKIE", "")
    if ck: return ck
    print("ERROR: No cookie", file=sys.stderr); sys.exit(1)

def load_org():
    if os.path.exists("/tmp/claude_hk_config.json"):
        try:
            cfg = json.load(open("/tmp/claude_hk_config.json"))
            org = cfg.get("org_id", "")
            if org: return org
        except: pass
    return os.environ.get("CLAUDE_HK_ORG", "")

# ── 同步 ───────────────────────────────────────────────────
sync_config()
CK = load_cookie()
ORG = load_org()
if not ORG:
    print("ERROR: No org_id. Update claude-hk-config/raw_curl.txt", file=sys.stderr)
    sys.exit(1)

BASE = f"https://claude.hk.cn/api/organizations/{ORG}"

PROMPT = f"""不要立刻查看所有内容,在你的linux上用tree、git branch 先查看架构. 使用 git clone --depth=1 进行clone(只拉最新commit, 不要完整历史)。没有tree你就apt install tree。 看看这个项目(upstream文件夹)关于代码移植的问题，我们需要每一个文件的每一行都用上。github.com/{REPO_URL} 鲁迅那样的拿法，别全权复制啊。我的意思是在mv的基础上，动态修改算法的20%的内容就行了。注意多写一点关于断点调试（或者print当前所有数据、结构体状态）的内容, 让我们在运行实验的时候能像现实世界开发一样得到反馈

分析完代码后，先输出JSON(这是最重要的,必须完整输出):
```json
{{
  "code_analysis": {{"src_lines": ..., "upstream_lines": ..., "top_changes": [{{"file":"...","added":...,"removed":...}}]}},
  "baselines": {{...}},
  "modifications_impact": [{{"name":"...","file":"...","lines_changed":...,"estimated_improvement_pct":...}}],
  "sota": [{{"name":"...","year":...}}],
  "expected_improvement": "..."
}}
```
然后再写分析文本。"""

def curl(url, method="GET", data=None, stream=False):
    cmd = ["curl", "-sf", "--max-time", str(TIMEOUT if stream else 15), url,
           "-H", "content-type: application/json", "-b", CK,
           "-H", "origin: https://claude.hk.cn", "-H", f"user-agent: {UA}"]
    if method == "POST": cmd += ["-X", "POST"]
    if method == "DELETE": cmd += ["-X", "DELETE"]
    if stream:
        cmd += ["-H", "accept: text/event-stream",
                "-H", "anthropic-client-platform: web_claude_ai",
                "-H", "referer: https://claude.hk.cn/new"]
    if data:
        cmd += ["--data-raw", json.dumps(data) if isinstance(data, dict) else data]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT+10)
    return r.stdout

t0 = time.time()
conv_raw = curl(f"{BASE}/chat_conversations", "POST", {"name":"","project_uuid":None,"model":None})
try:
    conv = json.loads(conv_raw)
    conv_id = conv["uuid"]
except:
    print(f"ERROR: conv create failed. Response: {conv_raw[:200]}", file=sys.stderr)
    sys.exit(1)

payload = {
    "prompt": PROMPT,
    "timezone": "Asia/Shanghai",
    "personalized_styles": [{"type":"default","key":"Default","name":"Normal",
        "nameKey":"normal_style_name","prompt":"Normal\n",
        "summary":"Default responses from Claude",
        "summaryKey":"normal_style_summary","isDefault":True}],
    "locale": "en-US", "model": MODEL, "effort": "high", "thinking_mode": "off",
    "tools": [
        {"type": "web_search_v0", "name": "web_search"},
        {"type": "artifacts_v0", "name": "artifacts"},
        {"type": "repl_v0", "name": "repl"}
    ],
    "turn_message_uuids": {"human_message_uuid": str(uuid.uuid4()), "assistant_message_uuid": str(uuid.uuid4())},
    "attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages",
    "create_conversation_params":{"name":"","model":MODEL,"include_conversation_preferences":True,
        "paprika_mode":None,"compass_mode":None,"tool_search_mode":"auto","is_temporary":False,"enabled_imagine":True}
}

sse_raw = curl(f"{BASE}/chat_conversations/{conv_id}/completion", data=payload, stream=True)

text_parts, tool_results, code_blocks = [], [], []
for line in sse_raw.split("\n"):
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
                try: code_blocks.append(json.loads(dc["json_block"]))
                except: pass
    elif t == "content_block_start":
        cb = d.get("content_block", {})
        if cb.get("type") == "tool_result":
            dc = cb.get("display_content", {})
            if dc and dc.get("type") == "json_block":
                try: tool_results.append(json.loads(dc["json_block"]))
                except: pass

full_text = "".join(text_parts)
full_text = re.sub(r'(?<=[:\s,])~(\d)', r'\1', full_text)
full_text = full_text.replace('"~', '"')

result = {"model": MODEL, "org": ORG, "text": full_text, "tool_results": tool_results,
          "code_executed": code_blocks, "num_tool_calls": len(tool_results),
          "elapsed_s": round(time.time() - t0, 1)}

for fn in [
    lambda: json.loads(full_text.strip()),
    lambda: json.loads(re.search(r'```(?:json)?\s*\n(.*?)\n```', full_text, re.DOTALL).group(1)),
    lambda: json.loads(re.search(r'\{.*\}', full_text, re.DOTALL).group(0)),
]:
    try: result["benchmark_data"] = fn(); break
    except: pass

for tr in tool_results:
    stdout = tr.get("stdout", "")
    if stdout: result["exec_stdout"] = result.get("exec_stdout", "") + stdout

json.dump(result, open(OUT_FILE, "w"), indent=2, ensure_ascii=False)
has_json = "benchmark_data" in result
print(f"tools={len(tool_results)}, text={len(full_text)}c, json={'✓' if has_json else '✗'}, {result['elapsed_s']}s [org={ORG[:8]}]")

try: curl(f"{BASE}/chat_conversations/{conv_id}", "DELETE")
except: pass