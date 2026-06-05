#!/usr/bin/env python3
"""从 raw_curl.txt 提取 cookie + org_id → cookie.txt + config.json"""
import re, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(here, "raw_curl.txt")
cookie_path = os.path.join(here, "cookie.txt")
config_path = os.path.join(here, "config.json")

if not os.path.exists(raw_path):
    print("raw_curl.txt not found", file=sys.stderr)
    sys.exit(1)

text = open(raw_path).read()

# 提取cookie
m = re.search(r"-b\s+['\"]([^'\"]+)['\"]", text)
if not m:
    m = re.search(r"--cookie\s+['\"]([^'\"]+)['\"]", text)
cookie = m.group(1) if m else ""

# 提取org_id（从URL里）
m = re.search(r"/organizations/([a-f0-9-]+)/", text)
org_id = m.group(1) if m else ""

# 提取model（从payload里）
m = re.search(r'"model"\s*:\s*"([^"]+)"', text)
model = m.group(1) if m else "claude-sonnet-4-6"

if cookie:
    open(cookie_path, "w").write(cookie.strip() + "\n")
    print(f"cookie: {len(cookie)} chars")
else:
    print("WARNING: no cookie found", file=sys.stderr)

config = {"org_id": org_id, "model": model}
# 保留已有config里的其他字段
if os.path.exists(config_path):
    try:
        old = json.load(open(config_path))
        old.update(config)
        config = old
    except: pass

json.dump(config, open(config_path, "w"), indent=2)
print(f"org_id: {org_id}")
print(f"model: {model}")