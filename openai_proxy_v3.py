#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TeleAgent OpenAI-Compatible Proxy · 极简版 v3.0.0
====================================================
把本地 TeleAgent（3.x）的 super-agent 暴露成 OpenAI 兼容接口。

设计原则（2026-11-26 定）：
  1. 只做一件事：/v1/chat/completions + /v1/models + /health。
     控制台、embeddings、来源追踪、tool 文本协议、确认注入接口全部删除。
  2. 最小痕迹：不伪造任何指纹（不碰 device-meta/installId/UA），
     服务端会话一律交给 super-agent 自动命名，不写人工标题。
     云端流量始终由官方 Go 二进制自己发出，检测面≈零。
  3. 凭据只在本机：session.key 0600，日志与错误信息脱敏，绝不外传。

端点：
  GET  /health               健康检查（含上游可达性）
  GET  /v1/models            列出 TeleAgent 中已配置的模型
  POST /v1/chat/completions  聊天补全（流式/非流式）

鉴权：Authorization: Bearer <config.json 中的 api_key>（常数时间比较）。
监听：默认 127.0.0.1:8088（本机自用）。

依赖：纯 Python 标准库。
"""

import argparse
import base64
import codecs
import hashlib
import hmac
import json
import os
import re
import secrets
import select
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

VERSION = "3.0.0"

# ============================================================
# 常量
# ============================================================
SUPER_AGENT_HOST = "127.0.0.1"
SUPER_AGENT_PORT = 4397
SUPER_AGENT_BASE = f"http://{SUPER_AGENT_HOST}:{SUPER_AGENT_PORT}"
SIGN_VERSION = "local-v1"

PROXY_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(PROXY_DIR, "session.key")
CONFIG_PATH = os.path.join(PROXY_DIR, "config.json")
LOG_DIR = os.path.join(PROXY_DIR, "logs")
EXTRACTOR = os.path.join(PROXY_DIR, "extract_sa_key.py")
_EXTRACTOR_ALIASES = ("extract_sa_key.py", "extract_sa_key_v3.py")


def _extractor_path():
    """容忍两种文件名，避免部署时改名导致自愈静默失效（实测踩过）。"""
    for name in _EXTRACTOR_ALIASES:
        p = os.path.join(PROXY_DIR, name)
        if os.path.exists(p):
            return p
    return None

DEFAULT_DATA_DIR = os.path.expanduser("~/.local/share/TeleAgent")
DEFAULT_PROVIDER = "NewApi"
DEFAULT_MODEL = "chat-pro"

# 会话缓存（仅显式 session_title 复用路径使用）
SESSION_CACHE_TTL = 120
SESSION_QUERY_LIMIT = 300

# 上游单请求超时（秒）：agent 会话首包可能较慢
UPSTREAM_TIMEOUT = 600
# 复用会话串行化最长等待
SESSION_LOCK_TIMEOUT = 120
# 401 后密钥重提取节流
EXTRACT_THROTTLE = 90
# 终态后继续排空事件流的宽限（等 step-finish 的 tokens 落袋）
DRAIN_GRACE = 2.0


# ============================================================
# 配置（config.json，0600）
# ============================================================
_config_lock = threading.Lock()
_config = {
    "api_key": "",
    "default_model": f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}",
    "auto_approve": True,
    "max_inflight": 2,
    "listen_host": "127.0.0.1",
    "listen_port": 8088,
}


def load_config():
    """启动时读取 config.json；缺失或无效时落一份默认配置。

    api_key 为空时自动生成并写盘，调用方无需手工准备。
    """
    global _config
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in _config:
                if k in data and data[k] is not None:
                    _config[k] = data[k]
    except Exception:
        pass
    if not _config.get("api_key"):
        _config["api_key"] = "sk-local-" + secrets.token_urlsafe(24)
    save_config()
    return _config


def save_config():
    """原子写盘 + 0600（含凭据，不能给其他人读）。"""
    with _config_lock:
        tmp = CONFIG_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_config, f, ensure_ascii=False, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, CONFIG_PATH)
        except Exception as e:
            print(f"[CONFIG] 保存失败: {e}", file=sys.stderr)


def get_default_model_parts():
    model = _config.get("default_model") or f"{DEFAULT_PROVIDER}/{DEFAULT_MODEL}"
    if "/" in model:
        provider_id, model_id = model.split("/", 1)
    else:
        provider_id, model_id = DEFAULT_PROVIDER, model
    return provider_id, model_id


# ============================================================
# 密钥：环境变量 > session.key > 旧 scheduler 环境 > 提取器
# ============================================================
_cached_session_key = None
_cached_session_key_time = 0.0
_session_key_lock = threading.Lock()
_extract_lock = threading.Lock()
_last_extract_time = 0.0


def _read_key_file():
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
        return key or None
    except Exception:
        return None


def _read_key_from_scheduler_env():
    """TeleAgent 2.x 兼容路径（3.x 已无 scheduler，正常不会命中）。"""
    try:
        out = subprocess.run(["pgrep", "-f", "scheduler/index.cjs"],
                             capture_output=True, text=True, timeout=5).stdout
        for pid in out.strip().split("\n"):
            pid = pid.strip()
            if not pid:
                continue
            try:
                line = subprocess.run(["ps", "eww", pid], capture_output=True,
                                      text=True, timeout=5).stdout
            except Exception:
                continue
            m = re.search(r"SUPER_AGENT_LOCAL_SESSION_KEY=(\S+)", line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def get_session_key(force=False):
    """取签名密钥。缓存 60s，避免每请求都读盘。

    优先级：环境变量 → session.key → 旧 scheduler 环境。
    401 时由 refresh_session_key() 触发提取器刷新。
    """
    global _cached_session_key, _cached_session_key_time
    with _session_key_lock:
        if (not force and _cached_session_key
                and time.time() - _cached_session_key_time < 60):
            return _cached_session_key

    key = os.environ.get("SUPER_AGENT_LOCAL_SESSION_KEY") or _read_key_file()
    if not key:
        key = _read_key_from_scheduler_env()
    with _session_key_lock:
        if key:
            _cached_session_key = key
            _cached_session_key_time = time.time()
        return _cached_session_key


def refresh_session_key():
    """401/403 后刷新密钥：优先跑提取器重新提取。

    注意：不能先「重读 session.key」——文件里通常就是刚被判定无效的那个旧值，
    重读等于什么都没做（v3.0.0 实测踩过）。只有提取器不可用时才退回重读文件。
    """
    global _cached_session_key, _cached_session_key_time, _last_extract_time

    extractor = _extractor_path()
    if extractor:
        if not _extract_lock.acquire(timeout=30):
            return None
        try:
            now = time.time()
            if now - _last_extract_time < EXTRACT_THROTTLE:
                return None
            _last_extract_time = now
            print("[KEY] session key 失效，运行提取器刷新…", file=sys.stderr)
            r = subprocess.run([sys.executable, extractor],
                               capture_output=True, text=True,
                               timeout=300, cwd=PROXY_DIR)
            key = _read_key_file()
            if r.returncode == 0 and key:
                with _session_key_lock:
                    _cached_session_key = key
                    _cached_session_key_time = time.time()
                print(f"[KEY] 已刷新（{key[:6]}…）", file=sys.stderr)
                return key
            print(f"[KEY] 提取器未成功退出 (exit={r.returncode})；"
                  f"stderr: {(r.stderr or '')[-200:]}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("[KEY] 提取器超时", file=sys.stderr)
        except Exception as e:
            print(f"[KEY] 提取器异常: {e}", file=sys.stderr)
        finally:
            _extract_lock.release()

    # 兜底：提取器不在/失败时，重读文件（外部可能已手工更新）
    key = _read_key_file()
    if key:
        with _session_key_lock:
            _cached_session_key = key
            _cached_session_key_time = time.time()
        return key
    return None


def sign_request(method, path_with_query):
    """X-SA local-v1：HMAC-SHA256(key, "local-v1\\nMETHOD\\nPATH\\nTS\\nNONCE")，base64url 无填充。"""
    key = get_session_key()
    if not key:
        raise RuntimeError(
            "无法获取会话密钥：请确认 TeleAgent 正在运行，"
            "并已执行过 extract_sa_key.py 生成 session.key")
    ts = str(int(time.time() * 1000))
    nonce = uuid.uuid4().hex[:24]
    payload = "\n".join([SIGN_VERSION, method.upper(), path_with_query, ts, nonce])
    sig = hmac.new(key.encode(), payload.encode(), hashlib.sha256).digest()
    return {
        "X-SA-Sign-Version": SIGN_VERSION,
        "X-SA-Timestamp": ts,
        "X-SA-Nonce": nonce,
        "X-SA-Signature": base64.urlsafe_b64encode(sig).decode().rstrip("="),
    }


def sa_request(method, path, body=None, timeout=30, _retry=True):
    """调 super-agent。401/403 时刷新密钥重试一次（不重复刷）。"""
    try:
        headers = sign_request(method, path)
    except Exception as e:
        return None, str(e), {}

    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(f"{SUPER_AGENT_BASE}{path}", data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", "replace")
        if _retry and e.code in (401, 403):
            if refresh_session_key():
                return sa_request(method, path, body, timeout, _retry=False)
        return e.code, body_text, dict(e.headers)
    except Exception as e:
        return None, str(e), {}


# ============================================================
# 会话
# ============================================================
_session_cache = {}
_session_cache_lock = threading.Lock()
_title_create_locks = {}

_TIME_SEG_RE = re.compile(r"^\d{1,2}:\d{2}$")
_DATE_SEG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _session_match_key(title):
    """标题匹配键：仅剥掉「…|YYYY-MM-DD|HH:MM」尾部时间，其余原样比较。"""
    if not title:
        return title
    parts = title.split("|")
    while (len(parts) > 2
           and _TIME_SEG_RE.match(parts[-1].strip())
           and _DATE_SEG_RE.match(parts[-2].strip())):
        parts.pop()
    return "|".join(parts)


def create_session(directory=None, title=None):
    body = {}
    if directory:
        body["directory"] = directory
    if title:
        body["title"] = title
    status, resp, _ = sa_request("POST", "/session", body)
    if status != 200:
        return None
    try:
        data = json.loads(resp)
    except Exception:
        return None
    return data.get("id") or (data.get("data") or {}).get("id")


def find_session_by_title(title, directory=None):
    if not title:
        return None
    status, resp, _ = sa_request("GET", f"/session?limit={SESSION_QUERY_LIMIT}", timeout=15)
    if status != 200:
        return None
    try:
        sessions = json.loads(resp)
    except Exception:
        return None
    if not isinstance(sessions, list):
        return None
    key = _session_match_key(title)
    matched = []
    for s in sessions:
        if not isinstance(s, dict):
            continue
        if _session_match_key(s.get("title") or "") != key:
            continue
        s_dir = s.get("directory") or ""
        if directory and s_dir and os.path.normpath(s_dir) != os.path.normpath(directory):
            continue
        matched.append(s)
    if not matched:
        return None
    matched.sort(key=lambda s: ((s.get("time") or {}).get("updated") or 0), reverse=True)
    return matched[0].get("id")


def get_or_create_session(directory=None, title=None):
    """显式标题 → 按标题复用；无标题 → 新建（交 super-agent 自动命名）。

    返回 (session_id, reused)。同名并发创建以标题级锁互斥。
    """
    if not title:
        return create_session(directory=directory), False

    cache_key = _session_match_key(title)
    now = time.time()
    with _session_cache_lock:
        cached = _session_cache.get(cache_key)
        if cached and now - cached[1] < SESSION_CACHE_TTL:
            return cached[0], True

    sid = find_session_by_title(title, directory)
    if sid:
        with _session_cache_lock:
            _session_cache[cache_key] = (sid, now)
        return sid, True

    with _session_cache_lock:
        lock = _title_create_locks.setdefault(cache_key, threading.Lock())
    with lock:
        sid = find_session_by_title(title, directory)
        if sid:
            with _session_cache_lock:
                _session_cache[cache_key] = (sid, time.time())
            return sid, True
        sid = create_session(directory=directory, title=title)
        if sid:
            with _session_cache_lock:
                _session_cache[cache_key] = (sid, time.time())
        return sid, False


_prompt_locks = {}
_prompt_locks_lock = threading.Lock()


def session_lock_for(session_id):
    with _prompt_locks_lock:
        return _prompt_locks.setdefault(session_id, threading.Lock())


# ============================================================
# 全局在飞闸门（模拟单人使用节奏 + 保护上游）
# ============================================================
_inflight_sem = None


def init_inflight(n):
    global _inflight_sem
    _inflight_sem = threading.BoundedSemaphore(max(1, int(n or 1)))


# ============================================================
# 上行 prompt 组装
# ============================================================
def _content_to_text(content):
    """content 支持 string / parts / null。图片走 data URI 落临时文件。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            parts.append(part.get("text") or "")
        elif ptype == "image_url":
            img = part.get("image_url")
            url = img.get("url") if isinstance(img, dict) else (img or "")
            if not isinstance(url, str) or not url:
                continue
            if url.startswith("data:image"):
                try:
                    head, b64 = url.split(",", 1)
                    ext = "png"
                    m = re.match(r"data:image/(\w+)", head)
                    if m:
                        ext = m.group(1).replace("jpeg", "jpg")
                    raw = base64.b64decode(b64)
                except Exception:
                    parts.append("[图片解析失败]")
                    continue
                path = _save_vision_image(raw, ext)
                if path:
                    parts.append(f"请用 image_understanding 工具查看图片 {path} 并回答问题。")
                else:
                    parts.append("[图片保存失败]")
            else:
                # 非 data URI 一律不代取：避免 SSRF 与"代下载"这种非原生行为
                parts.append("[仅支持 data URI 图片]")
    return "\n".join(p for p in parts if p)


def _save_vision_image(raw, ext):
    """图片落盘到数据目录（不使用 /tmp 固定路径，避免跨进程冲突）。"""
    try:
        d = os.path.join(DEFAULT_DATA_DIR, "proxy-vision")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"img_{int(time.time()*1000)}_{secrets.token_hex(3)}.{ext}")
        with open(path, "wb") as f:
            f.write(raw)
        os.chmod(path, 0o600)
        return path
    except Exception:
        return None


def cleanup_vision_images(max_age=86400):
    """清理过期图片（原先只写不删会无限增长）。"""
    d = os.path.join(DEFAULT_DATA_DIR, "proxy-vision")
    try:
        now = time.time()
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            try:
                if now - os.path.getmtime(p) > max_age:
                    os.unlink(p)
            except Exception:
                continue
    except Exception:
        pass


def build_prompt(messages, session_reused=False):
    """OpenAI messages → 单条 prompt 文本。

    复用会话时只发最新一条 user 消息：TeleAgent 侧已有历史，
    全量重发会让上下文双重累积（fork v1.8.1 的教训）。
    """
    system_parts = []
    body_parts = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _content_to_text(msg.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
        elif role == "user":
            body_parts.append(text)
        elif role == "assistant":
            if text:
                body_parts.append(f"[assistant] {text}")
            else:
                names = []
                for tc in (msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        names.append(fn.get("name") or tc.get("id") or "?")
                if names:
                    body_parts.append(f"[assistant] (此前调用了工具: {', '.join(names)})")
        elif role == "tool":
            tcid = msg.get("tool_call_id") or ""
            prefix = f"[tool_result:{tcid}] " if tcid else "[tool_result] "
            body_parts.append(prefix + text)
        # 其它 role 忽略（OpenAI 侧校验已保证只有合法 role）

    if session_reused and len(body_parts) > 1:
        body_parts = [body_parts[-1]]

    if len(body_parts) == 1 and not system_parts:
        return body_parts[0]
    return "\n\n".join(system_parts + body_parts)


def send_prompt_async(session_id, prompt_text, provider_id, model_id, directory=None):
    body = {"sessionID": session_id, "parts": [{"type": "text", "text": prompt_text}]}
    if directory:
        body["directory"] = directory
    if provider_id and model_id:
        body["model"] = {"providerID": provider_id, "modelID": model_id}
    status, resp, _ = sa_request("POST", f"/session/{session_id}/prompt_async", body, timeout=20)
    return status in (200, 204), resp


def get_messages(session_id):
    status, resp, _ = sa_request("GET", f"/session/{session_id}/message", timeout=20)
    if status != 200:
        return []
    try:
        data = json.loads(resp)
    except Exception:
        return []
    return data if isinstance(data, list) else (data.get("data") or [])


def extract_assistant_text(messages, min_time=None):
    """兜底取回答。min_time 过滤掉上一轮的旧消息（复用会话的关键防错）。"""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        info = msg.get("info") or msg
        if info.get("role") != "assistant":
            continue
        t = (info.get("time") or {}).get("created")
        if min_time is not None:
            if t is None or t <= min_time:
                continue
        parts = msg.get("parts") or []
        texts = [p.get("text") for p in parts
                 if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
        if texts:
            return "\n".join(texts)
    return None


# ============================================================
# 确认请求自动答复（避免 API 客户端挂死）
# ============================================================
def reply_confirmation(conf_id, kind, decision="once"):
    if not conf_id:
        return False
    if kind == "question":
        # 问题类需要答案列表；自动批准场景下用第一个选项
        answers = ["确认"]
        status, _, _ = sa_request("POST", f"/question/{conf_id}/reply",
                                  {"answers": answers}, timeout=15)
    else:
        status, _, _ = sa_request("POST", f"/permission/{conf_id}/reply",
                                  {"reply": decision}, timeout=15)
    ok = status in (200, 204)
    print(f"[CONF] 自动答复 {kind} id={conf_id} decision={decision} -> {status}", file=sys.stderr)
    return ok


# ============================================================
# SSE 监听器
# ============================================================
class SSEListener:
    """订阅 /global/event 并按 sessionID 过滤，增量解码 UTF-8。

    终态由 session.status=idle 或消息 error 决定；回调只触发一次。
    """

    def __init__(self, session_id, on_delta=None, on_reasoning=None,
                 on_complete=None, on_error=None, on_confirmation=None,
                 timeout=UPSTREAM_TIMEOUT):
        self.session_id = session_id
        self.timeout = timeout
        self.on_delta = on_delta
        self.on_reasoning = on_reasoning
        self.on_complete = on_complete
        self.on_error = on_error
        self.on_confirmation = on_confirmation

        self.sock = None
        self._stop = threading.Event()
        self.completed = False
        self.error = None
        self.tokens = None
        self.text_chunks = []
        self.reasoning_chunks = []

        self._assistant_msg_ids = set()
        self._text_part_ids = set()
        self._reasoning_part_ids = set()

        self._terminal_lock = threading.Lock()
        self._terminal_fired = False

    # ---------- 生命周期 ----------
    def stop(self):
        self._stop.set()
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

    def _fire(self, cb, *args):
        if cb is None:
            return False
        try:
            cb(*args)
            return True
        except Exception as e:
            print(f"[SSE] 回调异常: {e}", file=sys.stderr)
            return False

    def _fire_terminal(self, kind, payload=None):
        """完成/错误终态只触发一次（禁止双 [DONE]）。"""
        with self._terminal_lock:
            if self._terminal_fired:
                return False
            self._terminal_fired = True
        if kind == "complete":
            self._fire(self.on_complete)
        else:
            self._fire(self.on_error, payload)
        return True

    # ---------- 连接 ----------
    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((SUPER_AGENT_HOST, SUPER_AGENT_PORT))
        headers = sign_request("GET", "/global/event")
        headers["Accept"] = "text/event-stream"
        headers["Host"] = f"{SUPER_AGENT_HOST}:{SUPER_AGENT_PORT}"
        lines = [f"GET /global/event HTTP/1.1"]
        for k, v in headers.items():
            lines.append(f"{k}: {v}")
        lines.append("Connection: keep-alive")
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        return sock

    # ---------- 事件处理 ----------
    def _handle(self, event):
        payload = event.get("payload") or {}
        etype = payload.get("type") or ""
        props = payload.get("properties") or {}

        sid = props.get("sessionID") or ""
        if sid and sid != self.session_id:
            return

        if etype == "message.updated":
            info = props.get("info") or {}
            mid = info.get("id")
            if mid:
                if info.get("role") == "assistant":
                    self._assistant_msg_ids.add(mid)
            if info.get("finish") == "error" or info.get("error"):
                self.error = info.get("error") or {"message": "upstream error"}
                self._fire_terminal("error", self.error)

        elif etype == "message.part.updated":
            part = props.get("part") or {}
            ptype = part.get("type") or ""
            pid = part.get("id") or ""
            mid = part.get("messageID") or props.get("messageID") or ""
            text = part.get("text") or ""

            if ptype == "text" and text and mid in self._assistant_msg_ids:
                self._text_part_ids.add(pid)
                if not self.text_chunks:
                    self.text_chunks.append(text)
                    self._fire(self.on_delta, text)
            elif ptype == "reasoning" and mid in self._assistant_msg_ids:
                self._reasoning_part_ids.add(pid)
                if text and not self.reasoning_chunks:
                    self.reasoning_chunks.append(text)
                    self._fire(self.on_reasoning, text)
            elif ptype == "step-finish":
                tokens = part.get("tokens")
                if tokens and not self.tokens:
                    self.tokens = tokens

            if part.get("error"):
                self.error = part["error"]
                self._fire_terminal("error", self.error)

        elif etype == "message.part.delta":
            pid = props.get("partID") or ""
            field = props.get("field") or ""
            delta = props.get("delta") or ""
            if not delta:
                return
            if field == "text" and pid in self._text_part_ids:
                self.text_chunks.append(delta)
                self._fire(self.on_delta, delta)
            elif field == "reasoning" or pid in self._reasoning_part_ids:
                self.reasoning_chunks.append(delta)
                self._fire(self.on_reasoning, delta)

        elif etype == "session.status":
            status = props.get("status") or {}
            if status.get("type") == "idle":
                self.completed = True
                self._fire_terminal("complete")

        elif etype in ("permission.asked", "question.asked"):
            cid = (props.get("id") or props.get("permissionID")
                   or props.get("permissionId") or props.get("questionID")
                   or props.get("questionId") or "")
            if not cid:
                return
            kind = "permission" if etype == "permission.asked" else "question"
            if _config.get("auto_approve"):
                reply_confirmation(cid, kind)
            if self.on_confirmation:
                self._fire(self.on_confirmation, cid, kind, props)

    # ---------- 主循环 ----------
    def run(self):
        deadline = time.time() + self.timeout
        sock = None
        try:
            sock = self._connect()
            self.sock = sock
            sock.settimeout(2)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            buf = ""
            header_ok = False
            # 终态触发后仍短暂排空事件流：上游的 step-finish（tokens）常在
            # session.status=idle 之后才到达，立即 break 会丢掉 usage
            drain_until = 0.0

            while not self._stop.is_set() and time.time() < deadline:
                if self._terminal_fired and time.time() > drain_until:
                    break
                r, _, _ = select.select([sock], [], [], 1)
                if not r:
                    continue
                chunk = sock.recv(16384)
                if not chunk:
                    break
                buf += decoder.decode(chunk)
                if not header_ok:
                    if "\r\n\r\n" not in buf:
                        continue
                    head, buf = buf.split("\r\n\r\n", 1)
                    status_line = head.split("\r\n", 1)[0]
                    m = re.match(r"HTTP/1\.\d\s+(\d{3})", status_line)
                    if not m or not m.group(1).startswith("2"):
                        self.error = {"message": f"upstream stream handshake failed: {status_line}"}
                        self._fire_terminal("error", self.error)
                        return
                    header_ok = True

                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:
                    line = line.rstrip("\r")
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        event = json.loads(data)
                    except Exception:
                        continue
                    try:
                        self._handle(event)
                        if self._terminal_fired and not drain_until:
                            drain_until = time.time() + DRAIN_GRACE
                    except Exception as e:
                        print(f"[SSE] 事件处理异常: {e}", file=sys.stderr)
        except Exception as e:
            if not self._terminal_fired:
                self.error = {"message": f"{type(e).__name__}: {e}"}
                self._fire_terminal("error", self.error)
        finally:
            try:
                if sock:
                    sock.close()
            except Exception:
                pass
            self.sock = None


# ============================================================
# 日志（纯追加 JSONL，按天分文件）
# ============================================================
_log_lock = threading.Lock()


def _log_path(date=None):
    date = date or time.strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"requests-{date}.jsonl")


def log_request(entry):
    """追加一行。凭据字段一律脱敏后才落盘。"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = dict(entry)
        entry.pop("api_key", None)
        entry.pop("authorization", None)
        with _log_lock:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOG] 落盘失败: {e}", file=sys.stderr)


# ============================================================
# 上游可用性 / 模型目录
# ============================================================
def upstream_health():
    status, resp, _ = sa_request("GET", "/global/health", timeout=10)
    return status, resp


def list_models():
    """只列出在 TeleAgent 中真正配置好的 provider 与模型（source=config）。"""
    status, resp, _ = sa_request("GET", "/provider", timeout=15)
    if status != 200:
        return None
    try:
        data = json.loads(resp)
    except Exception:
        return None
    out = []
    for p in data.get("all") or []:
        if not isinstance(p, dict):
            continue
        if p.get("source") != "config":
            continue
        models = p.get("models") or {}
        if not isinstance(models, dict):
            continue
        for mid in models:
            out.append(f"{p.get('id')}/{mid}")
    return out


def validate_model(provider_id, model_id):
    models = list_models()
    if models is None:
        return False
    return f"{provider_id}/{model_id}" in models


# ============================================================
# HTTP 处理器
# ============================================================
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 120
    MAX_BODY = 32 * 1024 * 1024
    client_gone = False

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}", file=sys.stderr)

    # ---------- 基础设施 ----------
    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.client_gone = True
            return False

    def _error(self, status, message, type_="invalid_request_error"):
        return self._send_json(status, {"error": {"message": message, "type": type_}})

    def _read_json_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            self.close_connection = True
            self._error(400, "invalid Content-Length")
            return None
        if n <= 0:
            self._error(400, "request body required")
            return None
        if n > self.MAX_BODY:
            self.close_connection = True
            self._error(413, "request body too large")
            return None
        try:
            raw = self.rfile.read(n)
        except Exception:
            self.client_gone = True
            self.close_connection = True
            return None
        try:
            data = json.loads(raw)
        except Exception as e:
            self._error(400, f"Invalid JSON: {e}")
            return None
        if not isinstance(data, dict):
            self._error(400, "request body must be a JSON object")
            return None
        return data

    def _authorized(self):
        """Bearer 常数时间比较。未配置 key 时拒绝（fail-closed）。"""
        expected = _config.get("api_key") or ""
        if not expected:
            return False
        hdr = self.headers.get("Authorization") or ""
        if not hdr.startswith("Bearer "):
            return False
        got = hdr[7:].strip()
        return hmac.compare_digest(got, expected)

    def _require_auth(self):
        if self._authorized():
            return True
        self._error(401, "invalid api key", "authentication_error")
        return False

    # ---------- SSE 写回 ----------
    def _send_sse_headers(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.client_gone = True
            self.close_connection = True
            return False

    def _sse(self, obj):
        try:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.client_gone = True
            return False

    def _sse_done(self):
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.client_gone = True
        self.close_connection = True

    # ---------- 路由 ----------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/"):
            return self._handle_health()
        if path == "/v1/models":
            if not self._require_auth():
                return
            return self._handle_models()
        return self._error(404, "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            if not self._require_auth():
                return
            return self._handle_chat()
        return self._error(404, "not found")

    # ---------- 端点 ----------
    def _handle_health(self):
        status, _ = upstream_health()
        ok = status == 200
        self._send_json(200 if ok else 503, {
            "status": "ok" if ok else "degraded",
            "version": VERSION,
            "upstream": {"port": SUPER_AGENT_PORT, "status": status},
            "key_present": bool(get_session_key()),
        })

    def _handle_models(self):
        models = list_models()
        if models is None:
            return self._error(502, "cannot query models from TeleAgent", "api_error")
        created = int(time.time())
        self._send_json(200, {
            "object": "list",
            "data": [{"id": m, "object": "model", "created": created,
                      "owned_by": "teleagent"} for m in models],
        })

    def _handle_chat(self):
        req = self._read_json_body()
        if req is None:
            return

        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._error(400, "messages is required and must be a non-empty array")
        for m in messages:
            if not isinstance(m, dict):
                return self._error(400, "each message must be an object")
            role = m.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                return self._error(400, f"unsupported role: {role!r}")

        stream = bool(req.get("stream"))
        stream_options = req.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}

        model = req.get("model")
        if model is None or model == "":
            provider_id, model_id = get_default_model_parts()
        elif not isinstance(model, str):
            return self._error(400, "model must be a string")
        elif "/" in model:
            provider_id, model_id = model.split("/", 1)
        else:
            provider_id = get_default_model_parts()[0]
            model_id = model

        explicit_title = req.get("session_title")
        if explicit_title is not None and not isinstance(explicit_title, str):
            return self._error(400, "session_title must be a string")

        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model_name = f"{provider_id}/{model_id}"
        started = time.time()

        entry = {
            "id": request_id, "ts": started, "model": model_name, "stream": stream,
            "session_title": explicit_title or None, "client": self.client_address[0],
        }

        if _inflight_sem is None or not _inflight_sem.acquire(timeout=UPSTREAM_TIMEOUT):
            entry.update(status="error", error="too many concurrent requests")
            log_request(entry)
            return self._error(429, "too many concurrent requests", "rate_limit_error")

        try:
            session_id, reused = get_or_create_session(directory=None, title=explicit_title)
            if not session_id:
                entry.update(status="error", error="failed to create session")
                log_request(entry)
                return self._error(502, "failed to create session on TeleAgent", "api_error")

            entry["session_id"] = session_id
            entry["session_reused"] = reused

            lock = session_lock_for(session_id) if (explicit_title or reused) else None
            if lock is not None and not lock.acquire(timeout=SESSION_LOCK_TIMEOUT):
                entry.update(status="error", error="session busy")
                log_request(entry)
                return self._error(503, "same session is busy, retry later", "server_error")
            try:
                if stream:
                    return self._do_stream(request_id, created, model_name, provider_id,
                                           model_id, messages, session_id, reused,
                                           stream_options, entry, started)
                return self._do_nonstream(request_id, created, model_name, provider_id,
                                          model_id, messages, session_id, reused,
                                          entry, started)
            finally:
                if lock is not None:
                    lock.release()
        finally:
            _inflight_sem.release()

    # ---------- 流式 ----------
    def _do_stream(self, rid, created, model_name, provider_id, model_id,
                   messages, session_id, reused, stream_options, entry, started):
        if not self._send_sse_headers():
            entry.update(status="client_disconnected", error="headers write failed",
                         duration_ms=int((time.time() - started) * 1000))
            log_request(entry)
            return

        include_usage = bool(stream_options.get("include_usage"))
        role_sent = {"v": False}
        text_acc, reasoning_acc = [], []
        final_tokens = {"v": None}
        terminal = {"done": False}
        tlock = threading.Lock()
        has_error = {"v": False}

        def _chunk(delta, finish=None):
            return {"id": rid, "object": "chat.completion.chunk", "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        def _ensure_role():
            if role_sent["v"]:
                return
            role_sent["v"] = True
            self._sse(_chunk({"role": "assistant"}))

        def on_delta(text):
            if self.client_gone:
                listener.stop()
                return
            _ensure_role()
            text_acc.append(text)
            self._sse(_chunk({"content": text}))

        def on_reasoning(text):
            if self.client_gone:
                listener.stop()
                return
            _ensure_role()
            reasoning_acc.append(text)
            self._sse(_chunk({"reasoning_content": text}))

        def _finalize(status, err=None):
            with tlock:
                if terminal["done"]:
                    return
                terminal["done"] = True

            # 直接读监听器上的 tokens：上游 step-finish 先于 session.status=idle 到达，
            # 此刻 listener.tokens 已就绪（外部 final_tokens 变量在 join 后才赋值）
            tokens = listener.tokens
            final_tokens["v"] = tokens

            if err is not None:
                has_error["v"] = True
                entry["error"] = err

            if not self.client_gone:
                if err is not None:
                    self._sse(_chunk({"content": f"[Error: {err}]"}, "stop"))
                else:
                    if not role_sent["v"]:
                        _ensure_role()
                    self._sse(_chunk({}, "stop"))

                usage = _usage_payload(tokens, text_acc, reasoning_acc)
                if include_usage and usage:
                    self._sse({"id": rid, "object": "chat.completion.chunk",
                               "created": created, "model": model_name,
                               "choices": [], "usage": usage})
                self._sse_done()

            entry.update(
                status=status,
                duration_ms=int((time.time() - started) * 1000),
                response_preview=("".join(text_acc))[:120],
                reasoning_preview=("".join(reasoning_acc))[:80],
                tokens=tokens,
            )
            log_request(entry)

        def on_complete():
            _finalize("success")

        def on_error(err):
            msg = err.get("message") if isinstance(err, dict) else str(err)
            _finalize("error", str(msg or "upstream error"))

        def on_confirmation(cid, kind, props):
            _ensure_role()
            self._sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": model_name,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                       "confirmation": {"id": cid, "type": kind, "session_id": session_id}})

        listener = SSEListener(session_id, on_delta=on_delta, on_reasoning=on_reasoning,
                              on_complete=on_complete, on_error=on_error,
                              on_confirmation=on_confirmation)
        th = threading.Thread(target=listener.run, daemon=True)
        th.start()

        prompt = build_prompt(messages, session_reused=reused)
        ok, resp = send_prompt_async(session_id, prompt, provider_id, model_id)
        if not ok:
            listener.stop()
            return _finalize("error", f"failed to send prompt: {str(resp)[:120]}")

        # 等到终态；客户端一旦断开立即收兵，不再空等满 UPSTREAM_TIMEOUT
        deadline = time.time() + UPSTREAM_TIMEOUT
        while th.is_alive() and time.time() < deadline and not self.client_gone:
            th.join(timeout=1)
        listener.stop()
        th.join(timeout=5)
        final_tokens["v"] = listener.tokens

        if not terminal["done"]:
            if listener.error:
                on_error(listener.error)
            elif listener.completed:
                _finalize("success")
            elif self.client_gone:
                # 客户端先走了：这是断开，不是上游超时（记账要分清）
                _finalize("client_disconnected")
            else:
                _finalize("error", "response timeout")

        if self.client_gone and entry.get("status") == "success":
            entry["status"] = "client_disconnected"

    # ---------- 非流式 ----------
    def _do_nonstream(self, rid, created, model_name, provider_id, model_id,
                      messages, session_id, reused, entry, started):
        state = {"text": [], "reasoning": [], "tokens": None,
                 "done": False, "error": None, "event": threading.Event()}
        tlock = threading.Lock()

        def _settle(err=None):
            with tlock:
                if state["done"]:
                    return
                state["done"] = True
                state["error"] = err
            state["event"].set()

        def on_delta(t):
            state["text"].append(t)

        def on_reasoning(t):
            state["reasoning"].append(t)

        listener = SSEListener(
            session_id,
            on_delta=on_delta, on_reasoning=on_reasoning,
            on_complete=lambda: _settle(),
            on_error=lambda e: _settle(e.get("message") if isinstance(e, dict) else str(e)),
        )
        th = threading.Thread(target=listener.run, daemon=True)
        th.start()

        prompt = build_prompt(messages, session_reused=reused)
        ok, resp = send_prompt_async(session_id, prompt, provider_id, model_id)
        if not ok:
            listener.stop()
            entry.update(status="error", error="failed to send prompt",
                         duration_ms=int((time.time() - started) * 1000))
            log_request(entry)
            return self._error(502, f"failed to send prompt: {str(resp)[:120]}", "api_error")

        state["event"].wait(timeout=UPSTREAM_TIMEOUT)
        listener.stop()
        th.join(timeout=5)
        state["tokens"] = listener.tokens

        text = "".join(state["text"])
        reasoning = "".join(state["reasoning"])
        err = state["error"]

        if not text and not err:
            # 兜底：只接受晚于本次请求开始的消息
            fallback = extract_assistant_text(get_messages(session_id),
                                              min_time=started - 1)
            if fallback:
                text = fallback

        if err and not text:
            entry.update(status="error", error=str(err),
                         duration_ms=int((time.time() - started) * 1000))
            log_request(entry)
            return self._error(502, f"upstream error: {err}", "api_error")

        message = {"role": "assistant", "content": text}
        if reasoning:
            message["reasoning_content"] = reasoning

        payload = {
            "id": rid, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": _usage_payload(state["tokens"], [text], [reasoning]),
        }
        entry.update(status="success" if not err else "error",
                     error=str(err) if err else None,
                     duration_ms=int((time.time() - started) * 1000),
                     response_preview=text[:120],
                     reasoning_preview=reasoning[:80],
                     tokens=state["tokens"])
        log_request(entry)
        self._send_json(200, payload)


def _usage_payload(tokens, text_parts, reasoning_parts):
    """上游 step-finish 的 tokens → OpenAI usage。

    上游口径（真机实测 3.1.1）：
      total = input + output + cache.read
      input 是「未命中缓存」的 prompt 部分，cache.read 是命中的部分，
      因此 OpenAI 的 prompt_tokens = input + cache.read。
    缺 tokens 时按字符粗略估算并标注 estimated。
    """
    if isinstance(tokens, dict) and tokens:
        raw_input = int(tokens.get("input") or 0)
        completion = int(tokens.get("output") or 0)
        reasoning = int(tokens.get("reasoning") or 0)
        cache = tokens.get("cache") or {}
        cached = int(cache.get("read") or 0) if isinstance(cache, dict) else 0
        prompt = raw_input + cached
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        if reasoning:
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
        if cached:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return usage
    text = "".join(text_parts) + "".join(reasoning_parts)
    if not text:
        return None
    completion = max(1, len(text) // 2)
    return {"prompt_tokens": 0, "completion_tokens": completion,
            "total_tokens": completion, "estimated": True}


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ============================================================
# 入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="TeleAgent OpenAI Proxy (minimal)")
    ap.add_argument("--host", default=None, help="监听地址（默认取 config.json，127.0.0.1）")
    ap.add_argument("--port", type=int, default=None, help="监听端口（默认取 config.json，8088）")
    ap.add_argument("--check", action="store_true", help="只做自检后退出")
    args = ap.parse_args()

    cfg = load_config()
    host = args.host or cfg.get("listen_host") or "127.0.0.1"
    port = args.port or int(cfg.get("listen_port") or 8088)
    init_inflight(cfg.get("max_inflight") or 2)

    if not get_session_key():
        print("[WARN] 暂无会话密钥；服务照常启动，首个请求会尝试自动提取。",
              file=sys.stderr)
        print("       如自动提取失败，请手动运行：python3 extract_sa_key.py",
              file=sys.stderr)
    status, resp = upstream_health()
    if status != 200:
        # 不因上游暂时不可用而拒绝启动：TeleAgent 可能正在重启换 key，
        # 首个请求会经 401 分支自愈；这里只告警。
        print(f"[WARN] super-agent 当前不可用 (status={status}): {str(resp)[:160]}",
              file=sys.stderr)

    if args.check:
        models = list_models()
        print(json.dumps({"version": VERSION, "upstream": status,
                          "models": models, "key": True}, ensure_ascii=False, indent=2))
        return 0 if models else 2

    cleanup_vision_images()
    print(f"""
TeleAgent OpenAI Proxy v{VERSION}（极简版）
  监听      : {host}:{port}
  上游      : {SUPER_AGENT_BASE}
  默认模型  : {cfg.get('default_model')}
  自动批准  : {cfg.get('auto_approve')}
  并发上限  : {cfg.get('max_inflight')}
  端点      : GET /health · GET /v1/models · POST /v1/chat/completions
""", file=sys.stderr)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
