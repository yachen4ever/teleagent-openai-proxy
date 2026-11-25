#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_sa_key.py — 提取 TeleAgent 3.x 的 super-agent 本地签名密钥（fork 适配版）
============================================================================
原理（与上游 2.0.0 一致，额外做了容错与检索面收敛）：
  1. 从 ais 的 process.json 定位 ais PID（3.x 沙箱内 pgrep 不可用，用 ps 兜底）
  2. sudo tcpdump 抓 4397 端口的本地回环入向流量，取出 ais 发出的真实签名请求
     （X-Sa-* 头；优先无 query 的 GET 作为干净样本）
  3. sudo lldb attach + save-core 转储 ais 堆内存（90s 超时 + 残留调试器清理）
  4. 堆内 43 位 base64url 串作候选（出现≥2 次者优先）逐个验签
  5. 命中后对其余样本交叉验证，再对 super-agent 实机验证（GET /plugin）
  6. 原子写入 session.key（0600），立即删除 core 与 pcap

用法：
  python3 extract_sa_key.py          提取并写 session.key
  python3 extract_sa_key.py --check  只校验现有 session.key 是否仍然有效

退出码：0 成功 / 1 前置失败 / 2 无 key 文件 / 3 端口不可达 / 4 key 失效
        5 转储失败 / 6 无样本 / 7 验签未命中 / 8 交叉或实机验证失败

注意：本脚本只在本机运行，密钥只写本地文件，不上传、不打印完整值。
"""

import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

PROXY_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(PROXY_DIR, "session.key")
LOCK_FILE = os.path.join(tempfile.gettempdir(), "extract_sa_key.lock")

# 3.x：super-agent 在 4397 起，ais 可能让它顺延；一并探测
PORTS = (4397, 4398, 4399)
CAP_SECONDS = 22
LLDB_TIMEOUT = 90
MIN_CORE_BYTES = 1_000_000
KEY_RE = re.compile(rb"[A-Za-z0-9_\-]{43}")

AIS_PROCESS_JSON = os.path.expanduser("~/.local/share/TeleAgent/ais/process.json")


def log(msg):
    print(f"[extract] {msg}", file=sys.stderr, flush=True)


def b64u(raw):
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def sh(cmd, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- 定位
def find_ais_pid():
    try:
        with open(AIS_PROCESS_JSON, "r", encoding="utf-8") as f:
            pid = int(json.load(f).get("pid") or 0)
        if pid > 0:
            os.kill(pid, 0)
            return pid
    except Exception:
        pass
    try:
        out = sh(["ps", "axo", "pid,command"], timeout=10).stdout
        for line in out.splitlines():
            if "runtimes/ais/bin/ais" in line and "grep" not in line:
                parts = line.split(None, 1)
                if parts and parts[0].isdigit():
                    return int(parts[0])
    except Exception:
        pass
    return None


def find_sa_port():
    for port in PORTS:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return port
        except urllib.error.HTTPError:
            # 401/404 也说明端口上有 HTTP 服务
            return port
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- 调试器安全
def orphan_debuggers(pid):
    found = []
    try:
        out = sh(["ps", "axo", "pid,command"], timeout=10).stdout
        for line in out.splitlines():
            if "grep" in line or "axo" in line:
                continue
            if f"lldb -p {pid}" in line or ("debugserver" in line and "native-regs" in line):
                p = line.split(None, 1)[0]
                if p.isdigit():
                    found.append(int(p))
    except Exception:
        pass
    return found


def cleanup_debuggers(pid):
    """确保 ais 不会因为我们停在 frozen 状态。"""
    for sig in ("-TERM", "-KILL"):
        pids = orphan_debuggers(pid)
        if not pids:
            break
        for p in pids:
            try:
                sh(["sudo", "-n", "kill", sig, str(p)], timeout=5)
            except Exception:
                pass
        time.sleep(2)
    try:
        sh(["sudo", "-n", "kill", "-CONT", str(pid)], timeout=5)
    except Exception:
        pass


def dump_core(pid, path):
    cleanup_debuggers(pid)
    if orphan_debuggers(pid):
        log("残留调试器无法清理，放弃本次转储")
        return False

    cmd = ["sudo", "-n", "/usr/bin/lldb", "-p", str(pid), "--batch",
           "-o", f"process save-core {path}", "-o", "detach"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        proc.wait(timeout=LLDB_TIMEOUT)
    except subprocess.TimeoutExpired:
        log(f"lldb 超时({LLDB_TIMEOUT}s)，终止并恢复 ais")
        for sig in ("-TERM", "-KILL"):
            try:
                sh(["sudo", "-n", "kill", sig, str(proc.pid)], timeout=5)
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue
        cleanup_debuggers(pid)
        return False

    cleanup_debuggers(pid)
    if not (os.path.exists(path) and os.path.getsize(path) >= MIN_CORE_BYTES):
        return False
    # lldb 以 root 运行，转储文件归 root；读取前交给当前用户
    try:
        sh(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", path], timeout=20)
        sh(["sudo", "-n", "chmod", "600", path], timeout=20)
    except Exception:
        pass
    return os.access(path, os.R_OK)


# 头名大小写：发送端代码用 X-SA-*，而抓到的字节流可能被网络栈规范化成 X-Sa-*。
# 用 IGNORECASE 兼容两种，避免某版本漂移后静默失效（结果为 exit 6 无从诊断）。
SAMPLE_RE = re.compile(
    rb"(GET|POST|PUT|DELETE) (\S+?) HTTP/1\.1\r?\n[^\x00]{0,800}?"
    rb"X-SA-Nonce: ([A-Za-z0-9]+)\r?\n[^\x00]{0,200}?"
    rb"X-SA-Signature: ([A-Za-z0-9_\-]{43})\r?\n[^\x00]{0,150}?"
    rb"X-SA-Timestamp: (\d+)",
    re.IGNORECASE,
)


def _secure_tmp(suffix):
    """预创建 0600 临时文件（mkstemp）而非 mktemp 返回的裸路径。

    mktemp 只返回路径不建文件，本机其它用户可在间隙里用 symlink 抢占，
    让随后以 root 执行的 tcpdump/lldb 写到任意位置（TOCTOU）。
    """
    fd, path = tempfile.mkstemp(prefix="sa_", suffix=suffix, dir=tempfile.gettempdir())
    try:
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    return path


def capture_samples(port):
    pcap = _secure_tmp(".pcap")
    filt = " or ".join(f"tcp dst port {p}" for p in PORTS)
    proc = subprocess.Popen(["sudo", "-n", "tcpdump", "-i", "lo0", filt,
                             "-s", "0", "-w", pcap],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(CAP_SECONDS)
    try:
        sh(["sudo", "-n", "kill", "-INT", str(proc.pid)], timeout=5)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    time.sleep(1)

    try:
        text = sh(["sudo", "-n", "tcpdump", "-r", pcap, "-A", "-nn"], timeout=90).stdout
        data = text.encode("utf-8", "ignore")
    except Exception:
        data = b""
    finally:
        try:
            sh(["sudo", "-n", "rm", "-f", pcap], timeout=15)
        except Exception:
            pass

    samples = [{"method": m.decode(), "path": p.decode(), "nonce": n.decode(),
                "sig": s.decode(), "ts": t.decode()}
               for m, p, n, s, t in SAMPLE_RE.findall(data)]
    clean = [s for s in samples if "?" not in s["path"] and s["method"] == "GET"]
    return clean or samples


# ---------------------------------------------------------------- 验签
def sign_with(key, sample):
    payload = f"local-v1\n{sample['method']}\n{sample['path']}\n{sample['ts']}\n{sample['nonce']}"
    return b64u(hmac.new(key, payload.encode(), hashlib.sha256).digest())


def verify_live(key, port):
    try:
        ts = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex[:24]
        sample = {"method": "GET", "path": "/plugin", "ts": ts, "nonce": nonce}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/plugin")
        for k, v in {"X-SA-Sign-Version": "local-v1", "X-SA-Timestamp": ts,
                     "X-SA-Nonce": nonce,
                     "X-SA-Signature": sign_with(key.encode(), sample)}.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------- 主流程
def acquire_lock():
    import fcntl
    try:
        f = open(LOCK_FILE, "w")
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except Exception:
        return None


def check_mode():
    if not os.path.exists(KEY_FILE):
        log("session.key 不存在")
        return 2
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()
    port = find_sa_port()
    if not port:
        log("super-agent 端口不可达")
        return 3
    if verify_live(key, port):
        log(f"session.key 有效（{key[:6]}…，端口 {port}）")
        return 0
    log("session.key 已失效，需要重新提取")
    return 4


def extract():
    pid = find_ais_pid()
    if not pid:
        log("未找到 ais 进程；请确认 TeleAgent 正在运行")
        return 1
    port = find_sa_port()
    if not port:
        log("super-agent 端口不可达（4397-4399）")
        return 1
    log(f"ais PID={pid}，super-agent 端口={port}")

    log(f"抓包 {CAP_SECONDS}s 采样签名请求…")
    samples = capture_samples(port)
    if not samples:
        log("窗口内没有签名请求；请在 TeleAgent 里随便发一句话后重试")
        return 6
    log(f"获得 {len(samples)} 条样本（首条 {samples[0]['method']} {samples[0]['path'][:60]}）")

    core = _secure_tmp(".core")
    log(f"转储 ais 内存（PID={pid}）…")
    try:
        if not dump_core(pid, core):
            log("转储失败")
            return 5
        with open(core, "rb") as f:
            blob = f.read()
    except Exception as e:
        log(f"读取转储失败: {e}")
        return 5
    finally:
        try:
            sh(["sudo", "-n", "rm", "-f", core], timeout=30)
        except Exception:
            pass
        for p in os.listdir(tempfile.gettempdir()):
            if p.startswith("sa_core_"):
                try:
                    os.unlink(os.path.join(tempfile.gettempdir(), p))
                except Exception:
                    pass
    log(f"转储 {len(blob) / 1e6:.1f} MB，已读取并删除")

    counts = {}
    for m in KEY_RE.finditer(blob):
        c = m.group()
        counts[c] = counts.get(c, 0) + 1
    strong = [c for c, n in counts.items()
              if n >= 2 and re.search(rb"[A-Z]", c)
              and re.search(rb"[a-z]", c) and re.search(rb"[0-9]", c)]
    weak = [c for c in counts if c not in strong]
    log(f"候选：强 {len(strong)} / 弱 {len(weak)}")

    head = samples[0]
    hit = None
    for pool in (strong, weak):
        for c in pool:
            if sign_with(c, head) == head["sig"]:
                hit = c
                break
        if hit:
            break
    if not hit:
        log("验签未命中（密钥格式或签名算法可能变更，需要更新本脚本）")
        return 7

    if len(samples) > 1:
        if not all(sign_with(hit, s) == s["sig"] for s in samples[1:]):
            log("多样本交叉验证失败（样本中混入了非 ais 流量）")
            return 8

    key = hit.decode()
    if not verify_live(key, port):
        log("内存命中但实机验证失败（super-agent 可能刚重启换 key）")
        return 8

    tmp = KEY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(key)
    os.chmod(tmp, 0o600)
    os.replace(tmp, KEY_FILE)
    log(f"密钥已写入 session.key（{key[:6]}…，端口 {port} 实机验证通过）")
    return 0


def main():
    if "--check" in sys.argv:
        return check_mode()

    lock = acquire_lock()
    if lock is None:
        log("已有提取进程在运行，跳过")
        return 1
    try:
        return extract()
    finally:
        try:
            lock.close()
            os.unlink(LOCK_FILE)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
