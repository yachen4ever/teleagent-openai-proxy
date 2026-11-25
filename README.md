---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'd90e64ea-64f4-49bb-acbf-8be5f9ac88cb'
  PropagateID: 'd90e64ea-64f4-49bb-acbf-8be5f9ac88cb'
  ReservedCode1: 'f5a009ce-34c5-4d59-8c67-b58a2ed012ff'
  ReservedCode2: 'f5a009ce-34c5-4d59-8c67-b58a2ed012ff'
---

# TeleAgent OpenAI-Compatible Proxy · 极简版 v3.0.0

把本地 [TeleAgent](https://teleagent.cn)（**3.1.x**）的 super-agent 暴露成 OpenAI 兼容接口。
纯 Python 标准库，零第三方依赖，两个文件即可部署。

> **v3.0.0 是一次重写，不是增量**：只有一个文件 `openai_proxy_v3.py`，
> 只保留 `/v1/chat/completions`、`/v1/models`、`/health`。
> 控制台、embeddings、来源追踪、tool 文本协议、确认注入接口全部删除。
> 取舍与理由见 [docs/上游2.0.0审计报告.md](docs/上游2.0.0审计报告.md)。

## 为什么需要它

TeleAgent 3.x 把签名密钥从「进程环境变量」搬到了 `ais` 进程的内存里（经 stdin boot handoff 注入，
不落盘、不进 env）。所以旧版代理「读 scheduler 环境变量」的路径彻底失效，
必须用 `extract_sa_key.py` 现场提取一次，之后代理就能长期无特权运行。

## 部署（以 macOS 为例）

```bash
mkdir -p ~/code/teleagent-openai-proxy-v3
cp openai_proxy_v3.py extract_sa_key_v3.py ~/code/teleagent-openai-proxy-v3/
cd ~/code/teleagent-openai-proxy-v3
mv extract_sa_key_v3.py extract_sa_key.py     # 代理按这个名字找它

# 1) 提取密钥（需 sudo：tcpdump + lldb，约 40 秒；密钥只写本地 session.key 0600）
python3 extract_sa_key.py

# 2) 自检（不启动服务）
python3 openai_proxy_v3.py --check

# 3) 启动
python3 openai_proxy_v3.py            # 默认 127.0.0.1:8088
```

首次启动会生成 `config.json`（0600），里面自动带一个随机 `api_key`，用它调用即可。

## 调用

```bash
KEY=$(python3 -c "import json;print(json.load(open('config.json'))['api_key'])")

curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"NewApi/chat-pro","messages":[{"role":"user","content":"你好"}]}'
```

Python SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key=KEY)
r = client.chat.completions.create(
    model="NewApi/chat-pro",
    messages=[{"role": "user", "content": "你好"}],
    extra_body={"session_title": "我的会话"},   # 可选：传了就按标题复用上下文
)
```

## 行为约定

| 项 | 行为 |
|---|---|
| 未传 `session_title` | **每次新建会话**，由 super-agent 自动命名（OpenAI 无状态语义） |
| 传了 `session_title` | 按标题复用已有会话；复用时只发最新一条消息（避免上下文双重累积） |
| 思考内容 | 上游 `reasoning` part 以 `reasoning_content` 下发（流式 delta / 非流式字段） |
| `include_usage` | 返回真实 token 数，`prompt_tokens = input + cache.read`，含 `cached_tokens` |
| 权限/问题确认 | 自动回 `once`（`config.json` 的 `auto_approve` 可关），避免客户端挂死 |
| 图片 | 只支持 `data:` URI；落临时文件后提示模型用 `image_understanding` 查看 |
| 工具调用 | **不注入 tools**。上游只有纯文本 prompt 通道，需要 function calling 请直连 |
| 密钥失效 | 401 时自动跑提取器刷新（90 秒节流 + 文件锁），TeleAgent 重启换 key 无需人工干预 |

## config.json

```json
{
  "api_key": "sk-local-…",          // 调用方鉴权（Bearer），自动生成，可改
  "default_model": "NewApi/chat-pro",
  "auto_approve": true,              // 确认请求自动答复
  "max_inflight": 2,                 // 全局并发上限（模拟单人节奏，保护上游）
  "listen_host": "127.0.0.1",        // 只监听本机
  "listen_port": 8088
}
```

## 关于"反代检测"

本方案**不做任何指纹伪装**：不改 device-meta/installId/UA，不代下载外部图片，
不写人工会话标题。云端请求始终由官方 Go 二进制自己发出，所以厂商看到的就是官方客户端。
残余可观测性主要在"用量曲线"（单账户高频会异常），因此请按自用级频率使用。
详细分析见审计报告第四节。

## 常驻运行（launchd）

```bash
cat > ~/Library/LaunchAgents/io.local.teleagent-proxy-v3.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.local.teleagent-proxy-v3</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/python3</string>
    <string>REPLACE_ME/openai_proxy_v3.py</string>
  </array>
  <key>WorkingDirectory</key><string>REPLACE_ME</string>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
</dict></plist>
PLIST
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/io.local.teleagent-proxy-v3.plist
```

## 测试

```bash
python3 -m unittest discover -s tests -v    # 27 项，无第三方依赖
```

## 与上游的关系

本仓库是 [lizhun781021/teleagent-openai-proxy](https://github.com/lizhun781021/teleagent-openai-proxy)
的 fork。上游 2.0.0 提供了 3.x 密钥提取的正确方案（本版继承），但其代码基线仍是 1.9.10，
缺少本 fork 自 v1.10.0 起积累的 25 项稳定性修复（增量 UTF-8、SSE 状态行校验、终态锁存、
断连收尾、会话串行化、兜底时间过滤等）。因此 v3.0.0 以本 fork 为基座合入密钥层，
而不是反向从上游拉取。逐条证据见审计报告第二节。
