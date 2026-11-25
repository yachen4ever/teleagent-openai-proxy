#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3.0.0 核心逻辑单测（标准库 unittest，无第三方依赖）。

运行：python3 -m unittest discover -s tests -v
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai_proxy_v3 as P  # noqa: E402


class TestSigning(unittest.TestCase):
    def setUp(self):
        self._old_env = os.environ.get("SUPER_AGENT_LOCAL_SESSION_KEY")
        os.environ["SUPER_AGENT_LOCAL_SESSION_KEY"] = "test-key-0123456789abcdefghijklmn"
        with P._session_key_lock:
            P._cached_session_key = None
            P._cached_session_key_time = 0

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop("SUPER_AGENT_LOCAL_SESSION_KEY", None)
        else:
            os.environ["SUPER_AGENT_LOCAL_SESSION_KEY"] = self._old_env
        with P._session_key_lock:
            P._cached_session_key = None
            P._cached_session_key_time = 0

    def test_signature_is_base64url_without_padding(self):
        headers = P.sign_request("GET", "/plugin")
        sig = headers["X-SA-Signature"]
        self.assertEqual(len(sig), 43)
        self.assertNotIn("=", sig)
        self.assertNotIn("+", sig)
        self.assertNotIn("/", sig)
        self.assertEqual(headers["X-SA-Sign-Version"], "local-v1")
        self.assertTrue(headers["X-SA-Timestamp"].isdigit())
        self.assertGreaterEqual(len(headers["X-SA-Nonce"]), 16)

    def test_signature_matches_spec(self):
        """独立复算签名串，确保与 super-agent 的 local-v1 规范一致。"""
        headers = P.sign_request("POST", "/session/abc/prompt_async")
        payload = "\n".join(["local-v1", "POST", "/session/abc/prompt_async",
                             headers["X-SA-Timestamp"], headers["X-SA-Nonce"]])
        expected = base64.urlsafe_b64encode(
            hmac.new(b"test-key-0123456789abcdefghijklmn",
                     payload.encode(), hashlib.sha256).digest()
        ).decode().rstrip("=")
        self.assertEqual(headers["X-SA-Signature"], expected)

    def test_nonce_unique(self):
        nonces = {P.sign_request("GET", "/health")["X-SA-Nonce"] for _ in range(20)}
        self.assertEqual(len(nonces), 20)


class TestKeyResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.dirname(os.path.abspath(__file__))
        self._old_key_file = P.KEY_FILE
        self._old_env = os.environ.pop("SUPER_AGENT_LOCAL_SESSION_KEY", None)
        with P._session_key_lock:
            P._cached_session_key = None
            P._cached_session_key_time = 0

    def tearDown(self):
        P.KEY_FILE = self._old_key_file
        if self._old_env:
            os.environ["SUPER_AGENT_LOCAL_SESSION_KEY"] = self._old_env
        with P._session_key_lock:
            P._cached_session_key = None
            P._cached_session_key_time = 0

    def test_env_var_wins(self):
        os.environ["SUPER_AGENT_LOCAL_SESSION_KEY"] = "env-key"
        self.assertEqual(P.get_session_key(), "env-key")

    def test_file_used_when_no_env(self):
        path = os.path.join(self.tmp, "session.key.testunit")
        with open(path, "w") as f:
            f.write("file-key")
        P.KEY_FILE = path
        try:
            self.assertEqual(P.get_session_key(), "file-key")
        finally:
            os.unlink(path)

    def test_refresh_rereads_file(self):
        path = os.path.join(self.tmp, "session.key.testunit2")
        with open(path, "w") as f:
            f.write("first")
        P.KEY_FILE = path
        try:
            self.assertEqual(P.get_session_key(), "first")
            with open(path, "w") as f:
                f.write("second")
            with P._session_key_lock:
                P._cached_session_key_time = 0
            self.assertEqual(P.get_session_key(), "second")
        finally:
            os.unlink(path)


class TestTitleMatching(unittest.TestCase):
    def test_strips_date_and_time(self):
        self.assertEqual(P._session_match_key("脚本|Python|2026-11-26|07:12"),
                         "脚本|Python|2026-11-26")

    def test_no_date_not_stripped(self):
        # 自定义标题里的时间不能被剥掉（否则不同时段会互相复用）
        self.assertEqual(P._session_match_key("每日汇报|09:00"), "每日汇报|09:00")

    def test_channel_title_untouched(self):
        self.assertEqual(P._session_match_key("企微|私聊|user1"), "企微|私聊|user1")

    def test_empty(self):
        self.assertEqual(P._session_match_key(""), "")
        self.assertIsNone(P._session_match_key(None))


class TestPromptBuild(unittest.TestCase):
    def test_single_user_no_system(self):
        msgs = [{"role": "user", "content": "你好"}]
        self.assertEqual(P.build_prompt(msgs), "你好")

    def test_system_plus_user(self):
        msgs = [{"role": "system", "content": "你是助手"},
                {"role": "user", "content": "在吗"}]
        self.assertEqual(P.build_prompt(msgs), "你是助手\n\n在吗")

    def test_reused_sends_only_last_user(self):
        msgs = [{"role": "user", "content": "旧问题"},
                {"role": "assistant", "content": "旧答案"},
                {"role": "user", "content": "新问题"}]
        out = P.build_prompt(msgs, session_reused=True)
        self.assertEqual(out, "新问题")

    def test_assistant_with_tool_calls_null_content(self):
        msgs = [
            {"role": "user", "content": "查天气"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}]},
        ]
        out = P.build_prompt(msgs)
        self.assertIn("get_weather", out)
        self.assertNotIn("None", out)

    def test_content_parts_text(self):
        msgs = [{"role": "user",
                 "content": [{"type": "text", "text": "第一段"},
                             {"type": "text", "text": "第二段"}]}]
        self.assertEqual(P.build_prompt(msgs), "第一段\n第二段")

    def test_tool_message_keeps_call_id(self):
        msgs = [{"role": "tool", "content": "晴 26 度", "tool_call_id": "call_9"}]
        self.assertIn("[tool_result:call_9]", P.build_prompt(msgs))

    def test_null_content_is_empty(self):
        self.assertEqual(P._content_to_text(None), "")
        self.assertEqual(P._content_to_text(""), "")


class TestUsagePayload(unittest.TestCase):
    def test_from_upstream_tokens(self):
        # 上游 total = input + output + cache.read；prompt = input + cache.read
        usage = P._usage_payload({"total": 24296, "input": 100, "output": 50,
                                  "reasoning": 10, "cache": {"read": 20, "write": 0}},
                                 ["x"], [])
        self.assertEqual(usage["prompt_tokens"], 120)
        self.assertEqual(usage["completion_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 170)
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 10)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 20)

    def test_estimates_when_missing(self):
        usage = P._usage_payload(None, ["四" * 20], [])
        self.assertTrue(usage["estimated"])
        self.assertGreater(usage["completion_tokens"], 0)

    def test_none_when_no_text(self):
        self.assertIsNone(P._usage_payload(None, [], []))


class TestFallbackFilter(unittest.TestCase):
    def test_rejects_old_message(self):
        msgs = [{"info": {"role": "assistant", "time": {"created": 100}},
                 "parts": [{"type": "text", "text": "旧回答"}]}]
        self.assertIsNone(P.extract_assistant_text(msgs, min_time=200))

    def test_accepts_new_message(self):
        msgs = [{"info": {"role": "assistant", "time": {"created": 300}},
                 "parts": [{"type": "text", "text": "新回答"}]}]
        self.assertEqual(P.extract_assistant_text(msgs, min_time=200), "新回答")

    def test_missing_time_is_untrusted(self):
        msgs = [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "无时间"}]}]
        self.assertIsNone(P.extract_assistant_text(msgs, min_time=1))


class TestListenerTerminal(unittest.TestCase):
    """终态锁存：complete 与 error 只能触发一次（禁止双 [DONE]）。"""

    def test_terminal_fires_once(self):
        calls = {"complete": 0, "error": 0}
        lsn = P.SSEListener("ses_x",
                            on_complete=lambda: calls.__setitem__("complete", calls["complete"] + 1),
                            on_error=lambda e: calls.__setitem__("error", calls["error"] + 1))
        lsn._handle({"payload": {"type": "session.status",
                                 "properties": {"sessionID": "ses_x",
                                                "status": {"type": "idle"}}}})
        lsn._handle({"payload": {"type": "message.updated",
                                 "properties": {"sessionID": "ses_x",
                                                "info": {"role": "assistant", "id": "m1",
                                                         "error": {"message": "boom"}}}}})
        self.assertEqual(calls["complete"], 1)
        self.assertEqual(calls["error"], 0)

    def test_other_session_ignored(self):
        seen = []
        lsn = P.SSEListener("ses_mine", on_delta=seen.append)
        lsn._handle({"payload": {"type": "message.part.updated",
                                 "properties": {"sessionID": "ses_other",
                                                "part": {"type": "text", "id": "p1",
                                                         "messageID": "m1", "text": "别人的"}}}})
        self.assertEqual(seen, [])

    def test_reasoning_routed_separately(self):
        text, reasoning = [], []
        lsn = P.SSEListener("s", on_delta=text.append, on_reasoning=reasoning.append)
        lsn._handle({"payload": {"type": "message.updated",
                                 "properties": {"sessionID": "s",
                                                "info": {"role": "assistant", "id": "m1"}}}})
        lsn._handle({"payload": {"type": "message.part.updated",
                                 "properties": {"sessionID": "s",
                                                "part": {"type": "reasoning", "id": "r1",
                                                         "messageID": "m1", "text": "先想"}}}})
        lsn._handle({"payload": {"type": "message.part.updated",
                                 "properties": {"sessionID": "s",
                                                "part": {"type": "text", "id": "t1",
                                                         "messageID": "m1", "text": "后说"}}}})
        self.assertEqual(text, ["后说"])
        self.assertEqual(reasoning, ["先想"])


    def test_reasoning_delta_has_text_field_upstream(self):
        """上游对 reasoning part 的增量也用 field='text'，只能靠 partID 区分。"""
        text, reasoning = [], []
        lsn = P.SSEListener("s", on_delta=text.append, on_reasoning=reasoning.append)
        lsn._handle({"payload": {"type": "message.updated",
                                 "properties": {"sessionID": "s",
                                                "info": {"role": "assistant", "id": "m1"}}}})
        # reasoning part 先登记，随后 delta 用 field="text" 到达
        lsn._handle({"payload": {"type": "message.part.updated",
                                 "properties": {"sessionID": "s",
                                                "part": {"type": "reasoning", "id": "r1",
                                                         "messageID": "m1"}}}})
        lsn._handle({"payload": {"type": "message.part.delta",
                                 "properties": {"sessionID": "s", "partID": "r1",
                                                "field": "text", "delta": "思考中"}}})
        # 正文 part 登记后同样用 field="text"
        lsn._handle({"payload": {"type": "message.part.updated",
                                 "properties": {"sessionID": "s",
                                                "part": {"type": "text", "id": "t1",
                                                         "messageID": "m1"}}}})
        lsn._handle({"payload": {"type": "message.part.delta",
                                 "properties": {"sessionID": "s", "partID": "t1",
                                                "field": "text", "delta": "正文"}}})
        self.assertEqual(reasoning, ["思考中"])
        self.assertEqual(text, ["正文"])

    def test_reasoning_field_delta_fallback(self):
        reasoning = []
        lsn = P.SSEListener("s", on_reasoning=reasoning.append)
        lsn._handle({"payload": {"type": "message.part.delta",
                                 "properties": {"sessionID": "s", "partID": "r1",
                                                "field": "reasoning", "delta": "想想"}}})
        self.assertEqual(reasoning, ["想想"])


class TestHandlerAuth(unittest.TestCase):
    def test_constant_time_compare(self):
        """不同长度也走 compare_digest，不抛异常。"""
        self.assertFalse(hmac.compare_digest("a", "longer-string"))
        self.assertTrue(hmac.compare_digest("same", "same"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
