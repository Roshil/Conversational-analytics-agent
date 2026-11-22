"""Regression test for the user-reported hang: google-genai's own retries slept through a 24-minute
retry-after on a daily-quota 429. A local fake server returns that exact error (no internet needed)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from analytics_agent.llm import GeminiStructuredClient, LLMError, _retry_after_seconds, classify_error, disable_sdk_retries

DAILY = ("You exceeded your current quota, please check your plan and billing details. See https://ai.google.dev/gemini-api/docs/rate-limits. "
         "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
         "description: Generate content requests per day limit exceeded on Free Tier. Please retry in 24m36.803s.")


class _Ping(BaseModel):
    status: str


@pytest.fixture()
def fake_server():
    state = {"n": 0, "message": DAILY, "retry_after": "1476"}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            state["n"] += 1
            body = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": state["message"]}}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", state["retry_after"])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()


def _client(base_url):
    from google import genai
    from google.genai import types
    c = genai.Client(api_key="test-key", http_options=types.HttpOptions(base_url=base_url))
    assert disable_sdk_retries(c)
    return c


def test_daily_quota_fails_fast_without_sdk_sleep(fake_server):
    base, state = fake_server
    llm = GeminiStructuredClient("gemini-x", client=_client(base), max_retries=2)
    t0 = time.time()
    with pytest.raises(LLMError) as e:
        llm.generate(system_instruction="s", prompt="p", schema=_Ping, max_output_tokens=32)
    assert e.value.kind == "quota" and state["n"] == 1 and time.time() - t0 < 5
    assert abs(e.value.retry_after_s - 1476.8) < 1


def test_long_retry_after_is_not_waited_for(fake_server):
    base, state = fake_server
    state["message"] = "Rate limit: requests per minute exceeded. Please retry in 90s."
    slept = []
    llm = GeminiStructuredClient("gemini-x", client=_client(base), max_retries=2, sleep=slept.append, max_wait_s=20)
    with pytest.raises(LLMError) as e:
        llm.generate(system_instruction="s", prompt="p", schema=_Ping, max_output_tokens=32)
    assert e.value.kind == "rate_limit" and slept == [] and state["n"] == 1 and "not waiting longer" in e.value.message


def test_short_retry_after_is_honoured_and_bounded(fake_server):
    base, state = fake_server
    state["message"] = "Rate limit: requests per minute exceeded. Please retry in 2s."
    slept = []
    llm = GeminiStructuredClient("gemini-x", client=_client(base), max_retries=2, sleep=slept.append)
    with pytest.raises(LLMError):
        llm.generate(system_instruction="s", prompt="p", schema=_Ping, max_output_tokens=32)
    assert slept == [2.0, 2.0] and state["n"] == 3            # 1 try + 2 bounded retries, server's 2 s honoured


def test_retry_after_parsing_and_classification():
    assert abs(_retry_after_seconds(DAILY) - 1476.803) < 0.01
    assert _retry_after_seconds("'retryDelay': '1476s'") == 1476.0
    from types import SimpleNamespace
    err = type("RateLimitError", (Exception,), {})("x")
    err.status_code, err.message = 429, DAILY
    assert classify_error(err).kind == "quota"
