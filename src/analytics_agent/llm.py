"""Thin wrapper around the Gemini Developer API (google-genai SDK, Interactions API).

* Structured output: the Pydantic model's JSON schema is sent as
  response_format, and the reply is validated client-side with
  model_validate_json (the docs recommend validating in application code).
* Errors are classified into stable kinds so callers can tell an invalid key
  from a rate limit, an unknown model, a network failure or malformed output.
* Retries are bounded and only for transient kinds (rate limit, server
  unavailable, network). Quota exhaustion, auth and bad requests fail fast.
* The API key is read from os.environ at construction and never logged.

Verified against google-genai 2.31.0 and the Gemini API structured-output docs
(updated 2026-10-21). Live calls were NOT executed in the build environment.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

TRANSIENT_KINDS = {"rate_limit", "unavailable", "network", "timeout"}


class LLMError(Exception):
    """kind is one of: not_configured, auth, invalid_request, model_not_found,
    rate_limit, quota, unavailable, network, timeout, invalid_output, blocked."""

    def __init__(self, kind: str, message: str, *, code: Optional[int] = None, attempts: int = 1,
                 retry_after_s: Optional[float] = None):
        super().__init__(f"[{kind}] {message}")
        self.kind, self.message, self.code, self.attempts = kind, message, code, attempts
        self.retry_after_s = retry_after_s       # server-suggested wait, parsed from the error message

    @property
    def transient(self) -> bool:
        return self.kind in TRANSIENT_KINDS


@dataclass
class LLMCallMeta:
    model: str
    attempts: int = 0
    latency_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thought_tokens: Optional[int] = None
    status: Optional[str] = None
    errors: list = field(default_factory=list)


def _mro_names(exc: BaseException) -> set:
    return {c.__name__ for c in type(exc).__mro__}


def _retry_after_seconds(text: str) -> Optional[float]:
    """Parse 'Please retry in 24m36.8s' / 'retryDelay': '1476s' / 'retry in 5s' from an error message."""
    import re as _re
    m = _re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", text)
    if m:
        h, mnt, sec = m.groups()
        return int(h or 0) * 3600 + int(mnt or 0) * 60 + float(sec)
    m = _re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", text)
    return float(m.group(1)) if m else None


def disable_sdk_retries(client) -> bool:
    """Turn off google-genai's own retries for the Interactions API.

    By default the SDK retries 408/409/429/5XX itself (up to 4 times) and sleeps for the server's
    retry-after with no upper bound, so after a quota error ('retry in 24m') a notebook cell can hang
    for many minutes before this code sees the error. This client does its own bounded retries instead.
    Uses a private SDK attribute (verified on google-genai 2.31.0 and 2.34.0); returns False if unavailable.
    """
    try:
        from google.genai._gaos.utils.retries import RetryConfig
        client.interactions.sdk_configuration.retry_config = RetryConfig("none", None, False)
        return True
    except Exception:  # noqa: BLE001
        return False


def _from_status(code: Optional[int], text: str) -> LLMError:
    low = text.lower()
    if code == 429 or "resource_exhausted" in low:
        after = _retry_after_seconds(text)
        if "perday" in low or "per day" in low:          # daily quota: retrying today is pointless
            return LLMError("quota", text, code=code, retry_after_s=after)
        if "per minute" in low or "perminute" in low:
            return LLMError("rate_limit", text, code=code, retry_after_s=after)
        is_quota = "quota" in low and "rate limit" not in low
        return LLMError("quota" if is_quota else "rate_limit", text, code=code, retry_after_s=after)
    if code in (401, 403) or "permission_denied" in low or "unauthenticated" in low:
        return LLMError("auth", text, code=code)
    if code == 400 and ("api key" in low or "api_key" in low):
        return LLMError("auth", text, code=code)
    if code == 404 or "not_found" in low:
        return LLMError("model_not_found", text, code=code)
    if code is not None and code >= 500:
        return LLMError("unavailable", text, code=code)
    return LLMError("invalid_request", text, code=code)


def classify_error(exc: BaseException) -> LLMError:
    """Map SDK / transport exceptions to LLMError kinds.

    google-genai has two error families: google.genai.errors.APIError (attribute
    `code`, used by client.models.*) and the Interactions API's errors
    (APIStatusError / APIConnectionError / APITimeoutError with `status_code`,
    defined in a private module). We classify by attributes and class names so
    both families are handled without importing private modules.
    """
    import httpx

    if isinstance(exc, LLMError):
        return exc
    names = _mro_names(exc)
    msg = str(getattr(exc, "message", "") or exc)
    body = getattr(exc, "body", None) or getattr(exc, "details", None)
    text = msg if body is None or str(body) in msg else f"{msg} {body}"
    if "APITimeoutError" in names or isinstance(exc, httpx.TimeoutException):
        return LLMError("timeout", msg)
    if "APIConnectionError" in names or isinstance(exc, (httpx.TransportError, ConnectionError)):
        return LLMError("network", msg)
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):                     # Interactions API errors
        return _from_status(status_code, text)
    code = getattr(exc, "code", None)
    if isinstance(code, int) and "APIError" in names:    # google.genai.errors.APIError family
        status = str(getattr(exc, "status", "") or "")
        return _from_status(code, f"{text} {status}".strip())
    if isinstance(exc, OSError):
        return LLMError("network", msg)
    return LLMError("invalid_request", f"{type(exc).__name__}: {msg}")


class GeminiStructuredClient:
    """Call Gemini and get back a validated Pydantic object."""

    def __init__(self, model: str, *, api_key: Optional[str] = None, timeout_s: float = 90.0,
                 max_retries: int = 2, backoff_s: float = 2.0, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep, max_wait_s: float = 20.0):
        self.model = model
        self.max_wait_s = max_wait_s              # never wait longer than this for a retry-after
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        if client is not None:          # injected (tests / offline simulation)
            self._client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not key or key.startswith("<"):
                raise LLMError("not_configured", "GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
            from google import genai
            self._client = genai.Client(api_key=key)
        self.sdk_retries_disabled = disable_sdk_retries(self._client) if client is None else None

    def generate(self, *, system_instruction: str, prompt: str, schema: Type[T],
                 temperature: float = 0.0, max_output_tokens: int = 8192) -> tuple[T, LLMCallMeta]:
        meta = LLMCallMeta(model=self.model)
        t0 = time.time()
        last: Optional[LLMError] = None
        for attempt in range(1, self.max_retries + 2):        # 1 try + max_retries retries
            meta.attempts = attempt
            try:
                interaction = self._client.interactions.create(
                    model=self.model,
                    input=prompt,
                    system_instruction=system_instruction,
                    store=False,          # Interactions API stores requests by default (1 day free / 55 days paid)
                    response_format={"type": "text", "mime_type": "application/json",
                                     "schema": schema.model_json_schema()},
                    generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
                    timeout=self.timeout_s,
                )
                meta.latency_s = round(time.time() - t0, 2)
                return self._parse(interaction, schema, meta), meta
            except Exception as exc:  # noqa: BLE001 - classified below
                err = classify_error(exc)
                err.attempts = attempt
                meta.errors.append(err.kind)
                last = err
                if not err.transient or attempt > self.max_retries:
                    break
                wait = err.retry_after_s if err.retry_after_s is not None else self.backoff_s * (2 ** (attempt - 1))
                if wait > self.max_wait_s:                # e.g. "retry in 24m": stop now instead of hanging
                    err.message += f" (server asked to wait {wait:.0f}s; not waiting longer than {self.max_wait_s:.0f}s)"
                    break
                self._sleep(wait)
        meta.latency_s = round(time.time() - t0, 2)
        assert last is not None
        last.meta = meta  # type: ignore[attr-defined]
        raise last

    @staticmethod
    def _parse(interaction: Any, schema: Type[T], meta: LLMCallMeta) -> T:
        meta.status = str(getattr(interaction, "status", None))
        usage = getattr(interaction, "usage", None)
        if usage is not None:
            meta.input_tokens = getattr(usage, "total_input_tokens", None)
            meta.output_tokens = getattr(usage, "total_output_tokens", None)
            meta.thought_tokens = getattr(usage, "total_thought_tokens", None)
        if meta.status not in ("completed", "None"):
            raise LLMError("invalid_output", f"interaction status {meta.status!r} (e.g. output truncated or blocked)")
        text = getattr(interaction, "output_text", None)
        if not text:
            raise LLMError("invalid_output", "empty response text")
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError("invalid_output", f"response does not match {schema.__name__}: "
                                             f"{exc.error_count()} validation error(s)") from exc
