"""Optional Langfuse tracing shared by the LangGraph pipeline and the ADK front door.

* NoopTracer is used when tracing is disabled or misconfigured: the agents behave identically.
* LangfuseTracer wraps the Langfuse Python SDK (v4.15.x; v5 requires OpenTelemetry >= 1.45, which
  conflicts with google-adk 2.12's <= 1.42.1 pin). A mask function redacts emails and key-like
  strings and truncates long lists before anything is exported. Langfuse spans, spans with gen_ai.*
  attributes and Google ADK's own spans (scope google.adk.*) are exported; other OTEL spans are not.
* Tracing never raises into the agent: every SDK call is wrapped.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-lf-[0-9A-Za-z\-]{8,}|pk-lf-[0-9A-Za-z\-]{8,})")


def redact(obj, max_list: int = 20):
    if isinstance(obj, str):
        return SECRET_RE.sub("<secret>", EMAIL_RE.sub("<email>", obj))
    if isinstance(obj, dict):
        return {k: redact(v, max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [redact(v, max_list) for v in list(obj)[:max_list]]
        return items + ([f"... {len(obj) - max_list} more"] if len(obj) > max_list else [])
    return obj


def langfuse_mask(*, data, **kwargs):
    try:
        return redact(data)
    except Exception:  # noqa: BLE001
        return "<unserializable>"


class _NoopSpan:
    def update(self, **kwargs):
        pass


class NoopTracer:
    enabled = False

    @contextmanager
    def span(self, name: str, as_type: str = "span", input=None, metadata=None):
        yield _NoopSpan()

    @contextmanager
    def session(self, session_id=None, user_id=None, tags=None):
        yield

    def score(self, name: str, value, trace_id=None, session_id=None, data_type=None, comment=None):
        pass

    def trace_url(self, trace_id):
        return None

    def current_trace_id(self):
        return None

    def flush(self):
        pass


class _SafeSpan:
    def __init__(self, obs):
        self._obs = obs

    def update(self, **kwargs):
        try:
            self._obs.update(**kwargs)
        except Exception:  # noqa: BLE001
            pass


class LangfuseTracer:
    enabled = True

    def __init__(self, public_key=None, secret_key=None, base_url=None, environment="notebook", client=None):
        if client is not None:
            self.client = client
            return
        from langfuse import Langfuse, is_default_export_span

        def export_filter(span) -> bool:
            scope = getattr(getattr(span, "instrumentation_scope", None), "name", "") or ""
            return is_default_export_span(span) or scope.startswith("google.adk")

        self.client = Langfuse(public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                               secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                               base_url=base_url or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com",
                               mask=langfuse_mask, environment=environment, should_export_span=export_filter)

    @contextmanager
    def span(self, name: str, as_type: str = "span", input=None, metadata=None):
        try:
            cm = self.client.start_as_current_observation(name=name, as_type=as_type, input=input, metadata=metadata)
            obs = cm.__enter__()
        except Exception:  # noqa: BLE001
            yield _NoopSpan()
            return
        try:
            yield _SafeSpan(obs)
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    @contextmanager
    def session(self, session_id=None, user_id=None, tags=None):
        """Group every trace created inside this block under one Langfuse session (e.g. one chat)."""
        try:
            from langfuse import propagate_attributes
            cm = propagate_attributes(session_id=session_id, user_id=user_id, tags=tags)
            cm.__enter__()
        except Exception:  # noqa: BLE001
            yield
            return
        try:
            yield
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    def score(self, name: str, value, trace_id=None, session_id=None, data_type=None, comment=None):
        """Attach a quality score to a trace (or session) in Langfuse."""
        try:
            self.client.create_score(name=name, value=value, trace_id=trace_id, session_id=session_id,
                                     data_type=data_type, comment=comment)
        except Exception:  # noqa: BLE001
            pass

    def trace_url(self, trace_id):
        try:
            return self.client.get_trace_url(trace_id=trace_id) if trace_id else None
        except Exception:  # noqa: BLE001
            return None

    def current_trace_id(self):
        try:
            return self.client.get_current_trace_id()
        except Exception:  # noqa: BLE001
            return None

    def flush(self):
        try:
            self.client.flush()
        except Exception:  # noqa: BLE001
            pass


def make_tracer(enabled: bool):
    """Langfuse tracer if enabled and keys exist and auth succeeds; otherwise NoopTracer. Returns (tracer, message)."""
    if not enabled:
        return NoopTracer(), "Langfuse disabled (no-op tracer)."
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return NoopTracer(), "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY missing - using the no-op tracer."
    try:
        t = LangfuseTracer()
        if t.client.auth_check():
            return t, f"Langfuse enabled ({os.getenv('LANGFUSE_BASE_URL') or 'https://cloud.langfuse.com'})."
        return NoopTracer(), "Langfuse auth_check failed - check keys and LANGFUSE_BASE_URL. Using the no-op tracer."
    except Exception as exc:  # noqa: BLE001
        return NoopTracer(), f"Langfuse could not start ({type(exc).__name__}) - using the no-op tracer."


# ---------------------------------------------------------------- local + tee tracers ----
class _LocalSpan:
    def __init__(self, rec):
        self.rec = rec

    def update(self, **kw):
        if kw.get("model"):
            self.rec["model"] = kw["model"]
        usage = kw.get("usage_details") or {}
        for k in ("input", "output", "reasoning"):
            if usage.get(k) is not None:
                self.rec[f"{k}_tokens"] = usage[k]
        out = kw.get("output")
        if isinstance(out, dict):
            for k in ("status", "ok", "approved", "source", "error", "intent"):
                if k in out:
                    self.rec[k] = out[k]


class LocalTracer:
    """Records the same spans, model/usage and scores Langfuse would receive, in memory (no network).
    Useful before Langfuse is configured, and for showing traces as a table in the notebook."""
    enabled = False
    kind = "local"

    def __init__(self):
        import itertools
        self.spans, self.scores = [], []
        self._stack, self._session, self._ids = [], None, itertools.count(1)

    @contextmanager
    def span(self, name: str, as_type: str = "span", input=None, metadata=None):
        import time as _t
        if not self._stack:
            self._trace = f"local-{next(self._ids):04d}"
        rec = {"trace_id": self._trace, "session_id": self._session, "depth": len(self._stack), "name": name, "type": as_type,
               "model": None, "input_tokens": None, "output_tokens": None, "seconds": None}
        self.spans.append(rec)
        self._stack.append(rec)
        t0 = _t.time()
        try:
            yield _LocalSpan(rec)
        finally:
            rec["seconds"] = round(_t.time() - t0, 3)
            self._stack.pop()

    @contextmanager
    def session(self, session_id=None, user_id=None, tags=None):
        prev, self._session = self._session, session_id
        try:
            yield
        finally:
            self._session = prev

    def score(self, name: str, value, trace_id=None, session_id=None, data_type=None, comment=None):
        self.scores.append({"trace_id": trace_id, "session_id": session_id, "name": name, "value": value,
                            "data_type": data_type, "comment": comment})

    def trace_url(self, trace_id):
        return None

    def current_trace_id(self):
        return self._stack[0]["trace_id"] if self._stack else None

    def flush(self):
        pass

    def spans_frame(self, trace_id=None):
        import pandas as pd
        df = pd.DataFrame(self.spans)
        if trace_id is not None and len(df):
            df = df[df.trace_id == trace_id]
        if len(df):
            df = df.assign(name=df.depth.map(lambda d: "  " * d) + df.name)
        return df

    def scores_frame(self):
        import pandas as pd
        return pd.DataFrame(self.scores)


class TeeTracer:
    """Send every span, session and score to several tracers (e.g. Langfuse + LocalTracer)."""

    def __init__(self, *tracers):
        self.tracers = [t for t in tracers if t is not None]
        self.enabled = any(getattr(t, "enabled", False) for t in self.tracers)

    class _Multi:
        def __init__(self, spans):
            self.spans = spans

        def update(self, **kw):
            for s in self.spans:
                s.update(**kw)

    @contextmanager
    def span(self, name: str, as_type: str = "span", input=None, metadata=None):
        from contextlib import ExitStack
        with ExitStack() as stack:
            yield TeeTracer._Multi([stack.enter_context(t.span(name, as_type=as_type, input=input, metadata=metadata))
                                    for t in self.tracers])

    @contextmanager
    def session(self, session_id=None, user_id=None, tags=None):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for t in self.tracers:
                stack.enter_context(t.session(session_id=session_id, user_id=user_id, tags=tags))
            yield

    def score(self, name: str, value, trace_id=None, session_id=None, data_type=None, comment=None):
        for t in self.tracers:
            t.score(name, value, trace_id=trace_id, session_id=session_id, data_type=data_type, comment=comment)

    def trace_url(self, trace_id):
        return next((u for t in self.tracers if (u := t.trace_url(trace_id))), None)

    def current_trace_id(self):
        return next((i for t in self.tracers if (i := t.current_trace_id())), None)

    def flush(self):
        for t in self.tracers:
            t.flush()
