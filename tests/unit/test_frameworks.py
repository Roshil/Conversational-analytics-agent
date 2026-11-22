"""LangGraph (human approval, checkpointer, streaming), Google ADK (multi-agent team, guardrail callback)
and Langfuse (sessions, scores, generation usage) features - offline, with the DuckDB emulator and stand-ins."""
import asyncio
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from analytics_agent.bq import BQSettings
from analytics_agent.agents.analytics_graph import (AnalyticsDeps, approve, ask_with_approval, build_analytics_graph,
                                                    build_approval_graph, reject, run_pipeline)
from analytics_agent.agents.data_quality_reviewer import DataQualityReviewer
from analytics_agent.data_generation.generator import GenerationConfig
from analytics_agent.dq_bigquery import validate_dataset
from analytics_agent.llm import LLMCallMeta
from analytics_agent.notebook_ui import BudgetedLLM, CountingRunner, GeminiBudget, OfflineStandInModel, budgeted_adk_model
from analytics_agent.observability import LangfuseTracer, NoopTracer
from analytics_agent.testing import DuckDBRunner

pytestmark = pytest.mark.skipif(not Path("data/generated/clean/parquet/orders.parquet").exists(), reason="generate data first")
S = BQSettings(project_id="test-project-123")
DS = {"clean": S.dataset_clean, "corrupted": S.dataset_corrupted}


@pytest.fixture(scope="module")
def env():
    runner = CountingRunner(DuckDBRunner(S, "data/generated"))
    reports = {v: validate_dataset(runner, ds, GenerationConfig().as_of_ts) for v, ds in DS.items()}
    budget = GeminiBudget(10 ** 6)
    llm = BudgetedLLM(OfflineStandInModel(), budget)

    def deps(v, tracer=None):
        return AnalyticsDeps(llm=llm, runner=runner, settings=S, dataset=DS[v], tracer=tracer, use_llm_sql_reviewer=False,
                             dq_report_provider=lambda v=v: reports[v], dq_reviewer=DataQualityReviewer(None))
    return SimpleNamespace(runner=runner, budget=budget, deps=deps)


# ---------------------------------------------------------------- LangGraph ----
def test_human_approval_pauses_then_executes(env, monkeypatch):
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")          # all state types must be registered
    g = build_approval_graph(env.deps("clean"))
    before = env.runner.runs
    p = ask_with_approval("What was our monthly net sales over the last six months?", env.deps("clean"), g)
    assert p.awaiting_approval and p.sql and p.period == "2026-01-01..2026-06-30" and env.runner.runs == before
    r = approve(g, p, env.deps("clean"))
    assert r.status == "answered" and r.row_count == 6 and env.runner.runs == before + 1


def test_human_rejection_executes_nothing(env):
    g = build_approval_graph(env.deps("clean"))
    p = ask_with_approval("Which product categories generate the most sales?", env.deps("clean"), g)
    before = env.runner.runs
    r = reject(g, p, "not needed")
    assert r.status == "refused" and env.runner.runs == before and r.steps[-1]["step"] == "human_review"
    assert ask_with_approval("Delete all orders", env.deps("clean"), g).finished.status == "refused"


def test_streaming_shows_every_node(env):
    graph = build_analytics_graph(env.deps("clean"))
    nodes = [list(u)[0] for u in graph.stream({"question": "What is our order cancellation rate?", "history": [], "steps": []},
                                              stream_mode="updates")]
    assert nodes == ["precheck", "plan", "generate_sql", "validate", "review_sql", "execute", "answer", "verify", "caveats", "finalize"]


# ---------------------------------------------------------------- Langfuse ----
class _FakeObs:
    def __init__(self, log, name):
        self.log, self.name = log, name
    def update(self, **kw):
        self.log.append((self.name, kw))


class _FakeLangfuse:
    def __init__(self):
        self.updates, self.scores = [], []
    @contextmanager
    def start_as_current_observation(self, name, as_type="span", input=None, metadata=None):
        yield _FakeObs(self.updates, name)
    def get_current_trace_id(self):
        return "trace-1"
    def create_score(self, **kw):
        self.scores.append(kw)
    def get_trace_url(self, trace_id):
        return f"https://langfuse.example/trace/{trace_id}"
    def flush(self):
        pass


def test_langfuse_generation_usage_scores_and_session(env):
    fake = _FakeLangfuse()
    tracer = LangfuseTracer(client=fake)

    class MetaLLM:                                   # stand-in that reports token usage like Gemini does
        model = "gemini-test"
        def __init__(self, inner): self.inner = inner
        def generate(self, **kw):
            out, _ = self.inner.generate(**kw)
            return out, LLMCallMeta(model="gemini-test", attempts=1, input_tokens=120, output_tokens=30)
    d = env.deps("clean", tracer)
    d.llm = MetaLLM(OfflineStandInModel())
    with tracer.session(session_id="chat-1", user_id="analyst", tags=["test"]):
        r = run_pipeline("What is our order cancellation rate?", [], d)
    gen = [kw for name, kw in fake.updates if name in ("plan", "generate_sql", "generate_answer") and "usage_details" in kw]
    assert gen and gen[0]["model"] == "gemini-test" and gen[0]["usage_details"] == {"input": 120, "output": 30}
    tracer.score("verified", 1, trace_id=r.trace_id, data_type="BOOLEAN")
    assert fake.scores[0]["name"] == "verified" and fake.scores[0]["trace_id"] == "trace-1"
    assert tracer.trace_url("trace-1").endswith("trace-1")
    noop = NoopTracer()
    with noop.session(session_id="x"):
        noop.score("a", 1)
    assert noop.trace_url("x") is None


# ---------------------------------------------------------------- Google ADK team ----
def test_adk_team_routes_audits_and_blocks(env):
    pytest.importorskip("google.adk")
    from analytics_agent.agents.adk_team import AnalyticsTeam, offline_team_models
    team = AnalyticsTeam({v: env.deps(v) for v in DS}, {k: budgeted_adk_model(m, env.budget) for k, m in offline_team_models().items()})
    fanout = (f"SELECT SUM(o.order_total) AS total FROM `{S.project_id}.{DS['clean']}.orders` o JOIN "
              f"`{S.project_id}.{DS['clean']}.order_items` li ON li.order_id = o.order_id WHERE o.order_status IN ('delivered')")
    canonical_ok = (f"SELECT COUNT(*) AS orders FROM `{S.project_id}.{DS['clean']}.orders` WHERE order_date BETWEEN @start_date AND @end_date")

    async def go():
        sid = await team.new_session()
        out = [await team.ask(t, sid) for t in [
            "What were monthly net sales in the first half of 2026?",
            "And which product category contributed the most in that period?",
            "Can I trust these numbers? Check the corrupted dataset.",
            f"Please audit this SQL: {fanout}",
            f"Please audit this SQL: {canonical_ok}",
            "Show total net sales from the staging dataset."]]
        return out, await team.state(sid)
    turns, state = asyncio.run(go())
    assert turns[0].route == ["coordinator", "sales_analyst"] and turns[0].verified
    assert "Electronics" in turns[1].answer
    assert "data_steward" in turns[2].route and "not fit for use" in turns[2].answer
    assert "NEEDS CHANGES" in turns[3].answer and "fan-out" in turns[3].answer
    assert "APPROVED" in turns[4].answer
    assert turns[5].blocked_calls and "not approved" in turns[5].answer
    assert state["blocked_tool_calls"] == 1 and state["tool_calls"] >= 6


def test_adk_team_turn_is_bounded(env):
    pytest.importorskip("google.adk")
    from analytics_agent.agents.adk_team import AnalyticsTeam
    from analytics_agent.testing import adk_call, make_scripted_adk_llm
    ping_pong = {"coordinator": make_scripted_adk_llm(lambda r: adk_call("transfer_to_agent", agent_name="sales_analyst")),
                 "sales_analyst": make_scripted_adk_llm(lambda r: adk_call("transfer_to_agent", agent_name="data_steward")),
                 "data_steward": make_scripted_adk_llm(lambda r: adk_call("transfer_to_agent", agent_name="sales_analyst")),
                 "sql_auditor": make_scripted_adk_llm(lambda r: adk_call("transfer_to_agent", agent_name="sales_analyst"))}
    team = AnalyticsTeam({v: env.deps(v) for v in DS}, ping_pong, max_llm_calls_per_turn=6)

    async def go():
        sid = await team.new_session()
        try:
            return await asyncio.wait_for(team.ask("loop forever", sid), timeout=30)
        except Exception as exc:  # ADK raises when the limit is hit
            return exc
    out = asyncio.run(go())
    assert not isinstance(out, asyncio.TimeoutError)      # bounded: it ends instead of looping forever
