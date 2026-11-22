"""Tests for notebook helpers: offline stand-in, Gemini budget, counting runner, SQL formatting, ADK stand-in."""
import asyncio
import re
from pathlib import Path

import pytest

from analytics_agent.bq import BQSettings
from analytics_agent.agents.analytics_graph import AnalyticsDeps, AnalyticsSession, run_pipeline
from analytics_agent.agents.data_quality_reviewer import DataQualityReviewer
from analytics_agent.evaluation import EVAL_SPECS, compare, load_gold
from analytics_agent.llm import LLMError
from analytics_agent.notebook_ui import (BudgetedLLM, CountingRunner, GeminiBudget, OfflineStandInModel, format_sql,
                                         preview_sql)
from analytics_agent import semantic_layer as sem
from analytics_agent.testing import DuckDBRunner

pytestmark = pytest.mark.skipif(not Path("data/generated/clean/parquet/orders.parquet").exists(), reason="generate data first")
S = BQSettings(project_id="test-project-123")


@pytest.fixture(scope="module")
def runner():
    return CountingRunner(DuckDBRunner(S, "data/generated"))


def deps(runner, llm, variant="clean"):
    return AnalyticsDeps(llm=llm, runner=runner, settings=S, dataset=S.dataset(variant), use_llm_sql_reviewer=False,
                         dq_reviewer=DataQualityReviewer(None))


def test_format_sql_changes_only_whitespace():
    for m in sem.METRICS:
        sql = m.render(S.refs(S.dataset_clean))
        assert re.sub(r"\s+", "", sql) == re.sub(r"\s+", "", format_sql(sql)), m.name
    assert "BETWEEN @start_date AND @end_date" in format_sql(sem.METRICS_BY_NAME["cancellation_rate"].render(S.refs(S.dataset_clean)))


def test_stand_in_answers_gold_questions(runner):
    gold = load_gold("data/manifests/expected_results_clean.json")
    llm = BudgetedLLM(OfflineStandInModel(), GeminiBudget(10_000))
    ok = sum(compare(s, gold[s.test_id]["expected_result"], run_pipeline(s.question, [], deps(runner, llm)).rows)[0] for s in EVAL_SPECS)
    assert ok == len(EVAL_SPECS)


def test_stand_in_follow_up_and_refusals(runner):
    llm = BudgetedLLM(OfflineStandInModel(), GeminiBudget(10_000))
    sess = AnalyticsSession(deps(runner, llm))
    sess.ask("What was our monthly net sales over the last six months?")
    r = sess.ask("Now break that down by product category.")
    assert r.metrics == ["net_sales_by_dimension"] and r.period == "2026-01-01..2026-06-30"
    assert run_pipeline("What is our customer acquisition cost?", [], deps(runner, llm)).status == "refused"
    assert run_pipeline("Show sales for the East region.", [], deps(runner, llm)).status == "clarification_needed"


def test_budget_blocks_calls_and_pipeline_degrades(runner):
    budget = GeminiBudget(1)
    llm = BudgetedLLM(OfflineStandInModel(), budget)
    before = runner.runs
    r = run_pipeline("What is our order cancellation rate?", [], deps(runner, llm))
    assert budget.used == 1 and r.status == "error" and "quota" in r.answer and runner.runs == before
    with pytest.raises(LLMError):
        budget.charge()


def test_preview_does_not_execute(runner):
    llm = BudgetedLLM(OfflineStandInModel(), GeminiBudget(10_000))
    before = runner.runs
    p = preview_sql("What is the average order value by region in Q2 2026?", deps(runner, llm))
    assert p.status == "ready" and p.period == "2026-04-01 to 2026-06-30" and runner.runs == before


def test_corrupted_rows_with_null_key_are_summarised(runner):
    llm = BudgetedLLM(OfflineStandInModel(), GeminiBudget(10_000))
    r = run_pipeline("What is the average order value by region?", [], deps(runner, llm, "corrupted"))
    assert r.answer.startswith("Across") and "Unknown" in r.answer


def test_offline_adk_front_door(runner):
    pytest.importorskip("google.adk")
    from analytics_agent.agents.adk_frontend import AnalyticsFrontDoor
    from analytics_agent.notebook_ui import budgeted_adk_model, offline_adk_model
    budget = GeminiBudget(10_000)
    llm = BudgetedLLM(OfflineStandInModel(), budget)
    fd = AnalyticsFrontDoor({"clean": deps(runner, llm), "corrupted": deps(runner, llm, "corrupted")},
                            budgeted_adk_model(offline_adk_model(), budget))

    async def go():
        sid = await fd.new_session()
        a = await fd.ask("What were monthly net sales in the first half of 2026?", sid)
        b = await fd.ask("And which product category contributed the most in that period?", sid)
        c = await fd.ask("What metrics can you calculate?", sid)
        return a, b, c
    a, b, c = asyncio.run(go())
    assert a.verified and b.verified and c.verified
    assert "Electronics" in b.answer and c.tool_calls[0]["tool"] == "list_metrics"
    assert budget.used > 0                         # ADK model calls are charged to the shared budget
