"""Integrity checks, the semantic-layer report, Gemini insight workflows, model routing and multi-month periods."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from analytics_agent.bq import BQSettings
from analytics_agent.agents.analytics_graph import AnalyticsDeps, run_pipeline
from analytics_agent.agents.data_quality_reviewer import DataQualityReviewer
from analytics_agent.agents.insights import (DrillDownFindings, DrillDownPlan, ExecutiveSummary, collect_kpis, drill_down,
                                             executive_summary)
from analytics_agent.evaluation import load_gold
from analytics_agent.integrity import dataset_inventory, integrity_checks, integrity_examples
from analytics_agent.llm import LLMError
from analytics_agent.notebook_ui import BudgetedLLM, GeminiBudget, ModelRouter, OfflineStandInModel
from analytics_agent.period import resolve_period
from analytics_agent.reports import semantic_layer_report
from analytics_agent.testing import DuckDBRunner, SchemaRoutedLLM

pytestmark = pytest.mark.skipif(not Path("data/generated/clean/parquet/orders.parquet").exists(), reason="generate data first")
S = BQSettings(project_id="test-project-123")
DS = {"clean": S.dataset_clean, "corrupted": S.dataset_corrupted}


@pytest.fixture(scope="module")
def runner():
    return DuckDBRunner(S, "data/generated")


def deps(runner, llm, variant="clean"):
    return AnalyticsDeps(llm=llm, runner=runner, settings=S, dataset=DS[variant], use_llm_sql_reviewer=False,
                         dq_reviewer=DataQualityReviewer(None))


def test_integrity_matches_planted_defects(runner):
    inv, issues = dataset_inventory(runner, S, DS)
    rows = {(r.variant, r.table): r.row_count for r in inv.itertuples()}
    assert rows[("corrupted", "customers")] == 2005 and rows[("corrupted", "payments")] == 15819 and issues.empty
    assert (integrity_checks(runner, S, DS["clean"]).issue_count == 0).all()
    bad = integrity_checks(runner, S, DS["corrupted"]).set_index(["table", "check", "columns"]).issue_count
    assert bad[("customers", "PRIMARY_KEY_DUPLICATE", "customer_id")] == 5
    assert bad[("payments", "BUSINESS_KEY_DUPLICATE", "provider_transaction_id")] == 15
    assert bad[("orders", "FOREIGN_KEY_ORPHAN", "customer_id")] == 8
    ex = integrity_examples(runner, S, DS["corrupted"], limit=5)
    assert len(ex) > 0 and "@example.com" not in str(ex.record.tolist())


def test_semantic_layer_report(runner, tmp_path):
    summary, sections, path = semantic_layer_report(runner, S, DS["clean"], load_gold("data/manifests/expected_results_clean.json"),
                                                    tmp_path / "r.md")
    assert (summary.result == "PASS").all() and len(sections) == 11 and "Semantic layer vs gold" in path.read_text()


def test_executive_summary_verification(runner):
    kpis = collect_kpis(runner, S, DS["clean"], "2026-01-01", "2026-06-30")
    ns = kpis["net_sales"]["rows"]
    good = ExecutiveSummary(headline=f"Net sales reached {ns[-1]['net_sales']:,} in {ns[-1]['month']}.",
                            highlights=[f"Lowest month {ns[1]['month']}: {ns[1]['net_sales']:,}"])
    assert executive_summary(SchemaRoutedLLM({ExecutiveSummary: lambda p: good}), kpis, "2026-01-01", "2026-06-30").source == "gemini_verified"
    bad = good.model_copy(update={"headline": "Sales grew 18% because of a campaign."})
    r = executive_summary(SchemaRoutedLLM({ExecutiveSummary: lambda p: bad}), kpis, "2026-01-01", "2026-06-30")
    assert r.source == "deterministic_fallback" and any("causal" in i for i in r.issues) and any("18" in i for i in r.issues)
    assert executive_summary(SchemaRoutedLLM({}), kpis, "2026-01-01", "2026-06-30").source == "deterministic_fallback"


def test_drill_down_with_gemini_like_model(runner):
    standin = OfflineStandInModel()
    plan = DrillDownPlan(questions=["Net sales by month for May 2026 and June 2026", "What is our order cancellation rate?"])

    def synth(prompt):
        import json, re
        results = json.loads(prompt.rsplit("<results>", 1)[1].split("</results>")[0])
        cancel = results[1]["rows"][0]["cancellation_rate"]
        return DrillDownFindings(summary=f"The cancellation rate was {cancel}.", findings=[f"Cancellation rate: {cancel}"],
                                 limitations="Causes outside the warehouse cannot be observed.")
    llm = SchemaRoutedLLM({**standin.handlers, DrillDownPlan: lambda p: plan, DrillDownFindings: synth})
    llm.generate = lambda **kw: (standin.generate(**kw) if kw["schema"] not in (DrillDownPlan, DrillDownFindings)
                                 else SchemaRoutedLLM.generate(llm, **kw))
    d = drill_down("Why did sales change between May and June 2026?", deps(runner, llm))
    assert d.plan == plan.questions and d.sub_results[0].period == "2026-05-01..2026-06-30"
    assert d.summary.source == "gemini_verified" and "cancellation rate" in d.summary.text


def test_router_falls_back_on_quota(runner):
    class Quota:
        model = "gemini-x"
        def generate(self, **kw):
            raise LLMError("quota", "daily limit")
    router = ModelRouter(Quota(), OfflineStandInModel(), "Gemini x")
    r = run_pipeline("What is our order cancellation rate?", [], deps(runner, router))
    assert r.status == "answered" and router.fallen_back and router.counts["gemini"] == 0 and "NOT Gemini" in router.label()


def test_multi_month_periods():
    assert (resolve_period("Net sales for May 2026 and June 2026").start_iso, resolve_period("Net sales for May 2026 and June 2026").end_iso) == ("2026-05-01", "2026-06-30")
    p = resolve_period("Net sales in December 2025 and January 2026")
    assert (p.start_iso, p.end_iso) == ("2025-12-01", "2026-01-31")
    assert resolve_period("May is a month") is None


def test_standin_honours_region_filter_and_explicit_dimension(runner):
    """Regression: region-specific drill-down questions must not return company-wide numbers."""
    llm = OfflineStandInModel()
    total = run_pipeline("Net sales by month for May 2026 and June 2026", [], deps(runner, llm))
    sw = run_pipeline("Net sales by month in the Southwest region for May 2026 and June 2026", [], deps(runner, llm))
    assert sw.status == "answered" and "'Southwest'" in sw.sql
    assert all(a["net_sales"] < b["net_sales"] for a, b in zip(sw.rows, total.rows))          # a subset of the total
    cat = run_pipeline("Net sales by product category in the Southwest region for June 2026", [], deps(runner, llm))
    assert "category" in cat.rows[0] and "'Southwest'" in cat.sql and "region" not in cat.rows[0]


def test_executive_summary_fallback_describes_metric_column(runner):
    kpis = collect_kpis(runner, S, DS["clean"], "2026-01-01", "2026-06-30")
    text = executive_summary(SchemaRoutedLLM({}), kpis, "2026-01-01", "2026-06-30").text
    aov_line = next(l for l in text.splitlines() if "AOV" in l or "verage order value" in l)
    assert "average_order_value" in aov_line and "orders =" not in aov_line
