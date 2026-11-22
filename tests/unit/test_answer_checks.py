"""The notebook's independent answer checks: correct on the stand-in, and tolerant of other correct layouts."""
from types import SimpleNamespace

import pytest

from analytics_agent.agents.analytics_graph import AnalyticsDeps, run_pipeline
from analytics_agent.agents.data_quality_reviewer import DataQualityReviewer
from analytics_agent.answer_checks import AnswerChecks, label_column, same_numbers
from analytics_agent.bq import BQSettings
from analytics_agent.customer_eval import customer_gold
from analytics_agent.notebook_ui import OfflineStandInModel
from analytics_agent.testing import DuckDBRunner

S = BQSettings(project_id="offline-emulator")
DATA = "data/generated"
Q2, JUNE, H1 = ("2026-04-01", "2026-06-30"), ("2026-06-01", "2026-06-30"), ("2026-01-01", "2026-06-30")


@pytest.fixture(scope="module")
def checks():
    return AnswerChecks(DATA, {v: customer_gold(v, data_dir=DATA) for v in ("clean", "corrupted")})


@pytest.fixture(scope="module")
def ask():
    runner, llm = DuckDBRunner(S, DATA), OfflineStandInModel()
    datasets = {"clean": S.dataset_clean, "corrupted": S.dataset_corrupted}

    def run(question, variant="clean"):
        deps = AnalyticsDeps(llm=llm, runner=runner, settings=S, dataset=datasets[variant],
                             use_llm_sql_reviewer=False, dq_reviewer=DataQualityReviewer(None))
        return run_pipeline(question, [], deps)
    return run


def answer(rows, status="answered"):
    return SimpleNamespace(status=status, rows=rows)


@pytest.mark.parametrize("question, variant, make_check", [
    ("How many customers are in the table?", "clean", lambda c: c.total_customers("clean")),
    ("How many customers do we actually have, not just rows?", "corrupted",
     lambda c: c.total_customers("corrupted", rows_too=True)),
    ("Show customer counts by region.", "clean", lambda c: c.customers_by_region("clean")),
    ("Show customer counts by region. Where do customers with no region go?", "corrupted",
     lambda c: c.customers_by_region("corrupted")),
    ("What percentage of customers are in each segment?", "clean", lambda c: c.segment_shares("clean")),
    ("Which region had the most signups during the last three months?", "clean",
     lambda c: c.top_signup_region("clean", *Q2)),
    ("How many duplicate customer IDs are there, and how many extra rows do they add?", "corrupted",
     lambda c: c.duplicates("corrupted")),
    ("Which product category had the highest net sales in June 2026?", "clean",
     lambda c: c.top_net_sales("category", *JUNE)),
    ("What was our monthly net sales over the last six months?", "clean",
     lambda c: c.net_sales_by_dimension("month", *H1)),
    ("What was the average order value last quarter? Don't count tax or shipping.", "clean",
     lambda c: c.average_order_value(*Q2)),
    ("What was the average order value by region last quarter?", "clean", lambda c: c.average_order_value(*Q2)),
    ("What were net sales by sales channel in the first half of 2026?", "clean",
     lambda c: c.net_sales_by_dimension("sales_channel", *H1)),
    ("What were net sales by product category last quarter?", "clean",
     lambda c: c.net_sales_by_dimension("category", *Q2)),
    ("How did net sales change month over month in June 2026?", "clean", lambda c: c.mom_change("2026-06")),
    ("Which region had the biggest month-over-month drop in net sales in June 2026?", "clean",
     lambda c: c.biggest_mom_drop("2026-06")),
    ("What is the total net amount of order ORD0000002?", "clean", lambda c: c.order_net_total("ORD0000002")),
    ("What were net sales by product category in 2027?", "clean", lambda c: c.no_data()),
])
def test_stand_in_answers_pass_independent_checks(checks, ask, question, variant, make_check):
    passed, detail = make_check(checks)(ask(question, variant))
    assert passed, detail


def test_checks_accept_other_correct_layouts(checks, ask):
    lines = ask("What is the total net amount of order ORD0000002?").rows
    total = round(sum(float(r["net_amount"]) for r in lines), 2)
    assert checks.order_net_total("ORD0000002")(answer([{"total_net_amount": total}]))[0]

    ranking = ask("Which region had the most signups during the last three months?").rows
    top_only = [max(ranking, key=lambda r: r["signups"])]
    assert checks.top_signup_region("clean", *Q2)(answer(top_only))[0]

    category = ask("Which product category had the highest net sales in June 2026?").rows
    best = max(category, key=lambda r: r["net_sales"])
    assert checks.top_net_sales("category", *JUNE)(answer([{"product_category": best["category"],
                                                            "revenue": best["net_sales"]}]))[0]


def test_checks_reject_wrong_answers(checks, ask):
    rows = ask("What was the average order value last quarter? Don't count tax or shipping.").rows
    assert not checks.average_order_value(*Q2)(answer([{"average_order_value": rows[0]["average_order_value"] + 5}]))[0]
    assert not checks.order_net_total("ORD0000002")(answer([{"total_net_amount": 1.0}]))[0]
    assert not checks.total_customers("clean")(answer([], status="error"))[0]
    assert not checks.no_data()(answer([{"net_sales": 0}]))[0]


def test_helpers():
    assert label_column([{"net_sales": 1.0, "note": "x", "region": "West"}]) == "region"
    assert same_numbers({"a": 1.0}, {"a": 1.004})[0]
    assert not same_numbers({"a": 1.0, "b": 2.0}, {"a": 1.0})[0]          # an extra group is a wrong grouping


def test_follow_up_breakdown_uses_the_new_dimension_and_labels_the_stand_in():
    from analytics_agent.agents.analytics_graph import AnalyticsSession
    deps = AnalyticsDeps(llm=OfflineStandInModel(), runner=DuckDBRunner(S, DATA), settings=S, dataset=S.dataset_clean,
                         use_llm_sql_reviewer=False, dq_reviewer=DataQualityReviewer(None))
    session = AnalyticsSession(deps)
    first = session.ask("What were net sales by sales channel in the first half of 2026?")
    follow_up = session.ask("Now break that down by product category.")
    assert "sales_channel" in first.rows[0]
    assert "category" in follow_up.rows[0] and "sales_channel" not in follow_up.rows[0]
    assert follow_up.period == first.period                              # the period carries over
    assert follow_up.answer_source == "stand_in_verified"                # never reported as Gemini
