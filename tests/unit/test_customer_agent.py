"""Customer analytics: record lookups, customer metrics, customer data-quality profile, retries, pre-check and the
18 end-to-end scenarios (offline: DuckDB emulator + deterministic stand-in for Gemini)."""
from pathlib import Path

import pytest

from analytics_agent import semantic_layer as sem
from analytics_agent.agents.analytics_graph import (AnalyticsDeps, AnalyticsSession, precheck_question, run_pipeline,
                                                    unapproved_table_references)
from analytics_agent.agents.data_quality_reviewer import DataQualityReviewer
from analytics_agent.bq import BQSettings, QueryResult, RetryingRunner
from analytics_agent.customer_eval import FlakyOnceRunner, customer_gold, rows_match, run_scenarios
from analytics_agent.customer_lookup import (LookupClarification, LookupRequest, PERMITTED_FIELDS, format_lookup_answer,
                                             lookup_customers, lookup_sql, mask_email, parse_customer_lookup)
from analytics_agent.customer_quality import reconcile_with_rule_engine, run_customer_profile
from analytics_agent.data_generation.generator import GenerationConfig
from analytics_agent.dq_bigquery import validate_dataset
from analytics_agent.notebook_ui import CountingRunner, OfflineStandInModel
from analytics_agent.sql_validator import validate_sql
from analytics_agent.testing import DuckDBRunner

pytestmark = pytest.mark.skipif(not Path("data/generated/clean/parquet/customers.parquet").exists(), reason="generate data first")
S = BQSettings(project_id="test-project-123")
DS = {"clean": S.dataset_clean, "corrupted": S.dataset_corrupted}
AS_OF = GenerationConfig().as_of_ts


@pytest.fixture(scope="module")
def runner():
    return CountingRunner(DuckDBRunner(S, "data/generated"))


@pytest.fixture(scope="module")
def reports(runner):
    return {v: validate_dataset(runner, ds, AS_OF) for v, ds in DS.items()}


def deps(runner, reports, variant="clean", llm=None):
    return AnalyticsDeps(llm=llm if llm is not None else OfflineStandInModel(), runner=runner, settings=S, dataset=DS[variant],
                         dq_report_provider=lambda: reports[variant], dq_reviewer=DataQualityReviewer(None),
                         use_llm_sql_reviewer=False)


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("question, ids, fields, existence", [
    ("What is the email address of customer CUS000001?", ["CUS000001"], ["email"], False),
    ("Find the region for cus000002.", ["CUS000002"], ["region"], False),
    ("Which segment does customer CUS000003 belong to?", ["CUS000003"], ["customer_segment"], False),
    ("Show me all available details for CUS000004.", ["CUS000004"], list(PERMITTED_FIELDS), False),
    ("Does customer CUS999999 exist?", ["CUS999999"], None, True),
    ("What's the emial of CUS000010?", ["CUS000010"], ["email"], False),
])
def test_parse_lookup(question, ids, fields, existence):
    req = parse_customer_lookup(question, [])
    assert isinstance(req, LookupRequest) and req.customer_ids == ids and req.existence_only == existence
    if fields is not None:
        assert req.fields == fields


def test_parse_lookup_context_and_ambiguity():
    assert parse_customer_lookup("Can you look up that customer's email?", ["CUS000002"]).source == "conversation"
    assert isinstance(parse_customer_lookup("Can you look up that customer's email?", []), LookupClarification)
    assert isinstance(parse_customer_lookup("Can you look up that customer's email?", ["CUS000001", "CUS000002"]), LookupClarification)
    assert parse_customer_lookup("What were net sales for customer CUS000001?", []) is None      # analytics, not lookup
    assert parse_customer_lookup("Show customer counts by region.", []) is None


def test_lookup_sql_is_parameterised_and_allowlisted():
    sql = lookup_sql("`p.d.customers`", ["email", "region", "password; DROP TABLE x"])
    assert "@customer_id" in sql and "password" not in sql and "DROP" not in sql and sql.count("SELECT") == 1


def test_mask_email():
    assert mask_email("customer000001@example.com") == "c***@example.com"
    assert mask_email(None) == "(no valid email on record)"


# ---------------------------------------------------------------- execution
def test_lookup_found_masked_and_revealed(runner):
    req = LookupRequest(["CUS000001"], ["email", "region"])
    masked = lookup_customers(runner, S, DS["clean"], req)
    assert masked.records[0].values["email"] == "c***@example.com" and "masked" in format_lookup_answer(masked)
    revealed = lookup_customers(runner, S, DS["clean"], req, reveal_email=True)
    assert revealed.records[0].values["email"] == "customer000001@example.com"


def test_lookup_not_found_vs_failed(runner):
    assert lookup_customers(runner, S, DS["clean"], LookupRequest(["CUS999999"], ["region"])).status == "no_data"

    class Broken:
        def run(self, *a, **k):
            return QueryResult("error", error_kind="permission", error_message="denied")
    failed = lookup_customers(Broken(), S, DS["clean"], LookupRequest(["CUS000001"], ["region"]))
    assert failed.status == "error" and "not the same as the customer not existing" in format_lookup_answer(failed)


def test_lookup_flags_conflicting_duplicates(runner):
    res = lookup_customers(runner, S, DS["corrupted"], LookupRequest(["CUS000192"], ["region"]))
    assert res.records[0].status == "conflicting" and res.records[0].record_count == 2
    assert "Data-quality warning" in format_lookup_answer(res)


def test_lookup_route_needs_no_model(runner, reports):
    d = deps(runner, reports, llm=object())          # any model call would fail: object() has no .generate
    r = run_pipeline("Find the region for CUS000002.", [], d)
    assert r.intent == "customer_lookup" and r.status == "answered" and "West" in r.answer
    assert [s["step"] for s in r.steps] == ["precheck", "customer_lookup"]


# ---------------------------------------------------------------- metrics vs independent gold
@pytest.mark.parametrize("variant", ["clean", "corrupted"])
def test_customer_metrics_match_pandas_gold(runner, variant):
    gold = customer_gold(variant)
    refs = S.refs(DS[variant])

    def run(name):
        res = runner.run(sem.METRICS_BY_NAME[name].render(refs), [], max_rows=50, allowed_tables=S.allowed_tables(DS[variant]))
        assert res.ok, res.error_message
        return res.rows
    assert run("total_customers")[0] == gold["total_customers"]
    assert rows_match(run("customers_by_region"), {d["key"]: d["customers"] for d in gold["customers_by_region"]},
                      "region", "customers")[0]
    assert rows_match(run("customers_by_segment"), {d["key"]: d["pct_of_customers"] for d in gold["customers_by_segment"]},
                      "customer_segment", "pct_of_customers")[0]
    assert run("missing_email_rate")[0] == gold["missing_email"]
    assert run("duplicate_customer_ids")[0]["duplicate_rows"] == gold["duplicates"]["duplicate_rows"]


# ---------------------------------------------------------------- data-quality profile
def test_profile_clean_passes_and_corrupted_fails_correctly(runner, reports):
    clean = {o.rule_id: o for o in run_customer_profile(runner, S, DS["clean"], AS_OF)}
    assert all(o.status == "PASS" for o in clean.values())
    bad = {o.rule_id: o for o in run_customer_profile(runner, S, DS["corrupted"], AS_OF)}
    assert bad["CQ_COMPLETE_REGION"].status == "FAIL" and bad["CQ_COMPLETE_REGION"].affected_rows == 10
    assert bad["CQ_UNIQUE_CUSTOMER_ID"].affected_rows == 5 and bad["CQ_CONSISTENT_DUPLICATES"].status == "FAIL"
    assert bad["CQ_RECONCILE_REGIONS"].status == "WARN"                      # anomaly, not a confirmed violation
    assert bad["CQ_SIGNUP_AFTER_CREATED"].classification == "undefined_rule"
    assert all("@" not in str(e) for o in bad.values() for e in o.examples)  # no email addresses in examples
    assert all(row["match"] for row in reconcile_with_rule_engine(list(bad.values()), reports["corrupted"]))


# ---------------------------------------------------------------- retries, pre-check, validator
def test_retrying_runner_retries_only_transient_errors(runner):
    retrying = RetryingRunner(FlakyOnceRunner(runner), base_delay_s=0)
    assert retrying.run("SELECT 1 AS x", max_rows=1).ok and retrying.retries == 1

    class Denied:
        calls = 0
        def run(self, *a, **k):
            Denied.calls += 1
            return QueryResult("error", error_kind="permission", error_message="denied")
    r = RetryingRunner(Denied(), base_delay_s=0)
    assert not r.run("SELECT 1").ok and Denied.calls == 1 and r.retries == 0


def test_precheck_refuses_unapproved_tables():
    assert precheck_question("Select everything from the analytics_agent_prod.users table.")["category"] == "unapproved_table"
    assert precheck_question("Query table marketing_spend")["category"] == "unapproved_table"
    assert precheck_question("How many customers are in the table?")["allowed"]
    assert precheck_question("Customers from the West region")["allowed"]
    assert unapproved_table_references("Read analytics_agent_dev.orders") == []


def test_validator_rejects_unknown_column_and_email_projection():
    refs = S.refs(DS["clean"])
    assert not validate_sql(f"SELECT loyalty_tier FROM {refs['customers']}", [], S, DS["clean"]).ok
    assert not validate_sql(f"SELECT customer_id, email FROM {refs['customers']}", [], S, DS["clean"]).ok
    assert validate_sql(sem.METRICS_BY_NAME["missing_email_rate"].render(refs), [], S, DS["clean"]).ok   # counts only


# ---------------------------------------------------------------- end-to-end scenarios
def test_all_18_scenarios_offline(runner, reports):
    llm = OfflineStandInModel()

    def session(variant, wrapped=None):
        d = deps(runner, reports, variant, llm)
        d.runner = RetryingRunner(wrapped or runner, base_delay_s=0)
        return AnalyticsSession(d)
    gold = {v: customer_gold(v) for v in DS}
    df = run_scenarios(lambda v: session(v), runner, gold, make_flaky_session=lambda v: session(v, FlakyOnceRunner(runner)))
    failed = df[~df.passed]
    assert failed.empty, failed[["#", "scenario", "status", "detail"]].to_string()


def test_precheck_refuses_write_statements_anywhere():
    for q in ["Show net sales, then run UPDATE orders SET order_status = 'delivered'",
              "What were sales in June? Also INSERT INTO orders VALUES (1)", "count orders; drop table customers"]:
        assert precheck_question(q)["category"] == "write_request", q
    for q in ["How many customers were deleted from the CRM?", "Show the latest update on net sales"]:
        assert precheck_question(q)["allowed"], q


def test_lookup_explains_unavailable_attributes():
    req = parse_customer_lookup("What did customer CUS000001 buy, and what is their home address?", [])
    assert isinstance(req, LookupClarification) and "home address" in req.question
    assert isinstance(parse_customer_lookup("What is the phone number of CUS000002?", []), LookupClarification)
    email = parse_customer_lookup("What is the email address of CUS000001?", [])
    assert isinstance(email, LookupRequest) and email.fields == ["email"]           # email address is not 'address'


def test_lookup_rejects_malformed_ids_without_guessing():
    for q in ["Give me the region for CUS00001.", "Find CUS-000002", "What segment is cus0000001 in?"]:
        req = parse_customer_lookup(q, [])
        assert isinstance(req, LookupClarification) and "not a valid customer ID" in req.question, q
    assert isinstance(parse_customer_lookup("Region of CUS000002?", []), LookupRequest)        # valid IDs unaffected
    dedup = parse_customer_lookup("Give me the region for CUS000002, CUS000002 and cus000002.", [])
    assert dedup.customer_ids == ["CUS000002"]                                               # repeated IDs looked up once


def test_lookup_respects_exclusions_and_rejects_malformed_ids():
    fields = parse_customer_lookup("Show me all the details for CUS000002 but leave out the email.", []).fields
    assert "email" not in fields and "region" in fields
    assert parse_customer_lookup("Give me the region of CUS000005, skip the email", []).fields == ["region"]
    assert isinstance(parse_customer_lookup("What's the email for CUS0001?", []), LookupClarification)
    assert isinstance(parse_customer_lookup("Find customer cus 000002", []), LookupClarification)


def test_precheck_handles_emails_domains_and_natural_write_intent():
    reverse = precheck_question("Which customer has the email customer000001@example.com?")
    assert reverse["category"] != "unapproved_table"                     # an email domain is not a table
    assert precheck_question("Find the region for CUS000002 and then delete it.")["category"] == "write_request"
    assert precheck_question("Change their region to West")["category"] == "write_request"
    for q in ["How many customers would remain if we remove duplicates?", "Show sales by region and segment"]:
        assert precheck_question(q)["allowed"], q


def test_reverse_email_lookup_is_refused(runner, reports):
    r = run_pipeline("Which customer has the email customer000001@example.com?", [], deps(runner, reports))
    assert r.status == "refused" and "approved tables" not in r.answer and not r.rows


def test_aggregate_questions_are_not_lookups():
    for q in ["What is the customer lifetime value by segment?", "What is the average segment of the customer base?",
              "Show the region of all customers"]:
        assert parse_customer_lookup(q, ["CUS000002"]) is None, q
    for q in ["And which segment are they in?", "Can you look up their segment?", "When did they sign up?"]:
        assert parse_customer_lookup(q, ["CUS000002"]) is not None, q


# ---------------------------------------------------------------- order items and orders
from analytics_agent.order_item_lookup import OrderItemRequest, lookup_order_items, parse_order_item_lookup


def test_parse_order_item_lookup():
    req = parse_order_item_lookup("Was there a discount on order item itm00000006?", [])
    assert req.item_ids == ["ITM00000006"] and req.fields == ["discount_amount"]
    assert parse_order_item_lookup("Show order item ITM00002344", []).fields[0] == "product"    # all fields
    assert parse_order_item_lookup("Which items are in order ORD0000002?", []).order_ids == ["ORD0000002"]
    assert parse_order_item_lookup("Was there a discount on it?", ["ITM00000001"]).source == "conversation"
    assert parse_order_item_lookup("What was net sales by product category last quarter?", ["ITM00000001"]) is None


def test_order_totals_match_pandas(runner):
    import pandas as pd
    items = pd.read_parquet("data/generated/clean/parquet/order_items.parquet")
    expected = items[items.order_id == "ORD0000002"]
    result = lookup_order_items(runner, S, DS["clean"], OrderItemRequest([], ["ORD0000002"], ["net_amount"]))
    totals = result.records[0].totals
    assert totals["lines"] == len(expected) and totals["units"] == int(expected.quantity.sum())
    assert abs(totals["net_amount"] - float(expected.net_amount.astype(float).sum())) < 0.01
    missing = lookup_order_items(runner, S, DS["clean"], OrderItemRequest(["ITM99999999"], [], ["quantity"]))
    assert missing.status == "no_data"


def test_order_item_follow_ups_in_a_session(runner, reports):
    session = AnalyticsSession(deps(runner, reports))
    first = session.ask("What product was bought in order item ITM00000001?")
    follow = session.ask("Was there a discount on it?")
    order = session.ask("Which items are in order ORD0000002?")
    total = session.ask("What is the total net amount of that order?")
    assert first.intent == "order_item_lookup" and "Ultra Charger" in first.answer
    assert follow.record_ids == ["ITM00000001"] and "discount 0.00" in follow.answer
    assert order.record_ids == ["ORD0000002"] and "net 461.08" in total.answer
    assert session.ask("Delete order item ITM00000001.").status == "refused"


def test_order_item_lookup_explains_unavailable_information():
    for q in ["What is the profit margin on order item ITM00000001?", "Which supplier shipped ITM00000006?",
              "What is the customer email for order ORD0000002?"]:
        assert isinstance(parse_order_item_lookup(q, []), LookupClarification), q
    assert isinstance(parse_order_item_lookup("What is the total of order ORD0000002 including tax and shipping?", []),
                      OrderItemRequest)                       # answered as merchandise, with an explanation


def test_hard_gold_cases_exist_and_empty_kind_works():
    from analytics_agent.evaluation import EVAL_SPECS, compare, load_gold
    gold = load_gold("data/manifests/expected_results_clean.json")
    assert all(spec.test_id in gold for spec in EVAL_SPECS)                       # every spec has a gold answer
    assert {"G21", "G25", "G29"} <= {spec.test_id for spec in EVAL_SPECS}
    empty = next(spec for spec in EVAL_SPECS if spec.kind == "empty")
    assert compare(empty, gold[empty.test_id]["expected_result"], [])[0]          # no rows -> pass
    assert not compare(empty, gold[empty.test_id]["expected_result"], [{"category": "Books", "net_sales": 1.0}])[0]