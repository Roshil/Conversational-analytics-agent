"""Tests for the notebook 05 answer helpers (analytics_agent.notebook_answers)."""
from analytics_agent.agents.analytics_graph import AnalyticsResponse
from analytics_agent import notebook_answers as na


def response(**kwargs):
    base = {"question": "q", "answer": "", "status": "answered"}
    base.update(kwargs)
    return AnalyticsResponse(**base)


def test_error_is_never_reported_as_empty_result():
    r = response(status="error", answer="Gemini could not generate SQL (invalid_output). Nothing was executed.",
                 intent=None, rows=[])
    lines = na.final_answer(r)
    text = " ".join(lines)
    assert "could not be answered" in text
    assert "failure, not an empty result" in text
    assert "ran successfully" not in text


def test_no_data_metric_says_ran_successfully():
    r = response(status="no_data", intent="metric_question", metrics=["net_sales_by_dimension"],
                 period="2027-01-01..2027-12-31", rows=[])
    assert "ran successfully but returned no rows" in na.final_answer(r)[0]


def test_refusal_uses_precheck_category():
    r = response(status="refused", answer="I can't modify data.", steps=[{"step": "precheck", "category": "write_request"}])
    lines = na.final_answer(r)
    assert lines[0].startswith("No query was run")
    assert "read-only" in lines[1]


def test_percent_uses_column_name_not_value():
    assert na.percent(1.0, "late_rate") == "100.00%"          # a fraction column
    assert na.percent(1.0, "pct_of_customers") == "1.00%"     # already a percentage: no longer shown as 100%
    assert na.percent(0.1872, "late_rate") == "18.72%"
    assert na.percent(None, "late_rate") == "n/a"


def test_month_over_month_with_missing_change_does_not_crash():
    rows = [{"region": "West", "previous": None, "net_sales": 10.0, "mom_pct": None},
            {"region": "East", "previous": 100.0, "net_sales": 80.0, "mom_pct": -20.0}]
    lines = na.describe_month_over_month(rows, "June 2026")
    assert "East" in lines[0] and "-20.00%" in lines[0]
    assert any("No change could be calculated for West" in line for line in lines)


def test_month_over_month_all_missing():
    rows = [{"region": "West", "mom_pct": None}]
    assert "No month-over-month change" in na.describe_month_over_month(rows, "June 2026")[0]


def test_net_sales_share_with_zero_total_does_not_divide_by_zero():
    rows = [{"category": "A", "net_sales": 0.0}, {"category": "B", "net_sales": 0.0}]
    lines = na.describe_net_sales_by_dimension(rows, "June 2026")
    assert "A 0" in lines[0]


def test_net_sales_without_text_column_falls_back_to_rows():
    rows = [{"net_sales": 5.0}]
    assert na.describe_net_sales_by_dimension(rows, "p") == ["net_sales = 5.00"]


def test_customer_lookup_masks_note_and_not_found():
    r = response(intent="customer_lookup", customer_ids=["CUS000001", "CUS999999"],
                 rows=[{"customer_id": "CUS000001", "email": "c***@example.com", "lookup_status": "found",
                        "record_count": 1},
                       {"lookup_status": "not_found", "record_count": 0}])
    lines = na.final_answer(r, reveal_email=False)
    assert "the email address is c***@example.com" in lines[0]
    assert "CUS999999 does not exist" in lines[1]
    assert any("masked by policy" in line for line in lines)


def test_model_used_and_warning():
    assert na.model_used({"gemini": 1, "stand-in": 0}, {"gemini": 4, "stand-in": 0}) == na.GEMINI
    assert na.model_used({"gemini": 1, "stand-in": 0}, {"gemini": 1, "stand-in": 3}) == na.STAND_IN
    assert na.model_used({"gemini": 1, "stand-in": 0}, {"gemini": 2, "stand-in": 1}) == na.MIXED
    assert na.model_used({"gemini": 1, "stand-in": 0}, {"gemini": 1, "stand-in": 0}) == na.NO_MODEL
    records = [{"model": na.GEMINI}, {"model": na.STAND_IN}, {"model": na.NO_MODEL}]
    assert "1 of 3 answers" in na.model_warning(records, "test")
    assert na.model_warning([{"model": na.GEMINI}], "test") is None


def test_email_scan_covers_result_rows(tmp_path):
    r = response(answer="Found it.", rows=[{"customer_id": "CUS1", "email": "someone@example.com"}])
    checked, findings = na.scan_records_for_emails({"lookups": [{"question": "q", "response": r,
                                                                 "final_answer": ["ok"]}]})
    assert checked == 1
    assert findings and findings[0]["where"] == "result rows"
    report = tmp_path / "r.txt"
    report.write_text("email: someone@example.com")
    assert na.scan_files_for_emails([report])[0]["emails"] == 1


def test_save_report_states_models(tmp_path):
    r = response(status="answered", rows=[{"x": 1}])
    records = [{"question": "q", "dataset": "clean", "response": r, "model": na.STAND_IN, "final_answer": ["x = 1"]}]
    path = na.save_report(records, "t", "Test", report_dir=tmp_path, backend="duckdb", reveal_email=False)
    text = path.read_text()
    assert "Models    : stand-in: 1" in text
    assert "NOT Gemini" in text
