import json

import pandas as pd
import pytest

from analytics_agent.data_quality import (RULE_SPECS, detection_metrics, redact_identifier, validate_tables)


@pytest.fixture(scope="module")
def reports(clean, corrupted, cfg):
    return validate_tables(clean, cfg.as_of_ts, "clean"), validate_tables(corrupted, cfg.as_of_ts, "corrupted")


def test_rule_ids_unique_and_documented():
    ids = [s.rule_id for s in RULE_SPECS]
    assert len(ids) == len(set(ids)) == 50
    for s in RULE_SPECS:
        assert s.severity in {"low", "medium", "high", "critical"}
        assert s.recommended_action and s.business_impact and s.expected_condition


def test_clean_all_pass(reports):
    clean_rep, _ = reports
    assert clean_rep.counts() == {"pass": 50, "fail": 0, "warning": 0, "error": 0}
    assert clean_rep.overall_status == "pass"


def test_corrupted_detection_is_exact(reports):
    _, rep = reports
    assert rep.overall_status == "fail"
    assert rep.counts()["fail"] == 20 and rep.counts()["error"] == 0


def test_detection_metrics_against_catalog(reports, defects):
    _, rep = reports
    m = detection_metrics(rep, [d.to_dict() for d in defects])
    assert m.missed.sum() == 0 and m.false_pos.sum() == 0
    assert m.true_pos.sum() == m.expected.sum() == 154


def test_report_fields_and_serialization(reports, tmp_path):
    _, rep = reports
    d = rep.to_dict()
    assert {"run_id", "run_timestamp", "dataset_label", "overall_status", "counts", "results"} <= set(d)
    r = rep.get("PAY_TXN_UNIQUE")
    assert r.status == "fail" and r.observed_count == 15 and r.severity == "critical"
    assert r.example_ids and r.recommended_action
    j, md = rep.save(tmp_path)
    assert json.loads(j.read_text())["counts"]["fail"] == 20
    assert "PAY_TXN_UNIQUE" in md.read_text()


def test_emails_are_redacted(reports):
    _, rep = reports
    text = json.dumps(rep.to_dict())
    assert "@example.com" not in text
    assert redact_identifier("customer000001@example.com").startswith("email#")
    assert redact_identifier("ORD0000001") == "ORD0000001"


def test_warning_is_not_fail(clean, cfg):
    rep = validate_tables(clean, cfg.as_of_ts + pd.Timedelta(days=30), "stale")
    assert rep.get("FRESH_ORDERS").status == "warning"
    assert rep.counts()["fail"] == 0 and rep.overall_status == "warning"


def test_errors_are_not_passes(clean, cfg):
    broken = dict(clean)
    broken["shipments"] = clean["shipments"].drop(columns=["delivered_timestamp"])
    rep = validate_tables(broken, cfg.as_of_ts, "broken")
    assert rep.get("SCHEMA_COLUMNS_SHIPMENTS").status == "fail"
    assert rep.get("SHP_DELIVERY_AFTER_SHIP").status == "error"
    assert rep.overall_status == "error"
    missing = dict(clean); del missing["refunds"]
    rep2 = validate_tables(missing, cfg.as_of_ts, "missing")
    assert all(r.status == "error" for r in rep2.results if r.table == "refunds")
