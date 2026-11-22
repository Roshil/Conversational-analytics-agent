import pandas as pd

from analytics_agent.data_generation.corruption import expected_findings, legitimate_cases
from analytics_agent.data_generation.reference_checks import RULES, run_reference_checks


def test_clean_data_passes_every_rule(clean, cfg):
    results = run_reference_checks(clean, cfg.as_of_ts)
    assert {k: v for k, v in results.items() if v} == {}


def test_corrupted_findings_equal_catalog_exactly(corrupted, defects, cfg):
    got = run_reference_checks(corrupted, cfg.as_of_ts)
    exp = expected_findings(defects)
    for rule in RULES:
        assert got[rule.rule_id] == exp.get(rule.rule_id, set()), rule.rule_id


def test_every_defect_is_documented(defects):
    assert len(defects) >= 10
    ids = [d.defect_id for d in defects]
    assert len(ids) == len(set(ids))
    rule_ids = {r.rule_id for r in RULES}
    for d in defects:
        assert d.affected_ids and d.expected_findings
        assert d.detection_rule in d.expected_findings
        assert set(d.expected_findings) <= rule_ids
        assert d.severity in {"critical", "high", "medium", "low"}
        assert d.business_impact and d.expected_outcome


def test_required_defect_types_covered(defects):
    types = {d.defect_type for d in defects}
    for required in ["duplicate_primary_key", "missing_required_value", "invalid_category", "invalid_status",
                     "orphan_foreign_key", "invalid_timestamp_sequence", "negative_quantity",
                     "incorrect_monetary_total", "duplicate_transaction", "refund_exceeds_refundable",
                     "inconsistent_state", "duplicate_business_key", "duplicate_rows"]:
        assert required in types, required


def test_corruption_does_not_mutate_clean_input(clean):
    from analytics_agent.data_generation.corruption import corrupt
    before = {k: v.copy() for k, v in clean.items()}
    corrupt(clean, seed=777)
    for k in clean:
        pd.testing.assert_frame_equal(before[k], clean[k])


def test_corruption_is_deterministic(clean):
    from analytics_agent.data_generation.corruption import corrupt
    a, da = corrupt(clean, seed=777)
    b, db = corrupt(clean, seed=777)
    assert [d.to_dict() for d in da] == [d.to_dict() for d in db]
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_legitimate_cases_exist_and_are_not_flagged(clean, cfg):
    cases = {c["case"]: c["count"] for c in legitimate_cases(clean)}
    for must in ["pending_payment_orders", "failed_then_captured_payments", "partial_item_refunds",
                 "cancelled_orders_with_captured_payment_and_full_refund", "pending_refunds",
                 "in_transit_shipments"]:
        assert cases[must] > 0, must
    assert not any(run_reference_checks(clean, cfg.as_of_ts).values())


def test_corrupted_has_expected_extra_rows(clean, corrupted):
    assert len(corrupted["customers"]) == len(clean["customers"]) + 5      # D01
    assert len(corrupted["order_items"]) == len(clean["order_items"]) + 5 + 12  # D09 + D10
    assert len(corrupted["payments"]) == len(clean["payments"]) + 15       # D11
