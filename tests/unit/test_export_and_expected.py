import json
from dataclasses import replace
from decimal import Decimal

import pandas as pd
import pyarrow.parquet as pq
import pytest

from analytics_agent import schema_registry as sr
from analytics_agent.data_generation.expected_results import build_expected_results
from analytics_agent.data_generation.export import (arrow_schema, write_bigquery_schemas, write_variant)
from analytics_agent.data_generation.generator import GenerationConfig, generate_clean


@pytest.fixture(scope="module")
def small_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen")
    t = generate_clean(replace(GenerationConfig(), n_customers=300, n_products=60, n_orders=1500))
    files = write_variant(t, out, "clean")
    return t, out, files


def test_parquet_types_and_rows(small_export):
    t, out, _ = small_export
    for name in sr.TABLE_LOAD_ORDER:
        tbl = pq.read_table(out / "clean" / "parquet" / f"{name}.parquet")
        assert tbl.num_rows == len(t[name])
        expected = arrow_schema(name)
        for f in expected:
            assert tbl.schema.field(f.name).type == f.type, (name, f.name)


def test_parquet_money_exact(small_export):
    t, out, _ = small_export
    tbl = pq.read_table(out / "clean" / "parquet" / "orders.parquet")
    first = tbl.column("order_total")[0].as_py()
    assert isinstance(first, Decimal)
    assert first == Decimal(f"{t['orders'].order_total.iloc[0]:.2f}")


def test_csv_round_trip(small_export):
    t, out, _ = small_export
    df = pd.read_csv(out / "clean" / "csv" / "orders.csv", dtype=str, keep_default_na=False)
    assert list(df.columns) == sr.ORDERS.column_names
    assert len(df) == len(t["orders"])
    assert df.order_timestamp.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC$").all()
    assert df.order_total.str.match(r"^-?\d+\.\d{2}$").all()
    cancelled = df[df.order_status == "cancelled"]
    assert (df.loc[df.order_status != "cancelled", "cancelled_timestamp"] == "").all()
    assert (cancelled.cancelled_timestamp != "").all()


def test_manifest_hashes_match_files(small_export):
    import hashlib
    _, _, files = small_export
    for f in files:
        assert hashlib.sha256(open(f["path"], "rb").read()).hexdigest() == f["sha256"]


def test_csv_files_reproducible(tmp_path):
    cfg = replace(GenerationConfig(), n_customers=300, n_products=60, n_orders=1500)
    a = write_variant(generate_clean(cfg), tmp_path / "a", "clean")
    b = write_variant(generate_clean(cfg), tmp_path / "b", "clean")
    ha = {(f["table"], f["format"]): f["sha256"] for f in a if f["format"] == "csv"}
    hb = {(f["table"], f["format"]): f["sha256"] for f in b if f["format"] == "csv"}
    assert ha == hb


def test_bigquery_schema_files(tmp_path):
    paths = write_bigquery_schemas(tmp_path)
    assert len(paths) == 7
    for p in paths:
        fields = json.load(open(p))
        assert all(f["type"] in {"STRING", "INTEGER", "NUMERIC", "BOOLEAN", "DATE", "TIMESTAMP"} for f in fields)


def test_expected_results_structure(clean):
    exp = build_expected_results(clean)
    assert len(exp["cases"]) >= 15
    keys = {"test_id", "question", "expected_business_definition", "expected_sql_properties",
            "expected_result", "tolerance", "expected_reviewer_behavior", "expected_safety_behavior"}
    for c in exp["cases"]:
        assert keys <= set(c), c["test_id"]
    assert len({c["test_id"] for c in exp["cases"]}) == len(exp["cases"])
    json.dumps(exp, default=str)


def test_expected_planted_answers(clean):
    cases = {c["test_id"]: c for c in build_expected_results(clean)["cases"]}
    assert cases["G06"]["expected_result"]["largest_decline_region"] == "Southwest"
    assert cases["G07"]["expected_result"]["bridge_difference"] == 0.0
    months = [r["month"] for r in cases["G01"]["expected_result"]]
    assert months == ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def test_known_wrong_answers_differ_from_correct(clean):
    for w in build_expected_results(clean)["known_wrong_answers"]:
        assert w["wrong_value"] != w["correct_value"], w["id"]


def test_corruption_changes_answers(clean, corrupted):
    c = {x["test_id"]: x for x in build_expected_results(clean)["cases"]}
    k = {x["test_id"]: x for x in build_expected_results(corrupted)["cases"]}
    assert c["G07"]["expected_result"]["captured_payments"] < k["G07"]["expected_result"]["captured_payments"]  # D11
    assert c["G02"]["expected_result"] != k["G02"]["expected_result"]                                           # D03/D10


def test_b19_documented_impacts_still_hold(clean, corrupted):
    """Guards the numbers quoted in evaluation_cases.B19."""
    from analytics_agent import metrics as m
    W = ("2025-07-01", "2026-06-30")
    total_c = m.net_sales(clean, *W).net_sales.iloc[0]
    total_x = m.net_sales(corrupted, *W).net_sales.iloc[0]
    assert round(total_x - total_c, 2) == -2539.30
    reg = m.net_sales(corrupted, *W, group_by=["region"])
    assert round(reg[reg.region.isna()].net_sales.iloc[0], 2) == 19296.81
    assert round(reg.net_sales.sum() - total_x, 2) == 13225.08       # D01 fan-out


def test_net_sales_paths_agree(clean, corrupted):
    """metrics.net_sales (no dimension) must equal the reconciliation's net sales on BOTH datasets.
    Regression test for an over-join bug: joining customers when no customer
    dimension is requested exposed totals to duplicate-key fan-out."""
    from analytics_agent import metrics as m
    W = ("2025-07-01", "2026-06-30")
    for t in (clean, corrupted):
        assert round(m.net_sales(t, *W).net_sales.iloc[0], 2) == m.reconciliation(t, *W)["net_sales"]
