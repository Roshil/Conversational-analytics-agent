from dataclasses import replace

import pandas as pd
import pytest

from analytics_agent import schema_registry as sr
from analytics_agent.data_generation.generator import GenerationConfig, generate_clean

EXPECTED_ROW_RANGES = {
    "customers": (2000, 2000), "products": (250, 250), "orders": (15000, 15000),
    "order_items": (30000, 36000), "payments": (15000, 17000),
    "shipments": (13500, 15000), "refunds": (1500, 2600),
}


def test_row_counts_in_expected_ranges(clean):
    for name, (lo, hi) in EXPECTED_ROW_RANGES.items():
        assert lo <= len(clean[name]) <= hi, (name, len(clean[name]))


def test_columns_match_schema_order(clean):
    for name, df in clean.items():
        assert list(df.columns) == sr.get_table(name).column_names


def test_required_columns_not_null(clean):
    for name, df in clean.items():
        req = sr.get_table(name).required_columns
        assert not df[req].isna().any().any(), name


def test_primary_keys_unique(clean):
    for name, df in clean.items():
        pk = list(sr.get_table(name).primary_key)
        assert not df.duplicated(pk).any(), name


def test_deterministic_with_same_seed():
    small = replace(GenerationConfig(), n_customers=300, n_products=60, n_orders=1500)
    a, b = generate_clean(small), generate_clean(small)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_different_seed_changes_data():
    small = replace(GenerationConfig(), n_customers=300, n_products=60, n_orders=1500)
    a = generate_clean(small)
    b = generate_clean(replace(small, seed=small.seed + 1))
    assert not a["orders"].equals(b["orders"])


def test_nothing_after_snapshot(clean, cfg):
    as_of = cfg.as_of_ts
    checks = [("orders", "order_timestamp"), ("payments", "payment_timestamp"),
              ("shipments", "shipped_timestamp"), ("shipments", "delivered_timestamp"),
              ("refunds", "refund_timestamp")]
    for table, col in checks:
        assert (clean[table][col].dropna() <= as_of).all(), (table, col)


def test_orders_within_window(clean, cfg):
    d = pd.to_datetime(clean["orders"].order_date)
    assert d.min() >= pd.Timestamp(cfg.start_date) and d.max() <= pd.Timestamp(cfg.end_date)


def test_signup_before_orders(clean):
    m = clean["orders"].merge(clean["customers"], on="customer_id")
    assert (pd.to_datetime(m.signup_date) <= pd.to_datetime(m.order_date)).all()


def test_order_date_is_utc_date_of_timestamp(clean):
    o = clean["orders"]
    assert (o.order_timestamp.dt.date == o.order_date).all()


def test_money_identities(clean):
    o, i = clean["orders"], clean["order_items"]
    assert ((i.quantity * i.unit_price - i.gross_amount).abs() < 0.005).all()
    assert ((i.gross_amount - i.discount_amount - i.net_amount).abs() < 0.005).all()
    calc = o.items_subtotal - o.discount_amount + o.shipping_amount + o.tax_amount
    assert ((calc - o.order_total).abs() < 0.005).all()


def test_shipping_rule(clean):
    o = clean["orders"]
    net = o.items_subtotal - o.discount_amount
    assert (o.loc[net >= 75, "shipping_amount"] == 0).all()
    assert (o.loc[net < 75, "shipping_amount"] == 6.99).all()


def test_status_lifecycle_consistency(clean):
    o = clean["orders"].set_index("order_id")
    s = clean["shipments"]
    assert set(s.order_id).isdisjoint(o.index[o.order_status.isin(["processing", "pending_payment", "cancelled"])])
    assert (o.loc[s[s.shipment_status == "delivered"].order_id, "order_status"] == "delivered").all()
    assert (o.loc[s[s.shipment_status == "in_transit"].order_id, "order_status"] == "shipped").all()
    assert len(o[o.order_status.isin(["shipped", "delivered"])]) == len(s)


def test_planted_patterns_exist(clean):
    o = clean["orders"]
    month = pd.to_datetime(o.order_date).dt.month
    assert (month == 12).sum() > (month == 2).sum() * 1.5          # Q4 seasonality
    s = clean["shipments"].dropna(subset=["delivered_timestamp"])
    late = s.delivered_timestamp.dt.date > s.promised_delivery_date
    by_carrier = late.groupby(s.carrier).mean()
    assert by_carrier["metroexpress"] > by_carrier["swiftship"]      # carrier difference
    assert 0.05 < late.mean() < 0.20


@pytest.mark.parametrize("case", ["pending_payment", "processing", "shipped", "cancelled"])
def test_legitimate_edge_statuses_present(clean, case):
    assert (clean["orders"].order_status == case).sum() > 0
