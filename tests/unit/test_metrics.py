"""Metric logic tests on a tiny hand-built dataset whose answers were computed by hand,
plus identity checks on the full clean dataset."""
import math

import pandas as pd
import pytest

from analytics_agent import metrics as m
from analytics_agent.data_generation.tiny_dataset import build_tiny_dataset


@pytest.fixture()
def tiny():
    return build_tiny_dataset()


W = ("2026-01-01", "2026-02-28")


def test_tiny_net_sales_by_month(tiny):
    df = m.net_sales(tiny, *W, group_by=["month"]).set_index("month")
    # Jan: merch 20+18+20 = 58, completed item refund 5 -> 53. Feb: only O5 (20).
    assert df.loc["2026-01", "net_sales"] == 53.00
    assert df.loc["2026-02", "net_sales"] == 20.00
    total = m.net_sales(tiny, *W)
    assert total.net_sales.iloc[0] == 73.00     # pending refund R2 and cancelled/pending orders excluded


def test_tiny_aov(tiny):
    assert m.average_order_value(tiny, *W).aov.iloc[0] == 26.00          # 78 / 3 booked orders
    by = m.average_order_value(tiny, *W, group_by=["region"]).set_index("region").aov
    assert by["West"] == 38.00 and by["Midwest"] == 20.00


def test_tiny_refund_and_delivery_rates(tiny):
    r = m.order_refund_rate(tiny, *W)
    assert (r["booked_orders"], r["refunded_orders"]) == (3, 1)
    d = m.delivery_performance(tiny, *W).iloc[0]
    assert (d.delivered, d.late, d.late_rate) == (2, 1, 0.5)            # in-transit excluded


def test_tiny_reconciliation(tiny):
    rec = m.reconciliation(tiny, *W)
    assert rec["captured_payments"] == 115.31
    assert rec["merch_sales_booked"] == 78.00
    assert rec["shipping_booked"] == 20.97
    assert rec["tax_booked"] == 5.64
    assert rec["captured_on_cancelled_orders"] == 10.70
    assert rec["bridge_difference"] == 0.0
    assert rec["net_sales"] == 73.00
    assert rec["net_cash"] == round(115.31 - 5.00 - 10.70, 2)
    assert rec["pending_refunds_excluded"] == 20.00


def test_tiny_payment_failure_rate(tiny):
    assert m.payment_failure_rate(tiny, *W)["failure_rate"] == round(1 / 6, 4)


def test_mom_first_month_null_and_zero_previous():
    df = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "v": [0.0, 10.0, 15.0]})
    out = m.mom_change(df, "v")
    assert math.isnan(out.mom_pct.iloc[0]) and math.isnan(out.mom_pct.iloc[1])
    assert out.mom_pct.iloc[2] == 50.0


def test_full_reconciliation_bridge_is_zero(clean):
    rec = m.reconciliation(clean, "2025-07-01", "2026-06-30")
    assert rec["bridge_difference"] == 0.0
    # every paid-then-cancelled order is refunded in full (some refunds may still be pending)
    assert rec["captured_on_cancelled_orders"] >= rec["completed_cancellation_refunds"]


def test_dimension_breakdowns_sum_to_total(clean):
    W = ("2025-07-01", "2026-06-30")
    total = m.net_sales(clean, *W).net_sales.iloc[0]
    for dim in ["month", "category", "region", "channel", "segment"]:
        assert abs(m.net_sales(clean, *W, group_by=[dim]).net_sales.sum() - total) < 0.05, dim


def test_fanout_pitfall_inflates(clean):
    booked = m.booked_orders(clean, "2025-07-01", "2026-06-30")
    wrong = clean["order_items"].merge(booked[["order_id", "order_total"]], on="order_id").order_total.sum()
    assert wrong > booked.order_total.sum() * 2
