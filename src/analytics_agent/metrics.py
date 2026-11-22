"""Canonical business metric definitions (pandas reference implementation).

These functions are the *independent ground truth* for the gold evaluation set.
They implement the definitions in docs/02_dataset_specification.md section 6 and
deliberately mirror how a correct SQL query behaves (inner joins to booked
orders, left joins for dimensions, child tables aggregated before combining),
so that when run on the corrupted dataset they show what a *correct query on bad
data* returns.

Stage 6 adds the equivalent BigQuery SQL templates; tests will assert that SQL
and these functions agree on the clean dataset.

Conventions
-----------
* Money in USD, rounded to cents only at the end.
* Dates are UTC. Sales are attributed to the ORDER date (refunds too: a refund
  reduces net sales of the month in which its order was placed).
* "Booked" orders = order_status in (processing, shipped, delivered).
* Date filters are inclusive [start, end].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics_agent import schema_registry as sr

DIMENSIONS = {
    "month": "Calendar month of order_date (YYYY-MM)",
    "region": "customers.region of the ordering customer",
    "segment": "customers.customer_segment",
    "category": "products.category",
    "product_id": "products.product_id",
    "channel": "orders.sales_channel",
}


def _d(x) -> pd.Timestamp:
    return pd.Timestamp(x)


def _in_window(dates: pd.Series, start, end) -> pd.Series:
    d = pd.to_datetime(dates)
    return (d >= _d(start)) & (d <= _d(end))


def booked_orders(t: dict, start=None, end=None) -> pd.DataFrame:
    o = t["orders"]
    o = o[o.order_status.isin(sr.BOOKED_ORDER_STATUSES)]
    if start is not None:
        o = o[_in_window(o.order_date, start, end)]
    return o


CUSTOMER_DIMS = {"region", "segment"}
PRODUCT_DIMS = {"category", "product_id", "product_name"}


def _enriched_items(t: dict, start, end, dims=()) -> pd.DataFrame:
    """Order lines of booked orders in the window, with only the dimensions requested.

    Inner join items->orders (orphan lines drop out, as in SQL). Dimension tables
    are LEFT-joined *only when a requested dimension needs them*, exactly as a
    correct SQL query would: joining customers when no customer attribute is
    needed adds nothing but exposes the result to fan-out from duplicate
    customer keys. When a dimension IS requested, fan-out from duplicate parent
    keys is not hidden, because SQL would not hide it either.
    """
    dims = set(dims or ())
    o = booked_orders(t, start, end)[["order_id", "order_date", "sales_channel", "customer_id"]]
    i = t["order_items"].merge(o, on="order_id", how="inner")
    if dims & CUSTOMER_DIMS:
        i = i.merge(t["customers"][["customer_id", "region", "customer_segment"]], on="customer_id", how="left")
    if dims & PRODUCT_DIMS:
        i = i.merge(t["products"][["product_id", "category", "product_name"]], on="product_id", how="left")
    i["month"] = pd.to_datetime(i.order_date).dt.strftime("%Y-%m")
    return i.rename(columns={"sales_channel": "channel", "customer_segment": "segment"})


def _dim_cols(group_by):
    return list(group_by) if group_by else []


def net_sales(t: dict, start, end, group_by=None) -> pd.DataFrame:
    """Net sales = SUM(order_items.net_amount) of booked orders
                   - SUM(completed item-level refunds on those lines).

    Excludes tax, shipping, cancelled and pending-payment orders, and pending
    refunds. Grain: order item. Returns merch_sales, refunds, net_sales.
    """
    g = _dim_cols(group_by)
    items = _enriched_items(t, start, end, g)
    r = t["refunds"]
    r = r[(r.refund_status == "completed") & r.order_item_id.notna()][["order_item_id", "refund_amount"]]
    # Aggregate each child separately, then combine (avoids fan-out).
    sales = items.groupby(g, dropna=False).net_amount.sum() if g else pd.Series({"total": items.net_amount.sum()})
    ref = r.merge(items[["order_item_id"] + g].drop_duplicates() if g else items[["order_item_id"]].drop_duplicates(),
                  on="order_item_id", how="inner")
    refunds = ref.groupby(g, dropna=False).refund_amount.sum() if g else pd.Series({"total": ref.refund_amount.sum()})
    out = pd.DataFrame({"merch_sales": sales}).join(pd.DataFrame({"refunds": refunds}), how="left").fillna({"refunds": 0.0})
    out["net_sales"] = out.merch_sales - out.refunds
    return out.round(2).reset_index().rename(columns={"index": "scope"})


def average_order_value(t: dict, start, end, group_by=None) -> pd.DataFrame:
    """AOV = SUM(order_items.net_amount) / COUNT(DISTINCT order_id), booked orders.

    Merchandise value after discounts, before refunds, excluding tax and
    shipping. Grain: order (numerator summed from items, denominator distinct orders).
    """
    g = _dim_cols(group_by)
    items = _enriched_items(t, start, end, g)
    if g:
        agg = items.groupby(g, dropna=False).agg(merch_sales=("net_amount", "sum"), orders=("order_id", "nunique"))
    else:
        agg = pd.DataFrame({"merch_sales": [items.net_amount.sum()], "orders": [items.order_id.nunique()]})
    agg["aov"] = agg.merch_sales / agg.orders
    return agg.round(2).reset_index()


def product_refund_rates(t: dict, start, end, min_units: int = 30, top_n: int = 10) -> pd.DataFrame:
    """Value-based refund rate per product = completed item refunds / merchandise sales.

    Only products with >= min_units units sold in the window (avoid tiny denominators).
    """
    items = _enriched_items(t, start, end, {"product_id", "category"})
    per = items.groupby(["product_id", "product_name", "category"], dropna=False).agg(
        units=("quantity", "sum"), merch_sales=("net_amount", "sum")).reset_index()
    r = t["refunds"]
    r = r[(r.refund_status == "completed") & r.order_item_id.notna()]
    r = r.merge(items[["order_item_id", "product_id"]].drop_duplicates(), on="order_item_id", how="inner")
    refunds = r.groupby("product_id").refund_amount.sum()
    per["refunds"] = per.product_id.map(refunds).fillna(0.0)
    per = per[per.units >= min_units].copy()
    per["refund_rate"] = per.refunds / per.merch_sales
    per = per.sort_values(["refund_rate", "product_id"], ascending=[False, True]).head(top_n)
    return per.round({"merch_sales": 2, "refunds": 2, "refund_rate": 4}).reset_index(drop=True)


def order_refund_rate(t: dict, start, end) -> dict:
    """Share of booked orders with at least one completed item-level refund."""
    o = booked_orders(t, start, end)
    r = t["refunds"]
    refunded = set(r[(r.refund_status == "completed") & r.order_item_id.notna()].order_id)
    n = len(set(o.order_id))
    k = len(set(o.order_id) & refunded)
    return {"booked_orders": n, "refunded_orders": k, "order_refund_rate": round(k / n, 4)}


def delivery_performance(t: dict, start, end, group_by=None) -> pd.DataFrame:
    """Late delivery rate = delivered shipments with DATE(delivered_timestamp) > promised_delivery_date
                            / delivered shipments with a delivered_timestamp.

    Grain: shipment. Window applies to the order_date of the shipped order.
    In-transit shipments are excluded from the denominator.
    """
    g = _dim_cols(group_by)
    o = t["orders"][["order_id", "order_date"]]
    s = t["shipments"].merge(o, on="order_id", how="inner")
    s = s[_in_window(s.order_date, start, end)]
    s = s[(s.shipment_status == "delivered") & s.delivered_timestamp.notna()].copy()
    s["is_late"] = pd.to_datetime(s.delivered_timestamp).dt.date > s.promised_delivery_date
    if g:
        agg = s.groupby(g).agg(delivered=("shipment_id", "count"), late=("is_late", "sum"))
    else:
        agg = pd.DataFrame({"delivered": [len(s)], "late": [int(s.is_late.sum())]})
    agg["late_rate"] = (agg.late / agg.delivered).round(4)
    agg["on_time_rate"] = (1 - agg.late / agg.delivered).round(4)
    return agg.reset_index() if g else agg.reset_index(drop=True)


def mom_change(series: pd.DataFrame, value_col: str, period_col: str = "month", by: str | None = None) -> pd.DataFrame:
    """MoM % = (current - previous) / previous * 100; NULL when previous is 0 or missing."""
    df = series.sort_values(([by] if by else []) + [period_col]).copy()
    prev = df.groupby(by)[value_col].shift(1) if by else df[value_col].shift(1)
    df["previous"] = prev
    df["mom_pct"] = np.where(prev > 0, (df[value_col] - prev) / prev * 100, np.nan)
    df["mom_pct"] = df["mom_pct"].round(2)
    return df


def reconciliation(t: dict, start, end) -> dict:
    """Bridge between net sales and cash for orders placed in the window.

    captured_payments (on window orders)
      = merch_sales_booked + shipping_booked + tax_booked + captured_on_cancelled_orders
    net_sales = merch_sales_booked - completed_item_refunds
    net_cash  = captured_payments - all completed refunds (item + cancellation)
    """
    o = t["orders"][_in_window(t["orders"].order_date, start, end)]
    booked = o[o.order_status.isin(sr.BOOKED_ORDER_STATUSES)]
    cancelled = o[o.order_status == "cancelled"]
    p = t["payments"]
    cap = p[(p.payment_status == "captured") & p.order_id.isin(o.order_id)]
    r = t["refunds"]
    r = r[(r.refund_status == "completed") & r.order_id.isin(o.order_id)]
    items = t["order_items"][t["order_items"].order_id.isin(booked.order_id)]
    item_ref = r[r.order_item_id.notna() & r.order_id.isin(booked.order_id)].refund_amount.sum()
    res = {
        "merch_sales_booked": items.net_amount.sum(),
        "shipping_booked": booked.shipping_amount.sum(),
        "tax_booked": booked.tax_amount.sum(),
        "captured_on_cancelled_orders": cap[cap.order_id.isin(cancelled.order_id)].amount.sum(),
        "captured_payments": cap.amount.sum(),
        "pending_payments": p[(p.payment_status == "pending") & p.order_id.isin(o.order_id)].amount.sum(),
        "completed_item_refunds": item_ref,
        "completed_cancellation_refunds": r[r.order_item_id.isna()].refund_amount.sum(),
        "pending_refunds_excluded": t["refunds"][(t["refunds"].refund_status == "pending") &
                                                 t["refunds"].order_id.isin(o.order_id)].refund_amount.sum(),
    }
    res["net_sales"] = res["merch_sales_booked"] - res["completed_item_refunds"]
    res["net_cash"] = res["captured_payments"] - res["completed_item_refunds"] - res["completed_cancellation_refunds"]
    res["bridge_difference"] = res["captured_payments"] - (res["merch_sales_booked"] + res["shipping_booked"] +
                                                          res["tax_booked"] + res["captured_on_cancelled_orders"])
    return {k: round(float(v), 2) + 0.0 for k, v in res.items()}  # + 0.0 normalises -0.0


def order_status_counts(t: dict) -> dict:
    return {k: int(v) for k, v in t["orders"].order_status.value_counts().sort_index().items()}


def cancellation_rate(t: dict, start, end) -> dict:
    o = t["orders"][_in_window(t["orders"].order_date, start, end)]
    n, k = len(o), int((o.order_status == "cancelled").sum())
    return {"orders": n, "cancelled": k, "cancellation_rate": round(k / n, 4)}


def payment_failure_rate(t: dict, start, end) -> dict:
    """Failed attempts / all attempts, by payment date. Grain: payment attempt."""
    p = t["payments"]
    p = p[_in_window(pd.to_datetime(p.payment_timestamp).dt.tz_convert(None).dt.normalize(), start, end)]
    n, k = len(p), int((p.payment_status == "failed").sum())
    return {"attempts": n, "failed": k, "failure_rate": round(k / n, 4)}


def captured_payments_by_month(t: dict, start, end) -> pd.DataFrame:
    """SUM(amount) of captured payments by payment month (cash view). Grain: payment."""
    p = t["payments"][t["payments"].payment_status == "captured"].copy()
    d = pd.to_datetime(p.payment_timestamp).dt.tz_convert(None)
    p = p[(d.dt.normalize() >= _d(start)) & (d.dt.normalize() <= _d(end))]
    p["month"] = pd.to_datetime(p.payment_timestamp).dt.strftime("%Y-%m")
    return p.groupby("month").amount.sum().round(2).reset_index(name="captured_payments")


def refunds_by_reason(t: dict, start, end) -> pd.DataFrame:
    """Completed refund amount by reason, by refund date. Grain: refund transaction."""
    r = t["refunds"][t["refunds"].refund_status == "completed"].copy()
    d = pd.to_datetime(r.refund_timestamp).dt.tz_convert(None).dt.normalize()
    r = r[(d >= _d(start)) & (d <= _d(end))]
    out = r.groupby("refund_reason").agg(refunds=("refund_id", "count"), amount=("refund_amount", "sum"))
    return out.round(2).sort_values("amount", ascending=False).reset_index()


def active_customers(t: dict, start, end) -> int:
    """Distinct customers with >= 1 booked order in the window."""
    return int(booked_orders(t, start, end).customer_id.nunique())


def basket_size(t: dict, start, end) -> dict:
    items = _enriched_items(t, start, end)
    n_orders = items.order_id.nunique()
    return {"avg_lines_per_order": round(len(items) / n_orders, 4),
            "avg_units_per_order": round(items.quantity.sum() / n_orders, 4)}
