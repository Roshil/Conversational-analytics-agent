"""Reference data-quality rules implemented in pandas.

Purpose: a *test oracle* for the generator, NOT the production data-quality
engine. The production engine (Stage 5) runs equivalent rules as BigQuery SQL
with the same rule IDs, and its findings are compared against this oracle and
against the defect catalog.

Each rule returns the set of offending record identifiers (as strings). An empty
set means the rule passes. Rules are written so that a record already failing a
"domain" rule (e.g. unknown status) is not double-reported by a dependent
consistency rule; cross-rule cascades that genuinely exist are listed in the
defect catalog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from analytics_agent import schema_registry as sr

MONEY_TOL = 0.005  # half a cent


@dataclass(frozen=True)
class RefRule:
    rule_id: str
    table: str
    severity: str           # critical | high | medium | low
    description: str
    id_column: str          # what the returned identifiers refer to
    fn: Callable[[dict, pd.Timestamp], set]


def _dupes(s: pd.Series) -> set:
    return set(s[s.duplicated(keep=False)].dropna().astype(str))


def _orphans(child: pd.DataFrame, col: str, parent: pd.DataFrame, pcol: str, id_col: str) -> set:
    c = child[child[col].notna()]
    return set(c.loc[~c[col].isin(parent[pcol]), id_col].astype(str))


def _not_allowed(df, col, allowed, id_col) -> set:
    d = df[df[col].notna()]
    return set(d.loc[~d[col].isin(allowed), id_col].astype(str))


def _required_nulls(df: pd.DataFrame, table: str, id_col: str) -> set:
    req = sr.get_table(table).required_columns
    mask = df[req].isna().any(axis=1)
    return set(df.loc[mask, id_col].astype(str))


def _ts(s):
    return pd.to_datetime(s, utc=True)


# ---------------------------------------------------------------- rules ----
def r_ord_total(t, as_of):
    o = t["orders"]
    o = o[o.order_status.isin(sr.ORDER_STATUSES)]
    calc = o.items_subtotal - o.discount_amount + o.shipping_amount + o.tax_amount
    return set(o.loc[(calc - o.order_total).abs() > MONEY_TOL, "order_id"])


def r_ord_subtotal_items(t, as_of):
    o, i = t["orders"], t["order_items"]
    sums = i.groupby("order_id").gross_amount.sum()
    o = o[o.order_status.isin(sr.ORDER_STATUSES)]
    item_sum = o.order_id.map(sums).fillna(0.0)
    return set(o.loc[(item_sum - o.items_subtotal).abs() > MONEY_TOL, "order_id"])


def r_ord_cancel_ts(t, as_of):
    o = t["orders"]
    o = o[o.order_status.isin(sr.ORDER_STATUSES)]
    bad = (o.order_status == "cancelled") != o.cancelled_timestamp.notna()
    return set(o.loc[bad, "order_id"])


def r_ord_not_future(t, as_of):
    o = t["orders"]
    return set(o.loc[_ts(o.order_timestamp) > as_of, "order_id"])


def r_itm_qty(t, as_of):
    i = t["order_items"]
    return set(i.loc[(i.quantity < 1) | (i.quantity > 10), "order_item_id"])


def r_itm_amounts(t, as_of):
    i = t["order_items"]
    bad = ((i.quantity * i.unit_price - i.gross_amount).abs() > MONEY_TOL) | \
          ((i.gross_amount - i.discount_amount - i.net_amount).abs() > MONEY_TOL)
    return set(i.loc[bad, "order_item_id"])


def r_itm_line_unique(t, as_of):
    i = t["order_items"]
    d = i[i.duplicated(["order_id", "line_number"], keep=False)]
    return set(d.order_item_id)


def r_pay_after_order(t, as_of):
    p = t["payments"].merge(t["orders"][["order_id", "order_timestamp"]], on="order_id", how="inner")
    return set(p.loc[_ts(p.payment_timestamp) < _ts(p.order_timestamp), "payment_id"])


def r_pay_captured_matches(t, as_of):
    o = t["orders"]
    booked = o[o.order_status.isin(sr.BOOKED_ORDER_STATUSES)]
    cap = t["payments"][t["payments"].payment_status == "captured"]
    g = cap.groupby("order_id").agg(n=("payment_id", "size"), amt=("amount", "sum"))
    b = booked.set_index("order_id").join(g, how="left")
    bad = (b.n.fillna(0) != 1) | ((b.amt.fillna(0) - b.order_total).abs() > MONEY_TOL)
    return set(b.index[bad])


def r_pay_failure_reason(t, as_of):
    p = t["payments"]
    p = p[p.payment_status.isin(sr.PAYMENT_STATUSES)]
    bad = (p.payment_status == "failed") != p.failure_reason.notna()
    return set(p.loc[bad, "payment_id"])


def r_shp_delivered_ts(t, as_of):
    s = t["shipments"]
    s = s[s.shipment_status.isin(sr.SHIPMENT_STATUSES)]
    bad = (s.shipment_status == "delivered") != s.delivered_timestamp.notna()
    return set(s.loc[bad, "shipment_id"])


def r_shp_after_ship(t, as_of):
    s = t["shipments"]
    s = s[s.delivered_timestamp.notna()]
    return set(s.loc[_ts(s.delivered_timestamp) < _ts(s.shipped_timestamp), "shipment_id"])


def r_shp_status_matches_order(t, as_of):
    s = t["shipments"].merge(t["orders"][["order_id", "order_status"]], on="order_id", how="inner")
    s = s[s.order_status.isin(sr.ORDER_STATUSES) & s.shipment_status.isin(sr.SHIPMENT_STATUSES)]
    expected = s.shipment_status.map({"delivered": "delivered", "in_transit": "shipped"})
    return set(s.loc[expected != s.order_status, "shipment_id"])


def r_ref_within_refundable(t, as_of):
    r, i = t["refunds"], t["order_items"]
    item_refunds = r[r.order_item_id.notna()]
    per_item = item_refunds.groupby("order_item_id").refund_amount.sum()
    net = i.drop_duplicates("order_item_id").set_index("order_item_id").net_amount
    over_items = per_item.index[(per_item - net.reindex(per_item.index)) > MONEY_TOL]
    bad = set(item_refunds.loc[item_refunds.order_item_id.isin(over_items), "refund_id"])
    # order-level refunds vs captured amount of the referenced payment
    lvl = r[r.order_item_id.isna()].merge(
        t["payments"][["payment_id", "amount"]].drop_duplicates("payment_id"), on="payment_id", how="left")
    bad |= set(lvl.loc[(lvl.refund_amount - lvl.amount) > MONEY_TOL, "refund_id"])
    return bad


def r_ref_item_null_only_cancel(t, as_of):
    r = t["refunds"]
    bad = r.order_item_id.isna() != (r.refund_reason == "order_cancelled")
    return set(r.loc[bad, "refund_id"])


RULES: tuple[RefRule, ...] = (
    # customers
    RefRule("CUS_PK_UNIQUE", "customers", "critical", "customer_id is unique", "customer_id",
            lambda t, a: _dupes(t["customers"].customer_id)),
    RefRule("CUS_REQUIRED_NOT_NULL", "customers", "high", "Required customer columns are not null", "customer_id",
            lambda t, a: _required_nulls(t["customers"], "customers", "customer_id")),
    RefRule("CUS_REGION_ALLOWED", "customers", "medium", "region in allowed values", "customer_id",
            lambda t, a: _not_allowed(t["customers"], "region", sr.REGIONS, "customer_id")),
    RefRule("CUS_EMAIL_UNIQUE", "customers", "medium", "email is unique", "email",
            lambda t, a: _dupes(t["customers"].email)),
    # products
    RefRule("PRD_PK_UNIQUE", "products", "critical", "product_id is unique", "product_id",
            lambda t, a: _dupes(t["products"].product_id)),
    RefRule("PRD_SKU_UNIQUE", "products", "high", "sku (business key) is unique", "sku",
            lambda t, a: _dupes(t["products"].sku)),
    RefRule("PRD_CATEGORY_ALLOWED", "products", "medium", "category in allowed values", "product_id",
            lambda t, a: _not_allowed(t["products"], "category", sr.CATEGORIES, "product_id")),
    RefRule("PRD_PRICE_VALID", "products", "high", "list_price > 0 and 0 < unit_cost < list_price", "product_id",
            lambda t, a: set(t["products"].loc[(t["products"].list_price <= 0) | (t["products"].unit_cost <= 0) |
                                               (t["products"].unit_cost >= t["products"].list_price), "product_id"])),
    # orders
    RefRule("ORD_PK_UNIQUE", "orders", "critical", "order_id is unique", "order_id",
            lambda t, a: _dupes(t["orders"].order_id)),
    RefRule("ORD_REQUIRED_NOT_NULL", "orders", "high", "Required order columns are not null", "order_id",
            lambda t, a: _required_nulls(t["orders"], "orders", "order_id")),
    RefRule("ORD_STATUS_ALLOWED", "orders", "medium", "order_status in allowed values", "order_id",
            lambda t, a: _not_allowed(t["orders"], "order_status", sr.ORDER_STATUSES, "order_id")),
    RefRule("ORD_CUSTOMER_FK", "orders", "high", "customer_id exists in customers", "order_id",
            lambda t, a: _orphans(t["orders"], "customer_id", t["customers"], "customer_id", "order_id")),
    RefRule("ORD_TOTAL_CONSISTENT", "orders", "high",
            "order_total = items_subtotal - discount_amount + shipping_amount + tax_amount", "order_id", r_ord_total),
    RefRule("ORD_SUBTOTAL_MATCHES_ITEMS", "orders", "high", "items_subtotal = SUM(order_items.gross_amount)",
            "order_id", r_ord_subtotal_items),
    RefRule("ORD_CANCELLED_TS_CONSISTENT", "orders", "medium", "cancelled_timestamp set iff status = cancelled",
            "order_id", r_ord_cancel_ts),
    RefRule("ORD_NOT_FUTURE", "orders", "medium", "order_timestamp <= snapshot time", "order_id", r_ord_not_future),
    # order_items
    RefRule("ITM_PK_UNIQUE", "order_items", "critical", "order_item_id is unique", "order_item_id",
            lambda t, a: _dupes(t["order_items"].order_item_id)),
    RefRule("ITM_LINE_UNIQUE", "order_items", "high", "(order_id, line_number) is unique", "order_item_id",
            r_itm_line_unique),
    RefRule("ITM_ORDER_FK", "order_items", "high", "order_id exists in orders", "order_item_id",
            lambda t, a: _orphans(t["order_items"], "order_id", t["orders"], "order_id", "order_item_id")),
    RefRule("ITM_PRODUCT_FK", "order_items", "high", "product_id exists in products", "order_item_id",
            lambda t, a: _orphans(t["order_items"], "product_id", t["products"], "product_id", "order_item_id")),
    RefRule("ITM_QUANTITY_RANGE", "order_items", "high", "quantity between 1 and 10", "order_item_id", r_itm_qty),
    RefRule("ITM_AMOUNTS_CONSISTENT", "order_items", "high",
            "gross = quantity * unit_price and net = gross - discount", "order_item_id", r_itm_amounts),
    # payments
    RefRule("PAY_PK_UNIQUE", "payments", "critical", "payment_id is unique", "payment_id",
            lambda t, a: _dupes(t["payments"].payment_id)),
    RefRule("PAY_TXN_UNIQUE", "payments", "critical", "provider_transaction_id is unique", "provider_transaction_id",
            lambda t, a: _dupes(t["payments"].provider_transaction_id)),
    RefRule("PAY_ORDER_FK", "payments", "high", "order_id exists in orders", "payment_id",
            lambda t, a: _orphans(t["payments"], "order_id", t["orders"], "order_id", "payment_id")),
    RefRule("PAY_STATUS_ALLOWED", "payments", "medium", "payment_status in allowed values", "payment_id",
            lambda t, a: _not_allowed(t["payments"], "payment_status", sr.PAYMENT_STATUSES, "payment_id")),
    RefRule("PAY_AFTER_ORDER", "payments", "medium", "payment_timestamp >= order_timestamp", "payment_id",
            r_pay_after_order),
    RefRule("PAY_CAPTURED_MATCHES_ORDER", "payments", "critical",
            "Each booked order has exactly one captured payment equal to order_total", "order_id",
            r_pay_captured_matches),
    RefRule("PAY_FAILURE_REASON_CONSISTENT", "payments", "low", "failure_reason set iff status = failed",
            "payment_id", r_pay_failure_reason),
    # shipments
    RefRule("SHP_PK_UNIQUE", "shipments", "critical", "shipment_id is unique", "shipment_id",
            lambda t, a: _dupes(t["shipments"].shipment_id)),
    RefRule("SHP_ONE_PER_ORDER", "shipments", "medium", "At most one shipment per order (v1)", "order_id",
            lambda t, a: _dupes(t["shipments"].order_id)),
    RefRule("SHP_ORDER_FK", "shipments", "high", "order_id exists in orders", "shipment_id",
            lambda t, a: _orphans(t["shipments"], "order_id", t["orders"], "order_id", "shipment_id")),
    RefRule("SHP_DELIVERED_TS_CONSISTENT", "shipments", "high", "delivered_timestamp set iff status = delivered",
            "shipment_id", r_shp_delivered_ts),
    RefRule("SHP_DELIVERY_AFTER_SHIP", "shipments", "high", "delivered_timestamp >= shipped_timestamp",
            "shipment_id", r_shp_after_ship),
    RefRule("SHP_STATUS_MATCHES_ORDER", "shipments", "medium",
            "delivered shipment <-> delivered order; in_transit <-> shipped order", "shipment_id",
            r_shp_status_matches_order),
    # refunds
    RefRule("REF_PK_UNIQUE", "refunds", "critical", "refund_id is unique", "refund_id",
            lambda t, a: _dupes(t["refunds"].refund_id)),
    RefRule("REF_ORDER_FK", "refunds", "high", "order_id exists in orders", "refund_id",
            lambda t, a: _orphans(t["refunds"], "order_id", t["orders"], "order_id", "refund_id")),
    RefRule("REF_ITEM_FK", "refunds", "high", "order_item_id (when set) exists in order_items", "refund_id",
            lambda t, a: _orphans(t["refunds"], "order_item_id", t["order_items"], "order_item_id", "refund_id")),
    RefRule("REF_PAYMENT_FK", "refunds", "high", "payment_id exists in payments", "refund_id",
            lambda t, a: _orphans(t["refunds"], "payment_id", t["payments"], "payment_id", "refund_id")),
    RefRule("REF_WITHIN_REFUNDABLE", "refunds", "critical",
            "Item refunds <= item net_amount; order-level refunds <= captured amount", "refund_id",
            r_ref_within_refundable),
    RefRule("REF_ITEM_NULL_ONLY_CANCEL", "refunds", "medium",
            "order_item_id is NULL only for order_cancelled refunds", "refund_id", r_ref_item_null_only_cancel),
)


def run_reference_checks(tables: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, set]:
    """Run every reference rule; returns {rule_id: set_of_offending_ids}."""
    return {rule.rule_id: rule.fn(tables, as_of) for rule in RULES}


def summarize(results: dict[str, set]) -> pd.DataFrame:
    rows = []
    by_id = {r.rule_id: r for r in RULES}
    for rid, ids in results.items():
        r = by_id[rid]
        rows.append({"rule_id": rid, "table": r.table, "severity": r.severity,
                     "status": "pass" if not ids else "fail", "violations": len(ids),
                     "example_ids": ", ".join(sorted(ids)[:3]), "description": r.description})
    return pd.DataFrame(rows)
