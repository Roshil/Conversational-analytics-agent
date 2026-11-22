"""Controlled defect injection: clean dataset -> corrupted dataset + defect catalog.

Rules of the game
-----------------
* Defects are injected deterministically (own seed) into small, disjoint sets of
  records. Each order is touched by at most one defect so effects never mix.
* Every defect records the exact affected identifiers and the full set of rule
  findings it should produce, including genuine cascades (e.g. a duplicated
  order line also makes the order header disagree with its lines).
* Legitimate edge cases (pending payments, failed-then-captured attempts,
  cancelled-and-refunded orders, partial refunds, in-transit shipments) are
  left untouched and must NOT be flagged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from analytics_agent import schema_registry as sr
from analytics_agent.data_generation.generator import normalize_dtypes


@dataclass
class Defect:
    defect_id: str
    table: str
    columns: list
    defect_type: str
    severity: str
    description: str
    detection_rule: str
    business_impact: str
    expected_outcome: str
    affected_ids: list = field(default_factory=list)
    expected_findings: dict = field(default_factory=dict)   # rule_id -> sorted list of ids

    def to_dict(self) -> dict:
        d = asdict(self)
        d["affected_count"] = len(self.affected_ids)
        return d


class _Picker:
    """Deterministic sampling that never reuses an order across defects."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.used_orders: set[str] = set()
        self.used_customers: set[str] = set()

    def pick(self, candidates, n, order_ids=None):
        candidates = list(candidates)
        if order_ids is not None:
            mapping = dict(zip(candidates, order_ids))
            candidates = [c for c in candidates if mapping[c] not in self.used_orders]
        if len(candidates) < n:
            raise ValueError("not enough candidates for defect injection")
        idx = self.rng.choice(len(candidates), size=n, replace=False)
        chosen = sorted(candidates[i] for i in idx)
        if order_ids is not None:
            self.used_orders.update(mapping[c] for c in chosen)
        return chosen


def corrupt(clean: dict[str, pd.DataFrame], seed: int = 777) -> tuple[dict[str, pd.DataFrame], list[Defect]]:
    t = {k: v.copy() for k, v in clean.items()}
    pk = _Picker(seed)
    rng = pk.rng
    defects: list[Defect] = []

    orders = t["orders"]
    booked = orders[orders.order_status.isin(sr.BOOKED_ORDER_STATUSES)]
    delivered = orders[orders.order_status == "delivered"]
    ord_status = orders.set_index("order_id").order_status

    # ---- D01 duplicate customer business key -------------------------------------------------
    custs_with_orders = sorted(set(orders.customer_id))
    ids = pk.pick(custs_with_orders, 5)
    pk.used_customers.update(ids)
    dup = t["customers"][t["customers"].customer_id.isin(ids)].copy()
    dup["region"] = [sr.REGIONS[(sr.REGIONS.index(r) + 1) % len(sr.REGIONS)] for r in dup.region]
    dup["created_at"] = dup.created_at + pd.Timedelta(days=1)
    t["customers"] = pd.concat([t["customers"], dup], ignore_index=True)
    emails = sorted(dup.email)
    defects.append(Defect(
        "D01", "customers", ["customer_id", "email", "region"], "duplicate_primary_key", "critical",
        "5 customers re-inserted with the same customer_id and email but a different region (e.g. a botched CRM sync).",
        "CUS_PK_UNIQUE",
        "Joining orders to customers doubles these customers' orders, inflating regional sales and AOV; the customer has two conflicting regions.",
        "Fail CUS_PK_UNIQUE with 5 keys; also CUS_EMAIL_UNIQUE (same emails). Analytics joining customers must be flagged as unreliable for these customers.",
        ids, {"CUS_PK_UNIQUE": ids, "CUS_EMAIL_UNIQUE": emails}))

    # ---- D02 missing required value ----------------------------------------------------------
    cand = [c for c in t["customers"].customer_id.unique() if c not in pk.used_customers]
    ids = pk.pick(cand, 10)
    pk.used_customers.update(ids)
    t["customers"].loc[t["customers"].customer_id.isin(ids), "region"] = None
    defects.append(Defect(
        "D02", "customers", ["region"], "missing_required_value", "high",
        "region set to NULL for 10 customers.", "CUS_REQUIRED_NOT_NULL",
        "Their orders fall into a NULL region bucket in regional reports.",
        "Fail CUS_REQUIRED_NOT_NULL with 10 customers. Regional answers must disclose an 'unknown region' share.",
        ids, {"CUS_REQUIRED_NOT_NULL": ids}))

    # ---- D03 invalid category values ---------------------------------------------------------
    prods = t["products"]
    ids = pk.pick(prods.product_id, 3)
    bad_values = ["Electronix", "apparel", "Home and Kitchen"]
    for pid, val in zip(ids, bad_values):
        prods.loc[prods.product_id == pid, "category"] = val
    defects.append(Defect(
        "D03", "products", ["category"], "invalid_category", "medium",
        "3 products get non-canonical categories: a typo, a case variant and a spelling variant.",
        "PRD_CATEGORY_ALLOWED",
        "Category totals split into phantom categories; canonical categories are under-reported.",
        "Fail PRD_CATEGORY_ALLOWED with 3 products.", ids, {"PRD_CATEGORY_ALLOWED": ids}))

    # ---- D04 duplicate SKU business key ------------------------------------------------------
    remaining = [p for p in prods.product_id if p not in ids]
    pair_ids = pk.pick(remaining, 4)
    skus = []
    for a, b in [(pair_ids[0], pair_ids[1]), (pair_ids[2], pair_ids[3])]:
        sku_a = prods.loc[prods.product_id == a, "sku"].iloc[0]
        prods.loc[prods.product_id == b, "sku"] = sku_a
        skus.append(sku_a)
    defects.append(Defect(
        "D04", "products", ["sku"], "duplicate_business_key", "high",
        "2 pairs of different products share a SKU.", "PRD_SKU_UNIQUE",
        "Inventory/SKU-level reporting merges unrelated products.",
        "Fail PRD_SKU_UNIQUE with 2 SKU values (4 products).", pair_ids, {"PRD_SKU_UNIQUE": sorted(skus)}))

    # ---- D05 orphan customer FK on orders ----------------------------------------------------
    ids = pk.pick(booked.order_id, 8, order_ids=booked.order_id)
    for k, oid in enumerate(ids):
        orders.loc[orders.order_id == oid, "customer_id"] = f"CUS9{k:05d}"
    defects.append(Defect(
        "D05", "orders", ["customer_id"], "orphan_foreign_key", "high",
        "8 booked orders reference customer_ids that do not exist.", "ORD_CUSTOMER_FK",
        "INNER JOIN to customers silently drops these orders' sales; LEFT JOIN yields NULL region.",
        "Fail ORD_CUSTOMER_FK with 8 orders.", ids, {"ORD_CUSTOMER_FK": ids}))

    # ---- D06 invalid order status ------------------------------------------------------------
    ids = pk.pick(delivered.order_id, 6, order_ids=delivered.order_id)
    orders.loc[orders.order_id.isin(ids), "order_status"] = "complete"
    defects.append(Defect(
        "D06", "orders", ["order_status"], "invalid_status", "medium",
        "6 delivered orders carry the non-canonical status 'complete'.", "ORD_STATUS_ALLOWED",
        "Filters on booked statuses exclude these orders, understating sales.",
        "Fail ORD_STATUS_ALLOWED with 6 orders. Consistency rules skip records already failing the status domain.",
        ids, {"ORD_STATUS_ALLOWED": ids}))

    # ---- D07 order money components do not reconcile -----------------------------------------
    cand = booked[booked.tax_amount > 0]
    ids = pk.pick(cand.order_id, 10, order_ids=cand.order_id)
    orders.loc[orders.order_id.isin(ids), "tax_amount"] = 0.0
    defects.append(Defect(
        "D07", "orders", ["tax_amount", "order_total"], "incorrect_monetary_total", "high",
        "tax_amount zeroed on 10 booked orders while order_total (and the captured payment) still include tax.",
        "ORD_TOTAL_CONSISTENT",
        "Tax reporting understated; order_total no longer equals its components.",
        "Fail ORD_TOTAL_CONSISTENT with 10 orders. Payments still reconcile to order_total, so PAY_CAPTURED_MATCHES_ORDER passes.",
        ids, {"ORD_TOTAL_CONSISTENT": ids}))

    items = t["order_items"]
    item_order = dict(zip(items.order_item_id, items.order_id))
    refunded_items = set(t["refunds"].order_item_id.dropna())

    # ---- D08 negative quantities -------------------------------------------------------------
    cand = items[items.order_id.isin(booked.order_id) & ~items.order_item_id.isin(refunded_items)]
    ids = pk.pick(cand.order_item_id, 5, order_ids=[item_order[i] for i in cand.order_item_id])
    items.loc[items.order_item_id.isin(ids), "quantity"] *= -1
    defects.append(Defect(
        "D08", "order_items", ["quantity"], "negative_quantity", "high",
        "quantity negated on 5 lines; money columns left unchanged.", "ITM_QUANTITY_RANGE",
        "Unit totals understated; quantity * unit_price no longer equals gross_amount.",
        "Fail ITM_QUANTITY_RANGE and ITM_AMOUNTS_CONSISTENT with the same 5 lines.",
        ids, {"ITM_QUANTITY_RANGE": ids, "ITM_AMOUNTS_CONSISTENT": ids}))

    # ---- D09 orphan order lines (inserted) ---------------------------------------------------
    pool = items[~items.order_item_id.isin(ids)]          # never copy a D08-modified line
    src = pool.iloc[pk.rng.choice(len(pool), size=5, replace=False)].copy()
    new_ids = [f"ITM9{k:07d}" for k in range(5)]
    src["order_item_id"] = new_ids
    src["order_id"] = [f"ORD9{k:06d}" for k in range(5)]
    src["line_number"] = 1
    t["order_items"] = items = pd.concat([items, src], ignore_index=True)
    defects.append(Defect(
        "D09", "order_items", ["order_id"], "orphan_foreign_key", "high",
        "5 order lines inserted that reference non-existent orders.", "ITM_ORDER_FK",
        "Item-level sales that do not join to an order header; item-only queries over-report.",
        "Fail ITM_ORDER_FK with 5 lines.", new_ids, {"ITM_ORDER_FK": new_ids}))

    # ---- D10 exact duplicate order lines -----------------------------------------------------
    clean_items = t["order_items"][~t["order_items"].order_item_id.str.startswith("ITM9")]
    cand = clean_items[clean_items.order_id.isin(booked.order_id) & ~clean_items.order_item_id.isin(refunded_items)]
    ids = pk.pick(cand.order_item_id, 12, order_ids=[item_order[i] for i in cand.order_item_id])
    dup = t["order_items"][t["order_items"].order_item_id.isin(ids)].copy()
    t["order_items"] = items = pd.concat([t["order_items"], dup], ignore_index=True)
    dup_orders = sorted({item_order[i] for i in ids})
    defects.append(Defect(
        "D10", "order_items", ["*"], "duplicate_rows", "critical",
        "12 order lines loaded twice (exact duplicate rows).", "ITM_PK_UNIQUE",
        "Item-based sales metrics are inflated by the duplicated lines' net_amount (fan-out).",
        "Fail ITM_PK_UNIQUE and ITM_LINE_UNIQUE with 12 lines; cascade: ORD_SUBTOTAL_MATCHES_ITEMS fails for their 12 orders.",
        ids, {"ITM_PK_UNIQUE": ids, "ITM_LINE_UNIQUE": ids, "ORD_SUBTOTAL_MATCHES_ITEMS": dup_orders}))

    pays = t["payments"]
    captured = pays[(pays.payment_status == "captured") & pays.order_id.isin(booked.order_id)]

    # ---- D11 duplicate payment transactions --------------------------------------------------
    ids = pk.pick(captured.payment_id, 15, order_ids=captured.order_id)
    dup = pays[pays.payment_id.isin(ids)].copy()
    dup["payment_id"] = [f"PAY9{k:06d}" for k in range(len(dup))]
    dup["payment_timestamp"] = dup.payment_timestamp + pd.Timedelta(seconds=2)
    t["payments"] = pays = pd.concat([pays, dup], ignore_index=True)
    defects.append(Defect(
        "D11", "payments", ["provider_transaction_id"], "duplicate_transaction", "critical",
        "15 captured payments recorded twice under new payment_ids with the same provider_transaction_id.",
        "PAY_TXN_UNIQUE",
        "Total payments received is overstated by the duplicated amounts; payments no longer reconcile to orders.",
        "Fail PAY_TXN_UNIQUE with 15 transaction ids; cascade: PAY_CAPTURED_MATCHES_ORDER fails for the 15 orders.",
        sorted(dup.payment_id), {"PAY_TXN_UNIQUE": sorted(dup.provider_transaction_id),
                                 "PAY_CAPTURED_MATCHES_ORDER": sorted(dup.order_id)}))

    # ---- D12 payment before order ------------------------------------------------------------
    captured = pays[(pays.payment_status == "captured") & pays.order_id.isin(booked.order_id) &
                    ~pays.payment_id.str.startswith("PAY9")]
    ids = pk.pick(captured.payment_id, 5, order_ids=captured.order_id)
    ots = orders.set_index("order_id").order_timestamp
    for pid in ids:
        oid = pays.loc[pays.payment_id == pid, "order_id"].iloc[0]
        pays.loc[pays.payment_id == pid, "payment_timestamp"] = ots[oid] - pd.Timedelta(hours=int(rng.integers(1, 48)))
    defects.append(Defect(
        "D12", "payments", ["payment_timestamp"], "invalid_timestamp_sequence", "medium",
        "5 captured payments timestamped 1-48 hours before their order was placed.", "PAY_AFTER_ORDER",
        "Payment-date reports shift cash into earlier periods; indicates clock or ETL timezone problems.",
        "Fail PAY_AFTER_ORDER with 5 payments.", ids, {"PAY_AFTER_ORDER": ids}))

    # ---- D13 refunds exceeding refundable amount ---------------------------------------------
    ref = t["refunds"]
    cand = ref[(ref.refund_status == "completed") & ref.order_item_id.notna()]
    ids = pk.pick(cand.refund_id, 6, order_ids=cand.order_id)
    net = items.drop_duplicates("order_item_id").set_index("order_item_id").net_amount
    for rid in ids:
        item_id = ref.loc[ref.refund_id == rid, "order_item_id"].iloc[0]
        ref.loc[ref.refund_id == rid, "refund_amount"] = round(float(net[item_id]) * 3, 2)
    defects.append(Defect(
        "D13", "refunds", ["refund_amount"], "refund_exceeds_refundable", "critical",
        "6 completed item refunds set to 3x the line's net amount.", "REF_WITHIN_REFUNDABLE",
        "Net sales understated; potential financial loss or fraud signal.",
        "Fail REF_WITHIN_REFUNDABLE with 6 refunds. Legitimate partial refunds must still pass.",
        ids, {"REF_WITHIN_REFUNDABLE": ids}))

    shp = t["shipments"]
    dlv = shp[shp.shipment_status == "delivered"]

    # ---- D14 delivered before shipped --------------------------------------------------------
    ids = pk.pick(dlv.shipment_id, 7, order_ids=dlv.order_id)
    for sid in ids:
        s_ts = shp.loc[shp.shipment_id == sid, "shipped_timestamp"].iloc[0]
        shp.loc[shp.shipment_id == sid, "delivered_timestamp"] = s_ts - pd.Timedelta(hours=int(rng.integers(2, 48)))
    defects.append(Defect(
        "D14", "shipments", ["delivered_timestamp"], "invalid_timestamp_sequence", "high",
        "7 delivered shipments have delivered_timestamp earlier than shipped_timestamp.", "SHP_DELIVERY_AFTER_SHIP",
        "Delivery-time metrics become negative/flattering; on-time rate overstated.",
        "Fail SHP_DELIVERY_AFTER_SHIP with 7 shipments.", ids, {"SHP_DELIVERY_AFTER_SHIP": ids}))

    # ---- D15 delivered without delivery timestamp --------------------------------------------
    ids = pk.pick(dlv.shipment_id, 6, order_ids=dlv.order_id)
    shp.loc[shp.shipment_id.isin(ids), "delivered_timestamp"] = pd.NaT
    defects.append(Defect(
        "D15", "shipments", ["shipment_status", "delivered_timestamp"], "inconsistent_state", "high",
        "6 shipments marked delivered with no delivered_timestamp.", "SHP_DELIVERED_TS_CONSISTENT",
        "These shipments silently drop out of on-time/late calculations.",
        "Fail SHP_DELIVERED_TS_CONSISTENT with 6 shipments.", ids, {"SHP_DELIVERED_TS_CONSISTENT": ids}))

    out = {name: normalize_dtypes(name, df) for name, df in t.items()}
    return out, defects


def expected_findings(defects: list[Defect]) -> dict[str, set]:
    """Union of all expected findings by rule id (what a perfect validator reports)."""
    agg: dict[str, set] = {}
    for d in defects:
        for rule_id, ids in d.expected_findings.items():
            agg.setdefault(rule_id, set()).update(map(str, ids))
    return agg


def legitimate_cases(clean: dict[str, pd.DataFrame]) -> list[dict]:
    """Patterns that look unusual but are correct, with counts. Validators must not flag them."""
    o, p, r, s = clean["orders"], clean["payments"], clean["refunds"], clean["shipments"]
    items = clean["order_items"]
    item_refunds = r[r.order_item_id.notna()].merge(items[["order_item_id", "net_amount"]], on="order_item_id")
    failed_then_captured = set(p[p.payment_status == "failed"].order_id) & set(p[p.payment_status == "captured"].order_id)
    cancelled_paid = set(o[o.order_status == "cancelled"].order_id) & set(p[p.payment_status == "captured"].order_id)
    return [
        {"case": "pending_payment_orders", "count": int((o.order_status == "pending_payment").sum()),
         "why_legitimate": "Recent orders awaiting payment confirmation; excluded from sales, not an error."},
        {"case": "failed_then_captured_payments", "count": len(failed_then_captured),
         "why_legitimate": "A declined attempt followed by a successful one; only the captured row counts."},
        {"case": "cancelled_orders_with_captured_payment_and_full_refund", "count": len(cancelled_paid),
         "why_legitimate": "Paid then cancelled; an order-level refund returns the full amount."},
        {"case": "cancelled_orders_without_payment", "count": int(len(set(o[o.order_status == 'cancelled'].order_id) - set(p.order_id))),
         "why_legitimate": "Cancelled before any payment attempt."},
        {"case": "partial_item_refunds", "count": int((item_refunds.refund_amount < item_refunds.net_amount - 0.005).sum()),
         "why_legitimate": "Refund below the line's net amount (e.g. goodwill credit)."},
        {"case": "pending_refunds", "count": int((r.refund_status == "pending").sum()),
         "why_legitimate": "Initiated within the last days of the snapshot; not yet settled."},
        {"case": "in_transit_shipments", "count": int((s.shipment_status == "in_transit").sum()),
         "why_legitimate": "No delivered_timestamp yet; excluded from on-time denominators."},
        {"case": "processing_orders_without_shipment", "count": int((o.order_status == "processing").sum()),
         "why_legitimate": "Paid but not yet shipped."},
        {"case": "customers_without_orders", "count": int((~clean["customers"].customer_id.isin(o.customer_id)).sum()),
         "why_legitimate": "Registered but never purchased."},
        {"case": "products_never_sold", "count": int((~clean["products"].product_id.isin(items.product_id)).sum()),
         "why_legitimate": "Catalog items with no sales in the window."},
    ]
