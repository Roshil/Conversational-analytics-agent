"""Deterministic generator for the CLEAN synthetic e-commerce dataset.

Design principles
-----------------
* One fixed seed -> byte-identical output (numpy Generator / PCG64).
* Order lifecycle is simulated against a snapshot time (`as_of`), so statuses
  are realistic: recent orders are pending/processing/in transit, older ones
  delivered. Nothing is dated after `as_of`.
* Money is rounded to cents at the line level and totals are built from rounded
  parts, so every documented identity holds exactly (to the cent).
* A few deliberate business patterns exist so analytical questions have real
  answers: Q4 seasonality, a Southwest decline in May-June 2026, carriers with
  different delivery speed, products with elevated refund propensity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from analytics_agent import schema_registry as sr

TAX_RATES = {"Northeast": 0.0725, "Southeast": 0.065, "Midwest": 0.06, "Southwest": 0.0675, "West": 0.08}
FREE_SHIPPING_THRESHOLD = 75.00
SHIPPING_FEE = 6.99
PROMISE_DAYS = 5

CATEGORY_SPEC = {
    # category: (share of catalog, min price, max price, base refund propensity, sku code)
    "Electronics": (0.18, 25, 900, 0.060, "ELE"),
    "Home & Kitchen": (0.18, 12, 300, 0.045, "HOM"),
    "Apparel": (0.20, 10, 150, 0.110, "APP"),
    "Beauty": (0.12, 5, 80, 0.030, "BEA"),
    "Sports & Outdoors": (0.12, 15, 400, 0.050, "SPO"),
    "Books": (0.10, 8, 60, 0.015, "BOO"),
    "Toys": (0.10, 10, 120, 0.040, "TOY"),
}
ADJECTIVES = ["Classic", "Compact", "Deluxe", "Eco", "Essential", "Premium", "Smart", "Ultra", "Urban", "Vintage"]
NOUNS = {
    "Electronics": ["Speaker", "Headphones", "Charger", "Monitor", "Keyboard", "Webcam", "Router", "Tablet"],
    "Home & Kitchen": ["Blender", "Knife Set", "Kettle", "Lamp", "Cookware Set", "Throw Blanket", "Mug Set"],
    "Apparel": ["Jacket", "T-Shirt", "Jeans", "Sneakers", "Hoodie", "Dress", "Scarf"],
    "Beauty": ["Face Serum", "Shampoo", "Lip Balm", "Sunscreen", "Moisturizer", "Perfume"],
    "Sports & Outdoors": ["Yoga Mat", "Tent", "Water Bottle", "Backpack", "Dumbbell Set", "Bike Light"],
    "Books": ["Cookbook", "Novel", "Travel Guide", "Biography", "Puzzle Book", "Atlas"],
    "Toys": ["Puzzle", "Building Set", "Plush Bear", "Board Game", "Toy Car", "Art Kit"],
}
REGION_BASE_WEIGHTS = {"Northeast": 0.23, "Southeast": 0.23, "Midwest": 0.19, "Southwest": 0.15, "West": 0.20}
# Monthly seasonality multipliers by calendar month.
SEASONALITY = {1: 0.85, 2: 0.82, 3: 0.95, 4: 0.97, 5: 1.00, 6: 1.02,
               7: 0.95, 8: 0.97, 9: 0.98, 10: 1.05, 11: 1.35, 12: 1.50}
CARRIER_SHARE = {"swiftship": 0.45, "parcelpro": 0.35, "metroexpress": 0.20}
CARRIER_TRANSIT_MEAN_DAYS = {"swiftship": 1.7, "parcelpro": 2.1, "metroexpress": 2.8}


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 20260701
    n_customers: int = 2000
    n_products: int = 250
    n_orders: int = 15000
    start_date: str = "2025-07-01"       # first order date (inclusive)
    end_date: str = "2026-06-30"         # last order date (inclusive)
    as_of: str = "2026-07-03 00:00:00"   # snapshot time, UTC
    customer_history_start: str = "2023-01-01"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def as_of_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of, tz="UTC")


def _money(x) -> np.ndarray:
    """Round to cents. Works on scalars and arrays."""
    return np.round(np.asarray(x, dtype="float64") + 0.0, 2)


def _txn_ids(rng: np.random.Generator, n: int) -> list[str]:
    raw = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    ids = [f"txn_{v:016x}" for v in raw]
    if len(set(ids)) != n:  # astronomically unlikely; keeps the guarantee explicit
        raise RuntimeError("provider_transaction_id collision; change seed")
    return ids


def _gen_customers(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_customers
    regions = rng.choice(sr.REGIONS, size=n, p=[REGION_BASE_WEIGHTS[r] for r in sr.REGIONS])
    segments = rng.choice(sr.SEGMENTS, size=n, p=[0.75, 0.18, 0.07])
    hist_start = pd.Timestamp(cfg.customer_history_start)
    window_start = pd.Timestamp(cfg.start_date)
    window_end = pd.Timestamp(cfg.end_date)
    # 70% signed up before the order window, 30% during it.
    before = rng.random(n) < 0.70
    days_before = (window_start - hist_start).days
    days_during = (window_end - window_start).days
    offsets = np.where(before, rng.integers(0, days_before, n), days_before + rng.integers(0, days_during + 1, n))
    signup = hist_start + pd.to_timedelta(offsets, unit="D")
    created = (signup + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")).tz_localize("UTC")
    ids = [f"CUS{i:06d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "customer_id": ids,
        "email": [f"customer{i:06d}@example.com" for i in range(1, n + 1)],
        "region": regions,
        "country_code": "US",
        "customer_segment": segments,
        "signup_date": signup.date,
        "created_at": created,
    })


def _gen_products(cfg: GenerationConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n = cfg.n_products
    cats = list(CATEGORY_SPEC)
    shares = np.array([CATEGORY_SPEC[c][0] for c in cats])
    category = rng.choice(cats, size=n, p=shares / shares.sum())
    lo = np.array([CATEGORY_SPEC[c][1] for c in category], dtype=float)
    hi = np.array([CATEGORY_SPEC[c][2] for c in category], dtype=float)
    # Log-uniform price, ending in .99
    price = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    list_price = _money(np.floor(price) + 0.99)
    unit_cost = _money(list_price * rng.uniform(0.45, 0.70, n))
    names = [f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS[c])}" for c in category]
    ids = [f"PRD{i:05d}" for i in range(1, n + 1)]
    skus = [f"SKU-{CATEGORY_SPEC[c][4]}-{i:05d}" for i, c in enumerate(category, start=1)]
    products = pd.DataFrame({
        "product_id": ids,
        "sku": skus,
        "product_name": names,
        "category": category,
        "list_price": list_price,
        "unit_cost": unit_cost,
        "is_active": rng.random(n) < 0.92,
        "created_at": pd.Timestamp("2023-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    })
    # Hidden (not exported) attributes that drive behaviour.
    popularity = rng.zipf(1.6, n).astype(float)
    popularity = np.minimum(popularity, 50) * rng.uniform(0.5, 1.5, n)
    popularity = popularity / popularity.sum()
    refund_prop = np.array([CATEGORY_SPEC[c][3] for c in category]) * rng.lognormal(0, 0.35, n)
    defective = rng.choice(n, size=8, replace=False)      # a "bad batch" with elevated returns
    refund_prop[defective] *= 3.5
    refund_prop = np.clip(refund_prop, 0, 0.6)
    return products, popularity, refund_prop


def _order_timestamps(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DatetimeIndex:
    days = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    month_idx = np.asarray((days.year - days[0].year) * 12 + days.month - days[0].month, dtype=float)
    weekday = np.where(np.asarray(days.day_of_week) >= 5, 1.10, 1.0)
    w = np.array([SEASONALITY[m] for m in days.month]) * (1 + 0.012 * month_idx) * weekday
    counts = rng.multinomial(cfg.n_orders, w / w.sum())
    day_of_order = np.repeat(days.values, counts)
    secs = rng.integers(0, 86400, cfg.n_orders)
    ts = pd.DatetimeIndex(day_of_order) + pd.to_timedelta(secs, unit="s")
    return ts.sort_values().tz_localize("UTC")


def _region_weights_for_month(period: pd.Period) -> np.ndarray:
    w = dict(REGION_BASE_WEIGHTS)
    # Planted pattern: Southwest demand falls in May and sharply in June 2026.
    if period == pd.Period("2026-05", "M"):
        w["Southwest"] *= 0.85
    elif period == pd.Period("2026-06", "M"):
        w["Southwest"] *= 0.55
    arr = np.array([w[r] for r in sr.REGIONS])
    return arr / arr.sum()


def generate_clean(cfg: GenerationConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate all seven clean tables. Returns {table_name: DataFrame} in schema column order."""
    cfg = cfg or GenerationConfig()
    rng = np.random.default_rng(cfg.seed)
    as_of = cfg.as_of_ts

    customers = _gen_customers(cfg, rng)
    products, popularity, refund_prop = _gen_products(cfg, rng)

    # ---- choose a customer for each order, respecting signup date and regional mix ----
    order_ts = _order_timestamps(cfg, rng)
    order_dates = order_ts.tz_convert(None).normalize()
    cust_by_region = {}
    for r in sr.REGIONS:
        sub = customers[customers.region == r].sort_values("signup_date")
        cust_by_region[r] = (sub.customer_id.to_numpy(), pd.to_datetime(sub.signup_date).to_numpy())
    periods = order_dates.to_period("M")
    cust_ids = np.empty(cfg.n_orders, dtype=object)
    cust_region = np.empty(cfg.n_orders, dtype=object)
    for i in range(cfg.n_orders):
        region = rng.choice(sr.REGIONS, p=_region_weights_for_month(periods[i]))
        ids, signups = cust_by_region[region]
        eligible = np.searchsorted(signups, order_dates[i].to_datetime64(), side="right")
        if eligible == 0:  # nobody in this region signed up yet; fall back to earliest customer
            eligible = 1
        # Skew toward earlier (loyal) customers: u^1.3 favours lower indices.
        j = int((rng.random() ** 1.3) * eligible)
        cust_ids[i] = ids[min(j, eligible - 1)]
        cust_region[i] = region

    order_ids = np.array([f"ORD{i:07d}" for i in range(1, cfg.n_orders + 1)])
    channels = rng.choice(sr.SALES_CHANNELS, size=cfg.n_orders, p=[0.55, 0.35, 0.10])

    # ---- order items ----
    n_lines = np.minimum(1 + rng.poisson(1.2, cfg.n_orders), 6)
    item_rows = []
    item_counter = 0
    prod_ids = products.product_id.to_numpy()
    prod_price = products.list_price.to_numpy()
    for i in range(cfg.n_orders):
        chosen = rng.choice(len(prod_ids), size=n_lines[i], replace=False, p=popularity)
        for line_no, p_idx in enumerate(chosen, start=1):
            item_counter += 1
            qty = int(rng.choice([1, 2, 3, 4, 5], p=[0.70, 0.20, 0.07, 0.02, 0.01]))
            unit_price = float(prod_price[p_idx])
            gross = round(qty * unit_price, 2)
            disc_pct = rng.choice([0.05, 0.10, 0.15, 0.20, 0.30]) if rng.random() < 0.25 else 0.0
            discount = round(gross * disc_pct, 2)
            item_rows.append((f"ITM{item_counter:08d}", order_ids[i], line_no, prod_ids[p_idx], qty,
                              unit_price, gross, discount, round(gross - discount, 2), int(p_idx)))
    items = pd.DataFrame(item_rows, columns=sr.ORDER_ITEMS.column_names + ["_p_idx"])
    for col in ("unit_price", "gross_amount", "discount_amount", "net_amount"):
        items[col] = _money(items[col])

    agg = items.groupby("order_id", sort=False).agg(items_subtotal=("gross_amount", "sum"),
                                                     discount_amount=("discount_amount", "sum"))
    agg = agg.reindex(order_ids)
    subtotal = _money(agg.items_subtotal.to_numpy())
    discount = _money(agg.discount_amount.to_numpy())
    merch_net = _money(subtotal - discount)
    shipping = np.where(merch_net >= FREE_SHIPPING_THRESHOLD, 0.0, SHIPPING_FEE)
    tax = _money(merch_net * np.array([TAX_RATES[r] for r in cust_region]))
    total = _money(merch_net + shipping + tax)

    # ---- lifecycle simulation ----
    status = np.empty(cfg.n_orders, dtype=object)
    cancelled_ts = [pd.NaT] * cfg.n_orders
    pay_rows, ship_rows, refund_rows = [], [], []
    pay_seq = ship_seq = ref_seq = 0
    txn_pool = iter(_txn_ids(rng, cfg.n_orders * 3))
    captured_payment_of = {}
    shipment_of = {}

    def add_payment(order_idx, ts, method, pstatus, amount, reason=None):
        nonlocal pay_seq
        pay_seq += 1
        pid = f"PAY{pay_seq:07d}"
        pay_rows.append((pid, order_ids[order_idx], next(txn_pool), ts, method, pstatus, amount, reason))
        return pid

    def add_refund(order_idx, item_id, pay_id, ts, amount, reason, rstatus):
        nonlocal ref_seq
        ref_seq += 1
        refund_rows.append((f"REF{ref_seq:06d}", order_ids[order_idx], item_id, pay_id, ts,
                            round(amount, 2), reason, rstatus))

    refund_pending_window = pd.Timedelta(days=5)
    for i in range(cfg.n_orders):
        ots = order_ts[i]
        method = rng.choice(sr.PAYMENT_METHODS, p=[0.50, 0.20, 0.22, 0.08])
        first_try = ots + pd.Timedelta(seconds=int(rng.integers(10, 600)))

        if rng.random() < 0.045:  # ---------------- cancelled ----------------
            status[i] = "cancelled"
            c_ts = ots + pd.Timedelta(minutes=int(rng.integers(30, 24 * 60)))
            cancelled_ts[i] = c_ts
            u = rng.random()
            if u < 0.40:      # payment failed, customer abandoned
                add_payment(i, first_try, method, "failed", float(total[i]), rng.choice(sr.FAILURE_REASONS))
            elif u < 0.75:    # paid, then cancelled -> full order-level refund
                pid = add_payment(i, first_try, method, "captured", float(total[i]))
                r_ts = c_ts + pd.Timedelta(minutes=int(rng.integers(30, 24 * 60)))
                r_status = "pending" if r_ts > as_of - pd.Timedelta(days=3) else "completed"
                add_refund(i, None, pid, r_ts, float(total[i]), "order_cancelled", r_status)
            # else: cancelled before any payment attempt -> no payment row
            continue

        if ots > as_of - pd.Timedelta(days=4) and rng.random() < 0.20:  # -------- pending payment --------
            status[i] = "pending_payment"
            add_payment(i, first_try, method, "pending", float(total[i]))
            continue

        # ---------------- booked (payment captured) ----------------
        cap_ts = first_try
        if rng.random() < 0.07:  # one failed attempt first
            add_payment(i, first_try, rng.choice(["credit_card", "debit_card"]), "failed",
                        float(total[i]), rng.choice(sr.FAILURE_REASONS))
            cap_ts = first_try + pd.Timedelta(seconds=int(rng.integers(60, 1800)))
        cap_id = add_payment(i, cap_ts, method, "captured", float(total[i]))
        captured_payment_of[i] = cap_id

        ship_ts = ots + pd.Timedelta(hours=float(rng.gamma(2.0, 12.0)) + 2)
        if ship_ts > as_of:
            status[i] = "processing"
            continue
        carrier = rng.choice(list(CARRIER_SHARE), p=list(CARRIER_SHARE.values()))
        peak = 1.35 if ots.month in (11, 12) else 1.0
        transit_days = float(rng.gamma(4.0, CARRIER_TRANSIT_MEAN_DAYS[carrier] * peak / 4.0)) + 0.3
        deliv_ts = ship_ts + pd.Timedelta(days=transit_days)
        promised = (ots.tz_convert(None).normalize() + pd.Timedelta(days=PROMISE_DAYS)).date()
        ship_seq += 1
        sid = f"SHP{ship_seq:07d}"
        if deliv_ts > as_of:
            status[i] = "shipped"
            ship_rows.append((sid, order_ids[i], carrier, "in_transit", ship_ts, promised, pd.NaT))
            continue
        status[i] = "delivered"
        ship_rows.append((sid, order_ids[i], carrier, "delivered", ship_ts, promised, deliv_ts))
        shipment_of[i] = (deliv_ts, deliv_ts.date() > promised)

    # ---- item-level refunds on delivered orders ----
    items_by_order = items.groupby("order_id", sort=False)
    order_pos = {oid: k for k, oid in enumerate(order_ids)}
    reasons_regular = ["damaged", "wrong_item", "not_as_described", "changed_mind"]
    for oid, grp in items_by_order:
        k = order_pos[oid]
        if status[k] != "delivered":
            continue
        deliv_ts, was_late = shipment_of[k]
        for row in grp.itertuples(index=False):
            p = 0.75 * refund_prop[row._9] * (1.6 if was_late else 1.0)
            if rng.random() >= p:
                continue
            r_ts = deliv_ts + pd.Timedelta(hours=float(rng.uniform(12, 25 * 24)))
            if r_ts > as_of:
                continue  # customer has not requested it yet as of the snapshot
            partial = rng.random() < 0.30
            amount = round(row.net_amount * rng.uniform(0.2, 0.6), 2) if partial else row.net_amount
            amount = max(amount, 0.01)
            reason = "late_delivery" if (was_late and rng.random() < 0.5) else rng.choice(reasons_regular)
            r_status = "pending" if r_ts > as_of - refund_pending_window else "completed"
            add_refund(k, row.order_item_id, captured_payment_of[k], r_ts, amount, reason, r_status)

    # ---- assemble frames in schema order ----
    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": cust_ids.astype(str),
        "order_timestamp": order_ts,
        "order_date": order_dates.date,
        "order_status": status.astype(str),
        "sales_channel": channels,
        "currency_code": "USD",
        "items_subtotal": subtotal,
        "discount_amount": discount,
        "shipping_amount": _money(shipping),
        "tax_amount": tax,
        "order_total": total,
        "cancelled_timestamp": pd.to_datetime(pd.Series(cancelled_ts), utc=True),
    })
    payments = pd.DataFrame(pay_rows, columns=sr.PAYMENTS.column_names)
    shipments = pd.DataFrame(ship_rows, columns=sr.SHIPMENTS.column_names)
    refunds = pd.DataFrame(refund_rows, columns=sr.REFUNDS.column_names)
    refunds = refunds.sort_values(["refund_timestamp", "refund_id"]).reset_index(drop=True)
    # Re-number refunds chronologically so IDs read naturally.
    refunds["refund_id"] = [f"REF{i:06d}" for i in range(1, len(refunds) + 1)]

    tables = {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": items.drop(columns=["_p_idx"]),
        "payments": payments,
        "shipments": shipments,
        "refunds": refunds,
    }
    return {name: normalize_dtypes(name, df) for name, df in tables.items()}


def normalize_dtypes(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame to canonical in-memory dtypes for its schema (order + types)."""
    t = sr.get_table(table_name)
    out = df.loc[:, t.column_names].copy()
    for c in t.columns:
        s = out[c.name]
        if c.bq_type == "STRING":
            out[c.name] = s.astype("object").where(s.notna(), None)
        elif c.bq_type == "INT64":
            out[c.name] = s.astype("int64")
        elif c.bq_type == "NUMERIC":
            out[c.name] = _money(s.astype("float64"))
        elif c.bq_type == "BOOL":
            out[c.name] = s.astype("bool")
        elif c.bq_type == "DATE":
            out[c.name] = pd.to_datetime(s).dt.date
        elif c.bq_type == "TIMESTAMP":
            out[c.name] = pd.to_datetime(s, utc=True).dt.floor("s")
    return out.reset_index(drop=True)
