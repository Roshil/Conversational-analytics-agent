"""A tiny hand-built dataset with answers computed by hand.

Used by tests/unit/test_metrics.py and notebook 01 to check metric logic
independently of the random generator. Expected answers (window Jan-Feb 2026):
net sales Jan 53.00, Feb 20.00; AOV 26.00; late rate 0.5; captured 115.31.
"""
import pandas as pd

from analytics_agent.data_generation.generator import normalize_dtypes


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def build_tiny_dataset() -> dict:
    """Five orders covering booked, cancelled-and-refunded, pending-payment and in-transit cases."""
    customers = pd.DataFrame({
        "customer_id": ["C1", "C2"], "email": ["a@example.com", "b@example.com"],
        "region": ["West", "Midwest"], "country_code": "US", "customer_segment": "consumer",
        "signup_date": [pd.Timestamp("2025-01-01").date()] * 2, "created_at": [_ts("2025-01-01")] * 2})
    products = pd.DataFrame({
        "product_id": ["P1", "P2"], "sku": ["S1", "S2"], "product_name": ["Book", "Toy"],
        "category": ["Books", "Toys"], "list_price": [10.0, 20.0], "unit_cost": [5.0, 9.0],
        "is_active": True, "created_at": [_ts("2025-01-01")] * 2})
    o = [  # id, cust, ts, status, subtotal, disc, ship, tax, total, cancelled_ts
        ("O1", "C1", "2026-01-10 10:00", "delivered", 40.0, 2.0, 6.99, 3.04, 48.03, None),
        ("O2", "C2", "2026-01-20 10:00", "delivered", 20.0, 0.0, 6.99, 1.20, 28.19, None),
        ("O3", "C1", "2026-02-05 10:00", "cancelled", 10.0, 0.0, 0.00, 0.70, 10.70, "2026-02-05 12:00"),
        ("O4", "C2", "2026-02-10 10:00", "pending_payment", 10.0, 0.0, 6.99, 0.60, 17.59, None),
        ("O5", "C2", "2026-02-15 10:00", "shipped", 20.0, 0.0, 6.99, 1.40, 28.39, None),
    ]
    orders = pd.DataFrame({
        "order_id": [r[0] for r in o], "customer_id": [r[1] for r in o],
        "order_timestamp": [_ts(r[2]) for r in o], "order_date": [_ts(r[2]).date() for r in o],
        "order_status": [r[3] for r in o], "sales_channel": "web", "currency_code": "USD",
        "items_subtotal": [r[4] for r in o], "discount_amount": [r[5] for r in o],
        "shipping_amount": [r[6] for r in o], "tax_amount": [r[7] for r in o], "order_total": [r[8] for r in o],
        "cancelled_timestamp": [_ts(r[9]) if r[9] else pd.NaT for r in o]})
    items = pd.DataFrame([
        ("I1", "O1", 1, "P1", 2, 10.0, 20.0, 0.0, 20.0),
        ("I2", "O1", 2, "P2", 1, 20.0, 20.0, 2.0, 18.0),
        ("I3", "O2", 1, "P2", 1, 20.0, 20.0, 0.0, 20.0),
        ("I4", "O3", 1, "P1", 1, 10.0, 10.0, 0.0, 10.0),
        ("I5", "O4", 1, "P1", 1, 10.0, 10.0, 0.0, 10.0),
        ("I6", "O5", 1, "P2", 1, 20.0, 20.0, 0.0, 20.0),
    ], columns=["order_item_id", "order_id", "line_number", "product_id", "quantity", "unit_price",
                "gross_amount", "discount_amount", "net_amount"])
    payments = pd.DataFrame([
        ("Y1", "O1", "t1", _ts("2026-01-10 10:01"), "credit_card", "captured", 48.03, None),
        ("Y2", "O2", "t2", _ts("2026-01-20 10:01"), "credit_card", "failed", 28.19, "card_declined"),
        ("Y3", "O2", "t3", _ts("2026-01-20 10:05"), "paypal", "captured", 28.19, None),
        ("Y4", "O3", "t4", _ts("2026-02-05 10:01"), "credit_card", "captured", 10.70, None),
        ("Y5", "O4", "t5", _ts("2026-02-10 10:01"), "credit_card", "pending", 17.59, None),
        ("Y6", "O5", "t6", _ts("2026-02-15 10:01"), "credit_card", "captured", 28.39, None),
    ], columns=["payment_id", "order_id", "provider_transaction_id", "payment_timestamp", "payment_method",
                "payment_status", "amount", "failure_reason"])
    shipments = pd.DataFrame([
        ("S1", "O1", "swiftship", "delivered", _ts("2026-01-11"), pd.Timestamp("2026-01-15").date(), _ts("2026-01-15 18:00")),
        ("S2", "O2", "parcelpro", "delivered", _ts("2026-01-21"), pd.Timestamp("2026-01-25").date(), _ts("2026-01-26 09:00")),
        ("S3", "O5", "parcelpro", "in_transit", _ts("2026-02-16"), pd.Timestamp("2026-02-20").date(), pd.NaT),
    ], columns=["shipment_id", "order_id", "carrier", "shipment_status", "shipped_timestamp",
                "promised_delivery_date", "delivered_timestamp"])
    refunds = pd.DataFrame([
        ("R1", "O1", "I1", "Y1", _ts("2026-01-20"), 5.00, "damaged", "completed"),
        ("R2", "O2", "I3", "Y3", _ts("2026-02-01"), 20.00, "changed_mind", "pending"),
        ("R3", "O3", None, "Y4", _ts("2026-02-05 13:00"), 10.70, "order_cancelled", "completed"),
    ], columns=["refund_id", "order_id", "order_item_id", "payment_id", "refund_timestamp", "refund_amount",
                "refund_reason", "refund_status"])
    raw = dict(customers=customers, products=products, orders=orders, order_items=items,
               payments=payments, shipments=shipments, refunds=refunds)
    return {k: normalize_dtypes(k, v) for k, v in raw.items()}
