"""Independently computed expected results (gold answers) for analytical test cases.

Computed with deterministic pandas code in analytics_agent.metrics, never by an
LLM. Each case records the question, the business definition, required SQL
properties, the expected result, a numeric tolerance and expected reviewer
behaviour. "Known wrong answers" are computed too, so Stage 6 reviewer tests
can check that classic mistakes are caught.
"""
from __future__ import annotations

import math

import pandas as pd

from analytics_agent import metrics as m
from analytics_agent import schema_registry as sr

WINDOW = ("2025-07-01", "2026-06-30")
Q1_2026 = ("2026-01-01", "2026-03-31")
Q2_2026 = ("2026-04-01", "2026-06-30")
H1_2026 = ("2026-01-01", "2026-06-30")
MONEY_TOL = 0.01
RATE_TOL = 0.0001


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and math.isnan(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            clean[str(k)] = v
        out.append(clean)
    return out


def _case(cid, question, category, definition, grain, filters, sql_props, result, tolerance,
          reviewer, notes=""):
    return {"test_id": cid, "question": question, "category": category,
            "expected_business_definition": definition, "grain": grain, "filters": filters,
            "expected_sql_properties": sql_props, "expected_result": result, "tolerance": tolerance,
            "expected_reviewer_behavior": reviewer, "expected_safety_behavior": "read-only SELECT on approved tables; passes SQL validator",
            "notes": notes}


def build_expected_results(t: dict) -> dict:
    s, e = WINDOW
    cases = []

    monthly = m.net_sales(t, *H1_2026, group_by=["month"])
    cases.append(_case(
        "G01", "What was our monthly net sales over the last six months?", "monthly_trend",
        "Net sales = SUM(order_items.net_amount) for booked orders - SUM(completed item-level refunds), attributed to order month.",
        "order item", {"order_date": list(H1_2026), "last_six_months_relative_to": "as_of 2026-07-03 -> Jan-Jun 2026 (complete months)"},
        ["filters order_status IN booked statuses", "aggregates refunds separately from items (no fan-out)",
         "excludes pending refunds", "groups by FORMAT_DATE('%Y-%m', order_date)", "excludes tax and shipping"],
        _records(monthly), MONEY_TOL, "approve",
        "Agent must state the resolved date range because 'last six months' is relative."))

    by_cat = m.net_sales(t, s, e, group_by=["category"]).sort_values("net_sales", ascending=False)
    cases.append(_case(
        "G02", "Which product categories generate the most sales?", "multi_table_join",
        "Net sales by products.category, full window, ranked descending.", "order item",
        {"order_date": list(WINDOW)},
        ["joins order_items -> orders -> products", "booked statuses only", "refunds aggregated separately",
         "ORDER BY net_sales DESC"], _records(by_cat), MONEY_TOL, "approve",
        "Ambiguity note: 'sales' resolved to the documented net sales metric; agent should say so."))

    aov = m.average_order_value(t, s, e, group_by=["region"]).sort_values("region")
    cases.append(_case(
        "G03", "What is the average order value by region?", "aov",
        "AOV = SUM(order_items.net_amount)/COUNT(DISTINCT order_id), booked orders, by customers.region.", "order",
        {"order_date": list(WINDOW)},
        ["COUNT(DISTINCT order_id) denominator", "does not use AVG(order_total)", "excludes tax/shipping",
         "joins customers for region"], _records(aov), MONEY_TOL, "approve"))

    prr = m.product_refund_rates(t, s, e, min_units=30, top_n=10)
    cases.append(_case(
        "G04", "Which products have the highest refund rates?", "refund_adjusted",
        "Refund rate = completed item refunds / merchandise sales per product; products with >= 30 units sold; top 10.",
        "order item", {"order_date": list(WINDOW), "min_units": 30},
        ["value-based rate with documented denominator", "minimum-volume threshold disclosed",
         "completed refunds only"], _records(prr), RATE_TOL, "approve",
        "Without a volume threshold, tiny-volume products dominate; the agent should disclose the threshold."))

    dp = m.delivery_performance(t, s, e)
    cases.append(_case(
        "G05", "What percentage of orders were delivered late?", "delivery",
        "Late rate = delivered shipments with DATE(delivered_timestamp) > promised_delivery_date / delivered shipments.",
        "shipment", {"order_date": list(WINDOW)},
        ["denominator excludes in_transit shipments", "compares DATE(delivered_timestamp) to promised_delivery_date"],
        _records(dp)[0], RATE_TOL, "approve",
        "Question says 'orders'; one shipment per order in v1 so shipment and order rates coincide - agent should note the grain."))

    reg = m.net_sales(t, "2026-05-01", "2026-06-30", group_by=["region", "month"])
    reg = m.mom_change(reg, "net_sales", by="region")
    june = reg[reg.month == "2026-06"].sort_values("mom_pct")
    cases.append(_case(
        "G06", "Which regions experienced the largest month-over-month sales decline?", "mom_change",
        "Net sales by region, June 2026 vs May 2026 (latest complete month); MoM % = (Jun - May)/May*100.",
        "order item", {"order_date": ["2026-05-01", "2026-06-30"]},
        ["computes both months then compares (LAG or self-join)", "uses latest COMPLETE month",
         "orders regions ascending by mom_pct"],
        {"largest_decline_region": june.iloc[0].region, "ranking": _records(june[["region", "previous", "net_sales", "mom_pct"]])},
        RATE_TOL, "approve"))

    rec = m.reconciliation(t, s, e)
    cases.append(_case(
        "G07", "Why do total payments differ from net sales?", "reconciliation",
        "Captured payments = merch sales + shipping + tax + captured on cancelled orders; net sales = merch sales - completed item refunds.",
        "mixed (payment, order, item, refund)", {"order_date": list(WINDOW)},
        ["aggregates each table separately", "explains each bridge component", "bridge_difference = 0"],
        rec, MONEY_TOL, "approve", "Explanatory question; the answer must cite each component with its value."))

    cases.append(_case(
        "G08", "How many orders do we have in each status?", "simple_aggregation",
        "COUNT(*) of orders grouped by order_status, full snapshot.", "order", {},
        ["GROUP BY order_status", "no joins"], m.order_status_counts(t), 0, "approve"))

    items = m._enriched_items(t, s, e)
    cases.append(_case(
        "G09", "What were gross item sales, discounts and merchandise sales after discounts?", "simple_aggregation",
        "SUM(gross_amount), SUM(discount_amount), SUM(net_amount) of order_items on booked orders.", "order item",
        {"order_date": list(WINDOW)}, ["sums item columns, not order header columns"],
        {"gross_item_sales": round(float(items.gross_amount.sum()), 2),
         "discounts": round(float(items.discount_amount.sum()), 2),
         "merch_sales": round(float(items.net_amount.sum()), 2)}, MONEY_TOL, "approve"))

    full_monthly = m.mom_change(m.net_sales(t, s, e, group_by=["month"]), "net_sales")
    cases.append(_case(
        "G10", "Show month-over-month change in net sales.", "mom_change",
        "MoM % of total net sales per month; first month has no prior value (NULL).", "order item",
        {"order_date": list(WINDOW)}, ["LAG over month", "NULL (not 0 or error) for first month"],
        _records(full_monthly[["month", "net_sales", "previous", "mom_pct"]]), RATE_TOL, "approve"))

    cases.append(_case(
        "G11", "What share of orders had a refund?", "refund_adjusted",
        "Booked orders with >= 1 completed item-level refund / booked orders.", "order",
        {"order_date": list(WINDOW)}, ["COUNT(DISTINCT order_id) numerator and denominator", "excludes cancellation refunds"],
        m.order_refund_rate(t, s, e), RATE_TOL, "approve"))

    cases.append(_case(
        "G12", "How does late delivery rate compare across carriers?", "delivery",
        "Late rate by shipments.carrier (delivered shipments only).", "shipment", {"order_date": list(WINDOW)},
        ["GROUP BY carrier", "delivered-only denominator"],
        _records(m.delivery_performance(t, s, e, group_by=["carrier"])), RATE_TOL, "approve"))

    cases.append(_case(
        "G13", "What were net sales by customer segment in Q1 2026?", "filtering",
        "Net sales by customers.customer_segment, order_date in Q1 2026.", "order item", {"order_date": list(Q1_2026)},
        ["date filter on order_date", "joins customers"],
        _records(m.net_sales(t, *Q1_2026, group_by=["segment"])), MONEY_TOL, "approve"))

    cases.append(_case(
        "G14", "What is our order cancellation rate?", "simple_aggregation",
        "Cancelled orders / all orders placed in the window.", "order", {"order_date": list(WINDOW)},
        ["denominator = all orders, not booked orders"], m.cancellation_rate(t, s, e), RATE_TOL, "approve"))

    cases.append(_case(
        "G15", "Net sales by sales channel last quarter?", "filtering",
        "Net sales by orders.sales_channel, Q2 2026 (last complete quarter relative to as_of).", "order item",
        {"order_date": list(Q2_2026)}, ["date filter", "GROUP BY sales_channel"],
        _records(m.net_sales(t, *Q2_2026, group_by=["channel"])), MONEY_TOL, "approve",
        "Relative period; agent must state Q2 2026."))

    cases.append(_case(
        "G16", "How many customers placed at least one order in Q2 2026?", "simple_aggregation",
        "COUNT(DISTINCT customer_id) over booked orders in Q2 2026.", "customer", {"order_date": list(Q2_2026)},
        ["COUNT(DISTINCT customer_id)", "booked statuses"], {"active_customers": m.active_customers(t, *Q2_2026)},
        0, "approve"))

    cases.append(_case(
        "G17", "How many items does a typical order contain?", "aov",
        "Average order lines and average units per booked order.", "order", {"order_date": list(WINDOW)},
        ["COUNT(*) lines / COUNT(DISTINCT order_id)"], m.basket_size(t, s, e), RATE_TOL, "approve"))

    cases.append(_case(
        "G18", "What are completed refunds by reason?", "refund_adjusted",
        "SUM(refund_amount) of completed refunds by refund_reason, by refund date in the window.", "refund",
        {"refund_date": list(WINDOW)}, ["refund_status = 'completed'", "includes order_cancelled reason as its own row"],
        _records(m.refunds_by_reason(t, s, e)), MONEY_TOL, "approve"))

    cases.append(_case(
        "G19", "What is our payment failure rate?", "simple_aggregation",
        "Failed payment attempts / all payment attempts, by payment date.", "payment attempt", {"payment_date": list(WINDOW)},
        ["payment grain (not order grain)"], m.payment_failure_rate(t, s, e), RATE_TOL, "approve"))

    cases.append(_case(
        "G20", "What were captured payments by month in H1 2026?", "monthly_trend",
        "SUM(amount) of captured payments by payment month. Cash view: includes tax, shipping and later-cancelled orders.",
        "payment", {"payment_date": list(H1_2026)}, ["payment_status = 'captured'", "groups by payment month, not order month"],
        _records(m.captured_payments_by_month(t, *H1_2026)), MONEY_TOL, "approve",
        "Must not be presented as net sales."))

    # ------------------------------------------------------------ hard cases (traps in wording) ----
    JUNE_2026 = ("2026-06-01", "2026-06-30")
    aov_q2 = m.average_order_value(t, *Q2_2026, group_by=["region"])
    cases.append(_case(
        "G21", "Excluding tax and shipping, what was the average order value by region last quarter?", "hard_aov",
        "AOV = SUM(order_items.net_amount)/COUNT(DISTINCT order_id) over booked orders, by region, Q2 2026.", "order",
        {"order_date": list(Q2_2026)}, ["relative period 'last quarter' -> Q2 2026", "does not use AVG(order_total)"],
        _records(aov_q2), MONEY_TOL, "approve", "Trap: tax/shipping wording and a relative period."))

    cases.append(_case(
        "G22", "In Q1 2026, how did net sales split across customer segments?", "hard_filtering",
        "Net sales by customers.customer_segment, order_date in Q1 2026.", "order item", {"order_date": list(Q1_2026)},
        ["date filter on order_date", "joins customers"], _records(m.net_sales(t, *Q1_2026, group_by=["segment"])),
        MONEY_TOL, "approve", "Trap: 'split across' wording instead of 'by'."))

    by_channel = m.net_sales(t, *Q2_2026, group_by=["channel"]).sort_values("net_sales", ascending=False)
    cases.append(_case(
        "G23", "Which sales channel brought in the most net sales last quarter?", "hard_ranking",
        "Top orders.sales_channel by net sales in Q2 2026.", "order item", {"order_date": list(Q2_2026)},
        ["GROUP BY sales_channel", "ORDER BY net_sales DESC"],
        {"top_channel": str(by_channel.iloc[0]["channel"]), "ranking": _records(by_channel)}, MONEY_TOL, "approve",
        "Trap: a ranking question with a relative period."))

    cases.append(_case(
        "G24", "Last month, what share of delivered orders arrived late for each carrier?", "hard_delivery",
        "Late rate by carrier, delivered shipments only, orders placed in June 2026.", "shipment",
        {"order_date": list(JUNE_2026)}, ["delivered-only denominator", "'last month' -> June 2026"],
        _records(m.delivery_performance(t, *JUNE_2026, group_by=["carrier"])), RATE_TOL, "approve",
        "Trap: in-transit shipments must not be in the denominator."))

    by_category = m.net_sales(t, *JUNE_2026, group_by=["category"]).sort_values("net_sales", ascending=False)
    cases.append(_case(
        "G25", "Which product category had the highest net sales in June 2026?", "hard_ranking",
        "Top products.category by net sales, June 2026.", "order item", {"order_date": list(JUNE_2026)},
        ["joins products for category", "ORDER BY net_sales DESC"],
        {"top_category": str(by_category.iloc[0]["category"]), "ranking": _records(by_category)}, MONEY_TOL, "approve"))

    customers = t["customers"]
    signup = pd.to_datetime(customers["signup_date"])
    h1 = customers[(signup >= H1_2026[0]) & (signup <= H1_2026[1])]
    signups_by_region = (h1.assign(region=h1["region"].fillna("Unknown")).groupby("region")["customer_id"].nunique()
                         .rename("signups").reset_index().sort_values("signups", ascending=False))
    cases.append(_case(
        "G26", "How many new customers signed up in each region during the first half of 2026?", "hard_customers",
        "COUNT(DISTINCT customer_id) by region for signup_date in H1 2026.", "customer", {"signup_date": list(H1_2026)},
        ["filters signup_date, not order_date", "COUNT(DISTINCT customer_id)"], _records(signups_by_region), 0,
        "approve", "Trap: signups are dated by signup_date, not by orders."))

    distinct_total = customers["customer_id"].nunique()
    shares = (customers.assign(segment=customers["customer_segment"].fillna("Unknown")).groupby("segment")["customer_id"]
              .nunique().rename("customers").reset_index())
    shares["pct_of_customers"] = (100 * shares["customers"] / distinct_total).round(2)
    cases.append(_case(
        "G27", "What percentage of customers are in each segment?", "hard_customers",
        "COUNT(DISTINCT customer_id) per segment / all distinct customers x 100.", "customer", {},
        ["distinct customers as the denominator", "shares add up to 100"], _records(shares), 0.011, "approve"))

    cases.append(_case(
        "G28", "How many distinct customers placed at least one booked order in the first half of 2026?", "hard_customers",
        "COUNT(DISTINCT customer_id) over booked orders in H1 2026.", "customer", {"order_date": list(H1_2026)},
        ["COUNT(DISTINCT customer_id)", "booked statuses"], {"active_customers": m.active_customers(t, *H1_2026)},
        0, "approve", "Trap: customers who ordered, not customers in the table."))

    cases.append(_case(
        "G29", "What were net sales by product category in 2027?", "hard_empty",
        "Net sales by category for 2027: the data ends in June 2026, so the correct result is empty.", "order item",
        {"order_date": ["2027-01-01", "2027-12-31"]}, ["keeps the requested period", "returns no rows"], [], 0,
        "approve", "Trap: the agent must report no data, never invent values."))

    # ------------------------------------------------------------ known wrong answers ----
    booked = m.booked_orders(t, s, e)
    fanout = t["order_items"].merge(booked[["order_id", "order_total"]], on="order_id").order_total.sum()
    all_ship = t["shipments"].merge(t["orders"][["order_id", "order_date"]], on="order_id")
    all_ship = all_ship[m._in_window(all_ship.order_date, s, e)]
    late_wrong = (all_ship.delivered_timestamp.notna() &
                  (pd.to_datetime(all_ship.delivered_timestamp).dt.date > all_ship.promised_delivery_date)).sum() / len(all_ship)
    r_all = t["refunds"][t["refunds"].order_item_id.notna() & t["refunds"].order_id.isin(booked.order_id)]
    wrong = [
        {"id": "W01", "related_case": "G09", "mistake": "SUM(orders.order_total) after joining order_items (join fan-out)",
         "wrong_value": round(float(fanout), 2), "correct_reference": "sum each order once",
         "correct_value": round(float(booked.order_total.sum()), 2), "expected_reviewer_behavior": "reject: fan-out"},
        {"id": "W02", "related_case": "G07", "mistake": "Reporting captured payments as net sales",
         "wrong_value": rec["captured_payments"], "correct_value": rec["net_sales"],
         "expected_reviewer_behavior": "reject: metric definition mismatch"},
        {"id": "W03", "related_case": "G05", "mistake": "Late rate with all shipments (incl. in_transit) as denominator",
         "wrong_value": round(float(late_wrong), 4), "correct_value": _records(dp)[0]["late_rate"],
         "expected_reviewer_behavior": "reject: wrong denominator"},
        {"id": "W04", "related_case": "G07", "mistake": "Net sales subtracting pending as well as completed refunds",
         "wrong_value": round(float(items.net_amount.sum() - r_all.refund_amount.sum()), 2),
         "correct_value": rec["net_sales"], "expected_reviewer_behavior": "reject: pending refunds included"},
        {"id": "W05", "related_case": "G03", "mistake": "AOV as AVG(order_total) (includes tax and shipping)",
         "wrong_value": round(float(booked.order_total.mean()), 2),
         "correct_value": round(float(items.net_amount.sum() / items.order_id.nunique()), 2),
         "expected_reviewer_behavior": "reject or revise: metric definition mismatch"},
    ]
    return {"window": list(WINDOW), "as_of": "2026-07-03T00:00:00Z", "cases": cases, "known_wrong_answers": wrong}