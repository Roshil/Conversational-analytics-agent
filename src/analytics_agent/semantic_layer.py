"""Semantic layer: business metric definitions, join rules and canonical BigQuery SQL.

This is the single source of business logic for the conversational agent. It is
inserted into the SQL-generation prompt (so Gemini follows the documented
definitions instead of guessing) and used by the SQL & Analytics Reviewer.
Every canonical query is tested against the independently computed gold
answers (data/manifests/expected_results_clean.json).

Placeholders: {orders}, {order_items}, ... are replaced by `project.dataset.table`;
@start_date / @end_date are DATE query parameters (inclusive).
"""
from __future__ import annotations

from dataclasses import dataclass

from analytics_agent import schema_registry as sr

BOOKED = "('processing', 'shipped', 'delivered')"
AS_OF_DATE = "2026-07-03"             # snapshot date of the dummy data; relative periods resolve against it
DEFAULT_WINDOW = ("2025-07-01", "2026-06-30")


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    definition: str
    grain: str
    canonical_sql: str
    example_question: str

    def render(self, refs: dict) -> str:
        return self.canonical_sql.format(**refs)


_LINES = """lines AS (
  SELECT li.order_item_id, li.net_amount, li.quantity, li.product_id, o.order_id, o.order_date, o.customer_id, o.sales_channel
  FROM {{order_items}} li
  JOIN {{orders}} o ON o.order_id = li.order_id
  WHERE o.order_status IN {booked}
    AND o.order_date BETWEEN @start_date AND @end_date
),
line_refunds AS (
  SELECT order_item_id, SUM(refund_amount) AS refunded
  FROM {{refunds}}
  WHERE refund_status = 'completed' AND order_item_id IS NOT NULL
  GROUP BY order_item_id
)""".format(booked=BOOKED)

METRICS: tuple[Metric, ...] = (
    Metric("net_sales", "Net sales",
           "SUM(order_items.net_amount) of booked orders (status processing/shipped/delivered) minus completed item-level "
           "refunds (refund_status='completed' AND order_item_id IS NOT NULL). Excludes tax, shipping, cancelled and "
           "pending-payment orders and pending refunds. Attributed to the ORDER date.",
           "order item",
           "WITH " + _LINES + """
SELECT FORMAT_DATE('%Y-%m', lines.order_date) AS month,
       ROUND(SUM(lines.net_amount), 2) AS merch_sales,
       ROUND(SUM(IFNULL(line_refunds.refunded, 0)), 2) AS refunds,
       ROUND(SUM(lines.net_amount) - SUM(IFNULL(line_refunds.refunded, 0)), 2) AS net_sales
FROM lines
LEFT JOIN line_refunds ON line_refunds.order_item_id = lines.order_item_id
GROUP BY month
ORDER BY month""", "What was monthly net sales over the last six months?"),
    Metric("net_sales_by_dimension", "Net sales by a dimension",
           "Same as net_sales, grouped by a dimension: products.category (join products on product_id), customers.region or "
           "customers.customer_segment (join customers on customer_id), or orders.sales_channel.",
           "order item",
           "WITH " + _LINES + """
SELECT p.category,
       ROUND(SUM(lines.net_amount) - SUM(IFNULL(line_refunds.refunded, 0)), 2) AS net_sales
FROM lines
JOIN {products} p ON p.product_id = lines.product_id
LEFT JOIN line_refunds ON line_refunds.order_item_id = lines.order_item_id
GROUP BY p.category
ORDER BY net_sales DESC""", "Which product categories generate the most sales?"),
    Metric("average_order_value", "Average order value (AOV)",
           "SUM(order_items.net_amount) / COUNT(DISTINCT order_id) over booked orders. Merchandise after discounts, before "
           "refunds, excluding tax and shipping. Never AVG(orders.order_total). The canonical query is by region; when the "
           "question asks for one overall figure (no breakdown), drop the dimension and GROUP BY and return a single row.",
           "order",
           f"""SELECT c.region,
       ROUND(SUM(li.net_amount) / COUNT(DISTINCT o.order_id), 2) AS average_order_value,
       COUNT(DISTINCT o.order_id) AS orders
FROM {{orders}} o
JOIN {{order_items}} li ON li.order_id = o.order_id
JOIN {{customers}} c ON c.customer_id = o.customer_id
WHERE o.order_status IN {BOOKED}
  AND o.order_date BETWEEN @start_date AND @end_date
GROUP BY c.region
ORDER BY c.region""", "What is the average order value by region?"),
    Metric("product_refund_rate", "Product refund rate",
           "Completed item refunds / merchandise sales per product, booked orders, only products with >= 30 units sold "
           "(disclose the threshold).",
           "order item",
           "WITH " + _LINES + """,
per_product AS (
  SELECT p.product_id, p.product_name, p.category,
         SUM(lines.quantity) AS units,
         SUM(lines.net_amount) AS merch_sales,
         SUM(IFNULL(line_refunds.refunded, 0)) AS refunds
  FROM lines
  JOIN {products} p ON p.product_id = lines.product_id
  LEFT JOIN line_refunds ON line_refunds.order_item_id = lines.order_item_id
  GROUP BY p.product_id, p.product_name, p.category
)
SELECT product_id, product_name, category, units, ROUND(merch_sales, 2) AS merch_sales, ROUND(refunds, 2) AS refunds,
       ROUND(SAFE_DIVIDE(refunds, merch_sales), 4) AS refund_rate
FROM per_product
WHERE units >= 30
ORDER BY refund_rate DESC, product_id
LIMIT 10""", "Which products have the highest refund rates?"),
    Metric("late_delivery_rate", "Late delivery rate",
           "Delivered shipments with DATE(delivered_timestamp) > promised_delivery_date / delivered shipments that have a "
           "delivered_timestamp. In-transit shipments are excluded from the denominator. Window on the order_date.",
           "shipment",
           """SELECT COUNT(*) AS delivered_shipments,
       COUNTIF(DATE(s.delivered_timestamp) > s.promised_delivery_date) AS late_shipments,
       ROUND(SAFE_DIVIDE(COUNTIF(DATE(s.delivered_timestamp) > s.promised_delivery_date), COUNT(*)), 4) AS late_rate
FROM {shipments} s
JOIN {orders} o ON o.order_id = s.order_id
WHERE s.shipment_status = 'delivered' AND s.delivered_timestamp IS NOT NULL
  AND o.order_date BETWEEN @start_date AND @end_date""", "What percentage of orders were delivered late?"),
    Metric("captured_payments", "Total payments received",
           "SUM(payments.amount) WHERE payment_status = 'captured', by payment date. A cash view: includes tax, shipping and "
           "payments on later-cancelled orders. NOT net sales.",
           "payment",
           """SELECT FORMAT_TIMESTAMP('%Y-%m', payment_timestamp) AS month,
       ROUND(SUM(amount), 2) AS captured_payments
FROM {payments}
WHERE payment_status = 'captured'
  AND DATE(payment_timestamp) BETWEEN @start_date AND @end_date
GROUP BY month
ORDER BY month""", "What were captured payments by month?"),
    Metric("order_refund_rate", "Order refund rate",
           "Booked orders with at least one completed item-level refund / booked orders.",
           "order",
           f"""WITH booked AS (
  SELECT order_id FROM {{orders}}
  WHERE order_status IN {BOOKED} AND order_date BETWEEN @start_date AND @end_date
),
refunded AS (
  SELECT DISTINCT order_id FROM {{refunds}}
  WHERE refund_status = 'completed' AND order_item_id IS NOT NULL
)
SELECT COUNT(*) AS booked_orders,
       COUNTIF(refunded.order_id IS NOT NULL) AS refunded_orders,
       ROUND(SAFE_DIVIDE(COUNTIF(refunded.order_id IS NOT NULL), COUNT(*)), 4) AS order_refund_rate
FROM booked
LEFT JOIN refunded ON refunded.order_id = booked.order_id""", "What share of orders had a refund?"),
    Metric("cancellation_rate", "Cancellation rate",
           "Cancelled orders / all orders placed in the window (all statuses in the denominator).",
           "order",
           """SELECT COUNT(*) AS orders,
       COUNTIF(order_status = 'cancelled') AS cancelled,
       ROUND(SAFE_DIVIDE(COUNTIF(order_status = 'cancelled'), COUNT(*)), 4) AS cancellation_rate
FROM {orders}
WHERE order_date BETWEEN @start_date AND @end_date""", "What is our order cancellation rate?"),
    Metric("payment_failure_rate", "Payment failure rate",
           "Failed payment attempts / all payment attempts, by payment date (payment-attempt grain).",
           "payment attempt",
           """SELECT COUNT(*) AS attempts,
       COUNTIF(payment_status = 'failed') AS failed,
       ROUND(SAFE_DIVIDE(COUNTIF(payment_status = 'failed'), COUNT(*)), 4) AS failure_rate
FROM {payments}
WHERE DATE(payment_timestamp) BETWEEN @start_date AND @end_date""", "What is our payment failure rate?"),
    Metric("gross_item_sales", "Gross item sales, discounts, merchandise sales",
           "SUM of order_items.gross_amount, discount_amount and net_amount for booked orders (item columns, never order header totals).",
           "order item",
           f"""SELECT ROUND(SUM(li.gross_amount), 2) AS gross_item_sales,
       ROUND(SUM(li.discount_amount), 2) AS discounts,
       ROUND(SUM(li.net_amount), 2) AS merch_sales
FROM {{order_items}} li
JOIN {{orders}} o ON o.order_id = li.order_id
WHERE o.order_status IN {BOOKED}
  AND o.order_date BETWEEN @start_date AND @end_date""", "What were gross item sales and discounts?"),
    Metric("active_customers", "Active customers",
           "COUNT(DISTINCT customer_id) over booked orders in the window.",
           "customer",
           f"""SELECT COUNT(DISTINCT customer_id) AS active_customers
FROM {{orders}}
WHERE order_status IN {BOOKED}
  AND order_date BETWEEN @start_date AND @end_date""", "How many customers placed at least one order in Q2 2026?"),
    Metric("refunds_by_reason", "Completed refunds by reason",
           "SUM(refund_amount) and count of completed refunds by refund_reason, by refund date.",
           "refund",
           """SELECT refund_reason, COUNT(*) AS refunds, ROUND(SUM(refund_amount), 2) AS amount
FROM {refunds}
WHERE refund_status = 'completed'
  AND DATE(refund_timestamp) BETWEEN @start_date AND @end_date
GROUP BY refund_reason
ORDER BY amount DESC""", "What are completed refunds by reason?"),
    Metric("mom_change", "Month-over-month change",
           "Compute the metric per month (and dimension), then (current - previous) / previous * 100 with LAG; NULL when "
           "previous is 0 or missing. Read data from the month BEFORE the period start so the first month has a comparison, "
           "and report the last month of the period. Compare complete months only.",
           "period",
           "WITH " + _LINES.replace("o.order_date BETWEEN @start_date AND @end_date",
                                    "o.order_date BETWEEN DATE_TRUNC(DATE_SUB(@start_date, INTERVAL 1 MONTH), MONTH) AND @end_date") + """,
monthly AS (
  SELECT c.region, FORMAT_DATE('%Y-%m', lines.order_date) AS month,
         SUM(lines.net_amount) - SUM(IFNULL(line_refunds.refunded, 0)) AS net_sales
  FROM lines
  JOIN {customers} c ON c.customer_id = lines.customer_id
  LEFT JOIN line_refunds ON line_refunds.order_item_id = lines.order_item_id
  GROUP BY c.region, month
),
with_prev AS (
  SELECT region, month, net_sales, LAG(net_sales) OVER (PARTITION BY region ORDER BY month) AS previous
  FROM monthly
)
SELECT region, month, ROUND(previous, 2) AS previous, ROUND(net_sales, 2) AS net_sales,
       ROUND(100 * SAFE_DIVIDE(net_sales - previous, previous), 2) AS mom_pct
FROM with_prev
WHERE month = FORMAT_DATE('%Y-%m', @end_date)
ORDER BY mom_pct""", "Which regions had the largest month-over-month sales decline?"),
    Metric("order_status_counts", "Orders by status",
           "COUNT(*) of orders grouped by order_status (full snapshot unless a period is given).",
           "order",
           """SELECT order_status, COUNT(*) AS orders
FROM {orders}
GROUP BY order_status
ORDER BY orders DESC""", "How many orders are in each status?"),
    # ---- customer metrics (customers table; a customer = one distinct customer_id) ----
    Metric("total_customers", "Total customers",
           "Number of customers in the customers table. customers = COUNT(DISTINCT customer_id); customer_rows = COUNT(*) "
           "is also returned so duplicate rows are visible (they should be equal). Snapshot: no date filter.",
           "customer",
           """SELECT COUNT(*) AS customer_rows,
       COUNT(DISTINCT customer_id) AS customers
FROM {customers}""", "How many customers are in the table?"),
    Metric("customers_by_region", "Customers by region",
           "COUNT(DISTINCT customer_id) per region, with each region's share of all distinct customers (pct_of_customers). "
           "NULL regions are reported as 'Unknown', never dropped. Ties are ordered by region name. Snapshot: no date filter.",
           "customer",
           """SELECT COALESCE(region, 'Unknown') AS region,
       COUNT(DISTINCT customer_id) AS customers,
       ROUND(100 * COUNT(DISTINCT customer_id) / (SELECT COUNT(DISTINCT customer_id) FROM {customers}), 2) AS pct_of_customers
FROM {customers}
GROUP BY region
ORDER BY customers DESC, region""", "Show customer counts by region."),
    Metric("customers_by_segment", "Customers by segment",
           "COUNT(DISTINCT customer_id) per customer_segment (consumer, small_business, corporate), with each segment's share "
           "of all distinct customers (pct_of_customers). NULL segments are reported as 'Unknown'. Snapshot: no date filter.",
           "customer",
           """SELECT COALESCE(customer_segment, 'Unknown') AS customer_segment,
       COUNT(DISTINCT customer_id) AS customers,
       ROUND(100 * COUNT(DISTINCT customer_id) / (SELECT COUNT(DISTINCT customer_id) FROM {customers}), 2) AS pct_of_customers
FROM {customers}
GROUP BY customer_segment
ORDER BY customers DESC, customer_segment""", "What percentage of customers are in each segment?"),
    Metric("signups_by_month", "New signups by month",
           "COUNT(DISTINCT customer_id) by calendar month of signup_date (a DATE, no timezone conversion) within the period. "
           "Rows with a NULL signup_date are excluded and reported by the data-quality checks.",
           "customer",
           """SELECT FORMAT_DATE('%Y-%m', signup_date) AS month,
       COUNT(DISTINCT customer_id) AS signups
FROM {customers}
WHERE signup_date BETWEEN @start_date AND @end_date
GROUP BY month
ORDER BY month""", "How many customers signed up each month?"),
    Metric("signups_by_region", "New signups by region",
           "COUNT(DISTINCT customer_id) per region for customers whose signup_date falls in the period. NULL regions are "
           "reported as 'Unknown'. Ties are ordered by region name.",
           "customer",
           """SELECT COALESCE(region, 'Unknown') AS region,
       COUNT(DISTINCT customer_id) AS signups
FROM {customers}
WHERE signup_date BETWEEN @start_date AND @end_date
GROUP BY region
ORDER BY signups DESC, region""", "Which region had the most signups during the last three months?"),
    Metric("missing_email_rate", "Missing-email percentage",
           "Customer rows whose email is NULL or blank (after trimming) divided by all customer rows, as a percentage. "
           "Counts only; email addresses are never returned. Snapshot: no date filter.",
           "customer row",
           """SELECT COUNT(*) AS customer_rows,
       COUNTIF(email IS NULL OR TRIM(email) = '') AS missing_email,
       ROUND(100 * SAFE_DIVIDE(COUNTIF(email IS NULL OR TRIM(email) = ''), COUNT(*)), 2) AS missing_email_pct
FROM {customers}""", "What percentage of customers have no email address?"),
    Metric("duplicate_customer_ids", "Duplicate customer IDs",
           "Rows that repeat an existing customer_id: COUNT(*) - COUNT(DISTINCT customer_id), plus how many IDs are affected. "
           "0 on a healthy table. Snapshot: no date filter.",
           "customer row",
           """SELECT COUNT(*) AS customer_rows,
       COUNT(DISTINCT customer_id) AS distinct_customer_ids,
       COUNT(*) - COUNT(DISTINCT customer_id) AS duplicate_rows,
       (SELECT COUNT(*) FROM (SELECT customer_id FROM {customers} GROUP BY customer_id HAVING COUNT(*) > 1)) AS duplicated_ids
FROM {customers}""", "Are there duplicate customer IDs?"),
)
METRICS_BY_NAME = {m.name: m for m in METRICS}

JOIN_RULES = """Join rules and traps:
- orders.order_id = order_items.order_id = payments.order_id = shipments.order_id = refunds.order_id
- order_items.product_id = products.product_id; orders.customer_id = customers.customer_id
- refunds.order_item_id = order_items.order_item_id (NULL for order-level cancellation refunds)
- FAN-OUT: joining orders to order_items, payments, refunds or shipments repeats each order row. Never SUM/AVG
  order-level columns (orders.order_total, items_subtotal, shipping_amount, tax_amount, orders.discount_amount)
  after such a join, and use COUNT(DISTINCT order_id) for order counts. Aggregate each child table in its own CTE first.
- Never join two child tables of orders directly (e.g. order_items and payments) without aggregating one of them first.
- Payments: filter payment_status = 'captured' for money received. Refunds: filter refund_status = 'completed'.
- Booked orders = order_status IN ('processing', 'shipped', 'delivered'); use them for every sales metric."""


def schema_block(refs: dict, blocked_columns: set) -> str:
    lines = []
    for name in sr.TABLE_LOAD_ORDER:
        t = sr.get_table(name)
        lines.append(f"Table {refs[name]}  -- {t.purpose} Grain: {t.grain}")
        for c in t.columns:
            if c.name in blocked_columns:
                lines.append(f"  - {c.name} ({c.bq_type}) BLOCKED: personal data, never select it")
                continue
            allowed = f" Allowed: {', '.join(c.allowed_values)}." if c.allowed_values else ""
            lines.append(f"  - {c.name} ({c.bq_type}): {c.description}{allowed}")
    return "\n".join(lines)


def metrics_block(refs: dict, include_sql: bool = True) -> str:
    out = []
    for m in METRICS:
        out.append(f"### {m.name} — {m.label} (grain: {m.grain})\n{m.definition}\nExample question: {m.example_question}")
        if include_sql:
            out.append("Canonical SQL:\n" + m.render(refs))
    return "\n\n".join(out)
