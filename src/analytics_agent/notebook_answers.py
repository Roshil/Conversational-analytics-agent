"""Plain-language final answers, a question runner and text reports for notebook 05.

Notebook 05 used to redefine ``ask()``, ``final_answer()`` and the report writers in six different cells, each a
little different. They live here once, so every section builds answers and reports the same way.

* final_answer           detailed answer built only from the returned rows, or the reason nothing was returned.
                         A failed question (status ``error``) is always reported as a failure, never as "no rows".
* QuestionRunner         runs one question, records route, queries, timing and WHICH MODEL answered it
                         (Gemini, the offline stand-in, both, or no model call), and displays it.
* record_as_text / save_report / model_usage / model_warning
                         text reports and an explicit statement of how many answers came from Gemini.
* find_emails / scan_records_for_emails / scan_files_for_emails
                         privacy checks over answers, result rows and saved report files.
"""
from __future__ import annotations

import re
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

LINE_WIDTH = 100

FIELD_PHRASES = {
    "email": "the email address is {}",
    "region": "the region is {}",
    "country_code": "the country is {}",
    "customer_segment": "the segment is {}",
    "signup_date": "the signup date is {}",
    "created_at": "the record was created at {}",
}
ITEM_FIELDS = ["product", "quantity", "unit_price", "gross_amount", "discount_amount", "net_amount"]

REFUSAL_REASONS = {
    "write_request": "The agent is read-only. Any request to change, delete or insert data is refused by the pre-check "
                     "before a model is called or a query is run, so the database cannot be modified.",
    "instruction_override": "The question tried to override the agent's instructions. Such requests are refused "
                            "before any model call or query.",
    "prompt_injection": "The question tried to override the agent's instructions. Such requests are refused before "
                        "any model call or query.",
    "unapproved_table": "The question referred to a table outside the approved list, so nothing was run.",
    "out_of_scope": "The question asked for tables or metadata outside the approved analytics dataset.",
}

EMAIL_RE = re.compile(r"\b[\w.+-]{2,}@[\w-]+\.[\w.]+\b")
_INJECTION_RE = re.compile(r"\bor\s+1\s*=\s*1\b|\bunion\b|--|;")


# ---------------------------------------------------------------------------
# Formatting helpers (None-safe and zero-safe)
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fmt(value) -> str:
    """Integers with thousands separators, floats with 2 decimals (4 below 1), everything else as text."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) >= 1 or value == 0 else f"{value:.4f}"
    return str(value)


def money(value) -> str:
    return "n/a" if value is None else f"{float(value):,.2f}"


def signed_pct(value) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"


def plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def percent(value, column: str = "") -> str:
    """Format a rate using the column name, not the size of the value.

    Columns named ``*pct*`` / ``*percent*`` already hold percentages (24.30 means 24.30%). Any other rate
    (``late_rate``, ``share``, ``ratio``) is a fraction (0.1872 means 18.72%). Guessing from the value would show a
    1% rate stored as ``1.0`` as 100%.
    """
    if value is None:
        return "n/a"
    value = float(value)
    name = column.lower()
    if "pct" in name or "percent" in name:
        return f"{value:.2f}%"
    return f"{value * 100:.2f}%"


def first_line(text: Optional[str]) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0] if lines else ""


def find_column(row: dict, *words: str) -> Optional[str]:
    """First column whose name contains one of the words (columns are matched by meaning, not exact name)."""
    return next((c for c in row if any(w in c.lower() for w in words)), None)


def label_column(rows: list) -> Optional[str]:
    """The first text column (category, region, month, sales channel, ...), or None."""
    if not rows:
        return None
    return next((c for c in rows[0] if isinstance(rows[0][c], str)), None)


def _period(response, default: str = "the whole data window") -> str:
    return (response.period or default).replace("..", " to ")


def _sort_key(column: str, descending: bool = True):
    """Sort key that puts rows with a missing value last."""
    def key(row):
        value = row.get(column)
        if value is None:
            return (1, 0.0)
        return (0, -float(value) if descending else float(value))
    return key


# ---------------------------------------------------------------------------
# Questions that returned no data: refused, clarification, error, empty
# ---------------------------------------------------------------------------

def not_answered(response) -> Optional[list[str]]:
    """Lines for a question that produced no result, or None when there is a result to describe.

    ``error`` is checked first and never reported as an empty result: a failed query or a failed model call is not
    the same as a query that ran and found nothing.
    """
    reply = first_line(response.answer)
    if response.status == "error":
        return [f"The question could not be answered: {reply or 'the pipeline failed'}",
                "This is a failure, not an empty result: no figures are reported. Re-run the question, or check the "
                "connection, the model quota and the SQL attempts shown above."]
    if response.status == "refused":
        lines = [f"No query was run. {reply}"]
        precheck = next((s for s in response.steps if s.get("step") == "precheck"), {})
        reason = REFUSAL_REASONS.get(precheck.get("category"))
        if reason:
            lines.append(reason)
        return lines
    if response.status == "clarification_needed":
        lines = [f"No query was run. {reply}"]
        if "don't have" in (response.answer or ""):
            lines.append("Rather than returning other fields instead, the agent states that the requested information "
                         "does not exist in the data, so nothing misleading is shown.")
        else:
            lines.append("The agent asked for clarification instead of guessing, so no wrong numbers are shown.")
        return lines
    if response.status == "no_data" and response.intent not in ("customer_lookup", "order_item_lookup"):
        return [f"The query ran successfully but returned no rows for {_period(response)}, so there is nothing to "
                "report (no data was invented).",
                "The data covers July 2025 to June 2026; no figures are produced for periods outside it."]
    return None


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def customer_answer(response, reveal_email: bool = False) -> list[str]:
    """One sentence per customer (found / not found / failed / conflicting / duplicate) plus safety notes."""
    lines = []
    conflict = re.search(r"disagree on ([\w, ]+?);", response.answer or "")
    for index, row in enumerate(response.rows):
        customer = row.get("customer_id") or (
            response.customer_ids[index] if index < len(response.customer_ids) else "?")
        status, count = row.get("lookup_status"), row.get("record_count") or 0
        if status == "not_found":
            lines.append(f"Customer {customer} does not exist in the customers table. The lookup query ran "
                         "successfully and returned no rows, so this is a confirmed 'not found', not an error.")
            continue
        if status == "error":
            lines.append(f"The lookup for {customer} failed, so no answer is available. This does not mean the "
                         "customer is missing; check the connection and permissions.")
            continue
        values = [FIELD_PHRASES[k].format(v) for k, v in row.items() if k in FIELD_PHRASES and v is not None]
        lines.append(f"For customer {customer}, " + "; ".join(values) + "." if values
                     else f"Yes, customer {customer} exists.")
        if status == "conflicting":
            fields = conflict.group(1) if conflict else "some fields"
            lines.append(f"Data-quality warning for {customer}: {count} records share this customer ID and they "
                         f"disagree on {fields}. The values above come from the first record; resolve the duplicate "
                         "before using this customer in reports (check CUS_PK_UNIQUE).")
        elif count > 1:
            lines.append(f"Note for {customer}: {count} records share this customer ID. They agree on the fields "
                         "shown here but may differ in others (duplicate customer ID).")
    step = next((s for s in response.steps if s.get("step") == "customer_lookup"), {})
    if step.get("source") == "conversation":
        lines.append("The question did not name a customer, so the agent used the one from the previous turn: "
                     f"{', '.join(response.customer_ids)}. It does this only when exactly one customer was discussed.")
    injection = _INJECTION_RE.search((response.question or "").lower())
    if injection and response.parameters:
        ids = ", ".join(str(p.get("value")) for p in response.parameters if isinstance(p, dict))
        lines.append(f"Only {ids} was looked up: the customer ID is passed to BigQuery as a bound parameter "
                     f"(@customer_id), so the extra text '{injection.group(0).strip()}' cannot change the query.")
    if not reveal_email and any("email" in r for r in response.rows):
        lines.append("Email addresses are masked by policy (CUSTOMER_LOOKUP_REVEAL_EMAIL=true in .env shows them "
                     "to authorised users).")
    return lines


def describe_line(line: dict, fields: list) -> str:
    """One order line in words."""
    product = line["product_id"] + (f" ({line['product_name']}, {line.get('category')})"
                                    if line.get("product_name") else "")
    parts = []
    if "product" in fields:
        parts.append(f"the product is {product}")
    if "quantity" in fields:
        parts.append(f"the quantity is {line['quantity']}")
    if "unit_price" in fields:
        parts.append(f"the unit price is {money(line['unit_price'])}")
    if "gross_amount" in fields:
        parts.append(f"the gross amount is {money(line['gross_amount'])}")
    if "discount_amount" in fields:
        parts.append("there was no discount" if float(line["discount_amount"] or 0) == 0
                     else f"the discount was {money(line['discount_amount'])}")
    if "net_amount" in fields:
        parts.append(f"the net amount is {money(line['net_amount'])}")
    return "; ".join(parts)


def arithmetic_check(line: dict) -> str:
    """Check quantity x unit price = gross and gross - discount = net on the stored values."""
    gross = float(line["quantity"]) * float(line["unit_price"])
    net = float(line["gross_amount"]) - float(line["discount_amount"])
    ok = abs(gross - float(line["gross_amount"])) < 0.01 and abs(net - float(line["net_amount"])) < 0.01
    return (f"Arithmetic check: {line['quantity']} x {money(line['unit_price'])} = {money(gross)} gross; "
            f"{money(line['gross_amount'])} - {money(line['discount_amount'])} discount = {money(net)}; stored net "
            f"amount {money(line['net_amount'])}: "
            + ("consistent." if ok else "INCONSISTENT, the stored values do not add up (a data-quality issue)."))


def order_item_answer(response) -> list[str]:
    """Order items: values and an arithmetic check. Whole orders: what was asked first, then lines and totals."""
    step = next((s for s in response.steps if s.get("step") == "order_item_lookup"), {})
    fields = [f for f in step.get("fields", ITEM_FIELDS) if f in ITEM_FIELDS] or ITEM_FIELDS
    asked_everything = set(ITEM_FIELDS) <= set(fields)
    asks_tax = any(w in (response.question or "").lower() for w in ("tax", "shipping"))
    lines = []
    for record_id in response.record_ids:
        rows = [r for r in response.rows if r.get("record_id") == record_id]
        status = rows[0].get("lookup_status") if rows else "not_found"
        kind = "order item" if record_id.upper().startswith("ITM") else "order"
        if status == "not_found":
            lines.append(f"The {kind} {record_id} does not exist in the order_items table (the query ran and returned "
                         "no rows, so this is a confirmed 'not found', not an error).")
            continue
        if status == "error":
            lines.append(f"The lookup for {record_id} failed, so no answer is available.")
            continue
        if kind == "order item":
            line = rows[0]
            lines.append(f"Order item {record_id} is line {line['line_number']} of order {line['order_id']}: "
                         f"{describe_line(line, fields)}.")
            lines.append(arithmetic_check(line))
            if status == "conflicting":
                lines.append(f"Warning: {line['record_count']} records exist for this ID and they disagree.")
            elif (line.get("record_count") or 0) > 1:
                lines.append(f"Data-quality warning: {line['record_count']} records share the order item ID "
                             f"{record_id} (a duplicate ID). Counting both would double this line in sales totals.")
            continue
        units = sum(int(r["quantity"]) for r in rows)
        gross = sum(float(r["gross_amount"]) for r in rows)
        discount = sum(float(r["discount_amount"]) for r in rows)
        net = sum(float(r["net_amount"]) for r in rows)
        discounted = [r for r in rows if float(r["discount_amount"]) > 0]
        totals = (f"gross {money(gross)} - discount {money(discount)} = net {money(net)} "
                  "(merchandise only; tax and shipping are not in order_items)")
        if not asked_everything and fields == ["net_amount"]:
            lines.append(f"The total net amount of order {record_id} is {money(net)}: {totals}.")
        elif not asked_everything and "discount_amount" in fields:
            if discounted:
                lines.append(f"In order {record_id}, " + "; ".join(
                    f"line {r['line_number']} ({r['order_item_id']}, {r.get('product_name') or r['product_id']}) had "
                    f"a discount of {money(r['discount_amount'])}" for r in discounted) + ".")
                lines.append(f"The other {plural(len(rows) - len(discounted), 'line')} had no discount. "
                             f"Order total: {totals}.")
            else:
                lines.append(f"No line in order {record_id} had a discount. Order total: {totals}.")
        else:
            lines.append(f"Order {record_id} has {plural(len(rows), 'line item')} with {plural(units, 'unit')} in total.")
            for r in rows:
                lines.append(f"Line {r['line_number']} ({r['order_item_id']}): {describe_line(r, fields)}.")
            lines.append(f"Order totals from its lines: {totals}.")
        if asks_tax:
            lines.append("Tax and shipping are not part of order_items, so they are not included and were not "
                         "estimated. They are stored at order level (in the orders table), which this lookup does not read.")
    return lines


# ---------------------------------------------------------------------------
# Metric answers
# ---------------------------------------------------------------------------

def ranking(rows: list, key: str, value: str, share: Optional[str] = None) -> list[str]:
    """'Southeast 486 (24.30%), ...' plus the largest and smallest, or the tie at the top."""
    rows = [r for r in rows if r.get(value) is not None] or rows
    parts = [f"{r[key] if r.get(key) is not None else 'Unknown'} {fmt(r.get(value))}"
             + (f" ({percent(r[share], share)})" if share and r.get(share) is not None else "") for r in rows]
    lines = [", ".join(parts) + "."]
    if len(rows) < 2:
        return lines
    top, bottom = rows[0], rows[-1]
    tied = [r[key] for r in rows if r.get(value) == top.get(value)]
    if len(tied) > 1:
        lines.append(f"There is a tie at the top: {', '.join(map(str, tied))} each have {fmt(top.get(value))}.")
    else:
        def share_of(r):
            return f" ({percent(r[share], share)})" if share and r.get(share) is not None else ""
        lines.append(f"The largest is {top[key]} with {fmt(top.get(value))}{share_of(top)}; the smallest is "
                     f"{bottom[key]} with {fmt(bottom.get(value))}{share_of(bottom)}.")
    return lines


def describe_net_sales_by_dimension(rows: list, period: str) -> list[str]:
    key = label_column(rows)
    if not key:
        return generic_rows(rows)
    ordered = sorted(rows, key=_sort_key("net_sales"))
    total = sum(float(r["net_sales"] or 0) for r in rows)
    name = key.replace("_", " ")
    parts = [f"{r[key]} {fmt(r['net_sales'])}"
             + (f" ({100 * float(r['net_sales'] or 0) / total:.2f}%)" if total else "") for r in ordered]
    lines = [f"Net sales by {name} for {period}: " + ", ".join(parts) + "."]
    if len(ordered) > 1:
        lines.append(f"The largest is {ordered[0][key]} and the smallest is {ordered[-1][key]}.")
    lines.append(f"Together the {len(rows)} {name} groups add up to {fmt(round(total, 2))} in net sales.")
    note = "Only completed item-level refunds are subtracted (pending refunds are not)"
    if "categor" in key:
        note += ", and the category comes from the products table, joined on product_id"
    lines.append(note + ".")
    return lines


def describe_monthly_net_sales(rows: list, period: str) -> list[str]:
    valued = [r for r in rows if r.get("net_sales") is not None]
    if not valued:
        return generic_rows(rows)
    total = sum(float(r["net_sales"]) for r in valued)
    best = max(valued, key=lambda r: float(r["net_sales"]))
    worst = min(valued, key=lambda r: float(r["net_sales"]))
    lines = [f"Monthly net sales for {period}: " + ", ".join(f"{r['month']} {fmt(r.get('net_sales'))}" for r in rows) + ".",
             f"The strongest month was {best['month']} ({fmt(best['net_sales'])}) and the weakest was "
             f"{worst['month']} ({fmt(worst['net_sales'])}); the total for the period is {fmt(round(total, 2))}."]
    if "merch_sales" in rows[0] and "refunds" in rows[0]:
        first = rows[0]
        lines.append(f"Net sales are merchandise sales minus completed refunds, for example {first['month']}: "
                     f"{fmt(first['merch_sales'])} - {fmt(first['refunds'])} = {fmt(first['net_sales'])}.")
    return lines


def describe_average_order_value(rows: list, period: str) -> list[str]:
    aov = find_column(rows[0], "average_order_value", "aov", "average")
    if not aov:
        return generic_rows(rows)
    orders = next((c for c in rows[0] if c in ("orders", "order_count")), None)
    key = label_column(rows)
    note = ("Tax and shipping are excluded: AOV is merchandise value after discounts divided by the number of "
            "booked orders, not AVG(order_total).")
    if len(rows) == 1 or not key:
        count = f" over {rows[0][orders]:,} booked orders" if orders and rows[0].get(orders) is not None else ""
        return [f"The average order value for {period} is {fmt(rows[0][aov])}{count}.", note]
    ordered = sorted(rows, key=_sort_key(aov))
    lines = [f"Average order value by {key} for {period}: " + ", ".join(
        f"{r[key]} {fmt(r[aov])}" + (f" ({r[orders]:,} orders)" if orders and r.get(orders) is not None else "")
        for r in ordered) + ".",
             f"The highest is {ordered[0][key]} ({fmt(ordered[0][aov])}) and the lowest is "
             f"{ordered[-1][key]} ({fmt(ordered[-1][aov])})."]
    if orders:
        counted = [r for r in rows if r.get(orders) and r.get(aov) is not None]
        total_orders = sum(r[orders] for r in counted)
        if total_orders:
            weighted = sum(float(r[aov]) * r[orders] for r in counted) / total_orders
            lines.append(f"Across all {total_orders:,} booked orders, the order-weighted average is about {weighted:,.2f}.")
    lines.append(note)
    return lines


def describe_month_over_month(rows: list, period: str) -> list[str]:
    change = find_column(rows[0], "pct", "change")
    if not change:
        return generic_rows(rows)
    previous = find_column(rows[0], "prev")
    current = next((c for c in rows[0] if c == "net_sales"), None) or find_column(rows[0], "sales", "current")
    key = label_column(rows)
    known = sorted([r for r in rows if r.get(change) is not None], key=lambda r: float(r[change]))
    unknown = [r for r in rows if r.get(change) is None]
    if not known:
        return ["No month-over-month change could be calculated: the previous month has no net sales for any group."]
    worst, best = known[0], known[-1]
    name = (lambda r: r.get(key) if key else r.get("month", "the period"))
    lines = []
    if float(worst[change]) < 0:
        detail = (f": net sales fell from {fmt(worst.get(previous))} in the previous month to {fmt(worst.get(current))}"
                  if previous and current else "")
        lines.append(f"The biggest month-over-month drop for {period} was in {name(worst)}{detail}, "
                     f"a change of {signed_pct(worst[change])}.")
    else:
        lines.append(f"No group had a drop for {period}; the smallest change was {name(worst)} "
                     f"({signed_pct(worst[change])}).")
    if len(known) > 1:
        lines.append(f"The biggest increase was in {name(best)} ({signed_pct(best[change])}). All changes: "
                     + ", ".join(f"{name(r)} {signed_pct(r[change])}" for r in known) + ".")
    if unknown:
        lines.append("No change could be calculated for " + ", ".join(str(name(r)) for r in unknown)
                     + " (no net sales in the previous month).")
    lines.append("Each month is compared with the month before it, so the month before the period is read as well; "
                 "net sales use completed refunds only.")
    return lines


def describe_late_rate(rows: list, period: str) -> list[str]:
    rate = find_column(rows[0], "late_rate", "rate", "share", "pct")
    if not rate:
        return generic_rows(rows)
    delivered = find_column(rows[0], "delivered")
    late = next((c for c in rows[0] if "late" in c.lower() and c != rate), None)
    key = label_column(rows)
    if len(rows) == 1 or not key:
        return [f"For {period}, {percent(rows[0][rate], rate)} of delivered orders arrived late."]
    ordered = sorted(rows, key=_sort_key(rate))
    lines = [f"Late deliveries by {key} for {period}:"]
    for position, r in enumerate(ordered):
        counts = (f"{r[late]:,} of {r[delivered]:,} delivered orders arrived late, "
                  if late and delivered and r.get(late) is not None and r.get(delivered) is not None else "")
        tag = " (the highest)" if position == 0 else (" (the lowest)" if position == len(ordered) - 1 else "")
        lines.append(f"{r[key]}: {counts}{percent(r[rate], rate)}{tag}.")
    lines.append("Only delivered shipments are counted; shipments still in transit are excluded from the denominator.")
    return lines


def generic_rows(rows: list, limit: int = 12) -> list[str]:
    lines = [", ".join(f"{k} = {fmt(v)}" for k, v in row.items()) for row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more rows.")
    return lines


def describe_metric(response, metric_definitions: Optional[dict] = None) -> list[str]:
    """A detailed, plain-language answer for a metric question, built only from the returned rows.

    Args:
        response: AnalyticsResponse with status ``answered``.
        metric_definitions: optional ``{name: Metric}`` (``semantic_layer.METRICS_BY_NAME``) to explain how each
            metric was calculated.
    """
    rows, metrics = response.rows, list(response.metrics or [])
    if not rows:
        return [f"The query ran successfully but returned no rows for {_period(response)}, so there is nothing to "
                "report (no data was invented)."]
    period = _period(response)
    metric = metrics[0] if metrics else None
    first = rows[0]
    lines: list[str] = []
    if metric == "total_customers" and "customers" in first:
        lines.append(f"There are {fmt(first['customers'])} distinct customers in the customers table.")
        if first.get("customer_rows") == first["customers"]:
            lines.append(f"The table has {fmt(first['customer_rows'])} rows, one per customer, so there are no "
                         "duplicate customer IDs.")
        elif first.get("customer_rows") is not None:
            lines.append(f"The table has {fmt(first['customer_rows'])} rows: "
                         f"{fmt(first['customer_rows'] - first['customers'])} more than the number of customers, "
                         "because some customer IDs appear more than once. Counting rows would over-report customers.")
    elif metric in ("customers_by_region", "customers_by_segment") and "customers" in first:
        key = "region" if metric == "customers_by_region" else "customer_segment"
        word = key.replace("customer_", "")
        total = sum(r["customers"] or 0 for r in rows)
        ranked = ranking(rows, key, "customers", "pct_of_customers")
        lines.append(f"Distinct customers by {word}: " + ranked[0])
        lines += ranked[1:]
        if "pct_of_customers" in first:
            share_total = round(sum(float(r["pct_of_customers"] or 0) for r in rows), 2)
            lines.append(f"The groups add up to {fmt(total)} customers and {share_total}% (100% apart from rounding).")
        unknown = next((r for r in rows if r.get(key) in (None, "Unknown")), None)
        if unknown:
            lines.append(f"{fmt(unknown['customers'])} customers have no {word} recorded and are shown as 'Unknown' "
                         "rather than being dropped.")
    elif metric in ("signups_by_region", "signups_by_month") and "signups" in first:
        key = "region" if metric == "signups_by_region" else "month"
        total = sum(r["signups"] or 0 for r in rows)
        lines.append(f"New customer signups by {key} for {period}: "
                     + ", ".join(f"{r.get(key)} {fmt(r['signups'])}" for r in rows) + ".")
        lines += ranking(sorted(rows, key=_sort_key("signups")), key, "signups")[1:]
        lines.append(f"In total, {fmt(total)} customers signed up in this period.")
    elif metric == "missing_email_rate" and "missing_email" in first:
        lines.append(f"{fmt(first['missing_email'])} of {fmt(first.get('customer_rows'))} customer records have no "
                     f"email address ({percent(first.get('missing_email_pct'), 'missing_email_pct')}). Only counts are "
                     "returned; no email addresses are shown.")
    elif metric == "duplicate_customer_ids" and "duplicate_rows" in first:
        lines.append(f"{fmt(first.get('duplicated_ids'))} customer IDs appear more than once, adding "
                     f"{fmt(first['duplicate_rows'])} extra rows ({fmt(first.get('customer_rows'))} rows for "
                     f"{fmt(first.get('distinct_customer_ids'))} distinct IDs).")
    elif "mom_change" in metrics:
        lines += describe_month_over_month(rows, period)
    elif "average_order_value" in metrics:
        lines += describe_average_order_value(rows, period)
    elif "late_delivery_rate" in metrics:
        lines += describe_late_rate(rows, period)
    elif "net_sales_by_dimension" in metrics and "net_sales" in first:
        lines += describe_net_sales_by_dimension(rows, period)
    elif "net_sales" in metrics and "month" in first and "net_sales" in first:
        lines += describe_monthly_net_sales(rows, period)
    else:
        lines += generic_rows(rows)

    attempts = [s for s in response.steps if s.get("step") == "generate_sql"]
    if len(attempts) > 1:
        reasons = [", ".join(s.get("issues") or s.get("errors") or []) for s in response.steps
                   if s.get("step") in ("validate", "review_sql") and not s.get("ok", s.get("approved", True))]
        lines.append(f"The SQL was corrected before it ran ({len(attempts)} attempts); rejected because: "
                     f"{'; '.join(r for r in reasons if r) or 'see the SQL attempts above'}.")
    if metric_definitions:
        definitions = [f"{metric_definitions[m].label}: {metric_definitions[m].definition}"
                       for m in metrics if m in metric_definitions]
        if definitions:
            lines.append("How it was calculated: " + " ".join(definitions))
    for assumption in response.assumptions:
        lines.append(f"Assumption: {assumption}")
    if response.verification:
        lines.append("Every number in the agent's answer was checked against the query result."
                     if response.verification.get("ok") else
                     "The agent's wording could not be verified, so only the query result is reported.")
    return lines


def describe_data_quality(response) -> list[str]:
    lines = ["No SQL was generated: the answer comes from the deterministic data-quality report (50 checks run in "
             "the warehouse)."]
    lines += [line.strip("- ") for line in (response.answer or "").splitlines() if line.strip()][:6]
    return lines


def final_answer(response, reveal_email: bool = False, metric_definitions: Optional[dict] = None) -> list[str]:
    """The detailed final answer for any response: lookups, metrics, data quality, refusals and failures."""
    stopped = not_answered(response)
    if stopped is not None:
        return stopped
    if response.intent == "data_quality":
        return describe_data_quality(response)
    if response.intent == "customer_lookup":
        return customer_answer(response, reveal_email) if response.rows else [first_line(response.answer) or "No answer."]
    if response.intent == "order_item_lookup":
        return order_item_answer(response)
    return describe_metric(response, metric_definitions)


def conversation_note(response, previous_period: Optional[str]) -> Optional[str]:
    """Say when a follow-up reused the period of the previous turn."""
    if not response.period or response.period != previous_period:
        return None
    if any(w in (response.question or "").lower() for w in ("month", "quarter", "year", "week", "20")):
        return None
    return (f"Period {response.period.replace('..', ' to ')} was carried over from the previous question, because "
            "this one did not name a period.")


# ---------------------------------------------------------------------------
# Which model answered
# ---------------------------------------------------------------------------

GEMINI, STAND_IN, MIXED, NO_MODEL = "Gemini", "stand-in", "Gemini + stand-in", "no model call"


def model_used(before: dict, after: dict) -> str:
    """Compare ``ModelRouter.counts`` before and after one question."""
    gemini = after.get("gemini", 0) - before.get("gemini", 0)
    standin = after.get("stand-in", 0) - before.get("stand-in", 0)
    if gemini and standin:
        return MIXED
    if gemini:
        return GEMINI
    if standin:
        return STAND_IN
    return NO_MODEL


def model_usage(records: Iterable[dict]) -> dict:
    """``{model: number of answers}`` for recorded results."""
    usage: dict[str, int] = {}
    for record in records:
        usage[record.get("model", NO_MODEL)] = usage.get(record.get("model", NO_MODEL), 0) + 1
    return usage


def model_warning(records: list, what: str) -> Optional[str]:
    """A loud statement when any answer in ``records`` was produced by the offline stand-in, else None."""
    usage = model_usage(records)
    affected = usage.get(STAND_IN, 0) + usage.get(MIXED, 0)
    if not affected:
        return None
    return (f"WARNING: {affected} of {len(records)} answers in {what} were produced fully or partly by the offline "
            "stand-in (deterministic templates, NOT Gemini). Those results test the template SQL, not the language "
            "model. Raise GEMINI_CALL_BUDGET (section 2) or run this section in a fresh kernel to measure Gemini.")


def gemini_budget_check(router, budget, expected_calls: int, section: str) -> str:
    """One line saying whether the Gemini budget covers a section (call it before Gemini-heavy sections)."""
    if router.fallen_back:
        return (f"{section}: Gemini is not available ({router.reason}); every model step will use the offline "
                "stand-in, and each answer is labelled.")
    if budget.remaining < expected_calls:
        return (f"{section}: needs about {expected_calls} Gemini calls but only {budget.remaining} of {budget.limit} "
                "remain. Later answers will come from the offline stand-in and are labelled as such.")
    return f"{section}: about {expected_calls} Gemini calls needed, {budget.remaining} of {budget.limit} remain."


# ---------------------------------------------------------------------------
# Run, display and record one question
# ---------------------------------------------------------------------------

class QuestionRunner:
    """Runs questions through the pipeline and records route, queries, timing and the model that answered.

    Args:
        make_deps: ``(variant) -> AnalyticsDeps``.
        run_pipeline: the pipeline entry point.
        runner: the CountingRunner (its ``runs`` counter proves how many queries ran).
        router: the ModelRouter (its ``counts`` say whether Gemini or the stand-in answered).
        show: ``show(response, **kwargs)`` display helper.
        graph: compiled graph, for per-question route diagrams (optional).
        reveal_email: whether lookups are allowed to show full emails (from .env, never hard-coded).
        metric_definitions: ``semantic_layer.METRICS_BY_NAME``.
    """

    def __init__(self, *, make_deps: Callable, run_pipeline: Callable, runner, router, show: Callable,
                 graph=None, reveal_email: bool = False, metric_definitions: Optional[dict] = None):
        self.make_deps, self.run_pipeline, self.runner, self.router = make_deps, run_pipeline, runner, router
        self.show, self.graph, self.reveal_email = show, graph, reveal_email
        self.metric_definitions = metric_definitions

    def answer_lines(self, response) -> list[str]:
        return final_answer(response, self.reveal_email, self.metric_definitions)

    def run(self, question: str, variant: str = "clean", session=None) -> dict:
        """Run one question without displaying anything; return the record."""
        queries_before, counts_before = self.runner.runs, dict(self.router.counts)
        started = time.perf_counter()
        response = session.ask(question) if session else self.run_pipeline(question, [], self.make_deps(variant))
        seconds = time.perf_counter() - started
        return {"question": question, "dataset": variant, "seconds": seconds,
                "queries": self.runner.runs - queries_before,
                "model": model_used(counts_before, self.router.counts),
                "route": [step["step"] for step in response.steps], "response": response}

    def ask(self, question: str, variant: str = "clean", session=None, *, draw_route: bool = False,
            note: Optional[str] = None, note_label: str = "What makes it hard", log: Optional[list] = None,
            show_attempts: bool = False, show_sql: bool = True, show_rows: bool = True,
            answer_fn: Optional[Callable] = None) -> dict:
        """Run one question, display route, answer, SQL and the detailed final answer, and optionally log it."""
        from IPython.display import SVG, Markdown, display

        if note:
            display(Markdown(f"---\n#### {question}\n\n*Dataset:* `{variant}` · *{note_label}:* {note}"))
        record = self.run(question, variant, session)
        response = record["response"]
        print(f"[{response.status}] route: {' -> '.join(record['route'])}  ({record['seconds']:.1f}s, "
              f"{record['queries']} queries, model: {record['model']})")
        if draw_route and self.graph is not None:
            from analytics_agent.notebook_ui import graph_svg
            display(SVG(graph_svg(self.graph.get_graph(),
                                  run_nodes=["__start__", *record["route"], "finalize", "__end__"],
                                  title=f"Route: {question[:70]}")))
        self.show(response, show_attempts=show_attempts, show_sql=show_sql, show_rows=show_rows)
        lines = (answer_fn or self.answer_lines)(response)
        display(Markdown("**Final answer:**\n\n" + "\n".join(f"- {line}" for line in lines)))
        record.update(note=note, final_answer=lines)
        if log is not None:
            log.append(record)
        return record


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------

def _field(label: str, value, width: int = 17) -> str:
    return textwrap.fill(f"{label:<{width - 2}}: {value}", LINE_WIDTH, subsequent_indent=" " * width)


def record_as_text(number: int, record: dict, dataset_names: Optional[dict] = None, format_sql=None,
                   params_text=None, extra: Iterable[tuple] = (), max_rows: int = 25) -> str:
    """One recorded question, its result, SQL and the final answer as plain text."""
    response = record["response"]
    dataset = record.get("dataset", "")
    lines = ["=" * LINE_WIDTH, f"{number}. {record['question']}", "-" * LINE_WIDTH]
    if record.get("note"):
        lines.append(_field("Why it's hard", record["note"]))
    for label, value in extra:
        lines.append(_field(label, value))
    lines += [_field("Dataset", (dataset_names or {}).get(dataset, dataset)),
              _field("Status", response.status),
              _field("Model", record.get("model", "-")),
              _field("Route", " -> ".join(record.get("route", []))),
              _field("Time", f"{record.get('seconds', 0):.2f}s, {record.get('queries', 0)} queries executed")]
    if response.customer_ids:
        lines.append(_field("Customers", ", ".join(response.customer_ids)))
    if response.record_ids:
        lines.append(_field("Items/orders", ", ".join(response.record_ids)))
    if response.metrics:
        lines.append(_field("Metrics", ", ".join(response.metrics)))
    if response.period:
        lines.append(_field("Period", response.period.replace("..", " to ")))
    if response.rows:
        lines += ["", f"Result rows ({response.row_count if response.row_count is not None else len(response.rows)}):"]
        lines += ["    " + ", ".join(f"{k}={v}" for k, v in row.items()) for row in response.rows[:max_rows]]
    if response.caveats:
        lines += ["", "Data-quality caveats:"] + [f"    - {c}" for c in response.caveats]
    if response.sql:
        sql = format_sql(response.sql) if format_sql else response.sql
        lines += ["", "SQL:", textwrap.indent(sql, "    ")]
        if params_text:
            lines.append(f"    -- parameters: {params_text(response.parameters)}")
    lines += ["", "FINAL ANSWER:"]
    lines += [textwrap.fill(line, LINE_WIDTH, initial_indent="    - ", subsequent_indent="      ")
              for line in record.get("final_answer", [])]
    return "\n".join(lines) + "\n"


def save_report(records: list, name: str, title: str, *, report_dir: Path, backend: str, reveal_email: bool,
                header_lines: Iterable[str] = (), formatter: Optional[Callable] = None, **record_kwargs) -> Path:
    """Write the recorded questions to ``report_dir/<name>_<timestamp>.txt`` and return the path.

    The header always states how many answers came from Gemini and how many from the offline stand-in.
    """
    statuses: dict[str, int] = {}
    for record in records:
        statuses[record["response"].status] = statuses.get(record["response"].status, 0) + 1
    usage = model_usage(records)
    header = ["=" * LINE_WIDTH, title.upper(), "=" * LINE_WIDTH,
              f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
              f"Backend   : {backend}",
              f"Models    : {', '.join(f'{k}: {v}' for k, v in usage.items()) or '-'}",
              f"Emails    : {'shown in full (authorised in .env)' if reveal_email else 'masked'}",
              f"Questions : {len(records)} ({', '.join(f'{k}: {v}' for k, v in statuses.items())})",
              *header_lines]
    warning = model_warning(records, "this report")
    if warning:
        header.append(textwrap.fill(warning, LINE_WIDTH))
    header.append("")
    fmt_record = formatter or (lambda n, r: record_as_text(n, r, **record_kwargs))
    body = [fmt_record(n, r) for n, r in enumerate(records, 1)]
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text("\n".join(header) + "\n" + "\n".join(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Privacy checks
# ---------------------------------------------------------------------------

def find_emails(text: str) -> list[str]:
    return sorted(set(EMAIL_RE.findall(text or "")))


def _record_texts(record) -> list[tuple[str, str]]:
    """(part, text) pairs for one recorded result: agent answer, final answer and result rows."""
    response = record.get("response")
    parts = [("answer", getattr(response, "answer", "") or ""),
             ("final answer", " ".join(record.get("final_answer") or []))]
    rows = getattr(response, "rows", None) or []
    parts.append(("result rows", " ".join(str(v) for row in rows for v in row.values())))
    return parts


def scan_records_for_emails(sources: dict) -> tuple[int, list[dict]]:
    """Scan recorded results (lists of records, or DataFrames) for full email addresses.

    Checks the agent's answer, the final answer AND the result rows (tables shown in the notebook).

    Returns:
        ``(items_checked, findings)``.
    """
    import pandas as pd

    checked, findings = 0, []
    for source, value in sources.items():
        if isinstance(value, pd.DataFrame):
            question_col = "scenario" if "scenario" in value.columns else value.columns[0]
            for _, row in value.iterrows():
                checked += 1
                emails = find_emails(" ".join(str(v) for v in row.values))
                if emails:
                    findings.append({"source": source, "question": str(row[question_col])[:70], "where": "table row",
                                     "emails shown": ", ".join(emails[:3])})
            continue
        for record in value or []:
            checked += 1
            for part, text in _record_texts(record):
                emails = find_emails(text)
                if emails:
                    findings.append({"source": source, "question": str(record.get("question", "?"))[:70],
                                     "where": part, "emails shown": ", ".join(emails[:3])})
    return checked, findings


def scan_files_for_emails(paths: Iterable[Path]) -> list[dict]:
    """Scan saved report files for full email addresses."""
    findings = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        emails = find_emails(path.read_text(encoding="utf-8", errors="replace"))
        if emails:
            findings.append({"file": path.name, "emails": len(emails), "examples": ", ".join(emails[:3])})
    return findings
