"""Independent checks of the agent's answers.

Every expected value is computed with pandas from the source Parquet files, never from the agent's SQL, so a
check can catch a wrong join, filter, period or denominator.

The checks look at what an answer means rather than how it is laid out. A correct answer may name its columns
differently (``total_net_amount`` instead of ``net_amount``), return only the top row of a ranking, or give one
overall figure instead of a breakdown. Each check accepts any of these shapes when the numbers are right.

Usage::

    checks = AnswerChecks(data_dir, gold)                 # gold: {variant: customer_eval.customer_gold(...)}
    check = checks.top_net_sales("category", "2026-06-01", "2026-06-30")
    passed, detail = check(response)                      # response: AnalyticsResponse (status, rows)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

BOOKED = ("processing", "shipped", "delivered")
# Columns that name a group, in the order they are preferred as the label of a row.
DIMENSION_COLUMNS = ("category", "product_category", "region", "sales_channel", "channel", "customer_segment",
                     "segment", "carrier", "month")

Check = Callable[[object], tuple[bool, str]]


# ---------------------------------------------------------------------------
# Reading result rows whatever their layout
# ---------------------------------------------------------------------------

def find_column(row: dict, *names: str, contains: Iterable[str] = (), exclude: Iterable[str] = ()) -> Optional[str]:
    """Return the first column of *row* named in *names*, else the first whose name contains one of *contains*."""
    for name in names:
        if name in row:
            return name
    for column in row:
        low = column.lower()
        if any(part in low for part in contains) and not any(part in low for part in exclude):
            return column
    return None


def label_column(rows: list[dict]) -> Optional[str]:
    """The column that names each row's group: a known dimension first, otherwise the first text column."""
    if not rows:
        return None
    known = next((c for c in DIMENSION_COLUMNS if c in rows[0]), None)
    if known:
        return known
    return next((c for c in rows[0] if any(isinstance(r.get(c), str) for r in rows)), None)


def as_float(value) -> Optional[float]:
    return None if value is None else float(value)


def show(value) -> str:
    """A number for a check's message: whole numbers without decimals."""
    if value is None:
        return "-"
    value = float(value)
    return f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"


def value_by_key(rows: list[dict], key: str, value: str) -> dict:
    return {str(r.get(key)): as_float(r.get(value)) for r in rows}


def same_numbers(got: dict, expected: dict, tolerance: float = 0.01) -> tuple[bool, str]:
    """True when exactly the expected keys appear, each with the expected value (within *tolerance*)."""
    missing = [k for k in expected if k not in got]
    wrong = [k for k in expected if k in got and (got[k] is None or abs(got[k] - float(expected[k])) > tolerance)]
    extra = [k for k in got if k not in expected and got[k]]
    problems = []
    if missing:
        problems.append(f"missing {missing[:3]}")
    if wrong:
        problems.append("wrong " + ", ".join(f"{k}: {got[k]} vs {expected[k]}" for k in wrong[:3]))
    if extra:
        problems.append(f"unexpected {extra[:3]}")
    return not problems, "; ".join(problems)


def answered(check: Check) -> Check:
    """A value check only makes sense on an answer with rows; anything else is a failure with the status shown."""
    def wrapped(response):
        if response.status != "answered" or not response.rows:
            return False, f"status {response.status} with {len(response.rows or [])} rows (expected an answer)"
        return check(response)
    wrapped.__name__ = getattr(check, "__name__", "check")
    return wrapped


# ---------------------------------------------------------------------------
# Expected answers from the source files
# ---------------------------------------------------------------------------

class AnswerChecks:
    """Builds check functions whose expected values come from pandas on the source Parquet files.

    Args:
        data_dir: Folder holding ``<variant>/parquet/<table>.parquet``.
        gold: ``{variant: customer_eval.customer_gold(variant)}``; its ``records`` frame is the customers table.
    """

    def __init__(self, data_dir, gold: dict):
        self.data_dir = Path(data_dir)
        self.gold = gold
        self._tables: dict = {}

    # ---- source data ----
    def load(self, variant: str, table: str) -> pd.DataFrame:
        key = (variant, table)
        if key not in self._tables:
            self._tables[key] = pd.read_parquet(self.data_dir / variant / "parquet" / f"{table}.parquet")
        return self._tables[key]

    def customers(self, variant: str) -> pd.DataFrame:
        return self.gold[variant]["records"].reset_index()

    def booked_lines(self, variant: str, start: str, end: str) -> pd.DataFrame:
        """Order lines of booked orders in the period, with completed item refunds and the dimensions joined in."""
        orders = self.load(variant, "orders")
        dates = pd.to_datetime(orders.order_date)
        orders = orders[orders.order_status.isin(BOOKED) & (dates >= start) & (dates <= end)]
        lines = self.load(variant, "order_items").merge(
            orders[["order_id", "order_date", "customer_id", "sales_channel"]], on="order_id")
        refunds = self.load(variant, "refunds")
        completed = refunds[(refunds.refund_status == "completed") & refunds.order_item_id.notna()]
        refund_by_item = completed.assign(amount=completed.refund_amount.astype(float)).groupby("order_item_id").amount.sum()
        lines["merch"] = lines.net_amount.astype(float)
        lines["refund"] = lines.order_item_id.map(refund_by_item).fillna(0.0)
        lines["net"] = lines.merch - lines.refund
        lines["month"] = pd.to_datetime(lines.order_date).dt.strftime("%Y-%m")
        lines = lines.merge(self.load(variant, "products")[["product_id", "category"]], on="product_id", how="left")
        regions = self.customers(variant)[["customer_id", "region"]].drop_duplicates("customer_id")
        return lines.merge(regions, on="customer_id", how="left")

    def net_sales_by(self, variant: str, start: str, end: str, column: str) -> dict:
        return self.booked_lines(variant, start, end).groupby(column).net.sum().round(2).to_dict()

    # ---- customer checks ----
    def total_customers(self, variant: str, *, rows_too: bool = False) -> Check:
        """Distinct customers (and, with *rows_too*, the raw row count, which duplicates inflate)."""
        gold = self.gold[variant]["total_customers"]

        @answered
        def check(r):
            row = r.rows[0]
            distinct = find_column(row, "customers", "distinct_customers", "customer_count",
                                   contains=("distinct", "customers"), exclude=("row",))
            got = as_float(row.get(distinct)) if distinct else None
            ok, detail = got == gold["customers"], f"customers {show(got)} (expected {show(gold['customers'])})"
            if rows_too:
                rows_col = find_column(row, "customer_rows", "rows", "row_count", contains=("row",))
                got_rows = as_float(row.get(rows_col)) if rows_col else None
                ok = ok and got_rows == gold["customer_rows"]
                detail += f", rows {show(got_rows)} (expected {show(gold['customer_rows'])})"
            return ok, detail
        return check

    def customers_by_region(self, variant: str) -> Check:
        """Distinct customers per region, NULL regions reported as 'Unknown'."""
        expected = {d["key"]: d["customers"] for d in self.gold[variant]["customers_by_region"]}

        @answered
        def check(r):
            value = find_column(r.rows[0], "customers", "customer_count", contains=("customer", "count"))
            ok, problems = same_numbers(value_by_key(r.rows, label_column(r.rows), value), expected, 0)
            unknown = expected.get("Unknown")
            note = f", 'Unknown' {unknown}" if unknown else ""
            return ok, (f"{len(expected)} regions match{note}" if ok else problems)
        return check

    def segment_shares(self, variant: str) -> Check:
        expected = {d["key"]: d["pct_of_customers"] for d in self.gold[variant]["customers_by_segment"]}

        @answered
        def check(r):
            value = find_column(r.rows[0], "pct_of_customers", contains=("pct", "percent", "share"))
            got = value_by_key(r.rows, label_column(r.rows), value)
            ok, problems = same_numbers(got, expected, 0.05)
            total = round(sum(v or 0 for v in got.values()), 2)
            ok = ok and abs(total - 100) <= 0.1
            return ok, (f"{len(expected)} segments match, total {total}%" if ok else problems or f"total {total}%")
        return check

    def top_signup_region(self, variant: str, start: str, end: str) -> Check:
        """The region with the most signups. A full ranking or only the top row are both accepted."""
        frame = self.customers(variant)
        dates = pd.to_datetime(frame.signup_date)
        frame = frame[(dates >= start) & (dates <= end)]
        counts = frame.assign(region=frame.region.fillna("Unknown")).groupby("region").customer_id.nunique()
        expected = {k: int(v) for k, v in counts.items()}
        top = min(expected, key=lambda k: (-expected[k], k))              # ties ordered by name

        @answered
        def check(r):
            value = find_column(r.rows[0], "signups", contains=("signup", "sign_up", "customers", "count"))
            got = value_by_key(r.rows, label_column(r.rows), value)
            got_top = min(got, key=lambda k: (-(got[k] or 0), k))
            wrong = [k for k in got if k in expected and got[k] != expected[k]]
            ok = got_top == top and not wrong
            return ok, f"top {got_top} (expected {top}, {expected[top]} signups)" + (f"; wrong {wrong}" if wrong else "")
        return check

    def duplicates(self, variant: str) -> Check:
        """Customer IDs that repeat versus the extra rows they create: two different numbers."""
        frame = self.customers(variant)
        expected_rows = len(frame) - frame.customer_id.nunique()
        expected_ids = int((frame.customer_id.value_counts() > 1).sum())

        @answered
        def check(r):
            row = r.rows[0]
            rows_col = find_column(row, "duplicate_rows", "extra_rows", contains=("extra", "duplicate_rows"))
            ids_col = find_column(row, "duplicated_ids", "duplicate_ids", contains=("_ids",), exclude=("distinct",))
            got_rows, got_ids = as_float(row.get(rows_col)), as_float(row.get(ids_col))
            ok = got_rows == expected_rows and got_ids == expected_ids
            return ok, f"{show(got_ids)} IDs repeated, {show(got_rows)} extra rows (expected {expected_ids}, {expected_rows})"
        return check

    @staticmethod
    def no_data() -> Check:
        """A period with no data must be reported as no data, not as zero."""
        def check(r):
            ok = r.status == "no_data" or (r.status == "answered" and not r.rows)
            return ok, f"status {r.status}, {len(r.rows or [])} rows"
        return check

    # ---- order-item checks ----
    def net_sales_by_dimension(self, column: str, start: str, end: str, variant: str = "clean") -> Check:
        """Every group's net sales (booked orders, minus completed item refunds)."""
        expected = self.net_sales_by(variant, start, end, column)

        @answered
        def check(r):
            value = find_column(r.rows[0], "net_sales", contains=("net_sales", "sales", "revenue"))
            ok, problems = same_numbers(value_by_key(r.rows, label_column(r.rows), value), expected)
            return ok, (f"{len(expected)} groups match, total {sum(expected.values()):,.2f}" if ok else problems)
        return check

    def top_net_sales(self, column: str, start: str, end: str, variant: str = "clean") -> Check:
        """The group with the highest net sales. A full ranking or only the top row are both accepted."""
        expected = self.net_sales_by(variant, start, end, column)
        top = max(expected, key=expected.get)

        @answered
        def check(r):
            value = find_column(r.rows[0], "net_sales", contains=("net_sales", "sales", "revenue"))
            best = max(r.rows, key=lambda row: as_float(row.get(value)) or 0)
            got, amount = best.get(label_column(r.rows)), as_float(best.get(value))
            ok = got == top and amount is not None and abs(amount - expected[top]) <= 0.01
            return ok, f"top {got} {show(amount)} (expected {top}, {expected[top]:,.2f})"
        return check

    def average_order_value(self, start: str, end: str, variant: str = "clean") -> Check:
        """AOV = merchandise after discounts / distinct booked orders. One overall figure or a breakdown by region."""
        lines = self.booked_lines(variant, start, end)
        overall = round(lines.merch.sum() / lines.order_id.nunique(), 2)
        grouped = lines.groupby("region").agg(merch=("merch", "sum"), orders=("order_id", "nunique"))
        by_region = (grouped.merch / grouped.orders).round(2).to_dict()

        @answered
        def check(r):
            value = find_column(r.rows[0], "average_order_value", "aov", contains=("average", "aov"))
            if "region" in r.rows[0]:
                ok, problems = same_numbers(value_by_key(r.rows, "region", value), by_region)
                return ok, (f"{len(by_region)} regions match (overall would be {overall:,.2f})" if ok else problems)
            got = as_float(r.rows[0].get(value))
            ok = len(r.rows) == 1 and got is not None and abs(got - overall) <= 0.01
            return ok, f"overall AOV {show(got)} (expected {overall:,.2f} over {lines.order_id.nunique():,} orders)"
        return check

    def _month_over_month(self, month: str, variant: str = "clean") -> tuple[dict, float]:
        current = pd.Period(month, "M")
        previous = current - 1
        now = self.net_sales_by(variant, str(current.start_time.date()), str(current.end_time.date()), "region")
        before = self.net_sales_by(variant, str(previous.start_time.date()), str(previous.end_time.date()), "region")
        by_region = {k: round((now[k] - before[k]) / before[k] * 100, 2) for k in now if before.get(k)}
        total = round((sum(now.values()) - sum(before.values())) / sum(before.values()) * 100, 2)
        return by_region, total

    def mom_change(self, month: str, variant: str = "clean") -> Check:
        """Month-over-month % change in net sales for *month* (YYYY-MM): overall, or per region."""
        by_region, total = self._month_over_month(month, variant)

        @answered
        def check(r):
            pct = find_column(r.rows[0], "mom_pct", contains=("pct", "percent", "change"))
            if "region" in r.rows[0]:
                rows = [row for row in r.rows if str(row.get("month", month)) == month]
                ok, problems = same_numbers(value_by_key(rows, "region", pct), by_region, 0.05)
                return ok, (f"{len(by_region)} regions match (overall {total:+.2f}%)" if ok else problems)
            got = as_float(r.rows[-1].get(pct))
            ok = got is not None and abs(got - total) <= 0.05
            return ok, f"overall change {got} (expected {total:+.2f}%)"
        return check

    def biggest_mom_drop(self, month: str, variant: str = "clean") -> Check:
        """The region with the largest month-over-month percentage fall (a percentage, not an absolute difference)."""
        by_region, _ = self._month_over_month(month, variant)
        worst = min(by_region, key=by_region.get)

        @answered
        def check(r):
            pct = find_column(r.rows[0], "mom_pct", contains=("pct", "percent", "change"))
            known = [row for row in r.rows if pct and row.get(pct) is not None]
            got = min(known, key=lambda row: float(row[pct])).get("region") if known else None
            return got == worst, f"biggest drop {got} (expected {worst}, {by_region[worst]:+.2f}%)"
        return check

    def order_net_total(self, order_id: str, variant: str = "clean") -> Check:
        """Merchandise total of one order: the sum of its lines' net amounts, with no tax or shipping added."""
        lines = self.load(variant, "order_items")
        expected = round(lines[lines.order_id == order_id].net_amount.astype(float).sum(), 2)

        @answered
        def check(r):
            first = r.rows[0]
            if len(r.rows) > 1 or "order_item_id" in first:              # one row per order line
                value = find_column(first, "net_amount", contains=("net",))
                got = round(sum(as_float(row.get(value)) or 0 for row in r.rows), 2)
            else:                                                       # one total row
                value = find_column(first, "total_net_amount", "net_total", "net_amount",
                                    contains=("total", "net"), exclude=("tax", "shipping"))
                got = as_float(first.get(value))
            ok = got is not None and abs(got - expected) <= 0.01
            return ok, f"order net total {show(got)} (expected {expected:,.2f}; merchandise only, no tax or shipping)"
        return check
