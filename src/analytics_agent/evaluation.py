"""Compare agent results with the independently computed gold answers.

The agent may name columns differently ("net_sales", "total_net_sales", ...) and may
express rates as fractions or percentages, so matching is by content:
  * mapping cases: some string column must contain exactly the gold keys and some numeric
    column must match the gold values within tolerance (x100 allowed for rates);
  * scalar cases: some numeric cell must match the gold value within tolerance (x100 allowed);
  * top cases: the first row must contain the expected key;
  * empty cases: the agent must return no rows (for example a period outside the data).
Gold answers come from deterministic pandas code (data_generation/expected_results.py), never an LLM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldCase:
    test_id: str
    question: str
    kind: str                 # mapping | scalar | top | empty
    key: str = ""             # gold key field (mapping/top)
    value: str = ""           # gold value field
    tolerance: float = 0.011
    rate: bool = False        # allow fraction <-> percent


EVAL_SPECS = [
    GoldCase("G01", "What was our monthly net sales over the last six months?", "mapping", "month", "net_sales"),
    GoldCase("G02", "Which product categories generate the most sales?", "mapping", "category", "net_sales"),
    GoldCase("G03", "What is the average order value by region?", "mapping", "region", "aov"),
    GoldCase("G05", "What percentage of orders were delivered late?", "scalar", value="late_rate", tolerance=0.0006, rate=True),
    GoldCase("G06", "Which region experienced the largest month-over-month net sales decline in June 2026?", "top", key="largest_decline_region"),
    GoldCase("G08", "How many orders do we have in each status?", "mapping", "order_status", "orders"),
    GoldCase("G11", "What share of orders had a refund?", "scalar", value="order_refund_rate", tolerance=0.0006, rate=True),
    GoldCase("G12", "How does late delivery rate compare across carriers?", "mapping", "carrier", "late_rate", 0.0006, True),
    GoldCase("G14", "What is our order cancellation rate?", "scalar", value="cancellation_rate", tolerance=0.0006, rate=True),
    GoldCase("G16", "How many customers placed at least one order in Q2 2026?", "scalar", value="active_customers", tolerance=0.0),
    GoldCase("G19", "What is our payment failure rate?", "scalar", value="failure_rate", tolerance=0.0006, rate=True),
    GoldCase("G20", "What were captured payments by month in H1 2026?", "mapping", "month", "captured_payments"),
    # ---- hard cases: relative periods, exclusions, rankings, different wording, customers vs orders, empty results
    GoldCase("G21", "Excluding tax and shipping, what was the average order value by region last quarter?",
             "mapping", "region", "aov"),
    GoldCase("G22", "In Q1 2026, how did net sales split across customer segments?", "mapping", "segment", "net_sales"),
    GoldCase("G23", "Which sales channel brought in the most net sales last quarter?", "top", key="top_channel"),
    GoldCase("G24", "Last month, what share of delivered orders arrived late for each carrier?",
             "mapping", "carrier", "late_rate", 0.0006, True),
    GoldCase("G25", "Which product category had the highest net sales in June 2026?", "top", key="top_category"),
    GoldCase("G26", "How many new customers signed up in each region during the first half of 2026?",
             "mapping", "region", "signups", 0.0),
    GoldCase("G27", "What percentage of customers are in each segment?", "mapping", "segment", "pct_of_customers"),
    GoldCase("G28", "How many distinct customers placed at least one booked order in the first half of 2026?",
             "scalar", value="active_customers", tolerance=0.0),
    GoldCase("G29", "What were net sales by product category in 2027?", "empty"),
]


def load_gold(path: str | Path) -> dict:
    return {c["test_id"]: c for c in json.loads(Path(path).read_text())["cases"]}


def _gold_mapping(case: GoldCase, gold_result) -> dict:
    if isinstance(gold_result, dict):           # e.g. G08 status counts {status: n}
        return {str(k): float(v) for k, v in gold_result.items()}
    return {str(r[case.key]): float(r[case.value]) for r in gold_result}


def _close(a: float, b: float, tol: float, rate: bool) -> bool:
    if abs(a - b) <= tol:
        return True
    return rate and (abs(a - b * 100) <= tol * 100 or abs(a * 100 - b) <= tol * 100)


def compare(case: GoldCase, gold_result, rows: list[dict]) -> tuple[bool, str]:
    if case.kind == "empty":
        return (not rows), ("no rows, as expected" if not rows else f"expected no rows, got {len(rows)}")
    if not rows:
        return False, "no rows"
    if case.kind == "scalar":
        target = float(gold_result[case.value] if isinstance(gold_result, dict) else gold_result)
        nums = [float(v) for r in rows for v in r.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        ok = any(_close(n, target, case.tolerance, case.rate) for n in nums)
        return ok, f"gold {target} | agent numbers {nums[:6]}"
    if case.kind == "top":
        target = str(gold_result[case.key]).lower()
        first = {str(v).lower() for v in rows[0].values()}
        return target in first, f"gold {target} | agent first row {rows[0]}"
    gold = _gold_mapping(case, gold_result)
    keys_lower = {k.lower() for k in gold}
    cols = list(rows[0].keys())
    key_cols = [c for c in cols if {str(r.get(c)).lower() for r in rows} >= keys_lower]
    if not key_cols:
        return False, f"no column contains all gold keys {sorted(gold)[:5]}"
    kc = key_cols[0]
    for vc in cols:
        if vc == kc:
            continue
        try:
            agent = {str(r[kc]).lower(): float(r[vc]) for r in rows if r.get(vc) is not None}
        except (TypeError, ValueError):
            continue
        if all(k.lower() in agent and _close(agent[k.lower()], v, case.tolerance, case.rate) for k, v in gold.items()):
            return True, f"matched on {kc} -> {vc}"
    return False, f"keys matched on {kc}, but no numeric column matches the gold values"