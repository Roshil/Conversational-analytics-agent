"""Descriptive Markdown report: canonical semantic-layer metrics vs independently computed gold answers.

Ported from the user's notebook cell (Part 3 report). Every number in the report comes from the rows
BigQuery (or the emulator) returned; descriptions are computed deterministically, not by an LLM.
"""
from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analytics_agent import semantic_layer as sem
from analytics_agent.bq import SqlParam
from analytics_agent.evaluation import EVAL_SPECS, compare

GOLD_CHECKS = [("G01", "net_sales", "2026-01-01", "2026-06-30"), ("G02", "net_sales_by_dimension", *sem.DEFAULT_WINDOW),
               ("G03", "average_order_value", *sem.DEFAULT_WINDOW), ("G05", "late_delivery_rate", *sem.DEFAULT_WINDOW),
               ("G06", "mom_change", "2026-05-01", "2026-06-30"), ("G08", "order_status_counts", *sem.DEFAULT_WINDOW),
               ("G11", "order_refund_rate", *sem.DEFAULT_WINDOW), ("G14", "cancellation_rate", *sem.DEFAULT_WINDOW),
               ("G16", "active_customers", "2026-04-01", "2026-06-30"), ("G19", "payment_failure_rate", *sem.DEFAULT_WINDOW),
               ("G20", "captured_payments", "2026-01-01", "2026-06-30")]
GOLD_TO_COLUMN = {"aov": "average_order_value"}


def fmt(col: str, v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if "rate" in col:
        return f"{v * 100:.2f}%" if abs(v) <= 1 else f"{v:,.2f}%"
    if col.endswith("_pct"):
        return f"{v:+.2f}%"
    return f"{v:,}" if isinstance(v, int) else f"{v:,.2f}"


def md_table(rows: list, max_rows: int = 30) -> str:
    if not rows:
        return "_no rows_"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(c, r.get(c)) for c in cols) + " |" for r in rows[:max_rows]]
    return "\n".join(out) + (f"\n\n_showing {max_rows} of {len(rows)} rows_" if len(rows) > max_rows else "")


def describe(metric: str, rows: list) -> list:
    """Plain-language observations computed from the rows (highest/lowest, totals, first-to-last trend)."""
    if not rows:
        return ["The query returned no rows."]
    cols = list(rows[0])
    is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    keys = [c for c in cols if any(isinstance(r.get(c), str) for r in rows)]
    nums = [c for c in cols if any(is_num(r.get(c)) for r in rows)]
    if len(rows) == 1 or not keys or not nums:
        return [", ".join(f"**{c.replace('_', ' ')}** = {fmt(c, rows[0].get(c))}" for c in cols) + "."]
    k = keys[0]
    main = "mom_pct" if "mom_pct" in nums else "net_sales" if "net_sales" in nums else next((c for c in nums if "rate" in c), nums[0])
    valid = [r for r in rows if r.get(main) is not None]
    name = lambda r: r.get(k) if r.get(k) is not None else "unknown"
    hi, lo = max(valid, key=lambda r: r[main]), min(valid, key=lambda r: r[main])
    obs = [f"{len(rows)} {k.replace('_', ' ')} values were returned.",
           f"Highest {main.replace('_', ' ')}: **{name(hi)}** with {fmt(main, hi[main])}; lowest: **{name(lo)}** with {fmt(main, lo[main])}."]
    if not any(x in main for x in ("rate", "pct", "average")):
        obs.append(f"Total {main.replace('_', ' ')} across all rows: {fmt(main, sum(r[main] for r in valid))}.")
    if k == "month" and len(valid) >= 2:
        first, last = valid[0], valid[-1]
        change = (last[main] - first[main]) / first[main] * 100 if first[main] else None
        direction = "rose" if last[main] > first[main] else "fell" if last[main] < first[main] else "was flat"
        obs.append(f"From {first[k]} to {last[k]}, {main.replace('_', ' ')} {direction} from {fmt(main, first[main])} to "
                   f"{fmt(main, last[main])}" + (f" ({change:+.1f}%)." if change is not None else "."))
    return obs


def gold_comparison(spec, gold_result, rows: list) -> str:
    if spec.kind == "scalar":
        col = GOLD_TO_COLUMN.get(spec.value, spec.value)
        got = next((r.get(col) for r in rows if col in r), None)
        return f"| measure | result | gold |\n|---|---|---|\n| {col} | {fmt(col, got)} | {fmt(col, gold_result[spec.value])} |"
    if spec.kind == "top":
        return f"| measure | result | gold |\n|---|---|---|\n| largest decline | {rows[0].get('region') if rows else '-'} | {gold_result[spec.key]} |"
    gmap = {str(k): v for k, v in gold_result.items()} if isinstance(gold_result, dict) else {str(r[spec.key]): r[spec.value] for r in gold_result}
    key_col = next((c for c in rows[0] if {str(r.get(c)) for r in rows} >= set(gmap)), None) if rows else None
    val_col = GOLD_TO_COLUMN.get(spec.value, spec.value)
    if key_col is None or val_col not in rows[0]:
        return "_could not align the result with the gold answer_"
    got = {str(r[key_col]): r[val_col] for r in rows}
    lines = [f"| {key_col} | result {val_col} | gold {spec.value} | difference |", "|---|---|---|---|"]
    for key, gv in gmap.items():
        bv = got.get(key)
        lines.append(f"| {key} | {fmt(val_col, bv)} | {fmt(val_col, gv)} | "
                     f"{fmt(val_col, bv - gv) if isinstance(bv, (int, float)) else '-'} |")
    return "\n".join(lines)


def semantic_layer_report(runner, settings, dataset: str, gold: dict, out_path: str | Path = "reports/semantic_layer_vs_gold.md"):
    """Run every canonical metric, compare with gold, write a descriptive Markdown report. Returns (summary_df, sections, path)."""
    spec = {s.test_id: s for s in EVAL_SPECS}
    refs = settings.refs(dataset)
    out, sections = [], []
    for gid, metric, a, b in GOLD_CHECKS:
        m = sem.METRICS_BY_NAME[metric]
        params = [SqlParam("start_date", "DATE", dt.date.fromisoformat(a)), SqlParam("end_date", "DATE", dt.date.fromisoformat(b))]
        res = runner.run(m.render(refs), params, allowed_tables=settings.allowed_tables(dataset))
        ok, why = compare(spec[gid], gold[gid]["expected_result"], res.rows) if res.ok else (False, res.error_message)
        out.append({"gold case": gid, "metric": metric, "result": "PASS" if ok else "FAIL", "bytes billed": res.bytes_billed, "detail": why[:80]})
        period = "full snapshot (no date filter)" if metric == "order_status_counts" else f"{a} to {b}"
        sec = [f"## {gid} — {m.label} — {'✅ PASS' if ok else '❌ FAIL'}", "", f"- **Question:** {spec[gid].question}",
               f"- **Metric:** `{metric}` (grain: {m.grain})", f"- **Definition:** {m.definition}", f"- **Period:** {period}",
               f"- **Execution:** {res.status} · {res.row_count} rows · bytes billed {res.bytes_billed} · job `{res.job_id}`", ""]
        if res.ok:
            sec += ["### What the data shows", ""] + [f"- {o}" for o in describe(metric, res.rows)] + [""]
            sec += ["### Result", "", md_table(res.rows), "", "### Comparison with the gold answer", "",
                    gold_comparison(spec[gid], gold[gid]["expected_result"], res.rows), "", f"_Match detail: {why}_", ""]
        else:
            sec += [f"**Query failed:** [{res.error_kind}] {res.error_message}", ""]
        sec += ["<details><summary>SQL</summary>", "", "```sql", m.render(refs), "```", "", "</details>", ""]
        sections.append("\n".join(sec))
    summary = pd.DataFrame(out)
    passed = int((summary.result == "PASS").sum())
    header = ["# Semantic layer vs gold answers", "", f"- **Run:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
              f"- **Backend:** {runner.backend}" + (" (OFFLINE emulator - not BigQuery)" if runner.backend != "bigquery" else ""),
              f"- **Dataset:** `{settings.project_id}.{dataset}`", f"- **Result:** {passed}/{len(out)} metrics match the gold answers",
              f"- **Total bytes billed:** {sum(r['bytes billed'] or 0 for r in out):,}", "",
              "Gold answers were computed independently in pandas, never by an LLM.", ""]
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header) + "\n---\n\n" + "\n---\n\n".join(sections), encoding="utf-8")
    return summary, sections, path
