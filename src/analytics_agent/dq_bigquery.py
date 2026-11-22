"""Deterministic data-quality rules executed as BigQuery SQL (Stage 5 backend).

Same rule IDs, severities and guidance as the local pandas engine
(analytics_agent.data_quality), so the Data Quality Reviewer agent, the
reports and the detection metrics work unchanged. Each rule is a SELECT that
returns the identifiers of offending records in a column named `id`; an empty
result means the rule passes. Schema rules use table metadata from the runner.

The SQL here is trusted, notebook-authored code (not model-generated), but it
still runs through the guarded runner with maximum_bytes_billed and a timeout.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from analytics_agent import schema_registry as sr
from analytics_agent.bq import SqlParam
from analytics_agent.data_quality import RULE_SPECS, MAX_EXAMPLES, RuleResult, ValidationReport, redact_identifier

TOL = "0.005"


def _in(values) -> str:
    return ", ".join("'" + v.replace("'", "\\'") + "'" for v in values)


def _required_not_null(table: str, id_col: str, t: str) -> str:
    cols = sr.get_table(table).required_columns
    cond = " OR ".join(f"{c} IS NULL" for c in cols)
    return f"SELECT DISTINCT IFNULL(CAST({id_col} AS STRING), 'None') AS id FROM {t} WHERE {cond}"


def _dupes(col: str, t: str) -> str:
    return f"SELECT CAST({col} AS STRING) AS id FROM {t} WHERE {col} IS NOT NULL GROUP BY {col} HAVING COUNT(*) > 1"


def _orphans(child: str, col: str, id_col: str, parent: str, pcol: str) -> str:
    return (f"SELECT DISTINCT CAST(c.{id_col} AS STRING) AS id FROM {child} c WHERE c.{col} IS NOT NULL "
            f"AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.{pcol} = c.{col})")


def build_rule_sql(refs: dict[str, str]) -> dict[str, str]:
    """{rule_id: SQL} for every SQL-backed rule. refs maps table name -> `project.dataset.table`."""
    c, p, o, i, pay, s, r = (refs[t] for t in ("customers", "products", "orders", "order_items", "payments", "shipments", "refunds"))
    OS, BOOKED = _in(sr.ORDER_STATUSES), _in(sr.BOOKED_ORDER_STATUSES)
    return {
        # customers
        "CUS_PK_UNIQUE": _dupes("customer_id", c),
        "CUS_REQUIRED_NOT_NULL": _required_not_null("customers", "customer_id", c),
        "CUS_REGION_ALLOWED": f"SELECT DISTINCT customer_id AS id FROM {c} WHERE region IS NOT NULL AND region NOT IN ({_in(sr.REGIONS)})",
        "CUS_EMAIL_UNIQUE": _dupes("email", c),
        # products
        "PRD_PK_UNIQUE": _dupes("product_id", p),
        "PRD_SKU_UNIQUE": _dupes("sku", p),
        "PRD_CATEGORY_ALLOWED": f"SELECT DISTINCT product_id AS id FROM {p} WHERE category IS NOT NULL AND category NOT IN ({_in(sr.CATEGORIES)})",
        "PRD_PRICE_VALID": f"SELECT DISTINCT product_id AS id FROM {p} WHERE list_price <= 0 OR unit_cost <= 0 OR unit_cost >= list_price",
        # orders
        "ORD_PK_UNIQUE": _dupes("order_id", o),
        "ORD_REQUIRED_NOT_NULL": _required_not_null("orders", "order_id", o),
        "ORD_STATUS_ALLOWED": f"SELECT DISTINCT order_id AS id FROM {o} WHERE order_status IS NOT NULL AND order_status NOT IN ({OS})",
        "ORD_CUSTOMER_FK": _orphans(o, "customer_id", "order_id", c, "customer_id"),
        "ORD_TOTAL_CONSISTENT": (f"SELECT DISTINCT order_id AS id FROM {o} WHERE order_status IN ({OS}) AND "
                                 f"ABS((items_subtotal - discount_amount + shipping_amount + tax_amount) - order_total) > {TOL}"),
        "ORD_SUBTOTAL_MATCHES_ITEMS": (f"SELECT DISTINCT ord.order_id AS id FROM {o} ord LEFT JOIN "
                                       f"(SELECT order_id, SUM(gross_amount) AS s FROM {i} GROUP BY order_id) li ON li.order_id = ord.order_id "
                                       f"WHERE ord.order_status IN ({OS}) AND ABS(IFNULL(li.s, 0) - ord.items_subtotal) > {TOL}"),
        "ORD_CANCELLED_TS_CONSISTENT": (f"SELECT DISTINCT order_id AS id FROM {o} WHERE order_status IN ({OS}) AND "
                                        f"((order_status = 'cancelled') != (cancelled_timestamp IS NOT NULL))"),
        "ORD_NOT_FUTURE": f"SELECT DISTINCT order_id AS id FROM {o} WHERE order_timestamp > @as_of",
        # order_items
        "ITM_PK_UNIQUE": _dupes("order_item_id", i),
        "ITM_LINE_UNIQUE": (f"SELECT DISTINCT order_item_id AS id FROM (SELECT order_item_id, "
                            f"COUNT(*) OVER (PARTITION BY order_id, line_number) AS n FROM {i}) WHERE n > 1"),
        "ITM_ORDER_FK": _orphans(i, "order_id", "order_item_id", o, "order_id"),
        "ITM_PRODUCT_FK": _orphans(i, "product_id", "order_item_id", p, "product_id"),
        "ITM_QUANTITY_RANGE": f"SELECT DISTINCT order_item_id AS id FROM {i} WHERE quantity < 1 OR quantity > 10",
        "ITM_AMOUNTS_CONSISTENT": (f"SELECT DISTINCT order_item_id AS id FROM {i} WHERE ABS(quantity * unit_price - gross_amount) > {TOL} "
                                   f"OR ABS(gross_amount - discount_amount - net_amount) > {TOL}"),
        # payments
        "PAY_PK_UNIQUE": _dupes("payment_id", pay),
        "PAY_TXN_UNIQUE": _dupes("provider_transaction_id", pay),
        "PAY_ORDER_FK": _orphans(pay, "order_id", "payment_id", o, "order_id"),
        "PAY_STATUS_ALLOWED": f"SELECT DISTINCT payment_id AS id FROM {pay} WHERE payment_status IS NOT NULL AND payment_status NOT IN ({_in(sr.PAYMENT_STATUSES)})",
        "PAY_AFTER_ORDER": (f"SELECT DISTINCT pm.payment_id AS id FROM {pay} pm JOIN {o} ord ON ord.order_id = pm.order_id "
                            f"WHERE pm.payment_timestamp < ord.order_timestamp"),
        "PAY_CAPTURED_MATCHES_ORDER": (f"WITH cap AS (SELECT order_id, COUNT(*) AS n, SUM(amount) AS amt FROM {pay} "
                                       f"WHERE payment_status = 'captured' GROUP BY order_id) "
                                       f"SELECT DISTINCT ord.order_id AS id FROM {o} ord LEFT JOIN cap ON cap.order_id = ord.order_id "
                                       f"WHERE ord.order_status IN ({BOOKED}) AND (IFNULL(cap.n, 0) != 1 OR ABS(IFNULL(cap.amt, 0) - ord.order_total) > {TOL})"),
        "PAY_FAILURE_REASON_CONSISTENT": (f"SELECT DISTINCT payment_id AS id FROM {pay} WHERE payment_status IN ({_in(sr.PAYMENT_STATUSES)}) "
                                          f"AND ((payment_status = 'failed') != (failure_reason IS NOT NULL))"),
        # shipments
        "SHP_PK_UNIQUE": _dupes("shipment_id", s),
        "SHP_ONE_PER_ORDER": _dupes("order_id", s),
        "SHP_ORDER_FK": _orphans(s, "order_id", "shipment_id", o, "order_id"),
        "SHP_DELIVERED_TS_CONSISTENT": (f"SELECT DISTINCT shipment_id AS id FROM {s} WHERE shipment_status IN ({_in(sr.SHIPMENT_STATUSES)}) "
                                        f"AND ((shipment_status = 'delivered') != (delivered_timestamp IS NOT NULL))"),
        "SHP_DELIVERY_AFTER_SHIP": (f"SELECT DISTINCT shipment_id AS id FROM {s} WHERE delivered_timestamp IS NOT NULL "
                                    f"AND delivered_timestamp < shipped_timestamp"),
        "SHP_STATUS_MATCHES_ORDER": (f"SELECT DISTINCT sh.shipment_id AS id FROM {s} sh JOIN {o} ord ON ord.order_id = sh.order_id "
                                     f"WHERE ord.order_status IN ({OS}) AND sh.shipment_status IN ({_in(sr.SHIPMENT_STATUSES)}) "
                                     f"AND ord.order_status != CASE sh.shipment_status WHEN 'delivered' THEN 'delivered' "
                                     f"WHEN 'in_transit' THEN 'shipped' END"),
        # refunds
        "REF_PK_UNIQUE": _dupes("refund_id", r),
        "REF_ORDER_FK": _orphans(r, "order_id", "refund_id", o, "order_id"),
        "REF_ITEM_FK": _orphans(r, "order_item_id", "refund_id", i, "order_item_id"),
        "REF_PAYMENT_FK": _orphans(r, "payment_id", "refund_id", pay, "payment_id"),
        "REF_WITHIN_REFUNDABLE": (f"WITH li AS (SELECT order_item_id, ANY_VALUE(net_amount) AS net FROM {i} GROUP BY order_item_id), "
                                  f"over_items AS (SELECT rf.order_item_id FROM {r} rf JOIN li ON li.order_item_id = rf.order_item_id "
                                  f"GROUP BY rf.order_item_id, li.net HAVING SUM(rf.refund_amount) - li.net > {TOL}), "
                                  f"pays AS (SELECT payment_id, ANY_VALUE(amount) AS amount FROM {pay} GROUP BY payment_id) "
                                  f"SELECT DISTINCT rf.refund_id AS id FROM {r} rf WHERE rf.order_item_id IN (SELECT order_item_id FROM over_items) "
                                  f"UNION DISTINCT "
                                  f"SELECT DISTINCT rf.refund_id AS id FROM {r} rf JOIN pays ON pays.payment_id = rf.payment_id "
                                  f"WHERE rf.order_item_id IS NULL AND rf.refund_amount - pays.amount > {TOL}"),
        "REF_ITEM_NULL_ONLY_CANCEL": (f"SELECT DISTINCT refund_id AS id FROM {r} WHERE "
                                      f"((order_item_id IS NULL) != (refund_reason = 'order_cancelled'))"),
        # soft (warning) rules
        # lag = whole elapsed 24-hour periods (seconds DIV 86400): same meaning in every engine
        "FRESH_ORDERS": (f"SELECT CONCAT('latest_order=', FORMAT_TIMESTAMP('%Y-%m-%d', MAX(order_timestamp)), ';lag_days=', "
                         f"CAST(DIV(TIMESTAMP_DIFF(@as_of, MAX(order_timestamp), SECOND), 86400) AS STRING)) AS id FROM {o} "
                         f"HAVING DIV(TIMESTAMP_DIFF(@as_of, MAX(order_timestamp), SECOND), 86400) > 7"),
        "VOL_MONTHLY_ORDERS": (f"WITH m AS (SELECT FORMAT_DATE('%Y-%m', order_date) AS month, COUNT(*) AS n FROM {o} GROUP BY month), "
                               f"w AS (SELECT month, n, AVG(n) OVER (ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS trailing, "
                               f"COUNT(*) OVER (ORDER BY month ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS k FROM m) "
                               f"SELECT month AS id FROM w WHERE k = 3 AND ABS(n - trailing) / trailing > 0.5"),
    }


def _schema_violations(live: dict | None, table: str) -> set:
    if live is None:
        return {"missing:table"}
    expected = {c.name: c.bq_type for c in sr.get_table(table).columns}
    out = {f"missing:{c}" for c in expected if c not in live} | {f"unexpected:{c}" for c in live if c not in expected}
    out |= {f"type:{c}:{expected[c]}->{live[c]}" for c in expected if c in live and live[c] != expected[c]}
    return out


def validate_dataset(runner, dataset: str, as_of, *, dataset_label: str | None = None,
                     progress: Callable[[str], None] | None = None) -> ValidationReport:
    """Run every rule against one BigQuery dataset (or the DuckDB emulator in tests)."""
    settings = runner.settings
    refs = settings.refs(dataset)
    sql_by_rule = build_rule_sql(refs)
    params = [SqlParam("as_of", "TIMESTAMP", as_of.to_pydatetime() if hasattr(as_of, "to_pydatetime") else as_of)]
    report = ValidationReport(
        run_id=f"dq_bq_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}",
        run_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_label=dataset_label or dataset, source=f"{runner.backend}:{settings.project_id}.{dataset}", as_of=str(as_of))
    counts_sql = "SELECT " + ", ".join(f"(SELECT COUNT(*) FROM {ref}) AS {t}" for t, ref in refs.items())
    counts = runner.run(counts_sql, max_rows=1, dry_run_first=False)
    row_counts = counts.rows[0] if counts.ok and counts.rows else {}
    for spec in RULE_SPECS:
        base = dict(rule_id=spec.rule_id, table=spec.table, severity=spec.severity, description=spec.description,
                    expected_condition=spec.expected_condition, rows_checked=int(row_counts.get(spec.table) or 0),
                    recommended_action=spec.recommended_action, business_impact=spec.business_impact,
                    affected_metrics=list(spec.affected_metrics))
        if progress:
            progress(spec.rule_id)
        if spec.rule_id.startswith("SCHEMA_COLUMNS_"):
            ids = _schema_violations(runner.table_schema(dataset, spec.table), spec.table)
        else:
            sql = sql_by_rule[spec.rule_id]
            res = runner.run(sql, params if "@as_of" in sql else None, max_rows=10_000, dry_run_first=False)
            if not res.ok:
                report.results.append(RuleResult(status="error", observed_count=0, example_ids=[],
                                                 error_message=f"[{res.error_kind}] {res.error_message}"[:300], **base))
                continue
            ids = {str(row["id"]) for row in res.rows}
        status = "pass" if not ids else ("fail" if spec.kind == "fail_rule" else "warning")
        report._offending[spec.rule_id] = ids
        report.results.append(RuleResult(status=status, observed_count=len(ids),
                                         example_ids=[redact_identifier(x) for x in sorted(ids)[:MAX_EXAMPLES]], **base))
    return report


def parity_with_local(bq_report: ValidationReport, local_report: ValidationReport) -> list[dict]:
    """Compare a BigQuery report with the local pandas report rule by rule (record-level for fail rules)."""
    rows = []
    for r in bq_report.results:
        lr = local_report.get(r.rule_id)
        b, l = bq_report.offending_ids(r.rule_id), local_report.offending_ids(r.rule_id)
        warning_rule = r.rule_id in ("VOL_MONTHLY_ORDERS",)       # id formats differ; compare counts
        schema_rule = r.rule_id.startswith("SCHEMA_COLUMNS_")
        same = (r.status == lr.status) and (len(b) == len(l) if (warning_rule or schema_rule) else b == l)
        rows.append({"rule_id": r.rule_id, "bigquery": r.status, "local": lr.status, "bq_count": len(b),
                     "local_count": len(l), "match": same})
    return rows
