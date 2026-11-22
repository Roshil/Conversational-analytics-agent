"""Deterministic data-quality engine.

Runs stable, identified rules against a dataset and produces an auditable
report (JSON + Markdown). This is the source of truth for data quality: an
LLM reviewer may *explain* the report but can never change a status.

Backends
--------
* local (this module): pandas over the generated Parquet/CSV files. Used now,
  before BigQuery is connected, and as the oracle for the SQL backend.
* bigquery (Stage 5): the same rule IDs implemented in BigQuery SQL.

Statuses
--------
pass     rule ran, condition holds
fail     rule ran, condition violated (a confirmed defect)
warning  rule ran, a soft threshold was crossed (suspicious, not proven wrong)
error    rule could not run (missing table/column, exception) - NOT a pass
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from analytics_agent import schema_registry as sr
from analytics_agent.data_generation.generator import normalize_dtypes
from analytics_agent.data_generation.reference_checks import RULES as REFERENCE_RULES

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", re.I)
MAX_EXAMPLES = 5


# ---------------------------------------------------------------- rule metadata ----
@dataclass(frozen=True)
class RuleSpec:
    """Immutable specification of one deterministic data-quality rule.

    Attributes:
        rule_id: Stable identifier shared with the BigQuery backend and the
            defect catalog (e.g. ``"CUS_PK_UNIQUE"``).
        table: Primary table the rule inspects.
        severity: Relative importance (``"low"``, ``"medium"``, ``"high"``,
            ``"critical"``).
        description: Short human-readable description of the rule.
        expected_condition: Statement of what must hold for the rule to pass.
        kind: ``"fail_rule"`` (non-empty result → status ``fail``) or
            ``"warning_rule"`` (non-empty result → status ``warning``).
        fn: Callable ``(tables: dict[str, DataFrame], as_of: Timestamp) -> set``
            that returns the set of offending identifiers (or diagnostic tokens).
        recommended_action: Concrete remediation advice shown in the report.
        business_impact: One-sentence explanation of why the finding matters.
        affected_metrics: Tuple of metric names that are distorted when the
            rule fails.
    """

    rule_id: str
    table: str
    severity: str
    description: str
    expected_condition: str
    kind: str  # fail_rule or warning_rule
    fn: Callable[[dict, pd.Timestamp], set]
    recommended_action: str
    business_impact: str
    affected_metrics: tuple = ()


# What each failure means for the business and what to do. Kept in code (not in
# a prompt) so the deterministic reviewer and the LLM reviewer share one source.
RULE_GUIDANCE: dict[str, tuple[str, str, tuple]] = {
    # rule_id: (recommended_action, business_impact, affected_metrics)
    "CUS_PK_UNIQUE": (
        "Deduplicate customers in the source CRM sync; keep the latest record per customer_id and add a uniqueness check to the load.",
        "Joins from orders to customers duplicate those customers' orders, inflating regional/segment sales and AOV.",
        ("net sales by region", "net sales by segment", "AOV by region"),
    ),
    "CUS_REQUIRED_NOT_NULL": (
        "Backfill missing required customer attributes from the source system; reject loads with NULL required fields.",
        "Orders of these customers fall into an 'unknown' bucket in regional reports.",
        ("net sales by region", "AOV by region"),
    ),
    "CUS_REGION_ALLOWED": (
        "Map non-canonical regions to the allowed list at ingestion.",
        "Phantom regions split regional totals.",
        ("net sales by region",),
    ),
    "CUS_EMAIL_UNIQUE": (
        "Investigate together with CUS_PK_UNIQUE; merge accounts sharing an email.",
        "Customer counts overstated; possible duplicate identities.",
        ("customer counts",),
    ),
    "PRD_PK_UNIQUE": (
        "Deduplicate the product catalog.",
        "Joins to products duplicate order lines.",
        ("net sales by category",),
    ),
    "PRD_SKU_UNIQUE": (
        "Assign unique SKUs; investigate catalog merges.",
        "SKU-level reporting merges unrelated products.",
        ("SKU/product reports",),
    ),
    "PRD_CATEGORY_ALLOWED": (
        "Map category values to the canonical list (typo/case/spelling variants) and add an allowed-values constraint.",
        "Category totals split into phantom categories; canonical categories are under-reported.",
        ("net sales by category",),
    ),
    "PRD_PRICE_VALID": (
        "Correct prices/costs in the catalog.",
        "Margin and price analyses are wrong.",
        ("margin",),
    ),
    "ORD_PK_UNIQUE": (
        "Deduplicate orders.",
        "Order counts and sales double-counted.",
        ("all order metrics",),
    ),
    "ORD_REQUIRED_NOT_NULL": (
        "Reject or repair orders with NULL required fields.",
        "Orders may be dropped from filters.",
        ("all order metrics",),
    ),
    "ORD_STATUS_ALLOWED": (
        "Map non-canonical statuses (e.g. 'complete' -> 'delivered') at ingestion after confirming with the order system owner.",
        "Orders with unknown statuses are excluded by booked-status filters, understating sales.",
        ("net sales", "order counts", "AOV"),
    ),
    "ORD_CUSTOMER_FK": (
        "Find the missing customers in the source or quarantine the orders; enforce the FK in the pipeline.",
        "INNER JOINs to customers silently drop these orders; LEFT JOINs put them in a NULL region.",
        ("net sales by region", "AOV by region"),
    ),
    "ORD_TOTAL_CONSISTENT": (
        "Recompute order header components from source; check tax calculation.",
        "Tax/shipping reporting is wrong; order_total no longer equals its parts.",
        ("tax reporting", "reconciliation"),
    ),
    "ORD_SUBTOTAL_MATCHES_ITEMS": (
        "Compare header and lines; usually caused by duplicated or missing order lines.",
        "Item-based and header-based totals disagree.",
        ("net sales", "reconciliation"),
    ),
    "ORD_CANCELLED_TS_CONSISTENT": (
        "Fix cancellation timestamps.",
        "Cancellation timing analyses wrong.",
        ("cancellation rate",),
    ),
    "ORD_NOT_FUTURE": (
        "Check clocks/timezones in the order system.",
        "Future-dated orders distort period totals.",
        ("monthly trends",),
    ),
    "ITM_PK_UNIQUE": (
        "Remove duplicate order lines (exact re-loads) and make the load idempotent.",
        "Item-based sales metrics are inflated by the duplicated lines (fan-out).",
        ("net sales", "net sales by category", "AOV"),
    ),
    "ITM_LINE_UNIQUE": (
        "Same root cause as ITM_PK_UNIQUE in most cases.",
        "Duplicate lines within an order.",
        ("net sales",),
    ),
    "ITM_ORDER_FK": (
        "Quarantine order lines whose order does not exist; trace the loading job.",
        "Item-only queries over-report sales; joined queries silently drop the lines.",
        ("item-level sales",),
    ),
    "ITM_PRODUCT_FK": (
        "Find missing products or quarantine the lines.",
        "Category attribution impossible for these lines.",
        ("net sales by category",),
    ),
    "ITM_QUANTITY_RANGE": (
        "Correct quantities from the source order system; reject non-positive quantities.",
        "Unit totals understated; quantity no longer consistent with amounts.",
        ("units sold", "basket size"),
    ),
    "ITM_AMOUNTS_CONSISTENT": (
        "Recompute line amounts or fix quantities.",
        "Line money no longer reconciles.",
        ("net sales",),
    ),
    "PAY_PK_UNIQUE": (
        "Deduplicate payments.",
        "Cash overstated.",
        ("captured payments",),
    ),
    "PAY_TXN_UNIQUE": (
        "Deduplicate payments by provider_transaction_id and reconcile with the payment provider's settlement report.",
        "Total payments received is overstated by the duplicated amounts.",
        ("captured payments", "net cash", "reconciliation"),
    ),
    "PAY_ORDER_FK": (
        "Trace orphan payments with finance.",
        "Unattributable cash.",
        ("reconciliation",),
    ),
    "PAY_STATUS_ALLOWED": (
        "Map payment statuses.",
        "Payments excluded from status filters.",
        ("captured payments",),
    ),
    "PAY_AFTER_ORDER": (
        "Check timezone handling in the payment ETL.",
        "Payment-date reports shift cash into earlier periods.",
        ("captured payments by month",),
    ),
    "PAY_CAPTURED_MATCHES_ORDER": (
        "Reconcile captured payments to orders; usually caused by duplicate payments.",
        "Payments no longer reconcile to order totals.",
        ("captured payments", "reconciliation"),
    ),
    "PAY_FAILURE_REASON_CONSISTENT": (
        "Fix failure reasons.",
        "Failure analyses incomplete.",
        ("payment failure rate",),
    ),
    "SHP_PK_UNIQUE": (
        "Deduplicate shipments.",
        "Delivery metrics double-count.",
        ("late delivery rate",),
    ),
    "SHP_ONE_PER_ORDER": (
        "Confirm whether split shipments are now allowed; update the model if so.",
        "Assumption of one shipment per order broken.",
        ("late delivery rate",),
    ),
    "SHP_ORDER_FK": (
        "Quarantine orphan shipments.",
        "Unattributable shipments.",
        ("late delivery rate",),
    ),
    "SHP_DELIVERED_TS_CONSISTENT": (
        "Backfill delivered timestamps from the carrier feed.",
        "These shipments silently drop out of on-time/late calculations.",
        ("late delivery rate", "on-time rate"),
    ),
    "SHP_DELIVERY_AFTER_SHIP": (
        "Check carrier feed timestamps/timezones.",
        "Delivery-time metrics become negative or flattering; on-time rate overstated.",
        ("late delivery rate", "on-time rate"),
    ),
    "SHP_STATUS_MATCHES_ORDER": (
        "Align order status with shipment status.",
        "Status-based reports disagree.",
        ("order counts",),
    ),
    "REF_PK_UNIQUE": (
        "Deduplicate refunds.",
        "Refunds double-counted.",
        ("net sales", "refund rate"),
    ),
    "REF_ORDER_FK": (
        "Trace orphan refunds with finance.",
        "Unattributable refunds.",
        ("refund rate",),
    ),
    "REF_ITEM_FK": (
        "Trace refunds to missing lines.",
        "Refunds cannot be attributed to products.",
        ("product refund rate",),
    ),
    "REF_PAYMENT_FK": (
        "Trace refunds to missing payments.",
        "Refunds cannot be reconciled.",
        ("reconciliation",),
    ),
    "REF_WITHIN_REFUNDABLE": (
        "Freeze and audit these refunds with finance (possible error or fraud); enforce refund <= refundable amount.",
        "Net sales understated; potential financial loss.",
        ("net sales", "refund rate", "product refund rate"),
    ),
    "REF_ITEM_NULL_ONLY_CANCEL": (
        "Fix refund attribution.",
        "Order-level refunds cannot be allocated to products.",
        ("net sales by category",),
    ),
    "SCHEMA_COLUMNS": (
        "Align the table with the approved schema before any analysis.",
        "Queries may fail or read the wrong columns.",
        ("all metrics",),
    ),
    "FRESH_ORDERS": (
        "Check the order load job; data may be stale.",
        "Recent periods are incomplete; latest-month comparisons are misleading.",
        ("monthly trends", "MoM change"),
    ),
    "VOL_MONTHLY_ORDERS": (
        "Investigate the month with abnormal volume before reporting it.",
        "A load gap or duplicate load may distort the month.",
        ("monthly trends", "MoM change"),
    ),
}


def _schema_rule(table: str) -> Callable[[dict, pd.Timestamp], set]:
    """Return a rule function that detects missing or unexpected columns.

    Args:
        table: Table name whose schema must match the registry.

    Returns:
        A callable that, given the tables dict, returns a set of tokens of the
        form ``"missing:<col>"`` or ``"unexpected:<col>"``.
    """

    def fn(t, as_of):
        cols = list(t[table].columns)
        want = sr.get_table(table).column_names
        missing = [c for c in want if c not in cols]
        extra = [c for c in cols if c not in want]
        return {f"missing:{c}" for c in missing} | {f"unexpected:{c}" for c in extra}

    return fn


def _freshness(t, as_of, max_lag_days=7):
    """Flag the orders table when its newest timestamp lags *as_of* by more than *max_lag_days*.

    Returns:
        A singleton set containing a diagnostic token, or an empty set when the
        data is fresh enough.
    """
    latest = pd.to_datetime(t["orders"].order_timestamp, utc=True).max()
    lag = (as_of - latest).days
    return {f"latest_order={latest:%Y-%m-%d};lag_days={lag}"} if lag > max_lag_days else set()


def _volume(t, as_of, threshold=0.5):
    """Flag months whose order volume deviates more than *threshold* from the trailing 3-month mean.

    Returns:
        A set of diagnostic tokens ``"YYYY-MM:+N%"`` for every flagged month.
    """
    o = t["orders"]
    counts = pd.to_datetime(o.order_date).dt.to_period("M").value_counts().sort_index()
    trailing = counts.rolling(3).mean().shift(1)
    dev = (counts - trailing) / trailing
    flagged = dev[dev.abs() > threshold].dropna()
    return {f"{p}:{v:+.0%}" for p, v in flagged.items()}


def build_rule_specs() -> list[RuleSpec]:
    """Assemble the complete ordered list of rule specifications.

    Includes:

    * one schema-column check per table in ``TABLE_LOAD_ORDER``;
    * every reference rule imported from the data-generation package;
    * the two soft warning rules (freshness and monthly volume).

    Returns:
        A list of frozen ``RuleSpec`` instances ready for execution.
    """
    specs = []
    for name in sr.TABLE_LOAD_ORDER:
        act, imp, mets = RULE_GUIDANCE["SCHEMA_COLUMNS"]
        specs.append(
            RuleSpec(
                f"SCHEMA_COLUMNS_{name.upper()}",
                name,
                "critical",
                f"{name} has exactly the approved columns",
                "no missing or unexpected columns",
                "fail_rule",
                _schema_rule(name),
                act,
                imp,
                mets,
            )
        )
    for r in REFERENCE_RULES:
        act, imp, mets = RULE_GUIDANCE[r.rule_id]
        specs.append(
            RuleSpec(
                r.rule_id,
                r.table,
                r.severity,
                r.description,
                f"0 violating records ({r.id_column})",
                "fail_rule",
                r.fn,
                act,
                imp,
                mets,
            )
        )
    for rid, fn, desc, cond in (
        (
            "FRESH_ORDERS",
            _freshness,
            "Orders are fresh relative to the snapshot",
            "latest order within 7 days of as_of",
        ),
        (
            "VOL_MONTHLY_ORDERS",
            _volume,
            "Monthly order volume has no abnormal jumps",
            "|monthly count vs trailing 3-month mean| <= 50%",
        ),
    ):
        act, imp, mets = RULE_GUIDANCE[rid]
        specs.append(
            RuleSpec(rid, "orders", "medium", desc, cond, "warning_rule", fn, act, imp, mets)
        )
    return specs


RULE_SPECS = build_rule_specs()


# ---------------------------------------------------------------- results ----
@dataclass
class RuleResult:
    """Outcome of executing a single rule.

    Attributes:
        rule_id: Identifier of the rule that produced this result.
        table: Table the rule inspected.
        severity: Severity inherited from the rule specification.
        status: ``"pass"``, ``"fail"``, ``"warning"`` or ``"error"``.
        description: Human-readable description of the rule.
        expected_condition: Statement of the condition that should hold.
        observed_count: Number of offending identifiers (0 on pass/error).
        rows_checked: Number of rows present in the table at evaluation time.
        example_ids: Up to ``MAX_EXAMPLES`` redacted identifiers that illustrate
            the finding.
        recommended_action: Remediation advice copied from the rule guidance.
        business_impact: Business impact copied from the rule guidance.
        affected_metrics: Metric names distorted by this failure.
        error_message: Populated only when ``status == "error"``.
    """

    rule_id: str
    table: str
    severity: str
    status: str
    description: str
    expected_condition: str
    observed_count: int
    rows_checked: int
    example_ids: list
    recommended_action: str
    business_impact: str
    affected_metrics: list
    error_message: str = ""


@dataclass
class ValidationReport:
    """Complete, auditable result of a data-quality run.

    The report is the single source of truth for pass/fail/warning/error
    statuses.  An LLM may explain the report but must never alter a status.

    Attributes:
        run_id: Unique identifier of this run (timestamp + short uuid).
        run_timestamp: ISO-8601 timestamp of when the run started.
        dataset_label: Human-readable label of the dataset that was validated
            (e.g. ``"clean"`` or ``"corrupted"``).
        source: Backend that produced the findings (``"local"`` or
            ``"bigquery"``).
        as_of: Snapshot timestamp used for freshness / temporal rules.
        results: Ordered list of ``RuleResult`` instances.
        _offending: Private mapping of rule_id → full set of offending
            identifiers.  Kept in memory for detection-metric calculations and
            never serialised to disk.
    """

    run_id: str
    run_timestamp: str
    dataset_label: str
    source: str
    as_of: str
    results: list = field(default_factory=list)
    _offending: dict = field(default_factory=dict, repr=False)  # full id sets, kept local, never serialized

    # ---- views ----
    def summary_frame(self) -> pd.DataFrame:
        """Return a DataFrame of all rule results with verbose fields omitted.

        Columns exclude ``business_impact`` and ``affected_metrics`` so the
        frame stays compact for display and CSV export.
        """
        return pd.DataFrame(
            [
                {
                    k: v
                    for k, v in asdict(r).items()
                    if k not in ("business_impact", "affected_metrics")
                }
                for r in self.results
            ]
        )

    def counts(self) -> dict:
        """Return a dictionary of status → count for the four possible statuses."""
        out = {"pass": 0, "fail": 0, "warning": 0, "error": 0}
        for r in self.results:
            out[r.status] += 1
        return out

    def by_status(self, *statuses) -> list[RuleResult]:
        """Return the subset of results whose status is one of *statuses*."""
        return [r for r in self.results if r.status in statuses]

    def get(self, rule_id: str) -> RuleResult:
        """Look up a single result by rule identifier.

        Raises:
            KeyError: If no result with that ``rule_id`` exists.
        """
        for r in self.results:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)

    def offending_ids(self, rule_id: str) -> set:
        """Return the full (unredacted) set of offending identifiers for a rule.

        The set is empty when the rule passed or was never recorded.
        """
        return set(self._offending.get(rule_id, set()))

    @property
    def overall_status(self) -> str:
        """Derive the overall run status from the individual counts.

        Precedence: error > fail > warning > pass.
        """
        c = self.counts()
        if c["error"]:
            return "error"
        if c["fail"]:
            return "fail"
        return "warning" if c["warning"] else "pass"

    # ---- serialization ----
    def to_dict(self) -> dict:
        """Serialise the report to a JSON-ready dictionary (no private fields)."""
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "dataset_label": self.dataset_label,
            "source": self.source,
            "as_of": self.as_of,
            "overall_status": self.overall_status,
            "counts": self.counts(),
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        """Render a compact Markdown summary of the non-passing rules.

        Returns:
            A Markdown string containing the overall status, a table of
            failing / warning / error rules sorted by severity, and a short
            action column.
        """
        c = self.counts()
        lines = [
            f"# Data-quality report — {self.dataset_label}",
            "",
            f"Run `{self.run_id}` at {self.run_timestamp} · source: {self.source} · snapshot as_of {self.as_of}",
            "",
            f"**Overall: {self.overall_status.upper()}** — {c['pass']} pass, {c['fail']} fail, "
            f"{c['warning']} warning, {c['error']} error",
            "",
        ]
        bad = [r for r in self.results if r.status != "pass"]
        if not bad:
            lines.append("All rules passed.")
        else:
            lines += [
                "| Rule | Table | Severity | Status | Observed | Expected | Examples | Action |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in sorted(bad, key=lambda x: (-SEVERITY_ORDER[x.severity], x.rule_id)):
                ex = ", ".join(map(str, r.example_ids[:3])) or r.error_message
                lines.append(
                    f"| `{r.rule_id}` | {r.table} | {r.severity} | {r.status} | {r.observed_count} | "
                    f"{r.expected_condition} | {ex} | {r.recommended_action} |"
                )
        return "\n".join(lines) + "\n"

    def save(self, out_dir: str | Path = "reports/dq") -> tuple[Path, Path]:
        """Write the report to JSON and Markdown files under *out_dir*.

        Args:
            out_dir: Directory that will receive ``{run_id}.json`` and
                ``{run_id}.md``.  Created if it does not exist.

        Returns:
            A pair ``(json_path, markdown_path)``.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        j = out / f"{self.run_id}.json"
        m = out / f"{self.run_id}.md"
        j.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        m.write_text(self.to_markdown())
        return j, m


# engine ----
def redact_identifier(value) -> str:
    """Mask PII-looking identifiers (emails) with a stable short hash.

    Non-email values are returned unchanged.  Embedded emails inside a larger
    string are replaced in-place.

    Args:
        value: Any identifier that may contain an email address.

    Returns:
        A string safe for inclusion in reports and logs.
    """
    s = str(value)
    if EMAIL_RE.fullmatch(s):
        return "email#" + hashlib.sha256(s.lower().encode()).hexdigest()[:10]
    return EMAIL_RE.sub(
        lambda m: "email#" + hashlib.sha256(m.group(0).lower().encode()).hexdigest()[:10],
        s,
    )


def validate_tables(
    tables: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    dataset_label: str,
    source: str = "local",
    specs: list[RuleSpec] | None = None,
) -> ValidationReport:
    """Run every rule specification against an in-memory set of tables.

    This is the local (pandas) backend of the data-quality engine.  It is the
    source of truth used before BigQuery is connected and remains the oracle
    against which the SQL backend is validated.

    Args:
        tables: Mapping of table name → DataFrame.  Missing tables produce an
            ``error`` status for rules that depend on them.
        as_of: Snapshot timestamp used by freshness and temporal rules.
        dataset_label: Label recorded in the report (e.g. ``"clean"``).
        source: Backend label recorded in the report (default ``"local"``).
        specs: Optional override of the rule list; defaults to the module-level
            ``RULE_SPECS``.

    Returns:
        A fully populated ``ValidationReport``.  Statuses are never altered
        after this function returns.
    """
    specs = specs or RULE_SPECS
    report = ValidationReport(
        run_id=f"dq_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}",
        run_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_label=dataset_label,
        source=source,
        as_of=str(as_of),
    )
    schema_broken = set()
    for spec in specs:
        rows = int(len(tables[spec.table])) if spec.table in tables else 0
        base = dict(
            rule_id=spec.rule_id,
            table=spec.table,
            severity=spec.severity,
            description=spec.description,
            expected_condition=spec.expected_condition,
            rows_checked=rows,
            recommended_action=spec.recommended_action,
            business_impact=spec.business_impact,
            affected_metrics=list(spec.affected_metrics),
        )
        if spec.table not in tables:
            report.results.append(
                RuleResult(
                    status="error",
                    observed_count=0,
                    example_ids=[],
                    error_message=f"table {spec.table!r} not found",
                    **base,
                )
            )
            continue
        try:
            ids = spec.fn(tables, as_of)
        except Exception as exc:  # noqa: BLE001 - recorded as 'error', never as 'pass'
            report.results.append(
                RuleResult(
                    status="error",
                    observed_count=0,
                    example_ids=[],
                    error_message=f"{type(exc).__name__}: {exc}"[:300],
                    **base,
                )
            )
            continue
        ids = {str(i) for i in ids}
        if spec.rule_id.startswith("SCHEMA_COLUMNS_") and ids:
            schema_broken.add(spec.table)
        status = "pass" if not ids else ("fail" if spec.kind == "fail_rule" else "warning")
        report._offending[spec.rule_id] = ids
        report.results.append(
            RuleResult(
                status=status,
                observed_count=len(ids),
                example_ids=[redact_identifier(i) for i in sorted(ids)[:MAX_EXAMPLES]],
                **base,
            )
        )
    return report


def load_tables_from_files(
    variant: str, fmt: str = "parquet", root: Path | None = None
) -> dict[str, pd.DataFrame]:
    """Load one dataset variant from the generated data directory.

    The files are the same ones that are later uploaded to BigQuery, so the
    local engine and the BigQuery engine operate on identical data.

    Args:
        variant: Dataset variant name (``"clean"`` or ``"corrupted"``).
        fmt: File format – ``"parquet"`` (default) or ``"csv"``.
        root: Project root.  When ``None`` it is discovered via
            ``find_project_root``.

    Returns:
        Mapping of table name → normalised DataFrame.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    from analytics_agent.config import find_project_root

    root = root or find_project_root()
    tables = {}
    for name in sr.TABLE_LOAD_ORDER:
        path = root / "data/generated" / variant / fmt / f"{name}.{fmt}"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/generate_data.py"
            )
        if fmt == "parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", None)
            for c in sr.get_table(name).columns:
                if c.bq_type == "TIMESTAMP":
                    df[c.name] = pd.to_datetime(
                        df[c.name].str.replace(" UTC", "", regex=False), utc=True
                    )
                elif c.bq_type == "BOOL":
                    df[c.name] = df[c.name].map({"true": True, "false": False})
        tables[name] = normalize_dtypes(name, df)
    return tables


def detection_metrics(
    report: ValidationReport, defect_catalog: list[dict]
) -> pd.DataFrame:
    """Compare engine findings with the defect catalog at record level.

    For every rule that has either expected or found identifiers a row is
    produced with true-positive, missed and false-positive counts.

    Args:
        report: Validation report produced by ``validate_tables``.
        defect_catalog: List of defect dictionaries, each containing an
            ``expected_findings`` mapping of rule_id → list of identifiers.

    Returns:
        A DataFrame with columns ``rule_id``, ``status``, ``expected``,
        ``found``, ``true_pos``, ``missed`` and ``false_pos``.
    """
    expected: dict[str, set] = {}
    for d in defect_catalog:
        for rid, ids in d["expected_findings"].items():
            expected.setdefault(rid, set()).update(map(str, ids))
    rows = []
    for r in report.results:
        found = report.offending_ids(r.rule_id)
        exp = expected.get(r.rule_id, set())
        if not found and not exp:
            continue
        tp, fn, fp = len(found & exp), len(exp - found), len(found - exp)
        rows.append(
            {
                "rule_id": r.rule_id,
                "status": r.status,
                "expected": len(exp),
                "found": len(found),
                "true_pos": tp,
                "missed": fn,
                "false_pos": fp,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["rule_id", "status", "expected", "found", "true_pos", "missed", "false_pos"],
    )