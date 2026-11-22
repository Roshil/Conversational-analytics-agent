"""Customer data-quality profile.

Deterministic SQL checks on the customers table, grouped by dimension (completeness, uniqueness, validity, consistency,
timeliness, reconciliation). Each check is classified so findings are not over-stated:

* confirmed_violation - breaks a documented rule (schema registry: required column, primary/business key, allowed
  values). Status FAIL when records are affected.
* potential_anomaly  - unusual but not forbidden by a documented rule (stray whitespace, stale data). Status WARN.
* undefined_rule     - the business has not decided whether this is an error (e.g. signup date after the record was
  created). Status WARN, with a note that the rule needs defining.

Counts come only from SQL executed on BigQuery (or the emulator); no model is involved. Examples are customer IDs only:
email addresses are never returned.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from analytics_agent import schema_registry as sr
from analytics_agent.bq import SqlParam

CONFIRMED, ANOMALY, UNDEFINED = "confirmed_violation", "potential_anomaly", "undefined_rule"
FRESHNESS_THRESHOLD_DAYS = 7  # newest customer record older than this -> WARN
COVERAGE_MONTHS = 6  # complete months before the snapshot that must each contain signups
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


@dataclass(frozen=True)
class CustomerCheck:
    """Definition of a single deterministic data-quality check on the customers table.

    Attributes:
        rule_id: Stable identifier (e.g. ``"CQ_UNIQUE_CUSTOMER_ID"``).
        name: Short human-readable description of the rule.
        dimension: One of completeness, uniqueness, validity, consistency,
            timeliness or reconciliation.
        classification: ``CONFIRMED``, ``ANOMALY`` or ``UNDEFINED`` – controls
            whether a non-zero count becomes FAIL or WARN.
        severity: Relative importance (``"critical"``, ``"high"``, ``"medium"``,
            ``"low"``).
        count_sql: Parameterised SQL that returns columns ``affected`` and
            ``total`` (and optionally extra diagnostic columns).
        examples_sql: Optional SQL that returns a ``customer_id`` column used to
            illustrate failing rows.  ``None`` when examples are not useful.
        business_impact: One-sentence explanation of why the finding matters.
        remediation: Concrete suggestion for fixing the underlying data issue.
    """

    rule_id: str
    name: str
    dimension: str
    classification: str
    severity: str
    count_sql: str  # returns columns: affected, total
    examples_sql: Optional[str]  # returns column: customer_id
    business_impact: str
    remediation: str


@dataclass
class CheckOutcome:
    """Result of executing one ``CustomerCheck``.

    Attributes:
        rule_id: Identifier of the check that produced this outcome.
        name: Human-readable name of the check.
        dimension: Quality dimension the check belongs to.
        classification: Classification inherited from the check definition.
        table: Always ``"customers"`` for this profile.
        validated_at: ISO-8601 timestamp of when the check was executed.
        severity: Severity inherited from the check definition.
        status: ``"PASS"``, ``"FAIL"``, ``"WARN"`` or ``"ERROR"``.
        affected_rows: Number of rows that violate the rule (``None`` on error).
        total_rows: Total rows examined (``None`` on error).
        failure_pct: Percentage of affected rows, or ``None`` when not applicable
            (e.g. pure timeliness checks).
        examples: Up to five customer IDs that illustrate the finding.
        business_impact: Copied from the check definition.
        remediation: Copied from the check definition.
        note: Additional diagnostic text (classification note, extra SQL columns,
            or the error message on failure).
        execution: Timing / job metadata (``duration_s``, ``job_id``,
            ``bytes_processed``).
    """

    rule_id: str
    name: str
    dimension: str
    classification: str
    table: str
    validated_at: str
    severity: str
    status: str  # PASS | FAIL | WARN | ERROR
    affected_rows: Optional[int]
    total_rows: Optional[int]
    failure_pct: Optional[float]
    examples: list = field(default_factory=list)
    business_impact: str = ""
    remediation: str = ""
    note: str = ""
    execution: dict = field(default_factory=dict)


def _allowed(column: str) -> tuple:
    """Return the allowed-values tuple for a customers-table column from the schema registry."""
    return next(c.allowed_values for c in sr.TABLES["customers"].columns if c.name == column)


def _in_list(values) -> str:
    """Format a sequence of allowed values as a SQL ``IN`` list literal.

    Values originate exclusively from the schema registry and are therefore
    safe to embed; they are never taken from user input.
    """
    return ", ".join(f"'{v}'" for v in values)


def _row_check(
    rule_id,
    name,
    dimension,
    classification,
    severity,
    predicate,
    impact,
    remediation,
) -> CustomerCheck:
    """Helper that builds a simple row-level check from a SQL predicate.

    The generated ``count_sql`` returns ``COUNTIF(predicate)`` / ``COUNT(*)``
    and the ``examples_sql`` returns up to five matching ``customer_id`` values.
    """
    return CustomerCheck(
        rule_id,
        name,
        dimension,
        classification,
        severity,
        f"SELECT COUNTIF({predicate}) AS affected, COUNT(*) AS total FROM {{customers}}",
        f"SELECT customer_id FROM {{customers}} WHERE {predicate} ORDER BY customer_id LIMIT 5",
        impact,
        remediation,
    )


def build_checks() -> list[CustomerCheck]:
    """Construct the full ordered list of customer data-quality checks.

    Checks are grouped by dimension:

    * **A. Completeness** – required columns must be non-null / non-blank.
    * **B. Uniqueness** – primary key, business key and exact-row uniqueness.
    * **C. Validity** – format, allowed values and plausible date ranges.
    * **D. Consistency** – whitespace/casing anomalies and conflicting duplicates.
    * **E. Timeliness** – freshness of the newest record and monthly signup coverage.
    * **F. Reconciliation** – internal consistency of aggregate counts.

    Returns:
        A list of frozen ``CustomerCheck`` instances ready for execution.
    """
    regions, segments, countries = (
        _allowed("region"),
        _allowed("customer_segment"),
        _allowed("country_code"),
    )
    checks = []

    #A. completeness
    for column, impact in [
        (
            "customer_id",
            "Rows without an ID cannot be joined to orders and are dropped from customer counts.",
        ),
        (
            "email",
            "Customers cannot be contacted; email-based deduplication misses them.",
        ),
        ("region", "Customers fall into an 'Unknown' bucket in regional reports."),
        (
            "customer_segment",
            "Customers fall into an 'Unknown' bucket in segment reports.",
        ),
        ("country_code", "Country-level reporting is incomplete."),
        ("signup_date", "Customers are missing from signup trends."),
        ("created_at", "Freshness and audit checks cannot use these rows."),
    ]:
        is_text = column not in ("signup_date", "created_at")
        predicate = f"{column} IS NULL" + (f" OR TRIM({column}) = ''" if is_text else "")
        checks.append(
            _row_check(
                f"CQ_COMPLETE_{column.upper()}",
                f"{column} is present",
                "completeness",
                CONFIRMED,
                "critical" if column == "customer_id" else "high",
                predicate,
                impact,
                f"Backfill {column} from the source system; make the column required in the load job.",
            )
        )

    #B. uniqueness
    checks.append(
        CustomerCheck(
            "CQ_UNIQUE_CUSTOMER_ID",
            "customer_id is unique",
            "uniqueness",
            CONFIRMED,
            "critical",
            "SELECT COUNT(*) - COUNT(DISTINCT customer_id) AS affected, COUNT(*) AS total FROM {customers}",
            "SELECT customer_id FROM {customers} GROUP BY customer_id HAVING COUNT(*) > 1 ORDER BY customer_id LIMIT 5",
            "Duplicate rows inflate customer counts and duplicate orders in joins (regional sales, AOV).",
            "Deduplicate on customer_id (keep the earliest created_at) and add a uniqueness check to the load job.",
        )
    )
    checks.append(
        CustomerCheck(
            "CQ_UNIQUE_EMAIL",
            "email is unique (documented business key)",
            "uniqueness",
            CONFIRMED,
            "medium",
            "SELECT COUNTIF(email IS NOT NULL) - COUNT(DISTINCT LOWER(TRIM(email))) AS affected, COUNT(*) AS total FROM {customers}",
            "SELECT MIN(customer_id) AS customer_id FROM {customers} WHERE email IS NOT NULL "
            "GROUP BY LOWER(TRIM(email)) HAVING COUNT(*) > 1 ORDER BY customer_id LIMIT 5",
            "One person may be counted as several customers.",
            "Merge customer records that share an email after confirming they are the same person.",
        )
    )
    checks.append(
        CustomerCheck(
            "CQ_UNIQUE_FULL_ROW",
            "no exact duplicate rows",
            "uniqueness",
            ANOMALY,
            "medium",
            "SELECT COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT customer_id, email, region, country_code, "
            "customer_segment, signup_date, created_at FROM {customers})) AS affected, COUNT(*) AS total FROM {customers}",
            None,
            "Exact copies usually come from a load running twice.",
            "Check the load job for re-runs; remove exact copies.",
        )
    )

    #C. validity
    checks.append(
        _row_check(
            "CQ_VALID_EMAIL_FORMAT",
            "email has a valid format",
            "validity",
            CONFIRMED,
            "medium",
            f"email IS NOT NULL AND NOT REGEXP_CONTAINS(email, r'{EMAIL_PATTERN}')",
            "Messages to these customers will bounce.",
            "Correct the addresses at the source.",
        )
    )
    checks.append(
        _row_check(
            "CQ_VALID_REGION",
            "region is an allowed value",
            "validity",
            CONFIRMED,
            "medium",
            f"region IS NOT NULL AND region NOT IN ({_in_list(regions)})",
            "Unexpected regions create extra rows in regional reports.",
            f"Map values to the allowed regions: {', '.join(regions)}.",
        )
    )
    checks.append(
        _row_check(
            "CQ_VALID_SEGMENT",
            "customer_segment is an allowed value",
            "validity",
            CONFIRMED,
            "medium",
            f"customer_segment IS NOT NULL AND customer_segment NOT IN ({_in_list(segments)})",
            "Unexpected segments break segment reports.",
            f"Map values to the allowed segments: {', '.join(segments)}.",
        )
    )
    checks.append(
        _row_check(
            "CQ_VALID_COUNTRY",
            "country_code is an allowed value",
            "validity",
            CONFIRMED,
            "low",
            f"country_code IS NOT NULL AND country_code NOT IN ({_in_list(countries)})",
            "Customers outside the supported countries distort country reporting.",
            f"Allowed countries: {', '.join(countries)}; correct or document new ones.",
        )
    )
    checks.append(
        _row_check(
            "CQ_VALID_SIGNUP_DATE",
            "signup_date is a possible date",
            "validity",
            CONFIRMED,
            "high",
            "signup_date < DATE '2000-01-01' OR signup_date > @as_of_date",
            "Impossible dates distort signup trends.",
            "Correct the dates at the source.",
        )
    )
    checks.append(
        _row_check(
            "CQ_VALID_CREATED_AT",
            "created_at is not in the future",
            "validity",
            CONFIRMED,
            "high",
            "DATE(created_at) > @as_of_date",
            "Future timestamps break freshness checks.",
            "Check the source system clock and time zones.",
        )
    )
    checks.append(
        _row_check(
            "CQ_SIGNUP_AFTER_CREATED",
            "signup_date is not after the record was created",
            "validity",
            UNDEFINED,
            "low",
            "signup_date > DATE(created_at)",
            "If signups must precede record creation, these rows are wrong; the rule is not documented yet.",
            "Agree the business rule; then promote this check to a confirmed rule or drop it.",
        )
    )

    #D. consistency
    checks.append(
        _row_check(
            "CQ_CONSISTENT_FORMATTING",
            "no stray whitespace or casing in coded fields",
            "consistency",
            ANOMALY,
            "low",
            "region != TRIM(region) OR customer_segment != TRIM(customer_segment) OR "
            "country_code != UPPER(TRIM(country_code)) OR customer_segment != LOWER(customer_segment) OR "
            "email != TRIM(email)",
            "The same value written two ways splits groups in reports.",
            "Trim and normalise casing in the load job.",
        )
    )
    checks.append(
        CustomerCheck(
            "CQ_CONSISTENT_DUPLICATES",
            "duplicate IDs carry the same values",
            "consistency",
            CONFIRMED,
            "critical",
            "SELECT COUNT(*) AS affected, (SELECT COUNT(DISTINCT customer_id) FROM {customers}) AS total FROM ("
            "SELECT customer_id FROM {customers} GROUP BY customer_id HAVING COUNT(*) > 1 AND ("
            "COUNT(DISTINCT region) > 1 OR COUNT(DISTINCT customer_segment) > 1 OR COUNT(DISTINCT email) > 1 OR "
            "COUNT(DISTINCT signup_date) > 1))",
            "SELECT customer_id FROM {customers} GROUP BY customer_id HAVING COUNT(*) > 1 AND (COUNT(DISTINCT region) > 1 "
            "OR COUNT(DISTINCT customer_segment) > 1) ORDER BY customer_id LIMIT 5",
            "Conflicting records for one customer make lookups and regional reports ambiguous.",
            "Decide which record is correct (usually the earliest) and remove the others.",
        )
    )

    #E. timeliness
    checks.append(
        CustomerCheck(
            "CQ_FRESHNESS",
            f"newest record is at most {FRESHNESS_THRESHOLD_DAYS} days old",
            "timeliness",
            ANOMALY,
            "medium",
            f"SELECT CASE WHEN DATE_DIFF(@as_of_date, DATE(MAX(created_at)), DAY) > {FRESHNESS_THRESHOLD_DAYS} THEN 1 ELSE 0 END "
            "AS affected, 1 AS total, DATE_DIFF(@as_of_date, DATE(MAX(created_at)), DAY) AS age_days, "
            "CAST(MAX(created_at) AS STRING) AS latest_created_at FROM {customers}",
            None,
            "Stale data under-reports recent signups.",
            "Check that the customer load is running on schedule.",
        )
    )
    checks.append(
        CustomerCheck(
            "CQ_SIGNUP_COVERAGE",
            f"each of the last {COVERAGE_MONTHS} complete months has signups",
            "timeliness",
            ANOMALY,
            "medium",
            f"SELECT {COVERAGE_MONTHS} - COUNT(DISTINCT FORMAT_DATE('%Y-%m', signup_date)) AS affected, {COVERAGE_MONTHS} AS total "
            "FROM {customers} WHERE signup_date BETWEEN @coverage_start AND @coverage_end",
            None,
            "A month without any signups usually means a missing load.",
            "Re-run the load for the missing months.",
        )
    )

    #F. reconciliation
    checks.append(
        CustomerCheck(
            "CQ_RECONCILE_ROWS",
            "row count equals distinct customer IDs",
            "reconciliation",
            CONFIRMED,
            "high",
            "SELECT COUNT(*) - COUNT(DISTINCT customer_id) AS affected, COUNT(*) AS total, COUNT(*) AS customer_rows, "
            "COUNT(DISTINCT customer_id) AS distinct_customers FROM {customers}",
            None,
            "If rows and distinct customers differ, 'number of customers' depends on how it is counted.",
            "Always count customers as COUNT(DISTINCT customer_id) (the metric registry does); fix the duplicates.",
        )
    )
    checks.append(
        CustomerCheck(
            "CQ_RECONCILE_REGIONS",
            "per-region customer counts add up to the total",
            "reconciliation",
            ANOMALY,
            "medium",
            "SELECT (SELECT SUM(n) FROM (SELECT COUNT(DISTINCT customer_id) AS n FROM {customers} GROUP BY region)) "
            "- COUNT(DISTINCT customer_id) AS affected, COUNT(DISTINCT customer_id) AS total FROM {customers}",
            None,
            "A customer recorded in two regions is counted in both, so regional counts add up to more than the total.",
            "Resolve conflicting duplicate records (see CQ_CONSISTENT_DUPLICATES).",
        )
    )
    return checks


CHECKS = build_checks()


def _status(check: CustomerCheck, affected: int) -> tuple[str, str]:
    """Derive the outcome status and an optional explanatory note from a check and its count.

    Args:
        check: The check definition whose classification drives the decision.
        affected: Number of rows that matched the check predicate.

    Returns:
        A pair ``(status, note)`` where *status* is ``"PASS"``, ``"FAIL"`` or
        ``"WARN"`` and *note* is an empty string or a short classification note.
    """
    if not affected:
        return "PASS", ""
    if check.classification == CONFIRMED:
        return "FAIL", ""
    if check.classification == UNDEFINED:
        return "WARN", "Business rule not defined yet: this is reported, not counted as an error."
    return "WARN", "Potential anomaly: not forbidden by a documented rule; investigate."


def run_customer_profile(
    runner, settings, dataset: str, as_of: datetime
) -> list[CheckOutcome]:
    """Execute every customer data-quality check against one dataset.

    Each check is run as its own read-only query.  Parameter binding, allow-list
    enforcement and error classification are delegated to the supplied runner.
    Example customer IDs are collected only when the check defines an
    ``examples_sql`` and the count is non-zero; email addresses are never
    returned.

    Args:
        runner: Query runner that implements ``.run(...)``.
        settings: Settings object exposing ``.refs(dataset)`` and
            ``.allowed_tables(dataset)``.
        dataset: Dataset variant (``"clean"`` or ``"corrupted"``).
        as_of: Reference timestamp used for freshness and date-validity
            checks.  May be a ``datetime`` or a ``date``.

    Returns:
        A list of ``CheckOutcome`` instances, one per check, in the same order
        as ``CHECKS``.
    """
    refs = {"customers": settings.refs(dataset)["customers"]}
    as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    first_of_month = as_of_date.replace(day=1)
    coverage_end = first_of_month - timedelta(days=1)
    coverage_start = first_of_month
    for _ in range(COVERAGE_MONTHS):
        coverage_start = (coverage_start - timedelta(days=1)).replace(day=1)
    params = [
        SqlParam("as_of_date", "DATE", as_of_date),
        SqlParam("coverage_start", "DATE", coverage_start),
        SqlParam("coverage_end", "DATE", coverage_end),
    ]
    validated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    outcomes = []
    for check in CHECKS:
        sql = check.count_sql.format(**refs)
        used = [p for p in params if f"@{p.name}" in sql]
        started = time.perf_counter()
        res = runner.run(
            sql,
            used,
            max_rows=1,
            allowed_tables=settings.allowed_tables(dataset),
            dry_run_first=False,
        )
        execution = {
            "duration_s": round(time.perf_counter() - started, 3),
            "job_id": getattr(res, "job_id", None),
            "bytes_processed": getattr(res, "bytes_processed", None),
        }
        if not res.ok:
            outcomes.append(
                CheckOutcome(
                    check.rule_id,
                    check.name,
                    check.dimension,
                    check.classification,
                    "customers",
                    validated_at,
                    check.severity,
                    "ERROR",
                    None,
                    None,
                    None,
                    business_impact=check.business_impact,
                    remediation=check.remediation,
                    note=f"{res.error_kind}: {res.error_message}",
                    execution=execution,
                )
            )
            continue
        row = res.rows[0]
        affected, total = int(row["affected"] or 0), int(row["total"] or 0)
        status, note = _status(check, affected)
        extra = {k: v for k, v in row.items() if k not in ("affected", "total")}
        if extra:
            note = (note + " " if note else "") + ", ".join(
                f"{k}={v}" for k, v in extra.items()
            )
        examples = []
        if affected and check.examples_sql:
            ex_sql = check.examples_sql.format(**refs)
            ex = runner.run(
                ex_sql,
                [p for p in params if f"@{p.name}" in ex_sql],
                max_rows=5,
                allowed_tables=settings.allowed_tables(dataset),
                dry_run_first=False,
            )
            examples = [r["customer_id"] for r in ex.rows] if ex.ok else []
        pct = (
            round(100 * affected / total, 2)
            if total and check.dimension != "timeliness"
            else None
        )
        outcomes.append(
            CheckOutcome(
                check.rule_id,
                check.name,
                check.dimension,
                check.classification,
                "customers",
                validated_at,
                check.severity,
                status,
                affected,
                total,
                pct,
                examples,
                check.business_impact,
                check.remediation,
                note.strip(),
                execution,
            )
        )
    return outcomes


def profile_frame(outcomes: list[CheckOutcome]):
    """Convert a list of ``CheckOutcome`` objects into a pandas DataFrame.

    Args:
        outcomes: Results produced by ``run_customer_profile``.

    Returns:
        A DataFrame whose columns match the fields of ``CheckOutcome``.
    """
    import pandas as pd

    return pd.DataFrame([asdict(o) for o in outcomes])


# Overlapping rules in the main 50-rule engine: both must report the same number of affected records.
RULE_ENGINE_EQUIVALENTS = {
    "CQ_UNIQUE_CUSTOMER_ID": "CUS_PK_UNIQUE",
    "CQ_UNIQUE_EMAIL": "CUS_EMAIL_UNIQUE",
    "CQ_VALID_REGION": "CUS_REGION_ALLOWED",
}


def reconcile_with_rule_engine(outcomes: list[CheckOutcome], report) -> list[dict]:
    """Compare overlapping checks with the main rule-engine report.

    A mismatch indicates that one of the two implementations is wrong and
    should be investigated.

    Args:
        outcomes: Results from ``run_customer_profile``.
        report: Rule-engine report object whose ``.results`` is an iterable of
            objects exposing ``.rule_id`` and ``.observed_count``.

    Returns:
        A list of dictionaries, one per overlapping rule, containing
        ``profile_rule``, ``engine_rule``, ``profile_affected``,
        ``engine_affected`` and a boolean ``match`` flag.
    """
    engine = {r.rule_id: r for r in report.results}
    by_id = {o.rule_id: o for o in outcomes}
    rows = []
    for profile_id, engine_id in RULE_ENGINE_EQUIVALENTS.items():
        o, e = by_id.get(profile_id), engine.get(engine_id)
        if o is None or e is None:
            continue
        rows.append(
            {
                "profile_rule": profile_id,
                "engine_rule": engine_id,
                "profile_affected": o.affected_rows,
                "engine_affected": e.observed_count,
                "match": o.affected_rows == e.observed_count,
            }
        )
    return rows