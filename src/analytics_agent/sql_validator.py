"""Deterministic SQL validator for model-generated BigQuery SQL (the model has no say).

A query may run only if sqlglot (BigQuery dialect) can prove that it is:
  * exactly one statement, a SELECT (CTEs allowed), with no DDL/DML/script nodes;
  * reading only the seven approved tables of ONE configured dataset (plus its own CTEs);
  * free of table functions, INFORMATION_SCHEMA, external sources and denied functions;
  * referencing only columns that exist in the approved tables (or defined aliases);
  * not selecting blocked personal-data columns (customers.email), not using SELECT *;
  * passing user values (dates, IDs) as typed @parameters, not literals;
  * row-limited (LIMIT appended to the ORIGINAL text if missing, then re-parsed).

The result is a ValidatedQuery signed with a per-process HMAC key; executors refuse
anything without a valid signature. Parsing does not replace IAM: the BigQuery
identity must still be read-only, and the runner's dry run re-checks referenced tables.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Optional

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from analytics_agent import schema_registry as sr
from analytics_agent.bq import ANALYTICS_TABLES, BQSettings, SqlParam

_SIGNING_KEY = secrets.token_bytes(32)
FORBIDDEN_NODE_TYPES = tuple(
    getattr(exp, n)
    for n in (
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Into",
        "TruncateTable",
        "Export",
        "Grant",
        "Revoke",
        "Transaction",
        "Commit",
        "Rollback",
        "Use",
        "Set",
        "Copy",
        "LoadData",
        "Declare",
        "Analyze",
        "Describe",
        "Pragma",
    )
    if hasattr(exp, n)
)
DENIED_FUNCTIONS = {
    "EXTERNAL_QUERY",
    "SESSION_USER",
    "EXTERNAL_OBJECT_TRANSFORM",
    "VECTOR_SEARCH",
    "APPENDS",
    "CHANGES",
}
DENIED_FUNCTION_PREFIXES = (
    "AI_",
    "AI.",
    "ML_",
    "ML.",
    "AEAD.",
    "KEYS.",
    "OBJ.",
    "PREDICT",
    "VECTOR_",
)
BLOCKED_COLUMNS = {"email"}
PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
DATE_LITERAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T].*)?$")
ID_LITERAL_RE = re.compile(r"^(CUS|ORD|ITM|PRD|PAY|SHP|REF)\d+$")
MAX_SQL_CHARS, MAX_ARRAY_VALUES = 20_000, 50
TABLE_COLUMNS = {t: set(sr.get_table(t).column_names) for t in ANALYTICS_TABLES}
ALL_COLUMNS = set().union(*TABLE_COLUMNS.values())


class QueryParam(BaseModel):
    """Parameter as proposed by the model (values as strings; converted and type-checked here).

    Attributes:
        name: Parameter name without the leading ``@`` (e.g. ``start_date``).
        type: BigQuery scalar type used for conversion and binding.
        values: Exactly one value for a scalar parameter; several for an array.
        is_array: When ``True`` the parameter is bound as an array.
    """

    name: str = Field(description="Parameter name without @, e.g. start_date")
    type: Literal["STRING", "INT64", "FLOAT64", "BOOL", "DATE", "TIMESTAMP"]
    values: list[str] = Field(
        description=(
            "Exactly one value for a scalar parameter; several for an array parameter."
        )
    )
    is_array: bool = False


@dataclass(frozen=True)
class ValidatedQuery:
    """HMAC-signed query that has passed every deterministic safety check.

    Executors must call ``is_authentic()`` before running the SQL; a mismatched
    signature means the object was tampered with or built outside this module.

    Attributes:
        sql: Final SQL text (possibly with an appended ``LIMIT``).
        params: Tuple of converted ``SqlParam`` values ready for the runner.
        dataset: Dataset id the query was validated against.
        tables: Sorted tuple of approved base-table names referenced by the query.
        signature: Per-process HMAC of ``(sql, dataset, params)``.
    """

    sql: str
    params: tuple  # SqlParam
    dataset: str
    tables: tuple  # approved base tables referenced
    signature: str

    def is_authentic(self) -> bool:
        """Return ``True`` when the HMAC signature matches the current payload."""
        return hmac.compare_digest(
            self.signature, _sign(self.sql, self.params, self.dataset)
        )

    def params_summary(self) -> list[dict]:
        """Return a JSON-safe summary of bound parameters for logging / display."""
        return [
            {
                "name": p.name,
                "type": p.type,
                "value": (
                    [str(v) for v in p.value] if p.is_array else str(p.value)
                ),
            }
            for p in self.params
        ]


def _sign(sql: str, params: tuple, dataset: str) -> str:
    """Compute the per-process HMAC signature for a validated query payload."""
    payload = json.dumps(
        [
            sql,
            dataset,
            [(p.name, p.type, str(p.value), p.is_array) for p in params],
        ]
    )
    return hmac.new(
        _SIGNING_KEY, payload.encode(), hashlib.sha256
    ).hexdigest()


@dataclass
class ValidationResult:
    """Outcome of ``validate_sql``.

    Attributes:
        ok: ``True`` when the query passed every check and ``validated`` is set.
        errors: Blocking problems; non-empty implies ``ok`` is ``False``.
        warnings: Non-blocking notes (unused parameters, auto-added LIMIT, …).
        validated: The signed ``ValidatedQuery`` when ``ok`` is ``True``.
    """

    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    validated: Optional[ValidatedQuery] = None


def _in_projection(node: exp.Expression) -> bool:
    """Return ``True`` if *node* sits in the SELECT list of its nearest SELECT.

    Uses identity comparison against the SELECT's projection expressions so
    columns buried inside aggregations or sub-expressions are still detected.
    """
    sel = node.find_ancestor(exp.Select)
    if sel is None:
        return False
    cur = node
    while cur.parent is not None and cur.parent is not sel:
        cur = cur.parent
    return any(cur is e for e in sel.expressions)


def convert_param(p: QueryParam) -> SqlParam:
    """Convert a model-proposed ``QueryParam`` into a typed ``SqlParam``.

    Validates the parameter name, value count, individual value length and
    type conversion (including a sane year range for dates).

    Args:
        p: Parameter as emitted by the model (string values only).

    Returns:
        A ``SqlParam`` ready for the query runner.

    Raises:
        ValueError: On invalid name, empty/oversized values, or failed
            type conversion.
    """
    if not PARAM_NAME_RE.fullmatch(p.name):
        raise ValueError(f"invalid parameter name {p.name!r}")
    if not p.values:
        raise ValueError(f"@{p.name} has no value")
    if not p.is_array and len(p.values) != 1:
        raise ValueError(
            f"scalar parameter @{p.name} must have exactly one value"
        )
    if p.is_array and len(p.values) > MAX_ARRAY_VALUES:
        raise ValueError(
            f"@{p.name} has more than {MAX_ARRAY_VALUES} values"
        )

    def conv(v: str):
        if len(v) > 256:
            raise ValueError(f"@{p.name} value is too long")
        try:
            if p.type == "STRING":
                return v
            if p.type == "BOOL":
                if v.lower() not in ("true", "false"):
                    raise ValueError
                return v.lower() == "true"
            if p.type == "INT64":
                return int(v)
            if p.type == "FLOAT64":
                return float(v)
            if p.type == "DATE":
                d = date.fromisoformat(v)
                if not (1990 <= d.year <= 2100):
                    raise ValueError
                return d
            if p.type == "TIMESTAMP":
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise ValueError(
                f"@{p.name}: cannot convert {v!r} to {p.type}"
            ) from None
        raise ValueError(f"unsupported type {p.type}")

    values = [conv(v) for v in p.values]
    return SqlParam(
        p.name,
        p.type,
        tuple(values) if p.is_array else values[0],
        p.is_array,
    )


def validate_sql(
    sql: str,
    params: list[QueryParam],
    settings: BQSettings,
    dataset: str,
    max_rows: Optional[int] = None,
) -> ValidationResult:
    """Decide deterministically whether model-generated SQL may run against *dataset*.

    Performs, in order:

    1. Length / emptiness checks and BigQuery-dialect parsing.
    2. Single-SELECT shape and forbidden-node scan (DDL/DML/script).
    3. Table allow-list (project + dataset + approved table names; CTEs allowed).
    4. Denied function / table-function scan.
    5. Column existence, ``SELECT *`` ban, and blocked personal-data columns.
    6. Rejection of date / ID / email-like string literals (must be parameters).
    7. Parameter name matching, type conversion and binding.
    8. Row-limit enforcement (append ``LIMIT`` to the original text if missing).
    9. HMAC signing of the final payload.

    Args:
        sql: Model-generated BigQuery Standard SQL.
        params: Parameters proposed by the model (string values).
        settings: BigQuery settings supplying project id and row limits.
        dataset: Dataset id the query is allowed to read.
        max_rows: Optional override of ``settings.max_result_rows``.

    Returns:
        A ``ValidationResult``.  When ``ok`` is ``True``, ``validated`` holds
        the signed ``ValidatedQuery`` that executors may run.
    """
    errors, warnings = [], []
    max_rows = max_rows or settings.max_result_rows
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ValidationResult(False, ["empty SQL"])
    if len(sql) > MAX_SQL_CHARS:
        return ValidationResult(
            False, [f"SQL longer than {MAX_SQL_CHARS} characters"]
        )
    try:
        statements = [
            s for s in sqlglot.parse(sql, read="bigquery") if s is not None
        ]
    except (ParseError, TokenError) as e:
        return ValidationResult(
            False,
            [
                "SQL could not be parsed (BigQuery dialect): "
                f"{str(e).splitlines()[0][:200]}"
            ],
        )
    if len(statements) != 1:
        return ValidationResult(
            False,
            [
                f"exactly one statement is allowed (found {len(statements)})"
            ],
        )
    root = statements[0]
    if not isinstance(root, exp.Select):
        return ValidationResult(
            False,
            [
                "only a single SELECT (optionally with WITH) is allowed, "
                f"got {type(root).__name__.upper()}"
            ],
        )
    bad = sorted(
        {
            type(n).__name__.upper()
            for n in root.find_all(*FORBIDDEN_NODE_TYPES)
        }
    )
    if bad:
        errors.append(f"forbidden statement types inside the query: {bad}")

    # ---- tables ----
    cte_names = {c.alias_or_name for c in root.find_all(exp.CTE)}
    alias_to_table: dict[str, str] = {}
    qualifiers = set(cte_names)
    used_tables = set()
    for t in root.find_all(exp.Table):
        full = ".".join(x for x in (t.catalog, t.db, t.name) if x)
        if not t.name:
            errors.append("table functions / external sources are not allowed")
        elif not t.catalog and not t.db and t.name in cte_names:
            qualifiers.add(t.alias_or_name)
        elif (
            t.catalog == settings.project_id
            and t.db == dataset
            and t.name in ANALYTICS_TABLES
        ):
            used_tables.add(t.name)
            alias_to_table[t.alias_or_name] = t.name
            alias_to_table[t.name] = t.name
        elif "INFORMATION_SCHEMA" in full.upper():
            errors.append("INFORMATION_SCHEMA is not allowed")
        else:
            errors.append(
                f"unapproved table {full!r}; only "
                f"`{settings.project_id}.{dataset}."
                f"<{'|'.join(ANALYTICS_TABLES)}>` may be used"
            )
    qualifiers |= set(alias_to_table)
    for sq in root.find_all(exp.Subquery):
        if sq.alias:
            qualifiers.add(sq.alias)

    # ---- functions ----
    for f in root.find_all(exp.Func):
        name = (
            f.name if isinstance(f, exp.Anonymous) else f.sql_name()
        ).upper()
        if (
            name in DENIED_FUNCTIONS
            or name.startswith(DENIED_FUNCTION_PREFIXES)
            or (isinstance(f, exp.Anonymous) and "." in name)
        ):
            errors.append(f"function {name} is not allowed")

    # ---- columns, stars, blocked columns ----
    aliases = {a.alias for a in root.find_all(exp.Alias)}
    for s in root.find_all(exp.Star):
        if not isinstance(s.parent, exp.Count):
            errors.append(
                "SELECT * is not allowed; list the columns explicitly"
            )
            break
    for c in root.find_all(exp.Column):
        if c.name == "*":
            continue
        if c.table:
            if c.table not in qualifiers:
                errors.append(
                    f"unknown table qualifier {c.table!r} in {c.sql('bigquery')}"
                )
            elif (
                c.table in alias_to_table
                and c.name not in TABLE_COLUMNS[alias_to_table[c.table]]
                and c.name not in aliases
            ):
                errors.append(
                    f"column {c.name!r} does not exist in table "
                    f"{alias_to_table[c.table]!r}"
                )
        elif c.name not in ALL_COLUMNS and c.name not in aliases:
            errors.append(f"unknown column {c.name!r}")
        if (
            c.name in BLOCKED_COLUMNS
            and not c.find_ancestor(exp.AggFunc)
            and _in_projection(c)
        ):
            errors.append(
                f"column {c.name!r} is personal data and may not be selected"
            )

    # ---- literals that should be parameters ----
    for lit in root.find_all(exp.Literal):
        if lit.is_string:
            v = lit.this
            if DATE_LITERAL_RE.match(v):
                errors.append(
                    f"date literal {v!r}: pass dates as "
                    "@start_date / @end_date DATE parameters"
                )
            elif ID_LITERAL_RE.match(v):
                errors.append(
                    f"identifier literal {v!r}: pass identifiers as @parameters"
                )
            elif "@" in v:
                errors.append("email-like literals are not allowed")

    # ---- parameters ----
    used = {p.name for p in root.find_all(exp.Parameter)}
    provided = {p.name: p for p in params}
    missing = sorted(used - set(provided))
    if missing:
        errors.append(f"parameters used in SQL but not provided: {missing}")
    if set(provided) - used:
        warnings.append(
            f"ignored unused parameters: {sorted(set(provided) - used)}"
        )
    converted = []
    for name in sorted(used & set(provided)):
        try:
            converted.append(convert_param(provided[name]))
        except ValueError as e:
            errors.append(f"invalid parameter: {e}")

    if errors:
        return ValidationResult(False, list(dict.fromkeys(errors)), warnings)

    # ---- row limit on the ORIGINAL text (sqlglot re-rendering can change details) ----
    final_sql = sql
    limit = root.args.get("limit")
    if limit is None:
        final_sql = f"{sql}\nLIMIT {max_rows}"
        warnings.append(f"added LIMIT {max_rows}")
        try:
            again = [
                s
                for s in sqlglot.parse(final_sql, read="bigquery")
                if s is not None
            ]
        except (ParseError, TokenError):
            again = []
        if (
            len(again) != 1
            or not isinstance(again[0], exp.Select)
            or again[0].args.get("limit") is None
        ):
            return ValidationResult(
                False,
                [
                    "could not safely apply a row limit; add LIMIT to the query"
                ],
                warnings,
            )
    else:
        lim = limit.expression
        if not (isinstance(lim, exp.Literal) and not lim.is_string):
            return ValidationResult(
                False, ["LIMIT must be a number"], warnings
            )
        if int(lim.this) > max_rows:
            warnings.append(
                f"LIMIT {lim.this} exceeds {max_rows}; "
                "the runner caps returned rows"
            )
    params_t = tuple(converted)
    vq = ValidatedQuery(
        final_sql,
        params_t,
        dataset,
        tuple(sorted(used_tables)),
        _sign(final_sql, params_t, dataset),
    )
    return ValidationResult(True, [], warnings, vq)