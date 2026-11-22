"""BigQuery access layer: settings, a guarded query runner and the dataset loader.

Every query in the project goes through `BigQueryRunner.run`, which:
  * optionally dry-runs first (bytes estimate + BigQuery's own list of referenced
    tables, checked against an allowlist),
  * executes with maximum_bytes_billed, a job timeout and a row cap,
  * classifies errors (auth, permission, not_found, invalid_sql, cost_limit,
    quota, timeout, unavailable) and never fabricates rows after an error.

Authentication is Application Default Credentials (gcloud auth
application-default login). The Gemini API key is unrelated and never used here.
"""
from __future__ import annotations

import base64
import concurrent.futures
import decimal
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from analytics_agent import schema_registry as sr

ANALYTICS_TABLES = tuple(sr.TABLE_LOAD_ORDER)


# ---------------------------------------------------------------- settings ----
@dataclass(frozen=True)
class BQSettings:
    """Immutable configuration for BigQuery access.

    Holds project/dataset names, location, cost and timeout limits, and helpers
    that build fully-qualified table identifiers and allow-lists used by the
    guarded runner.
    """

    project_id: str
    dataset_clean: str = "analytics_agent_dev"
    dataset_corrupted: str = "analytics_agent_corrupted"
    location: Optional[str] = "US"
    max_bytes_billed: int = 100_000_000
    timeout_s: int = 60
    max_result_rows: int = 1000

    def dataset(self, variant: str) -> str:
        """Return the dataset name for the given variant.

        Args:
            variant: Either ``"clean"`` or ``"corrupted"``.

        Returns:
            The corresponding dataset id.

        Raises:
            ValueError: If *variant* is not one of the two allowed values.
        """
        if variant not in ("clean", "corrupted"):
            raise ValueError("variant must be 'clean' or 'corrupted'")
        return self.dataset_clean if variant == "clean" else self.dataset_corrupted

    def full_id(self, dataset: str, table: str) -> str:
        """Return the fully-qualified table id ``project.dataset.table``."""
        return f"{self.project_id}.{dataset}.{table}"

    def sql_ref(self, dataset: str, table: str) -> str:
        """Return a backtick-quoted SQL reference for the table."""
        return f"`{self.full_id(dataset, table)}`"

    def refs(self, dataset: str) -> dict[str, str]:
        """Map each approved table name to its backtick-quoted SQL reference.

        Args:
            dataset: Dataset id (clean or corrupted).

        Returns:
            ``{table_name: `project.dataset.table`}`` for every table in
            ``ANALYTICS_TABLES``.
        """
        return {t: self.sql_ref(dataset, t) for t in ANALYTICS_TABLES}

    def allowed_tables(self, dataset: str) -> set[str]:
        """Return the set of fully-qualified table ids that may be queried."""
        return {self.full_id(dataset, t) for t in ANALYTICS_TABLES}

    @classmethod
    def from_app_settings(cls, s) -> "BQSettings":
        """Build a ``BQSettings`` instance from ``analytics_agent.config.Settings``.

        Args:
            s: Application settings object that exposes a ``.bigquery`` attribute
               containing the individual BigQuery configuration fields.

        Returns:
            A frozen ``BQSettings`` populated from *s*.
        """
        b = s.bigquery
        return cls(
            project_id=b.project_id,
            dataset_clean=b.dataset_clean,
            dataset_corrupted=b.dataset_corrupted,
            location=b.location,
            max_bytes_billed=b.maximum_bytes_billed,
            timeout_s=b.timeout_seconds,
            max_result_rows=b.max_result_rows,
        )


# ---------------------------------------------------------------- parameters/results ----
@dataclass(frozen=True)
class SqlParam:
    """Backend-neutral query parameter.

    Converted to a BigQuery (or DuckDB) parameter object by the runner.
    """

    name: str
    type: str  # STRING, INT64, FLOAT64, BOOL, DATE, TIMESTAMP, NUMERIC
    value: Any
    is_array: bool = False


@dataclass
class QueryResult:
    """Result of a single query execution (success, rejection or error).

    On success the result contains the returned rows (JSON-safe) together with
    cost and timing metadata. On failure ``status`` is ``"error"`` or
    ``"rejected"`` and ``error_kind`` / ``error_message`` describe the problem.
    Rows are never fabricated after an error.
    """

    status: str  # success | error | rejected
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)  # JSON-safe dicts
    row_count: int = 0
    total_rows: Optional[int] = None
    truncated: bool = False
    job_id: Optional[str] = None
    bytes_estimated: Optional[int] = None
    bytes_processed: Optional[int] = None
    bytes_billed: Optional[int] = None
    cache_hit: Optional[bool] = None
    elapsed_s: Optional[float] = None
    referenced_tables: list = field(default_factory=list)
    error_kind: Optional[str] = None
    error_message: Optional[str] = None
    backend: str = "bigquery"

    @property
    def ok(self) -> bool:
        """True when the query completed successfully."""
        return self.status == "success"

    @property
    def empty(self) -> bool:
        """True when the query succeeded but returned zero rows."""
        return self.ok and self.row_count == 0

    def metadata(self) -> dict:
        """Return a compact dictionary of status and cost/timing fields."""
        return {
            k: getattr(self, k)
            for k in (
                "status",
                "backend",
                "row_count",
                "total_rows",
                "truncated",
                "job_id",
                "bytes_estimated",
                "bytes_processed",
                "bytes_billed",
                "cache_hit",
                "elapsed_s",
                "error_kind",
            )
        }


def to_jsonable(value: Any) -> Any:
    """Convert a value to a JSON-serialisable Python object.

    Handles ``datetime``, ``date``, ``Decimal``, ``bytes``, and recursively
    processes lists, tuples and dicts. All other types are returned unchanged.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def classify_bigquery_error(
    exc: BaseException, max_bytes: int = 0, timeout_s: int = 0
) -> tuple[str, str]:
    """Map a client/API exception to an error kind and a beginner-friendly message.

    Args:
        exc: The exception raised by the BigQuery client or underlying libraries.
        max_bytes: The configured maximum-bytes-billed limit (used in cost-limit
            messages).
        timeout_s: The configured job timeout in seconds (used in timeout
            messages).

    Returns:
        A pair ``(kind, message)`` where *kind* is one of
        ``auth``, ``permission``, ``not_found``, ``invalid_sql``,
        ``cost_limit``, ``quota``, ``timeout``, ``unavailable`` or the generic
        ``error``.
    """
    from google.api_core import exceptions as gexc
    from google.auth import exceptions as auth_exc

    if isinstance(exc, (auth_exc.DefaultCredentialsError, auth_exc.RefreshError)):
        return (
            "auth",
            "Google credentials missing or expired. Run: gcloud auth application-default login",
        )
    reasons = [
        e.get("reason")
        for e in (getattr(exc, "errors", None) or [])
        if isinstance(e, dict) and e.get("reason")
    ]
    msg = str(getattr(exc, "message", "") or exc)
    low = (msg + " " + " ".join(reasons)).lower()
    if "bytesbilledlimitexceeded" in low or "bytes billed" in low:
        return "cost_limit", f"Query would exceed the {max_bytes:,}-byte billing limit."
    if any(r in low for r in ("quotaexceeded", "ratelimitexceeded", "quota exceeded")):
        return "quota", "BigQuery quota or rate limit reached. Wait and try again."
    if isinstance(exc, gexc.Forbidden) or "accessdenied" in low:
        return (
            "permission",
            (
                "Permission denied. The identity needs roles/bigquery.jobUser on the project and "
                "roles/bigquery.dataViewer on the dataset."
            ),
        )
    if isinstance(exc, gexc.NotFound):
        return (
            "not_found",
            "Table, dataset or project not found (check names and the dataset location).",
        )
    if isinstance(exc, gexc.BadRequest) or "invalidquery" in low:
        return "invalid_sql", f"BigQuery rejected the SQL: {msg[:300]}"
    if (
        isinstance(exc, (concurrent.futures.TimeoutError, gexc.DeadlineExceeded))
        or "timeout" in low
    ):
        return "timeout", f"Query did not finish within {timeout_s}s."
    if isinstance(
        exc, (gexc.ServiceUnavailable, gexc.InternalServerError, gexc.BadGateway)
    ):
        return "unavailable", "BigQuery is temporarily unavailable. Try again shortly."
    return "error", f"{type(exc).__name__}: {msg[:300]}"


def to_bigquery_params(params) -> list:
    """Convert a sequence of ``SqlParam`` (or already-built BigQuery parameters)
    into the list expected by ``QueryJobConfig.query_parameters``.

    Args:
        params: Iterable of ``SqlParam`` instances or pre-built BigQuery
            parameter objects. May be ``None``.

    Returns:
        List of ``bigquery.ScalarQueryParameter`` / ``ArrayQueryParameter``
        (or the original objects when they were already BigQuery parameters).
    """
    from google.cloud import bigquery

    out = []
    for p in params or []:
        if isinstance(p, SqlParam):
            out.append(
                bigquery.ArrayQueryParameter(p.name, p.type, list(p.value))
                if p.is_array
                else bigquery.ScalarQueryParameter(p.name, p.type, p.value)
            )
        else:  # already a bigquery parameter object
            out.append(p)
    return out


# ---------------------------------------------------------------- runner ----
class BigQueryRunner:
    """Guarded BigQuery query executor.

    Every call to ``run`` optionally performs a dry-run first (byte estimate +
    referenced-table allow-list check), then executes the query under the
    configured ``maximum_bytes_billed``, job timeout and row cap. Errors are
    classified and returned inside a ``QueryResult``; the method never raises
    for query-level failures and never invents rows after an error.
    """

    backend = "bigquery"

    def __init__(self, client, settings: BQSettings):
        """Initialise the runner.

        Args:
            client: An authenticated ``google.cloud.bigquery.Client``.
            settings: Immutable ``BQSettings`` that supply limits and identifiers.
        """
        self.client, self.settings = client, settings

    def run(
        self,
        sql: str,
        params=None,
        *,
        max_rows: Optional[int] = None,
        allowed_tables: Optional[set] = None,
        dry_run_first: bool = True,
    ) -> QueryResult:
        """Execute one read-only query under cost, timeout and row limits.

        Args:
            sql: The SQL statement to run.
            params: Optional sequence of ``SqlParam`` (or BigQuery parameter
                objects) to bind.
            max_rows: Maximum number of rows to return. Defaults to
                ``settings.max_result_rows``.
            allowed_tables: Optional set of fully-qualified table ids that the
                query may reference. When supplied, the dry-run's
                ``referenced_tables`` are checked against this set.
            dry_run_first: When ``True`` (default) a dry-run is performed first
                to obtain a byte estimate and the list of referenced tables.

        Returns:
            A ``QueryResult`` whose ``status`` is ``"success"``, ``"rejected"``
            (cost or unapproved table) or ``"error"``. On success the result
            contains the rows (capped) and cost/timing metadata. On failure
            ``error_kind`` and ``error_message`` describe the problem; no rows
            are fabricated.
        """
        from google.cloud import bigquery

        s = self.settings
        max_rows = max_rows or s.max_result_rows
        t0 = time.time()
        base = dict(
            use_legacy_sql=False,
            maximum_bytes_billed=s.max_bytes_billed,
            query_parameters=to_bigquery_params(params),
            labels={"app": "analytics-agent"},
        )
        estimated, job, refs = None, None, []
        try:
            if dry_run_first:
                dry = self.client.query(
                    sql,
                    job_config=bigquery.QueryJobConfig(
                        dry_run=True, use_query_cache=False, **base
                    ),
                    location=s.location,
                )
                estimated = int(dry.total_bytes_processed or 0)
                refs = sorted(
                    f"{t.project}.{t.dataset_id}.{t.table_id}"
                    for t in (dry.referenced_tables or [])
                )
                if estimated > s.max_bytes_billed:
                    return QueryResult(
                        "rejected",
                        bytes_estimated=estimated,
                        referenced_tables=refs,
                        error_kind="cost_limit",
                        error_message=(
                            f"Dry run estimates {estimated:,} bytes, above the "
                            f"{s.max_bytes_billed:,}-byte limit."
                        ),
                        elapsed_s=round(time.time() - t0, 2),
                    )
                if allowed_tables is not None and set(refs) - set(allowed_tables):
                    return QueryResult(
                        "rejected",
                        bytes_estimated=estimated,
                        referenced_tables=refs,
                        error_kind="unapproved_table",
                        error_message=(
                            f"Query references unapproved tables: "
                            f"{sorted(set(refs) - set(allowed_tables))}"
                        ),
                        elapsed_s=round(time.time() - t0, 2),
                    )
            cfg = bigquery.QueryJobConfig(**base)
            cfg.job_timeout_ms = s.timeout_s * 1000
            job = self.client.query(
                sql, job_config=cfg, location=s.location, timeout=s.timeout_s
            )
            it = job.result(timeout=s.timeout_s, max_results=max_rows + 1)
            columns = [f.name for f in (it.schema or [])]
            rows = [{k: to_jsonable(v) for k, v in row.items()} for row in it]
            total = it.total_rows if it.total_rows is not None else len(rows)
            truncated = len(rows) > max_rows or (total or 0) > max_rows
            rows = rows[:max_rows]
            return QueryResult(
                "success",
                columns,
                rows,
                len(rows),
                total,
                truncated,
                job.job_id,
                estimated,
                job.total_bytes_processed,
                job.total_bytes_billed,
                job.cache_hit,
                round(time.time() - t0, 2),
                refs,
            )
        except Exception as exc:  # noqa: BLE001 - classified and returned
            if isinstance(exc, concurrent.futures.TimeoutError) and job is not None:
                try:
                    self.client.cancel_job(job.job_id, location=s.location)
                except Exception:  # noqa: BLE001
                    pass
            kind, message = classify_bigquery_error(
                exc, s.max_bytes_billed, s.timeout_s
            )
            return QueryResult(
                "error",
                job_id=getattr(job, "job_id", None),
                bytes_estimated=estimated,
                referenced_tables=refs,
                error_kind=kind,
                error_message=message,
                elapsed_s=round(time.time() - t0, 2),
            )

    def table_schema(self, dataset: str, table: str) -> Optional[dict]:
        """Return the column-name → type mapping for a table, or ``None`` if missing.

        Args:
            dataset: Dataset id.
            table: Table name.

        Returns:
            ``{column: normalised_type}`` or ``None`` when the table does not
            exist. Types are normalised (e.g. ``INTEGER`` → ``INT64``).
        """
        from google.api_core import exceptions as gexc

        try:
            t = self.client.get_table(self.settings.full_id(dataset, table))
        except gexc.NotFound:
            return None
        return {f.name: _normalize_type(f.field_type) for f in t.schema}


def _normalize_type(t: str) -> str:
    """Normalise BigQuery type names to the canonical forms used by the schema registry."""
    return {"INTEGER": "INT64", "FLOAT": "FLOAT64", "BOOLEAN": "BOOL"}.get(t, t)


def make_client(settings: BQSettings):
    """Create a BigQuery client that uses Application Default Credentials.

    Args:
        settings: ``BQSettings`` supplying the project id and location.

    Returns:
        An authenticated ``google.cloud.bigquery.Client``.
    """
    from google.cloud import bigquery

    return bigquery.Client(project=settings.project_id, location=settings.location)


# ---------------------------------------------------------------- loader ----
def load_variant(
    client,
    settings: BQSettings,
    variant: str,
    data_dir: Path,
    *,
    allow_overwrite: bool = False,
    create_dataset: bool = True,
) -> list[dict]:
    """Load one dataset variant (clean / corrupted) from Parquet into BigQuery.

    * Creates the dataset (in ``settings.location``) if it is missing and
      ``create_dataset`` is true.
    * Refuses to touch existing tables unless ``allow_overwrite=True`` (then
      ``WRITE_TRUNCATE`` is used).
    * Parquet preserves exact types: ``DECIMAL(38,9)`` → ``NUMERIC``,
      ``date32`` → ``DATE``, ``timestamp(us, UTC)`` → ``TIMESTAMP``.
    * After each load the live schema is verified against the schema registry
      and the row count is checked against the Parquet file.

    Args:
        client: Authenticated BigQuery client.
        settings: ``BQSettings`` that supply project / dataset names and location.
        variant: ``"clean"`` or ``"corrupted"``.
        data_dir: Root directory that contains ``{variant}/parquet/{table}.parquet``.
        allow_overwrite: When ``True``, existing tables are truncated and
            reloaded; otherwise they are skipped.
        create_dataset: When ``True``, create the dataset if it does not exist.

    Returns:
        A list of per-table result dictionaries containing status, row counts
        and schema-verification details.

    Raises:
        FileNotFoundError: If a required Parquet file is missing.
        google.api_core.exceptions.NotFound: If the dataset is missing and
            ``create_dataset`` is ``False``.
    """
    from google.api_core import exceptions as gexc
    from google.cloud import bigquery
    import pyarrow.parquet as pq

    dataset_id = settings.dataset(variant)
    ds_ref = f"{settings.project_id}.{dataset_id}"
    try:
        client.get_dataset(ds_ref)
    except gexc.NotFound:
        if not create_dataset:
            raise
        ds = bigquery.Dataset(ds_ref)
        ds.location = settings.location
        client.create_dataset(ds)
    results = []
    for table in ANALYTICS_TABLES:
        path = Path(data_dir) / variant / "parquet" / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run: python scripts/generate_data.py"
            )
        full = settings.full_id(dataset_id, table)
        try:
            existing = client.get_table(full)
        except gexc.NotFound:
            existing = None
        if existing is not None and not allow_overwrite:
            results.append(
                {"table": table, "status": "skipped (exists)", "rows": existing.num_rows}
            )
            continue
        cfg = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=(
                bigquery.WriteDisposition.WRITE_TRUNCATE
                if existing is not None
                else bigquery.WriteDisposition.WRITE_EMPTY
            ),
            decimal_target_types=[
                bigquery.DecimalTargetType.NUMERIC,
                bigquery.DecimalTargetType.BIGNUMERIC,
            ],
        )
        with open(path, "rb") as fh:
            job = client.load_table_from_file(
                fh, full, job_config=cfg, location=settings.location
            )
        job.result(timeout=300)
        loaded = client.get_table(full)
        live = {f.name: _normalize_type(f.field_type) for f in loaded.schema}
        expected = {c.name: c.bq_type for c in sr.get_table(table).columns}
        mismatches = {
            c: (expected.get(c), live.get(c))
            for c in set(expected) | set(live)
            if expected.get(c) != live.get(c)
        }
        file_rows = pq.ParquetFile(path).metadata.num_rows
        results.append(
            {
                "table": table,
                "status": "loaded",
                "rows": loaded.num_rows,
                "file_rows": file_rows,
                "row_count_ok": loaded.num_rows == file_rows,
                "schema_ok": not mismatches,
                "schema_mismatches": mismatches,
            }
        )
    return results


# ---------------------------------------------------------------- bounded retries ----
class RetryingRunner:
    """Retry a query only when the failure is transient.

    Only errors whose ``error_kind`` is in ``RETRYABLE`` (currently only
    ``"unavailable"`` – BigQuery 5xx) are retried. Authentication, permission,
    invalid SQL, cost-limit, unapproved-table and quota errors are never
    retried: retrying cannot fix them and may waste quota. Retries are bounded
    and use exponential backoff.
    """

    RETRYABLE = frozenset({"unavailable"})

    def __init__(
        self,
        inner,
        max_retries: int = 2,
        base_delay_s: float = 1.0,
        sleep=time.sleep,
    ):
        """Wrap an inner runner with bounded exponential-backoff retries.

        Args:
            inner: The underlying runner (normally a ``BigQueryRunner``).
            max_retries: Maximum number of additional attempts after the first
                failure.
            base_delay_s: Initial sleep duration; doubled after each retry.
            sleep: Callable used to pause between attempts (injectable for tests).
        """
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self._sleep = sleep
        self.retries = 0  # total retries performed (for observability)

    def run(self, sql, params=None, **kwargs):
        """Execute the query, retrying only on transient failures.

        Args:
            sql: SQL statement.
            params: Optional query parameters.
            **kwargs: Forwarded unchanged to the inner runner.

        Returns:
            The final ``QueryResult`` (successful or the last non-retryable /
            exhausted error).
        """
        result = self.inner.run(sql, params, **kwargs)
        attempt = 0
        while (
            result.status == "error"
            and result.error_kind in self.RETRYABLE
            and attempt < self.max_retries
        ):
            attempt += 1
            self.retries += 1
            self._sleep(self.base_delay_s * (2 ** (attempt - 1)))
            result = self.inner.run(sql, params, **kwargs)
        return result

    def __getattr__(self, name):
        """Delegate any attribute not defined on this class to the inner runner."""
        return getattr(self.inner, name)