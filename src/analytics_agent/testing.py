"""Offline test doubles. Clearly NOT a model: they replay scripted responses.

ScriptedGeminiClient mimics the small part of google-genai used by
analytics_agent.llm (client.interactions.create -> object with status,
output_text, usage). Use it to test retry, revision and escalation logic
without network access or API cost.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def fake_api_error(code: int, status: str, message: str):
    """Build the exception the Interactions API raises for an HTTP error.

    Uses the SDK's own error factory (private module, pinned SDK version), so
    tests exercise the real exception types. Falls back to
    ``google.genai.errors`` if that module moves in a future SDK.

    Args:
        code: HTTP status code (e.g. 429, 500).
        status: API status string (e.g. ``"RESOURCE_EXHAUSTED"``).
        message: Human-readable error message.

    Returns:
        An exception instance of the same type the real SDK would raise.
    """
    import httpx

    body = {"error": {"code": code, "status": status, "message": message}}
    try:
        from google.genai._gaos.lib import compat_errors as ce

        response = httpx.Response(
            code, request=httpx.Request("POST", _URL), json=body
        )
        return ce.APIError.generate(code, body, None, response)
    except ImportError:  # pragma: no cover
        from google.genai import errors

        return (
            errors.ServerError if code >= 500 else errors.ClientError
        )(code, body)


def fake_connection_error(timeout: bool = False):
    """Build the Interactions API's connection or timeout error.

    Args:
        timeout: When ``True`` return a timeout error; otherwise a connection
            error.

    Returns:
        An exception matching the real SDK's connection/timeout types.
    """
    import httpx

    req = httpx.Request("POST", _URL)
    try:
        from google.genai._gaos.lib import compat_errors as ce

        return (
            ce.APITimeoutError(req)
            if timeout
            else ce.APIConnectionError(request=req)
        )
    except ImportError:  # pragma: no cover
        return (
            httpx.ReadTimeout("timeout", request=req)
            if timeout
            else httpx.ConnectError("no route", request=req)
        )


def fake_models_api_error(code: int, status: str, message: str):
    """Build a ``google.genai.errors.ClientError`` / ``ServerError``.

    Used for the ``client.models.*`` family of calls (as opposed to the
    Interactions API).

    Args:
        code: HTTP status code.
        status: API status string.
        message: Human-readable error message.

    Returns:
        A ``ClientError`` (4xx) or ``ServerError`` (5xx) instance.
    """
    from google.genai import errors

    body = {"error": {"code": code, "status": status, "message": message}}
    return (
        errors.ServerError if code >= 500 else errors.ClientError
    )(code, body)


class _Interactions:
    """Internal stand-in for ``client.interactions``.

    Each call to ``create`` pops the next item from the parent's script and
    either raises it (when it is an exception) or returns a
    ``SimpleNamespace`` mimicking a completed interaction response.
    """

    def __init__(self, parent: "ScriptedGeminiClient"):
        self.parent = parent

    def create(self, **kwargs) -> Any:
        """Replay the next scripted response (or raise the next scripted error).

        Args:
            **kwargs: Arguments that would have been sent to the real API;
                recorded on ``parent.calls`` for assertions.

        Returns:
            A ``SimpleNamespace`` with ``status``, ``output_text`` and
            ``usage`` attributes.

        Raises:
            RuntimeError: When the script is exhausted.
            BaseException: When the next script item is an exception instance.
        """
        self.parent.calls.append(kwargs)
        if not self.parent.script:
            raise RuntimeError("ScriptedGeminiClient: script exhausted")
        item = self.parent.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        status = "completed"
        if isinstance(item, tuple):
            status, item = item
        return SimpleNamespace(
            status=status,
            output_text=item,
            usage=SimpleNamespace(
                total_input_tokens=1000,
                total_output_tokens=500,
                total_thought_tokens=0,
            ),
        )


class ScriptedGeminiClient:
    """Test double for the small subset of google-genai used by the LLM client.

    Mimics ``client.interactions.create`` so retry, revision and escalation
    logic can be exercised offline without network access or API cost.

    Args:
        script: Ordered list of responses.  Each item is either:

            * a string (returned as ``output_text`` with status ``"completed"``);
            * a ``(status, output_text)`` tuple; or
            * an exception instance that is raised on that call.

    Attributes:
        script: Remaining items still to be replayed.
        calls: Every kwargs dict passed to ``interactions.create``.
        interactions: The ``_Interactions`` stand-in.
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []
        self.interactions = _Interactions(self)


# =============================================================================================
# DuckDB emulator for BigQuery SQL (TEST DOUBLE ONLY).
# Executes the project's BigQuery Standard SQL on the local Parquet files by transpiling it with
# sqlglot to DuckDB. Used to test SQL logic offline against the pandas oracle and gold answers.
# It is NOT BigQuery: dialect differences are possible, which is why the notebook also runs a
# live parity check against BigQuery.
# =============================================================================================


class DuckDBRunner:
    """Offline BigQuery stand-in that runs SQL against local Parquet files via DuckDB.

    BigQuery Standard SQL is transpiled with sqlglot; fully-qualified table
    references are remapped to DuckDB views backed by the generated Parquet
    files.  Useful for testing SQL logic offline against the pandas oracle and
    gold answers.  **Not** a substitute for live BigQuery parity checks.
    """

    backend = "duckdb-emulator"

    def __init__(self, settings, data_dir, variants=("clean", "corrupted")):
        """Create DuckDB views for every analytics table in each variant.

        Args:
            settings: Settings object exposing ``.dataset(variant)`` and
                ``.max_result_rows``.
            data_dir: Root directory containing
                ``{variant}/parquet/{table}.parquet``.  Relative paths are
                resolved against the project root.
            variants: Dataset variants to register (default both clean and
                corrupted).

        Raises:
            FileNotFoundError: When any required Parquet file is missing.
        """
        import duckdb
        from pathlib import Path

        from analytics_agent.bq import ANALYTICS_TABLES

        data_dir = Path(data_dir)
        if not data_dir.is_absolute():  # resolve against the project root, not the CWD
            from analytics_agent.config import find_project_root

            data_dir = find_project_root() / data_dir
        missing = [
            str(data_dir / v / "parquet" / f"{t}.parquet")
            for v in variants
            for t in ANALYTICS_TABLES
            if not (data_dir / v / "parquet" / f"{t}.parquet").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} generated Parquet file(s) are missing, e.g. {missing[0]}.\n"
                "Generate them from the project root with:  python scripts/generate_data.py\n"
                "(the zip does not include data/generated; it is reproducible from the seed)."
            )
        self.settings = settings
        self.con = duckdb.connect()
        self.con.execute("SET TimeZone = 'UTC'")
        self.queries: list[str] = []
        for v in variants:
            ds = settings.dataset(v)
            for t in ANALYTICS_TABLES:
                path = Path(data_dir) / v / "parquet" / f"{t}.parquet"
                self.con.execute(
                    f"CREATE VIEW {ds}__{t} AS SELECT * FROM read_parquet('{path.as_posix()}')"
                )

    def _translate(self, sql: str) -> tuple[str, list]:
        """Transpile BigQuery SQL to DuckDB and collect fully-qualified table refs.

        Args:
            sql: BigQuery Standard SQL statement.

        Returns:
            A pair ``(duckdb_sql, referenced_tables)`` where *referenced_tables*
            is a list of ``project.dataset.table`` strings found in the original
            query.
        """
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="bigquery")
        refs = []

        def remap(node):
            if isinstance(node, exp.Table) and node.db:
                refs.append(f"{node.catalog}.{node.db}.{node.name}")
                return exp.Table(
                    this=exp.to_identifier(f"{node.db}__{node.name}"),
                    alias=node.args.get("alias"),
                )
            return node

        return tree.transform(remap).sql(dialect="duckdb"), refs

    def run(
        self,
        sql,
        params=None,
        *,
        max_rows=None,
        allowed_tables=None,
        dry_run_first=True,
    ):
        """Execute transpiled SQL against the local DuckDB views.

        Mirrors the ``BigQueryRunner.run`` signature so the same pipeline code
        can run offline.  ``dry_run_first`` is accepted but ignored (there is
        no cost model).

        Args:
            sql: BigQuery Standard SQL statement.
            params: Optional sequence of ``SqlParam`` (or BigQuery parameter
                objects).
            max_rows: Row cap (defaults to ``settings.max_result_rows``).
            allowed_tables: Optional set of fully-qualified table ids; queries
                that reference anything outside this set are rejected.
            dry_run_first: Accepted for API compatibility; not used.

        Returns:
            A ``QueryResult`` with ``backend="duckdb-emulator"``.
        """
        import time

        from analytics_agent.bq import QueryResult, SqlParam, to_jsonable

        t0 = time.time()
        max_rows = max_rows or self.settings.max_result_rows
        try:
            duck_sql, refs = self._translate(sql)
        except Exception as exc:  # noqa: BLE001
            return QueryResult(
                "error",
                error_kind="invalid_sql",
                error_message=f"parse: {exc}"[:300],
                backend=self.backend,
            )
        if allowed_tables is not None and set(refs) - set(allowed_tables):
            return QueryResult(
                "rejected",
                error_kind="unapproved_table",
                referenced_tables=sorted(refs),
                error_message=(
                    f"Query references unapproved tables: "
                    f"{sorted(set(refs) - set(allowed_tables))}"
                ),
                backend=self.backend,
            )
        values = {}
        for p in params or []:
            if isinstance(p, SqlParam):
                values[p.name] = list(p.value) if p.is_array else p.value
            else:  # bigquery parameter object
                values[p.name] = (
                    list(p.values)
                    if hasattr(p, "values") and not hasattr(p, "value")
                    else p.value
                )
        import re as _re

        used = set(_re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", duck_sql))
        values = {k: v for k, v in values.items() if k in used}
        try:
            self.queries.append(duck_sql)
            cur = (
                self.con.execute(duck_sql, values)
                if values
                else self.con.execute(duck_sql)
            )
            columns = [d[0] for d in cur.description]
            fetched = cur.fetchmany(max_rows + 1)
            rows = [
                {c: to_jsonable(v) for c, v in zip(columns, r)} for r in fetched
            ]
        except Exception as exc:  # noqa: BLE001
            return QueryResult(
                "error",
                error_kind="invalid_sql",
                error_message=str(exc)[:300],
                backend=self.backend,
                referenced_tables=sorted(set(refs)),
            )
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        return QueryResult(
            "success",
            columns,
            rows,
            len(rows),
            len(rows) if not truncated else None,
            truncated,
            job_id="duckdb-local",
            elapsed_s=round(time.time() - t0, 3),
            referenced_tables=sorted(set(refs)),
            backend=self.backend,
        )

    def table_schema(self, dataset, table):
        """Return the column-name → BigQuery-style type mapping for a view.

        Args:
            dataset: Dataset id (used as the view-name prefix).
            table: Table name.

        Returns:
            ``{column: type}`` or ``None`` when the view does not exist.
            DuckDB types are normalised toward the schema-registry conventions
            (e.g. ``VARCHAR`` → ``STRING``, ``DECIMAL…`` → ``NUMERIC``).
        """
        type_map = {
            "VARCHAR": "STRING",
            "BIGINT": "INT64",
            "INTEGER": "INT64",
            "BOOLEAN": "BOOL",
            "DATE": "DATE",
            "TIMESTAMP WITH TIME ZONE": "TIMESTAMP",
            "DOUBLE": "FLOAT64",
        }
        try:
            rows = self.con.execute(f"DESCRIBE {dataset}__{table}").fetchall()
        except Exception:  # noqa: BLE001
            return None
        return {
            r[0]: (
                "NUMERIC" if r[1].startswith("DECIMAL") else type_map.get(r[1], r[1])
            )
            for r in rows
        }


class SchemaRoutedLLM:
    """Test double implementing the ``GeminiStructuredClient.generate`` interface.

    Routes each call to a handler keyed by the Pydantic schema class.  Records
    every call for later assertions.

    Attributes:
        model: Fixed model name reported in call metadata.
        handlers: Mapping of schema class → callable ``(prompt) -> instance | Exception``.
        calls: List of ``(schema_name, prompt)`` pairs recorded in order.
    """

    model = "schema-routed-fake"

    def __init__(self, handlers: dict):
        """Initialise with a schema → handler mapping.

        Args:
            handlers: ``{PydanticSchemaClass: callable(prompt) -> instance | Exception}``.
        """
        self.handlers, self.calls = handlers, []

    def generate(
        self,
        *,
        system_instruction: str,
        prompt: str,
        schema,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
    ):
        """Dispatch to the handler registered for *schema*.

        Args:
            system_instruction: Ignored (present for interface compatibility).
            prompt: Forwarded to the handler.
            schema: Pydantic model class used as the routing key.
            temperature: Ignored.
            max_output_tokens: Ignored.

        Returns:
            A pair ``(parsed_instance, LLMCallMeta)``.

        Raises:
            LLMError: When no handler is registered for *schema*, or when the
                handler returns an exception instance (which is then raised).
        """
        from analytics_agent.llm import LLMCallMeta, LLMError

        self.calls.append((schema.__name__, prompt))
        h = self.handlers.get(schema)
        if h is None:
            raise LLMError(
                "invalid_request", f"no fake handler for {schema.__name__}"
            )
        out = h(prompt)
        if isinstance(out, Exception):
            raise out
        return out, LLMCallMeta(model=self.model, attempts=1)


def make_scripted_adk_llm(responder):
    """Build a test-double ADK ``BaseLlm`` whose replies come from *responder*.

    Defined lazily so importing ``analytics_agent.testing`` does not require
    ``google-adk``.

    Args:
        responder: Callable ``(LlmRequest) -> LlmResponse`` invoked once per
            generation request.

    Returns:
        A ``ScriptedAdkLlm`` instance ready to pass to ADK agents.
    """
    from typing import AsyncGenerator

    from google.adk.models import BaseLlm, LlmRequest, LlmResponse

    class ScriptedAdkLlm(BaseLlm):
        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield responder(llm_request)

    return ScriptedAdkLlm(model="scripted-adk")


def adk_text(text: str):
    """Build an ADK ``LlmResponse`` that contains a single text part.

    Args:
        text: Plain-text content for the model turn.

    Returns:
        An ``LlmResponse`` suitable for returning from a scripted ADK LLM.
    """
    from google.adk.models import LlmResponse
    from google.genai import types

    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part(text=text)]
        )
    )


def adk_call(name: str, **args):
    """Build an ADK ``LlmResponse`` that issues a function call.

    Args:
        name: Tool / function name to invoke.
        **args: Arguments passed to the function call.

    Returns:
        An ``LlmResponse`` containing a single ``FunctionCall`` part.
    """
    from google.adk.models import LlmResponse
    from google.genai import types

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name=name, args=args)
                )
            ],
        )
    )


def last_function_response(llm_request):
    """Return the most recent function-response payload in an ADK ``LlmRequest``.

    Walks ``llm_request.contents`` in reverse looking for a part that carries
    a ``function_response``.

    Args:
        llm_request: The ADK request object for the current turn.

    Returns:
        The ``response`` dict from the latest function response, or ``None``
        when none is present.
    """
    for content in reversed(llm_request.contents or []):
        for part in content.parts or []:
            if getattr(part, "function_response", None):
                return part.function_response.response
    return None