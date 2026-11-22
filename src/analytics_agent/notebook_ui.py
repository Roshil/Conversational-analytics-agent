"""Helpers for the problem-driven notebook (06): display, Gemini call budget, offline stand-in model.

* format_sql / show_response / show_turn / show_preview   readable output (only whitespace of SQL changes)
* GeminiBudget + BudgetedLLM + budgeted_adk_model          one shared call budget for the pipeline AND Google ADK,
                                                           so free-tier quotas (e.g. 20 requests/day/model) are respected
* OfflineStandInModel + offline_adk_model                  a deterministic stand-in for Gemini (NOT a language model):
                                                           maps questions to semantic-layer metrics so every problem can be
                                                           solved without an API key. Output is always labelled as such.
* CountingRunner                                           counts queries sent to BigQuery (or the emulator)
* preview_sql                                              plan -> SQL -> validate -> review -> dry run, without executing
* ProblemLog                                               records approach, outcome, Gemini calls and queries per problem
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from analytics_agent import semantic_layer as sem
from analytics_agent.schema_registry import REGIONS as REGION_NAMES
from analytics_agent.llm import LLMError, LLMCallMeta
from analytics_agent.testing import SchemaRoutedLLM


#formatting / display
_SQL_BREAKS = (r"(\bWITH\b|\bSELECT\b|\bFROM\b|\bLEFT JOIN\b|\bINNER JOIN\b|(?<!LEFT )(?<!INNER )\bJOIN\b|\bWHERE\b|"
               r"\bGROUP BY\b|\bHAVING\b|\bQUALIFY\b|\bORDER BY\b|\bLIMIT\b|\bUNION (?:ALL|DISTINCT)\b|\),\s*(?=\w+ AS \())")


def format_sql(sql: str) -> str:
    """Readable layout: line breaks before main clauses. Only whitespace changes; quoted strings are untouched."""
    parts = re.split(r"('(?:[^'\\]|\\.)*')", sql or "")
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            out.append(part)
            continue
        part = re.sub(r"\s+", " ", part)
        part = re.sub(_SQL_BREAKS, lambda m: ("),\n" if m.group(0).startswith(")") else "\n" + m.group(0)), part, flags=re.I)
        part = re.sub(r"\bBETWEEN\s+(\S+)\s+AND\s+", "BETWEEN \\1 \x00AND ", part, flags=re.I)
        part = re.sub(r"\s+(?<!\x00)(AND|OR)\s+", r"\n  \1 ", part, flags=re.I).replace("\x00", "")
        out.append(part)
    return re.sub(r"\n\s*\n", "\n", "".join(out)).strip()


def params_text(params) -> str:
    return ", ".join(f"@{p['name']} = {p['value']} ({p['type']})" for p in params or []) or "none"


def _display(obj):
    from IPython.display import display
    display(obj)


def _md(text: str):
    from IPython.display import Markdown
    _display(Markdown(text))


def show_response(resp, show_sql: bool = True, show_rows: bool = True, show_attempts: bool = True, model_label: str = "") -> None:
    """Descriptive rendering of an AnalyticsResponse: answer, period, checks, results table, executed SQL, attempts."""
    import pandas as pd
    period_src = next((st.get("period_source") for st in resp.steps if st.get("step") == "plan"), None)
    meta = [f"- **Status:** `{resp.status}` · intent `{resp.intent}` · metrics {resp.metrics or '-'} · answer source `{resp.answer_source}`"
            + (f" · model: {model_label}" if model_label else "")]
    if resp.period:
        meta.append(f"- **Period:** {resp.period.replace('..', ' to ')}" + (f" — {period_src}" if period_src else ""))
    for a in resp.assumptions:
        meta.append(f"- **Assumption:** {a}")
    if resp.row_count is not None:
        ex = resp.execution
        meta.append(f"- **Execution:** {resp.row_count} rows · backend {ex.get('backend')} · bytes billed {ex.get('bytes_billed')} · job `{ex.get('job_id')}`")
    if resp.review:
        meta.append(f"- **SQL review:** {'approved' if resp.review.get('sql_review_approved') else 'not approved'} "
                    f"({resp.review.get('sql_review_source')})")
    if resp.verification:
        meta.append(f"- **Answer verification:** {'passed' if resp.verification.get('ok') else 'failed → descriptive table shown instead'}"
                    + (f" ({'; '.join(resp.verification.get('issues') or [])})" if resp.verification.get("issues") else ""))
    if resp.trace_id:
        meta.append(f"- **Trace:** `{resp.trace_id}`")
    _md(f"#### Q: {resp.question}\n\n{resp.answer}\n\n" + "\n".join(meta))
    if show_rows and resp.rows and resp.answer_source != "deterministic_fallback":
        _display(pd.DataFrame(resp.rows).head(25))
    if show_sql:
        _md(f"**Executed SQL** (layout reformatted for reading):\n\n```sql\n{format_sql(resp.sql)}\n```\n\n**Parameters:** {params_text(resp.parameters)}"
            if resp.sql else "**Executed SQL:** none — answered without a query (data quality, clarification or refusal), "
                             "or no generated query passed validation.")
    attempts = [st for st in resp.steps if st.get("step") in ("generate_sql", "validate", "review_sql")]
    if show_attempts and sum(st["step"] == "generate_sql" for st in attempts) > 1:
        lines = ["**All SQL attempts** (rejected attempts are sent back with the reasons, at most twice):"]
        for st in attempts:
            if st["step"] == "generate_sql":
                lines.append(f"\n*Attempt {st.get('attempt')}:*\n\n```sql\n{format_sql(st.get('sql') or '')}\n```")
            elif st["step"] == "validate":
                lines.append(f"- validator: {'ok' if st.get('ok') else 'REJECTED: ' + '; '.join(st.get('errors') or [])}")
            else:
                lines.append(f"- reviewer: {'approved' if st.get('approved') else 'REJECTED: ' + '; '.join(st.get('issues') or [])}")
        _md("\n".join(lines))


def show_turn(text: str, turn, calls_used: Optional[int] = None) -> None:
    tools = [(c["tool"], c["args"]) for c in turn.tool_calls]
    _md(f"**USER:** {text}\n\n**ASSISTANT:** {turn.answer}\n\n"
        f"- tools called: `{tools}`\n"
        f"- verified against tool outputs: {turn.verified} · fallback used: {turn.fallback_used}"
        + (f" ({'; '.join(turn.issues)})" if turn.issues else "")
        + (f"\n- Gemini calls this turn: {calls_used}" if calls_used is not None else "")
        + (f"\n- trace: `{turn.trace_id}`" if turn.trace_id else ""))


#  Gemini budget 
class GeminiBudget:
    """One call budget shared by the pipeline (BudgetedLLM) and ADK (budgeted_adk_model)."""

    def __init__(self, limit: int):
        self.limit, self.used = limit, 0
        self.current = "setup"
        self.by_problem: dict[str, int] = {}

    def charge(self, source: str = "") -> None:
        if self.used >= self.limit:
            raise LLMError("quota", f"Notebook Gemini budget of {self.limit} calls reached (set GEMINI_CALL_BUDGET higher "
                                    f"or enable billing). No call was made.")
        self.used += 1
        self.by_problem[self.current] = self.by_problem.get(self.current, 0) + 1

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class BudgetedLLM:
    """GeminiStructuredClient-compatible wrapper that charges the shared budget before every call."""

    def __init__(self, inner, budget: GeminiBudget):
        self.inner, self.budget = inner, budget
        self.model = getattr(inner, "model", "")

    def generate(self, **kwargs):
        self.budget.charge("pipeline")
        return self.inner.generate(**kwargs)


def budgeted_adk_model(inner, budget: GeminiBudget):
    """Wrap an ADK model (e.g. Gemini) so every model call is charged to the shared budget."""
    from google.adk.models import BaseLlm

    class BudgetedAdkModel(BaseLlm):
        async def generate_content_async(self, llm_request, stream: bool = False):
            budget.charge("adk")
            async for response in inner.generate_content_async(llm_request, stream):
                yield response

    return BudgetedAdkModel(model=getattr(inner, "model", "budgeted"))


def is_quota_error(exc) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return getattr(exc, "kind", "") in ("quota", "rate_limit") or "RESOURCE_EXHAUSTED" in text or " 429" in text


#  runner counting 
class CountingRunner:
    """Pass-through runner that counts queries (to prove refused requests never reach BigQuery)."""

    def __init__(self, inner):
        self.inner, self.runs = inner, 0

    def run(self, *args, **kwargs):
        self.runs += 1
        return self.inner.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


#  offline stand-in model 
def _between(text: str, tag: str) -> str:
    """Content of the LAST <tag>...</tag> block (system texts may mention the tag name earlier)."""
    if f"</{tag}>" not in text:
        return ""
    return text.rsplit(f"<{tag}>", 1)[-1].split(f"</{tag}>", 1)[0].strip()


class OfflineStandInModel(SchemaRoutedLLM):
    """Deterministic stand-in for Gemini — NOT a language model.

    Planner: keyword rules + the semantic layer. SQL: canonical semantic-layer queries (with a few dimension
    variants). Answer: built only from the returned rows. Its purpose is to let every problem run without
    an API key; it is not a measure of how well Gemini performs.
    """
    model = "offline-stand-in (deterministic, not Gemini)"

    def __init__(self):
        from analytics_agent.agents.analytics_graph import AnalyticsPlan, AnswerDraft, SqlProposal
        self._Plan, self._Answer, self._Sql = AnalyticsPlan, AnswerDraft, SqlProposal
        super().__init__({AnalyticsPlan: self._plan, SqlProposal: self._sql, AnswerDraft: self._answer})

    def generate(self, *, system_instruction: str, prompt: str, schema, temperature: float = 0.0, max_output_tokens: int = 8192):
        # The SQL step needs the schema (table names), which lives in the system instruction.
        text = f"{system_instruction}\n\n{prompt}" if schema is self._Sql else prompt
        return super().generate(system_instruction=system_instruction, prompt=text, schema=schema)

    # ---- planner ----
    def _plan(self, prompt: str):
        from analytics_agent.agents.analytics_graph import normalize_metrics
        q = _between(prompt, "question") or prompt
        history = _between(prompt, "history")
        previous = re.findall(r"-> '(.*?)' \(intent metric_question", history)
        if previous and re.match(r"\s*(and|now|what about|how about|break|by|also|then)\b", q.lower()):
            q = f"{previous[-1]} {q}"                      # follow-up: join with the previous standalone question
        low = q.lower()
        counted = set(normalize_metrics([], q)) & {"missing_email_rate", "duplicate_customer_ids"}
        if counted:                                       # counts of missing/duplicate values are catalogue metrics
            return self._Plan(intent="metric_question", standalone_question=q, metrics=sorted(counted))
        if re.search(r"retention|churn|lifetime value|\bclv\b|\bltv\b", low):
            return self._Plan(intent="unsupported", standalone_question=q,
                              reason="Retention, churn and lifetime value are not defined in the metric registry; they need "
                                     "an agreed definition (for example repeat purchases within N days) before they can be calculated.")
        if re.search(r"acquisition cost|\bcac\b|forecast|predict|marketing spend|customer names?|e-?mail", low):
            return self._Plan(intent="unsupported", standalone_question=q, reason="That data is not in the warehouse tables.")
        if re.search(r"data[- ]quality|trust|duplicate|missing values|invalid data|issues? (with|in) the data", low):
            return self._Plan(intent="data_quality", standalone_question=q)
        if re.search(r"\beast\b|\bnorth\b|\bsouth\b(?!west|east)", low) and "region" in low:
            return self._Plan(intent="clarification", standalone_question=q, needs_clarification=True,
                              clarification_question="There is no region with that name. Did you mean Northeast, Southeast, Midwest, Southwest or West?")
        metrics = normalize_metrics([], q)
        if not metrics:
            return self._Plan(intent="clarification", standalone_question=q, needs_clarification=True,
                              clarification_question="Which metric do you mean? Ask 'what metrics can you calculate?' for the list.")
        return self._Plan(intent="metric_question", standalone_question=q, metrics=metrics)

    # ---- SQL ----
    def _sql(self, prompt: str):
        from analytics_agent.sql_validator import QueryParam
        m_refs = re.search(r"Table `([^`]+)\.customers`", prompt)
        if not m_refs:
            return self._Sql(needs_clarification=True, clarification_question="(stand-in) schema not found in the prompt.")
        refs = {t: f"`{m_refs.group(1)}.{t}`" for t in ("customers", "products", "orders", "order_items", "payments", "shipments", "refunds")}
        metrics = re.findall(r"'(\w+)'", (re.search(r"Metrics: \[(.*?)\]", prompt) or re.search(r"()", "")).group(1))
        question = _between(prompt, "question").lower()
        dates = re.search(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", prompt)
        params = ([QueryParam(name="start_date", type="DATE", values=[dates.group(1)]),
                   QueryParam(name="end_date", type="DATE", values=[dates.group(2)])] if dates else [])
        if not metrics:
            return self._Sql(needs_clarification=True, clarification_question="Which metric do you mean?")
        metric = next((m for m in metrics if m != "mom_change"), metrics[0]) if "mom_change" not in metrics else "mom_change"
        region = next((r for r in sorted(REGION_NAMES, key=len, reverse=True) if re.search(rf"\b{r.lower()}\b", question)), None)
        if metric in ("net_sales", "net_sales_by_dimension") and region:
            sql = self._net_sales_in_region(refs, question, region, by_dimension=(metric == "net_sales_by_dimension"))
        elif metric == "net_sales_by_dimension":
            sql = self._net_sales_by(refs, question)
        elif metric == "late_delivery_rate" and "carrier" in question:
            sql = (f"SELECT s.carrier, COUNT(*) AS delivered_shipments, "
                   f"COUNTIF(DATE(s.delivered_timestamp) > s.promised_delivery_date) AS late_shipments, "
                   f"ROUND(SAFE_DIVIDE(COUNTIF(DATE(s.delivered_timestamp) > s.promised_delivery_date), COUNT(*)), 4) AS late_rate "
                   f"FROM {refs['shipments']} s JOIN {refs['orders']} o ON o.order_id = s.order_id "
                   f"WHERE s.shipment_status = 'delivered' AND s.delivered_timestamp IS NOT NULL "
                   f"AND o.order_date BETWEEN @start_date AND @end_date GROUP BY s.carrier ORDER BY late_rate DESC")
        elif metric == "average_order_value" and not re.search(r"\b(by|per|each|across)\b.{0,20}\b(region|segment|channel|categor)", question):
            sql = self._overall_aov(refs)              # one overall figure unless a breakdown was asked for
        else:
            sql = sem.METRICS_BY_NAME[metric].render(refs)
        if "@start_date" not in sql:
            params = []
        return self._Sql(sql=sql, parameters=params, explanation=f"Canonical semantic-layer query for {metric}.")

    @staticmethod
    def _overall_aov(refs: dict) -> str:
        """Average order value as a single figure: merchandise after discounts / distinct booked orders."""
        return (f"SELECT ROUND(SUM(li.net_amount) / COUNT(DISTINCT o.order_id), 2) AS average_order_value,\n"
                f"       COUNT(DISTINCT o.order_id) AS orders\n"
                f"FROM {refs['orders']} o\nJOIN {refs['order_items']} li ON li.order_id = o.order_id\n"
                f"WHERE o.order_status IN {sem.BOOKED}\n  AND o.order_date BETWEEN @start_date AND @end_date")

    @staticmethod
    def _dimension(question: str):
        """(expr, alias, join-kind) for an explicit 'by <dimension>' request; otherwise the first dimension mentioned."""
        dims = {"categor": ("p.category", "category", "products"), "segment": ("c.customer_segment", "customer_segment", "customers"),
                "channel": ("lines.sales_channel", "sales_channel", ""), "region": ("c.region", "region", "customers")}
        found = re.findall(r"\bby\s+(?:product\s+|customer\s+|sales\s+)?(categor|segment|channel|region)", question)
        if found:                                      # the last one: a follow-up is appended to the previous question
            return dims[found[-1]]
        return next((dims[k] for k in ("categor", "segment", "channel", "region") if k in question), dims["categor"])

    @classmethod
    def _net_sales_in_region(cls, refs: dict, question: str, region: str, by_dimension: bool) -> str:
        """Net sales restricted to one customer region: monthly, or broken down by a dimension within the region."""
        lines = sem._LINES.format(**refs)
        joins = f"JOIN {refs['customers']} c ON c.customer_id = lines.customer_id"
        if by_dimension:
            expr, alias, kind = cls._dimension(question.replace(region.lower(), ""))
            if alias == "region":                      # "by region" inside one region makes no sense -> category
                expr, alias, kind = "p.category", "category", "products"
            if kind == "products":
                joins += f"\nJOIN {refs['products']} p ON p.product_id = lines.product_id"
            select, group, order = f"{expr} AS {alias}", alias, "net_sales DESC"
        else:
            select, group, order = "FORMAT_DATE('%Y-%m', lines.order_date) AS month", "month", "month"
        return (f"WITH {lines}\nSELECT {select}, "
                f"ROUND(SUM(lines.net_amount) - SUM(IFNULL(line_refunds.refunded, 0)), 2) AS net_sales\n"
                f"FROM lines {joins}\nLEFT JOIN line_refunds ON line_refunds.order_item_id = lines.order_item_id\n"
                f"WHERE c.region = '{region}'\nGROUP BY {group}\nORDER BY {order}")

    @staticmethod
    def _net_sales_by(refs: dict, question: str) -> str:
        dims = [("region", "c.region", "region", f"JOIN {refs['customers']} c ON c.customer_id = lines.customer_id"),
                ("segment", "c.customer_segment", "customer_segment", f"JOIN {refs['customers']} c ON c.customer_id = lines.customer_id"),
                ("channel", "lines.sales_channel", "sales_channel", ""),
                ("categor", "p.category", "category", f"JOIN {refs['products']} p ON p.product_id = lines.product_id")]
        explicit = re.findall(r"\bby\s+(?:product\s+|customer\s+|sales\s+)?(categor|segment|channel|region)", question)
        key, expr, alias, join = (next(d for d in dims if d[0] == explicit[-1]) if explicit
                                  else next((d for d in dims if d[0] in question), dims[-1]))
        lines = sem._LINES.format(**refs)
        return (f"WITH {lines}\nSELECT {expr} AS {alias}, "
                f"ROUND(SUM(lines.net_amount) - SUM(IFNULL(line_refunds.refunded, 0)), 2) AS net_sales\n"
                f"FROM lines {join}\nLEFT JOIN line_refunds ON line_refunds.order_item_id = lines.order_item_id\n"
                f"GROUP BY {alias}\nORDER BY net_sales DESC")

    # ---- answer (from rows only) ----
    def _answer(self, prompt: str):
        payload = json.loads(_between(prompt, "result") or "{}")
        rows, cols = payload.get("rows", []), payload.get("columns", [])
        if not rows:
            return self._Answer(summary="The query returned no rows.")
        fmt = lambda v: f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) >= 1000 else str(v)
        is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        num_cols = [c for c in cols if any(is_num(r.get(c)) for r in rows)]
        key_col = next((c for c in cols if any(isinstance(r.get(c), str) for r in rows)), None)   # NULLs allowed in some rows
        prefer = ("mom_pct", "net_sales", "average_order_value", "late_rate", "captured_payments", "refund_rate",
                  "cancellation_rate", "failure_rate", "order_refund_rate", "active_customers", "orders", "amount")
        metric_col = next((c for p in prefer for c in num_cols if c == p), num_cols[-1] if num_cols else None)
        if len(rows) == 1 or not key_col or not metric_col:
            r = rows[0]
            summary = "Result: " + ", ".join(f"{c} = {fmt(r[c])}" for c in cols) + "."
            points = [f"{c}: {fmt(r[c])}" for c in cols]
        else:
            valid = [r for r in rows if r.get(metric_col) is not None]
            hi = max(valid, key=lambda r: r[metric_col])
            lo = min(valid, key=lambda r: r[metric_col])
            name = lambda r: r[key_col] if r.get(key_col) is not None else "Unknown"
            summary = (f"Across {len(rows)} rows, {metric_col} is highest for {name(hi)} ({fmt(hi[metric_col])}) "
                       f"and lowest for {name(lo)} ({fmt(lo[metric_col])}).")
            points = [f"{r[key_col] if r[key_col] is not None else 'Unknown'}: {metric_col} = {fmt(r[metric_col])}" for r in rows[:8]]
        defs = re.search(r"Metric definitions: (.*?)\n", prompt)
        return self._Answer(summary=summary, key_points=points,
                            method=(defs.group(1)[:300] if defs else "Canonical semantic-layer query."))


def offline_adk_model():
    """Deterministic ADK stand-in (NOT Gemini): routes each turn to a tool and relays the tool's verified answer.
    Follow-ups (short questions starting with 'and', 'now', 'what about', ...) are joined to the previous question."""
    from analytics_agent.testing import adk_call, adk_text, last_function_response, make_scripted_adk_llm
    state = {"last_question": ""}

    def responder(req):
        last = req.contents[-1]
        if last.parts and getattr(last.parts[0], "function_response", None) is not None:
            payload = last_function_response(req) or {}
            if "metrics" in payload:
                return adk_text("I can calculate: " + "; ".join(m["label"] for m in payload["metrics"]) + ".")
            return adk_text(payload.get("answer") or "The tool returned no answer.")
        text = next((p.text for c in reversed(req.contents) if c.role == "user" for p in (c.parts or []) if p.text), "")
        low = text.lower()
        if re.search(r"metrics (can|do) you|what (can|metrics) you calculate|which metrics", low):
            return adk_call("list_metrics")
        if re.search(r"trust|data[- ]quality|duplicate|missing", low):
            return adk_call("check_data_quality", dataset="corrupted" if "corrupted" in low else "clean")
        follow_up = bool(re.match(r"\s*(and|now|what about|how about|break|by|also)\b", low)) and state["last_question"]
        question = f"{state['last_question']} {text}" if follow_up else text
        state["last_question"] = question
        return adk_call("ask_analytics", question=question, dataset="corrupted" if "corrupted" in low else "clean")

    return make_scripted_adk_llm(responder)


# ============================================================ SQL preview (no execution) ================
@dataclass
class SqlPreview:
    question: str
    status: str = "not_generated"         # ready | rejected | refused | clarification_needed | error
    sql: Optional[str] = None
    parameters: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    period: Optional[str] = None
    period_source: Optional[str] = None
    attempts: list = field(default_factory=list)
    dry_run_bytes: Optional[int] = None
    message: str = ""


def preview_sql(question: str, deps, bq_client=None, max_attempts: int = 2) -> SqlPreview:
    """Plan -> SQL -> validator -> SQL reviewer (-> BigQuery dry run). Never executes the query."""
    from analytics_agent.agents.analytics_graph import (AnalyticsPlan, SqlProposal, _resolve_dates, normalize_metrics,
                                                        plan_system, precheck_question, sql_system)
    from analytics_agent.agents.reviewers import review_sql
    from analytics_agent.bq import to_bigquery_params
    from analytics_agent.sql_validator import validate_sql
    out = SqlPreview(question)
    pc = precheck_question(question)
    if not pc["allowed"]:
        out.status, out.message = "refused", pc["reason"]
        return out
    try:
        plan, _ = deps.llm.generate(system_instruction=plan_system(), schema=AnalyticsPlan,
                                    prompt=f"<history>\n(no previous turns)\n</history>\n<question>\n{question}\n</question>")
    except LLMError as e:
        out.status, out.message = "error", f"planning failed ({e.kind})"
        return out
    if plan.intent != "metric_question":
        out.status = "clarification_needed" if plan.intent == "clarification" else "refused"
        out.message = plan.clarification_question or f"intent {plan.intent}: {plan.reason}"
        return out
    metrics = normalize_metrics(plan.metrics, f"{question} {plan.standalone_question}")
    start, end, label, _ = _resolve_dates({"question": question, "history": []}, plan, metrics)
    out.metrics, out.period, out.period_source = metrics, (f"{start} to {end}" if start else None), label
    feedback: list = []
    for attempt in range(1, max_attempts + 1):
        period_line = (f"Period (already resolved by the agent from the question; use exactly these dates for @start_date / "
                       f"@end_date): {start} to {end} - {label}" if start else "Period: none (snapshot metric; do not filter by date)")
        prompt = f"Metrics: {metrics}\n{period_line}\n<question>\n{plan.standalone_question}\n</question>"
        if feedback:
            prompt += "\n\nYour previous query was REJECTED. Fix every problem:\n- " + "\n- ".join(feedback)
        try:
            prop, _ = deps.llm.generate(system_instruction=sql_system(deps), prompt=prompt, schema=SqlProposal)
        except LLMError as e:
            out.status, out.message = "error", f"SQL generation failed ({e.kind})"
            return out
        if prop.needs_clarification:
            out.status, out.message = "clarification_needed", prop.clarification_question
            return out
        v = validate_sql(prop.sql, prop.parameters, deps.settings, deps.dataset)
        rec = {"attempt": attempt, "sql": prop.sql, "validator": "ok" if v.ok else v.errors, "reviewer": None}
        out.attempts.append(rec)
        if not v.ok:
            feedback = v.errors
            continue
        r = review_sql(v.validated, plan.standalone_question, metrics, start, end,
                       llm=deps.llm if deps.use_llm_sql_reviewer else None)
        rec["reviewer"] = "approved" if r.approved else r.issues
        if not r.approved:
            feedback = r.issues
            continue
        out.sql, out.parameters, out.status = v.validated.sql, v.validated.params_summary(), "ready"
        if bq_client is not None:
            from google.cloud import bigquery
            dry = bq_client.query(out.sql, job_config=bigquery.QueryJobConfig(
                dry_run=True, use_query_cache=False, query_parameters=to_bigquery_params(v.validated.params)),
                location=deps.settings.location)
            out.dry_run_bytes = dry.total_bytes_processed
        return out
    out.status, out.message = "rejected", f"no query passed validation and review in {max_attempts} attempts"
    return out


def show_preview(p: SqlPreview) -> None:
    lines = [f"#### SQL preview: {p.question}", "", f"- **Status:** `{p.status}`" + (f" — {p.message}" if p.message else "")]
    if p.metrics:
        lines.append(f"- **Metrics:** {p.metrics}")
    if p.period:
        lines.append(f"- **Period:** {p.period} — {p.period_source}")
    if p.dry_run_bytes is not None:
        lines.append(f"- **BigQuery dry run:** would process {p.dry_run_bytes:,} bytes (not executed)")
    if p.sql:
        lines += ["", "```sql", format_sql(p.sql), "```", "", f"**Parameters:** {params_text(p.parameters)}"]
    _md("\n".join(lines))


# ============================================================ problem log ===============================
class ProblemLog:
    """Collects, per problem: approach, tools, outcome, Gemini calls and queries run."""

    def __init__(self, budget: GeminiBudget, runner: CountingRunner, tracer=None):
        self.budget, self.runner, self.tracer = budget, runner, tracer
        self.rows: list[dict] = []

    @contextmanager
    def problem(self, pid: str, title: str, tools: str):
        rec = {"problem": pid, "title": title, "tools": tools, "outcome": "", "gemini_calls": 0, "queries": 0, "seconds": 0.0}
        self.budget.current = pid
        q0, t0 = self.runner.runs, time.time()
        span_cm = self.tracer.span(f"problem:{pid}", as_type="span", input={"title": title}) if self.tracer else None
        span = span_cm.__enter__() if span_cm else None
        try:
            yield rec
        finally:
            rec["gemini_calls"] = self.budget.by_problem.get(pid, 0)
            rec["queries"] = self.runner.runs - q0
            rec["seconds"] = round(time.time() - t0, 1)
            if span_cm:
                span.update(output={"outcome": rec["outcome"]})
                span_cm.__exit__(None, None, None)
            self.rows = [r for r in self.rows if r["problem"] != pid] + [rec]
            self.budget.current = "other"

    def frame(self):
        import pandas as pd
        return pd.DataFrame(sorted(self.rows, key=lambda r: int(r["problem"][1:])))


# ============================================================ Gemini -> stand-in routing ==============
class ModelRouter:
    """Use Gemini while the budget/quota allow; after a quota, rate-limit or budget error, switch to the offline stand-in
    for the rest of the run (clearly reported). Schemas the stand-in cannot produce raise, so callers use their own
    deterministic fallbacks (e.g. the DQ reviewer, executive summary, drill-down)."""

    def __init__(self, gemini=None, standin=None, label: str = "Gemini"):
        self.gemini, self.standin, self.label_gemini = gemini, standin or OfflineStandInModel(), label
        self.fallen_back = gemini is None
        self.reason = "" if gemini is not None else "Gemini not configured"
        self.counts = {"gemini": 0, "stand-in": 0}
        self.last = None
        self.model = getattr(gemini, "model", "") if gemini is not None else self.standin.model

    def generate(self, **kwargs):
        if not self.fallen_back:
            try:
                out = self.gemini.generate(**kwargs)
                self.counts["gemini"] += 1
                self.last = "gemini"
                return out
            except LLMError as e:
                if e.kind not in ("quota", "rate_limit"):
                    raise
                self.fallen_back, self.reason = True, f"{e.kind}: {e.message[:120]}"
                print(f"⚠ Gemini stopped ({e.kind}) — continuing with the offline stand-in for the rest of this run.")
        self.counts["stand-in"] += 1
        self.last = "stand-in"
        return self.standin.generate(**kwargs)

    def label(self) -> str:
        if self.fallen_back:
            return f"offline stand-in (NOT Gemini; {self.reason})"
        return self.label_gemini


def adk_model_with_fallback(gemini_model, offline_model):
    """ADK model that uses Gemini until a quota/rate-limit/budget error, then the deterministic offline ADK model."""
    from google.adk.models import BaseLlm
    state = {"fallen_back": gemini_model is None}

    class FallbackAdkModel(BaseLlm):
        async def generate_content_async(self, llm_request, stream: bool = False):
            if not state["fallen_back"]:
                try:
                    async for response in gemini_model.generate_content_async(llm_request, stream):
                        yield response
                    return
                except Exception as exc:  # noqa: BLE001
                    if not is_quota_error(exc):
                        raise
                    state["fallen_back"] = True
                    print(f"⚠ Gemini (ADK) stopped ({type(exc).__name__}) — continuing with the offline ADK stand-in.")
            async for response in offline_model.generate_content_async(llm_request, stream):
                yield response

    m = FallbackAdkModel(model=getattr(gemini_model, "model", "adk-fallback"))
    m.__dict__["_fallback_state"] = state
    return m


# ============================================================ LangGraph drawing (no extra packages) =======
from collections import Counter, defaultdict
from html import escape

NODE_ROLES = {
    "precheck": ("code", "Blocks write requests, prompt injection and unapproved tables before any model call; detects customer lookups."),
    "customer_lookup": ("code + BigQuery", "Record lookup by customer ID: fixed parameterised query, no model, emails masked by default."),
    "order_item_lookup": ("code + BigQuery", "Order item / order lookup by ID: fixed parameterised query, no model, order totals from the lines."),
    "plan": ("Gemini + code", "Gemini classifies the intent and picks documented metrics; the period is resolved in code."),
    "data_quality": ("code + reviewer", "Answers data-quality questions from the deterministic report, without writing SQL."),
    "generate_sql": ("Gemini", "Writes BigQuery SQL grounded in the schema, join rules and canonical metric SQL."),
    "validate": ("code", "sqlglot validator: one read-only SELECT, allowlisted tables, no email column, parameterised dates."),
    "review_sql": ("code (+ Gemini)", "SQL & Analytics Reviewer: fan-out, booked-status filters, refunds, denominators, period."),
    "execute": ("BigQuery", "Dry run, byte cap, timeout and row cap; then the query runs read-only."),
    "answer": ("Gemini", "Descriptive answer built only from the returned rows."),
    "verify": ("code", "Every number, date and name in the answer must appear in the rows; otherwise a table is shown."),
    "caveats": ("code", "Adds failing data-quality checks for the tables the query read."),
    "finalize": ("code", "Assembles the response: answer, SQL, parameters, checks, caveats and run trace."),
}


# ----------------------------------------------------------------------------- graph drawing
def main_path(graph_view) -> list[str]:
    """Follow the 'happy path' from start to finish; nodes off this path are drawn as side branches."""
    out = defaultdict(list)
    for e in graph_view.edges:
        out[e.source].append(e.target)
    path, seen, current = ["__start__"], {"__start__"}, "__start__"
    while True:
        nxt = [t for t in out[current] if t not in seen and t not in ("finalize", "__end__")]
        if not nxt:
            break
        current = max(nxt, key=lambda t: len(out[t]))      # the branch that continues the pipeline
        path.append(current)
        seen.add(current)
    return path + ["finalize", "__end__"]


def graph_svg(graph_view, run_nodes=None, title=None) -> str:
    """Draw the compiled LangGraph as SVG (no extra packages).
    Without run_nodes: the full graph. With run_nodes (the nodes one question visited, in order): the route taken is
    highlighted, retried steps get a xN badge and unused steps are faded."""
    path = main_path(graph_view)
    w, h, gap, cx, top = 170, 34, 56, 300, (70 if title else 40)
    pos = {n: (cx, top + i * gap) for i, n in enumerate(path)}
    side = [n for n in graph_view.nodes if n not in pos]
    for n in side:
        parent = next(e.source for e in graph_view.edges if e.target == n)
        pos[n] = (cx + 230, pos[parent][1] + gap / 2)
    visits = Counter(run_nodes or [])
    taken = set(zip(run_nodes, run_nodes[1:])) if run_nodes else set()
    height = top + len(path) * gap + 70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="660" height="{height}" font-family="sans-serif" font-size="12">',
             '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker>'
             '<marker id="g" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#15803d"/></marker>'
             '<marker id="f" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#cbd5e1"/></marker></defs>']
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')   # readable in dark themes too
    if title:
        parts.append(f'<text x="20" y="28" font-size="13" font-weight="bold" fill="#111827">{escape(title[:95])}</text>')
    right_lane, left_lane = 0, 0
    for e in graph_view.edges:
        (x1, y1), (x2, y2) = pos[e.source], pos[e.target]
        if run_nodes is None:
            color, width, marker = "#555", 1.3, "a"
        elif (e.source, e.target) in taken:
            color, width, marker = "#15803d", 2.6, "g"
        else:
            color, width, marker = "#cbd5e1", 1.1, "f"
        dash = ' stroke-dasharray="5,4"' if e.conditional and (e.source, e.target) not in taken else ""
        style = f'fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{marker})"'
        i1, i2 = (path.index(e.source) if e.source in path else None), (path.index(e.target) if e.target in path else None)
        if i1 is not None and i2 is not None and i2 == i1 + 1:                     # next step on the main path
            parts.append(f'<line x1="{x1}" y1="{y1 + h/2}" x2="{x2}" y2="{y2 - h/2 - 2}" {style}/>')
        elif e.target in side:                                                      # branch out to a side node
            parts.append(f'<path d="M{x1 + w/2},{y1} L{x2 - w/2 - 2},{y2}" {style}/>')
        elif i1 is not None and i2 is not None and i2 < i1:                         # retry loop (left side)
            lx = cx - w / 2 - 24 - left_lane * 16; left_lane += 1
            parts.append(f'<path d="M{x1 - w/2},{y1} L{lx},{y1} L{lx},{y2} L{x2 - w/2 - 2},{y2}" {style}/>')
            parts.append(f'<text x="{lx - 4}" y="{(y1 + y2) / 2}" text-anchor="end" fill="#b45309">retry</text>')
        else:                                                                       # early exit (right side)
            lx = cx + w / 2 + 30 + right_lane * 14 if e.source not in side else x1 + w / 2 + 20
            right_lane += e.source not in side
            parts.append(f'<path d="M{x1 + w/2},{y1} L{lx},{y1} L{lx},{y2} L{x2 + w/2 + 2},{y2}" {style}/>')
    for n, (x, y) in pos.items():
        terminal = n.startswith("__")
        role = NODE_ROLES.get(n, ("", ""))[0]
        fill = "#e5e7eb" if terminal else ("#dbeafe" if role.startswith("Gemini") else "#fef3c7" if role.startswith("BigQuery") else "#dcfce7")
        used = run_nodes is None or visits[n] > 0
        if not used:                                       # explicit grey instead of opacity (renders everywhere)
            fill, stroke, sw, text_color = "#f3f4f6", "#d1d5db", 1, "#9ca3af"
        elif run_nodes is not None:
            stroke, sw, text_color = "#15803d", 2.4, "#111827"
        else:
            stroke, sw, text_color = "#374151", 1, "#111827"
        label = {"__start__": "START", "__end__": "END"}.get(n, n)
        parts.append(f'<rect x="{x - w/2}" y="{y - h/2}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
                     f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-weight="bold" fill="{text_color}">{escape(label)}</text>')
        if visits[n] > 1:
            parts.append(f'<circle cx="{x + w/2}" cy="{y - h/2}" r="11" fill="#b45309"/>'
                         f'<text x="{x + w/2}" y="{y - h/2 + 4}" text-anchor="middle" fill="white" font-size="11" font-weight="bold">×{visits[n]}</text>')
    ly = height - 40
    legend = ("green = route taken by this question · faded = not used · ×N = step repeated (bounded retry)" if run_nodes is not None
              else "solid = always · dashed = conditional · left = bounded retry (max 2) · right = early exit (refusal, error)")
    parts += [f'<rect x="20" y="{ly}" width="14" height="14" fill="#dbeafe" stroke="#374151"/><text x="40" y="{ly + 11}" fill="#111827">Gemini</text>',
              f'<rect x="110" y="{ly}" width="14" height="14" fill="#dcfce7" stroke="#374151"/><text x="130" y="{ly + 11}" fill="#111827">deterministic code</text>',
              f'<rect x="270" y="{ly}" width="14" height="14" fill="#fef3c7" stroke="#374151"/><text x="290" y="{ly + 11}" fill="#111827">BigQuery</text>',
              f'<text x="20" y="{ly + 32}" fill="#374151">{legend}</text>', "</svg>"]
    return "\n".join(parts)




def show_graph(graph, run_nodes=None, title=None):
    """Display the compiled graph: LangGraph's Mermaid renderer if reachable, otherwise the local SVG drawing."""
    from IPython.display import SVG, Image
    import pandas as pd
    view = graph.get_graph()
    if run_nodes is None:
        try:
            _display(Image(view.draw_mermaid_png(max_retries=1, retry_delay=0.5)))
            return
        except Exception:  # noqa: BLE001 - offline or blocked: draw locally
            pass
    _display(SVG(graph_svg(view, run_nodes=run_nodes, title=title)))
    if run_nodes is None:
        _display(pd.DataFrame([{"node": n, "done by": r[0], "what it does": r[1]} for n, r in NODE_ROLES.items()]))