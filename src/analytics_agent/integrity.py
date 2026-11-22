"""Key and relationship integrity checks (ported from the user's notebook cells to the guarded runner).

Works on BigQuery and on the offline emulator because every query goes through runner.run
(maximum_bytes_billed, timeout). Keys come from the schema registry:
  * primary keys: NULL/blank values and duplicate rows (extra rows beyond the first);
  * business keys: duplicates (NULL/blank keys excluded);
  * foreign keys: non-NULL child values without a parent row.
Example records are returned with email values masked.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from analytics_agent import schema_registry as sr
from analytics_agent.observability import redact

KEY_COLUMNS = {name: list(t.primary_key) for name, t in sr.TABLES.items()}
UNIQUE_KEYS = {name: [list(k) for k in t.business_keys] for name, t in sr.TABLES.items() if t.business_keys}
FOREIGN_KEYS = {(name, fk.column): (fk.ref_table, fk.ref_column) for name, t in sr.TABLES.items() for fk in t.foreign_keys}


def _q(col: str) -> str:
    return "`" + col.replace("`", "\\`") + "`"


def _scalar(runner, sql: str) -> int:
    res = runner.run(sql, max_rows=1, dry_run_first=False)
    if not res.ok:
        raise RuntimeError(f"[{res.error_kind}] {res.error_message}")
    return int(next(iter(res.rows[0].values())) or 0)


def dataset_inventory(runner, settings, datasets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(inventory: rows and columns per table and dataset, schema differences between datasets)."""
    schemas, rows = {}, []
    for variant, ds in datasets.items():
        refs = settings.refs(ds)
        counts = runner.run("SELECT " + ", ".join(f"(SELECT COUNT(*) FROM {r}) AS {t}" for t, r in refs.items()),
                            max_rows=1, dry_run_first=False)
        schemas[variant] = {t: runner.table_schema(ds, t) for t in refs}
        for t in refs:
            sch = schemas[variant][t]
            rows.append({"variant": variant, "table": t, "row_count": counts.rows[0].get(t) if counts.ok else None,
                         "column_count": len(sch) if sch else 0})
    issues = []
    for table in sorted({t for s in schemas.values() for t in s}):
        present = [v for v in schemas if schemas[v].get(table)]
        for v in schemas:
            if not schemas[v].get(table):
                issues.append({"table": table, "column": "__TABLE__", "variants": v, "issue": "TABLE_MISSING", "details": "table not found"})
        for a, b in combinations(present, 2):
            sa, sb = schemas[a][table], schemas[b][table]
            for col in sorted(set(sa) | set(sb)):
                if col not in sa or col not in sb:
                    issues.append({"table": table, "column": col, "variants": f"{a} vs {b}", "issue": "COLUMN_MISSING",
                                   "details": f"missing in {a if col not in sa else b}"})
                elif sa[col] != sb[col]:
                    issues.append({"table": table, "column": col, "variants": f"{a} vs {b}", "issue": "TYPE_MISMATCH",
                                   "details": f"{a}: {sa[col]} | {b}: {sb[col]}"})
    return pd.DataFrame(rows), pd.DataFrame(issues, columns=["table", "column", "variants", "issue", "details"])


def integrity_checks(runner, settings, dataset: str) -> pd.DataFrame:
    refs = settings.refs(dataset)
    out = []

    def add(table, check, cols, n, details=""):
        out.append({"table": table, "check": check, "columns": ", ".join(cols), "issue_count": int(n), "details": details})

    schemas = {t: runner.table_schema(dataset, t) or {} for t in refs}

    def present(table, cols):
        return all(c in schemas.get(table, {}) for c in cols)

    def blank_cond(table, cols, negate=False):
        parts = []
        for c in cols:
            cond = f"{_q(c)} IS NULL" + (f" OR TRIM({_q(c)}) = ''" if schemas[table].get(c) == "STRING" else "")
            parts.append(f"({cond})")
        expr = " OR ".join(parts)
        return f"NOT ({expr})" if negate else expr

    for table, keys in KEY_COLUMNS.items():
        if not present(table, keys):
            add(table, "KEY_COLUMN_MISSING", keys, 1)
            continue
        add(table, "PRIMARY_KEY_NULL_OR_BLANK", keys, _scalar(runner, f"SELECT COUNTIF({blank_cond(table, keys)}) AS n FROM {refs[table]}"))
        add(table, "PRIMARY_KEY_DUPLICATE", keys, _scalar(runner,
            f"SELECT COALESCE(SUM(n - 1), 0) AS n FROM (SELECT COUNT(*) AS n FROM {refs[table]} "
            f"WHERE {blank_cond(table, keys, negate=True)} GROUP BY {', '.join(map(_q, keys))} HAVING COUNT(*) > 1)"))
    for table, key_sets in UNIQUE_KEYS.items():
        for keys in key_sets:
            if not present(table, keys):
                add(table, "BUSINESS_KEY_COLUMN_MISSING", keys, 1)
                continue
            add(table, "BUSINESS_KEY_DUPLICATE", keys, _scalar(runner,
                f"SELECT COALESCE(SUM(n - 1), 0) AS n FROM (SELECT COUNT(*) AS n FROM {refs[table]} "
                f"WHERE {blank_cond(table, keys, negate=True)} GROUP BY {', '.join(map(_q, keys))} HAVING COUNT(*) > 1)"),
                "uniqueness rule: " + ", ".join(keys))
    for (child, col), (parent, pcol) in FOREIGN_KEYS.items():
        if not (present(child, [col]) and present(parent, [pcol])):
            add(child, "FOREIGN_KEY_COLUMN_MISSING", [col], 1, f"expected {parent}.{pcol}")
            continue
        add(child, "FOREIGN_KEY_ORPHAN", [col], _scalar(runner,
            f"SELECT COUNT(*) AS n FROM {refs[child]} c WHERE c.{_q(col)} IS NOT NULL AND NOT EXISTS "
            f"(SELECT 1 FROM {refs[parent]} p WHERE p.{_q(pcol)} = c.{_q(col)})"), f"references {parent}.{pcol}")
    return pd.DataFrame(out)


def integrity_examples(runner, settings, dataset: str, limit: int = 20) -> pd.DataFrame:
    """Example offending records (duplicate keys and orphans); emails masked."""
    refs = settings.refs(dataset)
    examples = []
    checks = [(t, k, "PRIMARY_KEY_DUPLICATE") for t, k in KEY_COLUMNS.items()]
    checks += [(t, k, "BUSINESS_KEY_DUPLICATE") for t, ks in UNIQUE_KEYS.items() for k in ks]
    for table, keys, check in checks:
        on = " AND ".join(f"t.{_q(k)} = d.{_q(k)}" for k in keys)
        nonnull = " AND ".join(f"{_q(k)} IS NOT NULL" for k in keys)
        res = runner.run(f"WITH d AS (SELECT {', '.join(map(_q, keys))}, COUNT(*) AS duplicate_count FROM {refs[table]} "
                         f"WHERE {nonnull} GROUP BY {', '.join(map(_q, keys))} HAVING COUNT(*) > 1) "
                         f"SELECT t.*, d.duplicate_count FROM {refs[table]} t JOIN d ON {on} "
                         f"ORDER BY d.duplicate_count DESC LIMIT {int(limit)}", max_rows=limit, dry_run_first=False)
        examples += [{"table": table, "check": check, "columns": ", ".join(keys), "record": redact(r)} for r in (res.rows if res.ok else [])]
    for (child, col), (parent, pcol) in FOREIGN_KEYS.items():
        res = runner.run(f"SELECT c.* FROM {refs[child]} c WHERE c.{_q(col)} IS NOT NULL AND NOT EXISTS "
                         f"(SELECT 1 FROM {refs[parent]} p WHERE p.{_q(pcol)} = c.{_q(col)}) LIMIT {int(limit)}",
                         max_rows=limit, dry_run_first=False)
        examples += [{"table": child, "check": f"FOREIGN_KEY_ORPHAN -> {parent}.{pcol}", "columns": col, "record": redact(r)}
                     for r in (res.rows if res.ok else [])]
    return pd.DataFrame(examples, columns=["table", "check", "columns", "record"])
