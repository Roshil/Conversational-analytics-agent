"""Test the data FILES on disk (what you will upload to BigQuery), not in-memory data.

    python scripts/check_files.py                 # Parquet files (default)
    python scripts/check_files.py --format csv    # CSV files

For each variant it reloads the 7 tables from disk, checks schema and row counts
against the manifest, runs the 41 reference rules, and compares the corrupted
findings with the defect catalog. Exit code 0 = all good.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from analytics_agent import schema_registry as sr  # noqa: E402
from analytics_agent.data_generation.generator import GenerationConfig, normalize_dtypes  # noqa: E402
from analytics_agent.data_generation.reference_checks import run_reference_checks  # noqa: E402


def load(variant: str, fmt: str) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in sr.TABLE_LOAD_ORDER:
        path = ROOT / "data/generated" / variant / fmt / f"{name}.{fmt}"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python scripts/generate_data.py")
        if fmt == "parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, dtype=str, keep_default_na=False).replace("", None)
            for c in sr.get_table(name).columns:
                if c.bq_type == "TIMESTAMP":
                    df[c.name] = pd.to_datetime(df[c.name].str.replace(" UTC", "", regex=False), utc=True)
                elif c.bq_type == "BOOL":
                    df[c.name] = df[c.name].map({"true": True, "false": False})
        tables[name] = normalize_dtypes(name, df)
    return tables


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    fmt = ap.parse_args().format
    manifest = json.loads((ROOT / "data/manifests/manifest.json").read_text())
    rows = {(f["variant"], f["table"]): f["rows"] for f in manifest["files"] if f["format"] == fmt}
    catalog = json.loads((ROOT / "data/manifests/defect_catalog.json").read_text())
    expected = {}
    for d in catalog:
        for rule_id, ids in d["expected_findings"].items():
            expected.setdefault(rule_id, set()).update(map(str, ids))
    as_of = GenerationConfig().as_of_ts
    failures = 0

    for variant in ("clean", "corrupted"):
        print(f"\n=== {variant} ({fmt}) ===")
        t = load(variant, fmt)
        for name, df in t.items():
            ok = list(df.columns) == sr.get_table(name).column_names and len(df) == rows[(variant, name)]
            failures += not ok
            print(f"  {'OK  ' if ok else 'FAIL'} {name:<12} {len(df):>7,} rows, columns match schema")
        found = run_reference_checks(t, as_of)
        if variant == "clean":
            bad = {k: len(v) for k, v in found.items() if v}
            failures += bool(bad)
            print(f"  {'OK  ' if not bad else 'FAIL'} reference rules: {len(found) - len(bad)}/{len(found)} pass"
                  + (f"  failing: {bad}" if bad else ""))
        else:
            print(f"  {'rule':<30} {'expected':>8} {'found':>6} {'missed':>6} {'extra':>6}")
            for rule_id in sorted(found):
                e, g = expected.get(rule_id, set()), found[rule_id]
                if e or g:
                    miss, extra = len(e - g), len(g - e)
                    failures += bool(miss or extra)
                    flag = "OK  " if not (miss or extra) else "FAIL"
                    print(f"  {flag}{rule_id:<26} {len(e):>8} {len(g):>6} {miss:>6} {extra:>6}")
            print(f"  {len(catalog)} catalogued defects; rules silent where no defect was injected: "
                  f"{sum(1 for r in found if not found[r] and r not in expected)}")
    print("\nRESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
