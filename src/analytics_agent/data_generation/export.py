"""Write datasets, manifests, catalogs and BigQuery schema files to disk."""
from __future__ import annotations

import hashlib
import json
import platform
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analytics_agent import schema_registry as sr

# BigQuery NUMERIC is DECIMAL(38, 9); writing that exact Arrow type lets an
# explicit-schema Parquet load map 1:1 without float rounding.
ARROW_TYPES = {
    "STRING": pa.string(), "INT64": pa.int64(), "NUMERIC": pa.decimal128(38, 9),
    "BOOL": pa.bool_(), "DATE": pa.date32(), "TIMESTAMP": pa.timestamp("us", tz="UTC"),
}


def arrow_schema(table_name: str) -> pa.Schema:
    t = sr.get_table(table_name)
    return pa.schema([pa.field(c.name, ARROW_TYPES[c.bq_type], nullable=(c.mode == "NULLABLE"))
                      for c in t.columns])


def to_arrow(table_name: str, df: pd.DataFrame) -> pa.Table:
    t = sr.get_table(table_name)
    arrays = []
    for c in t.columns:
        s = df[c.name]
        if c.bq_type == "NUMERIC":
            vals = [None if pd.isna(v) else Decimal(f"{v:.2f}") for v in s]
            arrays.append(pa.array(vals, type=ARROW_TYPES["NUMERIC"]))
        elif c.bq_type == "TIMESTAMP":
            arrays.append(pa.array(pd.to_datetime(s, utc=True), type=ARROW_TYPES["TIMESTAMP"]))
        else:
            arrays.append(pa.array(s.tolist(), type=ARROW_TYPES[c.bq_type]))
    # Build without nullability constraints: corrupted data intentionally violates REQUIRED.
    return pa.Table.from_arrays(arrays, names=t.column_names)


def to_csv_frame(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Format values the way BigQuery CSV loads expect (explicit schema, no inference)."""
    t = sr.get_table(table_name)
    out = pd.DataFrame(index=df.index)
    for c in t.columns:
        s = df[c.name]
        if c.bq_type == "NUMERIC":
            out[c.name] = [None if pd.isna(v) else f"{v:.2f}" for v in s]
        elif c.bq_type == "TIMESTAMP":
            ts = pd.to_datetime(s, utc=True)
            out[c.name] = [None if pd.isna(v) else v.strftime("%Y-%m-%d %H:%M:%S UTC") for v in ts]
        elif c.bq_type == "DATE":
            out[c.name] = [None if pd.isna(v) else str(v) for v in s]
        elif c.bq_type == "BOOL":
            out[c.name] = np.where(s.astype(bool), "true", "false")
        else:
            out[c.name] = s
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_variant(tables: dict[str, pd.DataFrame], out_dir: Path, variant: str) -> list[dict]:
    files = []
    for fmt in ("csv", "parquet"):
        (out_dir / variant / fmt).mkdir(parents=True, exist_ok=True)
    for name in sr.TABLE_LOAD_ORDER:
        df = tables[name]
        csv_path = out_dir / variant / "csv" / f"{name}.csv"
        to_csv_frame(name, df).to_csv(csv_path, index=False, lineterminator="\n")
        pq_path = out_dir / variant / "parquet" / f"{name}.parquet"
        pq.write_table(to_arrow(name, df), pq_path, compression="snappy")
        for path, fmt in ((csv_path, "csv"), (pq_path, "parquet")):
            files.append({"variant": variant, "table": name, "format": fmt,
                          "path": path.as_posix(), "rows": int(len(df)),
                          "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return files


def write_bigquery_schemas(schema_dir: Path) -> list[str]:
    schema_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in sr.TABLE_LOAD_ORDER:
        p = schema_dir / f"{name}.json"
        p.write_text(json.dumps(sr.get_table(name).to_bigquery_json(), indent=2) + "\n")
        paths.append(p.as_posix())
    return paths


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def build_manifest(cfg, files: list[dict], corruption_seed: int) -> dict:
    import pandas, pyarrow
    return {
        "project": "analytics-agent",
        "dataset_version": "v1",
        "generation_config": cfg.to_dict(),
        "corruption_seed": corruption_seed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pandas.__version__, "pyarrow": pyarrow.__version__},
        "tables": {name: {"columns": [{"name": c.name, "type": c.bq_type, "mode": c.mode}
                                      for c in sr.get_table(name).columns]}
                   for name in sr.TABLE_LOAD_ORDER},
        "files": files,
        "notes": "Row counts for the corrupted variant include injected duplicate/orphan rows. "
                 "Re-running with the same config and package versions reproduces identical CSV files (checked by sha256).",
    }
