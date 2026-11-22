"""Generate clean + corrupted datasets, manifests, catalogs and expected results.

Usage (from the project root, virtual environment active):
    python scripts/generate_data.py                    # default volumes
    python scripts/generate_data.py --scale 0.2        # quick small run
    python scripts/generate_data.py --out data/generated --manifests data/manifests
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

from analytics_agent.data_generation.corruption import corrupt, expected_findings, legitimate_cases  # noqa: E402
from analytics_agent.data_generation.expected_results import build_expected_results  # noqa: E402
from analytics_agent.data_generation.export import (  # noqa: E402
    build_manifest, write_bigquery_schemas, write_json, write_variant)
from analytics_agent.data_generation.generator import GenerationConfig, generate_clean  # noqa: E402
from analytics_agent.data_generation.reference_checks import run_reference_checks, summarize  # noqa: E402

CORRUPTION_SEED = 777


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--manifests", default="data/manifests")
    ap.add_argument("--schemas", default="schemas/bigquery")
    ap.add_argument("--seed", type=int, default=GenerationConfig.seed)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies customer/product/order volumes")
    args = ap.parse_args()

    base = GenerationConfig()
    cfg = replace(base, seed=args.seed,
                  n_customers=max(200, int(base.n_customers * args.scale)),
                  n_products=max(60, int(base.n_products * args.scale)),
                  n_orders=max(1500, int(base.n_orders * args.scale)))
    out, man = ROOT / args.out, ROOT / args.manifests
    t0 = time.time()

    print(f"[1/6] Generating clean data with seed={cfg.seed} ...")
    clean = generate_clean(cfg)
    for k, v in clean.items():
        print(f"      {k:<12} {len(v):>7,} rows")

    print("[2/6] Running reference checks on clean data ...")
    clean_results = run_reference_checks(clean, cfg.as_of_ts)
    failing = {k: len(v) for k, v in clean_results.items() if v}
    if failing:
        print(f"      ERROR: clean data violates rules: {failing}")
        return 1
    print(f"      all {len(clean_results)} rules pass")

    print("[3/6] Injecting controlled defects ...")
    corrupted, defects = corrupt(clean, seed=CORRUPTION_SEED)
    got = run_reference_checks(corrupted, cfg.as_of_ts)
    exp = expected_findings(defects)
    mismatches = {r for r in set(got) | set(exp) if got.get(r, set()) != exp.get(r, set())}
    if mismatches:
        print(f"      ERROR: findings differ from catalog for rules {sorted(mismatches)}")
        return 1
    print(f"      {len(defects)} defects injected; findings match the catalog exactly")

    print("[4/6] Computing independent expected results ...")
    exp_clean = build_expected_results(clean)
    exp_corrupt = build_expected_results(corrupted)

    print("[5/6] Writing CSV + Parquet ...")
    files = write_variant(clean, out, "clean") + write_variant(corrupted, out, "corrupted")
    for f in files:
        f["path"] = Path(f["path"]).relative_to(ROOT).as_posix()

    print("[6/6] Writing manifests, catalogs and BigQuery schema files ...")
    write_bigquery_schemas(ROOT / args.schemas)
    write_json(build_manifest(cfg, files, CORRUPTION_SEED), man / "manifest.json")
    write_json([d.to_dict() for d in defects], man / "defect_catalog.json")
    write_json(legitimate_cases(clean), man / "legitimate_cases.json")
    write_json(exp_clean, man / "expected_results_clean.json")
    write_json(exp_corrupt, man / "expected_results_corrupted.json")
    write_json({"clean": summarize(clean_results).to_dict(orient="records"),
                "corrupted": summarize(got).to_dict(orient="records")}, man / "reference_check_results.json")
    print(f"Done in {time.time() - t0:.1f}s. Outputs: {out.relative_to(ROOT)} and {man.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
