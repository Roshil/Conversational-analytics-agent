"""Check that your locally generated CSV files are byte-identical to the reference build.

    python scripts/generate_data.py
    python scripts/verify_reproducibility.py

CSV is compared (Parquet embeds writer metadata that can vary by library build).
A mismatch usually means different numpy/pandas versions; see README troubleshooting.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ref = json.loads((ROOT / "data/manifests/reference_hashes.json").read_text())
bad = missing = 0
for rel, want in sorted(ref["csv_sha256"].items()):
    p = ROOT / rel
    if not p.exists():
        print(f"MISSING   {rel}")
        missing += 1
        continue
    have = hashlib.sha256(p.read_bytes()).hexdigest()
    ok = have == want
    bad += not ok
    print(f"{'OK      ' if ok else 'DIFFERENT'} {rel}")
print(f"\n{len(ref['csv_sha256']) - bad - missing} identical, {bad} different, {missing} missing")
print("Reference environment:", ref["environment"])
sys.exit(1 if bad or missing else 0)
