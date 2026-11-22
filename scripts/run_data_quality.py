"""Run the deterministic data-quality engine, and optionally the reviewer agent, from the terminal.

    python scripts/run_data_quality.py --variant corrupted             # validation only
    python scripts/run_data_quality.py --variant corrupted --review    # + Gemini review (needs GEMINI_API_KEY)
    python scripts/run_data_quality.py --variant corrupted --review --offline   # deterministic reviewer, no API

Writes reports to reports/dq/. Exit code: 0 pass/warning, 1 fail, 2 error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analytics_agent.agents.data_quality_reviewer import DataQualityReviewer, review_to_markdown  # noqa: E402
from analytics_agent.config import load_settings  # noqa: E402
from analytics_agent.data_generation.generator import GenerationConfig  # noqa: E402
from analytics_agent.data_quality import load_tables_from_files, validate_tables  # noqa: E402
from analytics_agent.llm import GeminiStructuredClient, LLMError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "corrupted"], default="corrupted")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--review", action="store_true", help="run the Data Quality Reviewer agent")
    ap.add_argument("--offline", action="store_true", help="use the deterministic reviewer (no Gemini call)")
    args = ap.parse_args()

    settings = load_settings(require_project=False)
    tables = load_tables_from_files(args.variant, args.format, ROOT)
    rep = validate_tables(tables, GenerationConfig().as_of_ts, dataset_label=args.variant)
    j, md = rep.save(ROOT / "reports/dq")
    c = rep.counts()
    print(f"Validation {rep.run_id}: {c['pass']} pass, {c['fail']} fail, {c['warning']} warning, {c['error']} error "
          f"-> {rep.overall_status.upper()}")
    for r in rep.by_status("fail", "warning", "error"):
        print(f"  {r.status:<7} {r.severity:<8} {r.rule_id:<30} observed={r.observed_count}")
    print(f"Reports: {j.relative_to(ROOT)}, {md.relative_to(ROOT)}")

    if args.review:
        llm = None
        if not args.offline:
            try:
                llm = GeminiStructuredClient(settings.gemini.model, timeout_s=settings.gemini.timeout_seconds,
                                             max_retries=settings.gemini.max_retries)
            except LLMError as e:
                print(f"Gemini not available ({e.kind}); using the deterministic reviewer.")
        res = DataQualityReviewer(llm, max_output_tokens=settings.gemini.max_output_tokens).review(rep)
        out = ROOT / "reports/dq" / f"{rep.run_id}_review.md"
        out.write_text(review_to_markdown(res))
        print(f"\nReview: outcome={res.outcome} reviewer={res.reviewer} guardrails="
              f"{'passed' if res.guardrail.passed else 'FAILED'} overall={res.review.overall_assessment} "
              f"human_review={res.requires_human_review}")
        if res.error:
            print("  note:", res.error)
        for f in res.review.findings:
            print(f"  [{f.severity}] {f.title} -> {', '.join(f.rule_ids)}")
        print(f"Review saved: {out.relative_to(ROOT)}")
    return {"pass": 0, "warning": 0, "fail": 1, "error": 2}[rep.overall_status]


if __name__ == "__main__":
    raise SystemExit(main())
