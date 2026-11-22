# Analytics Agent — BigQuery + Gemini conversational analytics

A local, reproducible prototype of a conversational analytics agent over BigQuery. This package delivers **Stage 1** (requirements, architecture, data model, metric definitions, evaluation plan) and **Stage 2** (synthetic data generator, controlled corruption, independently computed gold answers, tests and notebooks).

What has and has not been executed: the generator, the 55 unit tests and notebooks 00–02 were run successfully in the build environment (Python 3.12.3, Linux). Nothing has been uploaded to BigQuery and no Gemini or Langfuse call has been made yet; those start in Stages 3, 4 and 7.

## 1. Repository layout

```text
analytics-agent/
├── README.md, pyproject.toml, requirements.txt, .gitignore, .env.example
├── configs/config.example.yaml        # volumes, BigQuery limits, retry bounds (copy to config.yaml)
├── docs/
│   ├── 01_requirements_architecture.md   # requirements, Mermaid architecture, components, security model
│   ├── 02_dataset_specification.md       # ERD, grain, joins/fan-out, metric definitions, reconciliation
│   ├── data_dictionary.md                # generated from the schema registry
│   ├── sample_records.md                 # generated from the data
│   ├── 03_defect_catalog.md              # generated: 15 defects + legitimate edge cases
│   └── 04_evaluation_plan.md             # generated: 20 gold cases, 5 wrong answers, 20 behavioural cases
├── src/analytics_agent/
│   ├── schema_registry.py                # single source of truth for all 7 tables
│   ├── metrics.py                        # canonical metric definitions (ground truth)
│   ├── config.py                         # settings from configs/*.yaml + .env (key never stored)
│   ├── llm.py                            # Gemini structured-output client: error classes, bounded retries
│   ├── data_quality.py                   # deterministic DQ engine: 50 rules, JSON/Markdown reports
│   ├── agents/data_quality_reviewer.py   # Agent A: Gemini reviewer + guardrails + fallback
│   ├── testing.py                        # scripted fake Gemini client (offline tests only)
│   ├── evaluation_cases.py               # behavioural eval cases (B01–B20)
│   └── data_generation/
│       ├── generator.py                  # clean data, fixed seed
│       ├── corruption.py                 # controlled defects + catalog
│       ├── reference_checks.py           # 41 rules: test oracle (Stage 5 reuses the rule IDs in SQL)
│       ├── expected_results.py           # gold answers G01–G20 + known wrong answers W01–W05
│       ├── tiny_dataset.py               # 5-order dataset with hand-computed answers
│       └── export.py                     # CSV / Parquet / manifest / BigQuery schema JSON
├── scripts/
│   ├── generate_data.py                  # one command: generate, check, export
│   ├── build_docs.py                     # re-render generated docs
│   ├── verify_reproducibility.py         # compare your CSVs with the reference hashes
│   ├── check_files.py                    # re-test the exported files on disk
│   └── run_data_quality.py               # run validation (+ optional reviewer) from the terminal
├── notebooks/00_project_overview.ipynb, 01_dataset_design.ipynb, 02_generate_synthetic_data.ipynb,
│            04_data_quality_validation.ipynb, 04b_gemini_live_review.ipynb
├── schemas/bigquery/*.json               # explicit schemas for BigQuery loads (Stage 3)
├── data/manifests/*.json                 # manifest, defect catalog, gold answers (committed)
├── data/generated/                       # CSV + Parquet (git-ignored; regenerate in ~15 s)
├── tests/unit/                           # 55 tests
└── sql/{profiling,validation,examples}/  # filled in Stages 3, 5 and 6
```

Change from the suggested structure: `docs/`, `scripts/` and `schemas/bigquery/` were added so design documents, command-line entry points and load schemas each have one obvious home; `data_generation/` is a sub-package so generation code never mixes with the runtime agent.

## 2. One-time setup (local VS Code)

Requirements: Python **3.11 or 3.12** (tested on 3.12; `google-genai` 2.x needs 3.11+), VS Code with the **Python** and **Jupyter** extensions (Extensions panel → search "Python" and "Jupyter", both by Microsoft).

Open the folder: *File → Open Folder… → `analytics-agent`*. Open a terminal inside VS Code (*Terminal → New Terminal*); every command below runs from the project root.

**macOS / Linux (bash/zsh)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   (then re-run the line above)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `src/` package in editable mode, so `import analytics_agent` works in scripts, tests and notebooks and picks up code edits immediately.

Select the interpreter: `Ctrl+Shift+P` (macOS `Cmd+Shift+P`) → **Python: Select Interpreter** → choose the one inside `.venv`. In a notebook, use **Select Kernel** (top right) → **Python Environments** → `.venv`.

No `.env` is needed for Stages 1–2. When Stage 3 starts: `cp .env.example .env` (PowerShell: `Copy-Item .env.example .env`) and fill in placeholders. `.env` is git-ignored.

## 3. Step-by-step: run each piece of code and test the data

Run these in order the first time. Each step says what to expect; if the output differs, see §5.

### Step 1 — Run the full test suite

```bash
pytest -q
```

Expected: `55 passed` in roughly 30–45 s. The first test triggers one full data generation (~15 s) that the rest reuse.

### Step 2 — Generate both datasets and all artefacts

```bash
python scripts/generate_data.py
```

Expected output (row counts are exact because the seed is fixed):

```text
[1/6] Generating clean data with seed=20260701 ...
      customers      2,000 rows
      products         250 rows
      orders        15,000 rows
      order_items   32,823 rows
      payments      15,804 rows
      shipments     14,291 rows
      refunds        2,047 rows
[2/6] Running reference checks on clean data ...
      all 41 rules pass
[3/6] Injecting controlled defects ...
      15 defects injected; findings match the catalog exactly
[4/6] Computing independent expected results ...
[5/6] Writing CSV + Parquet ...
[6/6] Writing manifests, catalogs and BigQuery schema files ...
```

The script stops with an error (exit code 1) if the clean data violates any rule or the corrupted data's findings differ from the catalog, so a successful run is itself a data test. Quick experiment run: `python scripts/generate_data.py --scale 0.2` (smaller volumes; gold values will differ from the docs).

### Step 3 — Confirm your files are byte-identical to the reference build

```bash
python scripts/verify_reproducibility.py
```

Expected: `14 identical, 0 different, 0 missing`. Run it after Step 2 at default scale.

### Step 3b — Test the files on disk (what you will upload to BigQuery)

```bash
python scripts/check_files.py                 # Parquet
python scripts/check_files.py --format csv    # CSV
```

Reloads all 14 tables from disk, checks columns and row counts against the manifest, runs the 41 rules, and compares corrupted findings with the defect catalog. Expected last line: `RESULT: ALL CHECKS PASSED`.

### Step 4 — Re-render the generated documentation (optional)

```bash
python scripts/build_docs.py
```

Rewrites `docs/data_dictionary.md`, `docs/sample_records.md`, `docs/03_defect_catalog.md`, `docs/04_evaluation_plan.md`.

### Step 5 — Run the notebooks in VS Code

Open each notebook, select the `.venv` kernel, and click **Run All**, in this order:

| Notebook | What it does | Runtime | Key checkpoint |
|---|---|---|---|
| `00_project_overview` | Verifies interpreter, versions and package import | seconds | all versions print `ok` |
| `01_dataset_design` | Walks through tables, keys, FKs, fan-out and the net-sales definition on hand-checkable data | seconds | Jan = 53.00, Feb = 20.00, bridge = 0.00 |
| `02_generate_synthetic_data` | Generates, checks, corrupts, computes gold answers, exports and verifies files, runs pytest | ~1 min | `rules with mismatches: 0`, `Export verified: 28 files`, `55 passed` |

Each notebook starts with a bootstrap cell that finds the project root, so it works regardless of the folder VS Code starts the kernel in. Saved outputs come from the build environment and should match your run exactly.

### Step 6 — Run individual test files to test one aspect at a time

```bash
pytest tests/unit/test_schema_registry.py -v        # data model is internally consistent
pytest tests/unit/test_generator.py -v              # row counts, PK uniqueness, no future timestamps, money identities, lifecycle, determinism
pytest tests/unit/test_corruption_and_checks.py -v  # clean passes all 41 rules; corrupted fails exactly the catalogued ones
pytest tests/unit/test_metrics.py -v                # metric logic vs hand-computed answers; reconciliation bridge = 0
pytest tests/unit/test_export_and_expected.py -v    # Parquet/CSV types and round-trip, hashes, gold-case structure
pytest -k "reconciliation" -v                       # run tests by keyword
```

### Step 7 — Inspect the data yourself

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/generated/clean/parquet/orders.parquet'); print(df.head()); print(df.order_status.value_counts())"
python -c "import pandas as pd; print(pd.read_csv('data/generated/corrupted/csv/payments.csv').provider_transaction_id.duplicated().sum(), 'duplicate transaction ids in corrupted payments')"
python -c "import json; c = json.load(open('data/manifests/expected_results_clean.json'))['cases']; print([(x['test_id'], x['question']) for x in c][:5])"
```

The second command should print `15` (defect D11). On Windows PowerShell the same commands work; if quoting causes trouble, run them inside a notebook cell instead.

## 3b. Data-quality validation and the Data Quality Reviewer agent

The deterministic engine (`data_quality.py`, 50 rules) checks the data and is the source of truth. The reviewer agent (`agents/data_quality_reviewer.py`) asks Gemini to explain the report as structured JSON; deterministic guardrails reject reviews that cite unknown rules, drop failures, downgrade severity, call bad data fit, or lack evidence. Revisions and retries are bounded; failures fall back to a deterministic review and flag human review. Right now the engine runs locally on the generated files; the same rule IDs run as BigQuery SQL in Stage 5.

**Set up Gemini (optional, for live reviews):** create a key in Google AI Studio (Get API key), then `cp .env.example .env` and set `GEMINI_API_KEY=...`. `GEMINI_MODEL` defaults to `gemini-3.8-flash`. Everything below also works without a key (offline mode).

```bash
python scripts/run_data_quality.py --variant clean                         # expect 50 pass -> PASS
python scripts/run_data_quality.py --variant corrupted                     # expect 20 fail -> FAIL
python scripts/run_data_quality.py --variant corrupted --review --offline  # deterministic reviewer, no API call
python scripts/run_data_quality.py --variant corrupted --review            # live Gemini review (needs the key)
python -m pytest -q tests/unit/test_data_quality.py tests/unit/test_dq_reviewer.py   # 36 tests
```

Then open `notebooks/04_data_quality_validation.ipynb` and **Run All**. For the live Gemini review on its own, use `notebooks/04b_gemini_live_review.ipynb` (7 numbered steps; each checks the earlier ones ran). Checkpoints: clean 50/50 pass; corrupted `Defective records found: 154/154 ... false positives: 0`; every guardrail case `rejected = True`; scenario table outcomes accepted / revised / escalated as described; Part 4 runs only when the key is set. Reports are written to `reports/dq/` (git-ignored).

## 3c. BigQuery + conversational agents (notebook 05)

Notebook `05_bigquery_agents_adk_langgraph.ipynb` moves everything to BigQuery and adds the agents:

- **Load and verify.** `bq.load_variant` uploads the clean and corrupted Parquet files into `analytics_agent_dev` and `analytics_agent_corrupted`. It uses `WRITE_EMPTY` and refuses to overwrite existing tables unless you allow it, then checks schemas and row counts.
- **Data-quality rules in BigQuery.** `dq_bigquery.validate_dataset` runs the 50 rules as BigQuery SQL, with record-level parity with the local engine. The Data Quality Reviewer explains the result.
- **Semantic layer.** `semantic_layer.py` holds the metric definitions, join rules and canonical SQL, all checked against the gold answers.
- **Analytics agent (LangGraph).** `agents/analytics_graph.py` runs: plan → SQL → `sql_validator` → SQL & Analytics Reviewer → guarded execution → Answer Verification Reviewer → data-quality caveats.
- **Front door (Google ADK).** `agents/adk_frontend.py` is a multi-turn `LlmAgent` whose tools call the pipeline. Its final text is verified against the tool outputs.
- **Tracing (Langfuse, optional).** `observability.py` provides masked traces.

Setup for live mode, in `.env`:

- `GOOGLE_CLOUD_PROJECT=...` and `BQ_LOCATION=US`
- `GEMINI_API_KEY=...`
- optionally, `ENABLE_LANGFUSE=true` and the Langfuse keys

Then run `gcloud auth application-default login`. In the notebook, set `LOAD_DATA_TO_BIGQUERY = True` once to upload the data.

Without a project the notebook runs in a clearly labelled **offline test mode**: SQL runs on a local DuckDB emulator of BigQuery, and the agents use scripted models.

```bash
python -m pip install -r requirements.txt
python -m pytest -q                      # 144 tests (offline)
python -m pytest -q tests/unit/test_bigquery_stack.py
```

## 3d. Problem-driven notebook (notebook 06)

`notebooks/06_problem_driven_analytics_agents.ipynb` solves 11 business problems, each with the approach and tool that fits it:

1. Trust the warehouse: data-quality checks in BigQuery SQL plus the Data Quality Reviewer.
2. Correct metric definitions: the semantic layer checked against gold answers.
3. Plausible but wrong SQL: the SQL validator plus the SQL & Analytics Reviewer.
4. Plain-English questions: the LangGraph pipeline.
5. Follow-up questions.
6. Invented numbers: answer verification.
7. Bad data: data-quality caveats.
8. Unsafe requests: nothing is executed.
9. A multi-turn assistant: Google ADK.
10. Cost and quota control: dry runs, byte caps and a shared Gemini budget.
11. Accuracy against gold.

A shared `GEMINI_CALL_BUDGET` keeps the notebook within free-tier quotas. Without Gemini, a clearly labelled deterministic stand-in (`notebook_ui.OfflineStandInModel`) lets every problem run.

## 3e. Google ADK, LangGraph and Langfuse in depth (notebook 07)

`notebooks/07_adk_langgraph_langfuse_in_depth.ipynb` gives each framework a problem it is good at.

**LangGraph** (A1, A2):

- the graph diagram, generated from the code;
- step-by-step streaming;
- **human approval before any SQL runs** (`build_approval_graph`, `ask_with_approval`, `approve`, `reject`), using a checkpointer, `interrupt_before=["execute"]` and resume or reject.

**Google ADK** (B1, B2), in `agents/adk_team.py`:

- a **multi-agent team**: a coordinator, plus `sales_analyst`, `data_steward` and `sql_auditor`, which reviews pasted SQL without running it;
- a `before_tool_callback` guardrail;
- session state;
- a `RunConfig(max_llm_calls)` limit on every turn.

**Langfuse** (C1, C2):

- generation spans with the model and token usage;
- **scores** (verified, sql_review_approved, correct_vs_gold, answer_source);
- **sessions**;
- a `LocalTracer` and `TeeTracer` that show the same data without a Langfuse account.

The whole notebook uses about 40 model calls. Use `RUN_PART_A/B/C` or `ADK_GEMINI_MODEL` to stay within free-tier quotas.

## 3f. The complete notebook (notebook 08)

`notebooks/08_conversational_analytics_complete.ipynb` combines notebooks 05 and 06, plus the later features, into one problem-driven notebook. It covers 15 problems in five groups:

- **A. Foundations:** data-quality checks, key and relationship integrity, the descriptive metric report, and the SQL reviewer.
- **B. LangGraph agent:** streaming, follow-ups, answer verification, data-quality caveats, refusals, human approval, and cost control.
- **C. Gemini analysis:** a verified executive summary and a "why" drill-down.
- **D. Google ADK:** the multi-agent team and its guardrails.
- **E. Langfuse:** traces with token usage, scores, sessions, and accuracy against gold.

Use `RUN_GROUP_A` … `RUN_GROUP_E` and `GEMINI_CALL_BUDGET` to fit the free-tier quota. When Gemini runs out, `ModelRouter` continues on the labelled stand-in, and the SDK's hidden retries are switched off, so cells never hang on quota errors.

## 4. What the generated files are

| File | Purpose |
|---|---|
| `data/generated/{clean,corrupted}/parquet/*.parquet` | Recommended for the BigQuery load: exact NUMERIC (DECIMAL 38,9), UTC timestamps, real DATE/BOOL types. |
| `data/generated/{clean,corrupted}/csv/*.csv` | Same data as text: `YYYY-MM-DD HH:MM:SS UTC`, 2-decimal money, empty = NULL. Load only with an explicit schema. |
| `schemas/bigquery/*.json` | Explicit schemas for `bq load` / console uploads. Prevents type inference errors. |
| `data/manifests/manifest.json` | Generation settings, package versions, per-file row counts and SHA-256. |
| `data/manifests/defect_catalog.json` | Each defect: table, columns, severity, rule, affected IDs, expected findings. |
| `data/manifests/expected_results_clean.json` | Gold answers G01–G20 and wrong answers W01–W05. |
| `data/manifests/expected_results_corrupted.json` | Same metrics on corrupted data (shows impact). |
| `data/manifests/legitimate_cases.json` | Unusual-but-correct patterns that must not be flagged. |
| `data/manifests/behavioral_eval_cases.json` | B01–B20: ambiguity, follow-ups, abstention, injection, authorization, failures. |

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: analytics_agent` | Activate `.venv` and run `pip install -e .`; in notebooks, re-run the first cell and check the kernel. |
| `verify_reproducibility.py` reports DIFFERENT | Your numpy/pandas versions differ from `requirements.txt` (random streams and rounding can change). `pip install -r requirements.txt` in a fresh venv. |
| Notebook uses the wrong Python | Kernel picker → `.venv`. Notebook 00 Step 2 tells you if you're outside the venv. |
| `Activate.ps1 cannot be loaded` (Windows) | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. |
| `PermissionError` when writing CSVs | A CSV is open in Excel; close it. |
| pip build errors for numpy/pyarrow on Python 3.13+ | Use Python 3.12 (`py -3.12` / `python3.12`). |

## 6. Cleanup

Delete generated data: remove `data/generated/clean` and `data/generated/corrupted` (they regenerate in ~15 s). Remove the environment: deactivate and delete `.venv`. No cloud resources exist yet.

## 7. Next stage

**Stage 3** — create the BigQuery datasets (`analytics_agent_dev`, `analytics_agent_corrupted`), load the Parquet files with `schemas/bigquery/*.json` (console and `bq`/Python options), configure ADC, set up least-privilege read-only access, and build `03_bigquery_connection.ipynb` with dry-run and byte-cap cost controls.
