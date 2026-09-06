# Cheops AI (Multi-Channel Fraud Intelligence System)

> **Lost in the documents?** [DOCS.md](DOCS.md) is the map: what each file is
> for, who keeps it, and where to write a new thing. The three that matter:
> this README (how to run it), [PLAN.md](PLAN.md) (what is being built and in
> what order), [docs/RESULTS.md](docs/RESULTS.md) (every measurement, with what
> it does not prove).

Cheops AI is a local MVP for detection of multi-channel financial fraud patterns (legal + crypto).
It combines ML risk scoring, ETL for transaction logs, a FastAPI backend, and a Streamlit multipage frontend.

## Current Architecture
- `src/apris/` - core backend and ML modules.
- `src/apris/api/main.py` - FastAPI REST API (`/api/v1/*`, `/api/v2/*`).
- `src/apris/cheops/` - v2 clean architecture layers (`domain`, `application`, `infrastructure`, `interfaces`).
- `src/apris/risk_engine.py` - model inference, feature validation, explainability.
- `src/apris/etl.py` - CSV/JSON ingestion and operational-to-feature transformation.
- `src/apris/train_model.py` - model training, metrics, artifact export, MLflow logging.
- `src/apris/cheops/infrastructure/ml/tabular_v2.py` - tabular v2 bundle training (global + typology + isotonic calibration).
- `src/apris/frontend/api_client.py` - HTTP client used by Streamlit pages.
- `src/apris/frontend/scanner_pipeline.py` - optimized case builder for scanner page (feature/transaction modes).
- `pages/` - Streamlit multipage UI (network discovery, candidate dossier,
  validation, manual check). No page loads a model: scoring goes through the API.
- `tests/` - pytest-based test suite (`unit`, `api`, `smoke`).

## Runtime vs Source Directories
- Source code: `src/`, `pages/`, `tests/`, `scripts/`.
- Runtime/generated data: `artifacts/`, `mlruns/`, `.run/`.
- Virtual environments/backups: `.venv/`, `.venv_*`.

This repository keeps runtime directories for local experimentation. They are not required for code review and can be regenerated.

## Dependency Source of Truth
- Canonical dependency spec: `pyproject.toml` (`[project.dependencies]` and `[project.optional-dependencies].dev`).
- `requirements.txt` is kept in sync for convenience and mirrors runtime dependencies from `pyproject.toml`.

## Run (PowerShell-only)
Use the unified launcher:

```powershell
.\scripts\app.ps1 start
.\scripts\app.ps1 status
.\scripts\app.ps1 open
.\scripts\app.ps1 stop
```

What `start` does:
- bootstraps `.venv` if missing,
- installs project and dev dependencies from `pyproject.toml` via `pip install -e ".[dev]"`,
- starts FastAPI on `127.0.0.1:8000`,
- starts Streamlit on `127.0.0.1:8501`,
- writes PID files to `.run/`,
- writes logs to `.run/api.out.log`, `.run/api.err.log`, `.run/streamlit.out.log`, `.run/streamlit.err.log`.

Frontend API client environment:
- `CHEOPS_API_BASE_URL` (default: `http://127.0.0.1:8000`)
- `CHEOPS_API_TIMEOUT` in seconds (optional)
- `CHEOPS_API_RETRIES` for retryable GET calls (default: `1`)
- `CHEOPS_API_RETRY_BACKOFF` in seconds between retries (default: `0.2`)

## Run (Docker Compose)
Containerized run for API + UI:

Prerequisite:
- Docker Desktop / Docker Engine with Compose plugin.

```bash
docker compose up -d --build
docker compose ps
```

Open:
- UI: `http://127.0.0.1:8501`
- API: `http://127.0.0.1:8000`

Stop:

```bash
docker compose down -v
```

Notes:
- `ui` service uses `CHEOPS_API_BASE_URL=http://api:8000` inside Compose network.
- Runtime folders are mounted from host: `./artifacts`, `./mlruns`, `./.run`.

## Train Model
Train on synthetic data:

```powershell
.\.venv\Scripts\python.exe -m apris.train_model
```

Train on external data via ETL (`csv` or `json`):

```powershell
.\.venv\Scripts\python.exe -m apris.train_model --data your_real_data.csv
```

External datasets are validated for required training fields (`FEATURE_COLUMNS + label`) and do not require synthetic-only column `is_borderline`.

Training exports both legacy and v2 tabular artifacts:
- `artifacts/model.joblib` + `artifacts/feature_names.json` (legacy v1 path).
- `artifacts/cheops_v2_tabular.joblib` + `artifacts/cheops_v2_metrics.json` (Cheops v2 tabular branch).
- `artifacts/cheops_v2_sequence.joblib` + `artifacts/cheops_v2_sequence_metrics.json` (Cheops v2 trainable sequence branch).
- `artifacts/cheops_v2_graph.joblib` + `artifacts/cheops_v2_graph_metrics.json` (Cheops v2 trainable graph branch).
- `artifacts/cheops_v2_fusion_meta.joblib` + `artifacts/cheops_v2_fusion_metrics.json` (Cheops v2 logistic fusion meta-head).
- `artifacts/cheops_v2_feature_profile.json` (baseline feature profile for drift monitoring).
- `artifacts/cheops_v2_model_registry.json` (model governance registry with selected candidate and branch metrics).
- `cheops_v2_metrics.json` contains calibration-aware metrics (`roc_auc`, `brier`, `ece`) for global and typology heads.
- `cheops_v2_sequence_metrics.json` contains sequence branch metrics and heuristic fallback comparison.
- `cheops_v2_graph_metrics.json` contains graph branch metrics and heuristic fallback comparison.
- `cheops_v2_fusion_metrics.json` contains calibration-aware metrics for the fusion layer and weighted-fallback comparison.

Optional benchmark run:

```powershell
.\.venv\Scripts\python.exe -m apris.train_model --benchmark
.\.venv\Scripts\python.exe -m apris.train_model --benchmark --benchmark-lightgbm-only
```

Benchmark report artifact:
- `artifacts/cheops_v2_benchmark.json`.
- Each candidate now records calibration metric `ece` in addition to `roc_auc`, `accuracy`, and `brier`.
- Benchmark now stores explicit `selection_policy`, candidate `selection_score`, ranking, and `winner_reason`.

Optional drift check against another feature dataset:

```powershell
.\.venv\Scripts\python.exe -m apris.train_model --drift-data your_features_snapshot.csv
```

Drift artifact:
- `artifacts/cheops_v2_drift_report.json` with `overall_psi`, per-feature `psi`, and drift level (`stable|moderate|high`).

Runtime inference behavior for v2:
- If `cheops_v2_sequence.joblib` exists, sequence score uses calibrated trained surrogate branch.
- If sequence artifact is absent, sequence score falls back to deterministic heuristic.
- If `cheops_v2_graph.joblib` exists, graph score uses calibrated trained surrogate branch.
- If graph artifact is absent, graph score falls back to deterministic heuristic.
- If `cheops_v2_fusion_meta.joblib` exists, engine uses calibrated logistic fusion head for `global_risk`.
- If fusion artifact is absent, engine falls back to deterministic weighted fusion (v1-compatible behavior).

## Run the whole thing (one command)

```
python scripts/run_demo.py                  # quick world
python scripts/run_demo.py --preset full    # the world the audit was measured on
```

Then open **http://127.0.0.1:8501** and go to **Витрина Vertex** — the showcase
page. Seven tabs: the five topologies as graphs drawn from events, the crypto
channel, the five worlds, rarity with a slider that re-prices the measured ROC
curve, the evasion curve, the analyst queue, and buttons that launch the real
runs. That page is the demo; everything else in the interface is the machinery
behind it.

It builds the analyst queue, starts the API, waits for its health check
rather than sleeping, starts Streamlit, waits for that, and prints the URL.
Ctrl+C stops both. The health checks are the same ones `docker-compose.yml`
uses, so "it did not come up" is diagnosed the same way with or without
Docker.

The queue on its own, without the interface:

```
python scripts/run_pipeline.py --preset full --target-recall 0.8
```

World, discovery, features, detector, threshold — one call, writing
`artifacts/analyst_queue.json`. The threshold is read off the earlier
walk-forward folds and the queue is cut on the last one, which the model
never trained on, so the precision it reports is the precision it would have
on Monday. The queue's LENGTH is an output, not a setting: that is the
prevalence result (R6) made operational, since a fixed review budget is the
wrong policy once fraud is rare.

## Interface

Five pages, and the order is the workflow:

1. **Поиск сетей** — `discover_candidates` proposes clusters from the event
   stream alone. Labels are attached afterwards, so the coverage shown is a
   real ceiling on recall rather than 1.0 by construction.
2. **Досье кандидата** — the candidate's own transfer graph, its ten
   event-derived features, and the score returned by `/api/v2/score`. Branch
   modes are shown as they are: a branch running a heuristic says so.
3. **Валидация** — purged walk-forward, the quintile ladder, coverage and the
   naive-rule acceptance check, all computed when the page opens.
4. **Ручная проверка** — the original nine features for manual entry, with
   the uncalibrated thresholds labelled as such, and a button that computes
   the same nine from a pyramid organiser's real cash flow.
5. **Очередь аналитика** — the end product rather than a part of it: the
   cases the pipeline put on somebody's desk, the threshold they were cut
   by, and the price of that threshold in reviews per catch. Reads
   `artifacts/analyst_queue.json`; if no run has happened yet it says so
   instead of failing.

Three things the interface deliberately does not do, each a defect that was
removed rather than a precaution: draw a graph built from the features it
claims to support, draw a structure derived from the verdict, or report a
metric computed on a grouping the detector was handed in advance.
`tests/unit/test_scanner_architecture.py` parses every page and fails if any
of them comes back.

The dossier and manual pages need the API running. `scripts/run_demo.py`
does this for you; by hand it is:

```powershell
python -m uvicorn apris.api.main:app --port 8000
streamlit run app.py
```

Every page is rendered end to end by `tests/smoke/test_pages_render.py`, and
that test fails if a page file exists without a line in its list — a screen
nobody opens is how an interface rots behind a moving core.

## Case-Level Baseline

```powershell
python scripts/case_baseline.py --seed 42
```

Rebuilds the candidate-classification numbers from scratch — discovery,
features from events, purged walk-forward — and prints everything the
write-up is allowed to quote about case-level detection. Written because the
audit carried a baseline that no committed code reproduced.

## Simulation Layer and Event Features

`src/apris/cheops/infrastructure/simulation/` generates a synthetic world of
accounts, ATMs and transactions with known ground truth. It writes **events
only** — who paid whom, how much, when. Every feature is derived by the
detection layers from those events; the generator never writes a metric.

`src/apris/cheops/infrastructure/ml/event_features_v2.py` computes graph and
sequence features from that stream. It replaces the `*_from_tabular` builders,
which produced structural and temporal matrices as hand-written linear
combinations of nine period aggregates — so the graph branch never read a
graph and `burst_ratio_90s` was derived from a quantity measured in days.

```python
from apris.cheops.infrastructure.simulation import generate_world, SimulationConfig
from apris.cheops.infrastructure.simulation.acceptance import evaluate
from apris.cheops.infrastructure.simulation.cases import build_cases

world = generate_world(SimulationConfig())   # ~2 min, ~320k events
report = evaluate(world)                      # layer-0 acceptance criterion
cases = build_cases(world)                    # labelled cases for detectors
```

Details: [docs/SIMULATION_LAYER.md](docs/SIMULATION_LAYER.md).
Measured findings: [docs/reviews/AUDIT_FINDINGS_2026-09-04.md](docs/reviews/AUDIT_FINDINGS_2026-09-04.md).
Plan: [docs/RESEARCH_PLAN_RKNP_2026.md](docs/RESEARCH_PLAN_RKNP_2026.md).

## Test and Quality Workflow

Run every gate CI runs, in the same order:

```powershell
python scripts/check_all.py           # everything
python scripts/check_all.py --fast    # skip the slow test suite
```

Running a subset and reading it as "the gates pass" has now cost three red CI
runs in a day: ruff, mypy and pytest passing says nothing about bandit, and
bandit is what failed. The script also distinguishes FAIL from SUSPECT — a
gate that fails while printing nothing is almost always a broken console
script rather than broken code, which is exactly what `lint-imports.exe` did
after the project directory was renamed.

Install dev tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest -m smoke
```

Run quality checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests pages
.\.venv\Scripts\python.exe -m ruff format --check src tests pages
.\.venv\Scripts\python.exe -m mypy
```

Enable pre-commit hooks:

```powershell
.\.venv\Scripts\python.exe -m pre_commit install
```

## API Surface
Versioned API contract:
- `GET /api/v1/health`
- `POST /api/v1/predict`
- `POST /api/v1/predict/ops`
- `POST /api/v1/explain`
- `GET /api/v1/meta/features`
- `GET /api/v2/meta/typologies`
- `GET /api/v2/health/model`
- `GET /api/v2/health/model/details`
- `GET /api/v2/health/runtime`
- `POST /api/v2/score`
- `POST /api/v2/score/batch`
- `POST /api/v2/explain`

`/api/v2/explain` now includes branch-level outputs:
- `branch_scores` (`tabular`, `sequence`, `graph`, `fusion`)
- `branch_modes` (whether each branch is trained or fallback mode)

Operational observability:
- API responses include `X-Request-Id` header for request tracing.
- `/api/v2/health/runtime` returns aggregated runtime counters and per-endpoint latency/error snapshots.

## Release Readiness
- Regression and operational release checklist: `docs/RELEASE_CHECKLIST.md`.
- Scientific methodology and formulas: `docs/CHEOPS_AI_SCIENTIFIC_FOUNDATION.md`.
