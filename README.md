# Cheops AI (Multi-Channel Fraud Intelligence System)

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
- `pages/` - Streamlit multipage UI (dashboard, scanner, manual check).
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

## Test and Quality Workflow
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
