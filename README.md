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
- `src/apris/frontend/api_client.py` - HTTP client used by Streamlit pages.
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

## Train Model
Train on synthetic data:

```powershell
.\.venv\Scripts\python.exe -m apris.train_model
```

Train on external data via ETL (`csv` or `json`):

```powershell
.\.venv\Scripts\python.exe -m apris.train_model --data your_real_data.csv
```

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
- `POST /api/v2/score`
- `POST /api/v2/score/batch`
- `POST /api/v2/explain`

## Release Readiness
- Regression and operational release checklist: `docs/RELEASE_CHECKLIST.md`.
