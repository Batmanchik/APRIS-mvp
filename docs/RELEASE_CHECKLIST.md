# Cheops AI Release Checklist (Baseline v1-compatible)

## Pre-release Quality Gates
- `python -m ruff check src tests pages app.py`
- `python -m mypy`
- `python -m pytest --cov=src/apris`
- `python -m bandit -q -r src/apris -x src/apris/crypto_ponzi -s B101`
- `python -m radon cc src/apris/cheops -s -n B`
- `lint-imports`

## Runtime Smoke
- `scripts/app.ps1 start`
- `scripts/app.ps1 status`
- `scripts/app.ps1 open`
- `scripts/app.ps1 stop`
- `scripts/app.ps1 status`

## User Scenario Smoke
- Scanner batch run (synthetic mode)
- Dashboard opens and displays risk rows
- Manual check page scores a case and renders explanation
- API endpoints healthy: `/api/v1/health`, `/api/v2/health/model`

## Regression Notes
- v1 endpoints must remain available without payload/response breaking changes.
- UI scanner must use API batch path (`/api/v2/score/batch`) and must not call model inference directly.
- `CHEOPS_API_BASE_URL` and `CHEOPS_API_TIMEOUT` must be honored by frontend API client.
