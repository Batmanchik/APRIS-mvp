# Cheops AI Architecture Audit (2026-03-21)

## Scope
Audit focused on:
- repository structure hygiene;
- dead/unused code and helper scripts;
- runtime artifact hygiene;
- keeping all quality gates green after cleanup.

## Current Architecture Snapshot
- `src/apris/cheops/*` is the v2 clean core (`domain`, `application`, `infrastructure`, `interfaces`).
- `src/apris/api/main.py` is the API boundary.
- `pages/*` + `app.py` are UI entry and feature pages.
- `scripts/app.ps1` is local process launcher.
- `Dockerfile` + `docker-compose.yml` provide reproducible runtime path.

## Findings
1. Root contained obsolete one-off repair script (`fix_app.py`) not used by runtime/tests.
2. Multiple modules had dead variables/imports (noise and maintenance burden).
3. Runtime training outputs (`mlruns`, `artifacts/cheops_v2_*`) polluted git status after retrain.

## Cleanup Actions Applied
1. Removed obsolete file:
- deleted `fix_app.py`.

2. Removed dead code / unused symbols:
- `src/apris/cheops/infrastructure/ml/tabular_v2.py`:
  - removed unused `y_test_bin`.
- `src/apris/crypto_ponzi/aggregator.py`:
  - removed unused `total_in`;
  - removed no-op pre-loop with `day_i/day_idx`.
- `src/apris/crypto_ponzi/tx_generator.py`:
  - removed unused `exch_in_share`.
- `src/apris/crypto_ponzi/visualizations.py`:
  - removed unused `numpy` import.
- `src/apris/risk_engine.py`:
  - removed unused `FEATURE_COLUMNS` import.
- `src/apris/train_model.py`:
  - removed unused `DRIFT_REPORT_V2_PATH` import.
- `src/apris/cheops/infrastructure/ml/engine_v2.py`:
  - removed unused `_sigmoid`.

3. Repository hygiene:
- `.gitignore` updated with:
  - `mlruns/`
  - `artifacts/cheops_v2_*.joblib`
  - `artifacts/cheops_v2_*.json`

## Verification
- `ruff check .` -> passed
- `mypy src/apris` -> passed
- `pytest -q --maxfail=1` -> passed (`72 passed`)

## Residual Architecture Recommendations (Next)
1. Move all non-source operational docs to `docs/` index with explicit ownership/status tags.
2. Add lightweight module ownership map (`CODEOWNERS` + folder-level ADR pointers).
3. Add CI check preventing accidental commit of large runtime artifacts (size guard).
