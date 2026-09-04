# Cheops AI - Factual Change Report (2026-03-21)

## 1) Iteration objective
Continue product hardening after multi-branch ML rollout:
- improve explainability with explicit branch-level signals;
- add operational health details for trained artifacts and branch metrics;
- keep all quality gates green.

## 2) Explainability upgrade (branch-aware)

### Updated modules
- `src/apris/cheops/application/dto.py`
- `src/apris/cheops/interfaces/schemas_v2.py`
- `src/apris/cheops/infrastructure/ml/engine_v2.py`
- `src/apris/api/main.py`

### Implemented changes
- `ExplainOutput`/`ExplainV2Response` now include:
  - `branch_scores` (`tabular`, `sequence`, `graph`, `fusion`)
  - `branch_modes` (trained/fallback mode per branch)
- `engine_v2.explain_case(...)` now reports:
  - sequence factors from runtime sequence feature extractor,
  - graph factors from runtime graph feature extractor,
  - consistent branch score decomposition used in final fusion.

## 3) Operational observability upgrade

### New/updated API
- Added endpoint:
  - `GET /api/v2/health/model/details`

### Implemented behavior
- Returns:
  - current health snapshot,
  - artifact existence map,
  - key branch metrics from saved JSON artifacts (`roc_auc`, `ece`),
  - drift summary (`overall_level`, `overall_psi`, counts).
- Source implementation:
  - `src/apris/cheops/infrastructure/ml/engine_v2.py` (`health_details`)
  - `src/apris/api/main.py` (new route wiring)

## 4) Tests updated

### Updated tests
- `tests/api/test_api_v2_endpoints.py`
  - validates `/api/v2/health/model/details`
  - validates branch fields in `/api/v2/explain`
- Existing engine and ML tests remained green.

### Current result
```text
pytest -q --cov=src/apris --cov-report=term-missing
Result: 63 passed
Coverage: 81.08% (threshold 75% passed)
```

## 5) Static quality checks
```text
ruff check . -> All checks passed
mypy src/apris -> Success: no issues found in 29 source files
```

## 6) Runtime verification (fact)

### Direct API checks
- `/api/v2/health/model/details` returned `200` with populated branch metrics:
  - `tabular_global_roc_auc`
  - `sequence_roc_auc`
  - `graph_roc_auc`
  - `fusion_roc_auc`
- `/api/v2/explain` returned `200` with:
  - `branch_scores`
  - `branch_modes`

## 7) Documentation update
- `README.md` updated:
  - new endpoint in API surface,
  - explain response now documents `branch_scores` and `branch_modes`.

## 8) Practical outcome
- Explainability is now directly aligned with deployed branch architecture (`tabular + sequence + graph + fusion`).
- Operational monitoring can confirm artifact readiness and per-branch model quality without manual file inspection.

---

## 9) Iteration objective (ergonomics + optimization)
Continue Cheops AI hardening with focus on analyst UX and batch-scan performance:
- reduce scanner pipeline complexity on raw transaction mode;
- remove fragile dossier selection logic in dashboard;
- keep API-first workflow and preserve v1 compatibility.

## 10) Scanner optimization and reliability

### New module
- `src/apris/frontend/scanner_pipeline.py`

### Implemented changes
- Extracted scanner data-prep logic from `pages/2_Сканнер_транзакций.py` into reusable/testable pipeline.
- Replaced repeated full-table entity filtering with prebuilt vectorized entity-to-transaction index map.
- Added deterministic case id generator (`deterministic_case_id`) in shared scanner pipeline.
- Added normalization guard for low-volume entities before model feature conversion:
  - operational values are clipped to `OPERATIONAL_INPUT_BOUNDS`,
  - invalid relations are corrected (`referred_clients_current <= new_clients_current`, `top1_wallet_share <= top10_wallet_share`).
- Scanner page now records and displays batch duration and exposes optional v2 model health details in UI.

### Practical effect
- Better scalability on transaction uploads with many entities.
- Scanner no longer fails on small entities with very low `tx_count_total`.

## 11) Dashboard ergonomics upgrade

### Updated module
- `pages/1_Дашборд_аномалий.py`

### Implemented changes
- Rebuilt alert selection flow:
  - added risk filter (`All / Medium+ / High+ / Critical`),
  - added `Top N` limiter and entity search,
  - removed fragile parsing of selected label (`split(...)`), now selection is id-based and deterministic.
- Added explicit inbox table preview before dossier.
- Added `@st.cache_data` for synthetic crypto dossier generation to reduce repeated render cost.
- Integrated v2 explain path in dossier (`/api/v2/explain`) with fallback to v1 explain when needed.
- Updated risk labels/messages to be aligned with Cheops AI multi-channel fraud positioning.

### Practical effect
- Faster analyst navigation in large alert lists.
- Lower chance of dossier mismatch caused by string parsing.
- Better explainability context directly in dossier.

## 12) API client update
- `src/apris/frontend/api_client.py`
- Added:
  - `health_check_v2_model_details()` -> `GET /api/v2/health/model/details`

## 13) Tests added/updated for this iteration
- Added:
  - `tests/unit/test_scanner_pipeline.py`
- Updated:
  - `tests/unit/test_frontend_api_client.py` (new health-details client path check)

### New test coverage points
- deterministic case-id stability;
- feature-row to v2-case conversion;
- transaction-log to per-entity case conversion;
- low-volume entity handling without crash;
- API client routing for model health details.

## 14) Quality gate results (post-change)
```text
ruff check .                                -> All checks passed
mypy src/apris                              -> Success: no issues found in 30 source files
pytest -q --cov=src/apris --cov-report=... -> 69 passed, coverage 81.21%
```

## 15) Files changed in this iteration
- `src/apris/frontend/scanner_pipeline.py` (new)
- `pages/2_Сканнер_транзакций.py`
- `pages/1_Дашборд_аномалий.py`
- `src/apris/frontend/api_client.py`
- `tests/unit/test_scanner_pipeline.py` (new)
- `tests/unit/test_frontend_api_client.py`

---

## 16) Scientific section added

To support judge review with explicit methodology, a dedicated scientific document was added:
- `docs/CHEOPS_AI_SCIENTIFIC_FOUNDATION.md`

This section contains:
- notation and formal problem statement,
- explicit formulas for operational-to-feature transformation,
- deterministic typology rules and label construction logic,
- branch formulas for tabular/sequence/graph/fusion scoring,
- calibration (ECE) and drift (PSI) definitions,
- interpretation constraints (risk score is not a legal verdict).

---

## 17) ML governance step completed (roadmap continuation)

### Objective
Add explicit and auditable model-selection governance on top of existing v2 training pipeline.

### Implemented changes
- Added model registry module:
  - `src/apris/cheops/infrastructure/ml/model_registry_v2.py`
- Added governance artifact:
  - `artifacts/cheops_v2_model_registry.json`
- Upgraded benchmark report format (`benchmark_v2`):
  - `selection_policy`,
  - per-candidate `selection_score`,
  - ordered `ranking`,
  - `winner_reason`.
- Updated training flow (`src/apris/train_model.py`):
  - registry is saved on every train run,
  - registry is updated after benchmark run with selected winner and reason.
- Extended model-health details:
  - `/api/v2/health/model/details` now includes `registry` summary.

### Factual benchmark run executed
Command executed:
```text
python -m apris.train_model --benchmark --benchmark-lightgbm-only
```

Observed result:
- `benchmark_winner`: `lightgbm`
- `selection_score`: `0.9852068805848386`
- Policy used:
  - `0.65*roc_auc + 0.20*accuracy + 0.10*(1-brier) + 0.05*(1-ece)`

### Quality verification
- `ruff check` -> passed
- `mypy src/apris` -> passed
- `pytest -q --cov=src/apris --cov-report=term-missing` -> passed (`72` tests)

---

## 18) Platform hardening step completed (containerization + CI smoke)

### Objective
Move to reproducible runtime and add container-level operational smoke checks.

### Implemented changes
- Added `Dockerfile` for unified runtime image.
- Added `docker-compose.yml` with 2 services:
  - `api` (`uvicorn`, port `8000`, healthcheck `/api/v1/health`)
  - `ui` (`streamlit`, port `8501`, healthcheck `/_stcore/health`)
- Added `.dockerignore` for clean build context and faster container builds.
- Added CI job `docker-smoke` (`.github/workflows/ci.yml`):
  - build/start compose stack,
  - wait for API/UI health,
  - smoke-call `/api/v2/health/model/details`,
  - teardown with `docker compose down -v`.
- Updated docs:
  - `README.md` (`Run (Docker Compose)` section),
  - `docs/RELEASE_CHECKLIST.md` (`Container Smoke` section).

### Practical outcome
- Runtime now has cross-platform reproducible startup path beyond local PowerShell.
- CI validates not only Python quality gates but also real container boot/health behavior.

---

## 19) Architecture hygiene pass completed

### Objective
Before continuing feature development, reduce maintenance noise and remove technical clutter.

### Implemented cleanup
- Removed obsolete one-off script:
  - `fix_app.py`
- Removed dead variables/imports in multiple modules:
  - `tabular_v2.py`, `engine_v2.py`, `risk_engine.py`,
  - `train_model.py`, `crypto_ponzi/aggregator.py`,
  - `crypto_ponzi/tx_generator.py`, `crypto_ponzi/visualizations.py`.
- Updated `.gitignore` for runtime-generated outputs:
  - `mlruns/`,
  - `artifacts/cheops_v2_*.joblib`,
  - `artifacts/cheops_v2_*.json`.
- Added architecture audit document:
  - `docs/ARCHITECTURE_AUDIT_2026-03-21.md`.

### Quality verification
- `ruff check .` -> passed
- `mypy src/apris` -> passed
- `pytest -q --maxfail=1` -> passed (`72` tests)

---

## 20) Runtime observability surfaced in UI + icon rendering issue fixed

### Objective
Improve operational ergonomics and remove first-screen rendering artifacts without changing scoring contracts.

### Implemented changes
- Scanner page (`pages/2_Сканнер_транзакций.py`):
  - added runtime health rendering (`_render_runtime_health_snapshot`),
  - now calls frontend client `health_check_v2_runtime()` before batch run,
  - displays total request/error counters and endpoint-level p95 latency snapshot.
- Main app style bootstrap (`app.py`):
  - added font links for `Material Symbols Rounded` and `Material Icons`.
  - fixes ligature text artifacts where icon names were rendered as plain text.

### API/contract impact
- No breaking changes to v1/v2 scoring endpoints.
- Existing v2 observability endpoint is now actively used in UI.

### Quality verification
- `ruff check .` -> passed
- `mypy src/apris` -> passed
- `pytest -q --maxfail=1` -> passed (`74` tests)

---

## 21) Reliability hardening pass: retry policy and degraded UX

### Objective
Reduce operational fragility when API is temporarily unavailable or unstable.

### Changes
- Frontend API client now supports configurable retries for read-only endpoints:
  - `CHEOPS_API_RETRIES` (default `1`),
  - `CHEOPS_API_RETRY_BACKOFF` (default `0.2`).
- Added normalized exception type `ApiClientError` to keep UI messaging consistent.
- Scanner page now has explicit degraded paths:
  - actionable API unavailable state with retry button,
  - runtime degradation warning from `/api/v2/health/runtime` error rate,
  - batch degradation warning on high partial-failure ratio.

### Test evidence
- New unit tests in `tests/unit/test_frontend_api_client.py` validate:
  - retry on connection errors,
  - retry on HTTP 503,
  - no retries for POST batch scoring,
  - invalid JSON response handling.

### Quality result
- `ruff check .` -> passed
- `mypy src/apris` -> passed
- `pytest -q --maxfail=1` -> passed (`78` tests)

---

## 22) Step A factual delivery: entity contracts + deterministic entity resolution

### Scope
Start of the large-scale automation track (IP/TOO) with a stable identity layer.

### Code changes
- New module: `src/apris/cheops/domain/entity_resolution.py`
  - introduces `LegalEntityRecord` contract and schema validation,
  - introduces deterministic `resolve_entity_key` with anchor-priority strategy,
  - adds grouping helper for dedup/merge scenarios.
- Scanner pipeline integration: `src/apris/frontend/scanner_pipeline.py`
  - normalized sender/receiver IDs before case assembly,
  - tx-case fingerprint now derived from resolved entity identity.

### Test evidence
- Added tests: `tests/unit/test_cheops_entity_resolution.py`
- Updated tests: `tests/unit/test_scanner_pipeline.py`
- Full suite status:
  - `ruff` passed,
  - `mypy` passed,
  - `pytest` passed (`85` tests).

### Practical impact
- Better stability of entity identity across noisy formatting variants.
- Reduced duplicate-case inflation in batch analysis for the same legal subject represented with different textual IDs.
- Scanner runtime panel was aligned with actual observability payload keys and endpoint naming format.
