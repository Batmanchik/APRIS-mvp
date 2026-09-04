# История улучшений проекта APRIS

В этом файле фиксируются все архитектурные и алгоритмические изменения, внедренные по результатам ревью проекта.

---

## Блок 5: Научная секция и формализация методологии (21.03.2026, выполнено)

- Добавлен отдельный научный документ:
  - `docs/CHEOPS_AI_SCIENTIFIC_FOUNDATION.md`
- В документе зафиксированы:
  - математические обозначения и контракт события/кейса;
  - формулы преобразования операционных данных в признаки;
  - формальное определение "pyramid-like" в текущей ML-постановке;
  - правила типологий (`LEGAL_LAYERING`, `LEGAL_TO_CRYPTO_BRIDGE`, `CRYPTO_MIXING`, `STRUCTURED_SPLITTING`, `CASH_OUT`);
  - формулы веток `tabular + sequence + graph + fusion`;
  - формулы calibration/ECE и drift/PSI;
  - ограничения интерпретации (ML-скоринг не является юридическим вердиктом).
- Обновлен `README.md` ссылкой на научную секцию.

---

## Блок 8: Архитектурный аудит и code hygiene cleanup (21.03.2026, выполнено)

### Анализ
- Проведен архитектурный аудит структуры репозитория и runtime-контуров.
- Отчет добавлен:
  - `docs/ARCHITECTURE_AUDIT_2026-03-21.md`

### Чистка и оптимизация
- Удален устаревший скрипт:
  - `fix_app.py`.
- Удалены реально неиспользуемые переменные/импорты в ML и вспомогательных модулях.
- Удален мертвый helper `_sigmoid` в `engine_v2`.
- Обновлен `.gitignore` для runtime-мусора:
  - `mlruns/`
  - `artifacts/cheops_v2_*.joblib`
  - `artifacts/cheops_v2_*.json`

### Результат
- Кодовая база чище по стат-анализу и проще в сопровождении.
- `git status` больше не засоряется новыми v2 runtime-артефактами после переобучения.
- Quality gates сохранены зелеными.

---

## Блок 7: Этап 3 (Platform Hardening) — контейнеризация и CI smoke (21.03.2026, выполнено)

### Контейнеризация
- Добавлен `Dockerfile` для унифицированного запуска backend/frontend среды.
- Добавлен `docker-compose.yml` с сервисами:
  - `api` (FastAPI, порт `8000`, healthcheck `/api/v1/health`);
  - `ui` (Streamlit, порт `8501`, healthcheck `/_stcore/health`).
- В `ui` зафиксирован API-first маршрут через:
  - `CHEOPS_API_BASE_URL=http://api:8000`.
- Добавлен `.dockerignore` для ускорения сборки и чистого build-context.

### CI/Smoke
- В `.github/workflows/ci.yml` добавлен job `docker-smoke`:
  - `docker compose up -d --build`;
  - ожидание health API/UI;
  - smoke-запрос `/api/v2/health/model/details`;
  - гарантированный `docker compose down -v`.

### Документация
- `README.md` дополнен секцией `Run (Docker Compose)`.
- `docs/RELEASE_CHECKLIST.md` дополнен блоком `Container Smoke`.

---

## Блок 6: ML Governance и Benchmark-Selection Policy (21.03.2026, выполнено)

### Что добавлено
- Новый модуль:
  - `src/apris/cheops/infrastructure/ml/model_registry_v2.py`
- Новый артефакт:
  - `artifacts/cheops_v2_model_registry.json`

### Что изменено
- В benchmark (`benchmark_v2`) добавлена явная политика выбора winner:
  - формула `selection_score = 0.65*roc_auc + 0.20*accuracy + 0.10*(1-brier) + 0.05*(1-ece)`;
  - `selection_policy`, `ranking`, `winner_reason` в отчете.
- `train_model.py` теперь:
  - сохраняет model-registry после каждого обучения;
  - при `--benchmark` обновляет registry фактическим winner и reason.
- `/api/v2/health/model/details` дополнен блоком `registry` (выбранный табличный кандидат и причина выбора).

### Проверки
- Добавлены/обновлены тесты:
  - `tests/unit/test_cheops_model_registry_v2.py`
  - `tests/unit/test_cheops_benchmark_v2.py`
  - `tests/api/test_api_v2_endpoints.py`
  - `tests/smoke/test_e2e_smoke.py`
- Прогон:
  - `ruff check .` — passed
  - `mypy src/apris` — passed
  - `pytest --cov=src/apris` — passed (`72 passed`, coverage > 75%)

---

## Блок 4: Cheops AI — Эргономичность и оптимизация (21.03.2026, выполнено)

### Scanner (batch pipeline)
- Вынесена подготовка кейсов из `pages/2_Сканнер_транзакций.py` в отдельный модуль:
  - `src/apris/frontend/scanner_pipeline.py`.
- Оптимизирован путь для сырого transaction log:
  - вместо повторных full-scan фильтров по DataFrame для каждой сущности используется единая индексная карта `entity_id -> tx_idx`.
- Добавлена защита low-volume сущностей:
  - нормализация операционных полей по `OPERATIONAL_INPUT_BOUNDS` перед переводом в model-features.
- В Scanner UI добавлены:
  - длительность последнего скана,
  - вывод состояния/метрик модели через `/api/v2/health/model/details`.

### Alert Inbox / Dossier UX
- Полностью пересобран `pages/1_Дашборд_аномалий.py` с упором на рабочий сценарий аналитика:
  - фильтры риска (`All`, `Medium+`, `High+`, `Critical`),
  - ограничение `Top N`,
  - поиск по `entity_id`,
  - стабильный выбор кейса по id (убран хрупкий `split`-парсинг строки).
- Добавлен кэш генерации демонстрационного графа (`@st.cache_data`) для снижения времени перерисовки.
- В досье подключен explain v2 (`/api/v2/explain`) с fallback на v1 explain для отказоустойчивости.

### API client
- В `src/apris/frontend/api_client.py` добавлен метод:
  - `health_check_v2_model_details()`.

### Тесты и качество
- Добавлены unit-тесты scanner-пайплайна:
  - `tests/unit/test_scanner_pipeline.py`.
- Обновлены тесты API-клиента:
  - `tests/unit/test_frontend_api_client.py`.
- Результат quality-gate после изменений:
  - `ruff check .` — passed
  - `mypy src/apris` — passed
  - `pytest --cov=src/apris` — `69 passed`, coverage `81.21%`

---

## Блок 1: Архитектура и структура кода (Выполнено)

### Серверная часть (API & Backend)
- **Переезд в модульную структуру:** Весь логический код (`risk_engine.py`, `data_generator.py`, `graph_module.py` и др.) перенесён в пакет `src/apris/`. Ранее файлы лежали в корне.
- **Интеграция FastAPI:** Создан REST API сервер (`src/apris/api/main.py`), полностью отделивший ML-движок от фронтенда. Сервер предоставляет 5 эндпоинтов для скоринга, объяснений и метаданных. Добавлена автодокументация Swagger (порт 8000).

### Клиентская часть (Frontend)
- **Расщепление Streamlit:** Монолитный файл `app.py` (ранее ~1361 строка) уменьшен до ~160 строк. Оставлен только код для визуальных стилей и навигации.
- **Multipage-архитектура:** Логика четырёх независимых разделов вынесена в отдельные файлы в папке `pages/` (Оценка кейса, Live-демо, Crypto-Ponzi, Гид).
- **HTTP-клиент:** Создан `src/apris/frontend/api_client.py` для стандартизированных запросов от интерфейса к FastAPI.

### Инфраструктура
- Настроен `pyproject.toml` для локальной установки модуля APRIS (`pip install -e .`), позволяющий использовать абсолютные импорты вида `from apris.risk_engine import...` без конфликтов путей.
- Обновлен запускной `start_app.bat` — теперь он стартует два фоновых процесса: интерфейс (порт 8501) и API-сервер (порт 8000).

---

## Блок 2: Машинное обучение и данные (Выполнено)

### Алгоритм и метрики
- **Переход на LightGBM:** Модель `RandomForestClassifier` заменена на более производительный градиентный бустинг (`lightgbm.LGBMClassifier`). Это дало прирост в скорости без потери качества. Метрики для синтетического демо-датасета: Accuracy 0.962, ROC-AUC 0.993. Важности признаков были нормализованы для совместимости со старым UI.

### Трекинг экспериментов (MLOps)
- **Интеграция MLflow:** В файл `train_model.py` встроен инструментарий для сохранения каждого цикла обучения. Логируются: набор гиперпараметров, все ключевые метрики, график ROC-кривой и сама сериализованная модель (для быстрого отката).

### Работа с реальными данными (ETL)
- **Новый модуль ETL:** Написан `src/apris/etl.py` — слой интеграции (ETL Pipeline). Он умеет принимать реальные сырые выгрузки банковских или блокчейн-транзакций (в форматах `csv` и `json`).
- **Трансформация графа:** Написана логика для сведения плоского лога перечислений (`sender_id`, `receiver_id`, `amount`) к 12 укрупнённым операционным фактам, а затем — к 9 итоговым риск-признакам. 
- Теперь `train_model.py` может принимать внешний файл вместо синтетики командой:
  `python -m apris.train_model --data your_real_data.csv`

---

## Блок 3: Автоматизация и UI/UX (Выполнено)

### Переход от калькулятора к "Радару"
- Интерфейс кардинально переработан по требованиям бизнеса: вместо ручного ввода данных реализована система массового мониторинга.
- Скрыта форма "Оценки кейса" (перенесена в `Ручная проверка` для отладки).

### Сканнер транзакций (ETL Trigger)
- Создан модуль `Сканнер транзакций` — точка входа (File Drop) для загрузки логов (например, 1500 транзакций за день). Сканнер автоматически парсит сырой граф переводов, формирует бизнес-показатели субъектов и прогоняет батч через ML-модель.

### Дашборд аномалий (Alert Inbox)
- Основное окно аналитика: выводит таблицу (`Leaderboard`) всех отсканированных субъектов, отсортированную по убыванию риска.
- Реализован функционал **«Досье»**: при нажатии на подозрительного клиента разворачивается крипто-сеть платежей («паутина»), показывается гранулярная структура концентрации вкладов и объяснение (Explainability) от модели (какие признаки повлияли на высокий скор).

---

## Block 9: Runtime Observability + UI Icon Rendering Fix (21.03.2026, completed)

### Implemented changes
- API runtime observability endpoint integrated into frontend scanner UX:
  - Scanner now queries `GET /api/v2/health/runtime` via `api_client.health_check_v2_runtime()`.
  - Added runtime summary in scanner (requests/errors/error_rate).
  - Added endpoint-focused runtime table (`/api/v2/score/batch`, `/api/v2/score`, `/api/v2/explain`) with `p95` latency.
- Fixed Streamlit icon ligature rendering artifacts (e.g. `keyboard_double...` text shown instead of icon):
  - `app.py` now loads Google Material icon fonts (`Material Symbols Rounded`, `Material Icons`) together with Inter.

### Documentation updates
- `README.md`:
  - added `GET /api/v2/health/runtime` to public API list;
  - documented `X-Request-Id` tracing header and runtime observability endpoint semantics.
- `docs/RELEASE_CHECKLIST.md`:
  - runtime health endpoint included in required smoke health checks.

### Quality verification
- `python -m ruff check .` -> passed
- `python -m mypy src/apris` -> passed
- `python -m pytest -q --maxfail=1` -> passed (`74` tests)

---

## Block 10: Reliability hardening (API client retry + degraded scanner UX) (21.03.2026, completed)

### Implemented
- `src/apris/frontend/api_client.py`
  - Added `ApiClientError` for normalized user-facing failures.
  - Added configurable retry policy for retryable GET endpoints:
    - `CHEOPS_API_RETRIES` (default `1`)
    - `CHEOPS_API_RETRY_BACKOFF` (default `0.2` sec)
  - Added bounded retries for connection/timeout and HTTP 5xx on retryable calls.
  - Added explicit error for non-JSON API responses.
- `pages/2_Сканнер_транзакций.py`
  - Added actionable API-down state (`_render_api_unavailable`) with retry button.
  - Added degraded runtime warning when API error-rate is elevated.
  - Added degraded batch warning when failure ratio is high.

### Tests
- `tests/unit/test_frontend_api_client.py`
  - Added retry-path tests (connection error, HTTP 503 recovery).
  - Added no-retry assertion for POST batch scoring.
  - Added non-JSON response failure test.

### Verification
- `python -m ruff check .` -> passed
- `python -m mypy src/apris` -> passed
- `python -m pytest -q --maxfail=1` -> passed (`78` tests)

---

## Block 11: Step A started — Entity contracts and resolution for IP/TOO (21.03.2026, completed)

### Implemented
- Added new domain module:
  - `src/apris/cheops/domain/entity_resolution.py`
- Added entity contract primitives:
  - `LegalEntityRecord`
  - `normalize_entity_record`
  - `validate_entity_schema`
  - `resolve_entity_key`
  - `group_entity_records`
  - `normalize_external_entity_id`
- Entity key policy is deterministic and priority-based by strongest anchor:
  - BIN -> IIN -> registration_no -> tax_id -> source_entity_id -> name/aliases.

### Scanner integration
- Updated `src/apris/frontend/scanner_pipeline.py`:
  - sender/receiver IDs are normalized before case building;
  - case-id fingerprint for tx-mode now uses resolved `entity_key` instead of raw unnormalized entity id.

### Tests added/updated
- Added: `tests/unit/test_cheops_entity_resolution.py`
- Updated: `tests/unit/test_scanner_pipeline.py`
  - new test for ID normalization/dedup behavior in tx batch preparation.

### Quality verification
- `python -m ruff check .` -> passed
- `python -m mypy src/apris` -> passed
- `python -m pytest -q --maxfail=1` -> passed (`85` tests)
- Runtime health rendering in scanner corrected to current observability schema:
  - reads `requests_total/errors_total/error_rate_total`;
  - maps endpoint stats by suffix match (`POST /api/v2/...`) and uses `latency_p95_ms`.
