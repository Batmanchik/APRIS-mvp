# Cheops AI Manual Review Checklist

Используется для критичных модулей: `domain`, `risk scoring`, `etl`, `api contracts`.

## 1) Domain Invariants
- Все инварианты явно проверяются и тестируются.
- Ошибки валидации возвращают понятные причины.
- Нет неявных fallback, скрывающих поврежденные входные данные.

## 2) API Contract Integrity
- Request/response схемы версионированы.
- Изменения контрактов сопровождаются миграционной заметкой.
- Ошибки 4xx/5xx разделены корректно.

## 3) ML/Scoring Correctness
- Нет leakage между train/val/test.
- Калибровка не ломает ranking по high-risk кейсам.
- Fallback поведение детерминировано и наблюдаемо.

## 4) Code Quality
- Функции > 40 строк обоснованы или декомпозированы.
- Нет дублирования бизнес-логики между слоями.
- Исключения поднимаются с сохранением контекста.

## 5) Architecture Purity
- Слои не нарушены (проверка import-linter + ручная проверка PR diff).
- Interface слой не содержит бизнес-логику.
- Application слой не зависит от конкретного web framework.

## 6) Reliability & Observability
- Health-check и error-pathы покрыты тестами.
- Ключевые операции имеют структурируемые логи.
- Для деградаций задана стратегия (retry/fallback/abort) и она протестирована.

