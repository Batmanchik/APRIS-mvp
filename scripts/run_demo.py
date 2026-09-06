"""Поднять всю систему одной командой: API, интерфейс и свежая очередь.

    python scripts/run_demo.py                  # быстрый мир
    python scripts/run_demo.py --preset full    # полный мир
    python scripts/run_demo.py --no-pipeline    # не пересобирать очередь

Что делает по шагам:

1. собирает очередь аналитика (`scripts/run_pipeline.py`), чтобы страница
   «Очередь аналитика» открывалась с данными, а не с инструкцией;
2. поднимает FastAPI и **ждёт**, пока он ответит на health, а не спит
   фиксированные пять секунд;
3. поднимает Streamlit, дожидается его health и печатает адрес;
4. по Ctrl+C гасит оба процесса.

Зачем скрипт, если есть docker-compose: он для машины с докером, а на защите
и на ноутбуке чаще нужен запуск без него. Проверка здоровья здесь та же, что
в compose, поэтому «у меня не поднялось» разбирается одинаково в обоих
случаях.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

API_HEALTH = "http://127.0.0.1:{port}/api/v1/health"
UI_HEALTH = "http://127.0.0.1:{port}/_stcore/health"


def _wait_for(url: str, name: str, timeout: float = 120.0) -> bool:
    """Опрашивать health, пока не ответит. Возвращает False по таймауту."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    print(f"  {name}: не поднялся за {timeout:.0f} с — смотрите вывод выше")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    parser.add_argument("--no-pipeline", action="store_true",
                        help="не пересобирать очередь, взять последнюю")
    args = parser.parse_args()

    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")
    environment["CHEOPS_API_BASE_URL"] = f"http://127.0.0.1:{args.api_port}"

    if not args.no_pipeline:
        print("1/3  собираю очередь аналитика…")
        pipeline = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py",
             "--preset", args.preset, "--seed", str(args.seed)],
            env=environment,
            check=False,
        )
        if pipeline.returncode != 0:
            print("     конвейер упал — интерфейс поднимется, но очередь будет пустой")
    else:
        print("1/3  очередь не пересобираю")

    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    try:
        print("2/3  поднимаю API…")
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "apris.api.main:app",
             "--host", "127.0.0.1", "--port", str(args.api_port)],
            env=environment,
        )
        processes.append(("API", api))
        if not _wait_for(API_HEALTH.format(port=args.api_port), "API"):
            return 1
        print(f"     API готов: http://127.0.0.1:{args.api_port}/docs")

        print("3/3  поднимаю интерфейс…")
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.address", "127.0.0.1",
             "--server.port", str(args.ui_port),
             "--server.headless", "true"],
            env=environment,
        )
        processes.append(("Интерфейс", ui))
        if not _wait_for(UI_HEALTH.format(port=args.ui_port), "Интерфейс"):
            return 1

        print()
        print(f"  Готово. Интерфейс: http://127.0.0.1:{args.ui_port}")
        print("  Ctrl+C — остановить оба процесса.")
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n  {name} завершился с кодом {process.returncode}")
                    return process.returncode or 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  останавливаю…")
        return 0
    finally:
        for _, process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _, process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
