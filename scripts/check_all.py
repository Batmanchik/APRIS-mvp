"""Run every gate CI runs, in the same order, and report honestly.

Why this exists
---------------
Three CI failures in one day, and none of them was caught locally — not
because the checks are hard to run, but because a subset was run and the
result was read as "the gates pass". Ruff, mypy and pytest passing says
nothing about bandit, and bandit is what failed.

A second trap this closes: a broken console-script wrapper. After the project
directory was renamed, ``lint-imports.exe`` kept exiting 1 with no output at
all, which reads exactly like a contract violation and is not one. Any gate
that fails without saying anything is reported here as SUSPECT rather than
FAILED, so a tooling problem is never mistaken for a code problem.

Usage:
    python scripts/check_all.py            # everything
    python scripts/check_all.py --fast     # skip the slow test suite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

PYTHON = sys.executable

GATES: list[tuple[str, list[str], bool]] = [
    # name, command, is_slow
    ("Ruff", [PYTHON, "-m", "ruff", "check", "src", "tests", "pages", "app.py"], False),
    ("Mypy", [PYTHON, "-m", "mypy"], False),
    ("Import Linter", [PYTHON, "-m", "importlinter.cli"], False),
    ("Bandit", [PYTHON, "-m", "bandit", "-q", "-r", "src/apris",
                "-x", "src/apris/crypto_ponzi", "-s", "B101"], False),
    ("Radon", [PYTHON, "-m", "radon", "cc", "src/apris/cheops", "-s", "-n", "B"], False),
    ("Pytest", [PYTHON, "-m", "pytest", "-q", "--no-header"], True),
]


@dataclass
class GateResult:
    name: str
    status: str      # PASS | FAIL | SUSPECT | SKIP
    seconds: float
    tail: str


def _run_import_linter() -> tuple[int, str]:
    """Call import-linter through its Python API.

    The console script embeds an absolute path at install time and silently
    breaks if the project directory is renamed. Going through the API removes
    a whole class of false failure.
    """
    try:
        from importlinter.cli import EXIT_STATUS_SUCCESS, lint_imports
    except ImportError as exc:  # pragma: no cover - environment problem
        return 1, f"import-linter is not installed: {exc}"
    code = lint_imports()
    return (0 if code == EXIT_STATUS_SUCCESS else 1), ""


def run_gate(name: str, command: list[str]) -> GateResult:
    started = time.time()

    if name == "Import Linter":
        code, message = _run_import_linter()
        output = message
    else:
        completed = subprocess.run(command, capture_output=True, text=True)
        code = completed.returncode
        output = (completed.stdout + completed.stderr).strip()

    elapsed = time.time() - started
    lines = [line for line in output.splitlines() if line.strip()]
    tail = "\n".join(lines[-6:])

    if code == 0:
        status = "PASS"
    elif not lines:
        # A gate that fails without saying anything is almost always broken
        # tooling, not broken code. Flag it as such instead of raising a
        # false alarm about the source.
        status = "SUSPECT"
    else:
        status = "FAIL"

    return GateResult(name=name, status=status, seconds=elapsed, tail=tail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the slow test suite")
    args = parser.parse_args()

    results: list[GateResult] = []
    for name, command, is_slow in GATES:
        if is_slow and args.fast:
            results.append(GateResult(name, "SKIP", 0.0, ""))
            print(f"  {name:<16} SKIP")
            continue
        print(f"  {name:<16} running...", flush=True)
        result = run_gate(name, command)
        results.append(result)
        print(f"\033[F  {result.name:<16} {result.status:<8} {result.seconds:6.1f}s")

    print("\n" + "=" * 58)
    failed = [r for r in results if r.status == "FAIL"]
    suspect = [r for r in results if r.status == "SUSPECT"]

    for result in failed + suspect:
        print(f"\n--- {result.name}: {result.status} ---")
        print(result.tail or "(no output)")

    if suspect:
        print(
            "\nSUSPECT means the gate failed without printing anything, which "
            "usually means broken tooling rather than broken code. Reinstall "
            "the package that provides it before believing the failure."
        )

    if failed:
        print(f"\n{len(failed)} gate(s) FAILED. Do not push.")
        return 1
    if suspect:
        print(f"\n{len(suspect)} gate(s) SUSPECT. Investigate before pushing.")
        return 2

    skipped = sum(1 for r in results if r.status == "SKIP")
    note = f" ({skipped} skipped — rerun without --fast before pushing)" if skipped else ""
    print(f"\nAll gates pass{note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
