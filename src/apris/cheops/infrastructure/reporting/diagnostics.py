"""The five standard diagnostics, drawn.

Why this module exists
----------------------
Three of these five were already implemented — ``purged_walk_forward_splits``,
``quintile_ladder`` and ``permutation_importance_with_noise_floor`` all live
in ``ml/validation_v2.py`` and have since the validation layer was written.
They produced numbers in a JSON file and sentences in docstrings, and nobody
outside the code ever saw one. A diagnostic nobody can look at does not
persuade anybody, which is most of what a diagnostic is for.

The set is the standard quantitative-research checklist, adapted from returns
to payments:

    1  distribution against a fitted normal, plus a Q-Q plot
    2  stability of the signal over time
    3  the quintile ladder — does the score RANK, or only separate
    4  purged walk-forward splits — train always before test, with a gap
    5  permutation importance against a shuffled control

Four and five are the ones that decide whether a result is real. Three is the
one that decides whether it is usable: a score with a good AUC and a jumbled
ladder separates on a subset rather than ordering the population, and it will
not survive a threshold being moved.

Every panel carries its own measurement, so the figure states its result
rather than inviting the reader to read a shape and trust it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from apris.cheops.infrastructure.ml.validation_v2 import (
    ImportanceResult,
    LadderResult,
    WalkForwardSplit,
)
from apris.cheops.infrastructure.reporting.figures import (
    ACCENT,
    CASH,
    DEFAULT_FIGURE_DIR,
    FRAUD,
    HARD_NEGATIVE,
    HONEST,
    INK,
    INK_FAINT,
    INK_SOFT,
    RULE,
    _save,
)


# ==========================================================================
# 1 — know your data
# ==========================================================================


def plot_amount_distribution(
    amounts: Sequence[float],
    *,
    name: str = "diag_amounts",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """What payment amounts actually do, against the normal nobody should assume.

    The equivalent of the stylised-facts check in returns research. Payment
    sizes are heavy-tailed and multi-modal — salaries cluster, rents cluster,
    a cash-out is its own mode — so any method that assumes a bell curve is
    wrong before it starts. The Q-Q panel shows it directly rather than
    asserting it.
    """
    values = np.asarray([a for a in amounts if a > 0], dtype=float)
    logs = np.log10(values)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    left.hist(logs, bins=70, color=ACCENT, alpha=0.85, density=True,
              label="actual amounts")
    grid = np.linspace(logs.min(), logs.max(), 400)
    mean, sd = float(logs.mean()), float(logs.std())
    normal = np.exp(-((grid - mean) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))
    left.plot(grid, normal, color=HARD_NEGATIVE, label="normal fit")
    left.set_xlabel("payment amount, log₁₀ KZT")
    left.set_ylabel("density")
    left.set_title("Amounts are not normal")
    left.legend()

    ordered = np.sort(logs)
    probabilities = (np.arange(len(ordered)) + 0.5) / len(ordered)
    theoretical = mean + sd * np.sqrt(2) * _erfinv(2 * probabilities - 1)
    right.scatter(theoretical, ordered, s=4, color=ACCENT, alpha=0.5)
    line = np.linspace(theoretical.min(), theoretical.max(), 2)
    right.plot(line, line, color=HARD_NEGATIVE, linestyle="--")
    right.set_xlabel("theoretical quantiles")
    right.set_ylabel("sample quantiles")
    right.set_title("Q-Q: the tails are the whole story")

    kurtosis = float(((logs - mean) ** 4).mean() / sd**4 - 3.0)
    skew = float(((logs - mean) ** 3).mean() / sd**3)
    right.text(
        0.03, 0.95,
        f"excess kurtosis {kurtosis:+.2f}\nskew {skew:+.2f}",
        transform=right.transAxes, va="top", fontsize=10, color=INK,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EEF1F4", "edgecolor": RULE},
    )

    fig.suptitle("Know your data: what payment amounts actually do",
                 fontsize=14, fontweight="bold", color=INK)
    return _save(
        fig, name,
        f"{len(values):,} payments from the simulated world. Log scale: a linear "
        "axis shows one bar.",
        directory, note_y=-0.04,
    )


def _erfinv(x: np.ndarray) -> np.ndarray:
    """Inverse error function, good to about 4 decimals.

    Written out rather than pulled from scipy: this repository does not
    otherwise depend on scipy, and one function is not worth a dependency in
    the edge-function-sized budget the rest of the project keeps.
    """
    a = 0.147
    ln = np.log(1 - x**2)
    first = 2 / (np.pi * a) + ln / 2
    return np.sign(x) * np.sqrt(np.sqrt(first**2 - ln / a) - first)


# ==========================================================================
# 2 — is the signal stable, or did one period carry it
# ==========================================================================


def plot_signal_stability(
    period_labels: Sequence[str],
    scores: Sequence[float],
    *,
    metric: str = "ROC-AUC",
    name: str = "diag_stability",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """One bar per period. A signal that lives in one month is not a signal.

    The payments analogue of the stationarity check: the question there is
    whether a regression is spurious, and here it is whether the number in
    the headline is an average over periods that disagree.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(scores))
    values = np.asarray(scores, dtype=float)
    mean = float(np.nanmean(values))

    colors = [ACCENT if v >= mean else HARD_NEGATIVE for v in values]
    ax.bar(positions, values, color=colors, width=0.62)
    ax.axhline(mean, color=INK_SOFT, linestyle="--", linewidth=1.4)
    ax.axhline(0.5, color=INK_FAINT, linestyle=":", linewidth=1.2)
    ax.text(len(scores) - 0.4, mean, f"  mean {mean:.3f}", va="center",
            color=INK_SOFT, fontsize=10)
    ax.text(len(scores) - 0.4, 0.5, "  chance", va="center", color=INK_FAINT, fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(period_labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(metric)
    ax.set_title(f"Is the signal stable? Spread {values.max() - values.min():.3f}")
    return _save(fig, name,
                 "One fold of the purged walk-forward split per bar. Orange is below the mean.",
                 directory)


# ==========================================================================
# 3 — the quintile ladder
# ==========================================================================


def plot_quintile_ladder(
    ladder: LadderResult,
    *,
    name: str = "diag_ladder",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """Sort by score, bucket, read the fraud rate per bucket.

    A real signal RANKS: the rate climbs from the bottom bucket to the top.
    A good AUC with a jumbled ladder means the score separates a subset and
    does not order the population, and it will break the moment somebody
    moves the threshold — which is the only thing an operator ever does.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    rates = np.asarray(ladder.bucket_rates, dtype=float)
    positions = np.arange(len(rates))

    shade = np.linspace(0.35, 1.0, len(rates))
    colors = [(*_hex_to_rgb(FRAUD), s) for s in shade]
    ax.bar(positions, rates, color=colors, width=0.66, edgecolor=FRAUD, linewidth=1.0)

    for position, (rate, size) in enumerate(zip(rates, ladder.bucket_sizes)):
        ax.text(position, rate + 0.02, f"{rate:.2f}", ha="center", color=INK, fontsize=10)
        ax.text(position, -0.055, f"n={size}", ha="center", color=INK_FAINT, fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Q1\nlowest score", *[f"Q{i + 2}" for i in range(len(rates) - 2)], "Q5\nhighest score"]
    )
    ax.set_ylim(-0.09, min(1.05, max(rates.max() * 1.25, 0.3)))
    ax.set_ylabel("share that really is fraud")

    verdict = "monotonic" if ladder.monotonic else "NOT monotonic"
    colour = HONEST if ladder.monotonic else HARD_NEGATIVE
    ax.set_title("Does the score rank, or only separate?")
    ax.text(
        0.02, 0.95,
        f"{verdict}\ntop − bottom {ladder.spread:+.3f}\nrank corr {ladder.rank_correlation:+.3f}",
        transform=ax.transAxes, va="top", fontsize=11, color=colour, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EEF1F4", "edgecolor": RULE},
    )
    return _save(fig, name,
                 "Out-of-fold scores, sorted and cut into five equal buckets.",
                 directory, note_y=-0.06)


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


# ==========================================================================
# 4 — the honest backtest
# ==========================================================================


def plot_walk_forward(
    splits: Sequence[WalkForwardSplit],
    total: int,
    *,
    name: str = "diag_walk_forward",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """Train always before test, with a purge gap between them.

    Ordinary k-fold trains on the future to predict the past, which inflates
    every score quietly. A case here also SPANS a window of events, so a case
    straddling the boundary carries the test period into training even in a
    plain time split. The purge drops those rather than including them
    silently, and this figure is where the reader sees that it happened.
    """
    fig, ax = plt.subplots(figsize=(10.5, 1.15 * len(splits) + 2.4))

    for row, split in enumerate(splits):
        y = len(splits) - row - 1
        train_end = len(split.train) + len(split.purged)
        test_end = train_end + len(split.test)
        ax.barh(y, len(split.train), left=0, color=ACCENT, height=0.55)
        ax.barh(y, len(split.purged), left=len(split.train), color=FRAUD, height=0.55)
        ax.barh(y, len(split.test), left=train_end, color=HARD_NEGATIVE, height=0.55)
        ax.text(test_end + total * 0.012, y, f"purged {split.purged_count}",
                va="center", fontsize=9, color=INK_FAINT)

    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"split {len(splits) - i}" for i in range(len(splits))])
    ax.set_xlabel("cases, ordered in time")
    ax.set_xlim(0, total * 1.14)
    ax.set_title("The honest backtest: training always precedes the test")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
        plt.Rectangle((0, 0), 1, 1, color=FRAUD),
        plt.Rectangle((0, 0), 1, 1, color=HARD_NEGATIVE),
    ]
    ax.legend(handles, ["train", "purge gap", "test"], ncol=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, name,
                 "A case spans a window of events, so one straddling the boundary would "
                 "carry the test period into training. The gap drops it.",
                 directory, note_y=-0.10)


# ==========================================================================
# 5 — separate signal from noise
# ==========================================================================


def plot_importance(
    result: ImportanceResult,
    *,
    name: str = "diag_importance",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """Every feature against a deliberately random column.

    The control is not a convention, it is the measurement: a shuffled column
    is fitted and permuted exactly like a real feature, so whatever scores no
    better than it carries no information. Comparing to an eyeballed
    threshold instead is how a story survives its own evidence.

    Error bars are the standard deviation over repeats. The quantity is an
    average over random shuffles, so one run proves nothing.
    """
    order = np.argsort(result.means)[::-1]
    names = [result.feature_names[i] for i in order]
    means = np.asarray([result.means[i] for i in order], dtype=float)
    errors = np.asarray([result.errors[i] for i in order], dtype=float)

    fig, ax = plt.subplots(figsize=(max(9.5, 1.05 * len(names) + 3), 5.8))
    positions = np.arange(len(names))
    above = means > result.noise_floor
    colors = [CASH if hit else RULE for hit in above]

    ax.bar(positions, means, yerr=errors, color=colors, width=0.66,
           error_kw={"ecolor": INK_SOFT, "capsize": 4, "linewidth": 1.2})
    ax.bar([len(names)], [result.noise_floor], yerr=[result.noise_floor_error],
           color=INK_FAINT, width=0.66,
           error_kw={"ecolor": INK_SOFT, "capsize": 4, "linewidth": 1.2})
    ax.axhline(result.noise_floor, color=FRAUD, linestyle="--", linewidth=1.5)
    ax.text(len(names) + 0.45, result.noise_floor, "  noise floor",
            va="bottom", ha="right", color=FRAUD, fontsize=10)

    ax.set_xticks([*positions, len(names)])
    ax.set_xticklabels([*names, "shuffled\ncontrol"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("drop in ROC-AUC when the column is shuffled")
    ax.set_title(
        f"Signal against noise: {len(result.above_floor())} of "
        f"{len(names)} features clear the floor"
    )
    return _save(fig, name,
                 "A shuffled column is fitted and permuted like any real feature. "
                 "Anything at or below it carries no information.",
                 directory, note_y=-0.18)


__all__ = [
    "plot_amount_distribution",
    "plot_importance",
    "plot_quintile_ladder",
    "plot_signal_stability",
    "plot_walk_forward",
]
