"""Calibrators turn raw control metric values into a final per-metric score.

Each declares the control roles it needs, a per-perturbation combine, and a
cross-perturbation aggregate.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .types import Calibrator


def _drf_per_pert(raws, p):
    pos, neg = raws["positive"], raws["negative"]
    beyond_perfect = neg > p.perfect if p.better == "higher" else neg < p.perfect
    if not np.isfinite(neg) or beyond_perfect:
        return float("nan")
    if p.better == "higher":
        num, den = pos - neg, p.perfect - neg
    else:
        num, den = neg - pos, neg - p.perfect
    return float(np.clip(num / (den + 1e-6), -1.0, 1.0))


def _bds_per_pert(raws, p):
    pos, neg = raws["positive"], raws["negative"]
    wins = pos < neg if p.better == "lower" else pos > neg
    return float(wins)


def _paired_diff_per_pert(raws, p):
    """Per-perturbation gap, signed so positive means the ``positive`` role wins."""
    pos, neg = raws["positive"], raws["negative"]
    if not (np.isfinite(pos) and np.isfinite(neg)):
        return float("nan")
    return float(neg - pos) if p.better == "lower" else float(pos - neg)


def _bootstrap_ci(values, n_resamples=10_000, seed=42):
    """Percentile bootstrap 95% CI on the mean of ``values`` (nan entries dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    resampled = v[rng.integers(0, v.size, size=(n_resamples, v.size))]
    boot_means = resampled.mean(axis=1)
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.quantile(boot_means, 0.025)),
        "ci_high": float(np.quantile(boot_means, 0.975)),
    }


def _ttest_result(values):
    """Paired one-sided Student t-test (H1: mean diff > 0).

    Equivalent to a paired t-test on the two raw sources, since
    ``ttest_1samp(a - b, 0) == ttest_rel(a, b)``.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"mean": float("nan"), "statistic": float("nan"), "pvalue": float("nan")}
    statistic, pvalue = stats.ttest_1samp(v, popmean=0.0, alternative="greater")
    return {"mean": float(v.mean()), "statistic": float(statistic), "pvalue": float(pvalue)}


def _wilcoxon_result(values):
    """Paired one-sided Wilcoxon signed-rank test (H1: median diff > 0)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 1 or np.all(v == 0):
        return {"mean": float(v.mean()) if v.size else float("nan"), "statistic": float("nan"), "pvalue": float("nan")}
    statistic, pvalue = stats.wilcoxon(v, alternative="greater")
    return {"mean": float(v.mean()), "statistic": float(statistic), "pvalue": float(pvalue)}


#: ``{name: Calibrator}`` dict — ``drf`` and ``bds`` for calibration mode, ``score`` for prediction-scoring mode.
CALIBRATORS = {
    "drf": Calibrator(
        "drf",
        ("positive", "negative"),
        _drf_per_pert,
        lambda v: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))},
        description="Dynamic Range Fraction — mean/median over perturbations (Miller et al. 2025)",
    ),
    "bds": Calibrator(
        "bds",
        ("positive", "negative"),
        _bds_per_pert,
        lambda v: {"bds": float(np.nanmean(v))},
        description="Bound Discrimination Score — fraction of perturbations the positive control wins (SBB 2026)",
    ),
    "score": Calibrator(
        "score",
        ("prediction",),
        lambda raws, p: raws["prediction"],
        lambda v: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))},
        description="raw metric of a prediction vs ground truth — mean/median over perturbations (prediction-scoring mode)",
    ),
    "paired_ci": Calibrator(
        "paired_ci",
        ("positive", "negative"),
        _paired_diff_per_pert,
        _bootstrap_ci,
        description="10000-resample bootstrap 95% CI on the mean paired per-perturbation gap "
        "between --positive and --negative (e.g. a model prediction vs. a baseline source); "
        "the 'does it outperform' question behind Ahlmann-Eltze et al. 2025 and Miller et al. 2025",
    ),
    "ttest": Calibrator(
        "ttest",
        ("positive", "negative"),
        _paired_diff_per_pert,
        _ttest_result,
        description="paired one-sided Student t-test (H1: mean diff > 0) between --positive and "
        "--negative (Miller et al. 2025); reports the raw p-value — apply a Bonferroni "
        "correction yourself across however many comparisons you're running at once",
    ),
    "wilcoxon": Calibrator(
        "wilcoxon",
        ("positive", "negative"),
        _paired_diff_per_pert,
        _wilcoxon_result,
        description="paired one-sided Wilcoxon signed-rank test (H1: median diff > 0) between "
        "--positive and --negative (Miller et al. 2025); reports the raw p-value — apply a "
        "Bonferroni correction yourself across however many comparisons you're running at once",
    ),
}
