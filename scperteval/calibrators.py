"""Calibrators turn the raw metric values measured on each control into a final
per-metric score. Each declares the control roles it needs, a per-perturbation
combine, and a cross-perturbation aggregate.
"""

from __future__ import annotations

import numpy as np

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
}
