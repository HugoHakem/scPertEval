"""Calibrators turn raw control metric values into a final per-metric score.

Each declares the control roles it needs, a per-perturbation combine, and a
cross-perturbation aggregate.
"""

from __future__ import annotations

import numpy as np

from .types import Calibrator


def drf_per_pert(raws, p):
    """Dynamic Range Fraction for one perturbation (Miller et al. 2025), clipped to ``[-1, 1]``.

    ``raws`` is ``{"positive": ..., "negative": ...}`` — the two raw metric values to compare,
    already computed (e.g. by two separate :func:`~scperteval.api.score` calls). Pure function of
    those two numbers plus the protocol's own ``better``/``perfect`` metadata — no ``Context``
    needed, so it also works to compare two independently-scored prediction sets directly (the
    ``drf`` :data:`CALIBRATORS` entry can only compare within one ``Context``, e.g. a prediction
    against a dataset-native source; comparing two prediction files needs this function applied
    to their already-computed raw values instead — see ``models/compare.py``'s ``per_pert_drf``).
    """
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


#: ``{name: Calibrator}`` dict — ``drf`` and ``bds`` for calibration mode, ``score`` for prediction-scoring mode.
CALIBRATORS = {
    "drf": Calibrator(
        "drf",
        ("positive", "negative"),
        drf_per_pert,
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
}
