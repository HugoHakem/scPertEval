"""Pure-function tests for the new metric family: r2, l2, weighted_pearson, weighted_r2, nir."""

from __future__ import annotations

import numpy as np
import pytest

from scperteval.protocols import metrics as M


def test_r2_is_perfect_for_identical_profiles():
    gt = np.array([1.0, 2.0, 3.0, -1.0, 0.5])
    assert M.r2(gt, gt.copy(), None) == 1.0


class _UniformWeightCtx:
    """Fake Context whose DE statistic is constant, so mejia_weights falls back to uniform."""

    current_pert = "p"

    class cfg:
        truth = "truth"

    def de(self, pert, truth, ref):
        class _DE:
            statistic = np.ones(10)

        return _DE()


def test_weighted_pearson_matches_pearson_with_uniform_weights():
    rng = np.random.default_rng(0)
    gt, pred = rng.normal(size=10), rng.normal(size=10)
    assert M.weighted_pearson(gt, pred, _UniformWeightCtx(), exp=2.0) == pytest.approx(M.pearson(gt, pred, None))


def test_weighted_r2_matches_r2_with_uniform_weights():
    rng = np.random.default_rng(0)
    gt, pred = rng.normal(size=10), rng.normal(size=10)
    assert M.weighted_r2(gt, pred, _UniformWeightCtx(), exp=2.0) == pytest.approx(M.r2(gt, pred, None))


def test_nir_is_one_minus_transpose_rank():
    rng = np.random.default_rng(0)
    gt = [rng.normal(size=5) for _ in range(4)]
    pred = [rng.normal(size=5) for _ in range(4)]
    assert np.allclose(M.nir(gt, pred, None), 1.0 - M.rank_retrieval(gt, pred, None, transpose=True))
