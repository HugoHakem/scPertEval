"""The warm/prepare path fits embedding spaces (PCA) once, sized for the largest k requested.

Covers the ``prepare`` hook seam on the space registry: with several PCA dimensions requested,
PCA must be fit exactly once at ``max(k)`` (components are nested, so smaller ``pca_k`` slice from
it) and never refit — deterministically, independent of set iteration order. Also checks that
correctness does not depend on the hook: lazy use with no warm still works.
"""

from __future__ import annotations

import numpy as np
from conftest import make_cfg, make_dataset

from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols.resolve import resolve_protocols


def _spy_fit_pca(ctx):
    """Record the ``n_components`` of every ``_fit_pca`` call on ``ctx``."""
    calls: list[int] = []
    orig = ctx._fit_pca

    def spy(n_components):
        calls.append(n_components)
        return orig(n_components)

    ctx._fit_pca = spy  # instance attribute shadows the bound method
    return calls


def _ctx(ng=120):
    # ng large enough that pca_50 and pca_100 are both valid (k <= min(n_cells, n_genes)).
    cfg = make_cfg()
    return Context(Dataset(make_dataset(ng=ng), cfg), cfg)


def test_warm_fits_pca_once_at_max_k():
    """Two PCA dims requested (pca_50 + a larger pca_100): one fit, at the max k, no refit."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    protocols = resolve_protocols(["energy_distance_pca_k=50", "sinkhorn_w2_pca_k=100"])
    ctx.warm(protocols)

    assert calls == [100], f"expected a single fit at k=100, got {calls}"

    # Exercising the projections for both dims must not trigger any further fit.
    ctx.ref_projection("pca_50")
    ctx.ref_projection("pca_100")
    assert calls == [100], f"projection triggered a refit: {calls}"

    proj = ctx.ref_projection("pca_100")
    assert proj.shape[1] <= 100


def test_warm_is_order_independent():
    """The single fit is sized for the max k regardless of which spec comes first."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    ctx.warm(resolve_protocols(["sinkhorn_w2_pca_k=100", "energy_distance_pca_k=50"]))
    assert calls == [100]


def test_lazy_pca_without_warm_is_correct():
    """Correctness must not depend on the hook: no warm, projections still compute."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    proj = ctx.ref_projection("pca_50")
    assert proj.shape[0] == len(ctx.reference().cells)
    assert proj.shape[1] == 50
    assert np.isfinite(proj).all()
    # A single lazy fit happened (floored at 50), and it was not pre-warmed.
    assert calls == [50]


def test_single_dim_case_fits_once_at_floor():
    """The common single-dimension case still fits exactly once (no added work)."""
    ctx = _ctx()
    calls = _spy_fit_pca(ctx)
    ctx.warm(resolve_protocols(["energy_distance_pca_k=50"]))
    ctx.ref_projection("pca_50")
    assert calls == [50]
