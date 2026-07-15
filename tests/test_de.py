"""Tests for the differential-expression backends, focused on the scanpy
``t-test_overestim_var`` variant added as a selectable DE method."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from scperteval.blocks.de import DE_METHODS, _moments, de_ttest_overestim, mejia_weights, ttest_from_moments


def _scanpy_overestim(Xt, Xr):
    """scanpy's own t-test_overestim_var scores/pvals for `target` vs `reference`,
    re-indexed back to gene order."""
    ng = Xt.shape[1]
    names = [str(i) for i in range(ng)]
    adata = ad.AnnData(np.vstack([Xt, Xr]).astype(np.float64))
    adata.var_names = names
    adata.obs["g"] = ["target"] * Xt.shape[0] + ["reference"] * Xr.shape[0]
    adata.obs["g"] = adata.obs["g"].astype("category")
    sc.tl.rank_genes_groups(adata, "g", groups=["target"], reference="reference", method="t-test_overestim_var")
    res = adata.uns["rank_genes_groups"]
    order = np.array([int(n) for n in res["names"]["target"]])
    scores = np.empty(ng)
    scores[order] = np.asarray(res["scores"]["target"], dtype=np.float64)
    pvals = np.empty(ng)
    pvals[order] = np.asarray(res["pvals"]["target"], dtype=np.float64)
    return scores, pvals


def test_overestim_var_matches_scanpy():
    """Our backend reproduces scanpy's t-test_overestim_var statistic and p-values."""
    rng = np.random.default_rng(0)
    Xt = rng.poisson(1.0, (40, 60)).astype(np.float64)  # small target group
    Xr = rng.poisson(1.3, (90, 60)).astype(np.float64)  # larger reference
    de = de_ttest_overestim(Xt, Xr)
    sc_scores, sc_pvals = _scanpy_overestim(Xt, Xr)
    assert np.allclose(de.statistic, sc_scores, atol=1e-5, rtol=1e-4)
    assert np.allclose(de.pvalue, sc_pvals, atol=1e-6, rtol=1e-4)
    assert de.pvalue_adj.shape == de.statistic.shape


def test_overestim_var_differs_from_plain_ttest():
    """Sanity: the overestimated-variance variant is more conservative (|t| no larger)."""
    rng = np.random.default_rng(1)
    Xt = rng.poisson(1.0, (30, 50)).astype(np.float64)
    Xr = rng.poisson(1.4, (120, 50)).astype(np.float64)
    over = de_ttest_overestim(Xt, Xr)
    plain = DE_METHODS["t-test"](Xt, Xr)
    assert np.all(np.abs(over.statistic) <= np.abs(plain.statistic) + 1e-9)
    assert not np.allclose(over.statistic, plain.statistic)


def test_overestim_var_registered_and_selectable():
    """It is a registered DE method, hence a selectable `run --de-method` backend
    (choices = DE_METHODS.names()) so new protocols can use it."""
    assert "t-test_overestim_var" in DE_METHODS
    assert "t-test_overestim_var" in DE_METHODS.names()


def test_mejia_weights_basic():
    """Min-max normalises |score| to [0, 1]; the largest-magnitude gene gets weight 1."""
    w = mejia_weights(np.array([-4.0, 1.0, 0.0, 2.0]))
    assert w[0] == 1.0  # |-4| is the max
    assert w[2] == 0.0  # |0| is the min
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_mejia_weights_constant_score_is_zero():
    """No spread to normalise against -> every weight is 0, not NaN/inf."""
    w = mejia_weights(np.full(5, 3.0))
    assert np.array_equal(w, np.zeros(5))


def test_mejia_weights_handles_nonfinite():
    """Non-finite entries don't poison the min/max and come back as 0."""
    w = mejia_weights(np.array([1.0, np.nan, 3.0, np.inf]))
    assert np.isfinite(w).all()
    assert w[1] == 0.0  # nan input -> 0 weight


def test_overestim_var_runs_through_export_path():
    """Hits the real export path used by `scperteval de`: compute_de_export -> ctx.de
    -> DE_METHODS dispatch, on a tiny in-memory dataset."""
    from scperteval.context import Context
    from scperteval.dataset import Dataset
    from scperteval.runner import compute_de
    from scperteval.types import RunConfig

    rng = np.random.default_rng(2)
    ng = 40
    parts, labels = [], []
    for lab, mean, n in [("control", 1.0, 80), ("pertA", 1.6, 50), ("pertB", 0.7, 50)]:
        parts.append(rng.poisson(mean, (n, ng)))
        labels += [lab] * n
    adata = ad.AnnData(np.vstack(parts).astype(np.float64))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = labels
    cfg = RunConfig(
        dataset="-", protocols=[], de_method="t-test_overestim_var", subsample=200, seed=0, min_cells=10, workers=1
    )
    ctx = Context(Dataset(adata, cfg), cfg)
    ctx._ensure_ref_sums()
    stat, padj = compute_de(ctx)
    assert stat.shape == (len(ctx.perturbations), ng)
    assert padj.shape == stat.shape
    assert np.isfinite(stat).all()


def test_ttest_declares_moment_capability():
    """t-test carries its moment implementation as a registry capability (not a name special-case);
    a cells-only method like MWU does not."""
    assert DE_METHODS.meta("t-test").get("from_moments") is ttest_from_moments
    assert DE_METHODS.meta("MWU").get("from_moments") is None


def test_moment_capability_dispatches_through_cache():
    """A registered moment-based method routes through Context's shared moment cache — its
    `from_moments` is called, the cells path is not, and a source's moments are computed once
    and reused across perturbations (no per-comparison recompute)."""
    from scperteval.context import Context
    from scperteval.dataset import Dataset
    from scperteval.types import RunConfig

    calls = {"from_moments": 0, "cells": 0}

    def spy_from_moments(*moments):
        calls["from_moments"] += 1
        return ttest_from_moments(*moments)

    @DE_METHODS.register("spy_moment", description="test double", from_moments=spy_from_moments)
    def de_spy(target, reference):
        calls["cells"] += 1
        return spy_from_moments(*_moments(target), *_moments(reference))

    try:
        rng = np.random.default_rng(3)
        ng = 30
        parts, labels = [], []
        for lab, mean, n in [("control", 1.0, 60), ("pertA", 1.5, 40), ("pertB", 0.8, 40)]:
            parts.append(rng.poisson(mean, (n, ng)))
            labels += [lab] * n
        adata = ad.AnnData(np.vstack(parts).astype(np.float64))
        adata.var_names = [f"g{i}" for i in range(ng)]
        adata.obs["perturbation"] = labels
        cfg = RunConfig(
            dataset="-", protocols=[], de_method="spy_moment", subsample=200, seed=0, min_cells=10, workers=1
        )
        ctx = Context(Dataset(adata, cfg), cfg)

        perts = ctx.perturbations
        results = [ctx.de(p, "control", "all_perturbed") for p in perts]

        # The moment capability was used for every comparison; the cells path never was.
        assert calls["from_moments"] == len(perts)
        assert calls["cells"] == 0
        # The `control` source's moments were computed once and reused across perturbations.
        assert list(ctx._store.mom.keys()) == ["control"]
        # And it agrees with computing t-test directly from the same cached moments.
        for p, de in zip(perts, results, strict=True):
            expected = ttest_from_moments(*ctx._moments("control", p), *ctx._moments("all_perturbed", p))
            assert np.allclose(de.statistic, expected.statistic, equal_nan=True)
    finally:
        DE_METHODS._items.pop("spy_moment", None)
