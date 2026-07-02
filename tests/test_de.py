"""Tests for the differential-expression backends, focused on the scanpy
``t-test_overestim_var`` variant added as a selectable DE method."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from scperteval.blocks.de import DE_METHODS, de_ttest_overestim


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
    assert np.allclose(de.score, sc_scores, atol=1e-5, rtol=1e-4)
    assert np.allclose(de.pvalue, sc_pvals, atol=1e-6, rtol=1e-4)
    assert de.pvalue_adj.shape == de.score.shape


def test_overestim_var_differs_from_plain_ttest():
    """Sanity: the overestimated-variance variant is more conservative (|t| no larger)."""
    rng = np.random.default_rng(1)
    Xt = rng.poisson(1.0, (30, 50)).astype(np.float64)
    Xr = rng.poisson(1.4, (120, 50)).astype(np.float64)
    over = de_ttest_overestim(Xt, Xr)
    plain = DE_METHODS["t-test"](Xt, Xr)
    assert np.all(np.abs(over.score) <= np.abs(plain.score) + 1e-9)
    assert not np.allclose(over.score, plain.score)


def test_overestim_var_registered_and_selectable():
    """It is a registered DE method, hence a selectable `run --de-method` backend
    (choices = DE_METHODS.names()) so new protocols can use it."""
    assert "t-test_overestim_var" in DE_METHODS
    assert "t-test_overestim_var" in DE_METHODS.names()


def test_overestim_var_runs_through_export_path():
    """Hits the real export path used by `scperteval de`: compute_de_export -> ctx.de
    -> DE_METHODS dispatch, on a tiny in-memory dataset."""
    from scperteval.context import Context
    from scperteval.dataset import Dataset
    from scperteval.runner import compute_de_export
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
    out = compute_de_export(ctx, ["t-test_overestim_var"])
    stat, padj = out["t-test_overestim_var"]
    assert stat.shape == (len(ctx.perturbations), ng)
    assert padj.shape == stat.shape
    assert np.isfinite(stat).all()
