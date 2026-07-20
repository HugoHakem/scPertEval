"""Gene-subset feature spaces: heg_space, hvg_space, perturbed_space, and combine_space."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from conftest import make_cfg

from scperteval.blocks.spaces import SPACES, combine_space, heg_space, hvg_space, perturbed_space
from scperteval.context import Context
from scperteval.dataset import Dataset


def test_heg_space_picks_highest_control_expression_genes():
    rng = np.random.default_rng(0)
    ng = 10
    ctrl_means = np.array([1.0, 5.0, 2.0, 9.0, 0.5, 3.0, 8.0, 4.0, 7.0, 6.0])
    ctrl = rng.poisson(ctrl_means, size=(300, ng)).astype(np.float32)
    pert = rng.poisson(ctrl_means, size=(80, ng)).astype(np.float32)
    adata = ad.AnnData(np.vstack([ctrl, pert]))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 300 + ["pertA"] * 80

    cfg = make_cfg(min_cells=10)
    ctx = Context(Dataset(adata, cfg), cfg)
    name = heg_space(3)
    out = SPACES[name](ctx.ds.cells("pertA"), ctx, "pertA")

    assert out.shape[1] == 3
    # the 3 highest control-expressed genes are g3 (9.0), g6 (8.0), g8 (7.0), in that order
    expected_order = np.argsort(-ctrl_means)[:3]
    assert list(expected_order) == [3, 6, 8]
    for col, gene_idx in enumerate(expected_order):
        assert out[:, col].mean() == np.asarray(ctx.ds.cells("pertA"))[:, gene_idx].mean()


def test_hvg_space_picks_highest_dispersion_genes():
    # scanpy's "seurat" flavor bins genes by mean expression (20 bins by default) before
    # z-scoring dispersion within each bin, so this needs enough genes spread across a
    # realistic mean range for the binning to be meaningful -- a handful of genes (as in
    # test_heg_space_picks_highest_control_expression_genes) gives near-empty, degenerate bins.
    rng = np.random.default_rng(0)
    ng, n_ctrl, n_top = 200, 800, 5
    base_means = rng.uniform(0.5, 20.0, size=ng)
    base = rng.poisson(base_means, size=(n_ctrl, ng)).astype(np.float32)
    overdispersed = set(rng.choice(ng, size=n_top, replace=False).tolist())
    # mean-preserving variance inflation: a 0.1x/10x scale mixture around the same base mean
    # (p_lo chosen so p_lo*0.1 + (1 - p_lo)*10 == 1, i.e. E[scale] == 1).
    lo, hi = 0.1, 10.0
    p_lo = (hi - 1.0) / (hi - lo)
    for g in overdispersed:
        scale = rng.choice([lo, hi], size=n_ctrl, p=[p_lo, 1 - p_lo])
        base[:, g] = rng.poisson(base_means[g] * scale).astype(np.float32)
    pert = rng.poisson(base_means, size=(80, ng)).astype(np.float32)
    # scanpy's "seurat" flavor expects log1p'd input (it internally un-logs via expm1), matching
    # real pipeline data (see docs/user-guide/datasets.md) -- raw counts would give a meaningless
    # ranking.
    adata = ad.AnnData(np.log1p(np.vstack([base, pert])))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * n_ctrl + ["pertA"] * 80

    cfg = make_cfg(min_cells=10)
    ctx = Context(Dataset(adata, cfg), cfg)
    name = hvg_space(n_top)
    out = SPACES[name](ctx.ds.cells("pertA"), ctx, "pertA")

    dispersion = ctx.control_hvg_dispersion()
    expected_order = np.argsort(-dispersion)[:n_top]
    assert out.shape[1] == n_top
    assert set(expected_order.tolist()) == overdispersed
    for col, gene_idx in enumerate(expected_order):
        assert out[:, col].mean() == np.asarray(ctx.ds.cells("pertA"))[:, gene_idx].mean()


def test_perturbed_gene_indices_matches_var_names_and_skips_non_gene_labels():
    rng = np.random.default_rng(0)
    ng = 6
    adata = ad.AnnData(rng.poisson(1.0, size=(40, ng)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(ng)]
    # g1: single-gene perturbation; g2+g4: combo (+-delimited, see docs/user-guide/datasets.md);
    # drugX: a non-gene treatment label that doesn't match any var_names entry (skipped).
    adata.obs["perturbation"] = ["control"] * 10 + ["g1"] * 10 + ["g2+g4"] * 10 + ["drugX"] * 10

    cfg = make_cfg(min_cells=5)
    ds = Dataset(adata, cfg)
    assert sorted(ds.perturbed_gene_indices().tolist()) == [1, 2, 4]

    ctx = Context(ds, cfg)
    assert sorted(ctx.perturbed_genes().tolist()) == [1, 2, 4]


def _heg_and_perturbed_dataset():
    """10 genes; heg_space(3) picks {3, 6, 8}; perturbations target {0, 3, 5} (g3 overlaps)."""
    rng = np.random.default_rng(0)
    ng = 10
    ctrl_means = np.array([1.0, 5.0, 2.0, 9.0, 0.5, 3.0, 8.0, 4.0, 7.0, 6.0])
    ctrl = rng.poisson(ctrl_means, size=(300, ng)).astype(np.float32)
    pertA = rng.poisson(ctrl_means, size=(40, ng)).astype(np.float32)
    pertB = rng.poisson(ctrl_means, size=(40, ng)).astype(np.float32)
    adata = ad.AnnData(np.vstack([ctrl, pertA, pertB]))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 300 + ["g3+g0"] * 40 + ["g5"] * 40

    cfg = make_cfg(min_cells=10)
    return Context(Dataset(adata, cfg), cfg)


def test_combine_space_union_of_heg_and_perturbed():
    ctx = _heg_and_perturbed_dataset()
    name = combine_space(heg_space(3), perturbed_space())
    assert name == "heg_3+perturbed"
    out = SPACES[name](ctx.ds.cells("g5"), ctx, "g5")
    assert out.shape[1] == 5  # {3, 6, 8} | {0, 3, 5} == {0, 3, 5, 6, 8}


def test_combine_space_intersect_of_heg_and_perturbed():
    ctx = _heg_and_perturbed_dataset()
    name = combine_space(heg_space(3), perturbed_space(), op="intersect")
    assert name == "heg_3&perturbed"
    out = SPACES[name](ctx.ds.cells("g5"), ctx, "g5")
    assert out.shape[1] == 1  # {3, 6, 8} & {0, 3, 5} == {3}


def test_combine_space_diff_of_heg_and_perturbed():
    ctx = _heg_and_perturbed_dataset()
    name = combine_space(heg_space(3), perturbed_space(), op="diff")
    assert name == "heg_3-perturbed"
    out = SPACES[name](ctx.ds.cells("g5"), ctx, "g5")
    assert out.shape[1] == 2  # {3, 6, 8} - {0, 3, 5} == {6, 8}


def test_combine_space_rejects_bad_input():
    with pytest.raises(ValueError, match="at least two"):
        combine_space(heg_space(3))
    with pytest.raises(ValueError, match="unknown op"):
        combine_space(heg_space(3), perturbed_space(), op="xor")
    with pytest.raises(ValueError, match="indices metadata"):
        combine_space("full", perturbed_space())  # "full" isn't a gene-subset space
