"""Gene-subset feature spaces: heg, hvg, perturbed_genes, and combine_space.

Also covers the ``indices`` registry capability those spaces are built on: it is what makes a
space composable, so the composition tests double as its contract tests.
"""

from __future__ import annotations

from dataclasses import replace

import anndata as ad
import numpy as np
import pytest
from conftest import make_cfg

from scperteval.blocks.spaces import DEGS, FULL, HEG, HVG, MILLER_PANEL, PERTURBED_GENES, SPACES, TOP
from scperteval.calibrators import CALIBRATORS
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.predictions import PredictionSet
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import run_protocol


def _rule(name, ctx, pert):
    """The gene indices ``name`` selects — its registered ``indices`` rule."""
    return SPACES.meta(name)["select"](ctx, pert)


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
    name = SPACES.instance(HEG, 3)
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
    name = SPACES.instance(HVG, n_top)
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
    idx = ds.perturbed_gene_indices()
    assert sorted(idx.tolist()) == [1, 2, 4]
    assert np.issubdtype(idx.dtype, np.integer)  # float indices would fail to index X

    ctx = Context(ds, cfg)
    assert sorted(ctx.perturbed_gene_indices().tolist()) == [1, 2, 4]


def test_perturbed_gene_indices_raises_when_no_label_is_a_gene():
    """A drug/compound dataset has no targeted genes -- fail loudly, don't score an empty panel."""
    rng = np.random.default_rng(0)
    ng = 6
    adata = ad.AnnData(rng.poisson(1.0, size=(30, ng)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 10 + ["drugX"] * 10 + ["drugY"] * 10

    cfg = make_cfg(min_cells=5)
    with pytest.raises(ValueError, match="no perturbation label matches a gene"):
        Dataset(adata, cfg).perturbed_gene_indices()


def _heg_and_perturbed_dataset():
    """10 genes; SPACES.instance(HEG, 3) picks {3, 6, 8}; perturbations target {0, 3, 5} (g3 overlaps)."""
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


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("union", [0, 3, 5, 6, 8]),  # {3, 6, 8} | {0, 3, 5}
        ("intersect", [3]),  # {3, 6, 8} & {0, 3, 5}
        ("diff", [6, 8]),  # {3, 6, 8} - {0, 3, 5}
    ],
)
def test_combine_space_applies_the_set_operation(op, expected):
    ctx = _heg_and_perturbed_dataset()
    name = SPACES.combine(f"heg3_{op}_pert", SPACES.instance(HEG, 3), SPACES.instance(PERTURBED_GENES), op=op)
    assert sorted(_rule(name, ctx, "g5").tolist()) == expected
    assert SPACES[name](ctx.ds.cells("g5"), ctx, "g5").shape[1] == len(expected)


def test_combine_space_composes_composites():
    """A composite carries ``indices`` too, so it can be composed again -- and nesting is explicit."""
    ctx = _heg_and_perturbed_dataset()
    heg, pert, hvg = SPACES.instance(HEG, 3), SPACES.instance(PERTURBED_GENES), SPACES.instance(HVG, 4)
    inner = SPACES.combine("heg3_minus_pert", heg, pert, op="diff")
    assert sorted(_rule(inner, ctx, "g5").tolist()) == [6, 8]

    # (heg_3 \ perturbed_genes) U hvg_4 and heg_3 \ (perturbed_genes U hvg_4) are different sets,
    # and each caller names its own panel, so neither can silently shadow the other.
    left = SPACES.combine("left_grouping", inner, hvg, op="union")
    outer = SPACES.combine("pert_or_hvg", pert, hvg, op="union")
    right = SPACES.combine("right_grouping", heg, outer, op="diff")

    hvg_idx = set(_rule(hvg, ctx, "g5").tolist())
    assert set(_rule(left, ctx, "g5").tolist()) == {6, 8} | hvg_idx
    assert set(_rule(right, ctx, "g5").tolist()) == {3, 6, 8} - ({0, 3, 5} | hvg_idx)
    assert _rule(left, ctx, "g5").tolist() != _rule(right, ctx, "g5").tolist()


def test_combine_space_folds_three_spaces_left_to_right():
    ctx = _heg_and_perturbed_dataset()
    heg, pert, hvg = SPACES.instance(HEG, 3), SPACES.instance(PERTURBED_GENES), SPACES.instance(HVG, 4)
    name = SPACES.combine("three_way_diff", heg, pert, hvg, op="diff")
    expected = set(_rule(heg, ctx, "g5").tolist()) - set(_rule(pert, ctx, "g5").tolist())
    expected -= set(_rule(hvg, ctx, "g5").tolist())
    assert set(_rule(name, ctx, "g5").tolist()) == expected


def test_combine_space_is_global_only_when_every_part_is():
    """global_space gates the shared reference projection, so one per-perturbation part poisons it."""
    assert SPACES.meta(SPACES.combine("both_global", SPACES.instance(HEG, 3), SPACES.instance(PERTURBED_GENES)))[
        "global_space"
    ]
    mixed = SPACES.combine("one_per_pert", SPACES.instance(TOP, 50), SPACES.instance(HEG, 3))
    assert not SPACES.meta(mixed)["global_space"]


def test_combine_space_rejects_bad_input():
    with pytest.raises(ValueError, match="at least two"):
        SPACES.combine("too_few", SPACES.instance(HEG, 3))
    with pytest.raises(ValueError, match="unknown op"):
        SPACES.combine("bad_op", SPACES.instance(HEG, 3), SPACES.instance(PERTURBED_GENES), op="xor")
    with pytest.raises(ValueError, match="no genes to combine"):
        SPACES.combine("with_pca", "pca_50", SPACES.instance(PERTURBED_GENES))  # pca isn't a gene subset
    with pytest.raises(KeyError, match="unknown space"):
        SPACES.combine("with_typo", "heg_99999", SPACES.instance(PERTURBED_GENES))


def test_register_defaults_the_value_and_guards_it():
    assert SPACES.instance(HEG) == "heg_1000"  # the row's default value
    assert SPACES.instance(HEG, 250) == "heg_250"
    assert SPACES.instance(PERTURBED_GENES) == "perturbed_genes"
    with pytest.raises(TypeError, match="takes no parameter"):
        SPACES.instance(FULL, 5)
    # Distinct values that format to the same name must not silently share one registration.
    SPACES.instance(DEGS, 0.05)
    with pytest.raises(ValueError, match="already registered with value"):
        SPACES.instance(DEGS, 0.05000000001)


def test_full_selects_every_gene_without_copying():
    """The identity space returns a slice, so applying it is a view rather than a gather."""
    ctx = _heg_and_perturbed_dataset()
    assert _rule("full", ctx, "g5") == slice(None)
    cells = ctx.ds.cells("g5")
    out = SPACES["full"](cells, ctx, "g5")
    assert out.shape == cells.shape


def test_full_composes_as_a_complement():
    """full - heg_3 is the complement of heg_3, which is why full is a subset and not a transform."""
    ctx = _heg_and_perturbed_dataset()
    name = SPACES.combine("not_heg3", SPACES.instance(FULL), SPACES.instance(HEG, 3), op="diff")
    assert sorted(_rule(name, ctx, "g5").tolist()) == [0, 1, 2, 4, 5, 7, 9]  # 10 genes minus {3, 6, 8}


def test_composed_space_runs_end_to_end_through_the_runner(cfg_factory):
    """The whole path: warm() -> cached reference projection -> per-perturbation scoring."""
    rng = np.random.default_rng(0)
    ng, n_ctrl, n_pert = 80, 300, 120
    genes = [f"g{i}" for i in range(ng)]
    parts = [rng.poisson(1.0, (n_ctrl, ng)).astype(np.float32)]
    labels = ["control"] * n_ctrl
    for lab, block in {"g0": range(0, 6), "g1+g2": range(15, 21), "g3": range(30, 36)}.items():
        x = rng.poisson(1.0, (n_pert, ng)).astype(np.float32)
        x[:, list(block)] += 6.0
        parts.append(x)
        labels += [lab] * n_pert
    adata = ad.AnnData(np.log1p(np.vstack([parts[0], *parts[1:]])))
    adata.var_names = genes
    adata.obs["perturbation"] = labels

    panel = SPACES.combine("panel", SPACES.instance(HVG, 10), SPACES.instance(PERTURBED_GENES))
    proto = replace(PROTOCOLS["energy_distance_top_k"], name="ed_panel", space=panel, param=None)

    cfg = cfg_factory(truth="gt_all_cells", calibrator="score")
    ds = Dataset(adata, cfg)
    ctx = Context(ds, cfg)
    sub = adata[np.asarray(adata.obs["perturbation"]).astype(str) != "control"].copy()
    pred = ad.AnnData(np.asarray(sub.X, dtype=np.float32), obs=sub.obs.copy())
    pred.var_names = genes
    ctx.predictions = PredictionSet(pred, ds, cfg)

    ctx.warm([proto])
    assert panel in ctx._store.reference_projections  # global composite: projected once, shared
    agg, rows, _ = run_protocol(proto, ctx, CALIBRATORS["score"])
    assert len(rows) == 3
    assert np.isfinite(agg["mean"])


def test_miller_panel_is_registered_as_the_hvg_union_perturbed_genes_panel():
    """The panel the composition machinery exists for is built at import, not left to the user."""
    assert MILLER_PANEL in SPACES
    assert SPACES.meta(MILLER_PANEL)["description"] == "union of hvg_8192, perturbed_genes"

    ctx = _heg_and_perturbed_dataset()  # 10 genes, so hvg_8192 degrades to all of them
    panel = set(_rule(MILLER_PANEL, ctx, "g5").tolist())
    assert panel == set(_rule(SPACES.instance(HVG, 8192), ctx, "g5").tolist()) | set(
        _rule(SPACES.instance(PERTURBED_GENES), ctx, "g5").tolist()
    )
