"""Feature spaces: the catalog rules, instance registration, and composition.

The rules are plain functions, so most tests call them directly; the registry tests cover
turning a catalog entry into a registered instance.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import anndata as ad
import numpy as np
import pytest
from conftest import make_cfg

from scperteval.blocks.spaces import OPS, SPACES, combine_subsets
from scperteval.blocks.spaces.catalog import full, heg, hvg, miller_panel, perturbed_genes
from scperteval.calibrators import CALIBRATORS
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.predictions import PredictionSet
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import run_protocol


def test_heg_picks_highest_control_expression_genes():
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
    # the 3 highest control-expressed genes are g3 (9.0), g6 (8.0), g8 (7.0), in that order
    assert list(np.argsort(-ctrl_means)[:3]) == [3, 6, 8]
    assert heg(ctx, "pertA", 3).tolist() == [3, 6, 8]


def test_hvg_picks_highest_dispersion_genes():
    # scanpy's "seurat" flavor bins genes by mean expression (20 bins by default) before
    # z-scoring dispersion within each bin, so this needs enough genes spread across a
    # realistic mean range for the binning to be meaningful -- a handful of genes gives
    # near-empty, degenerate bins.
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
    # real pipeline data (see docs/user-guide/datasets.md) -- raw counts would be meaningless.
    adata = ad.AnnData(np.log1p(np.vstack([base, pert])))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * n_ctrl + ["pertA"] * 80

    cfg = make_cfg(min_cells=10)
    ctx = Context(Dataset(adata, cfg), cfg)
    assert set(hvg(ctx, "pertA", n_top).tolist()) == overdispersed


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


def _ctx_10_genes():
    """10 genes; heg(3) picks {3, 6, 8}; perturbations target {0, 3, 5} (g3 overlaps)."""
    rng = np.random.default_rng(0)
    ng = 10
    ctrl_means = np.array([1.0, 5.0, 2.0, 9.0, 0.5, 3.0, 8.0, 4.0, 7.0, 6.0])
    ctrl = rng.poisson(ctrl_means, size=(300, ng)).astype(np.float32)
    a = rng.poisson(ctrl_means, size=(40, ng)).astype(np.float32)
    b = rng.poisson(ctrl_means, size=(40, ng)).astype(np.float32)
    adata = ad.AnnData(np.log1p(np.vstack([ctrl, a, b])))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 300 + ["g3+g0"] * 40 + ["g5"] * 40

    cfg = make_cfg(min_cells=10)
    return Context(Dataset(adata, cfg), cfg)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (OPS.union, [0, 3, 5, 6, 8]),  # {3, 6, 8} | {0, 3, 5}
        (OPS.intersection, [3]),  # {3, 6, 8} & {0, 3, 5}
        (OPS.difference, [6, 8]),  # {3, 6, 8} - {0, 3, 5}
    ],
)
def test_combine_subsets_applies_the_set_operation(op, expected):
    ctx = _ctx_10_genes()
    got = combine_subsets(ctx, op, heg(ctx, "g5", 3), perturbed_genes(ctx, "g5"))
    assert got.tolist() == expected


def test_combine_subsets_canonicalises_a_slice_so_full_composes_as_a_complement():
    """full returns a slice; combining must still yield integer positions."""
    ctx = _ctx_10_genes()
    complement = combine_subsets(ctx, OPS.difference, full(ctx, "g5"), heg(ctx, "g5", 3))
    assert complement.tolist() == [0, 1, 2, 4, 5, 7, 9]  # 10 genes minus {3, 6, 8}


def test_combine_subsets_nests_to_any_depth():
    """A composed selection is just a selection, so it folds into another one."""
    ctx = _ctx_10_genes()
    inner = combine_subsets(ctx, OPS.difference, heg(ctx, "g5", 3), perturbed_genes(ctx, "g5"))
    outer = combine_subsets(ctx, OPS.union, inner, hvg(ctx, "g5", 4))
    assert set(outer.tolist()) == {6, 8} | set(hvg(ctx, "g5", 4).tolist())

    # the other grouping is a different set, and each is written explicitly
    other = combine_subsets(
        ctx,
        np.setdiff1d,
        heg(ctx, "g5", 3),
        combine_subsets(ctx, OPS.union, perturbed_genes(ctx, "g5"), hvg(ctx, "g5", 4)),
    )
    assert other.tolist() != outer.tolist()


def test_miller_panel_unions_hvg_with_the_targeted_genes():
    ctx = _ctx_10_genes()  # 10 genes, so hvg(8192) degrades to all of them
    assert set(miller_panel(ctx, "g5").tolist()) == set(hvg(ctx, "g5", 8192).tolist()) | set(
        perturbed_genes(ctx, "g5").tolist()
    )


def test_catalog_lists_definitions_and_says_what_each_takes():
    labels = {s.name: s.label for s in SPACES.catalog()}
    assert labels["heg"] == "heg_<k>"  # parameter name read from the rule's signature
    assert labels["degs"] == "degs_<padj>"
    assert labels["full"] == "full"  # trailing default => takes no parameter
    assert labels["miller_panel"] == "miller_panel"


def test_instance_registers_on_demand_and_guards_its_value():
    assert SPACES.instance("heg") == "heg_1000"  # the catalog default
    assert SPACES.instance("heg", 250) == "heg_250"
    assert SPACES.instance("perturbed_genes") == "perturbed_genes"
    assert SPACES.meta("heg_250")["global_space"] is True
    assert SPACES.meta(SPACES.instance("top", 5))["global_space"] is False  # per_pert
    with pytest.raises(KeyError, match="unknown space"):
        SPACES.instance("nope")
    with pytest.raises(TypeError, match="takes no parameter"):
        SPACES.instance("full", 5)
    # Distinct values that format to the same name must not silently share one registration.
    SPACES.instance("degs", 0.05)
    with pytest.raises(ValueError, match="already registered with value"):
        SPACES.instance("degs", 0.05000000001)


def test_a_rule_taking_a_parameter_must_declare_a_default():
    with pytest.raises(TypeError, match="needs a default"):

        @SPACES.subset("bad", description="no default for k")
        def bad(ctx, pert, k):
            return slice(None)


def test_full_is_a_view_not_a_copy():
    ctx = _ctx_10_genes()
    assert full(ctx, "g5") == slice(None)
    cells = ctx.ds.cells("g5")
    assert SPACES["full"](cells, ctx, "g5").shape == cells.shape


def test_space_runs_end_to_end_through_the_runner(cfg_factory):
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
    adata = ad.AnnData(np.log1p(np.vstack(parts)))
    adata.var_names = genes
    adata.obs["perturbation"] = labels

    panel = SPACES.instance("miller_panel")
    proto = replace(PROTOCOLS["energy_distance_top_k"], name="ed_panel", space=panel, param=None)

    cfg = cfg_factory(truth="gt_all_cells", calibrator="score")
    ds = Dataset(adata, cfg)
    ctx = Context(ds, cfg)
    sub = adata[np.asarray(adata.obs["perturbation"]).astype(str) != "control"].copy()
    pred = ad.AnnData(np.asarray(sub.X, dtype=np.float32), obs=sub.obs.copy())
    pred.var_names = genes
    ctx.predictions = PredictionSet(pred, ds, cfg)

    ctx.warm([proto])
    assert panel in ctx._store.reference_projections  # global space: projected once, shared
    agg, rows, _ = run_protocol(proto, ctx, CALIBRATORS["score"])
    assert len(rows) == 3
    assert np.isfinite(agg["mean"])


def test_importing_the_package_defines_the_catalog():
    """The catalog only exists because importing the package imports the rules that declare it.

    Checked in a fresh interpreter: this module imports ``catalog`` directly, which would define
    the spaces for the whole pytest session and hide a missing import in ``__init__``.
    """
    probe = (
        "from scperteval.blocks.spaces import SPACES;"
        "names = [s.name for s in SPACES.catalog()];"
        "assert 'heg' in names, names;"
        "assert SPACES['full'] is not None;"
        "print(len(names))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) >= 8
