"""Calibration mode: DRF/BDS over built-in positive/negative controls."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scperteval.blocks.spaces import SPACES, expr_space
from scperteval.calibrators import CALIBRATORS
from scperteval.cli import _concrete
from scperteval.context import Context
from scperteval.dataset import Dataset
from scperteval.protocols.table import PROTOCOLS
from scperteval.runner import run_protocol


def _run(name, calibrator, dataset_adata, cfg):
    ctx = Context(Dataset(dataset_adata, cfg), cfg)
    return run_protocol(_concrete(PROTOCOLS[name]), ctx, CALIBRATORS[calibrator])


def test_calibrators_registered():
    assert {"drf", "bds", "score", "paired_ci"} <= set(CALIBRATORS)
    # drf/bds/paired_ci need both controls; score needs only the prediction
    assert CALIBRATORS["drf"].requires == ("positive", "negative")
    assert CALIBRATORS["paired_ci"].requires == ("positive", "negative")
    assert CALIBRATORS["score"].requires == ("prediction",)


def test_drf_rows_have_control_columns(dataset_adata, cfg_factory):
    agg, rows, seconds = _run("pearson_ctrl", "drf", dataset_adata, cfg_factory())
    assert seconds >= 0.0
    assert len(rows) == 4  # one row per perturbation
    cols = set(rows[0])
    assert {"protocol", "perturbation", "raw_positive", "raw_negative", "drf"} <= cols
    assert {"mean", "median"} <= set(agg)


def test_drf_positive_for_real_signal(dataset_adata, cfg_factory):
    # the positive control (held-out replicate) should beat the uninformative baseline,
    # so mean DRF is clearly positive on a dataset with strong perturbation signal.
    for name in ("pearson_ctrl", "mse"):
        agg, _, _ = _run(name, "drf", dataset_adata, cfg_factory())
        assert agg["mean"] > 0.0, name


def test_bds_is_a_fraction(dataset_adata, cfg_factory):
    agg, rows, _ = _run("pearson_ctrl", "bds", dataset_adata, cfg_factory())
    assert 0.0 <= agg["bds"] <= 1.0
    assert all(r["bds"] in (0.0, 1.0) for r in rows)  # per-perturbation BDS is binary


def test_de_protocol_calibrates(dataset_adata, cfg_factory):
    # the de representation should produce a finite, well-defined auprc (distinct DE blocks)
    agg, _, _ = _run("de_auprc", "drf", dataset_adata, cfg_factory())
    assert np.isfinite(agg["mean"])


def test_paired_ci_reports_mean_and_bounds(dataset_adata, cfg_factory):
    agg, rows, _ = _run("pearson_ctrl", "paired_ci", dataset_adata, cfg_factory())
    assert {"mean", "ci_low", "ci_high"} <= set(agg)
    assert agg["ci_low"] <= agg["mean"] <= agg["ci_high"]
    assert {"raw_positive", "raw_negative", "paired_ci"} <= set(rows[0])


def test_paired_ci_positive_for_real_signal(dataset_adata, cfg_factory):
    # the positive control (held-out replicate) should beat the uninformative baseline,
    # so the paired gap's CI should sit above zero on a dataset with strong perturbation signal.
    agg, _, _ = _run("pearson_ctrl", "paired_ci", dataset_adata, cfg_factory())
    assert agg["ci_low"] > 0.0


# --- Ahlmann-Eltze et al. 2025: expr_space + l2 (const-ae's protocol, ported natively) ---


def test_expr_k_protocols_resolve_and_calibrate(dataset_adata, cfg_factory):
    for name in ("pearson_expr_k", "pearson_ctrl_expr_k", "l2_expr_k"):
        agg, rows, _ = _run(name, "drf", dataset_adata, cfg_factory())
        assert np.isfinite(agg["mean"]), name
        assert len(rows) == 4


def test_expr_space_picks_highest_control_expression_genes():
    rng = np.random.default_rng(0)
    ng = 10
    ctrl_means = np.array([1.0, 5.0, 2.0, 9.0, 0.5, 3.0, 8.0, 4.0, 7.0, 6.0])
    ctrl = rng.poisson(ctrl_means, size=(300, ng)).astype(np.float32)
    pert = rng.poisson(ctrl_means, size=(80, ng)).astype(np.float32)
    adata = ad.AnnData(np.vstack([ctrl, pert]))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = ["control"] * 300 + ["pertA"] * 80

    from tests.conftest import make_cfg

    cfg = make_cfg(min_cells=10)
    ctx = Context(Dataset(adata, cfg), cfg)
    name = expr_space(3)
    out = SPACES[name](ctx.ds.cells("pertA"), ctx, "pertA")

    assert out.shape[1] == 3
    # the 3 highest control-expressed genes are g3 (9.0), g6 (8.0), g8 (7.0), in that order
    expected_order = np.argsort(-ctrl_means)[:3]
    assert list(expected_order) == [3, 6, 8]
    for col, gene_idx in enumerate(expected_order):
        assert out[:, col].mean() == np.asarray(ctx.ds.cells("pertA"))[:, gene_idx].mean()
