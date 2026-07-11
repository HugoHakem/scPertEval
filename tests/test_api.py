"""The native Python API: prepare / calibrate / score / de (single protocol/method), and parity."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pandas as pd
import pytest

import scperteval as sp
from scperteval.api import DatasetDEResults, DEMethodName, EvalResult, Prepared
from scperteval.blocks.de import DE_METHODS
from scperteval.cli import main

FAST = dict(subsample=400, seed=0, min_cells=10, workers=1)


def test_public_surface():
    assert set(sp.__all__) == {
        "DatasetDEResults",
        "EvalResult",
        "Prepared",
        "__version__",
        "calibrate",
        "de",
        "prepare",
        "score",
    }
    # the trimmed / renamed names are gone
    for gone in ("differential_expression", "available_protocols", "DEResults", "DEMethodResult"):
        assert not hasattr(sp, gone)
    assert sp.__version__


def test_prepare_returns_reusable_handle(dataset_adata):
    prep = sp.prepare(dataset_adata, **FAST)
    assert isinstance(prep, Prepared)
    a = sp.calibrate(prep, "de_auprc")
    b = sp.calibrate(prep, "de_overlap_k")  # reuses the shared DE cache on the handle
    assert np.isfinite(a.aggregate["mean"]) and np.isfinite(b.aggregate["mean"])


def test_calibrate_single_protocol(dataset_adata):
    r = sp.calibrate(dataset_adata, "pearson_ctrl", **FAST)
    assert isinstance(r, EvalResult)
    assert set(r.aggregate) == {"mean", "median"}
    assert r.aggregate["mean"] > 0.0  # real signal beats the baseline
    assert len(r.per_perturbation) == 4  # one row per perturbation
    assert {"protocol", "perturbation", "raw_positive", "raw_negative", "drf"} <= set(r.per_perturbation.columns)


def test_calibrate_rejects_multiple_protocols(dataset_adata):
    with pytest.raises(ValueError, match="one protocol per call"):
        sp.calibrate(dataset_adata, "all", **FAST)


def test_calibrate_rejects_score_output(dataset_adata):
    with pytest.raises(ValueError, match="score"):
        sp.calibrate(dataset_adata, "mse", output="score", **FAST)


def test_calibrate_bds(dataset_adata):
    r = sp.calibrate(dataset_adata, "mse", output="bds", **FAST)
    assert set(r.aggregate) == {"bds"}
    assert 0.0 <= r.aggregate["bds"] <= 1.0


def test_score_single_protocol(dataset_adata, predictions_factory):
    pred = predictions_factory(dataset_adata, kind="degraded")
    r = sp.score(dataset_adata, pred, "pearson", **FAST)
    assert isinstance(r, EvalResult)
    assert "score" in r.per_perturbation.columns
    assert np.isfinite(r.aggregate["mean"])


def test_de_single_method(dataset_adata):
    d = sp.de(dataset_adata, "t-test", **FAST)
    assert isinstance(d, DatasetDEResults)
    assert isinstance(d.statistic, pd.DataFrame) and d.statistic.shape == (4, 60)
    assert list(d.statistic.columns) == [f"g{i}" for i in range(60)]
    assert np.isfinite(d.statistic.to_numpy()).all()
    stat, padj = d  # NamedTuple unpacks
    assert stat.shape == padj.shape == (4, 60)


def test_de_rejects_unknown_method(dataset_adata):
    with pytest.raises(ValueError, match="unknown DE method"):
        sp.de(dataset_adata, "nope", **FAST)


def test_de_method_literal_matches_registry():
    # the public Literal must stay in sync with the registry's built-in methods
    assert set(get_args(DEMethodName)) == set(DE_METHODS.names())


def test_in_memory_anndata_not_mutated(dataset_adata):
    before = dataset_adata.X.copy()
    sp.calibrate(dataset_adata, "de_auprc", **FAST)
    sp.de(dataset_adata, "t-test", **FAST)
    after = dataset_adata.X
    assert np.array_equal(np.asarray(before.todense() if hasattr(before, "todense") else before), np.asarray(after))


def test_prepare_matches_no_prepare(dataset_adata):
    fresh = sp.calibrate(dataset_adata, "mse", **FAST)
    prep = sp.calibrate(sp.prepare(dataset_adata, **FAST), "mse")
    assert fresh.aggregate == pytest.approx(prep.aggregate)


def test_out_dir_writes_files(dataset_adata, tmp_path):
    sp.calibrate(dataset_adata, "mse", out_dir=str(tmp_path), **FAST)
    assert len(list(tmp_path.glob("*__drf.csv"))) == 1
    sp.de(dataset_adata, "t-test", out_dir=str(tmp_path), **FAST)
    h5 = list(tmp_path.glob("*__de.h5"))
    assert len(h5) == 1
    import h5py

    with h5py.File(h5[0]) as f:
        assert "t-test" in f
        assert f["t-test"]["statistic"].shape == (4, 60)
        assert f["genes"].shape == (60,)


def test_api_matches_cli(dataset_path, tmp_path):
    """The API's per-perturbation table equals the CLI's CSV for the same single protocol."""
    cli_dir = tmp_path / "cli"
    main(
        [
            "calibrate",
            dataset_path,
            "-p",
            "pearson_ctrl",
            "--subsample",
            "400",
            "--seed",
            "0",
            "--min-cells",
            "10",
            "--workers",
            "1",
            "--out-dir",
            str(cli_dir),
            "--quiet",
        ]
    )
    cli_df = pd.read_csv(next(cli_dir.glob("*__drf.csv")))
    api = sp.calibrate(dataset_path, "pearson_ctrl", **FAST)
    pd.testing.assert_frame_equal(
        api.per_perturbation.reset_index(drop=True), cli_df.reset_index(drop=True), check_dtype=False
    )
