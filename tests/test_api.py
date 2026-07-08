"""The native Python API: calibrate / score / differential_expression, and CLI parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scperteval as sp
from scperteval.api import DEResults, EvalResult
from scperteval.cli import main


def test_public_surface_reexported():
    for name in ("calibrate", "score", "differential_expression", "de", "available_protocols"):
        assert hasattr(sp, name)
    assert sp.de is sp.differential_expression
    assert sp.__version__  # resolvable from installed metadata


def test_calibrate_returns_eval_result(dataset_adata):
    r = sp.calibrate(dataset_adata, ["pearson_ctrl", "mse"], subsample=400, seed=0, min_cells=10, workers=1)
    assert isinstance(r, EvalResult)
    # summary: one row per protocol, drf aggregate columns
    assert list(r.summary.index) == ["pearson_ctrl", "mse"]
    assert {"mean", "median"} <= set(r.summary.columns)
    # per_perturbation: one row per (protocol, perturbation) + provenance columns
    assert len(r.per_perturbation) == 2 * 4
    assert {"protocol", "perturbation", "raw_positive", "raw_negative", "drf", "dataset"} <= set(
        r.per_perturbation.columns
    )
    assert r.summary.loc["pearson_ctrl", "mean"] > 0.0  # real signal beats the baseline


def test_calibrate_accepts_path_and_string_spec(dataset_path):
    r = sp.calibrate(dataset_path, "pearson_ctrl,mse", subsample=400, seed=0, min_cells=10, workers=1)
    assert list(r.summary.index) == ["pearson_ctrl", "mse"]
    # the display name is derived from the path stem
    assert (r.per_perturbation["dataset"] == "dataset").all()


def test_calibrate_bds_output(dataset_adata):
    r = sp.calibrate(dataset_adata, "mse", output="bds", subsample=400, seed=0, min_cells=10, workers=1)
    assert list(r.summary.columns) == ["bds"]
    assert 0.0 <= r.summary.loc["mse", "bds"] <= 1.0


def test_calibrate_rejects_score_output(dataset_adata):
    with pytest.raises(ValueError, match="score"):
        sp.calibrate(dataset_adata, "mse", output="score")


def test_score_returns_eval_result(dataset_adata, predictions_factory):
    pred = predictions_factory(dataset_adata, kind="degraded")
    r = sp.score(dataset_adata, pred, ["pearson", "mse"], subsample=400, seed=0, min_cells=10, workers=1)
    assert isinstance(r, EvalResult)
    assert "score" in r.per_perturbation.columns  # the calibrator column is the raw score
    assert np.isfinite(r.summary.loc["pearson", "mean"])


def test_differential_expression(dataset_adata):
    d = sp.differential_expression(dataset_adata, ["t-test", "MWU"], subsample=400, seed=0, min_cells=10, workers=1)
    assert isinstance(d, DEResults)
    assert set(d) == {"t-test", "MWU"}
    stat = d["t-test"].statistic
    assert isinstance(stat, pd.DataFrame)
    assert stat.shape == (4, 60)  # 4 perturbations x 60 genes
    assert list(stat.columns) == [f"g{i}" for i in range(60)]
    assert np.isfinite(stat.to_numpy()).all()


def test_out_dir_writes_files(dataset_adata, tmp_path):
    sp.calibrate(dataset_adata, "mse", subsample=400, seed=0, min_cells=10, workers=1, out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*__drf.csv"))) == 1
    sp.differential_expression(
        dataset_adata, "t-test", subsample=400, seed=0, min_cells=10, workers=1, out_dir=str(tmp_path)
    )
    assert len(list(tmp_path.glob("*__de.h5"))) == 1


def test_to_csv_explicit_path(dataset_adata, tmp_path):
    r = sp.calibrate(dataset_adata, "mse", subsample=400, seed=0, min_cells=10, workers=1)
    out = tmp_path / "sub" / "rows.csv"
    r.to_csv(out)
    written = pd.read_csv(out)
    pd.testing.assert_frame_equal(written, r.per_perturbation)


def test_api_matches_cli(dataset_path, tmp_path):
    """The API's per-perturbation table equals the CLI's CSV for identical inputs."""
    cli_dir = tmp_path / "cli"
    main(
        [
            "calibrate",
            dataset_path,
            "-p",
            "pearson_ctrl,mse",
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
    cli_csv = next(cli_dir.glob("*__drf.csv"))
    cli_df = pd.read_csv(cli_csv)

    r = sp.calibrate(dataset_path, ["pearson_ctrl", "mse"], subsample=400, seed=0, min_cells=10, workers=1)
    # same columns, same values (the CLI derives the same 'dataset' stem from the path)
    pd.testing.assert_frame_equal(
        r.per_perturbation.reset_index(drop=True), cli_df.reset_index(drop=True), check_dtype=False
    )
