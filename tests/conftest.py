"""Shared fixtures and tiny in-memory dataset builders for the test suite."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scperteval.types import RunConfig

# Each perturbation gets a strong, *distinct* block of DE genes, so ground-truth DEGs
# form a proper subset (de_auprc/auroc are well-defined) and the perturbation signal is
# unambiguous for the calibration controls.
_DE_GENES = {
    "pertA": range(0, 6),
    "pertB": range(15, 21),
    "pertC": range(30, 36),
    "pertD": range(45, 51),
}


def make_dataset(seed: int = 0, ng: int = 60, n_ctrl: int = 150, n_pert: int = 120) -> ad.AnnData:
    """A tiny log-normalised-looking dataset: control + 4 perturbations with distinct DE blocks."""
    rng = np.random.default_rng(seed)
    parts = [rng.poisson(1.0, (n_ctrl, ng)).astype(np.float32)]
    labels = ["control"] * n_ctrl
    for lab, genes in _DE_GENES.items():
        x = rng.poisson(1.0, (n_pert, ng)).astype(np.float32)
        x[:, list(genes)] += 6.0
        parts.append(x)
        labels += [lab] * n_pert
    adata = ad.AnnData(np.vstack(parts))
    adata.var_names = [f"g{i}" for i in range(ng)]
    adata.obs["perturbation"] = labels
    return adata


def make_predictions(
    dataset: ad.AnnData, kind: str = "perfect", shuffle_genes: bool = False, seed: int = 1
) -> ad.AnnData:
    """Build a prediction AnnData from the dataset's perturbed cells.

    ``perfect`` is an exact replica (should score optimally); ``degraded`` shrinks each cell
    toward the control mean plus noise (a worse prediction). ``shuffle_genes`` permutes the
    gene columns to exercise the name-based alignment.
    """
    rng = np.random.default_rng(seed)
    pert = np.asarray(dataset.obs["perturbation"]).astype(str)
    mask = pert != "control"
    sub = dataset[mask].copy()
    x = np.asarray(sub.X, dtype=np.float32)
    if kind == "degraded":
        ctrl_mean = np.asarray(dataset.X[pert == "control"]).mean(0)
        x = np.clip(0.4 * x + 0.6 * ctrl_mean + rng.normal(0, 0.2, x.shape), 0, None).astype(np.float32)
    elif kind != "perfect":
        raise ValueError(f"unknown prediction kind {kind!r}")
    pred = ad.AnnData(x, obs=sub.obs.copy())
    pred.var_names = list(dataset.var_names)
    if shuffle_genes:
        pred = pred[:, rng.permutation(pred.n_vars)].copy()
    return pred


def make_cfg(**kw) -> RunConfig:
    """A RunConfig with small, fast, deterministic defaults for tests."""
    base = dict(dataset="-", protocols=[], de_method="t-test", subsample=400, seed=0, min_cells=10, workers=1)
    base.update(kw)
    return RunConfig(**base)


@pytest.fixture
def dataset_adata() -> ad.AnnData:
    return make_dataset()


@pytest.fixture
def dataset_path(tmp_path, dataset_adata) -> str:
    path = tmp_path / "dataset.h5ad"
    dataset_adata.write_h5ad(path)
    return str(path)


@pytest.fixture
def cfg_factory():
    return make_cfg


@pytest.fixture
def predictions_factory():
    return make_predictions
