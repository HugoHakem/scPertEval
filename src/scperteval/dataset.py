"""Thin wrapper over a preprocessed AnnData with a perturbation column."""

from __future__ import annotations

import zlib

import anndata as ad
import numpy as np
import scipy.sparse as sp

from .types import RunConfig


def _seed(seed: int, *tags) -> np.random.Generator:
    key = (seed, *(zlib.crc32(str(t).encode()) for t in tags))
    return np.random.default_rng(np.array(key, dtype=np.uint32))


class Dataset:
    """Loads the AnnData, splits each perturbation into halves, and serves cell sets."""

    def __init__(self, adata, cfg: RunConfig):
        self.adata = adata
        self.cfg = cfg
        self.var_names = np.asarray(adata.var_names)
        self.pert = np.asarray(adata.obs[cfg.perturbation_key]).astype(str)
        self.control_idx = np.where(self.pert == cfg.control_label)[0]
        self._index()

    @classmethod
    def load(cls, path: str, cfg: RunConfig) -> Dataset:
        """Load a dataset from a preprocessed ``.h5ad`` path."""
        return cls(ad.read_h5ad(path), cfg)

    def _index(self):
        rng = np.random.default_rng(self.cfg.seed)
        self.halves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.perturbations: list[str] = []
        means = []
        for p in np.unique(self.pert[self.pert != self.cfg.control_label]):
            idx = np.where(self.pert == p)[0]
            if len(idx) < self.cfg.min_cells:
                continue
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            h = len(shuffled) // 2
            self.halves[p] = (np.sort(shuffled[:h]), np.sort(shuffled[h:]))
            self.perturbations.append(p)
            means.append(np.asarray(self.adata.X[idx].mean(0)).ravel())
        self._mean_matrix = np.vstack(means) if means else np.zeros((0, len(self.var_names)))
        self._row = {p: i for i, p in enumerate(self.perturbations)}
        self._mean_sum = self._mean_matrix.sum(0)

    def cells(self, pert: str, half: str | None = None) -> np.ndarray:
        """Cells for ``pert``: the first/second split half, or all cells when ``half`` is None."""
        if half == "first":
            idx = self.halves[pert][0]
        elif half == "second":
            idx = self.halves[pert][1]
        else:
            idx = np.where(self.pert == pert)[0]
        return self.adata.X[idx]

    def control_cells(self, cap: int) -> np.ndarray:
        """A capped subsample of the non-targeting control cells."""
        return self.adata.X[self._cap(self.control_idx, cap, "control")]

    def all_perturbed_indices(self, cap: int) -> np.ndarray:
        """Indices of one all-perturbed subsample (the shared reference sample).

        The "pool" tag is a fixed reproducibility salt for the draw, not a public name.
        """
        return self._cap(np.where(self.pert != self.cfg.control_label)[0], cap, "pool")

    def allpert_mean_except(self, pert: str) -> np.ndarray:
        """Mean of all per-perturbation means, excluding ``pert`` (leave-one-out)."""
        k = len(self.perturbations)
        return (self._mean_sum - self._mean_matrix[self._row[pert]]) / max(k - 1, 1)

    def allpert_mean(self) -> np.ndarray:
        """Mean of all per-perturbation means (no target exclusion).

        A single vector shared across perturbations, used as the cross-perturbation ranking
        baseline.
        """
        return self._mean_sum / max(len(self.perturbations), 1)

    def control_mean(self) -> np.ndarray:
        """Pseudobulk centroid of the control cells."""
        return np.asarray(self.adata.X[self.control_idx].mean(0)).ravel()

    def _cap(self, idx: np.ndarray, cap: int, *tags) -> np.ndarray:
        if len(idx) <= cap:
            return np.sort(idx)
        chosen = _seed(self.cfg.seed, *tags).choice(idx, size=cap, replace=False)
        return np.sort(chosen)


def to_dense(X) -> np.ndarray:
    """Return ``X`` as a dense array (densifying if sparse)."""
    return X.toarray() if sp.issparse(X) else np.asarray(X)
