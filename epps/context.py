"""The per-run engine: lazily builds and caches the shared building blocks, and
turns a (perturbation, source) into the exact view a protocol consumes."""
from __future__ import annotations

import threading

import numpy as np

from .blocks.de import DE_METHODS, moments, ttest_from_moments
from .blocks.spaces import SPACES
from .dataset import Dataset, to_dense
from .reference import Reference
from .sources import SOURCES
from .types import Protocol, RunConfig


class Context:
    """Owns the dataset, caches DE / PCA / control mean, and dispatches views.

    Caches are keyed per perturbation, so the runner can fan perturbations out
    across threads; ``current_pert`` is thread-local for the same reason.
    """

    def __init__(self, dataset: Dataset, cfg: RunConfig):
        self.ds = dataset
        self.cfg = cfg
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._de: dict = {}
        self._mom: dict = {}
        self._weights: dict = {}
        self._pca = None
        self._pca_k = 0
        self._control_mean = None
        self._reference = None
        self._ref_proj: dict = {}
        self._ref_sums = None

    @property
    def perturbations(self):
        return self.ds.perturbations

    @property
    def current_pert(self):
        return getattr(self._local, "pert", None)

    @current_pert.setter
    def current_pert(self, value):
        self._local.pert = value

    def warm(self, protocols):
        """Precompute shared singletons before the parallel loop so per-perturbation
        threads only ever write per-perturbation cache keys."""
        self.control_mean()
        if any(p.representation in ("population", "de") for p in protocols):
            self.reference()
        if any(p.representation == "de" for p in protocols):
            self._ensure_ref_sums()
            self._moments("control", None)
        if any(p.space == "pca50" for p in protocols):
            self.pca()
        for space in {p.space for p in protocols
                      if p.representation == "population" and SPACES.meta(p.space).get("global_space")}:
            self.ref_projection(space)

    def view(self, pert: str, source: str, p: Protocol):
        if p.representation == "population":
            if source == "all_perturbed":
                return self._reference_population(p.space, pert)
            return SPACES[p.space](SOURCES[source](self, pert), self, pert)
        if p.representation == "centroid":
            v = self.centroid(pert, source, p.centering)
            return SPACES[p.space](v[None, :], self, pert).ravel()
        if p.representation == "de":
            return self._de_view(pert, source, p)
        raise ValueError(f"unknown protocol representation {p.representation!r}")

    def centroid(self, pert, source, centering):
        arr = SOURCES[source](self, pert)
        if SOURCES.meta(source).get("provides") == "centroid":
            v = np.asarray(arr, dtype=np.float64).ravel()
        else:
            v = np.asarray(to_dense(arr), dtype=np.float64).mean(0)
        if centering == "ctrl":
            v = v - self.control_mean()
        elif centering == "allpert":
            v = v - self.ds.allpert_mean_except(pert)
        return v

    def _de_view(self, pert, source, p):
        """GT -> truth labels (its DEResult); a candidate -> its |score| ranking.
        The negative candidate is tested against ``neg_reference`` (e.g. control)
        rather than ``reference`` (the all-perturbed sample), the hybrid DE setup."""
        if source == "gt":
            return self.de(pert, "gt", p.reference)
        reference = p.neg_reference if (source == p.negative and p.neg_reference) else p.reference
        return np.abs(self.de(pert, source, reference).score)

    def de(self, pert, source, reference="all_perturbed"):
        """DE for one (source vs reference) comparison; the reference moments are
        leave-one-out, so a perturbation is never compared against a sample of itself."""
        method = self.cfg.de_method
        key = (self._mom_key(source, pert), self._mom_key(reference, pert), method)
        if key not in self._de:
            if method == "t-test":
                self._de[key] = ttest_from_moments(*self._moments(source, pert),
                                                    *self._moments(reference, pert))
            else:
                self._de[key] = DE_METHODS[method](self._de_cells(source, pert),
                                                   self._de_cells(reference, pert))
        return self._de[key]

    def _moments(self, source, pert):
        if source == "all_perturbed":
            return self._reference_moments(pert)
        key = self._mom_key(source, pert)
        if key not in self._mom:
            self._mom[key] = moments(self._de_cells(source, pert))
        return self._mom[key]

    def _de_cells(self, source, pert):
        if source == "all_perturbed":
            return self.reference().subset(pert)
        if source == "control":
            return self.ds.control_cells(self.cfg.subsample)
        return SOURCES[source](self, pert)

    @staticmethod
    def _mom_key(source, pert):
        return source if source == "control" else (source, pert)

    def wmse_weights(self, pert):
        """Mejia DEG weights: min-max normalised |effect size| of GT vs the reference."""
        if pert not in self._weights:
            s = np.abs(self.de(pert, "gt", "all_perturbed").score)
            finite = np.isfinite(s)
            lo, hi = s[finite].min(), s[finite].max()
            w = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
            self._weights[pert] = np.nan_to_num(w, nan=0.0)
        return self._weights[pert]

    # -- the all-perturbed reference: one sample, served leave-one-out -------------

    def reference(self) -> Reference:
        """The all-perturbed sample (subsampled + densified once), with each cell's
        perturbation recorded so it can be served leave-one-out."""
        if self._reference is None:
            with self._init_lock:
                if self._reference is None:
                    idx = self.ds.all_perturbed_indices(self.cfg.subsample)
                    cells = to_dense(self.ds.adata.X[idx]).astype(np.float64)
                    self._reference = Reference(cells, self.ds.pert[idx])
        return self._reference

    def _reference_population(self, space, pert):
        """The reference in a feature space with the target perturbation removed:
        project the whole sample (cached for global spaces) then drop its rows."""
        ref = self.reference()
        if SPACES.meta(space).get("global_space"):
            proj = self.ref_projection(space)
        else:
            proj = SPACES[space](ref.cells, self, pert)
        keep = ref.keep(pert)
        return proj if keep is None else proj[keep]

    def ref_projection(self, space):
        """The reference projected to a perturbation-independent space, cached once."""
        if space not in self._ref_proj:
            with self._init_lock:
                if space not in self._ref_proj:
                    self._ref_proj[space] = SPACES[space](self.reference().cells, self, None)
        return self._ref_proj[space]

    def _ensure_ref_sums(self):
        """Cache the reference's column sums and sums-of-squares once, so leave-one-out
        moments are an O(target cells) subtraction rather than a re-densify per perturbation."""
        if self._ref_sums is None:
            with self._init_lock:
                if self._ref_sums is None:
                    X = self.reference().cells
                    self._ref_sums = (X.sum(0), np.einsum("ij,ij->j", X, X), len(X))
        return self._ref_sums

    def _reference_moments(self, pert):
        total, totalsq, n = self._ensure_ref_sums()
        keep = self.reference().keep(pert)
        if keep is None:
            s, sq, k = total, totalsq, n
        else:
            Xp = self.reference().cells[~keep]
            s = total - Xp.sum(0)
            sq = totalsq - np.einsum("ij,ij->j", Xp, Xp)
            k = int(keep.sum())
        mean = s / k
        var = np.maximum((sq / k - mean * mean) * (k / max(k - 1, 1)), 0.0)
        return mean, var, k

    def control_mean(self):
        if self._control_mean is None:
            with self._init_lock:
                if self._control_mean is None:
                    self._control_mean = self.ds.control_mean()
        return self._control_mean

    def pca(self, k=50):
        """A fitted PCA with at least ``k`` components (refit if a larger k is later asked for)."""
        if self._pca is None or self._pca_k < k:
            with self._init_lock:
                if self._pca is None or self._pca_k < k:
                    self._pca_k = max(k, 50)
                    self._pca = self._fit_pca(self._pca_k)
        return self._pca

    PCA_FIT_CAP = 50000

    def _fit_pca(self, n_components):
        """Fit PCA on (nearly) all cells; the subsample cap is for the O(n^2)
        distance populations, not the PCA basis, which needs many cells to be stable."""
        from sklearn.decomposition import PCA

        n = self.ds.adata.n_obs
        idx = np.arange(n)
        if n > self.PCA_FIT_CAP:
            idx = np.sort(np.random.default_rng(self.cfg.seed).choice(n, self.PCA_FIT_CAP, replace=False))
        X = to_dense(self.ds.adata.X[idx]).astype(np.float64)
        return PCA(n_components=min(n_components, *X.shape), random_state=self.cfg.seed).fit(X)
