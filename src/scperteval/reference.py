"""The comparison reference: a fixed cell sample served leave-one-out."""

from __future__ import annotations

import warnings

import numpy as np


class Reference:
    """A comparison sample of cells (the all-perturbed subsample, or non-targeting
    control), served leave-one-out.

    ``subset(P)`` returns the sample with perturbation ``P``'s own cells removed, so
    a perturbation is never scored against a reference that contains itself. When
    the sample has no per-cell perturbation labels (e.g. control), the exclusion is
    a no-op. The only logic here is owning the materialised sample and the
    leave-one-out rule -- callers project it or reduce it like any other population.
    """

    def __init__(self, cells, labels=None, warn_frac: float = 0.10):
        self.cells = cells  # densified once, (n_cells, n_genes)
        self.labels = labels  # per-cell perturbation, or None
        self.warn_frac = warn_frac
        self._n = len(cells)
        self._warned: set = set()

    def keep(self, exclude) -> np.ndarray | None:
        """Boolean mask of the cells to keep, or None when nothing is excluded."""
        if self.labels is None:
            return None
        mask = self.labels != exclude
        self._warn_if_large(exclude, self._n - int(mask.sum()))
        return mask

    def subset(self, exclude):
        mask = self.keep(exclude)
        return self.cells if mask is None else self.cells[mask]

    def _warn_if_large(self, exclude, dropped: int) -> None:
        if dropped > self.warn_frac * self._n and exclude not in self._warned:
            self._warned.add(exclude)
            warnings.warn(
                f"Excluding '{exclude}' removes {dropped}/{self._n} "
                f"({100 * dropped / self._n:.0f}%) of the reference sample -- this "
                f"perturbation is a large share of the dataset, so its leave-one-out "
                f"reference is much smaller than other perturbations'. Raise --subsample "
                f"(or expect noisier single-cell metrics for this perturbation).",
                stacklevel=3,
            )
