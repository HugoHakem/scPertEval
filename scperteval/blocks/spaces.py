"""Feature spaces: a transform applied to the gene axis before a protocol runs.

Spaces receive the raw (possibly sparse) cells and return a dense array, so a
gene-subset space densifies only its subset. The parameterised families
``top_<k>`` / ``pca_<k>`` / ``degs_<padj>`` are registered on demand by the
``top_space`` / ``pca_space`` / ``degs_space`` factories (used by the protocol
templates); the default instances created at import are what ``scperteval list spaces``
shows. ``description`` is shown by ``scperteval list spaces``.
"""
from __future__ import annotations

import numpy as np

from ..dataset import to_dense
from ..registry import Registry

SPACES = Registry("space")


@SPACES.register("full", global_space=True, description="all genes, no transform")
def space_full(X, ctx, pert):
    return to_dense(X)


def _field(de, name):
    return de.extra[name.split(":", 1)[1]] if name.startswith("extra:") else getattr(de, name)


def register_de_space(name, field, top=None, threshold=None, description=""):
    """Register a DE-derived gene subset selected from a field of the GT DEResult.

    Exactly one of ``top`` (select top-k by |value|) or ``threshold`` (a callable
    returning a boolean mask) must be provided.
    """

    def space(X, ctx, pert):
        values = _field(ctx.de(pert, "gt"), field)
        if top is not None:
            keep = np.argsort(-np.abs(values))[:top]
        else:
            assert threshold is not None, "either top or threshold must be provided"
            keep = np.where(threshold(values))[0]
        return to_dense(X[:, keep])

    SPACES.add(name, space, description=description)
    return name


def top_space(k: int) -> str:
    """top-k genes by |ground-truth effect size| (registered on demand)."""
    name = f"top_{k}"
    if name not in SPACES:
        register_de_space(name, field="score", top=k,
                          description=f"top {k} genes by ground-truth effect size, per perturbation")
    return name


def degs_space(padj: float) -> str:
    """ground-truth DEGs at adjusted p < padj (registered on demand)."""
    name = f"degs_{padj:g}"
    if name not in SPACES:
        register_de_space(name, field="pvalue_adj", threshold=(lambda v, p=padj: v < p),
                          description=f"ground-truth DEGs at adjusted p < {padj:g}, per perturbation")
    return name


def pca_space(k: int) -> str:
    """top-k principal components (registered on demand)."""
    name = f"pca_{k}"
    if name not in SPACES:
        SPACES.add(name, lambda X, ctx, pert, k=k: ctx.pca(k).transform(to_dense(X))[:, :k],
                   global_space=True, description=f"top {k} principal components (fit on the dataset)")
    return name


# Default instances — also what `scperteval list spaces` shows.
top_space(50)
pca_space(50)
degs_space(0.05)
