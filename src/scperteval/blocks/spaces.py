"""Feature spaces: a transform applied to the gene axis before a protocol runs.

A space receives raw (possibly sparse) cells and returns a dense array over a gene subset.
Three parameterised families are registered on demand by factory functions:

- ``top_<k>`` — top-k genes by ground-truth effect size (:func:`top_space`).
- ``degs_<padj>`` — ground-truth DEGs at adjusted p < padj (:func:`degs_space`).
- ``pca_<k>`` — top-k principal components (:func:`pca_space`).

Default instances (``top_50``, ``degs_0.05``, ``pca_50``) are created at import;
these are what ``scperteval list spaces`` shows.
"""

from __future__ import annotations

import numpy as np

from ..dataset import to_dense
from ..registry import Registry

SPACES = Registry("space")
"""Registry of feature-space transforms; keys are space names (e.g. ``"top_50"``).

Use :meth:`~scperteval.registry.Registry.register` to add a custom space::

    from scperteval.blocks.spaces import SPACES, to_dense

    @SPACES.register("hvg_100", global_space=True, description="100 highest-variance genes")
    def space_hvg(X, ctx, pert):
        keep = ...                    # indices of the 100 genes to keep
        return to_dense(X[:, keep])

Pass ``global_space=True`` if the transform does not depend on the perturbation
(so it can be computed once and shared across all perturbations in a run).

Embedding-style families (PCA today; learned/UMAP/scVI later) may also register an optional
``prepare`` hook — a callable ``(ctx, names) -> None`` where ``names`` is the *set* of that
family's requested space names in a run. :meth:`~scperteval.context.Context.warm` invokes each
distinct hook once with all its variants, so the family can do a single shared precompute sized
for the largest variant (e.g. fit PCA once at ``max(k)`` and slice smaller ``pca_k`` from it).
The hook is purely an optimisation: transforms must stay correct when it never runs.
"""


@SPACES.register("full", global_space=True, description="all genes, no transform")
def space_full(X, ctx, pert):
    """Identity space: all genes, densified, no transform."""
    return to_dense(X)


def _field(de, name):
    return de.extra[name.split(":", 1)[1]] if name.startswith("extra:") else getattr(de, name)


def register_de_space(name, field, top=None, threshold=None, description=""):
    r"""Register a DE-derived gene subset selected from a field of the GT PerturbationDEResult.

    Exactly one of ``top`` (select top-k by \|value\|) or ``threshold`` (a callable
    returning a boolean mask) must be provided.

    Parameters
    ----------
    name : str
        Registry key for the new space.
    field : str
        Attribute of :class:`~scperteval.types.PerturbationDEResult` to read
        (e.g. ``"statistic"``, ``"pvalue_adj"``).
    top : int or None
        If given, keep the top-k genes by absolute value of ``field``.
    threshold : Callable or None
        If given, a function ``(values) -> bool mask`` selecting genes to keep.
    description : str
        Human-readable description shown by ``scperteval list spaces``.

    Returns
    -------
    str
        The registered space name (same as ``name``).
    """

    def space(X, ctx, pert):
        values = _field(ctx.de(pert, ctx.cfg.truth), field)
        if top is not None:
            keep = np.argsort(-np.abs(values))[:top]
        else:
            assert threshold is not None  # register_de_space takes exactly one of top/threshold
            keep = np.where(threshold(values))[0]
        return to_dense(X[:, keep])

    SPACES.add(name, space, description=description)
    return name


def top_space(k: int) -> str:
    r"""top-k genes by absolute ground-truth effect size (registered on demand).

    Parameters
    ----------
    k : int
        Number of genes to keep (selected by \|ground-truth effect size\| per perturbation).

    Returns
    -------
    str
        Space name ``"top_<k>"`` (e.g. ``"top_50"``).
    """
    name = f"top_{k}"
    if name not in SPACES:
        register_de_space(
            name, field="statistic", top=k, description=f"top {k} genes by ground-truth effect size, per perturbation"
        )
    return name


def degs_space(padj: float) -> str:
    """ground-truth DEGs at adjusted p < padj (registered on demand).

    Parameters
    ----------
    padj : float
        Adjusted p-value threshold (e.g. 0.05).

    Returns
    -------
    str
        Space name ``"degs_<padj>"`` (e.g. ``"degs_0.05"``).
    """
    name = f"degs_{padj:g}"
    if name not in SPACES:
        register_de_space(
            name,
            field="pvalue_adj",
            threshold=(lambda v, p=padj: v < p),
            description=f"ground-truth DEGs at adjusted p < {padj:g}, per perturbation",
        )
    return name


def _pca_prepare(ctx, names):
    """Prepare hook for the ``pca_*`` family: fit PCA once at the largest k requested.

    ``names`` is the set of requested ``pca_<k>`` space names; PCA components are nested,
    so fitting at ``max(k)`` lets every smaller ``pca_k`` slice from the same fit with no refit.
    Order-independent (a ``max`` over a set), so ``warm()`` stays deterministic.
    """
    ctx.pca(max(int(name.rsplit("_", 1)[1]) for name in names))


def pca_space(k: int) -> str:
    """top-k principal components (registered on demand).

    PCA is fit once on (up to 50 000) cells from the full dataset, then applied
    to each cell population. The fitted transform is shared across perturbations.

    Parameters
    ----------
    k : int
        Number of principal components to retain.

    Returns
    -------
    str
        Space name ``"pca_<k>"`` (e.g. ``"pca_50"``).
    """
    name = f"pca_{k}"
    if name not in SPACES:
        SPACES.add(
            name,
            lambda X, ctx, pert, k=k: ctx.pca(k).transform(to_dense(X))[:, :k],
            global_space=True,
            prepare=_pca_prepare,
            description=f"top {k} principal components (fit on the dataset)",
        )
    return name


# Default instances — also what `scperteval list spaces` shows.
top_space(50)
pca_space(50)
degs_space(0.05)
