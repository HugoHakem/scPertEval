"""Every feature space, one decorated rule each.

To add one, write a rule and decorate it:

- A **subset** rule is ``(ctx, pert, value)`` returning a column selection into the *full* gene
  axis: an integer array, or a slice. Never positions into some earlier subset, so selections
  from different spaces can be folded together. Give the trailing argument a default to declare
  that the space takes no parameter, and pass ``per_pert=True`` when the choice varies by
  perturbation.
- A **transform** rule is ``(X, ctx, pert, value)`` returning the finished dense array, for a
  space that replaces the gene axis rather than narrowing it.

The rule runs once per perturbation per protocol, so anything computed over the whole dataset
belongs behind a :class:`~scperteval.context.Context` cache — ``ctx.control_mean()`` and friends —
rather than recomputed here.
"""

from __future__ import annotations

import numpy as np

from ...dataset import to_dense
from .registry import OPS, SPACES, combine_subsets


@SPACES.subset("full", description="all genes, no transform")
def full(ctx, pert, value=None):
    """Every gene. Returns a slice, so applying the identity space is a view rather than a copy."""
    return slice(None)


@SPACES.subset("top", default=50, per_pert=True, description="top {v} genes by ground-truth effect size")
def top(ctx, pert, k):
    """The k strongest ground-truth effect sizes for this perturbation, by absolute value."""
    return np.argsort(-np.abs(ctx.de(pert, ctx.cfg.truth).statistic))[:k]


@SPACES.subset("degs", default=0.05, per_pert=True, description="ground-truth DEGs at adjusted p < {v}")
def degs(ctx, pert, padj):
    """Ground-truth differentially expressed genes for this perturbation."""
    return np.where(ctx.de(pert, ctx.cfg.truth).pvalue_adj < padj)[0]


@SPACES.subset("heg", default=1000, description="top {v} genes by control-condition expression")
def heg(ctx, pert, k):
    """The k highest-expressed genes in the control cells — the criterion of Ahlmann-Eltze 2025.

    Dataset-wide, so the same panel serves every perturbation, unlike ``top``/``degs``.
    """
    return np.argsort(-ctx.control_mean())[:k]


@SPACES.subset("hvg", default=2000, description="top {v} genes by control-condition normalized dispersion")
def hvg(ctx, pert, k):
    """The k most variable genes in the control cells, by scanpy's ``"seurat"`` dispersion."""
    return np.argsort(-ctx.control_hvg_dispersion())[:k]


@SPACES.subset("perturbed_genes", description="genes targeted by a perturbation in the dataset")
def perturbed_genes(ctx, pert, value=None):
    """The genes the perturbations target — for a knockdown screen, the knocked-down genes.

    Their own expression is the most direct readout that a perturbation took effect, and they
    aren't necessarily variable, so this is meant to be unioned with another subset.
    """
    return ctx.perturbed_gene_indices()


@SPACES.subset("perturbed_and_hvgs", description="HVG union perturbed genes — a panel introduced in Miller et al. 2025")
def perturbed_and_hvgs(ctx, pert, value=None):
    """The gene panel of Miller et al. 2025: the top 8192 HVGs plus every targeted gene."""
    return combine_subsets(ctx, OPS.union, hvg(ctx, pert, 8192), perturbed_genes(ctx, pert))


def _fit_pca(ctx, names):
    """Fit every requested ``pca_<k>`` before the run.

    sklearn's PCA is not basis-stable across ``n_components``, so a smaller ``pca_k`` can't be
    sliced out of a larger fit — each size is fit and cached separately.
    """
    for name in names:
        ctx.pca(int(name.rsplit("_", 1)[1]))


@SPACES.transform("pca", default=50, prepare=_fit_pca, description="top {v} principal components (fit on the dataset)")
def pca(X, ctx, pert, k):
    """The top k principal components, from a PCA fit once on the dataset and shared."""
    return ctx.pca(k).transform(to_dense(X))[:, :k]


# A space is created when a protocol or a `Param` asks for it. `full` is created here because
# `Protocol.space` defaults to it, so that name has to resolve before any run.
SPACES.instance("full")
