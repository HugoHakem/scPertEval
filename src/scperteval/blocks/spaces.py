"""Feature spaces: a transform applied to the gene axis before a protocol runs.

A space receives raw (possibly sparse) cells and returns a dense array over a gene subset.
Two registration patterns:

- **Fixed space** — one decorated function (:func:`space_full`).
- **Parameterised family** — a factory that registers ``name_<value>`` on demand:
  ``top_<k>`` (:func:`top_space`), ``degs_<padj>`` (:func:`degs_space`),
  ``pca_<k>`` (:func:`pca_space`), ``heg_<k>`` (:func:`heg_space`), ``hvg_<k>`` (:func:`hvg_space`),
  ``perturbed`` (:func:`perturbed_space`, no parameters).

Default instances (``top_50``, ``degs_0.05``, ``pca_50``, ``heg_1000``, ``hvg_2000``,
``perturbed``) are created at import; these are what ``scperteval list spaces`` shows.

**Composing spaces.** Every space above except :func:`space_full`/:func:`pca_space` (which
aren't gene subsets — one keeps every gene, the other transforms to components) selects a gene
*subset*, and registers the ``(ctx, pert) -> indices`` callable that picked it as registry
metadata. :func:`combine_space` composes two or more such spaces by a set operation (union,
intersect, or diff) into a new registered space, e.g. Miller et al. 2025's HVG ∪
perturbed-genes gene panel: ``combine_space(hvg_space(8192), perturbed_space())``.
"""

from __future__ import annotations

from collections.abc import Callable

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

**Optional ``prepare`` hook.** A space family whose transform depends on some expensive, shared
structure (a fitted basis, a trained embedding model) can register a ``prepare`` hook to build
that structure once, up front, instead of lazily inside the per-perturbation loop::

    @SPACES.register("pca_50", global_space=True, prepare=my_prepare, description="…")
    def space(X, ctx, pert): ...

- Signature ``prepare(ctx, names) -> None`` — ``names`` is the *set* of that family's variant
  space names requested in the run (e.g. ``{"pca_30", "pca_50", "pca_100"}``).
- :meth:`~scperteval.context.Context.warm` calls each distinct hook **once** with all its
  variants, before any transform runs — so the family sees every variant at once and can build
  each variant's shared structure eagerly instead of on first use (e.g. fit each requested PCA
  size; a learned embedding might fit one model per variant). Store the result on ``ctx`` (e.g.
  ``ctx.pca(...)``, which caches on the shared store).
- It is **purely an optimisation and must be idempotent**: the transform has to stay correct if
  the hook never runs (a space run without being declared to ``prepare`` computes lazily), and the
  hook may be invoked again on an already-warm context. Do no per-perturbation work here — that
  belongs in the transform.
"""


# --- Fixed spaces: one registered function each ---


@SPACES.register("full", global_space=True, description="all genes, no transform")
def space_full(X, ctx, pert):
    """Identity space: all genes, densified, no transform."""
    return to_dense(X)


# --- Parameterised families: a factory registers name_<value> on demand ---


def _field(de, name):
    return de.extra[name.split(":", 1)[1]] if name.startswith("extra:") else getattr(de, name)


def _index_space(name, indices, *, global_space=False, prepare=None, description=""):
    """Register a gene-subset space: ``indices(ctx, pert) -> array`` picks the columns to keep.

    Shared by every subset-space factory below. Stores ``indices`` as registry metadata (not
    just the resulting transform) so :func:`combine_space` can compose spaces by their gene-index
    sets rather than their already-densified output.
    """

    def transform(X, ctx, pert):
        return to_dense(X[:, indices(ctx, pert)])

    SPACES.add(name, transform, indices=indices, global_space=global_space, prepare=prepare, description=description)
    return name


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

    def indices(ctx, pert):
        values = _field(ctx.de(pert, ctx.cfg.truth), field)
        if top is not None:
            return np.argsort(-np.abs(values))[:top]
        assert threshold is not None  # register_de_space takes exactly one of top/threshold
        return np.where(threshold(values))[0]

    return _index_space(name, indices, description=description)


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
    """Prepare hook for the ``pca_*`` family: fit each requested ``pca_<k>`` up front.

    ``names`` is the set of requested ``pca_<k>`` space names. Each distinct fit-size is fit once
    and cached independently (see :meth:`~scperteval.context.Context.pca`): sklearn's PCA is not
    basis-stable across ``n_components``, so a smaller ``pca_k`` cannot be sliced from a larger
    fit without changing its result. Set iteration order does not matter — every size is fit.
    """
    for name in names:
        ctx.pca(int(name.rsplit("_", 1)[1]))


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

        def transform(X, ctx, pert):
            return ctx.pca(k).transform(to_dense(X))[:, :k]

        SPACES.add(
            name,
            transform,
            global_space=True,
            prepare=_pca_prepare,
            description=f"top {k} principal components (fit on the dataset)",
        )
    return name


def heg_space(k: int) -> str:
    """top-k highly-expressed genes by control-condition mean (registered on demand).

    Ranks genes dataset-wide by their mean expression in control cells — the same panel
    for every perturbation, unlike :func:`top_space`/:func:`degs_space` which rank per
    perturbation by ground-truth effect size. This is the gene-selection criterion used by
    :cite:t:`AhlmannEltze_2025`.

    Parameters
    ----------
    k : int
        Number of genes to keep (highest control-mean expression, dataset-wide).

    Returns
    -------
    str
        Space name ``"heg_<k>"`` (e.g. ``"heg_1000"``).
    """
    name = f"heg_{k}"
    if name not in SPACES:

        def indices(ctx, pert):
            return np.argsort(-ctx.control_mean())[:k]

        _index_space(name, indices, global_space=True, description=f"top {k} genes by control-condition expression")
    return name


def hvg_space(k: int) -> str:
    """top-k highly-variable genes by control-condition dispersion (registered on demand).

    Ranks genes dataset-wide by scanpy's ``"seurat"``-flavor normalized dispersion
    (:meth:`~scperteval.dataset.Dataset.control_hvg_dispersion`) over control cells — the
    same mean-binned-dispersion statistic ``sc.pp.highly_variable_genes`` uses, computed once
    on log-normalised control cells and cached per dataset.

    Parameters
    ----------
    k : int
        Number of genes to keep (highest control-cell normalized dispersion, dataset-wide).

    Returns
    -------
    str
        Space name ``"hvg_<k>"`` (e.g. ``"hvg_2000"``).
    """
    name = f"hvg_{k}"
    if name not in SPACES:

        def indices(ctx, pert):
            return np.argsort(-ctx.control_hvg_dispersion())[:k]

        _index_space(
            name,
            indices,
            global_space=True,
            description=f"top {k} genes by control-condition normalized dispersion",
        )
    return name


def perturbed_space() -> str:
    """Genes targeted by any perturbation in the dataset (registered on demand, no parameters).

    Reads :meth:`~scperteval.context.Context.perturbed_genes` — perturbation labels are gene
    symbols matching ``var_names`` (``+``-delimited for combinations; see
    docs/user-guide/datasets.md). Meant to be composed with another subset space via
    :func:`combine_space` (e.g. Miller et al. 2025's HVG ∪ perturbed-genes gene panel) rather
    than used standalone.

    Returns
    -------
    str
        The registered space name, ``"perturbed"``.
    """
    name = "perturbed"
    if name not in SPACES:

        def indices(ctx, pert):
            return ctx.perturbed_genes()

        _index_space(name, indices, global_space=True, description="genes targeted by any perturbation in the dataset")
    return name


_COMBINE_OPS: dict[str, tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
    "union": ("+", np.union1d),
    "intersect": ("&", np.intersect1d),
    "diff": ("-", np.setdiff1d),
}


def combine_space(*names: str, op: str = "union") -> str:
    """Compose two or more registered gene-subset spaces by a set operation.

    Each ``name`` must already be a registered subset space carrying an ``indices(ctx, pert)``
    callable in its metadata — i.e. anything registered via :func:`_index_space`
    (:func:`top_space`, :func:`degs_space`, :func:`heg_space`, :func:`hvg_space`,
    :func:`perturbed_space`, or a custom space registered the same way). :func:`space_full`
    and :func:`pca_space` aren't gene subsets and can't be composed this way.

    Parameters
    ----------
    *names : str
        Two or more already-registered gene-subset space names.
    op : str
        ``"union"`` (default, symbol ``"+"``), ``"intersect"`` (``"&"``), or ``"diff"``
        (``"-"``, left-to-right: ``names[0]`` minus the rest).

    Returns
    -------
    str
        The composed space's name — ``op``'s symbol joining ``names`` in order
        (e.g. ``"hvg_2000+perturbed"``).
    """
    if len(names) < 2:
        raise ValueError("combine_space needs at least two space names")
    if op not in _COMBINE_OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_COMBINE_OPS)}")
    symbol, reduce_op = _COMBINE_OPS[op]
    name = symbol.join(names)
    if name not in SPACES:
        missing = [n for n in names if "indices" not in SPACES.meta(n)]
        if missing:
            raise ValueError(f"not gene-subset spaces (no indices metadata): {missing}")
        index_fns = [SPACES.meta(n)["indices"] for n in names]
        global_space = all(SPACES.meta(n).get("global_space", False) for n in names)

        def indices(ctx, pert):
            result = index_fns[0](ctx, pert)
            for fn in index_fns[1:]:
                result = reduce_op(result, fn(ctx, pert))
            return result

        _index_space(name, indices, global_space=global_space, description=f"{op} of {', '.join(names)}")
    return name


# Default instances — also what `scperteval list spaces` shows.
top_space(50)
pca_space(50)
degs_space(0.05)
heg_space(1000)
hvg_space(2000)
perturbed_space()
