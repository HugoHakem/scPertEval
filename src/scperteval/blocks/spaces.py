"""Feature spaces: a transform applied to the gene axis before a protocol runs.

A space receives raw (possibly sparse) cells and returns a dense array over a gene subset.
Two registration patterns:

- **Fixed space** — one decorated function (:func:`space_full`).
- **Parameterised family** — a factory that registers ``name_<value>`` on demand:
  ``top_<k>`` (:func:`top_space`), ``degs_<padj>`` (:func:`degs_space`),
  ``pca_<k>`` (:func:`pca_space`), ``heg_<k>`` (:func:`heg_space`), ``hvg_<k>``
  (:func:`hvg_space`), ``perturbed_genes`` (:func:`perturbed_genes_space`, no parameters).

Default instances (``top_50``, ``degs_0.05``, ``pca_50``, ``heg_1000``, ``hvg_2000``,
``perturbed_genes``) are created at import; these are what ``scperteval list spaces`` shows.

**Gene-subset spaces and composition.** Every space above except :func:`space_full` and
:func:`pca_space` (which aren't gene subsets — one keeps every gene, the other transforms to
components) is registered through :func:`register_subset_space`, which stores the space's
*selection rule* — ``indices(ctx, pert) -> gene positions`` — as registry metadata rather than
only the finished transform. :func:`combine_space` uses those rules to build a new space from
two or more existing ones by a set operation, e.g. the HVG ∪ perturbed-genes gene panel of
:cite:t:`Miller_2025`::

    combine_space("miller_panel", hvg_space(8192), perturbed_genes_space())
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from ..dataset import to_dense
from ..registry import Registry

SPACES = Registry("space")
"""Registry of feature-space transforms; keys are space names (e.g. ``"top_50"``).

Use :meth:`~scperteval.registry.Registry.register` to add a custom space::

    from scperteval.blocks.spaces import SPACES, to_dense

    @SPACES.register("my_panel", global_space=True, description="a hand-picked gene panel")
    def space_my_panel(X, ctx, pert):
        keep = ...                    # indices of the genes to keep
        return to_dense(X[:, keep])

Pass ``global_space=True`` if the transform does not depend on the perturbation
(so it can be computed once and shared across all perturbations in a run).

**Optional ``indices`` capability.** A space that selects a *subset of genes* (rather than
transforming them, as ``pca_<k>`` does) should register through
:func:`register_subset_space` instead of by hand. That records the selection rule under the
``indices`` metadata key, which is what marks the space as composable: :func:`combine_space`
builds new spaces by applying set operations to those rules. A space registered without it
still works everywhere else — it just can't be composed.

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


# --- Gene-subset spaces: one registration helper shared by every subset family ---


def register_subset_space(name, indices, *, global_space=False, description="") -> str:
    """Register a space that keeps a subset of the genes, selected by ``indices``.

    The transform is always ``to_dense(X[:, indices(ctx, pert)])``; only the selection rule
    differs between subset spaces, so it is the only thing a caller supplies. The rule is also
    stored as ``indices`` registry metadata, which is what lets :func:`combine_space` compose
    spaces (see the :data:`SPACES` docstring). Registration is idempotent — an already-registered
    ``name`` is left untouched.

    Parameters
    ----------
    name : str
        Registry key for the space (e.g. ``"heg_1000"``).
    indices : Callable
        ``indices(ctx, pert) -> np.ndarray`` returning **integer positions into the full gene
        axis** — not into some earlier subset — so that rules from different spaces are directly
        comparable. Called once per perturbation per protocol; put anything expensive and shared
        behind a :class:`~scperteval.context.Context` cache. May ignore ``pert``.
    global_space : bool
        ``True`` if the selection does not depend on ``pert``, so it can be computed once and
        shared across every perturbation in a run.
    description : str
        Human-readable description shown by ``scperteval list spaces``.

    Returns
    -------
    str
        The registered space name (same as ``name``).
    """
    if name not in SPACES:

        def transform(X, ctx, pert):
            return to_dense(X[:, indices(ctx, pert)])

        SPACES.add(name, transform, indices=indices, global_space=global_space, description=description)
    return name


# --- Selection rules: (ctx, pert) -> integer positions into the full gene axis ---


def _field(de, name):
    return de.extra[name.split(":", 1)[1]] if name.startswith("extra:") else getattr(de, name)


def _de_top(ctx, pert, *, field, k):
    """Top-k genes by absolute value of a ground-truth DE field, per perturbation."""
    return np.argsort(-np.abs(_field(ctx.de(pert, ctx.cfg.truth), field)))[:k]


def _de_threshold(ctx, pert, *, field, threshold):
    """Genes whose ground-truth DE field passes ``threshold``, per perturbation."""
    return np.where(threshold(_field(ctx.de(pert, ctx.cfg.truth), field)))[0]


def _below(values, *, p):
    return values < p


def _heg(ctx, pert, *, k):
    """Top-k genes by control-condition mean expression, dataset-wide."""
    return np.argsort(-ctx.control_mean())[:k]


def _hvg(ctx, pert, *, k):
    """Top-k genes by control-condition normalized dispersion, dataset-wide."""
    return np.argsort(-ctx.control_hvg_dispersion())[:k]


def _perturbed_genes(ctx, pert):
    """Genes targeted by a retained perturbation, dataset-wide."""
    return ctx.perturbed_gene_indices()


# --- Parameterised families: a factory registers name_<value> on demand ---


def register_de_space(name, field, top=None, threshold=None, description="") -> str:
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
    if top is not None:
        rule = partial(_de_top, field=field, k=top)
    else:
        assert threshold is not None  # register_de_space takes exactly one of top/threshold
        rule = partial(_de_threshold, field=field, threshold=threshold)
    return register_subset_space(name, rule, description=description)


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
    return register_de_space(
        f"top_{k}", field="statistic", top=k, description=f"top {k} genes by ground-truth effect size, per perturbation"
    )


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
    return register_de_space(
        f"degs_{padj:g}",
        field="pvalue_adj",
        threshold=partial(_below, p=padj),
        description=f"ground-truth DEGs at adjusted p < {padj:g}, per perturbation",
    )


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
    return register_subset_space(
        f"heg_{k}",
        partial(_heg, k=k),
        global_space=True,
        description=f"top {k} genes by control-condition expression",
    )


def hvg_space(k: int) -> str:
    """top-k highly-variable genes by control-condition dispersion (registered on demand).

    Ranks genes dataset-wide by scanpy's ``"seurat"``-flavor normalized dispersion
    (:meth:`~scperteval.dataset.Dataset.control_hvg_dispersion`) over control cells — the
    same mean-binned-dispersion statistic ``sc.pp.highly_variable_genes`` uses, computed once
    over the (log-normalised, per ``docs/user-guide/datasets.md``) control cells and cached
    per dataset.

    Parameters
    ----------
    k : int
        Number of genes to keep (highest control-cell normalized dispersion, dataset-wide).

    Returns
    -------
    str
        Space name ``"hvg_<k>"`` (e.g. ``"hvg_2000"``).
    """
    return register_subset_space(
        f"hvg_{k}",
        partial(_hvg, k=k),
        global_space=True,
        description=f"top {k} genes by control-condition normalized dispersion",
    )


def perturbed_genes_space() -> str:
    """Genes targeted by a perturbation in the dataset (registered on demand, no parameters).

    Reads :meth:`~scperteval.context.Context.perturbed_gene_indices` — perturbation labels are
    gene symbols matching ``var_names`` (``+``-delimited for combinations; see
    ``docs/user-guide/datasets.md``). For a knockdown screen these are the knocked-down genes,
    whose own expression is the most direct readout that a perturbation took effect; they are
    not necessarily highly variable, which is why this is meant to be *unioned* with another
    subset space via :func:`combine_space` rather than used standalone.

    Returns
    -------
    str
        The registered space name, ``"perturbed_genes"``.
    """
    return register_subset_space(
        "perturbed_genes",
        _perturbed_genes,
        global_space=True,
        description="genes targeted by a perturbation in the dataset",
    )


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

    Not a gene subset — it transforms genes into components rather than selecting among them —
    so it is registered directly rather than through :func:`register_subset_space`, and
    :func:`combine_space` rejects it.

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


# --- Composing gene-subset spaces ---

_COMBINE_OPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "union": np.union1d,
    "intersect": np.intersect1d,
    "diff": np.setdiff1d,
}


def _combined(ctx, pert, *, rules, reduce_op):
    """Fold the constituent selection rules left-to-right with a set operation."""
    result = rules[0](ctx, pert)
    for rule in rules[1:]:
        result = reduce_op(result, rule(ctx, pert))
    return result


def combine_space(name: str, *spaces: str, op: str = "union") -> str:
    """Register a new gene-subset space built from two or more existing ones by a set operation.

    Each of ``spaces`` must already be a registered *subset* space — i.e. one registered via
    :func:`register_subset_space`, which is every built-in space except ``full`` and ``pca_<k>``.
    Their selection rules are applied to the same ``(ctx, pert)`` and folded together, so the
    genes are combined as index sets and the cells are densified only once, at the end.

    Parameters
    ----------
    name : str
        Registry key for the new space. Chosen by the caller (as with
        :func:`register_de_space`) so composed panels get a meaningful name — e.g.
        ``"miller_panel"`` rather than a derived one.
    *spaces : str
        Two or more already-registered gene-subset space names.
    op : str
        ``"union"`` (default), ``"intersect"``, or ``"diff"`` — the latter left-to-right, i.e.
        ``spaces[0]`` minus all the rest.

    Returns
    -------
    str
        The registered space name (same as ``name``).

    Notes
    -----
    The numpy set operations return sorted output, so a composite's columns are in gene order
    even when its constituents' are in rank order (``heg_5`` yields the five highest-expressed
    genes ranked; a composite including it yields them sorted). Every metric here is
    column-order invariant, so this only matters to an order-sensitive consumer.

    The constituent rules are looked up once, at registration; re-registering one of ``spaces``
    afterwards does not change an already-composed space.

    Examples
    --------
    The HVG ∪ perturbed-genes gene panel of :cite:t:`Miller_2025`::

        combine_space("miller_panel", hvg_space(8192), perturbed_genes_space())

    (``8192`` here is a number of genes; it is unrelated to the identically-valued default
    ``subsample``, which counts reference *cells*.)
    """
    if len(spaces) < 2:
        raise ValueError("combine_space needs at least two space names")
    if op not in _COMBINE_OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(_COMBINE_OPS)}")
    unknown = [s for s in spaces if s not in SPACES]
    if unknown:
        raise KeyError(f"unknown {SPACES.kind}(s) {unknown}; available: {SPACES.names()}")
    not_subsets = [s for s in spaces if "indices" not in SPACES.meta(s)]
    if not_subsets:
        raise ValueError(
            f"not gene-subset spaces (no indices metadata): {not_subsets}; "
            f"only spaces registered via register_subset_space can be composed"
        )
    rules = [SPACES.meta(s)["indices"] for s in spaces]
    return register_subset_space(
        name,
        partial(_combined, rules=rules, reduce_op=_COMBINE_OPS[op]),
        global_space=all(SPACES.meta(s).get("global_space", False) for s in spaces),
        description=f"{op} of {', '.join(spaces)}",
    )


# Default instances — also what `scperteval list spaces` shows.
top_space(50)
pca_space(50)
degs_space(0.05)
heg_space(1000)
hvg_space(2000)
perturbed_genes_space()
