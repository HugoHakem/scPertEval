"""Feature spaces: which features a protocol scores on, chosen before the metric runs.

Every space is one row of :data:`SUBSETS` (keep some genes, drop the rest) or :data:`TRANSFORMS`
(replace the gene axis with something else), near the bottom of this file. To add one, write a
selection rule and add a row — see :doc:`/user-guide/building-blocks`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

import numpy as np

from ..dataset import to_dense
from ..registry import Registry

# =============================================================================
# Row types
# =============================================================================


@dataclass(frozen=True)
class Subset:
    """A space that keeps a subset of the genes. One row of :data:`SUBSETS`."""

    #: Space name. With a ``default``, instances are ``"<name>_<value>"`` (``"heg_1000"``).
    name: str
    #: ``select(ctx, pert, value)`` -> a column selection into the **full** gene axis: an integer
    #: array, or a slice. Runs once per perturbation per protocol, so anything expensive and
    #: shared belongs behind a :class:`~scperteval.context.Context` cache.
    select: Callable
    #: Parameter value for the instance registered at import; ``None`` if the space takes none.
    default: Any = None
    #: Shown by ``scperteval list spaces``; ``{v}`` is replaced by the parameter.
    description: str = ""
    #: Whether the selection depends on which perturbation is being scored.
    per_pert: bool = False


@dataclass(frozen=True)
class Transform:
    """A space that replaces the gene axis instead of narrowing it. One row of :data:`TRANSFORMS`.

    Has no gene selection, so :meth:`~scperteval.blocks.spaces.SpaceRegistry.combine` cannot compose it.
    """

    name: str
    #: ``apply(X, ctx, pert, value)`` -> the dense ``cells × features`` array, built directly.
    apply: Callable
    default: Any = None
    description: str = ""
    #: Optional ``prepare(ctx, names)`` run once before a run, with every requested variant name,
    #: to build shared structure up front (see :meth:`~scperteval.context.Context.warm`). Purely
    #: an optimisation: ``apply`` must stay correct if it never runs, and it must be idempotent.
    prepare: Callable | None = None


# =============================================================================
# The registry
# =============================================================================


def _bind(fn, value):
    """``fn`` with its trailing ``value`` bound, leaving ``(ctx, pert)`` or ``(X, ctx, pert)``."""

    def bound(*args):
        return fn(*args, value)

    return bound


def _instance_name(space, value):
    """``space``'s registry name at ``value``, with its ``default`` filled in when omitted."""
    if space.default is None:
        if value is not None:
            raise TypeError(f"space {space.name!r} takes no parameter, got {value!r}")
        return space.name, None
    value = space.default if value is None else value
    return f"{space.name}_{value:g}", value


class SpaceRegistry(Registry):
    """The feature-space registry: it holds the transforms *and* builds them from space definitions.

    A protocol names its space as a string, so a name must be registered here before a run can
    resolve it. :meth:`~scperteval.blocks.spaces.SpaceRegistry.instance` registers one variant of a :class:`Subset` or :class:`Transform`,
    :meth:`~scperteval.blocks.spaces.SpaceRegistry.combine` builds a new subset from existing ones, and :meth:`~scperteval.blocks.spaces.SpaceRegistry.add_subset` registers a
    subset defined outside this module.
    """

    #: Set operations :meth:`~scperteval.blocks.spaces.SpaceRegistry.combine` supports, by name.
    OPS: ClassVar[dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {
        "union": np.union1d,
        "intersect": np.intersect1d,
        "diff": np.setdiff1d,
    }

    def add_subset(self, name, select, *, global_space=False, description="", **meta) -> str:
        """Register a gene-subset space under ``name``, its genes chosen by ``select(ctx, pert)``.

        In-tree spaces go through :meth:`~scperteval.blocks.spaces.SpaceRegistry.instance` instead; this is the entry point for a subset
        defined outside this module. ``select`` returns a column selection into the full gene axis
        (an integer array or a slice); the transform and composability follow from it.

        Parameters
        ----------
        name : str
            Registry key for the space.
        select : Callable
            ``select(ctx, pert) -> column selection``, already bound to any parameter.
        global_space : bool
            ``True`` if the selection ignores ``pert``, so it can be computed once and shared.
        description : str
            Shown by ``scperteval list spaces``.
        **meta
            Extra registry metadata.

        Returns
        -------
        str
            The registered space name (same as ``name``).
        """

        def transform(X, ctx, pert):
            return to_dense(X[:, select(ctx, pert)])

        self.add(name, transform, select=select, global_space=global_space, description=description, **meta)
        return name

    def instance(self, space: Subset | Transform, value=None) -> str:
        """Register one variant of ``space`` and return its name, e.g. ``SPACES.instance(HEG, 250)``.

        Idempotent: a variant already registered at the same value is reused. Omit ``value`` for
        the space's ``default``, which is what :data:`DEFAULTS` does at import.

        Parameters
        ----------
        space : Subset or Transform
            The space definition, e.g. ``HEG``.
        value : optional
            Its parameter (e.g. ``k``). Defaults to ``space.default``; must be omitted for a space
            that takes no parameter.

        Returns
        -------
        str
            The registered name — ``"<name>_<value>"``, or ``"<name>"`` if it takes no parameter.
        """
        name, value = _instance_name(space, value)
        if name in self:
            # Distinct values can format to the same name (0.05 and 0.05000001 are both "0.05").
            # Registering the second silently under the first's rule would score the wrong genes.
            registered = self.meta(name).get("value")
            if registered != value:
                raise ValueError(f"{name!r} is already registered with value {registered!r}, not {value!r}")
            return name
        description = space.description.format(v=f"{value:g}" if value is not None else "")
        if isinstance(space, Subset):
            return self.add_subset(
                name,
                _bind(space.select, value),
                global_space=not space.per_pert,
                description=description,
                value=value,
            )
        self.add(
            name,
            _bind(space.apply, value),
            global_space=True,
            prepare=space.prepare,
            description=description,
            value=value,
        )
        return name

    def combine(self, name: str, *spaces: str, op: str = "union") -> str:
        """Register a gene-subset space built from two or more registered ones by a set operation.

        Each of ``spaces`` must name a registered subset — a :data:`SUBSETS` variant, a previous
        :meth:`~scperteval.blocks.spaces.SpaceRegistry.combine` result, or an :meth:`~scperteval.blocks.spaces.SpaceRegistry.add_subset` call. Their selections are taken on the
        same ``(ctx, pert)`` and folded together, so genes combine as index sets and the cells are
        densified once, at the end.

        Parameters
        ----------
        name : str
            Registry key for the new space, chosen by the caller so composed panels read
            meaningfully.
        *spaces : str
            Two or more registered gene-subset space names.
        op : str
            A key of ``OPS`` — ``"union"`` (default), ``"intersect"``, or ``"diff"``, the last
            left-to-right, i.e. ``spaces[0]`` minus all the rest.

        Returns
        -------
        str
            The registered space name (same as ``name``).

        Notes
        -----
        The set operations return sorted output, so a composite's columns are in gene order even
        when its parts' are in rank order. Every metric here is column-order invariant.
        Constituent selections are read once, at registration.
        """
        if len(spaces) < 2:
            raise ValueError("combine needs at least two space names")
        if op not in self.OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {sorted(self.OPS)}")
        unknown = [s for s in spaces if s not in self]
        if unknown:
            raise KeyError(f"unknown {self.kind}(s) {unknown}; available: {self.names()}")
        not_subsets = [s for s in spaces if "select" not in self.meta(s)]
        if not_subsets:
            raise ValueError(f"not gene-subset spaces, so they have no genes to combine: {not_subsets}")
        return self.add_subset(
            name,
            partial(_combined, rules=[self.meta(s)["select"] for s in spaces], reduce_op=self.OPS[op]),
            global_space=all(self.meta(s).get("global_space", False) for s in spaces),
            description=f"{op} of {', '.join(spaces)}",
        )


def _combined(ctx, pert, *, rules, reduce_op):
    # Canonicalise each selection to integer positions -- a rule may return a slice, and the set
    # operations need real indices. Every rule indexes the same full gene axis, so they compose.
    genes = np.arange(len(ctx.ds.var_names))
    result = genes[rules[0](ctx, pert)]
    for rule in rules[1:]:
        result = reduce_op(result, genes[rule(ctx, pert)])
    return result


SPACES = SpaceRegistry("space")
"""Every registered space, keyed by name (``"full"``, ``"heg_1000"``)."""


# =============================================================================
# Selection rules — (ctx, pert, value) -> a column selection into the full gene axis
# =============================================================================


def _all_genes(ctx, pert, value):
    return slice(None)  # a view, so the identity space costs nothing to apply


def _strongest_de(ctx, pert, k):
    return np.argsort(-np.abs(ctx.de(pert, ctx.cfg.truth).statistic))[:k]


def _significant_de(ctx, pert, padj):
    return np.where(ctx.de(pert, ctx.cfg.truth).pvalue_adj < padj)[0]


def _highest_expressed(ctx, pert, k):
    return np.argsort(-ctx.control_mean())[:k]


def _most_variable(ctx, pert, k):
    return np.argsort(-ctx.control_hvg_dispersion())[:k]


def _targeted_genes(ctx, pert, value):
    return ctx.perturbed_gene_indices()


def _principal_components(X, ctx, pert, k):
    return ctx.pca(k).transform(to_dense(X))[:, :k]


def _fit_pca(ctx, names):
    # sklearn's PCA is not basis-stable across n_components, so a smaller pca_k can't be sliced
    # out of a larger fit -- each size is fit and cached separately.
    for name in names:
        ctx.pca(int(name.rsplit("_", 1)[1]))


# =============================================================================
# Every space, one row each. To add one: write a rule above, add a row here.
# =============================================================================

FULL = Subset("full", _all_genes, None, "all genes, no transform")
TOP = Subset("top", _strongest_de, 50, "top {v} genes by ground-truth effect size, per perturbation", per_pert=True)
DEGS = Subset("degs", _significant_de, 0.05, "ground-truth DEGs at adjusted p < {v}, per perturbation", per_pert=True)
HEG = Subset("heg", _highest_expressed, 1000, "top {v} genes by control-condition expression")
HVG = Subset("hvg", _most_variable, 2000, "top {v} genes by control-condition normalized dispersion")
PERTURBED_GENES = Subset("perturbed_genes", _targeted_genes, None, "genes targeted by a perturbation in the dataset")

PCA = Transform("pca", _principal_components, 50, "top {v} principal components (fit on the dataset)", _fit_pca)

SUBSETS = [FULL, TOP, DEGS, HEG, HVG, PERTURBED_GENES]
"""Gene-subset spaces — they keep some genes and drop the rest, so :meth:`~scperteval.blocks.spaces.SpaceRegistry.combine` can
compose them."""

TRANSFORMS = [PCA]
"""Spaces that replace the gene axis rather than narrowing it. Not composable."""

DEFAULTS: list[str] = [SPACES.instance(s) for s in SUBSETS] + [SPACES.instance(s) for s in TRANSFORMS]
"""The instance of every space above registered at import, at its ``default`` value —
``["full", "top_50", "degs_0.05", "heg_1000", "hvg_2000", "perturbed_genes", "pca_50"]``. This is
what ``scperteval list spaces`` shows; other values register on demand via ``SPACES.instance(HEG, 250)``.
"""


# =============================================================================
# Composed panels — gene sets from the literature, built from the spaces above
# =============================================================================

MILLER_PANEL = SPACES.combine("miller_panel", SPACES.instance(HVG, 8192), SPACES.instance(PERTURBED_GENES))
"""The HVG ∪ perturbed-genes gene panel of :cite:t:`Miller_2025`, whose DRF calibrator
(:mod:`scperteval.calibrators`) and interpolated positive control
(``src_interpolated`` in :mod:`scperteval.sources`) scPertEval already implements. Registering it also
registers its ``hvg_8192`` constituent.
"""
