"""The feature-space registry: the catalog of space definitions, and the instances built from them.

Two levels, deliberately distinct:

- A **definition** is one entry in the catalog — a rule plus how to present it. There are a
  handful, declared by decorating a rule in ``catalog.py``.
  ``scperteval list spaces`` shows these.
- An **instance** is a definition at one parameter value, registered under a concrete name
  (``"heg_1000"``). A protocol names its space as a string, so an instance must exist before a
  run can resolve it. :meth:`SpaceRegistry.instance` creates them, and nothing is instantiated
  except what something actually references.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...dataset import to_dense
from ...registry import Registry


@dataclass(frozen=True)
class SetOps:
    """The set operations :func:`combine_subsets` folds with, named as Python's own set methods.

    Exposed as the singleton ``OPS``, so composing a space reads
    ``combine_subsets(ctx, OPS.union, ...)`` without importing numpy.
    """

    #: Genes in either selection.
    union: Callable = np.union1d
    #: Genes in both selections.
    intersection: Callable = np.intersect1d
    #: Genes in the first selection but not the rest.
    difference: Callable = np.setdiff1d


OPS = SetOps()
"""The set operations available to :func:`combine_subsets` — ``OPS.union``, ``OPS.intersection``,
``OPS.difference``."""


def combine_subsets(ctx, op: Callable, *selections):
    """Fold gene selections together with a set operation from ``OPS``.

    A composed space is an ordinary subset rule that calls the rules it composes, so composition
    needs no machinery and nests to any depth. Selections may be slices; each is canonicalised to
    integer positions first, since the set operations need real indices and every rule indexes the
    same full gene axis.

    Parameters
    ----------
    ctx : ~scperteval.context.Context
        Supplies the gene axis the selections index into.
    op : Callable
        One of ``OPS`` — ``OPS.union``, ``OPS.intersection``, or ``OPS.difference``. Applied
        left to right, so ``OPS.difference`` subtracts the rest from the first.
    *selections
        Two or more already-computed selections.

    Returns
    -------
    numpy.ndarray
        Integer gene positions.

    Examples
    --------
    The HVG panel unioned with the targeted genes, and the complement of the HVG panel::

        combine_subsets(ctx, OPS.union, hvg(ctx, pert, 8192), perturbed_genes(ctx, pert))
        combine_subsets(ctx, OPS.difference, full(ctx, pert), hvg(ctx, pert, 2000))
    """
    genes = np.arange(len(ctx.ds.var_names))
    result = genes[selections[0]]
    for selection in selections[1:]:
        result = op(result, genes[selection])
    return result


def _parameter_of(rule: Callable, lead: int) -> str | None:
    """The rule's parameter name, or ``None`` if it takes none.

    ``lead`` counts the fixed leading arguments — 2 for a selection rule ``(ctx, pert, …)``, 3 for
    a transform ``(X, ctx, pert, …)``. A trailing argument with a default means "no parameter", so
    ``full(ctx, pert, value=None)`` is unparameterised while ``heg(ctx, pert, k)`` takes ``k``.
    """
    params = list(inspect.signature(rule).parameters.values())
    if len(params) <= lead:
        return None
    tail = params[lead]
    return None if tail.default is not inspect.Parameter.empty else tail.name


@dataclass(frozen=True)
class Space:
    """One entry in the space catalog — what a decorated rule becomes."""

    #: Catalog name (``"heg"``). Instances are ``"<name>_<value>"``, or ``"<name>"`` unparameterised.
    name: str
    #: The rule: ``(ctx, pert, value)`` for a subset, ``(X, ctx, pert, value)`` for a transform.
    rule: Callable
    #: The rule's parameter name (``"k"``), read from its signature; ``None`` if it takes none.
    parameter: str | None
    #: Parameter value used when a caller doesn't supply one; ``None`` iff unparameterised.
    default: Any
    #: Human-readable, with ``{v}`` standing in for the parameter.
    description: str
    #: Whether the selection depends on which perturbation is scored (subsets only).
    per_pert: bool = False
    #: ``False`` for a space that replaces the gene axis rather than narrowing it.
    is_subset: bool = True
    #: Optional ``prepare(ctx, names)`` warm-up hook (transforms only).
    prepare: Callable | None = None

    @property
    def label(self) -> str:
        """How the space is written in listings and docs — ``"heg_<k>"`` or ``"full"``."""
        return f"{self.name}_<{self.parameter}>" if self.parameter else self.name

    def describe(self, value=None) -> str:
        """The description with ``{v}`` filled in — by ``value``, or by the parameter's name."""
        return self.description.format(v=f"{value:g}" if value is not None else self.parameter or "")


class SpaceRegistry(Registry):
    """A :class:`~scperteval.registry.Registry` that also holds the catalog spaces are built from.

    Add a space by decorating its rule with :meth:`SpaceRegistry.subset` or :meth:`SpaceRegistry.transform`, exactly as
    ``DE_METHODS`` and ``SOURCES`` are extended. :meth:`SpaceRegistry.instance` then builds a named instance on
    demand; the inherited ``__getitem__`` / ``meta`` / ``names`` see instances only.
    """

    def __init__(self, kind: str):
        super().__init__(kind)
        self._catalog: dict[str, Space] = {}

    # -- defining ------------------------------------------------------------------

    def subset(self, name: str, *, default=None, description="", per_pert=False) -> Callable:
        """Decorator: define a space that keeps a subset of the genes.

        The rule is ``(ctx, pert, value) -> column selection into the full gene axis`` — an
        integer array, or a slice. Give the trailing argument a default to declare that the space
        takes no parameter. Subsets can be folded together with
        ``combine_subsets``.

        Parameters
        ----------
        name : str
            Catalog name; instances are ``"<name>_<value>"``.
        default : optional
            Parameter value used when a caller doesn't supply one. Required if the rule takes a
            parameter, and must be omitted if it doesn't.
        description : str
            Shown by ``scperteval list spaces``; ``{v}`` stands in for the parameter.
        per_pert : bool
            ``True`` if the selection depends on which perturbation is being scored, so it can't
            be computed once and shared.
        """

        def deco(rule: Callable) -> Callable:
            self._define(Space(name, rule, _parameter_of(rule, 2), default, description, per_pert))
            return rule

        return deco

    def transform(self, name: str, *, default=None, description="", prepare=None) -> Callable:
        """Decorator: define a space that replaces the gene axis instead of narrowing it.

        The rule is ``(X, ctx, pert, value) -> dense cells × features array``, built directly, so
        the space has no gene selection and cannot be composed. ``prepare(ctx, names)`` optionally
        builds shared structure once before a run (see :meth:`~scperteval.context.Context.warm`);
        it is purely an optimisation and must be idempotent.
        """

        def deco(rule: Callable) -> Callable:
            self._define(Space(name, rule, _parameter_of(rule, 3), default, description, False, False, prepare))
            return rule

        return deco

    def _define(self, space: Space) -> None:
        if (space.parameter is None) != (space.default is None):
            raise TypeError(
                f"{self.kind} {space.name!r}: a rule taking a parameter needs a default and one "
                f"taking none must not have one (parameter={space.parameter!r}, default={space.default!r})"
            )
        self._catalog[space.name] = space

    def catalog(self) -> list[Space]:
        """Every defined space, by name — the palette, not the registered instances."""
        return [self._catalog[n] for n in sorted(self._catalog)]

    # -- instantiating -------------------------------------------------------------

    def instance(self, name: str, value=None) -> str:
        """Register one variant of a defined space and return its name, e.g. ``"heg_250"``.

        Idempotent: a variant already registered at the same value is reused. Omit ``value`` for
        the space's default.
        """
        if name not in self._catalog:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._catalog)}")
        space = self._catalog[name]
        if space.parameter is None:
            if value is not None:
                raise TypeError(f"{self.kind} {name!r} takes no parameter, got {value!r}")
            key = name
        else:
            value = space.default if value is None else value
            key = f"{name}_{value:g}"
        if key in self:
            # Distinct values can format to the same name (0.05 and 0.05000001 are both "0.05").
            # Registering the second silently under the first's rule would score the wrong genes.
            registered = self.meta(key).get("value")
            if registered != value:
                raise ValueError(f"{key!r} is already registered with value {registered!r}, not {value!r}")
            return key
        common = dict(description=space.describe(value), value=value)
        if space.is_subset:
            select = _bind_select(space.rule, value)

            def apply(X, ctx, pert):
                return to_dense(X[:, select(ctx, pert)])

            self.add(key, apply, select=select, global_space=not space.per_pert, **common)
        else:
            self.add(key, _bind_transform(space.rule, value), global_space=True, prepare=space.prepare, **common)
        return key


def _bind_select(rule, value):
    def select(ctx, pert):
        return rule(ctx, pert, value)

    return select


def _bind_transform(rule, value):
    def apply(X, ctx, pert):
        return rule(X, ctx, pert, value)

    return apply


SPACES = SpaceRegistry("space")
"""The feature-space registry: the catalog, and the instances registered from it."""
