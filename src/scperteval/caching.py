"""Caching for dataset-level computations shared across every call against one prepared dataset.

Anything computed over the whole dataset -- a per-gene statistic, a fitted basis -- has to be
computed once and reused, whichever building block needs it: a space rule, a centering source, a
metric. :func:`cached` does that: decorate the computation, and it is evaluated once per prepared
dataset and stored on the handle's shared :class:`~scperteval.context.CacheStore`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import Context
    from .dataset import Dataset

_MISSING = object()  # distinct from None, which a helper may legitimately return


@dataclass(frozen=True)
class DatasetScope:
    """Everything a cached computation is allowed to read.

    The dataset, plus the settings fixed when the handle was prepared. Per-call configuration
    (``de_method``, ``calibrator``, ``truth``) is deliberately absent: one cache is shared by
    every call against a prepared dataset, so a value that varied with those would be served to a
    call that set them differently.
    """

    #: The prepared dataset.
    ds: Dataset
    #: Reproducibility seed (``prepare(seed=...)``).
    seed: int
    #: Worker threads this run may use — the budget for BLAS-parallel work.
    threads: int
    #: Cell cap for sampled populations (``prepare(subsample=...)``).
    subsample: int


def cached(fn: Callable) -> Callable:
    """Compute a dataset-level value once per prepared dataset and reuse it.

    Call the wrapped function as ``fn(ctx, *params)``; its body receives a :class:`DatasetScope`
    in place of ``ctx``, so it can only depend on things the cache is valid over. Results are
    keyed by ``(function, params)``, so a parameterised computation caches one value per
    parameter.

    Example
    -------
    ::

        @cached
        def control_dispersion(scope: DatasetScope):
            return ...  # scope.ds, scope.seed, scope.threads


        control_dispersion(ctx)  # computed on first call, reused after
    """

    @wraps(fn)
    def call(ctx: Context, *params: Any) -> Any:
        key, store = (fn, params), ctx._store
        value = store.memo.get(key, _MISSING)
        if value is _MISSING:
            with store.lock:  # re-check under the lock: another thread may have filled it
                value = store.memo.get(key, _MISSING)
                if value is _MISSING:
                    value = fn(ctx.scope(), *params)
                    store.memo[key] = value
        return value

    return call
