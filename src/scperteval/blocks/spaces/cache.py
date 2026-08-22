"""Caching for the dataset-level computations a space rule depends on.

A rule runs once per perturbation per protocol, so anything computed over the whole dataset — a
per-gene statistic, a fitted basis — has to be computed once and reused. :func:`cached` does
that: decorate the computation, and it is evaluated once per prepared dataset and stored on the
handle's shared cache.

A space may also declare ``precompute=`` (see :class:`~scperteval.blocks.spaces.registry.Space`),
which only changes *when* that single evaluation happens — during
:meth:`~scperteval.context.Context.warm`, before the per-perturbation loop, rather than inside
it. Worth doing when the computation is heavy enough to want the machine's threads to itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

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
    ds: Any
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
        def control_dispersion(data):
            return ...  # data.ds, data.seed, data.threads


        control_dispersion(ctx)  # computed on first call, reused after
    """

    @wraps(fn)
    def call(ctx, *params):
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
