"""Core dataclasses shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np


@dataclass
class RunConfig:
    """Resolved options for a single run."""

    dataset: str
    protocols: list[str]
    de_method: str = "t-test"
    subsample: int = 8192
    seed: int = 42
    positive: str = "auto"
    negative: str = "auto"
    output: str = "drf"
    out_dir: str = "results"
    workers: int = 0
    min_cells: int = 30
    perturbation_key: str = "perturbation"
    control_label: str = "control"
    profile: bool = False


@dataclass(frozen=True)
class DEResult:
    """Per-gene differential expression for one target-vs-reference comparison."""

    score: np.ndarray
    pvalue: np.ndarray
    pvalue_adj: np.ndarray
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Param:
    """A protocol's tunable knob — how a CLI value (``k=30``, ``padj=0.05``) is cast,
    defaulted, and applied. ``space`` maps the value to a feature-space name; when it is
    ``None`` the value is passed straight to the metric as a keyword argument.
    """

    name: str
    cast: Callable
    default: float
    space: Callable | None = None


@dataclass(frozen=True)
class Protocol:
    """An evaluation protocol: a pure metric plus its data and control wiring.

    ``representation`` and ``scope`` are independent and together decide what the metric
    receives:

    - ``representation`` — the shape of one perturbation's datapoint: ``"centroid"`` (a 1-D
      pseudobulk vector), ``"population"`` (a cells × genes matrix), or ``"de"`` (a DEResult).
    - ``scope`` — ``"perturbation"`` (default): the metric is called once per perturbation,
      gets that perturbation's ``(gt, prediction)`` datapoints, and returns a scalar.
      ``"dataset"``: the metric is called once, gets the list of *every* perturbation's
      ``gt`` and ``prediction`` datapoints, and returns one score per perturbation (e.g. a
      cross-perturbation retrieval rank).

    Set ``param`` to make the protocol *tunable* — its feature space (or a metric argument)
    is then chosen per run from a CLI value, e.g. ``-p mse_top_k=30``; with no value the
    parameter's default is used. Leave ``param`` unset for a fully-specified protocol.

    ``better`` and ``perfect`` describe the metric's score scale and are independent:

    - ``better`` — which direction is an improvement, ``"higher"`` or ``"lower"``.
      Correlations and overlaps improve as they go up (``"higher"``); errors and
      distances improve as they go down (``"lower"``). This is the metric's *sense*,
      and it is not implied by ``perfect`` — e.g. perplexity has ``perfect=1.0`` yet
      ``better="lower"``, and a log-likelihood has ``perfect=0.0`` yet ``better="higher"``.
    - ``perfect`` — the value a flawless prediction attains (1.0 for a correlation,
      0.0 for an error). It anchors the top of the DRF scale.

    Together they let the calibrator orient the score: DRF measures how far the positive
    control moves from the negative-control floor toward ``perfect`` in the ``better``
    direction, and BDS counts the perturbations where the positive control is ``better``
    than the negative.
    """

    name: str
    metric: Callable
    representation: str
    scope: str = "perturbation"
    space: str = "full"
    centering: str | None = None
    reference: str = "all_perturbed"
    neg_reference: str | None = None
    better: str = "higher"
    perfect: float = 1.0
    positive: str = "auto"
    negative: str = "auto"
    group: str = ""
    param: Param | None = None

    @property
    def parameterised(self) -> bool:
        return self.param is not None

    def resolve(self, value) -> Protocol:
        """Concrete protocol for a tunable one at ``value`` (sets the space or metric arg)."""
        assert self.param is not None
        suffix = f"{value:g}" if isinstance(value, float) else str(value)
        name = f"{self.name}={suffix}"
        if self.param.space is not None:
            return replace(self, name=name, space=self.param.space(value), param=None)
        metric = partial(self.metric, **{self.param.name: value})
        return replace(self, name=name, metric=metric, param=None)


@dataclass(frozen=True)
class Calibrator:
    """Turns per-control raw metric values into per-perturbation and aggregate scores."""

    name: str
    requires: tuple[str, ...]
    per_pert: Callable
    aggregate: Callable
    description: str = ""
