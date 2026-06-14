"""Core dataclasses shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

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
class Protocol:
    """An evaluation protocol: a pure metric plus its data and control wiring.

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
    space: str = "full"
    centering: Optional[str] = None
    reference: str = "all_perturbed"
    neg_reference: Optional[str] = None
    better: str = "higher"
    perfect: float = 1.0
    positive: str = "auto"
    negative: str = "auto"
    group: str = ""


@dataclass(frozen=True)
class Calibrator:
    """Turns per-control raw metric values into per-perturbation and aggregate scores."""

    name: str
    requires: tuple[str, ...]
    per_pert: Callable
    aggregate: Callable
    description: str = ""
