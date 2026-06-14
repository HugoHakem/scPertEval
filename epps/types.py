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
    """An evaluation protocol: a pure algorithm plus its data and control wiring."""

    name: str
    algo: Callable
    kind: str
    space: str = "full"
    centering: Optional[str] = None
    reference: str = "all_perturbed"
    neg_reference: Optional[str] = None
    direction: str = "higher"
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
