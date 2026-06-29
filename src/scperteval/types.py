"""Core dataclasses shared across the package."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np


@dataclass
class RunConfig:
    """Resolved options for a single run.

    Attributes
    ----------
    dataset : str
        Path to the preprocessed ``.h5ad`` file.
    protocols : list of str
        Names of the resolved (concrete) protocols to run.
    de_method : str
        DE backend for every DE-dependent unit (default ``"t-test"``).
    subsample : int
        Number of cells in the all-perturbed reference sample (default 8192).
    seed : int
        Random seed for subsampling and reproducibility (default 42).
    positive : str
        Override the positive control source; ``"auto"`` defers to the protocol.
    negative : str
        Override the negative control source; ``"auto"`` defers to the protocol.
    output : str
        Calibrator to apply — ``"drf"`` or ``"bds"`` (default ``"drf"``).
    out_dir : str
        Directory for output CSV files (default ``"results"``).
    workers : int
        Number of worker threads; 0 auto-detects (default 0).
    min_cells : int
        Minimum cells required to evaluate a perturbation (default 30).
    perturbation_key : str
        ``adata.obs`` column holding perturbation labels (default ``"perturbation"``).
    control_label : str
        Label in ``perturbation_key`` that identifies control cells (default ``"control"``).
    profile : bool
        If ``True``, also write a per-protocol wall-clock timing CSV.
    """

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
    """Per-gene differential expression for one target-vs-reference comparison.

    Attributes
    ----------
    score : numpy.ndarray
        Per-gene test statistic (e.g. t-statistic or Cliff's delta), shape ``(G,)``.
    pvalue : numpy.ndarray
        Raw per-gene p-values, shape ``(G,)``.
    pvalue_adj : numpy.ndarray
        Benjamini-Hochberg adjusted p-values, shape ``(G,)``.
    extra : dict
        Optional method-specific extras (e.g. ``{"u": u_statistic}`` for MWU).
    """

    score: np.ndarray
    pvalue: np.ndarray
    pvalue_adj: np.ndarray
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Param:
    """A protocol's tunable knob — how a CLI value is cast, defaulted, and applied.

    ``space`` maps the value to a feature-space name; when it is ``None`` the value is
    passed straight to the metric as a keyword argument.

    Attributes
    ----------
    name : str
        Keyword argument name passed to the metric or space factory (e.g. ``"k"``).
    cast : Callable
        Type cast applied to the CLI string (e.g. ``int``, ``float``).
    default : float
        Default value used when no value is given on the CLI.
    space : Callable or None
        Factory mapping the value to a feature-space name (e.g. ``top_space``);
        ``None`` means the value is passed directly to the metric as a keyword argument.
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
      distances improve as they go down (``"lower"``).
    - ``perfect`` — the value a flawless prediction attains (1.0 for a correlation,
      0.0 for an error). It anchors the top of the DRF scale.

    Attributes
    ----------
    name : str
        Protocol identifier used on the CLI (``-p name``) and in output CSVs.
    metric : Callable
        Pure metric function ``(gt, prediction, ctx) -> float``.
    representation : str
        Shape of each datapoint: ``"centroid"``, ``"population"``, or ``"de"``.
    scope : str
        ``"perturbation"`` (default) or ``"dataset"`` — how many perturbations the metric sees at once.
    space : str
        Feature space applied before scoring (default ``"full"``).
    centering : str or None
        Baseline subtracted before scoring — ``"ctrl"``, ``"allpert"``, or ``None``.
    reference : str
        Source used as the reference for the GT DE computation (default ``"all_perturbed"``).
    neg_reference : str or None
        Reference for the negative-control DE computation; ``None`` uses ``reference``.
    better : str
        ``"higher"`` or ``"lower"`` — which direction improves the score.
    perfect : float
        Score a flawless prediction attains (e.g. 1.0 for correlations, 0.0 for errors).
    positive : str
        Positive control source name (default ``"auto"``, deferring to the protocol).
    negative : str
        Negative control source name (default ``"auto"``, deferring to the protocol).
    group : str
        Display group for ``scperteval list protocols`` (e.g. ``"pseudobulk"``).
    param : ~scperteval.types.Param or None
        If set, makes the protocol tunable from the CLI; ``None`` for fixed protocols.
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
    """Turns per-control raw metric values into per-perturbation and aggregate scores.

    Attributes
    ----------
    name : str
        Registry key and output column name (e.g. ``"drf"``).
    requires : tuple of str
        Control roles needed — typically ``("positive", "negative")``.
    per_pert : Callable
        ``(raws: dict, protocol: Protocol) -> float`` — combines raw control values
        into one per-perturbation calibrated score.
    aggregate : Callable
        ``(values: numpy.ndarray) -> dict`` — reduces per-perturbation scores into
        summary statistics (e.g. ``{"mean": …, "median": …}``).
    description : str
        Human-readable description shown by ``scperteval list calibrators``.
    """

    name: str
    requires: tuple[str, ...]
    per_pert: Callable
    aggregate: Callable
    description: str = ""
