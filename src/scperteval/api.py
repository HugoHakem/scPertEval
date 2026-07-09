"""The native Python API — run evaluation protocols and differential expression in-process.

These functions mirror the CLI (:mod:`scperteval.cli`) but take plain keyword arguments and
return in-memory results (pandas), so scPertEval can be used from a notebook or script without
shelling out. They are re-exported at the package root, e.g. ``scperteval.calibrate(...)``.

Everything here wraps the same engine the CLI uses; users never need to build a
:class:`~scperteval.types.RunConfig` or touch :class:`~scperteval.context.Context` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from . import io
from .blocks.de import DE_METHODS
from .blocks.spaces import SPACES
from .calibrators import CALIBRATORS
from .context import Context
from .dataset import Dataset
from .predictions import PredictionSet
from .protocols.resolve import resolve_protocols
from .protocols.table import TABLE
from .runner import compute_de_export, run_all
from .sources import SOURCES
from .types import RunConfig

if TYPE_CHECKING:  # annotation-only; keeps ``import scperteval`` from eagerly importing anndata
    from anndata import AnnData

__all__ = [
    "DEMethodResult",
    "DEResults",
    "EvalResult",
    "available_calibrators",
    "available_de_methods",
    "available_protocols",
    "available_sources",
    "available_spaces",
    "calibrate",
    "de",
    "differential_expression",
    "score",
]


# --------------------------------------------------------------------------- result types


@dataclass(frozen=True)
class EvalResult:
    """Results of a :func:`calibrate` or :func:`score` run.

    Attributes
    ----------
    summary : pandas.DataFrame
        One row per protocol, indexed by protocol name; columns are the calibrator's
        aggregate statistics (``mean``/``median`` for ``drf``/``score``, ``bds`` for ``bds``).
    per_perturbation : pandas.DataFrame
        One row per (protocol, perturbation) with the raw control values and the calibrated
        score — identical to the CSV the CLI writes.
    """

    summary: pd.DataFrame
    per_perturbation: pd.DataFrame

    def to_csv(self, path: str | Path) -> Path:
        """Write the ``per_perturbation`` table to ``path`` (the same layout the CLI writes)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.per_perturbation.to_csv(path, index=False)
        return path

    def __repr__(self) -> str:
        col = self.per_perturbation.get("perturbation")
        n = col.nunique() if col is not None else 0
        return f"EvalResult(protocols={list(self.summary.index)}, perturbations={n})"


@dataclass(frozen=True)
class DEMethodResult:
    """Per-gene differential expression for one method (perturbations x genes)."""

    #: Test statistic per (perturbation, gene).
    statistic: pd.DataFrame
    #: Benjamini-Hochberg adjusted p-value per (perturbation, gene).
    pvalue_adj: pd.DataFrame


class DEResults(Mapping):
    """Differential-expression results: a read-only mapping from method name to :class:`DEMethodResult`.

    Index it by method (``results["t-test"].statistic``) or iterate it to get the method names.
    The overrides below shadow :class:`~collections.abc.Mapping`'s inherited members with plain
    docstrings so the strict (``-W``) docs build stays clean.
    """

    def __init__(self, methods: dict[str, DEMethodResult]):
        self._methods = dict(methods)

    def keys(self):
        """The method names."""
        return self._methods.keys()

    def values(self):
        """The per-method :class:`DEMethodResult` objects."""
        return self._methods.values()

    def items(self):
        """``(method, DEMethodResult)`` pairs."""
        return self._methods.items()

    def get(self, method, default=None):
        """The result for ``method``, or ``default`` if it was not computed."""
        return self._methods.get(method, default)

    def __getitem__(self, key: str) -> DEMethodResult:
        return self._methods[key]

    def __iter__(self):
        return iter(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    def __repr__(self) -> str:
        any_result = next(iter(self._methods.values()), None)
        shape = any_result.statistic.shape if any_result is not None else (0, 0)
        return f"DEResults(methods={list(self._methods)}, perturbations={shape[0]}, genes={shape[1]})"


# --------------------------------------------------------------------------- helpers


def _display_name(dataset, name: str | None) -> str:
    """The label threaded into ``cfg.dataset`` (drives summary headers and output filenames)."""
    if name is not None:
        return name
    if isinstance(dataset, (str, Path)):
        return str(dataset)
    return "dataset"


def _to_dataset(dataset, cfg: RunConfig) -> Dataset:
    """Build a :class:`~scperteval.dataset.Dataset` from a path or an in-memory AnnData."""
    if isinstance(dataset, (str, Path)):
        return Dataset.load(str(dataset), cfg)
    return Dataset(dataset, cfg)  # an AnnData (referenced, never mutated)


def _to_predictions(predictions, ds: Dataset, cfg: RunConfig) -> PredictionSet:
    """Build a :class:`~scperteval.predictions.PredictionSet` from a path or an AnnData."""
    if isinstance(predictions, (str, Path)):
        return PredictionSet.load(str(predictions), ds, cfg)
    return PredictionSet(predictions, ds, cfg)


def _specs(protocols) -> list[str]:
    return [protocols] if isinstance(protocols, str) else list(protocols)


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H%M%S")


def _eval_result(cfg: RunConfig, aggregates: dict, rows: list, protocols) -> EvalResult:
    keys = sorted({k for v in aggregates.values() for k in v})
    summary = pd.DataFrame(
        {p.name: [aggregates.get(p.name, {}).get(k, float("nan")) for k in keys] for p in protocols},
    ).T
    summary.columns = keys
    summary.index.name = "protocol"
    return EvalResult(summary=summary, per_perturbation=io.rows_frame(cfg, rows))


# --------------------------------------------------------------------------- public functions


def calibrate(
    dataset: str | Path | AnnData,
    protocols: str | Iterable[str] = "all",
    *,
    de_method: str = "t-test",
    output: str | Iterable[str] = "drf",
    subsample: int = 8192,
    seed: int = 42,
    positive: str = "auto",
    negative: str = "auto",
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> EvalResult | dict[str, EvalResult]:
    """Calibrate protocols against built-in positive/negative controls (DRF or/and BDS).

    Parameters
    ----------
    dataset : str or pathlib.Path or anndata.AnnData
        A preprocessed ``.h5ad`` path, or an in-memory AnnData with the same layout.
    protocols : str or list of str, optional
        Protocol spec(s): ``"all"``, a group (``pseudobulk``/``distributional``/``de``), a
        protocol name, or a tunable one as ``name=value`` (e.g. ``"mse_top_k=30"``). Comma-
        separated tokens are allowed within a string. Defaults to ``"all"``.
    de_method : str, optional
        DE backend for every DE-dependent unit (default ``"t-test"``).
    output : {"drf", "bds"} or iterable of {"drf", "bds"}, optional
        Which calibrator(s) to apply (default ``"drf"``). Given more than one, the dataset is
        loaded and warmed only once and shared across every calibrator -- cheaper than calling
        :func:`calibrate` once per calibrator, which would redundantly repeat that precompute.
    subsample, seed, positive, negative, min_cells, perturbation_key, control_label, workers
        The same knobs as the CLI; see :class:`~scperteval.types.RunConfig`.
    name : str, optional
        Label used in the summary header and any written filename; defaults to the dataset
        path stem, or ``"dataset"`` for an in-memory AnnData.
    out_dir : str or pathlib.Path, optional
        If given, also write the per-perturbation CSV there (as the CLI does). By default
        nothing is written and results are returned in memory only.

    Returns
    -------
    EvalResult or dict[str, EvalResult]
        Aggregate ``summary`` and ``per_perturbation`` tables. A single :class:`EvalResult` for
        a single ``output``; a ``{output: EvalResult}`` dict when ``output`` is an iterable.
    """
    single = isinstance(output, str)
    outputs: list[str] = [output] if isinstance(output, str) else list(output)
    for o in outputs:
        if o not in ("drf", "bds"):
            raise ValueError(f"calibrate output must be 'drf' or 'bds', not {o!r} (use score() for predictions)")
    protos = resolve_protocols(_specs(protocols))
    cfg = RunConfig(
        dataset=_display_name(dataset, name),
        protocols=[p.name for p in protos],
        de_method=de_method,
        subsample=subsample,
        seed=seed,
        positive=positive,
        negative=negative,
        output=outputs[0],
        out_dir=str(out_dir) if out_dir is not None else "results",
        workers=workers,
        perturbation_key=perturbation_key,
        control_label=control_label,
        min_cells=min_cells,
    )
    ctx = Context(_to_dataset(dataset, cfg), cfg)
    results = {o: run_all(replace(cfg, output=o), protos, ctx) for o in outputs}
    eval_results = {}
    for o, (aggregates, rows, _) in results.items():
        o_cfg = replace(cfg, output=o)
        if out_dir is not None:
            io.write_rows(o_cfg, rows, _stamp())
        eval_results[o] = _eval_result(o_cfg, aggregates, rows, protos)
    return eval_results[outputs[0]] if single else eval_results


def score(
    dataset: str | Path | AnnData,
    predictions: str | Path | AnnData | Mapping[str, str | Path | AnnData],
    protocols: str | Iterable[str] = "all",
    *,
    de_method: str = "t-test",
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> EvalResult | dict[str, EvalResult]:
    """Score model predictions against ground truth (real cells), per protocol.

    Parameters
    ----------
    dataset : str or pathlib.Path or anndata.AnnData
        The ground-truth preprocessed ``.h5ad`` (or AnnData).
    predictions : str or pathlib.Path or anndata.AnnData, or a mapping to one of those
        Predicted cells — the same genes and perturbation labels as ``dataset``. Given a mapping
        of several prediction sets (e.g. one per model, keyed by name), the dataset is loaded
        and warmed only once and shared across every prediction set -- cheaper than calling
        :func:`score` once per model, which would redundantly repeat that precompute.
    protocols : str or list of str, optional
        Protocol spec(s); see :func:`calibrate`. Defaults to ``"all"``.
    de_method, subsample, seed, min_cells, perturbation_key, control_label, workers
        The same knobs as the CLI; see :class:`~scperteval.types.RunConfig`.
    name, out_dir
        As in :func:`calibrate`.

    Returns
    -------
    EvalResult or dict[str, EvalResult]
        Aggregate ``summary`` and ``per_perturbation`` tables (raw metric per perturbation). A
        single :class:`EvalResult` for a single ``predictions``; a ``{name: EvalResult}`` dict
        when ``predictions`` is a mapping.
    """
    single = not isinstance(predictions, Mapping)
    pred_map: dict[str, str | Path | AnnData] = (
        dict(predictions) if isinstance(predictions, Mapping) else {"predictions": predictions}
    )
    protos = resolve_protocols(_specs(protocols))
    cfg = RunConfig(
        dataset=_display_name(dataset, name),
        protocols=[p.name for p in protos],
        de_method=de_method,
        subsample=subsample,
        seed=seed,
        output="score",
        truth="gt_all_cells",
        out_dir=str(out_dir) if out_dir is not None else "results",
        workers=workers,
        perturbation_key=perturbation_key,
        control_label=control_label,
        min_cells=min_cells,
    )
    ds = _to_dataset(dataset, cfg)
    ctx = Context(ds, cfg)
    eval_results = {}
    for pred_name, preds in pred_map.items():
        ctx.predictions = _to_predictions(preds, ds, cfg)
        aggregates, rows, _ = run_all(cfg, protos, ctx)
        if out_dir is not None:
            # distinct filename per prediction set -- write_rows derives it from
            # Path(cfg.dataset).stem, so strip cfg.dataset's own extension first: appending
            # after it (".h5ad__gears") would make Path(...).stem swallow both as one suffix
            file_cfg = cfg if single else replace(cfg, dataset=f"{Path(cfg.dataset).stem}__{pred_name}")
            io.write_rows(file_cfg, rows, _stamp())
        eval_results[pred_name] = _eval_result(cfg, aggregates, rows, protos)
    return eval_results["predictions"] if single else eval_results


def differential_expression(
    dataset: str | Path | AnnData,
    methods: str | Sequence[str] = ("t-test", "MWU"),
    *,
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> DEResults:
    """Compute per-gene differential expression (ground-truth vs all-perturbed) per method.

    Parameters
    ----------
    dataset : str or pathlib.Path or anndata.AnnData
        A preprocessed ``.h5ad`` path, or an in-memory AnnData.
    methods : str or sequence of str, optional
        DE backends to compute (default ``("t-test", "MWU")``).
    subsample, seed, min_cells, perturbation_key, control_label, workers
        The same knobs as the CLI; see :class:`~scperteval.types.RunConfig`.
    name : str, optional
        Label used for any written filename (see :func:`calibrate`).
    out_dir : str or pathlib.Path, optional
        If given, also write the HDF5 export there (as the CLI does).

    Returns
    -------
    DEResults
        A mapping from method name to :class:`DEMethodResult` (statistic and adjusted-p
        DataFrames, perturbations x genes).
    """
    methods = [methods] if isinstance(methods, str) else list(methods)
    if not methods:
        raise ValueError("differential_expression requires at least one method")
    cfg = RunConfig(
        dataset=_display_name(dataset, name),
        protocols=[],
        de_method=methods[0],
        subsample=subsample,
        seed=seed,
        out_dir=str(out_dir) if out_dir is not None else "results",
        workers=workers,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
    )
    ctx = Context(_to_dataset(dataset, cfg), cfg)
    ctx._ensure_ref_sums()
    results = compute_de_export(ctx, methods)
    perts = list(ctx.perturbations)
    genes = [str(g) for g in ctx.ds.var_names]
    de_results = DEResults(
        {
            method: DEMethodResult(
                statistic=pd.DataFrame(stat, index=perts, columns=genes),
                pvalue_adj=pd.DataFrame(padj, index=perts, columns=genes),
            )
            for method, (stat, padj) in results.items()
        }
    )
    if out_dir is not None:
        io.write_de(cfg, ctx.ds.var_names, ctx.perturbations, results, _stamp())
    return de_results


#: Short alias for :func:`differential_expression` (matches the ``de`` CLI subcommand).
de = differential_expression


# --------------------------------------------------------------------------- discovery


def available_protocols() -> list[str]:
    """Names of the available evaluation protocols (see :func:`calibrate`'s ``protocols``)."""
    return [p.name for p in TABLE]


def available_de_methods() -> list[str]:
    """Names of the registered DE backends usable as ``de_method`` / ``methods``."""
    return list(DE_METHODS.names())


def available_spaces() -> list[str]:
    """Names of the registered feature spaces."""
    return list(SPACES.names())


def available_sources() -> list[str]:
    """Names of the registered control/reference sources."""
    return list(SOURCES.names())


def available_calibrators() -> list[str]:
    """Names of the available calibrators (``drf``/``bds`` for calibrate, ``score`` internally)."""
    return list(CALIBRATORS)
