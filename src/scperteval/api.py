"""The native Python API — evaluate one protocol, or compute one DE method, at a time.

These functions mirror the CLI engine but take plain keyword arguments and return in-memory
results (pandas). They are re-exported at the package root, e.g. ``scperteval.calibrate(...)``.

Each call evaluates a **single** protocol (``calibrate``/``score``) or a **single** DE method
(``de``). To run several, call once per protocol/method — and reuse the expensive dataset/DE/PCA
setup across those calls with :func:`prepare`. ``prepare`` is optional: if you pass a path or an
AnnData directly, the context is built on the fly (with lazy caching), just less efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import pandas as pd

from . import io
from .blocks.de import DE_METHODS
from .context import Context
from .dataset import Dataset
from .predictions import PredictionSet
from .protocols.resolve import resolve_protocols
from .runner import compute_de, run_all
from .types import RunConfig

if TYPE_CHECKING:  # annotation-only; keeps ``import scperteval`` from eagerly importing anndata
    from anndata import AnnData

__all__ = [
    "DatasetDEResults",
    "EvalResult",
    "Prepared",
    "calibrate",
    "de",
    "prepare",
    "score",
]

#: The calibration outputs selectable from :func:`calibrate` (closed set).
CalibratorName = Literal["drf", "bds"]
#: The DE backends selectable from :func:`de` — mirrors the ``DE_METHODS`` registry's built-ins
#: (kept in sync by ``tests/test_api.py::test_de_method_literal_matches_registry``).
DEMethodName = Literal["t-test", "MWU", "t-test_overestim_var"]


# --------------------------------------------------------------------------- result types


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating one protocol on one dataset.

    Attributes
    ----------
    aggregate : dict of str to float
        The protocol's summary statistics — ``{"mean": …, "median": …}`` for ``drf``/``score``,
        ``{"bds": …}`` for ``bds``.
    per_perturbation : pandas.DataFrame
        One row per perturbation (raw control values + the calibrated score, or the raw metric) —
        the same layout the CLI writes to CSV.
    """

    aggregate: dict[str, float]
    per_perturbation: pd.DataFrame

    def __repr__(self) -> str:
        col = self.per_perturbation.get("perturbation")
        n = col.nunique() if col is not None else 0
        return f"EvalResult(aggregate={self.aggregate}, perturbations={n})"


class DatasetDEResults(NamedTuple):
    """Per-gene differential expression across the whole dataset, for one method.

    Both frames are indexed by perturbation with genes as columns.
    """

    #: Test statistic per (perturbation, gene).
    statistic: pd.DataFrame
    #: Benjamini-Hochberg adjusted p-value per (perturbation, gene).
    pvalue_adj: pd.DataFrame


# --------------------------------------------------------------------------- prepared handle


class Prepared:
    """A reusable, prepared dataset + context.

    Build it once with :func:`prepare`, then pass it to many :func:`calibrate` / :func:`score` /
    :func:`de` calls: the dataset read/index and the DE/reference/PCA caches are computed once and
    shared across those calls. Treat it as opaque — its internals are not part of the public API.

    Reuse is intended for sequential calls (or your own outer orchestration); a single handle is
    not designed to be driven by concurrent calls that use conflicting configuration.
    """

    __slots__ = ("_cfg", "_context")

    def __init__(self, context: Context, cfg: RunConfig):
        self._context = context
        self._cfg = cfg

    def __repr__(self) -> str:
        ds = self._context.ds
        return (
            f"Prepared(name={Path(self._cfg.dataset).stem!r}, "
            f"perturbations={len(ds.perturbations)}, de_method={self._cfg.de_method!r})"
        )


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


def _single_protocol(protocol: str):
    """Resolve one protocol spec to exactly one concrete protocol (error otherwise)."""
    protos = resolve_protocols([protocol])
    if len(protos) != 1:
        raise ValueError(
            f"the API evaluates one protocol per call; {protocol!r} resolves to {len(protos)} "
            f"protocols (pass a single name, e.g. 'pearson_ctrl' or 'mse_top_k=30')"
        )
    return protos[0]


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H%M%S")


def _context(data, *, name, de_method, subsample, seed, min_cells, perturbation_key, control_label, workers):
    """Return ``(context, base_cfg)`` — reuse a :class:`Prepared` handle, or build fresh (lazy).

    When ``data`` is a :class:`Prepared`, the dataset/DE configuration comes from the handle and
    the shaping keyword arguments here are ignored.
    """
    if isinstance(data, Prepared):
        return data._context, data._cfg
    cfg = RunConfig(
        dataset=_display_name(data, name),
        protocols=[],
        de_method=de_method,
        subsample=subsample,
        seed=seed,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
        workers=workers,
    )
    return Context(_to_dataset(data, cfg), cfg), cfg


# --------------------------------------------------------------------------- public functions


def prepare(
    dataset: str | Path | AnnData,
    *,
    de_method: DEMethodName = "t-test",
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    protocols: str | list[str] | None = None,
) -> Prepared:
    """Build a reusable context for a dataset, so many calls reuse one setup.

    Reads and indexes the dataset once and returns an opaque :class:`Prepared` handle to pass to
    :func:`calibrate` / :func:`score` / :func:`de`. Those calls then share the dataset and the
    DE/reference/PCA caches instead of rebuilding them each time. Optional — omit it and pass the
    dataset directly to a verb, which builds the context on the fly (lazily, less efficiently).

    Parameters
    ----------
    dataset : str or pathlib.Path or anndata.AnnData
        A preprocessed ``.h5ad`` path, or an in-memory AnnData.
    de_method : str, optional
        DE backend cached for DE-dependent work (default ``"t-test"``).
    subsample, seed, min_cells, perturbation_key, control_label, workers, name
        The same dataset/run knobs as the verbs; see :class:`~scperteval.types.RunConfig`.
    protocols : str or list of str, optional
        A hint: if given, pre-warm the shared singletons these protocols need (otherwise they are
        computed lazily on first use).

    Returns
    -------
    Prepared
        An opaque, reusable handle.
    """
    cfg = RunConfig(
        dataset=_display_name(dataset, name),
        protocols=[],
        de_method=de_method,
        subsample=subsample,
        seed=seed,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
        workers=workers,
    )
    ctx = Context(_to_dataset(dataset, cfg), cfg)
    if protocols is not None:
        specs = [protocols] if isinstance(protocols, str) else list(protocols)
        ctx.warm(resolve_protocols(specs))
    return Prepared(ctx, cfg)


def calibrate(
    data: str | Path | AnnData | Prepared,
    protocol: str,
    *,
    output: CalibratorName = "drf",
    positive: str = "auto",
    negative: str = "auto",
    de_method: DEMethodName = "t-test",
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> EvalResult:
    """Calibrate one protocol against the built-in positive/negative controls (DRF or BDS).

    Parameters
    ----------
    data : str or pathlib.Path or anndata.AnnData or Prepared
        A dataset (path/AnnData) or a :func:`prepare` handle.
    protocol : str
        A single protocol spec — a name (``"pearson_ctrl"``) or a tunable one (``"mse_top_k=30"``).
    output : {"drf", "bds"}, optional
        Which calibrator to apply (default ``"drf"``).
    positive, negative, de_method, subsample, seed, min_cells, perturbation_key, control_label, workers, name
        The same knobs as the CLI; see :class:`~scperteval.types.RunConfig`. When ``data`` is a
        :class:`Prepared` handle, the dataset/DE knobs come from the handle.
    out_dir : str or pathlib.Path, optional
        If given, also write the per-perturbation CSV there (as the CLI does).

    Returns
    -------
    EvalResult
        ``.aggregate`` (the protocol's summary stats) and ``.per_perturbation`` (the detail table).
    """
    if output not in ("drf", "bds"):
        raise ValueError(f"calibrate output must be 'drf' or 'bds', not {output!r} (use score() for predictions)")
    proto = _single_protocol(protocol)
    ctx, base = _context(
        data,
        name=name,
        de_method=de_method,
        subsample=subsample,
        seed=seed,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
        workers=workers,
    )
    cfg = replace(
        base,
        protocols=[proto.name],
        output=output,
        positive=positive,
        negative=negative,
        out_dir=str(out_dir) if out_dir is not None else "results",
    )
    ctx.cfg = cfg
    aggregates, rows, _ = run_all(cfg, [proto], ctx)
    if out_dir is not None:
        io.write_rows(cfg, rows, _stamp())
    return EvalResult(aggregate=aggregates[proto.name], per_perturbation=io.rows_frame(cfg, rows))


def score(
    data: str | Path | AnnData | Prepared,
    predictions: str | Path | AnnData,
    protocol: str,
    *,
    de_method: DEMethodName = "t-test",
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> EvalResult:
    """Score model predictions against ground truth for one protocol.

    Parameters
    ----------
    data : str or pathlib.Path or anndata.AnnData or Prepared
        The ground-truth dataset (path/AnnData) or a :func:`prepare` handle.
    predictions : str or pathlib.Path or anndata.AnnData
        Predicted cells — the same genes and perturbation labels as the dataset.
    protocol : str
        A single protocol spec (see :func:`calibrate`).
    de_method, subsample, seed, min_cells, perturbation_key, control_label, workers, name, out_dir
        As in :func:`calibrate`.

    Returns
    -------
    EvalResult
        ``.aggregate`` (mean/median raw metric) and ``.per_perturbation`` (the detail table).
    """
    proto = _single_protocol(protocol)
    ctx, base = _context(
        data,
        name=name,
        de_method=de_method,
        subsample=subsample,
        seed=seed,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
        workers=workers,
    )
    cfg = replace(
        base,
        protocols=[proto.name],
        output="score",
        truth="gt_all_cells",
        out_dir=str(out_dir) if out_dir is not None else "results",
    )
    ctx.cfg = cfg
    ctx.predictions = _to_predictions(predictions, ctx.ds, cfg)
    aggregates, rows, _ = run_all(cfg, [proto], ctx)
    if out_dir is not None:
        io.write_rows(cfg, rows, _stamp())
    return EvalResult(aggregate=aggregates[proto.name], per_perturbation=io.rows_frame(cfg, rows))


def de(
    data: str | Path | AnnData | Prepared,
    method: DEMethodName = "t-test",
    *,
    subsample: int = 8192,
    seed: int = 42,
    min_cells: int = 30,
    perturbation_key: str = "perturbation",
    control_label: str = "control",
    workers: int = 0,
    name: str | None = None,
    out_dir: str | Path | None = None,
) -> DatasetDEResults:
    """Compute per-gene differential expression (ground truth vs all-perturbed) for one method.

    Parameters
    ----------
    data : str or pathlib.Path or anndata.AnnData or Prepared
        A dataset (path/AnnData) or a :func:`prepare` handle.
    method : str, optional
        The DE backend (default ``"t-test"``).
    subsample, seed, min_cells, perturbation_key, control_label, workers, name
        The same dataset/run knobs as the verbs; ignored when ``data`` is a :class:`Prepared`
        handle (except ``method``, which always selects the backend).
    out_dir : str or pathlib.Path, optional
        If given, also write the HDF5 export there (as the CLI does).

    Returns
    -------
    DatasetDEResults
        ``.statistic`` and ``.pvalue_adj`` DataFrames (perturbations × genes).
    """
    if method not in DE_METHODS:
        raise ValueError(f"unknown DE method {method!r}; available: {', '.join(DE_METHODS.names())}")
    ctx, base = _context(
        data,
        name=name,
        de_method=method,
        subsample=subsample,
        seed=seed,
        min_cells=min_cells,
        perturbation_key=perturbation_key,
        control_label=control_label,
        workers=workers,
    )
    cfg = replace(base, de_method=method, out_dir=str(out_dir) if out_dir is not None else "results")
    ctx.cfg = cfg
    ctx._ensure_ref_sums()
    statistic, pvalue_adj = compute_de(ctx)
    perts = list(ctx.perturbations)
    genes = [str(g) for g in ctx.ds.var_names]
    result = DatasetDEResults(
        statistic=pd.DataFrame(statistic, index=perts, columns=genes),
        pvalue_adj=pd.DataFrame(pvalue_adj, index=perts, columns=genes),
    )
    if out_dir is not None:
        io.write_de(cfg, ctx.ds.var_names, ctx.perturbations, {method: (statistic, pvalue_adj)}, _stamp())
    return result
