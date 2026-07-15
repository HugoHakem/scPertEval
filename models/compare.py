"""Score every model against the same ground truth under both papers' native protocols.

Compares every trained model in models/ (gears, scgpt, presage, sclambda, state, tabicl) plus
mean, no_change, and linear against Miller et al. 2025's and Ahlmann-Eltze et al. 2025's native
protocols, to show the ranking flip that motivates scPertEval — see
docs/user-guide/reproducing-literature-protocols.md.

Concatenates each model's per-fold test predictions (models/data/prepare_split.py's folds) into
one full-coverage prediction file, builds the two baselines directly from scPertEval's own
``Dataset`` (models/baselines/baselines.py), then reuses scPertEval's own scoring internals
(the same code path ``scperteval score`` runs) for the raw per-protocol score, and Miller et
al. 2025's/Ahlmann-Eltze et al. 2025's own convention for the model-comparison DRF/CI/
significance tables: the fold-scoped ``mean`` baseline as comparator for most metrics, but the
``no_change`` (control) baseline specifically for the ``allpert``-centered "delta pert" family
(see :func:`baseline_for`) -- not whichever native control a protocol happens to use for its
own calibration.

Run with the main project's environment, not a models/ pixi env — this needs ``scperteval``
installed, not GEARS/scGPT/torch::

    python models/compare.py
"""

from __future__ import annotations

import pickle
import sys
from functools import partial
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # baselines/ is a sibling package, not installed

from baselines.baselines import BASELINES, FOLD_SCOPED_BASELINES, build_predictions  # noqa: E402

import scperteval as sp  # noqa: E402
from scperteval.calibrators import drf_per_pert  # noqa: E402
from scperteval.protocols.resolve import resolve_protocols  # noqa: E402

RAW = HERE / "data" / "smoke_k562" / "raw.h5ad"
FOLD_DIR = HERE / "data" / "smoke_k562" / "folds"
N_FOLDS = 3
TRAINED_MODELS = {
    "gears": HERE / "gears",
    "scgpt": HERE / "scgpt",
    "presage": HERE / "presage",
    "sclambda": HERE / "sclambda",
    "state": HERE / "state",
    "tabicl": HERE / "tabicl",
}


def _paired_diff_per_pert(raws, p):
    """Per-perturbation gap, signed so positive means the ``positive`` role wins.

    Local copy of the package's former private helper of the same name -- moved here since this
    module is its only real consumer (the pairwise comparison between two independently-scored
    prediction sets that scPertEval's own role resolution can't express directly).
    """
    pos, neg = raws["positive"], raws["negative"]
    if not (np.isfinite(pos) and np.isfinite(neg)):
        return float("nan")
    return float(neg - pos) if p.better == "lower" else float(pos - neg)

# Each protocol's origin (see docs/user-guide/reproducing-literature-protocols.md). Miller et
# al. 2025's four delta-correlation metrics (PearsonDeltaCtrl, PearsonDeltaPerturbMean,
# R2DeltaCtrl, R2DeltaPerturbMean) each come in three gene-set variants — all genes, DEG
# (padj < 0.05), and a DE-effect-size-weighted variant (Vollenweider & Bühlmann 2026's
# "Weighted Pearson Delta" for the two Pearson ones, Miller's own "continuous weighting" for
# the two R2 ones). MSE has the same three-way split (mse / mse_degs_padj / wmse_exp2).
PROTOCOL_SPECS = [
    # --- Ahlmann-Eltze et al. 2025 ---
    "pearson_ctrl_expr_k=1000",  # headline "Pearson Delta"
    "l2_expr_k=1000",  # companion l2 metric
    # --- Miller et al. 2025: MSE (all genes / DEG / DE-weighted) ---
    "mse",
    "mse_degs_padj",
    "wmse_exp2",
    # --- Miller et al. 2025: PearsonDeltaCtrl ---
    "pearson_ctrl",
    "pearson_ctrl_degs_padj",
    "weighted_pearson_ctrl_exp2",  # Vollenweider & Bühlmann 2026's "Weighted Pearson Delta"
    # --- Miller et al. 2025: PearsonDeltaPerturbMean ---
    "pearson_pert",
    "pearson_pert_degs_padj",
    "weighted_pearson_pert_exp2",
    # --- Miller et al. 2025: R2DeltaCtrl ---
    "r2_ctrl",
    "r2_ctrl_degs_padj",
    "weighted_r2_ctrl_exp2",
    # --- Miller et al. 2025: R2DeltaPerturbMean ---
    "r2_pert",
    "r2_pert_degs_padj",
    "weighted_r2_pert_exp2",
    # --- Miller et al. 2025: Normalized Inverse Rank ---
    "nir",
]


def concat_fold_predictions(model_dir: Path, n_folds: int = N_FOLDS) -> ad.AnnData:
    """One model's per-fold test predictions, concatenated into full gene-coverage."""
    parts = [ad.read_h5ad(model_dir / "smoke_data" / f"smoke_k562_predictions_fold{i}.h5ad") for i in range(n_folds)]
    return ad.concat(parts)


def concat_fold_baseline(fold_baseline_fn, n_folds: int = N_FOLDS) -> ad.AnnData:
    """A fold-aware baseline's per-fold predictions, concatenated across folds.

    Works for :func:`baselines.fold_mean_baseline` or :func:`baselines.linear_baseline`,
    concatenated the same way a real model's per-fold predictions are (see
    :func:`concat_fold_predictions`) — each fold fit only on that fold's own training
    perturbations.
    """
    parts = []
    for i in range(n_folds):
        with (FOLD_DIR / f"fold_{i}.pkl").open("rb") as f:
            fold = pickle.load(f)
        parts.append(fold_baseline_fn(str(RAW), fold))
    return ad.concat(parts)


def build_all_predictions() -> dict[str, Path]:
    """Every model's full-coverage predictions, written to disk and returned as paths.

    Written out rather than kept in memory so each is also a standalone artifact —
    spot-checkable with the plain ``scperteval score`` CLI.
    """
    paths = {}
    for name, model_dir in TRAINED_MODELS.items():
        out = model_dir / "smoke_data" / "smoke_k562_predictions_all.h5ad"
        concat_fold_predictions(model_dir).write_h5ad(out)
        paths[name] = out

    baselines_dir = HERE / "baselines" / "smoke_data"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    for name, fold_baseline_fn in FOLD_SCOPED_BASELINES.items():
        out = baselines_dir / f"smoke_k562_predictions_{name}.h5ad"
        concat_fold_baseline(fold_baseline_fn).write_h5ad(out)
        paths[name] = out
    for name in BASELINES:
        if name in FOLD_SCOPED_BASELINES:
            continue  # superseded by the fold-scoped version above
        out = baselines_dir / f"smoke_k562_predictions_{name}.h5ad"
        build_predictions(name, str(RAW)).write_h5ad(out)
        paths[name] = out
    return paths


def calibration_scores(protocols) -> list[dict]:
    """Two model-independent views of whether this protocol can tell signal from noise at all.

    Positive/negative are the protocol's own built-in controls (interpolated duplicate vs.
    its native negative) -- no predictions involved (``scperteval.calibrate``). One
    ``sp.calibrate(..., output=["drf", "bds"])`` call shares a single ``Dataset``/``Context``/warm
    pass across both calibrators (PR #11's batching), instead of one call -- and one warm -- per
    calibrator:

    - ``drf``: Miller et al. 2025's Dynamic Range Fraction -- how much of the gap toward the
      metric's theoretical perfect score does the positive control close, on average.
    - ``bds``: Vollenweider & Bühlmann 2026's Bound Discrimination Score -- the fraction of
      *individual perturbations* where the positive control actually wins. A protocol can have
      a healthy mean/median DRF while still having a middling BDS, if the per-perturbation gap
      is noisy rather than consistent -- DRF alone can hide that.
    """
    names = [p.name for p in protocols]
    results = sp.calibrate(str(RAW), protocols=names, output=["drf", "bds"])
    rows = []
    for calibrator_name, result in results.items():
        for protocol_name, agg in result.summary.iterrows():
            rows.append({"protocol": protocol_name, "calibrator": calibrator_name, **agg.to_dict()})
    return rows


def protocol_gene_counts(protocols) -> pd.DataFrame:
    """Per-protocol number of genes contributing to its score.

    Every protocol's own DE reference is ``all_perturbed`` (and ``Context.wmse_weights`` is
    itself built from that same ground-truth-vs-all-perturbed comparison), so every gene-count
    below is read straight off one ``scperteval.differential_expression`` call instead of
    reaching into ``Context``/``Dataset`` internals -- the public API already has everything
    this needs.

    Fixed-size feature spaces (``expr_<k>``, ``top_<k>``) and the unrestricted ``full`` space
    report one constant, the same for every perturbation. DEGs-threshold spaces
    (``degs_<padj>``) vary per perturbation -- the ground-truth DEG count at that padj differs
    perturbation to perturbation. Continuously-weighted metrics (``wmse``/weighted Pearson/R2)
    have no hard cutoff at all, so there is no literal gene count -- ``sum(weight) / max(weight)``
    gives an effective one instead: since weights are min-max normalised to a max of 1, this is
    "how many full-weight genes' worth of contribution" the metric integrates over. Both variable
    cases are summarised by their median and [P25, P75] across perturbations.
    """
    de = sp.differential_expression(str(RAW), methods=["t-test"])["t-test"]
    perts = list(de.statistic.index)

    def wmse_weight(pert: str, exp: float) -> np.ndarray:
        """Mirrors ``Context.wmse_weights``: min-max normalised |effect size|, then **exp."""
        s = np.abs(de.statistic.loc[pert].to_numpy())
        finite = np.isfinite(s)
        if not finite.any():
            return np.zeros_like(s)
        lo, hi = s[finite].min(), s[finite].max()
        w = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        return np.nan_to_num(w, nan=0.0) ** exp

    rows = []
    for p in protocols:
        if isinstance(p.metric, partial) and "exp" in p.metric.keywords:
            exp = p.metric.keywords["exp"]
            counts = [float(w.sum() / w.max()) for pert in perts if (w := wmse_weight(pert, exp)).max() > 0]
            kind = "weighted (effective)"
        elif p.space.startswith("degs_"):
            padj = float(p.space.split("_", 1)[1])
            counts = [float(np.sum(de.pvalue_adj.loc[pert].to_numpy() < padj)) for pert in perts]
            kind = "DEGs (padj threshold)"
        elif p.space == "full":
            counts = [float(de.statistic.shape[1])]
            kind = "all genes"
        else:  # expr_<k> / top_<k> fixed-size spaces
            counts = [float(p.space.split("_")[-1])]
            kind = "fixed top-k"
        arr = np.asarray(counts, dtype=float)
        rows.append(
            {
                "protocol": p.name,
                "kind": kind,
                "n_median": float(np.median(arr)) if arr.size else float("nan"),
                "n_p25": float(np.percentile(arr, 25)) if arr.size else float("nan"),
                "n_p75": float(np.percentile(arr, 75)) if arr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def per_pert_raw_scores(predictions: dict[str, Path], protocols) -> dict[str, dict[str, dict[str, float]]]:
    """Every prediction set's raw per-perturbation score, for every protocol.

    ``{name: {protocol_name: {perturbation: raw_value}}}`` — one ``sp.score(..., predictions={...})``
    call (PR #11's batching) shares a single ``Dataset``/``Context``/warm pass across every
    prediction set, instead of one call -- and one warm -- per model/baseline. Each
    ``EvalResult.per_perturbation`` table (the same ``score`` calibrator the CLI's
    ``scperteval score`` uses) is reshaped from a flat per-(protocol, perturbation) frame into a
    nested dict so :func:`per_pert_drf` can pair two prediction sets' values by perturbation
    afterwards.
    """
    names = [p.name for p in protocols]
    results = sp.score(str(RAW), dict(predictions), protocols=names)
    out = {}
    for pred_name, result in results.items():
        by_protocol = result.per_perturbation.groupby("protocol")
        out[pred_name] = {
            protocol_name: dict(zip(g["perturbation"], g["raw_prediction"], strict=True))
            for protocol_name, g in by_protocol
        }
    return out

def baseline_for(p) -> str:
    """Which fitted baseline plays the paper's own comparator role for this protocol.

    Miller et al. 2025's and Ahlmann-Eltze et al. 2025's own convention: the (fold-scoped)
    mean baseline for most metrics, but the control baseline specifically for the
    ``allpert``-centered "delta pert" family -- matching scPertEval's own native
    ``p.negative`` wiring for these same protocols (``all_perturbed_mean``/``global_mean`` vs
    ``control`` in table.py's ``_PB``/``_PB_CTRL``/``_NIR`` bundles).
    """
    return "no_change" if p.negative == "control" else "mean"

def per_pert_drf(
    model_name: str, raw: dict[str, dict[str, dict[str, float]]], protocols
) -> dict[str, dict[str, float]]:
    """Per-protocol, per-perturbation DRF for one model against its own baseline comparator.

    ``{protocol_name: {perturbation: drf}}`` -- keyed by perturbation so DRF values can be
    aligned across models within one perturbation, e.g. to rank models per perturbation
    (:func:`~models.plots.compute_avg_drf_ranks`), or read as a plain array by callers that only
    need the values (models/plots.py's heatmap/forest figures). Omits protocols where
    ``model_name`` *is* the comparator (``model_name == baseline_for(p)``, e.g. comparing the mean
    baseline to itself is degenerate). The combine formula itself (``drf_per_pert``) is the
    package's own -- what's left here is just picking which two already-scored prediction sets to
    compare, which is orchestration/policy (:func:`baseline_for`), not something the package can
    decide on its own. The one place the pairing between two prediction files' raw scores happens,
    since scPertEval's role resolution can only point ``negative`` at a dataset-native source or
    the single loaded ``prediction`` set, never at a second prediction file.
    """
    out = {}
    for p in protocols:
        baseline_name = baseline_for(p)
        if model_name == baseline_name:
            continue
        m_scores, b_scores = raw[model_name][p.name], raw[baseline_name][p.name]
        drf = {}
        for pert in sorted(set(m_scores) & set(b_scores)):
            raws = {"positive": m_scores[pert], "negative": b_scores[pert]}
            if np.isfinite(raws["positive"]) and np.isfinite(raws["negative"]):
                drf[pert] = drf_per_pert(raws, p)
        out[p.name] = drf
    return out


def pairwise_diff(
    model_a: str, model_b: str, raw: dict[str, dict[str, dict[str, float]]], protocols
) -> dict[str, np.ndarray]:
    """Per-protocol array of signed per-perturbation gaps between two models directly.

    Positive means ``model_a`` wins, oriented by each protocol's own ``better`` direction (same
    convention as ``_paired_diff_per_pert``), matched by perturbation. Unlike :func:`per_pert_drf`,
    neither side is routed through :func:`baseline_for` -- both models are ordinary "prediction"
    sources here, not one baseline and one subject.
    """
    out = {}
    for p in protocols:
        a_scores, b_scores = raw[model_a][p.name], raw[model_b][p.name]
        diffs = []
        for pert in sorted(set(a_scores) & set(b_scores)):
            raws = {"positive": a_scores[pert], "negative": b_scores[pert]}
            if not (np.isfinite(raws["positive"]) and np.isfinite(raws["negative"])):
                continue
            diffs.append(_paired_diff_per_pert(raws, p))
        out[p.name] = np.asarray(diffs, dtype=float)
    return out


def pairwise_diff_drf(
    model_a: str, model_b: str, raw: dict[str, dict[str, dict[str, float]]], protocols
) -> dict[str, np.ndarray]:
    """Per-protocol array of per-perturbation DRF gaps between two candidate models.

    Unlike :func:`pairwise_diff`'s raw-score gap, this divides each perturbation's raw gap by
    that perturbation's own headroom (``perfect - baseline_raw_score``, via :func:`per_pert_drf`)
    -- not a uniform rescaling, since the headroom varies perturbation to perturbation, so this
    can disagree with the raw-score test about which pairs are significant. Empty for a protocol
    where either model *is* that protocol's baseline comparator (:func:`baseline_for`), since
    DRF isn't defined for the comparator itself.
    """
    drf_a, drf_b = per_pert_drf(model_a, raw, protocols), per_pert_drf(model_b, raw, protocols)
    out = {}
    for p in protocols:
        da, db = drf_a.get(p.name), drf_b.get(p.name)
        if da is None or db is None:
            out[p.name] = np.asarray([], dtype=float)
            continue
        perts = sorted(set(da) & set(db))
        out[p.name] = np.asarray([da[pert] - db[pert] for pert in perts], dtype=float)
    return out


def _wilcoxon_holm_from_pair_diffs(pair_diffs: dict[tuple[str, str], dict[str, np.ndarray]], protocols) -> pd.DataFrame:
    """Two-sided Wilcoxon + Holm correction.

    Shared by :func:`pairwise_wilcoxon_holm` and :func:`pairwise_wilcoxon_holm_drf` -- only the
    per-perturbation gap they test differs.
    """
    rows = []
    for p in protocols:
        pair_rows = []
        for (model_a, model_b), diffs_by_protocol in pair_diffs.items():
            diffs = diffs_by_protocol[p.name]
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size < 1 or np.all(diffs == 0):
                statistic, pvalue = float("nan"), float("nan")
            else:
                statistic, pvalue = stats.wilcoxon(diffs, alternative="two-sided")
            pair_rows.append(
                {
                    "protocol": p.name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": int(diffs.size),
                    "statistic": float(statistic),
                    "pvalue": float(pvalue),
                }
            )
        pvals = np.array([r["pvalue"] for r in pair_rows])
        finite = np.isfinite(pvals)
        holm = np.full(pvals.shape, np.nan)
        if finite.sum() > 0:
            holm[finite] = multipletests(pvals[finite], method="holm")[1]
        for r, h in zip(pair_rows, holm, strict=True):
            r["pvalue_holm"] = float(h)
        rows.extend(pair_rows)
    return pd.DataFrame(rows)


def pairwise_wilcoxon_holm(raw: dict[str, dict[str, dict[str, float]]], protocols, models: list[str]) -> pd.DataFrame:
    """Two-sided Wilcoxon for every model pair's raw-score gap, Holm-corrected within each protocol.

    Benavoli, Corani & Mangili 2016's recommended alternative to Demšar 2006's Nemenyi
    post-hoc: pairwise Wilcoxon signed-rank tests directly on the paired per-perturbation
    differences (Nemenyi's mean-rank comparison is under-powered and can be inconsistent),
    corrected with Holm's step-down procedure -- uniformly more powerful than Bonferroni while
    still controlling the family-wise error rate, with no omnibus Friedman/Nemenyi gatekeeping
    needed first.

    Blocked on *perturbations*, matching Miller et al. 2025's own Appendix H design and every
    other paired test in this module -- not blocked on protocols: the 18 protocols are a
    curated, correlated set rather than independent replicate "datasets", and pooling across
    them would presuppose the one thing this whole comparison is built to show doesn't hold: a
    stable, protocol-independent ranking. Correction is likewise scoped to one protocol's own
    ``C(len(models), 2)`` pairs at a time, not pooled across protocols.
    """
    pair_diffs = {(a, b): pairwise_diff(a, b, raw, protocols) for a, b in combinations(models, 2)}
    return _wilcoxon_holm_from_pair_diffs(pair_diffs, protocols)


def pairwise_wilcoxon_holm_drf(
    raw: dict[str, dict[str, dict[str, float]]], protocols, models: list[str]
) -> pd.DataFrame:
    """Same test as :func:`pairwise_wilcoxon_holm`, but on DRF gaps instead of raw-score gaps.

    See :func:`pairwise_diff_drf` for why the two aren't equivalent. A model that is a given
    protocol's baseline comparator is silently absent from that protocol's rows (``n=0``), same
    as :func:`per_pert_drf`.
    """
    pair_diffs = {(a, b): pairwise_diff_drf(a, b, raw, protocols) for a, b in combinations(models, 2)}
    return _wilcoxon_holm_from_pair_diffs(pair_diffs, protocols)


def _resample_win_rate(
    a_vals: np.ndarray, b_vals: np.ndarray, higher_wins: bool, n_resamples: int, rng: np.random.Generator
) -> tuple[int, float]:
    """One (model_a, model_b, protocol) cell's bootstrap win rate, from paired per-perturbation arrays.

    ``a_vals``/``b_vals`` must already be aligned by perturbation. Filters non-finite entries
    (a no-op for DRF inputs, which are already finite by construction -- see :func:`per_pert_drf`
    -- but needed for raw scores, e.g. an undefined correlation). Returns ``(0, nan)`` without
    resampling if fewer than one paired perturbation remains, exactly like the two callers'
    previous ``n < 1`` branches.
    """
    finite = np.isfinite(a_vals) & np.isfinite(b_vals)
    a_vals, b_vals = a_vals[finite], b_vals[finite]
    n = a_vals.size
    if n < 1:
        return 0, float("nan")
    idx = rng.integers(0, n, size=(n_resamples, n))
    s_a, s_b = a_vals[idx].mean(axis=1), b_vals[idx].mean(axis=1)
    a_wins = s_a > s_b if higher_wins else s_a < s_b
    return n, float(a_wins.mean())


def _pairwise_bootstrap(protocols, models: list[str], n_resamples: int, seed: int, values_for, higher_is_better):
    """Shared bootstrap loop behind :func:`pairwise_bayes_factor` and :func:`pairwise_bayes_factor_drf`.

    ``values_for(model, p)`` returns ``{perturbation: value}`` for that model/protocol (``None``
    or empty where undefined, e.g. DRF for a protocol's own baseline model). ``higher_is_better(p)``
    resolves the win direction -- ``p.better``-dependent for raw scores, always ``True`` for DRF.
    One ``rng``, consumed in the same protocol-then-pair order the two callers used previously,
    so results are unchanged from before this was unified.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in protocols:
        for model_a, model_b in ((a, b) for a in models for b in models if a != b):
            a_map, b_map = values_for(model_a, p), values_for(model_b, p)
            perts = sorted(set(a_map) & set(b_map)) if a_map and b_map else []
            a_vals = np.asarray([a_map[pert] for pert in perts], dtype=float)
            b_vals = np.asarray([b_map[pert] for pert in perts], dtype=float)
            n, p_a_wins = _resample_win_rate(a_vals, b_vals, higher_is_better(p), n_resamples, rng)
            rows.append({"protocol": p.name, "model_a": model_a, "model_b": model_b, "n": n, "p_a_wins": p_a_wins})
    return rows


def pairwise_bayes_factor(
    raw: dict[str, dict[str, dict[str, float]]], protocols, models: list[str], n_resamples: int = 10_000, seed: int = 42
) -> pd.DataFrame:
    """Bootstrap "Bayes Factor" for every ordered model pair.

    How robustly does A's aggregate score beat B's under resampling of the benchmark
    perturbation set? DREAM Challenge convention for bootstrap-based leaderboard robustness (Menden et al. 2019,
    the AstraZeneca-Sanger Drug Combination Challenge) -- despite the name, this is a purely
    bootstrap/frequentist statistic, not a Bayesian Bayes factor. Complements rather than
    replaces :func:`pairwise_wilcoxon_holm`: that asks whether an observed gap is distinguishable
    from noise (a hypothesis test on the real sample); this asks how often the observed ranking
    between two models would hold up if the 12-perturbation benchmark were resampled (a
    robustness statement about the aggregate itself).

    For each protocol and ordered pair ``(model_a, model_b)``, resamples the perturbations they
    share with replacement -- the same draw for both models each replicate, since their scores
    are correlated through shared perturbation difficulty (paired, like every other test in this
    module) -- recomputes each model's resampled mean, and tallies how often ``model_a`` wins
    (direction-aware via ``p.better``). ``bf = p_a_wins / (1 - p_a_wins)``: the odds that A's
    resampled aggregate beats B's. ``bf > 1`` favors A, ``bf < 1`` favors B, and
    ``bf(A, B) == 1 / bf(B, A)`` by construction (both directions are computed directly here
    rather than derived, so floating-point agreement isn't exact but should be close).

    At n=12 perturbations this estimate is itself noisy -- interpret alongside, not instead of,
    the Wilcoxon+Holm p-values.

    Also returns a Holm-corrected two-sided bootstrap p-value per unordered pair
    (``pvalue_boot``/``pvalue_boot_holm``), so "is this pair robustly separated" can be judged on
    the same footing as :func:`pairwise_wilcoxon_holm` -- Holm's step-down procedure is defined
    on p-values, not odds, so BF is converted rather than thresholded directly. ``1 / (1 + BF)``
    is the one-sided bootstrap probability that the *other* direction holds; twice the smaller of
    the pair's two directions is the standard percentile-bootstrap two-sided p-value, then
    Holm-corrected across the ``C(len(models), 2)`` pairs within each protocol, exactly like the
    Wilcoxon test.
    """
    rows = _pairwise_bootstrap(
        protocols,
        models,
        n_resamples,
        seed,
        values_for=lambda m, p: raw[m][p.name],
        higher_is_better=lambda p: p.better == "higher",
    )
    return _bayes_factor_from_win_rates(rows, protocols)


def pairwise_bayes_factor_drf(
    raw: dict[str, dict[str, dict[str, float]]], protocols, models: list[str], n_resamples: int = 10_000, seed: int = 42
) -> pd.DataFrame:
    """Same as :func:`pairwise_bayes_factor`, but on DRF instead of raw score.

    See :func:`pairwise_diff_drf` for why the two aren't equivalent. DRF is already oriented
    higher-is-better by construction, so no ``p.better`` branching is needed here (unlike the
    raw-score version). A model that is a given protocol's baseline comparator has no DRF there
    and is silently absent from that protocol's rows (``n=0``), same as :func:`per_pert_drf`.
    """
    drf_by_model = {m: per_pert_drf(m, raw, protocols) for m in models}
    rows = _pairwise_bootstrap(
        protocols,
        models,
        n_resamples,
        seed,
        values_for=lambda m, p: drf_by_model[m].get(p.name),
        higher_is_better=lambda p: True,
    )
    return _bayes_factor_from_win_rates(rows, protocols)


def _bayes_factor_from_win_rates(rows: list[dict], protocols) -> pd.DataFrame:
    """Shared BF/p-value/Holm post-processing.

    Used by :func:`pairwise_bayes_factor` and :func:`pairwise_bayes_factor_drf` --
    everything after the resampling loop is identical.
    """
    df = pd.DataFrame(rows)
    p_a_wins = df["p_a_wins"].to_numpy()
    odds = np.divide(p_a_wins, 1.0 - p_a_wins, out=np.full_like(p_a_wins, np.inf), where=(1.0 - p_a_wins) != 0)
    df["bf"] = np.where(np.isnan(p_a_wins), np.nan, odds)

    df["pair_key"] = [tuple(sorted((a, b))) for a, b in zip(df["model_a"], df["model_b"], strict=True)]
    one_sided = 1.0 / (1.0 + df["bf"])
    df["pvalue_boot"] = (2 * one_sided.groupby([df["protocol"], df["pair_key"]]).transform("min")).clip(upper=1.0)

    df["pvalue_boot_holm"] = np.nan
    for p in protocols:
        proto_mask = df["protocol"] == p.name
        unique_pairs = df.loc[proto_mask, ["pair_key", "pvalue_boot"]].drop_duplicates("pair_key")
        pvals = unique_pairs["pvalue_boot"].to_numpy()
        finite = np.isfinite(pvals)
        holm = np.full(pvals.shape, np.nan)
        if finite.sum() > 0:
            holm[finite] = multipletests(pvals[finite], method="holm")[1]
        holm_by_pair = dict(zip(unique_pairs["pair_key"], holm, strict=True))
        df.loc[proto_mask, "pvalue_boot_holm"] = df.loc[proto_mask, "pair_key"].map(holm_by_pair)
    return df.drop(columns="pair_key")


def main() -> None:
    """Build every model's/baseline's predictions, score them, and print the ranking tables."""
    protocols = resolve_protocols(PROTOCOL_SPECS)

    calib = pd.DataFrame(calibration_scores(protocols))

    print("=== calibration DRF — metric headroom on this dataset, model-independent (Miller et al. 2025) ===")
    drf_calib = calib[calib["calibrator"] == "drf"].set_index("protocol")[["mean", "median"]]
    print(drf_calib.to_string(float_format="%.3f"))

    print(
        "\n=== calibration BDS — fraction of perturbations the positive control wins, "
        "model-independent (Vollenweider & Bühlmann 2026) ==="
    )
    bds_calib = calib[calib["calibrator"] == "bds"].set_index("protocol")[["bds"]]
    print(bds_calib.to_string(float_format="%.3f"))

    print("\n=== gene counts — how many genes each protocol's score is actually computed over ===")
    gene_counts = protocol_gene_counts(protocols).set_index("protocol")
    print(gene_counts.to_string(float_format="%.1f"))

    predictions = build_all_predictions()
    raw = per_pert_raw_scores(predictions, protocols)

    print("\n=== pairwise Wilcoxon + Holm — which models are directly distinguishable, per protocol ===")
    pairwise = pairwise_wilcoxon_holm(raw, protocols, list(predictions))
    significant = pairwise[pairwise["pvalue_holm"] < 0.05]
    print(f"{len(significant)}/{len(pairwise)} pairwise comparisons Holm-significant at alpha=0.05")
    print(
        significant.set_index(["protocol", "model_a", "model_b"])[["n", "pvalue", "pvalue_holm"]].to_string(
            float_format="%.4f"
        )
    )

    pairwise_out = HERE / "compare_pairwise.csv"
    pairwise.to_csv(pairwise_out, index=False)
    print(f"\n-> {pairwise_out}")


if __name__ == "__main__":
    main()
