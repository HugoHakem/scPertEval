"""Exploratory figures built directly from models/compare.py's functions -- not a notebook yet.

Self-contained: re-invokes compare.calibration_scores/build_all_predictions/per_pert_raw_scores/
per_pert_drf/pairwise_wilcoxon_holm(_drf) for the DRF/BDS headroom, raw-per-perturbation,
per-perturbation-DRF, and pairwise-significance data (fast at smoke scale -- rebuilds the
fold-scoped baselines but does no training). Doesn't read any of models/compare.py's CSV output,
so `python models/compare.py` doesn't need to run first.

Usage::

    python models/plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare import (  # noqa: E402
    PROTOCOL_SPECS,
    baseline_for,
    build_all_predictions,
    calibration_scores,
    pairwise_bayes_factor,
    pairwise_bayes_factor_drf,
    pairwise_wilcoxon_holm,
    pairwise_wilcoxon_holm_drf,
    per_pert_drf,
    per_pert_raw_scores,
    protocol_gene_counts,
    resolve_protocols,
)

OUT = HERE / "plots"

MODELS = ["gears", "scgpt", "presage", "sclambda", "state", "tabicl", "mean", "linear", "no_change"]
COLORS = {
    "gears": "#2a78d6",
    "scgpt": "#1baf7a",
    "presage": "#d6542a",
    "sclambda": "#a72a9e",
    "state": "#2ac2c2",
    "tabicl": "#c2a12a",
    "mean": "#eda100",
    "linear": "#008300",
    "no_change": "#4a3aa7",
}
LOWER_BETTER = {"l2_expr_k=1000", "mse", "mse_degs_padj=0.05", "wmse_exp2"}

FAMILIES = {
    "Ahlmann-Eltze\n(headline)": ["pearson_ctrl_expr_k=1000", "l2_expr_k=1000"],
    "Miller MSE": ["mse", "mse_degs_padj=0.05", "wmse_exp2"],
    "Miller\nPearsonDeltaCtrl": ["pearson_ctrl", "pearson_ctrl_degs_padj=0.05", "weighted_pearson_ctrl_exp2"],
    "Miller\nPearsonDeltaPertMean": ["pearson_pert", "pearson_pert_degs_padj=0.05", "weighted_pearson_pert_exp2"],
    "Miller R2DeltaCtrl": ["r2_ctrl", "r2_ctrl_degs_padj=0.05", "weighted_r2_ctrl_exp2"],
    "Miller R2DeltaPertMean": ["r2_pert", "r2_pert_degs_padj=0.05", "weighted_r2_pert_exp2"],
    "NIR": ["nir"],
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
    }
)


def load_raw_per_pert() -> dict[str, dict[str, dict[str, float]]]:
    """Every model's/baseline's raw per-perturbation score, for every protocol.

    ``{model: {protocol: {perturbation: value}}}`` -- rebuilds the fold-scoped predictions
    (fast: no training, just re-concatenating every already-trained model's folds and refitting
    the mean/linear baselines) so every plot in this module works from the same per-perturbation
    values models/compare.py's own tables are built from.
    """
    protocols = resolve_protocols(PROTOCOL_SPECS)
    predictions = build_all_predictions()
    return per_pert_raw_scores(predictions, protocols)


def compute_avg_ranks(raw: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Model x protocol matrix of each model's mean rank across perturbations (1 = best).

    Ranks models within each perturbation first, then averages those per-perturbation ranks --
    the block-wise quantity :func:`compare.pairwise_wilcoxon_holm`'s paired Wilcoxon test
    actually compares, unlike ranking by the aggregate mean directly (which can disagree with
    it, e.g. under a single outlier perturbation). Ties within a perturbation use pandas'
    ``method="average"``: two models tied for best both get ``1.5`` rather than an arbitrary
    1/2 split.
    """
    protocol_order = [p for protos in FAMILIES.values() for p in protos]
    out = pd.DataFrame(index=MODELS, columns=protocol_order, dtype=float)
    for protocol in protocol_order:
        perts = sorted(set.intersection(*(set(raw[m][protocol]) for m in MODELS)))
        if not perts:
            continue
        table = pd.DataFrame({m: [raw[m][protocol][pert] for pert in perts] for m in MODELS}, index=perts)
        per_pert_ranks = table.rank(axis=1, ascending=protocol in LOWER_BETTER, method="average")
        out[protocol] = per_pert_ranks.mean(axis=0)
    return out


def compute_avg_drf_ranks(raw: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Model x protocol matrix of each model's mean DRF-based rank across perturbations (1 = best).

    DRF analog of :func:`compute_avg_ranks`: ranks candidate models within each perturbation by
    DRF (:func:`compare.per_pert_drf`) against their protocol's own baseline comparator
    (:func:`compare.baseline_for`), then averages those per-perturbation ranks -- the block-wise
    quantity :func:`compare.pairwise_wilcoxon_holm_drf` tests. DRF is already oriented
    higher-is-better, so no per-protocol direction lookup is needed. A protocol's own baseline
    model has no DRF there and is left ``NaN``, matching :func:`plot_drf_median_heatmap`'s
    ``"n/a"`` cells.

    Built on a *union* of perturbations, not an intersection: :func:`compare.per_pert_drf` omits
    a perturbation entirely wherever a model's raw score there is non-finite (e.g. a ctrl-centered
    Pearson protocol against the ``no_change`` model, whose ctrl-centered prediction is an exact
    zero vector -- correlation is undefined for every perturbation). Intersecting first would let
    one structurally-undefined candidate wipe out the whole protocol for every other model;
    pandas' NaN-aware ``rank``/``mean`` instead just excludes that one candidate where it's
    missing, ranking the rest normally.
    """
    protocols = resolve_protocols(PROTOCOL_SPECS)
    protocol_order = [p for protos in FAMILIES.values() for p in protos]
    drf_by_model = {m: per_pert_drf(m, raw, protocols) for m in MODELS}
    out = pd.DataFrame(index=MODELS, columns=protocol_order, dtype=float)
    for p in protocols:
        candidates = [m for m in MODELS if m != baseline_for(p)]
        per_model_drf = {m: drf_by_model[m][p.name] for m in candidates}
        perts = sorted(set.union(*(set(d) for d in per_model_drf.values())))
        if not perts:
            continue
        table = pd.DataFrame(
            {m: [per_model_drf[m].get(pert, float("nan")) for pert in perts] for m in candidates}, index=perts
        )
        per_pert_ranks = table.rank(axis=1, ascending=False, method="average")
        out.loc[candidates, p.name] = per_pert_ranks.mean(axis=0)
    return out


def plot_drf_headroom(calib: pd.DataFrame) -> None:
    """Calibration DRF per protocol -- model-independent metric headroom (sequential blue ramp)."""
    df = calib[calib["calibrator"] == "drf"].set_index("protocol")["mean"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=df.min(), vmax=df.max())
    colors = [cmap(0.3 + 0.6 * norm(v)) for v in df.to_numpy()]
    ax.barh(df.index, df.to_numpy(), color=colors, edgecolor="white")
    for i, v in enumerate(df.to_numpy()):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xlabel("Dynamic Range Fraction (model-independent headroom)")
    ax.set_title(
        "Calibration DRF per protocol\n(how much signal each metric has before any model is considered)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, max(float(df.max()) * 1.15, 0.1))
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "drf_headroom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bds_headroom(calib: pd.DataFrame) -> None:
    """Calibration BDS per protocol -- fraction of perturbations the positive control wins.

    Sequential aqua ramp (the palette's second sequential hue, since this is a second
    sequential context alongside :func:`plot_drf_headroom`'s blue) -- BDS is the
    per-perturbation-reliability companion to DRF's average: a protocol can have decent mean
    DRF while still having a middling BDS, if the gap is inconsistent across perturbations
    rather than uniformly positive.
    """
    df = calib[calib["calibrator"] == "bds"].set_index("protocol")["bds"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list("aqua_seq", ["#cdf3e6", "#1baf7a"])
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    colors = [cmap(norm(v)) for v in df.to_numpy()]
    ax.barh(df.index, df.to_numpy(), color=colors, edgecolor="white")
    for i, v in enumerate(df.to_numpy()):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xlabel("Bound Discrimination Score (fraction of perturbations the positive control wins)")
    ax.set_title(
        "Calibration BDS per protocol\n(per-perturbation reliability of that headroom, not just its average)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "bds_headroom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gene_counts(gene_counts: pd.DataFrame) -> None:
    """How many genes each protocol's score is actually computed over.

    Third model-independent "calibration" view alongside :func:`plot_drf_headroom`'s DRF and
    :func:`plot_bds_headroom`'s BDS (:func:`compare.protocol_gene_counts`). Log-x since the 18
    protocols span from ~97 (a continuously-weighted metric's effective count) to 8563 (the
    full gene set) -- a ~90x range a linear axis would flatten into indistinguishable bars.
    Sorted ascending, same convention as the other two headroom plots.

    A fixed-size protocol (``full``/``expr_k``) gets an exact count, marked ``n=``, no whisker
    (P25 == P75 == median by construction). A DEGs-threshold protocol's count varies by
    perturbation, marked ``n≈``. A continuously-weighted protocol has no hard gene cutoff at
    all, so it gets its own label, ``n_eff≈`` (:func:`compare.protocol_gene_counts`'s effective
    count, ``sum(weight) / max(weight)`` -- defined there, not repeated on the chart) rather
    than being conflated with an actual DEG count. Both variable cases get a whisker spanning
    ``[P25, P75]``.

    Sequential yellow -- the palette's third categorical hue, since this is a third sequential
    context alongside DRF's blue and BDS's aqua.
    """
    df = gene_counts.set_index("protocol").sort_values("n_median")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = mcolors.LinearSegmentedColormap.from_list("yellow_seq", ["#fbe4ad", "#eda100"])
    log_vals = np.log10(df["n_median"].to_numpy())
    norm = plt.Normalize(vmin=log_vals.min(), vmax=log_vals.max())
    colors = [cmap(norm(v)) for v in log_vals]
    ax.barh(y, df["n_median"].to_numpy(), color=colors, edgecolor="white")

    kind = df["kind"].to_numpy()
    is_weighted = kind == "weighted (effective)"
    variable = np.isin(kind, ["DEGs (padj threshold)", "weighted (effective)"])
    medians = df["n_median"].to_numpy()
    err_low = (df["n_median"] - df["n_p25"]).clip(lower=0).to_numpy()
    err_high = (df["n_p75"] - df["n_median"]).clip(lower=0).to_numpy()
    ax.errorbar(
        medians[variable],
        y[variable],
        xerr=[err_low[variable], err_high[variable]],
        fmt="none",
        ecolor="#52514e",
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    label_x = np.where(variable, df["n_p75"].to_numpy(), medians)
    prefix = np.where(is_weighted, r"$n_{\mathrm{eff}}$", "n")
    mark = np.where(variable, "≈", "=")
    for i, (pre, m, x, med) in enumerate(zip(prefix, mark, label_x, medians, strict=True)):
        ax.text(x * 1.08, i, f"{pre}{m}{round(med)}", va="center", fontsize=8, color="#0b0b0b")

    low_ends = np.where(variable, df["n_p25"].to_numpy(), medians)
    high_ends = np.where(variable, df["n_p75"].to_numpy(), medians)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(low_ends.min() * 0.7, high_ends.max() * 2.0)
    ax.set_xlabel("genes the score is computed over (log scale)")
    ax.set_title(
        "Gene count per protocol\n(n= exact fixed gene set; n≈ median DEG count; n_eff≈ median effective count)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "gene_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(raw: dict[str, dict[str, dict[str, float]]]) -> None:
    """Raw per-perturbation metric value by model, one small-multiple panel per protocol.

    Native metric scale (not DRF) -- each protocol gets its own y-axis since e.g. MSE and
    Pearson r aren't on comparable scales. Every perturbation is a point (n ~ 12 at smoke
    scale, small enough to show in full without collapsing to a summary statistic) laid out as
    a true non-overlapping beeswarm -- needs a wide-enough figure and small-enough markers that
    swarmplot's own placement never has to drop a point (it silently does, with a warning, if
    the panel is too cramped) -- with a black tick at each model's median.
    """
    protocol_order = [p for protos in FAMILIES.values() for p in protos]
    fig, axes = plt.subplots(3, 6, figsize=(28, 14))
    axes = axes.ravel()
    for ax, protocol in zip(axes, protocol_order, strict=False):
        long = pd.DataFrame(
            [{"model": model, "value": v} for model in MODELS for v in raw[model][protocol].values() if np.isfinite(v)]
        )
        sns.swarmplot(
            data=long,
            x="model",
            y="value",
            hue="model",
            palette=COLORS,
            order=MODELS,
            hue_order=MODELS,
            size=3.0,
            legend=False,
            ax=ax,
        )
        for i, model in enumerate(MODELS):
            vals = long.loc[long["model"] == model, "value"]
            if len(vals):
                ax.hlines(vals.median(), i - 0.3, i + 0.3, color="#0b0b0b", linewidth=1.8, zorder=5)
        arrow = "↓" if protocol in LOWER_BETTER else "↑"
        ax.set_title(f"{protocol} {arrow}", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Raw per-perturbation metric value by model (native scale, not DRF)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "beeswarm_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drf_median_heatmap(raw: dict[str, dict[str, dict[str, float]]]) -> None:
    """Model-comparison DRF per protocol: median [P25, P75] over perturbations, diverging color.

    Shows the actual calibrated magnitude (DRF is comparable across differently-scaled metrics,
    unlike the raw score) plus its spread -- a "1st place" rank can still hide a wide,
    unreliable distribution.
    """
    protocols = resolve_protocols(PROTOCOL_SPECS)
    protocol_order = [p for protos in FAMILIES.values() for p in protos]
    med = pd.DataFrame(index=protocol_order, columns=MODELS, dtype=float)
    p25, p75 = med.copy(), med.copy()
    for model in MODELS:
        for protocol, vals in per_pert_drf(model, raw, protocols).items():
            drf = np.asarray(list(vals.values()), dtype=float)
            drf = drf[np.isfinite(drf)]
            if drf.size:
                med.loc[protocol, model] = np.median(drf)
                p25.loc[protocol, model] = np.percentile(drf, 25)
                p75.loc[protocol, model] = np.percentile(drf, 75)

    fig, ax = plt.subplots(figsize=(7, 10))
    vmax = max(float(np.nanmax(np.abs(med.to_numpy()))), 1e-6)
    cmap = mcolors.LinearSegmentedColormap.from_list("div_blue_red", ["#e34948", "#f0efec", "#2a78d6"])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(med.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_yticks(range(len(protocol_order)))
    ax.set_yticklabels(protocol_order, fontsize=8)
    for i, protocol in enumerate(protocol_order):
        for j, model in enumerate(MODELS):
            m, lo, hi = med.loc[protocol, model], p25.loc[protocol, model], p75.loc[protocol, model]
            if np.isnan(m):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#898781")
                continue
            color = "white" if abs(m) > vmax * 0.6 else "#0b0b0b"
            ax.text(j, i - 0.15, f"{m:.2f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)
            ax.text(j, i + 0.2, f"[{lo:.2f}, {hi:.2f}]", ha="center", va="center", fontsize=6, color=color)
    ax.set_title(
        "Model-comparison DRF vs. each protocol's own baseline\nmedian [P25, P75] over perturbations",
        fontsize=11,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("median DRF")
    fig.tight_layout()
    fig.savefig(OUT / "drf_median_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bayes_factor_heatmap(
    bf: pd.DataFrame,
    title: str = (
        "Bootstrap Bayes Factor (row model vs. column model)\n"
        "blue = row robustly beats column; red = column robustly beats row -- log10(BF), capped at ±4; "
        "black border = Holm-significant (α=0.05)"
    ),
    out_name: str = "bayes_factor_heatmap.png",
    alpha: float = 0.05,
) -> None:
    """Bootstrap Bayes Factor for every model pair, one row-vs-column matrix per protocol.

    DREAM Challenge convention for bootstrap-based leaderboard robustness (Menden et al. 2019,
    the AstraZeneca-Sanger Drug Combination Challenge), via :func:`compare.pairwise_bayes_factor`
    (raw score) or :func:`compare.pairwise_bayes_factor_drf` (DRF) -- despite the name, a
    bootstrap/frequentist statistic, not a Bayesian Bayes factor. Complements
    :func:`plot_significance_cliques`'s hypothesis test with a robustness question: not "is this
    gap distinguishable from noise" but "how often would the row model's aggregate beat the
    column model's if this 12-perturbation benchmark were resampled."

    log10(BF) on a diverging blue-red scale centered at 0 (BF=1, a coin flip) -- BF is an odds
    ratio, so a linear scale would badly compress one side. Capped at +/-4 (BF of 10,000:1 either
    way) since an unbroken run of bootstrap replicates going the same direction gives BF=0 or
    infinity outright, which would otherwise blow out the shared color scale for every other
    cell -- cells actually beyond the cap are labelled ``>4``/``<-4`` rather than a misleadingly
    precise ``4.0``. The diagonal (a model vs. itself), and -- for the DRF variant -- a model
    that is a given protocol's own baseline comparator, are both undefined and left blank.

    A black border marks cells where ``pvalue_boot_holm < alpha`` -- deliberately not a fixed
    log10(BF) reference line, since Holm's step-down correction makes the actual significance
    boundary different for every pair (it depends on how many other pairs in that protocol also
    have small p-values), so no single BF value on the shared color scale would be correct for
    more than a few cells. The border is exact, computed from the same column
    :func:`plot_significance_cliques` uses for this data.
    """
    protocol_order = [p for protos in FAMILIES.values() for p in protos]
    log_cap = 4.0
    cmap = mcolors.LinearSegmentedColormap.from_list("div_blue_red", ["#e34948", "#f0efec", "#2a78d6"])
    norm = mcolors.TwoSlopeNorm(vmin=-log_cap, vcenter=0.0, vmax=log_cap)

    fig, axes = plt.subplots(3, 6, figsize=(24, 13))
    axes = axes.ravel()
    im = None
    for ax, protocol in zip(axes, protocol_order, strict=False):
        sub = bf[bf["protocol"] == protocol]
        mat = sub.pivot(index="model_a", columns="model_b", values="bf").reindex(index=MODELS, columns=MODELS)
        sig = (
            sub.pivot(index="model_a", columns="model_b", values="pvalue_boot_holm").reindex(
                index=MODELS, columns=MODELS
            )
            < alpha
        )
        with np.errstate(divide="ignore"):
            log_bf_raw = np.log10(mat.to_numpy(dtype=float))
        log_bf = np.clip(log_bf_raw, -log_cap, log_cap)
        im = ax.imshow(np.ma.masked_invalid(log_bf), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                v, v_raw = log_bf[i, j], log_bf_raw[i, j]
                if np.isnan(v):
                    continue
                label = ">4" if v_raw > log_cap else "<-4" if v_raw < -log_cap else f"{v:.1f}"
                color = "white" if abs(v) > log_cap * 0.6 else "#0b0b0b"
                ax.text(j, i, label, ha="center", va="center", fontsize=6, color=color)
                if sig.to_numpy()[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#0b0b0b", linewidth=1.8, zorder=4
                        )
                    )
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS, fontsize=6)
        ax.set_title(protocol, fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    assert im is not None  # protocol_order is non-empty, so the loop always runs at least once
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label("log10(BF)")
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _maximal_nonsignificant_cliques(order: list[str], significant: set[frozenset]) -> list[list[str]]:
    """Maximal contiguous (in ``order``) groups where every internal pair is non-significant.

    Only contiguous ranges matter here -- the diagram places models by rank on a line, and a
    connecting bar only makes visual sense over a contiguous span of that line, exactly how a
    critical-difference diagram's bars work.
    """
    n = len(order)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            group = order[i : j + 1]
            pairs_ok = all(
                frozenset((group[a], group[b])) not in significant
                for a in range(len(group))
                for b in range(a + 1, len(group))
            )
            if pairs_ok:
                candidates.append(group)
    return [g for g in candidates if not any(set(g) < set(other) for other in candidates)]


def _pack_levels(cliques: list[list[str]], x: dict[str, float]) -> list[tuple[list[str], int]]:
    """Assign each clique to the lowest level whose span it doesn't touch or overlap.

    Two cliques that only *touch* at a shared rank position (e.g. spans ending/starting at the
    same x) still conflict and must go on different levels -- otherwise they'd render as one
    unbroken line and misread as a single, larger (transitively) non-significant group, which
    is exactly the false impression a chain of pairwise-only non-significant results can give.
    Plain greedy interval-graph coloring, sorted by start position.
    """
    spans = sorted(cliques, key=lambda g: x[g[0]])
    level_ends: list[float] = []
    placed = []
    for g in spans:
        start, end = x[g[0]], x[g[-1]]
        level = next((lv for lv, e in enumerate(level_ends) if start > e), None)
        if level is None:
            level = len(level_ends)
            level_ends.append(end)
        else:
            level_ends[level] = end
        placed.append((g, level))
    return placed


def plot_significance_cliques(
    pairwise: pd.DataFrame,
    ranks: pd.DataFrame,
    quantity: str,
    title: str,
    out_name: str,
) -> None:
    """Which models are NOT distinguishable from each other, per protocol.

    Benavoli, Corani & Mangili 2016's alternative to a Nemenyi critical-difference diagram:
    models placed by their average per-perturbation rank on a shared x-axis, with a bar
    connecting any maximal contiguous group where every pairwise comparison inside it is
    non-significant -- a dot with no bar over it is significantly different from every other
    model shown. Position and test both operate block-wise on perturbations, so they're reading
    the same underlying quantity.

    ``pairwise`` must carry a boolean ``significant`` column, precomputed by the caller -- this
    function only cares about the resulting yes/no per pair, not how it was derived. Three
    criteria are used elsewhere in this module, each answering a different question and each
    with its own ``ranks``/``quantity`` pairing:

    - Holm-corrected Wilcoxon p < alpha on raw-score gaps (:func:`compare.pairwise_wilcoxon_holm`,
      ``ranks=compute_avg_ranks(raw)``, ``quantity="raw score"``) -- is the observed gap
      distinguishable from noise?
    - The same test on DRF gaps (:func:`compare.pairwise_wilcoxon_holm_drf`,
      ``ranks=compute_avg_drf_ranks(raw)``, ``quantity="DRF"``) -- can disagree with the raw-score
      version (see :func:`compare.pairwise_diff_drf`), since DRF reweights each perturbation's gap
      by its own headroom.
    - A bootstrap Bayes Factor past a threshold (:func:`compare.pairwise_bayes_factor`,
      ``ranks=compute_avg_ranks(raw)``, ``quantity="raw score"``) -- how often would this
      ordering hold up if the benchmark were resampled, DREAM Challenge-style?

    Cliques that only share an endpoint (e.g. {A, B} and {B, C}, with A vs. C significant) are
    placed on different levels via :func:`_pack_levels` and given explicit end-cap ticks, so
    they never visually merge into what would look like one non-transitive {A, B, C} clique.

    Models tied at the exact same average rank (e.g. two models that alternate which one wins
    per perturbation, averaging out identically) are stacked slightly *below* the baseline
    (never horizontal, since x is the real quantity being plotted and offsetting it would
    misrepresent a tie as distinct ranks; never above, to stay clear of the clique bars). A
    minimum bar width keeps their clique bar visible as a link rather than a zero-width tick.

    See :func:`plot_gene_counts` separately for how many genes each protocol's score is
    actually computed over -- a static, model-independent property that doesn't belong repeated
    on every model-comparison figure.
    """
    protocol_order = [p for protos in FAMILIES.values() for p in protos]
    min_bar_width = 0.15
    dot_stack_step = 0.14
    fig, axes = plt.subplots(len(protocol_order), 1, figsize=(7, len(protocol_order) * 0.7), sharex=True)
    for ax, protocol in zip(axes, protocol_order, strict=True):
        row = ranks[protocol].dropna()
        order = row.sort_values().index.tolist()
        sub = pairwise[pairwise["protocol"] == protocol]
        significant = {
            frozenset((r.model_a, r.model_b))
            for r in sub.itertuples()
            if r.model_a in order and r.model_b in order and r.significant
        }
        tied: dict[float, list[str]] = {}
        for model in order:
            tied.setdefault(round(row[model], 6), []).append(model)
        for members in tied.values():
            offsets = [0.0] if len(members) == 1 else [-dot_stack_step * (i + 1) for i in range(len(members))]
            for model, dy in zip(members, offsets, strict=True):
                ax.scatter(row[model], dy, color=COLORS[model], s=45, zorder=3, edgecolor="white", linewidth=1)
        cliques = _maximal_nonsignificant_cliques(order, significant)
        for group, level in _pack_levels(cliques, row.to_dict()):
            y = 0.3 + level * 0.4
            x0, x1 = row[group[0]], row[group[-1]]
            if x1 - x0 < min_bar_width:
                cx = (x0 + x1) / 2
                x0, x1 = cx - min_bar_width / 2, cx + min_bar_width / 2
            ax.plot([x0, x1], [y, y], color="#52514e", linewidth=2, solid_capstyle="butt", zorder=2)
            ax.plot([x0, x0], [y - 0.08, y + 0.08], color="#52514e", linewidth=2, zorder=2)
            ax.plot([x1, x1], [y - 0.08, y + 0.08], color="#52514e", linewidth=2, zorder=2)
        ax.set_yticks([0])
        ax.set_yticklabels([protocol], fontsize=7)
        ax.set_ylim(-0.5, 2.0)
        ax.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xlim(0.5, len(MODELS) + 0.5)
    axes[-1].set_xlabel(f"average rank across perturbations by {quantity} (1 = best)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[m], markersize=8, label=m) for m in MODELS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build every figure and write it to models/plots/."""
    OUT.mkdir(exist_ok=True)
    protocols = resolve_protocols(PROTOCOL_SPECS)
    calib = pd.DataFrame(calibration_scores(protocols))
    plot_drf_headroom(calib)
    plot_bds_headroom(calib)
    plot_gene_counts(protocol_gene_counts(protocols))

    raw = load_raw_per_pert()
    plot_beeswarm(raw)
    plot_drf_median_heatmap(raw)

    alpha = 0.05
    pairwise = pairwise_wilcoxon_holm(raw, protocols, MODELS)
    pairwise["significant"] = np.isfinite(pairwise["pvalue_holm"]) & (pairwise["pvalue_holm"] < alpha)
    plot_significance_cliques(
        pairwise,
        compute_avg_ranks(raw),
        quantity="raw score",
        title=f"Pairwise Wilcoxon + Holm on raw score (α={alpha:g}): bar = not significantly different (per protocol)",
        out_name="significance_cliques.png",
    )

    pairwise_drf = pairwise_wilcoxon_holm_drf(raw, protocols, MODELS)
    pairwise_drf["significant"] = np.isfinite(pairwise_drf["pvalue_holm"]) & (pairwise_drf["pvalue_holm"] < alpha)
    plot_significance_cliques(
        pairwise_drf,
        compute_avg_drf_ranks(raw),
        quantity="DRF",
        title=f"Pairwise Wilcoxon + Holm on DRF (α={alpha:g}): bar = not significantly different (per protocol)",
        out_name="significance_cliques_drf.png",
    )

    bf = pairwise_bayes_factor(raw, protocols, MODELS)
    plot_bayes_factor_heatmap(bf, alpha=alpha)
    # Holm-corrected two-sided bootstrap p-value (compare.pairwise_bayes_factor), not a raw BF
    # threshold -- puts this on the same alpha=0.05 footing as pairwise_wilcoxon_holm above.
    bf["significant"] = np.isfinite(bf["pvalue_boot_holm"]) & (bf["pvalue_boot_holm"] < alpha)
    plot_significance_cliques(
        bf,
        compute_avg_ranks(raw),
        quantity="raw score",
        title=f"Bootstrap Bayes Factor + Holm (α={alpha:g}): bar = not robustly separated (per protocol)",
        out_name="significance_cliques_bf.png",
    )

    bf_drf = pairwise_bayes_factor_drf(raw, protocols, MODELS)
    plot_bayes_factor_heatmap(
        bf_drf,
        title=(
            "Bootstrap Bayes Factor on DRF (row model vs. column model)\n"
            "blue = row robustly beats column; red = column robustly beats row -- log10(BF), capped at ±4; "
            "black border = Holm-significant (α=0.05)"
        ),
        out_name="bayes_factor_heatmap_drf.png",
        alpha=alpha,
    )
    bf_drf["significant"] = np.isfinite(bf_drf["pvalue_boot_holm"]) & (bf_drf["pvalue_boot_holm"] < alpha)
    plot_significance_cliques(
        bf_drf,
        compute_avg_drf_ranks(raw),
        quantity="DRF",
        title=f"Bootstrap Bayes Factor + Holm on DRF (α={alpha:g}): bar = not robustly separated (per protocol)",
        out_name="significance_cliques_bf_drf.png",
    )
    print(f"wrote plots to {OUT}")


if __name__ == "__main__":
    main()
