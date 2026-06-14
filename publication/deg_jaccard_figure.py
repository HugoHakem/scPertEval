"""DEG-method Jaccard figure from EPPS-recomputed DE (self-contained).

Consumes the per-gene DE HDF5 files written by ``epps de`` (t-test + MWU, GT
first-half vs the all-perturbed pool) and renders the composite figure: per-
perturbation Spearman histograms (top), 12x12 mean-Jaccard heatmaps (middle),
and a median DEG set-size table (bottom). All criteria / Jaccard / plotting code
is vendored here -- no external project dependency.

    python deg_jaccard_figure.py        # run after `epps de` for the datasets below
"""
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DE_DIR = ROOT / "de_outputs"
FIG_DIR = ROOT / "figures"

DATASETS = ["wessels23", "replogle22k562", "nadig25hepg2", "arch1"]

# 12 DEG-calling criteria: t-test |t| / padj, then MWU |delta| / padj.
TTEST_STAT_THR = [2.0, 4.0, 8.0]
MWU_CLIFF_THR = [0.15, 0.33, 0.47]
PADJ_THR = [0.1, 0.05, 0.01]
PADJ_FLOOR = 1e-300
SUPTITLE = "corr(-log10 p_adj t-test, -log10 p_adj MWU)"


# --------------------------------------------------------------------------- #
# Criteria
# --------------------------------------------------------------------------- #

def build_criteria(ttest_stat_thr, mwu_cliff_thr, padj_thr):
    """Ordered (label, method, kind, threshold); method in {ttest,mwu}, kind in {stat,padj}."""
    criteria = []
    for thr in ttest_stat_thr:
        criteria.append((f"t-test, |t| > {thr:g}", "ttest", "stat", thr))
    for thr in padj_thr:
        criteria.append((f"t-test, padj < {thr:g}", "ttest", "padj", thr))
    for thr in mwu_cliff_thr:
        criteria.append((f"MWU, |δ| > {thr:g}", "mwu", "stat", thr))
    for thr in padj_thr:
        criteria.append((f"MWU, padj < {thr:g}", "mwu", "padj", thr))
    return criteria


def _criterion_mask(criterion, ttest_stat, ttest_padj, mwu_stat, mwu_padj):
    """Boolean per-gene mask for one criterion. NaNs -> False."""
    _, method, kind, thr = criterion
    stat, padj = (ttest_stat, ttest_padj) if method == "ttest" else (mwu_stat, mwu_padj)
    with np.errstate(invalid="ignore"):
        if kind == "stat":
            return np.where(np.isnan(stat), False, np.abs(stat) > thr)
        return np.where(np.isnan(padj), False, padj < thr)


def neg_log10_padj(padj):
    return -np.log10(np.clip(padj, PADJ_FLOOR, None))


# --------------------------------------------------------------------------- #
# Per-dataset accumulation: 12x12 mean Jaccard + per-pert Spearman + set sizes
# --------------------------------------------------------------------------- #

def _jaccard(mi, mj):
    union = int(np.count_nonzero(mi | mj))
    return int(np.count_nonzero(mi & mj)) / union if union else 0.0


def _spearman_corr(arrays):
    """Per-perturbation Spearman of -log10(p_adj) between t-test and MWU."""
    from scipy.stats import spearmanr

    tt_padj, mw_padj = arrays["ttest_padj"], arrays["mwu_padj"]
    valid = ~(np.isnan(tt_padj) | np.isnan(mw_padj))
    if int(valid.sum()) < 2:
        return np.nan
    tt_nl, mw_nl = neg_log10_padj(tt_padj[valid]), neg_log10_padj(mw_padj[valid])
    if np.ptp(tt_nl) <= 0 or np.ptp(mw_nl) <= 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        s = spearmanr(tt_nl, mw_nl).correlation
    return float(s) if s is not None and np.isfinite(s) else np.nan


def compute_dataset(comp, criteria):
    """Single pass over a dataset: mean Jaccard matrix, per-pert Spearman, median set sizes."""
    n = len(criteria)
    sum_jacc = np.zeros((n, n), dtype=np.float64)
    count = 0
    corr_rows, size_rows = [], []
    for pert in comp.perturbations:
        arrays = comp.read_perturbation(pert)
        if arrays is None:
            continue
        count += 1
        corr_rows.append((pert, _spearman_corr(arrays)))
        masks = [_criterion_mask(c, arrays["ttest_stat"], arrays["ttest_padj"],
                                 arrays["mwu_stat"], arrays["mwu_padj"]) for c in criteria]
        size_rows.append([int(np.count_nonzero(m)) for m in masks])
        for i in range(n):
            for j in range(i, n):
                v = _jaccard(masks[i], masks[j])
                sum_jacc[i, j] += v
                if j != i:
                    sum_jacc[j, i] += v
    mean_jacc = sum_jacc / max(count, 1)
    median_sizes = (np.median(np.asarray(size_rows, dtype=np.float64), axis=0)
                    if size_rows else np.full(n, np.nan))
    return {
        "dataset": comp.dataset_name, "reference": comp.reference,
        "labels": [c[0] for c in criteria], "n_perturbations": count,
        "mean_jaccard": mean_jacc, "median_sizes": median_sizes,
        "corr_df": pd.DataFrame(corr_rows, columns=["perturbation", "spearman"]),
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _hist(ax, corr_df, dataset, show_y):
    vals = corr_df["spearman"].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.hist(vals, bins=20, range=(0.0, 1.0), color="#4477aa", edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.median(vals)), color="crimson", ls="--", lw=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(dataset, fontsize=9)
    ax.set_xlabel("spearman corr", fontsize=8)
    if show_y:
        ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)


def _heatmap(ax, mat, labels, title, show_y):
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="equal")
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=5.5, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels if show_y else [], fontsize=5.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(length=0)
    return im


def _count_table(ax, labels, datasets, size_matrix):
    ax.axis("off")
    ax.set_title("Set-size key for the Jaccard matrices above - median genes "
                 "selected per criterion\n(rows/cols match the heatmap order)",
                 fontsize=8, pad=3, y=0.99)
    cell_text = [["-" if not np.isfinite(size_matrix[r, c]) else f"{int(round(size_matrix[r, c]))}"
                  for c in range(len(datasets))] for r in range(len(labels))]
    table = ax.table(cellText=cell_text, rowLabels=labels, colLabels=datasets,
                     cellLoc="center", rowLoc="left", loc="upper center",
                     bbox=[0.0, 0.0, 1.0, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.4)
        cell.set_height(0.058)
        cell.PAD = 0.02
        if r == 0 or c == -1:
            cell.set_text_props(fontweight="bold")


def plot_figure(results, labels, out_path):
    """Three-band figure: histograms (top), 12x12 heatmaps (middle), set-size table (bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(results)
    datasets = [res["dataset"] for res in results]
    size_matrix = np.column_stack([res["median_sizes"] for res in results])

    col_w = 3.6
    fig_w = col_w * n
    left, right = 0.10, 0.90
    heat_h = (right - left) * fig_w / n          # square heatmap content
    hist_h, table_h = 1.7, 3.4
    fig = plt.figure(figsize=(fig_w, hist_h + heat_h + table_h + 1.6))
    gs = GridSpec(3, n, figure=fig, height_ratios=[hist_h, heat_h, table_h],
                  left=left, right=right, top=0.95, bottom=0.03, hspace=0.35, wspace=0.30)

    for col, res in enumerate(results):
        _hist(fig.add_subplot(gs[0, col]), res["corr_df"], res["dataset"], show_y=(col == 0))

    last_im = first_heat_ax = last_heat_ax = None
    for col, res in enumerate(results):
        ax = fig.add_subplot(gs[1, col])
        last_im = _heatmap(ax, res["mean_jaccard"], labels,
                           f"Mean Jaccard\nref={res['reference']}", show_y=(col == 0))
        first_heat_ax = first_heat_ax or ax
        last_heat_ax = ax

    ax_table = fig.add_subplot(gs[2, :])
    _count_table(ax_table, labels, datasets, size_matrix)
    fig.suptitle(SUPTITLE, fontsize=12, y=0.985)

    if last_im is not None:
        fig.canvas.draw()
        first_pos, last_pos = first_heat_ax.get_position(), last_heat_ax.get_position()
        tpos = ax_table.get_position()
        ax_table.set_position([first_pos.x0, tpos.y0, last_pos.x1 - first_pos.x0, tpos.height])
        cb_h = last_pos.height * 0.6
        cax = fig.add_axes([right + 0.015, last_pos.y0 + (last_pos.height - cb_h) / 2.0, 0.012, cb_h])
        fig.colorbar(last_im, cax=cax, label="Mean Jaccard")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# EPPS DE source + main
# --------------------------------------------------------------------------- #

class EPPSDESource:
    """Reads an ``epps de`` HDF5 file, yielding the {ttest,mwu}_{stat,padj} arrays per pert."""

    def __init__(self, h5path: str, dataset: str):
        self.dataset_name = dataset
        self.reference = "all_perturbed_8192"
        with h5py.File(h5path, "r") as f:
            self.perturbations = [p.decode() for p in f["perturbations"][:]]
            self._ts, self._tp = f["t-test/statistic"][:], f["t-test/pvalue_adj"][:]
            self._ms, self._mp = f["MWU/statistic"][:], f["MWU/pvalue_adj"][:]
        self._idx = {p: i for i, p in enumerate(self.perturbations)}

    def read_perturbation(self, pert):
        i = self._idx[pert]
        return {"ttest_stat": self._ts[i], "ttest_padj": self._tp[i],
                "mwu_stat": self._ms[i], "mwu_padj": self._mp[i]}


def latest_de(ds: str):
    cands = sorted(glob.glob(str(DE_DIR / f"{ds}*__de.h5")))
    return cands[-1] if cands else None


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    criteria = build_criteria(TTEST_STAT_THR, MWU_CLIFF_THR, PADJ_THR)
    labels = [c[0] for c in criteria]
    results = []
    for ds in DATASETS:
        h5 = latest_de(ds)
        if h5 is None:
            print(f"[SKIP] {ds}: no EPPS DE export in {DE_DIR}")
            continue
        res = compute_dataset(EPPSDESource(h5, ds), criteria)
        if res["n_perturbations"] == 0:
            print(f"[SKIP] {ds}: no perturbations")
            continue
        results.append(res)
        print(f"[OK]   {ds}: n_pert={res['n_perturbations']}")
    if not results:
        print("No datasets available; run `epps de` first.")
        return
    plot_figure(results, labels, FIG_DIR / "deg_jaccard_with_counts.png")
    rows = [(r["dataset"], lab, int(round(sz)) if np.isfinite(sz) else "")
            for r in results for lab, sz in zip(labels, r["median_sizes"])]
    pd.DataFrame(rows, columns=["dataset", "criterion", "median_set_size"]).to_csv(
        FIG_DIR / "deg_jaccard_with_counts.csv", index=False)
    print(f"wrote deg_jaccard_with_counts.{{png,csv}} ({len(results)} datasets)")


if __name__ == "__main__":
    main()
