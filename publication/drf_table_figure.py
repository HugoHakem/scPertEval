"""Main DRF table figure from EPPS outputs, styled after the drf figure.

Reads the latest per-dataset DRF parquet in ``drf_outputs/``, aggregates the
per-perturbation DRF to mean and median per protocol, and renders a
metric-row x dataset-column table (RdBu, signed 2-dp), one figure for each of
mean and median. EPPS implements 20 of the drf figure's 22 rows; the two rank
metrics (transpose_rank, rank) are not in EPPS and are omitted.

    python drf_table_figure.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
DRF_DIR = ROOT / "drf_outputs"
FIG_DIR = ROOT / "figures"

DATASETS = [
    ("wessels23", "Wessels23"),
    ("replogle22rpe1", "Replogle22\nRPE1"),
    ("replogle22k562", "Replogle22\nK562"),
    ("nadig25jurkat", "Nadig25\nJurkat"),
    ("nadig25hepg2", "Nadig25\nHepg2"),
    ("arch1", "ArcH1"),
    ("kaden25rpe1", "Kaden25\nRPE1"),
]

EM = "—"
# (protocol, Base Metric, Centering, Feature Space, Positive Control). Protocol names are
# the current EPPS names at their figure defaults (k=50 / padj=0.05).
ROWS = [
    ("pearson", "Pearson", EM, EM, "Interp."),
    ("pearson_ctrl", "Pearson", "Ctrl", EM, "Interp."),
    ("pearson_pert", "Pearson", "Pert", EM, "Interp."),
    ("pearson_pert_degs_padj=0.05", "Pearson", "Pert", "Sig. DEGs", "Interp."),
    ("pearson_pert_top_k=50", "Pearson", "Pert", "Top-50", "Interp."),
    ("mse", "MSE", EM, EM, "Interp."),
    ("mse_degs_padj=0.05", "MSE", EM, "Sig. DEGs", "Interp."),
    ("mse_top_k=50", "MSE", EM, "Top-50", "Interp."),
    ("wmse_exp1", "WMSE (γ=1)", EM, EM, "Interp."),
    ("wmse_exp2", "WMSE (γ=2)", EM, EM, "Interp."),
    ("wmse_exp4", "WMSE (γ=4)", EM, EM, "Interp."),
    ("transpose_rank", "Transp. Rank", EM, EM, "Interp."),
    ("rank", "Rank", EM, EM, "Interp."),
    ("de_overlap_k=50", "DE Overlap", EM, "Top-50", "Tech. dup."),
    ("de_auprc", "DE AUPRC", EM, EM, "Tech. dup."),
    ("de_auroc", "DE AUROC", EM, EM, "Tech. dup."),
    ("unbiased_mmd_median_top_k=50", "MMD", EM, "Top-50", "Tech. dup."),
    ("unbiased_mmd_median_pca_k=50", "MMD", EM, "PCA-50", "Tech. dup."),
    ("energy_distance_top_k=50", "Energy", EM, "Top-50", "Tech. dup."),
    ("energy_distance_pca_k=50", "Energy", EM, "PCA-50", "Tech. dup."),
    ("sinkhorn_w2_top_k=50", "$W_2$", EM, "Top-50", "Tech. dup."),
    ("sinkhorn_w2_pca_k=50", "$W_2$", EM, "PCA-50", "Tech. dup."),
]
DESC = ["Base Metric", "Centering", "Feature Space", "Positive Ctrl"]


def latest_drf(ds: str):
    cands = sorted(glob.glob(str(DRF_DIR / f"{ds}*__drf.csv")))
    return cands[-1] if cands else None


def aggregate() -> dict:
    """{dataset: {protocol: (mean_drf, median_drf)}}, mirroring EPPS's drf calibrator."""
    out = {}
    for ds, _ in DATASETS:
        p = latest_drf(ds)
        if p is None:
            out[ds] = {}
            continue
        g = pd.read_csv(p).groupby("protocol")["drf"]
        out[ds] = {proto: (float(np.nanmean(v)), float(np.nanmedian(v))) for proto, v in g}
    return out


def render(agg: dict, stat_idx: int, title: str, path: Path) -> None:
    cmap, norm = plt.cm.RdBu, Normalize(vmin=-1.0, vmax=1.0)
    headers = DESC + [lbl for _, lbl in DATASETS]
    ncol = len(headers)
    cells, colors = [], []
    for proto, *desc in ROWS:
        vals, vcolors = [], []
        for ds, _ in DATASETS:
            mm = agg.get(ds, {}).get(proto)
            v = mm[stat_idx] if mm is not None else float("nan")
            vals.append("-" if not np.isfinite(v) else f"{v:+.2f}")
            vcolors.append("#f0f0f0" if not np.isfinite(v) else cmap(norm(v)))
        cells.append(list(desc) + vals)
        colors.append(["white"] * len(DESC) + vcolors)

    fig, ax = plt.subplots(figsize=(0.95 * ncol + 2.5, 0.42 * len(ROWS) + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, cellColours=colors, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for j in range(ncol):
        c = tbl[0, j]
        c.set_text_props(weight="bold")
        c.set_facecolor("#e8e8e8")
    ax.set_title(title, fontsize=13, weight="bold", pad=16)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(cells, columns=[h.replace("\n", " ") for h in headers]).to_csv(
        str(path).replace(".png", ".csv"), index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg = aggregate()
    render(agg, 0, "EPPS — Mean DRF", FIG_DIR / "drf_table_mean.png")
    render(agg, 1, "EPPS — Median DRF", FIG_DIR / "drf_table_median.png")
    have = [ds for ds, _ in DATASETS if latest_drf(ds)]
    print(f"wrote drf_table_{{mean,median}}.{{png,csv}} ; datasets present: {have}")


if __name__ == "__main__":
    main()
