"""Per-metric wall-clock timing table from EPPS ``--profile`` outputs.

Reads the latest per-dataset timing parquet in ``drf_outputs/`` and renders a table
with the same metric rows/order as the DRF table figure, one wall-clock column per
dataset plus a per-row Total and a per-dataset TOTAL row.

    python timing_table.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drf_table_figure import DATASETS, DESC, DRF_DIR, FIG_DIR, ROWS


def latest_timing(ds: str):
    cands = sorted(glob.glob(str(DRF_DIR / f"{ds}*__timing.csv")))
    return cands[-1] if cands else None


def collect() -> dict:
    """{dataset: {protocol: seconds}}"""
    out = {}
    for ds, _ in DATASETS:
        p = latest_timing(ds)
        if p is None:
            out[ds] = {}
            continue
        t = pd.read_csv(p)
        out[ds] = dict(zip(t.protocol, t.seconds))
    return out


def render(tim: dict, path: Path) -> None:
    headers = DESC + [lbl for _, lbl in DATASETS] + ["Total (s)"]
    cells = []
    for proto, *desc in ROWS:
        secs = [tim.get(ds, {}).get(proto, float("nan")) for ds, _ in DATASETS]
        tot = np.nansum(secs) if np.any(np.isfinite(secs)) else float("nan")
        row = list(desc)
        row += ["-" if not np.isfinite(s) else f"{s:.1f}" for s in secs]
        row += ["-" if not np.isfinite(tot) else f"{tot:.1f}"]
        cells.append(row)
    # per-dataset TOTAL row
    totals = ["TOTAL", "", "", ""]
    grand = 0.0
    for ds, _ in DATASETS:
        s = float(np.nansum(list(tim.get(ds, {}).values()))) if tim.get(ds) else float("nan")
        grand += 0.0 if not np.isfinite(s) else s
        totals.append("-" if not np.isfinite(s) else f"{s:.0f}")
    totals.append(f"{grand:.0f}")
    cells.append(totals)

    ncol = len(headers)
    fig, ax = plt.subplots(figsize=(0.95 * ncol + 2.5, 0.42 * len(cells) + 1.6))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for j in range(ncol):
        c = tbl[0, j]
        c.set_text_props(weight="bold")
        c.set_facecolor("#e8e8e8")
    for j in range(ncol):
        c = tbl[len(cells), j]
        c.set_text_props(weight="bold")
        c.set_facecolor("#f3f3f3")
    ax.set_title("EPPS — per-metric wall-clock time (s)", fontsize=13, weight="bold", pad=16)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(cells, columns=[h.replace("\n", " ") for h in headers]).to_csv(
        str(path).replace(".png", ".csv"), index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render(collect(), FIG_DIR / "metric_timing_table.png")
    have = [ds for ds, _ in DATASETS if latest_timing(ds)]
    print(f"wrote metric_timing_table.{{png,csv}} ; datasets present: {have}")


if __name__ == "__main__":
    main()
