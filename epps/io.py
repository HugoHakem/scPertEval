"""Human-readable summary plus a per-perturbation CSV named with dataset + time."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def print_summary(cfg, aggregates: dict, calibrator, protocols) -> None:
    name = Path(cfg.dataset).stem
    print(f"\n{name} · {cfg.de_method} · subsample={cfg.subsample} · seed={cfg.seed} · output={cfg.output}\n")
    agg_keys = sorted({k for v in aggregates.values() for k in v})
    header = f"{'protocol':26s} {'kind':11s} {'space':9s} " + " ".join(f"{k:>9s}" for k in agg_keys)
    print(header)
    print("-" * len(header))
    for p in protocols:
        vals = aggregates.get(p.name, {})
        cells = " ".join(f"{vals.get(k, float('nan')):>9.3f}" for k in agg_keys)
        print(f"{p.name:26s} {p.kind:11s} {p.space:9s} {cells}")
    print()


def write_rows(cfg, rows: list, timestamp: str) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col, val in (("dataset", Path(cfg.dataset).stem), ("de_method", cfg.de_method),
                     ("subsample", cfg.subsample), ("seed", cfg.seed)):
        df[col] = val
    path = out_dir / f"{Path(cfg.dataset).stem}__{timestamp}__{cfg.output}.csv"
    df.to_csv(path, index=False)
    return path


def write_timing(cfg, timed: list, timestamp: str) -> Path:
    """Write per-protocol wall-clock seconds (one row per protocol)."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"dataset": Path(cfg.dataset).stem, "protocol": p.name,
             "kind": p.kind, "space": p.space, "seconds": seconds}
            for p, seconds in timed]
    path = out_dir / f"{Path(cfg.dataset).stem}__{timestamp}__timing.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_de(cfg, genes, perturbations, results: dict, timestamp: str) -> Path:
    """Write per-gene DE (statistic + adjusted p) per method to an HDF5 file.

    Layout: ``genes``, ``perturbations``, and one group per method holding
    ``statistic`` and ``pvalue_adj`` matrices (perturbations x genes)."""
    import h5py

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(cfg.dataset).stem}__{timestamp}__de.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("genes", data=np.asarray(genes, dtype="S"))
        f.create_dataset("perturbations", data=np.asarray(perturbations, dtype="S"))
        for method, (stat, padj) in results.items():
            g = f.create_group(method)
            g.create_dataset("statistic", data=stat)
            g.create_dataset("pvalue_adj", data=padj)
    return path
