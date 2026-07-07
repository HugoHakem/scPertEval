"""Score every model against the same ground truth under both papers' native protocols.

Compares GEARS, scGPT, mean, no_change, and linear against Miller et al. 2025's and
Ahlmann-Eltze et al. 2025's native protocols, to show the ranking flip that motivates
scPertEval — see docs/user-guide/reproducing-literature-protocols.md.

Concatenates each model's per-fold test predictions (models/data/prepare_split.py's folds) into
one full-coverage prediction file, builds the two baselines directly from scPertEval's own
``Dataset`` (models/baselines/baselines.py), then reuses scPertEval's own scoring internals
(the same code path ``scperteval score`` runs) for every (model, protocol) pair.

Run with the main project's environment, not a models/ pixi env — this needs ``scperteval``
installed, not GEARS/scGPT/torch::

    python models/compare.py
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # baselines/ is a sibling package, not installed

from baselines.baselines import BASELINES, FOLD_SCOPED_BASELINES, build_predictions  # noqa: E402

from scperteval.calibrators import CALIBRATORS  # noqa: E402
from scperteval.cli import resolve_protocols  # noqa: E402
from scperteval.context import Context  # noqa: E402
from scperteval.dataset import Dataset  # noqa: E402
from scperteval.predictions import PredictionSet  # noqa: E402
from scperteval.runner import run_protocol  # noqa: E402
from scperteval.types import RunConfig  # noqa: E402

RAW = HERE / "data" / "smoke_k562_raw.h5ad"
FOLD_DIR = HERE / "data" / "smoke_k562_folds"
N_FOLDS = 3
TRAINED_MODELS = {"gears": HERE / "gears", "scgpt": HERE / "scgpt"}

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


def calibration_drf(protocols) -> list[dict]:
    """Miller et al. 2025's first question: dataset-native, model-independent metric headroom.

    How much headroom does each protocol's *metric* have at all on this dataset, before any
    model is even considered? Positive/negative are the protocol's own built-in controls
    (interpolated duplicate vs. all_perturbed_mean) — no predictions involved
    (``scperteval calibrate``).
    """
    cfg = RunConfig(dataset=str(RAW), protocols=[p.name for p in protocols], output="drf")
    ds = Dataset.load(str(RAW), cfg)
    ctx = Context(ds, cfg)
    ctx.warm(protocols)
    calibrator = CALIBRATORS["drf"]
    rows = []
    for p in protocols:
        agg, _, _ = run_protocol(p, ctx, calibrator)
        rows.append({"protocol": p.name, **agg})
    return rows


def score_model(pred_path: Path, protocols) -> list[dict]:
    """Three views of one model's predictions against each protocol's own negative baseline.

    ``all_perturbed_mean`` for uncentered/``ctrl``-centered protocols, ``control`` for
    ``allpert``-centered ones — ``negative="auto"`` defers to each protocol's own wiring rather
    than forcing one baseline on every protocol: an ``allpert``-centered protocol's own
    ``all_perturbed_mean`` control coincides exactly with its centering reference, so the
    "negative" raw value would be an identically-zero vector — undefined for a correlation
    metric, since it divides by that vector's own variance:

    - ``score``: the raw per-protocol value (mean/median over perturbations).
    - ``drf``: Miller et al. 2025's *model-comparison* Dynamic Range Fraction — the same
      ``drf`` calibrator as :func:`calibration_drf`, but with ``positive`` overridden to the
      model's own prediction, so it answers "how much of the positive-control-vs-baseline gap
      does this *model* close?" rather than "how much headroom does this *metric* have?".
    - ``paired_ci``: Ahlmann-Eltze et al. 2025's bootstrap "does it outperform" question,
      generalized by scPertEval's own ``paired_ci`` calibrator.
    - ``ttest``/``wilcoxon``: Miller et al. 2025's paired one-sided Student t-test and Wilcoxon
      signed-rank test. Both report a *raw* p-value; :func:`main` applies the Bonferroni
      correction afterwards, across every (model, protocol) comparison actually run here —
      the calibrator itself has no way to know that count.
    """
    base_cfg = RunConfig(
        dataset=str(RAW),
        protocols=[p.name for p in protocols],
        predictions=str(pred_path),
        truth="gt_all_cells",
    )
    ds = Dataset.load(str(RAW), base_cfg)
    ctx = Context(ds, base_cfg)
    ctx.predictions = PredictionSet.load(str(pred_path), ds, base_cfg)
    ctx.warm(protocols)

    rows = []
    for calibrator_name, cfg in [
        ("score", base_cfg),
        ("drf", replace(base_cfg, output="drf", positive="prediction")),
        ("paired_ci", replace(base_cfg, output="paired_ci", positive="prediction")),
        ("ttest", replace(base_cfg, output="ttest", positive="prediction")),
        ("wilcoxon", replace(base_cfg, output="wilcoxon", positive="prediction")),
    ]:
        ctx.cfg = cfg
        calibrator = CALIBRATORS[calibrator_name]
        for p in protocols:
            agg, _, _ = run_protocol(p, ctx, calibrator)
            rows.append({"protocol": p.name, "calibrator": calibrator_name, **agg})
    return rows


def main() -> None:
    """Build every model's/baseline's predictions, score them, and print the ranking tables."""
    protocols = resolve_protocols(PROTOCOL_SPECS)

    print("=== calibration DRF — metric headroom on this dataset, model-independent (Miller et al. 2025) ===")
    calib = pd.DataFrame(calibration_drf(protocols)).set_index("protocol")[["mean", "median"]]
    print(calib.to_string(float_format="%.3f"))

    predictions = build_all_predictions()
    rows = []
    for model_name, pred_path in predictions.items():
        for row in score_model(pred_path, protocols):
            rows.append({"model": model_name, **row})
    table = pd.DataFrame(rows)

    print("\n=== raw per-protocol score (mean/median over perturbations) ===")
    raw = table[table["calibrator"] == "score"].pivot(index="model", columns="protocol", values="mean")
    print(raw.to_string(float_format="%.3f"))

    print("\n=== model-comparison DRF vs each protocol's own negative baseline (Miller et al. 2025) ===")
    drf = table[table["calibrator"] == "drf"].pivot(index="model", columns="protocol", values="mean")
    print(drf.to_string(float_format="%.3f"))

    print("\n=== paired_ci vs each protocol's own negative baseline (positive = model wins) ===")
    ci = table[table["calibrator"] == "paired_ci"].set_index(["protocol", "model"])[["mean", "ci_low", "ci_high"]]
    print(ci.to_string(float_format="%.3f"))

    print("\n=== paired one-sided t-test / Wilcoxon vs each protocol's own negative baseline ===")
    for calibrator_name in ("ttest", "wilcoxon"):
        sub = table[table["calibrator"] == calibrator_name].copy()
        n_comparisons = len(sub)  # Bonferroni: correct across every comparison run under this test
        sub["pvalue_bonferroni"] = np.minimum(sub["pvalue"] * n_comparisons, 1.0)
        print(f"\n--- {calibrator_name} (Bonferroni n={n_comparisons}) ---")
        print(
            sub.set_index(["protocol", "model"])[["mean", "pvalue", "pvalue_bonferroni"]].to_string(float_format="%.4f")
        )

    out = HERE / "compare_results.csv"
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
