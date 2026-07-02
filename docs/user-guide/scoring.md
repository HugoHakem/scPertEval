# Scoring predictions

Scoring answers the question: **how well does a model's predicted response match the real
perturbation response?** Each evaluation protocol is applied to the predicted cells against the
real cells, yielding one score per perturbation. It is the conventional benchmarking step — the
number you report in a paper.

This is distinct from [calibration](calibration.md), which asks a prior question: is a given
protocol trustworthy enough to report in the first place? Use calibration to select protocols,
then scoring to evaluate your model with them.

`scperteval score dataset.h5ad predictions.h5ad` runs the *same* protocol catalog as
[`calibrate`](calibration.md); only two pieces differ.

## Inputs

- **ground truth** — *all* of a perturbation's real cells (the `gt_all_cells` source). Unlike
  calibration, no half is held out and no positive/negative controls are built — the ground truth
  is the whole real population.
- **prediction** — the matching cells from your `predictions.h5ad` (the `prediction` source).
  The prediction file must contain the dataset's exact gene set (any order — columns are
  reordered by name so the comparison lines up gene-for-gene) and the same perturbation labels.
  A gene-set mismatch, or a perturbation present in the dataset but absent from the predictions,
  raises an error naming exactly what's wrong.

## Output

The `score` calibrator reports each protocol's raw metric value per perturbation and its
mean/median across perturbations, written to `<dataset>__<timestamp>__score.csv`. Higher- vs
lower-is-better follows each protocol's `better` field, exactly as in calibration.

```bash
scperteval score data/wessels23.h5ad predictions.h5ad -p pearson,mse,de_auprc,unbiased_mmd_median_pca_k=20
```

| protocol | representation | perfect prediction | degraded prediction |
|---|---|---|---|
| `pearson` | centroid | 1.000 | 0.993 |
| `mse` | centroid | 0.000 | 0.004 |
| `de_auprc` | de | 1.000 | 0.297 |
| `unbiased_mmd_median_pca_k=20` | population | ≈0 | 0.199 |

(An exact replica of the real cells scores optimally; a prediction degraded toward the control
mean scores worse on every representation.)

## How it relates to calibration

Architecturally this reuses everything — the per-perturbation loop, every metric, representation,
and feature space are shared with `calibrate`. The only differences are the **truth source**
(`gt_all_cells` instead of the held-out `gt_half`) and the **calibrator** (`score`, which needs
only the prediction, instead of `drf`/`bds`, which need both controls). The DE-derived feature
spaces (`top_k`, `degs`) and the WMSE weights are computed from this same all-cells ground truth.

Use `score` to measure how good a model's predictions are; use [`calibrate`](calibration.md) to
decide whether a given protocol is trustworthy enough to report those scores in the first place.

→ See [Usage](usage.md) for the full CLI reference, all options, and `--help` output.
