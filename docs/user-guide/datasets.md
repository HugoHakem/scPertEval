# Datasets

Seven preprocessed single-cell perturbation datasets are available from a public GCS bucket:

```
gs://scperteval/processed/
```

Each file is also reachable over plain HTTPS:

```
https://storage.googleapis.com/scperteval/processed/<dataset>_processed_complete.h5ad
```

## Available datasets

| Dataset | Cell line | Reference |
|---|---|---|
| `replogle22k562` | K562 | Replogle et al. 2022 |
| `replogle22rpe1` | RPE1 | Replogle et al. 2022 |
| `nadig25hepg2` | HepG2 | Nadig et al. 2025 |
| `nadig25jurkat` | Jurkat | Nadig et al. 2025 |
| `wessels23` | — | Wessels et al. 2023 |
| `arch1` | — | — |
| `kaden25rpe1` | RPE1 | Kaden et al. 2025 |

:::{note}
`wessels23` contains only **combination** perturbations. Single-gene perturbations (used as
additive-model building blocks) are excluded from evaluation targets.
:::

## Format

A scPertEval input is an ordinary AnnData (`.h5ad`). scPertEval reads only three things and
recomputes everything else (differential expression, controls, baselines) in memory:

| Where | What | Required |
|---|---|---|
| `adata.X` | **log-normalised** expression, cells × genes — `normalize_total(target_sum=1e4)` then `log1p`; sparse `float32` is ideal | yes |
| `adata.obs["perturbation"]` | one **perturbation label per cell**; all non-targeting control cells share the single label `control` | yes |
| `adata.var_names` | **gene names** (the `var` index) | yes |

The `obs` column name and the control label are configurable — `--perturbation-key` and
`--control-label` (defaults `perturbation` and `control`). Nothing else is read.

:::{note}
`X` must be **log-normalised**, not raw counts — scPertEval does not normalise for you. If your
file is *already* normalised, don't normalise it again.
:::

**Trimming.** Everything scPertEval doesn't read should be dropped: a redundant raw-count layer
(often the single biggest source of bloat — roughly doubling the file), embeddings in `obsm`,
graphs and precomputed results in `uns`, and surplus `obs`/`var` columns. Keeping only `X`, the
`perturbation` column, and the gene names — written sparse, `float32`, gzip-compressed — makes
files markedly smaller and faster to load. The hosted files above are trimmed this way.

## Preparing your own dataset

Starting from a raw perturb-seq `.h5ad`, three steps get you to the format above:

1. **Clean `obs["perturbation"]`** — one label per cell, all controls collapsed to `control`,
   any guide/plasmid suffixes stripped, combinations joined with `+`, and cells with a missing
   label dropped.
2. **Log-normalise `X`** — `normalize_total(target_sum=1e4)` + `log1p`, with light QC filtering
   (e.g. `filter_cells(min_genes=200)`, `filter_genes(min_cells=3)`).
3. **Trim** — keep only `X`, `obs["perturbation"]` and the gene names; write sparse `float32`,
   gzip-compressed.

Only step 1 really differs between datasets. The
[Preparing a dataset](../tutorials.md) tutorial works through it end-to-end on three real
datasets, each showing a different wrinkle:

| Dataset | Cleanup needed in step 1 |
|---|---|
| `replogle22k562` | none — labels are already clean gene symbols; just normalise + trim |
| `adamson16` | strip the plasmid suffix (`SEC61A1_pDS031` → `SEC61A1`); route parenthesised `(mod)` constructs to `control` |
| `wessels23` | join combination guides with `+` (`IKZF1_SMARCD1` → `IKZF1+SMARCD1`) |

:::{note}
Combinations are **relabelled, not removed** — `GENE1+GENE2` is a perturbation like any other.
Whether a study evaluates combos, singles, or both is a downstream modelling choice (for
`wessels23`, combos are the targets and singles are additive-model building blocks).
:::

## Downloading

```bash
# single file
gsutil cp gs://scperteval/processed/replogle22k562_processed_complete.h5ad .

# all files (parallel)
gsutil -m cp gs://scperteval/processed/*_processed_complete.h5ad .
```

Then pass the path directly to the CLI:

```bash
scperteval calibrate ./replogle22k562_processed_complete.h5ad -p all --de-method t-test
```
