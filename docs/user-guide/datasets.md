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

Every file is a standardised AnnData with:

- **`X`** — log-normalised expression (`normalize_total` + `log1p`), sparse `float32`.
- **`obs["perturbation"]`** — perturbation label per cell; non-targeting controls are labelled `control`.
- **`var`** — gene names (the index).

Upstream results (`uns`, `obsm`, `layers`, PCA, …) are stripped — scPertEval recomputes
differential expression and all baselines in memory.

## Downloading

```bash
# single file
gsutil cp gs://scperteval/processed/replogle22k562_processed_complete.h5ad .

# all files (parallel)
gsutil -m cp gs://scperteval/processed/*_processed_complete.h5ad .
```

Then pass the path directly to the CLI:

```bash
scperteval run ./replogle22k562_processed_complete.h5ad -p all --de-method t-test
```
