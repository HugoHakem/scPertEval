# Usage

## Input data

scPertEval reads one preprocessed AnnData (`.h5ad`) per dataset. Only three things are required:

- **`adata.X`** — normalized expression, cells × genes (e.g. `sc.pp.normalize_total` + `sc.pp.log1p`); sparse or dense float.
- **`adata.obs["perturbation"]`** — the perturbation label for each cell; control cells use the label `"control"`. Both names are configurable (`--perturbation-key` / `--control-label`).
- **`adata.var_names`** — gene identifiers, used as the DEG labels.

Perturbations with at least `--min-cells` cells (default 30) are evaluated. Nothing else is
needed — references, DE, and PCA are all recomputed in memory, so no `uns`/`obsm`/`layers` are read.

**Sample datasets.** Seven preprocessed perturbation datasets live in a public, read-only GCS
bucket and serve as a template for the format above:

```bash
gsutil ls gs://scperteval/processed/      # wessels23, replogle22{k562,rpe1}, nadig25{hepg2,jurkat}, arch1, kaden25rpe1
gsutil cp gs://scperteval/processed/wessels23_processed_complete.h5ad .
```

No gcloud account is needed — each file is also reachable over plain HTTPS at
`https://storage.googleapis.com/scperteval/processed/<dataset>_processed_complete.h5ad`.

## Run it

```bash
# protocols by name — including parameterised ones (set k / padj per protocol)
scperteval run data/wessels23.h5ad -p pearson_ctrl,unbiased_mmd_median_pca_k=20,de_overlap_k=10 --de-method t-test

# a parameterised protocol with no value uses its default (k=50, padj=0.05)
scperteval run data/wessels23.h5ad -p unbiased_mmd_median_top_k --de-method MWU

# a whole group, or everything (parameterised protocols use their defaults)
scperteval run data/wessels23.h5ad -p distributional --de-method MWU
scperteval run data/wessels23.h5ad -p all --de-method t-test

# DRF calibration only (compute DRF only; exclude BDS)
scperteval run data/wessels23.h5ad -p pearson_ctrl --de-method t-test --output drf

# DE only — writes per-gene statistic + adjusted p to HDF5 (no protocol calibration)
# Provided as a convenience, since DE methods are tightly coupled with some evaluation protocols
scperteval de data/wessels23.h5ad --methods MWU

# discover what's available
scperteval list protocols        # also: de-methods | spaces | sources | calibrators
```

Each run prints a summary table and writes a per-perturbation CSV
`<dataset>__<timestamp>__drf.csv` (the raw control values and the calibrated score for
every perturbation). `--profile` adds a per-protocol wall-clock timing CSV.

**DE backends** (`scperteval list de-methods`): `t-test` (default, Welch's, moment-based),
`MWU` (Cliff's δ via illico), and `t-test_overestim_var` (scanpy's conservative-variance
variant — the reference variance is scaled by the target's cell count). Select one with
`--de-method` for a `run`, or list several with `--methods` for a `de` export. The overestim
variant is a selectable backend for new protocols; no current protocol uses it.

<details><summary><code>scperteval run --help</code></summary>

```text
usage: scperteval run [-h] [-p PROTOCOLS] [--de-method {MWU,t-test,t-test_overestim_var}]
                [--subsample SUBSAMPLE] [--seed SEED] [--positive POSITIVE]
                [--negative NEGATIVE] [--output {drf,bds}] [--out-dir OUT_DIR]
                [--workers WORKERS] [--perturbation-key PERTURBATION_KEY]
                [--control-label CONTROL_LABEL] [--min-cells MIN_CELLS]
                [--profile] [--quiet]
                dataset

  -p, --protocols       comma-separated names (parameterised as name=value, e.g.
                        mse_top_k=30), a group (pseudobulk|distributional|de), or 'all'
  --de-method           {MWU, t-test, t-test_overestim_var}   DE backend for every DE unit:
                        the interpolated positive control, the top_k/degs spaces,
                        the de_* protocols, and the WMSE weights
  --subsample           cells in the single-cell reference sample (default 8192)
  --output              {drf, bds}      how per-perturbation values are calibrated
  --positive/--negative override a protocol's controls by source name
  --min-cells           skip perturbations with fewer cells
  --profile             also write a per-protocol wall-clock timing table
```

</details>

## Use it from Python

Install with `pip install scperteval` (or, from this repo,
`pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"`).
The simplest path mirrors the CLI — call it via subprocess, exactly as the figure notebook does:

```python
import subprocess, sys

subprocess.run([sys.executable, "-m", "scperteval", "run", "data/wessels23.h5ad",
                "-p", "all", "--de-method", "t-test", "--out-dir", "results"], check=True)
# -> results/wessels23__<timestamp>__drf.csv  (raw control values + calibrated DRF per perturbation)
```
