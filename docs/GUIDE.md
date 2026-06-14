# EPPS guide

## 1. Running the tool

EPPS computes per-metric calibration (DRF, or BDS) for the evaluation protocols
used to assess single-cell perturbation predictions, on one preprocessed dataset
per run.

```bash
# a few protocols
epps run data/wessels23.h5ad -p pearson_ctrl,mmd_top50,de_auprc

# everything, with the defaults (DRF, 8192 subsample, t-test)
epps run data/wessels23.h5ad -p all

# a group, a different DE backend, the BDS calibration
epps run data/wessels23.h5ad -p distributional --de-method MWU
epps run data/wessels23.h5ad -p all --output bds
```

Each run prints a summary table and writes a per-perturbation parquet named
`<dataset>__<timestamp>__<output>.parquet`, so running across datasets and
concatenating the files into a figure is a shell loop plus a `glob`.

```
$ epps run --help
usage: epps run [-h] [-p PROTOCOLS] [--de-method {t-test,MWU}] [--subsample SUBSAMPLE]
                [--seed SEED] [--positive POSITIVE] [--negative NEGATIVE]
                [--output {drf,bds}] [--out-dir OUT_DIR] [--workers WORKERS]
                [--perturbation-key PERTURBATION_KEY] [--control-label CONTROL_LABEL]
                [--quiet]
                dataset

positional arguments:
  dataset               preprocessed .h5ad

options:
  -p, --protocols       comma-separated names, a group (pseudobulk|distributional|de), or 'all'
  --de-method           t-test (default) or MWU
  --subsample           cell cap applied to every population (default 8192)
  --seed                default 42
  --positive/--negative control source, or 'auto' (default)
  --output              drf (default) or bds
  --workers             threads (0 = auto)
```

`epps list protocols | de-methods | spaces | sources | calibrators` shows what's available.

## 2. Reading a protocol's implementation

Every protocol is a pure function plus one declarative table row.

- The **algorithms** are in [`epps/protocols/algorithms.py`](../epps/protocols/algorithms.py) —
  read the function to see exactly what each metric computes (e.g. `mmd`, `pearson`,
  `de_auprc`).
- The **table** in [`epps/protocols/table.py`](../epps/protocols/table.py) wires each
  algorithm to its feature space, centering, controls, direction, and perfect score.

So `mmd_top50` is the `mmd` function evaluated in the `top50` space with the
technical-duplicate positive control and the all-perturbed-pool negative control —
all visible in one table row.

## 3. Adding a protocol

Two steps:

1. Write a pure function in `protocols/algorithms.py`:

   ```python
   def cosine(gt, cand, ctx):
       return 1.0 - float(gt @ cand / (np.linalg.norm(gt) * np.linalg.norm(cand)))
   ```

2. Add a row in `protocols/table.py`:

   ```python
   Protocol("cosine_ctrl", A.cosine, "profile", centering="ctrl",
            direction="lower", perfect=0.0, positive="interp", negative="mean", group="pseudobulk"),
   ```

`kind` is `profile` (1-D vectors), `population` (cells x genes), or `de`
(GT DEResult + candidate ranking). That's it — it shows up in `list`, `-p`, and groups.

## 4. Advanced building blocks

All of these are decorator registries, so a new one is a single drop-in.

**DE method** — `blocks/de.py`. Map two cell matrices to a `DEResult`
(`score`, `pvalue`, `pvalue_adj`, plus method-specific `extra`):

```python
@DE_METHODS.register("wilcoxon")
def de_wilcoxon(target, reference):
    ...
    return DEResult(score=..., pvalue=..., pvalue_adj=bh(...))
```

It is then usable via `--de-method wilcoxon`, and any DE-derived space or DE
protocol picks it up.

**Feature space** — `blocks/spaces.py`. A geometric space is a function
`(X, ctx, pert) -> dense array`; a DE-thresholded space is one line:

```python
register_de_space("cliff10", field="score", threshold=lambda v: np.abs(v) > 0.10)
register_de_space("top100",  field="score", top=100)
```

The uniform `DEResult` fields (`score`, `pvalue_adj`) mean a DE space works across
DE methods; `extra:<name>` reaches a method-specific output.

**Control source** — `sources.py`. A source yields a perturbation's `cells` or a
pseudobulk `profile`:

```python
@SOURCES.register("subsampled_control", provides="cells")
def src_subsampled_control(ctx, pert):
    ...
```

Then `--positive subsampled_control` (the runner checks cells-vs-profile compatibility).

**Calibration** — `calibrators.py`. A `Calibrator` declares the control roles it
needs, a per-perturbation combine, and a cross-perturbation aggregate:

```python
CALIBRATORS["my_score"] = Calibrator(
    "my_score", ("positive", "negative"), per_pert_fn, aggregate_fn)
```

Then `--output my_score`.
