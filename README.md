# scPertEval — Evaluation Protocols for Perturbation Sequencing

scPertEval is a command-line tool for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies. 

Evaluating predictions across a dataset's
perturbations reduces to a single question: how different is one group of cells from another? To answer this, an **evaluation protocol** is defined: a specific formulation of a metric, along with some representation of the perturbation data fed to the metric. However, there are a multitude of possibilities -- many already reflected in the literature -- and it can be challenging to compare and contrast protocols across the field and ultimately choose the right approach for a given dataset and problem space. 

scPertEval renders each protocol as a short, readable building block to run, read, reuse, and contribute back -- a place for
collaboration and alignment in the field.

The same catalog of protocols backs three commands, each a different use case:

- **`score`** — score a model's predictions against ground truth. Each protocol's metric is
  applied to your **predicted** cells vs the **real** cells, one score per perturbation — the
  conventional "how good is my prediction" evaluation (see
  [Scoring predictions](#scoring-predictions-against-ground-truth)).
- **`calibrate`** — calibrate a protocol against empirical positive/negative controls built from
  the dataset itself, reporting the **Dynamic Range Fraction (DRF)** and the **Bound Discrimination
  Score (BDS)** — quantifying how well the protocol separates real perturbation signal from an
  uninformative baseline (see [How calibration works](#how-calibration-works)). Use this to decide
  whether a metric is trustworthy in the first place.
- **`de`** — export per-gene differential expression (statistic + adjusted p) to HDF5, since DE
  is tightly coupled with several protocols.

Our accompanying publiciation: TODO_LINK_HERE

## Install

```bash
pip install -e .          # provides the `scperteval` command
```

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
scperteval calibrate data/wessels23.h5ad -p pearson_ctrl,unbiased_mmd_median_pca_k=20,de_overlap_k=10 --de-method t-test

# a parameterised protocol with no value uses its default (k=50, padj=0.05)
scperteval calibrate data/wessels23.h5ad -p unbiased_mmd_median_top_k --de-method MWU

# a whole group, or everything (parameterised protocols use their defaults)
scperteval calibrate data/wessels23.h5ad -p distributional --de-method MWU
scperteval calibrate data/wessels23.h5ad -p all --de-method t-test

# DRF calibration only (compute DRF only; exclude BDS)
scperteval calibrate data/wessels23.h5ad -p pearson_ctrl --de-method t-test --output drf

# SCORE predictions against ground truth — predicted cells vs real cells, per protocol.
# predictions.h5ad must have the same genes and perturbation labels as the dataset.
scperteval score data/wessels23.h5ad predictions.h5ad -p pearson,mse,de_auprc --de-method t-test

# DE only — writes per-gene statistic + adjusted p to HDF5 (no protocol calibration)
# Provided as a convenience, since DE methods are tightly coupled with some evaluation protocols
scperteval de data/wessels23.h5ad --methods MWU

# discover what's available
scperteval list protocols        # also: de-methods | spaces | sources | calibrators
```

Each command prints a summary table and writes a per-perturbation CSV named
`<dataset>__<timestamp>__<output>.csv`: `calibrate` writes the raw control values and the
calibrated DRF/BDS per perturbation (`…__drf.csv` / `…__bds.csv`); `score` writes the raw metric
value per perturbation (`…__score.csv`). `--profile` adds a per-protocol wall-clock timing CSV.

**DE backends** (`scperteval list de-methods`): `t-test` (default, Welch's, moment-based),
`MWU` (Cliff's δ via illico), and `t-test_overestim_var` (scanpy's conservative-variance
variant — the reference variance is scaled by the target's cell count). Select one with
`--de-method` for a `calibrate`/`score`, or list several with `--methods` for a `de` export. The overestim
variant is a selectable backend for new protocols; no current protocol uses it.

<details><summary><code>scperteval calibrate --help</code></summary>

```
usage: scperteval calibrate [-h] [-p PROTOCOLS] [--de-method {MWU,t-test,t-test_overestim_var}]
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

<details><summary><code>scperteval score --help</code></summary>

```
usage: scperteval score [-h] [-p PROTOCOLS] [--de-method {MWU,t-test,t-test_overestim_var}]
                [--subsample SUBSAMPLE] [--seed SEED] [--out-dir OUT_DIR] [--workers WORKERS]
                [--perturbation-key PERTURBATION_KEY] [--control-label CONTROL_LABEL]
                [--min-cells MIN_CELLS] [--profile] [--quiet]
                dataset predictions

  dataset               preprocessed .h5ad — the ground truth (real cells)
  predictions           predicted .h5ad — same genes and perturbation labels as the dataset
  -p, --protocols       comma-separated names, a group, or 'all'
  --de-method           DE backend for the de_* protocols, the top_k/degs spaces, and WMSE weights
  --subsample           cells in the all-perturbed reference (the ground truth is never subsampled)
```

Unlike `calibrate`, there are no `--positive`/`--negative`/`--output` options: the candidate is
always your prediction and the output is always the raw `score`.
</details>

## Use it from Python

Install with `pip install scperteval` (or, from this repo,
`pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"`).
The simplest path mirrors the CLI — call it via subprocess, exactly as the figure notebook does:

```python
import subprocess, sys

subprocess.run([sys.executable, "-m", "scperteval", "calibrate", "data/wessels23.h5ad",
                "-p", "all", "--de-method", "t-test", "--out-dir", "results"], check=True)
# -> results/wessels23__<timestamp>__drf.csv  (raw control values + calibrated DRF per perturbation)

# score predictions against ground truth instead:
subprocess.run([sys.executable, "-m", "scperteval", "score", "data/wessels23.h5ad",
                "predictions.h5ad", "-p", "all", "--out-dir", "results"], check=True)
# -> results/wessels23__<timestamp>__score.csv  (raw metric value per perturbation)
```

## Look up an Evaluation Protocol

Two files define each protocol:

- **[`scperteval/protocols/metrics.py`](scperteval/protocols/metrics.py)** — the metric, as a
  pure function of the ground truth and a `prediction` (the candidate being scored — a positive
  or negative control under `calibrate`, or your model's output under `score`). e.g. `mse`, `mmd`,
  `de_auprc`:
  ```python
  def mse(gt, prediction, ctx):
      return float(np.mean((gt - prediction) ** 2))
  ```
- **[`scperteval/protocols/table.py`](scperteval/protocols/table.py)** — one row wiring that function
  to its data: the data representation it receives (`representation`), feature space,
  reference centering, positive/negative controls, which direction is `better`
  (`"higher"`/`"lower"`), and the `perfect` score:
  ```python
  Protocol("mse", M.mse, representation="centroid",
           positive="interpolated", negative="all_perturbed_mean", better="lower", perfect=0.0)
  ```

The next section breaks these arguments down while building one up from scratch.

## Create a protocol

A protocol is two things: a pure metric **function** and a one-line **spec** that wires it
to data and scoring. We'll ease in — the simplest possible protocol first, then the spec
broken down, then a few richer examples.

### Start simple

Here is a complete new protocol: mean absolute error on the standard pseudobulk profiles.

1. Add a pure function to [`scperteval/protocols/metrics.py`](scperteval/protocols/metrics.py):
   ```python
   def mae(gt, prediction, ctx):
       return float(np.mean(np.abs(gt - prediction)))
   ```
   Every metric function has this signature. `gt` is one perturbation's ground-truth
   profile; `prediction` is the candidate being compared against it (under `calibrate`, scPertEval
   calls the function once for the positive control and once for the negative; under `score`, once
   with your model's prediction). `ctx` is the dataset context, needed by only a few metrics —
   ignore it otherwise. Return a single number.

2. Add a row to [`scperteval/protocols/table.py`](scperteval/protocols/table.py):
   ```python
   Protocol("mae", M.mae, representation="centroid",
            positive="interpolated", negative="all_perturbed_mean",
            better="lower", perfect=0.0)
   ```

Run it with `scperteval calibrate data.h5ad -p mae`. That is the whole protocol: MAE between each
perturbation's pseudobulk profile and its positive and negative controls, scored as
lower-is-better toward a perfect of 0.

### The spec

That row is the spec; parameters include:

| argument | meaning |
|---|---|
| `name` | selects the protocol on the CLI (`-p mae`) |
| `representation` | the shape of each datapoint your function receives (see below) |
| `scope` | `"perturbation"` (default) or `"dataset"` — how many perturbations at once (see below) |
| `space` | which features to score — `full` (default), or a feature space like `top_50` |
| `centering` | a baseline subtracted before scoring, e.g. `"ctrl"` (default: none) |
| `positive` / `negative` | the two control sources to compare |
| `better` | `"higher"` or `"lower"` — which direction is an improvement |
| `perfect` | the value a flawless prediction attains |
| `param` | optional — a parameter family (`top_k`, `pca_k`, `degs_padj`, `overlap_k`) that makes the protocol tunable from the CLI; omit for a fixed protocol |

**`representation`** decides the *shape* of each datapoint — the format `gt` and
`prediction` arrive in — so you never deal with sampling, references, or projection yourself:

| `representation` | a datapoint is |
|---|---|
| `centroid` | a 1-D pseudobulk vector (one value per gene) |
| `population` | a `(cells × genes)` matrix |
| `de` | a `DEResult` (for the ground truth) / per-gene `|score|` ranking (for a prediction) |

**`scope`** is the independent companion axis — *how many* perturbations the metric sees at once:

| `scope` | the metric is called |
|---|---|
| `perturbation` (default) | once per perturbation — gets that perturbation's `(gt, prediction)` datapoints and returns a scalar |
| `dataset` | once for the whole dataset — gets the **list** of every perturbation's `gt` and `prediction` datapoints and returns one score per perturbation (e.g. a retrieval `rank`) |

The two compose freely: `rank` is just `representation="centroid", scope="dataset"`; a
distributional retrieval metric would be `representation="population", scope="dataset"`.

Many rows repeat the same wiring, so the top of `table.py` predefines the common
combinations as plain dicts. You then unpack one into a row with `**` (Python's
keyword-expansion syntax) to avoid retyping it:
```python
_PB = dict(group="pseudobulk", positive="interpolated", negative="all_perturbed_mean")
_LOWER = dict(better="lower", perfect=0.0)
```
With those, the `mae` row above is exactly `Protocol("mae", M.mae,
representation="centroid", **_PB, **_LOWER)` — same protocol, less repetition. You'll see
these bundles reused throughout the table.

### Building blocks — the palette

The values those arguments take — feature spaces, control sources, DE methods, calibrators
— are registered building blocks. `scperteval list <category>` shows what's available
in each, with descriptions:

**Feature spaces** (the `space` argument)

```bash
$ scperteval list spaces
degs_0.05  — ground-truth DEGs at adjusted p < 0.05, per perturbation
full       — all genes, no transform
pca_50     — top 50 principal components (fit on the dataset)
top_50     — top 50 genes by ground-truth effect size, per perturbation
```

`top_<k>` / `pca_<k>` / `degs_<padj>` are parameterised families (the defaults are shown);
a protocol template picks the value. If the space you need isn't here, see
[Add a feature space](#add-a-feature-space).

**DE methods** (the `--de-method` choice)

```bash
$ scperteval list de-methods
MWU        — Mann-Whitney U / Cliff's delta effect size (via illico)
t-test     — Welch's t-test (default) — moment-based and fast
```

Chosen with `--de-method`; it applies to **every** DE-dependent unit (the `interpolated`
positive control, the `top_k`/`degs` spaces, the `de_*` protocols, and the WMSE weights).
To add another, see [Add a DE method](#add-a-de-method).

**Control sources** (the `positive` / `negative` arguments)

```bash
$ scperteval list sources
all_perturbed  (cells) — all-perturbed reference sample, leave-one-out (single-cell negative control)
all_perturbed_mean (centroid) — all-perturbed mean, excluding the target — leave-one-out (pseudobulk sibling of all_perturbed; pseudobulk negative control)
control        (cells) — non-targeting control cells
global_mean    (centroid) — mean of all perturbations — shared baseline for the ranking protocols
gt_all_cells   (cells) — ground truth — all of a perturbation's real cells (prediction-scoring truth)
gt_half        (cells) — ground truth — the first half of a perturbation's cells (calibration truth)
interpolated   (centroid) — interpolated duplicate — DE-weighted blend of the held-out half and the dataset mean (pseudobulk positive control)
prediction     (cells) — model-predicted cells for the perturbation, from the --predictions h5ad
tech_dup       (cells) — technical duplicate — the held-out second half (single-cell positive control)
```

The truth source is chosen by the command, not by a protocol: `calibrate` uses `gt_half` and
holds the other half out to build the positive control; `score` uses `gt_all_cells` and compares
it to `prediction`.

Each `provides` cells or a pseudobulk `centroid`. Use via `positive=`/`negative=` (or
`--positive`/`--negative`). To add another, see [Add a control source](#add-a-control-source).

**Calibrators** (the `--output` choice)

```bash
$ scperteval list calibrators
drf    — Dynamic Range Fraction — mean/median over perturbations (Miller et al. 2025)
bds    — Bound Discrimination Score — fraction of perturbations the positive control wins (SBB 2026)
score  — raw metric of a prediction vs ground truth — mean/median over perturbations (prediction-scoring mode)
```

`drf`/`bds` are chosen with `calibrate --output`; `score` is selected automatically by the
`score` command. To add another, see [Add a calibrator](#add-a-calibrator).

### More examples

With the spec and the palette in hand, richer protocols are just different combinations.

**Same wiring, different metric.** Cosine distance on pseudobulk reuses the bundles wholesale:
```python
def cosine(gt, prediction, ctx):
    return 1.0 - float(gt @ prediction / (np.linalg.norm(gt) * np.linalg.norm(prediction)))
```
```python
Protocol("cosine", M.cosine, representation="centroid", **_PB, **_LOWER)
```

**Restrict to a feature space.** Set `space` to score only some genes — e.g. MAE on the
top-50 DEGs:
```python
Protocol("mae_top50", M.mae, representation="centroid", space="top_50", **_PB, **_LOWER)
```

**Expose the space as a knob (parameterised).** To make `k` adjustable per invocation, add a
`param` to the same `Protocol(...)` row — nothing else changes. The row's name carries the
parameter, and the value is supplied on the CLI:
```python
Protocol("mae_top_k", M.mae, representation="centroid", param=top_k, **_PB, **_LOWER)
```
Then `scperteval calibrate data.h5ad -p mae_top_k=30` (or `mae_top_k` for the default `k=50`). The
families are `top_k` (top-k DEGs), `pca_k` (k PCs), and `degs_padj` (DEGs at adjusted
p < padj) for the space, and `overlap_k` to feed an integer straight to the metric.

**A metric over cells, not profiles.** Switch `representation` to `population` and your
function receives `(cells × genes)` matrices; pair it with the single-cell controls:
```python
def my_mmd(gt, prediction, ctx):      # gt, prediction are (cells × genes)
    ...
```
```python
Protocol("my_mmd_top50", M.my_mmd, representation="population", space="top_50",
         positive="tech_dup", negative="all_perturbed", better="lower", perfect=0.0)
```
This changes two pieces at once — the `representation` (so the function sees cells) and the controls
(the single-cell positive/negative) — which is the general pattern for a distributional
protocol.

By now you've seen every moving part: the function, the spec, the building blocks the spec
draws on, fixed and parameterised spaces, and switching the representation the function
sees. Most new metrics are some combination of these.

## Add a building block

Spaces, DE methods, control sources, and calibrators are registered units — add one when
the palette is missing what a new protocol needs. Each is a small function (or object) plus
a one-line registration.

### Add a feature space

A space is a function `(X, ctx, pert) -> dense (cells × genes) array` that transforms the
gene axis. Register it with `@SPACES.register` in
[`scperteval/blocks/spaces.py`](scperteval/blocks/spaces.py); pass `global_space=True` if it doesn't
depend on the perturbation (so it can be computed once and shared):

```python
@SPACES.register("hvg_100", global_space=True, description="100 highest-variance genes")
def space_hvg(X, ctx, pert):
    keep = ...                       # indices of the genes to keep
    return to_dense(X[:, keep])
```

For a per-perturbation subset derived from the ground-truth DE (like `top_k` / `degs`), use
the `register_de_space(name, field=..., top=...)` helper in the same file instead.

### Add a DE method

A DE method maps `(target_cells, reference_cells) -> DEResult(score, pvalue, pvalue_adj)`.
Register it with `@DE_METHODS.register` in [`scperteval/blocks/de.py`](scperteval/blocks/de.py) (the
`bh` helper there BH-adjusts p-values):

```python
@DE_METHODS.register("my_test", description="…")
def de_my_test(target, reference):
    score, pvalue = ...              # per-gene statistic and raw p-value
    return DEResult(score=score, pvalue=pvalue, pvalue_adj=bh(pvalue))
```

Then `--de-method my_test` routes every DE-dependent unit through it.

### Add a control source

A source maps `(ctx, pert) -> cells or a 1-D centroid`, declaring which with `provides`.
Register it with `@SOURCES.register` in [`scperteval/sources.py`](scperteval/sources.py):

```python
@SOURCES.register("my_baseline", provides="centroid", description="…")
def src_my_baseline(ctx, pert):
    return ...                       # a 1-D centroid (or cells, if provides="cells")
```

Use it as a control via `positive=`/`negative=` in a row, or `--positive`/`--negative` at
the CLI.

### Add a calibrator

A calibrator declares the control roles it needs, a per-perturbation combine, and a
cross-perturbation aggregate. Add a `Calibrator` to the `CALIBRATORS` dict in
[`scperteval/calibrators.py`](scperteval/calibrators.py):

```python
CALIBRATORS["my_score"] = Calibrator(
    "my_score", ("positive", "negative"),
    per_pert=lambda raws, p: ...,          # raws["positive"], raws["negative"] -> one number
    aggregate=lambda v: {"my_score": float(np.nanmean(v))},
    description="…",
)
```

Then `--output my_score` reports it.

## Scoring predictions against ground truth

`scperteval score dataset.h5ad predictions.h5ad` is the conventional evaluation: each protocol's
metric is applied to your **predicted** cells against the **real** cells, one score per
perturbation. It runs the *same* protocol catalog as `calibrate`; only two pieces differ.

- **ground truth** — *all* of a perturbation's real cells (the `gt_all_cells` source). Unlike
  calibration, no half is held out and no positive/negative controls are built — the ground
  truth is the whole real population.
- **prediction** — the matching cells from your `predictions.h5ad` (the `prediction` source).
  The prediction file must contain the dataset's exact gene set (any order — columns are
  reordered by name so the comparison lines up gene-for-gene) and the same perturbation labels.
  A gene-set mismatch, or a perturbation present in the dataset but absent from the predictions,
  raises an error naming exactly what's wrong.

The `score` calibrator reports each protocol's raw metric value per perturbation and its
mean/median across perturbations, written to `<dataset>__<timestamp>__score.csv`. Higher- vs
lower-is-better follows each protocol's `better` field, exactly as in calibration.

Architecturally this reuses everything — the per-perturbation loop, every metric, representation,
and feature space are shared with `calibrate`. The only differences are the **truth source**
(`gt_all_cells` instead of the held-out `gt_half`) and the **calibrator** (`score`, which needs
only the prediction, instead of `drf`/`bds`, which need both controls). The DE-derived feature
spaces (`top_k`, `degs`) and the WMSE weights are computed from this same all-cells ground truth.

## How calibration works

scPertEval's claim — a usable catalog of protocols — rests on **calibrating** each protocol
against two empirical controls per perturbation, so you can see whether a metric actually
separates signal from baseline rather than read a raw, uninterpretable number.

- **positive control** — the best realistic candidate: the **technical duplicate** (a
  held-out replicate) for single-cell protocols, the **interpolated duplicate** for pseudobulk.
- **negative control** — an uninformative baseline: the **all-perturbed reference,
  excluding the target perturbation** (a full-resolution mean for pseudobulk; an 8192-cell
  subsample for single-cell distances).

**Dynamic Range Fraction (DRF)** — where the protocol's value sits between the negative
control (floor) and the perfect score, anchored by the positive control:

```
DRF = (positive − negative) / (perfect − negative)        # per perturbation, clipped to [-1, 1]
```

`--output drf` reports the mean/median across perturbations. High DRF means the protocol
discriminates real signal; near zero means it doesn't. Introduced by Miller et al.,
*Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on
Well-Calibrated Metrics* (2025) — <https://doi.org/10.1101/2025.10.20.683304>.

**Bound Discrimination Score (BDS)** — the fraction of perturbations for which the positive
control beats the negative control under this protocol:

```
BDS = fraction of perturbations where  positive control beats negative control     # in [0, 1]
```

`--output bds` reports this fraction. It's a sensitivity check: a protocol with low BDS
can't even tell a technical replicate from an uninformative baseline, so its scores
shouldn't be trusted. Introduced by Vollenweider & Bühlmann, *Signal, Bounds, and
Baselines* (SBB, 2026) — <https://doi.org/10.64898/2026.04.20.719650> (code:
<https://github.com/michavol/sbb-perturbation-benchmark>).

---

**Contributing:** see [CONTRIBUTORS.md](CONTRIBUTORS.md).
