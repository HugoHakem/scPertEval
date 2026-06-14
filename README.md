# EPPS — Evaluation Protocols for Perturbation Sequencing

EPPS is a place to **experiment with, create, and share reference implementations of
evaluation protocols** for single-cell perturbation studies. The field scores predicted
perturbations in many different ways — different feature spaces, reference centerings,
and metrics. EPPS turns each one into a short, readable building block you can run, read,
reuse, and contribute back.

It does this by **calibrating** each protocol rather than reporting a raw, uninterpretable
number: for every protocol EPPS provides calibration data — the **Dynamic Range Fraction
(DRF)** and the **Bound Discrimination Score (BDS)** — that quantify how well the protocol
separates real perturbation signal from an uninformative baseline (see
[How scoring works](#how-scoring-works-the-calibration)). Every protocol is a **pure
function plus a one-line recipe**: read the function to see the maths, read the recipe to
see how it's wired and scored.

## Install

```bash
pip install -e .          # provides the `epps` command
```

The only input is one preprocessed `.h5ad`: log-normalized `X`, a `perturbation` obs
column, and a `control` label. Every reference (technical duplicate, interpolated
duplicate, all-perturbed sample, control) is derived in memory.

## Run it

```bash
# protocols by name — including parameterised ones (set k / padj per protocol)
epps run data/wessels23.h5ad -p pearson_ctrl,mmd_pca_k=20,de_overlap_k=10

# a parameterised protocol with no value uses its default (k=50, padj=0.05)
epps run data/wessels23.h5ad -p mmd_top_k

# a whole group, or everything (parameterised protocols use their defaults)
epps run data/wessels23.h5ad -p distributional
epps run data/wessels23.h5ad -p all

# discover what's available
epps list protocols        # also: de-methods | spaces | sources | calibrators
```

Each run prints a summary table and writes a per-perturbation CSV
`<dataset>__<timestamp>__drf.csv` (the raw control values and the calibrated score for
every perturbation). `--profile` adds a per-protocol wall-clock timing CSV. (`epps de`
writes per-gene DE matrices to HDF5, which stay binary.)

<details><summary><code>epps run --help</code></summary>

```
usage: epps run [-h] [-p PROTOCOLS] [--de-method {MWU,t-test}]
                [--subsample SUBSAMPLE] [--seed SEED] [--positive POSITIVE]
                [--negative NEGATIVE] [--output {drf,bds}] [--out-dir OUT_DIR]
                [--workers WORKERS] [--perturbation-key PERTURBATION_KEY]
                [--control-label CONTROL_LABEL] [--min-cells MIN_CELLS]
                [--profile] [--quiet]
                dataset

  -p, --protocols       comma-separated names (parameterised as name=value, e.g.
                        mmd_top_k=30), a group (pseudobulk|distributional|de), or 'all'
  --de-method           {MWU, t-test}   DE backend for EVERY DE-dependent unit:
                        the interp positive control, the top_k/degs spaces, the
                        de_* protocols, and the WMSE weights
  --subsample           cells in the single-cell reference sample (default 8192)
  --output              {drf, bds}      how per-perturbation values are calibrated
  --positive/--negative override a protocol's controls by source name
  --min-cells           skip perturbations with fewer cells
  --profile             also write a per-protocol wall-clock timing table
```
</details>

## Examine an existing Evaluation Protocol

Two files define each protocol:

- **[`epps/protocols/algorithms.py`](epps/protocols/algorithms.py)** — the metric, as a
  pure function of the ground truth and a `prediction` (whichever control is being scored,
  positive or negative). e.g. `mse`, `mmd`, `de_auprc`:
  ```python
  def mse(gt, prediction, ctx):
      return float(np.mean((gt - prediction) ** 2))
  ```
- **[`epps/protocols/table.py`](epps/protocols/table.py)** — one row wiring that function
  to its data: feature space, reference centering, positive/negative controls, direction,
  and perfect score:
  ```python
  Protocol("mse", A.mse, "centroid",
           positive="interp", negative="mean", direction="lower", perfect=0.0)
  ```

Look at the function to see the algorithm; look at the row to see the protocol.

## Create a protocol

### Common case — reuse existing pieces (no new DE or space)

Most new protocols are just a new metric over spaces and controls that already exist. Two
steps — here, adding mean absolute error:

1. Add a pure function to [`epps/protocols/algorithms.py`](epps/protocols/algorithms.py):
   ```python
   def mae(gt, prediction, ctx):
       return float(np.mean(np.abs(gt - prediction)))
   ```
2. Add a row to [`epps/protocols/table.py`](epps/protocols/table.py):
   ```python
   Protocol("mae", A.mae, "centroid", **_PB, **_LOWER)
   ```

Then `epps run data.h5ad -p mae`. Your function receives ready-to-use views — you never
touch sampling, references, or projection. The protocol's `kind` decides the views:

| kind | your function gets |
|---|---|
| `centroid` | two 1-D pseudobulk vectors `(gt, prediction)` |
| `population` | two `(cells × genes)` arrays |
| `de` | `(gt DEResult, prediction \|score\| ranking)` |

### Parameterised protocols (k / padj)

To expose a knob — top-k genes, PCA dimensions, a DEG threshold — register a **template**
instead of a plain row. Its name carries the parameter, and the value is supplied per
protocol at the CLI:

```python
# MAE on the top-k DEGs; the parameter selects the feature space
_space_tpl("mae_top_k", A.mae, "centroid", top_space, "k", int, 50,
           "MAE on the top-k DEGs", {**_PB, **_LOWER})
```

Then `epps run data.h5ad -p mae_top_k=30` (or `mae_top_k` for the default k=50). The
existing families are `top_space` (top-k DEGs), `pca_space` (k PCs), and `degs_space`
(DEGs at adjusted p < padj).

### Building blocks — the current palette

Protocols are assembled from these registered units; `epps list <kind>` shows each with a
description. A new protocol usually reuses them — add to a kind only when the catalog is
missing what you need.

**Feature spaces**

```bash
$ epps list spaces
degs_0.05  — ground-truth DEGs at adjusted p < 0.05, per perturbation
full       — all genes, no transform
pca_50     — top 50 principal components (fit on the dataset)
top_50     — top 50 genes by ground-truth effect size, per perturbation
```

`top_<k>` / `pca_<k>` / `degs_<padj>` are parameterised families (the defaults are shown);
a protocol template picks the value. Add one with `@SPACES.register(...)` /
`register_de_space(...)` in [`epps/blocks/spaces.py`](epps/blocks/spaces.py).

**DE methods**

```bash
$ epps list de-methods
MWU        — Mann-Whitney U / Cliff's delta effect size (via illico)
t-test     — Welch's t-test (default) — moment-based and fast
```

Chosen with `--de-method`; it applies to **every** DE-dependent unit (the `interp` positive
control, the `top_k`/`degs` spaces, the `de_*` protocols, and the WMSE weights). Add one
with `@DE_METHODS.register(...)` in [`epps/blocks/de.py`](epps/blocks/de.py).

**Control sources**

```bash
$ epps list sources
all_perturbed  (cells) — all-perturbed reference sample, leave-one-out (single-cell negative control)
control        (cells) — non-targeting control cells
global_mean    (centroid) — mean of all perturbations — shared baseline for the ranking protocols
gt             (cells) — ground truth — the first half of a perturbation's cells
interp         (centroid) — interpolated duplicate — DE-weighted blend of the held-out half and the dataset mean (pseudobulk positive control)
mean           (centroid) — mean of all perturbations except the target (pseudobulk negative control)
tech_dup       (cells) — technical duplicate — the held-out second half (single-cell positive control)
```

Each `provides` cells or a pseudobulk `centroid`. Use via `positive=`/`negative=` (or
`--positive`/`--negative`). Add one with `@SOURCES.register(..., provides="cells"|"centroid")`
in [`epps/sources.py`](epps/sources.py).

**Calibrators**

```bash
$ epps list calibrators
drf    — Dynamic Range Fraction — mean/median over perturbations (Miller et al. 2025)
bds    — Bound Discrimination Score — fraction of perturbations the positive control wins (SBB 2026)
```

Chosen with `--output`. Add one as a `Calibrator` in
[`epps/calibrators.py`](epps/calibrators.py).

## How scoring works (the calibration)

EPPS's claim — a usable catalog of protocols — rests on **calibrating** each protocol
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

**Contributing:** see [CONTRIBUTORS.md](CONTRIBUTORS.md). A deeper walkthrough lives in
[`docs/GUIDE.md`](docs/GUIDE.md).
