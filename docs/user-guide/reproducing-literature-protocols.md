# Reproducing evaluation protocols from the literature

*Draft — intended as the opening section of a tutorial notebook (not yet wired into the docs
build or the tutorials submodule).*

Two recent papers benchmark the same class of models — deep-learning perturbation-response
predictors — on overlapping perturbation datasets, and reach opposite headline conclusions:

- {cite}`Miller_2025`, *"Deep Learning-Based Genetic Perturbation Models Do Outperform
  Uninformative Baselines on Well-Calibrated Metrics"*
- {cite}`AhlmannEltze_2025`, *"Deep learning-based predictions of gene perturbation effects do
  not yet outperform simple linear baselines"*

The disagreement is not really about the models. It is about the **evaluation protocol**: which
representation, which metric, which baseline, and which notion of "outperform" each paper uses.
That is exactly the axis scPertEval is built to make explicit and swappable — a protocol here is
a declarative bundle of a representation, a metric, and a pair of controls (see
[Protocols](protocols.md) and [Calibration](calibration.md)), and different bundles can rank the
very same model predictions differently.

This page walks through both protocols so a follow-up tutorial can reproduce each one on the same
dataset and the same set of model predictions, and show concretely how the ranking changes
depending on which protocol is applied — which is, in a sentence, the whole premise of
scPertEval.

## Miller et al. 2025 — calibrate the metric before trusting it

Miller et al.'s starting observation is that a benchmark verdict ("model X beats the mean
baseline") is only meaningful if the metric used has room to distinguish a good prediction from a
bad one *for that dataset and that perturbation*. Many perturbations are weak — most genes simply
aren't differentially expressed — so a metric computed over the full transcriptome can have almost
no dynamic range between an uninformative baseline and the best a model could physically achieve.
Declaring a model "no better than the mean" on such a metric conflates "the model is bad" with "the
metric can't tell the difference."

Their fix, implemented in their `cellsimbench` framework, is to calibrate every metric with two
empirical controls before trusting its verdict:

- a **positive control** — a technical duplicate (a held-out half of a perturbation's own cells),
  or, for pseudobulk metrics, an **interpolated duplicate**: a per-gene blend of the duplicate and
  the dataset mean, weighted by how significantly differentially expressed each gene is. This
  approximates the best a real, unbiased predictor could achieve.
- a **negative control** — the dataset mean (or control-cell mean), an intentionally uninformative
  baseline.

**Dynamic Range Fraction (DRF)** then asks how much of the gap between these two controls a given
metric actually resolves:

$$
\operatorname{DRF} = \frac{s_{\text{negative}} - s_{\text{positive}}}{s_{\text{negative}} - s_{\text{perfect}} + \xi}
$$

A metric with high DRF has real headroom to reward a good model; a metric with DRF near zero
cannot distinguish signal from noise on that dataset, regardless of which model is plugged in.
Once restricted to well-calibrated metrics, Miller et al. find that deep-learning models *do*
separate from the mean/control baselines — hence the paper's title.

scPertEval's `calibrate` mode is already a close port of this framework — `tech_dup`,
`interpolated`, `all_perturbed_mean`, and the `drf` calibrator (`scperteval calibrate ... --output
drf`) implement exactly the controls and formula above (see [Calibration](calibration.md)).

**Splitting perturbations for cross-validation.** Held-out perturbations are chosen by genuine
5-fold cross-validation (2-fold for the double-perturbation datasets — Norman19, Wessels23, and,
despite its name, Sunshine23, which turns out to be combo-structured throughout, not single-gene).
A fixed, hardcoded seed (42) permutes the non-control conditions once per dataset and cuts them
into contiguous blocks, so every perturbation is held out as the test fold exactly once across all
folds; a model is retrained independently per fold, and the reported metrics are computed once on
the concatenation of every fold's test predictions. It is a full-coverage CV aggregate, not a
single held-out sample — every perturbation contributes exactly one test-time observation.

## Ahlmann-Eltze, Huber & Anders 2025 — score predictions directly, then bootstrap the comparison

The linear-baseline paper takes a much more direct route: no calibration step, no controls beyond
a single fixed baseline — just a head-to-head comparison of each model's predicted pseudobulk
profile against the observed one.

- **Gene set**: the 1,000 genes with the highest expression in the *control* condition (not the
  genes with the largest predicted or observed effect — this is a fixed, model-independent panel
  chosen before any prediction is scored).
- **Headline metric ("Pearson Delta")**: the Pearson correlation between the predicted and
  observed *change from control*,
  $$
  r_\Delta = \operatorname{corr}\big(\hat y - y_{\text{ctrl}},\; y - y_{\text{ctrl}}\big),
  $$
  computed per perturbation, per dataset, restricted to that 1,000-gene panel. A companion
  Euclidean-distance metric ($\ell_2$ on raw profiles) is reported alongside it.
- **Baselines**: a **Mean** baseline (the average pseudobulk profile across training
  perturbations) and a **linear model** — a low-rank, ridge-regularized bilinear regression
  trained on the same PCA-derived gene and perturbation embeddings — compared against several deep
  learning methods (GEARS, scGPT, and others).
- **"Outperform," operationalized**: rather than a p-value, the paper bootstraps (10,000
  resamples) the mean of each model's per-perturbation error *relative to the Mean baseline*
  ($\ell_2 / \ell_2^{\text{mean}}$) and reports the resulting 95% CI as a forest plot. A model is
  said to outperform the baseline only if its CI sits entirely below 1 (and doesn't overlap a
  competing baseline's CI) — most deep-learning models' CIs do not clear that bar, hence the
  paper's title.
- **Combinatorial (two-gene) perturbations** are scored the same way, against an **additive**
  baseline built from the two single-gene effects: $\widehat{AB} = y_{\text{ctrl}} + (y_A -
  y_{\text{ctrl}}) + (y_B - y_{\text{ctrl}})$. Deviations from this additive expectation are used
  separately to flag synergy/buffering, via an empirical-null model — a diagnostic, not part of
  the headline metric.

The gene-selection-by-control-expression space, the raw-profile `l2` metric, and the
bootstrap-CI comparison now exist natively:

```bash
# "Pearson Delta" and l2, restricted to the top-1000 control-expressed genes
scperteval score data.h5ad predictions.h5ad -p pearson_ctrl_expr_k=1000,l2_expr_k=1000

# the bootstrap-CI "outperform" question, prediction vs. a chosen baseline source
scperteval score data.h5ad predictions.h5ad -p pearson_ctrl_expr_k=1000 \
  --positive prediction --negative global_mean --output paired_ci
```

`pearson_expr_k`/`pearson_ctrl_expr_k`/`l2_expr_k` (`blocks/spaces.py`'s `expr_space`,
`protocols/metrics.py`'s `l2`) reproduce the paper's `r2`/`r2_delta`/`l2`. The `paired_ci`
calibrator (`calibrators.py`) reproduces the *shape* of both papers' significance question — a
bootstrapped 95% CI on a paired per-perturbation gap — as a signed difference oriented by the
protocol's own `better` direction, rather than Ahlmann-Eltze's specific ratio
($\ell_2/\ell_2^{\text{mean}}$), so it applies uniformly to correlation- and error-type metrics
alike; it does not implement Miller et al.'s cross-model Bonferroni correction, which needs several
models' p-values at once rather than a single run. The additive baseline source for combinatorial
perturbations still does not exist.

**Splitting perturbations for cross-validation.** Held-out perturbations are chosen very
differently from Miller et al.'s CV. Single-gene datasets use GEARS's `simulation` split, which
holds out a random 25% of *genes* (not conditions, and not a rotating fold) in one shot, repeated
for only **2** independent seeds — and the two seeds' held-out perturbations are then pooled
together for the headline plots as if they were one larger sample, rather than averaged per seed,
with no guarantee the two seeds' held-out genes jointly cover the full gene set. The
double-perturbation (Norman) analysis instead redraws its train/test/holdout combo split
independently for **5** seeds, again with no coverage guarantee across seeds. Either way, this is
repeated random subsampling of a fixed size, not exhaustive coverage — a materially different
notion of "held out" than Miller et al.'s 5-fold CV, and part of why the two papers' held-out
perturbation sets aren't even the same *kind* of object to compare.

## Why they disagree, and what a reproduction would show

Both papers draw on overlapping Perturb-seq datasets and overlapping model zoos, yet reach
opposite verdicts, because they disagree on essentially every axis a protocol fixes: whole
transcriptome vs. a fixed 1,000-gene control-expression panel; raw metric value vs.
controls-calibrated DRF; a single Mean baseline vs. an explicit noise-ceiling positive control; a
bootstrap CI on relative error vs. a dynamic-range fraction. Neither is "wrong" — they are
answering different questions about the same predictions.

That is precisely scPertEval's founding premise: a model ranking is a function of the protocol,
not just of the model. A follow-up tutorial should take one shared set of model predictions and
run:

| What the paper reports | scPertEval today |
|---|---|
| Miller et al.'s DRF/BDS calibration | `scperteval calibrate ... --output drf` — native |
| Miller et al.'s model-vs-baseline ranking (their second, model-comparison DRF) | `scperteval score ... --positive prediction --negative <baseline> --output drf` — native, reusing the same `drf` calibrator (clipped to `[-1, 1]`, not cellsimbench's `[-1, 2]`) |
| Ahlmann-Eltze's Pearson Delta / raw $r$ / $\ell_2$ on the top-1000 control-expressed genes | `pearson_ctrl_expr_k` / `pearson_expr_k` / `l2_expr_k` — native |
| Both papers' bootstrap-CI "outperform" question | `scperteval score ... --positive prediction --negative <baseline> --output paired_ci` — native, as a signed paired difference rather than Ahlmann-Eltze's specific ratio; no cross-model Bonferroni correction |
| The additive combinatorial baseline | not yet available — an `additive` source, built from constituent single-perturbation ground truth |
| Miller et al.'s 5-fold perturbation CV vs. Ahlmann-Eltze's repeated random gene-holdout | neither is a protocol concern in scPertEval today — held-out perturbations come from the dataset, not the protocol table; would need a dataset-preparation step exposing fold/seed columns, analogous to `get_data.py`'s `split_fold_N`, kept separate from the protocol/metric layer |

— and show, side by side, that the same models get ranked differently depending on which row of
that table is applied.

## Appendix: exact split mechanics (for byte-level reproduction)

**Miller et al. / cellsimbench**, per single-gene dataset's `get_data.py`: `np.random.seed(42)` →
permute non-control conditions → cut into 5 contiguous blocks, one pre-assigned as each fold's test
set → the remaining 80% per fold split ~69:11 train:val (the code's own comment claims a 70:10
split; the actual arithmetic — 13.75% of the *remaining* 80%, not of the total — yields ~11%, a
minor discrepancy worth knowing about rather than silently matching). Control cells are shuffled
and split the same way, independently of the technical-duplicate half-split. All 5 folds are always
run, each with an independently trained model, and metrics are computed once on the concatenation
of all folds' test predictions.

**Ahlmann-Eltze et al. / GEARS `simulation` split**, for non-Norman datasets: two nested calls to
GEARS's `get_simulation_split`, both reseeded with the same seed — first splits genes 75%
train-candidates / 25% held-out-test; the 75% is then re-split 90/10 into final train/val. Net:
≈67.5% train / 7.5% val / 25% test genes, a single random partition per seed (not a rotating fold),
with `ctrl` always forced into train. Single-gene benchmark jobs use only seeds `{1, 2}`; the
Norman double-perturbation job uses 5 independent seeds, each redrawing its own combo split from
scratch. Results from different seeds are pooled as additional independent observations in the
final plots, not averaged into a per-seed statistic.

