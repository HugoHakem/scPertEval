# Reproducing evaluation protocols from the literature

*Draft — intended as the opening section of a tutorial notebook (not yet wired into the docs
build or the tutorials submodule).*

Two recent papers benchmark the same class of models — deep-learning perturbation-response
predictors — on overlapping perturbation datasets, and reach opposite headline conclusions:

- {cite}`Miller_2025`, *"Deep Learning-Based Genetic Perturbation Models Do Outperform
  Uninformative Baselines on Well-Calibrated Metrics"*
- {cite}`AhlmannEltze_2025`, *"Deep learning-based predictions of gene perturbation effects do
  not yet outperform simple linear baselines"*

A third, related preprint, {cite}`Vollenweider_2026` (*"Signal, Bounds, and Baselines: Principles
for Rigorous Evaluation of High-Dimensional Biological Perturbation Prediction"*), builds directly
on Miller et al.'s metric family — most concretely, extending its DE-effect-size weighting scheme
to Pearson correlation as the **Weighted Pearson Delta** (see the Miller et al. section below).

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
  the dataset mean, weighted by how significantly differentially expressed each gene is (DEGs
  computed against *all other perturbed cells*, the same reference population used everywhere
  else in the paper — not control). This approximates the best a real, unbiased predictor could
  achieve.
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

**Their own metric family.** Calibration is orthogonal to *which* metric gets calibrated — Miller
et al. also propose a specific family of pseudobulk metrics, each scored on the prediction's
deviation from a reference point ("delta") rather than the raw profile:

- **MSE** — mean squared error on the raw profile.
- **PearsonDeltaCtrl** / **PearsonDeltaPerturbMean** — Pearson correlation between the predicted
  and observed change from control, or from the leave-one-out mean of all *other* perturbations.
- **R2DeltaCtrl** / **R2DeltaPerturbMean** — the same two deltas, scored by the coefficient of
  determination ($R^2 = 1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$, floored at $-1$) instead of
  Pearson $r$. Unlike Pearson, $R^2$ penalises bias and scale errors, not just linear association
  — a prediction that is perfectly correlated with, but off-scale from, the truth still loses $R^2$.
- **NIR (Normalized Inverse Rank)** — for each perturbation, rank its predicted centroid's
  distance to *every* perturbation's ground-truth centroid, normalise to $[0, 1]$, and invert so
  1 is a perfect top-1 retrieval.

Miller et al.'s own gene-set axis has only two levels per metric — MSE and the two Pearson
deltas each come as every gene or only significant DEGs ($p_{\text{adj}} < 0.05$); **WMSE**
(their weighted-MSE variant) and the two R² deltas additionally get a *third*, continuously
DE-effect-size-weighted variant, using the per-gene weight
$w_g \propto |t_g|^{\text{exp}} / \sum_{g'} |t_{g'}|^{\text{exp}}$ ($t_g$ the ground-truth DE
t-statistic). That gives Miller et al. 2025's own count of 13 metrics exactly: MSE + WMSE +
(Pearson: 2 baselines × 2 gene-sets = 4) + ($R^2$: 2 baselines × 3 gene-sets = 6) + NIR = 13.
{cite}`Vollenweider_2026` (*"Signal, Bounds, and Baselines"*) then extends the same
effect-size weighting to Pearson correlation too — the **Weighted Pearson Delta** — which is
*not* one of Miller et al.'s own 13.

All of this now exists natively:

| Metric | scPertEval protocol |
|---|---|
| MSE | `mse` (all genes) / `mse_degs_padj` (DEG) |
| WMSE | `wmse_exp1`, `wmse_exp2`, `wmse_exp4` (DE-weighted, three exponents) |
| PearsonDeltaCtrl | `pearson_ctrl` (all genes) / `pearson_ctrl_degs_padj` (DEG) |
| PearsonDeltaPerturbMean | `pearson_pert` (all genes) / `pearson_pert_degs_padj` (DEG) |
| R2DeltaCtrl | `r2_ctrl` (all genes) / `r2_ctrl_degs_padj` (DEG) / `weighted_r2_ctrl_exp2` (DE-weighted) |
| R2DeltaPerturbMean | `r2_pert` (all genes) / `r2_pert_degs_padj` (DEG) / `weighted_r2_pert_exp2` (DE-weighted) |
| NIR | `nir` |
| Weighted Pearson Delta ({cite}`Vollenweider_2026`, not one of Miller's 13) | `weighted_pearson_ctrl_exp2` / `weighted_pearson_pert_exp2` |

```bash
# Miller et al.'s DE-weighted delta correlation, restricted to significant DEGs / continuously weighted
scperteval score data.h5ad predictions.h5ad -p pearson_ctrl_degs_padj=0.05,weighted_r2_ctrl_exp2,nir
```

Every row can be combined with the DRF/BDS calibration above, or scored directly against a
model's predictions (`scperteval score`) — calibration and metric choice are independent axes
of the same protocol table.

**Statistical tests.** Miller et al. use three: a bootstrap 95% CI (shared with Ahlmann-Eltze et
al. below), a paired one-sided Student t-test, and a paired one-sided Wilcoxon signed-rank test.
These aren't scPertEval calibrators: `--positive`/`--negative` role resolution compares one
attached prediction (or dataset-native source) against another, but comparing several trained
models against each other — or against a baseline that is itself a fold-scoped, trained
prediction set rather than a dataset-native source — means pairing two independent prediction
sets directly, which role resolution can't express. `models/compare.py`'s worked example does
this directly instead: a two-sided Wilcoxon signed-rank test per model pair, Holm-corrected
within each protocol (uniformly more powerful than a flat Bonferroni across every comparison,
while still controlling the family-wise error rate), plus a bootstrap "Bayes Factor" robustness
check (Menden et al. 2019's DREAM Challenge convention) as a complementary view.

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
```

`pearson_expr_k`/`pearson_ctrl_expr_k`/`l2_expr_k` (`blocks/spaces.py`'s `expr_space`,
`protocols/metrics.py`'s `l2`) reproduce the paper's `r2`/`r2_delta`/`l2`. The bootstrap-CI
"outperform" question — is a model's per-perturbation gap against a baseline distinguishable from
zero — isn't a scPertEval calibrator here either: Ahlmann-Eltze et al.'s baselines (Mean, linear)
are themselves fold-scoped, trained prediction sets, so comparing a model against them means
pairing two independent prediction files, not one attached prediction against a dataset-native
source. `models/compare.py`'s bootstrap "Bayes Factor" (Menden et al. 2019's DREAM Challenge
convention) reproduces the *shape* of this question directly on two prediction sets' paired
per-perturbation gap, oriented by the protocol's own `better` direction rather than
Ahlmann-Eltze's specific ratio ($\ell_2/\ell_2^{\text{mean}}$), so it applies uniformly to
correlation- and error-type metrics alike.

The **Mean** and **linear** baselines (`models/baselines/baselines.py`'s `fold_mean_baseline`
and `linear_baseline`, ported from const-ae/linear_perturbation_prediction-Paper's
`run_linear_pretrained_model.R`) are implemented in the `models/` reproduction scaffold — a
baseline choice is a model/prediction-source decision, not a protocol-table concept, so it
lives alongside GEARS/scGPT rather than in `src/scperteval/`. The additive baseline source for
combinatorial perturbations still does not exist.

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
| Miller et al.'s model-vs-baseline ranking (their second, model-comparison DRF) | `models/compare.py`'s `per_pert_drf`, via `scperteval.calibrators.drf_per_pert` (clipped to `[-1, 1]`, not cellsimbench's `[-1, 2]`) applied to two independently-scored (`sp.score`) prediction sets — deliberately *not* `scperteval score --positive prediction --output drf`: that would calibrate against a dataset-native control (e.g. `global_mean`) computed over the whole ground-truth file passed in, which can leak information a model's own held-out fold shouldn't see |
| Miller et al.'s metric family — MSE/WMSE/PearsonDelta/R2Delta × all-genes/DEG/DE-weighted, plus NIR | `mse`/`mse_degs_padj`, `wmse_exp1/2/4`, `pearson_ctrl`/`pearson_pert` (+ `_degs_padj` siblings), `r2_ctrl`/`r2_pert` (+ `_degs_padj` and `weighted_..._exp2` siblings), `nir` — all native |
| Vollenweider & Bühlmann 2026's Weighted Pearson Delta | `weighted_pearson_ctrl_exp2` / `weighted_pearson_pert_exp2` — native |
| Ahlmann-Eltze's Pearson Delta / raw $r$ / $\ell_2$ on the top-1000 control-expressed genes | `pearson_ctrl_expr_k` / `pearson_expr_k` / `l2_expr_k` — native |
| Both papers' bootstrap-CI "outperform" question | `models/compare.py`'s bootstrap Bayes Factor — pairwise, on two independently-scored prediction sets (not a scPertEval calibrator: role resolution can't pair two prediction files directly) |
| Miller et al.'s paired t-test / Wilcoxon signed-rank tests | `models/compare.py`'s pairwise two-sided Wilcoxon + Holm correction — a more powerful standard alternative to Miller et al.'s own flat Bonferroni |
| Ahlmann-Eltze's Mean baseline (fold-scoped, not dataset-wide leave-one-out) and linear baseline (PCA + ridge bilinear regression) | `models/baselines/baselines.py`'s `fold_mean_baseline`/`linear_baseline` — implemented in the `models/` scaffold, not the protocol table (a baseline is a prediction source, like GEARS/scGPT, not a metric) |
| The additive combinatorial baseline | not yet available — an `additive` source, built from constituent single-perturbation ground truth |
| Miller et al.'s 5-fold perturbation CV vs. Ahlmann-Eltze's repeated random gene-holdout | neither is a protocol concern in scPertEval today — held-out perturbations come from the dataset, not the protocol table. `models/`'s own k-fold scaffold implements Miller's design (see `prepare_split.py`); scPertEval's `calibrate`/`score` modes themselves remain fold-agnostic by construction (a dataset-level diagnostic, not tied to any one model's CV loop) |
| Miller et al.'s exact filtering/normalization/gene-selection (B.2: min 12 cells/perturbation, target-sum 10k + log1p, gene set = top-8192 HVGs ∪ perturbed genes) | unverified — whether the shared `replogle22k562_processed_complete.h5ad` file matches these thresholds hasn't been checked, and `models/data/prepare_data.py`'s smoke-scale subsample uses a fixed cell cap and the full gene panel rather than Miller's dynamic cap and HVG-restricted gene set |

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

