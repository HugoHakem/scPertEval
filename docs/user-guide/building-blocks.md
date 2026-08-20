# Building blocks

Spaces, DE methods, control sources, and calibrators are registered units — add one when
the palette is missing what a new protocol needs. Each is a small function (or object) plus
a one-line registration. To author the protocol or metric that draws on these blocks, see
[Create a protocol](protocols.md#create-a-protocol).

## Add a feature space

A space decides which features a protocol scores on. Add one by writing a rule and decorating
it, the same way DE methods and control sources are registered. Everything lives in
[`src/scperteval/blocks/spaces/catalog.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces/catalog.py):

```python
@SPACES.subset("mito", default=20, description="top {v} mitochondrial genes by control expression")
def mito(ctx, pert, k):
    """The k highest-expressed mitochondrial genes."""
    mt = np.flatnonzero([g.startswith("MT-") for g in ctx.ds.var_names])
    return mt[np.argsort(-ctx.control_mean()[mt])][:k]
```

That's it — one edit, one place. `mito_<k>` now appears in `scperteval list spaces`, and a
protocol can use it at any `k`.

**The rule** returns a column selection into the *full* gene axis — an integer array, or a
slice — never positions into some earlier subset, so selections from different spaces can be
folded together. It receives `(ctx, pert, value)`. The rule runs once per perturbation per
protocol, so anything computed over the whole dataset belongs behind a `Context` cache (as
`ctx.control_mean()` is), not recomputed here.

**The decorator** carries the metadata. `default` is the parameter value used when a caller
doesn't supply one; `{v}` in the description is filled in with it. Pass `per_pert=True` when the
selection depends on which perturbation is being scored — including when the rule folds in a
per-perturbation rule such as `top` or `degs` — so scPertEval knows it can't compute the
selection once and share it.

**Whether a space takes a parameter is read from the rule's signature.** A trailing argument
with a default means it takes none:

```python
@SPACES.subset("perturbed_genes", description="genes targeted by a perturbation")
def perturbed_genes(ctx, pert, value=None):
    return ctx.perturbed_gene_indices()
```

`scperteval list spaces` shows `mito_<k>` and `perturbed_genes` accordingly. Declaring a
parameter without a default (or a default without a parameter) is an error at import.

**Adding a cached statistic.** If your rule needs a new per-gene statistic over the whole
dataset, follow `control_hvg_dispersion`: a method on `Dataset`, a slot on `CacheStore`, and a
double-checked-lock accessor on `Context`.

### Composing subsets

Because a rule is an ordinary function, a composed space is just a rule that calls other rules.
`combine_subsets` folds their selections with a set operation from `OPS`, and nests to any depth:

```python
@SPACES.subset("miller_panel", description="HVG union perturbed genes — the panel of Miller et al. 2025")
def miller_panel(ctx, pert, value=None):
    return combine_subsets(ctx, OPS.union, hvg(ctx, pert, 8192), perturbed_genes(ctx, pert))
```

`OPS` carries the three set operations — `OPS.union`, `OPS.intersection`, and
`OPS.difference`, which subtracts left to right. So
`combine_subsets(ctx, OPS.difference, full(ctx, pert), hvg(ctx, pert, 2000))` is the
complement of the HVG panel.

### Spaces that aren't gene subsets

`pca_<k>` replaces the gene axis with components instead of narrowing it, so it has no gene
selection and can't be composed. Use `@SPACES.transform`, whose rule takes the cells and returns
the finished array:

```python
@SPACES.transform("pca", default=50, prepare=_fit_pca, description="top {v} principal components")
def pca(X, ctx, pert, k):
    return ctx.pca(k).transform(to_dense(X))[:, :k]
```

The optional `prepare(ctx, names)` hook runs once before a run with every requested variant name,
for building shared structure up front. It is purely an optimisation: the rule must stay correct
if it never runs.

### Definitions vs. instances

The catalog above holds **definitions**. A protocol names its space as a concrete string, so a
definition is turned into a registered **instance** — `heg_1000` — by
`SPACES.instance("heg", 1000)`. That happens when something asks for it: a `Param` in
`protocols/table.py` calls it as the CLI value is resolved, so `-p mse_top_k=30` registers
`top_30` on the spot. Nothing is instantiated merely to appear in a listing —
`scperteval list spaces` shows the catalog.

## Add a DE method

A DE method maps `(target_cells, reference_cells) -> PerturbationDEResult(statistic, pvalue, pvalue_adj)`.
Register it with `@DE_METHODS.register` in [`src/scperteval/blocks/de.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/de.py) (the
`bh` helper there BH-adjusts p-values):

```python
@DE_METHODS.register("my_test", description="…")
def de_my_test(target, reference):
    statistic, pvalue = ...          # per-gene statistic and raw p-value
    return PerturbationDEResult(statistic=statistic, pvalue=pvalue, pvalue_adj=bh(pvalue))
```

Then `--de-method my_test` routes every DE-dependent unit through it.

A method whose statistic is expressible from per-gene moments (mean, variance, cell count) may
additionally declare `from_moments=<callable>` in its `register(...)` metadata to reuse
scPertEval's cached reference moments, as the built-in `t-test` does — the callable takes
`(mean_t, var_t, n_t, mean_r, var_r, n_r)` and returns a `PerturbationDEResult`. It's a pure
performance opt-in: correctness is identical without it, and the `(target, reference)` function
above is still required.

## Add a control source

A source maps `(ctx, pert) -> cells or a 1-D centroid`, declaring which with `provides`.
Register it with `@SOURCES.register` in [`src/scperteval/sources.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/sources.py):

```python
@SOURCES.register("my_baseline", provides="centroid", description="…")
def src_my_baseline(ctx, pert):
    return ...                       # a 1-D centroid (or cells, if provides="cells")
```

Use it as a control at the CLI via `--positive`/`--negative`, or make it a row's default with
`default_positive=`/`default_negative=` (only where the row deviates from the representation's
generic default; controls are otherwise resolved at runtime — see
[Protocols → Control sources](protocols.md)).

## Add a calibrator

A calibrator declares the control roles it needs, a per-perturbation combine, and a
cross-perturbation aggregate. Add a `Calibrator` to the `CALIBRATORS` dict in
[`src/scperteval/calibrators.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/calibrators.py):

```python
CALIBRATORS["my_score"] = Calibrator(
    "my_score", ("positive", "negative"),
    per_pert=lambda raws, p: ...,          # raws["positive"], raws["negative"] -> one number
    aggregate=lambda v: {"my_score": float(np.nanmean(v))},
    description="…",
)
```

Then `--calibrator my_score` reports it.
