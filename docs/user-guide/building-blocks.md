# Building blocks

Spaces, DE methods, control sources, and calibrators are registered units — add one when
the palette is missing what a new protocol needs. Each is a small function (or object) plus
a one-line registration. To author the protocol or metric that draws on these blocks, see
[Create a protocol](protocols.md#create-a-protocol).

## Add a feature space

A space decides which features a protocol scores on. Every space is **one line** near the bottom
of
[`src/scperteval/blocks/spaces.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces.py):

```python
FULL = Subset("full", _all_genes,        None, "all genes, no transform")
TOP  = Subset("top",  _strongest_de,       50, "top {v} genes by ground-truth effect size", per_pert=True)
HEG  = Subset("heg",  _highest_expressed, 1000, "top {v} genes by control-condition expression")
...

SUBSETS = [FULL, TOP, DEGS, HEG, HVG, PERTURBED_GENES]
DEFAULTS = [row.register() for row in SUBSETS] + [row.register() for row in TRANSFORMS]
```

Adding one is two edits, both in that file. Write the selection rule, then add the space and put
it in the list:

```python
def _mitochondrial(ctx, pert, k):                    # 1. the rule
    mito = np.flatnonzero(np.char.startswith(ctx.ds.var_names, "MT-"))
    return mito[np.argsort(-ctx.control_mean()[mito])][:k]

MITO = Subset("mito", _mitochondrial, 20,            # 2. the space
              "top {v} mitochondrial genes by control-condition expression")

SUBSETS = [FULL, TOP, DEGS, HEG, HVG, PERTURBED_GENES, MITO]
```

That is the whole thing. `mito_20` is registered at import and appears in `scperteval list
spaces`; `MITO.register(50)` registers `mito_50` on demand; and it composes with any other subset
via `combine_space`.

**The rule** returns a column selection into the *full* gene axis — an integer array, or a
slice — never positions into some earlier subset, so selections from different spaces can be
combined. It receives `(ctx, pert, value)`, where `value` is the space's parameter. Ignore `pert`
unless the choice actually varies per perturbation, and set `per_pert=True` when it does, so
scPertEval knows it can't compute the selection once and share it.

**The arguments** are `Subset(name, rule, default, description)`. `default` is the parameter
value of the instance registered at import, and instances are named `<name>_<value>` — so
`default=20` gives you `mito_20`. Use `None` when the space takes no parameter at all (as `full`
and `perturbed_genes` do); the instance is then just `<name>`. `{v}` in the description is
filled in with the parameter.

**Registration** is what makes a name usable: a protocol names its space as a string, so
`SPACES["mito_20"]` has to resolve. `DEFAULTS`, directly under the list, calls `register()` on
every space when the module is imported. Any other value registers the first time
`MITO.register(50)` is called, which is what a protocol template does when you pass
`-p <protocol>=<value>` — `table.py` wires that up as `Param("k", int, 20, space=MITO.register)`.

**Expensive statistics belong on the `Context`.** The rule runs once per perturbation per
protocol, so anything computed over the whole dataset should be cached rather than recomputed.
`_highest_expressed` calls `ctx.control_mean()` for this reason. Adding a new cached statistic
means three small edits following `control_hvg_dispersion` as the template: a method on
`Dataset`, a slot on `CacheStore`, and a double-checked-lock accessor on `Context`.

### Spaces that aren't gene subsets

`pca_<k>` replaces the gene axis with components instead of narrowing it, so it has no gene
selection and nothing to compose. Those are `Transform`s and supply the finished array:

```python
PCA = Transform("pca", _principal_components, 50, "top {v} principal components", _fit_pca)
```

The optional last argument is a `prepare(ctx, names)` hook, run once before a run with every
requested variant name, for building shared structure up front (PCA fits each requested size
there). It is purely an optimisation: `apply` must stay correct if it never runs.

### Combining subsets

`combine_space` builds a new space from existing subsets by a set operation. The built-in
`miller_panel` is defined this way, at the bottom of `spaces.py`:

```python
combine_space("miller_panel", HVG.register(8192), PERTURBED_GENES.register())
combine_space("not_hvg", FULL.register(), HVG.register(2000), op="diff")
```

### Defining a space outside the repo

`register_subset_space(name, select, ...)` registers one directly, for a space that shouldn't
live in this file. `select` takes `(ctx, pert)` — already bound to its parameter — and the
result composes like any other subset.

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
