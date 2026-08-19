# Building blocks

Spaces, DE methods, control sources, and calibrators are registered units — add one when
the palette is missing what a new protocol needs. Each is a small function (or object) plus
a one-line registration. To author the protocol or metric that draws on these blocks, see
[Create a protocol](protocols.md#create-a-protocol).

## Add a feature space

A space picks the features a protocol scores on. Almost every space is a **gene subset** — it
keeps some gene columns and drops the rest, differing only in *which* genes it picks. Everything
goes in
[`src/scperteval/blocks/spaces.py`](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/src/scperteval/blocks/spaces.py),
in three steps. Copy the nearest existing family — `heg_<k>` is the simplest complete example —
and change the name, the rule, and the description:

```python
# --- mito_<k> — the k mitochondrial genes with the highest control expression ---

def _mito(ctx, pert, *, k):                          # 1. the selection rule
    mito = np.flatnonzero(np.char.startswith(ctx.ds.var_names, "MT-"))
    return mito[np.argsort(-ctx.control_mean()[mito])][:k]

def mito_space(k: int) -> str:                       # 2. the factory
    return register_subset_space(
        f"mito_{k}", partial(_mito, k=k), global_space=True,
        description=f"top {k} mitochondrial genes by control-condition expression",
    )

mito_space(20)                                       # 3. a default instance, at the bottom
```

1. **The selection rule** returns integer positions into the *full* gene axis — never into some
   earlier subset — so rules from different spaces stay comparable. Ignore `pert` if the choice
   is dataset-wide. The rule runs once per perturbation per protocol, so anything expensive and
   shared belongs behind a `Context` cache (as `ctx.control_mean()` is), not recomputed here.
2. **The factory** binds the parameters with `partial` and calls `register_subset_space`, which
   supplies the slice-and-densify transform and returns the space name. Pass
   `global_space=True` when the rule ignores `pert`, so the result can be computed once and
   shared across perturbations.
3. **The default instance** is one call at the bottom of the file, which is what makes the space
   show up in `scperteval list spaces`.

That is also all it takes to make the space **composable**. `register_subset_space` records the
rule under the `indices` metadata key, and `combine_space` builds new spaces by unioning,
intersecting, or subtracting the gene sets of existing ones — how the HVG ∪ perturbed-genes
panel is expressed:

```python
combine_space("miller_panel", hvg_space(8192), perturbed_genes_space())
```

**Shortcut for DE-derived subsets.** If the genes are chosen from the ground-truth differential
expression (as `top_k` and `degs` are), `register_de_space` writes the rule for you — you supply
only which field to read and how to cut it:

```python
register_de_space("my_degs", field="pvalue_adj", threshold=lambda v: v < 0.01, description="…")
```

**Spaces that aren't gene subsets.** A space that *transforms* genes into something else rather
than selecting among them — `pca_<k>` is the only built-in — registers its transform directly:

```python
@SPACES.register("my_embedding", global_space=True, description="…")
def space_my_embedding(X, ctx, pert):
    return ...                       # a dense cells × features array
```

Such a space works everywhere else but can't be composed, which is why `combine_space` rejects
`full` and `pca_<k>`.

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
