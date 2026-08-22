"""Dataset-level computations the space rules in ``catalog.py`` depend on.

Each is a function of the dataset alone, so :func:`~scperteval.blocks.spaces.cache.cached`
evaluates it once per prepared dataset. Add one here when a new space needs a per-gene statistic
or a fitted structure, and call it from the rule as ``my_helper(ctx)``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
from threadpoolctl import threadpool_limits

from ...dataset import to_dense
from .cache import cached

#: Cells sampled for the PCA basis. The ``subsample`` cap is for the O(n^2) distance
#: populations, not the basis, which needs many cells to be stable.
PCA_FIT_CAP = 50000


@cached
def control_mean(data):
    """Pseudobulk centroid of the control cells."""
    return data.ds.control_mean()


@cached
def control_dispersion(data):
    """Per-gene normalized dispersion of the control cells (scanpy's ``"seurat"`` HVG statistic).

    ``X`` is passed through as-is: scanpy accepts sparse input and returns the identical
    statistic, so the control block is never densified (it can be the largest single population
    in the dataset).
    """
    import scanpy as sc

    view = ad.AnnData(X=data.ds.adata.X[data.ds.control_idx])
    df = sc.pp.highly_variable_genes(view, flavor="seurat", inplace=False)
    assert df is not None  # inplace=False always returns a DataFrame
    return df["dispersions_norm"].to_numpy()


@cached
def targeted_genes(data):
    """``var_names`` indices of every gene targeted by a retained perturbation.

    Retained means it passed the ``min_cells`` filter, so this is the set of genes targeted by
    the perturbations actually being evaluated. Labels are gene symbols matching ``var_names``;
    combinations are ``+``-delimited (e.g. ``"GENE1+GENE2"``, see docs/user-guide/datasets.md). A
    token that doesn't match any ``var_names`` entry (e.g. a non-gene treatment label) is skipped.

    Raises
    ------
    ValueError
        If no label matches a gene, which would leave an empty panel — the metrics would silently
        return ``nan`` rather than fail.
    """
    ds = data.ds
    pos = {g: i for i, g in enumerate(ds.var_names)}
    idx = {pos[gene] for pert in ds.perturbations for gene in pert.split("+") if gene in pos}
    if not idx:
        raise ValueError(
            "no perturbation label matches a gene in var_names, so the perturbed-gene panel "
            "would be empty. Labels are split on '+' and matched exactly against var_names "
            f"(labels e.g. {[str(x) for x in ds.perturbations[:3]]}, "
            f"genes e.g. {[str(g) for g in ds.var_names[:3]]}). "
            "This space needs genetic perturbations labelled with gene symbols."
        )
    return np.array(sorted(idx), dtype=int)


@cached
def fitted_pca(data, n_components):
    """PCA fit on (nearly) all cells, cached per fit size."""
    from sklearn.decomposition import PCA

    n = data.ds.adata.n_obs
    idx = np.arange(n)
    if n > PCA_FIT_CAP:
        idx = np.sort(np.random.default_rng(data.seed).choice(n, PCA_FIT_CAP, replace=False))
    X = to_dense(data.ds.adata.X[idx]).astype(np.float64)
    # sklearn's bundled BLAS/OpenMP only loads with the import above, after run_all()'s own
    # threadpool_limits already scanned -- re-scan here so it's actually caught and capped
    with threadpool_limits(limits=data.threads):
        return PCA(n_components=min(n_components, *X.shape), random_state=data.seed).fit(X)


def pca_for(ctx, k):
    """The fit backing ``pca_<k>``.

    Sizes below 50 share the 50-component fit, and a size is fit once and never replaced:
    sklearn's PCA is not basis-stable across ``n_components`` (the solver switches, and
    randomized SVD is not nested), so slicing a smaller ``pca_k`` out of a larger fit would
    silently change its result and desync anything (e.g. the cached reference projection) already
    projected through the old basis.
    """
    return fitted_pca(ctx, max(k, 50))
