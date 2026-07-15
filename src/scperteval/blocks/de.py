"""Differential-expression backends sharing one :class:`~scperteval.types.PerturbationDEResult` interface.

Each backend maps ``(target_cells, reference_cells) → PerturbationDEResult``. Three backends are
registered in :data:`DE_METHODS`:

- ``"t-test"`` — Welch's t-test, default and fastest (:func:`de_ttest`).
- ``"t-test_overestim_var"`` — :func:`scanpy.tl.rank_genes_groups`'s conservative variant
  (:func:`de_ttest_overestim`).
- ``"MWU"`` — Mann-Whitney U via illico; statistic is Cliff's delta (:func:`de_mwu`).

The t-test family is factored through :func:`ttest_from_moments` so the
runner can cache a shared reference's moments and reuse them cheaply across perturbations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import stats

from ..registry import Registry
from ..types import PerturbationDEResult

DE_METHODS = Registry("de-method")
"""Registry of DE backends; keys are the ``--de-method`` names.

Use :meth:`~scperteval.registry.Registry.register` to add a custom backend::

    from scperteval.blocks.de import DE_METHODS, bh
    from scperteval.types import PerturbationDEResult

    @DE_METHODS.register("my_test", description="My custom DE test")
    def de_my_test(target, reference):
        statistic = ...   # per-gene statistic, shape (G,)
        pvalue = ...      # per-gene raw p-values, shape (G,)
        return PerturbationDEResult(statistic=statistic, pvalue=pvalue, pvalue_adj=bh(pvalue))

Then ``--de-method my_test`` routes every DE-dependent unit through it.

**Opting into the moment cache.** A method whose statistic is a function of per-gene
moments (mean, sample variance, cell count) can declare a ``from_moments`` capability to
reuse :class:`~scperteval.context.Context`'s shared moment cache — the same optimisation
that makes ``t-test`` fast (a source's moments are computed once and the all-perturbed
reference's are served leave-one-out). Register a second callable with the signature
``(mean_t, var_t, n_t, mean_r, var_r, n_r) -> PerturbationDEResult``::

    @DE_METHODS.register("my_moment_test", description="...", from_moments=my_from_moments)
    def de_my_moment_test(target, reference):
        return my_from_moments(*_moments(target), *_moments(reference))

When a method provides ``from_moments`` the context dispatches through it (reading cached
moments); otherwise it calls the registered cells-based function. The registered function
stays the single source of truth — no method is privileged by name.
"""


def _moments(X):
    r"""Per-gene mean, sample variance, and cell count for a cell matrix.

    Sparse- and dense-aware; uses :math:`\text{Var}(X) = E[X^2] - E[X]^2`
    with Bessel's correction.

    Parameters
    ----------
    X : array-like, shape ``(n, G)``
        Cell matrix (sparse or dense).

    Returns
    -------
    mean : numpy.ndarray, shape ``(G,)``
        Per-gene sample mean.
    variance : numpy.ndarray, shape ``(G,)``
        Per-gene sample variance (floored at 0).
    n : int
        Number of cells.
    """
    n = X.shape[0]
    if sp.issparse(X):
        m = np.asarray(X.mean(0)).ravel()
        msq = np.asarray(X.multiply(X).mean(0)).ravel()
    else:
        X = np.asarray(X)
        m = X.mean(0)
        msq = (X * X).mean(0)
    v = (msq - m * m) * (n / max(n - 1, 1))
    return m, np.maximum(v, 0.0), n


def mejia_weights(score: np.ndarray) -> np.ndarray:
    """Min-max normalised ``|score|`` across genes :cite:p:`Mejia_2025`.

    The DEG weighting scheme behind the weighted metrics (e.g. ``wmse``): each gene's weight
    is its share of the strongest observed effect size, in ``[0, 1]``.

    Parameters
    ----------
    score : numpy.ndarray
        Per-gene DE statistic (e.g. a t-statistic), shape ``(G,)``.

    Returns
    -------
    numpy.ndarray
        Weights in ``[0, 1]``, same shape as ``score``; ``0.0`` where ``score`` is
        non-finite or every finite value is equal.
    """
    s = np.abs(score)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    lo, hi = s[finite].min(), s[finite].max()
    w = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
    return np.nan_to_num(w, nan=0.0)


def bh(pvalue: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values for FDR control :cite:p:`Vollenweider_2026`.

    Applied gene-wise inside each DE method to control the false discovery rate
    across genes. The same procedure is used across perturbations to summarise
    the overall sensitivity of a metric — see :cite:t:`Vollenweider_2026`.

    Parameters
    ----------
    pvalue : numpy.ndarray
        Array of raw p-values; non-finite values are carried through as ``nan``.

    Returns
    -------
    numpy.ndarray
        BH-adjusted p-values clipped to [0, 1]; same shape as ``pvalue``.
    """
    p = np.asarray(pvalue, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    adj = p[order] * idx.size / np.arange(1, idx.size + 1)
    out[order] = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0.0, 1.0)
    return out


def ttest_from_moments(mt, vt, nt, mr, vr, nr) -> PerturbationDEResult:
    """Welch's t-test from pre-computed per-gene moments (:func:`scanpy.tl.rank_genes_groups`'s convention).

    Accepts moments directly so the context can cache the reference's moments
    once and combine them cheaply for every perturbation. The ``statistic`` field
    of the returned :class:`~scperteval.types.PerturbationDEResult` is the t-statistic.

    Parameters
    ----------
    mt : numpy.ndarray, shape ``(G,)``
        Target per-gene means.
    vt : numpy.ndarray, shape ``(G,)``
        Target per-gene sample variances.
    nt : int
        Number of target cells.
    mr : numpy.ndarray, shape ``(G,)``
        Reference per-gene means.
    vr : numpy.ndarray, shape ``(G,)``
        Reference per-gene sample variances.
    nr : int
        Number of reference cells.

    Returns
    -------
    ~scperteval.types.PerturbationDEResult
        ``statistic`` is the Welch t-statistic; ``pvalue_adj`` is BH-adjusted.
    """
    se2 = vt / nt + vr / nr
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mt - mr) / np.sqrt(se2)
        df = se2**2 / ((vt / nt) ** 2 / max(nt - 1, 1) + (vr / nr) ** 2 / max(nr - 1, 1))
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    pval = np.nan_to_num(2.0 * stats.t.sf(np.abs(t), df), nan=1.0)
    return PerturbationDEResult(statistic=t, pvalue=pval, pvalue_adj=bh(pval))


@DE_METHODS.register(
    "t-test",
    description="Welch's t-test (default) — moment-based and fast",
    from_moments=ttest_from_moments,
)
def de_ttest(target, reference) -> PerturbationDEResult:
    """Welch's t-test between target and reference cell matrices.

    Registered with a ``from_moments`` capability (:func:`ttest_from_moments`), so the
    per-run :class:`~scperteval.context.Context` computes it from cached per-gene moments
    instead of re-densifying cells for every comparison — see :data:`DE_METHODS`.
    """
    return ttest_from_moments(*_moments(target), *_moments(reference))


@DE_METHODS.register(
    "t-test_overestim_var",
    description="scanpy's conservative t-test variant; reference variance scaled by the "
    "target's cell count (selectable backend; not used by any current protocol)",
)
def de_ttest_overestim(target, reference) -> PerturbationDEResult:
    """:func:`scanpy.tl.rank_genes_groups` with ``method='t-test_overestim_var'``.

    Identical to Welch's t-test except the reference group's cell count is replaced by the
    target's, which inflates the reference standard-error term ("overestimating" its variance
    for small target groups) and yields a more conservative statistic. Selectable as a DE
    backend (``--de-method``/``--methods``); no current protocol uses it.
    """
    mt, vt, nt = _moments(target)
    mr, vr, _nr = _moments(reference)
    return ttest_from_moments(mt, vt, nt, mr, vr, nt)


@DE_METHODS.register("MWU", description="Mann-Whitney U / Cliff's delta effect size (via illico)")
def de_mwu(target, reference) -> PerturbationDEResult:
    """Mann-Whitney U via illico (one-vs-reference); statistic = Cliff's delta.

    illico tests labelled groups in an AnnData against a reference group and
    returns a (group, gene) frame with raw ``p_value`` + U ``statistic``; we read
    the target group, map U to Cliff's delta in [-1, 1], and BH-adjust here.
    """
    import anndata as ad
    from illico import asymptotic_wilcoxon

    Xt = target.toarray() if sp.issparse(target) else np.asarray(target)
    Xr = reference.toarray() if sp.issparse(reference) else np.asarray(reference)
    nt, nr, ng = Xt.shape[0], Xr.shape[0], Xt.shape[1]
    genes = [str(i) for i in range(ng)]
    adata = ad.AnnData(np.vstack([Xt, Xr]).astype(np.float64))
    adata.var_names = genes
    adata.obs["_g"] = ["target"] * nt + ["reference"] * nr
    df = asymptotic_wilcoxon(
        adata,
        is_log1p=True,
        group_keys="_g",
        reference="reference",
        n_threads=1,
        alternative="two-sided",
        use_continuity=True,
        tie_correct=True,
        return_as_scanpy=False,
    )
    sub = df.xs("target", level=0).reindex(genes)  # pyright: ignore[reportAttributeAccessIssue]
    u = sub["statistic"].to_numpy(dtype=np.float64)
    pval = np.nan_to_num(sub["p_value"].to_numpy(dtype=np.float64), nan=1.0)
    cliff = 2.0 * u / (nt * nr) - 1.0
    return PerturbationDEResult(statistic=cliff, pvalue=pval, pvalue_adj=bh(pval), extra={"u": u})
