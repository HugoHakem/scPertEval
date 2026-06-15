"""Differential-expression backends sharing one DEResult interface.

A DE method maps (target cells, reference cells) -> DEResult. The t-test is
expressed through reusable ``moments`` / ``ttest_from_moments`` helpers so the
context can cache a shared reference's moments and combine them cheaply.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import stats

from ..registry import Registry
from ..types import DEResult

DE_METHODS = Registry("de-method")


def moments(X):
    """Per-gene (mean, sample variance, n) for a cell matrix, sparse- or dense-aware."""
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


def bh(pvalue: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalue, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    adj = p[order] * idx.size / np.arange(1, idx.size + 1)
    out[order] = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0.0, 1.0)
    return out


def ttest_from_moments(mt, vt, nt, mr, vr, nr) -> DEResult:
    """Welch's t-test (scanpy convention); score = t-statistic."""
    se2 = vt / nt + vr / nr
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mt - mr) / np.sqrt(se2)
        df = se2 ** 2 / ((vt / nt) ** 2 / max(nt - 1, 1) + (vr / nr) ** 2 / max(nr - 1, 1))
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    pval = np.nan_to_num(2.0 * stats.t.sf(np.abs(t), df), nan=1.0)
    return DEResult(score=t, pvalue=pval, pvalue_adj=bh(pval))


@DE_METHODS.register("t-test", description="Welch's t-test (default) — moment-based and fast")
def de_ttest(target, reference) -> DEResult:
    return ttest_from_moments(*moments(target), *moments(reference))


@DE_METHODS.register(
    "t-test_overestim_var",
    description="scanpy's conservative t-test variant; reference variance scaled by the "
                "target's cell count (selectable backend; not used by any current protocol)",
)
def de_ttest_overestim(target, reference) -> DEResult:
    """scanpy ``rank_genes_groups(method='t-test_overestim_var')``.

    Identical to Welch's t-test except the reference group's cell count is replaced by the
    target's, which inflates the reference standard-error term ("overestimating" its variance
    for small target groups) and yields a more conservative statistic. Selectable as a DE
    backend (``--de-method``/``--methods``) so new evaluation protocols can use it; no current
    protocol does.
    """
    mt, vt, nt = moments(target)
    mr, vr, _nr = moments(reference)
    return ttest_from_moments(mt, vt, nt, mr, vr, nt)


@DE_METHODS.register("MWU", description="Mann-Whitney U / Cliff's delta effect size (via illico)")
def de_mwu(target, reference) -> DEResult:
    """Mann-Whitney U via illico (one-vs-reference); score = Cliff's delta.

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
    df = asymptotic_wilcoxon(adata, is_log1p=True, group_keys="_g", reference="reference",
                             n_threads=1, alternative="two-sided", use_continuity=True,
                             tie_correct=True, return_as_scanpy=False)
    sub = df.xs("target", level=0).reindex(genes)
    u = sub["statistic"].to_numpy(dtype=np.float64)
    pval = np.nan_to_num(sub["p_value"].to_numpy(dtype=np.float64), nan=1.0)
    cliff = 2.0 * u / (nt * nr) - 1.0
    return DEResult(score=cliff, pvalue=pval, pvalue_adj=bh(pval), extra={"u": u})
