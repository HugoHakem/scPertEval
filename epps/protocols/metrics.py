"""Evaluation-protocol metrics — the exact implementation of every metric.

Each takes the ground-truth view and a prediction view (and the context, used only
where a metric needs an extra input such as DE weights) and returns one scalar.
The prediction is whichever control is being scored — positive or negative. The
shapes are set by the protocol's ``representation``: ``centroid`` -> 1-D pseudobulk vectors,
``population`` -> (cells x genes) arrays, ``de`` -> (gt DEResult, prediction |score| ranking).

Every metric is implemented in full here; only external numerical libraries (numpy,
scikit-learn, geomloss) are relied upon. So a metric is completely defined by its function
below plus its row in ``table.py`` — nothing is hidden behind another layer.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _sq_dists(X, Y):
    """Pairwise squared euclidean distances via ||x||^2 + ||y||^2 - 2 x.y.

    Routed through a BLAS matrix product, which releases the GIL so the
    per-perturbation thread pool actually parallelises.
    """
    xx = np.einsum("ij,ij->i", X, X)
    yy = np.einsum("ij,ij->i", Y, Y)
    sq = xx[:, None] + yy[None, :] - 2.0 * (X @ Y.T)
    return np.maximum(sq, 0.0)


def _within_unbiased(sq, n):
    """Unbiased (U-statistic) mean within-population euclidean distance."""
    if n <= 1:
        return 0.0
    return float(np.sqrt(sq).sum() / (n * (n - 1)))


def pearson(gt, prediction, ctx):
    return float(np.corrcoef(gt, prediction)[0, 1])


def mse(gt, prediction, ctx):
    return float(np.mean((gt - prediction) ** 2))


def weighted_mse(gt, prediction, ctx, exp=2.0):
    w = ctx.wmse_weights(ctx.current_pert) ** exp
    total = w.sum()
    w = w / total if total > 0 else np.full(w.size, 1.0 / w.size)
    return float(np.sum(w * (gt - prediction) ** 2))


def energy_distance(gt, prediction, ctx):
    """Szekely-Rizzo energy distance with bias-corrected within terms."""
    if len(gt) == 0 or len(prediction) == 0:
        return float("nan")
    X = gt.astype(np.float64)
    Y = prediction.astype(np.float64)
    cross = np.sqrt(_sq_dists(X, Y)).mean()
    xx = _within_unbiased(_sq_dists(X, X), len(X))
    yy = _within_unbiased(_sq_dists(Y, Y), len(Y))
    return float(2.0 * cross - xx - yy)


def unbiased_mmd_median(gt, prediction, ctx):
    """Unbiased RBF-MMD^2 with a single median-heuristic bandwidth (Gretton 2012)."""
    if len(gt) < 2 or len(prediction) < 2:
        return float("nan")
    X = gt.astype(np.float64)
    Y = prediction.astype(np.float64)
    nx, ny = len(X), len(Y)
    pooled = np.vstack([X, Y])
    euc = np.sqrt(_sq_dists(pooled, pooled))
    n = euc.shape[0]
    sigma = float(np.median(euc[~np.eye(n, dtype=bool)]))
    if sigma <= 0:
        return 0.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-gamma * _sq_dists(X, X))
    k_yy = np.exp(-gamma * _sq_dists(Y, Y))
    k_xy = np.exp(-gamma * _sq_dists(X, Y))
    xx = (k_xx.sum() - np.trace(k_xx)) / (nx * (nx - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (ny * (ny - 1))
    return float(xx + yy - 2.0 * k_xy.mean())


_geomloss_cache: dict = {}


def sinkhorn_w2(gt, prediction, ctx, blur=0.05):
    """Debiased Sinkhorn 2-Wasserstein distance (geomloss, p=2)."""
    if len(gt) == 0 or len(prediction) == 0:
        return float("nan")
    import torch
    from geomloss import SamplesLoss

    loss = _geomloss_cache.get(blur)
    if loss is None:
        torch.set_num_threads(1)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True, backend="tensorized")
        _geomloss_cache[blur] = loss
    Xt = torch.as_tensor(np.ascontiguousarray(gt), dtype=torch.float32)
    Yt = torch.as_tensor(np.ascontiguousarray(prediction), dtype=torch.float32)
    a = torch.full((len(gt),), 1.0 / len(gt), dtype=torch.float32)
    b = torch.full((len(prediction),), 1.0 / len(prediction), dtype=torch.float32)
    with torch.no_grad():
        val = float(loss(a, Xt, b, Yt))
    return float(np.sqrt(max(2.0 * val, 0.0)))


def rank_retrieval(prediction, gt, transpose=False):
    """Cross-perturbation retrieval rank (0 = best, lower is better).

    Unlike the per-perturbation metrics, this consumes the full prediction and
    GT centroid matrices (rows = perturbations) and returns one score per row. In
    the prediction-vs-GT squared-distance matrix, ``rank`` ranks each GT's own
    prediction against all predictions (column-wise); ``transpose_rank`` ranks each
    prediction's own GT against all GTs (row-wise). Normalised by n-1, with the
    drf tie-breaking noise (seed 42).
    """
    sq = _sq_dists(prediction, gt)
    if transpose:
        sq = sq.T
    n = sq.shape[0]
    noise = np.random.default_rng(42).uniform(0, 1e-12, size=sq.shape)
    ranks = np.argsort(np.argsort(sq + noise, axis=0), axis=0)
    return np.diag(ranks).astype(np.float64) / max(n - 1, 1)


def de_auprc(gt, prediction, ctx):
    labels = gt.pvalue_adj < 0.05
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(average_precision_score(labels, prediction))


def de_auroc(gt, prediction, ctx):
    labels = gt.pvalue_adj < 0.05
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    return float(roc_auc_score(labels, prediction))


def de_overlap(gt, prediction, ctx, k=50):
    truth = np.abs(gt.score)
    if k >= truth.size:
        return float("nan")
    top_truth = np.argpartition(-truth, k - 1)[:k]
    top_prediction = np.argpartition(-prediction, k - 1)[:k]
    return float(np.intersect1d(top_truth, top_prediction).size) / k
