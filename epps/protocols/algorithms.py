"""Evaluation-protocol algorithms.

Each takes the ground-truth view and a prediction view (and the context, used only
where an algorithm needs an extra input such as DE weights) and returns one scalar.
The prediction is whichever control is being scored — positive or negative. The view
shapes are set by the protocol's ``kind``: ``centroid`` -> 1-D pseudobulk vectors,
``population`` -> (cells x genes) arrays, ``de`` -> (gt DEResult, prediction |score| ranking).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..blocks.kernels import _sq_dists, energy_distance, mmd_unbiased_median, sinkhorn_w2


def pearson(gt, prediction, ctx):
    return float(np.corrcoef(gt, prediction)[0, 1])


def mse(gt, prediction, ctx):
    return float(np.mean((gt - prediction) ** 2))


def weighted_mse(gt, prediction, ctx, exp=2.0):
    w = ctx.wmse_weights(ctx.current_pert) ** exp
    total = w.sum()
    w = w / total if total > 0 else np.full(w.size, 1.0 / w.size)
    return float(np.sum(w * (gt - prediction) ** 2))


def mmd(gt, prediction, ctx):
    return mmd_unbiased_median(gt, prediction)


def energy(gt, prediction, ctx):
    return energy_distance(gt, prediction)


def sinkhorn(gt, prediction, ctx):
    return sinkhorn_w2(gt, prediction)


def rank_retrieval(prediction, gt, transpose=False):
    """Cross-perturbation retrieval rank (0 = best, lower is better).

    Unlike the per-perturbation algorithms, this consumes the full prediction and
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
