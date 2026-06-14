"""Distance kernels between two single-cell populations (rows = cells).

Pairwise distances go through a BLAS matrix product so the heavy work releases
the GIL and the per-perturbation thread pool actually parallelises.
"""
from __future__ import annotations

import numpy as np

_geomloss_cache: dict[float, object] = {}


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distances via ||x||^2 + ||y||^2 - 2 x.y."""
    xx = np.einsum("ij,ij->i", X, X)
    yy = np.einsum("ij,ij->i", Y, Y)
    sq = xx[:, None] + yy[None, :] - 2.0 * (X @ Y.T)
    return np.maximum(sq, 0.0)


def _within_unbiased(sq: np.ndarray, n: int) -> float:
    """Unbiased (U-statistic) mean within-population euclidean distance."""
    if n <= 1:
        return 0.0
    return float(np.sqrt(sq).sum() / (n * (n - 1)))


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Szekely-Rizzo energy distance with bias-corrected within terms."""
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    cross = np.sqrt(_sq_dists(X, Y)).mean()
    xx = _within_unbiased(_sq_dists(X, X), len(X))
    yy = _within_unbiased(_sq_dists(Y, Y), len(Y))
    return float(2.0 * cross - xx - yy)


def mmd_unbiased_median(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased RBF-MMD^2 with a single median-heuristic bandwidth (Gretton 2012)."""
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
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


def sinkhorn_w2(X: np.ndarray, Y: np.ndarray, blur: float = 0.05) -> float:
    """Debiased Sinkhorn 2-Wasserstein distance (geomloss, p=2)."""
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    import torch
    from geomloss import SamplesLoss

    loss = _geomloss_cache.get(blur)
    if loss is None:
        torch.set_num_threads(1)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur, debias=True, backend="tensorized")
        _geomloss_cache[blur] = loss
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
    Yt = torch.as_tensor(np.ascontiguousarray(Y), dtype=torch.float32)
    a = torch.full((len(X),), 1.0 / len(X), dtype=torch.float32)
    b = torch.full((len(Y),), 1.0 / len(Y), dtype=torch.float32)
    with torch.no_grad():
        val = float(loss(a, Xt, b, Yt))
    return float(np.sqrt(max(2.0 * val, 0.0)))
