"""Runs one protocol over every perturbation and applies the chosen calibrator."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np

from .sources import SOURCES
from .types import Calibrator, Protocol


def n_workers(cfg) -> int:
    return cfg.workers if cfg.workers > 0 else max(1, min(16, (os.cpu_count() or 2) - 2))


def resolve_controls(p: Protocol, cfg) -> dict:
    return {
        "positive": cfg.positive if cfg.positive != "auto" else p.positive,
        "negative": cfg.negative if cfg.negative != "auto" else p.negative,
    }


def run_protocol(p: Protocol, ctx, calibrator: Calibrator):
    """Return (aggregate scores, per-perturbation rows, wall-clock seconds) for one protocol."""
    if p.kind == "ranking":
        return _run_ranking(p, ctx, calibrator)
    roles = resolve_controls(p, ctx.cfg)
    needed = {role: roles[role] for role in calibrator.requires}
    _check_sources(p, needed)

    def work(pert):
        ctx.current_pert = pert
        gt = ctx.view(pert, "gt", p)
        raws = {role: p.algo(gt, ctx.view(pert, src, p), ctx) for role, src in needed.items()}
        return pert, raws, calibrator.per_pert(raws, p)

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers(ctx.cfg)) as pool:
        results = list(pool.map(work, ctx.perturbations))
    seconds = perf_counter() - start

    rows, per_pert = [], []
    for pert, raws, value in results:
        per_pert.append(value)
        rows.append({"protocol": p.name, "perturbation": pert,
                     **{f"raw_{role}": raws[role] for role in needed},
                     calibrator.name: value})
    return calibrator.aggregate(np.asarray(per_pert, dtype=float)), rows, seconds


def _run_ranking(p: Protocol, ctx, calibrator: Calibrator):
    """Cross-perturbation ranking protocols: build the candidate/GT profile matrices
    over all perturbations once, then score each perturbation's retrieval rank.

    Perturbations are treated as a single group (these datasets are single-covariate);
    drf instead ranks within each covariate group."""
    roles = resolve_controls(p, ctx.cfg)
    needed = {role: roles[role] for role in calibrator.requires}
    perts = ctx.perturbations

    start = perf_counter()
    gt = np.vstack([ctx.centroid(pert, "gt", p.centering) for pert in perts])
    scores = {role: p.algo(np.vstack([ctx.centroid(pert, src, p.centering) for pert in perts]), gt)
              for role, src in needed.items()}
    seconds = perf_counter() - start

    rows, per_pert = [], []
    for i, pert in enumerate(perts):
        raws = {role: float(scores[role][i]) for role in needed}
        value = calibrator.per_pert(raws, p)
        per_pert.append(value)
        rows.append({"protocol": p.name, "perturbation": pert,
                     **{f"raw_{role}": raws[role] for role in needed},
                     calibrator.name: value})
    return calibrator.aggregate(np.asarray(per_pert, dtype=float)), rows, seconds


def compute_de_export(ctx, methods):
    """{method: (statistic, pvalue_adj)} matrices (perturbations x genes) for each
    method's GT(first-half)-vs-all-perturbed differential expression."""
    out = {}
    for method in methods:
        ctx.cfg.de_method = method

        def work(pert):
            de = ctx.de(pert, "gt", "all_perturbed")
            return de.score, de.pvalue_adj

        with ThreadPoolExecutor(max_workers=n_workers(ctx.cfg)) as pool:
            res = list(pool.map(work, ctx.perturbations))
        out[method] = (np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res]))
    return out


def _check_sources(p: Protocol, roles: dict) -> None:
    if p.kind in ("population", "de"):
        for role, src in roles.items():
            if SOURCES.meta(src).get("provides") != "cells":
                raise ValueError(
                    f"{p.name}: {role} source {src!r} provides "
                    f"{SOURCES.meta(src).get('provides')}, but a single-cell protocol needs cells"
                )
