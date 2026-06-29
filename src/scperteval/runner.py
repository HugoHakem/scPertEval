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
    """Run one protocol over every perturbation and apply the calibrator.

    ``p.scope`` chooses the execution path: ``"perturbation"``-scope protocols run in a
    thread pool (one perturbation at a time); ``"dataset"``-scope protocols collect all
    perturbations' datapoints first, then call the metric once.

    Parameters
    ----------
    p : ~scperteval.types.Protocol
        The concrete (non-parameterised) protocol to evaluate.
    ctx : ~scperteval.context.Context
        Per-run context holding the dataset, caches, and building blocks.
    calibrator : ~scperteval.types.Calibrator
        Calibrator that converts raw positive/negative control scores into a final value.

    Returns
    -------
    aggregates : dict
        Aggregate scores across perturbations (e.g. ``{"mean": 0.42, "median": 0.38}``).
    rows : list of dict
        Per-perturbation records with raw control values and the calibrated score.
    seconds : float
        Wall-clock time for this protocol.
    """
    roles = resolve_controls(p, ctx.cfg)
    needed = {role: roles[role] for role in calibrator.requires}
    _check_sources(p, needed)
    run = _run_dataset if p.scope == "dataset" else _run_per_perturbation
    return run(p, ctx, calibrator, needed)


def _finalize(p, calibrator, perts, raws_list):
    """Per-perturbation rows + the aggregate, from each perturbation's raw control values."""
    per_pert = [calibrator.per_pert(raws, p) for raws in raws_list]
    rows = [
        {
            "protocol": p.name,
            "perturbation": pert,
            **{f"raw_{role}": raws[role] for role in raws},
            calibrator.name: value,
        }
        for pert, raws, value in zip(perts, raws_list, per_pert)
    ]
    return calibrator.aggregate(np.asarray(per_pert, dtype=float)), rows


def _run_per_perturbation(p: Protocol, ctx, calibrator: Calibrator, needed: dict):
    """Score one perturbation at a time (across a thread pool), gt vs each control."""

    def work(pert):
        ctx.current_pert = pert
        gt = ctx.view(pert, "gt", p)
        return {role: p.metric(gt, ctx.view(pert, src, p), ctx) for role, src in needed.items()}

    perts = ctx.perturbations
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers(ctx.cfg)) as pool:
        raws_list = list(pool.map(work, perts))
    seconds = perf_counter() - start
    agg, rows = _finalize(p, calibrator, perts, raws_list)
    return agg, rows, seconds


def _run_dataset(p: Protocol, ctx, calibrator: Calibrator, needed: dict):
    """Dataset-scope protocols: build every perturbation's gt and control datapoints, hand
    the metric the full lists at once, then read off each perturbation's score.

    Perturbations are treated as a single group (these datasets are single-covariate);
    drf instead ranks within each covariate group.
    """
    perts = ctx.perturbations

    def collect(source):
        out = []
        for pert in perts:
            ctx.current_pert = pert
            out.append(ctx.view(pert, source, p))
        return out

    start = perf_counter()
    gt = collect("gt")
    scores = {role: p.metric(gt, collect(src), ctx) for role, src in needed.items()}
    seconds = perf_counter() - start

    raws_list = [{role: float(scores[role][i]) for role in needed} for i in range(len(perts))]
    agg, rows = _finalize(p, calibrator, perts, raws_list)
    return agg, rows, seconds


def compute_de_export(ctx, methods):
    """{method: (statistic, pvalue_adj)} matrices (perturbations x genes) for each
    method's GT(first-half)-vs-all-perturbed differential expression.
    """
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
    if p.representation in ("population", "de"):
        for role, src in roles.items():
            if SOURCES.meta(src).get("provides") != "cells":
                raise ValueError(
                    f"{p.name}: {role} source {src!r} provides "
                    f"{SOURCES.meta(src).get('provides')}, but a single-cell protocol needs cells"
                )
