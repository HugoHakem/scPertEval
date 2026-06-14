"""The declarative table of evaluation protocols — every protocol in one list (``TABLE``).

Each row is one protocol:

- a fully-specified protocol is a ``Protocol(...)`` row;
- a *parameterised* protocol — one whose feature space or metric takes a value supplied at
  the CLI (e.g. ``-p mse_top_k=30``) — is a ``template(...)`` row carrying a parameter
  family (``top_k`` / ``pca_k`` / ``degs_padj`` select the space; ``overlap_k`` feeds the
  metric). With no value a template uses the family default.

To add a protocol, write a metric in ``metrics.py`` and add one row to ``TABLE`` below.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

from ..blocks.spaces import degs_space, pca_space, top_space
from ..types import Protocol
from . import metrics as M


# --- parameter families: a value supplied at the CLI selects a space (or feeds the metric) ---
@dataclass(frozen=True)
class Param:
    """How a CLI value (k / padj) is cast, defaulted, and applied to a parameterised row."""

    name: str
    cast: Callable
    default: float
    space: Optional[Callable] = None   # value -> space name; if None the value is a metric kwarg


top_k = Param("k", int, 50, space=top_space)              # top-k DEGs by effect size
pca_k = Param("k", int, 50, space=pca_space)              # k principal components
degs_padj = Param("padj", float, 0.05, space=degs_space)  # DEGs at adjusted p < padj
overlap_k = Param("k", int, 50)                           # feeds de_overlap's k (a metric arg)


# --- shared wiring bundles (controls + score scale), splatted into rows with ** ---
_PB = dict(group="pseudobulk", positive="interpolated", negative="all_perturbed_mean")
_PB_CTRL = dict(group="pseudobulk", positive="interpolated", negative="control")
_LOWER = dict(better="lower", perfect=0.0)
_DIST = dict(group="distributional", positive="tech_dup", negative="all_perturbed", better="lower", perfect=0.0)
_DE = dict(group="de", positive="tech_dup", negative="all_perturbed", reference="all_perturbed",
           neg_reference="control", better="higher", perfect=1.0)
_RANK = dict(group="pseudobulk", positive="interpolated", negative="global_mean", better="lower", perfect=0.0)


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


@dataclass(frozen=True)
class Template:
    """A parameterised protocol row; ``build(value)`` returns a concrete ``Protocol``."""

    name: str
    metric: Callable
    representation: str
    param: Param
    wiring: dict

    @property
    def group(self):
        return self.wiring["group"]

    @property
    def cast(self):
        return self.param.cast

    @property
    def default(self):
        return self.param.default

    def build(self, v) -> Protocol:
        name = f"{self.name}={_fmt(v)}"
        if self.param.space is not None:                       # the value selects a feature space
            return Protocol(name, self.metric, representation=self.representation,
                            space=self.param.space(v), **self.wiring)
        metric = partial(self.metric, **{self.param.name: v})  # the value feeds the metric
        return Protocol(name, metric, representation=self.representation, **self.wiring)


def template(name, metric, param, *, representation, **wiring) -> Template:
    """One parameterised protocol row — mirrors ``Protocol(...)`` plus a ``param`` family."""
    return Template(name, metric, representation, param, wiring)


TABLE = [
    # --- pseudobulk: correlation & error (positive = interpolated duplicate) ---
    Protocol("pearson", M.pearson, representation="centroid", **_PB),
    Protocol("pearson_ctrl", M.pearson, representation="centroid", centering="ctrl", **_PB),
    Protocol("pearson_pert", M.pearson, representation="centroid", centering="allpert", **_PB_CTRL),
    Protocol("mse", M.mse, representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp1", partial(M.weighted_mse, exp=1.0), representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp2", partial(M.weighted_mse, exp=2.0), representation="centroid", **_PB, **_LOWER),
    Protocol("wmse_exp4", partial(M.weighted_mse, exp=4.0), representation="centroid", **_PB, **_LOWER),
    template("mse_top_k", M.mse, top_k, representation="centroid", **_PB, **_LOWER),
    template("mse_degs_padj", M.mse, degs_padj, representation="centroid", **_PB, **_LOWER),
    template("pearson_pert_top_k", M.pearson, top_k, representation="centroid", centering="allpert", **_PB_CTRL),
    template("pearson_pert_degs_padj", M.pearson, degs_padj, representation="centroid", centering="allpert", **_PB_CTRL),

    # --- cross-perturbation retrieval rank ---
    Protocol("rank", partial(M.rank_retrieval, transpose=False), representation="ranking", **_RANK),
    Protocol("transpose_rank", partial(M.rank_retrieval, transpose=True), representation="ranking", **_RANK),

    # --- distributional: distances between cell populations (positive = technical duplicate) ---
    template("unbiased_mmd_median_top_k", M.unbiased_mmd_median, top_k, representation="population", **_DIST),
    template("unbiased_mmd_median_pca_k", M.unbiased_mmd_median, pca_k, representation="population", **_DIST),
    template("energy_distance_top_k", M.energy_distance, top_k, representation="population", **_DIST),
    template("energy_distance_pca_k", M.energy_distance, pca_k, representation="population", **_DIST),
    template("sinkhorn_w2_top_k", M.sinkhorn_w2, top_k, representation="population", **_DIST),
    template("sinkhorn_w2_pca_k", M.sinkhorn_w2, pca_k, representation="population", **_DIST),

    # --- differential expression: GT DEGs vs prediction ranking ---
    Protocol("de_auprc", M.de_auprc, representation="de", **_DE),
    Protocol("de_auroc", M.de_auroc, representation="de", **_DE),
    template("de_overlap_k", M.de_overlap, overlap_k, representation="de", **_DE),
]

PROTOCOL_TABLE = [r for r in TABLE if isinstance(r, Protocol)]
TEMPLATE_TABLE = [r for r in TABLE if isinstance(r, Template)]
PROTOCOLS = {p.name: p for p in PROTOCOL_TABLE}
TEMPLATES = {t.name: t for t in TEMPLATE_TABLE}
GROUPS = sorted({r.group for r in TABLE})
