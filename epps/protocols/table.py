"""The declarative list of evaluation protocols.

Concrete protocols are explicit rows in ``PROTOCOL_TABLE``. Parameterised ones are
templates (``TEMPLATE_TABLE``) named with their parameter — e.g. ``mmd_top_k`` — and a
value is supplied per protocol at the CLI (``-p mmd_top_k=30``), defaulting otherwise.
To add a protocol, write an algorithm in ``algorithms.py`` and append a row or template here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from ..blocks.spaces import degs_space, pca_space, top_space
from ..types import Protocol
from . import algorithms as A

_PB = dict(group="pseudobulk", positive="interp", negative="mean")
_PB_CTRL = dict(group="pseudobulk", positive="interp", negative="control")
_LOWER = dict(direction="lower", perfect=0.0)
_DIST = dict(group="distributional", positive="tech_dup", negative="all_perturbed", direction="lower", perfect=0.0)
_DE = dict(group="de", positive="tech_dup", negative="all_perturbed", reference="all_perturbed",
           neg_reference="control", direction="higher", perfect=1.0)
_RANK = dict(group="pseudobulk", positive="interp", negative="global_mean", direction="lower", perfect=0.0)

PROTOCOL_TABLE = [
    Protocol("pearson", A.pearson, "centroid", **_PB),
    Protocol("pearson_ctrl", A.pearson, "centroid", centering="ctrl", **_PB),
    Protocol("pearson_pert", A.pearson, "centroid", centering="allpert", **_PB_CTRL),
    Protocol("mse", A.mse, "centroid", **_PB, **_LOWER),
    Protocol("wmse_exp1", partial(A.weighted_mse, exp=1.0), "centroid", **_PB, **_LOWER),
    Protocol("wmse_exp2", partial(A.weighted_mse, exp=2.0), "centroid", **_PB, **_LOWER),
    Protocol("wmse_exp4", partial(A.weighted_mse, exp=4.0), "centroid", **_PB, **_LOWER),
    Protocol("rank", partial(A.rank_retrieval, transpose=False), "ranking", **_RANK),
    Protocol("transpose_rank", partial(A.rank_retrieval, transpose=True), "ranking", **_RANK),
    Protocol("de_auprc", A.de_auprc, "de", **_DE),
    Protocol("de_auroc", A.de_auroc, "de", **_DE),
]
PROTOCOLS = {p.name: p for p in PROTOCOL_TABLE}


@dataclass(frozen=True)
class Template:
    """A parameterised protocol; ``build(value)`` returns a concrete Protocol."""

    name: str
    group: str
    kind: str
    param: str          # display name of the parameter, e.g. "k" or "padj"
    cast: Callable      # int / float
    default: float
    build: Callable
    description: str


def _fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def _space_tpl(name, algo, kind, family, param, cast, default, desc, wiring, centering=None):
    """A template whose parameter selects the feature space (top_k / pca_k / degs_padj)."""
    def build(v):
        extra = {"centering": centering} if centering else {}
        return Protocol(f"{name}={_fmt(v)}", algo, kind, space=family(v), **extra, **wiring)
    return Template(name, wiring["group"], kind, param, cast, default, build, desc)


def _de_overlap_build(k):
    return Protocol(f"de_overlap_k={k}", partial(A.de_overlap, k=k), "de", **_DE)


TEMPLATE_TABLE = [
    _space_tpl("mse_top_k", A.mse, "centroid", top_space, "k", int, 50,
               "MSE on the top-k DEGs", {**_PB, **_LOWER}),
    _space_tpl("mse_degs_padj", A.mse, "centroid", degs_space, "padj", float, 0.05,
               "MSE on significant DEGs (adjusted p < padj)", {**_PB, **_LOWER}),
    _space_tpl("pearson_pert_top_k", A.pearson, "centroid", top_space, "k", int, 50,
               "Pearson vs all-perturbed mean on the top-k DEGs", _PB_CTRL, centering="allpert"),
    _space_tpl("pearson_pert_degs_padj", A.pearson, "centroid", degs_space, "padj", float, 0.05,
               "Pearson vs all-perturbed mean on significant DEGs", _PB_CTRL, centering="allpert"),
    _space_tpl("mmd_top_k", A.mmd, "population", top_space, "k", int, 50,
               "unbiased MMD on the top-k DEGs", _DIST),
    _space_tpl("mmd_pca_k", A.mmd, "population", pca_space, "k", int, 50,
               "unbiased MMD in k-dim PCA space", _DIST),
    _space_tpl("energy_top_k", A.energy, "population", top_space, "k", int, 50,
               "energy distance on the top-k DEGs", _DIST),
    _space_tpl("energy_pca_k", A.energy, "population", pca_space, "k", int, 50,
               "energy distance in k-dim PCA space", _DIST),
    _space_tpl("sinkhorn_top_k", A.sinkhorn, "population", top_space, "k", int, 50,
               "Sinkhorn 2-Wasserstein on the top-k DEGs", _DIST),
    _space_tpl("sinkhorn_pca_k", A.sinkhorn, "population", pca_space, "k", int, 50,
               "Sinkhorn 2-Wasserstein in k-dim PCA space", _DIST),
    Template("de_overlap_k", "de", "de", "k", int, 50, _de_overlap_build,
             "top-k gene overlap between GT and prediction DE rankings"),
]
TEMPLATES = {t.name: t for t in TEMPLATE_TABLE}
GROUPS = sorted({p.group for p in PROTOCOL_TABLE} | {t.group for t in TEMPLATE_TABLE})
