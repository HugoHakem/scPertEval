"""scPertEval command-line interface."""
from __future__ import annotations

import argparse
from datetime import datetime

from . import io
from .blocks.de import DE_METHODS
from .blocks.spaces import SPACES
from .calibrators import CALIBRATORS
from .context import Context
from .dataset import Dataset
from .predictions import PredictionSet
from .protocols.table import GROUPS, PROTOCOLS, TABLE
from .runner import compute_de_export, run_protocol
from .sources import SOURCES
from .types import Protocol, RunConfig


def _concrete(p: Protocol) -> Protocol:
    """A tunable protocol at its default value; a fixed protocol unchanged."""
    return p.resolve(p.param.default) if p.parameterised else p


def _resolve_token(token: str) -> list[Protocol]:
    if token == "all":
        return [_concrete(p) for p in TABLE]
    if token in GROUPS:
        return [_concrete(p) for p in TABLE if p.group == token]
    if "=" in token:                                     # a tunable protocol with a value, e.g. mse_top_k=30
        name, _, value = token.partition("=")
        p = PROTOCOLS.get(name)
        if p is None or not p.parameterised:
            raise SystemExit(f"unknown tunable protocol {name!r}; try `scperteval list protocols`")
        return [p.resolve(p.param.cast(value))]
    p = PROTOCOLS.get(token)
    if p is None:
        raise SystemExit(f"unknown protocol {token!r}; try `scperteval list protocols`")
    return [_concrete(p)]


def resolve_protocols(specs: list[str]) -> list[Protocol]:
    out: list[Protocol] = []
    for spec in specs:
        for token in spec.split(","):
            token = token.strip()
            if token:
                out += _resolve_token(token)
    by_name: dict[str, Protocol] = {}
    for p in out:
        by_name.setdefault(p.name, p)
    return list(by_name.values())


def _evaluate(cfg: RunConfig, protocols, ctx, quiet: bool) -> None:
    """Run every protocol over the dataset, print the summary, and write the per-perturbation
    CSV. Shared by ``calibrate`` and ``score`` (prediction vs ground truth); they differ only
    in how ``ctx`` is built and which calibrator ``cfg.output`` selects."""
    calibrator = CALIBRATORS[cfg.output]
    ctx.warm(protocols)
    aggregates, rows, timed = {}, [], []
    for p in protocols:
        agg, proto_rows, seconds = run_protocol(p, ctx, calibrator)
        aggregates[p.name] = agg
        rows += proto_rows
        timed.append((p, seconds))
    if not quiet:
        io.print_summary(cfg, aggregates, calibrator, protocols)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = io.write_rows(cfg, rows, stamp)
    print(f"-> {path}")
    if cfg.profile:
        print(f"-> {io.write_timing(cfg, timed, stamp)}")


def cmd_calibrate(args) -> None:
    protocols = resolve_protocols(args.protocols or ["all"])
    cfg = RunConfig(
        dataset=args.dataset, protocols=[p.name for p in protocols], de_method=args.de_method,
        subsample=args.subsample, seed=args.seed, positive=args.positive,
        negative=args.negative, output=args.output, out_dir=args.out_dir,
        workers=args.workers, perturbation_key=args.perturbation_key,
        control_label=args.control_label, min_cells=args.min_cells,
        profile=args.profile,
    )
    ctx = Context(Dataset.load(cfg.dataset, cfg), cfg)
    _evaluate(cfg, protocols, ctx, args.quiet)


def cmd_score(args) -> None:
    protocols = resolve_protocols(args.protocols or ["all"])
    cfg = RunConfig(
        dataset=args.dataset, protocols=[p.name for p in protocols], de_method=args.de_method,
        subsample=args.subsample, seed=args.seed, output="score", out_dir=args.out_dir,
        workers=args.workers, perturbation_key=args.perturbation_key,
        control_label=args.control_label, min_cells=args.min_cells, profile=args.profile,
        predictions=args.predictions, truth="gt_all_cells",
    )
    ds = Dataset.load(cfg.dataset, cfg)
    ctx = Context(ds, cfg)
    ctx.predictions = PredictionSet.load(cfg.predictions, ds, cfg)
    _evaluate(cfg, protocols, ctx, args.quiet)


def cmd_de(args) -> None:
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cfg = RunConfig(
        dataset=args.dataset, protocols=[], de_method=methods[0],
        subsample=args.subsample, seed=args.seed, out_dir=args.out_dir,
        workers=args.workers, min_cells=args.min_cells,
        perturbation_key=args.perturbation_key, control_label=args.control_label,
    )
    ctx = Context(Dataset.load(cfg.dataset, cfg), cfg)
    ctx._ensure_ref_sums()
    results = compute_de_export(ctx, methods)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = io.write_de(cfg, ctx.ds.var_names, ctx.perturbations, results, stamp)
    print(f"-> {path}  ({len(ctx.perturbations)} perturbations, methods={methods})")


def cmd_list(args) -> None:
    def reg(registry, fmt):
        return [fmt(n, registry.meta(n)) for n in registry.names()]

    if args.what == "protocols":
        def descr(p):
            scope = "" if p.scope == "perturbation" else f", {p.scope}-wide"
            knob = f"{p.param.name}=…" if p.parameterised else f"space={p.space}"
            return f"{p.group}, {p.representation}{scope}, {knob}"
        lines = [f"{p.name:24s} ({descr(p)})" for p in TABLE]
    elif args.what == "de-methods":
        lines = reg(DE_METHODS, lambda n, m: f"{n:10s} — {m.get('description', '')}")
    elif args.what == "spaces":
        lines = reg(SPACES, lambda n, m: f"{n:10s} — {m.get('description', '')}")
    elif args.what == "sources":
        lines = reg(SOURCES, lambda n, m: f"{n:14s} ({m.get('provides')}) — {m.get('description', '')}")
    elif args.what == "calibrators":
        lines = [f"{n:6s} — {c.description}" for n, c in CALIBRATORS.items()]
    print("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="scperteval", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    calibrate = sub.add_parser(
        "calibrate", help="calibrate protocols against positive/negative controls (DRF/BDS)")
    calibrate.add_argument("dataset", help="preprocessed .h5ad")
    calibrate.add_argument("-p", "--protocols", action="append", default=[],
                           help="comma-separated names, a group (pseudobulk|distributional|de), or 'all'")
    calibrate.add_argument("--de-method", choices=DE_METHODS.names(), default="t-test",
                           help="DE backend for EVERY DE-dependent unit in the run: the interpolated "
                                "positive control, the top_k/degs spaces, the de_* protocols, and the WMSE weights")
    calibrate.add_argument("--subsample", type=int, default=8192)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--positive", default="auto")
    calibrate.add_argument("--negative", default="auto")
    calibrate.add_argument(
        "--output", default="drf",
        choices=[n for n, c in CALIBRATORS.items() if "prediction" not in c.requires],
        help="how per-perturbation values are calibrated (drf/bds)")
    calibrate.add_argument("--out-dir", default="results")
    calibrate.add_argument("--workers", type=int, default=0, help="threads (0 = auto)")
    calibrate.add_argument("--perturbation-key", default="perturbation")
    calibrate.add_argument("--control-label", default="control")
    calibrate.add_argument("--min-cells", type=int, default=30,
                           help="skip perturbations with fewer cells")
    calibrate.add_argument("--profile", action="store_true",
                           help="also write a per-protocol wall-clock timing table")
    calibrate.add_argument("--quiet", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    score = sub.add_parser(
        "score", help="score model predictions against ground truth (real cells), per protocol")
    score.add_argument("dataset", help="preprocessed .h5ad — the ground truth (real cells)")
    score.add_argument("predictions", help="predicted .h5ad — same genes and perturbation labels")
    score.add_argument("-p", "--protocols", action="append", default=[],
                       help="comma-separated names, a group (pseudobulk|distributional|de), or 'all'")
    score.add_argument("--de-method", choices=DE_METHODS.names(), default="t-test",
                       help="DE backend for every DE-dependent unit (the top_k/degs spaces, the "
                            "de_* protocols, and the WMSE weights)")
    score.add_argument("--subsample", type=int, default=8192,
                       help="cells in the all-perturbed reference sample (the ground truth itself is never subsampled)")
    score.add_argument("--seed", type=int, default=42)
    score.add_argument("--out-dir", default="results")
    score.add_argument("--workers", type=int, default=0, help="threads (0 = auto)")
    score.add_argument("--perturbation-key", default="perturbation")
    score.add_argument("--control-label", default="control")
    score.add_argument("--min-cells", type=int, default=30,
                       help="skip perturbations with fewer cells")
    score.add_argument("--profile", action="store_true",
                       help="also write a per-protocol wall-clock timing table")
    score.add_argument("--quiet", action="store_true")
    score.set_defaults(func=cmd_score)

    de = sub.add_parser("de", help="write per-gene DE (statistic + adj p) per method to HDF5")
    de.add_argument("dataset", help="preprocessed .h5ad")
    de.add_argument("--methods", default="t-test,MWU",
                    help="comma-separated DE methods to compute (GT first-half vs all-perturbed)")
    de.add_argument("--subsample", type=int, default=8192)
    de.add_argument("--seed", type=int, default=42)
    de.add_argument("--out-dir", default="results")
    de.add_argument("--workers", type=int, default=0)
    de.add_argument("--min-cells", type=int, default=30)
    de.add_argument("--perturbation-key", default="perturbation")
    de.add_argument("--control-label", default="control")
    de.set_defaults(func=cmd_de)

    lst = sub.add_parser("list", help="list available building blocks")
    lst.add_argument("what", choices=["protocols", "de-methods", "spaces", "sources", "calibrators"])
    lst.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    args.func(args)
