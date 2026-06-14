"""EPPS command-line interface."""
from __future__ import annotations

import argparse
from datetime import datetime

from . import io
from .blocks.de import DE_METHODS
from .blocks.spaces import SPACES
from .calibrators import CALIBRATORS
from .context import Context
from .dataset import Dataset
from .protocols.table import GROUPS, PROTOCOL_TABLE, PROTOCOLS, TEMPLATE_TABLE, TEMPLATES
from .runner import compute_de_export, run_protocol
from .sources import SOURCES
from .types import Protocol, RunConfig


def _resolve_token(token: str) -> list[Protocol]:
    if token == "all":
        return list(PROTOCOL_TABLE) + [t.build(t.default) for t in TEMPLATE_TABLE]
    if token in GROUPS:
        return ([p for p in PROTOCOL_TABLE if p.group == token]
                + [t.build(t.default) for t in TEMPLATE_TABLE if t.group == token])
    if "=" in token:                                     # a template with a value, e.g. mmd_top_k=30
        name, _, value = token.partition("=")
        if name not in TEMPLATES:
            raise SystemExit(f"unknown parameterised protocol {name!r}; try `epps list protocols`")
        t = TEMPLATES[name]
        return [t.build(t.cast(value))]
    if token in PROTOCOLS:
        return [PROTOCOLS[token]]
    if token in TEMPLATES:                               # a template with no value -> its default
        t = TEMPLATES[token]
        return [t.build(t.default)]
    raise SystemExit(f"unknown protocol {token!r}; try `epps list protocols`")


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


def cmd_run(args) -> None:
    protocols = resolve_protocols(args.protocols or ["all"])
    cfg = RunConfig(
        dataset=args.dataset, protocols=[p.name for p in protocols], de_method=args.de_method,
        subsample=args.subsample, seed=args.seed, positive=args.positive,
        negative=args.negative, output=args.output, out_dir=args.out_dir,
        workers=args.workers, perturbation_key=args.perturbation_key,
        control_label=args.control_label, min_cells=args.min_cells,
        profile=args.profile,
    )
    calibrator = CALIBRATORS[cfg.output]
    ctx = Context(Dataset.load(cfg.dataset, cfg), cfg)
    ctx.warm(protocols)
    aggregates, rows, timed = {}, [], []
    for p in protocols:
        agg, proto_rows, seconds = run_protocol(p, ctx, calibrator)
        aggregates[p.name] = agg
        rows += proto_rows
        timed.append((p, seconds))
    if not args.quiet:
        io.print_summary(cfg, aggregates, calibrator, protocols)
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = io.write_rows(cfg, rows, stamp)
    print(f"-> {path}")
    if cfg.profile:
        print(f"-> {io.write_timing(cfg, timed, stamp)}")


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
        lines = [f"{p.name:24s} ({p.group}, {p.kind}, space={p.space})" for p in PROTOCOL_TABLE]
        lines += [f"{t.name:24s} ({t.group}, {t.kind}, {t.param}=…) — {t.description}"
                  for t in TEMPLATE_TABLE]
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
    parser = argparse.ArgumentParser(prog="epps", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="compute protocol calibration for one dataset")
    run.add_argument("dataset", help="preprocessed .h5ad")
    run.add_argument("-p", "--protocols", action="append", default=[],
                     help="comma-separated names, a group (pseudobulk|distributional|de), or 'all'")
    run.add_argument("--de-method", choices=DE_METHODS.names(), default="t-test",
                     help="DE backend for EVERY DE-dependent unit in the run: the interp "
                          "positive control, the top_k/degs spaces, the de_* protocols, and the WMSE weights")
    run.add_argument("--subsample", type=int, default=8192)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--positive", default="auto")
    run.add_argument("--negative", default="auto")
    run.add_argument("--output", choices=list(CALIBRATORS), default="drf")
    run.add_argument("--out-dir", default="results")
    run.add_argument("--workers", type=int, default=0, help="threads (0 = auto)")
    run.add_argument("--perturbation-key", default="perturbation")
    run.add_argument("--control-label", default="control")
    run.add_argument("--min-cells", type=int, default=30,
                     help="skip perturbations with fewer cells")
    run.add_argument("--profile", action="store_true",
                     help="also write a per-protocol wall-clock timing table")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

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
