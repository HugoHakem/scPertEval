"""Guard against the shared knob defaults drifting across RunConfig, the API, and the CLI."""

from __future__ import annotations

import dataclasses
import inspect

from scperteval import api, cli
from scperteval.types import RunConfig

# Knobs whose defaults are (necessarily) restated in more than one place.
SHARED = ["de_method", "subsample", "seed", "min_cells", "perturbation_key", "control_label", "workers"]


def _runconfig_defaults() -> dict:
    return {f.name: f.default for f in dataclasses.fields(RunConfig)}


def test_api_signature_defaults_match_runconfig():
    rc = _runconfig_defaults()
    for fn in (api.prepare, api.calibrate, api.score, api.de):
        params = inspect.signature(fn).parameters
        for key in SHARED:
            if key in params and params[key].default is not inspect.Parameter.empty:
                assert params[key].default == rc[key], (
                    f"{fn.__name__}: {key}={params[key].default!r} != RunConfig {rc[key]!r}"
                )


def test_cli_argparse_defaults_match_runconfig():
    rc = _runconfig_defaults()
    parser = cli.build_parser()
    # argparse stores "--de-method" as dest "de_method", etc.
    for cmd, args in (
        ("calibrate", ["x.h5ad"]),
        ("score", ["x.h5ad", "pred.h5ad"]),
        ("de", ["x.h5ad"]),
    ):
        ns = vars(parser.parse_args([cmd, *args]))
        for key in SHARED:
            if key in ns:
                assert ns[key] == rc[key], f"cli {cmd}: {key}={ns[key]!r} != RunConfig {rc[key]!r}"
