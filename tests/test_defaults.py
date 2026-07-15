"""Guard against the shared knob defaults drifting across RunConfig, the API, and the CLI."""

from __future__ import annotations

import dataclasses
import inspect

from scperteval import api, cli
from scperteval.types import RunConfig

# Dataset/run knobs that live on `prepare` (necessarily restated from RunConfig's defaults).
PREPARE_KNOBS = ["subsample", "seed", "min_cells", "perturbation_key", "control_label", "workers"]


def _runconfig_defaults() -> dict:
    return {f.name: f.default for f in dataclasses.fields(RunConfig)}


def test_prepare_defaults_match_runconfig():
    rc = _runconfig_defaults()
    params = inspect.signature(api.prepare).parameters
    for key in PREPARE_KNOBS:
        assert params[key].default == rc[key], f"prepare: {key}={params[key].default!r} != RunConfig {rc[key]!r}"


def test_verb_de_method_defaults_match_runconfig():
    rc = _runconfig_defaults()
    assert inspect.signature(api.calibrate).parameters["de_method"].default == rc["de_method"]
    assert inspect.signature(api.score).parameters["de_method"].default == rc["de_method"]
    assert inspect.signature(api.de).parameters["method"].default == rc["de_method"]


def test_cli_argparse_defaults_match_runconfig():
    rc = _runconfig_defaults()
    parser = cli.build_parser()
    for cmd, args in (("calibrate", ["x.h5ad"]), ("score", ["x.h5ad", "pred.h5ad"]), ("de", ["x.h5ad"])):
        ns = vars(parser.parse_args([cmd, *args]))
        for key in [*PREPARE_KNOBS, "de_method"]:
            if key in ns:
                assert ns[key] == rc[key], f"cli {cmd}: {key}={ns[key]!r} != RunConfig {rc[key]!r}"
