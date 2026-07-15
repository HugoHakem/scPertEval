"""Resolve protocol specs (the ``-p`` DSL) into concrete :class:`~scperteval.types.Protocol` objects.

Shared by the CLI and the Python API so both accept the exact same spec language: ``"all"``,
a group name (``pseudobulk``/``distributional``/``de``), a bare protocol name, or a tunable
protocol with a value (``name=value``, e.g. ``mse_top_k=30``).
"""

from __future__ import annotations

from ..types import Protocol
from .table import GROUPS, PROTOCOLS, TABLE


def _concrete(p: Protocol) -> Protocol:
    """A tunable protocol at its default value; a fixed protocol unchanged."""
    return p.resolve(p.param.default) if p.parameterised else p  # type: ignore[union-attr]


def _resolve_token(token: str) -> list[Protocol]:
    if token == "all":
        return [_concrete(p) for p in TABLE]
    if token in GROUPS:
        return [_concrete(p) for p in TABLE if p.group == token]
    if "=" in token:  # a tunable protocol with a value, e.g. mse_top_k=30
        name, _, value = token.partition("=")
        p = PROTOCOLS.get(name)
        if p is None or not p.parameterised:
            raise ValueError(f"unknown tunable protocol {name!r}; try `scperteval list protocols`")
        return [p.resolve(p.param.cast(value))]  # type: ignore[union-attr]
    p = PROTOCOLS.get(token)
    if p is None:
        raise ValueError(f"unknown protocol {token!r}; try `scperteval list protocols`")
    return [_concrete(p)]


def resolve_protocols(specs: list[str]) -> list[Protocol]:
    """Resolve protocol specs to a de-duplicated list of concrete protocols."""
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
