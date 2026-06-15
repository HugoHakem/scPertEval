"""A minimal decorator registry for the pluggable building blocks."""

from __future__ import annotations

from typing import Callable


class Registry:
    """Maps a name to a function plus optional metadata, populated by decoration."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, tuple[Callable, dict]] = {}

    def register(self, name: str, **meta) -> Callable:
        def deco(fn: Callable) -> Callable:
            self._items[name] = (fn, meta)
            return fn

        return deco

    def add(self, name: str, fn: Callable, **meta) -> None:
        self._items[name] = (fn, meta)

    def __getitem__(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._items[name][0]

    def meta(self, name: str) -> dict:
        return self._items[name][1]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)
