"""Feature spaces: which features a protocol scores on, chosen before the metric runs.

Every space is a decorated rule in ``catalog.py`` — that is the file to
open to see what exists or to add one. ``registry.py`` holds the
machinery: the catalog of definitions and the named instances built from them.
"""

from __future__ import annotations

from .registry import OPS, SPACES, SetOps, Space, SpaceRegistry, combine_subsets

__all__ = ["OPS", "SPACES", "SetOps", "Space", "SpaceRegistry", "combine_subsets"]
