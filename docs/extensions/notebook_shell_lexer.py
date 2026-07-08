"""Re-lex ``%%bash``/``%%sh`` notebook cells as real bash.

myst-nb applies one lexer (from the notebook's ``language_info``) to every
code cell's literal_block, and only records it as a *string* ``language``
attribute for Sphinx builds -- the actual pygments highlighting happens later,
in the HTML writer. So we don't need to re-run pygments ourselves: we just
need to find literal_blocks whose source starts with a ``%%bash``/``%%sh``
cell magic and overwrite that attribute before the HTML writer gets to it.

We search for literal_block nodes directly, rather than going through the
cell_code container myst-nb wraps them in: other doctree-read extensions
(e.g. notebook_cell_tabs) may already have unwrapped that container by the
time this runs, since doctree-read handler order isn't guaranteed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from docutils import nodes

if TYPE_CHECKING:
    from sphinx.application import Sphinx

CELL_MAGIC_RE = re.compile(r"^%%(bash|sh)\b")


def relex_shell_magic_cells(_app: Sphinx, doctree: nodes.document) -> None:
    """Override the pygments language for %%bash/%%sh cells to real bash."""
    for block in doctree.findall(nodes.literal_block):
        if CELL_MAGIC_RE.match(block.astext()):
            block["language"] = "bash"


def setup(app: Sphinx) -> dict:
    """App setup hook."""
    app.connect("doctree-read", relex_shell_magic_cells)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
