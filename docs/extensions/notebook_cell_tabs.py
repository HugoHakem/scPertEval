"""Group tagged, adjacent notebook code cells into a sphinx-design tab-set.

Tag cells ``tab-item:<label>`` (e.g. ``tab-item:CLI``, ``tab-item:Python``) to
have them rendered as tabs in the docs, while staying as plain, independently
runnable cells in the raw notebook (Jupyter/Colab/JupyterLab never see the
tab-set markup, so nothing degrades there -- see the discussion on why we
avoid writing ``{tab-set}``/``{tab-item}`` directives directly into notebook
markdown cells).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docutils import nodes
from sphinx_design.shared import create_component

if TYPE_CHECKING:
    from sphinx.application import Sphinx

TAG_PREFIX = "tab-item:"


def _is_cell_code(node: nodes.Node) -> bool:
    return isinstance(node, nodes.Element) and node.get("nb_element") == "cell_code"


def _tab_label(node: nodes.Node) -> str | None:
    if not isinstance(node, nodes.Element):
        return None
    tags = node.get("cell_metadata", {}).get("tags", [])
    for tag in tags:
        if tag.startswith(TAG_PREFIX):
            return tag[len(TAG_PREFIX) :]
    return None


def group_tagged_cells_into_tabs(_app: Sphinx, doctree: nodes.document) -> None:
    """Replace runs of adjacent ``tab-item:``-tagged cells with a tab-set."""
    consumed: set[int] = set()
    for cell_node in list(doctree.findall(_is_cell_code)):
        if id(cell_node) in consumed or _tab_label(cell_node) is None or cell_node.parent is None:
            continue
        parent = cell_node.parent
        siblings = parent.children
        start = siblings.index(cell_node)
        end = start
        while end < len(siblings) and _tab_label(siblings[end]) is not None:
            end += 1
        run = siblings[start:end]
        if len(run) < 2:
            continue  # a lone tagged cell isn't a tab-set
        consumed.update(id(member) for member in run)

        tab_set = create_component("tab-set", classes=["sd-tab-set"])
        for member in run:
            label = _tab_label(member)
            assert label is not None
            tab_item = create_component("tab-item", classes=["sd-tab-item"], selected=False)
            tab_label_node = nodes.rubric(label, "", nodes.Text(label), classes=["sd-tab-label"])
            tab_content = create_component("tab-content", classes=["sd-tab-content"])
            tab_content += member.children
            tab_item += tab_label_node
            tab_item += tab_content
            tab_set += tab_item

        parent.replace(run[0], tab_set)
        for member in run[1:]:
            parent.remove(member)


def setup(app: Sphinx) -> dict:
    """App setup hook."""
    app.connect("doctree-read", group_tagged_cells_into_tabs)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
