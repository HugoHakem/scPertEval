"""Sphinx directive that auto-generates a reference table of evaluation protocols."""
from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class ProtocolTableDirective(Directive):
    """Emit a table of all protocols from ``scperteval.protocols.TABLE``."""

    def run(self):
        from scperteval.protocols import TABLE

        table = nodes.table()
        tgroup = nodes.tgroup(cols=4)
        table += tgroup

        for _ in range(4):
            tgroup += nodes.colspec(colwidth=1)

        thead = nodes.thead()
        tgroup += thead
        header_row = nodes.row()
        for text in ("Name", "Group", "Representation", "Better"):
            entry = nodes.entry()
            entry += nodes.paragraph(text=text)
            header_row += entry
        thead += header_row

        tbody = nodes.tbody()
        tgroup += tbody
        for p in TABLE:
            row = nodes.row()

            # Name cell — link to the metric function page if available
            name_entry = nodes.entry()
            metric_name = p.metric.__name__ if hasattr(p.metric, "__name__") else p.name
            ref_id = f"scperteval.protocols.metrics.{metric_name}"
            ref = nodes.reference("", p.name, internal=True, refuri=f"generated/{ref_id}.html")
            name_para = nodes.paragraph()
            name_para += ref
            name_entry += name_para
            row += name_entry

            for text in (p.group, p.representation, p.better):
                entry = nodes.entry()
                entry += nodes.paragraph(text=text)
                row += entry

            tbody += row

        return [table]


def setup(app: Sphinx):
    app.add_directive("protocol-table", ProtocolTableDirective)
    return {"version": "0.1", "parallel_read_safe": True}
