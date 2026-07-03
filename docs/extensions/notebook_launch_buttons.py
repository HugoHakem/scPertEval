"""Correct Colab launch buttons for notebook pages sourced from a git submodule.

sphinx_book_theme's ``launch_buttons`` reads org/repo/branch from the single
global ``html_context`` (see ``header_buttons.get_repo_parts``), so a notebook
rendered from a submodule gets a Colab link pointing at the *parent* repo
instead of the submodule's own repo. We leave ``launch_buttons`` off in
conf.py and add the correct button here instead, deriving org/repo/commit
from the submodule's own git remote so it doesn't need to be hardcoded.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docutils.nodes import document
    from sphinx.application import Sphinx

# Sphinx docname prefix -> path (relative to docs/) of the submodule it's rendered from.
SUBMODULE_PAGES = {"notebooks/notebooks/": "notebooks"}


@lru_cache
def _submodule_origin(path: str) -> tuple[str, str, str] | None:
    """Return (org, repo, commit) for the git submodule checked out at `path`.

    Pins to the exact checked-out commit rather than a branch name: the
    submodule is normally checked out at a detached commit (pinned by the
    parent repo's gitlink), so there is no reliable branch to read, and
    pinning to the commit guarantees the Colab notebook always matches what
    the docs actually rendered.
    """
    try:
        url = subprocess.check_output(["git", "-C", path, "remote", "get-url", "origin"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    if "github.com" not in url:
        return None
    org, repo = url.split("github.com")[-1].strip("/:").removesuffix(".git").split("/")
    return org, repo, commit


def add_submodule_colab_button(
    app: Sphinx, pagename: str, _templatename: str, context: dict, _doctree: document | None
) -> None:
    """Add a Colab button pointing at the submodule's own repo, not the parent's."""
    for prefix, subpath in SUBMODULE_PAGES.items():
        if not pagename.startswith(prefix):
            continue
        origin = _submodule_origin(str(Path(app.confdir) / subpath))
        if origin is None:
            return
        org, repo, commit = origin
        rel_path = pagename[len(prefix) :] + ".ipynb"
        colab_url = f"https://colab.research.google.com/github/{org}/{repo}/blob/{commit}/{rel_path}"
        context["header_buttons"].append(
            {
                "type": "group",
                "tooltip": "Launch interactive content",
                "icon": "fas fa-rocket",
                "label": "submodule-launch-buttons",
                "buttons": [
                    {
                        "type": "link",
                        "text": "Colab",
                        "tooltip": "Launch on Colab",
                        "icon": "_static/images/logo_colab.png",
                        "url": colab_url,
                    }
                ],
            }
        )
        return


def setup(app: Sphinx) -> dict:
    """App setup hook."""
    # priority > 500 so context["header_buttons"] (initialized by the theme's
    # prep_header_buttons, priority 500) already exists.
    app.connect("html-page-context", add_submodule_colab_button, priority=600)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
