# Contributing guide

We welcome contributions! Please open an issue or pull request on [GitHub](https://github.com/Virtual-Cell-Research-Community/scPertEval).

## Installing dev dependencies

:::::{tab-set}
::::{tab-item} uv

```bash
uv sync --group dev
```

::::

::::{tab-item} pip

```bash
pip install -e ".[dev]"
```

::::
:::::

## Code style

This project uses [ruff][] for formatting and linting, and [mypy][]/[pyright][] for type checking.

```bash
uv run ruff format .
uv run ruff check .
uv run mypy scperteval
```

[ruff]: https://docs.astral.sh/ruff/

## Building the docs locally

```bash
uv sync --group doc
uv run sphinx-build -M html docs docs/_build -W
```

Then open `docs/_build/html/index.html`.

## Publishing a release

Update the version in `pyproject.toml`, commit, push, and create a GitHub release tagged `vX.Y.Z`.

## Writing documentation

- Use [numpy-style docstrings][numpydoc].
- Add tutorials as Jupyter notebooks in `docs/notebooks/`.
- Add intersphinx entries to `docs/conf.py` for cross-references to external packages.

[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
