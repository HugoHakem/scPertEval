# Installation

## From PyPI

```bash
pip install scperteval
```

## From source

```bash
pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"
```

Or, for an editable install from a local clone:

```bash
git clone https://github.com/Virtual-Cell-Research-Community/scPertEval.git
cd scPertEval
pip install -e .
```

## Development setup

Install all dev dependencies (linting + docs + tests):

```bash
uv sync --group dev
```

Run linters:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/scperteval
```

Build the docs locally with live reload:

```bash
uv sync --group docs
uv run sphinx-autobuild docs docs/_build/html
```
