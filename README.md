<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Virtual-Cell-Research-Community/scPertEval/main/docs/_static/logo/scPertEval-dark-logo.svg">
  <img alt="scPertEval" src="https://raw.githubusercontent.com/Virtual-Cell-Research-Community/scPertEval/main/docs/_static/logo/scPertEval-logo.svg" width="400">
</picture>

# scPertEval — Evaluation Protocols for Perturbation Sequencing

[![Stars][stars-badge]][stars-link]
[![PyPI][pypi-badge]][pypi-link]
[![PyPI Downloads][pepy-badge]][pepy-link]
[![Docs][docs-badge]][docs-link]
[![Lint][lint-badge]][lint-link]
[![Test][test-badge]][test-link]
[![Build][build-badge]][build-link]
[![Codecov][codecov-badge]][codecov-link]

scPertEval is a command-line tool for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies. The same catalog of protocols backs
three commands: **`score`** (score a model's predictions against ground truth), **`calibrate`**
(calibrate a protocol against empirical positive/negative controls per perturbation, reporting the
**Dynamic Range Fraction (DRF)** and **Bound Discrimination Score (BDS)**), and **`de`** (export
per-gene differential expression).

Our accompanying publication: TODO_LINK_HERE

**→ Full documentation at <https://scperteval.readthedocs.io/>**

## Install

```bash
pip install scperteval
```

Or from this repo:

```bash
pip install "scperteval @ git+https://github.com/Virtual-Cell-Research-Community/scPertEval.git"
```

## Quick start

```bash
# calibrate protocols against built-in controls (DRF/BDS)
scperteval calibrate data/wessels23.h5ad -p all --de-method t-test

# score a model's predictions against ground truth
scperteval score data/wessels23.h5ad predictions.h5ad -p all

scperteval list protocols   # also: de-methods | spaces | sources | calibrators
```

Sample datasets are available at
`https://storage.googleapis.com/scperteval/processed/<dataset>_processed_complete.h5ad`.

---

**Contributing:** see [CONTRIBUTORS.md](CONTRIBUTORS.md).

[stars-badge]: https://img.shields.io/github/stars/Virtual-Cell-Research-Community/scPertEval?style=flat&logo=GitHub&color=yellow
[stars-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/stargazers
[pypi-badge]: https://img.shields.io/pypi/v/scperteval.svg
[pypi-link]: https://pypi.org/project/scperteval
[pepy-badge]: https://static.pepy.tech/badge/scperteval
[pepy-link]: https://pepy.tech/project/scperteval
[docs-badge]: https://readthedocs.org/projects/scperteval/badge/?version=latest
[docs-link]: https://scperteval.readthedocs.io/
[lint-badge]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/lint.yaml/badge.svg
[lint-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/lint.yaml
[test-badge]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/test.yaml/badge.svg
[test-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/test.yaml
[build-badge]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/build.yaml/badge.svg
[build-link]: https://github.com/Virtual-Cell-Research-Community/scPertEval/actions/workflows/build.yaml
[codecov-badge]: https://codecov.io/gh/Virtual-Cell-Research-Community/scPertEval/branch/main/graph/badge.svg
[codecov-link]: https://codecov.io/gh/Virtual-Cell-Research-Community/scPertEval
