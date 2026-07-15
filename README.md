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

scPertEval is a toolkit for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies, usable both as a **command-line
interface** and as a **native Python API**. The same catalog of protocols backs three actions:
**`score`** (score a model's predictions against ground truth), **`calibrate`** (calibrate a
protocol against empirical positive/negative controls per perturbation, reporting the **Dynamic
Range Fraction (DRF)** and **Bound Discrimination Score (BDS)**), and **`de`** (export per-gene
differential expression).

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

The Sinkhorn / optimal-transport metrics (the `sinkhorn_w2_*` protocols) need PyTorch and
[GeomLoss](https://www.kernel-operations.io/geomloss/), which are optional to keep the base
install light. Enable them with the `sinkhorn` extra:

```bash
pip install "scperteval[sinkhorn]"
```

## Quick start

From the command line:

```bash
# calibrate protocols against built-in controls (DRF/BDS)
scperteval calibrate data/wessels23.h5ad -p all --de-method t-test

# score a model's predictions against ground truth
scperteval score data/wessels23.h5ad predictions.h5ad -p all

scperteval list protocols   # also: de-methods | spaces | sources | calibrators
```

Or from Python — the same protocols, returning results in memory (see the
[Python API guide](https://scperteval.readthedocs.io/en/latest/user-guide/python-api.html)):

```python
import scperteval as sp

prep = sp.prepare("data/wessels23.h5ad", "pearson_ctrl")   # read + index once, reusable
result = sp.calibrate(prep, "pearson_ctrl", de_method="t-test")
result.aggregate          # {"mean": …, "median": …} — calibrated DRF summary
result.per_perturbation   # the per-perturbation detail table
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
