# scPertEval — Evaluation Protocols for Perturbation Sequencing

scPertEval is a command-line tool for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies.

Evaluating predictions across a dataset's perturbations reduces to a single question: how
different is one group of cells from another? To answer this, an **evaluation protocol** is
defined: a specific formulation of a metric, along with some representation of the
perturbation data fed to the metric. However, there are a multitude of possibilities — many
already reflected in the literature — and it can be challenging to compare and contrast
protocols across the field and ultimately choose the right approach for a given dataset and
problem space.

scPertEval renders each protocol as a short, readable building block to run, read, reuse,
and contribute back — a place for collaboration and alignment in the field. Run the tool by
specifying a dataset, one or more protocols, and a method of differential expression; the
tool outputs calibration data: the **Dynamic Range Fraction (DRF)** and the **Bound
Discrimination Score (BDS)** — quantifying how well the protocol separates real perturbation
signal from an uninformative baseline (see [How scoring works](user-guide/scoring.md)).

## Quick start

```bash
pip install scperteval
scperteval run data/wessels23.h5ad -p all --de-method t-test
```

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} {octicon}`desktop-download;1em;` Installation
:link: installation
:link-type: doc
Get scPertEval installed and set up your development environment.
:::

:::{grid-item-card} {octicon}`book;1em;` User guide
:link: user-guide/index
:link-type: doc
Learn how to run protocols, interpret scores, and explore the building blocks.
:::

:::{grid-item-card} {octicon}`mortar-board;1em;` Tutorials
:link: tutorials
:link-type: doc
Step-by-step notebooks: CLI walkthrough, Python API, and extending the tool.
:::

:::{grid-item-card} {octicon}`code-square;1em;` API reference
:link: api
:link-type: doc
Full reference for the Python API.
:::

:::{grid-item-card} {octicon}`mark-github;1em;` GitHub
:link: https://github.com/Virtual-Cell-Research-Community/scPertEval
:link-type: url
Browse the source code, open issues, or contribute a pull request.
:::

::::

## Citation

If you use scPertEval, please cite {cite}`Schafer_2026`.

```bibtex
@unpublished{Schafer_2026,
    author = {Schäfer, Philipp S. L. and Reid, Kendall A. and Boldyga, Zach
              and Aksu, Ekin Deniz and Hakem, Hugo and Saez-Rodriguez, Julio},
    title  = {Towards a Principled Evaluation of Single-Cell Perturbation
              Response Prediction Models},
    note   = {In preparation},
    year   = {2026},
}
```

```{toctree}
:hidden: true
:maxdepth: 2

installation.md
user-guide/index
tutorials.md
api.md
changelog.md
Contributing <contributing.md>
references.md
```
