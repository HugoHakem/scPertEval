# scPertEval — Evaluation Protocols for Perturbation Sequencing

scPertEval is a command-line tool for **experimenting with and sharing reference implementations of
evaluation protocols** in single-cell perturbation studies. It calibrates each protocol
against empirical positive and negative controls per perturbation, outputting the
**Dynamic Range Fraction (DRF)** and the **Bound Discrimination Score (BDS)**.

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
scperteval run data/wessels23.h5ad -p all --de-method t-test
scperteval list protocols   # also: de-methods | spaces | sources | calibrators
```

Sample datasets are available at
`https://storage.googleapis.com/scperteval/processed/<dataset>_processed_complete.h5ad`.

---

**Contributing:** see [CONTRIBUTORS.md](CONTRIBUTORS.md).
