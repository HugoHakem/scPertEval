# scPertEval — Evaluation Protocols for Perturbation Sequencing

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
