# EPPS publication figures

Running the notebook `epps-publication-figures.ipynb` reproduces the DRF and DEG
(DEG-Jaccard) figures from the EPPS preprint.

The notebook downloads the 7 preprocessed datasets from the public GCS bucket
(`gs://perturb-seq-datasets/epps/processed/`), installs and runs EPPS to compute the DRF
table, the per-metric timing table, and the DEG-Jaccard composite, then renders the
figures. The three `*.py` scripts in this directory are the figure renderers it invokes.

To run: open `epps-publication-figures.ipynb` in Jupyter and run all cells. It needs the
`gsutil` SDK on `PATH` for the dataset download, and the download plus the EPPS run are
long-running.
