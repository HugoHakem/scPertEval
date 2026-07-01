# Protocols reference

Full reference for the built-in metric functions. Each function implements one protocol's
core computation: it receives two arrays (or `DEResult` objects) and returns a scalar score.

```{eval-rst}
.. module:: scperteval.protocols.metrics
.. currentmodule:: scperteval.protocols.metrics

.. automodule:: scperteval.protocols.metrics
   :no-members:
   :no-index:

.. autosummary::
    :toctree: ../generated

    pearson
    mse
    weighted_mse
    energy_distance
    unbiased_mmd_median
    sinkhorn_w2
    rank_retrieval
    de_auprc
    de_auroc
    de_overlap
```
