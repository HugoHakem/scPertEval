# API

## Core types

```{eval-rst}
.. module:: scperteval.types
.. currentmodule:: scperteval.types

.. autosummary::
    :toctree: generated

    RunConfig
    Protocol
    Calibrator
    DEResult
    Param
```

## Runner

```{eval-rst}
.. module:: scperteval.runner
.. currentmodule:: scperteval.runner

.. autosummary::
    :toctree: generated

    run_protocol
```

## Protocols

- `scperteval.protocols.TABLE` — list of all `Protocol` objects.
- `scperteval.protocols.PROTOCOLS` — `{name: Protocol}` dict.
- `scperteval.protocols.GROUPS` — sorted list of group names.

```{eval-rst}
.. protocol-table::
```

### Metrics

```{eval-rst}
.. module:: scperteval.protocols.metrics
.. currentmodule:: scperteval.protocols.metrics

.. automodule:: scperteval.protocols.metrics
   :no-members:
   :no-index:

.. autosummary::
    :toctree: generated

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

## Calibrators

```{eval-rst}
.. automodule:: scperteval.calibrators
   :no-members:
```

`scperteval.calibrators.CALIBRATORS` — `{name: Calibrator}` dict of built-in calibrators (`drf`, `bds`).
Add entries here to register a new calibrator; see [Add a calibrator](user-guide/building-blocks.md#add-a-calibrator).

## Building blocks

### Differential expression

```{eval-rst}
.. module:: scperteval.blocks.de
.. currentmodule:: scperteval.blocks.de

.. automodule:: scperteval.blocks.de
   :no-members:
   :no-index:

.. autosummary::
    :toctree: generated

    DE_METHODS
    moments
    bh
    ttest_from_moments
    de_ttest
    de_ttest_overestim
    de_mwu
```

### Feature spaces

```{eval-rst}
.. module:: scperteval.blocks.spaces
.. currentmodule:: scperteval.blocks.spaces

.. automodule:: scperteval.blocks.spaces
   :no-members:
   :no-index:

.. autosummary::
    :toctree: generated

    SPACES
    register_de_space
    top_space
    pca_space
    degs_space
```

### Control sources

```{eval-rst}
.. automodule:: scperteval.sources
   :no-members:
```

`scperteval.sources.SOURCES` — registry of all control/reference sources.
Add entries here to register a new source; see [Add a control source](user-guide/building-blocks.md#add-a-control-source).

## Context

```{eval-rst}
.. module:: scperteval.context
.. currentmodule:: scperteval.context

.. autosummary::
    :toctree: generated

    Context
```

## Registry

```{eval-rst}
.. module:: scperteval.registry
.. currentmodule:: scperteval.registry

.. autosummary::
    :toctree: generated

    Registry
```

## Dataset & I/O

```{eval-rst}
.. module:: scperteval.dataset
.. currentmodule:: scperteval.dataset

.. autosummary::
    :toctree: generated

    Dataset
    to_dense
```

```{eval-rst}
.. module:: scperteval.io
.. currentmodule:: scperteval.io

.. autosummary::
    :toctree: generated

    print_summary
    write_rows
    write_timing
    write_de
```
