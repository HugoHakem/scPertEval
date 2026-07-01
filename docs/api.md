# API reference

```{toctree}
:hidden:

api/types
api/protocols
api/extensions
api/io
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} Core types & runner
:link: api/types
:link-type: doc

`RunConfig`, `Protocol`, `Calibrator`, `DEResult`, `Param`, `run_protocol`
:::

:::{grid-item-card} Protocols
:link: api/protocols
:link-type: doc

Built-in metric functions: `pearson`, `mse`, `de_auprc`, `rank_retrieval`, …
:::

:::{grid-item-card} Extension API
:link: api/extensions
:link-type: doc

`Registry`, `SPACES`, `DE_METHODS`, `SOURCES`, `PredictionSet`, `Context`
:::

:::{grid-item-card} Dataset & I/O
:link: api/io
:link-type: doc

`Dataset`, `to_dense`, `write_rows`, `write_de`, …
:::

::::

## Protocols

- `scperteval.protocols.TABLE` — list of all `Protocol` objects.
- `scperteval.protocols.PROTOCOLS` — `{name: Protocol}` dict.
- `scperteval.protocols.GROUPS` — sorted list of group names.

```{eval-rst}
.. protocol-table::
```
