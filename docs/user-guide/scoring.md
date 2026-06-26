# How scoring works (the calibration)

scPertEval's claim — a usable catalog of protocols — rests on **calibrating** each protocol
against two empirical controls per perturbation, so you can see whether a metric actually
separates signal from baseline rather than read a raw, uninterpretable number.

- **positive control** — the best realistic candidate: the **technical duplicate** (a
  held-out replicate) for single-cell protocols, the **interpolated duplicate** for pseudobulk.
- **negative control** — an uninformative baseline: the **all-perturbed reference,
  excluding the target perturbation** (a full-resolution mean for pseudobulk; an 8192-cell
  subsample for single-cell distances).

## Dynamic Range Fraction (DRF)

Where the protocol's value sits between the negative control (floor) and the perfect score,
anchored by the positive control:

```text
DRF = (positive − negative) / (perfect − negative)        # per perturbation, clipped to [-1, 1]
```

`--output drf` reports the mean/median across perturbations. High DRF means the protocol
discriminates real signal; near zero means it doesn't. Introduced by {cite:t}`Miller_2025`.

## Bound Discrimination Score (BDS)

The fraction of perturbations for which the positive control beats the negative control
under this protocol:

```text
BDS = fraction of perturbations where  positive control beats negative control     # in [0, 1]
```

`--output bds` reports this fraction. It's a sensitivity check: a protocol with low BDS
can't even tell a technical replicate from an uninformative baseline, so its scores
shouldn't be trusted. Introduced by {cite:t}`Vollenweider_2026`.
