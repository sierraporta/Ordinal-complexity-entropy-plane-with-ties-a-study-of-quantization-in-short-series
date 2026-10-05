# Ordinal complexity–entropy plane with ties: a study of quantization in short series

Working repository for an ongoing study (D. Sierra-Porta, A. M. S. Borin Jr.).
**Work in progress**: the code and results here are reproducible, but the write-up is
still being written and reviewed. Comments, checks and extensions are welcome.

## The question

Ordinal (Bandt–Pompe) analysis assumes a series without ties, yet the records the
complexity–entropy plane is usually applied to are quantized: integer samples, particle
or photon counts, digitized signals. Ties are then unavoidable, and the common fix is to
break them with an arbitrary rule, which imposes an order the data do not contain. This
study asks what happens if we **keep the ties** instead, work on the generalized-ordinal-pattern
(weak-order) alphabet, and treat the quantization step as the variable of interest:

1. How does the finite-sample bias of the plane coordinates under overlapping windows behave
   on the tie alphabet?
2. How does the quantization step relocate a series in the plane, and can that shift be corrected?
3. Which null hypothesis is valid for testing a quantized series, once ties are present?

The scripts below are the numerical investigation behind these questions. They are
self-checking: each reproduces a specific result and prints its numbers or writes its figure.

## Requirements
Python 3.10+ with the packages in `requirements.txt` (`pip install -r requirements.txt`).

## Scripts

| script | what it investigates |
|---|---|
| `gop_bias_theory.py`    | exact finite-sample overlap bias on the tie alphabet; self-test that Δ→0 recovers the tie-free limit |
| `gop_theory_results.py` | compares the exact bias to Monte Carlo (m=3,4) and traces the exact plane path vs the quantization step |
| `gop_R3_sheppard.py` / `gop_R3_figure.py` | the leading-order quantization law and its test across source densities |
| `gop_R4_groundtruth.py` | tie-alphabet coordinates for white noise, AR(1), Poisson counts, the logistic map; AR(1) estimator demo |
| `gop_ar1_exact.py`      | exact quantized-AR(1) coordinates via multivariate-normal box probabilities (transfer operator) |
| `gop_R5_null.py`        | a tie-matched permutation null: its size and power vs the naive white-noise null |
| `gop_R6_rr.py`          | the ideas applied to a real RR-interval record (record 14134) |
| `derisk_gop.py`         | first sanity check: the overlap-bias machinery ports to the 75-symbol tie alphabet |

Shared helpers: `fsb_extensions.py`, `apply_diagnostics.py` (ordinal-estimator routines).
Cached console outputs are in `results_*.txt`, figures in `figure_*.pdf` / `figure_*.png`,
small numeric caches in `*.npy`.

Example:
```
python gop_theory_results.py
python gop_R5_null.py
python gop_R6_rr.py        # reads RealData/RR_14134.csv
```

## Data
`RealData/RR_14134.csv` — RR intervals (inter-beat times) in integer samples, record 14134 of
the MIT-BIH Long-Term ECG Database (PhysioNet; Goldberger et al., *Circulation* 2000), 128 Hz.

## Status
A draft write-up exists (`manuscript_ties.pdf`, read-only; the editable source is kept on
Overleaf) but it is unfinished and under review. This repository is the reproducible companion,
not the paper.
