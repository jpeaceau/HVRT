# HART / FastHART Benchmark Findings

**Date**: 2026-03-03
**Version**: hvrt 2.10.0
**Benchmark**: `run_benchmarks.py --datasets housing multimodal --tasks reduce expand`
**Datasets**: housing (log-normal/heavy-tailed, d=6), multimodal (3 Gaussian clusters, d=10)
**Reduction ratios**: 0.1, 0.2, 0.3, 0.5
**Expansion ratios**: 1×, 2×, 5× (max_n=300)

---

## What HART changes vs HVRT

| Component | HVRT / FastHVRT | HART / FastHART |
|---|---|---|
| Input normalisation | mean + std (StandardScaler) | median + 1.4826·MAD (_MADScaler) |
| Tree split criterion | `squared_error` | `absolute_error` |
| y-component | std-normalised extremeness | MAD-normalised extremeness |
| Partitioning signal | joint variance | joint absolute deviation |

---

## Reduction — Key Findings

### Gaussian / near-Gaussian data (multimodal, 3-cluster)

**HART ≈ HVRT on all metrics across all reduction ratios.**

| ratio | HART-size mf | HVRT-size mf | HART-size ml_delta | HVRT-size ml_delta |
|---|---|---|---|---|
| 0.10 | 0.806 | 0.806 | −0.002 | −0.002 |
| 0.20 | 0.859 | 0.859 | +0.001 | +0.001 |
| 0.30 | 0.890 | 0.891 | +0.001 | +0.001 |
| 0.50 | 0.934 | 0.933 | +0.001 | +0.001 |

> **Interpretation**: When within-partition distributions are locally Gaussian, median/MAD and
> mean/std produce equivalent normalisations. Partitioning is structurally identical.
> HART provides no benefit, but also no penalty.

### Heavy-tailed data (housing, log-normal features)

At aggressive ratios (10%), HVRT edges out HART on marginal fidelity:

| ratio | HART-size mf | FastHART-size mf | HVRT-size mf | FastHVRT-size mf |
|---|---|---|---|---|
| 0.10 | 0.692 | 0.804 | 0.784 | 0.785 |
| 0.20 | 0.787 | 0.861 | 0.840 | 0.849 |
| 0.30 | 0.837 | 0.892 | 0.876 | 0.885 |
| 0.50 | 0.901 | 0.933 | 0.922 | 0.928 |

**FastHART-size wins on marginal fidelity at all reduction ratios** on housing data,
beating both HVRT-size and FastHVRT-size.

- HART-size (pairwise) lags behind at low ratios — absolute_error splits create coarser
  boundaries on log-normal data than squared_error (MAE-optimal splits are less sensitive
  to large values, reducing the influence of the very outliers that define log-normal tails).
- FastHART-size (z-score sum) avoids this: the simpler O(d) target is better calibrated
  to the MAD-normalised feature space.

**ML utility** (ml_delta relative to TRTR baseline) is largely equivalent across all
HVRT-family variants on both datasets. Differences are within ±0.01 at all ratios.

### Reduction summary

| Recommendation | Condition |
|---|---|
| FastHART-size | Best marginal fidelity on heavy-tailed / log-normal data |
| HVRT-size or HART-size | Near-equivalent on Gaussian data; prefer HVRT-size for interpretability |
| All size-weighted variants | Consistently outperform variance-weighted on marginal fidelity |

---

## Expansion — Key Findings

### Heavy-tailed data (housing, log-normal)

| ratio | FastHART-size ml_delta | FastHVRT-size ml_delta | HART-size ml_delta | HVRT-size ml_delta |
|---|---|---|---|---|
| 1× | −0.089 | −0.108 | **−0.010** | −0.066 |
| 2× | **+0.078** | +0.037 | −0.136 | −0.108 |
| 5× | **+0.040** | +0.037 | −0.016 | −0.072 |

Notable at **1× expansion** (generation from same-size sample):
- **HART-size achieves ml_delta = −0.010** — nearly perfect ML utility retention,
  compared to −0.066 for HVRT-size and −0.108 for FastHVRT-size.
- The absolute-error tree criterion appears to produce better-calibrated partitions
  at small budgets on log-normal data.

At **2×–5× expansion**, FastHART-size leads (ml_delta +0.04–+0.08). HART-size
struggles at 2× due to MAD-based covariance estimation interacting with very skewed
partitions — partition budgets are larger and the KDE becomes over-smooth.

### Gaussian / near-Gaussian data (multimodal)

| ratio | HART-var ml_delta | HART-size ml_delta | HVRT-size ml_delta |
|---|---|---|---|
| 1× | **+0.003** | −0.000 | −0.002 |
| 2× | −0.001 | +0.005 | **+0.009** |
| 5× | **+0.010** | +0.008 | +0.007 |

- HART-var (variance-weighted budget allocation) wins at 5× expansion on multimodal
  data. The MAD-based partitioning provides slightly tighter local densities on
  cluster edges, improving generation quality at high expansion ratios.
- At 2×, HVRT-size leads (ml_delta +0.009 vs +0.005 for HART-size).
- Differences are small (within ±0.01); neither model dominates the other on
  near-Gaussian data.

### Expansion summary

| Recommendation | Condition |
|---|---|
| **HART-size** | 1× expansion on heavy-tailed data (best ML utility retention) |
| **FastHART-size** | 2×–5× expansion on heavy-tailed data |
| **HVRT-size** | 2× expansion on Gaussian data |
| **HART-var** | 5× expansion on Gaussian / multimodal data |

---

## Privacy (DCR)

HART produces DCR scores in the same range as HVRT (0.44–0.69 on housing,
0.46–0.64 on multimodal). Neither model systematically outperforms the other
on privacy. DCR is primarily governed by bandwidth, not by the normalisation
scheme. See `findings_dcr.md` for bandwidth-vs-DCR analysis.

---

## Discriminator accuracy

Discriminator scores are effectively identical between HART and HVRT across all
conditions (within ±0.02). Both models produce synthetic data that is as hard
to discriminate from real data as each other. The normalisation scheme does not
affect indistinguishability.

---

## When to choose HART over HVRT

| Scenario | Recommended model | Reason |
|---|---|---|
| Heavy-tailed / log-normal features | **FastHART** (expand) | +5–8pp ML utility at 2×–5× |
| Heavy-tailed, 1× replication | **HART-size** | −1pp vs TRTR; HVRT loses −7pp |
| Log-normal data, reduction | **FastHART-size** | Highest marginal fidelity |
| Gaussian / multimodal data | **HVRT** or **HART** (tie) | No meaningful difference |
| Unknown distribution | **HVRT** (default) | Equivalent on Gaussian; proven baseline |

HART is not a replacement for HVRT — it is an alternative that is better calibrated
for datasets with substantial skew or heavy tails, where mean/std normalisation
overstates the influence of outliers in the partitioning target.

---

## Bandwidth benchmark

Run with `--model hart` to assess bandwidth sensitivity for HART:

```bash
python benchmarks/bandwidth_benchmark.py --model hart --quick
python benchmarks/bandwidth_benchmark.py --model hart --ratios 2 5 10
```

The same bandwidth candidates (scott, silverman, h=0.10 … h=2.00, epanechnikov,
adaptive) apply to HART expansion. Preliminary expectation: h=0.10 remains the
strongest general-purpose bandwidth for HART, since the within-partition
distribution structure is similar after MAD normalisation. Full bandwidth sweep
for HART TBD.
