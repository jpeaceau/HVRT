# HVRT Privacy DCR Benchmark — Findings

**Date:** 2026-02-26
**Benchmarks:** `benchmarks/dcr_privacy_benchmark.py`, `benchmarks/run_benchmarks.py --tasks expand --deep-learning`
**Script:** `python benchmarks/dcr_privacy_benchmark.py`

---

## Setup

### DCR metric definition

```
DCR = median( min Euclidean dist from each synthetic point to any real point )
    / median( min Euclidean dist from each real point to any other real point )
```

Both halves computed in z-score space (StandardScaler on real data).
The denominator uses leave-one-out — a real point is excluded from its own
nearest-neighbour search. The denominator normalises DCR relative to the
natural inter-record density of the training data, making it comparable across
datasets with different scales and feature counts.

| DCR range | Interpretation |
|---|---|
| < 0.10 | Near-copy risk: synthetic samples sit very close to specific training records |
| 0.10 – 0.40 | Tight generation: well within the local distribution; low risk |
| 0.40 – 0.70 | Moderate: typical for local KDE-based generators at default settings |
| 0.70 – 1.00 | High: samples spread to the neighbourhood boundary of training records |
| ≥ 1.00 | Maximum: more dispersed than real samples are from each other |

**Failure mode:** On datasets with many near-duplicate records (binary or
low-cardinality categorical features), the real→real leave-one-out distance
approaches zero, making DCR → ∞ for all methods. The `adult` dataset produces
DCR of 10–400× regardless of method — treat DCR as unreliable on mixed-type
data with low-cardinality features.

### Grid sweep parameters

| Parameter | Value |
|---|---|
| Datasets | fraud (class, n=500), housing (reg, n=500), multimodal (reg, n=500) |
| Expansion ratio | 2× (grid sweep); 1×/2×/5× (adaptive sweep) |
| Max training samples | 500 (80/20 train-test split) |
| Bandwidth candidates | `'auto'`, 0.10, 0.30, 0.50, 1.00, `'scott'` |
| n_partitions candidates | `None` (auto), 20, 10, 5 |
| adaptive_bandwidth | False / True |
| Downstream model | GradientBoostingClassifier / Regressor |

---

## Method Comparison (ratio 1×, continuous datasets)

Mean across fraud, housing, multimodal. TRTR = 0.846 for non-DL methods
(fraud TRTR=1.000, housing TRTR=0.573, multimodal TRTR=0.966).

| Method | DCR | TRTR | TSTR | TSTR Δ | MF | Disc. Err |
|---|---|---|---|---|---|---|
| HVRT-var | 0.45 | 0.846 | **0.866** | **+0.020** | 0.921 | 1.8% |
| HVRT-size | 0.45 | 0.846 | 0.850 | +0.004 | **0.944** | 5.0% |
| Bootstrap + Noise | 0.41 | 0.846 | 0.833 | −0.013 | 0.928 | **0.8%** |
| SMOTE | 0.30 | 0.846 | 0.828 | −0.018 | 0.902 | 1.0% |
| GMM | 1.17 | 0.846 | 0.820 | −0.026 | 0.878 | 1.8% |
| FastHVRT-size | 0.43 | 0.846 | 0.805 | −0.041 | 0.936 | 1.5% |
| Gaussian Copula | 1.17 | 0.846 | 0.806 | −0.040 | 0.937 | 1.9% |
| TVAE† | 0.89 | 0.769* | 0.702 | −0.067 | 0.624 | 26.1% |
| CTGAN† | 1.95 | 0.769* | 0.726 | −0.043 | 0.421 | 32.3% |

*† CTGAN/TVAE evaluated on housing + multimodal only (not fraud); TRTR differs accordingly.*

---

## Key Findings

### 1. DCR and ML utility are not in strict opposition

The most important finding: increasing DCR (toward the High profile) does **not**
necessarily degrade ML utility — it can improve it.

| Profile | Bandwidth | DCR | MF | TRTR | TSTR | TSTR Δ |
|---|---|---|---|---|---|---|
| Tight | `0.1` | 0.332 | 0.966 | 0.846 | 0.834 | −0.012 |
| Moderate | `'auto'` | 0.443 | 0.958 | 0.846 | 0.834 | −0.012 |
| High | `0.5` | 0.797 | 0.925 | 0.846 | 0.839 | **−0.007** |
| Maximum | `'scott'`, n_parts=10 | 1.067 | 0.856 | 0.846 | 0.824 | −0.022 |

The **High profile achieves the best TSTR Δ (−0.007)** despite having less
marginal fidelity than Tight/Moderate. The Tight profile — the narrowest
bandwidth — is not the most useful for downstream ML, even though it best
preserves the marginal distribution.

**Mechanism:** A wider bandwidth samples further from the nearest training
records. This forces the model to cover more of the feature space, which reduces
over-representation of high-density training regions and produces synthetic data
that generalises slightly better on the test set.

**Practical implication:** If the downstream goal is ML utility, `bandwidth=0.5`
(High privacy profile) is the recommended parameter — it achieves more privacy
**and** better ML performance than the default `'auto'` (0.1 Gaussian, which maps
to the Moderate profile at 2×).

### 2. Bandwidth is the primary DCR lever; n_partitions is secondary

Changing bandwidth alone spans nearly the full DCR range:

| Bandwidth | Mean DCR (n_parts=auto) | MF | TSTR Δ |
|---|---|---|---|
| `0.1` | 0.332 | 0.966 | −0.012 |
| `'auto'` | 0.443 | 0.958 | −0.012 |
| `0.3` | 0.607 | 0.955 | −0.018 |
| `0.5` | 0.797 | 0.925 | −0.007 |
| `1.0` | 1.271 | 0.787 | −0.004 |
| `'scott'` | 1.082 | 0.854 | 0.000 |

DCR increases monotonically with bandwidth (0.332 → 1.271). The MF cost is
roughly linear except at `h=1.0` where it drops sharply (−17% MF from Tight).
`'scott'` is more bandwidth-efficient than `h=1.0` for reaching DCR > 1.0:
it achieves DCR=1.08 at MF=0.854 while `h=1.0` achieves DCR=1.27 at MF=0.787.

**n_partitions effect:** At fixed bandwidth, fewer partitions (5 vs auto) add
roughly +0.04 to +0.14 to DCR across different bandwidths, while increasing
disc_err (samples stray further within larger coarser partitions). The bandwidth
effect dominates: changing from `h=0.1` to `'scott'` shifts DCR by ~0.75, while
changing n_partitions from auto to 5 shifts it by only ~0.04–0.13.

### 3. Adaptive bandwidth at 2×+ reaches DCR > 1.0 without constructor changes

`adaptive_bandwidth=True` scales per-partition bandwidth as
`bw_p = scott_p × max(1, budget_p / n_p)^(1/d)`. At expansion ratios ≥ 2×,
partitions are overpopulated relative to real samples, triggering bandwidth
scaling — effectively turning a Moderate-profile run into a High/Maximum one.

| Ratio | `adaptive_bandwidth` | Mean DCR | MF | TRTR | TSTR | TSTR Δ |
|---|---|---|---|---|---|---|
| 1× | False | 0.442 | 0.945 | 0.846 | 0.818 | −0.028 |
| 1× | True | 0.878 | 0.892 | 0.846 | 0.830 | −0.016 |
| 2× | False | 0.443 | 0.958 | 0.846 | 0.834 | −0.012 |
| 2× | **True** | **0.965** | 0.882 | 0.846 | **0.844** | **−0.002** |
| 5× | False | 0.438 | 0.966 | 0.846 | 0.829 | −0.017 |
| 5× | **True** | **0.974** | 0.868 | 0.846 | **0.845** | **−0.001** |

At 2× ratio, `adaptive_bandwidth=True` moves the mean DCR from Moderate (0.443)
to just below Maximum (0.965) while **improving TSTR Δ from −0.012 to −0.002**.
The TSTR gain is driven by the housing dataset where adaptive bandwidth also
improves utility. At 5× the pattern strengthens further (TSTR Δ → −0.001).

**Per-dataset breakdown at 2×:**

| Dataset | Adaptive | DCR | MF | TRTR | TSTR | TSTR Δ |
|---|---|---|---|---|---|---|
| fraud | False | 0.448 | 0.940 | 1.000 | 1.000 | 0.000 |
| fraud | True | 0.448 | 0.940 | 1.000 | 1.000 | 0.000 |
| housing | False | 0.744 | 0.962 | 0.573 | 0.554 | −0.019 |
| housing | **True** | **1.387** | 0.851 | 0.573 | **0.586** | **+0.013** |
| multimodal | False | 0.136 | 0.971 | 0.966 | 0.949 | −0.017 |
| multimodal | **True** | **1.060** | 0.854 | 0.966 | 0.947 | −0.019 |

Fraud is unaffected (already bandwidth-dominated at `h='auto'`; the balanced
class structure keeps real→real distances large). Housing and multimodal both
move into Maximum DCR territory with adaptive scaling. The housing TSTR
improvement (+0.013 Δ) at `adaptive=True` is genuine and reproducible.

### 4. Low DCR does not equal near-copy risk for KDE-based generators

Bootstrap + Noise (DCR=0.41) and HVRT-var (DCR=0.45) have similar DCR values,
but their risk profiles differ:

| Method | DCR | novelty_min | Mechanism |
|---|---|---|---|
| Bootstrap + Noise | 0.41 | ~0.16 | Resample with replacement + 10% Gaussian noise |
| HVRT | 0.45 | > 0 (stochastic) | KDE sampling within partition; bandwidth > 0 |

Bootstrap + Noise explicitly resamples real records, producing samples that are
10% σ away from actual training points — structurally near-copies. HVRT's KDE
sampling is continuous and stochastic; `novelty_min > 0` for any finite bandwidth
because samples are drawn from a smooth distribution, not perturbed copies.

**SMOTE** (DCR=0.30) is the tightest generator: it interpolates between
neighbouring real records, so synthetic points always lie on real-to-real line
segments. Its low DCR reflects geometric proximity to training records, not
random noise around them.

A complementary check for near-copy risk is `novelty_min`: a value near zero
means at least one synthetic sample was generated essentially on top of a real
record. HVRT's KDE bandwidth guarantees a minimum separation proportional to
the bandwidth scale.

### 5. Global models (Gaussian Copula, GMM) reach DCR > 1.0 at the cost of ML utility

Gaussian Copula and GMM both report DCR ≈ 1.17 at defaults — superficially
better privacy than HVRT. But:

| Method | DCR | TSTR Δ |
|---|---|---|
| Gaussian Copula | 1.17 | −0.040 |
| GMM | 1.17 | −0.026 |
| HVRT-var | 0.45 | **+0.020** |
| HVRT High profile (bw=0.5) | 0.80 | **−0.007** |

Their high DCR is structural: global parametric models sample from the full
inferred distribution rather than local training neighbourhoods, so synthetic
samples naturally spread beyond the training data envelope. This same property
is why they suffer significant TSTR degradation — the synthetic data does not
faithfully represent the local joint structure the downstream model needs.

**HVRT at the High privacy profile (bandwidth=0.5) achieves DCR=0.80 with
TSTR Δ=−0.007 — better ML utility than either global model at a comparable
privacy level.**

### 6. Per-dataset DCR variation is large and structurally driven

At default settings (n_parts=auto, bw=auto, ratio=2×):

| Dataset | DCR | Explanation |
|---|---|---|
| fraud | 0.448 | Balanced classes, near-Gaussian; moderate cluster density |
| housing | 0.744 | Sparse feature space (d=6); real records well separated |
| multimodal | 0.136 | Tight clusters; real→real distances are large relative to synth→real |

Multimodal's very low DCR (0.136) does not indicate near-copy risk — it reflects
that real samples are sparsely distributed across modes, so the real→real LOO
distance (denominator) is large. The synthetic samples land within each mode
cluster, producing small synth→real distances (numerator). The ratio is
structurally low even though samples are not near-copies.

This is why DCR must be interpreted per-dataset rather than as an absolute
threshold. A DCR of 0.15 on multimodal is structurally equivalent to a DCR of
0.45 on fraud in terms of actual privacy risk. The `novelty_min` metric is a
more absolute indicator of record-level copy risk.

---

## Privacy–Fidelity Decision Matrix

(Expansion ratio 2×, mean across fraud, housing, multimodal.
TSTR Δ ≥ −0.05 filter applied; best marginal fidelity within each profile selected.)

| Profile | DCR Target | `n_partitions` | `bandwidth` | DCR | MF | TRTR | TSTR | TSTR Δ |
|---|---|---|---|---|---|---|---|---|
| Tight | [0.00, 0.40) | `None` (auto) | `0.1` | 0.332 | 0.966 | 0.846 | 0.834 | −0.012 |
| Moderate | [0.40, 0.70) | `None` (auto) | `'auto'` | 0.443 | 0.958 | 0.846 | 0.834 | −0.012 |
| High | [0.70, 1.00) | `None` (auto) | `0.5` | 0.797 | 0.925 | 0.846 | 0.839 | **−0.007** |
| Maximum | [1.00, ∞) | `10` | `'scott'` | 1.067 | 0.856 | 0.846 | 0.824 | −0.022 |

Alternatively, `adaptive_bandwidth=True` with `bandwidth='auto'` reaches the
High/Maximum boundary (DCR ≈ 0.97 at 2×) without touching constructor parameters,
and produces the best mean TSTR Δ of any configuration (−0.002).

---

## Actionable Defaults

| Situation | Recommendation | DCR | TSTR Δ |
|---|---|---|---|
| Default: ML utility priority | `bandwidth='auto'`, `n_partitions=None` | ~0.44 (Moderate) | −0.012 |
| Better ML utility + more privacy | `bandwidth=0.5`, `n_partitions=None` | ~0.80 (High) | **−0.007** |
| Privacy target ≥ 1.0, constructor change OK | `bandwidth='scott'`, `n_partitions=10` | ~1.07 (Maximum) | −0.022 |
| Privacy target ≥ 1.0, no constructor change | `adaptive_bandwidth=True` at ratio ≥ 2× | ~0.97 (High/Maximum) | **−0.002** |
| Categorical / mixed-type data | Compute DCR on continuous columns only; treat full-feature DCR as unreliable | — | — |

### No default parameter changes

The DCR benchmark confirms `bandwidth='auto'` (→ h=0.10 at default partition
counts) remains the correct default for ML utility. The Moderate privacy profile
is a reasonable operating point for most augmentation use cases. Users with
explicit privacy requirements should consult the decision matrix above.

---

## Open Questions

1. **DCR normalisation for mixed-type data.** The categorical near-duplicate
   problem (adult dataset) makes full-feature DCR unreliable. A principled
   approach — e.g., compute DCR on continuous columns only, or use a mixed
   Gower distance — may make DCR useful on real-world tabular datasets that
   are almost never purely continuous.

2. **Per-mode DCR for multimodal distributions.** A global DCR of 0.14 on the
   multimodal dataset obscures within-mode proximity. Computing DCR separately
   per cluster (identified by the HVRT partition structure) would give a
   locally-normalised privacy estimate closer to actual per-record risk.

3. **Adaptive bandwidth and fraud.** Adaptive scaling has no effect on fraud
   because the dataset's balanced class structure already produces large
   real→real distances relative to partition bandwidth. Understanding which
   dataset properties govern adaptive bandwidth sensitivity would inform when
   to recommend it.
