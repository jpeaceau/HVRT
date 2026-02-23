# HVRT Bandwidth & Kernel Benchmark — Findings

**Date:** 2026-02-23
**Benchmark:** `benchmarks/bandwidth_benchmark.py`
**Model:** HVRT (pairwise interaction target, O(n·d²))
**Script:** `python benchmarks/bandwidth_benchmark.py`

---

## Setup

| Parameter | Value |
|---|---|
| Datasets | adult (class), fraud (class), housing (reg), multimodal (reg), emergence_divergence (reg), emergence_bifurcation (reg) |
| Expansion ratios | 2×, 5×, 10× |
| Max training samples | 500 per dataset |
| CV | 5-fold × 3-repeat = 15 evaluations per condition |
| Downstream model | GradientBoostingClassifier / Regressor |
| Total conditions | 6 datasets × 3 ratios = 18 conditions |

### Bandwidth candidates

| Candidate | Description |
|---|---|
| `scott` | Scott's rule: `n^(−1/(d+4))` |
| `silverman` | Silverman's rule (slightly wider than Scott) |
| `h=0.10 … h=2.00` | Absolute covariance scale factors (`kernel_cov = h² × data_cov`) |
| `epanechnikov` | Product Epanechnikov kernel, Ahrens-Dieter sampling, Scott's h per partition |
| `adaptive` | Per-partition Scott × `(budget/n_part)^(1/d)` |

### Metrics

| Metric | Interpretation | Direction |
|---|---|---|
| `disc_err` | `\|balanced_accuracy − 0.5\|` from LR discriminator | ↓ lower = harder to distinguish from real |
| `mw1` | Mean per-feature Wasserstein-1 (marginal fidelity) | ↓ lower |
| `corr_mae` | MAE of pairwise Pearson correlation matrix | ↓ lower |
| `tstr_delta` | TSTR − TRTR (GBM AUC or R²) | ↑ higher |

---

## Win-Count Summary (18 conditions)

| Method | disc_err ↓ | W1 ↓ | Corr ↓ | TSTR ↑ | **Total** |
|---|---|---|---|---|---|
| `h=0.10` | 1 | **15** | 6 | **8** | **30** |
| `epanechnikov` | **6** | 0 | 7 | 3 | **16** |
| `h=0.30` | 1 | 0 | 2 | 5 | **8** |
| `adaptive` | 0 | 3 | 3 | 2 | **8** |
| `h=2.00` | 7 | 0 | 0 | 0 | **7** |
| `h=0.50` | 2 | 0 | 0 | 0 | **2** |
| `h=1.50` | 1 | 0 | 0 | 0 | **1** |
| `scott` | 0 | 0 | 0 | 0 | **0** |
| `silverman` | 0 | 0 | 0 | 0 | **0** |

---

## Per-Dataset Normality

Within-partition normality was measured on z-scored continuous features.
Hypothesis: HVRT partitions are locally Gaussian, making Scott's rule and
Epanechnikov jointly AMISE-optimal.

| Dataset | Partitions | Mean size | Mean \|skew\| | Mean \|ex.kurt\| | Assessment |
|---|---|---|---|---|---|
| adult | 18 | 27.8 | **1.367 ± 1.801** | **5.171 ± 8.972** | Moderately non-Gaussian |
| fraud | 20 | 25.0 | 0.487 ± 0.414 | 0.789 ± 0.633 | Near-Gaussian |
| housing | 17 | 29.4 | 0.740 ± 0.678 | 1.335 ± 2.095 | Moderately non-Gaussian |
| multimodal | 16 | 31.2 | 0.531 ± 0.447 | 0.999 ± 1.079 | Mildly non-Gaussian |
| emergence_divergence | 17 | 29.4 | 0.642 ± 0.450 | 0.841 ± 0.738 | Moderately non-Gaussian |
| emergence_bifurcation | 17 | 29.4 | 0.642 ± 0.450 | 0.841 ± 0.738 | Moderately non-Gaussian |

**Conclusion:** The original hypothesis (HVRT partitions ≈ Gaussian) is partially
supported. Partitions are meaningfully more homogeneous than the global distribution,
but not Gaussian enough to make Scott's AMISE-optimal conditions hold. The high
adult |skewness| is driven by categorical-heavy features; the high |ex.kurt| reflects
fat-tailed outlier clusters not fully isolated by 18 partitions.

---

## Key Findings

### 1. Scott's rule and Silverman's rule are reliably suboptimal

Scott wins 0 conditions out of 18. Silverman wins 0. Both perform worse than
`h=0.10` on every metric in nearly every condition.

**Mechanism:** Scott's rule is AMISE-optimal for iid Gaussian data, but HVRT
partitions are (a) not Gaussian and (b) already stratified along the primary
variance axes by the decision tree. The tree has already captured the main
variance structure, so the residual within-partition variance is narrower than
Scott's formula assumes, leading to systematic over-smoothing.

### 2. h=0.10 is the strongest general-purpose bandwidth

`h=0.10` wins 15/18 conditions for Wasserstein-1 (marginal fidelity) and 8/18
for TSTR (ML utility). It is effectively "local bootstrap with minimal noise" —
tight enough to rely on partition boundaries for structure rather than kernel shape.

This degrades gracefully with partition heterogeneity: when a partition contains
heterogeneous sub-structure that broader kernels would smear, h=0.10 simply
perturbs existing points and avoids generating samples in distributional voids.

### 3. Wide bandwidths (≥ 0.75) are actively harmful for regression

On emergence_divergence and emergence_bifurcation (complex non-linear interaction
structure), Scott's rule TSTR penalty reaches −0.24 to −0.41 R² at 2×.
`h=0.10` reduces this to −0.02 to −0.07. The factor-of-10 gap shows wide
Gaussian KDE actively corrupts the conditional structure the downstream model
needs.

**The disc_err paradox:** `h=2.00` achieves the lowest disc_err on multimodal
regression (0.009 at 5×). But its TSTR Δ is −0.016 (harmful). Wide bandwidths
spread the marginal distribution to match real-data histograms while destroying
joint structure. The LR discriminator only sees marginals; it is fooled.
**TSTR is the reliable metric; disc_err alone is insufficient.**

### 4. Epanechnikov wins heterogeneous classification; `adaptive` wins near-Gaussian

**Adult (heterogeneous classification, |skew|=1.37):** Epanechnikov is the only
method achieving positive TSTR across all ratios (+0.009 / +0.011 / +0.017). All
Gaussian methods remain negative. Epanechnikov also dominates Corr.MAE by 4–5×.

**Fraud (near-Gaussian classification, |skew|=0.49):** The picture is
data-regime-specific by ratio:
- 2×: `h=0.10` wins TSTR (−0.013); Epanechnikov is worst Gaussian at −0.040.
- 5×: `adaptive` wins TSTR (−0.008); `h=0.10` close (−0.017); Epanechnikov −0.024.
- 10×: `adaptive` wins TSTR (+0.002, the only positive); Epanechnikov second (−0.006).

`adaptive` also wins W1 and Corr.MAE for fraud at 5× and 10×. On near-Gaussian
data, per-partition Scott bandwidth scaling (`adaptive`) and narrow Gaussian
(`h=0.10`) outperform the bounded Epanechnikov kernel. **The blanket rule
"use Epanechnikov for classification" applies only to heterogeneous data;
for near-Gaussian classification, `'auto'`/`h=0.10` or `adaptive` is preferred.**

**Housing (regression):** Epanechnikov Corr.MAE is excellent (0.022–0.010) but
TSTR degrades at 5×/10× (−0.049, −0.045). `h=0.30` wins TSTR at 5× (−0.001),
`h=0.10` wins at 10× (+0.004).

**Why:** The product Epanechnikov kernel samples each feature independently.
This is appropriate when the decision boundary is driven by partition membership
(classification) but breaks inter-feature correlations that regression targets depend on.
For near-Gaussian data the bounded support of Epanechnikov is also a disadvantage:
without heavy tails there is no mass to clip, and the strict cutoff under-covers
the tails, causing TSTR degradation at low-to-mid ratios.

### 5. The discriminator cannot differentiate method quality

All methods produce disc_err in 0.009–0.039 (1–4% from random chance). The LR
discriminator passes everything. It becomes mildly informative only at high ratios
(5×–10×), when the larger synthetic dataset allows detection of distributional tails
that Epanechnikov's bounded support eliminates. Do not use disc_err as the sole
quality metric.

### 6. Partition count is the primary lever for complex datasets

Emergence datasets (17 partitions, mean size 29.4, complex non-linear structure)
show the worst TSTR degradation and the largest method-to-method spread. The
partition boundaries are insufficiently fine-grained to achieve local homogeneity.

**Implication:** When synthetic data quality is poor, the first remediation step
is to increase partition count (`n_partitions`) rather than tuning bandwidth.
More partitions → smaller, more homogeneous partitions → narrower effective
bandwidth needed → all methods converge in quality.

---

## Selected Raw Results

### adult — 10× ratio

| Method | disc_err | W1 | Corr.MAE | TSTR Δ |
|---|---|---|---|---|
| `scott` | 0.0259 | 9883 | 0.0344 | −0.0038 |
| `h=0.10` | 0.0265 | **2422** | 0.0371 | −0.0071 |
| `epanechnikov` | **0.0182** | 12872 | **0.0099** | **+0.0167** |
| `h=2.00` | 0.0270 | 26093 | 0.0320 | −0.0252 |

### emergence_divergence — 5× ratio

| Method | disc_err | W1 | Corr.MAE | TSTR Δ |
|---|---|---|---|---|
| `scott` | 0.0208 | 0.116 | 0.0206 | −0.177 |
| `h=0.10` | 0.0254 | **0.024** | **0.015** | **−0.004** |
| `epanechnikov` | 0.0223 | 0.049 | 0.017 | −0.060 |
| `adaptive` | 0.0230 | 0.185 | 0.026 | −0.239 |

### fraud — 10× ratio

| Method | disc_err | W1 | Corr.MAE | TSTR Δ |
|---|---|---|---|---|
| `scott` | 0.0341 | 0.169 | 0.0573 | −0.0248 |
| `silverman` | 0.0347 | 0.154 | 0.0566 | −0.0243 |
| `h=0.10` | 0.0393 | 0.074 | 0.0577 | −0.0188 |
| `epanechnikov` | **0.0192** | 0.063 | 0.0268 | −0.0057 |
| `adaptive` | 0.0208 | **0.046** | **0.0236** | **+0.0019** |

---

## Actionable Defaults

### Changes made to the codebase

| Parameter | Old value | New value | Rationale |
|---|---|---|---|
| `bandwidth` constructor default | `0.5` | `'auto'` | Auto-selects h=0.10 (Gaussian) or Epanechnikov based on mean partition size vs dimensionality threshold |
| `generation_strategy='epanechnikov'` | Not available | Built-in | Vectorized Ahrens-Dieter; wins at fine partitions and on heterogeneous classification |

The `'auto'` bandwidth logic:
- Computes mean partition size at expand-time
- If `mean_part_size >= max(15, 2 * n_continuous_features)` → uses `h=0.10` Gaussian
- Otherwise → uses Epanechnikov
- Can be overridden with any explicit value: `expand(bandwidth=0.3)` or `expand(generation_strategy='epanechnikov')`

### No further default changes (re-run confirmation, 2026-02-23)

The bandwidth benchmark was re-run after the v2.1.2 auto-tuner formula change
(`min_samples_leaf = max(n_features+2, sqrt(n))`). The partition structure at
auto-tuned counts is the same as before: 16–20 partitions, mean sizes 25–31
samples, for all 6 datasets at `max_n=500`. The `'auto'` threshold condition
evaluates to True for all 6 datasets, so `'auto'` continues to select `h=0.10`
at default partition counts.

**`bandwidth='auto'` (→ `h=0.10`) remains the correct default.**

`h=0.10` wins 30 of 18 conditions across all metrics. `adaptive_bandwidth=True`
is competitive on **near-Gaussian data at high expansion ratios** (wins fraud at
5×/10×, including the only positive TSTR Δ on that dataset), but loses broadly
(8 wins total vs 30). It is not recommended as the default.

### When to deviate from defaults

| Situation | Recommendation | Evidence |
|---|---|---|
| Heterogeneous / high-skew classification (mean \|skew\| ≳ 0.8) | `generation_strategy='epanechnikov'` | Adult: only positive TSTR, all ratios |
| Near-Gaussian data, high expansion ratio (≥5×) | `adaptive_bandwidth=True` | Fraud 5×/10×: beats h=0.10 on TSTR and marginal fidelity |
| Fine manual partition count (≥50 partitions) | `generation_strategy='epanechnikov'` | Crossover confirmed in epanechnikov_partitions_benchmark |
| Regression, any partition count | `bandwidth='auto'` or explicit `h=0.10` / `h=0.30` | h=0.10 wins TSTR; h=0.30 edges ahead on housing/multimodal at 5×–10× |

---

## Epanechnikov × Partition-Count Benchmark

**Script:** `benchmarks/epanechnikov_partitions_benchmark.py`
**Hypothesis:** Finer partitions close the Epanechnikov regression gap.

### Setup

| Parameter | Value |
|---|---|
| Datasets | housing (d=6), multimodal (d=10), emergence_divergence (d=5), emergence_bifurcation (d=5) |
| Candidates | h=0.10, h=0.30, epanechnikov |
| Partition counts | auto, 30, 50, 75, 100, 150, 200 |
| Expansion ratios | 5×, 10× |
| CV | 5-fold × 3-repeat = 15 evaluations per condition |
| max_n | 500 training samples |

### Key Results

#### Housing (d=6) — TSTR Δ by partition count

| n_partitions | h=0.10 | h=0.30 | epanechnikov | Winner |
|---|---|---|---|---|
| auto (≈18) | −0.006 | **−0.000** | −0.018 | h=0.30 |
| 30 | −0.021 | −0.020 | −0.021 | h=0.30 |
| 50 | −0.036 | −0.029 | **−0.021** | Epan |
| 75 | −0.031 | −0.045 | **−0.006** | Epan |
| 100 | −0.048 | −0.054 | **−0.005** | Epan |
| 200 | −0.114 | −0.114 | **−0.004** | Epan |

Crossover: Epanechnikov wins from 50 partitions onward. At 200 partitions, the Gaussian TSTR penalty is 28× larger.

#### Multimodal (d=10) — TSTR Δ by partition count (10× ratio)

| n_partitions | h=0.10 | h=0.30 | epanechnikov | Winner |
|---|---|---|---|---|
| auto (≈18) | +0.001 | **+0.004** | −0.001 | h=0.30 |
| 30 | −0.009 | −0.006 | **−0.000** | Epan |
| 75 | −0.012 | −0.010 | **+0.000** | Epan |
| 150 | −0.014 | −0.014 | **+0.001** | Epan |
| 200 | −0.011 | −0.011 | **+0.001** | Epan |

Crossover earlier than housing (d=10 vs d=6). At 30+ partitions Epanechnikov leads; at ≥75 it achieves near-zero or positive TSTR.

#### Emergence Divergence (d=5, non-linear) — TSTR Δ (10× ratio)

| n_partitions | h=0.10 | h=0.30 | epanechnikov | Winner |
|---|---|---|---|---|
| auto (≈18) | **+0.007** | −0.043 | −0.048 | h=0.10 |
| 30 | −0.057 | −0.064 | **−0.012** | Epan |
| 75 | −0.060 | −0.086 | **−0.005** | Epan |
| 150 | −0.088 | −0.095 | **+0.004** | Epan |
| 200 | −0.088 | −0.093 | **+0.003** | Epan |

Large gap at auto: h=0.10 (+0.007) vs Epanechnikov (−0.048). At fine partitions, Epanechnikov achieves positive TSTR — better than TRTR — while Gaussian degrades to −0.09.

#### Emergence Bifurcation (d=5, hard structure) — TSTR Δ (10× ratio)

| n_partitions | h=0.10 | h=0.30 | epanechnikov | Winner |
|---|---|---|---|---|
| auto (≈18) | **−0.022** | −0.131 | −0.166 | h=0.10 |
| 50 | −0.220 | −0.333 | **−0.177** | Epan |
| 100 | −0.219 | −0.258 | **−0.132** | Epan |
| 200 | −0.154 | −0.151 | **−0.102** | Epan |

All methods remain significantly negative — the bifurcation structure is genuinely hard. Epanechnikov is most robust at fine partitions but the absolute TSTR values show this dataset's structure exceeds what any kernel can capture without additional partition density.

### Cross-Dataset Findings

**1. Hypothesis confirmed.** Finer partitions consistently close and eventually eliminate the Epanechnikov regression gap. The pattern holds across all 4 datasets, 2 ratios, and 7 partition counts.

**2. Dimensionality shifts the crossover earlier.**
- d=10 (multimodal): crossover at ~30 partitions
- d=6 (housing): crossover at ~50 partitions
- d=5 (emergence): crossover at ~30–50 partitions

Higher dimensionality makes Gaussian KDE degenerate faster (covariance matrix conditioning worsens as partitions shrink), while Epanechnikov is always covariance-free.

**3. 'auto' correctly selects h=0.10 at the default partition count.**
At auto (≈18 partitions), mean partition size ≈ 22–28 for max_n=500. The threshold `max(15, 2×d)` evaluates to 15–20 for d=5–10. All datasets have mean_part_size > threshold at auto, so 'auto' selects h=0.10 — which wins or ties on TSTR for 3 of 4 datasets at auto partition count.

**4. Epanechnikov always wins Corr.MAE.**
Across all 4 datasets, all partition counts, all ratios: Epanechnikov achieves lower correlation-structure MAE. The advantage is consistent (typically 0.010–0.035 vs 0.040–0.050 for Gaussian). The product kernel's independence assumption does not harm pairwise correlation *magnitude* preservation even while it breaks joint structure.

**5. Emergence bifurcation is a hard limit.**
The bifurcation dataset (same X → bimodal y) is the only case where increasing partitions alone cannot bring TSTR near zero. The best result at 200 partitions is Epanechnikov at −0.102. This dataset likely requires fundamentally different handling (e.g., conditional generation with y as explicit input, or class-conditional oversampling).

**6. Gaussian h=0.30 underperforms h=0.10 at fine partitions.**
At 30+ partitions on regression datasets, h=0.30 is uniformly worse than h=0.10. The wider bandwidth generates samples outside partition boundaries, re-introducing the inter-partition smearing that the tree structure was meant to prevent.

---

## Two-Phase Pipeline Benchmark

**Script:** `benchmarks/two_phase_pipeline_benchmark.py`
**Hypothesis:** Generating a large intermediate synthetic dataset with Gaussian h=0.10
(Phase 1), re-fitting HVRT on it to get ≥50 auto-tuned partitions, then running
Epanechnikov on those data-rich partitions (Phase 2) yields better TSTR than
single-phase approaches.

### Setup

| Parameter | Value |
|---|---|
| Datasets | housing (d=6), multimodal (d=10), emergence_divergence (d=5), emergence_bifurcation (d=5) |
| CV | 5-fold × 3-repeat = 15 evaluations per condition |
| max_n | 500 training samples |
| Ratios | 5×, 10× |
| Baselines | `baseline_auto` (bandwidth='auto'), `baseline_epan_50p` (epanechnikov, n_partitions=50) |
| Two-phase variants | 5×, 10×, 25× intermediate expansion before Phase 2 Epanechnikov |

**Expected Phase 2 partition counts** (n_tr ≈ 400, d=6):

| Phase 1 mult | n_intermediate | min_samples_leaf | expected n_parts |
|---|---|---|---|
| 5× | 2000 | 44 | ~39 |
| 10× | 4000 | 63 | ~55 |
| 25× | 10000 | 100 | ~88 |

### Results Summary — TSTR Δ (higher is better)

| Dataset | Ratio | baseline_auto | baseline_epan_50p | two_phase_5x | two_phase_10x | two_phase_25x | Winner |
|---|---|---|---|---|---|---|---|
| housing | 5× | **−0.0058** | −0.0214 | −0.0107 | −0.0218 | −0.0084 | baseline_auto |
| housing | 10× | **+0.0035** | −0.0075 | −0.0159 | −0.0187 | −0.0015 | baseline_auto |
| multimodal | 5× | **+0.0004** | −0.0010 | −0.0005 | −0.0001 | +0.0003 | baseline_auto |
| multimodal | 10× | +0.0006 | −0.0003 | +0.0001 | +0.0007 | **+0.0013** | two_phase_25x |
| emerg. div | 5× | **−0.0044** | −0.0378 | −0.0260 | −0.0216 | −0.0137 | baseline_auto |
| emerg. div | 10× | **+0.0071** | −0.0112 | −0.0121 | −0.0113 | +0.0058 | baseline_auto |
| emerg. bif | 5× | **−0.0454** | −0.1827 | −0.1621 | −0.1349 | −0.1063 | baseline_auto |
| emerg. bif | 10× | **−0.0217** | −0.1771 | −0.1343 | −0.1327 | −0.1070 | baseline_auto |

**Score: baseline_auto 7/8, two_phase_25x 1/8.**

### Key Findings

**1. Hypothesis disconfirmed for TSTR.** Single-phase `baseline_auto` (h=0.10 at ~18
auto-tuned partitions on real data) wins TSTR in 7 of 8 conditions. The two-phase
approach does not improve ML utility despite achieving substantially finer partitions
in Phase 2 (~88 vs 18).

**2. Within the two-phase family, more intermediate data is always better.**
As phase1_mult increases (5×→10×→25×), Phase 2 partition count grows (39→55→88)
and TSTR monotonically improves. At 25× intermediate, two-phase closes within ~0.01
of baseline_auto on most datasets. But it never surpasses it.

**3. Compounding distribution drift is the cause.**
Phase 1 Gaussian KDE introduces smoothing — the intermediate data is a slightly
blurred version of the real distribution. Phase 2 HVRT fits a tree on this
intermediate, so its partition boundaries reflect the smoothed distribution.
Epanechnikov then samples from within these drift-affected partitions. Each phase
adds drift from ground truth, and these errors compound.

In contrast, single-phase uses real data for HVRT fitting, so the partition boundaries
reflect the true distribution exactly. Only one step of drift (the KDE expansion)
occurs.

**4. Corr.MAE: two_phase_25x wins at 5×, baseline wins at 10×.**
Correlation structure preservation is mixed. The large intermediate (88 partitions,
~100 samples each) gives Epanechnikov better covariance stability than forced-50p on
real data. But `baseline_auto` at 10× (more synthetic data → stronger signal) ties or
beats everything on Corr.MAE.

**5. The forced n_partitions=50 on real data (baseline_epan_50p) is uniformly worst.**
Forcing 50 partitions on n=400 real samples gives ~8 samples/partition — too few for
any method. This confirms: the issue with Epanechnikov at small n was always
sample-starved partitions, not the kernel itself.

**6. The "right" partition count is set by the real data.**
Two-phase tried to manufacture fine partitions through synthetic expansion. But the
partition quality depends on the real data structure — you cannot synthesize your way
to better partition boundaries. More real data would help; more synthetic data (two-
phase) doesn't.

### Practical Implication

**Do not use two-phase pipelines** for TSTR improvement. The compounding drift
outweighs any benefit from finer partitions. Use `bandwidth='auto'` (single-phase)
which correctly selects the optimal kernel for the auto-tuned partition count on
real data.

If fine-partition Epanechnikov is desired, the correct path is **more real training
data**, not synthetic bootstrapping. With a larger real dataset, auto-tuning will
naturally produce finer partitions (e.g., n=5000 → ~70 partitions) and `'auto'` will
switch to Epanechnikov automatically.

---

## Open Questions

1. **Regression + Epanechnikov:** Can a multivariate (non-product) Epanechnikov
   kernel be implemented efficiently to preserve inter-feature correlations while
   retaining bounded support? A copula-based approach (rank-transform → Epanechnikov
   → back-transform) may combine the best of both kernels.

2. **Bifurcation and multi-modal targets:** For datasets where the same X region maps
   to multiple y values, expanding without conditioning on y loses this structure. A
   conditional generation mode (expanding separately per class/quantile of y) may be
   the correct approach for these cases.

3. **Adaptive Epanechnikov:** The current Epanechnikov uses Scott's formula for h.
   An adaptive variant — wider h for small partitions, narrower for large ones —
   may improve performance at the auto partition count where Epanechnikov currently
   under-explores relative to the narrow Gaussian. The two-phase benchmark suggests
   the issue isn't bandwidth width but partition structure derived from real data;
   so the payoff for adaptive h may be limited unless applied to larger real datasets.
