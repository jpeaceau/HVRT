# HVRT: Hierarchical Variance-Retaining Transformer

[![PyPI version](https://img.shields.io/pypi/v/hvrt.svg)](https://pypi.org/project/hvrt/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Variance-aware sample transformation for tabular data: reduce, expand, or augment.
Fits once; operates many times.

---

## Model Family

Five model classes share the same primary API. They differ in how they partition data
and which geometric statistic drives the tree.

| Class | Partitioning signal | Normalisation | Use-case |
|---|---|---|---|
| `HVRT` | Pairwise feature interactions (O(d²)) | Z-score (mean/std) | Reduction; when pairwise dependencies matter |
| `FastHVRT` | Z-score sum (O(d)) | Z-score (mean/std) | Expansion; same quality as HVRT at lower cost |
| `HART` | Pairwise interactions, MAD criterion | Z-score (median/MAD) | Heavy-tailed data; reduction with outliers |
| `FastHART` | Z-score sum, MAD criterion | Z-score (median/MAD) | Heavy-tailed expansion; best general performer |
| `PyramidHART` | ℓ₁ cooperation statistic A = \|S\| − ‖z‖₁ | Z-score (median/MAD) | Sign-structured data; polyhedral geometry |

```python
from hvrt import HVRT, FastHVRT, HART, FastHART, PyramidHART
```

---

## Algorithm

### 1. Z-score normalisation

```
X_z = (X − median) / MAD    (HART / FastHART / PyramidHART)
X_z = (X − mean)   / std    (HVRT / FastHVRT)
```

### 2. Synthetic target construction

**HVRT / HART** — sum of normalised pairwise feature interactions (T-statistic):
```
T = Σ_{i<j}  normalise(X_z[:,i] ⊙ X_z[:,j])    O(n · d²)
```

**FastHVRT / FastHART** — sum of z-scores (S-statistic):
```
S = Σ_j  X_z[:, j]    O(n · d)
```

**PyramidHART** — ℓ₁ cooperation statistic (A-statistic):
```
A = |S| − ‖z‖₁    bounded, sign-aware, O(n · d)
```

### 3. Partitioning

A `DecisionTreeRegressor` is fitted on the synthetic target. Leaves form variance-
homogeneous partitions. Tree depth and leaf size are auto-tuned to dataset size.

### 4. Per-partition operations

**Reduce:** Select representatives within each partition using the chosen
[selection strategy](#selection-strategies). Budget proportional to partition size
(`variance_weighted=False`) or biased toward high-variance partitions (`=True`).

**Expand:** Draw synthetic samples within each partition using the chosen
[generation strategy](#generation-strategies). Budget allocation follows the same logic.

---

## Installation

```bash
pip install hvrt
```

```bash
git clone https://github.com/jpeaceau/HVRT.git
cd HVRT
pip install -e .
```

Optional extras:

```bash
pip install hvrt[fast]       # Numba-compiled kernels (3–13× speedup on fit/FPS)
pip install hvrt[optimizer]  # Optuna-backed HPO (HVRTOptimizer)
pip install hvrt[benchmarks] # xgboost, matplotlib, pandas for benchmark scripts
```

---

## Quick Start

### HVRT / FastHVRT

```python
from hvrt import HVRT, FastHVRT

# Fit once — reduce and expand from the same model
model = HVRT(random_state=42).fit(X_train, y_train)   # y optional
X_reduced, idx = model.reduce(ratio=0.3, return_indices=True)
X_synthetic    = model.expand(n=50000)
X_augmented    = model.augment(n=15000)

# FastHVRT — O(n·d) target; preferred for expansion
model = FastHVRT(random_state=42).fit(X_train)
X_synthetic = model.expand(n=50000)
```

### PyramidHART with geometry-aware strategies

```python
from hvrt import PyramidHART, geometry_stats

# Fit on features only — sign structure lives in X space
model = PyramidHART(random_state=42).fit(X_train)

# Expand using A-range enforcement (polyhedral rejection sampling)
X_synth = model.expand(n=50000, generation_strategy='a_range_rejection')

# Inspect ℓ₁ geometry per partition
stats = model.geometry_stats()
# [{'partition_id': 0, 'n': 42, 'S_mean': 1.3, 'A_mean': -0.8, ...}, ...]

# Compute geometry statistics on any z-scored array
from hvrt import compute_A
A = compute_A(X_z)   # shape (n,): A-statistic per sample
```

---

## API Reference

### `HVRT` / `FastHVRT`

Both expose identical constructor parameters. `HVRT` uses pairwise interactions (O(d²));
`FastHVRT` uses z-score sum (O(d)).

```python
from hvrt import HVRT, FastHVRT

model = HVRT(
    n_partitions=None,           # Max tree leaves; auto-tuned if None
    min_samples_leaf=None,       # Min samples per leaf; auto-tuned if None
    max_depth=None,              # Tree max depth; auto-tuned if None
    y_weight=0.0,                # 0.0 = unsupervised; 1.0 = y drives splits
    bandwidth='auto',            # KDE bandwidth: 'auto', float, 'scott', 'silverman'
    auto_tune=True,
    n_jobs=1,                    # Parallelism: -1 = all cores
    tree_splitter='best',        # 'best' or 'random' (10–50× faster fit)
    random_state=42,
    # Pipeline params (see Pipeline section)
    reduce_params=None,
    expand_params=None,
    augment_params=None,
)
```

### `HART` / `FastHART`

All constructor parameters identical to HVRT/FastHVRT. Differ in normalisation
(median/MAD instead of mean/std) and tree criterion (absolute_error instead of
squared_error). Robust to heavy tails and outliers.

```python
from hvrt import HART, FastHART

model = HART(random_state=42).fit(X_train, y_train)
model = FastHART(random_state=42).fit(X_train)
```

### `PyramidHART`

Extends HART. Uses the ℓ₁ cooperation statistic A = |S| − ‖z‖₁ as the partitioning
target. Exposes `geometry_stats()` for per-partition breakdown of S, Q, T, and A.

```python
from hvrt import PyramidHART

model = PyramidHART(random_state=42).fit(X_train)
stats = model.geometry_stats()   # list of per-partition geometry dicts
```

### `HVRTOptimizer`

Requires: `pip install hvrt[optimizer]`

```python
from hvrt import HVRTOptimizer

opt = HVRTOptimizer(
    n_trials=30,             # Optuna trials; use ≥50 in production
    n_jobs=1,                # Parallel trials (-1 = all cores)
    cv=3,                    # Cross-validation folds for the objective
    expansion_ratio=5.0,     # Synthetic-to-real ratio during evaluation
    task='auto',             # 'auto', 'regression', 'classification'
    timeout=None,            # Wall-clock time limit in seconds
    random_state=None,
    verbose=0,               # 0 = silent, 1 = Optuna trial progress
)
opt = opt.fit(X, y)          # y enables TSTR Δ objective; required for classification
```

Performs TPE-based Bayesian optimisation over `n_partitions`, `min_samples_leaf`,
`y_weight`, kernel / bandwidth, and `variance_weighted`. HVRT defaults are always
evaluated as trial 0 (warm start).

**Post-fit attributes:**

| Attribute | Type | Description |
|---|---|---|
| `best_score_` | float | Best mean TSTR Δ across CV folds |
| `best_params_` | dict | Best constructor kwargs |
| `best_expand_params_` | dict | Best expand kwargs |
| `best_model_` | HVRT | Refitted on full dataset |
| `study_` | optuna.Study | Full Optuna study |

```python
opt = HVRTOptimizer(n_trials=50, n_jobs=4, cv=3, random_state=42).fit(X, y)
print(f'Best TSTR Δ: {opt.best_score_:+.4f}')
X_synth = opt.expand(n=50000)         # y column stripped automatically
X_aug   = opt.augment(n=len(X) * 5)
```

### `fit`

```python
model.fit(X, y=None)
```

### `reduce`

```python
X_reduced = model.reduce(
    n=None,                  # Absolute target count
    ratio=None,              # Proportional (e.g. 0.3 = keep 30%)
    method='fps',            # Selection strategy; see Selection Strategies
    variance_weighted=True,  # Oversample high-variance partitions
    return_indices=False,
    n_partitions=None,       # Override tree granularity for this call only
)
```

### `expand`

```python
X_synth = model.expand(
    n=10000,
    variance_weighted=False,      # True = oversample tails
    bandwidth=None,               # Override instance bandwidth
    adaptive_bandwidth=False,     # Scale bandwidth with local expansion ratio
    generation_strategy=None,     # See Generation Strategies
    return_novelty_stats=False,
    n_partitions=None,
)
```

### `augment`

```python
X_aug = model.augment(n=15000, variance_weighted=False)
# n must exceed len(X); returns original X concatenated with synthetic samples
```

### Utility methods

```python
partitions = model.get_partitions()
# [{'id': 5, 'size': 120, 'mean_abs_z': 0.84, 'variance': 1.2}, ...]

novelty = model.compute_novelty(X_new)   # min z-space distance per point

params = HVRT.recommend_params(X)        # {'n_partitions': 180, ...}

# PyramidHART only
stats = model.geometry_stats()
# [{'partition_id': 0, 'n': 42, 'S_mean': 1.3, 'Q_mean': 0.5, 'T_mean': 1.1,
#   'A_mean': -0.8, 'mst_mean': 0.4, 'A_q05': -2.1, 'A_q95': 0.3}, ...]
```

---

## sklearn Pipeline

Operation parameters are declared at construction time via `ReduceParams`, `ExpandParams`,
or `AugmentParams`. The tree is fitted once during `fit()`; `transform()` calls the
corresponding operation.

```python
from hvrt import HVRT, FastHVRT, ReduceParams, ExpandParams, AugmentParams
from sklearn.pipeline import Pipeline

# Reduce
pipe = Pipeline([('hvrt', HVRT(reduce_params=ReduceParams(ratio=0.3)))])
X_red = pipe.fit_transform(X, y)

# Expand
pipe = Pipeline([('hvrt', FastHVRT(expand_params=ExpandParams(n=50000)))])
X_synth = pipe.fit_transform(X)

# Augment
pipe = Pipeline([('hvrt', HVRT(augment_params=AugmentParams(n=15000)))])
X_aug = pipe.fit_transform(X)
```

Import from `hvrt.pipeline` to make intent explicit:

```python
from hvrt.pipeline import HVRT, ReduceParams
```

### ReduceParams

```python
ReduceParams(
    n=None,
    ratio=None,              # e.g. 0.3
    method='fps',
    variance_weighted=True,
    return_indices=False,
    n_partitions=None,
)
```

### ExpandParams

```python
ExpandParams(
    n=50000,                 # required
    variance_weighted=False,
    bandwidth=None,
    adaptive_bandwidth=False,
    generation_strategy=None,
    return_novelty_stats=False,
    n_partitions=None,
)
```

### AugmentParams

```python
AugmentParams(
    n=15000,                 # required; must exceed len(X)
    variance_weighted=False,
    n_partitions=None,
)
```

---

## Generation Strategies

Seven built-in strategies: four general-purpose and three PyramidHART-specific.

| Strategy | Behaviour | Notes |
|---|---|---|
| `'multivariate_kde'` | Gaussian KDE via batch Cholesky. Full joint covariance. | Default at large partitions |
| `'epanechnikov'` | Product Epanechnikov kernel, Ahrens-Dieter sampling. Bounded support. | Recommended for classification; ≥5× ratios |
| `'bootstrap_noise'` | Resample with replacement + 10% Gaussian noise. | Fastest; no distributional assumptions |
| `'univariate_kde_copula'` | Per-feature 1-D KDE marginals + Gaussian copula. | Flexible per-feature marginals |
| `'a_range_rejection'` | Rejection-sampling: accepts only samples within per-partition A-value quantile bounds. Falls back to training point after `max_iter` rounds. | **PyramidHART only** — X-only fit; best for polyhedral constraint enforcement |
| `'sign_preserving_epanechnikov'` | Epanechnikov noise on feature magnitudes only; original z-signs restored. Samples never cross coordinate hyperplanes. | **PyramidHART only** — X-only fit; sign-coherent generation |
| `'minority_sign_resampler'` | Bootstraps target MST (−A/2) from training partition; scales minority-sign group to match; Gaussian noise on majority group. | **PyramidHART only** — X-only fit; MST-matching generation |

```python
from hvrt import FastHVRT, epanechnikov, univariate_kde_copula

model = FastHVRT(random_state=42).fit(X)

# By name (preferred)
X_synth = model.expand(n=10000, generation_strategy='epanechnikov')

# By reference
X_synth = model.expand(n=10000, generation_strategy=univariate_kde_copula)
```

### PyramidHART-specific strategies

These strategies encode assumptions about the ℓ₁ polyhedral geometry of PyramidHART:
the cooperation statistic A = |S| − ‖z‖₁ partitions feature space into sign-coherent
cones, and these strategies are designed to preserve or restore that structure.

**Important:** Fit on `X` only (no y column stacked). Stacking y introduces a sign
dimension unrelated to the geometric construction.

```python
from hvrt import PyramidHART

model = PyramidHART(random_state=42).fit(X_train)  # X only

# A-range enforcement: reject samples outside training A-quantile range
X_synth = model.expand(n=50000, generation_strategy='a_range_rejection')

# Sign-preserving: Epanechnikov on magnitudes, original signs restored
X_synth = model.expand(n=50000, generation_strategy='sign_preserving_epanechnikov')

# MST-matching: bootstrap minority-sign group to match training MST
X_synth = model.expand(n=50000, generation_strategy='minority_sign_resampler')
```

All three strategies produce the same results as standard Epanechnikov at the default
auto-tuned partition granularity (n≈500, ~18–20 leaves). Their geometric advantages
emerge at larger n and finer partitions (n_partitions≥50) where sign structure in A
is more pronounced.

### Custom strategy

```python
from hvrt import StatefulGenerationStrategy, PartitionContext
import numpy as np

class MyStrategy:
    def prepare(self, X_z, partition_ids, unique_partitions):
        # precompute partition metadata once; return a PartitionContext subclass
        ...
        return PartitionContext(X_z=X_z, ...)

    def generate(self, context, budgets, random_state):
        ...
        return X_synthetic   # shape (sum(budgets), n_features), z-score space

X_synth = model.expand(n=10000, generation_strategy=MyStrategy())
```

```python
from hvrt import BUILTIN_GENERATION_STRATEGIES
list(BUILTIN_GENERATION_STRATEGIES)
# ['multivariate_kde', 'univariate_kde_copula', 'bootstrap_noise', 'epanechnikov',
#  'a_range_rejection', 'sign_preserving_epanechnikov', 'minority_sign_resampler']
```

---

## Selection Strategies

```python
from hvrt import HVRT

model = HVRT(random_state=42).fit(X, y)

# By name (preferred)
X_red = model.reduce(ratio=0.2, method='fps')             # default
X_red = model.reduce(ratio=0.2, method='medoid_fps')
X_red = model.reduce(ratio=0.2, method='variance_ordered')
X_red = model.reduce(ratio=0.2, method='stratified')
```

| Strategy | Behaviour | Notes |
|---|---|---|
| `'fps'` / `'centroid_fps'` | Greedy FPS seeded at partition centroid. **Default.** | Best general-purpose diversity |
| `'medoid_fps'` | FPS seeded at partition medoid. | Robust to outliers; slightly slower |
| `'variance_ordered'` | Highest local k-NN variance (k=10). | **23–37× faster** with `n_jobs=-1` at large n |
| `'stratified'` | Fully-vectorised random sample. | **2.5–3× faster** than loop; best for repeated `reduce()` |

### Custom strategy

```python
from hvrt import StatefulSelectionStrategy, SelectionContext
import numpy as np

class MySelector:
    def prepare(self, X_z, partition_ids, unique_partitions):
        from hvrt.reduction_strategies import _build_selection_context
        return _build_selection_context(X_z, partition_ids, unique_partitions)

    def select(self, context, budgets, random_state, n_jobs=1):
        ...
        return selected_indices   # global indices into X_z

X_red = model.reduce(ratio=0.2, method=MySelector())
```

**Memory-conscious large-data workflow:**

```python
model = HVRT(n_jobs=-1).fit(X_large)          # n_jobs forwarded to select()
X_red = model.reduce(ratio=0.1, method='fps') # parallel FPS, O(partition size) memory per worker
```

---

## Cooperative Geometry

The `_geometry.py` module provides standalone functions for computing ℓ₁ cooperation
statistics. These are useful for model selection, diagnostics, and custom analysis.

### Definitions

| Symbol | Name | Formula |
|---|---|---|
| S | Sign sum | Σ_j z_j (z-score sum; FastHVRT target) |
| Q | Quadrature | ‖z‖₂² = Σ_j z_j² |
| T | Cooperation | S² − Q = (Σ z_j)² − Σ z_j² |
| A | ℓ₁ cooperation | \|S\| − ‖z‖₁ = \|Σ z_j\| − Σ \|z_j\| (PyramidHART target) |
| MST | Minority-sign total | −A/2 = count of features with sign opposite to the majority |

### Usage

```python
from hvrt import compute_A, geometry_stats

# Compute A-statistic on z-scored feature matrix
import numpy as np
X_z = (X - X.mean(0)) / X.std(0)
A = compute_A(X_z)   # shape (n,); A ∈ [−d/2, 0]

# Full geometry stats (S, Q, T, A) per sample
from hvrt._geometry import compute_S, compute_Q, compute_T, minority_sign_total
S = compute_S(X_z)   # shape (n,)
Q = compute_Q(X_z)   # shape (n,)
T = compute_T(X_z)   # S² − Q, shape (n,)
mst = minority_sign_total(X_z)   # shape (n,)

# Per-partition breakdown from a fitted PyramidHART model
model = PyramidHART().fit(X_train)
stats = model.geometry_stats()
# [{'partition_id': 0, 'n': 42, 'S_mean': 1.3, 'A_mean': -0.8, ...}, ...]
```

### When to use each model

| Question | Recommendation |
|---|---|
| General-purpose reduction (keep diversity) | `HVRT` — pairwise T captures interactions |
| General-purpose expansion (generate synthetic data) | `FastHVRT` — O(d) target, same quality |
| Data with heavy tails or outliers | `HART` / `FastHART` — MAD normalisation is robust |
| Sign structure matters (financial, directional data) | `PyramidHART` — A-statistic partitions sign cones |
| Need per-partition geometry diagnostics | `PyramidHART.geometry_stats()` |

---

## Recommendations

### `bandwidth='auto'` — the default

`bandwidth='auto'` requires no tuning for most datasets. At each `expand()` call it
inspects the fitted partition structure and picks the kernel most likely to produce
high-quality synthetic data.

**How it decides:**

| Condition | Chosen kernel | Reason |
|---|---|---|
| mean partition size **≥** `max(15, 2 × d)` | Narrow Gaussian `h=0.1` | Enough samples for stable covariance estimation |
| mean partition size **<** `max(15, 2 × d)` | Epanechnikov product kernel | Too few samples for covariance; product kernel is covariance-free |

**Why not Scott's rule:** Scott's rule assumes iid Gaussian data. HVRT partitions are
locally homogeneous but non-Gaussian (mean |skewness| 0.49–1.37 across benchmark
datasets). The decision tree already captures the primary variance structure, so the
residual within-partition variance is narrower than Scott's formula assumes, causing
systematic over-smoothing. Scott's rule won **0 of 18** benchmark conditions.

**When to override:**

- **Heterogeneous / high-skew classification** (mean |skew| ≳ 0.8): use
  `generation_strategy='epanechnikov'` directly.
- **Small dataset, coarse partitions, regression:** use `bandwidth=0.1` or `bandwidth=0.3`.

### Model selection guidance

| Scenario | Recommended model | Recommended strategy |
|---|---|---|
| Reduction from large dataset | `HVRT` | `method='fps'` (default) |
| Reduction, rare events | `HVRT` | `method='fps'`, `variance_weighted=True`, `y_weight=0.3` |
| Expansion, general purpose | `FastHVRT` or `FastHART` | `'epanechnikov'` (classification), default (regression) |
| Data with outliers / heavy tails | `HART` / `FastHART` | any strategy |
| Sign-structured data | `PyramidHART` | `'a_range_rejection'` (large n), `'sign_preserving_epanechnikov'` (general) |
| HPO | Any model | `HVRTOptimizer` (requires `[optimizer]`) |

### Hyperparameter optimisation (HPO)

Dataset heterogeneity is the primary driver of sensitivity to HVRT's parameters.
A well-behaved near-Gaussian dataset produces good synthetic data at defaults. A
dataset with distinct clusters or regime-switching needs finer partitions.

**Parameter search space:**

| Parameter | Default | Effect |
|---|---|---|
| `n_partitions` | auto | **Primary lever.** More partitions → finer local homogeneity |
| `bandwidth` | `'auto'` | `'auto'` is near-optimal once partition count is right |
| `variance_weighted` | `False` | `True` oversamples high-variance partitions; useful for tail-heavy distributions |
| `y_weight` | 0.0 | Weights y in the synthetic target; helps when y governs sub-populations |

**When HPO is worth running:**

- TSTR Δ is significantly negative (below −0.05)
- Dataset has known sub-populations, clusters, or regime changes
- Generating at high ratio (10×+)

```python
from hvrt import HVRTOptimizer

opt = HVRTOptimizer(n_trials=50, n_jobs=4, cv=3, random_state=42).fit(X, y)
print(f'Best TSTR Δ: {opt.best_score_:+.4f}')
X_synth = opt.expand(n=50000)
X_aug   = opt.augment(n=len(X) * 5)
```

---

## Benchmarks

### Sample reduction

Metric: GBM ROC-AUC on reduced training set as % of full-training-set AUC.
n=3 000 train / 2 000 test, seed=42.

| Scenario | Retention | HVRT-fps | HVRT-yw | Random | Stratified |
|---|---|---|---|---|---|
| Well-behaved (Gaussian, no noise) | 10% | 97.1% | 98.1% | 96.9% | 98.0% |
| Well-behaved (Gaussian, no noise) | 20% | 98.7% | 98.9% | 98.3% | 99.0% |
| Noisy labels (20% random flip) | 10% | **96.1%** | 91.1% | 93.3% | 90.4% |
| Noisy labels (20% random flip) | 20% | **95.2%** | 95.9% | 93.1% | 93.1% |
| Heavy-tail + label noise + junk features | 30% | **98.2%** | 98.2% | 94.3% | 95.2% |
| Rare events (5% positive class) | 10% | 98.0% | **99.4%** | 86.5% | 94.1% |
| Rare events (5% positive class) | 20% | 98.0% | **100.4%** | 97.9% | 99.0% |

*HVRT-fps: `method='fps'`, `variance_weighted=True`. HVRT-yw: same + `y_weight=0.3`.*

Reproduce: `python benchmarks/reduction_denoising_benchmark.py`

### Synthetic data expansion

Metrics: discriminator accuracy (target ≈ 50%), marginal fidelity, tail preservation
(target = 1.0), **Privacy DCR**, TSTR Δ.  `bandwidth='auto'`, `max_n=500` training
samples, expansion ratio 1×. Mean across continuous benchmark datasets (fraud, housing,
multimodal).

| Method | Marginal Fidelity | Disc. Err % ↓ | Tail Preservation | Privacy DCR | TRTR | TSTR | TSTR Δ | Fit time |
|---|---|---|---|---|---|---|---|---|
| **HVRT-size** | 0.930 | 1.4 | 1.055 | 0.65 | 0.966 | 0.951 | −0.016 | 0.001 s |
| **FastHVRT-var** | 0.885 | 5.6 | 1.034 | 0.63 | 0.966 | **0.961** | **−0.004** | 0.001 s |
| FastHVRT-size | 0.949 | 1.1 | 1.055 | 0.66 | 0.966 | 0.957 | −0.009 | 0.001 s |
| **HART-size** | 0.930 | 1.4 | 1.066 | 0.79 | 0.966 | 0.950 | −0.016 | 0.001 s |
| **FastHART-size** | 0.930 | 1.4 | 1.051 | 0.80 | 0.966 | 0.950 | −0.016 | 0.001 s |
| PyramidHART-size | 0.930 | 1.4 | 1.066 | 0.79 | 0.966 | 0.950 | −0.016 | 0.001 s |
| Gaussian Copula | 0.964 | 0.6 | 0.991 | 1.60 | 0.966 | 0.941 | −0.024 | 0.004 s |
| GMM (k≤20) | 0.895 | 2.5 | 1.030 | 1.19 | 0.966 | 0.949 | −0.016 | 0.004 s |
| Bootstrap + Noise | 0.952 | **0.0** | 1.009 | 0.27 | 0.966 | 0.944 | −0.022 | 0.000 s |
| TabDDPM§ | 0.960 | 2.0 | — | N/A | — | — | — | 120 s |
| MOSTLY AI§ | 0.975 | 1.0 | — | N/A | — | — | — | 60 s |

*† PyramidHART-ARejection uses X-only fit + proxy y — correct evaluation for geometry-aware strategies.*
*‡ CTGAN/TVAE run locally (`--deep-learning`). Poor Disc. Err reflects small n=400 training set.*
*§ Published numbers only — no local runner.*
*\* CTGAN/TVAE TRTR is 0.769 (housing + multimodal only; fraud not evaluated).*
*Disc. Err = |discriminator accuracy − 50%|. Lower = more indistinguishable from real.*

Reproduce: `python benchmarks/run_benchmarks.py --tasks expand --deep-learning`

### Privacy evaluation

The benchmark suite computes two privacy metrics for every expansion run.

**Distance-to-Closest-Record (DCR)**

```
DCR = median(min_dist(synthetic_i → real))
    / median(min_dist(real_i → real excluding itself))
```

| DCR range | Interpretation |
|---|---|
| < 0.1 | Near-copies: high record-linkage risk |
| 0.1 – 0.8 | Tight generation: fits local distribution well; low risk |
| ≈ 1.0 | Neutral: synthetic at typical real-to-real distances |
| > 1.0 | Spread: samples more dispersed than real data |

HVRT (DCR ≈ 0.45) is 3× safer than Bootstrap + Noise (DCR ≈ 0.16) and 1.5× safer
than SMOTE (DCR ≈ 0.30) on continuous data.

**Privacy–Fidelity Decision Matrix**

| Privacy Profile | DCR Target | `bandwidth` | DCR | Marginal Fidelity | TSTR Δ |
|---|---|---|---|---|---|
| Tight | [0.00, 0.40) | `0.1` | 0.332 | 0.966 | −0.012 |
| Moderate | [0.40, 0.70) | `'auto'` | 0.443 | 0.958 | −0.012 |
| High | [0.70, 1.00) | `0.5` | 0.797 | 0.925 | −0.007 |
| Maximum | [1.00, ∞) | `'scott'` + `n_partitions=10` | 1.067 | 0.856 | −0.022 |

*Reproduce: `python benchmarks/dcr_privacy_benchmark.py`*

---

## Benchmarking Scripts

```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --tasks reduce --datasets adult housing
python benchmarks/run_benchmarks.py --tasks expand
python benchmarks/pyramid_hart_benchmark.py          # PyramidHART vs HART/HVRT family
python benchmarks/pyramid_hart_benchmark.py --quick  # 2 datasets, fast check
python benchmarks/strategy_speedup_benchmark.py      # vectorization speedup
python benchmarks/speed_benchmark.py                 # serial vs parallel wall-clock
python benchmarks/reduction_denoising_benchmark.py
python benchmarks/hpo_benchmark.py                  # requires: pip install hvrt[optimizer]
python benchmarks/dcr_privacy_benchmark.py           # privacy–fidelity sweep
```

---

## Testing

```bash
pytest
pytest --cov=hvrt --cov-report=term-missing
```

---

## Citation

```bibtex
@software{hvrt2026,
  author = {Peace, Jake},
  title  = {HVRT: Hierarchical Variance-Retaining Transformer},
  year   = {2026},
  url    = {https://github.com/jpeaceau/HVRT}
}
```

---

## License

GNU Affero General Public License v3 or later (AGPL-3.0-or-later) — see [LICENSE](LICENSE).

## Acknowledgments

Development assisted by Claude (Anthropic).
