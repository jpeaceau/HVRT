# HVRT: Hierarchical Variance-Retaining Transformer

[![PyPI version](https://img.shields.io/pypi/v/hvrt.svg)](https://pypi.org/project/hvrt/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Variance-aware sample transformation for tabular data: reduce, expand, or augment.

---

## Overview

HVRT partitions a dataset into variance-homogeneous regions via a decision tree fitted on a synthetic extremeness target, then applies a configurable per-partition operation (selection for reduction, sampling for expansion). The tree is fitted once; `reduce()`, `expand()`, and `augment()` all draw from the same fitted model.

| Operation | Method | Description |
|---|---|---|
| **Reduce** | `model.reduce(ratio=0.3)` | Select a geometrically diverse representative subset |
| **Expand** | `model.expand(n=50000)` | Generate synthetic samples via per-partition KDE or other strategy |
| **Augment** | `model.augment(n=15000)` | Concatenate original data with synthetic samples |

---

## Algorithm

### 1. Z-score normalisation

```
X_z = (X - μ) / σ   per feature
```

Categorical features are integer-encoded then z-scored.

### 2. Synthetic target construction

**HVRT** — sum of normalised pairwise feature interactions:
```
For all feature pairs (i, j):
  interaction = X_z[:,i] ⊙ X_z[:,j]
  normalised  = (interaction - mean) / std
target = sum of all normalised interaction columns        O(n · d²)
```

**FastHVRT** — sum of z-scores per sample:
```
target_i = Σ_j  X_z[i, j]                               O(n · d)
```

### 3. Partitioning

A `DecisionTreeRegressor` is fitted on the synthetic target. Leaves form variance-homogeneous partitions. Tree depth and leaf size are auto-tuned to dataset size.

### 4. Per-partition operations

**Reduce:** Select representatives within each partition using the chosen [selection strategy](#selection-strategies). Budget is proportional to partition size (`variance_weighted=False`) or biased toward high-variance partitions (`variance_weighted=True`).

**Expand:** Draw synthetic samples within each partition using the chosen [generation strategy](#generation-strategies). Budget allocation follows the same logic.

---

## Installation

```bash
pip install hvrt
```

```bash
git clone https://github.com/hotprotato/hvrt.git
cd hvrt
pip install -e .
```

---

## Quick Start

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

---

## API Reference

### `HVRT`

```python
from hvrt import HVRT

model = HVRT(
    n_partitions=None,           # Max tree leaves; auto-tuned if None
    min_samples_leaf=None,       # Min samples per leaf; auto-tuned if None
    y_weight=0.0,                # 0.0 = unsupervised; 1.0 = y drives splits
    bandwidth='auto',            # KDE bandwidth: 'auto' (default), float, 'scott', 'silverman'
    auto_tune=True,
    random_state=42,
    # Pipeline params (see Pipeline section)
    reduce_params=None,
    expand_params=None,
    augment_params=None,
)
```

Target: sum of normalised pairwise feature interactions. O(n · d²). Preferred for reduction.

### `FastHVRT`

```python
from hvrt import FastHVRT

model = FastHVRT(bandwidth='auto', random_state=42)
```

Target: sum of z-scores. O(n · d). Equivalent quality to HVRT for expansion. All constructor parameters identical to HVRT.

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
`y_weight`, kernel / bandwidth, and `variance_weighted`. The HVRT defaults are always
evaluated as trial 0 (warm start), so HPO can only match or improve on the baseline.

**Post-fit attributes:**

| Attribute | Type | Description |
|---|---|---|
| `best_score_` | float | Best mean TSTR Δ across CV folds |
| `best_params_` | dict | Best constructor kwargs (`n_partitions`, `min_samples_leaf`, `y_weight`, `bandwidth`) |
| `best_expand_params_` | dict | Best expand kwargs (`variance_weighted`, optionally `generation_strategy`) |
| `best_model_` | HVRT | Refitted on the full dataset using `best_params_` |
| `study_` | optuna.Study | Full Optuna study for visualisation and diagnostics |

**After fitting:**

```python
opt = HVRTOptimizer(n_trials=50, n_jobs=4, cv=3, random_state=42).fit(X, y)
print(f'Best TSTR Δ: {opt.best_score_:+.4f}')
print(f'Best params: {opt.best_params_}')

X_synth = opt.expand(n=50000)         # y column stripped automatically
X_aug   = opt.augment(n=len(X) * 5)   # originals + synthetic
```

`expand()` and `augment()` strip the appended y column, returning arrays with the same
number of columns as the training X.

### `fit`

```python
model.fit(X, y=None, feature_types=None)
# feature_types: list of 'continuous' or 'categorical' per column
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
    bandwidth=None,               # Override instance bandwidth; accepts float, 'auto', 'scott'
    adaptive_bandwidth=False,     # Scale bandwidth with local expansion ratio
    generation_strategy=None,     # See Generation Strategies
    return_novelty_stats=False,
    n_partitions=None,
)
```

`adaptive_bandwidth=True` uses per-partition bandwidth `bw_p = scott_p × max(1, budget_p/n_p)^(1/d)`.

### `augment`

```python
X_aug = model.augment(n=15000, variance_weighted=False)
# n must exceed len(X); returns original X concatenated with (n - len(X)) synthetic samples
```

### Utility methods

```python
partitions = model.get_partitions()
# [{'id': 5, 'size': 120, 'mean_abs_z': 0.84, 'variance': 1.2}, ...]

novelty = model.compute_novelty(X_new)   # min z-space distance per point

params = HVRT.recommend_params(X)        # {'n_partitions': 180, ...}
```

---

## sklearn Pipeline

Operation parameters are declared at construction time via `ReduceParams`, `ExpandParams`, or `AugmentParams`. The tree is fitted once during `fit()`; `transform()` calls the corresponding operation.

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

Alternatively, import from `hvrt.pipeline` to make the intent explicit:

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

```python
from hvrt import FastHVRT, epanechnikov, univariate_kde_copula

model = FastHVRT(random_state=42).fit(X)

# By name (preferred)
X_synth = model.expand(n=10000, generation_strategy='epanechnikov')

# By reference
X_synth = model.expand(n=10000, generation_strategy=univariate_kde_copula)

# Custom strategy — implement StatefulGenerationStrategy
from hvrt import StatefulGenerationStrategy, PartitionContext
import numpy as np

class MyStrategy:
    def prepare(self, X_z, partition_ids, unique_partitions):
        # precompute partition metadata once
        ...
        return PartitionContext(X_z=X_z, ...)   # or a PartitionContext subclass

    def generate(self, context, budgets, random_state):
        ...
        return X_synthetic   # shape (sum(budgets), n_features), z-score space

X_synth = model.expand(n=10000, generation_strategy=MyStrategy())
```

| Strategy | Behaviour | Throughput (5K→25K, d=10) | Notes |
|---|---|---|---|
| `'multivariate_kde'` | Gaussian KDE via batch Cholesky (pure NumPy). Captures full joint covariance. | 2.3M samples/s | Default when partitions are large |
| `'epanechnikov'` | Product Epanechnikov kernel, Ahrens-Dieter sampling. Bounded support. | 2.6M samples/s | Recommended for classification; ≥5× ratios |
| `'bootstrap_noise'` | Resample with replacement + Gaussian noise at 10% of per-feature std. | **4.3M samples/s** | Fastest; no distributional assumptions |
| `'univariate_kde_copula'` | Per-feature 1-D KDE marginals + Gaussian copula. CDF grids precomputed at `fit()`. | ~1M samples/s | More flexible per-feature marginals |

All four built-in strategies implement `StatefulGenerationStrategy`: partition
metadata is precomputed once in `prepare()` (called at `fit()` time when the
strategy is declared) and reused across repeated `expand()` calls.

```python
from hvrt import BUILTIN_GENERATION_STRATEGIES
list(BUILTIN_GENERATION_STRATEGIES)
# ['multivariate_kde', 'univariate_kde_copula', 'bootstrap_noise', 'epanechnikov']
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

# By reference (module-level singleton)
from hvrt import centroid_fps, variance_ordered
X_red = model.reduce(ratio=0.2, method=variance_ordered)

# Custom strategy — implement StatefulSelectionStrategy
from hvrt import StatefulSelectionStrategy, SelectionContext
import numpy as np

class MySelector:
    def prepare(self, X_z, partition_ids, unique_partitions):
        # precompute partition metadata once (cached at fit() time)
        from hvrt.reduction_strategies import _build_selection_context
        return _build_selection_context(X_z, partition_ids, unique_partitions)

    def select(self, context, budgets, random_state, n_jobs=1):
        # context.sort_idx, context.part_starts, context.part_sizes available
        ...
        return selected_indices   # global indices into X_z

X_red = model.reduce(ratio=0.2, method=MySelector())
```

| Strategy | Behaviour | Notes |
|---|---|---|
| `'fps'` / `'centroid_fps'` | Greedy FPS seeded at partition centroid. **Default.** | Best general-purpose diversity |
| `'medoid_fps'` | FPS seeded at partition medoid. | Robust to outliers; slightly slower |
| `'variance_ordered'` | Highest local k-NN variance (k=10). | **23–37× faster** with `n_jobs=-1` at large n |
| `'stratified'` | Fully-vectorised random sample. | **2.5–3× faster** than loop; best for repeated `reduce()` |

All four built-in strategies implement `StatefulSelectionStrategy`: partition
metadata is precomputed once in `prepare()` (eagerly at `fit()` time when
declared via `reduce_params.method`) and cached across repeated `reduce()` calls.
The model's `n_jobs` is forwarded to `select()` automatically.

```python
from hvrt import BUILTIN_STRATEGIES
list(BUILTIN_STRATEGIES)
# ['centroid_fps', 'fps', 'medoid_fps', 'variance_ordered', 'stratified']
```

**Memory-conscious large-data workflow:** FPS strategies dispatch partitions to
loky workers independently, keeping per-worker memory O(partition size) regardless
of total dataset size.  Enable with `n_jobs=-1`:

```python
model = HVRT(n_jobs=-1).fit(X_large)          # n_jobs forwarded to select()
X_red = model.reduce(ratio=0.1, method='fps') # parallel FPS, bounded memory per worker
```

---

## Recommendations

Findings from a systematic bandwidth and kernel benchmark across 6 datasets,
3 expansion ratios (2×/5×/10×), and 11 methods (see `benchmarks/bandwidth_benchmark.py`
and `findings.md`).

### `bandwidth='auto'` — the default

`bandwidth='auto'` is the default and requires no tuning for most datasets. At each
`expand()` call it inspects the fitted partition structure and picks the kernel most
likely to produce high-quality synthetic data:

```python
model = HVRT().fit(X)          # bandwidth='auto' by default
X_synth = model.expand(n=50000)  # auto chooses at call-time
```

**How it decides:**

At call-time, `'auto'` computes the mean number of samples per partition and
compares it against a feature-scaled threshold: `max(15, 2 × n_continuous_features)`.

| Condition | Chosen kernel | Reason |
|---|---|---|
| mean partition size **≥** threshold | Narrow Gaussian `h=0.1` | Enough samples for stable multivariate covariance estimation; tight kernel stays within partition geometry |
| mean partition size **<** threshold | Epanechnikov product kernel | Too few samples for reliable covariance; product kernel requires no covariance matrix and bounded support keeps samples within the local region |

The threshold scales with dimensionality because the minimum samples needed for a
non-degenerate `d`-dimensional covariance matrix grows with `d`. At 5 features the
threshold is 15; at 15 features it is 30.

**Why not just always use one or the other:**

Benchmarking across 4 regression datasets showed a clean crossover depending on
partition size. With the default auto-tuned partition count (typically 15–20 partitions
at n=500), partitions hold ~25 samples and narrow Gaussian wins on TSTR. But when
partitions are finer — either because the dataset is large and the auto-tuner produces
more leaves, or because `n_partitions` is manually increased — Gaussian KDE degrades
as partitions become too small for stable covariance estimation, while Epanechnikov
holds steady or improves. For example, on the housing dataset (d=6) at 10× expansion:

| Partition count | Gaussian `h=0.1` TSTR | Epanechnikov TSTR |
|---|---|---|
| auto (~18) | +0.004 | −0.014 |
| 50 | −0.033 | **−0.008** |
| 100 | −0.037 | **−0.011** |
| 200 | −0.080 | **−0.008** |

The crossover point depends on dimensionality: higher-dimensional datasets shift it
earlier. On multimodal (d=10), Epanechnikov wins from 30 partitions onward (mean
partition size ~13 at n=500). On housing (d=6) and emergence_divergence (d=5),
the crossover is ~50 partitions. This is because higher dimensionality makes a
d×d covariance matrix harder to estimate stably from small samples, while
Epanechnikov is always covariance-free.

`'auto'` captures this automatically: when you call `expand(n_partitions=200)`,
`'auto'` sees the resulting small partition sizes and switches to Epanechnikov
without any manual intervention.

**When to override `'auto'`:**

- **Heterogeneous / high-skew classification task (mean |skew| ≳ 0.8):**
  `generation_strategy='epanechnikov'` directly — Epanechnikov wins consistently
  when within-partition data is non-Gaussian. On near-Gaussian classification data,
  `bandwidth='auto'` (`h=0.10`) or `adaptive_bandwidth=True` is competitive or
  better, particularly at 2×–5× expansion ratios.
- **Small dataset, coarse partitions, regression:** `bandwidth=0.1` or `bandwidth=0.3`
  — explicit narrow Gaussian if you know partition sizes are large and correlation
  structure matters.
- **Diagnostic / ablation:** pass explicit values (`bandwidth=0.3`, `bandwidth='scott'`)
  to isolate the bandwidth effect.

### Why Scott's rule underperforms

Scott's rule is AMISE-optimal for iid Gaussian data. HVRT partitions, while locally
more homogeneous than the global distribution, are not Gaussian enough for this to
hold (mean |skewness| 0.49–1.37 across benchmark datasets). More importantly, the
decision tree already captures the primary variance structure of each partition, so
the residual within-partition variance is narrower than Scott's formula assumes.
The result is systematic over-smoothing: synthetic samples bleed across partition
boundaries and dilute the local density structure. Scott's rule won 0 of 18
benchmark conditions.

Wide bandwidths (≥ 0.75) are actively harmful. They produce synthetic data that
degrades downstream ML models (TSTR Δ as low as −0.75 R²). Discriminator accuracy
can paradoxically *improve* with wide bandwidths on regression — a metric artifact
where spreading matches marginals while destroying joint structure. Use TSTR as the
primary quality signal, not disc_err.

### Partition granularity

If `'auto'` is already in use, increasing `n_partitions` will automatically trigger
the switch to Epanechnikov when partition sizes fall below the threshold. You can
also set it explicitly:

```python
# Finer partitions — 'auto' will pick Epanechnikov when sizes drop below threshold
model.expand(n=50000, n_partitions=150)

# Or fix at construction time
model = HVRT(n_partitions=150, min_samples_leaf=10).fit(X)
```

Benchmark evidence (regression datasets, 5×/10× expansion ratios):

| Dataset (d) | At auto (~18 parts) best TSTR | At 150 parts Epan TSTR |
|---|---|---|
| housing (d=6) | h=0.30: −0.001 | **−0.013** |
| multimodal (d=10) | h=0.30: +0.004 | **+0.001** |
| emergence_divergence (d=5) | h=0.10: +0.007 | **+0.004** |
| emergence_bifurcation (d=5) | h=0.10: −0.022 | **−0.118** |

Note: for the emergence_bifurcation dataset (where the same feature region maps
to a bimodal target), all methods remain significantly negative at any partition
count. This indicates a structural limit: if the same X values correspond to
multiple distinct y outcomes, expansion without conditioning on y cannot reproduce
that structure. In such cases consider conditioning expansion on y directly
(e.g., expand class-conditional subsets separately).

### Hyperparameter optimisation (HPO)

Dataset heterogeneity is the primary driver of how sensitive synthetic quality
is to HVRT's parameters. A well-behaved, near-Gaussian dataset with few
sub-populations produces good synthetic data at defaults with little room to
improve. A dataset with distinct clusters, non-linear interactions, or
regime-switching needs finer partitions to achieve local homogeneity within
each leaf — and the optimal settings are dataset-specific.

Benchmark evidence: on near-Gaussian data (fraud, housing at auto partition
count), TSTR varied by less than 0.01 across all bandwidth candidates. On
heterogeneous datasets (emergence_divergence, emergence_bifurcation), TSTR
varied by up to 0.20+ between the best and worst methods at the same partition
count. If your data is heterogeneous, HPO pays; if it is well-behaved, defaults
are sufficient.

**When HPO is worth running:**

- TSTR Δ is significantly negative on your downstream task (below −0.05 is a
  useful rule of thumb)
- Your dataset has known sub-populations, clusters, non-linear interactions, or
  regime changes (e.g., different dynamics at different feature values)
- You are generating at a high ratio (10×+) where compounding errors matter more

**Parameter search space:**

| Parameter | Default | Suggested search | Effect |
|---|---|---|---|
| `n_partitions` | auto | `None`, 20, 30, 50, 75, 100 | **Primary lever.** More partitions → finer local homogeneity. Start here. |
| `min_samples_leaf` | auto | 5, 10, 15, 20 | Controls auto-tuner floor; lower allows finer splits when n is large. |
| `bandwidth` | `'auto'` | `'auto'`, 0.05, 0.10, 0.30, `epanechnikov` | `'auto'` is usually near-optimal once partition count is right. |
| `variance_weighted` | `False` | `True`, `False` | `True` oversamples high-variance partitions; useful for tail-heavy distributions. |
| `y_weight` | 0.0 | 0.1, 0.3, 0.5 | Weights target in synthetic target; helps when y governs sub-population identity. |

**Evaluation metric:** Use **TSTR Δ** (train-on-synthetic, test-on-real minus
train-on-real baseline) as the HPO objective. Discriminator accuracy (`disc_err`)
is structurally insensitive — wide bandwidths can lower it by spreading marginals
while destroying joint structure. TSTR directly measures what matters: can a model
trained on synthetic data perform as well as one trained on real data?

**Example HPO loop:**

Use `HVRTOptimizer` for automated Bayesian optimisation with Optuna
(install the optional extra first: `pip install hvrt[optimizer]`):

```python
from hvrt import HVRTOptimizer

opt = HVRTOptimizer(n_trials=50, n_jobs=4, cv=3, random_state=42).fit(X, y)
print(f'Best TSTR Δ: {opt.best_score_:+.4f}')
print(f'Best params: {opt.best_params_}')

X_synth = opt.expand(n=50000)        # uses tuned kernel + params
X_aug   = opt.augment(n=len(X) * 5)  # originals + synthetic
```

`HVRTOptimizer` searches over `n_partitions`, `min_samples_leaf`,
`y_weight`, kernel / bandwidth, and `variance_weighted` using TPE
sampling, with TRTR pre-computed once to halve GBM fitting overhead.
The fitted `best_model_` is refitted on the full dataset after tuning.

For a custom objective or manual grid search:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from hvrt import HVRT

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def tstr_delta(n_partitions, bandwidth, variance_weighted=False, seed=42):
    XY_tr = np.column_stack([X_tr, y_tr.reshape(-1, 1)])
    model = HVRT(n_partitions=n_partitions, bandwidth=bandwidth,
                 random_state=seed).fit(XY_tr)
    XY_s = model.expand(n=len(X_tr) * 5, variance_weighted=variance_weighted)
    X_s, y_s = XY_s[:, :-1], XY_s[:, -1]
    trtr = r2_score(y_te, GradientBoostingRegressor(
                        random_state=seed).fit(X_tr, y_tr).predict(X_te))
    tstr = r2_score(y_te, GradientBoostingRegressor(
                        random_state=seed).fit(X_s, y_s).predict(X_te))
    return tstr - trtr

best_score, best_cfg = float('-inf'), {}
for n_parts in [None, 30, 50, 100]:   # None = let auto-tune decide
    for bw in ['auto', 0.10, 0.30]:
        score = tstr_delta(n_partitions=n_parts, bandwidth=bw)
        if score > best_score:
            best_score, best_cfg = score, {'n_partitions': n_parts, 'bandwidth': bw}

print(f'Best TSTR Δ={best_score:+.4f}  params={best_cfg}')
```

**Recommended tuning sequence:**

1. **Run with defaults.** Establish a baseline TSTR Δ. If it is close to zero, stop.
2. **Sweep `n_partitions`.** This has the largest effect on heterogeneous data. Try
   `None` (auto), 20, 30, 50, 75, 100. More partitions only help when `n` is large
   enough — a rule of thumb is at least 10–15 real samples per partition.
3. **Check `bandwidth`.** With `'auto'`, HVRT already picks the right kernel for
   the resulting partition size. If you have prior knowledge (classification → prefer
   `'epanechnikov'`; regression with large partitions → prefer `0.10`), override it.
4. **Try `variance_weighted=True`** if your dataset has a long tail or rare events
   you want the expansion to oversample.
5. **If TSTR remains poor at any partition count**, the dataset likely has inherently
   unpredictable local structure (e.g., the same feature region maps to multiple
   distinct outcomes). Consider conditioning: split by `y` quantile or class and
   expand each subset independently.

**What not to try:** Expanding synthetically and re-fitting HVRT on that output
("two-phase pipeline") to manufacture fine partitions does not improve TSTR.
Phase 1 Gaussian smoothing introduces distribution drift that Phase 2 amplifies,
and the net TSTR is worse than single-phase at the auto partition count. Finer
partitions must come from more *real* data.

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

Metrics: discriminator accuracy (target ≈ 50%), marginal fidelity, tail preservation (target = 1.0),
**Privacy DCR**, TSTR Δ.  `bandwidth='auto'`, `max_n=500` training samples, expansion ratio 1×.
Mean across continuous benchmark datasets (fraud, housing, multimodal).  Full results: `--print-table expand`.

| Method | Marginal Fidelity | Disc. Err % ↓ | Tail Preservation | Privacy DCR | TRTR | TSTR | TSTR Δ | Fit time |
|---|---|---|---|---|---|---|---|---|
| **HVRT-size** | **0.944** | 5.0 | 1.023 | 0.45 | 0.846 | 0.850 | +0.004 | 0.006 s |
| **HVRT-var** | 0.921 | 1.8 | 1.068 | 0.45 | 0.846 | **0.866** | **+0.020** | 0.007 s |
| FastHVRT-size | 0.936 | **1.5** | 1.018 | 0.43 | 0.846 | 0.805 | −0.041 | 0.006 s |
| Gaussian Copula | 0.937 | 1.9 | 0.983 | 1.17 | 0.846 | 0.806 | −0.040 | 0.002 s |
| GMM (k≤20) | 0.878 | 1.8 | 1.035 | 1.17 | 0.846 | 0.820 | −0.026 | 0.028 s |
| Bootstrap + Noise | 0.928 | **0.8** | 0.971 | 0.41 | 0.846 | 0.833 | −0.013 | 0.000 s |
| SMOTE | 0.902 | 1.0 | 0.889 | 0.30 | 0.846 | 0.828 | −0.018 | 0.003 s |
| CTGAN† | 0.421 | 32.3 | — | 1.95 | 0.769* | 0.726 | −0.043 | ~10 s |
| TVAE† | 0.624 | 26.1 | — | 0.89 | 0.769* | 0.702 | −0.067 | ~6 s |
| TabDDPM‡ | 0.960 | 2.0 | — | N/A‡ | — | — | — | 120 s |
| MOSTLY AI‡ | 0.975 | 1.0 | — | N/A‡ | — | — | — | 60 s |

*Disc. Err = |discriminator accuracy − 50%|. Lower = more indistinguishable from real. Target = 0.*
*Note: Bootstrap + Noise achieves 0.8% error by creating near-copies — low discriminator error without genuine novelty. TSTR Δ is the more reliable quality signal.*
*† CTGAN/TVAE run locally (`--deep-learning`); all metrics including DCR are computed. Poor Disc. Err reflects small n=400 training set — deep-learning methods need more data.*
*‡ Published numbers only — no local runner. DCR cannot be computed.*
*\* CTGAN/TVAE TRTR is 0.769 (mean over housing + multimodal); fraud was not evaluated for these methods, so their baseline differs from other methods' 0.846 (fraud + housing + multimodal).*
*Tail preservation = 1.0 is ideal.*

Reproduce: `python benchmarks/run_benchmarks.py --tasks expand --deep-learning`

### Privacy evaluation

The benchmark suite computes two data privacy metrics for every expansion run.
Both are available in the full JSON output, the ASCII results table (`--print-table expand`),
and the summary plot (`--plot`).

**Distance-to-Closest-Record (DCR)**

```
DCR = median(min_dist(synthetic_i → real))
    / median(min_dist(real_i → real excluding itself))
```

| DCR range | Interpretation |
|---|---|
| < 0.1 | Near-copies: synthetic samples sit very close to specific training records — high record-linkage risk |
| 0.1 – 0.8 | Tight generation: samples fit the local distribution well; low risk unless combined with auxiliary data |
| ≈ 1.0 | Neutral: synthetic samples at the same typical distances as real-to-real neighbours |
| > 1.0 | Spread generation: samples more dispersed than real data — global models cover empty regions |

A tight generative model (HVRT, Bootstrap + Noise) intentionally produces samples close to the
training distribution, yielding DCR < 1 on continuous data — this is correct behaviour, not a
privacy failure. Global models (Gaussian Copula, GMM) yield DCR ≈ 1.0–1.2 because they sample
from the full inferred distribution rather than local neighbourhoods.

The critical threshold is DCR < 0.1: at that level synthetic samples are essentially copies of
training records and present a realistic record-linkage risk. HVRT (DCR ≈ 0.45) is 3× safer
than Bootstrap + Noise (DCR ≈ 0.16) and 1.5× safer than SMOTE (DCR ≈ 0.30) on continuous data.

**Categorical data caveat**: on datasets with many near-duplicate records (e.g., binary or
low-cardinality features), the real→real NN distance approaches zero, making the DCR ratio
unstable. In the benchmark the adult dataset produces DCR values of 10–400× for all methods
due to this effect — treat those numbers as unreliable. DCR is most meaningful on
fully-continuous feature sets.

**Novelty min-distance** (`novelty_min`) is the minimum Euclidean distance from any synthetic
sample to any real sample. A value of 0 means at least one exact duplicate of a training record
was generated. HVRT's KDE sampling is strictly stochastic and produces `novelty_min > 0` for
any finite bandwidth.

Full privacy diagnostics per method are printed when you run:

```bash
python benchmarks/run_benchmarks.py --tasks expand --deep-learning
python benchmarks/report_results.py            # detailed rankings + radar chart including DCR
```

### Privacy–Fidelity Decision Matrix

`bandwidth` is the primary lever for controlling Privacy DCR. The table below shows the recommended
configuration per privacy profile, selected for highest marginal fidelity within each DCR range
(TSTR Δ ≥ −0.05 filter; expansion ratio 2×; averaged across continuous datasets: fraud, housing, multimodal).

| Privacy Profile | DCR Target | `n_partitions` | `bandwidth` | DCR | Marginal Fidelity | Disc. Err % | TRTR | TSTR | TSTR Δ |
|---|---|---|---|---|---|---|---|---|---|
| Tight | [0.00, 0.40) | `None` (auto) | `0.1` | 0.332 | 0.966 | 1.83% | 0.846 | 0.834 | −0.012 |
| Moderate | [0.40, 0.70) | `None` (auto) | `'auto'` | 0.443 | 0.958 | 0.79% | 0.846 | 0.834 | −0.012 |
| High | [0.70, 1.00) | `None` (auto) | `0.5` | 0.797 | 0.925 | 1.71% | 0.846 | 0.839 | −0.007 |
| Maximum | [1.00, ∞) | `10` | `'scott'` | 1.067 | 0.856 | 0.50% | 0.846 | 0.824 | −0.022 |

*Reproduce: `python benchmarks/dcr_privacy_benchmark.py`*

As expected, higher DCR trades off against marginal fidelity. The Moderate profile (`bandwidth='auto'`, `n_partitions=None`)
is the default HVRT behaviour. Choosing a privacy profile is a one-parameter decision — only `bandwidth` changes;
`n_partitions` stays at `None` for all profiles except Maximum.

```python
# High privacy profile
model = FastHVRT(bandwidth=0.5, random_state=42).fit(X)
X_synth = model.expand(n=50000)   # DCR ≈ 0.80, MF ≈ 0.925

# Maximum privacy profile
model = FastHVRT(n_partitions=10, bandwidth='scott', random_state=42).fit(X)
X_synth = model.expand(n=50000)   # DCR ≈ 1.07, MF ≈ 0.856
```

### Adaptive bandwidth and privacy

`adaptive_bandwidth=True` provides a significant DCR lift at expansion ratios above 1× without
changing constructor parameters. It scales per-partition bandwidth as
`bw_p = scott_p × max(1, budget_p/n_p)^(1/d)`.

| Expansion ratio | `adaptive_bandwidth` | Dataset | DCR | Marginal Fidelity | TRTR | TSTR | TSTR Δ |
|---|---|---|---|---|---|---|---|
| 2× | False | fraud | 0.448 | 0.940 | 1.000 | 1.000 | 0.000 |
| 2× | **True** | fraud | 0.448 | 0.940 | 1.000 | 1.000 | 0.000 |
| 2× | False | housing | 0.744 | 0.962 | 0.573 | 0.554 | −0.019 |
| 2× | **True** | housing | **1.387** | 0.851 | 0.573 | **0.586** | **+0.013** |
| 2× | False | multimodal | 0.136 | 0.971 | 0.966 | 0.949 | −0.017 |
| 2× | **True** | multimodal | **1.060** | 0.854 | 0.966 | 0.947 | −0.019 |
| 5× | False | housing | 0.739 | 0.975 | 0.573 | 0.541 | −0.032 |
| 5× | **True** | housing | **1.349** | 0.825 | 0.573 | **0.590** | **+0.017** |
| 5× | False | multimodal | 0.136 | 0.975 | 0.966 | 0.948 | −0.018 |
| 5× | **True** | multimodal | **1.131** | 0.831 | 0.966 | 0.947 | −0.019 |

`adaptive_bandwidth=True` moves housing from Moderate (0.74) to Maximum (1.39) and multimodal from
Tight (0.14) to Maximum (1.06) at 2× ratio, while also improving TSTR Δ on housing (+0.032 gain).
On fraud the dataset is already KDE-bandwidth-dominated and adaptive scaling has no additional effect.
Use `adaptive_bandwidth=True` when expanding at ratios ≥ 2× and a DCR ≥ 1.0 target is required.

---

## Benchmarking Scripts

```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --tasks reduce --datasets adult housing
python benchmarks/run_benchmarks.py --tasks expand
python benchmarks/strategy_speedup_benchmark.py   # vectorization speedup (new in v2.4)
python benchmarks/speed_benchmark.py              # serial vs parallel wall-clock times
python benchmarks/reduction_denoising_benchmark.py
python benchmarks/adaptive_kde_benchmark.py
python benchmarks/adaptive_full_benchmark.py
python benchmarks/heart_disease_benchmark.py      # requires: pip install ctgan
python benchmarks/bootstrap_failure_benchmark.py
python benchmarks/hpo_benchmark.py               # HPO vs defaults, nested CV (requires: pip install hvrt[optimizer])
python benchmarks/hpo_benchmark.py --quick       # 3 datasets, 10 trials, fast mode
python benchmarks/dcr_privacy_benchmark.py       # privacy–fidelity parameter sweep + decision matrix
python benchmarks/dcr_privacy_benchmark.py --no-adaptive  # grid-only, skip adaptive bandwidth sweep
```

---

## Backward Compatibility

The v1 API is still importable:

```python
from hvrt import HVRTSampleReducer, AdaptiveHVRTReducer

reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
X_reduced, y_reduced = reducer.fit_transform(X, y)
```

The `mode` constructor parameter is deprecated. Replace with params objects:

```python
# Deprecated
HVRT(mode='reduce')

# Replacement
HVRT(reduce_params=ReduceParams(ratio=0.3))
```

The plain callable protocol for generation strategies is deprecated (v2.4).
Custom callables still work and emit `HVRTDeprecationWarning`:

```python
# Deprecated (still works)
def my_strategy(X_z, partition_ids, unique_partitions, budgets, random_state):
    ...
    return X_synthetic

model.expand(n=50000, generation_strategy=my_strategy)

# Replacement: implement StatefulGenerationStrategy
from hvrt import StatefulGenerationStrategy

class MyStrategy:
    def prepare(self, X_z, partition_ids, unique_partitions):
        ...   # return a PartitionContext (or subclass)
    def generate(self, context, budgets, random_state):
        ...   # return (sum(budgets), d) ndarray

model.expand(n=50000, generation_strategy=MyStrategy())
```

The plain callable protocol for selection strategies is also deprecated.
Custom callables still work and emit `HVRTDeprecationWarning`:

```python
# Deprecated (still works)
def my_selector(X_z, partition_ids, unique_partitions, budgets, random_state):
    ...
    return selected_indices

model.reduce(ratio=0.3, method=my_selector)

# Replacement: implement StatefulSelectionStrategy
from hvrt import StatefulSelectionStrategy
from hvrt.reduction_strategies import _build_selection_context

class MySelector:
    def prepare(self, X_z, partition_ids, unique_partitions):
        return _build_selection_context(X_z, partition_ids, unique_partitions)
    def select(self, context, budgets, random_state, n_jobs=1):
        ...   # return 1-D int64 ndarray of global indices

model.reduce(ratio=0.3, method=MySelector())
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
  url    = {https://github.com/hotprotato/hvrt}
}
```

---

## License

GNU Affero General Public License v3 or later (AGPL-3.0-or-later) — see [LICENSE](LICENSE).

## Acknowledgments

Development assisted by Claude (Anthropic).
