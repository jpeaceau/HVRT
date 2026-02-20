# HVRT: Hierarchical Variance-Retaining Transformer

[![PyPI version](https://img.shields.io/pypi/v/hvrt.svg)](https://pypi.org/project/hvrt/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
    bandwidth=0.5,               # Default KDE bandwidth for expand()
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

model = FastHVRT(bandwidth=0.5, random_state=42)
```

Target: sum of z-scores. O(n · d). Equivalent quality to HVRT for expansion. All constructor parameters identical to HVRT.

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
    bandwidth=None,               # Override instance bandwidth
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
from hvrt import FastHVRT, univariate_kde_copula

model = FastHVRT(random_state=42).fit(X)

# By name
X_synth = model.expand(n=10000, generation_strategy='bootstrap_noise')

# By reference
X_synth = model.expand(n=10000, generation_strategy=univariate_kde_copula)

# Custom callable
def my_strategy(X_z, partition_ids, unique_partitions, budgets, random_state):
    ...
    return X_synthetic   # shape (sum(budgets), n_features), z-score space

X_synth = model.expand(n=10000, generation_strategy=my_strategy)
```

| Strategy | Behaviour | Notes |
|---|---|---|
| `'multivariate_kde'` | `scipy.stats.gaussian_kde` on all features jointly. Scott's rule. **Default.** | Captures full joint covariance |
| `'univariate_kde_copula'` | Per-feature 1-D KDE marginals + Gaussian copula. | More flexible per-feature marginals |
| `'bootstrap_noise'` | Resample with replacement + Gaussian noise at 10% of per-feature std. | Fastest; no distributional assumptions |

```python
from hvrt import BUILTIN_GENERATION_STRATEGIES
list(BUILTIN_GENERATION_STRATEGIES)
# ['multivariate_kde', 'univariate_kde_copula', 'bootstrap_noise']
```

---

## Selection Strategies

```python
from hvrt import HVRT

model = HVRT(random_state=42).fit(X, y)

X_red = model.reduce(ratio=0.2, method='fps')             # default
X_red = model.reduce(ratio=0.2, method='medoid_fps')
X_red = model.reduce(ratio=0.2, method='variance_ordered')
X_red = model.reduce(ratio=0.2, method='stratified')

# Custom callable
def my_selector(X_z, partition_ids, unique_partitions, budgets, random_state):
    ...
    return selected_indices   # global indices into X

X_red = model.reduce(ratio=0.2, method=my_selector)
```

| Strategy | Behaviour |
|---|---|
| `'fps'` / `'centroid_fps'` | Greedy Furthest Point Sampling seeded at partition centroid. **Default.** |
| `'medoid_fps'` | FPS seeded at the partition medoid. |
| `'variance_ordered'` | Select samples with highest local k-NN variance (k=10). |
| `'stratified'` | Random sample within each partition. |

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

Metric: discriminator accuracy (target 50% = indistinguishable), marginal KS fidelity, tail MSE.
bandwidth=0.5, synthetic-to-real ratio 1×.

| Method | Marginal Fidelity | Discriminator | Tail Error | Fit time |
|---|---|---|---|---|
| **HVRT** | 0.974 | **49.6%** | **0.004** | 0.07 s |
| Gaussian Copula | 0.998 | 49.4% | 0.017 | 0.02 s |
| GMM (k=10) | 0.989 | 49.2% | 0.093 | 1.06 s |
| Bootstrap + Noise | 0.994 | 49.7% | 0.131 | 0.00 s |
| SMOTE | 1.000 | 48.6% | 0.000 | 0.00 s |
| CTGAN† | 0.920 | 55.8% | 0.500 | 45 s |
| TVAE† | 0.940 | 53.5% | 0.450 | 40 s |
| TabDDPM† | 0.960 | 52.0% | 0.300 | 120 s |
| MOSTLY AI† | 0.975 | 51.0% | 0.150 | 60 s |

*† Published numbers. Discriminator = 50% is ideal. Tail error = 0 is ideal.*

Reproduce: `python benchmarks/run_benchmarks.py --tasks expand`

---

## Benchmarking Scripts

```bash
python benchmarks/run_benchmarks.py
python benchmarks/run_benchmarks.py --tasks reduce --datasets adult housing
python benchmarks/run_benchmarks.py --tasks expand
python benchmarks/reduction_denoising_benchmark.py
python benchmarks/adaptive_kde_benchmark.py
python benchmarks/adaptive_full_benchmark.py
python benchmarks/heart_disease_benchmark.py      # requires: pip install ctgan
python benchmarks/bootstrap_failure_benchmark.py
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

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

Development assisted by Claude (Anthropic).
