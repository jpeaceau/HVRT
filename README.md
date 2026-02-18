# HVRT: Hierarchical Variance Reduction Tree

[![PyPI version](https://img.shields.io/pypi/v/hvrt.svg)](https://pypi.org/project/hvrt/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**HVRT** is a deterministic, variance-based sample reduction method that intelligently selects training subsets while preserving predictive accuracy.

## Why HVRT?

Unlike random sampling which treats all samples equally, HVRT optimizes for **explained variance preservation** through:

- **Hierarchical partitioning** based on pairwise feature interactions
- **Diversity-based selection** via Furthest-Point Sampling (FPS)
- **100% deterministic** - same data → same subset every time
- **Hybrid mode** for heavy-tailed data and rare events

## Installation

Install from [PyPI](https://pypi.org/project/hvrt/):

```bash
pip install hvrt
```

Or install from source:

```bash
git clone https://github.com/hotprotato/hvrt.git
cd hvrt
pip install -e .
```

## Quick Start

```python
from hvrt import HVRTSampleReducer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load your data
X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reduce training set to 20% of original size
reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
X_train_reduced, y_train_reduced = reducer.fit_transform(X_train, y_train)

# Train any model on reduced data
model = RandomForestRegressor()
model.fit(X_train_reduced, y_train_reduced)
predictions = model.predict(X_test)
```

## When to Use HVRT

### ✅ HVRT Excels When

- **Regulatory/audit requirements** - 100% reproducible sample selection
- **Heavy-tailed distributions** - Financial data, extreme events, rare outliers
- **SVM training** - Makes large-scale SVM practical (25-40x speedup)
- **Aggressive reduction needed** - 5-20% retention where every sample counts
- **Small-to-medium datasets** - Up to 50k samples

### ⚠️ Random Sampling May Suffice When

- **Large, well-behaved datasets** (≥50k samples, normal distributions)
- **Modest reduction** (≥50% retention)
- No interpretability or determinism requirements

### ❌ Avoid HVRT For

- **Distance-based clustering tasks** (K-Means, DBSCAN)
- Very small datasets (n < 1000)

## Key Features

### 1. Deterministic Selection

```python
# Same random_state → identical samples every time
reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
X_reduced1, _ = reducer.fit_transform(X, y)

reducer2 = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
X_reduced2, _ = reducer2.fit_transform(X, y)

assert np.array_equal(X_reduced1, X_reduced2)  # ✓ Always True
```

### 2. Hybrid Mode for Heavy Tails

```python
# For heavy-tailed data or rare events
reducer = HVRTSampleReducer(
    reduction_ratio=0.2,
    y_weight=0.25,  # 25% weight on y-extremeness
    random_state=42
)
X_reduced, y_reduced = reducer.fit_transform(X, y)
```

**When to use `y_weight`:**
- `0.0` (default): Well-behaved data, interaction-driven variance
- `0.25-0.50`: Heavy-tailed distributions, rare events
- `0.50-1.0`: Extreme outlier detection

### 3. SVM Speedup Example

```python
from sklearn.svm import SVR
import time

# Without HVRT: 30 minutes at 50k samples
start = time.time()
svm = SVR()
svm.fit(X_train, y_train)  # 50k samples
print(f"Training time: {time.time() - start:.1f}s")  # ~1800s

# With HVRT: 47 seconds (38x faster!)
reducer = HVRTSampleReducer(reduction_ratio=0.2)
X_reduced, y_reduced = reducer.fit_transform(X_train, y_train)  # 10k samples

start = time.time()
svm.fit(X_reduced, y_reduced)
print(f"Training time: {time.time() - start:.1f}s")  # ~47s
```

## API Reference

### AdaptiveHVRTReducer (Recommended)

**Automatically finds optimal reduction level via accuracy testing.**

```python
from hvrt import AdaptiveHVRTReducer

reducer = AdaptiveHVRTReducer(
    accuracy_threshold=0.95,     # Min accuracy retention (95%)
    reduction_ratios=[0.5, 0.3, 0.2, 0.1],  # Levels to test
    validator=None,              # Auto: XGBoost (fast validation)
    scoring='accuracy',          # Metric: accuracy, f1, r2, custom, dict
    cv=3,                        # Cross-validation folds
    y_weight=0.0,                # Hybrid mode (0.25 for heavy tails)
    random_state=42
)

X_reduced, y_reduced = reducer.fit_transform(X, y)

# Review all tested reductions
print(reducer.get_reduction_summary())
for result in reducer.reduction_results_:
    print(f"{result['reduction_ratio']}: {result['accuracy_retention']:.1%}")

# Multiple metrics example
reducer_multi = AdaptiveHVRTReducer(
    accuracy_threshold=0.95,
    scoring={'accuracy': 'accuracy', 'f1': 'f1', 'recall': 'recall'}
)
reducer_multi.fit(X, y)
print(reducer_multi.best_reduction_['all_scores'])
```

**Scoring Options:**

Built-in metrics (str):
- Classification: `'accuracy'`, `'f1'`, `'precision'`, `'recall'`, `'roc_auc'`
- Regression: `'r2'`, `'neg_mean_absolute_error'`, `'neg_mean_squared_error'`

Custom scorer (callable):
```python
def custom_scorer(y_true, y_pred):
    return score  # Higher is better
```

Multiple metrics (dict):
```python
scoring={'acc': 'accuracy', 'f1': 'f1'}  # First key is primary
```

**Use Cases:**
- Unknown optimal reduction level
- SVM training (use XGBoost validation, get samples for SVM)
- Need accuracy guarantees with specific metrics

**Methods:**
- `fit(X, y)`: Test reductions, find best
- `transform()`: Return best reduced dataset
- `get_reduction_summary()`: Human-readable results table

**Attributes:**
- `reduction_results_`: List of all tested reductions
- `best_reduction_`: Optimal reduction meeting threshold
- `baseline_score_`: Baseline accuracy

---

### HVRTSampleReducer (Manual)

**Direct reduction when you know the ratio.**

```python
from hvrt import HVRTSampleReducer

reducer = HVRTSampleReducer(
    reduction_ratio=0.2,      # Target retention (0.2 = keep 20%)
    y_weight=0.0,             # Hybrid mode weight (0.0-1.0)
    max_leaf_nodes=None,      # Tree partitions (auto-tuned if None)
    min_samples_leaf=None,    # Min samples per partition (auto-tuned)
    auto_tune=True,           # Enable automatic hyperparameter tuning
    random_state=42           # Random seed for reproducibility
)

X_reduced, y_reduced = reducer.fit_transform(X, y)
```

**Methods:**
- `fit(X, y)`: Learn variance-based partitioning
- `transform(X, y=None)`: Return reduced samples
- `fit_transform(X, y)`: Fit and transform in one step
- `get_reduction_info()`: Get partitioning statistics

**Attributes:**
- `selected_indices_`: Indices of selected samples
- `n_partitions_`: Number of partitions created
- `tree_`: Fitted DecisionTreeRegressor

## Examples

See the [`examples/`](examples/) directory for complete demonstrations:

- [`basic_usage.py`](examples/basic_usage.py) - Simple 10-line example
- [`adaptive_reduction.py`](examples/adaptive_reduction.py) - **NEW:** Automatic reduction level selection
- [`adaptive_scoring_options.py`](examples/adaptive_scoring_options.py) - **NEW:** Custom metrics (MAE, F1, callable)
- [`svm_speedup_demo.py`](examples/svm_speedup_demo.py) - SVM training speedup
- [`heavy_tailed_data.py`](examples/heavy_tailed_data.py) - Hybrid mode for rare events
- [`regulatory_compliance.py`](examples/regulatory_compliance.py) - Determinism for regulated industries

Run any example:
```bash
python examples/adaptive_reduction.py        # Automatic optimal reduction
python examples/adaptive_scoring_options.py  # Custom scoring metrics
python examples/basic_usage.py               # Simple manual reduction
```

## Use Cases

### 1. Regulatory Compliance (Healthcare, Finance)

```python
# FDA submission requires reproducible model training
reducer = HVRTSampleReducer(reduction_ratio=0.3, random_state=42)
X_reduced, y_reduced = reducer.fit_transform(X_train, y_train)

# Audit trail: Decision tree shows why samples were selected
info = reducer.get_reduction_info()
print(f"Partitions: {info['n_partitions']}, Tree depth: {info['tree_depth']}")
```

### 2. Financial Data (Heavy Tails)

```python
# Rare extreme events (market crashes, fraud)
reducer = HVRTSampleReducer(
    reduction_ratio=0.2,
    y_weight=0.5,  # Prioritize extreme outcomes
    random_state=42
)
X_reduced, y_reduced = reducer.fit_transform(X_returns, y_volatility)
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Reduce data once, then use for all 100+ grid search trials
reducer = HVRTSampleReducer(reduction_ratio=0.2, random_state=42)
X_reduced, y_reduced = reducer.fit_transform(X_train, y_train)

# Grid search on reduced data (10-50x faster)
grid = GridSearchCV(SVR(), param_grid, cv=5)
grid.fit(X_reduced, y_reduced)
```

## How It Works

1. **Synthetic Target Construction**
   ```python
   y_synthetic = sum(pairwise_interactions(X)) + α × |y - median(y)|
   ```
   - Captures feature interaction patterns
   - Optional y-extremeness weighting for outliers

2. **Hierarchical Partitioning**
   - Decision tree partitions data by synthetic target
   - Captures variance heterogeneity across feature space
   - Auto-tunes partition count based on dataset size

3. **Diversity Selection**
   - Within each partition: Furthest-Point Sampling (FPS)
   - Removes density, preserves boundaries
   - Centroid-seeded for determinism

## Performance

Results from validation experiments with SVM feasibility testing on 10k samples:

| Scenario | Retention | HVRT Accuracy | Random Accuracy | Speedup | SNR Retention |
|----------|-----------|---------------|-----------------|---------|---------------|
| Well-behaved | 20% | 93.9% | 95.3% | 23.5x | **126.2%** |
| Heavy-tailed | 20% | **106.6%** | 85.3% | 24.0x | **130.1%** |

**Key Findings:**
- **Well-behaved data:** Both methods work (CLT holds), random slightly better on accuracy
- **Heavy-tailed data:** HVRT achieves +21pp accuracy advantage via intelligent noise filtering
- **SNR (Signal-to-Noise Ratio):** HVRT improves data quality by 26-30% vs baseline
- **SVM Speedup:** 24-38x training time reduction at scale (50k samples)

**Why >100% accuracy?** HVRT acts as an intelligent denoiser, removing low-signal samples and improving SNR by 30%, which leads to better generalization.

### Experimental Data

All experimental results are included in this repository for full transparency and reproducibility.

**📊 Primary Validation Study:**
- [`archive/experimental/results/svm_pilot/pilot_results_with_snr.json`](archive/experimental/results/svm_pilot/pilot_results_with_snr.json) - Complete SVM pilot data with SNR measurements (15 trials, 10k samples)

**📄 Analysis & Documentation:**
- [`archive/docs/SVM_PILOT_SNR_ANALYSIS.md`](archive/docs/SVM_PILOT_SNR_ANALYSIS.md) - Detailed SNR analysis explaining >100% accuracy
- [`archive/docs/SVM_PERFORMANCE_ANALYSIS.md`](archive/docs/SVM_PERFORMANCE_ANALYSIS.md) - SVM feasibility study (speedup, accuracy)

**🔬 Reproduce Results:**
```bash
cd archive/experimental/experiments
python exp_svm_pilot_with_snr.py  # ~30 seconds
```

**📑 Complete Archive:**
See [`archive/ARCHIVE_INDEX.md`](archive/ARCHIVE_INDEX.md) for complete experimental data catalog.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=hvrt --cov-report=term-missing
```

## Citation

If you use HVRT in your research, please cite:

```bibtex
@software{hvrt2025,
  author = {Peace, Jake},
  title = {HVRT: Hierarchical Variance Reduction Tree Sample Reduction},
  year = {2025},
  url = {https://github.com/hotprotato/hvrt}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

Development assisted by Claude (Anthropic) for rapid prototyping and conceptual refinement.

## Related Work

- **Random Sampling**: Simple but fails on heavy-tailed data
- **CoreSet Methods**: Require distance metrics (not suitable for all data)
- **Active Learning**: Requires iterative labeling (different use case)
- **HVRT**: Deterministic, variance-based, works without distance metrics
