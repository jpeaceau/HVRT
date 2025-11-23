# H-VRT Examples

This directory contains practical examples demonstrating how to use H-VRT for different use cases.

## Examples

### 1. `basic_usage.py`
**Quick start example** showing the simplest way to use H-VRT for sample reduction.
- Load data
- Apply H-VRT reduction
- Train a model
- Compare performance

**Run**: `python examples/basic_usage.py`

### 2. `svm_speedup_demo.py`
**SVM feasibility example** demonstrating how H-VRT makes SVM training practical on large datasets.
- Shows 25-40x training speedup
- Compares accuracy retention
- Demonstrates hyperparameter tuning workflow

**Run**: `python examples/svm_speedup_demo.py`

### 3. `heavy_tailed_data.py`
**Heavy-tailed data example** showing H-VRT's hybrid mode for non-well-behaved data.
- Generates Cauchy-distributed data with rare events
- Compares H-VRT (original) vs H-VRT (hybrid) vs Random sampling
- Shows rare event capture rates

**Run**: `python examples/heavy_tailed_data.py`

### 4. `regulatory_compliance.py`
**Deterministic sampling example** for regulatory/audit requirements.
- Demonstrates 100% reproducibility
- Shows cross-platform consistency
- Explains interpretable sample selection

**Run**: `python examples/regulatory_compliance.py`

## Requirements

All examples require:
```bash
pip install hvrt numpy scikit-learn
```

Some examples have additional optional dependencies:
- `svm_speedup_demo.py`: matplotlib (for plots)
- `heavy_tailed_data.py`: matplotlib (for visualization)

Install all example dependencies:
```bash
pip install hvrt[dev]
# or
pip install -r requirements-dev.txt
```

## Typical Output

Each example prints:
- Configuration used
- Timing measurements
- Accuracy metrics
- Comparison with baseline (random sampling)

Expected runtime: 10-60 seconds per example.
