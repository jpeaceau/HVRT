# HVRT Research

This folder contains explorations that informed the design choices in the main HVRT library. Each subfolder is self-contained and can be run independently.

---

## `generation/` — In-Partition Generation Method Comparison

Explores the question: **given a set of variance-homogeneous partitions, what is the best method for generating synthetic samples within each partition?**

The current library uses **multivariate KDE** (`scipy.stats.gaussian_kde` on the full feature matrix). This folder benchmarks that choice against several alternatives.

### Methods compared

| Method | Description | Correlation handling |
|---|---|---|
| `MultivariateKDE` | Full-matrix Gaussian KDE (current HVRT) | Native (joint density) |
| `UnivariateKDERankCoupled` | Per-feature KDE + Gaussian copula rank coupling | Gaussian copula |
| `UnivariateKDEIndependent` | Per-feature KDE, samples drawn independently | None — treats features as independent |
| `PartitionGMM` | Gaussian Mixture Model fitted per partition | Native (Gaussian mixture) |
| `KNNInterpolation` | SMOTE-style: interpolate between k nearest neighbours | Preserved by construction |
| `PartitionBootstrap` | Resample with replacement + Gaussian noise (baseline) | Preserved by construction |

### Modern and deep learning methods (optional)

The comparison script also supports external libraries when installed:

| Method | Library | Notes |
|---|---|---|
| SMOTE | `imbalanced-learn` | k-NN interpolation; no label required in research context |
| CTGAN | `ctgan` | Conditional tabular GAN |
| TVAE | `ctgan` | Tabular variational autoencoder |

Install optional dependencies:
```bash
pip install imbalanced-learn ctgan
```

### Running the comparison

```bash
# Quick comparison on a single synthetic dataset (fast)
python research/generation/compare.py

# Specific dataset from the benchmark suite
python research/generation/compare.py --dataset adult --n 2000

# All datasets, all methods, save results
python research/generation/compare.py --all --output research/generation/results/

# Include deep learning methods (requires ctgan)
python research/generation/compare.py --deep-learning
```

### Key findings

1. **Multivariate KDE** is the best general-purpose choice. It captures the full joint density with a single bandwidth parameter and achieves near-perfect tail preservation at `bandwidth=0.5`.

2. **Univariate KDE with rank coupling** is a strong alternative when the partition size is small (few samples → multivariate KDE may overfit). The Gaussian copula coupling preserves correlation structure without requiring a full d×d covariance estimate.

3. **Univariate KDE independent** performs well on marginal metrics but degrades on correlation fidelity — features are generated independently, so inter-feature relationships are lost.

4. **KNN interpolation (SMOTE-style)** generates samples that are convex combinations of existing points — by design, it cannot extrapolate beyond the observed range. Tail preservation suffers at aggressive expansion ratios.

5. **PartitionGMM** is competitive but slower than KDE and more sensitive to partition size (needs enough samples to estimate a full GMM).

6. **The bandwidth insight** (from empirical tuning): `bandwidth=0.5` is the optimal default for `MultivariateKDE`. Lower values (0.1–0.3) produce tighter marginals but undersample tails. Higher values (0.7–1.0) oversample tails at the expense of marginal fidelity.

---

## Folder Structure

```
research/
├── README.md
└── generation/
    ├── methods.py        # Implementations of all generation methods
    ├── compare.py        # Comparison runner (CLI + importable)
    └── results/          # Saved comparison outputs (git-ignored)
```
