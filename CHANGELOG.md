# Changelog

All notable changes to HVRT are documented here.

---

## [2.2.0] — 2026-02-24

### Added
- **`HVRTOptimizer`**: Optuna-backed hyperparameter optimiser for HVRT.
  Searches over `n_partitions`, `min_samples_leaf`, `y_weight`, kernel /
  bandwidth, and `variance_weighted` using TPE sampling. Objective is mean
  TSTR Δ (train-on-synthetic minus train-on-real) across CV folds, with TRTR
  pre-computed once per fold to halve GBM fitting overhead. Exposes `expand()`
  and `augment()` methods that delegate to the best fitted model; both strip
  the appended y column automatically so output always matches training-X shape.
- **`optimizer` optional extra**: `pip install hvrt[optimizer]` installs
  `optuna>=3.0.0`. The import guard is lazy — `from hvrt import HVRTOptimizer`
  always succeeds regardless of whether optuna is present.
- **HPO benchmark** (`benchmarks/hpo_benchmark.py`): evaluates `HVRTOptimizer`
  vs. HVRT defaults using a nested-CV protocol (outer CV for evaluation, inner
  HPO on the training split only) across 6 benchmark datasets, reporting
  TSTR Δ, disc_err, Wasserstein-1, and Corr.MAE.

### Fixed
- **`HVRTOptimizer` objective: classification y-snapping.** KDE generates
  continuous synthetic y values; `GradientBoostingClassifier` requires discrete
  class labels. The objective now snaps KDE-generated y to the nearest observed
  class label before downstream TSTR scoring, preventing every classification
  trial from returning `float('-inf')`.
- **`HVRTOptimizer` warm-start guarantee.** The HVRT default configuration
  (`n_partitions=auto`, `min_samples_leaf=auto`, `y_weight=0.0`,
  `kernel=auto`, `variance_weighted=False`) is always enqueued as trial 0
  via `study.enqueue_trial()` so that HPO is guaranteed to find at least as
  good a result as the untuned defaults.

---

## [2.1.2] — 2026-02-22

### Changed
- **Auto-tuner: data-informed minimum partition sample count for KDE generation.**
  The expansion/augmentation branch of `auto_tune_tree_params()` previously used
  a feature-agnostic formula (`max(5, 0.75·n^(2/3))`), which a dedicated
  benchmarking study (`benchmarks/auto_tuner/min_samples_study.py`) showed
  causes KDE failures in 20–100 % of partitions for small-n / high-d datasets
  (e.g. n=50 d=15 → 80 % failure; n=50 d=20 → 100 % failure).
  The formula is now:

  ```
  min_samples_leaf = max(n_features + 2, int(n_samples ** 0.5))
  ```

  The feature-aware floor (`n_features + 2`) guarantees a non-singular
  covariance matrix for `scipy.stats.gaussian_kde` in every partition across
  all (n, d) combinations. The `sqrt(n)` base grows slower than the previous
  `n^(2/3)`, producing finer partitions that better capture local density
  structure. Benchmark results: zero KDE failures, zero at-risk partitions,
  and best mean Wasserstein distance (0.142) across 700 evaluated
  (n, d, formula, seed) combinations.

### Fixed
- **Reduce: `reduce(ratio=1.0)` now correctly returns all samples.**
  Variance-weighted budget allocation could assign more budget to a partition
  than its actual sample count. The FPS strategy caps selection at partition
  size, causing the total returned to silently fall short of n_target.
  `compute_partition_budgets()` now clips each budget to its partition size
  and greedily redistributes any shortfall, guaranteeing the requested count
  is always met.

- **Expand: KDE fitting no longer raises on small external-X partitions.**
  `fit_partition_kdes()` previously only caught `np.linalg.LinAlgError` for
  degenerate partitions. Recent scipy versions raise `ValueError` when
  `n_part ≤ d`. Both exception types are now caught and the affected partition
  gracefully falls back to bootstrap-noise. An explicit pre-check
  (`n_part ≤ n_features`) short-circuits the KDE attempt before scipy is
  called, making the behaviour consistent across scipy versions.

---

## [2.1.1] — 2025

### Fixed
- **Auto-tuner: separated reduction and generation minimum sample constraints.**
  The 40:1 sample-to-feature ratio for `min_samples_leaf` was originally
  applied uniformly to both reduction and expansion. This over-constrained
  generation, producing too few, too-coarse partitions and degrading KDE
  quality for expansion/augmentation use cases. The auto-tuner now applies
  the 40:1 ratio only when `is_reduction=True` (reduction pipeline), and uses
  a looser dataset-size-driven formula for expansion and augmentation.

---

## [2.1.0] — 2025

### Added
- **Dual model classes**: `HVRT` (pairwise feature interactions, O(n·d²))
  and `FastHVRT` (z-score sum, O(n·d)); both expose the same primary API.
- **`expand()`**: per-partition multivariate KDE synthetic data generation
  with pluggable generation strategies and adaptive bandwidth scaling.
- **`augment()`**: returns original data concatenated with synthetic samples
  to reach a target total count.
- **Pipeline API**: `ReduceParams`, `ExpandParams`, `AugmentParams` dataclasses
  declare operation parameters at construction for sklearn `Pipeline` use;
  `fit_transform()` dispatches accordingly. `hvrt.pipeline` re-export package.
- **Selection strategies** (reduce): `centroid_fps` / `fps`, `medoid_fps`,
  `variance_ordered`, `stratified`; callable protocol for custom strategies.
- **Generation strategies** (expand): `multivariate_kde`, `univariate_kde_copula`,
  `bootstrap_noise`; callable protocol for custom strategies.
- **External X**: `reduce(X=X_test)` and `expand(X=X_test)` operate on data
  other than the training set using the fitted tree.
- **Categorical features**: LabelEncoder + StandardScaler; expansion samples
  from per-partition empirical distributions.
- **Utility methods**: `get_partitions()`, `compute_novelty()`,
  `recommend_params()`.
- Full benchmark suite: reduction (GBM ROC-AUC across 7 structured scenarios)
  and expansion (discriminator accuracy, marginal fidelity, tail MSE vs
  Gaussian Copula, GMM, Bootstrap, SMOTE, CTGAN, TVAE, TabDDPM, MOSTLY AI).

### Changed
- Tree fitted once in `fit()`; all operations reuse it. Per-call `n_partitions`
  override re-fits temporarily without a new `fit()` call.
- Major structural refactor: `_params.py`, `_preprocessing.py`,
  `_partitioning.py`, `_budgets.py`, `model/`, `legacy/`, `pipeline/`.
  `utils.py` deleted; presets removed; `_base.py` reduced from ~1 067 to
  ~520 lines.
- `mode=` parameter deprecated (`HVRTDeprecationWarning`); use
  `reduce_params` / `expand_params` / `augment_params` instead.

### Deprecated
- `HVRTSampleReducer` and `AdaptiveHVRTReducer` moved to `hvrt.legacy`;
  still importable from `hvrt` directly for backward compatibility.

---

## [2.0.0] — 2024

### Added
- **Synthetic data generation**: HVRT now supports sample expansion in
  addition to reduction, enabling full sample count control over tabular
  datasets.
- Renamed to reflect the expanded scope: *Hierarchical Variance-Retaining
  Transformer*.

### Deprecated
- v0.x / v1.x reduction-only methods deprecated; scheduled for removal.

---

## [0.1.0] — 2024

### Added
- Initial release: variance-aware **sample reduction** for tabular data.
- Decision-tree partitioning driven by a synthetic variance signal isolates
  structural outliers into dedicated leaves.
- Furthest Point Sampling within each partition preserves geometric diversity.
- Published to PyPI as `hvrt`.
