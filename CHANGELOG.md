# Changelog

All notable changes to HVRT are documented here.

---

## [2.5.0] — 2026-02-26

### Added
- **Two-stage stateful generation strategies** (`StatefulGenerationStrategy` protocol).
  All four built-in strategies (`epanechnikov`, `bootstrap_noise`, `multivariate_kde`,
  `univariate_kde_copula`) now implement `prepare(X_z, partition_ids, unique_partitions)`
  (called once at fit-time, result cached) and `generate(context, budgets, random_state)`
  (called per expand, no recomputation of partition metadata).
  - `EpanechnikovStrategy`, `BootstrapNoiseStrategy`, `MultivariateKDEStrategy`,
    `UnivariateCopulaStrategy` classes exported from `hvrt`.
  - Frozen context dataclasses: `PartitionContext`, `EpanechnikovContext`,
    `BootstrapNoiseContext`, `MultivariateKDEContext`, `UnivariateCopulaContext`.
  - Generation internals fully vectorised — per-partition Python loops replaced with
    batch NumPy/linalg ops (`np.linalg.cholesky` on stacked covariance matrices,
    batch matmul for Cholesky noise, single RNG block per expand call).
  - Speedup vs old protocol: bootstrap 2–5×, multivariate KDE 1.5–4×,
    epanechnikov 1.3–3× (small-to-medium n).

- **Two-stage stateful selection strategies** (`StatefulSelectionStrategy` protocol).
  All four built-in strategies implement `prepare(X_z, partition_ids, unique_partitions)`
  and `select(context, budgets, random_state, n_jobs=1)`.
  - `StratifiedStrategy`, `VarianceOrderedStrategy`, `CentroidFPSStrategy`,
    `MedoidFPSStrategy` classes exported from `hvrt`.
  - `SelectionContext` frozen dataclass exported from `hvrt`.
  - `StratifiedStrategy` fully vectorised via `np.lexsort` — eliminates per-partition
    Python loop entirely. Consistent **2.5–3× speedup** across all dataset sizes.
  - FPS and `VarianceOrderedStrategy` retain joblib parallelism (`n_jobs`) alongside
    the two-stage protocol; both paths are active simultaneously.

- **Context caching** in `_HVRTBase`. Prepared contexts are cached per strategy
  instance (keyed by `id(strategy)`) and reused across repeated `expand()` /
  `reduce()` calls with the same data. Cache is invalidated on `fit()` / `_fit_tree()`.
  Eager preparation at the end of `fit()` when `expand_params` or `reduce_params`
  carry a strategy at construction time.

- **`HVRTOptimizer.objective` callable parameter** (default `None`).
  Accepts a per-fold scoring function that receives a metrics dict
  (`tstr`, `trtr`, `tstr_delta`, `X_synth`, `X_real`, `y_synth`, `y_real`,
  `fold`, `n_synth`) and must return a float to maximise. Enables weighted
  combinations of ML utility and privacy metrics. Defaults to TSTR Δ
  when `None` (no behavioural change for existing code).

### Changed
- Module-level strategy singletons (`epanechnikov`, `bootstrap_noise`,
  `multivariate_kde`, `univariate_kde_copula`, `centroid_fps`, `medoid_fps`,
  `variance_ordered`, `stratified`) are now instances of the corresponding
  strategy classes rather than plain functions.

### Deprecated
- Passing a **plain callable** as `generation_strategy` or `method` to
  `expand()` / `reduce()` now emits `HVRTDeprecationWarning`. Implement
  `StatefulGenerationStrategy` or `StatefulSelectionStrategy` instead.
  Plain callables remain fully functional; no removal planned for the v2.x line.

---

## [2.4.0] — 2026-02-25

### Changed
- **License changed from MIT to GNU Affero General Public License v3 or later (AGPL-3.0-or-later).**
  All source code, derivatives, and network-facing deployments must now be released under the same
  license terms. See [LICENSE](LICENSE) for the full text.

---

## [2.3.0] — 2026-02-24

### Added
- **`n_jobs` constructor parameter** on `HVRT` and `FastHVRT` (default `1`).
  Parallelises all partition-level work using `joblib`:
  - *Generation strategies* (`multivariate_kde`, `univariate_kde_copula`,
    `bootstrap_noise`, `epanechnikov`) and KDE fitting/sampling in `expand()`
    use **joblib threads** — scipy / NumPy routines release the GIL, so
    threads are sufficient and avoid process-spawn overhead.
  - *Selection strategies* (`centroid_fps`, `medoid_fps`, `variance_ordered`,
    `stratified`) use **joblib loky processes** — the inner FPS greedy loops
    are GIL-bound and require true multiprocessing.
  - Parallelism is skipped when fewer than 6 partition tasks exist (overhead
    would dominate) or when `n_jobs=1` (the default — zero behavioural change
    for existing code).
  - `joblib` is already a transitive dependency of `scikit-learn`; no new
    install-time dependency is introduced.

- **`_epanechnikov_partition` private helper** extracted from the inline body
  of `epanechnikov()`, enabling both serial and parallel dispatch.
- **`_fit_kde_partition` and `_sample_kde_partition_simple` private helpers**
  in `expand.py`, enabling parallel KDE fitting and sampling.
- **Per-partition dispatch helpers** in `reduction_strategies.py`
  (`_centroid_fps_partition`, `_medoid_fps_partition`,
  `_variance_ordered_partition`, `_stratified_partition`).
- **`benchmarks/speed_benchmark.py`**: wall-clock benchmark comparing
  `n_jobs=1` vs `n_jobs=-1` across four dataset sizes for all built-in
  reduction and generation strategies.  Includes a joblib pool warm-up
  step so one-time pool initialisation cost does not corrupt timed results.

### Changed
- **`_budgets._compute_weights` vectorised** — replaced per-partition Python
  loops (`for pid in unique_partitions: np.sum(partition_ids == pid)`) with
  `np.searchsorted` + `np.bincount`. O(n) single pass instead of
  O(n × n_partitions).
- **`reduce.compute_partition_budgets` vectorised** — same replacement for the
  `partition_sizes` loop.
- **`_base._resolve_bandwidth_auto` vectorised** — partition size mean now
  computed via `np.bincount` instead of a list comprehension with
  `np.sum(partition_ids == pid)` per partition.
- **`_base._adaptive_bandwidths` vectorised** — partition sizes precomputed
  with `np.bincount`; loop now indexes the result array instead of calling
  `np.sum` per partition.
- **`n_jobs` added to `select_from_partitions`** (keyword-only, default 1).
  Passed to built-in strategies that declare it; custom strategies with the
  standard 5-argument signature are called unchanged.
- **`medoid_fps` approximate medoid for large partitions.**  The exact medoid
  requires O(n²·d) pairwise distances.  When ``n_part > 200``, a centroid-
  nearest candidate set of size ``max(30, ⌊√n_part⌋)`` is selected in O(n·d)
  using ``np.argpartition``, and the exact medoid is found within that set in
  O(k²·d).  Total cost is O(n·d), a factor of O(n/k) ≈ O(√n) cheaper than
  the exact computation.  For HVRT's variance-structured partitions the true
  medoid lies in the centroid-nearest region with high probability, so
  approximation quality is effectively indistinguishable from exact.  Partitions
  with ``n_part ≤ 200`` continue to use the exact O(n²) computation.
- **Dual loky dispatch guards in `reduction_strategies._run_parallel`:**
  parallelism is skipped when fewer than 6 tasks *or* when total samples
  across all tasks is below 3 000, preventing loky IPC overhead from
  dominating on small or finely-partitioned datasets.
- **`stratified` and `epanechnikov` RNG path changed.** Both now derive a
  per-partition seed via `rng.randint(0, 2³¹)` and create an independent
  `np.random.RandomState(seed)` inside each partition worker. This enables
  parallel execution with deterministic per-partition output. As a result,
  calling either strategy with the same `random_state` as v2.2.0 will produce
  **different (but statistically equivalent) values**. Reproducibility
  *across calls with the same seed* is preserved.

### Performance
Benchmark on a 32-core machine (`n_jobs=-1` vs `n_jobs=1`):

Benchmark results (32-core machine, pools pre-warmed, `n_jobs=-1` vs `n_jobs=1`):

| Operation | n=1 000 | n=5 000 | n=20 000 | n=50 000 |
|---|---|---|---|---|
| `reduce(method='variance_ordered')` | 1.5× | 1.4× | **33×** | **48×** |
| `reduce(method='centroid_fps')` | ≈1× | ≈1× | ≈1× | 1.8× |
| `reduce(method='medoid_fps')` | ≈1× | ≈1× | 1.3× | 1.6× |
| `expand(generation_strategy='bootstrap_noise')` | overhead | overhead | 1.1× | 1.3× |

`variance_ordered` benefits most because `sklearn.neighbors.NearestNeighbors`
is independently fitted per partition and releases the GIL (enabling true
parallel CPU usage).  FPS strategies gain from loky's true multiprocessing at
large n where the greedy distance loop is the bottleneck.  For generation
strategies, the per-partition scipy/NumPy compute is already so fast that
thread overhead is comparable to the computation at typical HVRT partition
sizes; meaningful speedup appears only at very large expansion counts.

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
