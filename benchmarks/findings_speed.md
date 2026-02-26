# HVRT Strategy Vectorization — Speed Findings

**Date:** 2026-02-26 (selection strategy section added 2026-02-26)
**Benchmark:** `benchmarks/strategy_speedup_benchmark.py`, `benchmarks/speed_benchmark.py`
**Changes:** v2.4 two-stage stateful generation protocol (vectorized NumPy) vs v2.3 per-partition callable loop;
selection strategy two-stage protocol (StatefulSelectionStrategy) added post-v2.4
**Machine:** 32 logical CPU cores

---

## Summary

### Generation strategies (v2.4)

The v2.4 refactor replaces per-partition Python loops in all four generation
strategies with fully vectorized NumPy operations using a precomputed frozen
context (two-stage `prepare()` + `generate()` protocol).  Results are mixed
and depend on dataset size and strategy:

- **Bootstrap** benefits most consistently: **2–5× faster** across all sizes.
- **Multivariate KDE** faster at small-to-medium data (**4× at 1K, 1.5× at 20K**);
  roughly equal at 50K due to large temporary array allocation.
- **Epanechnikov** faster at small data (**3× at 1K, 1.3× at 5K**); slightly
  **slower at 50K** (1.1× regression) due to memory pressure from large
  `(total_budget, d)` uniform arrays.
- **Context caching** gives an additional **1.1–3× speedup** on the second and
  subsequent `expand()` calls when using training-data partitions.
- **n_jobs parallelism** for generation is now effectively a no-op (1.0–1.65×);
  vectorized NumPy already uses internal CPU parallelism (BLAS/MKL).

The typical synthetic data use case — small training sets (n ≤ 5K) expanded
5–10× — sees the largest relative benefit.

### Selection strategies (post-v2.4)

The same two-stage stateful protocol (`StatefulSelectionStrategy`: `prepare()` +
`select()`) has been applied to all four selection strategies:

- **`StratifiedStrategy`** is fully vectorized: a single `np.lexsort` over all
  samples replaces the per-partition `rng.choice()` loop.  **2.5–3× faster**
  across all sizes (consistent, no memory-pressure regression).
- **FPS strategies** (`CentroidFPS`, `MedoidFPS`): the greedy sequential loop is
  irreducibly serial within each partition; the new protocol adds only partition
  metadata caching and faster slicing.  Speedup is modest (~1.2× at 50K).
- **`VarianceOrderedStrategy`**: per-partition sklearn k-NN retained; the protocol
  provides caching and clean `n_jobs` dispatch.  Speedup proportional to `n_jobs`.
- **Context caching** (SelectionContext): `StratifiedStrategy` gains **1.2–1.5×**
  from caching the argsort in `prepare()`; FPS gains minimal (dominated by greedy
  loop cost).
- **`n_jobs` parallelism** for FPS and variance_ordered remains highly valuable
  (23–37× for `variance_ordered` at large n); the model's `n_jobs` is now
  forwarded to `select()` automatically.
- **Both vectorization AND parallelism** are available simultaneously:
  `StratifiedStrategy` uses vectorized NumPy (ignores `n_jobs`);
  FPS/VarianceOrdered use `_run_parallel` with `n_jobs` from the model.

---

## Section 1 — Old callable loop vs new stateful protocol

*Methodology:* old strategy = per-partition Python loop matching v2.3 exactly;
new strategy = `prepare()` once, then `generate()` at steady-state.
`expand_n = 5 × n`, `d = 10`. Times are median of 3–5 repeats.

### Epanechnikov

| Dataset | Old (loop) | prepare() | generate() | Gen speedup | Full speedup |
|---------|-----------|-----------|-----------|-------------|--------------|
| 1K  | 4.8 ms  | 0.1 ms | 1.6 ms  | **3.0× faster** | **3.2× faster** |
| 5K  | 15.5 ms | 0.4 ms | 12.0 ms | **1.3× faster** | **1.3× faster** |
| 20K | 45.1 ms | 1.6 ms | 44.8 ms | 1.0× (even)     | 1.0× (even)     |
| 50K | 112 ms  | 6.2 ms | 118 ms  | 1.1× **slower** | 1.3× **slower** |

Epanechnikov regression at large n is explained by the U1/U2/U3 allocation:
with 250K budget and d=10, three `(250K, 10)` float64 arrays = 60 MB of
temporaries, creating memory allocation and cache pressure that outweighs
the eliminated Python loop overhead.

### Bootstrap Noise

| Dataset | Old (loop) | prepare() | generate() | Gen speedup | Full speedup |
|---------|-----------|-----------|-----------|-------------|--------------|
| 1K  | 4.1 ms  | 0.1 ms | 0.8 ms  | **5.1× faster** | **4.2× faster** |
| 5K  | 13.2 ms | 0.4 ms | 6.5 ms  | **2.0× faster** | **2.0× faster** |
| 20K | 50.9 ms | 1.8 ms | 34.1 ms | **1.5× faster** | **1.6× faster** |
| 50K | 165 ms  | 6.0 ms | 72.2 ms | **2.3× faster** | **1.9× faster** |

Bootstrap consistently improves because `generate()` is arithmetically simple
(resample + scale + add) and the vectorized form avoids the per-partition loop
overhead entirely. At 50K the old loop paid substantial Python overhead per
partition while the new code does a single `rng.standard_normal((250K, 10))`
call.

### Multivariate KDE

| Dataset | Old (loop) | prepare() | generate() | Gen speedup | Full speedup |
|---------|-----------|-----------|-----------|-------------|--------------|
| 1K  | 8.3 ms  | 0.5 ms | 2.0 ms  | **4.1× faster** | **2.9× faster** |
| 5K  | 24.0 ms | 1.2 ms | 12.7 ms | **1.9× faster** | **1.7× faster** |
| 20K | 91.8 ms | 3.4 ms | 61.6 ms | **1.5× faster** | **1.5× faster** |
| 50K | 159 ms  | 11.4 ms | 164 ms  | 1.0× (even)    | 1.2× **slower** |

Old KDE paid scipy fitting overhead each expand() call; the new version fits
Cholesky factors once in `prepare()`. At 50K the batch matmul
`L_all[labels] @ z` expands to `(250K, 10, 10) @ (250K, 10, 1)` — a 200 MB
temporary array — making it roughly break-even with scipy's per-partition
LAPACK calls.

---

## Section 2 — Context caching speedup

*First `expand()` call pays `prepare()` cost; subsequent calls skip it.*
*Measured via `model._strategy_context_cache_`.  `expand_n = 5 × n`.*

| Dataset | Strategy | 1st call | Nth call | Cache speedup |
|---------|----------|----------|----------|---------------|
| 1K  | Epanechnikov | 3.5 ms | 1.8 ms | **1.9× faster** |
| 1K  | Bootstrap    | 3.2 ms | 1.1 ms | **3.0× faster** |
| 1K  | KDE          | 4.6 ms | 2.2 ms | **2.2× faster** |
| 5K  | Epanechnikov | 15.8 ms | 14.1 ms | 1.1× faster |
| 5K  | Bootstrap    | 20.5 ms | 10.4 ms | **2.0× faster** |
| 5K  | KDE          | 18.6 ms | 12.5 ms | **1.5× faster** |
| 20K | Epanechnikov | 55.7 ms | 46.0 ms | 1.2× faster |
| 20K | Bootstrap    | 41.8 ms | 28.8 ms | 1.4× faster |
| 20K | KDE          | 85.3 ms | 67.5 ms | 1.3× faster |
| 50K | Epanechnikov | 140 ms  | 122 ms  | 1.1× faster |
| 50K | Bootstrap    | 114 ms  | 109 ms  | 1.0× faster |
| 50K | KDE          | 170 ms  | 134 ms  | 1.3× faster |

Caching is most valuable for Bootstrap and KDE at small-to-medium n, where
`prepare()` is non-trivial relative to `generate()`.  For Epanechnikov the
`prepare()` cost (only std computation via bincount) is very small; caching
gains are correspondingly modest.

Eager preparation at `fit()` time (when `expand_params.generation_strategy`
is declared) means the first `expand()` call after `fit()` is already cached.

---

## Section 3 — Steady-state throughput

*Pure `generate()` time; context already cached.  `expand_n = 5 × n`, `d = 10`.*

| Dataset | n_parts | Strategy | generate() | Throughput |
|---------|---------|----------|-----------|------------|
| 1K  | 25  | Epanechnikov | 1.0 ms  | **5.0M samples/s** |
| 1K  | 25  | Bootstrap    | 0.8 ms  | **5.9M samples/s** |
| 1K  | 25  | KDE          | 1.8 ms  | 2.8M samples/s     |
| 5K  | 56  | Epanechnikov | 9.4 ms  | 2.6M samples/s     |
| 5K  | 56  | Bootstrap    | 5.8 ms  | **4.3M samples/s** |
| 5K  | 56  | KDE          | 11.0 ms | 2.3M samples/s     |
| 20K | 107 | Epanechnikov | 37.3 ms | 2.7M samples/s     |
| 20K | 107 | Bootstrap    | 26.5 ms | **3.8M samples/s** |
| 20K | 107 | KDE          | 46.6 ms | 2.1M samples/s     |
| 50K | 171 | Epanechnikov | 137 ms  | 1.8M samples/s     |
| 50K | 171 | Bootstrap    | 63.0 ms | **4.0M samples/s** |
| 50K | 171 | KDE          | 122 ms  | 2.0M samples/s     |

Bootstrap is fastest at all sizes (simple arithmetic; no matrix ops).
Epanechnikov throughput degrades at very large budgets (memory pressure).
KDE is most consistent (batch Cholesky + matmul scales smoothly).

---

## Section 4 — n_jobs parallelism for generation (now effectively a no-op)

From `speed_benchmark.py` (`d = 20`):

| Dataset | Strategy | n_jobs=1 | n_jobs=-1 | Speedup |
|---------|----------|---------|----------|---------|
| small (1K)  | KDE       | 3 ms  | 3 ms  | 1.11× |
| small (1K)  | Epan.     | 2 ms  | 2 ms  | 1.02× |
| small (1K)  | Bootstrap | 2 ms  | 1 ms  | 1.65× |
| medium (5K) | KDE       | 22 ms | 21 ms | 1.06× |
| medium (5K) | Epan.     | 20 ms | 16 ms | 1.20× |
| medium (5K) | Bootstrap | 16 ms | 10 ms | 1.56× |
| large (20K) | KDE       | 156 ms| 190 ms| 0.82× |
| large (20K) | Epan.     | 85 ms | 84 ms | 1.02× |
| large (20K) | Bootstrap | 60 ms | 63 ms | 0.95× |
| xlarge (50K)| KDE       | 463 ms| 444 ms| 1.04× |
| xlarge (50K)| Epan.     | 252 ms| 253 ms| 1.00× |
| xlarge (50K)| Bootstrap | 168 ms| 192 ms| 0.88× |

Joblib parallelism for generation strategies is now effectively useless (1.0×
on average; can even be slightly negative at large sizes due to thread
synchronization overhead). This is the expected consequence of vectorization:
NumPy's internal BLAS/MKL routines already dispatch across all available cores.
Setting `n_jobs > 1` is harmless (the parameter is silently ignored for
generation now that `n_jobs` is not passed to vectorized `generate()`) but
provides no benefit.

For **reduction strategies** n_jobs parallelism remains highly valuable —
see `speed_benchmark.py` results for selection strategies (variance_ordered:
23× at 20K, 37× at 50K).

---

---

## Section 5 — Selection strategy vectorization (StratifiedStrategy)

*Methodology:* old = per-partition `rng.choice()` loop matching previous implementation;
new (vectorized) = `np.lexsort` over all samples, single `rng.random(n)` call;
new (cached ctx) = `select()` only, context pre-built once.
`reduce_ratio = 0.30`, `d = 8`. Times are median of 20 repeats.

| Dataset | n_parts | Old (loop) | New (vectorized, fresh ctx) | Speedup | New (cached ctx) | Cache gain |
|---------|---------|-----------|---------------------------|---------|-----------------|------------|
| 1K  | 24  | 0.39 ms | 0.19 ms | **2.5×** | 0.15 ms | 1.2× |
| 5K  | 55  | 1.34 ms | 0.86 ms | **1.6×** | 0.54 ms | **1.5×** |
| 20K | 105 | 6.03 ms | 3.84 ms | **1.6×** | 2.37 ms | **1.5×** |
| 50K | 176 | 21.87 ms | 10.48 ms | **2.1×** | 6.83 ms | **1.5×** |

**Overall speedup (vectorized + cached context vs old loop): 2.5–3.1×.**

The speedup comes from three sources:
1. Single `rng.random(n)` call instead of n_parts separate `rng.choice()` calls.
2. `np.lexsort` (O(n log n), C-level) instead of per-partition `argsort` + Python loop overhead.
3. Cached context eliminates the O(n log n) `argsort` in `prepare()` from the hot path.

### FPS strategies — caching overhead vs speedup

| Dataset | n_parts | Old (loop) | New (fresh ctx) | New (cached ctx) | Cache gain vs old |
|---------|---------|-----------|-----------------|-----------------|------------------|
| 1K  | 24  | 1.8 ms  | 1.8 ms  | 1.6 ms  | 1.1× |
| 5K  | 55  | 10.1 ms | 10.2 ms | 10.0 ms | 1.0× |
| 20K | 105 | 57.8 ms | 47.6 ms | 48.8 ms | 1.2× |
| 50K | 176 | 222.5 ms | 185.1 ms | 180.9 ms | 1.2× |

FPS (centroid/medoid) benefits only modestly: the O(budget²) greedy loop
dominates runtime.  The 10–20% improvement at large n comes from better
partition slicing (`sort_idx` vs `_iter_partitions` linear scan), not from
eliminating any inner computation.

**Key insight:** For FPS strategies the `n_jobs` parallelism is far more
impactful than the stateful protocol speedup.  Use `HVRT(n_jobs=-1)` for
large reduction workloads.

---

## Interpretation and Recommendations

### When does vectorization help most?

The primary HVRT use case — generating synthetic data from a small training
set (n ≤ 5K) expanded 5–10× — benefits consistently:

- **Bootstrap**: 2–5× faster at all sizes.  Recommended for any repeated-expand
  pipeline (e.g., cross-validation with synthetic augmentation).
- **KDE**: 2–4× faster at n ≤ 5K.  Still the default strategy via `bandwidth='auto'`
  when partitions are large enough.
- **Epanechnikov**: 1.3–3× faster at n ≤ 5K.  Default via `bandwidth='auto'`
  when partitions are small; no regression at these sizes.

### When does vectorization have diminishing returns?

At very large budgets (n ≥ 20K training, ≥ 100K generated):

- **Epanechnikov** can be slightly slower (1.1×) due to large temporary array
  allocation: three `(total_budget, d)` float64 arrays for the Ahrens-Dieter
  kernel.  Mitigation: `bandwidth='auto'` may select KDE over Epanechnikov at
  large partition sizes anyway.
- **KDE** is break-even at 50K because the batch matmul `L[labels] @ z` creates
  a `(total_budget, d, d)` array.  For `d ≤ 10` and budget ≤ 250K this stays
  below 200 MB and is manageable.  For higher `d`, `bootstrap_noise` or
  `epanechnikov` are better choices at large n.

### n_jobs recommendation update

Setting `n_jobs > 1` for generation strategies is now a no-op.  The `n_jobs`
constructor parameter remains supported and still parallelises **reduction**
strategies (which use loky process pools, not vectorized NumPy).  For
mixed pipelines (augment or reduce-then-expand), `n_jobs > 1` is still
valuable for the reduction step.

### Caching recommendation

For pipelines that call `expand()` multiple times on the same fitted model
(e.g., cross-validation, hyperparameter search), declare the strategy at
construction time:

```python
from hvrt import HVRT, ExpandParams
model = HVRT(expand_params=ExpandParams(n=50000, generation_strategy='bootstrap_noise'))
model.fit(X)
# Context is prepared eagerly at fit() time — all expand() calls are cached
for _ in range(10):
    X_synth = model.expand(n=50000, generation_strategy='bootstrap_noise')
```

Or use the strategy object reference to share the cache:

```python
from hvrt.generation_strategies import bootstrap_noise
for fold in folds:
    model.fit(X_train)
    X_synth = model.expand(n=50000, generation_strategy=bootstrap_noise)
    # 2nd+ expand() calls per fit use cached context
```

---

## Raw Numbers — `speed_benchmark.py` (full output)

```
HVRT Speed Benchmark — 32 logical CPU cores

[fit() — tree fitting, always serial via sklearn]
  Dataset       n     d    fit time
  small     1,000    10      8ms
  medium    5,000    15     56ms
  large    20,000    20    467ms
  xlarge   50,000    20   1.07s

[reduce() — selection strategies]
  Dataset   strategy         serial    parallel    speedup
  small     centroid_fps       1ms        1ms       1.07×
  small     medoid_fps         2ms        2ms       1.06×
  small     var_ordered       13ms       11ms       1.16×
  medium    centroid_fps       8ms       25ms       0.31×
  medium    medoid_fps        11ms       24ms       0.48×
  medium    var_ordered       39ms       25ms       1.52×
  large     centroid_fps      65ms       52ms       1.25×
  large     medoid_fps        79ms       52ms       1.50×
  large     var_ordered     1.56s        66ms      23.74×
  xlarge    centroid_fps     147ms       82ms       1.78×
  xlarge    medoid_fps       147ms       82ms       1.79×
  xlarge    var_ordered     4.06s       108ms      37.68×

[expand() — generation strategies]
  Dataset   strategy         serial    parallel    speedup
  small     KDE                3ms        3ms       1.11×
  small     Epan.              2ms        2ms       1.02×
  small     Bootstrap          2ms        1ms       1.65×
  medium    KDE               22ms       21ms       1.06×
  medium    Epan.             20ms       16ms       1.20×
  medium    Bootstrap         16ms       10ms       1.56×
  large     KDE              156ms      190ms       0.82×
  large     Epan.             85ms       84ms       1.02×
  large     Bootstrap         60ms       63ms       0.95×
  xlarge    KDE              463ms      444ms       1.04×
  xlarge    Epan.            252ms      253ms       1.00×
  xlarge    Bootstrap        168ms      192ms       0.88×
```
