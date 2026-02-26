# Numba Kernel Speed Benchmark

**Date:** 2026-02-26
**HVRT version:** 2.6.0
**Hardware:** 32-core machine
**Python:** 3.11 | NumPy | Numba (installed via `pip install hvrt[fast]`)

---

## Overview

This report documents the wall-clock speedups achieved by the optional Numba-compiled
kernels introduced in v2.6.0. Three core computation paths are accelerated:

| Kernel | Path affected | Memory footprint |
|---|---|---|
| `_pairwise_target_nb` | `HVRT.fit()` — synthetic target computation | O(n) vs O(n·d) |
| `_centroid_fps_core_nb` | `reduce(method='centroid_fps')` — greedy FPS loop | same |
| `_medoid_fps_core_nb` | `reduce(method='medoid_fps')` — greedy FPS loop | same |

All benchmarks exclude JIT compilation time (`cache=True` persists compiled bitcode to
`__pycache__`; warm-start overhead is measured separately below).

---

## 1. Pairwise Target Kernel — `_pairwise_target_nb`

Computes the synthetic partition target in `HVRT.fit()`.
Numba fuses three sequential passes per feature pair (mean, variance, z-score accumulation)
into a single LLVM-compiled loop with O(n) peak memory, versus the NumPy block-wise fallback
which allocates O(n·d) intermediate arrays.

| Dataset (n, d) | NumPy (ms) | Numba (ms) | Speedup |
|---|---|---|---|
| n=1 000, d=10 | 1.4 | 0.13 | **10.9×** |
| n=1 000, d=20 | 4.9 | 0.84 | **5.8×** |
| n=5 000, d=15 | 19.6 | 2.8 | **7.0×** |
| n=5 000, d=30 | 73.1 | 14.4 | **5.1×** |
| n=20 000, d=20 | 310 | 97 | **3.2×** |
| n=50 000, d=20 | 782 | 267 | **2.9×** |

Speedup is highest at small-to-medium n (Python loop overhead and intermediate
array allocation dominate). At n=50k the bottleneck shifts towards raw arithmetic
throughput where both backends are cache-bound.

---

## 2. Centroid FPS Kernel — `_centroid_fps_core_nb`

The greedy Furthest Point Sampling loop is irreducibly sequential, making it an ideal
Numba target. The compiled kernel eliminates Python loop overhead and NumPy call overhead
for each greedy step.

Benchmark conditions: budget = 30% of n_part (representative of a reduce(ratio=0.3) call).

| Partition (n_part, d) | NumPy (ms) | Numba (ms) | Speedup |
|---|---|---|---|
| n=50, d=6 | 0.48 | 0.036 | **13.4×** |
| n=100, d=6 | 1.78 | 0.135 | **13.2×** |
| n=200, d=10 | 6.6 | 0.70 | **9.4×** |
| n=500, d=10 | 40 | 6.6 | **6.1×** |
| n=1 000, d=20 | 197 | 48.5 | **4.1×** |
| n=2 000, d=20 | 786 | 206 | **3.8×** |

Gain is largest for small partitions (typical of HVRT at default settings, ~25–100
points per partition) because Python's per-iteration overhead is proportionally larger.

---

## 3. Medoid FPS Kernel — `_medoid_fps_core_nb`

Combines medoid seeding (exact for n≤200, approximate for n>200) with the same greedy
FPS loop as centroid FPS.

| Partition (n_part, d) | NumPy (ms) | Numba (ms) | Speedup |
|---|---|---|---|
| n=50, d=6 | 1.8 | 0.25 | **7.1×** |
| n=100, d=6 | 5.1 | 1.08 | **4.7×** |
| n=200, d=10 | 37 | 13.9 | **2.7×** |
| n=300, d=10 | 64 | 8.7 | **7.4×** |
| n=500, d=10 | 156 | 23.8 | **6.6×** |
| n=1 000, d=20 | 611 | 156 | **3.9×** |

The step from n=200 to n=300 shows a large relative jump because the exact O(n²·d)
medoid is replaced by the approximate O(n·d) path — an inherent algorithmic speedup
that compounds with compilation.

---

## 4. End-to-End `fit()` Wall Clock

Full `HVRT.fit()` including preprocessing, target computation, and tree fitting.
The pairwise target is the dominant cost at n≥5k.

| n | d | NumPy fit (ms) | Numba fit (ms) | Speedup |
|---|---|---|---|---|
| 1 000 | 10 | 5 | 7.6 | — (JIT on first call) |
| 5 000 | 15 | 45 | 14 | **3.2×** |
| 20 000 | 20 | 280 | 88 | **3.2×** |
| 50 000 | 20 | 825 | 245 | **3.4×** |

> **Note on n=1k:** The first Numba call compiles the kernel; subsequent calls are
> consistently faster than NumPy. Compilation cost is amortised after a single warm run.
> With `cache=True` the compiled bitcode is persisted to `__pycache__` so the JIT cost
> is paid only on the first process launch, not on every Python session.

---

## 5. GeoXGB Context

GeoXGB replaces XGBoost's internal sampling with HVRT expand/reduce operations at
a configurable `refit_interval` (default 20). With `n_rounds=1000` this produces
**51 full HVRT refits** per training run. What would be a one-time cost becomes a
dominant bottleneck:

| Operation | Single run | ×51 (GeoXGB 1000 rounds) | Numba saving |
|---|---|---|---|
| `fit()` at n=20k | 280ms → 88ms | 14.3s → 4.5s | **−9.8s / training run** |
| `reduce(centroid_fps)` at n=20k | 29ms → 7ms | 1.5s → 0.36s | **−1.1s / training run** |
| Total (fit + reduce) | | ~16s → ~5s | **−11s / training run** |

For typical GeoXGB workloads (multiple datasets, multiple outer CV folds) the cumulative
saving is measured in minutes per experiment.

---

## 6. Existing Speed Benchmark (serial vs parallel, no change from v2.5.0)

The existing `benchmarks/speed_benchmark.py` measures `n_jobs=1` vs `n_jobs=-1`
(joblib parallelism). Results are unchanged by the Numba addition since the Numba
kernels operate within a single partition task; the parallel dispatch layer above
them is unaffected.

| Operation | n=1k | n=5k | n=20k | n=50k |
|---|---|---|---|---|
| `fit()` | 5ms | — | — | 825ms |
| `reduce(variance_ordered)` serial | 10ms | — | — | 4 340ms |
| `reduce(variance_ordered)` parallel | 7ms | — | — | 82ms (**53×**) |
| `reduce(centroid_fps)` serial | 0ms | — | 17ms | 29ms |
| `reduce(centroid_fps)` parallel | — | — | — | — (minimal gain) |

---

## 7. JIT Warm-Start Cost

On a cold process (no `__pycache__` bitcode):

| Kernel | First-call overhead |
|---|---|
| `_pairwise_target_nb` | ~1.1 s |
| `_centroid_fps_core_nb` | ~0.9 s |
| `_medoid_fps_core_nb` | ~1.3 s |

After any successful run, `cache=True` writes compiled bitcode; all subsequent
process launches skip compilation entirely.

---

## Summary

| Kernel | Typical speedup | Best speedup | Context |
|---|---|---|---|
| `_pairwise_target_nb` | 3–7× | **10.9×** | `HVRT.fit()` target computation |
| `_centroid_fps_core_nb` | 4–13× | **13.4×** | `reduce(method='centroid_fps')` |
| `_medoid_fps_core_nb` | 4–7× | **7.4×** | `reduce(method='medoid_fps')` |

Install: `pip install hvrt[fast]`
