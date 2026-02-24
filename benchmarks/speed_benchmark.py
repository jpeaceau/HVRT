"""
Speed benchmark: serial (n_jobs=1) vs parallel (n_jobs=-1).

Measures wall-clock time for fit(), reduce(), and expand() across several
dataset sizes.  Prints a summary table with absolute times and speedup ratios.

Usage
-----
    python benchmarks/speed_benchmark.py
"""

from __future__ import annotations

import time
import multiprocessing
import numpy as np
from hvrt import HVRT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(n: int, d: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, d)


def time_call(fn, repeats: int = 3) -> float:
    """Return the median wall-clock time (seconds) over ``repeats`` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def fmt(t: float) -> str:
    if t >= 1.0:
        return f"{t:6.2f}s"
    return f"{t * 1000:5.0f}ms"


def speedup(serial: float, parallel: float) -> str:
    if parallel == 0:
        return "  —  "
    r = serial / parallel
    return f"{r:5.2f}×"


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

CONFIGS = [
    # (label,         n,      d,   n_partitions,  expand_n,  repeats)
    ("small   ",    1_000,  10,   None,          5_000,     5),
    ("medium  ",    5_000,  15,   None,          25_000,    3),
    ("large   ",   20_000,  20,   None,          100_000,   3),
    ("xlarge  ",   50_000,  20,   None,          250_000,   2),
]

STRATEGIES = [
    ("multivariate_kde", "KDE      "),
    ("epanechnikov",     "Epan.    "),
    ("bootstrap_noise",  "Bootstrap"),
]

REDUCE_METHODS = [
    ("fps",              "centroid_fps "),
    ("medoid_fps",       "medoid_fps   "),
    ("variance_ordered", "var_ordered  "),
]


def run_reduce_bench(X: np.ndarray, n_jobs: int, method: str, n_partitions) -> float:
    model = HVRT(n_jobs=n_jobs, random_state=42).fit(X)
    n_target = max(10, len(X) // 5)
    return time_call(lambda: model.reduce(n=n_target, method=method), repeats=3)


def run_expand_bench(X: np.ndarray, n_jobs: int, strategy: str, expand_n: int, repeats: int) -> float:
    model = HVRT(n_jobs=n_jobs, bandwidth=0.1, random_state=42).fit(X)
    return time_call(lambda: model.expand(n=expand_n, generation_strategy=strategy), repeats=repeats)


def run_fit_bench(X: np.ndarray, repeats: int) -> float:
    return time_call(lambda: HVRT(random_state=42).fit(X), repeats=repeats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _warmup_pools():
    """
    Prime the joblib loky worker pool and the thread pool before any timed
    section.  Both pools are lazily created on first use; without a warm-up
    the first parallel benchmark call absorbs 1–2 s of pool-init overhead,
    distorting the "small" dataset results.
    """
    from joblib import Parallel, delayed
    # Loky (process pool) — used by selection strategies
    Parallel(n_jobs=-1)(delayed(abs)(i) for i in range(8))
    # Thread pool — used by generation strategies and KDE fitting
    Parallel(n_jobs=-1, prefer='threads')(delayed(abs)(i) for i in range(8))


def main():
    n_cores = multiprocessing.cpu_count()
    print(f"\nHVRT Speed Benchmark  — {n_cores} logical CPU cores")
    print("=" * 80)

    print("\nWarming up joblib worker pools (one-time cost, not benchmarked)...")
    _warmup_pools()
    print("Done.\n")

    # --- Fit timing (tree fit is single-threaded sklearn; shown for reference) ---
    print("\n[fit() — tree fitting, always serial via sklearn]\n")
    print(f"  {'Dataset':<10}  {'n':>6}  {'d':>3}  {'fit time':>10}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*3}  {'-'*10}")
    for label, n, d, n_part, expand_n, reps in CONFIGS:
        X = make_data(n, d)
        t = run_fit_bench(X, reps)
        print(f"  {label}  {n:>6,}  {d:>3}  {fmt(t):>10}")

    # --- Reduce timing ---
    print("\n[reduce() — selection strategies]\n")
    header = f"  {'Dataset':<10}  {'strategy':<15}  {'serial':>10}  {'parallel':>10}  {'speedup':>8}"
    print(header)
    print(f"  {'-'*10}  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}")

    for label, n, d, n_part, expand_n, reps in CONFIGS:
        X = make_data(n, d)
        for method, method_label in REDUCE_METHODS:
            t1 = run_reduce_bench(X, n_jobs=1,  method=method, n_partitions=n_part)
            tp = run_reduce_bench(X, n_jobs=-1, method=method, n_partitions=n_part)
            print(
                f"  {label}  {method_label:<15}  {fmt(t1):>10}  {fmt(tp):>10}"
                f"  {speedup(t1, tp):>8}"
            )
        print()

    # --- Expand timing ---
    print("[expand() — generation strategies]\n")
    header = f"  {'Dataset':<10}  {'strategy':<15}  {'serial':>10}  {'parallel':>10}  {'speedup':>8}"
    print(header)
    print(f"  {'-'*10}  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}")

    for label, n, d, n_part, expand_n, reps in CONFIGS:
        X = make_data(n, d)
        for strategy, strategy_label in STRATEGIES:
            t1 = run_expand_bench(X, n_jobs=1,  strategy=strategy, expand_n=expand_n, repeats=reps)
            tp = run_expand_bench(X, n_jobs=-1, strategy=strategy, expand_n=expand_n, repeats=reps)
            print(
                f"  {label}  {strategy_label:<15}  {fmt(t1):>10}  {fmt(tp):>10}"
                f"  {speedup(t1, tp):>8}"
            )
        print()

    print("=" * 80)
    print("Done.\n")


if __name__ == "__main__":
    main()
