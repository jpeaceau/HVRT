"""
Strategy Vectorization Speedup Benchmark
==========================================

Measures the concrete speedup delivered by the v2.4 two-stage stateful
generation protocol vs the previous per-partition callable protocol.

Three axes are measured:

1. **prepare() cost** — one-time context construction per strategy.

2. **generate() cost (cached)** — cost per expand() call when context is
   already in cache (the steady-state after the first call).

3. **Full expand() 1st call vs amortised** — shows how much the first call
   pays for prepare() and how the cost drops on subsequent calls.

4. **Old-protocol reference** — a hand-rolled per-partition loop that mirrors
   the v2.3 implementation so we can show the actual before/after delta.

Configurations
--------------
- Dataset sizes: 1K / 5K / 20K / 50K samples, d=10 features
- Strategies: epanechnikov, bootstrap_noise, multivariate_kde
- Expansion ratio: 5× (budget = 5 * n)
- Repeats: 5 for small, 3 for larger configs

Usage
-----
    python benchmarks/strategy_speedup_benchmark.py
"""

from __future__ import annotations

import time
import numpy as np

from hvrt import HVRT
from hvrt.generation_strategies import (
    EpanechnikovStrategy,
    BootstrapNoiseStrategy,
    MultivariateKDEStrategy,
    StatefulGenerationStrategy,
)
from hvrt._budgets import _iter_partitions


# ---------------------------------------------------------------------------
# Reference: old per-partition loop (mirrors v2.3 implementation)
# ---------------------------------------------------------------------------

def _ref_epanechnikov_partition(X_part, budget, seed):
    rng = np.random.RandomState(seed)
    n_p, d = X_part.shape
    h = n_p ** (-1.0 / (d + 4))
    per_feature_std = np.maximum(X_part.std(axis=0), 0.01)
    idx = rng.choice(n_p, size=budget, replace=True)
    base = X_part[idx]
    U1 = rng.uniform(-1.0, 1.0, (budget, d))
    U2 = rng.uniform(-1.0, 1.0, (budget, d))
    U3 = rng.uniform(-1.0, 1.0, (budget, d))
    use_U2 = (np.abs(U3) >= np.abs(U2)) & (np.abs(U3) >= np.abs(U1))
    noise_unit = np.where(use_U2, U2, U3)
    return base + h * per_feature_std * noise_unit


def _ref_bootstrap_partition(X_part, budget, seed):
    n, d = X_part.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=budget, replace=True)
    base = X_part[idx]
    std = np.maximum(X_part.std(axis=0), 0.01)
    return base + rng.normal(0, std * 0.1, (budget, d))


def _ref_kde_partition(X_part, budget, seed):
    from scipy.stats import gaussian_kde
    n, d = X_part.shape
    if n < 2:
        rng = np.random.RandomState(seed)
        return X_part[[0]] * np.ones((budget, 1)) + rng.normal(0, 0.01, (budget, d))
    try:
        kde = gaussian_kde(X_part.T, bw_method='scott')
        return kde.resample(budget, seed=seed).T
    except np.linalg.LinAlgError:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=budget, replace=True)
        return X_part[idx] + rng.normal(0, max(float(X_part.std()), 0.01) * 0.1, (budget, d))


class _OldProtocolEpanechnikov:
    """Reference: old per-partition loop callable (v2.3 style)."""
    def __call__(self, X_z, partition_ids, unique_partitions, budgets, random_state):
        rng = np.random.RandomState(random_state)
        results = []
        for _, X_part, budget in _iter_partitions(X_z, partition_ids, unique_partitions, budgets):
            seed = int(rng.randint(0, 2**31))
            results.append(_ref_epanechnikov_partition(X_part, budget, seed))
        return np.vstack(results) if results else np.empty((0, X_z.shape[1]))


class _OldProtocolBootstrap:
    """Reference: old per-partition loop callable (v2.3 style)."""
    def __call__(self, X_z, partition_ids, unique_partitions, budgets, random_state):
        rng = np.random.RandomState(random_state)
        results = []
        for _, X_part, budget in _iter_partitions(X_z, partition_ids, unique_partitions, budgets):
            seed = int(rng.randint(0, 2**31))
            results.append(_ref_bootstrap_partition(X_part, budget, seed))
        return np.vstack(results) if results else np.empty((0, X_z.shape[1]))


class _OldProtocolKDE:
    """Reference: old per-partition scipy KDE (v2.3 style)."""
    def __call__(self, X_z, partition_ids, unique_partitions, budgets, random_state):
        rng = np.random.RandomState(random_state)
        results = []
        for _, X_part, budget in _iter_partitions(X_z, partition_ids, unique_partitions, budgets):
            seed = int(rng.randint(0, 2**31))
            results.append(_ref_kde_partition(X_part, budget, seed))
        return np.vstack(results) if results else np.empty((0, X_z.shape[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(n: int, d: int = 10, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randn(n, d)


def median_time(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def fmt_ms(t: float) -> str:
    if t >= 10.0:
        return f"{t:7.2f}s"
    if t >= 1.0:
        return f"{t:7.3f}s"
    return f"{t * 1000:7.1f}ms"


def speedup_str(t_old: float, t_new: float) -> str:
    r = t_old / max(t_new, 1e-9)
    if r >= 1.0:
        return f"{r:5.1f}x faster"
    return f"{1/r:5.1f}x slower"


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------

CONFIGS = [
    # (label,    n,      d,  repeats)
    ("1K ",    1_000,  10,   5),
    ("5K ",    5_000,  10,   5),
    ("20K",   20_000,  10,   3),
    ("50K",   50_000,  10,   2),
]

STRATEGIES = [
    ("Epanechnikov", EpanechnikovStrategy(), _OldProtocolEpanechnikov(), "epanechnikov"),
    ("Bootstrap   ", BootstrapNoiseStrategy(), _OldProtocolBootstrap(), "bootstrap_noise"),
    ("Multivar.KDE", MultivariateKDEStrategy(), _OldProtocolKDE(), "multivariate_kde"),
]


def bench_old_vs_new(X, expand_n, repeats):
    """
    Compare old callable protocol vs new stateful protocol.
    Both measured at steady-state (no warm-up difference).

    Returns dict: strategy_name -> (t_old, t_prepare, t_generate, t_full_first, t_full_amortised)
    """
    model = HVRT(bandwidth=0.1, random_state=42).fit(X)

    from hvrt._budgets import _partition_pos
    from hvrt.expand import compute_expansion_budgets

    n_cont = int(model.continuous_mask_.sum())
    X_z_cont = model.X_z_[:, :n_cont]
    partition_ids = model.partition_ids_
    unique_partitions = model.unique_partitions_
    budgets = compute_expansion_budgets(partition_ids, unique_partitions, expand_n, False, model.X_z_)

    results = {}
    for name, new_strat, old_strat, str_name in STRATEGIES:
        # --- old protocol ---
        t_old = median_time(
            lambda s=old_strat: s(X_z_cont, partition_ids, unique_partitions, budgets, 42),
            repeats,
        )

        # --- new: prepare() ---
        t_prepare = median_time(
            lambda s=new_strat: s.prepare(X_z_cont, partition_ids, unique_partitions),
            repeats,
        )

        # --- new: generate() with pre-built context ---
        ctx = new_strat.prepare(X_z_cont, partition_ids, unique_partitions)
        t_generate = median_time(
            lambda s=new_strat, c=ctx: s.generate(c, budgets, 42),
            repeats,
        )

        # --- full expand(): 1st call (pays prepare + generate) ---
        # simulate by prepare + generate without cache
        t_full_cold = median_time(
            lambda s=new_strat: s.generate(
                s.prepare(X_z_cont, partition_ids, unique_partitions), budgets, 42
            ),
            repeats,
        )

        results[name] = (t_old, t_prepare, t_generate, t_full_cold)

    return results


def bench_caching(X, expand_n, repeats):
    """
    Show real expand() call speedup: 1st call vs Nth call via model cache.
    """
    results = {}
    for name, _, _, str_name in STRATEGIES:
        # First call: includes prepare() inside expand()
        model = HVRT(bandwidth=0.1, random_state=42).fit(X)
        t_first = median_time(
            lambda m=model, s=str_name: m.expand(n=expand_n, generation_strategy=s),
            1,   # single call — we want the first-call cost
        )

        # Clear cache to re-measure first call, then prime it
        model._strategy_context_cache_ = {}
        model.expand(n=expand_n, generation_strategy=str_name)   # prime cache
        t_cached = median_time(
            lambda m=model, s=str_name: m.expand(n=expand_n, generation_strategy=s),
            repeats,
        )

        results[name] = (t_first, t_cached)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nHVRT Strategy Vectorization Speedup Benchmark")
    print("=" * 75)
    print("New: StatefulGenerationStrategy (prepare + generate, vectorized)")
    print("Old: per-partition Python loop (v2.3 reference)")
    print()

    # ==============================================================
    # Section 1: Old protocol vs new protocol (at steady-state)
    # ==============================================================
    print("\n[1] Old callable protocol vs new vectorized protocol")
    print("    (Both measured at steady-state; expand_n = 5 * n)\n")

    hdr = (f"  {'Dataset':>6}  {'Strategy':<14}  "
           f"{'Old (loop)':>12}  {'prepare()':>12}  {'generate()':>12}  "
           f"{'speedup(gen)':>13}  {'speedup(full)':>14}")
    print(hdr)
    print("  " + "-" * 73)

    for label, n, d, reps in CONFIGS:
        X = make_data(n, d)
        expand_n = 5 * n
        res = bench_old_vs_new(X, expand_n, reps)
        first = True
        for name, (t_old, t_prep, t_gen, t_cold) in res.items():
            lbl = label if first else " " * len(label)
            first = False
            gen_su   = speedup_str(t_old, t_gen)
            full_su  = speedup_str(t_old, t_cold)
            print(
                f"  {lbl:>6}  {name:<14}  "
                f"{fmt_ms(t_old):>12}  {fmt_ms(t_prep):>12}  {fmt_ms(t_gen):>12}  "
                f"{gen_su:>13}  {full_su:>14}"
            )
        print()

    # ==============================================================
    # Section 2: Context caching benefit in expand()
    # ==============================================================
    print("\n[2] Context caching: 1st expand() vs Nth expand() (cached prepare)")
    print("    (expand_n = 5 * n)\n")

    hdr2 = (f"  {'Dataset':>6}  {'Strategy':<14}  "
            f"{'1st call':>12}  {'Nth call':>12}  {'cache speedup':>14}")
    print(hdr2)
    print("  " + "-" * 58)

    for label, n, d, reps in CONFIGS:
        X = make_data(n, d)
        expand_n = 5 * n
        cres = bench_caching(X, expand_n, reps)
        first = True
        for name, (t_first, t_cached) in cres.items():
            lbl = label if first else " " * len(label)
            first = False
            su = speedup_str(t_first, t_cached)
            print(
                f"  {lbl:>6}  {name:<14}  "
                f"{fmt_ms(t_first):>12}  {fmt_ms(t_cached):>12}  {su:>14}"
            )
        print()

    # ==============================================================
    # Section 3: Absolute generate() time vs n_samples (steady-state)
    # ==============================================================
    print("\n[3] generate() steady-state time vs dataset/budget size")
    print("    (Pure generate cost; context already cached; expand_n = 5 * n)\n")

    hdr3 = f"  {'Dataset':>6}  {'n_parts':>7}  {'Strategy':<14}  {'generate()':>12}  {'samples/s':>12}"
    print(hdr3)
    print("  " + "-" * 56)

    for label, n, d, reps in CONFIGS:
        X = make_data(n, d)
        expand_n = 5 * n
        model = HVRT(bandwidth=0.1, random_state=42).fit(X)
        n_parts = model.n_partitions_

        from hvrt.expand import compute_expansion_budgets
        n_cont = int(model.continuous_mask_.sum())
        X_z_cont = model.X_z_[:, :n_cont]
        partition_ids = model.partition_ids_
        unique_partitions = model.unique_partitions_
        budgets = compute_expansion_budgets(partition_ids, unique_partitions, expand_n, False, model.X_z_)

        first = True
        for name, new_strat, _, _ in STRATEGIES:
            lbl = label if first else " " * len(label)
            n_p = n_parts if first else ""
            first = False
            ctx = new_strat.prepare(X_z_cont, partition_ids, unique_partitions)
            t = median_time(lambda s=new_strat, c=ctx: s.generate(c, budgets, 42), reps)
            rate = expand_n / t
            print(
                f"  {lbl:>6}  {str(n_p):>7}  {name:<14}  "
                f"{fmt_ms(t):>12}  {rate:>10,.0f}/s"
            )
        print()

    print("=" * 75)
    print("Done.\n")


if __name__ == "__main__":
    main()
