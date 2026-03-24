"""
Generation Strategies for HVRT Expansion
==========================================

Partition-aware generation strategies.

StatefulGenerationStrategy
--------------------------
Each strategy is a class with two methods:

    prepare(X_z, partition_ids, unique_partitions) -> PartitionContext
        Called once; precomputes all partition metadata (stds, etc.).
        The returned context is a frozen dataclass — safe to cache and
        reuse across multiple expand() calls.

    generate(context, budgets, random_state) -> ndarray (sum(budgets), d)
        Called per expand(); draws samples using precomputed context.
        Fully vectorized — no per-partition Python loops.

Built-in strategies
-------------------
epanechnikov            — product Epanechnikov kernel using Ahrens-Dieter (1980)
                          exact O(1) algorithm.  Bounded support.  Recommended
                          for classification and high expansion ratios (>= 5x).

Usage
-----
    from hvrt import HVRT
    from hvrt.generation_strategies import epanechnikov

    model = HVRT(random_state=42).fit(X)

    # Built-in strategy by name (preferred)
    X_synth = model.expand(n=5000, generation_strategy='epanechnikov')

    # Built-in strategy by reference
    X_synth = model.expand(n=5000, generation_strategy=epanechnikov)

    # Custom strategy (implement StatefulGenerationStrategy)
    class MyStrategy:
        def prepare(self, X_z, partition_ids, unique_partitions):
            ...  # return a PartitionContext or subclass
        def generate(self, context, budgets, random_state):
            ...  # return (sum(budgets), d) ndarray

    X_synth = model.expand(n=5000, generation_strategy=MyStrategy())
"""

import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._budgets import _partition_pos


# ---------------------------------------------------------------------------
# Stateful protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StatefulGenerationStrategy(Protocol):
    """
    Two-stage stateful protocol for partition-aware generation.

    ``prepare()`` is called once per model fit (or per unique dataset) and
    caches all partition metadata.  ``generate()`` is called per
    ``expand()`` invocation using the precomputed context.

    Built-in strategy: :class:`EpanechnikovStrategy`.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> 'PartitionContext':
        """
        Precompute partition metadata (called once at fit time).

        Parameters
        ----------
        X_z : ndarray (n_samples, n_cont)
            Z-score-normalised continuous feature matrix.
        partition_ids : ndarray (n_samples,)
        unique_partitions : ndarray

        Returns
        -------
        context : PartitionContext
            Frozen dataclass holding all precomputed state.
        """
        ...

    def generate(
        self,
        context: 'PartitionContext',
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        """
        Generate synthetic samples using precomputed context.

        Parameters
        ----------
        context : PartitionContext
        budgets : ndarray of int
            Per-partition sample budgets (same order as unique_partitions).
        random_state : int

        Returns
        -------
        X_synthetic : ndarray (sum(budgets), n_cont)
            Synthetic samples in z-score space.
        """
        ...


# ---------------------------------------------------------------------------
# Frozen context dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartitionContext:
    """Base context holding common partition indexing arrays."""
    X_z:          np.ndarray  # (n_samples, n_cont) — reference, no copy
    pos:          np.ndarray  # (n_samples,) partition index per sample
    sort_idx:     np.ndarray  # (n_samples,) stable argsort of pos
    part_starts:  np.ndarray  # (n_parts,) start index in sorted order
    part_sizes:   np.ndarray  # (n_parts,) int
    n_parts:      int
    n_features:   int


@dataclass(frozen=True)
class EpanechnikovContext(PartitionContext):
    """Context for Epanechnikov product kernel."""
    part_stds:  np.ndarray  # (n_parts, n_features)
    part_h:     np.ndarray  # (n_parts,) Scott's bandwidth factor


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _build_base_context(X_z, partition_ids, unique_partitions):
    """
    Build common partition indexing arrays shared by all strategies.

    Returns
    -------
    (pos, sort_idx, part_starts, part_sizes)
    """
    pos = _partition_pos(partition_ids, unique_partitions)
    sort_idx = np.argsort(pos, kind='stable')
    part_starts = np.searchsorted(pos[sort_idx], np.arange(len(unique_partitions)))
    part_sizes = np.bincount(pos, minlength=len(unique_partitions))
    return pos, sort_idx, part_starts, part_sizes


def _resample_base_points(context, labels, total_budget, rng):
    """
    Vectorized base-point resampling: for each synthetic sample, draw a
    random source sample from the same partition.

    Uses ``rng.random()`` (legacy RandomState compatible) instead of
    ``rng.integers()`` with an array high, keeping np.random.RandomState usable.

    Parameters
    ----------
    context : PartitionContext
    labels : ndarray (total_budget,) — partition index per synthetic sample
    total_budget : int
    rng : np.random.RandomState

    Returns
    -------
    base : ndarray (total_budget, n_features)
    """
    part_size_arr = np.maximum(context.part_sizes[labels], 1)
    local_idx = (rng.random(total_budget) * part_size_arr).astype(np.intp)
    local_idx = np.minimum(local_idx, part_size_arr - 1)   # guard fp rounding
    return context.X_z[context.sort_idx[context.part_starts[labels] + local_idx]]


def _compute_part_stds(X_z, pos, part_sizes, n_parts, d):
    """
    Compute per-partition per-feature standard deviations via vectorised
    bincount calls — no per-partition Python loop.

    Returns
    -------
    part_stds : ndarray (n_parts, d), clipped to minimum 0.01
    """
    safe_sizes = np.maximum(part_sizes, 1).astype(float)
    means = np.zeros((n_parts, d))
    sq_means = np.zeros((n_parts, d))
    for j in range(d):
        means[:, j] = (
            np.bincount(pos, weights=X_z[:, j], minlength=n_parts) / safe_sizes
        )
        sq_means[:, j] = (
            np.bincount(pos, weights=X_z[:, j] ** 2, minlength=n_parts) / safe_sizes
        )
    var = np.maximum(sq_means - means ** 2, 0.0)
    return np.sqrt(var).clip(min=0.01)


# ---------------------------------------------------------------------------
# EpanechnikovStrategy
# ---------------------------------------------------------------------------

class EpanechnikovStrategy:
    """
    Product Epanechnikov kernel with Scott's bandwidth.

    Uses the Ahrens-Dieter (1980) exact O(1) algorithm for sampling from
    the 1-D Epanechnikov kernel K(u) = (3/4)(1 - u^2) for |u| <= 1.
    Applied independently per feature (product kernel).

    Fully vectorized: a single RNG block covers all partitions and samples.
    No per-partition Python loop in generate().

    Particularly effective for classification tasks and high expansion ratios
    (>= 5x).  Bounded support prevents samples from escaping the local
    partition region.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> EpanechnikovContext:
        n_parts = len(unique_partitions)
        d = X_z.shape[1]
        pos, sort_idx, part_starts, part_sizes = _build_base_context(
            X_z, partition_ids, unique_partitions
        )
        part_stds = _compute_part_stds(X_z, pos, part_sizes, n_parts, d)
        part_h = np.maximum(part_sizes, 1).astype(float) ** (-1.0 / (d + 4))
        return EpanechnikovContext(
            X_z=X_z, pos=pos, sort_idx=sort_idx,
            part_starts=part_starts, part_sizes=part_sizes,
            n_parts=n_parts, n_features=d,
            part_stds=part_stds, part_h=part_h,
        )

    def generate(
        self,
        context: EpanechnikovContext,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        ctx = context
        rng = np.random.RandomState(random_state)
        d = ctx.n_features
        total_budget = int(budgets.sum())
        if total_budget == 0:
            return np.empty((0, d))

        labels = np.repeat(np.arange(ctx.n_parts), budgets)   # (total_budget,)
        base = _resample_base_points(ctx, labels, total_budget, rng)  # (total_budget, d)

        # Ahrens-Dieter exact O(1) Epanechnikov kernel — all partitions at once
        U1 = rng.uniform(-1.0, 1.0, (total_budget, d))
        U2 = rng.uniform(-1.0, 1.0, (total_budget, d))
        U3 = rng.uniform(-1.0, 1.0, (total_budget, d))
        use_U2 = (np.abs(U3) >= np.abs(U2)) & (np.abs(U3) >= np.abs(U1))
        noise_unit = np.where(use_U2, U2, U3)

        h_arr = ctx.part_h[labels][:, None]    # (total_budget, 1) — broadcast over d
        scale_arr = ctx.part_stds[labels]       # (total_budget, d)
        return base + h_arr * scale_arr * noise_unit


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

epanechnikov = EpanechnikovStrategy()


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

BUILTIN_GENERATION_STRATEGIES: dict = {
    'epanechnikov': epanechnikov,
}


def get_generation_strategy(name: str) -> StatefulGenerationStrategy:
    """
    Return a built-in generation strategy by name.

    Parameters
    ----------
    name : str
        Must be ``'epanechnikov'``.

    Returns
    -------
    strategy : StatefulGenerationStrategy

    Raises
    ------
    ValueError
        If ``name`` is not a recognised strategy.
    """
    if name not in BUILTIN_GENERATION_STRATEGIES:
        raise ValueError(
            f"Unknown generation strategy: {name!r}. "
            f"Available: {sorted(BUILTIN_GENERATION_STRATEGIES)}"
        )
    return BUILTIN_GENERATION_STRATEGIES[name]
