"""
Generation Strategies for HVRT Expansion
==========================================

Partition-aware generation strategies.  Two protocols are supported:

StatefulGenerationStrategy (new, preferred)
-------------------------------------------
Each strategy is a class with two methods:

    prepare(X_z, partition_ids, unique_partitions) -> PartitionContext
        Called once; precomputes all partition metadata (stds, Cholesky
        factors, CDF grids, etc.).  The returned context is a frozen
        dataclass — safe to cache and reuse across multiple expand() calls.

    generate(context, budgets, random_state) -> ndarray (sum(budgets), d)
        Called per expand(); draws samples using precomputed context.
        Fully vectorized — no per-partition Python loops.

GenerationStrategy (old, deprecated)
--------------------------------------
Any callable with the partition-aware signature::

    def my_strategy(
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:

is still accepted.  A HVRTDeprecationWarning is emitted when HVRT detects
a plain callable at dispatch time.  The built-in module-level singletons
(``epanechnikov``, ``bootstrap_noise``, etc.) are now ``StatefulGenerationStrategy``
instances; calling them directly as functions still works but emits the
same deprecation warning.

Built-in strategies
-------------------
multivariate_kde        — per-partition Gaussian KDE via batch Cholesky.
                          Captures joint correlation structure.
                          Prepared: covariance + Cholesky (pure NumPy, no scipy).
                          Generated: batch matmul sampling.

univariate_kde_copula   — per-feature 1-D Gaussian KDE marginals + Gaussian
                          copula for joint dependence.
                          Prepared: CDF grids (scipy KDE fitted once, not per-expand).
                          Generated: np.interp + batch copula sampling.

bootstrap_noise         — resample with replacement + per-feature Gaussian noise
                          scaled to 10 % within-partition std.  Fastest; no
                          distributional assumptions.

epanechnikov            — product Epanechnikov kernel using Ahrens-Dieter (1980)
                          exact O(1) algorithm.  Bounded support.  Recommended
                          for classification and high expansion ratios (≥ 5×).

Usage
-----
    from hvrt import HVRT
    from hvrt.generation_strategies import epanechnikov, bootstrap_noise

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

import warnings

import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._budgets import _partition_pos
from ._warnings import HVRTDeprecationWarning


# ---------------------------------------------------------------------------
# Old protocol — kept for backward compatibility
# ---------------------------------------------------------------------------

@runtime_checkable
class GenerationStrategy(Protocol):
    """
    Protocol for partition-aware synthetic sample generation.

    .. deprecated::
        Implement :class:`StatefulGenerationStrategy` instead.  Plain
        callables are still accepted but emit a
        :class:`~hvrt._warnings.HVRTDeprecationWarning` at dispatch time.
    """

    def __call__(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# New stateful protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StatefulGenerationStrategy(Protocol):
    """
    Two-stage stateful protocol for partition-aware generation.

    ``prepare()`` is called once per model fit (or per unique dataset) and
    caches all partition metadata.  ``generate()`` is called per
    ``expand()`` invocation using the precomputed context.

    Built-in strategies (:class:`EpanechnikovStrategy`,
    :class:`BootstrapNoiseStrategy`, :class:`MultivariateKDEStrategy`,
    :class:`UnivariateCopulaStrategy`) implement this protocol.
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


@dataclass(frozen=True)
class BootstrapNoiseContext(PartitionContext):
    """Context for bootstrap-with-noise."""
    part_stds:  np.ndarray  # (n_parts, n_features)


@dataclass(frozen=True)
class MultivariateKDEContext(PartitionContext):
    """Context for multivariate Gaussian KDE."""
    L_all:   np.ndarray  # (n_parts, n_features, n_features) Cholesky factors
    part_h:  np.ndarray  # (n_parts,) Scott's bandwidth factor


@dataclass(frozen=True)
class UnivariateCopulaContext(PartitionContext):
    """Context for univariate KDE + Gaussian copula."""
    corr_L:     np.ndarray  # (n_parts, n_features, n_features) corr Cholesky
    cdf_grids:  np.ndarray  # (n_parts, n_features, 2000) CDF y-values
    grid_x:     np.ndarray  # (n_parts, n_features, 2000) CDF x-values


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
    the 1-D Epanechnikov kernel K(u) = (3/4)(1 − u²) for |u| ≤ 1.
    Applied independently per feature (product kernel).

    Fully vectorized: a single RNG block covers all partitions and samples.
    No per-partition Python loop in generate().

    Particularly effective for classification tasks and high expansion ratios
    (≥ 5×).  Bounded support prevents samples from escaping the local
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

    def __call__(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        warnings.warn(
            "Calling a generation strategy as a plain function is deprecated. "
            "Built-in strategies are now StatefulGenerationStrategy instances. "
            "Pass the strategy object to expand(generation_strategy=...) directly "
            "or use the strategy name string.",
            HVRTDeprecationWarning,
            stacklevel=2,
        )
        ctx = self.prepare(X_z, partition_ids, unique_partitions)
        return self.generate(ctx, budgets, random_state)


# ---------------------------------------------------------------------------
# BootstrapNoiseStrategy
# ---------------------------------------------------------------------------

class BootstrapNoiseStrategy:
    """
    Bootstrap-with-Gaussian-noise.

    Resamples from each partition with replacement and adds per-feature
    Gaussian noise scaled to 10 % of the within-partition standard deviation,
    floored at 0.01 z-score units.

    Fastest strategy; no distributional assumptions.  Works well for dense,
    low-variance partitions.

    Fully vectorized: no per-partition Python loop in generate().
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> BootstrapNoiseContext:
        n_parts = len(unique_partitions)
        d = X_z.shape[1]
        pos, sort_idx, part_starts, part_sizes = _build_base_context(
            X_z, partition_ids, unique_partitions
        )
        part_stds = _compute_part_stds(X_z, pos, part_sizes, n_parts, d)
        return BootstrapNoiseContext(
            X_z=X_z, pos=pos, sort_idx=sort_idx,
            part_starts=part_starts, part_sizes=part_sizes,
            n_parts=n_parts, n_features=d,
            part_stds=part_stds,
        )

    def generate(
        self,
        context: BootstrapNoiseContext,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        ctx = context
        rng = np.random.RandomState(random_state)
        d = ctx.n_features
        total_budget = int(budgets.sum())
        if total_budget == 0:
            return np.empty((0, d))

        labels = np.repeat(np.arange(ctx.n_parts), budgets)
        base = _resample_base_points(ctx, labels, total_budget, rng)
        scale = ctx.part_stds[labels] * 0.1   # (total_budget, d)
        return base + rng.standard_normal((total_budget, d)) * scale

    def __call__(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        warnings.warn(
            "Calling a generation strategy as a plain function is deprecated. "
            "Built-in strategies are now StatefulGenerationStrategy instances. "
            "Pass the strategy object to expand(generation_strategy=...) directly "
            "or use the strategy name string.",
            HVRTDeprecationWarning,
            stacklevel=2,
        )
        ctx = self.prepare(X_z, partition_ids, unique_partitions)
        return self.generate(ctx, budgets, random_state)


# ---------------------------------------------------------------------------
# MultivariateKDEStrategy
# ---------------------------------------------------------------------------

class MultivariateKDEStrategy:
    """
    Multivariate Gaussian KDE via batch Cholesky decomposition.

    Fits a multivariate Gaussian to each partition using Scott's bandwidth
    rule.  All covariance matrices are computed in a single vectorised pass
    (O(d²) bincount calls) and Cholesky-factored in batch — no scipy, no
    per-partition Python loop in prepare().

    generate() uses batch matmul to sample from all partitions at once.

    Memory note: L_all[labels] is (total_budget, d, d).  For d ≤ 30 and
    budget ≤ 100 k this is ≤ 720 MB.  For very large d, consider chunking.

    Single-sample partitions and singular covariance matrices are handled
    by eigenvalue clipping (PSD reconstruction) before Cholesky.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> MultivariateKDEContext:
        n_parts = len(unique_partitions)
        d = X_z.shape[1]
        pos, sort_idx, part_starts, part_sizes = _build_base_context(
            X_z, partition_ids, unique_partitions
        )
        safe_sizes = np.maximum(part_sizes, 1).astype(float)

        # Per-partition means — O(d) bincount calls
        means = np.zeros((n_parts, d))
        for j in range(d):
            means[:, j] = (
                np.bincount(pos, weights=X_z[:, j], minlength=n_parts) / safe_sizes
            )

        # Per-partition covariance — O(d²) bincount calls
        cov = np.zeros((n_parts, d, d))
        for i in range(d):
            for j in range(i, d):
                E_ij = (
                    np.bincount(pos, weights=X_z[:, i] * X_z[:, j], minlength=n_parts)
                    / safe_sizes
                )
                cov_ij = E_ij - means[:, i] * means[:, j]
                if i == j:
                    cov[:, i, j] = np.maximum(cov_ij, 0.0)   # clip negative variance
                else:
                    cov[:, i, j] = cov_ij
                    cov[:, j, i] = cov_ij

        # Scale by squared Scott's bandwidth factor
        part_h = safe_sizes ** (-1.0 / (d + 4))
        cov_scaled = cov * (part_h[:, None, None] ** 2)

        # Ensure PSD via eigenvalue clipping (handles singular / near-singular cov)
        eigvals, eigvecs = np.linalg.eigh(cov_scaled)          # (n_parts, d), (n_parts, d, d)
        eigvals = np.maximum(eigvals, 1e-8)
        cov_psd = (eigvecs * eigvals[:, None, :]) @ eigvecs.transpose(0, 2, 1)

        # Batch Cholesky — (n_parts, d, d)
        L_all = np.linalg.cholesky(cov_psd + 1e-12 * np.eye(d))

        return MultivariateKDEContext(
            X_z=X_z, pos=pos, sort_idx=sort_idx,
            part_starts=part_starts, part_sizes=part_sizes,
            n_parts=n_parts, n_features=d,
            L_all=L_all, part_h=part_h,
        )

    def generate(
        self,
        context: MultivariateKDEContext,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        ctx = context
        rng = np.random.RandomState(random_state)
        d = ctx.n_features
        total_budget = int(budgets.sum())
        if total_budget == 0:
            return np.empty((0, d))

        labels = np.repeat(np.arange(ctx.n_parts), budgets)
        base = _resample_base_points(ctx, labels, total_budget, rng)

        # Batch matmul: sample from per-partition Gaussian kernel
        z = rng.standard_normal((total_budget, d, 1))
        L_arr = ctx.L_all[labels]          # (total_budget, d, d)
        noise = (L_arr @ z)[:, :, 0]      # (total_budget, d)
        return base + noise

    def __call__(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        warnings.warn(
            "Calling a generation strategy as a plain function is deprecated. "
            "Built-in strategies are now StatefulGenerationStrategy instances. "
            "Pass the strategy object to expand(generation_strategy=...) directly "
            "or use the strategy name string.",
            HVRTDeprecationWarning,
            stacklevel=2,
        )
        ctx = self.prepare(X_z, partition_ids, unique_partitions)
        return self.generate(ctx, budgets, random_state)


# ---------------------------------------------------------------------------
# UnivariateCopulaStrategy
# ---------------------------------------------------------------------------

class UnivariateCopulaStrategy:
    """
    Univariate KDE marginals + Gaussian copula.

    Each feature's marginal distribution is modelled by its own 1-D
    Gaussian KDE.  The joint dependence structure is captured by a Gaussian
    copula fitted on rank-normalised data.

    **prepare()** runs the expensive scipy KDE fitting once (not per expand):
    CDF grids (2000 points per feature per partition) and correlation Cholesky
    factors are stored in the frozen context.

    **generate()** uses only np.interp + batch matmul — no scipy at sample time.

    Falls back to identity correlation for partitions with fewer than 4 samples
    or a single feature.  Single-sample partitions use a uniform CDF grid.
    """

    _GRID_SIZE = 2000

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> UnivariateCopulaContext:
        from scipy.stats import gaussian_kde, norm

        n_parts = len(unique_partitions)
        d = X_z.shape[1]
        G = self._GRID_SIZE
        pos, sort_idx, part_starts, part_sizes = _build_base_context(
            X_z, partition_ids, unique_partitions
        )

        corr_L = np.zeros((n_parts, d, d))
        cdf_grids = np.zeros((n_parts, d, G))
        grid_x = np.zeros((n_parts, d, G))

        for p in range(n_parts):
            start = int(part_starts[p])
            size = int(part_sizes[p])
            X_part = X_z[sort_idx[start:start + size]]   # (size, d)

            # --- Gaussian copula correlation ---
            if size < 4 or d == 1:
                corr_L[p] = np.eye(d)
            else:
                # Rank → pseudo-uniform → probit → correlation
                U = np.empty_like(X_part)
                for j in range(d):
                    col = X_part[:, j]
                    ranks = np.argsort(np.argsort(col))
                    U[:, j] = (ranks + 1) / (size + 1)
                Z = norm.ppf(U)
                corr = np.corrcoef(Z.T)
                min_eig = float(np.linalg.eigvalsh(corr).min())
                if min_eig <= 0:
                    corr += (-min_eig + 1e-6) * np.eye(d)
                try:
                    corr_L[p] = np.linalg.cholesky(corr)
                except np.linalg.LinAlgError:
                    corr_L[p] = np.eye(d)

            # --- CDF grids per feature ---
            for j in range(d):
                col = X_part[:, j]
                if size < 2:
                    lo = float(col[0]) - 0.1
                    hi = float(col[0]) + 0.1
                    grid_x[p, j] = np.linspace(lo, hi, G)
                    cdf_grids[p, j] = np.linspace(0.0, 1.0, G)
                else:
                    col_std = float(col.std())
                    margin = 3.0 * max(col_std, 0.01)
                    lo = float(col.min()) - margin
                    hi = float(col.max()) + margin
                    grid = np.linspace(lo, hi, G)
                    kde_j = gaussian_kde(col, bw_method='scott')
                    pdf_vals = kde_j.evaluate(grid)
                    cdf_vals = np.cumsum(pdf_vals)
                    cdf_vals /= cdf_vals[-1]
                    grid_x[p, j] = grid
                    cdf_grids[p, j] = cdf_vals

        return UnivariateCopulaContext(
            X_z=X_z, pos=pos, sort_idx=sort_idx,
            part_starts=part_starts, part_sizes=part_sizes,
            n_parts=n_parts, n_features=d,
            corr_L=corr_L, cdf_grids=cdf_grids, grid_x=grid_x,
        )

    def generate(
        self,
        context: UnivariateCopulaContext,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        from scipy.stats import norm

        ctx = context
        rng = np.random.RandomState(random_state)
        d = ctx.n_features
        n_parts = ctx.n_parts
        total_budget = int(budgets.sum())
        if total_budget == 0:
            return np.empty((0, d))

        labels = np.repeat(np.arange(n_parts), budgets)

        # Gaussian copula sampling — batch matmul
        z_cop = rng.standard_normal((total_budget, d, 1))
        L_arr = ctx.corr_L[labels]                    # (total_budget, d, d)
        z_corr = (L_arr @ z_cop)[:, :, 0]            # (total_budget, d)
        u_samp = norm.cdf(z_corr)                     # (total_budget, d)

        # CDF inversion via np.interp — O(n_parts × d) Python iterations,
        # each iteration is vectorized over all samples in the partition
        X_synth = np.empty((total_budget, d))
        for p in range(n_parts):
            mask = labels == p
            if not mask.any():
                continue
            for j in range(d):
                X_synth[mask, j] = np.interp(
                    u_samp[mask, j], ctx.cdf_grids[p, j], ctx.grid_x[p, j]
                )
        return X_synth

    def __call__(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        warnings.warn(
            "Calling a generation strategy as a plain function is deprecated. "
            "Built-in strategies are now StatefulGenerationStrategy instances. "
            "Pass the strategy object to expand(generation_strategy=...) directly "
            "or use the strategy name string.",
            HVRTDeprecationWarning,
            stacklevel=2,
        )
        ctx = self.prepare(X_z, partition_ids, unique_partitions)
        return self.generate(ctx, budgets, random_state)


# ---------------------------------------------------------------------------
# Module-level singletons — same names as the old functions for drop-in compat
# ---------------------------------------------------------------------------

epanechnikov         = EpanechnikovStrategy()
bootstrap_noise      = BootstrapNoiseStrategy()
multivariate_kde     = MultivariateKDEStrategy()
univariate_kde_copula = UnivariateCopulaStrategy()


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

BUILTIN_GENERATION_STRATEGIES: dict = {
    'multivariate_kde':      multivariate_kde,
    'univariate_kde_copula': univariate_kde_copula,
    'bootstrap_noise':       bootstrap_noise,
    'epanechnikov':          epanechnikov,
}


def get_generation_strategy(name: str) -> StatefulGenerationStrategy:
    """
    Return a built-in generation strategy by name.

    Parameters
    ----------
    name : str
        One of: ``'multivariate_kde'``, ``'univariate_kde_copula'``,
        ``'bootstrap_noise'``, ``'epanechnikov'``.

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
