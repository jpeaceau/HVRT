"""
Generation Strategies for HVRT Expansion
==========================================

Partition-aware generation strategies.  Each strategy owns the full
iteration loop across all partitions and returns synthetic samples in
z-score space.

Built-in strategies
-------------------
multivariate_kde      — one scipy multivariate gaussian_kde per partition.
                        Captures joint correlation structure exactly.

univariate_kde_copula — per-feature 1-D Gaussian KDE marginals + Gaussian
                        copula for the joint dependence.  More flexible
                        marginal shapes than the multivariate KDE (which
                        applies the same global bandwidth to all features),
                        while still preserving pairwise correlations.
                        Useful when marginals vary strongly across features.

bootstrap_noise       — resample from the partition with replacement and
                        add isotropic Gaussian noise scaled to 10 % of the
                        within-partition per-feature standard deviation.
                        Fastest option; no distributional assumptions.
                        Works well for dense, low-variance partitions; may
                        under-explore high-variance or tail partitions.

epanechnikov          — product Epanechnikov kernel using the Ahrens-Dieter
                        (1980) exact O(1) algorithm.  Fully vectorized.
                        Bandwidth per feature: h × σ_j where
                        h = n_part^(−1/(d+4)) (Scott's formula).
                        Bounded support prevents samples from escaping the
                        local partition region.  Recommended for
                        classification tasks and high expansion ratios (≥5×).

Custom strategies
-----------------
Any callable with the partition-aware signature::

    def my_strategy(
        X_z: np.ndarray,              # (n_samples, d)  full z-scored matrix
        partition_ids: np.ndarray,    # (n_samples,)
        unique_partitions: np.ndarray,
        budgets: np.ndarray,          # int, per-partition
        random_state: int,
    ) -> np.ndarray:                  # (sum(budgets), d) in z-score space

is a valid strategy.  Pass it directly to ``expand(generation_strategy=...)``.

Usage
-----
    from hvrt import HVRT
    from hvrt.generation_strategies import epanechnikov, bootstrap_noise

    model = HVRT(random_state=42).fit(X)

    # Built-in strategy by name
    X_synth = model.expand(n=5000, generation_strategy='epanechnikov')

    # Built-in strategy by reference
    X_synth = model.expand(n=5000, generation_strategy=epanechnikov)

    # Custom strategy
    def my_strategy(X_z, partition_ids, unique_partitions, budgets, random_state):
        ...
        return X_synthetic  # (sum(budgets), d)

    X_synth = model.expand(n=5000, generation_strategy=my_strategy)
"""

import numpy as np
from typing import Protocol, runtime_checkable

from ._budgets import _iter_partitions


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GenerationStrategy(Protocol):
    """
    Protocol for partition-aware synthetic sample generation.

    A generation strategy is a callable that iterates over all partitions
    and returns synthetic samples in z-score space.
    """

    def __call__(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
        budgets: np.ndarray,
        random_state: int,
    ) -> np.ndarray:
        """
        Generate synthetic samples across all partitions.

        Parameters
        ----------
        X_z : ndarray (n_samples, n_features)
            Z-score-normalised continuous feature matrix.
        partition_ids : ndarray (n_samples,)
        unique_partitions : ndarray
        budgets : ndarray of int
            Per-partition synthetic sample budgets.
        random_state : int

        Returns
        -------
        X_synthetic : ndarray (sum(budgets), n_features)
            Synthetic samples in z-score space.
        """
        ...


# ---------------------------------------------------------------------------
# Private per-partition sampling helpers
# ---------------------------------------------------------------------------

def _kde_sample_partition(X_part: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Fit and sample a multivariate Gaussian KDE on one partition."""
    from scipy.stats import gaussian_kde

    n, d = X_part.shape

    if n < 2:
        rng = np.random.RandomState(seed)
        base = np.tile(X_part[0], (budget, 1))
        return base + rng.normal(0, 0.01, base.shape)

    try:
        kde = gaussian_kde(X_part.T, bw_method='scott')
        return kde.resample(budget, seed=seed).T
    except np.linalg.LinAlgError:
        # Singular covariance (e.g. constant-valued partition): fall back
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size=budget, replace=True)
        noise_scale = max(float(X_part.std()), 0.01) * 0.1
        return X_part[idx] + rng.normal(0, noise_scale, (budget, d))


def _copula_sample_partition(X_part: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Fit univariate KDE marginals + Gaussian copula on one partition."""
    from scipy.stats import gaussian_kde, norm

    n, d = X_part.shape

    if n < 4 or d == 1:
        return _kde_sample_partition(X_part, budget, seed)

    rng = np.random.RandomState(seed)

    # Step 1: rank → pseudo-uniform [0, 1]
    U = np.empty_like(X_part)
    for j in range(d):
        col = X_part[:, j]
        ranks = np.argsort(np.argsort(col))
        U[:, j] = (ranks + 1) / (n + 1)

    # Step 2: probit transform → copula space
    Z = norm.ppf(U)

    # Step 3: Gaussian copula correlation matrix
    corr = np.corrcoef(Z.T)
    min_eig = float(np.linalg.eigvalsh(corr).min())
    if min_eig <= 0:
        corr += (-min_eig + 1e-6) * np.eye(d)

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return _kde_sample_partition(X_part, budget, seed)

    # Step 4: sample from the Gaussian copula
    z_samples = rng.standard_normal((budget, d)) @ L.T
    u_samples = norm.cdf(z_samples)

    # Step 5: invert marginal KDE CDFs
    X_synthetic = np.empty((budget, d))
    for j in range(d):
        col = X_part[:, j]
        col_std = float(col.std())
        kde_j = gaussian_kde(col, bw_method='scott')

        lo = float(col.min()) - 3.0 * max(col_std, 0.01)
        hi = float(col.max()) + 3.0 * max(col_std, 0.01)
        grid = np.linspace(lo, hi, 2000)

        pdf_vals = kde_j.evaluate(grid)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]

        X_synthetic[:, j] = np.interp(u_samples[:, j], cdf_vals, grid)

    return X_synthetic


def _bootstrap_noise_partition(X_part: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Bootstrap-with-noise on one partition."""
    n, d = X_part.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=budget, replace=True)
    base = X_part[idx]
    per_feature_std = X_part.std(axis=0)
    noise_scale = np.maximum(per_feature_std * 0.1, 0.01)
    return base + rng.normal(0, noise_scale, (budget, d))


# ---------------------------------------------------------------------------
# Built-in strategy 1: multivariate KDE
# ---------------------------------------------------------------------------

def multivariate_kde(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Multivariate Gaussian KDE (default strategy).

    Fits a single ``scipy.stats.gaussian_kde`` on all features simultaneously
    using Scott's rule bandwidth.  This captures within-partition correlation
    structure exactly.

    Single-sample partitions fall back to bootstrap-with-tiny-noise.
    Singular covariance matrices (constant-valued partitions) fall back to
    bootstrap-with-noise.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    X_synthetic : ndarray (sum(budgets), n_features), z-score space
    """
    rng = np.random.RandomState(random_state)
    all_synthetic = []
    n_features = X_z.shape[1]

    for _, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        seed = int(rng.randint(0, 2 ** 31))
        all_synthetic.append(_kde_sample_partition(X_part, budget, seed))

    if not all_synthetic:
        return np.empty((0, n_features))
    return np.vstack(all_synthetic)


# ---------------------------------------------------------------------------
# Built-in strategy 2: univariate KDE + Gaussian copula
# ---------------------------------------------------------------------------

def univariate_kde_copula(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Univariate KDE marginals + Gaussian copula.

    Each feature's marginal distribution is modelled by its own 1-D
    Gaussian KDE.  The joint dependence structure is captured by a Gaussian
    copula fitted on the rank-normalised data.

    Falls back to ``_kde_sample_partition`` for partitions with fewer than
    4 samples or a single feature.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    X_synthetic : ndarray (sum(budgets), n_features), z-score space
    """
    rng = np.random.RandomState(random_state)
    all_synthetic = []
    n_features = X_z.shape[1]

    for _, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        seed = int(rng.randint(0, 2 ** 31))
        all_synthetic.append(_copula_sample_partition(X_part, budget, seed))

    if not all_synthetic:
        return np.empty((0, n_features))
    return np.vstack(all_synthetic)


# ---------------------------------------------------------------------------
# Built-in strategy 3: bootstrap + noise
# ---------------------------------------------------------------------------

def bootstrap_noise(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Bootstrap-with-Gaussian-noise.

    Resamples from each partition with replacement and adds isotropic
    Gaussian noise.  The noise scale is 10 % of the within-partition
    per-feature standard deviation, floored at 0.01 z-score units.

    This is the fastest generation strategy and requires no distributional
    assumptions.  It works well for dense, low-variance partitions.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    X_synthetic : ndarray (sum(budgets), n_features), z-score space
    """
    rng = np.random.RandomState(random_state)
    all_synthetic = []
    n_features = X_z.shape[1]

    for _, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        seed = int(rng.randint(0, 2 ** 31))
        all_synthetic.append(_bootstrap_noise_partition(X_part, budget, seed))

    if not all_synthetic:
        return np.empty((0, n_features))
    return np.vstack(all_synthetic)


# ---------------------------------------------------------------------------
# Built-in strategy 4: Epanechnikov product kernel
# ---------------------------------------------------------------------------

def epanechnikov(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Product Epanechnikov kernel with Scott's bandwidth (fully vectorized).

    Uses the Ahrens-Dieter (1980) exact O(1) algorithm for sampling from the
    1-D Epanechnikov kernel ``K(u) = (3/4)(1 − u²)`` for ``|u| ≤ 1``.
    Applied independently per feature (product kernel).

    Algorithm per dimension: draw U1, U2, U3 ~ Uniform(−1, 1).  Return U2
    if ``|U3| ≥ |U2|`` and ``|U3| ≥ |U1|``, else return U3.

    Bandwidth per feature: ``h × σ_j`` where ``h = n_part^(−1/(d+4))``
    (Scott's rule) and ``σ_j`` is the within-partition standard deviation of
    feature j.

    Bounded support means no synthetic sample can fall outside the local
    bandwidth window, keeping all samples within the partition's feature-space
    region.  This prevents the distributional bleed across partition boundaries
    that affects Gaussian tails.

    Particularly effective for classification tasks and high expansion ratios
    (≥ 5×).  For regression tasks with strong inter-feature correlations,
    the product (independence) assumption may reduce quality relative to
    multivariate Gaussian KDE with a narrow bandwidth.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    X_synthetic : ndarray (sum(budgets), n_features), z-score space
    """
    rng = np.random.RandomState(random_state)
    all_synthetic = []
    n_features = X_z.shape[1]

    for _, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        n_p, d = X_part.shape

        # Scott's bandwidth factor for d-dimensional data
        h = n_p ** (-1.0 / (d + 4))

        # Per-feature within-partition std; floored to avoid zero scale
        per_feature_std = np.maximum(X_part.std(axis=0), 0.01)

        # Resample base points (kernel centring)
        idx = rng.choice(n_p, size=budget, replace=True)
        base = X_part[idx]  # (budget, d)

        # Ahrens-Dieter vectorized over all samples and features at once
        U1 = rng.uniform(-1.0, 1.0, (budget, d))
        U2 = rng.uniform(-1.0, 1.0, (budget, d))
        U3 = rng.uniform(-1.0, 1.0, (budget, d))
        use_U2 = (np.abs(U3) >= np.abs(U2)) & (np.abs(U3) >= np.abs(U1))
        noise_unit = np.where(use_U2, U2, U3)  # (budget, d)

        # Scale by h × per-feature std and add to base
        all_synthetic.append(base + h * per_feature_std * noise_unit)

    if not all_synthetic:
        return np.empty((0, n_features))
    return np.vstack(all_synthetic)


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

BUILTIN_GENERATION_STRATEGIES: dict = {
    'multivariate_kde': multivariate_kde,
    'univariate_kde_copula': univariate_kde_copula,
    'bootstrap_noise': bootstrap_noise,
    'epanechnikov': epanechnikov,
}


def get_generation_strategy(name: str) -> GenerationStrategy:
    """
    Return a built-in generation strategy by name.

    Parameters
    ----------
    name : str
        One of: ``'multivariate_kde'``, ``'univariate_kde_copula'``,
        ``'bootstrap_noise'``.

    Returns
    -------
    strategy : GenerationStrategy

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
