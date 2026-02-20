"""
Generation Strategies for HVRT Expansion
==========================================

Type definitions and built-in strategies for within-partition synthetic
sample generation.  Each strategy is a stateless callable that receives
a single partition's z-scored data and produces a requested number of
synthetic samples in the same z-score space.

Built-in strategies
-------------------
multivariate_kde      — default: one scipy multivariate gaussian_kde per
                        partition.  Captures joint correlation structure
                        exactly.  Best overall quality; default for expand().

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

Custom strategies
-----------------
Any callable with the signature::

    def my_strategy(
        X_partition: np.ndarray,   # (n, d)  z-scored continuous features
        budget: int,
        random_state: int = 42,
    ) -> np.ndarray:               # (budget, d)  in z-score space

is a valid strategy.  Pass it directly to ``expand(generation_strategy=...)``.

Usage
-----
    from hvrt import HVRT
    from hvrt.generation_strategies import bootstrap_noise, univariate_kde_copula

    model = HVRT(random_state=42).fit(X)

    # Built-in strategy by name
    X_synth = model.expand(n=5000, generation_strategy='bootstrap_noise')

    # Built-in strategy by reference
    X_synth = model.expand(n=5000, generation_strategy=univariate_kde_copula)

    # Custom strategy
    def my_strategy(X_part, budget, random_state=42):
        ...
        return X_synthetic  # (budget, d)

    X_synth = model.expand(n=5000, generation_strategy=my_strategy)
"""

import numpy as np
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GenerationStrategy(Protocol):
    """
    Protocol for within-partition synthetic sample generation.

    A generation strategy is a callable that generates synthetic samples
    for a single partition.  Implementations must be stochastic (use
    random_state) but otherwise stateless — no persistent fitted state
    between calls.

    The input ``X_partition`` and all outputs are in z-score space
    (i.e. already normalised by the model's ``StandardScaler``).  The
    caller (``expand()``) handles inverse-transforming back to original
    feature scale.
    """

    def __call__(
        self,
        X_partition: np.ndarray,
        budget: int,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Generate synthetic samples for one partition.

        Parameters
        ----------
        X_partition : ndarray (n_partition, n_features)
            Z-score-normalised feature matrix for this partition.
            All values are continuous (categorical columns are handled
            separately by the caller).
        budget : int
            Number of synthetic samples to generate.  The returned array
            must have exactly ``budget`` rows.
        random_state : int, default=42
            Seed for reproducibility.

        Returns
        -------
        X_synthetic : ndarray (budget, n_features)
            Synthetic samples in z-score space.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in strategy 1: multivariate KDE
# ---------------------------------------------------------------------------

def multivariate_kde(
    X_partition: np.ndarray,
    budget: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Multivariate Gaussian KDE (default strategy).

    Fits a single ``scipy.stats.gaussian_kde`` on all features simultaneously
    using Scott's rule bandwidth.  This captures within-partition correlation
    structure exactly and is the default generation strategy.

    Single-sample partitions fall back to bootstrap-with-tiny-noise.
    Singular covariance matrices (constant-valued partitions) fall back to
    bootstrap-with-noise.
    """
    from scipy.stats import gaussian_kde

    n, d = X_partition.shape

    if n < 2:
        rng = np.random.RandomState(random_state)
        base = np.tile(X_partition[0], (budget, 1))
        return base + rng.normal(0, 0.01, base.shape)

    try:
        kde = gaussian_kde(X_partition.T, bw_method='scott')
        return kde.resample(budget, seed=random_state).T
    except np.linalg.LinAlgError:
        # Singular covariance (e.g. constant-valued partition): fall back
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=budget, replace=True)
        noise_scale = max(float(X_partition.std()), 0.01) * 0.1
        return X_partition[idx] + rng.normal(0, noise_scale, (budget, d))


# ---------------------------------------------------------------------------
# Built-in strategy 2: univariate KDE + Gaussian copula
# ---------------------------------------------------------------------------

def univariate_kde_copula(
    X_partition: np.ndarray,
    budget: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Univariate KDE marginals + Gaussian copula.

    Each feature's marginal distribution is modelled by its own 1-D
    Gaussian KDE.  The joint dependence structure is captured by a Gaussian
    copula fitted on the rank-normalised data.

    This gives more faithful per-feature marginal shapes than the
    multivariate KDE (which applies a single global bandwidth across all
    features) while still preserving pairwise correlations.

    Algorithm
    ---------
    1. Rank-transform each feature column to pseudo-uniform [0, 1]
       (empirical CDF with continuity correction).
    2. Probit-transform to standard normal — the Gaussian copula space.
    3. Estimate the correlation matrix from the copula-space data.
    4. Sample ``budget`` points from the multivariate normal with that
       correlation matrix.
    5. Map each column back through the inverse marginal KDE CDF
       (approximated via a dense grid + linear interpolation).

    Falls back to ``multivariate_kde`` for partitions with fewer than
    4 samples or a single feature.
    """
    from scipy.stats import gaussian_kde, norm

    n, d = X_partition.shape
    rng = np.random.RandomState(random_state)

    if n < 4 or d == 1:
        return multivariate_kde(X_partition, budget, random_state)

    # --- Step 1: rank → pseudo-uniform [0, 1] --------------------------
    U = np.empty_like(X_partition)
    for j in range(d):
        col = X_partition[:, j]
        ranks = np.argsort(np.argsort(col))     # 0-indexed ranks
        U[:, j] = (ranks + 1) / (n + 1)        # avoid exact 0 and 1

    # --- Step 2: probit transform → copula space -----------------------
    Z = norm.ppf(U)

    # --- Step 3: Gaussian copula correlation matrix --------------------
    corr = np.corrcoef(Z.T)
    # Ensure positive-definiteness with a small diagonal nudge if needed
    min_eig = float(np.linalg.eigvalsh(corr).min())
    if min_eig <= 0:
        corr += (-min_eig + 1e-6) * np.eye(d)

    # --- Step 4: sample from the Gaussian copula -----------------------
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return multivariate_kde(X_partition, budget, random_state)

    z_samples = rng.standard_normal((budget, d)) @ L.T   # (budget, d)
    u_samples = norm.cdf(z_samples)                       # (budget, d) in [0,1]

    # --- Step 5: invert marginal KDE CDFs ------------------------------
    X_synthetic = np.empty((budget, d))
    for j in range(d):
        col = X_partition[:, j]
        col_std = float(col.std())
        kde_j = gaussian_kde(col, bw_method='scott')

        # Build a dense grid spanning ±3 std beyond observed range
        lo = float(col.min()) - 3.0 * max(col_std, 0.01)
        hi = float(col.max()) + 3.0 * max(col_std, 0.01)
        grid = np.linspace(lo, hi, 2000)

        # Approximate CDF by integrating the PDF on the grid
        pdf_vals = kde_j.evaluate(grid)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]   # normalise to [0, 1]

        # Interpolate u_samples → feature values
        X_synthetic[:, j] = np.interp(u_samples[:, j], cdf_vals, grid)

    return X_synthetic


# ---------------------------------------------------------------------------
# Built-in strategy 3: bootstrap + noise
# ---------------------------------------------------------------------------

def bootstrap_noise(
    X_partition: np.ndarray,
    budget: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Bootstrap-with-Gaussian-noise (partition-level).

    Resamples from the partition with replacement and adds isotropic
    Gaussian noise.  The noise scale is 10 % of the within-partition
    per-feature standard deviation, floored at 0.01 z-score units.

    This is the fastest generation strategy and requires no distributional
    assumptions.  It works well for dense, low-variance partitions where
    the data is already well-sampled.  For high-variance or tail partitions
    it may under-explore relative to the KDE-based strategies.
    """
    n, d = X_partition.shape
    rng = np.random.RandomState(random_state)

    idx = rng.choice(n, size=budget, replace=True)
    base = X_partition[idx]

    # Per-feature noise scale: 10 % of within-partition std, min 0.01
    per_feature_std = X_partition.std(axis=0)
    noise_scale = np.maximum(per_feature_std * 0.1, 0.01)
    noise = rng.normal(0, noise_scale, (budget, d))

    return base + noise


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

BUILTIN_GENERATION_STRATEGIES: dict = {
    'multivariate_kde': multivariate_kde,
    'univariate_kde_copula': univariate_kde_copula,
    'bootstrap_noise': bootstrap_noise,
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
