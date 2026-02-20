"""
In-partition generation methods for HVRT research.

Each class implements the same interface:
    method.fit(X_part)                  → self
    method.sample(n_samples, rng=None)  → ndarray (n_samples, n_features)
    method.fit_sample(X_part, n, rng)   → ndarray (n, n_features)  [convenience]

All methods operate in z-score normalised feature space.  Callers are
responsible for inverse-transforming back to original scale if needed.

Methods
-------
MultivariateKDE            Full-matrix Gaussian KDE  (current HVRT default)
UnivariateKDERankCoupled   Per-feature KDE + Gaussian copula rank coupling
UnivariateKDEIndependent   Per-feature KDE, features sampled independently
PartitionGMM               Gaussian Mixture Model fitted to the partition
KNNInterpolation           SMOTE-style k-NN convex interpolation
PartitionBootstrap         Resample with replacement + Gaussian noise
"""

import warnings
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class _BaseMethod:
    """Shared interface for all in-partition generation methods."""

    name: str = 'base'

    def fit(self, X_part):
        raise NotImplementedError

    def sample(self, n_samples, rng=None):
        raise NotImplementedError

    def fit_sample(self, X_part, n_samples, rng=None):
        return self.fit(X_part).sample(n_samples, rng=rng)

    def _rng(self, rng):
        if isinstance(rng, np.random.RandomState):
            return rng
        if isinstance(rng, int):
            return np.random.RandomState(rng)
        return np.random.RandomState()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multivariate KDE  (current HVRT default)
# ─────────────────────────────────────────────────────────────────────────────

class MultivariateKDE(_BaseMethod):
    """
    Full-matrix multivariate Gaussian KDE.

    scipy.stats.gaussian_kde is fitted on X_part.T (shape: d × n).
    A single bandwidth scalar is applied isotropically across all features.

    Strengths
    ---------
    - Captures full joint density in one model
    - No independence assumption
    - Smooth interpolation across the partition manifold

    Weaknesses
    ----------
    - O(n²) evaluation for large partitions at prediction time
    - Bandwidth must be tuned (0.5 is empirically optimal for tabular data)
    - Degrades when n << d (few samples, many features)

    Parameters
    ----------
    bandwidth : float or 'scott' or 'silverman', default=0.5
        KDE bandwidth.  Scalar values are passed directly to gaussian_kde's
        bw_method argument.  'scott' and 'silverman' use adaptive rules.
    """

    name = 'MultivariateKDE'

    def __init__(self, bandwidth=0.5):
        self.bandwidth = bandwidth

    def fit(self, X_part):
        from scipy.stats import gaussian_kde
        X = np.asarray(X_part, dtype=float)
        if len(X) < 2:
            self._kde = None
            self._fallback = X
            return self
        if X.shape[0] < X.shape[1]:
            # More features than samples: fall back to bootstrap + noise
            self._kde = None
            self._fallback = X
            return self
        self._kde = gaussian_kde(X.T, bw_method=self.bandwidth)
        self._n_features = X.shape[1]
        return self

    def sample(self, n_samples, rng=None):
        if self._kde is None:
            # Fallback: bootstrap + small noise
            rng = self._rng(rng)
            idx = rng.choice(len(self._fallback), n_samples, replace=True)
            noise = rng.randn(n_samples, self._fallback.shape[1]) * 0.01
            return self._fallback[idx] + noise

        seed = self._rng(rng).randint(0, 2**31)
        raw = self._kde.resample(n_samples, seed=seed)
        return raw.T   # (n_samples, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Univariate KDE with rank coupling
# ─────────────────────────────────────────────────────────────────────────────

class UnivariateKDERankCoupled(_BaseMethod):
    """
    Per-feature univariate KDE coupled via a Gaussian copula.

    Approach
    --------
    1. Fit a univariate KDE on each feature column independently.
    2. Compute the empirical rank (CDF) of each sample on each feature.
    3. Map ranks to the standard normal space via Φ⁻¹ → Gaussian copula.
    4. Estimate the copula's covariance from the rank-normal values.
    5. At generation time:
       a. Sample from the copula (multivariate normal with estimated covariance).
       b. Apply Φ to get back uniform marginal ranks.
       c. For each feature, map ranks through the KDE's approximate quantile
          function to recover values in the original marginal distribution.

    This preserves:
    - Per-feature marginal distributions (via KDE)
    - Inter-feature rank correlations (via the Gaussian copula)

    The marginal KDE governs tail behaviour independently of the correlation
    structure — unlike the multivariate KDE where both are entangled.

    Strengths
    ---------
    - Robust to small n (per-feature KDE needs fewer samples than full matrix)
    - Explicit separation of marginal and dependence modelling
    - Can produce more faithful marginals per feature

    Weaknesses
    ----------
    - Gaussian copula cannot capture non-Gaussian tail dependence
    - Quantile inversion via KDE resampling is approximate

    Parameters
    ----------
    bandwidth : float or 'scott', default=0.5
    n_quantile_grid : int, default=5000
        Resolution of the KDE quantile grid used for inversion.
    """

    name = 'UnivariateKDERankCoupled'

    def __init__(self, bandwidth=0.5, n_quantile_grid=5000):
        self.bandwidth = bandwidth
        self.n_quantile_grid = n_quantile_grid

    def fit(self, X_part):
        from scipy.stats import gaussian_kde, norm, rankdata

        X = np.asarray(X_part, dtype=float)
        n, d = X.shape
        self._d = d
        self._kdes = []
        self._quantile_grids = []

        # Rank-normal transform for copula
        Z = np.zeros_like(X)
        for j in range(d):
            kde_j = gaussian_kde(X[:, j], bw_method=self.bandwidth)
            self._kdes.append(kde_j)
            # Pre-compute quantile grid: sorted KDE samples → uniform grid
            grid = np.sort(kde_j.resample(self.n_quantile_grid, seed=0).ravel())
            self._quantile_grids.append(grid)
            # Rank-normalise each column for copula fitting
            u = rankdata(X[:, j]) / (n + 1)
            Z[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

        self._copula_mu  = Z.mean(axis=0)
        if d > 1:
            self._copula_cov = np.cov(Z.T)
        else:
            self._copula_cov = np.array([[Z.var()]])
        # Regularise in case of near-singular covariance
        self._copula_cov += np.eye(d) * 1e-6
        return self

    def sample(self, n_samples, rng=None):
        from scipy.stats import norm
        rng = self._rng(rng)

        # Sample from the Gaussian copula
        Z_synth = rng.multivariate_normal(self._copula_mu, self._copula_cov, n_samples)
        U_synth = norm.cdf(Z_synth)  # → uniform marginals, shape (n_samples, d)

        # Map each feature through its KDE quantile grid
        X_synth = np.zeros((n_samples, self._d))
        for j in range(self._d):
            grid = self._quantile_grids[j]
            n_grid = len(grid)
            # Map uniform quantile to grid index, then to grid value
            idx = np.clip(
                (U_synth[:, j] * n_grid).astype(int), 0, n_grid - 1
            )
            X_synth[:, j] = grid[idx]

        return X_synth


# ─────────────────────────────────────────────────────────────────────────────
# 3. Univariate KDE independent
# ─────────────────────────────────────────────────────────────────────────────

class UnivariateKDEIndependent(_BaseMethod):
    """
    Per-feature univariate KDE with fully independent sampling.

    Each feature is modelled and sampled independently from its own KDE.
    No attempt is made to preserve inter-feature correlations.

    Useful as a diagnostic baseline: it shows what performance looks like
    when marginal fidelity is preserved but correlation structure is not.

    Strengths
    ---------
    - Very fast; trivially parallelisable
    - Exact marginal fidelity per feature

    Weaknesses
    ----------
    - Destroys inter-feature correlation structure
    - Generated samples may be physically impossible combinations

    Parameters
    ----------
    bandwidth : float or 'scott', default=0.5
    """

    name = 'UnivariateKDEIndependent'

    def __init__(self, bandwidth=0.5):
        self.bandwidth = bandwidth

    def fit(self, X_part):
        from scipy.stats import gaussian_kde
        X = np.asarray(X_part, dtype=float)
        self._d = X.shape[1]
        self._kdes = [
            gaussian_kde(X[:, j], bw_method=self.bandwidth)
            for j in range(self._d)
        ]
        return self

    def sample(self, n_samples, rng=None):
        rng = self._rng(rng)
        X_synth = np.column_stack([
            kde.resample(n_samples, seed=rng.randint(0, 2**31)).ravel()
            for kde in self._kdes
        ])
        return X_synth


# ─────────────────────────────────────────────────────────────────────────────
# 4. Partition GMM
# ─────────────────────────────────────────────────────────────────────────────

class PartitionGMM(_BaseMethod):
    """
    Gaussian Mixture Model fitted to the partition.

    Uses sklearn's GaussianMixture.  The number of components is chosen
    as min(n_components, n // min_samples_per_component) to avoid
    over-parameterisation on small partitions.

    Strengths
    ---------
    - Handles multi-modal partitions better than a single Gaussian
    - Full covariance captures correlations

    Weaknesses
    ----------
    - Slower fit than KDE, especially with many components
    - Requires enough samples per component for stable estimation

    Parameters
    ----------
    n_components : int, default=3
    min_samples_per_component : int, default=10
    covariance_type : str, default='full'
    """

    name = 'PartitionGMM'

    def __init__(self, n_components=3, min_samples_per_component=10,
                 covariance_type='full'):
        self.n_components = n_components
        self.min_samples_per_component = min_samples_per_component
        self.covariance_type = covariance_type

    def fit(self, X_part):
        from sklearn.mixture import GaussianMixture
        X = np.asarray(X_part, dtype=float)
        n = len(X)
        k = max(1, min(self.n_components, n // self.min_samples_per_component))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=42,
                max_iter=200,
                reg_covar=1e-4,
            )
            self._gmm.fit(X)
        return self

    def sample(self, n_samples, rng=None):
        seed = self._rng(rng).randint(0, 2**31)
        X_synth, _ = self._gmm.sample(n_samples)
        return X_synth


# ─────────────────────────────────────────────────────────────────────────────
# 5. KNN Interpolation  (SMOTE-style)
# ─────────────────────────────────────────────────────────────────────────────

class KNNInterpolation(_BaseMethod):
    """
    SMOTE-style k-nearest-neighbour convex interpolation.

    For each synthetic sample:
    1. Select a random original sample x_i.
    2. Select a random k-NN of x_i (call it x_j).
    3. Synthesise: x_new = x_i + α * (x_j - x_i),  α ~ Uniform(0, 1).

    No external library required; works for any continuous tabular data.

    Strengths
    ---------
    - Preserves correlation structure (interpolates between real points)
    - Fast; scales linearly

    Weaknesses
    ----------
    - Cannot extrapolate beyond the convex hull of the original data
    - Tail preservation is inherently limited (no samples beyond observed extremes)
    - Quality degrades when the partition has very few points (k → 1)

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbours to consider.
    """

    name = 'KNNInterpolation'

    def __init__(self, k=5):
        self.k = k

    def fit(self, X_part):
        from sklearn.neighbors import NearestNeighbors
        X = np.asarray(X_part, dtype=float)
        self._X = X
        n = len(X)
        k_eff = min(self.k + 1, n)   # +1 because point itself is included
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm='auto')
        nn.fit(X)
        # Precompute neighbour indices (excluding self)
        _, indices = nn.kneighbors(X)
        self._neighbours = indices[:, 1:]   # exclude self (first column)
        return self

    def sample(self, n_samples, rng=None):
        rng = self._rng(rng)
        n = len(self._X)
        n_neigh = self._neighbours.shape[1]

        # Select random base samples
        base_idx = rng.randint(0, n, size=n_samples)
        # Select random neighbour for each base sample
        neigh_local = rng.randint(0, max(1, n_neigh), size=n_samples)
        neigh_idx = self._neighbours[base_idx, neigh_local % max(1, n_neigh)]

        alpha = rng.rand(n_samples, 1)
        X_synth = self._X[base_idx] + alpha * (self._X[neigh_idx] - self._X[base_idx])
        return X_synth


# ─────────────────────────────────────────────────────────────────────────────
# 6. Partition Bootstrap  (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class PartitionBootstrap(_BaseMethod):
    """
    Baseline: resample with replacement and add Gaussian noise.

    Noise is scaled to `noise_level` × per-feature standard deviation.
    This is the simplest possible generation method and serves as the
    performance floor for all other approaches.

    Parameters
    ----------
    noise_level : float, default=0.1
    """

    name = 'PartitionBootstrap'

    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def fit(self, X_part):
        X = np.asarray(X_part, dtype=float)
        self._X = X
        stds = X.std(axis=0)
        self._stds = np.where(stds > 1e-10, stds, 1.0)
        return self

    def sample(self, n_samples, rng=None):
        rng = self._rng(rng)
        idx = rng.choice(len(self._X), n_samples, replace=True)
        X_boot = self._X[idx].copy()
        X_boot += rng.normal(0, self.noise_level, X_boot.shape) * self._stds
        return X_boot


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODS = {
    'MultivariateKDE':          lambda: MultivariateKDE(bandwidth=0.5),
    'UniKDE-RankCoupled':       lambda: UnivariateKDERankCoupled(bandwidth=0.5),
    'UniKDE-Independent':       lambda: UnivariateKDEIndependent(bandwidth=0.5),
    'PartitionGMM':             lambda: PartitionGMM(n_components=3),
    'KNNInterpolation':         lambda: KNNInterpolation(k=5),
    'PartitionBootstrap':       lambda: PartitionBootstrap(noise_level=0.1),
}
