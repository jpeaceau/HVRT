"""
Expansion helpers for HVRT v2.

Provides per-partition KDE fitting, budget allocation for expansion, KDE
sampling with optional novelty filtering, strategy-based sampling, and
categorical empirical-distribution sampling.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

from ._budgets import _compute_weights, allocate_budgets


# ---------------------------------------------------------------------------
# Private per-partition KDE helper (used by joblib workers)
# ---------------------------------------------------------------------------

def _fit_kde_partition(
    X_part: np.ndarray,
    n_features: int,
    bw,
) -> 'gaussian_kde | None':
    """
    Fit a multivariate gaussian_kde on one partition.

    Returns ``None`` when the partition is too small for a non-singular
    covariance matrix or when scipy raises a numerical error.
    """
    if len(X_part) < 2 or len(X_part) <= n_features:
        return None
    try:
        return gaussian_kde(X_part.T, bw_method=bw)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _sample_kde_partition_simple(
    kde: 'gaussian_kde | None',
    X_part: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    """
    Draw ``budget`` samples from a single-partition KDE (min_novelty=0 path).

    Falls back to bootstrap-with-tiny-noise when kde is None (single-sample
    or singular-covariance partition).
    """
    if kde is None:
        rng = np.random.RandomState(seed)
        base = np.tile(X_part[0], (budget, 1))
        return base + rng.normal(0, 0.01, base.shape)
    return kde.resample(budget, seed=seed).T


# ---------------------------------------------------------------------------
# KDE fitting
# ---------------------------------------------------------------------------

def fit_partition_kdes(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    bandwidth: Union[float, str, dict, None] = None,
    n_jobs: int = 1,
) -> dict:
    """
    Fit a multivariate ``gaussian_kde`` for every partition.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
        Z-score-normalised continuous features.
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    bandwidth : float, str, dict, or None
        Passed as ``bw_method`` to ``scipy.stats.gaussian_kde``.
        ``None`` → Scott's rule.  A ``dict {pid: float}`` enables
        per-partition adaptive bandwidths.
    n_jobs : int, default 1
        Number of parallel jobs for KDE fitting.  -1 uses all cores.

    Returns
    -------
    kdes : dict[int, gaussian_kde | None]
        ``None`` entries indicate single-sample partitions (no KDE fitted).
    """
    n_features = X_z.shape[1]

    # Collect per-partition data and resolve bandwidth
    pids = []
    tasks = []
    for pid in unique_partitions:
        X_part = X_z[partition_ids == pid]
        if isinstance(bandwidth, dict):
            bw = bandwidth.get(int(pid), 'scott')
        else:
            bw = bandwidth if bandwidth is not None else 'scott'
        pids.append(int(pid))
        tasks.append((X_part, n_features, bw))

    _MIN_PAR = 6
    if n_jobs == 1 or len(tasks) < _MIN_PAR:
        kdes_list = [_fit_kde_partition(*t) for t in tasks]
    else:
        from joblib import Parallel, delayed
        # prefer='threads': gaussian_kde fit releases the GIL (scipy/BLAS).
        kdes_list = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_fit_kde_partition)(*t) for t in tasks
        )

    return {pid: kde for pid, kde in zip(pids, kdes_list)}


# ---------------------------------------------------------------------------
# Budget allocation for expansion
# ---------------------------------------------------------------------------

def compute_expansion_budgets(
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    n_synthetic: int,
    variance_weighted: bool,
    X_z: np.ndarray,
) -> np.ndarray:
    """
    Allocate synthetic sample budget across partitions.

    Parameters
    ----------
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    n_synthetic : int
        Total synthetic samples to allocate.
    variance_weighted : bool
        ``True``  — weight by mean |z-score| (oversample tail partitions).
        ``False`` — weight proportionally to partition size.
    X_z : ndarray (n_samples, n_features)

    Returns
    -------
    budgets : ndarray of int, shape (len(unique_partitions),)
    """
    weights = _compute_weights(partition_ids, unique_partitions, variance_weighted, X_z)
    return allocate_budgets(weights, n_synthetic, floor=0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_from_kdes(
    kdes: dict,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    random_state: int,
    min_novelty: float = 0.0,
    oversample_factor: int = 5,
    max_attempts: int = 10,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Draw synthetic samples from per-partition multivariate KDEs.

    Parameters
    ----------
    kdes : dict[int, gaussian_kde | None]
        From ``fit_partition_kdes``.  ``None`` entries fall back to
        bootstrap-with-noise.
    unique_partitions : ndarray
    budgets : ndarray of int
    X_z : ndarray (n_samples, n_features)
        Original data in z-score space, used for novelty filtering.
    partition_ids : ndarray (n_samples,)
    random_state : int
    min_novelty : float, default 0.0
        Minimum Euclidean distance from any original sample.
        ``0.0`` disables filtering.
    oversample_factor : int, default 5
    max_attempts : int, default 10
    n_jobs : int, default 1
        Number of parallel jobs for the min_novelty=0 fast path.
        The novelty-filtering path is always serial (deprecated feature).

    Returns
    -------
    X_synthetic : ndarray (n_synthetic, n_features), z-score space
    """
    rng = np.random.RandomState(random_state)

    if min_novelty <= 0.0:
        # Fast path: simple KDE resample — fully parallelisable.
        tasks = []
        for pid, budget in zip(unique_partitions, budgets):
            if budget == 0:
                continue
            kde = kdes.get(int(pid))
            X_part = X_z[partition_ids == pid]
            seed = int(rng.randint(0, 2 ** 31))
            tasks.append((kde, X_part, budget, seed))

        if not tasks:
            return np.empty((0, X_z.shape[1]))

        _MIN_PAR = 6
        if n_jobs == 1 or len(tasks) < _MIN_PAR:
            results = [_sample_kde_partition_simple(*t) for t in tasks]
        else:
            from joblib import Parallel, delayed
            # prefer='threads': kde.resample() releases the GIL (scipy/BLAS).
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_sample_kde_partition_simple)(*t) for t in tasks
            )

        return np.vstack(results)

    # Novelty-filtering path (deprecated, always serial — complex retry logic).
    all_synthetic = []
    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue

        kde = kdes.get(int(pid))
        X_part = X_z[partition_ids == pid]

        if kde is None:
            base = np.tile(X_part[0], (budget, 1))
            base += rng.normal(0, 0.01, base.shape)
            all_synthetic.append(base)
            continue

        collected = []
        needed = budget
        attempts = 0

        while needed > 0 and attempts < max_attempts:
            n_draw = needed * oversample_factor
            drawn = kde.resample(n_draw, seed=int(rng.randint(0, 2 ** 31))).T
            dists = cdist(drawn, X_part, metric='euclidean').min(axis=1)
            novel = drawn[dists >= min_novelty]
            if len(novel) > 0:
                take = min(needed, len(novel))
                collected.append(novel[:take])
                needed -= take
            attempts += 1

        if needed > 0:
            fallback = kde.resample(needed, seed=int(rng.randint(0, 2 ** 31))).T
            collected.append(fallback)

        samples = np.vstack(collected) if collected else np.empty((0, X_z.shape[1]))
        all_synthetic.append(samples[:budget])

    if not all_synthetic:
        return np.empty((0, X_z.shape[1]))

    return np.vstack(all_synthetic)


# ---------------------------------------------------------------------------
# Categorical sampling from per-partition empirical distributions
# ---------------------------------------------------------------------------

def sample_categorical_from_freqs(
    cat_partition_freqs: dict,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Sample categorical column values from per-partition empirical distributions.

    Parameters
    ----------
    cat_partition_freqs : dict[int, list[tuple[ndarray, ndarray]]]
        Per-partition frequency tables from _preprocessing.build_cat_partition_freqs.
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    X_cat : ndarray (n_synthetic, n_cat_cols)
    """
    rng = np.random.RandomState(random_state + 1)
    all_cat = []
    n_cat_cols = None

    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue

        col_freqs = cat_partition_freqs[int(pid)]
        n_cat_cols = len(col_freqs)
        dtype = col_freqs[0][0].dtype

        cat_block = np.empty((budget, n_cat_cols), dtype=dtype)
        for j, (values, probs) in enumerate(col_freqs):
            idx = rng.choice(len(values), size=budget, p=probs)
            cat_block[:, j] = values[idx]

        all_cat.append(cat_block)

    if not all_cat:
        n_cols = n_cat_cols if n_cat_cols is not None else 0
        return np.empty((0, n_cols), dtype=float)

    return np.vstack(all_cat)


# ---------------------------------------------------------------------------
# Novelty distances
# ---------------------------------------------------------------------------

def compute_novelty_distances(
    X_synthetic: np.ndarray,
    X_original: np.ndarray,
    chunk_size: int = 1000,
) -> np.ndarray:
    """
    Compute minimum Euclidean distance from each synthetic sample to any
    original sample.  Chunked to bound peak memory usage.

    Parameters
    ----------
    X_synthetic : ndarray (n_synthetic, d)
    X_original : ndarray (n_original, d)
    chunk_size : int, default 1000

    Returns
    -------
    min_dists : ndarray (n_synthetic,)
    """
    n = len(X_synthetic)
    min_dists = np.empty(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X_synthetic[start:end]
        min_dists[start:end] = cdist(chunk, X_original, metric='euclidean').min(axis=1)

    return min_dists
