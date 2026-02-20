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
# KDE fitting
# ---------------------------------------------------------------------------

def fit_partition_kdes(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    bandwidth: Union[float, str, dict, None] = None,
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

    Returns
    -------
    kdes : dict[int, gaussian_kde | None]
        ``None`` entries indicate single-sample partitions (no KDE fitted).
    """
    kdes = {}
    for pid in unique_partitions:
        X_part = X_z[partition_ids == pid]
        if len(X_part) < 2:
            kdes[int(pid)] = None
        else:
            if isinstance(bandwidth, dict):
                bw = bandwidth.get(int(pid), 'scott')
            else:
                bw = bandwidth if bandwidth is not None else 'scott'
            try:
                kdes[int(pid)] = gaussian_kde(X_part.T, bw_method=bw)
            except np.linalg.LinAlgError:
                kdes[int(pid)] = None
    return kdes


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

    Returns
    -------
    X_synthetic : ndarray (n_synthetic, n_features), z-score space
    """
    rng = np.random.RandomState(random_state)
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

        if min_novelty <= 0.0:
            samples = kde.resample(budget, seed=int(rng.randint(0, 2**31))).T
        else:
            collected = []
            needed = budget
            attempts = 0

            while needed > 0 and attempts < max_attempts:
                n_draw = needed * oversample_factor
                drawn = kde.resample(n_draw, seed=int(rng.randint(0, 2**31))).T
                dists = cdist(drawn, X_part, metric='euclidean').min(axis=1)
                novel = drawn[dists >= min_novelty]
                if len(novel) > 0:
                    take = min(needed, len(novel))
                    collected.append(novel[:take])
                    needed -= take
                attempts += 1

            if needed > 0:
                fallback = kde.resample(needed, seed=int(rng.randint(0, 2**31))).T
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
