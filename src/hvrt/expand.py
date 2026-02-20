"""
Multivariate KDE-based sample expansion for HVRT v2.

Each partition receives its own multivariate gaussian_kde fitted on the
full (z-score-normalized) feature matrix of that partition's members.
Samples are drawn from the KDE and optionally filtered by a minimum
novelty distance threshold.

Design notes
------------
- KDEs are MULTIVARIATE: one kde per partition, fitted on all d features
  simultaneously. This captures within-partition correlation structure that
  per-feature univariate KDEs miss.
- Bandwidth: Scott's rule (scipy default) is used when bandwidth=None.
- Single-sample partitions fall back to bootstrap with tiny Gaussian noise.
- min_novelty filtering oversamples then discards samples too close to any
  original (in z-score space). Falls back gracefully if the budget cannot
  be met after max_attempts.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# KDE fitting
# ---------------------------------------------------------------------------

def fit_partition_kdes(X_z, partition_ids, unique_partitions, bandwidth=None):
    """
    Fit a multivariate gaussian_kde for every partition.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
        Z-score-normalized data.
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    bandwidth : float or None
        Scalar bandwidth factor. None → Scott's rule (scipy default).

    Returns
    -------
    kdes : dict {int pid -> gaussian_kde or None}
        None entries indicate single-sample partitions.
    """
    kdes = {}
    for pid in unique_partitions:
        X_part = X_z[partition_ids == pid]
        if len(X_part) < 2:
            kdes[int(pid)] = None
        else:
            # bandwidth: None / 'scott' / scalar → same for every partition.
            # dict {pid: float} → per-partition value (used by adaptive mode).
            if isinstance(bandwidth, dict):
                bw = bandwidth.get(int(pid), 'scott')
            else:
                bw = bandwidth if bandwidth is not None else 'scott'
            try:
                # gaussian_kde expects shape (n_features, n_samples)
                kdes[int(pid)] = gaussian_kde(X_part.T, bw_method=bw)
            except np.linalg.LinAlgError:
                # Singular covariance matrix (e.g. constant-valued partition)
                kdes[int(pid)] = None
    return kdes


# ---------------------------------------------------------------------------
# Budget allocation for expansion
# ---------------------------------------------------------------------------

def compute_expansion_budgets(
    partition_ids,
    unique_partitions,
    n_synthetic,
    variance_weighted,
    X_z,
):
    """
    Allocate synthetic samples across partitions.

    Parameters
    ----------
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    n_synthetic : int
    variance_weighted : bool
        True  → oversample high-variance (tail) partitions.
        False → proportional to partition size (preserves distribution).
    X_z : ndarray (n_samples, n_features)

    Returns
    -------
    budgets : ndarray of int, shape (len(unique_partitions),)
    """
    partition_sizes = np.array(
        [np.sum(partition_ids == pid) for pid in unique_partitions], dtype=float
    )

    if variance_weighted:
        weights = np.array(
            [np.mean(np.abs(X_z[partition_ids == pid])) for pid in unique_partitions]
        )
        weights = np.maximum(weights, 1e-10)
        weights = weights / weights.sum()
    else:
        weights = partition_sizes / partition_sizes.sum()

    budgets = np.maximum(0, (weights * n_synthetic).astype(int))

    while budgets.sum() > n_synthetic:
        budgets[np.argmax(budgets)] -= 1
    while budgets.sum() < n_synthetic:
        budgets[np.argmin(budgets)] += 1

    return budgets


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_from_kdes(
    kdes,
    unique_partitions,
    budgets,
    X_z,
    partition_ids,
    random_state,
    min_novelty=0.0,
    oversample_factor=5,
    max_attempts=10,
):
    """
    Draw synthetic samples from per-partition multivariate KDEs.

    Parameters
    ----------
    kdes : dict {pid -> gaussian_kde or None}
    unique_partitions : ndarray
    budgets : ndarray of int
    X_z : ndarray (n_samples, n_features)
        Original data in z-score space (used for novelty filtering).
    partition_ids : ndarray (n_samples,)
    random_state : int
    min_novelty : float
        Minimum Euclidean distance (in z-score space) from any original
        sample. 0.0 disables filtering.
    oversample_factor : int
        When novelty filtering is active, draw this many extra samples per
        needed sample before filtering.
    max_attempts : int
        Maximum oversampling rounds before falling back without constraint.

    Returns
    -------
    X_synthetic : ndarray (n_synthetic, n_features) in z-score space
    """
    rng = np.random.RandomState(random_state)
    all_synthetic = []

    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue

        kde = kdes.get(int(pid))
        X_part = X_z[partition_ids == pid]

        if kde is None:
            # Single-sample partition: bootstrap with tiny noise
            base = np.tile(X_part[0], (budget, 1))
            base += rng.normal(0, 0.01, base.shape)
            all_synthetic.append(base)
            continue

        if min_novelty <= 0.0:
            # No filtering: draw directly
            samples = kde.resample(budget, seed=int(rng.randint(0, 2**31))).T
        else:
            # Oversample → filter → repeat until budget is met
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
                # Fallback: fill remainder without novelty constraint
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
    cat_partition_freqs,
    unique_partitions,
    budgets,
    random_state,
):
    """
    Sample categorical column values from per-partition empirical distributions.

    Categorical values are drawn directly from the observed values in each
    partition with their empirical probabilities — no KDE or interpolation
    is applied.  This ensures categorical outputs are always valid members
    of the original category set.

    Parameters
    ----------
    cat_partition_freqs : dict {int pid -> list of (unique_values, probs)}
        Per-partition empirical frequency tables, one entry per categorical
        column.  ``unique_values`` is an ndarray of the original values seen
        in that partition; ``probs`` is the corresponding probability vector.
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    X_cat : ndarray (n_synthetic, n_cat_cols)
        Sampled categorical values with the same dtype as the stored values.
    """
    rng = np.random.RandomState(random_state + 1)  # offset to decorrelate from KDE seed
    all_cat = []
    n_cat_cols = None

    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue

        col_freqs = cat_partition_freqs[int(pid)]
        n_cat_cols = len(col_freqs)

        # Infer output dtype from the stored value arrays
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
# Strategy-based sampling (alternative to KDE path)
# ---------------------------------------------------------------------------

def sample_with_strategy(
    strategy_fn,
    unique_partitions,
    budgets,
    X_z,
    partition_ids,
    random_state,
    min_novelty=0.0,
    oversample_factor=5,
    max_attempts=10,
):
    """
    Draw synthetic samples using an arbitrary generation strategy callable.

    This mirrors the interface of ``sample_from_kdes`` but delegates
    per-partition sampling to ``strategy_fn`` instead of a fitted KDE.

    Parameters
    ----------
    strategy_fn : callable  (X_partition, budget, random_state) -> ndarray
        A ``GenerationStrategy``-compatible callable.
    unique_partitions : ndarray
    budgets : ndarray of int
    X_z : ndarray (n_samples, n_features)
        Original data in z-score space.
    partition_ids : ndarray (n_samples,)
    random_state : int
    min_novelty : float
        Minimum Euclidean distance from any original sample.  0.0 disables.
    oversample_factor : int
        Extra samples drawn per iteration when novelty filtering is active.
    max_attempts : int
        Maximum oversampling rounds before falling back without constraint.

    Returns
    -------
    X_synthetic : ndarray (n_synthetic, n_features) in z-score space
    """
    rng = np.random.RandomState(random_state)
    all_synthetic = []

    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue

        X_part = X_z[partition_ids == pid]
        seed = int(rng.randint(0, 2 ** 31))

        if min_novelty <= 0.0:
            samples = strategy_fn(X_part, budget, seed)
        else:
            # Oversample → filter → repeat until budget is met
            collected = []
            needed = budget
            attempts = 0

            while needed > 0 and attempts < max_attempts:
                n_draw = needed * oversample_factor
                drawn = strategy_fn(X_part, n_draw, int(rng.randint(0, 2 ** 31)))
                dists = cdist(drawn, X_part, metric='euclidean').min(axis=1)
                novel = drawn[dists >= min_novelty]
                if len(novel) > 0:
                    take = min(needed, len(novel))
                    collected.append(novel[:take])
                    needed -= take
                attempts += 1

            if needed > 0:
                fallback = strategy_fn(X_part, needed, int(rng.randint(0, 2 ** 31)))
                collected.append(fallback)

            samples = (
                np.vstack(collected) if collected
                else np.empty((0, X_z.shape[1]))
            )

        all_synthetic.append(samples[:budget])

    if not all_synthetic:
        return np.empty((0, X_z.shape[1]))

    return np.vstack(all_synthetic)


# ---------------------------------------------------------------------------
# Novelty distances
# ---------------------------------------------------------------------------

def compute_novelty_distances(X_synthetic, X_original, chunk_size=1000):
    """
    Compute minimum Euclidean distance from each synthetic sample to any
    original sample. Chunked for memory efficiency.

    Parameters
    ----------
    X_synthetic : ndarray (n_synthetic, d)
    X_original : ndarray (n_original, d)
    chunk_size : int

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
