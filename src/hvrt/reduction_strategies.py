"""
Selection Strategies for HVRT Sample Reduction

Partition-aware selection strategies.  Each strategy owns the full
iteration loop across all partitions and returns global indices into
the original dataset.

The legacy per-partition API (used by ``HVRTSampleReducer``) lives in
``legacy/selection_strategies.py``.
"""

import numpy as np
from typing import Protocol, runtime_checkable

from scipy.spatial.distance import cdist

from ._budgets import _iter_partitions


@runtime_checkable
class SelectionStrategy(Protocol):
    """
    Protocol for partition-aware selection strategies.

    A selection strategy is a callable that iterates over all partitions
    and returns global indices into the original dataset.
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
        Select samples across all partitions.

        Parameters
        ----------
        X_z : ndarray (n_samples, n_features)
            Z-score-normalised feature matrix.
        partition_ids : ndarray (n_samples,)
        unique_partitions : ndarray
        budgets : ndarray of int
            Per-partition sample budgets.
        random_state : int

        Returns
        -------
        indices : ndarray of int64
            Global indices of selected samples.
        """
        ...


# ---------------------------------------------------------------------------
# Private per-partition FPS cores (inner greedy loops only)
# ---------------------------------------------------------------------------

def _centroid_fps_core(X_part: np.ndarray, budget: int) -> np.ndarray:
    """Centroid-seeded FPS on a single partition; returns local indices."""
    n_points = len(X_part)
    centroid = X_part.mean(axis=0)
    diff = X_part - centroid
    sq_dists = np.sum(diff * diff, axis=1)
    seed_idx = int(np.argmin(sq_dists))
    selected = [seed_idx]
    min_sq_dists = np.full(n_points, np.inf)

    for _ in range(budget - 1):
        last_idx = selected[-1]
        diff = X_part - X_part[last_idx]
        sq_dists = np.sum(diff * diff, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)
        selected.append(int(np.argmax(min_sq_dists)))

    return np.array(selected, dtype=np.int64)


def _medoid_fps_core(X_part: np.ndarray, budget: int) -> np.ndarray:
    """Medoid-seeded FPS on a single partition; returns local indices."""
    n_points = len(X_part)
    pairwise_sq = cdist(X_part, X_part, metric='sqeuclidean')
    medoid_idx = int(np.argmin(pairwise_sq.sum(axis=1)))
    selected = [medoid_idx]
    min_sq_dists = np.full(n_points, np.inf)

    for _ in range(budget - 1):
        last_idx = selected[-1]
        diff = X_part - X_part[last_idx]
        sq_dists = np.sum(diff * diff, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)
        selected.append(int(np.argmax(min_sq_dists)))

    return np.array(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

def centroid_fps(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Centroid-seeded Furthest Point Sampling (default strategy).

    Iterates over all partitions and selects diverse samples by greedily
    choosing points farthest from the current selection, seeded at the
    partition centroid.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int
        Unused (algorithm is deterministic); kept for protocol compatibility.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples.
    """
    selected = []
    for global_indices, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        if len(global_indices) <= budget:
            selected.extend(global_indices.tolist())
        else:
            local = _centroid_fps_core(X_part, budget)
            selected.extend(global_indices[local].tolist())
    return np.array(selected, dtype=np.int64)


def medoid_fps(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Medoid-seeded Furthest Point Sampling.

    Like centroid FPS but seeds at the partition medoid (the actual sample
    minimising the sum of distances to all others).  More robust to outliers.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int
        Unused (algorithm is deterministic); kept for protocol compatibility.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples.
    """
    selected = []
    for global_indices, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        if len(global_indices) <= budget:
            selected.extend(global_indices.tolist())
        else:
            local = _medoid_fps_core(X_part, budget)
            selected.extend(global_indices[local].tolist())
    return np.array(selected, dtype=np.int64)


def variance_ordered(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Variance-Ordered Selection.

    Selects samples with the highest local k-NN variance.  Prioritises
    boundary and transition regions.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int
        Unused (algorithm is deterministic); kept for protocol compatibility.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples (ordered by variance, descending).
    """
    from sklearn.neighbors import NearestNeighbors

    selected = []
    for global_indices, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        n_points = len(global_indices)
        if n_points <= budget:
            selected.extend(global_indices.tolist())
        else:
            k = min(10, n_points - 1)
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
            nn.fit(X_part)
            distances, _ = nn.kneighbors(X_part)
            local_variance = distances.var(axis=1)
            local = np.argsort(-local_variance, kind='stable')[:budget].astype(np.int64)
            selected.extend(global_indices[local].tolist())
    return np.array(selected, dtype=np.int64)


def stratified(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """
    Stratified Random Sampling.

    Random sampling within each partition.  Provides a baseline for
    comparison with deterministic strategies.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    random_state : int

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples.
    """
    rng = np.random.RandomState(random_state)
    selected = []
    for global_indices, X_part, budget in _iter_partitions(
        X_z, partition_ids, unique_partitions, budgets
    ):
        n_points = len(global_indices)
        if n_points <= budget:
            selected.extend(global_indices.tolist())
        else:
            local = np.sort(rng.choice(n_points, size=budget, replace=False))
            selected.extend(global_indices[local].tolist())
    return np.array(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------

# 'fps' is the canonical alias for centroid_fps
BUILTIN_STRATEGIES = {
    'centroid_fps': centroid_fps,
    'fps': centroid_fps,
    'medoid_fps': medoid_fps,
    'variance_ordered': variance_ordered,
    'stratified': stratified,
}


def get_strategy(strategy_name: str) -> SelectionStrategy:
    """
    Get built-in selection strategy by name.

    Parameters
    ----------
    strategy_name : str
        One of: ``'fps'``, ``'centroid_fps'``, ``'medoid_fps'``,
        ``'variance_ordered'``, ``'stratified'``.

    Returns
    -------
    strategy : SelectionStrategy

    Raises
    ------
    ValueError
        If strategy_name is not recognised.
    """
    if strategy_name not in BUILTIN_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy_name!r}. "
            f"Available strategies: {list(BUILTIN_STRATEGIES.keys())}"
        )
    return BUILTIN_STRATEGIES[strategy_name]
