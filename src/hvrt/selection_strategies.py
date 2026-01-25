"""
Selection Strategies for H-VRT Sample Reduction

Type definitions and built-in strategies for within-partition sample selection.
"""

import numpy as np
from typing import Protocol, runtime_checkable

from scipy.spatial.distance import cdist


@runtime_checkable
class SelectionStrategy(Protocol):
    """
    Protocol for selection strategies.

    A selection strategy is a callable that selects samples from a partition.
    Implementations should be deterministic for reproducibility.
    """

    def __call__(
        self,
        X_partition: np.ndarray,
        n_select: int,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Select samples from a partition.

        Parameters
        ----------
        X_partition : ndarray of shape (n_partition_samples, n_features)
            Feature matrix for samples in this partition
        n_select : int
            Number of samples to select
        random_state : int, default=42
            Random seed for reproducibility

        Returns
        -------
        selected_indices : ndarray of shape (n_select,)
            Indices of selected samples (local to partition)
        """
        ...


def centroid_fps(
    X_partition: np.ndarray,
    n_select: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Centroid-seeded Furthest Point Sampling (Default Strategy).

    Deterministically selects diverse samples by iteratively choosing points
    farthest from the current selection, seeded at the partition centroid.

    Parameters
    ----------
    X_partition : ndarray of shape (n_partition_samples, n_features)
        Feature matrix for partition
    n_select : int
        Number of samples to select
    random_state : int, default=42
        Unused (kept for interface compatibility)

    Returns
    -------
    selected_indices : ndarray of shape (n_select,)
        Indices of selected samples
    """
    n_points, n_features = X_partition.shape

    # Handle edge case: partition smaller than requested samples
    if n_points <= n_select:
        return np.arange(n_points)

    # Compute centroid for deterministic seed
    centroid = X_partition.mean(axis=0)

    # Use squared distances (monotonic ordering, no sqrt needed - faster)
    diff = X_partition - centroid
    squared_distances = np.sum(diff * diff, axis=1)

    # Seed: Closest to centroid (deterministic)
    seed_idx = np.argmin(squared_distances)
    selected = [seed_idx]
    min_squared_distances = np.full(n_points, np.inf)

    # Greedy FPS: Select points maximizing minimum distance to selected set
    for _ in range(n_select - 1):
        last_idx = selected[-1]

        # Vectorized squared distance update
        diff = X_partition - X_partition[last_idx]
        squared_distances = np.sum(diff * diff, axis=1)
        min_squared_distances = np.minimum(min_squared_distances, squared_distances)

        # Select farthest point
        next_idx = np.argmax(min_squared_distances)
        selected.append(next_idx)

    return np.array(selected, dtype=np.int64)


def medoid_fps(
    X_partition: np.ndarray,
    n_select: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Medoid-seeded Furthest Point Sampling.

    Similar to centroid FPS but seeds at the partition medoid (the actual
    sample minimizing sum of distances to all others). More robust to outliers.

    Parameters
    ----------
    X_partition : ndarray of shape (n_partition_samples, n_features)
        Feature matrix for partition
    n_select : int
        Number of samples to select
    random_state : int, default=42
        Unused (kept for interface compatibility)

    Returns
    -------
    selected_indices : ndarray of shape (n_select,)
        Indices of selected samples
    """
    n_points, n_features = X_partition.shape

    if n_points <= n_select:
        return np.arange(n_points)

    # Compute medoid: sample minimizing sum of distances to all others
    # Use squared distances for efficiency
    pairwise_squared_dist = cdist(X_partition, X_partition, metric='sqeuclidean')
    medoid_idx = np.argmin(pairwise_squared_dist.sum(axis=1))

    # FPS from medoid (same algorithm as centroid FPS)
    selected = [medoid_idx]
    min_squared_distances = np.full(n_points, np.inf)

    for _ in range(n_select - 1):
        last_idx = selected[-1]

        diff = X_partition - X_partition[last_idx]
        squared_distances = np.sum(diff * diff, axis=1)
        min_squared_distances = np.minimum(min_squared_distances, squared_distances)

        next_idx = np.argmax(min_squared_distances)
        selected.append(next_idx)

    return np.array(selected, dtype=np.int64)


def variance_ordered(
    X_partition: np.ndarray,
    n_select: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Variance-Ordered Selection.

    Selects samples with the highest local variance based on k-NN distances.
    Prioritizes boundary and transition regions.

    Parameters
    ----------
    X_partition : ndarray of shape (n_partition_samples, n_features)
        Feature matrix for partition
    n_select : int
        Number of samples to select
    random_state : int, default=42
        Unused (kept for interface compatibility)

    Returns
    -------
    selected_indices : ndarray of shape (n_select,)
        Indices of selected samples (ordered by variance, descending)
    """
    n_points, n_features = X_partition.shape

    if n_points <= n_select:
        return np.arange(n_points)

    # Compute local variance using k-nearest neighbors
    from sklearn.neighbors import NearestNeighbors

    k_neighbors = min(10, n_points - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto')
    nn.fit(X_partition)

    distances, _ = nn.kneighbors(X_partition)

    # Local variance = variance of distances to k nearest neighbors
    local_variance = distances.var(axis=1)

    # Select top-n by variance (stable sort for determinism)
    # Negative to get descending order
    sorted_indices = np.argsort(-local_variance, kind='stable')

    return sorted_indices[:n_select].astype(np.int64)


def stratified(
    X_partition: np.ndarray,
    n_select: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Stratified Random Sampling.

    Random sampling within partition. Provides a baseline for comparison
    with deterministic strategies.

    Parameters
    ----------
    X_partition : ndarray of shape (n_partition_samples, n_features)
        Feature matrix for partition
    n_select : int
        Number of samples to select
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    selected_indices : ndarray of shape (n_select,)
        Randomly selected sample indices
    """
    n_points = len(X_partition)

    if n_points <= n_select:
        return np.arange(n_points)

    rng = np.random.RandomState(random_state)
    selected_indices = rng.choice(n_points, size=n_select, replace=False)

    # Sort for deterministic ordering
    return np.sort(selected_indices).astype(np.int64)


# Registry of built-in strategies
BUILTIN_STRATEGIES = {
    'centroid_fps': centroid_fps,
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
        One of: 'centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified'

    Returns
    -------
    strategy : SelectionStrategy
        Strategy callable

    Raises
    ------
    ValueError
        If strategy_name is not recognized
    """
    if strategy_name not in BUILTIN_STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available strategies: {list(BUILTIN_STRATEGIES.keys())}"
        )
    return BUILTIN_STRATEGIES[strategy_name]
