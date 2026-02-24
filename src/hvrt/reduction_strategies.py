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


# Partitions smaller than this use the exact O(n²) medoid.
# Above the threshold the approximate O(n·d) method is used instead.
_MEDOID_EXACT_THRESHOLD = 200


def _approximate_medoid_idx(X_part: np.ndarray) -> int:
    """
    Approximate the medoid index via a centroid-nearest candidate set.

    Computing all n² pairwise distances for the exact medoid is O(n²·d).
    For HVRT's variance-structured partitions, the true medoid almost always
    lies close to the centroid.  This function:

    1. Computes Euclidean distance from each point to the centroid — O(n·d).
    2. Selects the k = max(30, ⌊√n⌋) centroid-nearest candidates using
       ``np.argpartition`` — O(n).
    3. Finds the exact medoid within that candidate set — O(k²·d).

    Total cost: O(n·d + k²·d) = O(n·d) since k = O(√n).

    The approximation quality is high for HVRT partitions because they are
    compact, roughly unimodal regions: the true medoid is virtually always
    within the top-√n centroid-nearest points.
    """
    n = len(X_part)
    k = min(max(int(n ** 0.5), 30), n)

    centroid = X_part.mean(axis=0)
    dists_sq = np.sum((X_part - centroid) ** 2, axis=1)
    # argpartition: O(n), avoids a full sort
    candidate_idx = np.argpartition(dists_sq, k - 1)[:k]

    # Exact medoid within the candidate set
    pairwise_sq = cdist(
        X_part[candidate_idx], X_part[candidate_idx], metric='sqeuclidean'
    )
    local_best = int(np.argmin(pairwise_sq.sum(axis=1)))
    return int(candidate_idx[local_best])


def _medoid_fps_core(X_part: np.ndarray, budget: int) -> np.ndarray:
    """Medoid-seeded FPS on a single partition; returns local indices."""
    n_points = len(X_part)

    if n_points <= _MEDOID_EXACT_THRESHOLD:
        # Small partition: exact O(n²·d) medoid
        pairwise_sq = cdist(X_part, X_part, metric='sqeuclidean')
        medoid_idx = int(np.argmin(pairwise_sq.sum(axis=1)))
    else:
        # Large partition: approximate O(n·d) medoid via centroid-nearest set
        medoid_idx = _approximate_medoid_idx(X_part)

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
# Private per-partition dispatch helpers (for joblib)
# ---------------------------------------------------------------------------

def _centroid_fps_partition(
    global_indices: np.ndarray, X_part: np.ndarray, budget: int
) -> np.ndarray:
    """Return global indices selected by centroid FPS on one partition."""
    if len(global_indices) <= budget:
        return global_indices
    local = _centroid_fps_core(X_part, budget)
    return global_indices[local]


def _medoid_fps_partition(
    global_indices: np.ndarray, X_part: np.ndarray, budget: int
) -> np.ndarray:
    """Return global indices selected by medoid FPS on one partition."""
    if len(global_indices) <= budget:
        return global_indices
    local = _medoid_fps_core(X_part, budget)
    return global_indices[local]


def _variance_ordered_partition(
    global_indices: np.ndarray, X_part: np.ndarray, budget: int
) -> np.ndarray:
    """Return global indices selected by variance-ordered on one partition."""
    from sklearn.neighbors import NearestNeighbors

    n_points = len(global_indices)
    if n_points <= budget:
        return global_indices
    k = min(10, n_points - 1)
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(X_part)
    distances, _ = nn.kneighbors(X_part)
    local_variance = distances.var(axis=1)
    local = np.argsort(-local_variance, kind='stable')[:budget].astype(np.int64)
    return global_indices[local]


def _stratified_partition(
    global_indices: np.ndarray, X_part: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    """Return global indices selected by stratified random on one partition."""
    n_points = len(global_indices)
    if n_points <= budget:
        return global_indices
    rng = np.random.RandomState(seed)
    local = np.sort(rng.choice(n_points, size=budget, replace=False))
    return global_indices[local]


# ---------------------------------------------------------------------------
# Internal dispatch helper
# ---------------------------------------------------------------------------

_MIN_PARALLEL_TASKS = 6      # don't dispatch to loky for trivially few partitions
_MIN_PARALLEL_SAMPLES = 3000  # minimum total samples to justify loky IPC overhead


def _run_parallel(fn, tasks, n_jobs):
    """
    Execute ``fn(*task)`` for each task, in parallel when n_jobs != 1.

    Uses the ``loky`` (process) backend.  Although each FPS iteration
    contains NumPy calls that release the GIL, the Python for-loop
    bookkeeping between those calls holds the GIL long enough that threads
    serialise against each other and see no throughput benefit.  True
    multiprocessing via loky gives genuine CPU parallelism for the
    GIL-bound portions.  The loky worker pool is lazily created on the
    first call and then reused within the Python session; pool initialisation
    is a one-time overhead of ~1–2 s on Windows.

    Two guards prevent dispatching when overhead would dominate:
    - Fewer than ``_MIN_PARALLEL_TASKS`` non-empty partitions.
    - Fewer than ``_MIN_PARALLEL_SAMPLES`` total samples across all tasks
      (t[1] is the per-partition X_part array in every task format).

    Returns results in the same order as ``tasks``.
    """
    if n_jobs == 1 or len(tasks) < _MIN_PARALLEL_TASKS:
        return [fn(*t) for t in tasks]
    total_samples = sum(len(t[1]) for t in tasks)
    if total_samples < _MIN_PARALLEL_SAMPLES:
        return [fn(*t) for t in tasks]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(fn)(*t) for t in tasks)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

def centroid_fps(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Number of parallel jobs.  -1 uses all available cores.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples.
    """
    tasks = [
        (global_indices, X_part, budget)
        for global_indices, X_part, budget in _iter_partitions(
            X_z, partition_ids, unique_partitions, budgets
        )
    ]
    results = _run_parallel(_centroid_fps_partition, tasks, n_jobs)
    if not results:
        return np.array([], dtype=np.int64)
    return np.concatenate(results).astype(np.int64)


def medoid_fps(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Number of parallel jobs.  -1 uses all available cores.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples.
    """
    tasks = [
        (global_indices, X_part, budget)
        for global_indices, X_part, budget in _iter_partitions(
            X_z, partition_ids, unique_partitions, budgets
        )
    ]
    results = _run_parallel(_medoid_fps_partition, tasks, n_jobs)
    if not results:
        return np.array([], dtype=np.int64)
    return np.concatenate(results).astype(np.int64)


def variance_ordered(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Number of parallel jobs.  -1 uses all available cores.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples (ordered by variance, descending).
    """
    tasks = [
        (global_indices, X_part, budget)
        for global_indices, X_part, budget in _iter_partitions(
            X_z, partition_ids, unique_partitions, budgets
        )
    ]
    results = _run_parallel(_variance_ordered_partition, tasks, n_jobs)
    if not results:
        return np.array([], dtype=np.int64)
    return np.concatenate(results).astype(np.int64)


def stratified(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    random_state: int,
    n_jobs: int = 1,
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
    n_jobs : int, default 1
        Number of parallel jobs.  -1 uses all available cores.

    Returns
    -------
    indices : ndarray of int64
        Global indices of selected samples.
    """
    rng = np.random.RandomState(random_state)
    tasks = [
        (global_indices, X_part, budget, int(rng.randint(0, 2 ** 31)))
        for global_indices, X_part, budget in _iter_partitions(
            X_z, partition_ids, unique_partitions, budgets
        )
    ]
    results = _run_parallel(_stratified_partition, tasks, n_jobs)
    if not results:
        return np.array([], dtype=np.int64)
    return np.concatenate(results).astype(np.int64)


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
