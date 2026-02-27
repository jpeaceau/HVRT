"""
Selection Strategies for HVRT Sample Reduction

Partition-aware selection strategies.  Each strategy owns the full
iteration loop across all partitions and returns global indices into
the original dataset.

Two-stage stateful protocol
---------------------------
Each strategy implements ``StatefulSelectionStrategy``:

- ``prepare(X_z, partition_ids, unique_partitions) -> SelectionContext``
  Called once at fit() time (or lazily on first reduce() call).  Precomputes
  partition metadata — sorted indices, slice offsets, partition sizes — and
  returns a frozen ``SelectionContext``.  The context is cached by
  ``HVRT._base._get_strategy_context()`` and reused across reduce() calls
  on the same tree, avoiding redundant computation.

- ``select(context, budgets, random_state, n_jobs=1) -> ndarray[int]``
  Called per reduce() invocation.  Uses the cached context to perform the
  selection.  ``n_jobs`` is forwarded from the model's constructor parameter.
"""

import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from scipy.spatial.distance import cdist

from ._budgets import _iter_partitions, _partition_pos
from ._kernels import (
    _NUMBA_AVAILABLE,
    _centroid_fps_core_nb,
    _medoid_fps_core_nb,
)


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionContext:
    """
    Immutable partition metadata for stateful selection strategies.

    Produced by ``prepare()`` and consumed by ``select()``.  All fields
    are required — no defaults — so frozen-dataclass inheritance is safe
    on Python 3.8+.

    Attributes
    ----------
    X_z : ndarray (n_samples, n_features)
        Full z-score-normalised feature matrix (by reference; no copy).
    pos : ndarray (n_samples,)
        Integer partition index (0..n_parts-1) for each sample.
    sort_idx : ndarray (n_samples,)
        Stable argsort of ``pos``; groups samples by partition.
    part_starts : ndarray (n_parts,)
        Start offset of each partition's block in ``sort_idx``.
    part_sizes : ndarray (n_parts,)
        Number of samples in each partition.
    n_parts : int
    n_features : int
    """
    X_z: np.ndarray
    pos: np.ndarray
    sort_idx: np.ndarray
    part_starts: np.ndarray
    part_sizes: np.ndarray
    n_parts: int
    n_features: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StatefulSelectionStrategy(Protocol):
    """
    Two-stage stateful protocol for partition-aware selection strategies.

    ``prepare()`` is called once (cacheable) to precompute partition
    metadata.  ``select()`` is called per reduce() invocation and uses
    the cached context.

    ``n_jobs`` in ``select()`` is forwarded from the model's ``n_jobs``
    constructor parameter.  Strategies that are fully vectorised
    (``StratifiedStrategy``) ignore it; FPS strategies use it for
    cross-partition parallelism via joblib.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> SelectionContext:
        """Build and return a frozen SelectionContext."""
        ...

    def select(
        self,
        context: SelectionContext,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Select samples and return their global indices.

        Parameters
        ----------
        context : SelectionContext
        budgets : ndarray of int, shape (n_parts,)
        random_state : int
        n_jobs : int, default 1

        Returns
        -------
        indices : ndarray of int64
        """
        ...


# ---------------------------------------------------------------------------
# Shared context builder
# ---------------------------------------------------------------------------

def _build_selection_context(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
) -> SelectionContext:
    """
    Build a :class:`SelectionContext` from raw partition arrays.

    Computes integer partition positions, stable argsort, partition slice
    offsets and sizes.  All results are stored by reference (no copy of
    ``X_z``).
    """
    n_parts = len(unique_partitions)
    d = X_z.shape[1]
    pos = _partition_pos(partition_ids, unique_partitions)
    sort_idx = np.argsort(pos, kind='stable')
    part_starts = np.searchsorted(pos[sort_idx], np.arange(n_parts))
    part_sizes = np.bincount(pos, minlength=n_parts)
    return SelectionContext(
        X_z=X_z,
        pos=pos,
        sort_idx=sort_idx,
        part_starts=part_starts,
        part_sizes=part_sizes,
        n_parts=n_parts,
        n_features=d,
    )


# ---------------------------------------------------------------------------
# Private per-partition FPS cores (inner greedy loops only)
# ---------------------------------------------------------------------------

def _centroid_fps_core(X_part: np.ndarray, budget: int) -> np.ndarray:
    """
    Centroid-seeded FPS on a single partition; returns local indices.

    Dispatches to the Numba-compiled kernel when ``numba`` is installed
    (``pip install hvrt[fast]``), otherwise runs the pure-NumPy path.
    Both paths are algorithmically identical and produce the same indices
    on non-degenerate data.
    """
    if _NUMBA_AVAILABLE:
        return _centroid_fps_core_nb(
            np.ascontiguousarray(X_part, dtype=np.float64), int(budget)
        )
    # --- pure-NumPy fallback ---
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
    """
    Medoid-seeded FPS on a single partition; returns local indices.

    Dispatches to the Numba-compiled kernel when ``numba`` is installed
    (``pip install hvrt[fast]``), otherwise runs the pure-NumPy/scipy path.
    Both paths use the same threshold (200) for exact vs approximate medoid.
    """
    if _NUMBA_AVAILABLE:
        return _medoid_fps_core_nb(
            np.ascontiguousarray(X_part, dtype=np.float64), int(budget)
        )
    # --- pure-NumPy/scipy fallback ---
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
# Strategy classes — StatefulSelectionStrategy implementations
# ---------------------------------------------------------------------------


class StratifiedStrategy:
    """
    Fully-vectorised stratified random selection.

    Assigns a random priority key to every sample in a single
    ``rng.random(n_samples)`` call, then uses ``np.lexsort`` to sort all
    samples by ``(partition, key)`` in one pass.  Samples appear in random
    order within each partition's block; the first ``budget[p]`` are
    selected.  No Python loop over partitions.

    Memory: O(n_samples) extra — one float64 key array and one int64 order
    array, both of length n_samples.

    Parameters
    ----------
    None — this strategy has no tunable constructor parameters.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> SelectionContext:
        """Build SelectionContext (pre-sort partition structure)."""
        return _build_selection_context(X_z, partition_ids, unique_partitions)

    def select(
        self,
        context: SelectionContext,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Select samples using a fully-vectorised lexsort.

        ``n_jobs`` is accepted for protocol compatibility but ignored
        (the implementation has no per-partition parallelism to exploit).
        """
        n = context.X_z.shape[0]
        rng = np.random.RandomState(random_state)

        # Assign a uniform random priority key to every sample
        keys = rng.random(n)

        # Single lexsort: primary key = partition pos, secondary = random key.
        # Within each partition's block the samples are in random key order.
        order = np.lexsort((keys, context.pos))

        # Within-partition rank: position of order[j] within its partition block.
        # part_starts[p] is the same in `order` and `sort_idx` because both
        # sort by pos as the primary key (stable sort; partition groups are
        # contiguous at the same offsets regardless of within-group ordering).
        pos_ordered = context.pos[order]
        rank = np.arange(n, dtype=np.intp) - context.part_starts[pos_ordered]
        budget_per_sample = budgets[pos_ordered]
        mask = rank < budget_per_sample

        return order[mask]


class VarianceOrderedStrategy:
    """
    Variance-Ordered Selection.

    Selects samples with the highest local k-NN variance.  Prioritises
    boundary and transition regions.  The k-NN search is performed
    within each partition (cross-partition neighbours are excluded by
    construction) using scikit-learn's ``NearestNeighbors``.

    The :class:`SelectionContext` caches the sorted partition structure so
    that per-partition array slicing is O(1) instead of O(n_samples).

    ``n_jobs`` controls cross-partition parallelism via loky.  For large
    datasets with many partitions, ``n_jobs=-1`` can give significant
    speedups; for small data the overhead dominates.

    Parameters
    ----------
    None — constructor accepts no parameters.  Pass the instance directly
    to ``HVRT.reduce(method=...)``; the model's ``n_jobs`` is forwarded
    to ``select()`` automatically.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> SelectionContext:
        """Build SelectionContext."""
        return _build_selection_context(X_z, partition_ids, unique_partitions)

    def select(
        self,
        context: SelectionContext,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Select samples by descending local k-NN variance."""
        tasks = []
        for p in range(context.n_parts):
            start = int(context.part_starts[p])
            size = int(context.part_sizes[p])
            global_idx = context.sort_idx[start:start + size]
            X_part = context.X_z[global_idx]
            budget = int(budgets[p])
            tasks.append((global_idx, X_part, budget))

        results = _run_parallel(_variance_ordered_partition, tasks, n_jobs)
        if not results:
            return np.array([], dtype=np.int64)
        return np.concatenate(results).astype(np.int64)


class CentroidFPSStrategy:
    """
    Centroid-seeded Furthest Point Sampling (default strategy).

    Iterates over all partitions and selects diverse samples by greedily
    choosing points farthest from the current selection, seeded at the
    partition centroid.

    The greedy FPS loop is O(budget²) and is irreducibly sequential within
    each partition.  Cross-partition parallelism is available via ``n_jobs``
    (loky multiprocessing).  The :class:`SelectionContext` provides
    O(1) partition slicing vs. the per-call ``_iter_partitions`` scan.

    Parameters
    ----------
    None — constructor accepts no parameters.  The model's ``n_jobs`` is
    forwarded to ``select()`` automatically.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> SelectionContext:
        """Build SelectionContext."""
        return _build_selection_context(X_z, partition_ids, unique_partitions)

    def select(
        self,
        context: SelectionContext,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Select samples via centroid-seeded FPS."""
        tasks = []
        for p in range(context.n_parts):
            start = int(context.part_starts[p])
            size = int(context.part_sizes[p])
            global_idx = context.sort_idx[start:start + size]
            X_part = context.X_z[global_idx]
            budget = int(budgets[p])
            tasks.append((global_idx, X_part, budget))

        results = _run_parallel(_centroid_fps_partition, tasks, n_jobs)
        if not results:
            return np.array([], dtype=np.int64)
        return np.concatenate(results).astype(np.int64)


class MedoidFPSStrategy:
    """
    Medoid-seeded Furthest Point Sampling.

    Like :class:`CentroidFPSStrategy` but seeds at the partition medoid
    (the actual sample minimising the sum of distances to all others).
    More robust to outliers at the cost of the extra medoid computation.

    For partitions > 200 samples, an approximate O(n·d) medoid is used
    (centroid-nearest √n candidates evaluated for exact medoid within
    that set) in place of the exact O(n²·d) computation.

    Parameters
    ----------
    None — constructor accepts no parameters.
    """

    def prepare(
        self,
        X_z: np.ndarray,
        partition_ids: np.ndarray,
        unique_partitions: np.ndarray,
    ) -> SelectionContext:
        """Build SelectionContext."""
        return _build_selection_context(X_z, partition_ids, unique_partitions)

    def select(
        self,
        context: SelectionContext,
        budgets: np.ndarray,
        random_state: int,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Select samples via medoid-seeded FPS."""
        tasks = []
        for p in range(context.n_parts):
            start = int(context.part_starts[p])
            size = int(context.part_sizes[p])
            global_idx = context.sort_idx[start:start + size]
            X_part = context.X_z[global_idx]
            budget = int(budgets[p])
            tasks.append((global_idx, X_part, budget))

        results = _run_parallel(_medoid_fps_partition, tasks, n_jobs)
        if not results:
            return np.array([], dtype=np.int64)
        return np.concatenate(results).astype(np.int64)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

#: Centroid-seeded FPS (default strategy).  Alias: ``'fps'``, ``'centroid_fps'``.
centroid_fps = CentroidFPSStrategy()

#: Medoid-seeded FPS.  More robust to outliers.  Alias: ``'medoid_fps'``.
medoid_fps = MedoidFPSStrategy()

#: Variance-ordered selection (highest local k-NN variance first).
#: Alias: ``'variance_ordered'``.
variance_ordered = VarianceOrderedStrategy()

#: Fully-vectorised stratified random selection.  Alias: ``'stratified'``.
stratified = StratifiedStrategy()


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


def get_strategy(strategy_name: str) -> StatefulSelectionStrategy:
    """
    Get built-in selection strategy by name.

    Parameters
    ----------
    strategy_name : str
        One of: ``'fps'``, ``'centroid_fps'``, ``'medoid_fps'``,
        ``'variance_ordered'``, ``'stratified'``.

    Returns
    -------
    strategy : StatefulSelectionStrategy

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
