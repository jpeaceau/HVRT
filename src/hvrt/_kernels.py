"""
Optional Numba-compiled kernels for performance-critical HVRT operations.

Install with::

    pip install "hvrt[fast]"

When numba is not installed every symbol defined here still exists and works
correctly — the decorated functions are plain Python/NumPy implementations.
All dispatch callsites (``reduction_strategies._centroid_fps_core``,
``reduction_strategies._medoid_fps_core``, ``HVRT._compute_x_component``)
check ``_NUMBA_AVAILABLE`` and select the appropriate backend automatically.

Design principles
-----------------
* ``@njit(cache=True, fastmath=True)`` — compiled bitcode is persisted to
  ``__pycache__`` so the ~1-2 s JIT cost occurs only on the *first ever*
  process run.  Subsequent calls (including the 2 000+ FPS invocations in a
  GeoXGB training run) pay zero JIT overhead.  ``fastmath=True`` allows LLVM
  to use FMA instructions and reorder FP operations; safe here because all
  inputs are z-scored (values in [-6, 6]) and squared-distance accumulations
  have bounded magnitude.
* No-op fallback — when numba is absent, ``njit`` is replaced by a
  decorator that returns the function unchanged.  All Numba-decorated
  functions therefore remain callable as ordinary Python.
* Strict dtype contract — every kernel expects C-contiguous float64
  input.  Callers are responsible for ensuring this (typically via
  ``np.ascontiguousarray(arr, dtype=np.float64)`` before dispatch).

Numeric equivalence
-------------------
``_pairwise_target_nb`` uses three sequential passes per feature pair to
compute mean, variance, and z-score accumulation.  NumPy's block-wise
fallback (``_pairwise_target_numpy``) uses the same algorithm but with
pairwise summation internally.  Results agree to within ~1e-8 for
z-scored data (values in [-6, 6]).

The FPS kernels (``_centroid_fps_core_nb``, ``_medoid_fps_core_nb``) are
algorithmically identical to their NumPy counterparts; on non-degenerate
random data they produce **exactly** the same integer index sequences
because every comparison uses strict inequalities with the same tie-
breaking rule (first occurrence of the extreme value).
"""

import numpy as np

from ._cpp_backend import _CPP_AVAILABLE, _cpp_pairwise_target

# ---------------------------------------------------------------------------
# Numba import — fail gracefully
# ---------------------------------------------------------------------------

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE: bool = True

    def njit(*args, **kwargs):
        """Thin wrapper so call-sites always use the same name."""
        return _njit(*args, **kwargs)

except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op decorator used when numba is not installed."""
        # @njit used directly without arguments: njit(func)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # @njit(cache=True, ...) used as a factory: njit(...)(func)
        return lambda f: f


# ---------------------------------------------------------------------------
# Pairwise interaction target kernel
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _pairwise_target_nb(X_z):
    """
    Fused pairwise-interaction synthetic target (O(n) peak memory).

    For every feature pair (i, j) with i < j:

      1. Mean of X_z[:, i] * X_z[:, j]  (pass 1 over n rows)
      2. Std  of X_z[:, i] * X_z[:, j]  (pass 2 over n rows)
      3. Accumulate z-score into ``scores``           (pass 3 over n rows)

    Constant interactions (std ≤ 1e-10) contribute zero — identical to the
    NumPy fallback.

    Parameters
    ----------
    X_z : C-contiguous float64 array, shape (n_samples, n_features)

    Returns
    -------
    scores : float64 array, shape (n_samples,)
    """
    n = X_z.shape[0]
    d = X_z.shape[1]
    scores = np.zeros(n)

    for i in range(d - 1):
        for j in range(i + 1, d):
            # --- pass 1: mean ---
            m = 0.0
            for k in range(n):
                m += X_z[k, i] * X_z[k, j]
            m /= float(n)

            # --- pass 2: variance (biased) ---
            var = 0.0
            for k in range(n):
                v = X_z[k, i] * X_z[k, j] - m
                var += v * v
            std = (var / float(n)) ** 0.5

            # --- pass 3: accumulate z-scores ---
            if std > 1e-10:
                inv_std = 1.0 / std
                for k in range(n):
                    scores[k] += (X_z[k, i] * X_z[k, j] - m) * inv_std

    return scores


def _pairwise_target_numpy(X_z):
    """
    Block-wise NumPy pairwise interaction target.

    O(n·d) peak memory — one row-block at a time.  Mathematically
    equivalent to the Numba kernel and the old ``PolynomialFeatures``
    path, without sklearn overhead or the O(n·d²) intermediate matrix.

    Used as the fallback when numba is not installed.

    Parameters
    ----------
    X_z : float array, shape (n_samples, n_features)

    Returns
    -------
    scores : float64 array, shape (n_samples,)
    """
    n_samples, n_features = X_z.shape
    scores = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_features - 1):
        # Block of interactions: X[:, i] * X[:, i+1:], shape (n, n_features-i-1)
        inter_block = X_z[:, i:i + 1] * X_z[:, i + 1:]
        means = inter_block.mean(axis=0)
        stds = inter_block.std(axis=0)
        valid = stds > 1e-10
        stds_safe = np.where(valid, stds, 1.0)
        inter_z = (inter_block - means) / stds_safe
        inter_z[:, ~valid] = 0.0
        scores += inter_z.sum(axis=1)

    return scores


def _pairwise_target(X_z: np.ndarray) -> np.ndarray:
    """Dispatch: C++ → Numba → NumPy."""
    if _CPP_AVAILABLE:
        return _cpp_pairwise_target(np.ascontiguousarray(X_z, dtype=np.float64))
    if _NUMBA_AVAILABLE:
        return _pairwise_target_nb(np.ascontiguousarray(X_z, dtype=np.float64))
    return _pairwise_target_numpy(X_z)


# ---------------------------------------------------------------------------
# Centroid-seeded FPS kernel
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _centroid_fps_core_nb(X_part, budget):
    """
    Centroid-seeded Furthest Point Sampling — compiled greedy loop.

    Semantically identical to ``_centroid_fps_core`` in
    ``reduction_strategies.py``.  LLVM-compiled via Numba, eliminating the
    Python-loop overhead that is the bottleneck for the 2 000+ FPS calls
    incurred during a GeoXGB training run.

    Tie-breaking: strict ``<`` / ``>`` comparisons with first-occurrence
    semantics — identical to NumPy ``argmin`` / ``argmax`` behaviour.

    Parameters
    ----------
    X_part : C-contiguous float64 array, shape (n_points, n_features)
             Caller guarantees ``budget <= n_points``.
    budget : int  (> 0)

    Returns
    -------
    indices : int64 array, shape (budget,)
        Local indices into ``X_part`` (not global dataset indices).
    """
    n = X_part.shape[0]
    d = X_part.shape[1]

    # --- compute centroid ---
    centroid = np.zeros(d)
    for i in range(n):
        for j in range(d):
            centroid[j] += X_part[i, j]
    for j in range(d):
        centroid[j] /= float(n)

    # --- seed: point closest to centroid ---
    min_dist = np.inf
    seed = 0
    for i in range(n):
        dist = 0.0
        for j in range(d):
            diff = X_part[i, j] - centroid[j]
            dist += diff * diff
        if dist < min_dist:
            min_dist = dist
            seed = i

    # --- greedy FPS loop ---
    result = np.empty(budget, dtype=np.int64)
    result[0] = seed
    min_sq_dists = np.full(n, np.inf)

    for s in range(1, budget):
        last = result[s - 1]
        for i in range(n):
            dist = 0.0
            for j in range(d):
                diff = X_part[i, j] - X_part[last, j]
                dist += diff * diff
            if dist < min_sq_dists[i]:
                min_sq_dists[i] = dist
        farthest = 0
        max_dist = -1.0
        for i in range(n):
            if min_sq_dists[i] > max_dist:
                max_dist = min_sq_dists[i]
                farthest = i
        result[s] = farthest

    return result


# ---------------------------------------------------------------------------
# Medoid helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _exact_medoid_nb(X_part):
    """
    Exact O(n²·d) medoid: minimise sum of squared distances to all others.

    Equivalent to ``argmin(cdist(X, X, 'sqeuclidean').sum(axis=1))``.
    """
    n = X_part.shape[0]
    d = X_part.shape[1]
    best_idx = 0
    best_sum = np.inf
    for i in range(n):
        s = 0.0
        for j in range(n):
            dist = 0.0
            for k in range(d):
                diff = X_part[i, k] - X_part[j, k]
                dist += diff * diff
            s += dist
        if s < best_sum:
            best_sum = s
            best_idx = i
    return best_idx


@njit(cache=True, fastmath=True)
def _approx_medoid_nb(X_part, k):
    """
    Approximate O(n·d) medoid via centroid-nearest k candidates.

    Equivalent to ``_approximate_medoid_idx`` in ``reduction_strategies.py``
    with ``np.argsort`` used in place of ``np.argpartition`` (the latter is
    not supported in Numba nopython mode).  For random float64 data the
    candidate sets are identical because ties at position k are negligible.

    Steps
    -----
    1. Distance from each point to the centroid — O(n·d).
    2. Sort all distances; take the k nearest — O(n log n).
    3. Exact medoid within the k-candidate set — O(k²·d).
    """
    n = X_part.shape[0]
    d = X_part.shape[1]

    # Centroid
    centroid = np.zeros(d)
    for i in range(n):
        for j in range(d):
            centroid[j] += X_part[i, j]
    for j in range(d):
        centroid[j] /= float(n)

    # Distance to centroid
    dists = np.empty(n)
    for i in range(n):
        dist = 0.0
        for j in range(d):
            diff = X_part[i, j] - centroid[j]
            dist += diff * diff
        dists[i] = dist

    # Top-k centroid-nearest candidates
    order = np.argsort(dists)
    k_actual = min(k, n)
    candidates = order[:k_actual]

    # Exact medoid within candidates
    best_local = 0
    best_sum = np.inf
    for ii in range(k_actual):
        i = candidates[ii]
        s = 0.0
        for jj in range(k_actual):
            j_idx = candidates[jj]
            dist = 0.0
            for kk in range(d):
                diff = X_part[i, kk] - X_part[j_idx, kk]
                dist += diff * diff
            s += dist
        if s < best_sum:
            best_sum = s
            best_local = ii
    return int(candidates[best_local])


# Must stay in sync with reduction_strategies._MEDOID_EXACT_THRESHOLD
_MEDOID_EXACT_THRESHOLD_NB = 200


@njit(cache=True, fastmath=True)
def _medoid_fps_core_nb(X_part, budget):
    """
    Medoid-seeded Furthest Point Sampling — compiled greedy loop.

    Semantically identical to ``_medoid_fps_core`` in
    ``reduction_strategies.py``.  Uses exact O(n²·d) medoid for partitions
    ≤ 200 and approximate O(n·d) medoid above that threshold, matching the
    pure-Python threshold exactly.

    Parameters
    ----------
    X_part : C-contiguous float64 array, shape (n_points, n_features)
             Caller guarantees ``budget <= n_points``.
    budget : int  (> 0)

    Returns
    -------
    indices : int64 array, shape (budget,)
        Local indices into ``X_part``.
    """
    n = X_part.shape[0]

    # --- find medoid seed ---
    if n <= _MEDOID_EXACT_THRESHOLD_NB:
        medoid_idx = _exact_medoid_nb(X_part)
    else:
        k = min(max(int(n ** 0.5), 30), n)
        medoid_idx = _approx_medoid_nb(X_part, k)

    # --- greedy FPS loop (identical to centroid FPS after seeding) ---
    d = X_part.shape[1]
    result = np.empty(budget, dtype=np.int64)
    result[0] = medoid_idx
    min_sq_dists = np.full(n, np.inf)

    for s in range(1, budget):
        last = result[s - 1]
        for i in range(n):
            dist = 0.0
            for j in range(d):
                diff = X_part[i, j] - X_part[last, j]
                dist += diff * diff
            if dist < min_sq_dists[i]:
                min_sq_dists[i] = dist
        farthest = 0
        max_dist = -1.0
        for i in range(n):
            if min_sq_dists[i] > max_dist:
                max_dist = min_sq_dists[i]
                farthest = i
        result[s] = farthest

    return result
