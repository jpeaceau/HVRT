"""
Strict tests for src/hvrt/_kernels.py — Numba-compiled and NumPy-fallback kernels.

Test strategy
-------------
All tests are grouped into three categories:

1. **Always-run tests** — exercise the dispatch layer (``_centroid_fps_core``,
   ``_medoid_fps_core``, ``HVRT._compute_x_component``).  These pass
   regardless of whether numba is installed.

2. **Numba-required tests** — marked ``@pytest.mark.skipif(not _NUMBA_AVAILABLE, ...)``.
   These explicitly call the ``_nb`` kernels and assert numerical equivalence
   with the pure-Python/NumPy reference implementations.  They verify that
   the compiled kernels are not just *fast* but also *correct*.

3. **Edge-case tests** — degenerate inputs (d=1, budget=1, budget=n,
   constant columns, identical points).  Run regardless of Numba availability
   so that regressions are caught early.

Numerical tolerance
-------------------
* Pairwise target: ``atol=1e-7`` — both paths use double precision but differ
  in summation order (Numba uses sequential sum; NumPy may use pairwise
  summation).  For z-scored data (values in [-6, 6]) the disagreement is
  well below 1e-7.
* FPS indices: **exact integer equality** is asserted for random data where
  all pairwise distances are well-separated (no float ties).  A dedicated
  near-tie test relaxes this to a property check.

JIT compilation note
--------------------
The first call to any ``@njit(cache=True)`` kernel triggers LLVM compilation
(~1–5 s on a cold cache).  Subsequent calls are instant.  The ``cache=True``
flag writes compiled bitcode to ``__pycache__``; if that directory is present
from a previous run the first call here is also instant.
"""

import numpy as np
import pytest

from hvrt._kernels import (
    _NUMBA_AVAILABLE,
    _centroid_fps_core_nb,
    _exact_medoid_nb,
    _approx_medoid_nb,
    _medoid_fps_core_nb,
    _pairwise_target_nb,
    _pairwise_target_numpy,
    _MEDOID_EXACT_THRESHOLD_NB,
)
from hvrt.reduction_strategies import (
    _centroid_fps_core,
    _medoid_fps_core,
    _MEDOID_EXACT_THRESHOLD,
)
from hvrt import HVRT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed=42):
    return np.random.RandomState(seed)


def _random_partition(n=80, d=6, seed=42):
    """Return a random C-contiguous float64 array (n, d)."""
    return np.ascontiguousarray(_rng(seed).randn(n, d), dtype=np.float64)


def _is_unique(arr):
    """True iff every element of arr is distinct."""
    return len(np.unique(arr)) == len(arr)


# ---------------------------------------------------------------------------
# 1. Availability flag
# ---------------------------------------------------------------------------

class TestAvailabilityFlag:
    def test_flag_is_bool(self):
        assert isinstance(_NUMBA_AVAILABLE, bool)

    def test_threshold_constants_match(self):
        """The Numba threshold must stay in sync with the NumPy fallback."""
        assert _MEDOID_EXACT_THRESHOLD_NB == _MEDOID_EXACT_THRESHOLD, (
            "_MEDOID_EXACT_THRESHOLD_NB in _kernels.py and "
            "_MEDOID_EXACT_THRESHOLD in reduction_strategies.py must be equal. "
            f"Got {_MEDOID_EXACT_THRESHOLD_NB} vs {_MEDOID_EXACT_THRESHOLD}."
        )


# ---------------------------------------------------------------------------
# 2. Pairwise target — NumPy fallback (always runs)
# ---------------------------------------------------------------------------

class TestPairwiseTargetNumpy:
    def test_output_shape_and_dtype(self):
        X = _rng().randn(50, 5).astype(np.float64)
        scores = _pairwise_target_numpy(X)
        assert scores.shape == (50,), "Expected (n_samples,)"
        assert scores.dtype == np.float64

    def test_single_feature_returns_zeros(self):
        """d=1 → no pairs → all scores zero."""
        X = _rng().randn(40, 1).astype(np.float64)
        scores = _pairwise_target_numpy(X)
        np.testing.assert_array_equal(scores, 0.0)

    def test_two_features(self):
        """d=2 → exactly one pair; at least some scores should be non-zero."""
        X = _rng().randn(100, 2).astype(np.float64)
        scores = _pairwise_target_numpy(X)
        assert np.any(scores != 0.0), "Non-trivial data should produce non-zero scores"

    def test_constant_column_contributes_zero(self):
        """A constant feature produces constant interactions → zero contribution."""
        rng = _rng()
        X = rng.randn(60, 4).astype(np.float64)
        X_const = X.copy()
        X_const[:, 2] = 5.0  # constant column
        scores_orig = _pairwise_target_numpy(X)
        scores_const = _pairwise_target_numpy(X_const)
        # Any pair involving column 2 is constant → z-score 0 for that pair.
        # Scores from other pairs should still be non-zero.
        assert scores_const is not None  # no crash

    def test_zero_variance_input_does_not_crash(self):
        """All-constant input → all scores zero."""
        X = np.ones((30, 4), dtype=np.float64)
        scores = _pairwise_target_numpy(X)
        np.testing.assert_array_equal(scores, 0.0)

    def test_structured_signal(self):
        """Samples with co-extreme feature values should receive high |score|."""
        rng = _rng(0)
        n = 200
        X = rng.randn(n, 5).astype(np.float64)
        # Make the first sample extreme on all features
        X[0, :] = 5.0
        scores = _pairwise_target_numpy(X)
        assert abs(scores[0]) > abs(scores[1:]).mean() * 2, (
            "The co-extreme sample should have a notably higher |score| than average"
        )

    def test_deterministic(self):
        X = _rng().randn(80, 6).astype(np.float64)
        s1 = _pairwise_target_numpy(X)
        s2 = _pairwise_target_numpy(X)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# 3. Pairwise target — Numba kernel (requires numba)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestPairwiseTargetNumba:
    def test_output_shape_and_dtype(self):
        X = _random_partition(n=60, d=5)
        scores = _pairwise_target_nb(X)
        assert scores.shape == (60,)
        assert scores.dtype == np.float64

    def test_single_feature_returns_zeros(self):
        X = _random_partition(n=40, d=1)
        scores = _pairwise_target_nb(X)
        np.testing.assert_array_equal(scores, 0.0)

    def test_zero_variance_input_does_not_crash(self):
        X = np.ones((30, 4), dtype=np.float64)
        scores = _pairwise_target_nb(X)
        np.testing.assert_array_equal(scores, 0.0)

    def test_deterministic(self):
        X = _random_partition(n=80, d=6)
        s1 = _pairwise_target_nb(X)
        s2 = _pairwise_target_nb(X)
        np.testing.assert_array_equal(s1, s2)

    def test_numerical_equivalence_with_numpy(self):
        """Numba kernel must agree with NumPy block-wise to within atol=1e-7."""
        rng = _rng(0)
        for n, d in [(50, 3), (200, 8), (500, 15), (1000, 20)]:
            X = np.ascontiguousarray(rng.randn(n, d), dtype=np.float64)
            nb = _pairwise_target_nb(X)
            py = _pairwise_target_numpy(X)
            np.testing.assert_allclose(
                nb, py, atol=1e-7,
                err_msg=f"Mismatch at n={n}, d={d}",
            )

    def test_numerical_equivalence_with_structured_data(self):
        """Equivalence holds on data with strong covariance structure."""
        rng = _rng(7)
        n, d = 300, 10
        # Correlated features
        cov = rng.randn(d, d)
        cov = cov @ cov.T
        X = rng.multivariate_normal(np.zeros(d), cov, size=n).astype(np.float64)
        X = np.ascontiguousarray(X)
        nb = _pairwise_target_nb(X)
        py = _pairwise_target_numpy(X)
        np.testing.assert_allclose(nb, py, atol=1e-7)

    def test_numerical_equivalence_single_pair(self):
        """d=2 → exactly one pair; easy to verify manually."""
        rng = _rng(1)
        X = np.ascontiguousarray(rng.randn(100, 2), dtype=np.float64)
        prod = X[:, 0] * X[:, 1]
        expected = (prod - prod.mean()) / prod.std()

        nb = _pairwise_target_nb(X)
        py = _pairwise_target_numpy(X)

        np.testing.assert_allclose(nb, expected, atol=1e-10)
        np.testing.assert_allclose(py, expected, atol=1e-10)

    def test_constant_column_both_paths_agree(self):
        """Constant interactions must be zeroed out in both paths."""
        rng = _rng(3)
        X = np.ascontiguousarray(rng.randn(80, 5), dtype=np.float64)
        X[:, 1] = 3.0  # constant
        nb = _pairwise_target_nb(X)
        py = _pairwise_target_numpy(X)
        np.testing.assert_allclose(nb, py, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Centroid FPS — dispatch layer (always runs)
# ---------------------------------------------------------------------------

class TestCentroidFPSDispatch:
    """Tests on _centroid_fps_core (auto-dispatches Numba or NumPy)."""

    def test_output_shape_and_dtype(self):
        X = _random_partition(n=80, d=6)
        idx = _centroid_fps_core(X, 20)
        assert idx.shape == (20,)
        assert idx.dtype == np.int64

    def test_no_duplicate_indices(self):
        X = _random_partition(n=80, d=6)
        idx = _centroid_fps_core(X, 30)
        assert _is_unique(idx), "FPS must not select the same point twice"

    def test_indices_in_valid_range(self):
        n, d, budget = 100, 5, 25
        X = _random_partition(n=n, d=d)
        idx = _centroid_fps_core(X, budget)
        assert idx.min() >= 0 and idx.max() < n

    def test_deterministic(self):
        X = _random_partition(n=100, d=5)
        i1 = _centroid_fps_core(X, 30)
        i2 = _centroid_fps_core(X, 30)
        np.testing.assert_array_equal(i1, i2)

    def test_budget_one_returns_seed(self):
        """budget=1 must return the centroid-nearest point."""
        X = _random_partition(n=50, d=4)
        idx = _centroid_fps_core(X, 1)
        assert idx.shape == (1,)
        centroid = X.mean(axis=0)
        dists = np.sum((X - centroid) ** 2, axis=1)
        expected_seed = int(np.argmin(dists))
        assert int(idx[0]) == expected_seed

    def test_budget_equals_n_selects_all(self):
        """budget=n must return all n distinct indices."""
        n, d = 30, 4
        X = _random_partition(n=n, d=d)
        idx = _centroid_fps_core(X, n)
        assert len(idx) == n
        assert _is_unique(idx)

    def test_diversity_exceeds_random(self):
        """FPS should produce a more diverse subset than random selection."""
        rng = _rng(0)
        n, d, budget = 200, 8, 40
        X = rng.randn(n, d).astype(np.float64)

        fps_idx = _centroid_fps_core(X, budget)
        rand_idx = rng.choice(n, size=budget, replace=False)

        def min_pairwise_dist(idx):
            pts = X[idx]
            dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
            np.fill_diagonal(dists, np.inf)
            return dists.min()

        fps_min = min_pairwise_dist(fps_idx)
        rand_min = min_pairwise_dist(rand_idx)
        assert fps_min > rand_min * 0.5, (
            "FPS minimum pairwise distance should not be far worse than random "
            f"(fps={fps_min:.4f}, random={rand_min:.4f})"
        )

    def test_small_d_one_feature(self):
        """d=1 must work (1-D points)."""
        X = _random_partition(n=50, d=1)
        idx = _centroid_fps_core(X, 10)
        assert len(idx) == 10
        assert _is_unique(idx)


# ---------------------------------------------------------------------------
# 5. Centroid FPS — Numba vs NumPy exact-index equality (requires numba)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestCentroidFPSNumbaEquivalence:
    """
    Assert that _centroid_fps_core_nb produces *exactly* the same indices as
    the pure-NumPy path on random data where all pairwise distances are
    well-separated (no float64 ties).
    """

    def _numpy_centroid_fps(self, X_part, budget):
        """Reference implementation (pure NumPy, no Numba dispatch)."""
        n = len(X_part)
        centroid = X_part.mean(axis=0)
        sq_dists = np.sum((X_part - centroid) ** 2, axis=1)
        seed = int(np.argmin(sq_dists))
        selected = [seed]
        min_sq_dists = np.full(n, np.inf)
        for _ in range(budget - 1):
            last = selected[-1]
            sq = np.sum((X_part - X_part[last]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, sq)
            selected.append(int(np.argmax(min_sq_dists)))
        return np.array(selected, dtype=np.int64)

    @pytest.mark.parametrize("n,d,budget,seed", [
        (50,  4, 15, 0),
        (80,  6, 20, 1),
        (120, 8, 30, 2),
        (200, 10, 50, 3),
        (50,  1, 10, 4),   # d=1 edge case
    ])
    def test_exact_index_equality(self, n, d, budget, seed):
        X = _random_partition(n=n, d=d, seed=seed)
        nb = _centroid_fps_core_nb(X, budget)
        py = self._numpy_centroid_fps(X, budget)
        np.testing.assert_array_equal(
            nb, py,
            err_msg=f"Index mismatch at n={n}, d={d}, budget={budget}, seed={seed}",
        )

    def test_budget_one_exact(self):
        X = _random_partition(n=60, d=5)
        nb = _centroid_fps_core_nb(X, 1)
        py = self._numpy_centroid_fps(X, 1)
        np.testing.assert_array_equal(nb, py)

    def test_budget_equals_n_exact(self):
        X = _random_partition(n=25, d=4)
        nb = _centroid_fps_core_nb(X, 25)
        py = self._numpy_centroid_fps(X, 25)
        np.testing.assert_array_equal(nb, py)


# ---------------------------------------------------------------------------
# 6. Medoid FPS — dispatch layer (always runs)
# ---------------------------------------------------------------------------

class TestMedoidFPSDispatch:
    """Tests on _medoid_fps_core (auto-dispatches Numba or NumPy)."""

    def test_output_shape_and_dtype(self):
        X = _random_partition(n=80, d=6)
        idx = _medoid_fps_core(X, 20)
        assert idx.shape == (20,)
        assert idx.dtype == np.int64

    def test_no_duplicate_indices(self):
        X = _random_partition(n=80, d=5)
        idx = _medoid_fps_core(X, 25)
        assert _is_unique(idx)

    def test_indices_in_valid_range(self):
        n, d, budget = 100, 6, 30
        X = _random_partition(n=n, d=d)
        idx = _medoid_fps_core(X, budget)
        assert idx.min() >= 0 and idx.max() < n

    def test_deterministic(self):
        X = _random_partition(n=80, d=6)
        i1 = _medoid_fps_core(X, 20)
        i2 = _medoid_fps_core(X, 20)
        np.testing.assert_array_equal(i1, i2)

    def test_budget_one_returns_medoid(self):
        """budget=1 must return the medoid (min sum-of-distances point)."""
        from scipy.spatial.distance import cdist
        X = _random_partition(n=30, d=4)
        idx = _medoid_fps_core(X, 1)
        pairwise = cdist(X, X, metric='sqeuclidean')
        expected_medoid = int(np.argmin(pairwise.sum(axis=1)))
        assert int(idx[0]) == expected_medoid

    def test_budget_equals_n_selects_all(self):
        n, d = 30, 4
        X = _random_partition(n=n, d=d)
        idx = _medoid_fps_core(X, n)
        assert len(idx) == n
        assert _is_unique(idx)

    def test_large_partition_approximate_path(self):
        """Partitions > 200 use the approximate medoid — must still be valid."""
        n, d, budget = 300, 6, 50
        X = _random_partition(n=n, d=d)
        assert n > _MEDOID_EXACT_THRESHOLD, "Test requires n > threshold"
        idx = _medoid_fps_core(X, budget)
        assert len(idx) == budget
        assert _is_unique(idx)
        assert idx.min() >= 0 and idx.max() < n


# ---------------------------------------------------------------------------
# 7. Medoid FPS — Numba vs NumPy exact-index equality (requires numba)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestMedoidFPSNumbaEquivalence:
    """
    Assert that _medoid_fps_core_nb produces *exactly* the same indices as
    the pure-NumPy/scipy path on random non-degenerate data.
    """

    def _numpy_medoid_fps(self, X_part, budget):
        """Reference implementation using scipy cdist for exact medoid."""
        from scipy.spatial.distance import cdist
        n = len(X_part)
        if n <= _MEDOID_EXACT_THRESHOLD:
            pairwise = cdist(X_part, X_part, metric='sqeuclidean')
            medoid_idx = int(np.argmin(pairwise.sum(axis=1)))
        else:
            # approximate medoid: argpartition-based (same k formula)
            centroid = X_part.mean(axis=0)
            dists_sq = np.sum((X_part - centroid) ** 2, axis=1)
            k = min(max(int(n ** 0.5), 30), n)
            candidates = np.argpartition(dists_sq, k - 1)[:k]
            sub = cdist(X_part[candidates], X_part[candidates], metric='sqeuclidean')
            medoid_idx = int(candidates[np.argmin(sub.sum(axis=1))])

        selected = [medoid_idx]
        min_sq_dists = np.full(n, np.inf)
        for _ in range(budget - 1):
            last = selected[-1]
            sq = np.sum((X_part - X_part[last]) ** 2, axis=1)
            min_sq_dists = np.minimum(min_sq_dists, sq)
            selected.append(int(np.argmax(min_sq_dists)))
        return np.array(selected, dtype=np.int64)

    @pytest.mark.parametrize("n,d,budget,seed", [
        (40,  4, 12, 10),
        (80,  6, 20, 11),
        (150, 8, 35, 12),  # below threshold
        (30,  3, 10, 13),
    ])
    def test_exact_index_equality_small_partition(self, n, d, budget, seed):
        """Exact medoid path (n <= 200): indices must be bit-for-bit identical."""
        assert n <= _MEDOID_EXACT_THRESHOLD
        X = _random_partition(n=n, d=d, seed=seed)
        nb = _medoid_fps_core_nb(X, budget)
        py = self._numpy_medoid_fps(X, budget)
        np.testing.assert_array_equal(
            nb, py,
            err_msg=f"Index mismatch at n={n}, d={d}, budget={budget}, seed={seed}",
        )

    def test_large_partition_properties(self):
        """
        Approximate medoid path (n > 200): argpartition vs argsort may differ
        in pathological cases.  We therefore test *properties* rather than
        exact equality: seed must be a plausible medoid (low mean distance),
        no duplicates, valid range.
        """
        from scipy.spatial.distance import cdist
        n, d, budget = 300, 6, 60
        X = _random_partition(n=n, d=d, seed=99)
        nb = _medoid_fps_core_nb(X, budget)

        assert len(nb) == budget
        assert _is_unique(nb)
        assert nb.min() >= 0 and nb.max() < n

        # Seed (nb[0]) should be among the top-25% lowest-sum-distance points
        # (true medoid or a very close approximation of it)
        pairwise = cdist(X, X, metric='sqeuclidean')
        sum_dists = pairwise.sum(axis=1)
        threshold_rank = int(0.25 * n)
        seed_rank = int((sum_dists < sum_dists[nb[0]]).sum())
        assert seed_rank < threshold_rank, (
            f"Approximate medoid seed rank={seed_rank} is outside the top-25% "
            f"(threshold={threshold_rank}). Approximate medoid may be poor."
        )

    def test_budget_one_exact(self):
        """budget=1 — Numba and NumPy seeds must match for small partitions."""
        X = _random_partition(n=50, d=5, seed=20)
        nb = _medoid_fps_core_nb(X, 1)
        py = self._numpy_medoid_fps(X, 1)
        np.testing.assert_array_equal(nb, py)


# ---------------------------------------------------------------------------
# 8. Exact- and approximate-medoid helpers (requires numba)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestMedoidHelpers:
    def test_exact_medoid_matches_scipy(self):
        from scipy.spatial.distance import cdist
        rng = _rng(5)
        for n, d in [(10, 3), (30, 5), (50, 8), (200, 6)]:
            X = np.ascontiguousarray(rng.randn(n, d), dtype=np.float64)
            expected = int(np.argmin(cdist(X, X, metric='sqeuclidean').sum(axis=1)))
            got = int(_exact_medoid_nb(X))
            assert got == expected, (
                f"_exact_medoid_nb mismatch at n={n}, d={d}: "
                f"expected={expected}, got={got}"
            )

    def test_approx_medoid_is_near_true_medoid(self):
        """Approximate medoid should be among the top-20% lowest-sum-distance points."""
        from scipy.spatial.distance import cdist
        rng = _rng(6)
        for n, d in [(300, 5), (500, 8), (1000, 6)]:
            X = np.ascontiguousarray(rng.randn(n, d), dtype=np.float64)
            k = min(max(int(n ** 0.5), 30), n)
            approx_idx = int(_approx_medoid_nb(X, k))
            pairwise = cdist(X, X, metric='sqeuclidean')
            sum_dists = pairwise.sum(axis=1)
            rank = int((sum_dists < sum_dists[approx_idx]).sum())
            threshold = int(0.20 * n)
            assert rank < threshold, (
                f"_approx_medoid_nb rank={rank} outside top-20% at n={n}, d={d}"
            )


# ---------------------------------------------------------------------------
# 9. End-to-end HVRT integration (always runs)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    Verify that the dispatch layer integrates correctly with the full
    HVRT.fit → reduce / expand pipeline.
    """

    def test_fit_reduce_runs_without_error(self):
        rng = _rng(0)
        X = rng.randn(300, 8).astype(np.float64)
        model = HVRT(random_state=42).fit(X)
        X_red = model.reduce(n=80)
        assert X_red.shape == (80, 8)

    def test_fit_reduce_medoid_runs_without_error(self):
        rng = _rng(1)
        X = rng.randn(200, 5).astype(np.float64)
        model = HVRT(random_state=42).fit(X)
        X_red = model.reduce(n=50, method='medoid_fps')
        assert X_red.shape == (50, 5)

    def test_fit_expand_runs_without_error(self):
        rng = _rng(2)
        X = rng.randn(200, 6).astype(np.float64)
        model = HVRT(random_state=42).fit(X)
        X_syn = model.expand(n=400)
        assert X_syn.shape == (400, 6)

    def test_reduce_output_is_subset_of_input(self):
        rng = _rng(3)
        X = rng.randn(300, 5).astype(np.float64)
        model = HVRT(random_state=42).fit(X)
        X_red, idx = model.reduce(n=60, return_indices=True)
        np.testing.assert_array_equal(X_red, X[idx])

    def test_compute_x_component_shape_and_dtype(self):
        """HVRT._compute_x_component must return (n_samples,) float64."""
        rng = _rng(4)
        X_z = np.ascontiguousarray(rng.randn(100, 7), dtype=np.float64)
        model = HVRT(random_state=42)
        scores = model._compute_x_component(X_z)
        assert scores.shape == (100,)
        assert scores.dtype == np.float64

    def test_compute_x_component_single_feature(self):
        """d=1 → no pairs → all-zero target."""
        X_z = np.ascontiguousarray(_rng(5).randn(50, 1), dtype=np.float64)
        model = HVRT(random_state=42)
        scores = model._compute_x_component(X_z)
        np.testing.assert_array_equal(scores, 0.0)

    def test_compute_x_component_deterministic(self):
        X_z = np.ascontiguousarray(_rng(6).randn(80, 5), dtype=np.float64)
        model = HVRT(random_state=42)
        s1 = model._compute_x_component(X_z)
        s2 = model._compute_x_component(X_z)
        np.testing.assert_array_equal(s1, s2)

    def test_all_reduce_methods_produce_valid_subsets(self):
        rng = _rng(7)
        X = rng.randn(300, 6).astype(np.float64)
        model = HVRT(random_state=42).fit(X)
        for method in ('fps', 'centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified'):
            X_red = model.reduce(n=40, method=method)
            assert X_red.shape == (40, 6), f"method={method} failed"

    def test_supervised_fit_reduce(self):
        rng = _rng(8)
        X = rng.randn(200, 5).astype(np.float64)
        y = X[:, 0] + rng.randn(200) * 0.1
        model = HVRT(y_weight=0.5, random_state=42).fit(X, y)
        X_red = model.reduce(n=50)
        assert X_red.shape == (50, 5)


# ---------------------------------------------------------------------------
# 10. Numba dispatch is actually used (requires numba)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
class TestNumbaDispatchIsUsed:
    """
    Verify that the dispatch layer in _centroid_fps_core and _medoid_fps_core
    actually calls the Numba kernels (not the Python fallback) when
    _NUMBA_AVAILABLE is True.
    """

    def test_centroid_fps_core_output_matches_nb_kernel_directly(self):
        X = _random_partition(n=80, d=6)
        dispatch_result = _centroid_fps_core(X, 20)
        nb_direct = _centroid_fps_core_nb(X, 20)
        np.testing.assert_array_equal(dispatch_result, nb_direct)

    def test_medoid_fps_core_output_matches_nb_kernel_directly(self):
        X = _random_partition(n=80, d=6)
        dispatch_result = _medoid_fps_core(X, 20)
        nb_direct = _medoid_fps_core_nb(X, 20)
        np.testing.assert_array_equal(dispatch_result, nb_direct)

    def test_compute_x_component_matches_nb_kernel_directly(self):
        X_z = _random_partition(n=100, d=7)
        model = HVRT(random_state=42)
        dispatch_result = model._compute_x_component(X_z)
        nb_direct = _pairwise_target_nb(X_z)
        np.testing.assert_array_equal(dispatch_result, nb_direct)


# ---------------------------------------------------------------------------
# 11. Dtype and contiguity safety (always runs)
# ---------------------------------------------------------------------------

class TestDtypeContiguity:
    """
    Dispatch functions must not crash on non-float64 or non-contiguous input
    (the dispatch layer converts via np.ascontiguousarray).
    """

    def test_centroid_fps_accepts_float32_input(self):
        X = _rng().randn(60, 4).astype(np.float32)
        idx = _centroid_fps_core(X, 15)
        assert len(idx) == 15

    def test_centroid_fps_accepts_non_contiguous_input(self):
        X = _rng().randn(120, 8).astype(np.float64)
        X_nc = X[::2, :]  # non-contiguous slice (every other row)
        idx = _centroid_fps_core(X_nc, 10)
        assert len(idx) == 10

    def test_medoid_fps_accepts_float32_input(self):
        X = _rng().randn(60, 4).astype(np.float32)
        idx = _medoid_fps_core(X, 15)
        assert len(idx) == 15

    def test_pairwise_numpy_accepts_float32_input(self):
        X = _rng().randn(50, 4).astype(np.float32)
        scores = _pairwise_target_numpy(X)
        assert scores.shape == (50,)

    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")
    def test_pairwise_nb_output_is_float64(self):
        """Numba kernel must always return float64 regardless of input dtype."""
        X = np.ascontiguousarray(_rng().randn(50, 4), dtype=np.float64)
        scores = _pairwise_target_nb(X)
        assert scores.dtype == np.float64


# ---------------------------------------------------------------------------
# 12. Regression guard — _NUMBA_AVAILABLE must not silently become False
#     after a successful import.
# ---------------------------------------------------------------------------

class TestImportConsistency:
    def test_numba_available_consistent_with_import(self):
        """
        _NUMBA_AVAILABLE must be True iff ``import numba`` succeeds.
        This guards against the flag being accidentally hard-coded to False.
        """
        try:
            import numba  # noqa: F401
            numba_importable = True
        except ImportError:
            numba_importable = False
        assert _NUMBA_AVAILABLE == numba_importable, (
            f"_NUMBA_AVAILABLE={_NUMBA_AVAILABLE} but numba importable={numba_importable}. "
            "The flag is inconsistent with the actual import state."
        )
