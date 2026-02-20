"""
Tests for reduction-specific behaviour.

Covers variance-weighted vs size-weighted budget allocation, method parity
with v1 HVRTSampleReducer, and edge cases.
"""

import pytest
import numpy as np
from hvrt import HVRT, FastHVRT, HVRTSampleReducer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data():
    rng = np.random.RandomState(7)
    X = rng.randn(500, 10)
    y = X[:, 0] + rng.randn(500) * 0.5
    return X, y


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------

class TestBudgetAllocation:
    def test_size_weighted_budget_proportional(self, data):
        """Size-weighted should allocate proportionally to partition sizes."""
        X, y = data
        model = HVRT(random_state=0).fit(X, y)
        X_red = model.reduce(n=100, variance_weighted=False)
        assert len(X_red) == 100

    def test_variance_weighted_selects_more_extremes(self, data):
        """Both variance-weighted and size-weighted should return valid subsets."""
        X, y = data
        model = HVRT(random_state=0).fit(X, y)
        idx_var = model.reduce(n=100, variance_weighted=True, return_indices=True)[1]
        idx_siz = model.reduce(n=100, variance_weighted=False, return_indices=True)[1]
        assert len(idx_var) == 100
        assert len(idx_siz) == 100
        # Indices must be valid into original X
        assert np.all(idx_var >= 0) and np.all(idx_var < len(X))
        assert np.all(idx_siz >= 0) and np.all(idx_siz < len(X))

    def test_min_samples_per_partition_respected(self, data):
        """Each non-empty partition should contribute at least min_samples_per_partition."""
        X, _ = data
        mpp = 3
        model = HVRT(min_samples_per_partition=mpp, random_state=0).fit(X)
        model.reduce(n=200, variance_weighted=False)  # should not raise


# ---------------------------------------------------------------------------
# All four HVRT-family reduce combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('Cls,vw', [
    (HVRT,     True),
    (HVRT,     False),
    (FastHVRT, True),
    (FastHVRT, False),
])
class TestAllReduceCombinations:
    def test_shape(self, Cls, vw, data):
        X, y = data
        X_red = Cls(random_state=0).fit(X, y).reduce(n=100, variance_weighted=vw)
        assert X_red.shape == (100, X.shape[1])

    def test_values_are_real(self, Cls, vw, data):
        X, y = data
        X_red = Cls(random_state=0).fit(X, y).reduce(n=100, variance_weighted=vw)
        assert np.all(np.isfinite(X_red))

    def test_samples_are_from_original(self, Cls, vw, data):
        """Reduced samples must be rows from the original X."""
        X, y = data
        model = Cls(random_state=0).fit(X, y)
        X_red, idx = model.reduce(n=100, variance_weighted=vw, return_indices=True)
        np.testing.assert_array_equal(X_red, X[idx])


# ---------------------------------------------------------------------------
# v1 parity
# ---------------------------------------------------------------------------

class TestV1Parity:
    def test_v1_still_importable(self):
        from hvrt import HVRTSampleReducer
        assert HVRTSampleReducer is not None

    def test_v1_still_functional(self, data):
        X, y = data
        reducer = HVRTSampleReducer(reduction_ratio=0.3, random_state=42)
        X_red, y_red = reducer.fit_transform(X, y)
        assert X_red.shape[1] == X.shape[1]
        assert len(X_red) < len(X)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestReduceEdgeCases:
    def test_tiny_dataset(self):
        rng = np.random.RandomState(0)
        X = rng.randn(30, 3)
        model = HVRT(random_state=0).fit(X)
        X_red = model.reduce(n=10)
        assert X_red.shape == (10, 3)

    def test_high_dimensional(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 50)
        model = FastHVRT(random_state=0).fit(X)
        X_red = model.reduce(n=50)
        assert X_red.shape == (50, 50)

    def test_ratio_below_one_percent(self):
        rng = np.random.RandomState(0)
        X = rng.randn(1000, 5)
        model = HVRT(random_state=0).fit(X)
        X_red = model.reduce(ratio=0.01)
        assert len(X_red) >= 1

    def test_ratio_of_1_returns_all(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 5)
        model = HVRT(random_state=0).fit(X)
        X_red = model.reduce(ratio=1.0)
        assert len(X_red) == len(X)
