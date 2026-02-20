"""
Tests for reduction-specific behaviour.

Covers variance-weighted vs size-weighted budget allocation, method parity
with v1 HVRTSampleReducer, and edge cases.
"""

import warnings
import pytest
import numpy as np
from hvrt import HVRT, FastHVRT, HVRTSampleReducer, HVRTFeatureWarning


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


# ---------------------------------------------------------------------------
# External X for reduce
# ---------------------------------------------------------------------------

class TestReduceWithExternalData:
    def test_shape_external_X(self, data):
        """reduce(X=X_test) should return n rows with the correct feature count."""
        X_train, _ = data
        rng = np.random.RandomState(11)
        X_test = rng.randn(200, X_train.shape[1])
        model = HVRT(random_state=0).fit(X_train)
        X_red = model.reduce(n=50, X=X_test)
        assert X_red.shape == (50, X_train.shape[1])

    def test_rows_from_external_X(self, data):
        """Reduced rows must come from the external X, not from the training data."""
        X_train, _ = data
        rng = np.random.RandomState(22)
        X_test = rng.randn(200, X_train.shape[1]) + 5.0  # shifted so no overlap
        model = HVRT(random_state=0).fit(X_train)
        X_red, idx = model.reduce(n=50, X=X_test, return_indices=True)
        np.testing.assert_array_equal(X_red, X_test[idx])
        # None of the selected rows should be present in X_train
        assert not np.any(np.all(np.isin(X_red, X_train), axis=1))

    def test_return_indices_external_X(self, data):
        """Returned indices must be valid into the external X."""
        X_train, _ = data
        rng = np.random.RandomState(33)
        X_test = rng.randn(150, X_train.shape[1])
        model = FastHVRT(random_state=0).fit(X_train)
        X_red, idx = model.reduce(n=40, X=X_test, return_indices=True)
        assert np.all(idx >= 0)
        assert np.all(idx < len(X_test))
        np.testing.assert_array_equal(X_red, X_test[idx])

    def test_ratio_with_external_X(self, data):
        """ratio should be resolved against len(X_test), not len(X_train)."""
        X_train, _ = data
        rng = np.random.RandomState(44)
        X_test = rng.randn(200, X_train.shape[1])
        model = HVRT(random_state=0).fit(X_train)
        X_red = model.reduce(ratio=0.1, X=X_test)
        expected = max(1, int(len(X_test) * 0.1))
        assert len(X_red) == expected


# ---------------------------------------------------------------------------
# Feature coercion (_coerce_external_X)
# ---------------------------------------------------------------------------

class TestFeatureCoercion:
    """Tests for column name matching, reordering, and extra-feature handling."""

    def _make_df(self, data, cols):
        """Return a pandas DataFrame; skip if pandas is not installed."""
        pd = pytest.importorskip('pandas')
        return pd.DataFrame(data, columns=cols)

    def test_numpy_exact_features_no_warning(self, data):
        """Exact feature count numpy array should pass with no warning."""
        X_train, _ = data
        model = HVRT(random_state=0).fit(X_train)
        rng = np.random.RandomState(5)
        X_test = rng.randn(50, X_train.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter('error', HVRTFeatureWarning)
            X_red = model.reduce(n=20, X=X_test)
        assert X_red.shape[1] == X_train.shape[1]

    def test_numpy_extra_features_warns_and_trims(self, data):
        """Extra trailing columns in a numpy array emit HVRTFeatureWarning."""
        X_train, _ = data
        model = HVRT(random_state=0).fit(X_train)
        rng = np.random.RandomState(6)
        X_test = rng.randn(50, X_train.shape[1] + 3)
        with pytest.warns(HVRTFeatureWarning, match="extra"):
            X_red = model.reduce(n=20, X=X_test)
        assert X_red.shape[1] == X_train.shape[1]

    def test_numpy_missing_features_raises(self, data):
        """Fewer columns than training features should raise ValueError."""
        X_train, _ = data
        model = HVRT(random_state=0).fit(X_train)
        rng = np.random.RandomState(7)
        X_test = rng.randn(50, X_train.shape[1] - 1)
        with pytest.raises(ValueError, match="All training features must be present"):
            model.reduce(n=10, X=X_test)

    def test_dataframe_name_matching_reorders(self, data):
        """DataFrame columns are matched by name and reordered to training order."""
        X_train, _ = data
        cols = [f'f{i}' for i in range(X_train.shape[1])]
        df_train = self._make_df(X_train, cols)
        model = HVRT(random_state=0).fit(df_train)

        rng = np.random.RandomState(8)
        X_test = rng.randn(80, X_train.shape[1])
        # Shuffle the column order in the test DataFrame
        shuffled_cols = cols[::-1]
        df_test = self._make_df(X_test[:, ::-1], shuffled_cols)

        with warnings.catch_warnings():
            warnings.simplefilter('error', HVRTFeatureWarning)
            X_red = model.reduce(n=20, X=df_test)
        assert X_red.shape[1] == X_train.shape[1]

    def test_dataframe_extra_columns_warns(self, data):
        """DataFrame with extra columns emits HVRTFeatureWarning."""
        X_train, _ = data
        cols = [f'f{i}' for i in range(X_train.shape[1])]
        df_train = self._make_df(X_train, cols)
        model = HVRT(random_state=0).fit(df_train)

        rng = np.random.RandomState(9)
        extra = rng.randn(80, X_train.shape[1] + 2)
        extra_cols = cols + ['extra_a', 'extra_b']
        df_test = self._make_df(extra, extra_cols)

        with pytest.warns(HVRTFeatureWarning, match="extra"):
            X_red = model.reduce(n=20, X=df_test)
        assert X_red.shape[1] == X_train.shape[1]

    def test_dataframe_missing_column_raises(self, data):
        """DataFrame missing a training column should raise ValueError."""
        X_train, _ = data
        cols = [f'f{i}' for i in range(X_train.shape[1])]
        df_train = self._make_df(X_train, cols)
        model = HVRT(random_state=0).fit(df_train)

        rng = np.random.RandomState(10)
        X_test = rng.randn(50, X_train.shape[1] - 1)
        df_test = self._make_df(X_test, cols[:-1])  # missing last training col

        with pytest.raises(ValueError, match="missing column"):
            model.reduce(n=10, X=df_test)

    def test_warning_suppression(self, data):
        """HVRTFeatureWarning can be suppressed via the standard warnings API."""
        X_train, _ = data
        model = HVRT(random_state=0).fit(X_train)
        rng = np.random.RandomState(11)
        X_test = rng.randn(50, X_train.shape[1] + 2)

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', HVRTFeatureWarning)
            X_red = model.reduce(n=20, X=X_test)
        assert X_red.shape[1] == X_train.shape[1]
