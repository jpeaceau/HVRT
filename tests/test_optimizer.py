"""
Tests for HVRTOptimizer.

The entire module is skipped if optuna is not installed.
Use n_trials=3, cv=2, n=200 samples throughout for speed.
"""
import numpy as np
import pytest

optuna = pytest.importorskip('optuna')

from hvrt import HVRTOptimizer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reg_data():
    """200-sample regression dataset (continuous y, many unique values)."""
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = X[:, 0] * 2.0 + X[:, 1] + rng.randn(200) * 0.5
    return X, y


@pytest.fixture
def clf_data():
    """200-sample binary classification dataset (y ∈ {0, 1})."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


FAST = dict(n_trials=3, cv=2, random_state=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fit_returns_self(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST)
    result = opt.fit(X, y)
    assert result is opt


def test_best_score_finite(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    assert np.isfinite(opt.best_score_)


def test_best_params_keys(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    for key in ('n_partitions', 'bandwidth', 'y_weight', 'min_samples_leaf'):
        assert key in opt.best_params_, f"Missing key in best_params_: {key!r}"


def test_best_expand_params_keys(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    assert 'variance_weighted' in opt.best_expand_params_, (
        "best_expand_params_ must contain 'variance_weighted'"
    )


def test_expand_shape(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    X_synth = opt.expand(n=100)
    assert X_synth.shape == (100, X.shape[1]), (
        f"Expected (100, {X.shape[1]}), got {X_synth.shape}"
    )


def test_augment_shape(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    n_total = 300
    X_aug = opt.augment(n=n_total)
    assert X_aug.shape == (n_total, X.shape[1]), (
        f"Expected ({n_total}, {X.shape[1]}), got {X_aug.shape}"
    )
    # First rows must match original X exactly
    np.testing.assert_array_equal(
        X_aug[:len(X)], X,
        err_msg="augment() first rows do not match original X"
    )


def test_study_stored(reg_data):
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    assert isinstance(opt.study_, optuna.Study)


def test_n_jobs_2(reg_data):
    """n_jobs=2 should complete without error."""
    X, y = reg_data
    opt = HVRTOptimizer(n_trials=6, cv=2, n_jobs=2, random_state=42).fit(X, y)
    assert hasattr(opt, 'best_model_')


def test_task_regression_auto(reg_data):
    """Continuous y (many unique values) → 'regression' detected automatically."""
    X, y = reg_data
    assert len(np.unique(y)) > 20, "Fixture must have > 20 unique y values"
    opt = HVRTOptimizer(**FAST).fit(X, y)
    assert hasattr(opt, 'best_score_')


def test_task_classification_auto(clf_data):
    """Binary y (≤ 20 unique values) → 'classification' detected automatically."""
    X, y = clf_data
    assert len(np.unique(y)) <= 20, "Fixture must have ≤ 20 unique y values"
    opt = HVRTOptimizer(**FAST).fit(X, y)
    assert hasattr(opt, 'best_score_')


def test_user_override_in_expand(reg_data):
    """kwargs passed to expand() should override best_expand_params_."""
    X, y = reg_data
    opt = HVRTOptimizer(**FAST).fit(X, y)
    # Explicitly override variance_weighted; should not raise
    X_synth = opt.expand(n=50, variance_weighted=True)
    assert X_synth.shape == (50, X.shape[1])
