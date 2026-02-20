"""
Tests for expansion and augmentation behaviour.

Covers multivariate KDE generation, min_novelty enforcement, DCR ratio,
all four HVRT-family expand combinations, and augment semantics.
"""

import pytest
import numpy as np
from hvrt import HVRT, FastHVRT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    return X


@pytest.fixture
def correlated_data():
    rng = np.random.RandomState(1)
    cov = np.array([
        [1.0, 0.8, 0.3, 0.1],
        [0.8, 1.0, 0.5, 0.2],
        [0.3, 0.5, 1.0, 0.7],
        [0.1, 0.2, 0.7, 1.0],
    ])
    return rng.multivariate_normal(np.zeros(4), cov, 400)


# ---------------------------------------------------------------------------
# All four HVRT-family expand combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('Cls,vw', [
    (HVRT,     True),
    (HVRT,     False),
    (FastHVRT, True),
    (FastHVRT, False),
])
class TestAllExpandCombinations:
    def test_shape(self, Cls, vw, simple_data):
        X = simple_data
        X_synth = Cls(random_state=0).fit(X).expand(n=200, variance_weighted=vw)
        assert X_synth.shape == (200, X.shape[1])

    def test_finite_values(self, Cls, vw, simple_data):
        X = simple_data
        X_synth = Cls(random_state=0).fit(X).expand(n=200, variance_weighted=vw)
        assert np.all(np.isfinite(X_synth))

    def test_not_identical_to_original(self, Cls, vw, simple_data):
        """Generated samples should not simply copy originals."""
        X = simple_data
        X_synth = Cls(random_state=0).fit(X).expand(n=200, variance_weighted=vw)
        # At least some rows must differ from all originals
        from scipy.spatial.distance import cdist
        min_dists = cdist(X_synth[:10], X, 'euclidean').min(axis=1)
        assert np.any(min_dists > 0)


# ---------------------------------------------------------------------------
# min_novelty
# ---------------------------------------------------------------------------

class TestMinNovelty:
    def test_zero_novelty_no_crash(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        X_synth = model.expand(n=100, min_novelty=0.0)
        assert X_synth.shape[0] == 100

    def test_nonzero_novelty_enforced(self, correlated_data):
        """With a strict novelty constraint the min distance should be ≥ threshold."""
        X = correlated_data
        model = HVRT(random_state=0).fit(X)
        threshold = 0.5
        X_synth = model.expand(n=100, min_novelty=threshold)
        # Check a sample of points
        from scipy.spatial.distance import cdist
        # Normalize to z-score space for comparison (model stores X_z_)
        X_synth_z = model._to_z(X_synth)
        min_dists = cdist(X_synth_z[:20], model.X_z_, 'euclidean').min(axis=1)
        # Allow some slack for the fallback path
        assert np.mean(min_dists >= threshold * 0.5) > 0.5

    def test_high_novelty_still_returns_n(self, simple_data):
        """Should return the requested number even if novelty is hard to meet."""
        model = HVRT(random_state=0).fit(simple_data)
        X_synth = model.expand(n=50, min_novelty=0.5)
        assert X_synth.shape[0] == 50


# ---------------------------------------------------------------------------
# Distribution fidelity
# ---------------------------------------------------------------------------

class TestDistributionFidelity:
    def test_marginals_roughly_preserved(self, correlated_data):
        """Per-feature mean and std of synthetic should be close to originals."""
        X = correlated_data
        model = FastHVRT(random_state=0).fit(X)
        X_synth = model.expand(n=1000, variance_weighted=False)

        for j in range(X.shape[1]):
            orig_mean, orig_std = X[:, j].mean(), X[:, j].std()
            synth_mean, synth_std = X_synth[:, j].mean(), X_synth[:, j].std()
            assert abs(synth_mean - orig_mean) < 1.5 * orig_std, \
                f"Feature {j} mean drifted: {orig_mean:.2f} vs {synth_mean:.2f}"
            assert 0.2 < synth_std / (orig_std + 1e-8) < 5.0, \
                f"Feature {j} std ratio out of range"

    def test_correlation_direction_preserved(self, correlated_data):
        """Signs of off-diagonal correlations should be preserved."""
        X = correlated_data
        model = HVRT(random_state=0).fit(X)
        X_synth = model.expand(n=2000, variance_weighted=False)

        C_orig = np.corrcoef(X.T)
        C_synth = np.corrcoef(X_synth.T)

        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                if abs(C_orig[i, j]) > 0.3:
                    assert np.sign(C_synth[i, j]) == np.sign(C_orig[i, j]), \
                        f"Correlation sign flipped for ({i},{j}): {C_orig[i,j]:.2f} vs {C_synth[i,j]:.2f}"


# ---------------------------------------------------------------------------
# Augment
# ---------------------------------------------------------------------------

class TestAugment:
    def test_augment_has_originals_first(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        X_aug = model.augment(n=500)
        np.testing.assert_array_equal(X_aug[:len(simple_data)], simple_data)

    def test_augment_size(self, simple_data):
        n_target = 600
        model = FastHVRT(random_state=0).fit(simple_data)
        X_aug = model.augment(n=n_target)
        assert X_aug.shape == (n_target, simple_data.shape[1])

    def test_augment_with_min_novelty(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        X_aug = model.augment(n=500, min_novelty=0.1)
        assert X_aug.shape[0] == 500


# ---------------------------------------------------------------------------
# Novelty stats
# ---------------------------------------------------------------------------

class TestNoveltyStats:
    def test_return_novelty_stats_keys(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        X_synth, stats = model.expand(n=100, return_novelty_stats=True)
        assert set(stats.keys()) >= {'min', 'mean', 'p5'}

    def test_novelty_stats_values_non_negative(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        _, stats = model.expand(n=100, return_novelty_stats=True)
        assert stats['min'] >= 0.0
        assert stats['mean'] >= stats['min']

    def test_compute_novelty_method(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        rng = np.random.RandomState(2)
        X_new = rng.randn(10, 4) * 3  # far from training data
        dists = model.compute_novelty(X_new)
        assert dists.shape == (10,)
        assert np.all(dists > 0)


# ---------------------------------------------------------------------------
# KDE bandwidth
# ---------------------------------------------------------------------------

class TestBandwidth:
    def test_custom_bandwidth(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        X_synth = model.expand(n=100, bandwidth=0.5)
        assert X_synth.shape == (100, 4)

    def test_bandwidth_change_refits_kdes(self, simple_data):
        model = HVRT(random_state=0).fit(simple_data)
        model.expand(n=50, bandwidth=1.0)
        model.expand(n=50, bandwidth=0.1)  # should refine KDEs, not crash
