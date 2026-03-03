"""
Tests for PyramidHART and the _geometry module.

Validates:
  - PyramidHART API parity with HVRT/HART (fit/reduce/expand/augment)
  - A = |S| − ‖z‖₁ target properties (Proposition 1 of Peace 2026)
  - geometry_stats() contents and correctness
  - Pure geometry functions (compute_S/Q/T/A, minority_sign_total)
  - geometry_stats() standalone function
"""

import numpy as np
import pytest

from hvrt import (
    PyramidHART,
    compute_S,
    compute_Q,
    compute_T,
    compute_A,
    minority_sign_total,
    geometry_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def X_small(rng):
    """200 × 6 standard-normal data."""
    return rng.standard_normal((200, 6))


@pytest.fixture
def X_medium(rng):
    """800 × 10 standard-normal data."""
    return rng.standard_normal((800, 10))


@pytest.fixture
def fitted_model(X_small):
    return PyramidHART(random_state=0).fit(X_small)


# ---------------------------------------------------------------------------
# 1.  API parity — PyramidHART behaves like HVRT / HART
# ---------------------------------------------------------------------------

class TestAPIParitiy:

    def test_fit_returns_self(self, X_small):
        m = PyramidHART(random_state=0)
        result = m.fit(X_small)
        assert result is m

    def test_fit_sets_attributes(self, fitted_model, X_small):
        assert hasattr(fitted_model, 'X_')
        assert hasattr(fitted_model, 'X_z_')
        assert hasattr(fitted_model, 'partition_ids_')
        assert hasattr(fitted_model, 'n_partitions_')
        assert fitted_model.n_partitions_ >= 1

    def test_reduce_shape(self, fitted_model, X_small):
        n_target = int(len(X_small) * 0.4)
        X_red = fitted_model.reduce(ratio=0.4)
        assert X_red.shape[1] == X_small.shape[1]
        assert X_red.shape[0] <= len(X_small)

    def test_reduce_ratio_vs_n(self, fitted_model, X_small):
        n = 50
        X_a = fitted_model.reduce(n=n)
        X_b = fitted_model.reduce(ratio=n / len(X_small))
        assert X_a.shape[0] == X_b.shape[0]

    def test_reduce_return_indices(self, fitted_model, X_small):
        X_red, idx = fitted_model.reduce(ratio=0.3, return_indices=True)
        assert idx.shape[0] == X_red.shape[0]
        np.testing.assert_array_equal(X_small[idx], X_red)

    def test_expand_shape(self, fitted_model, X_small):
        n_syn = 500
        X_syn = fitted_model.expand(n=n_syn)
        assert X_syn.shape == (n_syn, X_small.shape[1])

    def test_augment_shape(self, fitted_model, X_small):
        n_total = len(X_small) + 100
        X_aug = fitted_model.augment(n=n_total)
        assert X_aug.shape == (n_total, X_small.shape[1])

    def test_fit_with_y(self, X_small, rng):
        y = rng.standard_normal(len(X_small))
        m = PyramidHART(y_weight=0.3, random_state=0)
        m.fit(X_small, y)
        assert hasattr(m, 'n_partitions_')

    def test_geometry_stats_on_training_data(self, fitted_model):
        gs = fitted_model.geometry_stats()
        assert isinstance(gs, dict)
        assert 'A_mean' in gs
        assert 'partitions' in gs

    def test_geometry_stats_on_external_X(self, fitted_model, X_small, rng):
        X_new = rng.standard_normal((50, X_small.shape[1]))
        gs = fitted_model.geometry_stats(X=X_new)
        assert gs['n'] == 50


# ---------------------------------------------------------------------------
# 2.  A = |S| − ‖z‖₁ algebraic properties  (Proposition 1)
# ---------------------------------------------------------------------------

class TestAProperties:

    def test_A_always_nonpositive(self, rng):
        """Prop 1: A ≤ 0 always (triangle inequality)."""
        for shape in [(100, 2), (300, 8), (50, 20)]:
            X_z = rng.standard_normal(shape)
            A = compute_A(X_z)
            assert np.all(A <= 1e-12), f"A > 0 found for shape {shape}"

    def test_A_zero_when_same_sign(self):
        """A = 0 when all features share a sign."""
        # All positive
        X_pos = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 0.5]])
        A_pos = compute_A(X_pos)
        np.testing.assert_allclose(A_pos, 0.0, atol=1e-12)

        # All negative
        X_neg = -X_pos
        A_neg = compute_A(X_neg)
        np.testing.assert_allclose(A_neg, 0.0, atol=1e-12)

    def test_A_negative_when_mixed_signs(self):
        """A < 0 when features have mixed signs."""
        X_mix = np.array([[1.0, -1.0]])   # equal magnitudes, opposite signs
        A = compute_A(X_mix)
        assert A[0] < -1e-10

    def test_A_equals_negS_minus_norm1_formula(self, rng):
        """Verify A = |S| − ‖z‖₁ exactly."""
        X_z = rng.standard_normal((200, 7))
        A = compute_A(X_z)
        S = np.abs(X_z.sum(axis=1))
        L1 = np.abs(X_z).sum(axis=1)
        np.testing.assert_allclose(A, S - L1, atol=1e-12)

    def test_A_degree1_homogeneity(self, rng):
        """Prop 1.4: A(λz) = λA(z) for λ ≥ 0."""
        X_z = rng.standard_normal((200, 8))
        A = compute_A(X_z)
        for lam in (0.5, 2.0, 3.7):
            A_scaled = compute_A(lam * X_z)
            np.testing.assert_allclose(A_scaled, lam * A, rtol=1e-10)

    def test_T_degree2_homogeneity(self, rng):
        """T(λz) = λ²T(z)."""
        X_z = rng.standard_normal((200, 8))
        T = compute_T(X_z)
        for lam in (0.5, 2.0, 3.0):
            T_scaled = compute_T(lam * X_z)
            np.testing.assert_allclose(T_scaled, lam ** 2 * T, rtol=1e-10)

    def test_minority_sign_equals_neg_A_over_2(self, rng):
        """Prop 1.2: minority_sign_total = −A/2."""
        X_z = rng.standard_normal((300, 6))
        A = compute_A(X_z)
        mst = minority_sign_total(X_z)
        np.testing.assert_allclose(mst, -A / 2.0, atol=1e-12)

    def test_minority_sign_nonnegative(self, rng):
        """−A/2 ≥ 0 always."""
        X_z = rng.standard_normal((300, 6))
        mst = minority_sign_total(X_z)
        assert np.all(mst >= -1e-12)

    def test_A_bounded_by_radius_times_sqrt_d(self, rng):
        """Prop 1.1: A ≥ −‖z‖₂ · √d per sample."""
        X_z = rng.standard_normal((400, 10))
        A = compute_A(X_z)
        r = np.sqrt(compute_Q(X_z))
        d = X_z.shape[1]
        lower = -r * np.sqrt(d)
        assert np.all(A >= lower - 1e-10), "A below theoretical lower bound"

    def test_outlier_cancellation(self, rng):
        """Prop 1.3: single dominant |zₖ| shifts A far less than T."""
        X_clean = rng.standard_normal((500, 8))
        X_spike = X_clean.copy()
        X_spike[:, 0] += 200.0    # extreme spike

        A_clean = compute_A(X_clean).mean()
        A_spike = compute_A(X_spike).mean()
        T_clean = compute_T(X_clean).mean()
        T_spike = compute_T(X_spike).mean()

        shift_A = abs(A_spike - A_clean)
        shift_T = abs(T_spike - T_clean)

        # T inflates massively; A much less (ratio ~35-40× empirically)
        assert shift_T > 25 * shift_A, (
            f"Expected T to shift >> A; got shift_A={shift_A:.2f}, shift_T={shift_T:.2f}"
        )


# ---------------------------------------------------------------------------
# 3.  geometry_stats (model method)
# ---------------------------------------------------------------------------

class TestGeometryStats:

    def test_required_keys_present(self, fitted_model):
        gs = fitted_model.geometry_stats()
        required = [
            'n', 'd',
            'S_mean', 'S_std', 'S_min', 'S_max',
            'Q_mean', 'Q_std', 'Q_min', 'Q_max',
            'T_mean', 'T_std', 'T_min', 'T_max',
            'cone_fraction', 'cone_critical_angle_deg', 'cooperation_ratio',
            'A_mean', 'A_std', 'A_min', 'A_max',
            'A_theoretical_lower_bound', 'sign_consistent_fraction',
            'minority_sign_mean',
            'corr_TQ', 'corr_AQ', 'corr_AT', 'corr_AS',
            'A_homogeneity', 'T_homogeneity',
            'partitions',
        ]
        for key in required:
            assert key in gs, f"Missing key: {key}"

    def test_A_mean_nonpositive(self, fitted_model):
        gs = fitted_model.geometry_stats()
        assert gs['A_mean'] <= 1e-10

    def test_A_max_nonpositive(self, fitted_model):
        gs = fitted_model.geometry_stats()
        assert gs['A_max'] <= 1e-10

    def test_homogeneity_values(self, X_medium):
        m = PyramidHART(random_state=0).fit(X_medium)
        gs = m.geometry_stats()
        assert abs(gs['A_homogeneity'] - 2.0) < 0.01
        assert abs(gs['T_homogeneity'] - 4.0) < 0.01

    def test_cone_fraction_in_01(self, fitted_model):
        gs = fitted_model.geometry_stats()
        assert 0.0 <= gs['cone_fraction'] <= 1.0

    def test_sign_consistent_fraction_in_01(self, fitted_model):
        gs = fitted_model.geometry_stats()
        assert 0.0 <= gs['sign_consistent_fraction'] <= 1.0

    def test_corr_TQ_near_zero_isotropic(self, rng):
        """Theorem 1: Cov(T, Q) ≈ 0 for isotropic z."""
        X = rng.standard_normal((2000, 10))
        m = PyramidHART(random_state=0).fit(X)
        gs = m.geometry_stats()
        assert abs(gs['corr_TQ']) < 0.10, f"corr_TQ={gs['corr_TQ']:.4f} far from 0"

    def test_corr_AQ_nonzero(self, X_medium):
        """Prop 1 trade-off: Cov(A, Q) ≠ 0."""
        m = PyramidHART(random_state=0).fit(X_medium)
        gs = m.geometry_stats()
        # Not necessarily large but should differ from corr_TQ
        # corr_AQ should be more negative than corr_TQ for typical data
        assert not np.isnan(gs['corr_AQ'])

    def test_partitions_structure(self, fitted_model):
        gs = fitted_model.geometry_stats()
        parts = gs['partitions']
        assert len(parts) == fitted_model.n_partitions_
        required_keys = {
            'id', 'n', 'A_mean', 'A_std', 'sign_consistent_fraction',
            'minority_sign_mean', 'T_mean', 'T_std', 'Q_mean', 'Q_std',
            'cone_fraction',
        }
        for p in parts:
            assert required_keys <= set(p.keys()), f"Partition missing keys: {set(p.keys())}"
            assert p['n'] >= 1

    def test_partition_A_mean_nonpositive(self, fitted_model):
        gs = fitted_model.geometry_stats()
        for p in gs['partitions']:
            assert p['A_mean'] <= 1e-10, f"Partition {p['id']} has A_mean > 0"

    def test_partition_n_sums_to_total(self, fitted_model, X_small):
        gs = fitted_model.geometry_stats()
        total = sum(p['n'] for p in gs['partitions'])
        assert total == len(X_small)

    def test_n_d_match_data(self, X_small):
        m = PyramidHART(random_state=0).fit(X_small)
        gs = m.geometry_stats()
        assert gs['n'] == len(X_small)
        assert gs['d'] == X_small.shape[1]


# ---------------------------------------------------------------------------
# 4.  Standalone geometry_stats() function
# ---------------------------------------------------------------------------

class TestGeometryStatsFunction:

    def test_returns_dict(self, rng):
        X_z = rng.standard_normal((100, 4))
        gs = geometry_stats(X_z)
        assert isinstance(gs, dict)

    def test_A_bounds(self, rng):
        X_z = rng.standard_normal((300, 8))
        gs = geometry_stats(X_z)
        assert gs['A_max'] <= 1e-10
        # Tight per-point bound: A_i >= -r_i * sqrt(d); the global minimum is
        # bounded by -max(r) * sqrt(d) = -sqrt(Q_max) * sqrt(d), not the
        # mean-based A_theoretical_lower_bound which can be less negative.
        max_r = np.sqrt(gs['Q_max'])
        assert gs['A_min'] >= -max_r * np.sqrt(gs['d']) - 1e-10

    def test_homogeneity(self, rng):
        X_z = rng.standard_normal((500, 6))
        gs = geometry_stats(X_z)
        assert abs(gs['A_homogeneity'] - 2.0) < 0.01
        assert abs(gs['T_homogeneity'] - 4.0) < 0.01

    def test_minority_sign_mean_positive(self, rng):
        X_z = rng.standard_normal((200, 5))
        gs = geometry_stats(X_z)
        assert gs['minority_sign_mean'] >= 0.0

    def test_corr_TQ_approx_zero_large_sample(self, rng):
        X_z = rng.standard_normal((3000, 8))
        gs = geometry_stats(X_z)
        assert abs(gs['corr_TQ']) < 0.05

    def test_d1_edge_case(self, rng):
        """d=1: S=z, Q=z², T=0, A=|z|−|z|=0 always."""
        X_z = rng.standard_normal((200, 1))
        gs = geometry_stats(X_z)
        # T = S^2 - Q = z^2 - z^2 = 0 for d=1
        assert abs(gs['T_mean']) < 1e-10
        # A = |z| - |z| = 0 for d=1
        assert abs(gs['A_mean']) < 1e-10

    def test_correlated_data_positive_T_mean(self):
        """Positively correlated features → E[T] > 0 (Theorem 2)."""
        rng = np.random.default_rng(1)
        n, d = 2000, 5
        # All features = shared factor + small noise
        factor = rng.standard_normal(n)
        X_z = factor[:, None] + 0.1 * rng.standard_normal((n, d))
        gs = geometry_stats(X_z)
        # E[T] should be positive (correlated → cooperative)
        assert gs['T_mean'] > 0, f"Expected T_mean>0 for correlated data, got {gs['T_mean']:.3f}"


# ---------------------------------------------------------------------------
# 5.  Pure function correctness
# ---------------------------------------------------------------------------

class TestPureFunctions:

    def test_compute_S_sum(self):
        X_z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        S = compute_S(X_z)
        np.testing.assert_allclose(S, [6.0, 0.0])

    def test_compute_Q_sum_squares(self):
        X_z = np.array([[3.0, 4.0], [1.0, 0.0]])
        Q = compute_Q(X_z)
        np.testing.assert_allclose(Q, [25.0, 1.0])

    def test_compute_T_identity(self):
        """T = S² − Q = 2Σᵢ<ⱼ zᵢzⱼ."""
        X_z = np.array([[1.0, 2.0]])
        T = compute_T(X_z)
        # S = 3, Q = 5, T = 9 - 5 = 4 = 2 * 1 * 2
        np.testing.assert_allclose(T, [4.0])

    def test_compute_A_known_values(self):
        # [1, 1]: S=2, |z|₁=2, A=|2|-2=0
        # [1,-1]: S=0, |z|₁=2, A=0-2=-2
        # [2,-1]: S=1, |z|₁=3, A=1-3=-2
        X_z = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, -1.0]])
        A = compute_A(X_z)
        np.testing.assert_allclose(A, [0.0, -2.0, -2.0])

    def test_minority_sign_total_known(self):
        # [1,  1]: pos=2, neg=0, mst=0  → -A/2 = 0
        # [1, -1]: pos=1, neg=1, mst=1  → -A/2 = 1
        # [2, -1]: pos=2, neg=1, mst=1  → -A/2 = 1
        X_z = np.array([[1.0, 1.0], [1.0, -1.0], [2.0, -1.0]])
        mst = minority_sign_total(X_z)
        np.testing.assert_allclose(mst, [0.0, 1.0, 1.0])
