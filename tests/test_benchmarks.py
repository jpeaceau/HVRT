"""
Smoke tests for the benchmark suite.

These tests verify that every component runs end-to-end without error on
small inputs. They do NOT validate result quality (that is the purpose of
running the full benchmark suite externally).
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

class TestDatasets:
    @pytest.mark.parametrize('ds_key,expected_shape', [
        ('adult',                (200, 8)),
        ('fraud',                (200, 15)),
        ('housing',              (200, 6)),
        ('multimodal',           (200, 10)),
        ('emergence_divergence', (200, 5)),
        ('emergence_bifurcation',(200, 5)),
    ])
    def test_generator_shape(self, ds_key, expected_shape):
        from hvrt.benchmarks.datasets import BENCHMARK_DATASETS
        fn = BENCHMARK_DATASETS[ds_key]
        X, y, names = fn(n=expected_shape[0], random_state=0)
        assert X.shape == expected_shape
        assert len(y) == expected_shape[0]
        assert len(names) == expected_shape[1]

    def test_all_datasets_registered(self):
        from hvrt.benchmarks.datasets import BENCHMARK_DATASETS
        expected = {
            'adult', 'fraud', 'housing', 'multimodal',
            'emergence_divergence', 'emergence_bifurcation',
        }
        assert set(BENCHMARK_DATASETS.keys()) == expected

    def test_reproducible(self):
        from hvrt.benchmarks.datasets import make_adult_like
        X1, _, _ = make_adult_like(n=50, random_state=0)
        X2, _, _ = make_adult_like(n=50, random_state=0)
        np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture(autouse=True)
    def _data(self):
        rng = np.random.RandomState(0)
        self.X = rng.randn(200, 5)
        self.X_perturbed = self.X + rng.randn(200, 5) * 0.1

    def test_marginal_fidelity_perfect(self):
        from hvrt.benchmarks.metrics import marginal_fidelity
        score = marginal_fidelity(self.X, self.X)
        assert abs(score - 1.0) < 1e-6

    def test_marginal_fidelity_degraded(self):
        from hvrt.benchmarks.metrics import marginal_fidelity
        rng = np.random.RandomState(1)
        X_bad = rng.randn(200, 5) * 5  # completely different distribution
        score = marginal_fidelity(self.X, X_bad)
        assert score < 1.0

    def test_correlation_fidelity_perfect(self):
        from hvrt.benchmarks.metrics import correlation_fidelity
        score = correlation_fidelity(self.X, self.X)
        assert abs(score - 1.0) < 1e-6

    def test_tail_preservation_perfect(self):
        from hvrt.benchmarks.metrics import tail_preservation
        score = tail_preservation(self.X, self.X)
        assert abs(score - 1.0) < 1e-4

    def test_discriminator_accuracy_identical(self):
        """Logistic regression on real-vs-identical should be ~50%."""
        from hvrt.benchmarks.metrics import discriminator_accuracy
        acc = discriminator_accuracy(self.X, self.X)
        assert 0.0 <= acc <= 1.0

    def test_privacy_dcr_positive(self):
        from hvrt.benchmarks.metrics import privacy_dcr
        rng = np.random.RandomState(2)
        X_synth = rng.randn(100, 5) * 2
        dcr = privacy_dcr(self.X, X_synth)
        assert dcr >= 0.0

    def test_novelty_min_self_zero(self):
        from hvrt.benchmarks.metrics import novelty_min
        # Comparing data to itself: some points are identical → min = 0
        score = novelty_min(self.X, self.X)
        assert score == pytest.approx(0.0, abs=1e-8)

    def test_novelty_min_far_data(self):
        from hvrt.benchmarks.metrics import novelty_min
        rng = np.random.RandomState(3)
        X_far = rng.randn(50, 5) + 100  # very far from self.X
        score = novelty_min(self.X, X_far)
        assert score > 0

    def test_ml_utility_tstr_runs(self):
        from hvrt.benchmarks.metrics import ml_utility_tstr
        rng = np.random.RandomState(4)
        X_tr, y_tr = rng.randn(100, 5), rng.randn(100)
        X_te, y_te = rng.randn(50, 5), rng.randn(50)
        score = ml_utility_tstr(X_tr, y_tr, X_te, y_te, task='regression')
        assert isinstance(score, float)

    def test_emergence_score_runs(self):
        from hvrt.benchmarks.metrics import emergence_score
        rng = np.random.RandomState(5)
        X = rng.randn(100, 5)
        y = X[:, 0] + rng.randn(100) * 0.3
        score = emergence_score(X, y, X, y)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Runners (smoke tests on small data)
# ---------------------------------------------------------------------------

class TestRunners:
    def test_run_reduction_benchmark_smoke(self):
        from hvrt.benchmarks.runners import run_reduction_benchmark
        results = run_reduction_benchmark(
            datasets=['adult'],
            methods=['HVRT-size', 'FastHVRT-size', 'Random'],
            ratios=[0.5],
            random_state=0,
            verbose=False,
            max_n=300,
        )
        assert len(results) == 3  # 1 dataset × 1 ratio × 3 methods
        for r in results:
            assert r['task'] == 'reduce'
            assert 'metrics' in r
            assert 'marginal_fidelity' in r['metrics']

    def test_run_expansion_benchmark_smoke(self):
        from hvrt.benchmarks.runners import run_expansion_benchmark
        results = run_expansion_benchmark(
            datasets=['housing'],
            methods=['FastHVRT-size', 'GMM', 'Bootstrap-Noise'],
            expansion_ratios=[1.0],
            random_state=0,
            verbose=False,
            max_n=300,
        )
        assert len(results) == 3
        for r in results:
            assert r['task'] == 'expand'
            assert 'discriminator_accuracy' in r['metrics']

    def test_run_full_benchmark_smoke(self):
        from hvrt.benchmarks.runners import run_full_benchmark
        results = run_full_benchmark(
            datasets=['multimodal'],
            tasks=['reduce', 'expand'],
            random_state=0,
            verbose=False,
            max_n=300,
        )
        tasks = {r['task'] for r in results}
        assert 'reduce' in tasks
        assert 'expand' in tasks

    def test_load_results(self, tmp_path):
        from hvrt.benchmarks.runners import run_reduction_benchmark, load_results
        out = str(tmp_path / 'test_results.json')
        run_reduction_benchmark(
            datasets=['adult'],
            methods=['Random'],
            ratios=[0.5],
            random_state=0,
            save_path=out,
            verbose=False,
            max_n=200,
        )
        loaded = load_results(out)
        assert len(loaded) >= 1


# ---------------------------------------------------------------------------
# Visualization (smoke tests — no display)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_print_results_table(self):
        import io
        from hvrt.benchmarks.runners import run_reduction_benchmark
        from hvrt.benchmarks.visualization import print_results_table

        results = run_reduction_benchmark(
            datasets=['adult'],
            methods=['HVRT-size', 'Random'],
            ratios=[0.3],
            random_state=0,
            verbose=False,
        )
        buf = io.StringIO()
        print_results_table(results, task='reduce', file=buf)
        output = buf.getvalue()
        assert 'HVRT' in output
        assert 'Random' in output

    def test_results_to_string(self):
        from hvrt.benchmarks.runners import run_reduction_benchmark
        from hvrt.benchmarks.visualization import results_to_string

        results = run_reduction_benchmark(
            datasets=['adult'],
            methods=['HVRT-size'],
            ratios=[0.3],
            random_state=0,
            verbose=False,
        )
        s = results_to_string(results, task='reduce')
        assert isinstance(s, str) and len(s) > 0
