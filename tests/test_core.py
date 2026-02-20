"""
Tests for HVRT and FastHVRT core classes.

Covers: fit, reduce, expand, augment, fit_transform (params-based and legacy
mode), utility methods, presets, unsupervised/supervised modes.
"""

import pytest
import numpy as np
from hvrt import HVRT, FastHVRT, HVRTDeprecationWarning, ReduceParams, ExpandParams, AugmentParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_X():
    rng = np.random.RandomState(0)
    return rng.randn(300, 5)


@pytest.fixture
def small_Xy():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 5)
    y = X[:, 0] + 0.5 * X[:, 1] + rng.randn(300) * 0.3
    return X, y


@pytest.fixture(params=[HVRT, FastHVRT])
def ModelCls(request):
    return request.param


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, ModelCls, small_X):
        model = ModelCls(random_state=0)
        result = model.fit(small_X)
        assert result is model

    def test_fit_sets_attributes(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        assert hasattr(model, 'X_')
        assert hasattr(model, 'X_z_')
        assert hasattr(model, 'partition_ids_')
        assert hasattr(model, 'unique_partitions_')
        assert hasattr(model, 'n_partitions_')
        assert hasattr(model, 'tree_')

    def test_fit_with_y(self, ModelCls, small_Xy):
        X, y = small_Xy
        model = ModelCls(random_state=0).fit(X, y)
        assert hasattr(model, 'X_')

    def test_fit_supervised_y_weight(self, ModelCls, small_Xy):
        X, y = small_Xy
        model = ModelCls(y_weight=0.5, random_state=0).fit(X, y)
        assert model.n_partitions_ >= 1

    def test_n_partitions_positive(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        assert model.n_partitions_ >= 1

    def test_fit_stores_original_X(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        np.testing.assert_array_equal(model.X_, small_X)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------

class TestReduce:
    def test_reduce_ratio_shape(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_red = model.reduce(ratio=0.3)
        assert X_red.shape[1] == small_X.shape[1]
        assert len(X_red) <= int(len(small_X) * 0.35)

    def test_reduce_n_shape(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_red = model.reduce(n=50)
        assert X_red.shape == (50, small_X.shape[1])

    def test_reduce_return_indices(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_red, idx = model.reduce(n=50, return_indices=True)
        assert len(idx) == 50
        np.testing.assert_array_equal(X_red, small_X[idx])

    def test_reduce_variance_weighted(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X1 = model.reduce(n=50, variance_weighted=True)
        X2 = model.reduce(n=50, variance_weighted=False)
        assert X1.shape == X2.shape

    def test_reduce_methods(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        for method in ('fps', 'medoid_fps', 'stratified', 'variance_ordered'):
            X_red = model.reduce(n=30, method=method)
            assert X_red.shape == (30, small_X.shape[1])

    def test_reduce_requires_n_or_ratio(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        with pytest.raises(ValueError):
            model.reduce()

    def test_reduce_not_both_n_ratio(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        with pytest.raises(ValueError):
            model.reduce(n=50, ratio=0.3)

    def test_reduce_deterministic(self, ModelCls, small_X):
        model = ModelCls(random_state=42).fit(small_X)
        X1 = model.reduce(n=60)
        X2 = model.reduce(n=60)
        np.testing.assert_array_equal(X1, X2)

    def test_reduce_requires_fit(self, ModelCls):
        model = ModelCls()
        with pytest.raises(ValueError):
            model.reduce(n=10)


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------

class TestExpand:
    def test_expand_shape(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_synth = model.expand(n=100)
        assert X_synth.shape == (100, small_X.shape[1])

    def test_expand_large(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_synth = model.expand(n=1000)
        assert X_synth.shape == (1000, small_X.shape[1])

    def test_expand_variance_weighted(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X1 = model.expand(n=100, variance_weighted=True)
        X2 = model.expand(n=100, variance_weighted=False)
        assert X1.shape == X2.shape

    def test_expand_min_novelty(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        with pytest.warns(HVRTDeprecationWarning, match="min_novelty is deprecated"):
            X_synth = model.expand(n=100, min_novelty=0.1)
        assert X_synth.shape[0] == 100

    def test_expand_novelty_stats(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_synth, stats = model.expand(n=50, return_novelty_stats=True)
        assert X_synth.shape[0] == 50
        assert 'min' in stats and 'mean' in stats and 'p5' in stats

    def test_expand_requires_fit(self, ModelCls):
        model = ModelCls()
        with pytest.raises(ValueError):
            model.expand(n=10)

    def test_expand_output_scale_consistent(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_synth = model.expand(n=500)
        orig_std = small_X.std(axis=0)
        synth_std = X_synth.std(axis=0)
        ratio = synth_std / (orig_std + 1e-10)
        assert np.all(ratio > 0.1) and np.all(ratio < 10)


# ---------------------------------------------------------------------------
# augment
# ---------------------------------------------------------------------------

class TestAugment:
    def test_augment_shape(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_aug = model.augment(n=500)
        assert X_aug.shape == (500, small_X.shape[1])

    def test_augment_contains_originals(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_aug = model.augment(n=500)
        np.testing.assert_array_equal(X_aug[:len(small_X)], small_X)

    def test_augment_requires_n_gt_original(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        with pytest.raises(ValueError):
            model.augment(n=100)

    def test_augment_requires_fit(self, ModelCls):
        model = ModelCls()
        with pytest.raises(ValueError):
            model.augment(n=1000)


# ---------------------------------------------------------------------------
# fit_transform — params-based pipeline API
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_fit_transform_reduce(self, ModelCls, small_X):
        model = ModelCls(reduce_params=ReduceParams(ratio=0.3), random_state=0)
        X_red = model.fit_transform(small_X)
        assert X_red.shape[1] == small_X.shape[1]
        assert len(X_red) < len(small_X)

    def test_fit_transform_expand(self, ModelCls, small_X):
        model = ModelCls(expand_params=ExpandParams(n=200), random_state=0)
        X_synth = model.fit_transform(small_X)
        assert X_synth.shape == (200, small_X.shape[1])

    def test_fit_transform_augment(self, ModelCls, small_X):
        model = ModelCls(augment_params=AugmentParams(n=500), random_state=0)
        X_aug = model.fit_transform(small_X)
        assert X_aug.shape == (500, small_X.shape[1])

    def test_fit_transform_kwargs_override_params(self, ModelCls, small_X):
        """kwargs passed to fit_transform() override the params object fields."""
        model = ModelCls(reduce_params=ReduceParams(ratio=0.3), random_state=0)
        X_red = model.fit_transform(small_X, n=40, ratio=None)
        assert len(X_red) == 40

    def test_fit_transform_no_params_raises(self, ModelCls, small_X):
        """fit_transform() with no params and no mode should raise ValueError."""
        model = ModelCls(random_state=0).fit(small_X)
        with pytest.raises(ValueError):
            model.fit_transform(small_X)


# ---------------------------------------------------------------------------
# fit_transform — deprecated mode API
# ---------------------------------------------------------------------------

class TestFitTransformLegacyMode:
    def test_mode_reduce_warns(self, ModelCls, small_X):
        model = ModelCls(mode='reduce', random_state=0)
        with pytest.warns(HVRTDeprecationWarning, match="mode.*deprecated"):
            X_red = model.fit_transform(small_X, ratio=0.3)
        assert X_red.shape[1] == small_X.shape[1]
        assert len(X_red) < len(small_X)

    def test_mode_expand_warns(self, ModelCls, small_X):
        model = ModelCls(mode='expand', random_state=0)
        with pytest.warns(HVRTDeprecationWarning, match="mode.*deprecated"):
            X_synth = model.fit_transform(small_X, n=200)
        assert X_synth.shape == (200, small_X.shape[1])

    def test_mode_augment_warns(self, ModelCls, small_X):
        model = ModelCls(mode='augment', random_state=0)
        with pytest.warns(HVRTDeprecationWarning, match="mode.*deprecated"):
            X_aug = model.fit_transform(small_X, n=500)
        assert X_aug.shape == (500, small_X.shape[1])

    def test_mode_invalid_raises(self, ModelCls, small_X):
        model = ModelCls(mode='invalid', random_state=0)
        model.fit(small_X)
        with pytest.warns(HVRTDeprecationWarning):
            with pytest.raises(ValueError):
                model.fit_transform(small_X)


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_get_partitions(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        parts = model.get_partitions()
        assert isinstance(parts, list)
        assert len(parts) == model.n_partitions_
        for p in parts:
            assert 'id' in p and 'size' in p and 'mean_abs_z' in p

    def test_compute_novelty(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        rng = np.random.RandomState(1)
        X_new = rng.randn(20, 5)
        dists = model.compute_novelty(X_new)
        assert dists.shape == (20,)
        assert np.all(dists >= 0)

    def test_recommend_params(self, ModelCls, small_X):
        params = ModelCls.recommend_params(small_X)
        assert 'n_partitions' in params
        assert 'min_samples_leaf' in params

    def test_n_partitions_override_reduce(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        default_parts = model.n_partitions_
        X_red = model.reduce(n=50, n_partitions=5)
        assert X_red.shape == (50, small_X.shape[1])
        assert model.n_partitions_ <= 5
        model.reduce(n=50, n_partitions=default_parts)
        assert model.n_partitions_ >= 1

    def test_n_partitions_override_expand(self, ModelCls, small_X):
        model = ModelCls(random_state=0).fit(small_X)
        X_synth = model.expand(n=100, n_partitions=4)
        assert X_synth.shape == (100, small_X.shape[1])
        assert model.n_partitions_ <= 4

    def test_fit_once_reduce_and_expand(self, ModelCls, small_X):
        """Tree is fitted once; reduce and expand can both be called without refitting."""
        model = ModelCls(random_state=0).fit(small_X)
        tree_id = id(model.tree_)
        X_red = model.reduce(ratio=0.3)
        X_synth = model.expand(n=100)
        assert id(model.tree_) == tree_id, "Tree should not be replaced by reduce/expand calls"
        assert X_red.shape[1] == small_X.shape[1]
        assert X_synth.shape == (100, small_X.shape[1])


# ---------------------------------------------------------------------------
# HVRT vs FastHVRT differ in target computation
# ---------------------------------------------------------------------------

class TestHVRTvsFastHVRT:
    def test_different_partition_counts(self):
        rng = np.random.RandomState(0)
        X = rng.randn(500, 8)
        m1 = HVRT(random_state=0).fit(X)
        m2 = FastHVRT(random_state=0).fit(X)
        assert m1.n_partitions_ >= 1
        assert m2.n_partitions_ >= 1

    def test_both_produce_valid_reductions(self):
        rng = np.random.RandomState(0)
        X = rng.randn(500, 8)
        for Cls in (HVRT, FastHVRT):
            X_red = Cls(random_state=0).fit(X).reduce(n=100)
            assert X_red.shape == (100, 8)

    def test_both_produce_valid_expansions(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 5)
        for Cls in (HVRT, FastHVRT):
            X_synth = Cls(random_state=0).fit(X).expand(n=300)
            assert X_synth.shape == (300, 5)


# ---------------------------------------------------------------------------
# Pipeline package
# ---------------------------------------------------------------------------

class TestPipelinePackage:
    def test_pipeline_imports(self):
        from hvrt.pipeline import HVRT as PHVRT, FastHVRT as PFastHVRT
        from hvrt.pipeline import ReduceParams, ExpandParams, AugmentParams
        assert PHVRT is HVRT
        assert PFastHVRT is FastHVRT

    def test_sklearn_pipeline_reduce(self, small_X):
        from sklearn.pipeline import Pipeline
        from hvrt.pipeline import HVRT as PHVRT, ReduceParams
        pipe = Pipeline([('hvrt', PHVRT(reduce_params=ReduceParams(ratio=0.3), random_state=0))])
        X_red = pipe.fit_transform(small_X)
        assert X_red.shape[1] == small_X.shape[1]
        assert len(X_red) < len(small_X)

    def test_sklearn_pipeline_expand(self, small_X):
        from sklearn.pipeline import Pipeline
        from hvrt.pipeline import FastHVRT as PFastHVRT, ExpandParams
        pipe = Pipeline([('hvrt', PFastHVRT(expand_params=ExpandParams(n=200), random_state=0))])
        X_synth = pipe.fit_transform(small_X)
        assert X_synth.shape == (200, small_X.shape[1])
