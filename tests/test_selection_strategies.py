"""
Unit tests for selection strategies (partition-aware API).
"""

import numpy as np
import pytest
from hvrt import (
    HVRTSampleReducer,
    centroid_fps,
    medoid_fps,
    variance_ordered,
    stratified,
)


# ---------------------------------------------------------------------------
# Helpers for single-partition test scenarios
# ---------------------------------------------------------------------------

def _single_partition_args(X, budget):
    """
    Build the 5-arg partition-aware call for a single-partition scenario.

    Returns (X, partition_ids, unique_partitions, budgets, random_state)
    with all samples in partition 0.
    """
    n = len(X)
    partition_ids = np.zeros(n, dtype=np.int64)
    unique_partitions = np.array([0], dtype=np.int64)
    budgets = np.array([budget], dtype=np.int64)
    return X, partition_ids, unique_partitions, budgets, 42


# ---------------------------------------------------------------------------
# Determinism / reproducibility tests (partition-aware API)
# ---------------------------------------------------------------------------

def test_centroid_fps_determinism():
    """Centroid FPS should be fully deterministic."""
    X = np.random.randn(100, 5)
    args = _single_partition_args(X, 20)

    indices1 = centroid_fps(*args)
    indices2 = centroid_fps(*args)

    assert np.array_equal(indices1, indices2), "centroid_fps should be deterministic"
    assert len(indices1) == 20
    assert len(np.unique(indices1)) == 20, "No duplicate selections"


def test_medoid_fps_determinism():
    """Medoid FPS should be fully deterministic."""
    X = np.random.randn(100, 5)
    args = _single_partition_args(X, 20)

    indices1 = medoid_fps(*args)
    indices2 = medoid_fps(*args)

    assert np.array_equal(indices1, indices2), "medoid_fps should be deterministic"
    assert len(indices1) == 20
    assert len(np.unique(indices1)) == 20, "No duplicate selections"


def test_variance_ordered_determinism():
    """variance_ordered should be fully deterministic."""
    X = np.random.randn(100, 5)
    args = _single_partition_args(X, 20)

    indices1 = variance_ordered(*args)
    indices2 = variance_ordered(*args)

    assert np.array_equal(indices1, indices2), "variance_ordered should be deterministic"
    assert len(indices1) == 20


def test_stratified_reproducibility():
    """stratified should be reproducible with the same seed."""
    X = np.random.randn(100, 5)
    args = _single_partition_args(X, 20)

    indices1 = stratified(*args)
    indices2 = stratified(*args)

    assert np.array_equal(indices1, indices2), "stratified should be reproducible with seed"
    assert len(indices1) == 20


# ---------------------------------------------------------------------------
# Return-type / index-range checks
# ---------------------------------------------------------------------------

def test_strategies_return_global_indices():
    """Strategies should return valid global indices into X."""
    n = 100
    X = np.random.randn(n, 5)
    budget = 30
    args = _single_partition_args(X, budget)

    for fn in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        idx = fn(*args)
        assert idx.dtype == np.int64, f"{fn.__name__} dtype"
        assert len(idx) == budget, f"{fn.__name__} count"
        assert idx.min() >= 0 and idx.max() < n, f"{fn.__name__} index range"


def test_multi_partition_selection():
    """Strategies should handle multiple partitions and return correct count."""
    rng = np.random.RandomState(0)
    n = 200
    X = rng.randn(n, 4)
    # Two equal partitions
    partition_ids = np.repeat([0, 1], n // 2).astype(np.int64)
    unique_partitions = np.array([0, 1], dtype=np.int64)
    budgets = np.array([15, 15], dtype=np.int64)

    for fn in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        idx = fn(X, partition_ids, unique_partitions, budgets, 42)
        assert len(idx) == 30, f"{fn.__name__} total count"
        assert idx.min() >= 0 and idx.max() < n


# ---------------------------------------------------------------------------
# Edge case: partition smaller than budget
# ---------------------------------------------------------------------------

def test_edge_case_small_partition():
    """Strategies should return all samples when n_select > partition size."""
    X = np.random.randn(10, 3)
    budget = 20  # request more than available

    args = _single_partition_args(X, budget)

    for fn in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        indices = fn(*args)
        assert len(indices) == len(X), (
            f"{fn.__name__} should return all samples when budget > partition size"
        )
        assert len(np.unique(indices)) == len(X), "No duplicates"


# ---------------------------------------------------------------------------
# Legacy HVRTSampleReducer tests (string-strategy path unchanged)
# ---------------------------------------------------------------------------

def test_reducer_with_builtin_strategies():
    """HVRTSampleReducer with all built-in strategies should work."""
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)

    strategies = ['centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified']

    for strategy in strategies:
        reducer = HVRTSampleReducer(
            reduction_ratio=0.2,
            selection_strategy=strategy,
            maintain_ratio=False,
            random_state=42,
        )
        reducer.fit(X, y)

        assert hasattr(reducer, 'selected_indices_')
        assert len(reducer.selected_indices_) > 0
        assert len(reducer.selected_indices_) <= len(X) * 0.2 * 1.1
        assert reducer.selection_strategy_name_ == strategy


def test_reducer_with_custom_strategy():
    """HVRTSampleReducer with a custom callable strategy should work."""
    def custom_strategy(X_partition, n_select, random_state):
        return np.arange(min(n_select, len(X_partition)))

    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.randn(500)

    reducer = HVRTSampleReducer(
        reduction_ratio=0.2,
        selection_strategy=custom_strategy,
        random_state=42,
    )
    reducer.fit(X, y)

    assert hasattr(reducer, 'selected_indices_')
    assert len(reducer.selected_indices_) > 0
    assert reducer.selection_strategy_name_ == 'custom_strategy'


def test_reducer_determinism_across_strategies():
    """Reducer should be deterministic for deterministic strategies."""
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.randn(500)

    for strategy in ['centroid_fps', 'medoid_fps', 'variance_ordered']:
        r1 = HVRTSampleReducer(
            reduction_ratio=0.2, selection_strategy=strategy, random_state=42
        )
        r1.fit(X, y)

        r2 = HVRTSampleReducer(
            reduction_ratio=0.2, selection_strategy=strategy, random_state=42
        )
        r2.fit(X, y)

        assert np.array_equal(r1.selected_indices_, r2.selected_indices_), (
            f"{strategy} should produce identical results with same seed"
        )


def test_invalid_strategy_name():
    """Invalid strategy name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        HVRTSampleReducer(selection_strategy='invalid_strategy')


def test_invalid_strategy_type():
    """Non-string, non-callable strategy should raise TypeError."""
    with pytest.raises(TypeError, match="selection_strategy must be str or callable"):
        HVRTSampleReducer(selection_strategy=123)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
