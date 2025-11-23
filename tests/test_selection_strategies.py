"""
Unit tests for selection strategies.
"""

import sys
from pathlib import Path

# Add src to path for local imports

import numpy as np
import pytest
from hvrt import (
    HVRTSampleReducer,
    centroid_fps,
    medoid_fps,
    variance_ordered,
    stratified
)


def test_centroid_fps_determinism():
    """Test that centroid FPS is fully deterministic."""
    X = np.random.randn(100, 5)
    n_select = 20

    # Run twice
    indices1 = centroid_fps(X, n_select, random_state=42)
    indices2 = centroid_fps(X, n_select, random_state=42)

    assert np.array_equal(indices1, indices2), "Centroid FPS should be deterministic"
    assert len(indices1) == n_select
    assert len(np.unique(indices1)) == n_select, "No duplicate selections"


def test_medoid_fps_determinism():
    """Test that medoid FPS is fully deterministic."""
    X = np.random.randn(100, 5)
    n_select = 20

    indices1 = medoid_fps(X, n_select, random_state=42)
    indices2 = medoid_fps(X, n_select, random_state=42)

    assert np.array_equal(indices1, indices2), "Medoid FPS should be deterministic"
    assert len(indices1) == n_select


def test_variance_ordered_determinism():
    """Test that variance_ordered is fully deterministic."""
    X = np.random.randn(100, 5)
    n_select = 20

    indices1 = variance_ordered(X, n_select, random_state=42)
    indices2 = variance_ordered(X, n_select, random_state=42)

    assert np.array_equal(indices1, indices2), "Variance-ordered should be deterministic"
    assert len(indices1) == n_select


def test_stratified_reproducibility():
    """Test that stratified is reproducible with same seed."""
    X = np.random.randn(100, 5)
    n_select = 20

    indices1 = stratified(X, n_select, random_state=42)
    indices2 = stratified(X, n_select, random_state=42)

    assert np.array_equal(indices1, indices2), "Stratified should be reproducible with seed"
    assert len(indices1) == n_select


def test_reducer_with_builtin_strategies():
    """Test HVRTSampleReducer with all built-in strategies."""
    np.random.seed(42)
    X = np.random.randn(1000, 10)  # Larger dataset to avoid ratio adjustment
    y = np.random.randn(1000)

    strategies = ['centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified']

    for strategy in strategies:
        reducer = HVRTSampleReducer(
            reduction_ratio=0.2,
            selection_strategy=strategy,
            maintain_ratio=False,  # Disable ratio adjustment for test
            random_state=42
        )

        reducer.fit(X, y)

        assert hasattr(reducer, 'selected_indices_')
        assert len(reducer.selected_indices_) > 0
        assert len(reducer.selected_indices_) <= len(X) * 0.2 * 1.1  # Allow 10% tolerance
        assert reducer.selection_strategy_name_ == strategy


def test_reducer_with_custom_strategy():
    """Test HVRTSampleReducer with custom callable strategy."""
    def custom_strategy(X_partition, n_select, random_state):
        """Simple custom strategy: select first n_select samples."""
        return np.arange(min(n_select, len(X_partition)))

    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.randn(500)

    reducer = HVRTSampleReducer(
        reduction_ratio=0.2,
        selection_strategy=custom_strategy,
        random_state=42
    )

    reducer.fit(X, y)

    assert hasattr(reducer, 'selected_indices_')
    assert len(reducer.selected_indices_) > 0
    assert reducer.selection_strategy_name_ == 'custom_strategy'


def test_reducer_determinism_across_strategies():
    """Test that reducer is deterministic for deterministic strategies."""
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.randn(500)

    for strategy in ['centroid_fps', 'medoid_fps', 'variance_ordered']:
        reducer1 = HVRTSampleReducer(
            reduction_ratio=0.2,
            selection_strategy=strategy,
            random_state=42
        )
        reducer1.fit(X, y)

        reducer2 = HVRTSampleReducer(
            reduction_ratio=0.2,
            selection_strategy=strategy,
            random_state=42
        )
        reducer2.fit(X, y)

        assert np.array_equal(
            reducer1.selected_indices_,
            reducer2.selected_indices_
        ), f"{strategy} should produce identical results with same seed"


def test_invalid_strategy_name():
    """Test that invalid strategy name raises error."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        HVRTSampleReducer(selection_strategy='invalid_strategy')


def test_invalid_strategy_type():
    """Test that invalid strategy type raises error."""
    with pytest.raises(TypeError, match="selection_strategy must be str or callable"):
        HVRTSampleReducer(selection_strategy=123)


def test_edge_case_small_partition():
    """Test strategies handle partitions smaller than n_select."""
    X = np.random.randn(10, 3)  # Small partition
    n_select = 20  # Request more than available

    for strategy in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        indices = strategy(X, n_select, random_state=42)
        assert len(indices) == len(X), f"{strategy.__name__} should return all samples when n_select > n_partition"
        assert len(np.unique(indices)) == len(X), "No duplicates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
