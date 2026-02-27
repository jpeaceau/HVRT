"""
Unit tests for selection strategies (partition-aware API).
"""

import numpy as np
import pytest
from hvrt import (
    centroid_fps,
    medoid_fps,
    variance_ordered,
    stratified,
)


# ---------------------------------------------------------------------------
# Helpers for single-partition test scenarios
# ---------------------------------------------------------------------------

def _prepare_single_partition(strategy, X, budget):
    """
    Prepare a single-partition context and return (ctx, budgets, random_state).
    """
    n = len(X)
    partition_ids = np.zeros(n, dtype=np.int64)
    unique_partitions = np.array([0], dtype=np.int64)
    budgets = np.array([budget], dtype=np.int64)
    ctx = strategy.prepare(X, partition_ids, unique_partitions)
    return ctx, budgets, 42


def _select_single_partition(strategy, X, budget):
    """Run prepare+select for a single-partition scenario."""
    ctx, budgets, rs = _prepare_single_partition(strategy, X, budget)
    return strategy.select(ctx, budgets, random_state=rs)


# ---------------------------------------------------------------------------
# Determinism / reproducibility tests (partition-aware API)
# ---------------------------------------------------------------------------

def test_centroid_fps_determinism():
    """Centroid FPS should be fully deterministic."""
    X = np.random.randn(100, 5)

    indices1 = _select_single_partition(centroid_fps, X, 20)
    indices2 = _select_single_partition(centroid_fps, X, 20)

    assert np.array_equal(indices1, indices2), "centroid_fps should be deterministic"
    assert len(indices1) == 20
    assert len(np.unique(indices1)) == 20, "No duplicate selections"


def test_medoid_fps_determinism():
    """Medoid FPS should be fully deterministic."""
    X = np.random.randn(100, 5)

    indices1 = _select_single_partition(medoid_fps, X, 20)
    indices2 = _select_single_partition(medoid_fps, X, 20)

    assert np.array_equal(indices1, indices2), "medoid_fps should be deterministic"
    assert len(indices1) == 20
    assert len(np.unique(indices1)) == 20, "No duplicate selections"


def test_variance_ordered_determinism():
    """variance_ordered should be fully deterministic."""
    X = np.random.randn(100, 5)

    indices1 = _select_single_partition(variance_ordered, X, 20)
    indices2 = _select_single_partition(variance_ordered, X, 20)

    assert np.array_equal(indices1, indices2), "variance_ordered should be deterministic"
    assert len(indices1) == 20


def test_stratified_reproducibility():
    """stratified should be reproducible with the same seed."""
    X = np.random.randn(100, 5)

    indices1 = _select_single_partition(stratified, X, 20)
    indices2 = _select_single_partition(stratified, X, 20)

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

    for strategy in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        idx = _select_single_partition(strategy, X, budget)
        assert idx.dtype == np.int64, f"{type(strategy).__name__} dtype"
        assert len(idx) == budget, f"{type(strategy).__name__} count"
        assert idx.min() >= 0 and idx.max() < n, f"{type(strategy).__name__} index range"


def test_multi_partition_selection():
    """Strategies should handle multiple partitions and return correct count."""
    rng = np.random.RandomState(0)
    n = 200
    X = rng.randn(n, 4)
    # Two equal partitions
    partition_ids = np.repeat([0, 1], n // 2).astype(np.int64)
    unique_partitions = np.array([0, 1], dtype=np.int64)
    budgets = np.array([15, 15], dtype=np.int64)

    for strategy in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        ctx = strategy.prepare(X, partition_ids, unique_partitions)
        idx = strategy.select(ctx, budgets, random_state=42)
        assert len(idx) == 30, f"{type(strategy).__name__} total count"
        assert idx.min() >= 0 and idx.max() < n


# ---------------------------------------------------------------------------
# Edge case: partition smaller than budget
# ---------------------------------------------------------------------------

def test_edge_case_small_partition():
    """Strategies should return all samples when n_select > partition size."""
    X = np.random.randn(10, 3)
    budget = 20  # request more than available

    for strategy in [centroid_fps, medoid_fps, variance_ordered, stratified]:
        indices = _select_single_partition(strategy, X, budget)
        assert len(indices) == len(X), (
            f"{type(strategy).__name__} should return all samples when budget > partition size"
        )
        assert len(np.unique(indices)) == len(X), "No duplicates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
