"""
Reduction helpers for HVRT v2.

Provides partition-budget allocation (size-weighted or variance-weighted)
and multi-strategy within-partition sample selection.
"""

from __future__ import annotations

from typing import Callable, Literal, Union

import numpy as np

from ._budgets import _compute_weights, allocate_budgets
from .reduction_strategies import get_strategy

SelectionMethod = Union[
    Literal['fps', 'centroid_fps', 'medoid_fps', 'variance_ordered', 'stratified'],
    Callable,
]


def compute_partition_budgets(
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    n_target: int,
    min_per_partition: int,
    variance_weighted: bool,
    X_z: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the number of samples to select from each partition.

    Parameters
    ----------
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    n_target : int
        Total samples to select across all partitions.
    min_per_partition : int
        Floor on samples selected from any single partition.
    variance_weighted : bool
        If True, weight partitions by mean |z-score| (favours tail partitions).
        If False, weight proportionally to partition size.
    X_z : ndarray (n_samples, n_features) or None
        Required when variance_weighted=True.

    Returns
    -------
    budgets : ndarray of int, shape (len(unique_partitions),)
    """
    partition_sizes = np.array(
        [np.sum(partition_ids == pid) for pid in unique_partitions], dtype=int
    )
    weights = _compute_weights(partition_ids, unique_partitions, variance_weighted, X_z)
    budgets = allocate_budgets(weights, n_target, floor=min_per_partition)

    # Variance-weighted allocation can assign more budget to a partition than
    # it actually contains.  Clip each budget to its partition size, then
    # greedily redistribute any shortfall to partitions with remaining capacity
    # so the total stays equal to n_target.
    np.clip(budgets, 0, partition_sizes, out=budgets)
    remaining_capacity = partition_sizes - budgets
    shortfall = n_target - int(budgets.sum())
    for _ in range(shortfall):
        candidates = np.where(remaining_capacity > 0)[0]
        if not len(candidates):
            break
        pick = candidates[np.argmax(remaining_capacity[candidates])]
        budgets[pick] += 1
        remaining_capacity[pick] -= 1

    return budgets


def select_from_partitions(
    X_z: np.ndarray,
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    budgets: np.ndarray,
    method: SelectionMethod,
    random_state: int,
) -> np.ndarray:
    """
    Select samples from each partition using the given strategy.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    method : str or callable
        Built-in string aliases or a partition-aware callable
        ``(X_z, partition_ids, unique_partitions, budgets, random_state) -> ndarray[int]``.
    random_state : int

    Returns
    -------
    indices : ndarray of int
    """
    strategy = get_strategy(method) if isinstance(method, str) else method
    return strategy(X_z, partition_ids, unique_partitions, budgets, random_state)
