"""
Reduction helpers for HVRT v2.

Provides partition-budget allocation (size-weighted or variance-weighted)
and multi-strategy within-partition selection.
"""

import numpy as np
from .utils import centroid_fps
from .selection_strategies import get_strategy


def compute_partition_budgets(
    partition_ids,
    unique_partitions,
    n_target,
    min_per_partition,
    variance_weighted,
    X_z=None,
):
    """
    Compute how many samples to select from each partition.

    Parameters
    ----------
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    n_target : int
        Total samples to select.
    min_per_partition : int
        Minimum samples taken from any partition.
    variance_weighted : bool
        If True, weight partitions by mean absolute z-score (favour tails).
        If False, weight proportionally to partition size.
    X_z : ndarray (n_samples, n_features) or None
        Required when variance_weighted=True.

    Returns
    -------
    budgets : ndarray of int, shape (len(unique_partitions),)
    """
    partition_sizes = np.array(
        [np.sum(partition_ids == pid) for pid in unique_partitions], dtype=float
    )

    if variance_weighted and X_z is not None:
        weights = np.array(
            [np.mean(np.abs(X_z[partition_ids == pid])) for pid in unique_partitions]
        )
        weights = np.maximum(weights, 1e-10)
        weights = weights / weights.sum()
    else:
        weights = partition_sizes / partition_sizes.sum()

    budgets = np.maximum(min_per_partition, (weights * n_target).astype(int))

    # Trim or pad to exact target
    while budgets.sum() > n_target:
        budgets[np.argmax(budgets)] -= 1
    while budgets.sum() < n_target:
        budgets[np.argmin(budgets)] += 1

    return budgets


def select_from_partitions(
    X_z,
    partition_ids,
    unique_partitions,
    budgets,
    method,
    random_state,
):
    """
    Select samples from each partition using the given strategy.

    Parameters
    ----------
    X_z : ndarray (n_samples, n_features)
        Normalized feature matrix (used for FPS-based methods).
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    budgets : ndarray of int
    method : str
        'fps' uses centroid-seeded FPS; any other string is looked up in
        the selection_strategies registry.
    random_state : int

    Returns
    -------
    indices : ndarray of int
        Global indices into X_z (and the original X stored in the model).
    """
    use_fps = method in ('fps', 'centroid_fps')
    strategy = None if use_fps else get_strategy(method)

    selected = []
    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue
        mask = partition_ids == pid
        part_indices = np.where(mask)[0]

        if len(part_indices) <= budget:
            selected.extend(part_indices.tolist())
        else:
            X_part = X_z[mask]
            if use_fps:
                local = centroid_fps(X_part, budget)
            else:
                local = strategy(X_part, budget, random_state)
            selected.extend(part_indices[local].tolist())

    return np.array(selected, dtype=np.int64)
