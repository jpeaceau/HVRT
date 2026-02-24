"""
Shared partition-budget utilities for reduction and expansion.

Provides the single canonical _iter_partitions generator (previously duplicated
in reduction_strategies and generation_strategies) and the shared weight-
computation + budget-adjustment logic (previously duplicated between reduce.py
and expand.py).
"""

from __future__ import annotations

import numpy as np


def _iter_partitions(X_z, partition_ids, unique_partitions, budgets):
    """Yield (global_indices, X_part, budget) for each non-empty partition."""
    for pid, budget in zip(unique_partitions, budgets):
        if budget == 0:
            continue
        mask = partition_ids == pid
        yield np.where(mask)[0], X_z[mask], budget


def _partition_pos(
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
) -> np.ndarray:
    """
    Map every element of ``partition_ids`` to its index in ``unique_partitions``.

    ``unique_partitions`` is always sorted (output of ``np.unique``), so
    ``np.searchsorted`` gives the position of each sample's partition without
    a Python loop.  Used to enable vectorised ``np.bincount`` aggregations
    in place of per-partition Python loops.
    """
    return np.searchsorted(unique_partitions, partition_ids)


def _compute_weights(
    partition_ids: np.ndarray,
    unique_partitions: np.ndarray,
    variance_weighted: bool,
    X_z: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute normalised per-partition weights.

    Parameters
    ----------
    partition_ids : ndarray (n_samples,)
    unique_partitions : ndarray
    variance_weighted : bool
        True  — weight by mean |z-score| per partition (favours tails).
        False — weight proportionally to partition size.
    X_z : ndarray (n_samples, n_features) or None
        Required when variance_weighted=True.

    Returns
    -------
    weights : ndarray of float, shape (len(unique_partitions),)
    """
    pos = _partition_pos(partition_ids, unique_partitions)
    n_parts = len(unique_partitions)
    partition_sizes = np.bincount(pos, minlength=n_parts).astype(float)

    if variance_weighted and X_z is not None:
        # Mean |z-score| per partition — fully vectorised via bincount.
        # np.abs(X_z).mean(axis=1) gives the mean absolute z per sample
        # across all features; bincount then sums those per-sample means
        # within each partition and we divide by partition size to get the
        # per-partition mean.  This is mathematically identical to the
        # previous per-partition loop.
        mean_abs_z_per_sample = np.abs(X_z).mean(axis=1)  # (n,)
        weights = (
            np.bincount(pos, weights=mean_abs_z_per_sample, minlength=n_parts)
            / np.maximum(partition_sizes, 1.0)
        )
        weights = np.maximum(weights, 1e-10)
        weights /= weights.sum()
    else:
        weights = partition_sizes / partition_sizes.sum()

    return weights


def allocate_budgets(
    weights: np.ndarray,
    n_target: int,
    floor: int = 0,
) -> np.ndarray:
    """
    Convert normalised weights to integer per-partition budgets summing to n_target.

    Parameters
    ----------
    weights : ndarray of float
    n_target : int
        Total samples to allocate.
    floor : int, default 0
        Minimum budget for any single partition.

    Returns
    -------
    budgets : ndarray of int, shape (len(weights),)
    """
    budgets = np.maximum(floor, (weights * n_target).astype(int))

    while budgets.sum() > n_target:
        budgets[np.argmax(budgets)] -= 1
    while budgets.sum() < n_target:
        budgets[np.argmin(budgets)] += 1

    return budgets
