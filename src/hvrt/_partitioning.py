"""
Tree-fitting utilities for HVRT partitioning.

Centralises all decision-tree-related logic: hyperparameter auto-tuning,
tree construction, and per-operation granularity resolution.

Previously split across utils.py (auto_tune_tree_params) and _base.py
(fit_hvrt_tree as a module-level function, _resolve_tree_params as a method).
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def auto_tune_tree_params(n_samples, n_features, max_leaf_nodes=None, min_samples_leaf=None):
    """
    Auto-tune decision tree hyperparameters from dataset shape.

    Uses the 3x-finer partitioning strategy: more leaves naturally isolate
    outliers into dedicated partitions, improving variance preservation.

    Parameters
    ----------
    n_samples : int
    n_features : int
    max_leaf_nodes : int or None
        Returned unchanged if provided (manual override).
    min_samples_leaf : int or None
        Returned unchanged if provided (manual override).

    Returns
    -------
    max_leaf_nodes : int
    min_samples_leaf : int
    """
    if min_samples_leaf is None:
        # Maintain 40:1 sample-to-feature ratio inside each leaf,
        # relaxed by 2/3 to support 3x more partitions.
        min_samples_leaf = max(5, (n_features * 40 * 2) // 3)

    if max_leaf_nodes is None:
        # 3x multiplier: finer partitions for better outlier isolation
        max_leaf_nodes = max(30, min(1500, 3 * n_samples // (min_samples_leaf * 2)))

    return max_leaf_nodes, min_samples_leaf


def fit_hvrt_tree(X, target, max_leaf_nodes, min_samples_leaf, max_depth, random_state):
    """
    Fit and return the HVRT partitioning DecisionTreeRegressor.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Normalised feature matrix.
    target : ndarray (n_samples,)
        Synthetic or real target driving partitioning.
    max_leaf_nodes : int
    min_samples_leaf : int
    max_depth : int or None
    random_state : int

    Returns
    -------
    tree : DecisionTreeRegressor (fitted)
    """
    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        min_impurity_decrease=0.0,
        random_state=random_state,
    )
    tree.fit(X, target)
    return tree


def resolve_tree_params(
    n_samples,
    n_features,
    n_partitions_override=None,
    n_partitions=None,
    min_samples_leaf=None,
    auto_tune=True,
):
    """
    Resolve max_leaf_nodes and min_samples_leaf for a tree fit.

    n_partitions_override takes precedence over n_partitions and auto-tuning,
    enabling per-operation tree granularity without a new fit() call.

    Parameters
    ----------
    n_samples, n_features : int
    n_partitions_override : int or None
        Per-operation override.
    n_partitions : int or None
        Instance-level default (may be None → auto-tuned).
    min_samples_leaf : int or None
        Instance-level default (may be None → auto-tuned).
    auto_tune : bool

    Returns
    -------
    max_leaf_nodes : int
    min_samples_leaf : int
    """
    max_leaf_nodes = (
        n_partitions_override if n_partitions_override is not None else n_partitions
    )

    if n_partitions_override is not None:
        if min_samples_leaf is None:
            min_samples_leaf = max(2, n_samples // (n_partitions_override * 3))
    elif auto_tune:
        max_leaf_nodes, min_samples_leaf = auto_tune_tree_params(
            n_samples, n_features, max_leaf_nodes, min_samples_leaf
        )
    else:
        if max_leaf_nodes is None:
            max_leaf_nodes = max(2, n_samples // 100)
        if min_samples_leaf is None:
            min_samples_leaf = max(5, n_samples // 1000)

    return max_leaf_nodes, min_samples_leaf
