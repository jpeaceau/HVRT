"""
Shared utilities for HVRT and FastHVRT.

Covers z-score normalization, decision-tree fitting, and centroid-seeded
Furthest Point Sampling (FPS).
"""

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
        If not None, returned unchanged (manual override).
    min_samples_leaf : int or None
        If not None, returned unchanged (manual override).

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
        Normalized feature matrix.
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


def centroid_fps(X, n_select):
    """
    Centroid-seeded Furthest Point Sampling.

    Deterministically selects n_select diverse samples by greedily picking
    the point farthest from the current selection, seeded at the centroid.
    Uses squared distances (no sqrt) for speed.

    Parameters
    ----------
    X : ndarray (n_points, n_features)
    n_select : int

    Returns
    -------
    selected : ndarray of int, shape (n_select,)
        Local indices into X.
    """
    n = len(X)
    if n <= n_select:
        return np.arange(n, dtype=np.int64)

    centroid = X.mean(axis=0)
    diff = X - centroid
    sq_dist = (diff * diff).sum(axis=1)
    seed = int(np.argmin(sq_dist))

    selected = [seed]
    min_sq = np.full(n, np.inf)

    for _ in range(n_select - 1):
        last = selected[-1]
        diff = X - X[last]
        sq = (diff * diff).sum(axis=1)
        np.minimum(min_sq, sq, out=min_sq)
        selected.append(int(np.argmax(min_sq)))

    return np.array(selected, dtype=np.int64)
