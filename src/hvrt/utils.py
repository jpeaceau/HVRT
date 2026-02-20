"""
Shared utilities for HVRT and FastHVRT.

Covers auto-tuning of decision-tree hyperparameters.
"""

import numpy as np


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



