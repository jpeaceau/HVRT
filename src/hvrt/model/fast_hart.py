"""
FastHART: Hierarchical Absolute-deviation Retaining Transformer — z-score sum (fast).

MAD-based analog to FastHVRT.  Uses median+MAD scaling, an absolute_error tree
criterion, and MAD-normalised y for supervised partitioning.

The x-component is identical to FastHVRT (z-score sum) but operates on
MAD-normalised input, shifting the signal from total variance to total
absolute deviation.

Complexity: O(n · d) — significantly faster than HART for high-d data.

Preferred default for sample *expansion* on heavy-tailed data at scale.
"""

import numpy as np

from .._hart_base import _HARTBase


class FastHART(_HARTBase):
    """
    Hierarchical Absolute-deviation Retaining Transformer — z-score sum (fast).

    MAD-based analog to FastHVRT.  Replaces mean/std normalisation and
    squared-error splits with median/MAD normalisation and absolute-error
    splits, making partitioning robust to heavy-tailed distributions and
    outliers.

    The z-score-sum x-component is identical to FastHVRT but applied to
    MAD-normalised features.

    Complexity: O(n · d) — significantly faster than HART for high-d data.

    Preferred default for sample *expansion* on heavy-tailed data at scale;
    use HART for reduction tasks.

    Parameters
    ----------
    n_partitions : int or None
        Maximum leaf nodes.  Auto-tuned if None.
    min_samples_leaf : int or None
        Minimum samples per leaf.  Auto-tuned if None.
    max_depth : int or None
        Maximum tree depth.
    min_samples_per_partition : int, default=5
    y_weight : float, default=0.0
        0.0 = unsupervised; 1.0 = supervised (MAD-normalised y drives splits).
    auto_tune : bool, default=True
    bandwidth : float or 'auto', default='auto'
        KDE bandwidth for expand().
    random_state : int, default=42

    Examples
    --------
    >>> from hvrt import FastHART
    >>> model = FastHART().fit(X)
    >>> X_synth   = model.expand(n=50000)
    >>> X_reduced = model.reduce(ratio=0.3)
    """

    def _compute_x_component(self, X_z):
        """
        Z-score sum synthetic target (O(d)).

        target_i = sum_j X_z[i, j]

        Captures "total absolute deviation from typical" per sample on
        MAD-normalised features.
        """
        return X_z.sum(axis=1)
