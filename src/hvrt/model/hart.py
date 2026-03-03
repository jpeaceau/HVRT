"""
HART: Hierarchical Absolute-deviation Retaining Transformer — pairwise interactions.

MAD-based analog to HVRT.  Uses median+MAD scaling, an absolute_error tree
criterion, and MAD-normalised y for supervised partitioning.

The x-component is identical to HVRT (pairwise z-score interactions) but
operates on MAD-normalised input, so the structural signal reflects joint
absolute deviation rather than joint variance.

Complexity: O(n · d²) for target computation (d = number of features).

Preferred for sample *reduction* on heavy-tailed or outlier-rich data.
"""

import numpy as np

from .._hart_base import _HARTBase
from .._kernels import _pairwise_target


class HART(_HARTBase):
    """
    Hierarchical Absolute-deviation Retaining Transformer — pairwise interactions.

    MAD-based analog to HVRT.  Replaces mean/std normalisation and squared-error
    splits with median/MAD normalisation and absolute-error splits, making
    partitioning robust to heavy-tailed distributions and outliers.

    The pairwise x-component is identical to HVRT but applied to MAD-normalised
    features, shifting the structural signal from joint variance to joint
    absolute deviation.

    Complexity: O(n · d²) for target computation (d = number of features).

    Preferred for sample *reduction* on heavy-tailed or outlier-rich data;
    use FastHART for expansion at scale.

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
    >>> from hvrt import HART
    >>> model = HART().fit(X, y)
    >>> X_reduced = model.reduce(ratio=0.3)
    >>> X_synth   = model.expand(n=50000)
    >>> X_aug     = model.augment(n=30000)
    """

    def _compute_x_component(self, X_z):
        """
        Pairwise feature interaction synthetic target (O(d²) ops, O(n·d) peak memory).

        Identical implementation to HVRT._compute_x_component, but operates on
        MAD-normalised X_z so joint absolute deviation drives the signal.
        """
        return _pairwise_target(np.ascontiguousarray(X_z, dtype=np.float64))
