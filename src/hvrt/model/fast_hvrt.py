"""
FastHVRT: Hierarchical Variance-Retaining Transformer — z-score sum (fast).

Uses the sum of z-scores across features as the synthetic partitioning
target.  Samples with simultaneously extreme values across many features
receive a high |target| and are isolated into dedicated tree leaves.

Complexity: O(n · d) — significantly faster than HVRT for high-d data.

Preferred default for sample *expansion* at scale; benchmarks show no
meaningful quality gap vs HVRT for generation tasks.
"""

import numpy as np

from .._base import _HVRTBase


class FastHVRT(_HVRTBase):
    """
    Hierarchical Variance-Retaining Transformer — z-score sum (fast).

    Uses the sum of z-scores across features as the synthetic partitioning
    target.  Samples with simultaneously extreme values across many features
    receive a high |target| and are isolated into dedicated tree leaves.

    Complexity: O(n · d) — significantly faster than HVRT for high-d data.

    Preferred default for sample *expansion* at scale; benchmarks show no
    meaningful quality gap vs HVRT for generation tasks.

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
        0.0 = unsupervised; 1.0 = supervised (y-extremeness drives splits).
    auto_tune : bool, default=True
    mode : str, default='reduce'
        Default operation for fit_transform().
    bandwidth : float, default=0.5
        KDE bandwidth for expand().  0.5 achieves near-perfect tail
        preservation (error 0.004) while maintaining strong marginal
        fidelity (0.974).  Can be overridden per expand() call.
    random_state : int, default=42

    Examples
    --------
    >>> from hvrt import FastHVRT
    >>> model = FastHVRT().fit(X)
    >>> X_synth   = model.expand(n=50000, min_novelty=0.2)
    >>> X_reduced = model.reduce(ratio=0.3)
    """

    def _compute_x_component(self, X_z):
        """
        Z-score sum synthetic target (O(d)).

        target_i = sum_j X_z[i, j]

        Captures "total deviation from typical" per sample: high |target|
        indicates overall extremeness across the feature set.
        """
        return X_z.sum(axis=1)
