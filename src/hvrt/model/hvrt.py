"""
HVRT: Hierarchical Variance-Retaining Transformer — pairwise interactions.

Uses the sum of z-score-normalised pairwise feature interactions as the
synthetic partitioning target.  Points with co-occurring extreme values
across feature pairs receive high |target|, naturally isolating structural
outliers into dedicated tree leaves.

Complexity: O(n · d²) for target computation (d = number of features).

Preferred for sample *reduction*; comparable to FastHVRT for expansion.
"""

import numpy as np

from .._base import _HVRTBase
from .._kernels import _NUMBA_AVAILABLE, _pairwise_target_nb, _pairwise_target_numpy


class HVRT(_HVRTBase):
    """
    Hierarchical Variance-Retaining Transformer — pairwise interactions.

    Uses the sum of z-score-normalised pairwise feature interactions as the
    synthetic partitioning target.  Points with co-occurring extreme values
    across feature pairs receive high |target|, naturally isolating structural
    outliers into dedicated tree leaves.

    Complexity: O(n · d²) for target computation (d = number of features).

    Preferred for sample *reduction*; comparable to FastHVRT for expansion.

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
    >>> from hvrt import HVRT
    >>> model = HVRT().fit(X)
    >>> X_reduced = model.reduce(ratio=0.3)
    >>> X_synth   = model.expand(n=50000, min_novelty=0.2)
    >>> X_aug     = model.augment(n=30000)

    >>> # Sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> pipe = Pipeline([('hvrt', HVRT(mode='reduce', ratio=0.3))])
    >>> X_reduced = pipe.fit_transform(X)
    """

    def _compute_x_component(self, X_z):
        """
        Pairwise feature interaction synthetic target (O(d²) ops, O(n·d) peak memory).

        For all feature pairs (i, j), i < j:
          1. Compute element-wise product X_z[:,i] ⊙ X_z[:,j]
          2. Z-score normalise each interaction column
          3. Sum across all interaction columns

        High |score| indicates samples where many feature pairs are jointly
        extreme — the defining characteristic of structural outliers.

        Implementation
        --------------
        When ``numba`` is installed (``pip install hvrt[fast]``), delegates to
        ``_pairwise_target_nb`` — a fused LLVM-compiled kernel with O(n) peak
        memory (no intermediate arrays).  Falls back to a block-wise NumPy loop
        (O(n·d) peak memory) otherwise.  Both paths are numerically equivalent
        to within ~1e-8 on z-scored data.
        """
        X_z = np.ascontiguousarray(X_z, dtype=np.float64)
        if _NUMBA_AVAILABLE:
            return _pairwise_target_nb(X_z)
        return _pairwise_target_numpy(X_z)
